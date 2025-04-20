import numpy as np
import tensorflow as tf
from collections import deque
import os

from solution.geometry import Vector
from solution.simulation import DroneInfo


# Import the model creation functions (we will create models directly in PPOAgent)
from solution.executor import create_actor_model, create_critic_model, MAX_LIDAR_RANGE


class PPOAgent:
    def __init__(
        self,
        state_size,
        action_size,
        arena_bounds,
        dt,
        base_position,
        target_detection_radius,
        ppo_epochs,
        batch_size,
        save_dir,
    ):
        self.state_size = state_size
        self.action_size = action_size

        # Use the model creation functions
        # If actor outputted mean/log_std: self.actor = create_actor_model(...)
        # Here, actor outputs only mean [0,1]
        self.actor = create_actor_model(state_size, action_size)
        self.critic = create_critic_model(state_size)

        # Optimizers
        self.actor_optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.0003
        )  # Common LR for PPO
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        # Hyperparameters (tuned for PPO)
        self.gamma_discount = 0.99  # Discount factor
        self.gae_lambda = 0.95  # For Generalized Advantage Estimation (GAE)
        self.clip_ratio = 0.2  # PPO clipping parameter
        self.ppo_epochs = 10  # Number of times to update using the same batch
        self.batch_size = (
            2048  # Size of experience batch (recommend collecting from multiple envs)
        )
        self.value_coeff = 0.5  # Coefficient for critic loss
        self.entropy_coeff = 0.01  # Coefficient for entropy bonus

        # Exploration noise scale (for getting actions during data collection)
        # A proper PPO samples from a distribution. This is a hacky way
        # to add exploration if actor only outputs mean.
        self.explore_noise_scale = 0.1

        # Buffer to store transitions (state_vector, action, reward, value, done, log_prob)
        self.buffer = deque(maxlen=self.batch_size * 2)  # Use deque for max length

        # Instantiate the mock simulation
        self.target_detection_radius = target_detection_radius

        # Directory to save models
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # Load weights if available
        # self.load_weights(self.save_dir) # Uncomment to load previous training

    def prepare_state_vector(self, params: DroneInfo, target: Vector) -> np.ndarray:
        """
        Prepares the input state vector for the RL model from drone info and target.
        Handles lidar data where 0 means max range (10).
        Should match the logic in DroneExecutor.
        """
        raw_lidars = (
            params.lidars.get("lidars", [0.0] * 10) if params.lidars else [0.0] * 10
        )
        # Process lidar data: if 0, set to MAX_LIDAR_RANGE. Otherwise, use the value.
        # Also, scale lidar values by max range for normalization.
        processed_lidars = [
            (v if v > 1e-6 else MAX_LIDAR_RANGE) / MAX_LIDAR_RANGE  # Scale to [0, 1]
            for v in raw_lidars
        ]

        # Calculate vector to target and distance
        try:
            target_vec = target - params.possition
            dist_to_target = params.possition.distance_to(target)
        except Exception as e:
            print(f"Error calculating target vector/distance in agent: {e}")
            target_vec = Vector(0, 0, 0)
            dist_to_target = 1000.0

        state = (
            [
                params.possition.x,
                params.possition.y,
                params.possition.z,
                params.velocity.x,
                params.velocity.y,
                params.velocity.z,
                params.angle[0],
                params.angle[1],
                params.angle[2],
                params.angular_velocity[0],
                params.angular_velocity[1],
                params.angular_velocity[2],
            ]
            + processed_lidars
            + [target_vec.x, target_vec.y, target_vec.z, dist_to_target]
        )

        # Convert to numpy array (without batch dim for buffer)
        return np.array(state, dtype=np.float32)

    # For PPO, we need to sample from a distribution and get the log probability
    # If the actor outputs mean and log_std, this function is straightforward.
    # If the actor outputs only mean (as implemented), we'll sample from a Gaussian
    # centered at the mean, with a fixed or learned stddev. The log_prob calculation
    # for clipped actions is tricky.

    def get_action_and_log_prob(self, state_vector, explore=True):
        """Gets action by sampling from a Gaussian centered at actor mean (with noise for exploration)
        and calculates approximate log probability."""
        # state_vector = state_vector.reshape(1, -1)  # Add batch dimension
        state_vector = np.expand_dims(state_vector, axis=0)  # Преобразует (n,) в (1, n)

        # Get the mean action from the actor
        print("State vector shape:", state_vector.shape)
        mean_action = self.actor(state_vector)[0]  # Shape (8,)
        mean_action_np = mean_action.numpy()  # Convert to numpy

        # Define a base standard deviation (could be learned by actor)
        # Let's use a fixed stddev for simplicity with the current actor output
        std_dev = tf.constant(
            [self.explore_noise_scale] * self.action_size, dtype=tf.float32
        )  # Fixed std dev

        # Create a Gaussian distribution centered at the mean action
        # Note: this Gaussian is defined in the *unclipped* action space.
        # We need to handle the clipping. The log_prob of the *clipped* action
        # is what's strictly needed, which is complex (requires density of clipped distribution).
        # A common approximation is to calculate the log_prob of the original sample
        # from the unclipped distribution. This is what we'll do here, but be aware
        # of the inaccuracy when actions are clipped.

        # For exploration during training, sample from the distribution
        if explore:
            # Sample from the Gaussian distribution
            action = tf.random.normal(
                shape=(self.action_size,), mean=mean_action, stddev=std_dev
            )
            # Calculate log probability of the sampled action (in the unclipped space)
            # log_prob = tf.reduce_sum(tf.math.log(1e-6 + 1.0 / (std_dev * tf.sqrt(2 * np.pi))) - ((action - mean_action) / std_dev)**2 / 2, axis=-1)
            # Using tf.math.log_prob directly on the normal distribution is cleaner
            normal_dist = tf.compat.v1.distributions.Normal(
                loc=mean_action, scale=std_dev
            )
            log_prob = tf.reduce_sum(
                normal_dist.log_prob(action), axis=-1
            )  # Sum log probs across actions

            # Clamp the action to [0, 1] after sampling
            action = tf.clip_by_value(action, 0.0, 1.0)

        else:  # For inference/evaluation, use the mean action directly
            action = mean_action
            log_prob = tf.constant(
                0.0, dtype=tf.float32
            )  # Log prob is not needed/meaningful for deterministic inference
            action = tf.clip_by_value(action, 0.0, 1.0)

        return (
            action.numpy().tolist(),
            log_prob.numpy(),
        )  # Return action as list, log_prob as scalar

    def get_value(self, state_vector):
        """Gets the state value from the critic model."""
        state_vector = state_vector.reshape(1, -1)  # Add batch dimension
        value = self.critic(state_vector)
        return tf.squeeze(value).numpy()  # Return scalar value

    def store_transition(self, state, action, reward, next_state, done, log_prob):
        """Stores a transition in the buffer."""
        # Store state_vector, action, reward, next_state_vector, done, log_prob
        self.buffer.append((state, action, reward, next_state, done, log_prob))

    def update_models(self):
        """Updates the Actor and Critic models using the collected buffer (PPO algorithm)."""
        if len(self.buffer) < self.batch_size:
            return  # Not enough data to update

        # Get a batch of data from the buffer (e.g., the last batch_size transitions)
        # For on-policy methods like PPO, it's common to use the *entire* collected data
        # since the last update, or a fixed-size batch if buffer is larger.
        # Let's use the entire buffer content collected since last update.
        # This assumes update is called when buffer >= batch_size.
        # If buffer size is fixed, use random sampling or take last N.
        # Using the whole buffer (and clearing it) is simpler for illustration.
        experiences = list(self.buffer)
        self.buffer.clear()  # Clear buffer after getting data for update

        states, actions, rewards, next_states, dones, log_probs_old = zip(*experiences)

        # Convert to tensors
        states = tf.convert_to_tensor(np.array(states, dtype=np.float32))
        actions = tf.convert_to_tensor(np.array(actions, dtype=np.float32))
        rewards = tf.convert_to_tensor(
            np.array(rewards, dtype=np.float32), dtype=tf.float32
        )
        next_states = tf.convert_to_tensor(np.array(next_states, dtype=np.float32))
        dones = tf.convert_to_tensor(
            np.array(dones, dtype=np.float32), dtype=tf.float32
        )
        log_probs_old = tf.convert_to_tensor(
            np.array(log_probs_old, dtype=np.float32), dtype=tf.float32
        )

        # Calculate returns (Discounted future rewards) and Advantages (GAE)
        # Get values for all states in the batch
        values = self.critic(states)[:, 0]  # Shape (batch_size,)
        next_values = self.critic(next_states)[:, 0]  # Shape (batch_size,)

        # Calculate returns (G) - Simplified: just sum discounted rewards (Monte Carlo like)
        # or more commonly in Actor-Critic/PPO: G = R + gamma * V(next_S)
        # Let's calculate returns using V(s') for non-terminal states
        returns = rewards + self.gamma_discount * next_values * (1 - dones)

        # Calculate Advantages using GAE (Simplified - requires iterating backward)
        # A = R_t + gamma*V(S_{t+1}) - V(S_t) + gamma*lambda*A_{t+1}
        # Let's use a simplified TD(0) advantage for illustration: A = R + gamma*V(S') - V(S)
        # Proper GAE requires iterating backwards through a trajectory.
        # For batch update, calculate deltas first, then advantages.
        deltas = rewards + self.gamma_discount * next_values * (1 - dones) - values

        # Simple advantage calculation (TD error)
        advantages = deltas
        # To implement full GAE: iterate backwards over deltas: A_t = delta_t + gamma*lambda*A_{t+1}
        # For batch update, this needs careful indexing or using a library function.
        # Let's stick to the TD(0) advantage for simplicity in this manual example.

        # Normalize advantages
        advantages = (advantages - tf.reduce_mean(advantages)) / (
            tf.math.sqrt(tf.reduce_variance(advantages)) + 1e-8
        )

        # --- PPO Update Loop ---
        # Perform multiple epochs over the same batch
        for _ in range(self.ppo_epochs):
            # Recalculate log probabilities with the current policy weights
            # Pass states through actor again
            # Using the simplified log_prob calculation from get_action_and_log_prob
            mean_new = self.actor(states)  # Shape (batch_size, action_size)

            # Need std_dev for log_prob calculation. Using the fixed one from get_action_and_log_prob
            std_dev_tensor = tf.constant(
                [self.explore_noise_scale] * self.action_size, dtype=tf.float32
            )  # Same stddev

            # Create a Gaussian distribution for log_prob calculation
            # IMPORTANT: log_prob calculation is sensitive to how actions were sampled
            # and if they were clipped. This is an approximation.
            normal_dist_new = tf.compat.v1.distributions.Normal(
                loc=mean_new, scale=std_dev_tensor
            )
            # Calculate log prob of the *original actions* (from buffer) under the *new* policy
            log_probs_new = tf.reduce_sum(normal_dist_new.log_prob(actions), axis=-1)

            # Calculate the ratio of new vs old probabilities
            # Avoid division by zero or log(0)
            ratio = tf.exp(log_probs_new - log_probs_old)

            # Calculate the clipped objective function (Actor Loss)
            clipped_ratio = tf.clip_by_value(
                ratio, 1 - self.clip_ratio, 1 + self.clip_ratio
            )
            actor_loss_1 = ratio * advantages
            actor_loss_2 = clipped_ratio * advantages
            actor_loss = -tf.reduce_mean(
                tf.minimum(actor_loss_1, actor_loss_2)
            )  # Negative sign for maximization

            # Add Entropy Bonus to Actor Loss
            # Entropy for Gaussian: 0.5 * log(2 * pi * e * std^2)
            entropy = tf.reduce_sum(
                0.5 * tf.math.log(2 * np.pi * np.e * std_dev_tensor**2), axis=-1
            )
            entropy_bonus = tf.reduce_mean(entropy) * self.entropy_coeff
            actor_loss = (
                actor_loss - entropy_bonus
            )  # Add entropy bonus (maximize entropy)

            # Update Actor
            with tf.GradientTape() as tape:
                # Need to re-calculate the actor loss within the tape scope for gradients
                mean_new_tape = self.actor(states)
                normal_dist_new_tape = tf.compat.v1.distributions.Normal(
                    loc=mean_new_tape, scale=std_dev_tensor
                )
                log_probs_new_tape = tf.reduce_sum(
                    normal_dist_new_tape.log_prob(actions), axis=-1
                )
                ratio_tape = tf.exp(log_probs_new_tape - log_probs_old)
                clipped_ratio_tape = tf.clip_by_value(
                    ratio_tape, 1 - self.clip_ratio, 1 + self.clip_ratio
                )
                actor_loss_tape = -tf.reduce_mean(
                    tf.minimum(ratio_tape * advantages, clipped_ratio_tape * advantages)
                )
                # Add entropy bonus again
                entropy_tape = tf.reduce_sum(
                    0.5 * tf.math.log(2 * np.pi * np.e * std_dev_tensor**2), axis=-1
                )
                actor_loss_tape = (
                    actor_loss_tape - tf.reduce_mean(entropy_tape) * self.entropy_coeff
                )

            actor_grads = tape.gradient(actor_loss_tape, self.actor.trainable_variables)
            # Clip gradients if needed (often helps stability)
            # actor_grads, _ = tf.clip_by_global_norm(actor_grads, 0.5) # Example clipping
            self.actor_optimizer.apply_gradients(
                zip(actor_grads, self.actor.trainable_variables)
            )

            # Update Critic
            with tf.GradientTape() as tape:
                # Critic predicts state values
                predicted_values = self.critic(states)[:, 0]  # Shape (batch_size,)

                # Target values for the critic are the calculated returns (advantages + old_values)
                # GAE advantages combined with V(S) provide a low-variance target
                # This target V_target = A + V_old is a common PPO critic target
                target_values = advantages + values

                # Mean Squared Error loss for the Critic
                critic_loss = tf.reduce_mean(
                    tf.square(target_values - predicted_values)
                )

            critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
            # critic_grads, _ = tf.clip_by_global_norm(critic_grads, 0.5) # Example clipping
            self.critic_optimizer.apply_gradients(
                zip(critic_grads, self.critic.trainable_variables)
            )

        # After PPO epochs, the buffer is cleared for the next collection phase.

    def save_weights(self, path):
        """Saves model weights."""
        actor_path = os.path.join(path, "actor_weights.h5")
        critic_path = os.path.join(path, "critic_weights.h5")
        self.actor.save_weights(actor_path)
        self.critic.save_weights(critic_path)
        print(f"Model weights saved to {actor_path} and {critic_path}")

    def load_weights(self, path):
        """Loads model weights from specified path."""
        actor_path = os.path.join(path, "actor_weights.h5")
        critic_path = os.path.join(path, "critic_weights.h5")
        try:
            # Ensure models are built before loading weights
            dummy_state = np.zeros((1, self.state_size), dtype=np.float32)
            self.actor(dummy_state)
            self.critic(dummy_state)

            self.actor.load_weights(actor_path)
            self.critic.load_weights(critic_path)
            print(f"Loaded model weights from {path}.")
        except tf.errors.NotFoundError:
            print(f"No weights found in {path}. Starting from scratch.")
        except Exception as e:
            print(f"Error loading weights: {e}")
