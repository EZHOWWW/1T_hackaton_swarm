# executor.py
import os
from solution.geometry import Vector
from solution.simulation import DroneInfo

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Assume DroneInfo and Vector are defined and imported as previously discussed
# (Placeholder definitions omitted for brevity, assume they are available)

MAX_LIDAR_RANGE = 10.0


# Define the RL Model (Actor Network)
def create_actor_model(state_size, action_size):
    input_layer = layers.Input(shape=(state_size,))
    hidden_layer_1 = layers.Dense(
        256, activation="relu", kernel_initializer="he_normal"
    )(input_layer)  # Increased neurons, He init
    hidden_layer_2 = layers.Dense(
        256, activation="relu", kernel_initializer="he_normal"
    )(hidden_layer_1)
    # Output mean and log_std for a Gaussian distribution (for PPO stochastic policy)
    mean_output = layers.Dense(action_size, activation="sigmoid", name="mean_action")(
        hidden_layer_2
    )  # Mean between 0 and 1
    # It's common to have log_std as trainable parameters, not as a direct output
    # Let's simplify slightly for the model architecture, but use the concept
    # of a distribution with learned mean and stddev in the PPOAgent.
    # For this model definition, let's return the mean directly, and handle stddev elsewhere
    # OR, better for PPO, output mean and log_std from the model itself.

    # Option 1: Model outputs mean and log_std
    # log_std_output = layers.Dense(action_size, activation='tanh', name='log_std_action')(hidden_layer_2) # Tanh helps bound log_std
    # model = models.Model(inputs=input_layer, outputs=[mean_output, log_std_output])
    # return model

    # Option 2: Model outputs only mean (simpler for now, PPOAgent adds noise or uses fixed std)
    # Let's stick to the original simple actor outputting mean [0,1] and handle
    # stochasticity/log_prob in PPOAgent for clarity with basic model structure.
    model = models.Model(inputs=input_layer, outputs=mean_output)
    return model


# Define the RL Model (Critic Network)
def create_critic_model(state_size):
    input_layer = layers.Input(shape=(state_size,))
    hidden_layer_1 = layers.Dense(
        256, activation="relu", kernel_initializer="he_normal"
    )(input_layer)
    hidden_layer_2 = layers.Dense(
        256, activation="relu", kernel_initializer="he_normal"
    )(hidden_layer_1)
    output_layer = layers.Dense(1, activation="linear", name="state_value")(
        hidden_layer_2
    )
    model = models.Model(inputs=input_layer, outputs=output_layer)
    return model


class DroneExecutor:
    def __init__(self, drone):
        # Define state and action sizes
        self.state_size = (
            3 + 3 + 3 + 3 + 10 + 3 + 1
        )  # pos, vel, angle, ang_vel, lidars, target_vec, dist_to_target
        self.action_size = 8  # 8 motor powers

        # Create the Actor and Critic models
        # Note: If create_actor_model returned mean, log_std, adjust calls here
        self.actor = create_actor_model(self.state_size, self.action_size)
        self.critic = create_critic_model(self.state_size)

        # Load pre-trained weights (for deployment)
        # self.load_weights("./drone_rl_models") # Path where learn.py saves

    def prepare_state_vector(self, params: DroneInfo, target: Vector) -> np.ndarray:
        """
        Prepares the input state vector for the RL model from drone info and target.
        Handles lidar data where 0 means max range (10).
        """
        # Get lidar data, treating 0 as MAX_LIDAR_RANGE
        # Assuming lidars dict structure is {'lidars': [v1, v2, ...]}
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
        # Ensure Vector class supports subtraction and distance_to
        try:
            target_vec = target - params.possition
            dist_to_target = params.possition.distance_to(target)
        except Exception as e:
            print(f"Error calculating target vector/distance: {e}")
            # Provide dummy safe values if Vector operations fail
            target_vec = Vector(0, 0, 0)
            dist_to_target = 1000.0  # Large distance

        # Concatenate all state components
        state = (
            [
                params.possition.x,
                params.possition.y,
                params.possition.z,
                params.velocity.x,
                params.velocity.y,
                params.velocity.z,
                # Angles and angular velocities might benefit from wrapping/normalization
                # For simplicity, use raw values first.
                params.angle[0],
                params.angle[1],
                params.angle[2],  # roll, pitch, yaw in degrees
                params.angular_velocity[0],
                params.angular_velocity[1],
                params.angular_velocity[2],  # deg/s
            ]
            + processed_lidars
            + [target_vec.x, target_vec.y, target_vec.z, dist_to_target]
        )

        # Consider adding some form of normalization or scaling for positions, velocities, etc.
        # based on arena bounds if values vary widely. E.g., pos.x / (ARENA_SIZE_X / 2)

        # Convert to numpy array and reshape for the model (batch size of 1)
        return np.array(state, dtype=np.float32).reshape(1, -1)

    def move_to(self, params: DroneInfo, target: Vector, dt: float) -> list[float]:
        """
        Uses the trained RL model (Actor) to predict motor commands.
        This method is for INFERENCE/DEPLOYMENT after training.
        """
        if not params.is_alive:
            # If drone is dead, return zero motor power
            return [0.0] * self.action_size

        state_vector = self.prepare_state_vector(params, target)

        # In inference, we use the mean of the policy distribution
        # If actor model outputs mean, log_std: mean, log_std = self.actor.predict(state_vector)
        # If actor model outputs just mean:
        predicted_actions = self.actor.predict(state_vector)

        # The output is a numpy array, convert to a list of floats
        motor_powers = np.squeeze(predicted_actions).tolist()

        # Ensure actions are strictly within [0, 1]
        motor_powers = [max(0.0, min(1.0, p)) for p in motor_powers]

        return motor_powers

    # --- Methods used by learn.py for TRAINING ---
    # These would typically be in the Agent class in learn.py, but keeping them
    # minimal here for clarity as per your original structure.
    # In the PPOAgent class, we will use the actor/critic models directly.

    def load_weights(self, path):
        """Loads model weights from specified path."""
        actor_path = os.path.join(path, "actor_weights.h5")
        critic_path = os.path.join(path, "critic_weights.h5")
        try:
            # Ensure models are built before loading weights (call predict or a dummy forward pass)
            dummy_state = np.zeros((1, self.state_size), dtype=np.float32)
            self.actor(dummy_state)
            self.critic(dummy_state)

            self.actor.load_weights(actor_path)
            self.critic.load_weights(critic_path)
            print(f"Loaded model weights from {path}.")
        except tf.errors.NotFoundError:
            print(f"No weights found at {path}. Starting from scratch.")
        except Exception as e:
            print(f"Error loading weights: {e}")
