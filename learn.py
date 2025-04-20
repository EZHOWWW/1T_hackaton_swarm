# learn.py
# learn.py
import numpy as np
import tensorflow as tf
import random
from collections import deque
import os
import time  # For logging training time

# Assuming these are correctly defined and importable from your project structure
from solution.geometry import Vector
from solution.simulation import (
    Simulation,
    DroneInfo,
    DronesInfo,
)  # Assuming DronesInfo is list[DroneInfo]

# Assuming PPOAgent, create_actor_model, create_critic_model, MAX_LIDAR_RANGE are in solution.RL.model
from solution.RL.model import (
    PPOAgent,
    create_actor_model,
    create_critic_model,
    MAX_LIDAR_RANGE,
)


# --- Simulation Parameters ---
ARENA_BOUNDS = (
    -100,
    100,
    0,
    20,
    -100,
    100,
)  # (min_x, max_x, min_y, max_y, min_z, max_z)
SIMULATION_DT = 0.1  # Time step
BASE_POSITION = Vector(
    -77.01, 0.48, 75.31
)  # Fixed start position for the drone (e.g., on the ground at center)
TARGET_DETECTION_RADIUS = (
    2.0  # How close drone needs to be to consider target reached XZ
)

# --- RL Reward Coefficients ---
# These need tuning based on the simulation scale and desired behavior
REWARD_COEFF_DISTANCE_XZ = 0.5  # Penalty for XZ distance (per meter)
REWARD_COEFF_HEIGHT_PENALTY = 5.0  # Penalty for being below target Y (per meter below)
REWARD_COEFF_TIME = 0.05  # Penalty per simulation step
REWARD_BONUS_REACH = 2000.0  # Bonus for reaching the target
REWARD_PENALTY_CRASH = 5000.0  # Large penalty for crashing

# --- Training Parameters ---
STATE_SIZE = (
    3 + 3 + 3 + 3 + 10 + 3 + 1
)  # pos, vel, angle, ang_vel, lidars, target_vec, dist
ACTION_SIZE = 8  # motor powers
BATCH_SIZE = 256  # Number of experiences per agent to collect before update
PPO_EPOCHS = 10  # Number of times to update using the same batch


class LearnSimulation:
    def __init__(self, sim_proxy: Simulation, num_drones: int = 5):
        self.sim_proxy = sim_proxy
        self.num_drones = num_drones

        # Create a separate PPOAgent for each drone
        self.agents: list[PPOAgent] = []
        for i in range(self.num_drones):
            agent = PPOAgent(
                state_size=STATE_SIZE,
                action_size=ACTION_SIZE,
                arena_bounds=ARENA_BOUNDS,  # Pass simulation bounds to agent
                dt=SIMULATION_DT,  # Pass simulation dt to agent
                base_position=BASE_POSITION,  # Pass base position (might not be used by agent directly, but sim needs it)
                target_detection_radius=TARGET_DETECTION_RADIUS,  # Pass target radius
                # Pass reward coefficients if agent calculates rewards internally (better to calculate in LearnSimulation)
                # Or just note that agent's internal reward logic should match LearnSimulation's
                # Let's calculate rewards here and pass them to agent.store_transition
                ppo_epochs=PPO_EPOCHS,  # Pass PPO hyperparams
                batch_size=BATCH_SIZE,  # Pass PPO hyperparams
                save_dir=f"./drone_rl_models/agent_{i}",  # Separate save dir for each agent
            )
            self.agents.append(agent)

        self.episode_rewards: list[deque] = [
            deque(maxlen=100) for _ in range(self.num_drones)
        ]  # To track recent avg reward per drone
        self._current_targets: list[
            Vector
        ] = []  # Store targets for the current episode
        self._step_count = 0
        self._max_steps_per_episode = 1000  # Max steps before episode timeout

        # Store info from the *previous* step to calculate transition (s, a, r, s', done)
        # We need state_vector, action, log_prob from t, and reward, next_state_vector, done from t+1
        self._prev_step_data: list[dict] = [{} for _ in range(self.num_drones)]

    def start_learning(
        self, episodes: int = 10000, save_interval: int = 200, log_interval: int = 10
    ):
        """main function to start learning"""
        print(
            f"Starting training for {self.num_drones} drones over {episodes} episodes..."
        )

        # Load initial weights for all agents
        # Note: Consider if you want agents to start with same weights or different
        # Loading from separate dirs means they start independent unless you copy weights
        for i, agent in enumerate(self.agents):
            agent.load_weights(agent.save_dir)  # Each agent loads from its own dir

        total_steps = 0
        for ep in range(1, episodes + 1):
            episode_start_time = time.time()
            episode_steps = self.learn_episod()
            total_steps += episode_steps
            episode_end_time = time.time()

            # Log progress
            if ep % log_interval == 0:
                self.log(ep, total_steps, episode_end_time - episode_start_time)

            # Save weights
            if ep % save_interval == 0:
                self.save(ep)

        print("\nTraining finished.")
        self.save(episodes)  # Save final weights

    def learn_episod(self):
        """one learn episod"""
        # Reset the simulation and get initial info for all drones
        # Assuming sim_proxy.reset() resets all drones to BASE_POSITION and initial state
        self.sim_proxy.reset()
        current_info = self.sim_proxy.get_drones_info()  # Get info *after* reset

        # Generate random targets for each drone
        self._current_targets = self._get_random_targets(ARENA_BOUNDS)

        # Reset step counter and previous step data storage
        self._step_count = 0
        self._prev_step_data = [
            {} for _ in range(self.num_drones)
        ]  # Clear previous data

        stop = False
        episode_steps = 0

        # Main episode loop
        while not stop and self._step_count < self._max_steps_per_episode:
            episode_steps += 1
            self._step_count += 1

            # Get actions for all drones for the current state
            all_engines = self._get_actions_for_drones(
                current_info, self._current_targets
            )

            # Apply actions to the simulation
            self.sim_proxy.set_drones(all_engines)

            # The simulation advances. Get the *next* state information
            next_info = self.sim_proxy.get_drones_info()

            # Process transitions, calculate rewards, and update agents
            self._process_step_transitions(
                current_info, all_engines, next_info, self._current_targets
            )

            # Update current info for the next step
            current_info = next_info

            # Check if the episode should stop
            stop = self._stop_episod(next_info, self._current_targets)

            # Check if any agent buffer is full and perform update
            for i, agent in enumerate(self.agents):
                if len(agent.buffer) >= agent.batch_size:
                    # print(f"Agent {i} buffer full ({len(agent.buffer)}). Updating models.")
                    agent.update_models()
                    # Note: In a truly parallel setup, updates might happen on a separate thread/process

        # --- Episode End ---
        # Process the last step's transition with done=True
        # Rewards for the final step (crash/reach/timeout) are already included by _process_step_transitions
        # If the episode timed out, the 'done' flag needs to be true for the last transition
        # _process_step_transitions handles the done flag correctly based on sim state and target check

        # Log episode reward for each agent
        # The total reward for an episode is the sum of rewards collected over steps
        # We need to accumulate rewards per drone throughout the episode
        # Let's add an episode_reward accumulator to each agent or store it here
        # Store reward in _prev_step_data and sum it up per episode

        # Sum rewards from the last step and update episode reward accumulators
        # Need to get final rewards for the step that ended the episode
        final_rewards = [
            data.get("reward", 0.0) for data in self._prev_step_data
        ]  # Get reward calculated in the last _process_step_transitions

        # Add logging for each drone's episode reward
        # Need to track episode reward per drone over the episode duration
        # Let's add episode_reward tracking here in LearnSimulation
        episode_rewards_this_ep = [0.0] * self.num_drones
        # This requires accumulating rewards per drone step by step.
        # Let's adjust _process_step_transitions to return rewards and accumulate them here.

        # A simpler way for logging: sum up rewards collected in buffer? No, buffer clears.
        # Let's add episode_reward accumulators to each agent temporarily during the episode.
        # Or just sum up rewards from the buffer *before* update? No, buffer is for batches.
        # Best way: Accumulate rewards here in LearnSimulation step-by-step.

        # Let's refactor _process_step_transitions slightly to return rewards and done flags
        # and sum rewards here.

        # Re-think learn_episod loop slightly to handle rewards and done flags properly
        # Init accumulators: episode_rewards_this_ep = [0.0] * self.num_drones

        # Inside the loop, after _process_step_transitions, which returns rewards and done flags:
        # step_rewards, step_dones, step_statuses = self._process_step_transitions(...)
        # for i in range(self.num_drones):
        #     episode_rewards_this_ep[i] += step_rewards[i]
        #     # Optionally log drone status if it just became done
        #     if step_dones[i] and self._prev_step_data[i].get('was_alive', True): # Check if it just died/finished
        #          print(f"  Drone {i} finished episode: Status={step_statuses[i]}")

        # At the end of the episode loop (after while not stop):
        # for i in range(self.num_drones):
        #      self.episode_rewards[i].append(episode_rewards_this_ep[i]) # Store total episode reward for this drone

        # Let's implement this more robust accumulation

        # --- Revised learn_episod loop ---
        self.sim_proxy.reset()
        current_info = self.sim_proxy.get_drones_info()
        self._current_targets = self._get_random_targets(ARENA_BOUNDS)
        self._step_count = 0
        self._prev_step_data = [{} for _ in range(self.num_drones)]
        episode_rewards_this_ep = [0.0] * self.num_drones
        drones_done_status = [False] * self.num_drones  # Track if each drone is done

        while (
            not all(drones_done_status)
            and self._step_count < self._max_steps_per_episode
        ):
            self._step_count += 1

            # 1. Get actions and log_probs for the *current* state of each active drone
            all_engines = []
            current_state_vectors = []  # Store state vectors for transitions
            current_log_probs = []  # Store log probs for transitions

            for i in range(self.num_drones):
                if not drones_done_status[i]:  # Only get action for active drones
                    drone_info = current_info.drones[
                        i
                    ]  # Assuming info is DronesInfo with .drones list
                    target = self._current_targets[i]
                    agent = self.agents[i]

                    state_vector = agent.prepare_state_vector(drone_info, target)
                    action, log_prob = agent.get_action_and_log_prob(
                        state_vector, explore=True
                    )

                    all_engines.append(action)
                    current_state_vectors.append(state_vector)
                    current_log_probs.append(log_prob)

                    # Store pre-step data for transition (match by drone index)
                    self._prev_step_data[i]["state_vector"] = state_vector
                    self._prev_step_data[i]["action"] = action
                    self._prev_step_data[i]["log_prob"] = log_prob
                    self._prev_step_data[i]["was_active"] = (
                        True  # Mark as active this step
                    )
                else:
                    # If drone is done, send zero engines or last engines (check sim requirements)
                    # Sending zeros is safer
                    all_engines.append([0.0] * ACTION_SIZE)
                    self._prev_step_data[i]["was_active"] = False  # Mark as inactive

            # Ensure all_engines has 5 entries, even if some drones are done
            if len(all_engines) < self.num_drones:
                # This should not happen if loop is correct, but as a safeguard
                print("Warning: all_engines incomplete. Padding with zeros.")
                while len(all_engines) < self.num_drones:
                    all_engines.append([0.0] * ACTION_SIZE)

            # 2. Apply actions to the simulation
            self.sim_proxy.set_drones(all_engines)

            # 3. The simulation advances. Get the *next* state information
            # Add a small delay if sim needs real time (usually not in ML training sims)
            # time.sleep(SIMULATION_DT) # Only if your sim doesn't run in fixed steps internally

            next_info = self.sim_proxy.get_drones_info()

            # 4. Process transitions, calculate rewards, and update agents
            # Iterate through drones to get rewards and done flags for the step that just happened
            for i in range(self.num_drones):
                # Only process transitions for drones that were active *before* this step
                if self._prev_step_data[i].get("was_active", False):
                    drone_info = current_info.drones[i]  # State *before* action
                    next_drone_info = next_info.drones[i]  # State *after* action
                    target = self._current_targets[i]
                    agent = self.agents[i]

                    # Calculate reward and done for this specific drone
                    reward, done, status = self._calculate_reward_and_done(
                        next_drone_info, target, self._step_count
                    )

                    # Accumulate episode reward
                    episode_rewards_this_ep[i] += reward

                    # Store the full transition for the agent's buffer
                    # Retrieve the pre-step data
                    state_vector = self._prev_step_data[i]["state_vector"]
                    action = self._prev_step_data[i]["action"]
                    log_prob = self._prev_step_data[i]["log_prob"]
                    next_state_vector = agent.prepare_state_vector(
                        next_drone_info, target
                    )  # Prepare next state vector

                    # Store transition in the agent's buffer
                    agent.store_transition(
                        state_vector, action, reward, next_state_vector, done, log_prob
                    )

                    # Update drone's done status
                    if done:
                        drones_done_status[i] = True
                        # print(f"  Drone {i} finished at step {self._step_count}: Status={status}, Reward={episode_rewards_this_ep[i]:.2f}")

            # Update current info for the next iteration of the while loop
            current_info = next_info

            # Check if any agent buffer is full and perform update
            # Updating multiple agents in one simulation step can be done here.
            # If batch_size is reached for an agent, update its model.
            for i, agent in enumerate(self.agents):
                # Check buffer size and ensure there are enough experiences for a meaningful update
                # PPO updates are typically done on batches of experience collected over several steps/episodes.
                # Updating *within* the episode after batch_size is reached is a common strategy.
                if len(agent.buffer) >= agent.batch_size:
                    # print(f"Agent {i} buffer full ({len(agent.buffer)}). Updating models...")
                    agent.update_models()

        # --- End of Episode ---
        # When the while loop finishes (all drones done or max steps reached)
        # Add total episode reward to tracking deque for each drone
        for i in range(self.num_drones):
            self.episode_rewards[i].append(episode_rewards_this_ep[i])
            # If a drone didn't finish (e.g., others finished or timeout), make sure its last transition is marked as done
            # This is handled if _calculate_reward_and_done sets done=True on timeout or crash

        return episode_steps  # Return how many steps the episode took

    def _get_actions_for_drones(
        self, info: DronesInfo, targets: list[Vector]
    ) -> list[list[float]]:
        """Calculates motor commands for all drones using their respective agents."""
        all_engines = []
        # Assuming info.drones is a list of DroneInfo, ordered by drone index (0 to 4)
        for i in range(self.num_drones):
            drone_info = info.drones[i]
            target = targets[i]
            agent = self.agents[i]  # Get the specific agent for this drone

            # Get action from the agent (with exploration during training)
            # The method get_action_and_log_prob also stores log_prob if needed for PPO
            # We store the action and log_prob in _prev_step_data *before* the sim step
            # Use explore=True during training
            # action, log_prob = agent.get_action_and_log_prob(agent.prepare_state_vector(drone_info, target), explore=True)
            # The get_action_and_log_prob method in PPOAgent should return action and log_prob
            # and not store the transition itself. Storage happens *after* getting reward/next_state.

            # So, here we get action and log_prob, and *store* them in _prev_step_data
            state_vector = agent.prepare_state_vector(drone_info, target)
            action, log_prob = agent.get_action_and_log_prob(state_vector, explore=True)

            all_engines.append(action)
            # Store pre-step info for processing the transition *after* sim step
            # This is handled directly in the main loop now

        return all_engines

    def _calculate_reward_and_done(
        self, drone_info: DroneInfo, target: Vector, current_step: int
    ):
        """Calculates reward and done status for a single drone."""
        pos = drone_info.possition
        is_alive = drone_info.is_alive

        # Check if target is reached
        dist_xz = pos.distance_xz_to(target)
        is_above_target_y = pos.y >= target.y
        target_reached = dist_xz < TARGET_DETECTION_RADIUS and is_above_target_y

        # Check termination conditions
        done = False
        status = "flying"
        reward = 0.0

        if not is_alive:
            # Crash penalty (sim reported not alive)
            reward = -REWARD_PENALTY_CRASH
            done = True
            status = "crashed"
        elif target_reached:
            # Target reached bonus
            reward = REWARD_BONUS_REACH
            done = True
            status = "reached_target"
        elif current_step >= self._max_steps_per_episode:
            # Timeout penalty (or just end episode without extra penalty if desired)
            # Let's add a small penalty on timeout if target wasn't reached
            if not target_reached:  # Should be true if it's a timeout and not reached
                reward = (
                    -REWARD_PENALTY_CRASH * 0.1
                )  # Smaller penalty for timeout vs crash
            done = True
            status = "timeout"
        else:
            # Step reward (progress towards goal + time penalty)
            reward_distance_xz = -REWARD_COEFF_DISTANCE_XZ * dist_xz
            reward_height_penalty = -REWARD_COEFF_HEIGHT_PENALTY * max(
                0, target.y - pos.y
            )
            reward_time_penalty = -REWARD_COEFF_TIME

            reward = reward_distance_xz + reward_height_penalty + reward_time_penalty
            status = "flying"  # Keep status as flying if not terminated

        return reward, done, status

    def _stop_episod(self, info: DronesInfo, targets: list[Vector]) -> bool:
        """
        Checks if the episode should stop.
        Episode stops if ALL drones are either crashed or at their target.
        Also stops if max steps per episode is reached (handled in learn_episod loop).
        """
        all_drones_done = True
        # Assuming info.drones is a list of DroneInfo, ordered by drone index
        for i in range(self.num_drones):
            drone_info = info.drones[i]
            target = targets[i]

            if drone_info.is_alive:  # If the drone is still alive
                # Check if it has reached its target
                dist_xz = drone_info.possition.distance_xz_to(target)
                is_above_target_y = drone_info.possition.y >= target.y
                target_reached = dist_xz < TARGET_DETECTION_RADIUS and is_above_target_y

                if not target_reached:
                    # If drone is alive and hasn't reached its target, episode is NOT over yet
                    all_drones_done = False
                    break  # No need to check other drones

        # The timeout check is handled in the calling learn_episod loop
        return all_drones_done  # True if all are crashed or reached target

    def _get_random_targets(self, arena_bounds: tuple[float]) -> list[Vector]:
        """Generates random targets within arena bounds."""
        min_b, max_b = (
            Vector(arena_bounds[0], arena_bounds[2], arena_bounds[4]),
            Vector(arena_bounds[1], arena_bounds[3], arena_bounds[5]),
        )
        targets = []
        for _ in range(self.num_drones):
            target = Vector(
                random.uniform(min_b.x, max_b.x),
                random.uniform(
                    min_b.y + 5.0, max_b.y - 1.0
                ),  # Target height well above ground, below ceiling
                random.uniform(min_b.z, max_b.z),
            )
            # Optional: Ensure targets are not too close to BASE_POSITION or each other
            while target.distance_to(BASE_POSITION) < 20:  # Min 20m from base
                target = Vector(
                    random.uniform(min_b.x, max_b.x),
                    random.uniform(min_b.y + 5.0, max_b.y - 1.0),
                    random.uniform(min_b.z, max_b.z),
                )
            targets.append(target)
        return targets

    def save(self, episode_num: int):
        """Saves the weights of all agents."""
        print(f"Saving model weights at episode {episode_num}...")
        for i, agent in enumerate(self.agents):
            agent.save_weights(agent.save_dir)
        print("All agents' weights saved.")

    def log(self, episode_num: int, total_steps: int, episode_duration: float):
        """Logs training progress."""
        print(f"\n--- Episode {episode_num} ---")
        print(f"Total Steps: {total_steps}")
        print(f"Episode Duration: {episode_duration:.2f} seconds")

        for i in range(self.num_drones):
            avg_reward = (
                np.mean(self.episode_rewards[i]) if self.episode_rewards[i] else 0.0
            )
            print(
                f"  Drone {i}: Recent Avg Episode Reward ({len(self.episode_rewards[i])} eps): {avg_reward:.2f}"
            )

        # You might want to log other metrics, like success rate, crash rate etc.
        # This requires tracking episode outcomes (_stop_episod status) per drone.
        # You could add counters for reached_target, crashed, timeout in _calculate_reward_and_done
        # and display them here.


# --- Main Execution ---
if __name__ == "__main__":
    # Instantiate your actual simulation proxy
    # Ensure this connects to your sim and provides required methods:
    # sim.reset() -> Resets all drones, environment etc.
    # sim.get_drones_info() -> Returns DronesInfo (list of DroneInfo for 5 drones)
    # sim.set_drones(list[list[float]]) -> Takes list of 8 motor values for each of 5 drones
    # Note: The actual simulation needs to handle the physics step after set_drones.
    # The next get_drones_info() should reflect the state *after* that physics step.

    try:
        # Example of how you might instantiate your sim proxy
        print("Connecting to simulation...")
        s = Simulation()
        # s.connect_to_server() # Uncomment if your sim requires explicit connection
        print("Simulation connected.")

        # Instantiate the learning manager
        l = LearnSimulation(s, num_drones=5)

        # Start the training process
        l.start_learning(
            episodes=10000, save_interval=500, log_interval=50
        )  # Adjusted intervals for potentially longer training

    except Exception as e:
        print(f"\nAn error occurred during simulation or training: {e}")
        import traceback

        traceback.print_exc()