# learn.py
import numpy as np
import tensorflow as tf
from collections import deque
import random
import os
import time

# Assuming these are correctly defined and importable
from solution.geometry import Vector
from solution.simulation import (
    Simulation,
    DroneInfo,
    DronesInfo,  # Assuming DronesInfo is list[DroneInfo]
)

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
)
SIMULATION_DT = 0.1
BASE_POSITION = Vector(-77.01, 0.48, 75.31)
TARGET_DETECTION_RADIUS = 2.0


# --- RL Reward Coefficients ---
REWARD_COEFF_DISTANCE_XZ = 0.5
REWARD_COEFF_HEIGHT_PENALTY = 5.0
REWARD_COEFF_TIME = 0.05
REWARD_BONUS_REACH = 2000.0
REWARD_PENALTY_CRASH = 5000.0


# --- Training Parameters ---
STATE_SIZE = 3 + 3 + 3 + 3 + 10 + 3 + 1
ACTION_SIZE = 8
BATCH_SIZE = 256
PPO_EPOCHS = 10


class LearnSimulation:
    def __init__(self, sim_proxy: Simulation, num_drones: int = 5):
        self.sim_proxy = sim_proxy
        self.num_drones = num_drones

        self.agents: list[PPOAgent] = []
        for i in range(self.num_drones):
            agent = PPOAgent(
                state_size=STATE_SIZE,
                action_size=ACTION_SIZE,
                arena_bounds=ARENA_BOUNDS,
                dt=SIMULATION_DT,
                base_position=BASE_POSITION,
                target_detection_radius=TARGET_DETECTION_RADIUS,
                ppo_epochs=PPO_EPOCHS,
                batch_size=BATCH_SIZE,
                save_dir=f"./drone_rl_models/agent_{i}",
            )
            self.agents.append(agent)

        self.episode_rewards: list[deque] = [
            deque(maxlen=100) for _ in range(self.num_drones)
        ]
        self._current_targets: list[Vector] = []
        self._step_count = 0
        self._max_steps_per_episode = 10000000

        # Temporary storage for data from the current step before the simulation advances
        self._current_step_data: list[dict] = [{} for _ in range(self.num_drones)]

    def start_learning(
        self, episodes: int = 10000, save_interval: int = 200, log_interval: int = 10
    ):
        print(
            f"Starting training for {self.num_drones} drones over {episodes} episodes..."
        )

        for i, agent in enumerate(self.agents):
            agent.load_weights(agent.save_dir)

        total_steps = 0
        for ep in range(1, episodes + 1):
            episode_start_time = time.time()
            episode_steps = self.learn_episod()
            total_steps += episode_steps
            episode_end_time = time.time()

            if ep % log_interval == 0:
                self.log(ep, total_steps, episode_end_time - episode_start_time)

            if ep % save_interval == 0:
                self.save(ep)

        print("\nTraining finished.")
        self.save(episodes)

    def learn_episod(self):
        """Runs one learning episode for all drones."""
        self.sim_proxy.reset()
        # time.sleep(3)
        current_info = self.sim_proxy.get_drones_info()

        # Targets are fireplace positions
        self._current_targets = self._get_random_targets()

        self._step_count = 0
        # Initialize current step data storage
        self._current_step_data = [{} for _ in range(self.num_drones)]
        episode_rewards_this_ep = [0.0] * self.num_drones
        drones_done_status = [False] * self.num_drones

        while (
            not all(drones_done_status)
            and self._step_count < self._max_steps_per_episode
        ):
            self._step_count += 1

            # 1. Get actions and log_probs for the *current* state of each drone
            all_engines = self._get_actions_for_drones(
                current_info, self._current_targets, drones_done_status
            )

            # 2. Apply actions to the simulation
            self.sim_proxy.set_drones(all_engines)

            # 3. The simulation advances. Get the *next* state information
            next_info = self.sim_proxy.get_drones_info()
            print(next_info)

            # 4. Process transitions, calculate rewards, and update agents
            # Iterate through drones to get rewards and done flags for the step that just happened
            for i in range(self.num_drones):
                # Only process transitions for drones that were active *before* this step
                # Check if 'state_vector' exists in _current_step_data[i] meaning drone was active
                if "state_vector" in self._current_step_data[i]:
                    # Retrieve data from *before* the sim step
                    state_vector = self._current_step_data[i]["state_vector"]
                    action = self._current_step_data[i]["action"]
                    log_prob = self._current_step_data[i]["log_prob"]

                    next_drone_info = next_info[i]  # State *after* action
                    target = self._current_targets[i]
                    agent = self.agents[i]

                    # Calculate reward and done for this specific drone based on next_state
                    reward, done, status = self._calculate_reward_and_done(
                        next_drone_info, target, self._step_count
                    )

                    # Accumulate episode reward
                    episode_rewards_this_ep[i] += reward

                    # Prepare next state vector
                    next_state_vector = agent.prepare_state_vector(
                        next_drone_info, target
                    )

                    # Store full transition in the agent's buffer
                    agent.store_transition(
                        state_vector, action, reward, next_state_vector, done, log_prob
                    )

                    # Update drone's overall done status for the episode
                    if done:
                        drones_done_status[i] = True
                        # Optional: log status if a drone just finished
                        # print(f"  Drone {i} finished at step {self._step_count}: Status={status}, Reward={episode_rewards_this_ep[i]:.2f}")
            # print(drones_done_status, self._step_count, self._max_steps_per_episode)

            # Update current info for the next iteration of the while loop
            current_info = next_info

            # Check if any agent buffer is full and perform update
            for i, agent in enumerate(self.agents):
                if len(agent.buffer) >= agent.batch_size:
                    agent.update_models()

        # --- End of Episode ---
        # Add total episode reward to tracking deque for each drone
        for i in range(self.num_drones):
            self.episode_rewards[i].append(episode_rewards_this_ep[i])

        return self._step_count  # Return how many steps the episode took

    def _get_actions_for_drones(
        self, info: DronesInfo, targets: list[Vector], drones_done_status: list[bool]
    ) -> list[list[float]]:
        """
        Calculates motor commands for all drones using their respective agents.
        Stores pre-step state, action, log_prob for later transition processing.
        """
        all_engines = []
        # Assuming info is a list of DroneInfo, ordered by drone index (0 to 4)
        for i in range(self.num_drones):
            # Get the specific agent for this drone
            agent = self.agents[i]

            if not drones_done_status[i]:  # Only get action for active drones
                drone_info = info[i]  # Assuming info is DronesInfo with .drones list
                target = targets[i]

                # Prepare state vector
                state_vector = agent.prepare_state_vector(drone_info, target)

                # Get action and log_prob from the agent (with exploration)
                action, log_prob = agent.get_action_and_log_prob(
                    state_vector, explore=True
                )

                # Store pre-step data needed for the transition
                self._current_step_data[i]["state_vector"] = state_vector
                self._current_step_data[i]["action"] = action
                self._current_step_data[i]["log_prob"] = log_prob

                all_engines.append(action)
            else:
                # If drone is already done, send zero engines
                all_engines.append([0.0] * ACTION_SIZE)
                # Clear data for done drones to avoid processing old transitions
                self._current_step_data[i] = {}  # Mark as inactive for this step

        # Ensure all_engines always has the correct number of entries (5)
        while len(all_engines) < self.num_drones:
            all_engines.append([0.0] * ACTION_SIZE)

        return all_engines

    def _process_step_transitions(
        self,
        current_info: DronesInfo,
        actions: list[list[float]],
        next_info: DronesInfo,
        targets: list[Vector],
    ):
        """
        Processes the results of one simulation step for all drones.
        Calculates rewards and done status, and stores transitions in agent buffers.
        """
        # Iterate through drones to get rewards and done flags for the step that just happened
        for i in range(self.num_drones):
            # Only process transitions for drones that were active *before* this step
            if "state_vector" in self._current_step_data[i]:
                # Retrieve data from *before* the sim step
                state_vector = self._current_step_data[i]["state_vector"]
                action = self._current_step_data[i]["action"]
                log_prob = self._current_step_data[i]["log_prob"]

                next_drone_info = next_info[i]  # State *after* action
                target = targets[i]
                agent = self.agents[i]

                # Calculate reward and done for this specific drone based on next_state
                reward, done, status = self._calculate_reward_and_done(
                    next_drone_info, target, self._step_count
                )

                # Prepare next state vector
                next_state_vector = agent.prepare_state_vector(next_drone_info, target)

                # Store full transition in the agent's buffer
                agent.store_transition(
                    state_vector, action, reward, next_state_vector, done, log_prob
                )

                # Clear the data for this step after processing
                # This is done in _get_actions_for_drones now if drone is done.
                # If not done, the slot remains for the next step's pre-data.
                # If done *in this step*, the 'done' flag in the transition handles it.
                # The _current_step_data[i] will be overwritten in the next call to _get_actions_for_drones
                # or cleared if the drone is marked done.
                # Let's ensure it's explicitly cleared if the drone becomes done in this step.
                # This is handled by the 'drones_done_status' check in _get_actions_for_drones next iteration.
                pass  # No need to clear here if done status is tracked

    def _calculate_reward_and_done(
        self, drone_info: DroneInfo, target: Vector, current_step: int
    ):
        """Calculates reward and done status for a single drone."""
        pos = drone_info.possition
        is_alive = drone_info.is_alive

        dist_xz = pos.distance_xz_to(target)
        is_above_target_y = pos.y >= target.y
        target_reached = dist_xz < TARGET_DETECTION_RADIUS and is_above_target_y

        done = False
        status = "flying"
        reward = 0.0

        if not is_alive:
            reward = -REWARD_PENALTY_CRASH
            done = True
            status = "crashed"
        elif target_reached:
            reward = REWARD_BONUS_REACH
            done = True
            status = "reached_target"
        elif current_step >= self._max_steps_per_episode:
            if not target_reached:
                reward = -REWARD_PENALTY_CRASH * 0.1
            done = True
            status = "timeout"
        else:
            reward_distance_xz = -REWARD_COEFF_DISTANCE_XZ * dist_xz
            reward_height_penalty = -REWARD_COEFF_HEIGHT_PENALTY * max(
                0, target.y - pos.y
            )
            reward_time_penalty = -REWARD_COEFF_TIME

            reward = reward_distance_xz + reward_height_penalty + reward_time_penalty
            status = "flying"

        return reward, done, status

    def _stop_episod(self, info: DronesInfo, targets: list[Vector]) -> bool:
        """
        Checks if the episode should stop.
        Episode stops if ALL drones are either crashed or at their target.
        Max steps check is handled in learn_episod.
        """
        all_drones_done = True
        # Assuming info is list[DroneInfo]
        for i in range(self.num_drones):
            drone_info = info[i]
            target = targets[i]

            if drone_info.is_alive:
                dist_xz = drone_info.possition.distance_xz_to(target)
                is_above_target_y = drone_info.possition.y >= target.y
                target_reached = dist_xz < TARGET_DETECTION_RADIUS and is_above_target_y

                if not target_reached:
                    all_drones_done = False
                    break  # Found an active drone not at target

        return all_drones_done

    def _get_random_targets(self) -> list[Vector]:
        """Gets fireplace positions as targets from the simulation."""
        # Targets are now based on fireplaces provided by the simulation
        try:
            fireplaces = self.sim_proxy.get_fireplaces_info()[:5]
            if len(fireplaces) != self.num_drones:
                print(
                    f"Warning: Number of fireplaces ({len(fireplaces)}) does not match number of drones ({self.num_drones}). Using available fireplaces as targets."
                )
            # Use the positions of the fireplaces as targets
            targets = [
                fp.possition for fp in fireplaces[: self.num_drones]
            ]  # Take up to num_drones targets

            # If there are fewer fireplaces than drones, assign None or reuse targets for remaining drones
            while len(targets) < self.num_drones:
                print(
                    f"Warning: Not enough fireplaces for drone {len(targets)}. Reusing a target or assigning dummy."
                )
                # Option 1: Reuse a target (e.g., the last one)
                targets.append(
                    targets[-1] if targets else Vector(0, 1, 0)
                )  # Append last target or a dummy base target
                # Option 2: Assign a random target within bounds if no fireplaces
                # targets.append(Vector(random.uniform(ARENA_BOUNDS[0], ARENA_BOUNDS[1]), random.uniform(ARENA_BOUNDS[2]+5.0, ARENA_BOUNDS[3]-1.0), random.uniform(ARENA_BOUNDS[4], ARENA_BOUNDS[5])))

            return targets

        except Exception as e:
            print(f"Error getting fireplace info: {e}")
            # Fallback: generate random targets if fireplace info is unavailable
            print("Falling back to generating random targets within arena bounds.")
            min_b, max_b = (
                Vector(ARENA_BOUNDS[0], ARENA_BOUNDS[2], ARENA_BOUNDS[4]),
                Vector(ARENA_BOUNDS[1], ARENA_BOUNDS[3], ARENA_BOUNDS[5]),
            )
            targets = []
            for _ in range(self.num_drones):
                target = Vector(
                    random.uniform(min_b.x, max_b.x),
                    random.uniform(min_b.y + 5.0, max_b.y - 1.0),
                    random.uniform(min_b.z, max_b.z),
                )
                # Optional: Ensure targets are not too close to BASE_POSITION or each other
                while target.distance_to(BASE_POSITION) < 20:
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
            # Ensure directory exists for each agent
            os.makedirs(agent.save_dir, exist_ok=True)
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
            # Get the status of the drone in the last step if available
            last_step_status = self._calculate_reward_and_done(
                self.sim_proxy.get_drones_info()[i],
                self._current_targets[i],
                self._step_count,
            )[2]  # Get status without recalculating reward/done

            print(
                f"  Drone {i}: Recent Avg Episode Reward ({len(self.episode_rewards[i])} eps): {avg_reward:.2f} | Last Status: {last_step_status}"
            )


# --- Main Execution ---
async def main():
    try:
        print("Connecting to simulation...")
        s = Simulation()
        # Assuming connect_to_server is an async method
        await s.connect_to_server()
        print("Simulation connected.")

        s.get_fireplaces_info()
        # Instantiate the learning manager
        l = LearnSimulation(s, num_drones=5)

        # Start the training process
        l.start_learning(
            episodes=10000,
            save_interval=10,
            log_interval=1,  # Adjusted intervals
        )

    except Exception as e:
        print(f"\nAn error occurred during simulation or training: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
