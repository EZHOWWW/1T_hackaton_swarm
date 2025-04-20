# learn.py
import numpy as np
import tensorflow as tf
import random
from collections import deque
from solution.geometry import Vector
from solution.simulation import DroneInfo
from solution.executor import create_actor_model, create_critic_model, MAX_LIDAR_RANGE
from solution.RL.model import PPOAgent
from solution.simulation import Simulation, DronesInfo

# --- Simulation Parameters ---
# REPLACE WITH YOUR ACTUAL SIMULATION BOUNDS AND DT
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


class LearnSimulation:
    def __init__(self, sim_proxy: Simulation):
        self.sim_proxy = sim_proxy

        self.agents = None  # 5 agents

    def start_learning(
        self, episodes: int = 1000, save_interval: int = 10, log_interval: int = 10
    ):
        """main function to start learning"""
        for ep in range(episodes):
            self.learn_episod()

            if ep % save_interval == 0:
                self.save()
            if ep % log_interval == 0:
                self.log()

    def learn_episod(self):
        """one learn episod"""
        stop = False
        # targets = 5 random vectors
        targets = get_random_targets(ARENA_BOUNDS)
        while not stop:
            info = self.sim.get_drones_info()
            eng = self.step_in_episod(info, targets)
            self.sim.set_drones(eng)
            stop = self.stop_episod()

    def step_in_episod(
        self, info: list[DronesInfo], targets: list[Vector]
    ) -> list[float]:
        """here is one step of simulation and learn, return engines"""
        pass

    def stop_episod(self, info: list[DronesInfo], targets: list[Vector]) -> bool:
        # True if all drones is crushed or ib target
        pass

    def save(self):
        pass

    def log(self):
        pass


def get_random_targets(arena_bounds: tuple[float]) -> list[Vector]:
    min_b, max_b = (
        Vector(arena_bounds[0], arena_bounds[2], arena_bounds[4]),
        Vector(arena_bounds[1], arena_bounds[3], arena_bounds[5]),
    )
    targets = [
        Vector(
            random.uniform(min_b.x, max_b.x),
            random.uniform(
                min_b.y, max_b.y
            ),  # Target height well above ground, below ceiling
            random.uniform(min_b.z, max_b.z),
        )
        for _ in range(5)
    ]
    return targets


# --- Mock Simulation Environment ---
# REPLACE THIS ENTIRE CLASS WITH YOUR ACTUAL SIMULATION INTERFACE
class MockSimulation:
    def __init__(self, sim, arena_bounds, dt, base_position, target_detection_radius):
        self.arena_bounds = arena_bounds
        self.dt = dt
        self.base_position = base_position
        self.target_detection_radius = target_detection_radius

        self._state = None
        self._target = None
        self._steps_in_episode = 0
        self._max_steps_per_episode = 1000  # Increased max steps for larger arena
        self.sim = sim

    def reset(self):
        """Resets the simulation for a new episode with fixed base start and random target."""
        self._steps_in_episode = 0
        self.sim.reset()

        # Start at the base position
        self._state = DroneInfo(
            id=0,
            possition=Vector(
                self.base_position.x, self.base_position.y, self.base_position.z
            ),
            velocity=Vector(),
            angle=(0.0, 0.0, 0.0),
            angular_velocity=(0.0, 0.0, 0.0),
            lidars={"lidars": [0.0] * 10},  # Assume initially clear view
            is_alive=True,
        )

        # Set random target position within arena bounds (avoid too close to min_y)
        min_b, max_b = (
            Vector(self.arena_bounds[0], self.arena_bounds[2], self.arena_bounds[4]),
            Vector(self.arena_bounds[1], self.arena_bounds[3], self.arena_bounds[5]),
        )
        self._target = Vector(
            random.uniform(min_b.x, max_b.x),
            random.uniform(
                min_b.y + 5.0, max_b.y - 1.0
            ),  # Target height well above ground, below ceiling
            random.uniform(min_b.z, max_b.z),
        )

        # Ensure target is not too close to the start position if needed
        while (
            self._state.possition.distance_to(self._target) < 10
        ):  # Min 10m distance from start
            self._target = Vector(
                random.uniform(min_b.x, max_b.x),
                random.uniform(min_b.y + 5.0, max_b.y - 1.0),
                random.uniform(min_b.z, max_b.z),
            )

        print(
            f"Episode started. Start: {self._state.possition}, Target: {self._target}"
        )

        # In a real simulation, you'd pass start_pos and target_pos to the sim API
        # and get the initial state back.
        return self.get_state(), self.get_target()

    def step(self, motor_powers: list[float]):
        """
        Applies motor actions and advances the simulation by dt.
        Returns next_state, reward, done, info.
        REPLACE WITH YOUR SIMULATION'S STEP FUNCTION
        """
        if not self._state.is_alive:
            # If drone is already dead, just return current state and 0 reward
            return self._state, 0.0, True, {"status": "dead"}

        self._steps_in_episode += 1

        # Clamp motor powers to [0, 1]
        motor_powers = [max(0.0, min(1.0, p)) for p in motor_powers]

        # --- Simple Mock Physics Update ---
        # Total upward thrust (sum of motor powers * scale)
        total_thrust_force_magnitude = sum(motor_powers) * self.THRUST_SCALE
        net_force_y = total_thrust_force_magnitude + self.GRAVITY.y

        net_force = Vector(
            -self.DRAG_COEFF * self._state.velocity.x,  # Simple drag model
            net_force_y - self.DRAG_COEFF * self._state.velocity.y,
            -self.DRAG_COEFF * self._state.velocity.z,
        )

        # Update velocity and position (Euler integration)
        acceleration = Vector(
            net_force.x, net_force.y, net_force.z
        )  # Assuming mass = 1 for simplicity
        self._state.velocity = Vector(
            self._state.velocity.x + acceleration.x * self.dt,
            self._state.velocity.y + acceleration.y * self.dt,
            self._state.velocity.z + self._state.velocity.z * self.dt,
        )
        self._state.possition = Vector(
            self._state.possition.x + self._state.velocity.x * self.dt,
            self._state.possition.y + self._state.velocity.y * self.dt,
            self._state.possition.z + self._state.velocity.z * self.dt,
        )

        # --- Update Lidars (Mock) ---
        # This needs proper ray casting in a real sim.
        # Mock: simplified check against obstacles and arena bounds.
        # Return 0 if clear up to MAX_LIDAR_RANGE, otherwise distance <= MAX_LIDAR_RANGE
        updated_lidars = [0.0] * 10  # Assume clear initially

        # Simple check for distance to nearest point on obstacle bounding box
        # This is a simplification; real lidars are rays from origin/orientation.
        current_pos = self._state.possition
        for (
            obs_center_x,
            obs_center_y,
            obs_center_z,
            size_x,
            size_y,
            size_z,
        ) in self._obstacles:
            min_obs = Vector(
                obs_center_x - size_x / 2,
                obs_center_y - size_y / 2,
                obs_center_z - size_z / 2,
            )
            max_obs = Vector(
                obs_center_x + size_x / 2,
                obs_center_y + size_y / 2,
                obs_center_z + size_z / 2,
            )

            # Find closest point on AABB to drone position (simplified)
            closest_point_on_obs = Vector(
                max(min_obs.x, min(max_obs.x, current_pos.x)),
                max(min_obs.y, min(max_obs.y, current_pos.y)),
                max(min_obs.z, min(max_obs.z, current_pos.z)),
            )
            dist_to_closest = (
                current_pos.distance_to(closest_point_on_obs) - 0.5
            )  # Subtract drone radius approx

            # If within range, set lidar values (very crude approximation)
            if dist_to_closest < MAX_LIDAR_RANGE and dist_to_closest > 0:
                # This doesn't simulate directionality. A real sim would cast rays.
                # Just setting all lidars to this distance if close to *any* obstacle.
                # This is a major mock simplification.
                updated_lidars = [
                    min(d if d > 0 else MAX_LIDAR_RANGE, dist_to_closest)
                    for d in updated_lidars
                ]

        # Check distance to walls/floor/ceiling
        dist_to_floor = current_pos.y - self.arena_bounds[2]  # current_y - min_y
        dist_to_ceiling = self.arena_bounds[3] - current_pos.y  # max_y - current_y
        dist_to_wall_x_min = current_pos.x - self.arena_bounds[0]  # current_x - min_x
        dist_to_wall_x_max = self.arena_bounds[1] - current_pos.x  # max_x - current_x
        dist_to_wall_z_min = current_pos.z - self.arena_bounds[4]  # current_z - min_z
        dist_to_wall_z_max = self.arena_bounds[5] - current_pos.z  # max_z - current_z

        # Mock lidar directions (very rough map to indices)
        # 0-7: horizontal, 8: up, 9: down
        # Update relevant lidar readings if closer than current (or 0) and within MAX_LIDAR_RANGE
        if dist_to_ceiling < MAX_LIDAR_RANGE and dist_to_ceiling > 0:
            updated_lidars[8] = min(
                updated_lidars[8] if updated_lidars[8] > 0 else MAX_LIDAR_RANGE,
                dist_to_ceiling,
            )
        if dist_to_floor < MAX_LIDAR_RANGE and dist_to_floor > 0:
            updated_lidars[9] = min(
                updated_lidars[9] if updated_lidars[9] > 0 else MAX_LIDAR_RANGE,
                dist_to_floor,
            )
        # Horizontal - need to map actual lidar angles to XZ directions
        # This requires drone's yaw angle. Mock simplified: check nearest wall distances
        # and assign to 'general' horizontal lidars if closer than 10.
        horizontal_dists = [
            dist_to_wall_x_min,
            dist_to_wall_x_max,
            dist_to_wall_z_min,
            dist_to_wall_z_max,
        ]
        for i in range(8):  # Horizontal lidars 0-7
            closest_wall_dist = min(horizontal_dists)  # Still not directional
            if closest_wall_dist < MAX_LIDAR_RANGE and closest_wall_dist > 0:
                updated_lidars[i] = min(
                    updated_lidars[i] if updated_lidars[i] > 0 else MAX_LIDAR_RANGE,
                    closest_wall_dist,
                )

        # Final processing: any lidar still 0 means clear up to max range (as per user info)
        # We already initialized to 0.0, and updated only if < MAX_LIDAR_RANGE and > 0.
        # This matches the rule: 0 if clear, otherwise distance up to 10.

        self._state.lidars = {"lidars": updated_lidars}

        # --- Check for Termination Conditions ---
        done = False
        reward = 0.0
        status = "flying"

        # 1. Check if drone is marked not alive by the *actual* simulation
        if not self._state.is_alive:
            reward = -5000.0  # Very large penalty for crash/death
            done = True
            status = "crashed_sim_alive_flag"
            # print(f"Crashed: Sim reported dead at {self._state.possition}") # Keep logging low during training

        # 2. Check for collision with arena bounds (if sim doesn't handle this via is_alive)
        # Assuming sim handles this and sets is_alive, but as a fallback:
        pos = self._state.possition
        if (
            pos.x < self.arena_bounds[0]
            or pos.x > self.arena_bounds[1]
            or pos.y < self.arena_bounds[2]
            or pos.y > self.arena_bounds[3]
            or pos.z < self.arena_bounds[4]
            or pos.z > self.arena_bounds[5]
        ):
            if (
                self._state.is_alive
            ):  # Only apply penalty if not already dead by sim flag
                self._state.is_alive = False
                reward = -5000.0  # Very large penalty
                done = True
                status = "crashed_bounds_fallback"
                # print(f"Crashed: Out of bounds at {self._state.possition}")

        # 3. Check if target is reached (only if not crashed)
        dist_xz = self._state.possition.distance_xz_to(self._target)
        is_above_target_y = self._state.possition.y >= self._target.y

        if not done and dist_xz < self.target_detection_radius and is_above_target_y:
            reward = +2000.0  # Positive reward for reaching (scaled for larger arena)
            done = True
            status = "reached_target"
            # print(f"Reached target at {self._target} at step {self._steps_in_episode}")

        # 4. Check for maximum steps
        if not done and self._steps_in_episode >= self._max_steps_per_episode:
            done = True
            status = "timeout"
            # print("Episode timed out")

        # --- Calculate Step Reward (if not terminated by crash or reach) ---
        if not done:
            # Reward components (adjust coefficients for 200x20x200 arena)
            # Max XZ distance ~ sqrt(200^2 + 200^2) = sqrt(80000) ~ 280
            # Max Y below target ~ 20
            alpha = 0.5  # Penalty for XZ distance (was 1.0)
            beta = 5.0  # Penalty for being below target Y (was 2.0)
            delta = 0.05  # Time penalty per step (was 0.1)

            # Scale distance penalty by arena diagonal for some normalization idea
            max_arena_dist = Vector(
                self.arena_bounds[1], self.arena_bounds[3], self.arena_bounds[5]
            ).distance_to(
                Vector(self.arena_bounds[0], self.arena_bounds[2], self.arena_bounds[4])
            )
            # reward_distance_xz = -alpha * (dist_xz / max_arena_dist) # Alternative normalization
            reward_distance_xz = -alpha * dist_xz  # Simple linear penalty

            reward_height_penalty = -beta * max(
                0, self._target.y - self._state.possition.y
            )
            reward_time_penalty = -delta

            reward = reward_distance_xz + reward_height_penalty + reward_time_penalty

        # In a real sim, this would come from the sim API: next_state, reward, done, info
        return self._state, reward, done, {"status": status}

    def get_state(self):
        return self._state

    def get_target(self):
        return self._target

    def get_state_size(self):
        return 3 + 3 + 3 + 3 + 10 + 3 + 1

    def get_action_size(self):
        return 8


# --- PPO Specific Components ---
# Using a simplified Actor that outputs mean only and adding noise manually for exploration.
# For a proper PPO implementation with continuous actions, the Actor should output
# mean and stddev, and actions should be sampled from the resulting distribution.
# Log probability calculation is crucial and complex with action clipping.
# The code below uses a simplified log_prob calculation that might be inaccurate
# when actions are clipped. Consider using a library like TF-Agents or Stable-Baselines3
# for production-level PPO.


# --- Training Loop ---
if __name__ == "__main__":
    # Define training parameters
    TOTAL_EPISODES = 10000  # Increased episodes for larger arena
    SAVE_INTERVAL = 200  # Save weights every N episodes
    LOG_INTERVAL = 10  # Print log every N episodes

    # Create the RL agent
    agent = PPOAgent(
        state_size=3
        + 3
        + 3
        + 3
        + 10
        + 3
        + 1,  # pos, vel, angle, ang_vel, lidars, target_vec, dist
        action_size=8,  # motor powers
        arena_bounds=ARENA_BOUNDS,
        dt=SIMULATION_DT,
        base_position=BASE_POSITION,
    )

    # Start training
    print("Starting training...")
    episode_rewards = deque(maxlen=100)  # To track recent average reward

    for episode in range(1, TOTAL_EPISODES + 1):
        # Reset environment for a new episode (fixed start, random target)
        current_state, target = agent.sim.reset()
        done = False
        episode_reward = 0
        step_count = 0

        while not done:
            # Prepare state vector for the model
            state_vector = agent.prepare_state_vector(current_state, target)

            # Select action using the agent's policy (with exploration) and get log_prob
            action, log_prob = agent.get_action_and_log_prob(state_vector, explore=True)

            # Perform the action in the simulation
            # REPLACE agent.sim.step() WITH YOUR ACTUAL SIMULATION'S STEP FUNCTION
            next_state, reward, done, info = agent.sim.step(action)

            # Prepare next state vector for buffer
            next_state_vector = agent.prepare_state_vector(next_state, target)

            # Store the transition in the agent's buffer
            agent.store_transition(
                state_vector, action, reward, next_state_vector, done, log_prob
            )

            current_state = next_state
            episode_reward += reward
            step_count += 1

            # If buffer is full or episode ends, perform a model update
            # PPO often updates after collecting a fixed number of steps across multiple environments
            # In a single-environment setup, update when buffer reaches batch_size or episode ends
            if len(agent.buffer) >= agent.batch_size or done:
                if (
                    len(agent.buffer) >= 10
                ):  # Only update if buffer has some data, even if episode ends early
                    agent.update_models()

        # Log episode results
        episode_rewards.append(episode_reward)
        average_reward = np.mean(episode_rewards)

        if episode % LOG_INTERVAL == 0:
            print(
                f"Episode {episode}/{TOTAL_EPISODES} | Avg Reward: {average_reward:.2f} | Last Reward: {episode_reward:.2f} | Steps: {step_count} | Status: {info.get('status', 'unknown')}"
            )

        # Save model weights periodically
        if episode % SAVE_INTERVAL == 0:
            agent.save_weights(agent.save_dir)
        break

    print("Training finished.")
    # Save final weights
    agent.save_weights(agent.save_dir)
