"""Robotized complex navigation environment with v/w actions and footprint collision."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from filters.bounded_kalman_command_smoother import BoundedKalmanCommandSmoother


class RobotDiffDriveComplexEnv(gym.Env):
    """Fixed-map 2D differential-drive navigation with command smoothing."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        use_kf: bool = True,
        aug_prev_action: bool = False,
        seed: int | None = None,
        map_size: float = 80.0,
        start: tuple[float, float] = (6.0, 6.0),
        goal: tuple[float, float] = (74.0, 74.0),
        start_theta: float = 0.0,
        max_steps: int = 1000,
        max_linear_speed: float = 1.2,
        max_angular_speed: float = 1.5,
        dt: float = 1.0,
        goal_radius: float = 2.5,
        robot_radius: float = 0.0,
        obstacle_margin: float = 0.7,
        boundary_margin: float = 0.5,
        obstacle_safe_margin: float = 3.0,
        obstacle_penalty_weight: float = 0.9,
        heading_reward_weight: float = 0.1,
        progress_reward_weight: float = 4.0,
        distance_penalty_weight: float = 0.05,
        stall_penalty_weight: float = 0.2,
        stall_progress_threshold: float = 0.01,
        stall_speed_threshold: float = 0.05,
        timeout_distance_penalty_weight: float = 75.0,
    ) -> None:
        super().__init__()
        self.map_size = float(map_size)
        self.start = np.asarray(start, dtype=np.float32)
        self.goal = np.asarray(goal, dtype=np.float32)
        self.start_theta = float(start_theta)
        self.max_steps = int(max_steps)
        self.max_linear_speed = float(max_linear_speed)
        self.max_angular_speed = float(max_angular_speed)
        self.dt = float(dt)
        self.goal_radius = float(goal_radius)
        self.robot_radius = float(robot_radius)
        self.obstacle_margin = float(obstacle_margin)
        self.boundary_margin = float(boundary_margin)
        self.obstacle_safe_margin = float(obstacle_safe_margin)
        self.obstacle_penalty_weight = float(obstacle_penalty_weight)
        self.heading_reward_weight = float(heading_reward_weight)
        self.progress_reward_weight = float(progress_reward_weight)
        self.distance_penalty_weight = float(distance_penalty_weight)
        self.stall_penalty_weight = float(stall_penalty_weight)
        self.stall_progress_threshold = float(stall_progress_threshold)
        self.stall_speed_threshold = float(stall_speed_threshold)
        self.timeout_distance_penalty_weight = float(timeout_distance_penalty_weight)
        self.diagonal = float(np.sqrt(2.0) * self.map_size)
        self.use_kf = bool(use_kf)
        self.aug_prev_action = bool(aug_prev_action)
        self._initial_seed = seed
        self._has_reset = False

        self.rectangles = [
            {"min": np.asarray([18.0, 12.0], dtype=np.float32), "max": np.asarray([21.0, 50.0], dtype=np.float32)},
            {"min": np.asarray([21.0, 24.0], dtype=np.float32), "max": np.asarray([46.0, 27.0], dtype=np.float32)},
            {"min": np.asarray([39.0, 29.0], dtype=np.float32), "max": np.asarray([42.0, 64.0], dtype=np.float32)},
            {"min": np.asarray([58.0, 14.0], dtype=np.float32), "max": np.asarray([61.0, 48.0], dtype=np.float32)},
            {"min": np.asarray([42.0, 62.0], dtype=np.float32), "max": np.asarray([68.0, 65.0], dtype=np.float32)},
            {"min": np.asarray([9.0, 52.0], dtype=np.float32), "max": np.asarray([30.0, 55.0], dtype=np.float32)},
        ]
        self.circles = [
            {"center": np.asarray([30.0, 14.0], dtype=np.float32), "radius": 2.6},
            {"center": np.asarray([48.0, 40.0], dtype=np.float32), "radius": 3.0},
            {"center": np.asarray([68.0, 44.0], dtype=np.float32), "radius": 3.4},
            {"center": np.asarray([28.0, 68.0], dtype=np.float32), "radius": 3.4},
            {"center": np.asarray([14.0, 38.0], dtype=np.float32), "radius": 2.6},
        ]
        self.waypoints = [
            np.asarray([28.0, 10.0], dtype=np.float32),
            np.asarray([52.0, 20.0], dtype=np.float32),
            np.asarray([52.0, 56.0], dtype=np.float32),
            np.asarray([73.0, 58.0], dtype=np.float32),
            np.asarray([73.0, 72.0], dtype=np.float32),
            self.goal.copy(),
        ]

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.obs_dim = 5 + (2 if self.aug_prev_action else 0)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )
        self.command_low = np.asarray([0.0, -self.max_angular_speed], dtype=np.float32)
        self.command_high = np.asarray([self.max_linear_speed, self.max_angular_speed], dtype=np.float32)
        self.command_smoother = BoundedKalmanCommandSmoother(
            command_low=self.command_low,
            command_high=self.command_high,
            use_kf=self.use_kf,
        )

        self.position = self.start.copy()
        self.theta = self.start_theta
        self.step_count = 0
        self.prev_distance = self._distance_to_goal()
        self.prev_raw_command = np.zeros(2, dtype=np.float32)
        self.prev_exec_command = np.zeros(2, dtype=np.float32)
        self.prev_exec_delta = np.zeros(2, dtype=np.float32)

    def set_use_kf(self, use_kf: bool) -> None:
        self.use_kf = bool(use_kf)
        self.command_smoother.use_kf = self.use_kf
        self.command_smoother.reset()
        self.prev_exec_command = np.zeros(2, dtype=np.float32)
        self.prev_exec_delta = np.zeros(2, dtype=np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        actual_seed = seed
        if actual_seed is None and not self._has_reset:
            actual_seed = self._initial_seed
        super().reset(seed=actual_seed)
        self._has_reset = True

        self.position = self.start.astype(np.float32).copy()
        self.theta = self._wrap_to_pi(self.start_theta)
        self.step_count = 0
        self.command_smoother.reset()
        self.prev_distance = self._distance_to_goal()
        self.prev_raw_command = np.zeros(2, dtype=np.float32)
        self.prev_exec_command = np.zeros(2, dtype=np.float32)
        self.prev_exec_delta = np.zeros(2, dtype=np.float32)

        info = self._build_info(
            raw_command=np.zeros(2, dtype=np.float32),
            exec_command=np.zeros(2, dtype=np.float32),
            raw_delta=np.zeros(2, dtype=np.float32),
            exec_delta=np.zeros(2, dtype=np.float32),
            success=False,
            collision=False,
            out_of_bounds=False,
            timeout=False,
        )
        return self._get_obs(), info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        raw_action = self._clip_normalized_action(action)
        raw_command = self.normalized_action_to_command(raw_action)
        prev_raw_command = self.prev_raw_command.copy()
        prev_exec_command = self.prev_exec_command.copy()
        previous_distance = self._distance_to_goal()

        exec_command = self.command_smoother.smooth(raw_command)
        v_exec = float(exec_command[0])
        w_exec = float(exec_command[1])
        self.position = (
            self.position
            + np.asarray(
                [
                    v_exec * np.cos(self.theta) * self.dt,
                    v_exec * np.sin(self.theta) * self.dt,
                ],
                dtype=np.float32,
            )
        ).astype(np.float32)
        self.theta = self._wrap_to_pi(self.theta + w_exec * self.dt)
        self.step_count += 1

        current_distance = self._distance_to_goal()
        success = current_distance < self.goal_radius
        collision = self._is_collision()
        out_of_bounds = self._is_out_of_bounds()
        terminated = bool(success or collision or out_of_bounds)
        truncated = bool(self.step_count >= self.max_steps and not terminated)
        timeout = bool(truncated)

        goal_progress = previous_distance - current_distance
        progress_reward = self.progress_reward_weight * goal_progress
        distance_penalty = self.distance_penalty_weight * (current_distance / self.diagonal)
        obstacle_penalty, min_obstacle_clearance = self._obstacle_proximity_penalty()
        heading_reward = self._heading_reward()
        stall_penalty = self._anti_stall_penalty(goal_progress, v_exec, current_distance)
        timeout_distance_penalty = (
            self.timeout_distance_penalty_weight * (current_distance / self.diagonal)
            if timeout
            else 0.0
        )

        reward = (
            progress_reward
            + heading_reward
            - obstacle_penalty
            - distance_penalty
            - stall_penalty
            - timeout_distance_penalty
            - 0.01
        )
        if success:
            reward += 150.0
        if collision or out_of_bounds:
            reward -= 150.0

        raw_delta = raw_command - prev_raw_command
        exec_delta = exec_command - prev_exec_command
        self.prev_raw_command = raw_command.astype(np.float32)
        self.prev_exec_command = exec_command.astype(np.float32)
        self.prev_exec_delta = exec_delta.astype(np.float32)
        self.prev_distance = current_distance

        info = self._build_info(
            raw_command=raw_command,
            exec_command=exec_command,
            raw_delta=raw_delta,
            exec_delta=exec_delta,
            success=success,
            collision=collision,
            out_of_bounds=out_of_bounds,
            timeout=timeout,
            heading_reward=heading_reward,
            obstacle_penalty=obstacle_penalty,
            min_obstacle_clearance=min_obstacle_clearance,
            goal_progress=goal_progress,
            progress_reward=progress_reward,
            distance_penalty=distance_penalty,
            stall_penalty=stall_penalty,
            timeout_distance_penalty=timeout_distance_penalty,
        )
        return self._get_obs(), float(reward), terminated, truncated, info

    def normalized_action_to_command(self, action: np.ndarray) -> np.ndarray:
        raw_action = self._clip_normalized_action(action)
        return np.asarray(
            [
                (float(raw_action[0]) + 1.0) * 0.5 * self.max_linear_speed,
                float(raw_action[1]) * self.max_angular_speed,
            ],
            dtype=np.float32,
        )

    def command_to_normalized_action(self, command: np.ndarray) -> np.ndarray:
        command = np.clip(
            np.asarray(command, dtype=np.float32).reshape(2),
            self.command_low,
            self.command_high,
        )
        v_norm = 2.0 * float(command[0]) / max(self.max_linear_speed, 1e-8) - 1.0
        w_norm = float(command[1]) / max(self.max_angular_speed, 1e-8)
        return np.asarray([v_norm, w_norm], dtype=np.float32)

    def obstacle_clearance(self, point: np.ndarray) -> float:
        return float(self.signed_distance_to_obstacles(point) - self.obstacle_margin)

    def boundary_clearance(self, point: np.ndarray) -> float:
        point = np.asarray(point, dtype=np.float32).reshape(2)
        return float(
            min(
                point[0] - self.boundary_margin,
                self.map_size - self.boundary_margin - point[0],
                point[1] - self.boundary_margin,
                self.map_size - self.boundary_margin - point[1],
            )
        )

    def signed_distance_to_obstacles(self, point: np.ndarray) -> float:
        point = np.asarray(point, dtype=np.float32).reshape(2)
        distances = [self._circle_signed_distance(point, circle) for circle in self.circles]
        distances.extend(self._rectangle_signed_distance(point, rect) for rect in self.rectangles)
        return float(min(distances)) if distances else float("inf")

    def _get_obs(self) -> np.ndarray:
        dist_goal = np.asarray([self._distance_to_goal() / self.diagonal], dtype=np.float32)
        base_obs = [
            self.position.astype(np.float32) / self.map_size,
            np.asarray([np.sin(self.theta), np.cos(self.theta)], dtype=np.float32),
            dist_goal,
        ]
        if self.aug_prev_action:
            prev_exec_norm = np.asarray(
                [
                    self.prev_exec_command[0] / max(self.max_linear_speed, 1e-8),
                    self.prev_exec_command[1] / max(self.max_angular_speed, 1e-8),
                ],
                dtype=np.float32,
            )
            base_obs.append(prev_exec_norm)
        obs = np.concatenate(base_obs).astype(np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high).astype(np.float32)

    def _clip_normalized_action(self, action: np.ndarray) -> np.ndarray:
        return np.clip(
            np.asarray(action, dtype=np.float32).reshape(2),
            self.action_space.low,
            self.action_space.high,
        ).astype(np.float32)

    def _distance_to_goal(self) -> float:
        return float(np.linalg.norm(self.goal - self.position))

    def _obstacle_proximity_penalty(self) -> tuple[float, float]:
        min_clearance = self.obstacle_clearance(self.position)
        if min_clearance >= self.obstacle_safe_margin:
            return 0.0, min_clearance

        penetration = max(0.0, self.obstacle_safe_margin - min_clearance)
        normalized_penetration = penetration / max(self.obstacle_safe_margin, 1e-6)
        penalty = self.obstacle_penalty_weight * normalized_penetration
        return float(penalty), min_clearance

    def _heading_reward(self) -> float:
        goal_vec = self.goal - self.position
        goal_dist = float(np.linalg.norm(goal_vec))
        if goal_dist < 1e-6:
            return 0.0
        heading_vec = np.asarray([np.cos(self.theta), np.sin(self.theta)], dtype=np.float32)
        alignment = float(np.dot(heading_vec, goal_vec) / goal_dist)
        return self.heading_reward_weight * alignment

    def _anti_stall_penalty(
        self,
        goal_progress: float,
        linear_speed: float,
        current_distance: float,
    ) -> float:
        if current_distance < self.goal_radius * 2.0:
            return 0.0
        if goal_progress >= self.stall_progress_threshold:
            return 0.0
        if linear_speed >= self.stall_speed_threshold:
            return 0.0

        progress_factor = np.clip(
            (self.stall_progress_threshold - goal_progress)
            / max(self.stall_progress_threshold, 1e-6),
            0.0,
            1.0,
        )
        speed_factor = np.clip(
            1.0 - linear_speed / max(self.stall_speed_threshold, 1e-6),
            0.0,
            1.0,
        )
        return float(self.stall_penalty_weight * progress_factor * speed_factor)

    def _is_collision(self) -> bool:
        return self.obstacle_clearance(self.position) <= 0.0

    def _is_out_of_bounds(self) -> bool:
        return self.boundary_clearance(self.position) < 0.0

    @staticmethod
    def _circle_signed_distance(point: np.ndarray, circle: dict[str, Any]) -> float:
        return float(np.linalg.norm(point - circle["center"]) - float(circle["radius"]))

    @staticmethod
    def _rectangle_signed_distance(point: np.ndarray, rect: dict[str, Any]) -> float:
        rect_min = rect["min"]
        rect_max = rect["max"]
        outside_delta = np.maximum(np.maximum(rect_min - point, point - rect_max), 0.0)
        outside_distance = float(np.linalg.norm(outside_delta))
        if outside_distance > 0.0:
            return outside_distance

        distances_to_edges = np.asarray(
            [
                point[0] - rect_min[0],
                rect_max[0] - point[0],
                point[1] - rect_min[1],
                rect_max[1] - point[1],
            ],
            dtype=np.float32,
        )
        return -float(np.min(distances_to_edges))

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    def _build_info(
        self,
        raw_command: np.ndarray,
        exec_command: np.ndarray,
        raw_delta: np.ndarray,
        exec_delta: np.ndarray,
        success: bool,
        collision: bool,
        out_of_bounds: bool,
        timeout: bool,
        heading_reward: float = 0.0,
        obstacle_penalty: float = 0.0,
        min_obstacle_clearance: float | None = None,
        goal_progress: float = 0.0,
        progress_reward: float = 0.0,
        distance_penalty: float = 0.0,
        stall_penalty: float = 0.0,
        timeout_distance_penalty: float = 0.0,
    ) -> dict[str, Any]:
        raw_command = np.asarray(raw_command, dtype=np.float32).reshape(2)
        exec_command = np.asarray(exec_command, dtype=np.float32).reshape(2)
        raw_delta = np.asarray(raw_delta, dtype=np.float32).reshape(2)
        exec_delta = np.asarray(exec_delta, dtype=np.float32).reshape(2)
        if min_obstacle_clearance is None:
            min_obstacle_clearance = self.obstacle_clearance(self.position)
        min_obstacle_signed_distance = self.signed_distance_to_obstacles(self.position)
        min_boundary_clearance = self.boundary_clearance(self.position)

        return {
            "position": self.position.copy(),
            "theta": float(self.theta),
            "goal": self.goal.copy(),
            "distance_to_goal": self._distance_to_goal(),
            "raw_command": raw_command.copy(),
            "executed_command": exec_command.copy(),
            "raw_command_norm": float(np.linalg.norm(raw_command)),
            "exec_command_norm": float(np.linalg.norm(exec_command)),
            "raw_command_delta_norm": float(np.linalg.norm(raw_delta)),
            "exec_command_delta_norm": float(np.linalg.norm(exec_delta)),
            "raw_action": raw_command.copy(),
            "executed_action": exec_command.copy(),
            "raw_action_norm": float(np.linalg.norm(raw_command)),
            "exec_action_norm": float(np.linalg.norm(exec_command)),
            "raw_action_delta_norm": float(np.linalg.norm(raw_delta)),
            "exec_action_delta_norm": float(np.linalg.norm(exec_delta)),
            "goal_progress": float(goal_progress),
            "progress_reward": float(progress_reward),
            "heading_reward": float(heading_reward),
            "obstacle_penalty": float(obstacle_penalty),
            "distance_penalty": float(distance_penalty),
            "stall_penalty": float(stall_penalty),
            "timeout_distance_penalty": float(timeout_distance_penalty),
            "min_obstacle_signed_distance": float(min_obstacle_signed_distance),
            "min_obstacle_clearance": float(min_obstacle_clearance),
            "min_boundary_clearance": float(min_boundary_clearance),
            "robot_radius": self.robot_radius,
            "obstacle_margin": self.obstacle_margin,
            "boundary_margin": self.boundary_margin,
            "success": bool(success),
            "collision": bool(collision),
            "out_of_bounds": bool(out_of_bounds),
            "timeout": bool(timeout),
            "step_count": self.step_count,
            "use_kf": self.use_kf,
            "aug_prev_action": self.aug_prev_action,
        }

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None
