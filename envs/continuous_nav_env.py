"""Continuous 2D navigation environment with KF-in-the-loop action smoothing."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from filters.kalman_action_smoother import KalmanActionSmoother


class ContinuousNavEnv(gym.Env):
    """Minimal continuous 2D point-mass navigation environment.

    The policy action passed to step() is the raw SAC action. If enabled, a
    Kalman filter converts it to an executed action used for position updates.
    The observation includes the previous executed action so the closed loop
    exposes the action smoother state summary to the policy.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        use_kf: bool = True,
        seed: int | None = None,
        map_size: float = 60.0,
        start: tuple[float, float] = (5.0, 5.0),
        goal: tuple[float, float] = (55.0, 55.0),
        obstacles: list[dict[str, Any]] | None = None,
        max_steps: int = 500,
        max_speed: float = 1.0,
        dt: float = 1.0,
        goal_radius: float = 2.0,
        obstacle_safe_margin: float = 4.0,
        obstacle_penalty_weight: float = 0.5,
        heading_reward_weight: float = 0.2,
        progress_reward_weight: float = 3.0,
        distance_penalty_weight: float = 0.05,
        stall_penalty_weight: float = 0.15,
        stall_progress_threshold: float = 0.01,
        stall_speed_threshold: float = 0.05,
        timeout_distance_penalty_weight: float = 50.0,
    ) -> None:
        super().__init__()
        self.map_size = float(map_size)
        self.start = np.asarray(start, dtype=np.float32)
        self.goal = np.asarray(goal, dtype=np.float32)
        self.max_steps = int(max_steps)
        self.max_speed = float(max_speed)
        self.dt = float(dt)
        self.goal_radius = float(goal_radius)
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
        self._initial_seed = seed
        self._has_reset = False

        if obstacles is None:
            obstacles = [
                {"center": [20.0, 20.0], "radius": 5.0},
                {"center": [35.0, 35.0], "radius": 6.0},
                {"center": [20.0, 45.0], "radius": 4.0},
                {"center": [45.0, 20.0], "radius": 4.0},
            ]
        self.obstacles = [
            {
                "center": np.asarray(obs["center"], dtype=np.float32),
                "radius": float(obs["radius"]),
            }
            for obs in obstacles
        ]

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        # obs = pos(2) + goal_rel(2) + goal_dist(1) + obstacles(4*3)
        #       + prev_exec_action(2) = 19 for the default four obstacles.
        self.obs_dim = 5 + len(self.obstacles) * 3 + 2
        self.observation_space = spaces.Box(
            low=-2.0,
            high=2.0,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        self.action_smoother = KalmanActionSmoother(dt=self.dt, use_kf=self.use_kf)

        self.position = self.start.copy()
        self.step_count = 0
        self.prev_distance = self._distance_to_goal()
        self.prev_raw_action = np.zeros(2, dtype=np.float32)
        self.prev_exec_action = np.zeros(2, dtype=np.float32)
        self.prev_exec_delta = np.zeros(2, dtype=np.float32)

    def set_use_kf(self, use_kf: bool) -> None:
        """Enable or disable KF execution."""
        self.use_kf = bool(use_kf)
        self.action_smoother.use_kf = self.use_kf
        self.action_smoother.reset()
        self.prev_exec_action = np.zeros(2, dtype=np.float32)
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
        self.step_count = 0
        self.action_smoother.reset()
        self.prev_distance = self._distance_to_goal()
        self.prev_raw_action = np.zeros(2, dtype=np.float32)
        self.prev_exec_action = np.zeros(2, dtype=np.float32)
        self.prev_exec_delta = np.zeros(2, dtype=np.float32)

        info = self._build_info(
            raw_action=np.zeros(2, dtype=np.float32),
            exec_action=np.zeros(2, dtype=np.float32),
            raw_delta=np.zeros(2, dtype=np.float32),
            exec_delta=np.zeros(2, dtype=np.float32),
            success=False,
            collision=False,
            out_of_bounds=False,
            timeout=False,
        )
        return self._get_obs(), info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        raw_action = np.asarray(action, dtype=np.float32).reshape(2)
        raw_action = np.clip(raw_action, self.action_space.low, self.action_space.high)

        prev_raw_action = self.prev_raw_action.copy()
        prev_exec_action = self.prev_exec_action.copy()
        previous_distance = self._distance_to_goal()

        exec_action = self.action_smoother.smooth(raw_action)
        self.position = (
            self.position + exec_action.astype(np.float32) * self.max_speed * self.dt
        ).astype(np.float32)
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
        obstacle_penalty, min_obstacle_signed_distance = self._obstacle_proximity_penalty()
        heading_reward = self._heading_reward(exec_action)
        stall_penalty = self._anti_stall_penalty(goal_progress, exec_action, current_distance)
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
            reward += 100.0
        if collision or out_of_bounds:
            reward -= 100.0

        raw_delta = raw_action - prev_raw_action
        exec_delta = exec_action - prev_exec_action

        self.prev_raw_action = raw_action.astype(np.float32)
        self.prev_exec_action = exec_action.astype(np.float32)
        self.prev_exec_delta = exec_delta.astype(np.float32)
        self.prev_distance = current_distance

        info = self._build_info(
            raw_action=raw_action,
            exec_action=exec_action,
            raw_delta=raw_delta,
            exec_delta=exec_delta,
            success=success,
            collision=collision,
            out_of_bounds=out_of_bounds,
            timeout=timeout,
            heading_reward=heading_reward,
            obstacle_penalty=obstacle_penalty,
            min_obstacle_signed_distance=min_obstacle_signed_distance,
            goal_progress=goal_progress,
            progress_reward=progress_reward,
            distance_penalty=distance_penalty,
            stall_penalty=stall_penalty,
            timeout_distance_penalty=timeout_distance_penalty,
        )
        return self._get_obs(), float(reward), terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        pos_norm = self.position / self.map_size
        rel_goal = (self.goal - self.position) / self.map_size
        dist_goal = np.array([self._distance_to_goal() / self.diagonal], dtype=np.float32)

        obstacle_features: list[float] = []
        for obstacle in self.obstacles:
            center = obstacle["center"]
            radius = obstacle["radius"]
            rel_center = (center - self.position) / self.map_size
            signed_distance = (np.linalg.norm(self.position - center) - radius) / self.map_size
            obstacle_features.extend([rel_center[0], rel_center[1], signed_distance])

        obs = np.concatenate(
            [
                pos_norm.astype(np.float32),
                rel_goal.astype(np.float32),
                dist_goal.astype(np.float32),
                np.asarray(obstacle_features, dtype=np.float32),
                self.prev_exec_action.astype(np.float32),
            ]
        )
        return obs.astype(np.float32)

    def _distance_to_goal(self) -> float:
        return float(np.linalg.norm(self.goal - self.position))

    def _min_obstacle_signed_distance(self) -> float:
        if not self.obstacles:
            return float("inf")
        return float(
            min(
                np.linalg.norm(self.position - obstacle["center"]) - obstacle["radius"]
                for obstacle in self.obstacles
            )
        )

    def _obstacle_proximity_penalty(self) -> tuple[float, float]:
        min_signed_distance = self._min_obstacle_signed_distance()
        if min_signed_distance >= self.obstacle_safe_margin:
            return 0.0, min_signed_distance

        penetration = max(0.0, self.obstacle_safe_margin - min_signed_distance)
        normalized_penetration = penetration / max(self.obstacle_safe_margin, 1e-6)
        penalty = self.obstacle_penalty_weight * normalized_penetration
        return float(penalty), min_signed_distance

    def _heading_reward(self, exec_action: np.ndarray) -> float:
        action_norm = float(np.linalg.norm(exec_action))
        if action_norm < 1e-6:
            return 0.0

        goal_vec = self.goal - self.position
        goal_dist = float(np.linalg.norm(goal_vec))
        if goal_dist < 1e-6:
            return 0.0

        alignment = float(np.dot(exec_action, goal_vec) / (action_norm * goal_dist))
        return self.heading_reward_weight * alignment

    def _anti_stall_penalty(
        self,
        goal_progress: float,
        exec_action: np.ndarray,
        current_distance: float,
    ) -> float:
        if current_distance < self.goal_radius * 2.0:
            return 0.0

        if goal_progress >= self.stall_progress_threshold:
            return 0.0

        exec_speed = float(np.linalg.norm(exec_action))
        if exec_speed >= self.stall_speed_threshold:
            return 0.0

        progress_gap = self.stall_progress_threshold - goal_progress
        progress_factor = np.clip(
            progress_gap / max(self.stall_progress_threshold, 1e-6),
            0.0,
            1.0,
        )
        speed_factor = np.clip(
            1.0 - exec_speed / max(self.stall_speed_threshold, 1e-6),
            0.0,
            1.0,
        )
        return float(self.stall_penalty_weight * progress_factor * speed_factor)

    def _is_collision(self) -> bool:
        for obstacle in self.obstacles:
            if np.linalg.norm(self.position - obstacle["center"]) <= obstacle["radius"]:
                return True
        return False

    def _is_out_of_bounds(self) -> bool:
        return bool(
            self.position[0] < 0.0
            or self.position[0] > self.map_size
            or self.position[1] < 0.0
            or self.position[1] > self.map_size
        )

    def _build_info(
        self,
        raw_action: np.ndarray,
        exec_action: np.ndarray,
        raw_delta: np.ndarray,
        exec_delta: np.ndarray,
        success: bool,
        collision: bool,
        out_of_bounds: bool,
        timeout: bool,
        heading_reward: float = 0.0,
        obstacle_penalty: float = 0.0,
        min_obstacle_signed_distance: float | None = None,
        goal_progress: float = 0.0,
        progress_reward: float = 0.0,
        distance_penalty: float = 0.0,
        stall_penalty: float = 0.0,
        timeout_distance_penalty: float = 0.0,
    ) -> dict[str, Any]:
        if min_obstacle_signed_distance is None:
            min_obstacle_signed_distance = self._min_obstacle_signed_distance()

        return {
            "position": self.position.copy(),
            "goal": self.goal.copy(),
            "distance_to_goal": self._distance_to_goal(),
            "raw_action": np.asarray(raw_action, dtype=np.float32).copy(),
            "executed_action": np.asarray(exec_action, dtype=np.float32).copy(),
            "raw_action_norm": float(np.linalg.norm(raw_action)),
            "exec_action_norm": float(np.linalg.norm(exec_action)),
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
            "success": bool(success),
            "collision": bool(collision),
            "out_of_bounds": bool(out_of_bounds),
            "timeout": bool(timeout),
            "step_count": self.step_count,
            "use_kf": self.use_kf,
        }

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None
