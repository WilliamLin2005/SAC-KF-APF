"""Complex static maze navigation environment with KF-in-the-loop smoothing."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from filters.kalman_action_smoother import SMOOTHER_OBS_DIM, make_action_smoother


class ComplexNavEnv(gym.Env):
    """2D unicycle navigation in a static maze-like obstacle field.

    The policy action is [v_norm, omega_norm]. The environment executes the
    smoothed action using unicycle kinematics. The observation does not include
    lidar. It includes a fixed-length smoother/KF feature vector.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        use_kf: bool = True,
        seed: int | None = None,
        map_size: float = 80.0,
        start: tuple[float, float] = (6.0, 6.0),
        goal: tuple[float, float] = (74.0, 74.0),
        max_steps: int = 800,
        max_linear_speed: float = 1.2,
        max_angular_speed: float = 1.0,
        initial_heading: float | None = None,
        smoother_type: str = "current_kf",
        smoother_kwargs: dict[str, Any] | None = None,
        dt: float = 1.0,
        goal_radius: float = 2.5,
        lidar_num_rays: int = 24,
        lidar_max_range: float = 30.0,
        lidar_step: float = 0.5,
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
        self.max_steps = int(max_steps)
        self.max_linear_speed = float(max_linear_speed)
        self.max_angular_speed = float(max_angular_speed)
        self.initial_heading = initial_heading
        self.heading = 0.0
        self.smoother_type = smoother_type
        self.smoother_kwargs = smoother_kwargs or {}
        self.dt = float(dt)
        self.goal_radius = float(goal_radius)
        self.lidar_num_rays = int(lidar_num_rays)
        self.lidar_max_range = float(lidar_max_range)
        self.lidar_step = float(lidar_step)
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

        self.rectangles = [
            {"min": np.asarray([18.0, 12.0], dtype=np.float32), "max": np.asarray([22.0, 52.0], dtype=np.float32)},
            {"min": np.asarray([22.0, 24.0], dtype=np.float32), "max": np.asarray([48.0, 28.0], dtype=np.float32)},
            {"min": np.asarray([38.0, 28.0], dtype=np.float32), "max": np.asarray([42.0, 66.0], dtype=np.float32)},
            {"min": np.asarray([58.0, 14.0], dtype=np.float32), "max": np.asarray([62.0, 50.0], dtype=np.float32)},
            {"min": np.asarray([42.0, 62.0], dtype=np.float32), "max": np.asarray([70.0, 66.0], dtype=np.float32)},
            {"min": np.asarray([8.0, 52.0], dtype=np.float32), "max": np.asarray([32.0, 56.0], dtype=np.float32)},
        ]
        self.circles = [
            {"center": np.asarray([30.0, 14.0], dtype=np.float32), "radius": 3.0},
            {"center": np.asarray([48.0, 40.0], dtype=np.float32), "radius": 3.5},
            {"center": np.asarray([68.0, 44.0], dtype=np.float32), "radius": 4.0},
            {"center": np.asarray([28.0, 68.0], dtype=np.float32), "radius": 4.0},
            {"center": np.asarray([14.0, 38.0], dtype=np.float32), "radius": 3.0},
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
        self.obs_dim = 2 + 2 + 1 + 2 + 2 + SMOOTHER_OBS_DIM
        self.observation_space = spaces.Box(
            low=-2.0,
            high=2.0,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        self.action_smoother = make_action_smoother(
            smoother_type=self.smoother_type,
            dt=self.dt,
            use_kf=self.use_kf,
            smoother_kwargs=self.smoother_kwargs,
        )
        self.position = self.start.copy()
        self.step_count = 0
        self.prev_distance = self._distance_to_goal()
        self.prev_raw_action = np.zeros(2, dtype=np.float32)
        self.prev_exec_action = np.zeros(2, dtype=np.float32)
        self.prev_exec_delta = np.zeros(2, dtype=np.float32)

    def set_use_kf(self, use_kf: bool) -> None:
        self.use_kf = bool(use_kf)
        self.action_smoother = make_action_smoother(
            smoother_type=self.smoother_type,
            dt=self.dt,
            use_kf=self.use_kf,
            smoother_kwargs=self.smoother_kwargs,
        )
        self.prev_exec_action = np.zeros(2, dtype=np.float32)
        self.prev_exec_delta = np.zeros(2, dtype=np.float32)

    def set_smoother_type(
        self,
        smoother_type: str,
        smoother_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.smoother_type = smoother_type
        self.smoother_kwargs = smoother_kwargs or {}
        self.action_smoother = make_action_smoother(
            smoother_type=self.smoother_type,
            dt=self.dt,
            use_kf=self.use_kf,
            smoother_kwargs=self.smoother_kwargs,
        )
        self.prev_exec_action = np.zeros(2, dtype=np.float32)
        self.prev_exec_delta = np.zeros(2, dtype=np.float32)

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    def _compute_initial_heading(self) -> float:
        if self.initial_heading is not None:
            return self._wrap_angle(float(self.initial_heading))
        goal_vec = self.goal - self.start
        return self._wrap_angle(float(np.arctan2(goal_vec[1], goal_vec[0])))

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
        self.heading = self._compute_initial_heading()
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

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        raw_action = np.asarray(action, dtype=np.float32).reshape(2)
        raw_action = np.clip(raw_action, self.action_space.low, self.action_space.high)

        prev_raw_action = self.prev_raw_action.copy()
        prev_exec_action = self.prev_exec_action.copy()
        previous_distance = self._distance_to_goal()

        exec_action = self.action_smoother.smooth(raw_action)
        v_norm = float(np.clip(exec_action[0], 0.0, 1.0))
        omega_norm = float(np.clip(exec_action[1], -1.0, 1.0))
        exec_action = np.asarray([v_norm, omega_norm], dtype=np.float32)

        linear_velocity = v_norm * self.max_linear_speed
        angular_velocity = omega_norm * self.max_angular_speed
        direction = np.asarray(
            [np.cos(self.heading), np.sin(self.heading)],
            dtype=np.float32,
        )
        self.position = (
            self.position + direction * np.float32(linear_velocity * self.dt)
        ).astype(np.float32)
        self.heading = self._wrap_angle(self.heading + angular_velocity * self.dt)
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
            reward += 150.0
        if collision or out_of_bounds:
            reward -= 150.0

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
        dist_goal = np.asarray([self._distance_to_goal() / self.diagonal], dtype=np.float32)
        heading_features = np.asarray(
            [np.sin(self.heading), np.cos(self.heading)],
            dtype=np.float32,
        )
        smoother_features = self.action_smoother.get_observation_features()
        smoother_features = np.asarray(smoother_features, dtype=np.float32).reshape(SMOOTHER_OBS_DIM)

        obs = np.concatenate(
            [
                pos_norm.astype(np.float32),
                rel_goal.astype(np.float32),
                dist_goal.astype(np.float32),
                heading_features,
                self.prev_exec_action.astype(np.float32),
                smoother_features,
            ]
        )
        return obs.astype(np.float32)

    def _lidar_ranges(self) -> np.ndarray:
        angles = np.linspace(0.0, 2.0 * np.pi, self.lidar_num_rays, endpoint=False)
        ranges = np.full(self.lidar_num_rays, self.lidar_max_range, dtype=np.float32)
        distances = np.arange(0.0, self.lidar_max_range + self.lidar_step, self.lidar_step)

        for ray_idx, angle in enumerate(angles):
            direction = np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float32)
            for distance in distances[1:]:
                point = self.position + direction * np.float32(distance)
                if self._is_out_of_bounds_point(point) or self._is_collision_point(point):
                    ranges[ray_idx] = float(distance)
                    break
        return ranges

    def _distance_to_goal(self) -> float:
        return float(np.linalg.norm(self.goal - self.position))

    def _min_obstacle_signed_distance(self) -> float:
        return float(self.signed_distance_to_obstacles(self.position))

    def signed_distance_to_obstacles(self, point: np.ndarray) -> float:
        point = np.asarray(point, dtype=np.float32)
        distances = [self._circle_signed_distance(point, circle) for circle in self.circles]
        distances.extend(self._rectangle_signed_distance(point, rect) for rect in self.rectangles)
        return float(min(distances)) if distances else float("inf")

    def _obstacle_proximity_penalty(self) -> tuple[float, float]:
        min_signed_distance = self._min_obstacle_signed_distance()
        if min_signed_distance >= self.obstacle_safe_margin:
            return 0.0, min_signed_distance

        penetration = max(0.0, self.obstacle_safe_margin - min_signed_distance)
        normalized_penetration = penetration / max(self.obstacle_safe_margin, 1e-6)
        penalty = self.obstacle_penalty_weight * normalized_penetration
        return float(penalty), min_signed_distance

    def _heading_reward(self, exec_action: np.ndarray) -> float:
        v_norm = max(0.0, float(exec_action[0]))

        goal_vec = self.goal - self.position
        goal_dist = float(np.linalg.norm(goal_vec))
        if goal_dist < 1e-6:
            return 0.0

        goal_heading = float(np.arctan2(goal_vec[1], goal_vec[0]))
        heading_error = self._wrap_angle(goal_heading - self.heading)
        alignment = float(np.cos(heading_error))
        return float(self.heading_reward_weight * alignment * v_norm)

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

        exec_speed = abs(float(exec_action[0]))
        if exec_speed >= self.stall_speed_threshold:
            return 0.0

        progress_factor = np.clip(
            (self.stall_progress_threshold - goal_progress)
            / max(self.stall_progress_threshold, 1e-6),
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
        return self._is_collision_point(self.position)

    def _is_collision_point(self, point: np.ndarray) -> bool:
        for circle in self.circles:
            if self._circle_signed_distance(point, circle) <= 0.0:
                return True
        for rect in self.rectangles:
            if self._rectangle_signed_distance(point, rect) <= 0.0:
                return True
        return False

    def _is_out_of_bounds(self) -> bool:
        return self._is_out_of_bounds_point(self.position)

    def _is_out_of_bounds_point(self, point: np.ndarray) -> bool:
        return bool(
            point[0] < 0.0
            or point[0] > self.map_size
            or point[1] < 0.0
            or point[1] > self.map_size
        )

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
            "heading": float(self.heading),
            "smoother_type": str(getattr(self.action_smoother, "smoother_type", self.smoother_type)),
            "linear_velocity": float(exec_action[0]) * self.max_linear_speed,
            "angular_velocity": float(exec_action[1]) * self.max_angular_speed,
            "v_norm": float(exec_action[0]),
            "omega_norm": float(exec_action[1]),
            "smoother_features": self.action_smoother.get_observation_features().copy(),
        }

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None
