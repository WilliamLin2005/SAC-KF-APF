"""Near-target slowdown reward wrapper for robot v/w KF ablations."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from envs.robot_command_smoothing_wrappers import make_robot_command_smoothing_env


class NearTargetSlowdownRewardWrapper(gym.Wrapper):
    """Apply a speed/angular-speed penalty only close to the target."""

    def __init__(
        self,
        env: gym.Env,
        slowdown_radius: float = 8.0,
        linear_weight: float = 0.35,
        angular_weight: float = 0.15,
    ) -> None:
        super().__init__(env)
        self.slowdown_radius = float(slowdown_radius)
        self.linear_weight = float(linear_weight)
        self.angular_weight = float(angular_weight)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        info = self._add_zero_penalty_info(info)
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        penalty_info = self._penalty_info(info)
        reward_without_penalty = float(reward)
        reward = reward_without_penalty - penalty_info["near_target_slowdown_penalty"]
        updated_info = dict(info)
        updated_info.update(penalty_info)
        updated_info["reward_without_near_target_slowdown"] = reward_without_penalty
        return obs, float(reward), terminated, truncated, updated_info

    def set_curriculum_progress(self, progress: float) -> None:
        method = getattr(self.env, "set_curriculum_progress", None)
        if callable(method):
            method(progress)

    def state_info(self) -> dict[str, float]:
        method = getattr(self.env, "state_info", None)
        if callable(method):
            return method()
        return {}

    def _add_zero_penalty_info(self, info: dict[str, Any]) -> dict[str, Any]:
        updated_info = dict(info)
        updated_info.update(
            {
                "reward_without_near_target_slowdown": 0.0,
                "near_target_slowdown_penalty": 0.0,
                "near_target_slowdown_factor": 0.0,
                "near_target_linear_penalty": 0.0,
                "near_target_angular_penalty": 0.0,
            }
        )
        return updated_info

    def _penalty_info(self, info: dict[str, Any]) -> dict[str, float]:
        base_env = self.unwrapped
        distance_to_goal = float(info.get("distance_to_goal", 0.0))
        goal_radius = float(getattr(base_env, "goal_radius", 0.0))
        if distance_to_goal > self.slowdown_radius:
            near_factor = 0.0
        else:
            near_factor = float(
                np.clip(
                    (self.slowdown_radius - distance_to_goal)
                    / max(self.slowdown_radius - goal_radius, 1e-8),
                    0.0,
                    1.0,
                )
            )

        executed = np.asarray(info.get("executed_command", np.zeros(2)), dtype=np.float32).reshape(2)
        max_linear_speed = max(float(getattr(base_env, "max_linear_speed", 1.0)), 1e-8)
        max_angular_speed = max(float(getattr(base_env, "max_angular_speed", 1.0)), 1e-8)
        v_norm = float(executed[0]) / max_linear_speed
        w_norm = float(executed[1]) / max_angular_speed
        linear_penalty = near_factor * self.linear_weight * (v_norm**2)
        angular_penalty = near_factor * self.angular_weight * (w_norm**2)
        penalty = linear_penalty + angular_penalty
        return {
            "near_target_slowdown_penalty": float(penalty),
            "near_target_slowdown_factor": float(near_factor),
            "near_target_linear_penalty": float(linear_penalty),
            "near_target_angular_penalty": float(angular_penalty),
        }


def make_robot_near_target_env(
    smoother: str = "fixed",
    obs_mode: str = "kf_state",
    kf_curriculum: str = "none",
    seed: int | None = None,
    slowdown_radius: float = 8.0,
    linear_weight: float = 0.35,
    angular_weight: float = 0.15,
) -> gym.Env:
    """Build fixed/adaptive command smoothing env with near-target reward shaping."""

    env = make_robot_command_smoothing_env(
        smoother=smoother,
        obs_mode=obs_mode,
        kf_curriculum=kf_curriculum,
        seed=seed,
    )
    return NearTargetSlowdownRewardWrapper(
        env,
        slowdown_radius=slowdown_radius,
        linear_weight=linear_weight,
        angular_weight=angular_weight,
    )
