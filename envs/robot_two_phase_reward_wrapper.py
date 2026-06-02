"""Two-phase reward wrapper for fixed-KF robot v/w docking experiments."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from envs.robot_command_smoothing_wrappers import make_robot_command_smoothing_env


class TwoPhaseDockingRewardWrapper(gym.Wrapper):
    """Use approach reward outside the docking zone and low-speed docking reward inside it."""

    def __init__(
        self,
        env: gym.Env,
        slowdown_radius: float = 8.0,
        approach_progress_weight: float = 4.0,
        docking_progress_weight: float = 2.0,
        docking_distance_weight: float = 0.5,
        docking_entry_bonus: float = 25.0,
        linear_speed_weight: float = 1.5,
        angular_speed_weight: float = 0.8,
        heading_penalty_weight: float = 0.05,
        inside_goal_fast_penalty_weight: float = 5.0,
        success_linear_threshold: float = 0.25,
        success_angular_threshold: float = 0.25,
        success_reward: float = 150.0,
        collision_penalty: float = 150.0,
        timeout_penalty: float = 50.0,
        step_penalty: float = 0.01,
    ) -> None:
        super().__init__(env)
        self.slowdown_radius = float(slowdown_radius)
        self.approach_progress_weight = float(approach_progress_weight)
        self.docking_progress_weight = float(docking_progress_weight)
        self.docking_distance_weight = float(docking_distance_weight)
        self.docking_entry_bonus = float(docking_entry_bonus)
        self.linear_speed_weight = float(linear_speed_weight)
        self.angular_speed_weight = float(angular_speed_weight)
        self.heading_penalty_weight = float(heading_penalty_weight)
        self.inside_goal_fast_penalty_weight = float(inside_goal_fast_penalty_weight)
        self.success_linear_threshold = float(success_linear_threshold)
        self.success_angular_threshold = float(success_angular_threshold)
        self.success_reward = float(success_reward)
        self.collision_penalty = float(collision_penalty)
        self.timeout_penalty = float(timeout_penalty)
        self.step_penalty = float(step_penalty)
        self.prev_distance: float | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        self.prev_distance = float(info.get("distance_to_goal", self._distance_to_goal()))
        return obs, self._add_reset_info(info)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        previous_distance = self.prev_distance
        if previous_distance is None:
            previous_distance = self._distance_to_goal()

        obs, base_reward, base_terminated, base_truncated, info = self.env.step(action)
        del base_terminated, base_truncated

        reward_info = self._reward_info(
            info=info,
            previous_distance=float(previous_distance),
            base_reward=float(base_reward),
        )
        self.prev_distance = reward_info["distance_to_goal"]

        terminated = bool(
            reward_info["success"]
            or reward_info["collision"]
            or reward_info["out_of_bounds"]
        )
        timeout = bool(self.unwrapped.step_count >= self.unwrapped.max_steps and not terminated)
        truncated = timeout

        updated_info = dict(info)
        updated_info.update(reward_info)
        updated_info["timeout"] = timeout
        updated_info["terminated_by_low_speed_success"] = bool(reward_info["success"])
        return obs, float(reward_info["two_phase_reward"]), terminated, truncated, updated_info

    def set_curriculum_progress(self, progress: float) -> None:
        method = getattr(self.env, "set_curriculum_progress", None)
        if callable(method):
            method(progress)

    def state_info(self) -> dict[str, float]:
        method = getattr(self.env, "state_info", None)
        if callable(method):
            return method()
        return {}

    def _distance_to_goal(self) -> float:
        base = self.unwrapped
        return float(np.linalg.norm(base.goal - base.position))

    def _add_reset_info(self, info: dict[str, Any]) -> dict[str, Any]:
        updated_info = dict(info)
        updated_info.update(
            {
                "base_reward": 0.0,
                "base_success": False,
                "success": False,
                "reward_phase": "approach",
                "reward_phase_id": 0.0,
                "in_docking_zone": False,
                "entered_docking_zone": False,
                "inside_goal_too_fast": False,
                "two_phase_reward": 0.0,
                "approach_reward": 0.0,
                "docking_reward": 0.0,
                "docking_entry_bonus": 0.0,
                "docking_distance_penalty": 0.0,
                "terminal_linear_speed_penalty": 0.0,
                "terminal_angular_speed_penalty": 0.0,
                "terminal_heading_penalty": 0.0,
                "inside_goal_fast_penalty": 0.0,
                "low_speed_success_reward": 0.0,
                "collision_or_oob_penalty": 0.0,
                "two_phase_timeout_penalty": 0.0,
                "terminal_v_exec": 0.0,
                "terminal_abs_w_exec": 0.0,
            }
        )
        return updated_info

    def _reward_info(
        self,
        info: dict[str, Any],
        previous_distance: float,
        base_reward: float,
    ) -> dict[str, Any]:
        base = self.unwrapped
        current_distance = float(info.get("distance_to_goal", self._distance_to_goal()))
        goal_radius = float(base.goal_radius)
        goal_progress = previous_distance - current_distance
        collision = bool(info.get("collision", False))
        out_of_bounds = bool(info.get("out_of_bounds", False))
        base_success = bool(info.get("success", False))

        executed = np.asarray(info.get("executed_command", np.zeros(2)), dtype=np.float32).reshape(2)
        v_exec = float(executed[0])
        abs_w_exec = float(abs(executed[1]))
        max_v = max(float(base.max_linear_speed), 1e-8)
        max_w = max(float(base.max_angular_speed), 1e-8)
        v_norm = abs(v_exec) / max_v
        w_norm = abs_w_exec / max_w

        in_docking_zone = current_distance <= self.slowdown_radius
        entered_docking_zone = previous_distance > self.slowdown_radius and in_docking_zone
        low_speed_success = (
            current_distance < goal_radius
            and abs(v_exec) <= self.success_linear_threshold
            and abs_w_exec <= self.success_angular_threshold
        )
        inside_goal_too_fast = current_distance < goal_radius and not low_speed_success

        safety_penalty = float(info.get("obstacle_penalty", 0.0))
        collision_or_oob_penalty = self.collision_penalty if collision or out_of_bounds else 0.0
        timeout = bool(base.step_count >= base.max_steps and not (low_speed_success or collision or out_of_bounds))
        timeout_penalty = self.timeout_penalty if timeout else 0.0

        approach_reward = 0.0
        docking_reward = 0.0
        docking_distance_penalty = 0.0
        linear_speed_penalty = 0.0
        angular_speed_penalty = 0.0
        heading_penalty = 0.0
        inside_goal_fast_penalty = 0.0
        low_speed_success_reward = self.success_reward if low_speed_success else 0.0
        entry_bonus = self.docking_entry_bonus if entered_docking_zone else 0.0

        if not in_docking_zone or entered_docking_zone:
            progress_reward = self.approach_progress_weight * goal_progress
            approach_reward = progress_reward - safety_penalty - self.step_penalty + entry_bonus
            phase = "approach"
            phase_id = 0.0
            shaped_reward = approach_reward
        else:
            progress_reward = self.docking_progress_weight * goal_progress
            docking_distance_penalty = self.docking_distance_weight * (
                current_distance / max(self.slowdown_radius, 1e-8)
            )
            linear_speed_penalty = self.linear_speed_weight * (v_norm**2)
            angular_speed_penalty = self.angular_speed_weight * (w_norm**2)
            heading_penalty = self._heading_penalty()
            if inside_goal_too_fast:
                linear_over = max(0.0, abs(v_exec) - self.success_linear_threshold) / max_v
                angular_over = max(0.0, abs_w_exec - self.success_angular_threshold) / max_w
                inside_goal_fast_penalty = self.inside_goal_fast_penalty_weight * (
                    linear_over + angular_over
                )
            docking_reward = (
                progress_reward
                - safety_penalty
                - docking_distance_penalty
                - linear_speed_penalty
                - angular_speed_penalty
                - heading_penalty
                - inside_goal_fast_penalty
                - self.step_penalty
                + low_speed_success_reward
            )
            phase = "docking"
            phase_id = 1.0
            shaped_reward = docking_reward

        shaped_reward = shaped_reward - collision_or_oob_penalty - timeout_penalty

        return {
            "base_reward": float(base_reward),
            "base_success": base_success,
            "success": bool(low_speed_success),
            "collision": collision,
            "out_of_bounds": out_of_bounds,
            "timeout": timeout,
            "distance_to_goal": current_distance,
            "goal_progress": float(goal_progress),
            "reward_phase": phase,
            "reward_phase_id": float(phase_id),
            "in_docking_zone": bool(in_docking_zone),
            "entered_docking_zone": bool(entered_docking_zone),
            "inside_goal_too_fast": bool(inside_goal_too_fast),
            "two_phase_reward": float(shaped_reward),
            "approach_reward": float(approach_reward),
            "docking_reward": float(docking_reward),
            "progress_reward": float(progress_reward),
            "safety_penalty": float(safety_penalty),
            "docking_entry_bonus": float(entry_bonus),
            "docking_distance_penalty": float(docking_distance_penalty),
            "terminal_linear_speed_penalty": float(linear_speed_penalty),
            "terminal_angular_speed_penalty": float(angular_speed_penalty),
            "terminal_heading_penalty": float(heading_penalty),
            "inside_goal_fast_penalty": float(inside_goal_fast_penalty),
            "low_speed_success_reward": float(low_speed_success_reward),
            "collision_or_oob_penalty": float(collision_or_oob_penalty),
            "two_phase_timeout_penalty": float(timeout_penalty),
            "terminal_v_exec": float(v_exec),
            "terminal_abs_w_exec": float(abs_w_exec),
            "success_linear_threshold": float(self.success_linear_threshold),
            "success_angular_threshold": float(self.success_angular_threshold),
            "slowdown_radius": float(self.slowdown_radius),
        }

    def _heading_penalty(self) -> float:
        base = self.unwrapped
        goal_vec = base.goal - base.position
        goal_dist = float(np.linalg.norm(goal_vec))
        if goal_dist < 1e-8:
            return 0.0
        heading_vec = np.asarray([np.cos(base.theta), np.sin(base.theta)], dtype=np.float32)
        alignment = float(np.dot(heading_vec, goal_vec) / goal_dist)
        alignment = float(np.clip(alignment, -1.0, 1.0))
        return float(self.heading_penalty_weight * (1.0 - alignment) * 0.5)


def make_robot_two_phase_env(
    smoother: str = "fixed",
    obs_mode: str = "kf_state",
    kf_curriculum: str = "none",
    seed: int | None = None,
    slowdown_radius: float = 8.0,
    approach_progress_weight: float = 4.0,
    docking_progress_weight: float = 2.0,
    docking_distance_weight: float = 0.5,
    docking_entry_bonus: float = 25.0,
    linear_speed_weight: float = 1.5,
    angular_speed_weight: float = 0.8,
    heading_penalty_weight: float = 0.05,
    inside_goal_fast_penalty_weight: float = 5.0,
    success_linear_threshold: float = 0.25,
    success_angular_threshold: float = 0.25,
    success_reward: float = 150.0,
    collision_penalty: float = 150.0,
    timeout_penalty: float = 50.0,
    step_penalty: float = 0.01,
) -> gym.Env:
    """Build fixed/adaptive command smoothing env with two-phase docking reward."""

    env = make_robot_command_smoothing_env(
        smoother=smoother,
        obs_mode=obs_mode,
        kf_curriculum=kf_curriculum,
        seed=seed,
    )
    return TwoPhaseDockingRewardWrapper(
        env,
        slowdown_radius=slowdown_radius,
        approach_progress_weight=approach_progress_weight,
        docking_progress_weight=docking_progress_weight,
        docking_distance_weight=docking_distance_weight,
        docking_entry_bonus=docking_entry_bonus,
        linear_speed_weight=linear_speed_weight,
        angular_speed_weight=angular_speed_weight,
        heading_penalty_weight=heading_penalty_weight,
        inside_goal_fast_penalty_weight=inside_goal_fast_penalty_weight,
        success_linear_threshold=success_linear_threshold,
        success_angular_threshold=success_angular_threshold,
        success_reward=success_reward,
        collision_penalty=collision_penalty,
        timeout_penalty=timeout_penalty,
        step_penalty=step_penalty,
    )
