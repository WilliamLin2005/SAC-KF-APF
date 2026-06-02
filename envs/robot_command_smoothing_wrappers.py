"""Composable command smoothing and KF-state observation wrappers."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.robot_diffdrive_complex_env import RobotDiffDriveComplexEnv
from filters.adaptive_kalman_command_smoother import (
    AdaptiveKalmanCommandSmoother,
    CommandSmoothingContext,
)
from filters.bounded_kalman_command_smoother import BoundedKalmanCommandSmoother


SMOOTHER_CHOICES = ("none", "fixed", "adaptive")
OBS_MODE_CHOICES = ("base", "prev_exec", "kf_state")
CURRICULUM_CHOICES = ("none", "continuous")


class NoCommandSmoother:
    """Interface-compatible no-op smoother."""

    def __init__(self, command_low: np.ndarray, command_high: np.ndarray) -> None:
        self.command_low = np.asarray(command_low, dtype=np.float64)
        self.command_high = np.asarray(command_high, dtype=np.float64)
        self.curriculum_progress = 1.0

    def reset(self) -> None:
        return None

    def set_curriculum_progress(self, progress: float) -> None:
        self.curriculum_progress = float(np.clip(progress, 0.0, 1.0))

    def smooth(
        self,
        raw_command: np.ndarray,
        context: CommandSmoothingContext | None = None,
    ) -> np.ndarray:
        del context
        return np.clip(
            np.asarray(raw_command, dtype=np.float64).reshape(2),
            self.command_low,
            self.command_high,
        ).astype(np.float32)

    def state_info(self) -> dict[str, float]:
        return {
            "kf_process_noise_v": 0.0,
            "kf_process_noise_w": 0.0,
            "kf_measurement_noise_v": 0.0,
            "kf_measurement_noise_w": 0.0,
            "kf_covariance_v": 0.0,
            "kf_covariance_w": 0.0,
            "kf_curriculum_progress": float(self.curriculum_progress),
            "kf_responsive_factor": 0.0,
            "kf_jitter_factor": 0.0,
        }


class FixedKalmanCommandSmoother:
    """Adapter around the existing bounded linear command KF."""

    def __init__(self, command_low: np.ndarray, command_high: np.ndarray) -> None:
        self.smoother = BoundedKalmanCommandSmoother(
            command_low=command_low,
            command_high=command_high,
            use_kf=True,
        )
        self.curriculum_progress = 1.0

    def reset(self) -> None:
        self.smoother.reset()

    def set_curriculum_progress(self, progress: float) -> None:
        self.curriculum_progress = float(np.clip(progress, 0.0, 1.0))

    def smooth(
        self,
        raw_command: np.ndarray,
        context: CommandSmoothingContext | None = None,
    ) -> np.ndarray:
        del context
        return self.smoother.smooth(raw_command)

    def state_info(self) -> dict[str, float]:
        covariance_diag = np.diag(self.smoother.P)
        return {
            "kf_process_noise_v": float(self.smoother.process_noise_std),
            "kf_process_noise_w": float(self.smoother.process_noise_std),
            "kf_measurement_noise_v": float(self.smoother.measurement_noise_std),
            "kf_measurement_noise_w": float(self.smoother.measurement_noise_std),
            "kf_covariance_v": float(covariance_diag[0]),
            "kf_covariance_w": float(covariance_diag[1]),
            "kf_curriculum_progress": float(self.curriculum_progress),
            "kf_responsive_factor": 0.0,
            "kf_jitter_factor": 0.0,
        }


class CommandSmoothingWrapper(gym.Wrapper):
    """Inject fixed/adaptive command smoothing before no-KF env execution."""

    def __init__(
        self,
        env: RobotDiffDriveComplexEnv,
        smoother: str = "adaptive",
        kf_curriculum: str = "continuous",
    ) -> None:
        if smoother not in SMOOTHER_CHOICES:
            raise ValueError(f"Unsupported smoother: {smoother}")
        if kf_curriculum not in CURRICULUM_CHOICES:
            raise ValueError(f"Unsupported kf_curriculum: {kf_curriculum}")
        super().__init__(env)
        self.smoother_mode = smoother
        self.kf_curriculum = kf_curriculum
        self.command_smoother = self._make_smoother(smoother)
        self.curriculum_progress = 0.0 if kf_curriculum == "continuous" else 1.0
        self.command_smoother.set_curriculum_progress(self.curriculum_progress)
        self.prev_raw_command = np.zeros(2, dtype=np.float32)
        self.prev_exec_command = np.zeros(2, dtype=np.float32)
        self.prev_exec_delta = np.zeros(2, dtype=np.float32)
        self.raw_exec_error = np.zeros(2, dtype=np.float32)

    def _make_smoother(self, smoother: str):
        base = self.unwrapped
        if smoother == "none":
            return NoCommandSmoother(base.command_low, base.command_high)
        if smoother == "fixed":
            return FixedKalmanCommandSmoother(base.command_low, base.command_high)
        return AdaptiveKalmanCommandSmoother(base.command_low, base.command_high)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        self.command_smoother.reset()
        self.command_smoother.set_curriculum_progress(self.curriculum_progress)
        self.prev_raw_command = np.zeros(2, dtype=np.float32)
        self.prev_exec_command = np.zeros(2, dtype=np.float32)
        self.prev_exec_delta = np.zeros(2, dtype=np.float32)
        self.raw_exec_error = np.zeros(2, dtype=np.float32)
        info = self._override_info(
            info=info,
            raw_command=np.zeros(2, dtype=np.float32),
            exec_command=np.zeros(2, dtype=np.float32),
            raw_delta=np.zeros(2, dtype=np.float32),
            exec_delta=np.zeros(2, dtype=np.float32),
        )
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        base = self.unwrapped
        raw_action = np.clip(
            np.asarray(action, dtype=np.float32).reshape(2),
            base.action_space.low,
            base.action_space.high,
        ).astype(np.float32)
        raw_command = base.normalized_action_to_command(raw_action)
        raw_delta = raw_command - self.prev_raw_command
        prev_exec_command = self.prev_exec_command.copy()
        context = self._build_context(raw_command_delta_norm=float(np.linalg.norm(raw_delta)))
        exec_command = self.command_smoother.smooth(raw_command, context=context)
        exec_action = base.command_to_normalized_action(exec_command)

        obs, reward, terminated, truncated, info = self.env.step(exec_action)

        exec_delta = exec_command - prev_exec_command
        self.prev_raw_command = raw_command.astype(np.float32)
        self.prev_exec_command = exec_command.astype(np.float32)
        self.prev_exec_delta = exec_delta.astype(np.float32)
        self.raw_exec_error = (raw_command - exec_command).astype(np.float32)
        info = self._override_info(
            info=info,
            raw_command=raw_command,
            exec_command=exec_command,
            raw_delta=raw_delta,
            exec_delta=exec_delta,
        )
        return obs, reward, terminated, truncated, info

    def set_curriculum_progress(self, progress: float) -> None:
        if self.kf_curriculum == "continuous":
            self.curriculum_progress = float(np.clip(progress, 0.0, 1.0))
        else:
            self.curriculum_progress = 1.0
        self.command_smoother.set_curriculum_progress(self.curriculum_progress)

    def state_info(self) -> dict[str, float]:
        info = self.command_smoother.state_info()
        info["smoother_mode"] = self.smoother_mode
        info["kf_curriculum"] = self.kf_curriculum
        return info

    def kf_state_observation(self) -> np.ndarray:
        base = self.unwrapped
        err_norm = np.asarray(
            [
                self.raw_exec_error[0] / max(base.max_linear_speed, 1e-8),
                self.raw_exec_error[1] / max(2.0 * base.max_angular_speed, 1e-8),
            ],
            dtype=np.float32,
        )
        exec_delta_norm = np.asarray(
            [
                self.prev_exec_delta[0] / max(base.max_linear_speed, 1e-8),
                self.prev_exec_delta[1] / max(2.0 * base.max_angular_speed, 1e-8),
            ],
            dtype=np.float32,
        )
        state = self.command_smoother.state_info()
        covariance_norm = np.asarray(
            [
                state.get("kf_covariance_v", 0.0),
                state.get("kf_covariance_w", 0.0),
            ],
            dtype=np.float32,
        )
        covariance_norm = covariance_norm / max(
            float(getattr(self.command_smoother, "initial_covariance", 1.0)),
            1e-8,
        )
        obs = np.concatenate([err_norm, exec_delta_norm, covariance_norm]).astype(np.float32)
        return np.clip(obs, -1.0, 1.0).astype(np.float32)

    def _build_context(self, raw_command_delta_norm: float) -> CommandSmoothingContext:
        base = self.unwrapped
        step_fraction = float(base.step_count) / max(float(base.max_steps), 1.0)
        return CommandSmoothingContext(
            distance_to_goal=float(np.linalg.norm(base.goal - base.position)),
            goal_radius=float(base.goal_radius),
            obstacle_clearance=float(base.obstacle_clearance(base.position)),
            boundary_clearance=float(base.boundary_clearance(base.position)),
            raw_command_delta_norm=float(raw_command_delta_norm),
            step_fraction=step_fraction,
        )

    def _override_info(
        self,
        info: dict[str, Any],
        raw_command: np.ndarray,
        exec_command: np.ndarray,
        raw_delta: np.ndarray,
        exec_delta: np.ndarray,
    ) -> dict[str, Any]:
        raw_command = np.asarray(raw_command, dtype=np.float32).reshape(2)
        exec_command = np.asarray(exec_command, dtype=np.float32).reshape(2)
        raw_delta = np.asarray(raw_delta, dtype=np.float32).reshape(2)
        exec_delta = np.asarray(exec_delta, dtype=np.float32).reshape(2)
        filter_mismatch = float(np.linalg.norm(raw_command - exec_command))
        updated = dict(info)
        updated.update(
            {
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
                "filter_mismatch_norm": filter_mismatch,
                "use_kf": self.smoother_mode != "none",
                "smoother_mode": self.smoother_mode,
                "kf_curriculum": self.kf_curriculum,
            }
        )
        updated.update(self.command_smoother.state_info())
        return updated


class KFStateObservationWrapper(gym.ObservationWrapper):
    """Append raw-exec error, executed delta, and KF covariance diag."""

    def __init__(self, env: CommandSmoothingWrapper) -> None:
        super().__init__(env)
        extra_low = -np.ones(6, dtype=np.float32)
        extra_high = np.ones(6, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.concatenate([env.observation_space.low, extra_low]).astype(np.float32),
            high=np.concatenate([env.observation_space.high, extra_high]).astype(np.float32),
            dtype=np.float32,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        extra = self.env.kf_state_observation()
        obs = np.concatenate([np.asarray(observation, dtype=np.float32), extra]).astype(np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high).astype(np.float32)

    def set_curriculum_progress(self, progress: float) -> None:
        self.env.set_curriculum_progress(progress)

    def state_info(self) -> dict[str, float]:
        return self.env.state_info()


def make_robot_command_smoothing_env(
    smoother: str = "adaptive",
    obs_mode: str = "kf_state",
    kf_curriculum: str = "continuous",
    seed: int | None = None,
) -> gym.Env:
    """Build the decoupled v/w command-smoothing ablation env."""

    if smoother not in SMOOTHER_CHOICES:
        raise ValueError(f"Unsupported smoother: {smoother}")
    if obs_mode not in OBS_MODE_CHOICES:
        raise ValueError(f"Unsupported obs_mode: {obs_mode}")
    if kf_curriculum not in CURRICULUM_CHOICES:
        raise ValueError(f"Unsupported kf_curriculum: {kf_curriculum}")

    base_aug = obs_mode in {"prev_exec", "kf_state"}
    base_env = RobotDiffDriveComplexEnv(
        use_kf=False,
        aug_prev_action=base_aug,
        seed=seed,
    )
    env: gym.Env = CommandSmoothingWrapper(
        base_env,
        smoother=smoother,
        kf_curriculum=kf_curriculum,
    )
    if obs_mode == "kf_state":
        env = KFStateObservationWrapper(env)  # type: ignore[arg-type]
    return env
