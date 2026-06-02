"""Wrappers used by complex-environment baseline experiments."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.complex_nav_env import ComplexNavEnv


class ActionDeltaPenaltyWrapper(gym.Wrapper):
    """Subtract a raw-action delta penalty from the environment reward."""

    def __init__(self, env: gym.Env, penalty_weight: float = 0.2) -> None:
        super().__init__(env)
        self.penalty_weight = float(penalty_weight)
        self.prev_raw_action = np.zeros(self.action_space.shape, dtype=np.float32)

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        self.prev_raw_action = np.zeros(self.action_space.shape, dtype=np.float32)
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        raw_action = self._clip_action(action)
        prev_raw_action = self.prev_raw_action.copy()
        obs, reward, terminated, truncated, info = self.env.step(raw_action)

        raw_delta = raw_action - prev_raw_action
        penalty = self.penalty_weight * float(np.dot(raw_delta, raw_delta))
        adjusted_reward = float(reward) - penalty
        self.prev_raw_action = raw_action.astype(np.float32)

        info = dict(info)
        info["base_reward"] = float(reward)
        info["action_delta_penalty"] = float(penalty)
        info["action_delta_penalty_weight"] = self.penalty_weight
        return obs, adjusted_reward, terminated, truncated, info

    def _clip_action(self, action: np.ndarray) -> np.ndarray:
        return np.clip(
            np.asarray(action, dtype=np.float32).reshape(self.action_space.shape),
            self.action_space.low,
            self.action_space.high,
        ).astype(np.float32)


class LowPassActionWrapper(gym.Wrapper):
    """Execute low-pass filtered actions while exposing raw actions to SAC."""

    def __init__(self, env: gym.Env, alpha: float = 0.35) -> None:
        super().__init__(env)
        if not 0.0 < float(alpha) <= 1.0:
            raise ValueError("alpha must be in (0, 1].")
        self.alpha = float(alpha)
        self.prev_raw_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self.prev_exec_action = np.zeros(self.action_space.shape, dtype=np.float32)

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        self.prev_raw_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self.prev_exec_action = np.zeros(self.action_space.shape, dtype=np.float32)
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        raw_action = self._clip_action(action)
        prev_raw_action = self.prev_raw_action.copy()
        prev_exec_action = self.prev_exec_action.copy()
        exec_action = (
            self.alpha * raw_action + (1.0 - self.alpha) * prev_exec_action
        ).astype(np.float32)
        exec_action = np.clip(exec_action, self.action_space.low, self.action_space.high).astype(np.float32)

        obs, reward, terminated, truncated, info = self.env.step(exec_action)

        raw_delta = raw_action - prev_raw_action
        exec_delta = exec_action - prev_exec_action
        self.prev_raw_action = raw_action.astype(np.float32)
        self.prev_exec_action = exec_action.astype(np.float32)

        info = dict(info)
        info["lowpass_alpha"] = self.alpha
        info["raw_action"] = raw_action.copy()
        info["executed_action"] = exec_action.copy()
        info["raw_action_norm"] = float(np.linalg.norm(raw_action))
        info["exec_action_norm"] = float(np.linalg.norm(exec_action))
        info["raw_action_delta_norm"] = float(np.linalg.norm(raw_delta))
        info["exec_action_delta_norm"] = float(np.linalg.norm(exec_delta))
        info["lowpass_wrapped"] = True
        return obs, float(reward), terminated, truncated, info

    def _clip_action(self, action: np.ndarray) -> np.ndarray:
        return np.clip(
            np.asarray(action, dtype=np.float32).reshape(self.action_space.shape),
            self.action_space.low,
            self.action_space.high,
        ).astype(np.float32)


class DropPrevExecActionObsWrapper(gym.ObservationWrapper):
    """Remove the final prev_exec_action(2) observation features."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        if len(self.observation_space.shape) != 1 or self.observation_space.shape[0] < 2:
            raise ValueError("DropPrevExecActionObsWrapper expects a 1D observation space.")
        low = np.asarray(self.observation_space.low[:-2], dtype=np.float32)
        high = np.asarray(self.observation_space.high[:-2], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return np.asarray(observation[:-2], dtype=np.float32)


def make_complex_baseline_env(
    baseline: str,
    seed: int | None = None,
    action_penalty_weight: float = 0.2,
    lowpass_alpha: float = 0.35,
) -> gym.Env:
    """Build a ComplexNavEnv plus wrappers for a named baseline."""
    if baseline == "action_delta_penalty":
        return ActionDeltaPenaltyWrapper(
            ComplexNavEnv(use_kf=False, seed=seed),
            penalty_weight=action_penalty_weight,
        )
    if baseline == "lowpass_in_loop":
        return LowPassActionWrapper(
            ComplexNavEnv(use_kf=False, seed=seed),
            alpha=lowpass_alpha,
        )
    if baseline == "lowpass_eval_only":
        return LowPassActionWrapper(
            ComplexNavEnv(use_kf=False, seed=seed),
            alpha=lowpass_alpha,
        )
    if baseline == "gsde":
        return ComplexNavEnv(use_kf=False, seed=seed)
    if baseline == "kf_no_aug":
        return DropPrevExecActionObsWrapper(ComplexNavEnv(use_kf=True, seed=seed))
    if baseline == "kf":
        return ComplexNavEnv(use_kf=True, seed=seed)
    if baseline == "no_kf":
        return ComplexNavEnv(use_kf=False, seed=seed)
    raise ValueError(f"Unknown complex baseline: {baseline}")


def get_complex_base_env(env: gym.Env) -> ComplexNavEnv:
    """Return the underlying ComplexNavEnv from wrappers/Monitor wrappers."""
    unwrapped = env.unwrapped
    if not isinstance(unwrapped, ComplexNavEnv):
        raise TypeError(f"Expected ComplexNavEnv, got {type(unwrapped)!r}.")
    return unwrapped
