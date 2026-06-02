"""Sanity checks for complex [v, omega] smoothers and environment wiring."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.complex_nav_env import ComplexNavEnv
from envs.continuous_nav_env import ContinuousNavEnv
from filters.kalman_action_smoother import (
    SMOOTHER_OBS_DIM,
    VALID_SMOOTHER_TYPES,
    make_action_smoother,
)
from utils.apf_complex import ComplexAPFPolicy


def check_smoothers() -> None:
    assert SMOOTHER_OBS_DIM == 8
    raw_action = np.asarray([0.5, -0.5], dtype=np.float32)
    for smoother_type in VALID_SMOOTHER_TYPES:
        smoother = make_action_smoother(smoother_type=smoother_type, use_kf=True)
        exec_action = smoother.smooth(raw_action)
        assert exec_action.shape == (2,), smoother_type
        assert exec_action.dtype == np.float32, smoother_type
        assert np.all(exec_action >= -1.0) and np.all(exec_action <= 1.0), smoother_type
        features = smoother.get_observation_features()
        assert features.shape == (SMOOTHER_OBS_DIM,), smoother_type
        assert features.dtype == np.float32, smoother_type
        smoother.reset()


def check_continuous_env() -> None:
    env = ContinuousNavEnv(use_kf=True, seed=0)
    try:
        obs, _ = env.reset(seed=0)
        action = np.asarray([0.5, -0.2], dtype=np.float32)
        next_obs, _, _, _, _ = env.step(action)
        assert obs.shape == env.observation_space.shape
        assert next_obs.shape == env.observation_space.shape
    finally:
        env.close()


def check_complex_env() -> None:
    env = ComplexNavEnv(use_kf=True, seed=0, smoother_type="rate_kf")
    try:
        obs, _ = env.reset(seed=0)
        action = np.asarray([0.5, -0.2], dtype=np.float32)
        next_obs, _, _, _, info = env.step(action)
        assert obs.shape == env.observation_space.shape
        assert next_obs.shape == env.observation_space.shape
        assert env.observation_space.shape[0] == 17
        assert len(obs) == 17
        assert len(next_obs) == 17
        for key in (
            "heading",
            "smoother_type",
            "linear_velocity",
            "angular_velocity",
            "v_norm",
            "omega_norm",
            "smoother_features",
        ):
            assert key in info, key
        assert np.asarray(info["smoother_features"]).shape == (SMOOTHER_OBS_DIM,)
        assert info["v_norm"] >= 0.0
    finally:
        env.close()


def check_complex_apf() -> None:
    env = ComplexNavEnv(use_kf=False, seed=0, smoother_type="none")
    try:
        env.reset(seed=0)
        out = ComplexAPFPolicy().act(env)
        assert out.action.shape == (2,)
        assert out.action.dtype == np.float32
        assert 0.0 <= out.action[0] <= 1.0
        assert -1.0 <= out.action[1] <= 1.0
    finally:
        env.close()


def main() -> None:
    check_smoothers()
    check_continuous_env()
    check_complex_env()
    check_complex_apf()
    print("Complex [v, omega] smoother sanity check passed.")


if __name__ == "__main__":
    main()
