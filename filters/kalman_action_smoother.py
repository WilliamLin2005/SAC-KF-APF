"""Online smoothers for generic two-channel normalized actions."""

from __future__ import annotations

import numpy as np


SMOOTHER_OBS_DIM = 8

VALID_SMOOTHER_TYPES = (
    "none",
    "current_kf",
    "rate_kf",
    "singer_kf",
    "ema",
    "second_order_lowpass",
)


def _clip_action(action: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(action, dtype=np.float64).reshape(2), -1.0, 1.0)


def _clip_features(features: np.ndarray) -> np.ndarray:
    return np.clip(features, -2.0, 2.0).astype(np.float32)


class IdentityActionSmoother:
    """Pass through a generic two-channel normalized action [a0, a1]."""

    smoother_type = "none"
    use_kf = False

    def __init__(self, dt: float = 1.0, use_kf: bool = False, **kwargs) -> None:
        del use_kf, kwargs
        self.dt = float(dt)
        self.use_kf = False

    def reset(self) -> None:
        pass

    def smooth(self, raw_action: np.ndarray) -> np.ndarray:
        return _clip_action(raw_action).astype(np.float32)

    def get_observation_features(self) -> np.ndarray:
        return np.zeros(SMOOTHER_OBS_DIM, dtype=np.float32)


class KalmanActionSmoother:
    """Smooth a generic two-channel normalized action [a0, a1].

    For ComplexNavEnv, [a0, a1] means [v_norm, omega_norm].
    For ContinuousNavEnv, the existing [vx, vy] meaning is preserved.
    """

    smoother_type = "current_kf"

    def __init__(
        self,
        dt: float = 1.0,
        process_noise_std: float = 0.15,
        measurement_noise_std: float = 0.3,
        initial_covariance: float = 1.0,
        use_kf: bool = True,
    ) -> None:
        self.dt = float(dt)
        self.process_noise_std = float(process_noise_std)
        self.measurement_noise_std = float(measurement_noise_std)
        self.initial_covariance = float(initial_covariance)
        self.use_kf = bool(use_kf)

        self.F = np.eye(2, dtype=np.float64)
        self.H = np.eye(2, dtype=np.float64)
        self.Q = (self.process_noise_std**2) * np.eye(2, dtype=np.float64)
        self.R = (self.measurement_noise_std**2) * np.eye(2, dtype=np.float64)
        self.I = np.eye(2, dtype=np.float64)

        self.x = np.zeros(2, dtype=np.float64)
        self.P = self.initial_covariance * np.eye(2, dtype=np.float64)
        self.reset()

    def reset(self) -> None:
        """Reset the filter state to zero action."""
        self.x = np.zeros(2, dtype=np.float64)
        self.P = self.initial_covariance * np.eye(2, dtype=np.float64)

    def predict(self) -> np.ndarray:
        """Run the constant-action KF prediction step."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, z: np.ndarray) -> np.ndarray:
        """Run the KF measurement update with raw action z."""
        z = np.asarray(z, dtype=np.float64).reshape(2)
        innovation = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ innovation
        # Joseph form improves numerical stability for long training runs.
        KH = K @ self.H
        self.P = (self.I - KH) @ self.P @ (self.I - KH).T + K @ self.R @ K.T
        return self.x.copy()

    def smooth(self, raw_action: np.ndarray) -> np.ndarray:
        """Return the executed action for a raw policy action."""
        raw_action = _clip_action(raw_action)
        if not self.use_kf:
            return raw_action.astype(np.float32)

        self.predict()
        self.update(raw_action)
        return np.clip(self.x, -1.0, 1.0).astype(np.float32)

    def get_observation_features(self) -> np.ndarray:
        features = np.asarray(
            [
                self.x[0],
                self.x[1],
                0.0,
                0.0,
                self.P[0, 0],
                self.P[1, 1],
                self.process_noise_std,
                self.measurement_noise_std,
            ],
            dtype=np.float32,
        )
        return _clip_features(features)


class ActionRateKalmanSmoother:
    """KF for a generic action [a0, a1] and its rate [da0, da1]."""

    smoother_type = "rate_kf"

    def __init__(
        self,
        dt: float = 1.0,
        velocity_process_noise_std: float = 0.05,
        rate_process_noise_std: float = 0.02,
        measurement_noise_std: float = 0.3,
        initial_covariance: float = 1.0,
        use_kf: bool = True,
    ) -> None:
        self.dt = float(dt)
        self.velocity_process_noise_std = float(velocity_process_noise_std)
        self.rate_process_noise_std = float(rate_process_noise_std)
        self.measurement_noise_std = float(measurement_noise_std)
        self.initial_covariance = float(initial_covariance)
        self.use_kf = bool(use_kf)

        self.F = np.asarray(
            [
                [1.0, 0.0, self.dt, 0.0],
                [0.0, 1.0, 0.0, self.dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.H = np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        self.Q = np.diag(
            [
                self.velocity_process_noise_std**2,
                self.velocity_process_noise_std**2,
                self.rate_process_noise_std**2,
                self.rate_process_noise_std**2,
            ]
        )
        self.R = (self.measurement_noise_std**2) * np.eye(2, dtype=np.float64)
        self.I = np.eye(4, dtype=np.float64)
        self.x = np.zeros(4, dtype=np.float64)
        self.P = self.initial_covariance * np.eye(4, dtype=np.float64)
        self.reset()

    def reset(self) -> None:
        self.x = np.zeros(4, dtype=np.float64)
        self.P = self.initial_covariance * np.eye(4, dtype=np.float64)

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64).reshape(2)
        innovation = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        KH = K @ self.H
        self.P = (self.I - KH) @ self.P @ (self.I - KH).T + K @ self.R @ K.T
        return self.x.copy()

    def smooth(self, raw_action: np.ndarray) -> np.ndarray:
        raw_action = _clip_action(raw_action)
        if not self.use_kf:
            return raw_action.astype(np.float32)

        self.predict()
        self.update(raw_action)
        return np.clip(self.x[:2], -1.0, 1.0).astype(np.float32)

    def get_observation_features(self) -> np.ndarray:
        features = np.asarray(
            [
                self.x[0],
                self.x[1],
                self.x[2],
                self.x[3],
                self.P[0, 0],
                self.P[1, 1],
                self.rate_process_noise_std,
                self.measurement_noise_std,
            ],
            dtype=np.float32,
        )
        return _clip_features(features)


class SingerActionRateKalmanSmoother(ActionRateKalmanSmoother):
    """Action-rate KF with first-order Gauss-Markov rate decay."""

    smoother_type = "singer_kf"

    def __init__(
        self,
        dt: float = 1.0,
        tau: float = 3.0,
        velocity_process_noise_std: float = 0.05,
        rate_process_noise_std: float = 0.03,
        measurement_noise_std: float = 0.3,
        initial_covariance: float = 1.0,
        use_kf: bool = True,
    ) -> None:
        self.tau = max(float(tau), 1e-6)
        super().__init__(
            dt=dt,
            velocity_process_noise_std=velocity_process_noise_std,
            rate_process_noise_std=rate_process_noise_std,
            measurement_noise_std=measurement_noise_std,
            initial_covariance=initial_covariance,
            use_kf=use_kf,
        )
        self.alpha = float(np.exp(-self.dt / self.tau))
        self.F[2, 2] = self.alpha
        self.F[3, 3] = self.alpha

    def get_observation_features(self) -> np.ndarray:
        features = np.asarray(
            [
                self.x[0],
                self.x[1],
                self.x[2],
                self.x[3],
                self.P[0, 0],
                self.P[1, 1],
                self.alpha,
                self.measurement_noise_std,
            ],
            dtype=np.float32,
        )
        return _clip_features(features)


class EMAActionSmoother:
    """EMA baseline for a generic two-channel normalized action [a0, a1]."""

    smoother_type = "ema"
    use_kf = False

    def __init__(self, dt: float = 1.0, beta: float = 0.85, use_kf: bool = False) -> None:
        del use_kf
        self.dt = float(dt)
        self.beta = float(beta)
        self.use_kf = False
        self.prev_action = np.zeros(2, dtype=np.float64)

    def reset(self) -> None:
        self.prev_action = np.zeros(2, dtype=np.float64)

    def smooth(self, raw_action: np.ndarray) -> np.ndarray:
        raw_action = _clip_action(raw_action)
        self.prev_action = self.beta * self.prev_action + (1.0 - self.beta) * raw_action
        return np.clip(self.prev_action, -1.0, 1.0).astype(np.float32)

    def get_observation_features(self) -> np.ndarray:
        features = np.asarray(
            [
                self.prev_action[0],
                self.prev_action[1],
                0.0,
                0.0,
                self.beta,
                self.beta,
                self.beta,
                0.0,
            ],
            dtype=np.float32,
        )
        return _clip_features(features)


class SecondOrderLowPassActionSmoother:
    """Second-order low-pass baseline for normalized action [a0, a1]."""

    smoother_type = "second_order_lowpass"
    use_kf = False

    def __init__(self, dt: float = 1.0, beta: float = 0.85, use_kf: bool = False) -> None:
        del use_kf
        self.dt = float(dt)
        self.beta = float(beta)
        self.use_kf = False
        self.stage1 = np.zeros(2, dtype=np.float64)
        self.stage2 = np.zeros(2, dtype=np.float64)

    def reset(self) -> None:
        self.stage1 = np.zeros(2, dtype=np.float64)
        self.stage2 = np.zeros(2, dtype=np.float64)

    def smooth(self, raw_action: np.ndarray) -> np.ndarray:
        raw_action = _clip_action(raw_action)
        self.stage1 = self.beta * self.stage1 + (1.0 - self.beta) * raw_action
        self.stage2 = self.beta * self.stage2 + (1.0 - self.beta) * self.stage1
        return np.clip(self.stage2, -1.0, 1.0).astype(np.float32)

    def get_observation_features(self) -> np.ndarray:
        memory_delta = self.stage1 - self.stage2
        features = np.asarray(
            [
                self.stage2[0],
                self.stage2[1],
                memory_delta[0],
                memory_delta[1],
                self.beta,
                self.beta,
                self.beta,
                0.0,
            ],
            dtype=np.float32,
        )
        return _clip_features(features)


def make_action_smoother(
    smoother_type: str = "current_kf",
    dt: float = 1.0,
    use_kf: bool = True,
    smoother_kwargs: dict | None = None,
):
    """Construct a smoother while passing only supported keyword arguments."""
    if smoother_type not in VALID_SMOOTHER_TYPES:
        raise ValueError(
            f"Unknown smoother_type {smoother_type!r}. "
            f"Expected one of: {', '.join(VALID_SMOOTHER_TYPES)}."
        )

    kwargs = dict(smoother_kwargs or {})
    if smoother_type == "none" or (smoother_type.endswith("_kf") and not use_kf):
        return IdentityActionSmoother(dt=dt)

    smoother_classes = {
        "current_kf": (
            KalmanActionSmoother,
            {"process_noise_std", "measurement_noise_std", "initial_covariance"},
        ),
        "rate_kf": (
            ActionRateKalmanSmoother,
            {
                "velocity_process_noise_std",
                "rate_process_noise_std",
                "measurement_noise_std",
                "initial_covariance",
            },
        ),
        "singer_kf": (
            SingerActionRateKalmanSmoother,
            {
                "tau",
                "velocity_process_noise_std",
                "rate_process_noise_std",
                "measurement_noise_std",
                "initial_covariance",
            },
        ),
        "ema": (EMAActionSmoother, {"beta"}),
        "second_order_lowpass": (SecondOrderLowPassActionSmoother, {"beta"}),
    }
    smoother_class, supported_keys = smoother_classes[smoother_type]
    filtered_kwargs = {key: value for key, value in kwargs.items() if key in supported_keys}
    return smoother_class(dt=dt, use_kf=use_kf, **filtered_kwargs)
