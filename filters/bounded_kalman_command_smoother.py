"""Bounded linear Kalman filter for physical velocity commands."""

from __future__ import annotations

import numpy as np


class BoundedKalmanCommandSmoother:
    """Smooth physical commands while respecting per-dimension bounds."""

    def __init__(
        self,
        command_low: np.ndarray,
        command_high: np.ndarray,
        process_noise_std: float = 0.15,
        measurement_noise_std: float = 0.30,
        initial_covariance: float = 1.0,
        use_kf: bool = True,
    ) -> None:
        self.command_low = np.asarray(command_low, dtype=np.float64).reshape(2)
        self.command_high = np.asarray(command_high, dtype=np.float64).reshape(2)
        if np.any(self.command_low >= self.command_high):
            raise ValueError("command_low must be strictly smaller than command_high.")

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
        self.x = np.clip(np.zeros(2, dtype=np.float64), self.command_low, self.command_high)
        self.P = self.initial_covariance * np.eye(2, dtype=np.float64)

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, measurement: np.ndarray) -> np.ndarray:
        z = self.clip_command(measurement).astype(np.float64)
        innovation = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ innovation
        KH = K @ self.H
        self.P = (self.I - KH) @ self.P @ (self.I - KH).T + K @ self.R @ K.T
        self.x = self.clip_command(self.x).astype(np.float64)
        return self.x.copy()

    def clip_command(self, command: np.ndarray) -> np.ndarray:
        return np.clip(
            np.asarray(command, dtype=np.float64).reshape(2),
            self.command_low,
            self.command_high,
        )

    def smooth(self, raw_command: np.ndarray) -> np.ndarray:
        raw_command = self.clip_command(raw_command)
        if not self.use_kf:
            return raw_command.astype(np.float32)

        self.predict()
        self.update(raw_command)
        return self.clip_command(self.x).astype(np.float32)
