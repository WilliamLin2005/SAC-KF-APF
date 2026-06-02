"""Adaptive bounded Kalman smoother for physical v/w commands."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CommandSmoothingContext:
    """State summary used by adaptive command smoothing."""

    distance_to_goal: float
    goal_radius: float
    obstacle_clearance: float
    boundary_clearance: float
    raw_command_delta_norm: float
    step_fraction: float


class AdaptiveKalmanCommandSmoother:
    """Linear KF command smoother with state-dependent Q/R scheduling."""

    def __init__(
        self,
        command_low: np.ndarray,
        command_high: np.ndarray,
        initial_covariance: float = 1.0,
        weak_process_noise_std: tuple[float, float] = (0.45, 0.70),
        weak_measurement_noise_std: tuple[float, float] = (0.08, 0.07),
        target_process_noise_std: tuple[float, float] = (0.12, 0.18),
        target_measurement_noise_std: tuple[float, float] = (0.35, 0.28),
        responsive_process_noise_std: tuple[float, float] = (0.38, 0.85),
        responsive_measurement_noise_std: tuple[float, float] = (0.08, 0.05),
        smooth_process_noise_std: tuple[float, float] = (0.07, 0.10),
        smooth_measurement_noise_std: tuple[float, float] = (0.55, 0.50),
        obstacle_response_clearance: float = 3.0,
        boundary_response_clearance: float = 2.0,
        goal_response_distance: float = 7.0,
        jitter_delta_threshold: float = 0.35,
        jitter_delta_full: float = 1.20,
        use_kf: bool = True,
    ) -> None:
        self.command_low = np.asarray(command_low, dtype=np.float64).reshape(2)
        self.command_high = np.asarray(command_high, dtype=np.float64).reshape(2)
        if np.any(self.command_low >= self.command_high):
            raise ValueError("command_low must be strictly smaller than command_high.")

        self.initial_covariance = float(initial_covariance)
        self.weak_process_noise_std = np.asarray(weak_process_noise_std, dtype=np.float64)
        self.weak_measurement_noise_std = np.asarray(weak_measurement_noise_std, dtype=np.float64)
        self.target_process_noise_std = np.asarray(target_process_noise_std, dtype=np.float64)
        self.target_measurement_noise_std = np.asarray(target_measurement_noise_std, dtype=np.float64)
        self.responsive_process_noise_std = np.asarray(responsive_process_noise_std, dtype=np.float64)
        self.responsive_measurement_noise_std = np.asarray(responsive_measurement_noise_std, dtype=np.float64)
        self.smooth_process_noise_std = np.asarray(smooth_process_noise_std, dtype=np.float64)
        self.smooth_measurement_noise_std = np.asarray(smooth_measurement_noise_std, dtype=np.float64)
        self.obstacle_response_clearance = float(obstacle_response_clearance)
        self.boundary_response_clearance = float(boundary_response_clearance)
        self.goal_response_distance = float(goal_response_distance)
        self.jitter_delta_threshold = float(jitter_delta_threshold)
        self.jitter_delta_full = float(jitter_delta_full)
        self.use_kf = bool(use_kf)

        self.F = np.eye(2, dtype=np.float64)
        self.H = np.eye(2, dtype=np.float64)
        self.I = np.eye(2, dtype=np.float64)
        self.curriculum_progress = 1.0
        self.x = np.zeros(2, dtype=np.float64)
        self.P = self.initial_covariance * np.eye(2, dtype=np.float64)
        self.last_process_noise_std = self.target_process_noise_std.copy()
        self.last_measurement_noise_std = self.target_measurement_noise_std.copy()
        self.last_responsive_factor = 0.0
        self.last_jitter_factor = 0.0
        self.reset()

    def reset(self) -> None:
        self.x = np.clip(np.zeros(2, dtype=np.float64), self.command_low, self.command_high)
        self.P = self.initial_covariance * np.eye(2, dtype=np.float64)
        self.last_process_noise_std = self.target_process_noise_std.copy()
        self.last_measurement_noise_std = self.target_measurement_noise_std.copy()
        self.last_responsive_factor = 0.0
        self.last_jitter_factor = 0.0

    def set_curriculum_progress(self, progress: float) -> None:
        self.curriculum_progress = float(np.clip(progress, 0.0, 1.0))

    def clip_command(self, command: np.ndarray) -> np.ndarray:
        return np.clip(
            np.asarray(command, dtype=np.float64).reshape(2),
            self.command_low,
            self.command_high,
        )

    def smooth(
        self,
        raw_command: np.ndarray,
        context: CommandSmoothingContext | None = None,
    ) -> np.ndarray:
        raw_command = self.clip_command(raw_command)
        if not self.use_kf:
            self.x = raw_command.astype(np.float64)
            return raw_command.astype(np.float32)

        process_std, measurement_std = self._noise_stds(context)
        self.last_process_noise_std = process_std.copy()
        self.last_measurement_noise_std = measurement_std.copy()
        q = np.diag(process_std**2)
        r = np.diag(measurement_std**2)

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + q

        innovation = raw_command - self.H @ self.x
        s = self.H @ self.P @ self.H.T + r
        k = self.P @ self.H.T @ np.linalg.inv(s)
        self.x = self.x + k @ innovation
        kh = k @ self.H
        self.P = (self.I - kh) @ self.P @ (self.I - kh).T + k @ r @ k.T
        self.x = self.clip_command(self.x)
        return self.x.astype(np.float32)

    def state_info(self) -> dict[str, float]:
        covariance_diag = np.diag(self.P)
        return {
            "kf_process_noise_v": float(self.last_process_noise_std[0]),
            "kf_process_noise_w": float(self.last_process_noise_std[1]),
            "kf_measurement_noise_v": float(self.last_measurement_noise_std[0]),
            "kf_measurement_noise_w": float(self.last_measurement_noise_std[1]),
            "kf_covariance_v": float(covariance_diag[0]),
            "kf_covariance_w": float(covariance_diag[1]),
            "kf_curriculum_progress": float(self.curriculum_progress),
            "kf_responsive_factor": float(self.last_responsive_factor),
            "kf_jitter_factor": float(self.last_jitter_factor),
        }

    def _noise_stds(
        self,
        context: CommandSmoothingContext | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        process_std = self._lerp(
            self.weak_process_noise_std,
            self.target_process_noise_std,
            self.curriculum_progress,
        )
        measurement_std = self._lerp(
            self.weak_measurement_noise_std,
            self.target_measurement_noise_std,
            self.curriculum_progress,
        )

        responsive_factor = 0.0
        jitter_factor = 0.0
        if context is not None:
            obstacle_factor = self._near_factor(
                context.obstacle_clearance,
                self.obstacle_response_clearance,
            )
            boundary_factor = self._near_factor(
                context.boundary_clearance,
                self.boundary_response_clearance,
            )
            goal_margin = max(0.0, context.distance_to_goal - context.goal_radius)
            goal_factor = self._near_factor(goal_margin, self.goal_response_distance)
            responsive_factor = max(obstacle_factor, boundary_factor, goal_factor)

            jitter_factor = np.clip(
                (context.raw_command_delta_norm - self.jitter_delta_threshold)
                / max(self.jitter_delta_full - self.jitter_delta_threshold, 1e-8),
                0.0,
                1.0,
            )
            jitter_factor *= 1.0 - responsive_factor

        if responsive_factor > 0.0:
            process_std = self._lerp(
                process_std,
                self.responsive_process_noise_std,
                responsive_factor,
            )
            measurement_std = self._lerp(
                measurement_std,
                self.responsive_measurement_noise_std,
                responsive_factor,
            )

        if jitter_factor > 0.0:
            smooth_strength = 0.75 * jitter_factor
            process_std = self._lerp(
                process_std,
                self.smooth_process_noise_std,
                smooth_strength,
            )
            measurement_std = self._lerp(
                measurement_std,
                self.smooth_measurement_noise_std,
                smooth_strength,
            )

        self.last_responsive_factor = float(responsive_factor)
        self.last_jitter_factor = float(jitter_factor)
        return process_std, measurement_std

    @staticmethod
    def _lerp(start: np.ndarray, end: np.ndarray, weight: float) -> np.ndarray:
        weight = float(np.clip(weight, 0.0, 1.0))
        return (1.0 - weight) * start + weight * end

    @staticmethod
    def _near_factor(clearance: float, response_distance: float) -> float:
        return float(
            np.clip(
                (response_distance - float(clearance)) / max(response_distance, 1e-8),
                0.0,
                1.0,
            )
        )
