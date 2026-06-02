"""Waypoint-guided APF policy for complex static maze warm-up."""

from __future__ import annotations

import numpy as np

from utils.apf import APFOutput


class ComplexAPFPolicy:
    """APF controller used only to populate the SAC replay buffer."""

    def __init__(
        self,
        waypoint_radius: float = 4.5,
        attractive_gain: float = 1.4,
        repulsive_gain: float = 18.0,
        tangential_gain: float = 2.0,
        influence_radius: float = 10.0,
        max_force: float = 8.0,
        linear_gain: float = 1.0,
        angular_gain: float = 2.0,
        min_forward_alignment: float = 0.0,
    ) -> None:
        self.waypoint_radius = float(waypoint_radius)
        self.attractive_gain = float(attractive_gain)
        self.repulsive_gain = float(repulsive_gain)
        self.tangential_gain = float(tangential_gain)
        self.influence_radius = float(influence_radius)
        self.max_force = float(max_force)
        self.linear_gain = float(linear_gain)
        self.angular_gain = float(angular_gain)
        self.min_forward_alignment = float(min_forward_alignment)
        self.current_waypoint_idx = 0

    def reset(self) -> None:
        self.current_waypoint_idx = 0

    def act(self, env) -> APFOutput:
        position = np.asarray(env.position, dtype=np.float64)
        target = self._current_target(env, position)
        target_vec = target - position
        target_dist = float(np.linalg.norm(target_vec))
        target_dir = target_vec / (target_dist + 1e-8)

        attractive_force = self.attractive_gain * target_dir
        repulsive_force = np.zeros(2, dtype=np.float64)
        tangential_force = np.zeros(2, dtype=np.float64)
        repulsive_potential = 0.0

        for circle in env.circles:
            away_dir, signed_dist = self._circle_away(position, circle)
            repulsive, tangential, potential = self._obstacle_force(
                away_dir=away_dir,
                signed_dist=signed_dist,
                target_dir=target_dir,
            )
            repulsive_force += repulsive
            tangential_force += tangential
            repulsive_potential += potential

        for rect in env.rectangles:
            away_dir, signed_dist = self._rectangle_away(position, rect)
            repulsive, tangential, potential = self._obstacle_force(
                away_dir=away_dir,
                signed_dist=signed_dist,
                target_dir=target_dir,
            )
            repulsive_force += repulsive
            tangential_force += tangential
            repulsive_potential += potential

        force = attractive_force + repulsive_force + tangential_force
        force_norm = float(np.linalg.norm(force))
        if force_norm > self.max_force:
            force = force / force_norm * self.max_force
            force_norm = self.max_force

        if force_norm > 1e-8:
            desired_heading = float(np.arctan2(force[1], force[0]))
            heading_error = self._wrap_angle(desired_heading - float(env.heading))
            speed_scale = float(np.clip(force_norm / max(self.max_force, 1e-6), 0.0, 1.0))
            alignment = float(np.cos(heading_error))
            forward_scale = max(self.min_forward_alignment, alignment)
            forward_scale = float(np.clip(forward_scale, 0.0, 1.0))
            v_norm = float(np.clip(self.linear_gain * speed_scale * forward_scale, 0.0, 1.0))
            omega_norm = float(np.clip(self.angular_gain * heading_error / np.pi, -1.0, 1.0))
        else:
            v_norm = 0.0
            omega_norm = 0.0
        action = np.asarray([v_norm, omega_norm], dtype=np.float32)

        attractive_potential = 0.5 * self.attractive_gain * (target_dist / env.diagonal) ** 2
        potential = float(attractive_potential + repulsive_potential)
        state_value = float(np.clip(1.0 / (1.0 + potential), 0.0, 1.0))
        return APFOutput(action=action, potential=potential, state_value=state_value)

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    def _current_target(self, env, position: np.ndarray) -> np.ndarray:
        while self.current_waypoint_idx < len(env.waypoints):
            waypoint = np.asarray(env.waypoints[self.current_waypoint_idx], dtype=np.float64)
            if np.linalg.norm(waypoint - position) > self.waypoint_radius:
                return waypoint
            self.current_waypoint_idx += 1
        return np.asarray(env.goal, dtype=np.float64)

    def _obstacle_force(
        self,
        away_dir: np.ndarray,
        signed_dist: float,
        target_dir: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        if signed_dist >= self.influence_radius:
            return np.zeros(2, dtype=np.float64), np.zeros(2, dtype=np.float64), 0.0

        safe_dist = max(float(signed_dist), 0.25)
        rep_scale = self.repulsive_gain * (
            (1.0 / safe_dist) - (1.0 / self.influence_radius)
        ) / (safe_dist**2)
        repulsive = rep_scale * away_dir

        tangent_ccw = np.asarray([-away_dir[1], away_dir[0]], dtype=np.float64)
        tangent_cw = -tangent_ccw
        tangent = tangent_ccw if np.dot(tangent_ccw, target_dir) >= np.dot(tangent_cw, target_dir) else tangent_cw
        proximity = max(0.0, 1.0 - safe_dist / self.influence_radius)
        tangential = self.tangential_gain * proximity * tangent

        potential = 0.5 * self.repulsive_gain * (
            (1.0 / safe_dist) - (1.0 / self.influence_radius)
        ) ** 2
        return repulsive, tangential, float(potential)

    @staticmethod
    def _circle_away(position: np.ndarray, circle: dict) -> tuple[np.ndarray, float]:
        center = np.asarray(circle["center"], dtype=np.float64)
        radius = float(circle["radius"])
        away = position - center
        center_dist = float(np.linalg.norm(away))
        away_dir = away / (center_dist + 1e-8)
        return away_dir, center_dist - radius

    @staticmethod
    def _rectangle_away(position: np.ndarray, rect: dict) -> tuple[np.ndarray, float]:
        rect_min = np.asarray(rect["min"], dtype=np.float64)
        rect_max = np.asarray(rect["max"], dtype=np.float64)
        closest = np.minimum(np.maximum(position, rect_min), rect_max)
        away = position - closest
        outside_distance = float(np.linalg.norm(away))
        if outside_distance > 1e-8:
            return away / outside_distance, outside_distance

        edge_distances = np.asarray(
            [
                position[0] - rect_min[0],
                rect_max[0] - position[0],
                position[1] - rect_min[1],
                rect_max[1] - position[1],
            ],
            dtype=np.float64,
        )
        edge_idx = int(np.argmin(edge_distances))
        normals = [
            np.asarray([-1.0, 0.0], dtype=np.float64),
            np.asarray([1.0, 0.0], dtype=np.float64),
            np.asarray([0.0, -1.0], dtype=np.float64),
            np.asarray([0.0, 1.0], dtype=np.float64),
        ]
        return normals[edge_idx], -float(edge_distances[edge_idx])
