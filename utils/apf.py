"""Artificial potential field helper used for replay-buffer warm-up."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class APFOutput:
    action: np.ndarray
    potential: float
    state_value: float


class APFPolicy:
    """Simple APF controller for the fixed 2D navigation task.

    This controller is deliberately kept outside the environment reward and SAC
    update. It only generates warm-up transitions and diagnostic values.
    """

    def __init__(
        self,
        attractive_gain: float = 1.0,
        repulsive_gain: float = 16.0,
        tangential_gain: float = 1.25,
        influence_radius: float = 12.0,
        max_force: float = 10.0,
    ) -> None:
        self.attractive_gain = float(attractive_gain)
        self.repulsive_gain = float(repulsive_gain)
        self.tangential_gain = float(tangential_gain)
        self.influence_radius = float(influence_radius)
        self.max_force = float(max_force)

    def act(self, env) -> APFOutput:
        position = np.asarray(env.position, dtype=np.float64)
        goal = np.asarray(env.goal, dtype=np.float64)
        goal_vec = goal - position
        goal_dist = float(np.linalg.norm(goal_vec))
        goal_dir = goal_vec / (goal_dist + 1e-8)

        attractive_force = self.attractive_gain * goal_dir
        repulsive_force = np.zeros(2, dtype=np.float64)
        tangential_force = np.zeros(2, dtype=np.float64)
        repulsive_potential = 0.0

        for obstacle in env.obstacles:
            center = np.asarray(obstacle["center"], dtype=np.float64)
            radius = float(obstacle["radius"])
            away = position - center
            center_dist = float(np.linalg.norm(away))
            signed_dist = center_dist - radius

            if signed_dist >= self.influence_radius:
                continue

            away_dir = away / (center_dist + 1e-8)
            safe_dist = max(signed_dist, 0.25)
            rep_scale = self.repulsive_gain * (
                (1.0 / safe_dist) - (1.0 / self.influence_radius)
            ) / (safe_dist**2)
            repulsive_force += rep_scale * away_dir
            repulsive_potential += 0.5 * self.repulsive_gain * (
                (1.0 / safe_dist) - (1.0 / self.influence_radius)
            ) ** 2

            # Tangential component helps APF roll around circular obstacles.
            tangent_ccw = np.array([-away_dir[1], away_dir[0]], dtype=np.float64)
            tangent_cw = -tangent_ccw
            tangent = tangent_ccw if np.dot(tangent_ccw, goal_dir) >= np.dot(tangent_cw, goal_dir) else tangent_cw
            proximity = max(0.0, 1.0 - safe_dist / self.influence_radius)
            tangential_force += self.tangential_gain * proximity * tangent

        force = attractive_force + repulsive_force + tangential_force
        force_norm = float(np.linalg.norm(force))
        if force_norm > self.max_force:
            force = force / force_norm * self.max_force
            force_norm = self.max_force

        action_norm = float(np.linalg.norm(force))
        if action_norm > 1e-8:
            action = force / max(1.0, action_norm)
        else:
            action = np.zeros(2, dtype=np.float64)
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        attractive_potential = 0.5 * self.attractive_gain * (goal_dist / env.diagonal) ** 2
        potential = float(attractive_potential + repulsive_potential)
        state_value = float(np.clip(1.0 / (1.0 + potential), 0.0, 1.0))
        return APFOutput(action=action, potential=potential, state_value=state_value)
