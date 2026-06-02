"""Spline helpers used only for post-training trajectory visualization."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import splev, splprep


def smooth_trajectory_bspline(
    positions: np.ndarray,
    num_points: int = 300,
    smoothing: float = 2.0,
    degree: int = 3,
) -> np.ndarray:
    """Return a B-spline smoothed trajectory, falling back safely on failure."""
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape (T, 2)")
    if len(positions) < 4:
        return positions.copy()

    # splprep cannot handle long runs of identical points.
    keep = [0]
    for idx in range(1, len(positions)):
        if np.linalg.norm(positions[idx] - positions[keep[-1]]) > 1e-8:
            keep.append(idx)
    unique_positions = positions[keep]

    if len(unique_positions) < 4:
        return positions.copy()

    try:
        k = min(int(degree), len(unique_positions) - 1)
        tck, _ = splprep(
            [unique_positions[:, 0], unique_positions[:, 1]],
            s=float(smoothing),
            k=k,
        )
        u_new = np.linspace(0.0, 1.0, int(num_points))
        x_new, y_new = splev(u_new, tck)
        return np.column_stack([x_new, y_new]).astype(np.float32)
    except Exception:
        return positions.copy().astype(np.float32)
