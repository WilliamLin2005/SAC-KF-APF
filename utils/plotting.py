"""Matplotlib plotting utilities for navigation evaluation outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def plot_path(
    env,
    trajectory: np.ndarray,
    smoothed_trajectory: np.ndarray,
    save_path: str | Path,
    title: str = "SAC + KF Navigation Path",
) -> None:
    ensure_parent_dir(save_path)
    trajectory = np.asarray(trajectory, dtype=np.float32)
    smoothed_trajectory = np.asarray(smoothed_trajectory, dtype=np.float32)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(title)
    ax.set_xlim(0, env.map_size)
    ax.set_ylim(0, env.map_size)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.3)

    for obstacle in env.obstacles:
        circle = plt.Circle(
            obstacle["center"],
            obstacle["radius"],
            color="tab:red",
            alpha=0.25,
            ec="tab:red",
            linewidth=1.5,
        )
        ax.add_patch(circle)

    ax.scatter(env.start[0], env.start[1], c="tab:green", s=80, label="Start", zorder=4)
    ax.scatter(env.goal[0], env.goal[1], c="tab:purple", s=100, marker="*", label="Goal", zorder=4)

    if len(trajectory) > 0:
        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color="tab:blue",
            linewidth=2.0,
            alpha=0.75,
            label="Executed trajectory",
        )
    if len(smoothed_trajectory) > 0:
        ax.plot(
            smoothed_trajectory[:, 0],
            smoothed_trajectory[:, 1],
            color="tab:orange",
            linewidth=2.5,
            alpha=0.95,
            label="B-spline smoothed",
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_reward_curve(
    rewards: np.ndarray,
    save_path: str | Path,
    title: str = "Evaluation Reward Curve",
) -> None:
    ensure_parent_dir(save_path)
    rewards = np.asarray(rewards, dtype=np.float32)
    steps = np.arange(len(rewards))
    cumulative = np.cumsum(rewards)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title(title)
    ax.plot(steps, rewards, label="Step reward", color="tab:blue", alpha=0.75)
    ax.plot(steps, cumulative, label="Cumulative reward", color="tab:orange", linewidth=2.0)
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_step_distance(
    distances: np.ndarray,
    save_path: str | Path,
    title: str = "Distance to Goal During Evaluation",
) -> None:
    ensure_parent_dir(save_path)
    distances = np.asarray(distances, dtype=np.float32)
    steps = np.arange(len(distances))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title(title)
    ax.plot(steps, distances, color="tab:green", linewidth=2.0)
    ax.set_xlabel("Step")
    ax.set_ylabel("Distance to goal")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_action_smoothing(
    raw_actions: np.ndarray,
    exec_actions: np.ndarray,
    raw_delta_norms: np.ndarray,
    exec_delta_norms: np.ndarray,
    save_path: str | Path,
    title: str = "Raw Action vs KF Executed Action",
) -> None:
    ensure_parent_dir(save_path)
    raw_actions = np.asarray(raw_actions, dtype=np.float32)
    exec_actions = np.asarray(exec_actions, dtype=np.float32)
    raw_delta_norms = np.asarray(raw_delta_norms, dtype=np.float32)
    exec_delta_norms = np.asarray(exec_delta_norms, dtype=np.float32)
    steps = np.arange(len(raw_actions))

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.suptitle(title)

    if len(raw_actions) > 0:
        axes[0].plot(steps, raw_actions[:, 0], label="vx_raw", color="tab:blue", alpha=0.7)
        axes[0].plot(steps, exec_actions[:, 0], label="vx_exec", color="tab:orange", linewidth=2.0)
        axes[1].plot(steps, raw_actions[:, 1], label="vy_raw", color="tab:blue", alpha=0.7)
        axes[1].plot(steps, exec_actions[:, 1], label="vy_exec", color="tab:orange", linewidth=2.0)
        axes[2].plot(steps, raw_delta_norms, label="||delta raw||", color="tab:red", alpha=0.75)
        axes[2].plot(steps, exec_delta_norms, label="||delta exec||", color="tab:green", linewidth=2.0)

    axes[0].set_ylabel("vx")
    axes[1].set_ylabel("vy")
    axes[2].set_ylabel("Delta norm")
    axes[2].set_xlabel("Step")
    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
