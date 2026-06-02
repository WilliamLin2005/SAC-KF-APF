"""Deterministic three-way comparison for robot v/w ablation models."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
from stable_baselines3 import SAC

from envs.robot_diffdrive_complex_env import RobotDiffDriveComplexEnv
from evaluate.robot_diffdrive_complex import (
    add_episode_metrics,
    episode_status,
    run_episode,
    set_eval_seed,
    summarize,
)


OUTPUT_DIR = Path("outputs/vw/ablation")
EPISODES = 50
SEED = 0
DEVICE = "cpu"


@dataclass(frozen=True)
class Condition:
    key: str
    label: str
    model_path: Path
    use_kf: bool
    aug_prev_action: bool = True
    color: str = "tab:blue"
    linestyle: str = "-"


CONDITIONS = [
    Condition(
        key="full_kf_in_loop",
        label="Full KF-in-loop",
        model_path=OUTPUT_DIR / "apf_sac_kf_in_loop_aug.zip",
        use_kf=True,
        color="tab:blue",
        linestyle="-",
    ),
    Condition(
        key="train_no_kf_eval_external_kf",
        label="Train no-KF + eval external KF",
        model_path=OUTPUT_DIR / "vw_apf_sac_train_no_kf_aug.zip",
        use_kf=True,
        color="tab:orange",
        linestyle="--",
    ),
    Condition(
        key="train_eval_no_kf",
        label="Train/eval no-KF",
        model_path=OUTPUT_DIR / "vw_apf_sac_train_no_kf_aug.zip",
        use_kf=False,
        color="tab:green",
        linestyle="-.",
    ),
]


def require_models() -> None:
    missing = sorted({str(cond.model_path) for cond in CONDITIONS if not cond.model_path.exists()})
    if missing:
        raise FileNotFoundError("Missing model file(s): " + ", ".join(missing))


def evaluate_condition(condition: Condition) -> tuple[RobotDiffDriveComplexEnv, list[dict], dict]:
    env = RobotDiffDriveComplexEnv(
        use_kf=condition.use_kf,
        aug_prev_action=condition.aug_prev_action,
        seed=SEED,
    )
    model = SAC.load(str(condition.model_path), env=env, device=DEVICE)
    episodes: list[dict] = []
    print(
        f"Evaluating {condition.label}: model={condition.model_path}, "
        f"use_kf={int(condition.use_kf)}, deterministic=1"
    )
    for episode_idx in range(EPISODES):
        episode_seed = SEED + episode_idx
        set_eval_seed(episode_seed)
        episode = run_episode(
            model=model,
            env=env,
            seed=episode_seed,
            deterministic=True,
        )
        episode["episode_seed"] = episode_seed
        episode["deterministic"] = True
        add_episode_metrics(episode, env)
        episodes.append(episode)

    summary = summarize(episodes)
    print(
        f"  return={summary['average_return']:.3f}, "
        f"success={summary['success_rate']:.3f}, "
        f"steps={summary['average_steps']:.3f}, "
        f"exec_delta={summary['mean_executed_command_delta']:.6f}"
    )
    return env, episodes, summary


def best_episode(episodes: list[dict]) -> dict:
    return max(episodes, key=lambda ep: ep["return"])


def draw_map(ax: plt.Axes, env: RobotDiffDriveComplexEnv) -> None:
    ax.set_xlim(0, env.map_size)
    ax.set_ylim(0, env.map_size)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.25)

    valid_size = env.map_size - 2.0 * env.boundary_margin
    ax.add_patch(
        Rectangle(
            (env.boundary_margin, env.boundary_margin),
            valid_size,
            valid_size,
            fill=False,
            linestyle="--",
            linewidth=1.4,
            edgecolor="0.25",
            label="Boundary margin",
        )
    )

    for rect in env.rectangles:
        rect_min = rect["min"]
        rect_max = rect["max"]
        ax.add_patch(
            Rectangle(
                rect_min,
                rect_max[0] - rect_min[0],
                rect_max[1] - rect_min[1],
                facecolor="tab:red",
                edgecolor="tab:red",
                alpha=0.20,
            )
        )
        inflated_min = rect_min - env.obstacle_margin
        ax.add_patch(
            Rectangle(
                inflated_min,
                rect_max[0] - rect_min[0] + 2.0 * env.obstacle_margin,
                rect_max[1] - rect_min[1] + 2.0 * env.obstacle_margin,
                fill=False,
                linestyle=":",
                linewidth=1.1,
                edgecolor="tab:red",
                alpha=0.70,
            )
        )

    for circle in env.circles:
        ax.add_patch(
            Circle(
                circle["center"],
                circle["radius"],
                facecolor="tab:red",
                edgecolor="tab:red",
                alpha=0.20,
            )
        )
        ax.add_patch(
            Circle(
                circle["center"],
                circle["radius"] + env.obstacle_margin,
                fill=False,
                linestyle=":",
                linewidth=1.1,
                edgecolor="tab:red",
                alpha=0.70,
            )
        )

    waypoints = np.asarray(env.waypoints, dtype=np.float32)
    ax.plot(waypoints[:, 0], waypoints[:, 1], "--", color="0.55", linewidth=1.0, label="APF waypoints")
    ax.scatter(env.start[0], env.start[1], c="tab:green", s=80, label="Start", zorder=4)
    ax.scatter(env.goal[0], env.goal[1], c="tab:purple", s=110, marker="*", label="Goal", zorder=4)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def plot_path(results: dict[str, dict], env: RobotDiffDriveComplexEnv, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("Deterministic Robot v/w Path Comparison")
    draw_map(ax, env)
    for condition in CONDITIONS:
        episode = results[condition.key]["selected"]
        trajectory = episode["trajectory"]
        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color=condition.color,
            linestyle=condition.linestyle,
            linewidth=2.2,
            label=(
                f"{condition.label} "
                f"({episode_status(episode)}, R={episode['return']:.1f}, steps={episode['steps']})"
            ),
        )
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def plot_reward(results: dict[str, dict], save_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=False)
    fig.suptitle("Deterministic Robot v/w Reward Comparison")
    for condition in CONDITIONS:
        rewards = results[condition.key]["selected"]["rewards"]
        steps = np.arange(len(rewards))
        axes[0].plot(
            steps,
            rewards,
            color=condition.color,
            linestyle=condition.linestyle,
            linewidth=1.4,
            alpha=0.85,
            label=condition.label,
        )
        axes[1].plot(
            steps,
            np.cumsum(rewards),
            color=condition.color,
            linestyle=condition.linestyle,
            linewidth=2.0,
            label=condition.label,
        )
    axes[0].set_ylabel("Step reward")
    axes[1].set_ylabel("Cumulative reward")
    axes[1].set_xlabel("Step")
    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def plot_distance(results: dict[str, dict], save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.set_title("Deterministic Robot v/w Distance-to-Goal Comparison")
    for condition in CONDITIONS:
        distances = results[condition.key]["selected"]["distances"]
        steps = np.arange(len(distances))
        ax.plot(
            steps,
            distances,
            color=condition.color,
            linestyle=condition.linestyle,
            linewidth=2.0,
            label=condition.label,
        )
    ax.set_xlabel("Step")
    ax.set_ylabel("Distance to goal")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def plot_commands(results: dict[str, dict], save_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 9.5), sharex=False)
    fig.suptitle("Deterministic Robot v/w Command Smoothing Comparison")
    for condition in CONDITIONS:
        episode = results[condition.key]["selected"]
        raw = episode["raw_commands"]
        executed = episode["exec_commands"]
        steps = np.arange(len(raw))
        if len(raw) == 0:
            continue
        axes[0].plot(
            steps,
            raw[:, 0],
            color=condition.color,
            linestyle=":",
            linewidth=1.2,
            alpha=0.65,
            label=f"{condition.label} v_raw",
        )
        axes[0].plot(
            steps,
            executed[:, 0],
            color=condition.color,
            linestyle=condition.linestyle,
            linewidth=2.0,
            label=f"{condition.label} v_exec",
        )
        axes[1].plot(
            steps,
            raw[:, 1],
            color=condition.color,
            linestyle=":",
            linewidth=1.2,
            alpha=0.65,
            label=f"{condition.label} w_raw",
        )
        axes[1].plot(
            steps,
            executed[:, 1],
            color=condition.color,
            linestyle=condition.linestyle,
            linewidth=2.0,
            label=f"{condition.label} w_exec",
        )
        axes[2].plot(
            steps,
            episode["raw_delta_norms"],
            color=condition.color,
            linestyle=":",
            linewidth=1.2,
            alpha=0.65,
            label=f"{condition.label} raw delta",
        )
        axes[2].plot(
            steps,
            episode["exec_delta_norms"],
            color=condition.color,
            linestyle=condition.linestyle,
            linewidth=2.0,
            label=f"{condition.label} exec delta",
        )
    axes[0].set_ylabel("v")
    axes[1].set_ylabel("w")
    axes[2].set_ylabel("Command delta norm")
    axes[2].set_xlabel("Step")
    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def write_summary_csv(results: dict[str, dict], save_path: Path) -> None:
    fieldnames = [
        "condition",
        "model_path",
        "use_kf",
        "aug_prev_action",
        "episodes",
        "deterministic",
        "average_return",
        "std_return",
        "average_steps",
        "success_rate",
        "collision_rate",
        "out_of_bounds_rate",
        "timeout_rate",
        "average_final_distance",
        "average_path_length",
        "average_min_obstacle_clearance",
        "worst_min_obstacle_clearance",
        "average_min_boundary_clearance",
        "worst_min_boundary_clearance",
        "mean_raw_command_delta",
        "mean_executed_command_delta",
        "smoothing_ratio",
        "selected_return",
        "selected_steps",
        "selected_final_distance",
        "selected_path_length",
        "selected_min_obstacle_clearance",
        "selected_min_boundary_clearance",
        "selected_status",
    ]
    with save_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for condition in CONDITIONS:
            item = results[condition.key]
            summary = item["summary"]
            selected = item["selected"]
            row = {
                "condition": condition.label,
                "model_path": condition.model_path,
                "use_kf": int(condition.use_kf),
                "aug_prev_action": int(condition.aug_prev_action),
                "episodes": EPISODES,
                "deterministic": 1,
                "selected_status": episode_status(selected),
            }
            row.update(summary)
            writer.writerow(row)


def write_readme(save_path: Path) -> None:
    content = """# Deterministic Robot v/w Ablation Comparison

This folder contains a deterministic three-way comparison for:

- Full KF-in-loop: `apf_sac_kf_in_loop_aug.zip`, eval with KF.
- Train no-KF + eval external KF: `vw_apf_sac_train_no_kf_aug.zip`, eval with KF.
- Train/eval no-KF: `vw_apf_sac_train_no_kf_aug.zip`, eval without KF.

Settings:

- Environment: `RobotDiffDriveComplexEnv`
- Episodes: `50`
- Seed range: `0..49`
- Deterministic policy: `1`
- Device: `cpu`
- Observation mode: augmented previous executed command

PNG outputs:

- `deterministic_compare_path.png`: executed trajectories on the same complex map.
- `deterministic_compare_reward_curve.png`: step reward and cumulative reward.
- `deterministic_compare_step_distance.png`: distance-to-goal curves.
- `deterministic_compare_command_smoothing.png`: raw/executed `[v,w]` commands and delta norms.

`deterministic_compare_metrics_summary.csv` stores the same aggregate metrics used to choose each displayed best-return episode.
"""
    save_path.write_text(content, encoding="utf-8")


def main() -> None:
    require_models()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    reference_env: RobotDiffDriveComplexEnv | None = None
    for condition in CONDITIONS:
        env, episodes, summary = evaluate_condition(condition)
        if reference_env is None:
            reference_env = env
        results[condition.key] = {
            "episodes": episodes,
            "summary": summary,
            "selected": best_episode(episodes),
        }
        if env is not reference_env:
            env.close()

    if reference_env is None:
        raise RuntimeError("No comparison conditions were evaluated.")

    plot_path(results, reference_env, OUTPUT_DIR / "deterministic_compare_path.png")
    plot_reward(results, OUTPUT_DIR / "deterministic_compare_reward_curve.png")
    plot_distance(results, OUTPUT_DIR / "deterministic_compare_step_distance.png")
    plot_commands(results, OUTPUT_DIR / "deterministic_compare_command_smoothing.png")
    write_summary_csv(results, OUTPUT_DIR / "deterministic_compare_metrics_summary.csv")
    write_readme(OUTPUT_DIR / "deterministic_compare_README.md")
    reference_env.close()

    print("\nSaved deterministic comparison outputs:")
    for name in [
        "deterministic_compare_path.png",
        "deterministic_compare_reward_curve.png",
        "deterministic_compare_step_distance.png",
        "deterministic_compare_command_smoothing.png",
        "deterministic_compare_metrics_summary.csv",
        "deterministic_compare_README.md",
    ]:
        print(f"  {OUTPUT_DIR / name}")


if __name__ == "__main__":
    main()
