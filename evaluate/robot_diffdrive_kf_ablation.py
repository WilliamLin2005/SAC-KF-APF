"""Evaluate v/w SAC policies with decoupled KF ablation wrappers."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle, Rectangle
from stable_baselines3 import SAC

from envs.robot_command_smoothing_wrappers import (
    CURRICULUM_CHOICES,
    OBS_MODE_CHOICES,
    SMOOTHER_CHOICES,
    make_robot_command_smoothing_env,
)
from train.robot_diffdrive_kf_ablation import default_model_path, default_run_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate v/w SAC with decoupled KF wrappers.")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--smoother", type=str, default="adaptive", choices=SMOOTHER_CHOICES)
    parser.add_argument("--obs-mode", type=str, default="kf_state", choices=OBS_MODE_CHOICES)
    parser.add_argument("--kf-curriculum", type=str, default="continuous", choices=CURRICULUM_CHOICES)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", type=int, default=0, choices=[0, 1])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save-dir", type=str, default="outputs")
    return parser.parse_args()


def set_eval_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_out_dir(save_dir: Path, smoother: str, obs_mode: str) -> Path:
    return save_dir / "vw" / f"{default_run_name(smoother, obs_mode)}_eval"


def run_episode(model: SAC, env, seed: int, deterministic: bool) -> dict:
    obs, info = env.reset(seed=seed)
    trajectory = [info["position"].copy()]
    thetas = [float(info["theta"])]
    distances = [float(info["distance_to_goal"])]
    raw_commands = []
    exec_commands = []
    rewards = []
    raw_delta_norms = []
    exec_delta_norms = []
    filter_mismatches = []
    kf_process_noise_v = []
    kf_process_noise_w = []
    kf_measurement_noise_v = []
    kf_measurement_noise_w = []
    kf_covariance_v = []
    kf_covariance_w = []

    terminated = False
    truncated = False
    final_info = info

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, final_info = env.step(action)
        trajectory.append(final_info["position"].copy())
        thetas.append(float(final_info["theta"]))
        distances.append(float(final_info["distance_to_goal"]))
        raw_commands.append(final_info["raw_command"].copy())
        exec_commands.append(final_info["executed_command"].copy())
        rewards.append(float(reward))
        raw_delta_norms.append(float(final_info.get("raw_command_delta_norm", 0.0)))
        exec_delta_norms.append(float(final_info.get("exec_command_delta_norm", 0.0)))
        filter_mismatches.append(float(final_info.get("filter_mismatch_norm", 0.0)))
        kf_process_noise_v.append(float(final_info.get("kf_process_noise_v", 0.0)))
        kf_process_noise_w.append(float(final_info.get("kf_process_noise_w", 0.0)))
        kf_measurement_noise_v.append(float(final_info.get("kf_measurement_noise_v", 0.0)))
        kf_measurement_noise_w.append(float(final_info.get("kf_measurement_noise_w", 0.0)))
        kf_covariance_v.append(float(final_info.get("kf_covariance_v", 0.0)))
        kf_covariance_w.append(float(final_info.get("kf_covariance_w", 0.0)))

    return {
        "trajectory": np.asarray(trajectory, dtype=np.float32),
        "thetas": np.asarray(thetas, dtype=np.float32),
        "distances": np.asarray(distances, dtype=np.float32),
        "raw_commands": np.asarray(raw_commands, dtype=np.float32).reshape(-1, 2),
        "exec_commands": np.asarray(exec_commands, dtype=np.float32).reshape(-1, 2),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "raw_delta_norms": np.asarray(raw_delta_norms, dtype=np.float32),
        "exec_delta_norms": np.asarray(exec_delta_norms, dtype=np.float32),
        "filter_mismatches": np.asarray(filter_mismatches, dtype=np.float32),
        "kf_process_noise_v": np.asarray(kf_process_noise_v, dtype=np.float32),
        "kf_process_noise_w": np.asarray(kf_process_noise_w, dtype=np.float32),
        "kf_measurement_noise_v": np.asarray(kf_measurement_noise_v, dtype=np.float32),
        "kf_measurement_noise_w": np.asarray(kf_measurement_noise_w, dtype=np.float32),
        "kf_covariance_v": np.asarray(kf_covariance_v, dtype=np.float32),
        "kf_covariance_w": np.asarray(kf_covariance_w, dtype=np.float32),
        "return": float(np.sum(rewards)),
        "steps": len(rewards),
        "success": bool(final_info.get("success", False)),
        "collision": bool(final_info.get("collision", False)),
        "timeout": bool(final_info.get("timeout", False)),
        "out_of_bounds": bool(final_info.get("out_of_bounds", False)),
    }


def episode_status(episode: dict) -> str:
    if episode["success"]:
        return "success"
    if episode["collision"]:
        return "collision"
    if episode["out_of_bounds"]:
        return "out_of_bounds"
    if episode["timeout"]:
        return "timeout"
    return "terminated"


def add_episode_metrics(episode: dict, env) -> None:
    base_env = env.unwrapped
    trajectory = np.asarray(episode["trajectory"], dtype=np.float32)
    raw_commands = episode["raw_commands"]
    exec_commands = episode["exec_commands"]
    raw_delta_norms = episode["raw_delta_norms"]
    exec_delta_norms = episode["exec_delta_norms"]
    mismatches = episode["filter_mismatches"]

    path_length = 0.0
    if len(trajectory) > 1:
        path_length = float(np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1)))

    obstacle_clearances = [base_env.obstacle_clearance(point) for point in trajectory]
    boundary_clearances = [base_env.boundary_clearance(point) for point in trajectory]

    episode["final_distance"] = float(episode["distances"][-1])
    episode["path_length"] = path_length
    episode["min_obstacle_clearance"] = (
        float(np.min(obstacle_clearances)) if obstacle_clearances else float("inf")
    )
    episode["min_boundary_clearance"] = (
        float(np.min(boundary_clearances)) if boundary_clearances else float("inf")
    )
    episode["mean_raw_command_norm"] = (
        float(np.mean(np.linalg.norm(raw_commands, axis=1))) if len(raw_commands) else 0.0
    )
    episode["mean_executed_command_norm"] = (
        float(np.mean(np.linalg.norm(exec_commands, axis=1))) if len(exec_commands) else 0.0
    )
    episode["mean_raw_command_delta"] = (
        float(np.mean(raw_delta_norms)) if len(raw_delta_norms) else 0.0
    )
    episode["mean_executed_command_delta"] = (
        float(np.mean(exec_delta_norms)) if len(exec_delta_norms) else 0.0
    )
    episode["episode_smoothing_ratio"] = episode["mean_executed_command_delta"] / (
        episode["mean_raw_command_delta"] + 1e-8
    )
    episode["mean_filter_mismatch"] = float(np.mean(mismatches)) if len(mismatches) else 0.0
    episode["max_filter_mismatch"] = float(np.max(mismatches)) if len(mismatches) else 0.0


def array_mean(episodes: list[dict], key: str) -> float:
    values = [ep[key] for ep in episodes if len(ep[key]) > 0]
    if not values:
        return 0.0
    return float(np.mean(np.concatenate(values)))


def summarize(episodes: list[dict]) -> dict[str, float]:
    returns = np.asarray([ep["return"] for ep in episodes], dtype=np.float32)
    steps = np.asarray([ep["steps"] for ep in episodes], dtype=np.float32)
    successes = np.asarray([ep["success"] for ep in episodes], dtype=np.float32)
    collisions = np.asarray([ep["collision"] for ep in episodes], dtype=np.float32)
    out_of_bounds = np.asarray([ep["out_of_bounds"] for ep in episodes], dtype=np.float32)
    timeouts = np.asarray([ep["timeout"] for ep in episodes], dtype=np.float32)
    final_distances = np.asarray([ep["final_distance"] for ep in episodes], dtype=np.float32)
    path_lengths = np.asarray([ep["path_length"] for ep in episodes], dtype=np.float32)
    obstacle_clearances = np.asarray([ep["min_obstacle_clearance"] for ep in episodes], dtype=np.float32)
    boundary_clearances = np.asarray([ep["min_boundary_clearance"] for ep in episodes], dtype=np.float32)
    selected = max(episodes, key=lambda ep: ep["return"])
    mean_raw_delta = array_mean(episodes, "raw_delta_norms")
    mean_exec_delta = array_mean(episodes, "exec_delta_norms")
    return {
        "episodes": float(len(episodes)),
        "average_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "average_steps": float(np.mean(steps)),
        "success_rate": float(np.mean(successes)),
        "collision_rate": float(np.mean(collisions)),
        "out_of_bounds_rate": float(np.mean(out_of_bounds)),
        "timeout_rate": float(np.mean(timeouts)),
        "average_final_distance": float(np.mean(final_distances)),
        "average_path_length": float(np.mean(path_lengths)),
        "average_min_obstacle_clearance": float(np.mean(obstacle_clearances)),
        "worst_min_obstacle_clearance": float(np.min(obstacle_clearances)),
        "average_min_boundary_clearance": float(np.mean(boundary_clearances)),
        "worst_min_boundary_clearance": float(np.min(boundary_clearances)),
        "mean_raw_command_delta": mean_raw_delta,
        "mean_executed_command_delta": mean_exec_delta,
        "smoothing_ratio": mean_exec_delta / (mean_raw_delta + 1e-8),
        "mean_filter_mismatch": array_mean(episodes, "filter_mismatches"),
        "mean_kf_process_noise_v": array_mean(episodes, "kf_process_noise_v"),
        "mean_kf_process_noise_w": array_mean(episodes, "kf_process_noise_w"),
        "mean_kf_measurement_noise_v": array_mean(episodes, "kf_measurement_noise_v"),
        "mean_kf_measurement_noise_w": array_mean(episodes, "kf_measurement_noise_w"),
        "mean_kf_covariance_v": array_mean(episodes, "kf_covariance_v"),
        "mean_kf_covariance_w": array_mean(episodes, "kf_covariance_w"),
        "selected_return": float(selected["return"]),
        "selected_steps": float(selected["steps"]),
        "selected_final_distance": float(selected["final_distance"]),
        "selected_path_length": float(selected["path_length"]),
        "selected_filter_mismatch": float(selected["mean_filter_mismatch"]),
    }


def draw_map(ax: plt.Axes, env) -> None:
    base_env = env.unwrapped
    ax.set_xlim(0, base_env.map_size)
    ax.set_ylim(0, base_env.map_size)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.3)

    valid_size = base_env.map_size - 2.0 * base_env.boundary_margin
    ax.add_patch(
        Rectangle(
            (base_env.boundary_margin, base_env.boundary_margin),
            valid_size,
            valid_size,
            fill=False,
            linestyle="--",
            linewidth=1.4,
            edgecolor="0.25",
            label="Boundary margin",
        )
    )
    for rect in base_env.rectangles:
        rect_min = rect["min"]
        rect_max = rect["max"]
        ax.add_patch(
            Rectangle(
                rect_min,
                rect_max[0] - rect_min[0],
                rect_max[1] - rect_min[1],
                facecolor="tab:red",
                edgecolor="tab:red",
                alpha=0.25,
            )
        )
        inflated_min = rect_min - base_env.obstacle_margin
        ax.add_patch(
            Rectangle(
                inflated_min,
                rect_max[0] - rect_min[0] + 2.0 * base_env.obstacle_margin,
                rect_max[1] - rect_min[1] + 2.0 * base_env.obstacle_margin,
                fill=False,
                linestyle=":",
                linewidth=1.2,
                edgecolor="tab:red",
                alpha=0.75,
            )
        )
    for circle in base_env.circles:
        ax.add_patch(
            Circle(
                circle["center"],
                circle["radius"],
                facecolor="tab:red",
                edgecolor="tab:red",
                alpha=0.25,
            )
        )
        ax.add_patch(
            Circle(
                circle["center"],
                circle["radius"] + base_env.obstacle_margin,
                fill=False,
                linestyle=":",
                linewidth=1.2,
                edgecolor="tab:red",
                alpha=0.75,
            )
        )
    waypoints = np.asarray(base_env.waypoints, dtype=np.float32)
    ax.plot(waypoints[:, 0], waypoints[:, 1], "--", color="0.55", linewidth=1.2, label="APF waypoints")
    ax.scatter(base_env.start[0], base_env.start[1], c="tab:green", s=80, label="Start", zorder=4)
    ax.scatter(base_env.goal[0], base_env.goal[1], c="tab:purple", s=110, marker="*", label="Goal", zorder=4)


def plot_path(env, episode: dict, save_path: Path) -> None:
    trajectory = episode["trajectory"]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("Robot v/w KF Ablation Path")
    draw_map(ax, env)
    ax.plot(trajectory[:, 0], trajectory[:, 1], color="tab:blue", linewidth=2.0, label="Executed center path")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_reward(episode: dict, save_path: Path) -> None:
    rewards = episode["rewards"]
    steps = np.arange(len(rewards))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_title("Robot v/w KF Ablation Reward")
    ax.plot(steps, rewards, color="tab:blue", alpha=0.55, linewidth=1.2, label="step reward")
    ax.plot(steps, np.cumsum(rewards), color="tab:orange", linewidth=2.0, label="cumulative reward")
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_distance(episode: dict, save_path: Path) -> None:
    distances = episode["distances"]
    steps = np.arange(len(distances))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_title("Robot v/w KF Ablation Distance to Goal")
    ax.plot(steps, distances, color="tab:green", linewidth=2.0)
    ax.set_xlabel("Step")
    ax.set_ylabel("Distance to goal")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_commands(episode: dict, save_path: Path) -> None:
    raw = episode["raw_commands"]
    executed = episode["exec_commands"]
    steps = np.arange(len(raw))
    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    fig.suptitle("Robot v/w KF Ablation Command Smoothing")
    if len(raw) > 0:
        axes[0].plot(steps, raw[:, 0], label="v_raw", color="tab:blue", alpha=0.7)
        axes[0].plot(steps, executed[:, 0], label="v_exec", color="tab:orange", linewidth=2.0)
        axes[1].plot(steps, raw[:, 1], label="w_raw", color="tab:blue", alpha=0.7)
        axes[1].plot(steps, executed[:, 1], label="w_exec", color="tab:orange", linewidth=2.0)
        axes[2].plot(steps, episode["raw_delta_norms"], label="||delta raw||", color="tab:red", alpha=0.75)
        axes[2].plot(steps, episode["exec_delta_norms"], label="||delta exec||", color="tab:green", linewidth=2.0)
        axes[3].plot(steps, episode["filter_mismatches"], label="||raw - executed||", color="tab:purple")
    axes[0].set_ylabel("v")
    axes[1].set_ylabel("w")
    axes[2].set_ylabel("Delta norm")
    axes[3].set_ylabel("Mismatch")
    axes[3].set_xlabel("Step")
    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def write_episode_csv(episodes: list[dict], save_path: Path) -> None:
    fieldnames = [
        "episode_index",
        "episode_seed",
        "deterministic",
        "return",
        "steps",
        "status",
        "success",
        "collision",
        "out_of_bounds",
        "timeout",
        "final_distance",
        "path_length",
        "min_obstacle_clearance",
        "min_boundary_clearance",
        "mean_raw_command_delta",
        "mean_executed_command_delta",
        "episode_smoothing_ratio",
        "mean_filter_mismatch",
        "max_filter_mismatch",
    ]
    with save_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for episode_idx, episode in enumerate(episodes):
            writer.writerow(
                {
                    "episode_index": episode_idx,
                    "episode_seed": episode["episode_seed"],
                    "deterministic": int(episode["deterministic"]),
                    "return": episode["return"],
                    "steps": episode["steps"],
                    "status": episode_status(episode),
                    "success": int(episode["success"]),
                    "collision": int(episode["collision"]),
                    "out_of_bounds": int(episode["out_of_bounds"]),
                    "timeout": int(episode["timeout"]),
                    "final_distance": episode["final_distance"],
                    "path_length": episode["path_length"],
                    "min_obstacle_clearance": episode["min_obstacle_clearance"],
                    "min_boundary_clearance": episode["min_boundary_clearance"],
                    "mean_raw_command_delta": episode["mean_raw_command_delta"],
                    "mean_executed_command_delta": episode["mean_executed_command_delta"],
                    "episode_smoothing_ratio": episode["episode_smoothing_ratio"],
                    "mean_filter_mismatch": episode["mean_filter_mismatch"],
                    "max_filter_mismatch": episode["max_filter_mismatch"],
                }
            )


def write_summary_csv(episodes: list[dict], save_path: Path) -> None:
    summary = summarize(episodes)
    with save_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in summary.items():
            writer.writerow({"metric": key, "value": value})


def write_readme(args: argparse.Namespace, episodes: list[dict], out_dir: Path, model_path: Path) -> None:
    summary = summarize(episodes)
    content = f"""# Robot v/w KF Ablation Eval

## Setup

- Model path: `{model_path}`
- smoother: `{args.smoother}`
- obs_mode: `{args.obs_mode}`
- kf_curriculum: `{args.kf_curriculum}`
- Episodes: `{args.episodes}`
- Seeds: `{args.seed}` to `{args.seed + args.episodes - 1}`
- Deterministic: `{args.deterministic}`
- Device: `{args.device}`

## Outputs

- `path_complex.png`: selected best-return trajectory.
- `eval_reward_curve_complex.png`: selected episode reward and cumulative reward.
- `eval_step_distance_complex.png`: selected episode distance-to-goal.
- `command_smoothing_complex.png`: raw/executed `[v,w]`, command deltas, and filter mismatch.
- `episode_metrics.csv`: per-episode metrics.
- `metrics_summary.csv`: aggregate metrics.

## Key Summary

- Success rate: `{summary['success_rate']:.3f}`
- Average return: `{summary['average_return']:.3f}`
- Average steps: `{summary['average_steps']:.3f}`
- Mean executed command delta: `{summary['mean_executed_command_delta']:.6f}`
- Smoothing ratio: `{summary['smoothing_ratio']:.6f}`
- Mean filter mismatch: `{summary['mean_filter_mismatch']:.6f}`
"""
    (out_dir / "README.md").write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    model_path = (
        Path(args.model_path)
        if args.model_path
        else default_model_path(save_dir, smoother=args.smoother, obs_mode=args.obs_mode)
    )
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else default_out_dir(save_dir, smoother=args.smoother, obs_mode=args.obs_mode)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    env = make_robot_command_smoothing_env(
        smoother=args.smoother,
        obs_mode=args.obs_mode,
        kf_curriculum=args.kf_curriculum,
        seed=args.seed,
    )
    if args.kf_curriculum == "continuous":
        env.set_curriculum_progress(1.0)
    model = SAC.load(str(model_path), env=env, device=args.device)

    episodes = []
    print(
        f"Evaluating v/w KF ablation: model={model_path}, smoother={args.smoother}, "
        f"obs_mode={args.obs_mode}, episodes={args.episodes}"
    )
    for episode_idx in range(args.episodes):
        episode_seed = args.seed + episode_idx
        set_eval_seed(episode_seed)
        episode = run_episode(
            model=model,
            env=env,
            seed=episode_seed,
            deterministic=bool(args.deterministic),
        )
        episode["episode_seed"] = episode_seed
        episode["deterministic"] = bool(args.deterministic)
        add_episode_metrics(episode, env)
        episodes.append(episode)
        print(
            f"Episode {episode_idx + 1}: seed={episode_seed}, "
            f"return={episode['return']:.2f}, steps={episode['steps']}, "
            f"status={episode_status(episode)}, mismatch={episode['mean_filter_mismatch']:.4f}"
        )

    selected = max(episodes, key=lambda ep: ep["return"])
    plot_path(env, selected, out_dir / "path_complex.png")
    plot_reward(selected, out_dir / "eval_reward_curve_complex.png")
    plot_distance(selected, out_dir / "eval_step_distance_complex.png")
    plot_commands(selected, out_dir / "command_smoothing_complex.png")
    write_episode_csv(episodes, out_dir / "episode_metrics.csv")
    write_summary_csv(episodes, out_dir / "metrics_summary.csv")
    write_readme(args, episodes, out_dir, model_path)
    env.close()

    summary = summarize(episodes)
    print(f"\nAverage return: {summary['average_return']:.3f}")
    print(f"Success rate: {summary['success_rate']:.3f}")
    print(f"Mean executed command delta: {summary['mean_executed_command_delta']:.6f}")
    print(f"Mean filter mismatch: {summary['mean_filter_mismatch']:.6f}")
    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
