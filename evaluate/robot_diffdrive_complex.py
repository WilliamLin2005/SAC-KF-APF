"""Evaluate RobotDiffDriveComplexEnv policies."""

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

from envs.robot_diffdrive_complex_env import RobotDiffDriveComplexEnv
from train.robot_diffdrive_complex import ROBOT_VW_GROUP_DIR, default_model_path, run_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SAC on RobotDiffDriveComplexEnv.")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--use-kf", type=int, default=1, choices=[0, 1])
    parser.add_argument("--aug-prev-action", type=int, default=0, choices=[0, 1])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", type=int, default=0, choices=[0, 1])
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def set_eval_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_out_dir(use_kf: bool, aug_prev_action: bool) -> Path:
    return Path("outputs") / ROBOT_VW_GROUP_DIR / f"{run_name(use_kf, aug_prev_action)}_eval"


def run_episode(
    model: SAC,
    env: RobotDiffDriveComplexEnv,
    seed: int,
    deterministic: bool,
) -> dict:
    obs, info = env.reset(seed=seed)
    trajectory = [info["position"].copy()]
    thetas = [float(info["theta"])]
    distances = [float(info["distance_to_goal"])]
    raw_commands = []
    exec_commands = []
    rewards = []
    raw_delta_norms = []
    exec_delta_norms = []

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
        raw_delta_norms.append(final_info["raw_command_delta_norm"])
        exec_delta_norms.append(final_info["exec_command_delta_norm"])

    return {
        "trajectory": np.asarray(trajectory, dtype=np.float32),
        "thetas": np.asarray(thetas, dtype=np.float32),
        "distances": np.asarray(distances, dtype=np.float32),
        "raw_commands": np.asarray(raw_commands, dtype=np.float32).reshape(-1, 2),
        "exec_commands": np.asarray(exec_commands, dtype=np.float32).reshape(-1, 2),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "raw_delta_norms": np.asarray(raw_delta_norms, dtype=np.float32),
        "exec_delta_norms": np.asarray(exec_delta_norms, dtype=np.float32),
        "return": float(np.sum(rewards)),
        "steps": len(rewards),
        "success": bool(final_info.get("success", False)),
        "collision": bool(final_info.get("collision", False)),
        "timeout": bool(final_info.get("timeout", False)),
        "out_of_bounds": bool(final_info.get("out_of_bounds", False)),
    }


def add_episode_metrics(episode: dict, env: RobotDiffDriveComplexEnv) -> None:
    trajectory = np.asarray(episode["trajectory"], dtype=np.float32)
    raw_commands = episode["raw_commands"]
    exec_commands = episode["exec_commands"]
    raw_delta_norms = episode["raw_delta_norms"]
    exec_delta_norms = episode["exec_delta_norms"]

    if len(trajectory) > 1:
        path_segments = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
        path_length = float(np.sum(path_segments))
    else:
        path_length = 0.0

    obstacle_clearances = [env.obstacle_clearance(point) for point in trajectory]
    boundary_clearances = [env.boundary_clearance(point) for point in trajectory]
    mean_raw_delta = float(np.mean(raw_delta_norms)) if len(raw_delta_norms) else 0.0
    mean_exec_delta = float(np.mean(exec_delta_norms)) if len(exec_delta_norms) else 0.0

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
    episode["mean_raw_command_delta"] = mean_raw_delta
    episode["mean_executed_command_delta"] = mean_exec_delta
    episode["episode_smoothing_ratio"] = mean_exec_delta / (mean_raw_delta + 1e-8)


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
    raw_delta_all = np.concatenate([ep["raw_delta_norms"] for ep in episodes])
    exec_delta_all = np.concatenate([ep["exec_delta_norms"] for ep in episodes])
    mean_raw_delta = float(np.mean(raw_delta_all)) if len(raw_delta_all) else 0.0
    mean_exec_delta = float(np.mean(exec_delta_all)) if len(exec_delta_all) else 0.0
    selected = max(episodes, key=lambda ep: ep["return"])

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
        "selected_return": float(selected["return"]),
        "selected_steps": float(selected["steps"]),
        "selected_final_distance": float(selected["final_distance"]),
        "selected_path_length": float(selected["path_length"]),
        "selected_min_obstacle_clearance": float(selected["min_obstacle_clearance"]),
        "selected_min_boundary_clearance": float(selected["min_boundary_clearance"]),
    }


def plot_path(env: RobotDiffDriveComplexEnv, episode: dict, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory = episode["trajectory"]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("Robot v/w Complex Path")
    ax.set_xlim(0, env.map_size)
    ax.set_ylim(0, env.map_size)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.3)

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
                alpha=0.25,
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
                linewidth=1.2,
                edgecolor="tab:red",
                alpha=0.75,
            )
        )

    for circle in env.circles:
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
                circle["radius"] + env.obstacle_margin,
                fill=False,
                linestyle=":",
                linewidth=1.2,
                edgecolor="tab:red",
                alpha=0.75,
            )
        )

    waypoints = np.asarray(env.waypoints, dtype=np.float32)
    ax.plot(waypoints[:, 0], waypoints[:, 1], "--", color="0.55", linewidth=1.2, label="APF waypoints")
    ax.scatter(env.start[0], env.start[1], c="tab:green", s=80, label="Start", zorder=4)
    ax.scatter(env.goal[0], env.goal[1], c="tab:purple", s=110, marker="*", label="Goal", zorder=4)
    if len(trajectory) > 0:
        ax.plot(trajectory[:, 0], trajectory[:, 1], color="tab:blue", linewidth=2.0, label="Executed center path")
        if env.robot_radius > 0.0:
            stride = max(1, len(trajectory) // 12)
            for point in trajectory[::stride]:
                ax.add_patch(
                    Circle(
                        point,
                        env.robot_radius,
                        fill=False,
                        edgecolor="tab:blue",
                        linewidth=0.8,
                        alpha=0.35,
                    )
                )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_reward(episode: dict, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    rewards = episode["rewards"]
    steps = np.arange(len(rewards))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_title("Robot v/w Reward")
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
    save_path.parent.mkdir(parents=True, exist_ok=True)
    distances = episode["distances"]
    steps = np.arange(len(distances))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_title("Robot v/w Distance to Goal")
    ax.plot(steps, distances, color="tab:green", linewidth=2.0)
    ax.set_xlabel("Step")
    ax.set_ylabel("Distance to goal")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_commands(episode: dict, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    raw = episode["raw_commands"]
    executed = episode["exec_commands"]
    steps = np.arange(len(raw))
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.suptitle("Robot v/w Raw Command vs Executed Command")
    if len(raw) > 0:
        axes[0].plot(steps, raw[:, 0], label="v_raw", color="tab:blue", alpha=0.7)
        axes[0].plot(steps, executed[:, 0], label="v_exec", color="tab:orange", linewidth=2.0)
        axes[1].plot(steps, raw[:, 1], label="w_raw", color="tab:blue", alpha=0.7)
        axes[1].plot(steps, executed[:, 1], label="w_exec", color="tab:orange", linewidth=2.0)
        axes[2].plot(steps, episode["raw_delta_norms"], label="||delta raw||", color="tab:red", alpha=0.75)
        axes[2].plot(steps, episode["exec_delta_norms"], label="||delta exec||", color="tab:green", linewidth=2.0)
    axes[0].set_ylabel("v")
    axes[1].set_ylabel("w")
    axes[2].set_ylabel("Delta norm")
    axes[2].set_xlabel("Step")
    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def write_episode_csv(episodes: list[dict], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
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
        "mean_raw_command_norm",
        "mean_executed_command_norm",
        "mean_raw_command_delta",
        "mean_executed_command_delta",
        "episode_smoothing_ratio",
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
                    "mean_raw_command_norm": episode["mean_raw_command_norm"],
                    "mean_executed_command_norm": episode["mean_executed_command_norm"],
                    "mean_raw_command_delta": episode["mean_raw_command_delta"],
                    "mean_executed_command_delta": episode["mean_executed_command_delta"],
                    "episode_smoothing_ratio": episode["episode_smoothing_ratio"],
                }
            )


def write_summary_csv(episodes: list[dict], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize(episodes)
    with save_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in summary.items():
            writer.writerow({"metric": key, "value": value})


def write_readme(
    args: argparse.Namespace,
    episodes: list[dict],
    out_dir: Path,
    model_path: Path,
    env: RobotDiffDriveComplexEnv,
) -> None:
    summary = summarize(episodes)
    content = f"""# Robot v/w Complex Eval

## Setup

- Model path: `{model_path}`
- use_kf: `{args.use_kf}`
- aug_prev_action: `{args.aug_prev_action}`
- Episodes: `{args.episodes}`
- Seeds: `{args.seed}` to `{args.seed + args.episodes - 1}`
- Deterministic: `{args.deterministic}`
- Device: `{args.device}`
- Robot radius: `{env.robot_radius}`
- Obstacle margin: `{env.obstacle_margin}`
- Boundary margin: `{env.boundary_margin}`

## Outputs

- `path_complex.png`: selected best-return trajectory with footprint and inflated obstacles.
- `eval_reward_curve_complex.png`: selected episode reward and cumulative reward.
- `eval_step_distance_complex.png`: selected episode distance-to-goal.
- `command_smoothing_complex.png`: raw/executed `[v,w]` commands and command deltas.
- `episode_metrics.csv`: per-episode metrics.
- `metrics_summary.csv`: aggregate metrics.

## Key Summary

- Success rate: `{summary['success_rate']:.3f}`
- Collision rate: `{summary['collision_rate']:.3f}`
- Out-of-bounds rate: `{summary['out_of_bounds_rate']:.3f}`
- Average return: `{summary['average_return']:.3f}`
- Average steps: `{summary['average_steps']:.3f}`
- Average final distance: `{summary['average_final_distance']:.6f}`
- Average min obstacle clearance: `{summary['average_min_obstacle_clearance']:.6f}`
- Average min boundary clearance: `{summary['average_min_boundary_clearance']:.6f}`
- Mean executed command delta: `{summary['mean_executed_command_delta']:.6f}`
- Smoothing ratio: `{summary['smoothing_ratio']:.6f}`
"""
    (out_dir / "README.md").write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    use_kf = bool(args.use_kf)
    aug_prev_action = bool(args.aug_prev_action)
    model_path = (
        Path(args.model_path)
        if args.model_path
        else default_model_path(Path("outputs"), use_kf=use_kf, aug_prev_action=aug_prev_action)
    )
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir(use_kf, aug_prev_action)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = RobotDiffDriveComplexEnv(use_kf=use_kf, aug_prev_action=aug_prev_action, seed=args.seed)
    model = SAC.load(str(model_path), env=env, device=args.device)
    episodes = []
    print(
        f"Evaluating robot v/w: model={model_path}, use_kf={int(use_kf)}, "
        f"aug_prev_action={int(aug_prev_action)}, episodes={args.episodes}"
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
            f"status={episode_status(episode)}"
        )

    selected = max(episodes, key=lambda ep: ep["return"])
    plot_path(env, selected, out_dir / "path_complex.png")
    plot_reward(selected, out_dir / "eval_reward_curve_complex.png")
    plot_distance(selected, out_dir / "eval_step_distance_complex.png")
    plot_commands(selected, out_dir / "command_smoothing_complex.png")
    write_episode_csv(episodes, out_dir / "episode_metrics.csv")
    write_summary_csv(episodes, out_dir / "metrics_summary.csv")
    write_readme(args, episodes, out_dir, model_path, env)
    env.close()

    summary = summarize(episodes)
    print(f"\nAverage return: {summary['average_return']:.3f}")
    print(f"Average steps: {summary['average_steps']:.3f}")
    print(f"Success rate: {summary['success_rate']:.3f}")
    print(f"Collision rate: {summary['collision_rate']:.3f}")
    print(f"Out-of-bounds rate: {summary['out_of_bounds_rate']:.3f}")
    print(f"Mean executed command delta: {summary['mean_executed_command_delta']:.6f}")
    print(f"Smoothing ratio: {summary['smoothing_ratio']:.6f}")
    print(f"Saved robot v/w evaluation outputs to: {out_dir}")


if __name__ == "__main__":
    main()
