"""Evaluate fixed-KF v/w SAC policies with two-phase docking reward."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC

from envs.robot_two_phase_reward_wrapper import make_robot_two_phase_env
from evaluate.robot_diffdrive_kf_ablation import (
    episode_status,
    plot_commands,
    plot_distance,
    plot_path,
    plot_reward,
)
from train.robot_diffdrive_kf_two_phase import DEFAULT_MODEL_PATH, DEFAULT_RUN_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fixed-KF two-phase v/w SAC.")
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", type=int, default=0, choices=[0, 1])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save-dir", type=str, default="outputs")
    parser.add_argument("--slowdown-radius", type=float, default=8.0)
    parser.add_argument("--docking-entry-bonus", type=float, default=25.0)
    parser.add_argument("--docking-progress-weight", type=float, default=2.0)
    parser.add_argument("--docking-distance-weight", type=float, default=0.5)
    parser.add_argument("--linear-speed-weight", type=float, default=1.5)
    parser.add_argument("--angular-speed-weight", type=float, default=0.8)
    parser.add_argument("--heading-penalty-weight", type=float, default=0.05)
    parser.add_argument("--inside-goal-fast-penalty-weight", type=float, default=5.0)
    parser.add_argument("--success-linear-threshold", type=float, default=0.25)
    parser.add_argument("--success-angular-threshold", type=float, default=0.25)
    return parser.parse_args()


def set_eval_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_out_dir(save_dir: Path) -> Path:
    return save_dir / "vw" / f"{DEFAULT_RUN_NAME}_eval"


def make_env(seed: int | None, args: argparse.Namespace):
    return make_robot_two_phase_env(
        smoother="fixed",
        obs_mode="kf_state",
        kf_curriculum="none",
        seed=seed,
        slowdown_radius=args.slowdown_radius,
        docking_entry_bonus=args.docking_entry_bonus,
        docking_progress_weight=args.docking_progress_weight,
        docking_distance_weight=args.docking_distance_weight,
        linear_speed_weight=args.linear_speed_weight,
        angular_speed_weight=args.angular_speed_weight,
        heading_penalty_weight=args.heading_penalty_weight,
        inside_goal_fast_penalty_weight=args.inside_goal_fast_penalty_weight,
        success_linear_threshold=args.success_linear_threshold,
        success_angular_threshold=args.success_angular_threshold,
    )


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
    phase_ids = []
    docking_entries = []
    inside_goal_fast = []
    linear_speed_penalties = []
    angular_speed_penalties = []
    heading_penalties = []
    docking_distance_penalties = []

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
        phase_ids.append(float(final_info.get("reward_phase_id", 0.0)))
        docking_entries.append(float(final_info.get("entered_docking_zone", False)))
        inside_goal_fast.append(float(final_info.get("inside_goal_too_fast", False)))
        linear_speed_penalties.append(float(final_info.get("terminal_linear_speed_penalty", 0.0)))
        angular_speed_penalties.append(float(final_info.get("terminal_angular_speed_penalty", 0.0)))
        heading_penalties.append(float(final_info.get("terminal_heading_penalty", 0.0)))
        docking_distance_penalties.append(float(final_info.get("docking_distance_penalty", 0.0)))

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
        "phase_ids": np.asarray(phase_ids, dtype=np.float32),
        "docking_entries": np.asarray(docking_entries, dtype=np.float32),
        "inside_goal_fast": np.asarray(inside_goal_fast, dtype=np.float32),
        "linear_speed_penalties": np.asarray(linear_speed_penalties, dtype=np.float32),
        "angular_speed_penalties": np.asarray(angular_speed_penalties, dtype=np.float32),
        "heading_penalties": np.asarray(heading_penalties, dtype=np.float32),
        "docking_distance_penalties": np.asarray(docking_distance_penalties, dtype=np.float32),
        "return": float(np.sum(rewards)),
        "steps": len(rewards),
        "success": bool(final_info.get("success", False)),
        "collision": bool(final_info.get("collision", False)),
        "timeout": bool(final_info.get("timeout", False)),
        "out_of_bounds": bool(final_info.get("out_of_bounds", False)),
        "base_success": bool(final_info.get("base_success", False)),
    }


def add_episode_metrics(episode: dict, env) -> None:
    base_env = env.unwrapped
    trajectory = np.asarray(episode["trajectory"], dtype=np.float32)
    raw_commands = episode["raw_commands"]
    exec_commands = episode["exec_commands"]
    raw_delta_norms = episode["raw_delta_norms"]
    exec_delta_norms = episode["exec_delta_norms"]
    mismatches = episode["filter_mismatches"]
    phase_ids = episode["phase_ids"]
    last10_exec = exec_commands[-10:] if len(exec_commands) else np.zeros((0, 2), dtype=np.float32)
    last10_exec_delta = exec_delta_norms[-10:] if len(exec_delta_norms) else np.zeros(0, dtype=np.float32)

    path_length = 0.0
    if len(trajectory) > 1:
        path_length = float(np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1)))

    obstacle_clearances = [base_env.obstacle_clearance(point) for point in trajectory]
    boundary_clearances = [base_env.boundary_clearance(point) for point in trajectory]
    episode["final_distance"] = float(episode["distances"][-1])
    episode["path_length"] = path_length
    episode["min_obstacle_clearance"] = float(np.min(obstacle_clearances)) if obstacle_clearances else float("inf")
    episode["min_boundary_clearance"] = float(np.min(boundary_clearances)) if boundary_clearances else float("inf")
    episode["mean_raw_command_delta"] = float(np.mean(raw_delta_norms)) if len(raw_delta_norms) else 0.0
    episode["mean_executed_command_delta"] = float(np.mean(exec_delta_norms)) if len(exec_delta_norms) else 0.0
    episode["episode_smoothing_ratio"] = episode["mean_executed_command_delta"] / (
        episode["mean_raw_command_delta"] + 1e-8
    )
    episode["mean_filter_mismatch"] = float(np.mean(mismatches)) if len(mismatches) else 0.0
    episode["terminal_v_exec"] = float(exec_commands[-1, 0]) if len(exec_commands) else 0.0
    episode["terminal_abs_w_exec"] = float(abs(exec_commands[-1, 1])) if len(exec_commands) else 0.0
    episode["mean_last10_v_exec"] = float(np.mean(last10_exec[:, 0])) if len(last10_exec) else 0.0
    episode["mean_last10_abs_w_exec"] = float(np.mean(np.abs(last10_exec[:, 1]))) if len(last10_exec) else 0.0
    episode["mean_last10_exec_delta"] = float(np.mean(last10_exec_delta)) if len(last10_exec_delta) else 0.0
    episode["docking_fraction"] = float(np.mean(phase_ids)) if len(phase_ids) else 0.0
    episode["docking_entry_count"] = float(np.sum(episode["docking_entries"]))
    episode["inside_goal_fast_steps"] = float(np.sum(episode["inside_goal_fast"]))
    episode["mean_terminal_speed_penalty"] = float(
        np.mean(episode["linear_speed_penalties"] + episode["angular_speed_penalties"])
    ) if len(episode["linear_speed_penalties"]) else 0.0
    episode["mean_docking_distance_penalty"] = float(
        np.mean(episode["docking_distance_penalties"])
    ) if len(episode["docking_distance_penalties"]) else 0.0


def array_mean(episodes: list[dict], key: str) -> float:
    values = [ep[key] for ep in episodes if len(ep[key]) > 0]
    if not values:
        return 0.0
    return float(np.mean(np.concatenate(values)))


def scalar_mean(episodes: list[dict], key: str) -> float:
    values = [float(ep[key]) for ep in episodes]
    return float(np.mean(values)) if values else 0.0


def summarize(episodes: list[dict]) -> dict[str, float]:
    returns = np.asarray([ep["return"] for ep in episodes], dtype=np.float32)
    successes = np.asarray([ep["success"] for ep in episodes], dtype=np.float32)
    collisions = np.asarray([ep["collision"] for ep in episodes], dtype=np.float32)
    out_of_bounds = np.asarray([ep["out_of_bounds"] for ep in episodes], dtype=np.float32)
    timeouts = np.asarray([ep["timeout"] for ep in episodes], dtype=np.float32)
    selected = max(episodes, key=lambda ep: ep["return"])
    mean_raw_delta = array_mean(episodes, "raw_delta_norms")
    mean_exec_delta = array_mean(episodes, "exec_delta_norms")
    return {
        "episodes": float(len(episodes)),
        "average_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "average_steps": scalar_mean(episodes, "steps"),
        "success_rate": float(np.mean(successes)),
        "collision_rate": float(np.mean(collisions)),
        "out_of_bounds_rate": float(np.mean(out_of_bounds)),
        "timeout_rate": float(np.mean(timeouts)),
        "average_final_distance": scalar_mean(episodes, "final_distance"),
        "average_path_length": scalar_mean(episodes, "path_length"),
        "average_min_obstacle_clearance": scalar_mean(episodes, "min_obstacle_clearance"),
        "worst_min_obstacle_clearance": float(min(ep["min_obstacle_clearance"] for ep in episodes)),
        "average_min_boundary_clearance": scalar_mean(episodes, "min_boundary_clearance"),
        "worst_min_boundary_clearance": float(min(ep["min_boundary_clearance"] for ep in episodes)),
        "mean_raw_command_delta": mean_raw_delta,
        "mean_executed_command_delta": mean_exec_delta,
        "smoothing_ratio": mean_exec_delta / (mean_raw_delta + 1e-8),
        "mean_filter_mismatch": array_mean(episodes, "filter_mismatches"),
        "mean_docking_fraction": scalar_mean(episodes, "docking_fraction"),
        "mean_docking_entry_count": scalar_mean(episodes, "docking_entry_count"),
        "mean_inside_goal_fast_steps": scalar_mean(episodes, "inside_goal_fast_steps"),
        "mean_terminal_speed_penalty": scalar_mean(episodes, "mean_terminal_speed_penalty"),
        "mean_docking_distance_penalty": scalar_mean(episodes, "mean_docking_distance_penalty"),
        "mean_terminal_v_exec": scalar_mean(episodes, "terminal_v_exec"),
        "mean_terminal_abs_w_exec": scalar_mean(episodes, "terminal_abs_w_exec"),
        "mean_last10_v_exec": scalar_mean(episodes, "mean_last10_v_exec"),
        "mean_last10_abs_w_exec": scalar_mean(episodes, "mean_last10_abs_w_exec"),
        "mean_last10_exec_delta": scalar_mean(episodes, "mean_last10_exec_delta"),
        "selected_return": float(selected["return"]),
        "selected_steps": float(selected["steps"]),
        "selected_final_distance": float(selected["final_distance"]),
        "selected_path_length": float(selected["path_length"]),
    }


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
        "docking_fraction",
        "docking_entry_count",
        "inside_goal_fast_steps",
        "mean_terminal_speed_penalty",
        "terminal_v_exec",
        "terminal_abs_w_exec",
        "mean_last10_v_exec",
        "mean_last10_abs_w_exec",
        "mean_last10_exec_delta",
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
                    "docking_fraction": episode["docking_fraction"],
                    "docking_entry_count": episode["docking_entry_count"],
                    "inside_goal_fast_steps": episode["inside_goal_fast_steps"],
                    "mean_terminal_speed_penalty": episode["mean_terminal_speed_penalty"],
                    "terminal_v_exec": episode["terminal_v_exec"],
                    "terminal_abs_w_exec": episode["terminal_abs_w_exec"],
                    "mean_last10_v_exec": episode["mean_last10_v_exec"],
                    "mean_last10_abs_w_exec": episode["mean_last10_abs_w_exec"],
                    "mean_last10_exec_delta": episode["mean_last10_exec_delta"],
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
    content = f"""# Robot v/w Fixed-KF Two-Phase Eval

## Setup

- Model path: `{model_path}`
- smoother: `fixed`
- obs_mode: `kf_state`
- slowdown radius: `{args.slowdown_radius}`
- success thresholds: `v_exec <= {args.success_linear_threshold}`, `|w_exec| <= {args.success_angular_threshold}`
- Episodes: `{args.episodes}`
- Seeds: `{args.seed}` to `{args.seed + args.episodes - 1}`
- Deterministic: `{args.deterministic}`

## Outputs

- `path_complex.png`: selected best-return trajectory.
- `eval_reward_curve_complex.png`: selected episode reward and cumulative reward.
- `eval_step_distance_complex.png`: selected episode distance-to-goal.
- `command_smoothing_complex.png`: raw/executed `[v,w]`, command deltas, and filter mismatch.
- `episode_metrics.csv`: per-episode two-phase and terminal command metrics.
- `metrics_summary.csv`: aggregate metrics.

## Key Summary

- Success rate: `{summary['success_rate']:.3f}`
- Average return: `{summary['average_return']:.3f}`
- Average steps: `{summary['average_steps']:.3f}`
- Mean docking fraction: `{summary['mean_docking_fraction']:.6f}`
- Mean terminal v_exec: `{summary['mean_terminal_v_exec']:.6f}`
- Mean terminal |w_exec|: `{summary['mean_terminal_abs_w_exec']:.6f}`
- Mean last-10 v_exec: `{summary['mean_last10_v_exec']:.6f}`
- Mean last-10 |w_exec|: `{summary['mean_last10_abs_w_exec']:.6f}`
"""
    (out_dir / "README.md").write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir(Path(args.save_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(args.seed, args)
    model = SAC.load(str(model_path), env=env, device=args.device)
    episodes = []
    print(
        f"Evaluating fixed-KF two-phase: model={model_path}, "
        f"episodes={args.episodes}, deterministic={args.deterministic}"
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
            f"status={episode_status(episode)}, terminal_v={episode['terminal_v_exec']:.3f}, "
            f"terminal_abs_w={episode['terminal_abs_w_exec']:.3f}"
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
    print(f"Mean terminal v_exec: {summary['mean_terminal_v_exec']:.6f}")
    print(f"Mean terminal |w_exec|: {summary['mean_terminal_abs_w_exec']:.6f}")
    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
