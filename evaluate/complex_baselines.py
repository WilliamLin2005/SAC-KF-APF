"""Evaluate complex-env baseline variants."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import SAC

from envs.baseline_wrappers import get_complex_base_env, make_complex_baseline_env
from evaluate.compare_complex_ab import (
    EvalResult,
    add_complex_map,
    add_episode_metrics,
    episode_status,
    print_summary,
    set_eval_seed,
    summarize,
)
from evaluate.complex_env import run_episode


EVAL_BASELINES = (
    "action_delta_penalty",
    "lowpass_in_loop",
    "lowpass_eval_only",
    "gsde",
    "kf_no_aug",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate complex-env baseline variants.")
    parser.add_argument("--baseline", type=str, required=True, choices=EVAL_BASELINES)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", type=int, default=0, choices=[0, 1])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--action-penalty-weight", type=float, default=0.2)
    parser.add_argument("--lowpass-alpha", type=float, default=0.35)
    parser.add_argument("--eval-lowpass", type=int, default=0, choices=[0, 1])
    return parser.parse_args()


def default_model_path(baseline: str) -> Path:
    defaults = {
        "action_delta_penalty": "outputs/ablations/group1_complex/D_action_penalty.zip",
        "lowpass_in_loop": "outputs/ablations/group1_complex/E_lowpass_in_loop.zip",
        "lowpass_eval_only": "outputs/ablations/group1_complex/A_sac_train_no_kf.zip",
        "gsde": "outputs/ablations/group1_complex/F_gsde.zip",
        "kf_no_aug": "outputs/ablations/group1_complex/G_kf_no_aug.zip",
    }
    return Path(defaults[baseline])


def default_out_dir(baseline: str) -> Path:
    return Path("outputs/ablations/group1_complex") / f"{baseline}_eval"


def eval_env_name(args: argparse.Namespace) -> str:
    if args.eval_lowpass:
        return "lowpass_eval_only"
    return args.baseline


def evaluate_baseline(args: argparse.Namespace) -> EvalResult:
    model_path = Path(args.model_path) if args.model_path else default_model_path(args.baseline)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    env = make_complex_baseline_env(
        baseline=eval_env_name(args),
        seed=args.seed,
        action_penalty_weight=args.action_penalty_weight,
        lowpass_alpha=args.lowpass_alpha,
    )
    base_env = get_complex_base_env(env)
    model = SAC.load(str(model_path), env=env, device=args.device)

    episodes = []
    print(
        f"Evaluating {args.baseline}: model={model_path}, "
        f"eval_env={eval_env_name(args)}, episodes={args.episodes}"
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
        add_episode_metrics(episode, base_env)
        episodes.append(episode)
        print(
            f"Episode {episode_idx + 1}: seed={episode_seed}, "
            f"return={episode['return']:.2f}, steps={episode['steps']}, "
            f"status={episode_status(episode)}"
        )

    env.close()
    return EvalResult(label=args.baseline, use_kf=eval_env_name(args) == "kf_no_aug", episodes=episodes)


def plot_path(env, result: EvalResult, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(f"Complex Env Path: {result.label}")
    add_complex_map(ax, env)
    trajectory = result.selected["trajectory"]
    ax.plot(trajectory[:, 0], trajectory[:, 1], color="tab:blue", linewidth=2.2, label=result.label)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_reward(result: EvalResult, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    rewards = result.selected["rewards"]
    steps = np.arange(len(rewards))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_title(f"Complex Env Reward: {result.label}")
    ax.plot(steps, rewards, color="tab:blue", alpha=0.55, linewidth=1.2, label="step reward")
    ax.plot(steps, np.cumsum(rewards), color="tab:orange", linewidth=2.0, label="cumulative reward")
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_distance(result: EvalResult, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    distances = result.selected["distances"]
    steps = np.arange(len(distances))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_title(f"Complex Env Distance to Goal: {result.label}")
    ax.plot(steps, distances, color="tab:green", linewidth=2.0)
    ax.set_xlabel("Step")
    ax.set_ylabel("Distance to goal")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_actions(result: EvalResult, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    episode = result.selected
    raw_actions = episode["raw_actions"]
    exec_actions = episode["exec_actions"]
    raw_steps = np.arange(len(raw_actions))
    delta_steps = np.arange(len(episode["raw_delta_norms"]))
    raw_norm = np.linalg.norm(raw_actions, axis=1)
    exec_norm = np.linalg.norm(exec_actions, axis=1)

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=False)
    fig.suptitle(f"Complex Env Actions: {result.label}")
    axes[0].plot(raw_steps, raw_norm, color="tab:blue", linewidth=1.8, label="raw")
    axes[1].plot(raw_steps, exec_norm, color="tab:orange", linewidth=1.8, label="executed")
    axes[2].plot(delta_steps, episode["raw_delta_norms"], color="tab:blue", linestyle=":", linewidth=1.4, label="raw delta")
    axes[2].plot(delta_steps, episode["exec_delta_norms"], color="tab:orange", linewidth=1.8, label="executed delta")
    axes[0].set_title("Raw action norm")
    axes[1].set_title("Executed action norm")
    axes[2].set_title("Action delta norm")
    axes[2].set_xlabel("Step")
    for ax in axes:
        ax.set_ylabel("Norm")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def write_episode_csv(result: EvalResult, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "condition",
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
        "mean_raw_action_norm",
        "mean_executed_action_norm",
        "mean_raw_action_delta",
        "mean_executed_action_delta",
        "episode_smoothing_ratio",
    ]
    with save_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for episode_idx, episode in enumerate(result.episodes):
            writer.writerow(
                {
                    "condition": result.label,
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
                    "mean_raw_action_norm": episode["mean_raw_action_norm"],
                    "mean_executed_action_norm": episode["mean_executed_action_norm"],
                    "mean_raw_action_delta": episode["mean_raw_action_delta"],
                    "mean_executed_action_delta": episode["mean_executed_action_delta"],
                    "episode_smoothing_ratio": episode["episode_smoothing_ratio"],
                }
            )


def write_summary_csv(result: EvalResult, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize(result)
    with save_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in summary.items():
            writer.writerow({"metric": key, "value": value})


def write_readme(args: argparse.Namespace, result: EvalResult, out_dir: Path) -> None:
    summary = summarize(result)
    text = f"""# Complex Baseline Eval: {result.label}

## Setup

- Baseline: `{args.baseline}`
- Evaluation env: `{eval_env_name(args)}`
- Model path: `{Path(args.model_path) if args.model_path else default_model_path(args.baseline)}`
- Episodes: `{args.episodes}`
- Seeds: `{args.seed}` to `{args.seed + args.episodes - 1}`
- Deterministic: `{int(args.deterministic)}`
- Low-pass alpha: `{args.lowpass_alpha}`
- Action penalty weight: `{args.action_penalty_weight}`

## Outputs

- `path_complex.png`: selected best-return episode trajectory.
- `eval_reward_curve_complex.png`: selected episode reward and cumulative reward.
- `eval_step_distance_complex.png`: selected episode distance-to-goal.
- `action_smoothing_complex.png`: selected episode action norms and action deltas.
- `{args.baseline}_episode_metrics.csv`: per-episode metrics.
- `{args.baseline}_metrics_summary.csv`: aggregate metrics.

## Key Summary

- Success rate: `{summary['success_rate']:.3f}`
- Collision rate: `{summary['collision_rate']:.3f}`
- Average return: `{summary['average_return']:.3f}`
- Average steps: `{summary['average_steps']:.3f}`
- Average final distance: `{summary['average_final_distance']:.6f}`
- Average min obstacle clearance: `{summary['average_min_obstacle_clearance']:.6f}`
- Mean executed action delta: `{summary['mean_executed_action_delta']:.6f}`
- Smoothing ratio: `{summary['smoothing_ratio']:.6f}`
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir(args.baseline)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = evaluate_baseline(args)
    plot_wrapper_env = make_complex_baseline_env(
        baseline=eval_env_name(args),
        seed=args.seed,
        action_penalty_weight=args.action_penalty_weight,
        lowpass_alpha=args.lowpass_alpha,
    )
    plot_env = get_complex_base_env(plot_wrapper_env)

    try:
        plot_path(plot_env, result, out_dir / "path_complex.png")
        plot_reward(result, out_dir / "eval_reward_curve_complex.png")
        plot_distance(result, out_dir / "eval_step_distance_complex.png")
        plot_actions(result, out_dir / "action_smoothing_complex.png")
        write_episode_csv(result, out_dir / f"{args.baseline}_episode_metrics.csv")
        write_summary_csv(result, out_dir / f"{args.baseline}_metrics_summary.csv")
        write_readme(args, result, out_dir)
        print_summary(result)
        print(f"\nSaved baseline evaluation outputs to: {out_dir}")
    finally:
        plot_wrapper_env.close()


if __name__ == "__main__":
    main()
