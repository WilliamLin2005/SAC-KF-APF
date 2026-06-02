"""Evaluate two-phase v/w SAC policies with configurable command smoothing."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC

from envs.robot_command_smoothing_wrappers import OBS_MODE_CHOICES
from envs.robot_two_phase_reward_wrapper import make_robot_two_phase_env
from evaluate.robot_diffdrive_kf_two_phase import (
    add_episode_metrics,
    episode_status,
    plot_commands,
    plot_distance,
    plot_path,
    plot_reward,
    run_episode,
    summarize,
    write_episode_csv,
    write_summary_csv,
)
from train.robot_diffdrive_two_phase_ablation import (
    CURRICULUM_CHOICES,
    SMOOTHER_CHOICES,
    default_model_path,
    default_run_name,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate two-phase v/w SAC with configurable no-KF/fixed-KF smoothing."
    )
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--smoother", type=str, default="none", choices=SMOOTHER_CHOICES)
    parser.add_argument("--obs-mode", type=str, default="prev_exec", choices=OBS_MODE_CHOICES)
    parser.add_argument("--kf-curriculum", type=str, default="none", choices=CURRICULUM_CHOICES)
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


def default_out_dir(save_dir: Path, smoother: str, obs_mode: str) -> Path:
    return save_dir / "vw" / f"{default_run_name(smoother, obs_mode)}_eval"


def make_env(seed: int | None, args: argparse.Namespace):
    return make_robot_two_phase_env(
        smoother=args.smoother,
        obs_mode=args.obs_mode,
        kf_curriculum=args.kf_curriculum,
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


def write_readme(args: argparse.Namespace, episodes: list[dict], out_dir: Path, model_path: Path) -> None:
    summary = summarize(episodes)
    content = f"""# Robot v/w Two-Phase Ablation Eval

## Setup

- Model path: `{model_path}`
- smoother: `{args.smoother}`
- obs_mode: `{args.obs_mode}`
- kf_curriculum: `{args.kf_curriculum}`
- slowdown radius: `{args.slowdown_radius}`
- success thresholds: `v_exec <= {args.success_linear_threshold}`, `|w_exec| <= {args.success_angular_threshold}`
- Episodes: `{args.episodes}`
- Seeds: `{args.seed}` to `{args.seed + args.episodes - 1}`
- Deterministic: `{args.deterministic}`
- Device: `{args.device}`

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

    env = make_env(args.seed, args)
    model = SAC.load(str(model_path), env=env, device=args.device)
    episodes = []
    print(
        f"Evaluating two-phase v/w ablation: model={model_path}, "
        f"smoother={args.smoother}, obs_mode={args.obs_mode}, "
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
            f"status={episode_status(episode)}, "
            f"terminal_v={episode['terminal_v_exec']:.3f}, "
            f"terminal_abs_w={episode['terminal_abs_w_exec']:.3f}, "
            f"mismatch={episode['mean_filter_mismatch']:.4f}"
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
    print(f"Mean executed command delta: {summary['mean_executed_command_delta']:.6f}")
    print(f"Mean filter mismatch: {summary['mean_filter_mismatch']:.6f}")
    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
