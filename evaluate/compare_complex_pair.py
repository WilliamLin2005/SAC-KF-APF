"""Generic pairwise comparison for group-1 complex-env ablations."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from envs.complex_nav_env import ComplexNavEnv
from evaluate.compare_complex_ab import (
    EvalResult,
    add_complex_map,
    episode_status,
    evaluate_condition,
    print_summary,
    summarize,
)


@dataclass(frozen=True)
class ConditionSpec:
    key: str
    label: str
    csv_column: str
    model_path: str
    use_kf: bool
    description: str


CONDITIONS = {
    "A": ConditionSpec(
        key="A",
        label="A no KF",
        csv_column="A_no_KF",
        model_path="outputs/ablations/group1_complex/A_sac_train_no_kf.zip",
        use_kf=False,
        description="SAC train no KF, eval no KF.",
    ),
    "B": ConditionSpec(
        key="B",
        label="B external KF",
        csv_column="B_external_KF",
        model_path="outputs/ablations/group1_complex/A_sac_train_no_kf.zip",
        use_kf=True,
        description="SAC train no KF, eval with external KF.",
    ),
    "C": ConditionSpec(
        key="C",
        label="C KF-in-loop",
        csv_column="C_KF_in_loop",
        model_path="outputs/ablations/group1_complex/C_sac_train_with_kf.zip",
        use_kf=True,
        description="SAC train with KF-in-loop, eval with KF.",
    ),
}


def parse_pair_args(description: str, default_out_dir: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--out-dir", type=str, default=default_out_dir)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", type=int, default=0, choices=[0, 1])
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def run_pair_comparison(left_key: str, right_key: str, args: argparse.Namespace) -> None:
    left = CONDITIONS[left_key]
    right = CONDITIONS[right_key]
    out_dir = Path(args.out_dir)
    pair_slug = f"{left.key.lower()}{right.key.lower()}"

    for spec in [left, right]:
        model_path = Path(spec.model_path)
        if not model_path.exists():
            print(f"{spec.key} model file not found: {model_path}", file=sys.stderr)
            sys.exit(1)

    if bool(args.deterministic):
        print(
            "Note: deterministic=1 with the fixed ComplexNavEnv is expected to "
            "produce identical episodes even when episode seeds differ. Use "
            "--deterministic 0 to sample stochastic SAC actions under different seeds."
        )

    left_result = evaluate_condition(
        label=left.label,
        use_kf=left.use_kf,
        model_path=Path(left.model_path),
        episodes=args.episodes,
        seed=args.seed,
        deterministic=bool(args.deterministic),
        device=args.device,
    )
    right_result = evaluate_condition(
        label=right.label,
        use_kf=right.use_kf,
        model_path=Path(right.model_path),
        episodes=args.episodes,
        seed=args.seed,
        deterministic=bool(args.deterministic),
        device=args.device,
    )

    plot_env = ComplexNavEnv(use_kf=False, seed=args.seed)
    plot_path_comparison(
        plot_env,
        left_result,
        right_result,
        out_dir / f"path_{pair_slug}_complex.png",
        pair_label=f"{left.key} vs {right.key}",
    )
    plot_env.close()
    plot_reward_comparison(
        left_result,
        right_result,
        out_dir / f"reward_curve_{pair_slug}_complex.png",
        pair_label=f"{left.key} vs {right.key}",
    )
    plot_distance_comparison(
        left_result,
        right_result,
        out_dir / f"step_distance_{pair_slug}_complex.png",
        pair_label=f"{left.key} vs {right.key}",
    )
    plot_action_comparison(
        left_result,
        right_result,
        out_dir / f"action_smoothing_{pair_slug}_complex.png",
        pair_label=f"{left.key} vs {right.key}",
    )
    write_summary_csv(
        left,
        right,
        left_result,
        right_result,
        out_dir / f"{pair_slug}_metrics_summary.csv",
    )
    write_episode_csv(
        left,
        right,
        left_result,
        right_result,
        out_dir / f"{pair_slug}_episode_metrics.csv",
    )
    write_readme(
        left,
        right,
        left_result,
        right_result,
        out_dir / "README.md",
        pair_slug=pair_slug,
        args=args,
    )

    print_summary(left_result)
    print_summary(right_result)
    print(f"\nSaved comparison outputs to: {out_dir}")


def plot_path_comparison(
    env: ComplexNavEnv,
    left: EvalResult,
    right: EvalResult,
    save_path: Path,
    pair_label: str,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(f"Complex Env Path Comparison: {pair_label}")
    add_complex_map(ax, env)

    ax.plot(
        left.selected["trajectory"][:, 0],
        left.selected["trajectory"][:, 1],
        color="tab:blue",
        linewidth=2.2,
        label=left.label,
    )
    ax.plot(
        right.selected["trajectory"][:, 0],
        right.selected["trajectory"][:, 1],
        color="tab:orange",
        linewidth=2.2,
        label=right.label,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_reward_comparison(
    left: EvalResult,
    right: EvalResult,
    save_path: Path,
    pair_label: str,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_title(f"Complex Env Reward Comparison: {pair_label}")

    for result, color in [(left, "tab:blue"), (right, "tab:orange")]:
        rewards = result.selected["rewards"]
        steps = np.arange(len(rewards))
        ax.plot(
            steps,
            rewards,
            color=color,
            alpha=0.45,
            linewidth=1.2,
            label=f"{result.label} step reward",
        )
        ax.plot(
            steps,
            np.cumsum(rewards),
            color=color,
            linewidth=2.0,
            label=f"{result.label} cumulative reward",
        )

    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_distance_comparison(
    left: EvalResult,
    right: EvalResult,
    save_path: Path,
    pair_label: str,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_title(f"Complex Env Distance-to-Goal Comparison: {pair_label}")

    for result, color in [(left, "tab:blue"), (right, "tab:orange")]:
        distances = result.selected["distances"]
        steps = np.arange(len(distances))
        ax.plot(steps, distances, color=color, linewidth=2.0, label=result.label)

    ax.set_xlabel("Step")
    ax.set_ylabel("Distance to goal")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_action_comparison(
    left: EvalResult,
    right: EvalResult,
    save_path: Path,
    pair_label: str,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=False)
    fig.suptitle(f"Complex Env Action Comparison: {pair_label}")

    for result, color in [(left, "tab:blue"), (right, "tab:orange")]:
        episode = result.selected
        raw_actions = episode["raw_actions"]
        exec_actions = episode["exec_actions"]
        raw_steps = np.arange(len(raw_actions))
        delta_steps = np.arange(len(episode["raw_delta_norms"]))

        raw_norm = np.linalg.norm(raw_actions, axis=1)
        exec_norm = np.linalg.norm(exec_actions, axis=1)

        axes[0].plot(raw_steps, raw_norm, color=color, linewidth=1.8, label=f"{result.label} raw")
        axes[1].plot(raw_steps, exec_norm, color=color, linewidth=1.8, label=f"{result.label} executed")
        axes[2].plot(
            delta_steps,
            episode["raw_delta_norms"],
            color=color,
            linestyle=":",
            linewidth=1.4,
            label=f"{result.label} raw delta",
        )
        axes[2].plot(
            delta_steps,
            episode["exec_delta_norms"],
            color=color,
            linewidth=1.8,
            label=f"{result.label} executed delta",
        )

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


def write_episode_csv(
    left_spec: ConditionSpec,
    right_spec: ConditionSpec,
    left: EvalResult,
    right: EvalResult,
    save_path: Path,
) -> None:
    del left_spec, right_spec
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "condition",
        "use_kf",
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
        for result in [left, right]:
            for episode_idx, episode in enumerate(result.episodes):
                writer.writerow(
                    {
                        "condition": result.label,
                        "use_kf": int(result.use_kf),
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


def write_summary_csv(
    left_spec: ConditionSpec,
    right_spec: ConditionSpec,
    left: EvalResult,
    right: EvalResult,
    save_path: Path,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    left_summary = summarize(left)
    right_summary = summarize(right)
    rows = [
        ("episodes", "higher", "Both conditions are evaluated with the same episode seeds."),
        ("success_rate", "higher", "Higher success rate is better."),
        ("collision_rate", "lower", "Lower collision rate indicates safer navigation."),
        ("out_of_bounds_rate", "lower", "Lower out-of-bounds rate is better."),
        ("timeout_rate", "lower", "Lower timeout rate is better."),
        ("average_return", "higher", "Average return summarizes task quality and efficiency."),
        ("std_return", "lower", "Lower return standard deviation indicates more stable episodes."),
        ("average_steps", "lower", "Lower average steps means faster goal convergence."),
        ("average_final_distance", "lower", "Lower final distance is better."),
        ("average_path_length", "lower", "Shorter path is usually more direct; interpret with safety metrics."),
        ("average_min_obstacle_clearance", "higher", "Higher clearance indicates larger safety margin."),
        ("worst_min_obstacle_clearance", "higher", "Worst-case clearance reveals close-passing risk."),
        ("mean_raw_action_delta", "neutral", "Raw action delta is useful for explaining policy behavior."),
        ("mean_executed_action_delta", "lower", "Lower executed action delta means smoother executed control."),
        ("smoothing_ratio", "lower", "Lower ratio means executed action is smoother relative to raw action."),
        ("selected_return", "higher", "Return of the representative episode used for figures."),
        ("selected_steps", "lower", "Steps of the representative episode used for figures."),
        ("selected_final_distance", "lower", "Final distance of the representative episode used for figures."),
        ("selected_path_length", "lower", "Path length of the representative episode used for figures."),
        ("selected_min_obstacle_clearance", "higher", "Clearance of the representative episode used for figures."),
    ]

    with save_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "metric",
                left_spec.csv_column,
                right_spec.csv_column,
                f"{right_spec.key}_minus_{left_spec.key}",
                "relative_change_percent",
                "comparison_conclusion",
            ],
        )
        writer.writeheader()
        for metric, direction, base_comment in rows:
            left_value = left_summary[metric]
            right_value = right_summary[metric]
            delta = right_value - left_value
            relative = delta / (abs(left_value) + 1e-8) * 100.0
            writer.writerow(
                {
                    "metric": metric,
                    left_spec.csv_column: left_value,
                    right_spec.csv_column: right_value,
                    f"{right_spec.key}_minus_{left_spec.key}": delta,
                    "relative_change_percent": relative,
                    "comparison_conclusion": build_conclusion(
                        right_label=right_spec.label,
                        direction=direction,
                        delta=delta,
                        base_comment=base_comment,
                    ),
                }
            )


def build_conclusion(right_label: str, direction: str, delta: float, base_comment: str) -> str:
    tolerance = 1e-6
    if abs(delta) <= tolerance:
        trend = f"{right_label} is approximately the same as the left condition."
    elif direction == "higher":
        trend = f"{right_label} is higher, which is favorable." if delta > 0.0 else f"{right_label} is lower, which is unfavorable."
    elif direction == "lower":
        trend = f"{right_label} is lower, which is favorable." if delta < 0.0 else f"{right_label} is higher, which is unfavorable."
    else:
        trend = "This metric is diagnostic and should not be judged alone."
    return f"{trend} {base_comment}"


def write_readme(
    left_spec: ConditionSpec,
    right_spec: ConditionSpec,
    left: EvalResult,
    right: EvalResult,
    save_path: Path,
    pair_slug: str,
    args: argparse.Namespace,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    left_summary = summarize(left)
    right_summary = summarize(right)
    content = f"""# {left_spec.key}/{right_spec.key} Complex Evaluation Comparison

## Experiment Setup

- {left_spec.key}: {left_spec.description}
- {right_spec.key}: {right_spec.description}
- {left_spec.key} model: `{left_spec.model_path}`
- {right_spec.key} model: `{right_spec.model_path}`
- Episodes: {args.episodes}
- Episode seeds: `{args.seed}` to `{args.seed + args.episodes - 1}`
- Deterministic policy eval: `{int(args.deterministic)}`
- Device: `{args.device}`

When `--deterministic 0` is used, SAC samples stochastic policy actions and the episode seeds produce different trajectories. When `--deterministic 1` is used on the fixed `ComplexNavEnv`, repeated seeds are expected to produce identical trajectories.

## Figures

The four PNG figures are representative-episode visualizations, not averages over all seeds. For each condition, the selected episode is the one with the highest return among the evaluated episodes.

- {left_spec.key} selected episode: seed {left.selected['episode_seed']}, return {left.selected['return']:.3f}, steps {left.selected['steps']}.
- {right_spec.key} selected episode: seed {right.selected['episode_seed']}, return {right.selected['return']:.3f}, steps {right.selected['steps']}.

Files:

- `path_{pair_slug}_complex.png`: selected-episode executed trajectories on the complex map.
- `reward_curve_{pair_slug}_complex.png`: selected-episode step reward and cumulative reward.
- `step_distance_{pair_slug}_complex.png`: selected-episode distance-to-goal curves.
- `action_smoothing_{pair_slug}_complex.png`: selected-episode raw/executed action and action-delta comparison.

## Tables

- `{pair_slug}_episode_metrics.csv`: per-episode metrics for all seeds.
- `{pair_slug}_metrics_summary.csv`: aggregate metric comparison with a short conclusion per metric.

## Key Summary

- {left_spec.key} success rate: {left_summary['success_rate']:.3f}; {right_spec.key} success rate: {right_summary['success_rate']:.3f}.
- {left_spec.key} average steps: {left_summary['average_steps']:.3f}; {right_spec.key} average steps: {right_summary['average_steps']:.3f}.
- {left_spec.key} mean executed action delta: {left_summary['mean_executed_action_delta']:.6f}; {right_spec.key} mean executed action delta: {right_summary['mean_executed_action_delta']:.6f}.
- {left_spec.key} smoothing ratio: {left_summary['smoothing_ratio']:.6f}; {right_spec.key} smoothing ratio: {right_summary['smoothing_ratio']:.6f}.
- {left_spec.key} average minimum obstacle clearance: {left_summary['average_min_obstacle_clearance']:.6f}; {right_spec.key} average minimum obstacle clearance: {right_summary['average_min_obstacle_clearance']:.6f}.

## Interpretation Hint

For group-1 ablation, the most important comparison is B vs C: both are evaluated with KF, but only C is trained with KF-in-loop. If C preserves smooth executed actions while improving success, steps, final distance, or clearance relative to B, it supports the claim that KF-in-loop training reduces deployment dynamics mismatch compared with external post-policy filtering.
"""
    save_path.write_text(content, encoding="utf-8")
