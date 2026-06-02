"""Compare complex-env ablation A and B on shared plots.

A: SAC train no KF, eval no KF.
B: SAC train no KF, eval with external KF.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle, Rectangle
from stable_baselines3 import SAC

from envs.complex_nav_env import ComplexNavEnv
from evaluate.complex_env import run_episode


@dataclass(frozen=True)
class EvalResult:
    label: str
    use_kf: bool
    episodes: list[dict]

    @property
    def selected(self) -> dict:
        return max(self.episodes, key=lambda ep: ep["return"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot A/B complex-env eval comparison.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="outputs/ablations/group1_complex/A_sac_train_no_kf.zip",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/ablations/group1_complex/AB_compare",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", type=int, default=1, choices=[0, 1])
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def evaluate_condition(
    label: str,
    use_kf: bool,
    model_path: Path,
    episodes: int,
    seed: int,
    deterministic: bool,
    device: str,
) -> EvalResult:
    env = ComplexNavEnv(use_kf=use_kf, seed=seed)
    model = SAC.load(str(model_path), env=env, device=device)
    collected = []

    print(f"\nRunning {label}: use_kf={int(use_kf)}, episodes={episodes}")
    for episode_idx in range(episodes):
        episode_seed = seed + episode_idx
        set_eval_seed(episode_seed)
        episode = run_episode(
            model=model,
            env=env,
            seed=episode_seed,
            deterministic=deterministic,
        )
        episode["episode_seed"] = episode_seed
        episode["deterministic"] = deterministic
        add_episode_metrics(episode, env)
        collected.append(episode)
        print(
            f"{label} episode {episode_idx + 1}: "
            f"seed={episode_seed}, "
            f"return={episode['return']:.2f}, steps={episode['steps']}, "
            f"status={episode_status(episode)}"
        )

    env.close()
    return EvalResult(label=label, use_kf=use_kf, episodes=collected)


def set_eval_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_episode_metrics(episode: dict, env: ComplexNavEnv) -> None:
    trajectory = episode["trajectory"]
    raw_actions = episode["raw_actions"]
    exec_actions = episode["exec_actions"]
    raw_delta_norms = episode["raw_delta_norms"]
    exec_delta_norms = episode["exec_delta_norms"]

    if len(trajectory) > 1:
        path_segments = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
        path_length = float(np.sum(path_segments))
    else:
        path_length = 0.0

    clearances = [
        env.signed_distance_to_obstacles(point)
        for point in np.asarray(trajectory, dtype=np.float32)
    ]
    mean_raw_delta = float(np.mean(raw_delta_norms)) if len(raw_delta_norms) else 0.0
    mean_exec_delta = float(np.mean(exec_delta_norms)) if len(exec_delta_norms) else 0.0

    episode["final_distance"] = float(episode["distances"][-1])
    episode["path_length"] = path_length
    episode["min_obstacle_clearance"] = float(np.min(clearances)) if clearances else float("inf")
    episode["mean_raw_action_norm"] = (
        float(np.mean(np.linalg.norm(raw_actions, axis=1))) if len(raw_actions) else 0.0
    )
    episode["mean_executed_action_norm"] = (
        float(np.mean(np.linalg.norm(exec_actions, axis=1))) if len(exec_actions) else 0.0
    )
    episode["mean_raw_action_delta"] = mean_raw_delta
    episode["mean_executed_action_delta"] = mean_exec_delta
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


def add_complex_map(ax: plt.Axes, env: ComplexNavEnv) -> None:
    ax.set_xlim(0, env.map_size)
    ax.set_ylim(0, env.map_size)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.3)

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

    waypoints = np.asarray(env.waypoints, dtype=np.float32)
    ax.plot(
        waypoints[:, 0],
        waypoints[:, 1],
        "--",
        color="0.55",
        linewidth=1.2,
        label="APF waypoints",
    )
    ax.scatter(env.start[0], env.start[1], c="tab:green", s=80, label="Start", zorder=4)
    ax.scatter(env.goal[0], env.goal[1], c="tab:purple", s=110, marker="*", label="Goal", zorder=4)


def plot_path_comparison(env: ComplexNavEnv, a: EvalResult, b: EvalResult, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("Complex Env Path Comparison: A vs B")
    add_complex_map(ax, env)

    ax.plot(
        a.selected["trajectory"][:, 0],
        a.selected["trajectory"][:, 1],
        color="tab:blue",
        linewidth=2.2,
        label="A eval no KF",
    )
    ax.plot(
        b.selected["trajectory"][:, 0],
        b.selected["trajectory"][:, 1],
        color="tab:orange",
        linewidth=2.2,
        label="B eval external KF",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_reward_comparison(a: EvalResult, b: EvalResult, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_title("Complex Env Reward Comparison: A vs B")

    for result, color in [(a, "tab:blue"), (b, "tab:orange")]:
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


def plot_distance_comparison(a: EvalResult, b: EvalResult, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_title("Complex Env Distance-to-Goal Comparison: A vs B")

    for result, color in [(a, "tab:blue"), (b, "tab:orange")]:
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


def plot_action_comparison(a: EvalResult, b: EvalResult, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=False)
    fig.suptitle("Complex Env Action Comparison: A vs B")

    for result, color in [(a, "tab:blue"), (b, "tab:orange")]:
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


def summarize(result: EvalResult) -> dict[str, float]:
    returns = np.asarray([ep["return"] for ep in result.episodes], dtype=np.float32)
    steps = np.asarray([ep["steps"] for ep in result.episodes], dtype=np.float32)
    successes = np.asarray([ep["success"] for ep in result.episodes], dtype=np.float32)
    collisions = np.asarray([ep["collision"] for ep in result.episodes], dtype=np.float32)
    out_of_bounds = np.asarray([ep["out_of_bounds"] for ep in result.episodes], dtype=np.float32)
    timeouts = np.asarray([ep["timeout"] for ep in result.episodes], dtype=np.float32)
    raw_delta_all = np.concatenate([ep["raw_delta_norms"] for ep in result.episodes])
    exec_delta_all = np.concatenate([ep["exec_delta_norms"] for ep in result.episodes])
    mean_raw_delta = float(np.mean(raw_delta_all)) if len(raw_delta_all) else 0.0
    mean_exec_delta = float(np.mean(exec_delta_all)) if len(exec_delta_all) else 0.0
    final_distances = np.asarray([ep["final_distance"] for ep in result.episodes], dtype=np.float32)
    path_lengths = np.asarray([ep["path_length"] for ep in result.episodes], dtype=np.float32)
    min_clearances = np.asarray(
        [ep["min_obstacle_clearance"] for ep in result.episodes],
        dtype=np.float32,
    )
    selected = result.selected

    return {
        "episodes": float(len(result.episodes)),
        "average_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "average_steps": float(np.mean(steps)),
        "success_rate": float(np.mean(successes)),
        "collision_rate": float(np.mean(collisions)),
        "out_of_bounds_rate": float(np.mean(out_of_bounds)),
        "timeout_rate": float(np.mean(timeouts)),
        "average_final_distance": float(np.mean(final_distances)),
        "average_path_length": float(np.mean(path_lengths)),
        "average_min_obstacle_clearance": float(np.mean(min_clearances)),
        "worst_min_obstacle_clearance": float(np.min(min_clearances)),
        "mean_raw_action_delta": mean_raw_delta,
        "mean_executed_action_delta": mean_exec_delta,
        "smoothing_ratio": mean_exec_delta / (mean_raw_delta + 1e-8),
        "selected_return": float(selected["return"]),
        "selected_steps": float(selected["steps"]),
        "selected_final_distance": float(selected["final_distance"]),
        "selected_path_length": float(selected["path_length"]),
        "selected_min_obstacle_clearance": float(selected["min_obstacle_clearance"]),
    }


def print_summary(result: EvalResult) -> None:
    summary = summarize(result)
    print(f"\n{result.label} summary")
    print(f"Average return: {summary['average_return']:.3f}")
    print(f"Average steps: {summary['average_steps']:.3f}")
    print(f"Success rate: {summary['success_rate']:.3f}")
    print(f"Collision rate: {summary['collision_rate']:.3f}")
    print(f"Out-of-bounds rate: {summary['out_of_bounds_rate']:.3f}")
    print(f"Timeout rate: {summary['timeout_rate']:.3f}")
    print(f"Mean raw action delta: {summary['mean_raw_action_delta']:.6f}")
    print(f"Mean executed action delta: {summary['mean_executed_action_delta']:.6f}")
    print(f"Smoothing ratio: {summary['smoothing_ratio']:.6f}")
    print(f"Average final distance: {summary['average_final_distance']:.6f}")
    print(f"Average path length: {summary['average_path_length']:.6f}")
    print(f"Average min obstacle clearance: {summary['average_min_obstacle_clearance']:.6f}")
    print(
        "Selected episode: "
        f"return={summary['selected_return']:.3f}, steps={summary['selected_steps']:.0f}"
    )


def write_episode_csv(a: EvalResult, b: EvalResult, save_path: Path) -> None:
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
        for result in [a, b]:
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


def write_summary_csv(a: EvalResult, b: EvalResult, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    a_summary = summarize(a)
    b_summary = summarize(b)
    rows = [
        ("episodes", "higher", "A/B 使用同一 no-KF 训练模型；B 只在评估时打开 external KF。"),
        ("success_rate", "higher", "成功率相同或接近，说明 external KF 没有造成灾难性任务失败。"),
        ("collision_rate", "lower", "碰撞率越低越好；若 B 不升高，说明外部滤波没有明显破坏避障。"),
        ("out_of_bounds_rate", "lower", "越界率越低越好。"),
        ("timeout_rate", "lower", "超时率越低越好；外部 KF 若升高，通常表示响应滞后。"),
        ("average_return", "higher", "平均回报综合反映任务完成质量和效率。"),
        ("std_return", "lower", "回报标准差越低，episode 间稳定性越好。"),
        ("average_steps", "lower", "平均步数越低到达越快；B 更高通常表示 external KF 带来执行延迟。"),
        ("average_final_distance", "lower", "最终距离越低越好。"),
        ("average_path_length", "lower", "路径长度越短通常越直接，但需要结合 clearance 和碰撞率判断。"),
        ("average_min_obstacle_clearance", "higher", "平均最小障碍物间隙越高，路径安全裕度越大。"),
        ("worst_min_obstacle_clearance", "higher", "最差最小间隙用于观察是否存在贴边或碰撞风险。"),
        ("mean_raw_action_delta", "neutral", "raw action 由同一 no-KF policy 产生；差异来自 external KF 改变后的闭环轨迹。"),
        ("mean_executed_action_delta", "lower", "executed action delta 越低，执行动作越平滑。"),
        ("smoothing_ratio", "lower", "ratio 越低说明执行动作相对 raw action 被滤波得越明显。"),
        ("selected_return", "higher", "用于绘图的最佳 episode 回报。"),
        ("selected_steps", "lower", "用于绘图的最佳 episode 到达步数。"),
        ("selected_final_distance", "lower", "用于绘图的最佳 episode 最终距离。"),
        ("selected_path_length", "lower", "用于绘图的最佳 episode 路径长度。"),
        ("selected_min_obstacle_clearance", "higher", "用于绘图的最佳 episode 最小障碍物间隙。"),
    ]

    with save_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "metric",
                "A_no_KF",
                "B_external_KF",
                "B_minus_A",
                "relative_change_percent",
                "comparison_conclusion",
            ],
        )
        writer.writeheader()
        for metric, direction, base_comment in rows:
            a_value = a_summary[metric]
            b_value = b_summary[metric]
            delta = b_value - a_value
            relative = delta / (abs(a_value) + 1e-8) * 100.0
            writer.writerow(
                {
                    "metric": metric,
                    "A_no_KF": a_value,
                    "B_external_KF": b_value,
                    "B_minus_A": delta,
                    "relative_change_percent": relative,
                    "comparison_conclusion": build_conclusion(
                        metric=metric,
                        direction=direction,
                        delta=delta,
                        base_comment=base_comment,
                    ),
                }
            )


def build_conclusion(metric: str, direction: str, delta: float, base_comment: str) -> str:
    tolerance = 1e-6
    if abs(delta) <= tolerance:
        trend = "A 与 B 基本相同。"
    elif direction == "higher":
        trend = "B 更高，倾向于更好。" if delta > 0.0 else "B 更低，倾向于更差。"
    elif direction == "lower":
        trend = "B 更低，倾向于更好。" if delta < 0.0 else "B 更高，倾向于更差。"
    elif metric == "mean_raw_action_delta":
        trend = "该指标不单独判定优劣，主要用于解释闭环轨迹变化。"
    else:
        trend = "该指标用于辅助解释，不单独判定优劣。"
    return f"{trend} {base_comment}"


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    out_dir = Path(args.out_dir)

    if not model_path.exists():
        print(f"Model file not found: {model_path}", file=sys.stderr)
        sys.exit(1)
    if bool(args.deterministic):
        print(
            "Note: deterministic=1 with the fixed ComplexNavEnv is expected to "
            "produce identical episodes even when episode seeds differ. Use "
            "--deterministic 0 to sample stochastic SAC actions under different seeds."
        )

    a_result = evaluate_condition(
        label="A no KF",
        use_kf=False,
        model_path=model_path,
        episodes=args.episodes,
        seed=args.seed,
        deterministic=bool(args.deterministic),
        device=args.device,
    )
    b_result = evaluate_condition(
        label="B external KF",
        use_kf=True,
        model_path=model_path,
        episodes=args.episodes,
        seed=args.seed,
        deterministic=bool(args.deterministic),
        device=args.device,
    )

    plot_env = ComplexNavEnv(use_kf=False, seed=args.seed)
    plot_path_comparison(plot_env, a_result, b_result, out_dir / "path_ab_complex.png")
    plot_env.close()
    plot_reward_comparison(a_result, b_result, out_dir / "reward_curve_ab_complex.png")
    plot_distance_comparison(a_result, b_result, out_dir / "step_distance_ab_complex.png")
    plot_action_comparison(a_result, b_result, out_dir / "action_smoothing_ab_complex.png")
    write_summary_csv(a_result, b_result, out_dir / "ab_metrics_summary.csv")
    write_episode_csv(a_result, b_result, out_dir / "ab_episode_metrics.csv")

    print_summary(a_result)
    print_summary(b_result)
    print(f"\nSaved comparison figures to: {out_dir}")
    print(f"Saved summary CSV to: {out_dir / 'ab_metrics_summary.csv'}")
    print(f"Saved episode CSV to: {out_dir / 'ab_episode_metrics.csv'}")


if __name__ == "__main__":
    main()
