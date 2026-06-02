"""Evaluate a trained SAC policy and generate navigation plots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from envs.continuous_nav_env import ContinuousNavEnv
from utils.plotting import (
    plot_action_smoothing,
    plot_path,
    plot_reward_curve,
    plot_step_distance,
)
from utils.spline import smooth_trajectory_bspline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SAC navigation policy.")
    parser.add_argument("--model-path", type=str, default="outputs/models/sac_kf_nav.zip")
    parser.add_argument("--use-kf", type=int, default=1, choices=[0, 1])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fig-dir", type=str, default="outputs/figures")
    parser.add_argument("--deterministic", type=int, default=1, choices=[0, 1])
    return parser.parse_args()


def run_episode(model: SAC, env: ContinuousNavEnv, seed: int, deterministic: bool) -> dict:
    obs, info = env.reset(seed=seed)
    trajectory = [info["position"].copy()]
    distances = [info["distance_to_goal"]]
    raw_actions = []
    exec_actions = []
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
        distances.append(final_info["distance_to_goal"])
        raw_actions.append(final_info["raw_action"].copy())
        exec_actions.append(final_info["executed_action"].copy())
        rewards.append(float(reward))
        raw_delta_norms.append(final_info["raw_action_delta_norm"])
        exec_delta_norms.append(final_info["exec_action_delta_norm"])

    return {
        "trajectory": np.asarray(trajectory, dtype=np.float32),
        "distances": np.asarray(distances, dtype=np.float32),
        "raw_actions": np.asarray(raw_actions, dtype=np.float32).reshape(-1, 2),
        "exec_actions": np.asarray(exec_actions, dtype=np.float32).reshape(-1, 2),
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


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print(
            f"Model file not found: {model_path}\n"
            "Train first, for example:\n"
            "  python -m train.simple_env --total-steps 100000 --use-kf 1 --seed 0",
            file=sys.stderr,
        )
        sys.exit(1)

    env = ContinuousNavEnv(use_kf=bool(args.use_kf), seed=args.seed)
    model = SAC.load(str(model_path), env=env)

    episodes = []
    for episode_idx in range(args.episodes):
        episode = run_episode(
            model=model,
            env=env,
            seed=args.seed + episode_idx,
            deterministic=bool(args.deterministic),
        )
        episodes.append(episode)
        if episode["success"]:
            status = "success"
        elif episode["collision"]:
            status = "collision"
        elif episode["out_of_bounds"]:
            status = "out_of_bounds"
        elif episode["timeout"]:
            status = "timeout"
        else:
            status = "terminated"
        print(
            f"Episode {episode_idx + 1}: return={episode['return']:.2f}, "
            f"steps={episode['steps']}, status={status}"
        )

    env.close()

    selected = max(episodes, key=lambda ep: ep["return"])
    smoothed = smooth_trajectory_bspline(selected["trajectory"], num_points=300, smoothing=2.0)

    plot_path(
        env,
        selected["trajectory"],
        smoothed,
        fig_dir / "path_kf_bspline.png",
    )
    plot_reward_curve(selected["rewards"], fig_dir / "eval_reward_curve.png")
    plot_step_distance(selected["distances"], fig_dir / "eval_step_distance.png")
    plot_action_smoothing(
        selected["raw_actions"],
        selected["exec_actions"],
        selected["raw_delta_norms"],
        selected["exec_delta_norms"],
        fig_dir / "action_smoothing_comparison.png",
    )

    returns = np.asarray([ep["return"] for ep in episodes], dtype=np.float32)
    steps = np.asarray([ep["steps"] for ep in episodes], dtype=np.float32)
    successes = np.asarray([ep["success"] for ep in episodes], dtype=np.float32)
    collisions = np.asarray([ep["collision"] for ep in episodes], dtype=np.float32)
    out_of_bounds = np.asarray([ep["out_of_bounds"] for ep in episodes], dtype=np.float32)
    timeouts = np.asarray([ep["timeout"] for ep in episodes], dtype=np.float32)
    raw_delta_all = np.concatenate([ep["raw_delta_norms"] for ep in episodes])
    exec_delta_all = np.concatenate([ep["exec_delta_norms"] for ep in episodes])
    mean_raw_delta = float(np.mean(raw_delta_all)) if len(raw_delta_all) else 0.0
    mean_exec_delta = float(np.mean(exec_delta_all)) if len(exec_delta_all) else 0.0
    smoothing_ratio = mean_exec_delta / (mean_raw_delta + 1e-8)

    print("\nEvaluation summary")
    print(f"Average return: {float(np.mean(returns)):.3f}")
    print(f"Average steps: {float(np.mean(steps)):.3f}")
    print(f"Success rate: {float(np.mean(successes)):.3f}")
    print(f"Collision rate: {float(np.mean(collisions)):.3f}")
    print(f"Out-of-bounds rate: {float(np.mean(out_of_bounds)):.3f}")
    print(f"Timeout rate: {float(np.mean(timeouts)):.3f}")
    print(f"Mean raw action delta: {mean_raw_delta:.6f}")
    print(f"Mean executed action delta: {mean_exec_delta:.6f}")
    print(f"Smoothing ratio: {smoothing_ratio:.6f}")
    print(f"Saved figures to: {fig_dir}")


if __name__ == "__main__":
    main()
