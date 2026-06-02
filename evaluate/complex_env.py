"""Evaluate SAC + KF on the complex static maze environment without B-spline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
from stable_baselines3 import SAC

from envs.complex_nav_env import ComplexNavEnv
from filters.kalman_action_smoother import SMOOTHER_OBS_DIM, VALID_SMOOTHER_TYPES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SAC policy on ComplexNavEnv.")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--use-kf", type=int, default=1, choices=[0, 1])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fig-dir", type=str, default=None)
    parser.add_argument("--deterministic", type=int, default=1, choices=[0, 1])
    parser.add_argument("--smoother-type", type=str, default="current_kf", choices=VALID_SMOOTHER_TYPES)
    parser.add_argument("--smoother-beta", type=float, default=0.85)
    parser.add_argument("--singer-tau", type=float, default=3.0)
    parser.add_argument("--velocity-process-noise-std", type=float, default=0.05)
    parser.add_argument("--rate-process-noise-std", type=float, default=0.02)
    parser.add_argument("--measurement-noise-std", type=float, default=0.3)
    parser.add_argument("--max-linear-speed", type=float, default=1.2)
    parser.add_argument("--max-angular-speed", type=float, default=1.0)
    return parser.parse_args()


def run_episode(model: SAC, env: ComplexNavEnv, seed: int, deterministic: bool) -> dict:
    obs, info = env.reset(seed=seed)
    trajectory = [info["position"].copy()]
    distances = [info["distance_to_goal"]]
    raw_actions = []
    exec_actions = []
    rewards = []
    raw_delta_norms = []
    exec_delta_norms = []
    headings = []
    linear_velocities = []
    angular_velocities = []
    smoother_features = []

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
        headings.append(final_info["heading"])
        linear_velocities.append(final_info["linear_velocity"])
        angular_velocities.append(final_info["angular_velocity"])
        smoother_features.append(final_info["smoother_features"].copy())

    return {
        "trajectory": np.asarray(trajectory, dtype=np.float32),
        "distances": np.asarray(distances, dtype=np.float32),
        "raw_actions": np.asarray(raw_actions, dtype=np.float32).reshape(-1, 2),
        "exec_actions": np.asarray(exec_actions, dtype=np.float32).reshape(-1, 2),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "raw_delta_norms": np.asarray(raw_delta_norms, dtype=np.float32),
        "exec_delta_norms": np.asarray(exec_delta_norms, dtype=np.float32),
        "headings": np.asarray(headings, dtype=np.float32),
        "linear_velocities": np.asarray(linear_velocities, dtype=np.float32),
        "angular_velocities": np.asarray(angular_velocities, dtype=np.float32),
        "smoother_features": np.asarray(smoother_features, dtype=np.float32).reshape(-1, SMOOTHER_OBS_DIM),
        "return": float(np.sum(rewards)),
        "steps": len(rewards),
        "success": bool(final_info.get("success", False)),
        "collision": bool(final_info.get("collision", False)),
        "timeout": bool(final_info.get("timeout", False)),
        "out_of_bounds": bool(final_info.get("out_of_bounds", False)),
    }


def action_jerk_energy(actions: np.ndarray) -> float:
    if len(actions) < 3:
        return 0.0
    jerk = actions[2:] - 2.0 * actions[1:-1] + actions[:-2]
    return float(np.mean(np.sum(jerk * jerk, axis=1)))


def high_freq_energy_ratio(actions: np.ndarray, cutoff_ratio: float = 0.25) -> float:
    if len(actions) < 8:
        return 0.0
    centered = actions - np.mean(actions, axis=0, keepdims=True)
    fft_vals = np.fft.rfft(centered, axis=0)
    power = np.sum(np.abs(fft_vals) ** 2, axis=1)
    if len(power) <= 1 or float(np.sum(power[1:])) <= 1e-12:
        return 0.0
    cutoff_idx = max(1, int(len(power) * cutoff_ratio))
    high = float(np.sum(power[cutoff_idx:]))
    total = float(np.sum(power[1:]))
    return high / max(total, 1e-12)


def sign_change_rate(actions: np.ndarray) -> float:
    if len(actions) < 3:
        return 0.0
    delta = np.diff(actions, axis=0)
    signs = np.sign(delta)
    changes = signs[1:] * signs[:-1] < 0
    return float(np.mean(changes))


def plot_complex_path(env: ComplexNavEnv, trajectory: np.ndarray, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("SAC + KF Complex Navigation Path")
    ax.set_xlim(0, env.map_size)
    ax.set_ylim(0, env.map_size)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.3)

    for rect in env.rectangles:
        rect_min = rect["min"]
        rect_max = rect["max"]
        patch = Rectangle(
            rect_min,
            rect_max[0] - rect_min[0],
            rect_max[1] - rect_min[1],
            facecolor="tab:red",
            edgecolor="tab:red",
            alpha=0.28,
        )
        ax.add_patch(patch)

    for circle in env.circles:
        patch = Circle(
            circle["center"],
            circle["radius"],
            facecolor="tab:red",
            edgecolor="tab:red",
            alpha=0.28,
        )
        ax.add_patch(patch)

    waypoints = np.asarray(env.waypoints, dtype=np.float32)
    ax.plot(waypoints[:, 0], waypoints[:, 1], "--", color="0.55", linewidth=1.2, label="APF waypoints")
    ax.scatter(env.start[0], env.start[1], c="tab:green", s=80, label="Start", zorder=4)
    ax.scatter(env.goal[0], env.goal[1], c="tab:purple", s=110, marker="*", label="Goal", zorder=4)
    if len(trajectory) > 0:
        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color="tab:blue",
            linewidth=2.0,
            label="Executed trajectory",
        )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_reward_curve(rewards: np.ndarray, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    steps = np.arange(len(rewards))
    cumulative = np.cumsum(rewards)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title("Complex Env Evaluation Reward")
    ax.plot(steps, rewards, label="Step reward", color="tab:blue", alpha=0.75)
    ax.plot(steps, cumulative, label="Cumulative reward", color="tab:orange", linewidth=2.0)
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_step_distance(distances: np.ndarray, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    steps = np.arange(len(distances))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title("Complex Env Distance to Goal")
    ax.plot(steps, distances, color="tab:green", linewidth=2.0)
    ax.set_xlabel("Step")
    ax.set_ylabel("Distance to goal")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_action_smoothing(
    raw_actions: np.ndarray,
    exec_actions: np.ndarray,
    raw_delta_norms: np.ndarray,
    exec_delta_norms: np.ndarray,
    smoother_type: str,
    save_path: Path,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    steps = np.arange(len(raw_actions))
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.suptitle(f"Complex Env Raw [v, omega] vs Executed Action ({smoother_type})")

    if len(raw_actions) > 0:
        axes[0].plot(steps, raw_actions[:, 0], label="v_raw", color="tab:blue", alpha=0.7)
        axes[0].plot(steps, exec_actions[:, 0], label="v_exec", color="tab:orange", linewidth=2.0)
        axes[1].plot(steps, raw_actions[:, 1], label="omega_raw", color="tab:blue", alpha=0.7)
        axes[1].plot(steps, exec_actions[:, 1], label="omega_exec", color="tab:orange", linewidth=2.0)
        axes[2].plot(steps, raw_delta_norms, label="||delta raw||", color="tab:red", alpha=0.75)
        axes[2].plot(steps, exec_delta_norms, label="||delta exec||", color="tab:green", linewidth=2.0)

    axes[0].set_ylabel("v_norm")
    axes[1].set_ylabel("omega_norm")
    axes[2].set_ylabel("Delta norm")
    axes[2].set_xlabel("Step")
    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    model_path = (
        Path(args.model_path)
        if args.model_path
        else Path("outputs") / "models" / "complex" / f"sac_{args.smoother_type}_vw_complex_nav_seed{args.seed}.zip"
    )
    fig_dir = (
        Path(args.fig_dir)
        if args.fig_dir
        else Path("outputs") / "figures" / "complex_env" / f"{args.smoother_type}_seed{args.seed}"
    )
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print(
            f"Model file not found: {model_path}\n"
            "Train first, for example:\n"
            "  python -m train.complex_env --total-steps 500000 --use-kf 1 --apf-warmup-episodes 2000 --seed 0",
            file=sys.stderr,
        )
        sys.exit(1)

    smoother_kwargs = {
        "beta": args.smoother_beta,
        "tau": args.singer_tau,
        "velocity_process_noise_std": args.velocity_process_noise_std,
        "rate_process_noise_std": args.rate_process_noise_std,
        "measurement_noise_std": args.measurement_noise_std,
    }
    env = ComplexNavEnv(
        use_kf=bool(args.use_kf),
        seed=args.seed,
        smoother_type=args.smoother_type,
        smoother_kwargs=smoother_kwargs,
        max_linear_speed=args.max_linear_speed,
        max_angular_speed=args.max_angular_speed,
    )
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

    selected = max(episodes, key=lambda ep: ep["return"])
    plot_complex_path(env, selected["trajectory"], fig_dir / f"path_{args.smoother_type}_vw_complex.png")
    plot_reward_curve(selected["rewards"], fig_dir / f"eval_reward_curve_{args.smoother_type}_vw_complex.png")
    plot_step_distance(selected["distances"], fig_dir / f"eval_step_distance_{args.smoother_type}_vw_complex.png")
    plot_action_smoothing(
        selected["raw_actions"],
        selected["exec_actions"],
        selected["raw_delta_norms"],
        selected["exec_delta_norms"],
        args.smoother_type,
        fig_dir / f"action_smoothing_{args.smoother_type}_vw_complex.png",
    )
    env.close()

    returns = np.asarray([ep["return"] for ep in episodes], dtype=np.float32)
    steps = np.asarray([ep["steps"] for ep in episodes], dtype=np.float32)
    successes = np.asarray([ep["success"] for ep in episodes], dtype=np.float32)
    collisions = np.asarray([ep["collision"] for ep in episodes], dtype=np.float32)
    out_of_bounds = np.asarray([ep["out_of_bounds"] for ep in episodes], dtype=np.float32)
    timeouts = np.asarray([ep["timeout"] for ep in episodes], dtype=np.float32)
    raw_delta_all = np.concatenate([ep["raw_delta_norms"] for ep in episodes])
    exec_delta_all = np.concatenate([ep["exec_delta_norms"] for ep in episodes])
    raw_actions_all = np.concatenate([ep["raw_actions"] for ep in episodes], axis=0)
    exec_actions_all = np.concatenate([ep["exec_actions"] for ep in episodes], axis=0)
    mean_raw_delta = float(np.mean(raw_delta_all)) if len(raw_delta_all) else 0.0
    mean_exec_delta = float(np.mean(exec_delta_all)) if len(exec_delta_all) else 0.0
    smoothing_ratio = mean_exec_delta / (mean_raw_delta + 1e-8)
    raw_jerk = action_jerk_energy(raw_actions_all)
    exec_jerk = action_jerk_energy(exec_actions_all)
    jerk_ratio = exec_jerk / (raw_jerk + 1e-8)
    raw_hf = high_freq_energy_ratio(raw_actions_all)
    exec_hf = high_freq_energy_ratio(exec_actions_all)
    hf_reduction = 1.0 - exec_hf / (raw_hf + 1e-8)
    raw_sign = sign_change_rate(raw_actions_all)
    exec_sign = sign_change_rate(exec_actions_all)
    sign_ratio = exec_sign / (raw_sign + 1e-8)

    print("\nComplex evaluation summary")
    print(f"Average return: {float(np.mean(returns)):.3f}")
    print(f"Average steps: {float(np.mean(steps)):.3f}")
    print(f"Success rate: {float(np.mean(successes)):.3f}")
    print(f"Collision rate: {float(np.mean(collisions)):.3f}")
    print(f"Out-of-bounds rate: {float(np.mean(out_of_bounds)):.3f}")
    print(f"Timeout rate: {float(np.mean(timeouts)):.3f}")
    print(f"Mean raw action delta: {mean_raw_delta:.6f}")
    print(f"Mean executed action delta: {mean_exec_delta:.6f}")
    print(f"Smoothing ratio: {smoothing_ratio:.6f}")
    print(f"Raw jerk energy: {raw_jerk:.6f}")
    print(f"Executed jerk energy: {exec_jerk:.6f}")
    print(f"Jerk ratio: {jerk_ratio:.6f}")
    print(f"Raw high frequency energy ratio: {raw_hf:.6f}")
    print(f"Executed high frequency energy ratio: {exec_hf:.6f}")
    print(f"High frequency energy ratio reduction: {hf_reduction:.6f}")
    print(f"Raw sign change rate: {raw_sign:.6f}")
    print(f"Executed sign change rate: {exec_sign:.6f}")
    print(f"Sign change rate ratio: {sign_ratio:.6f}")
    print(f"Saved figures to: {fig_dir}")


if __name__ == "__main__":
    main()
