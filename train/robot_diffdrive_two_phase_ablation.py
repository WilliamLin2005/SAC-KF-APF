"""Train two-phase v/w SAC with configurable command smoothing ablations."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from envs.robot_command_smoothing_wrappers import OBS_MODE_CHOICES
from envs.robot_two_phase_reward_wrapper import make_robot_two_phase_env
from train.robot_diffdrive_complex import add_transition_to_replay_buffer, replay_buffer_size
from train.robot_diffdrive_kf_ablation import (
    KFAblationMetricsCallback,
    call_set_curriculum_progress,
)
from train.robot_diffdrive_kf_two_phase import TwoPhaseMetricsCallback
from utils.apf_diffdrive import DiffDriveAPFPolicy
from utils.training_callbacks import TensorboardMetricsCallback, TqdmTrainingCallback


SMOOTHER_CHOICES = ("none", "fixed")
CURRICULUM_CHOICES = ("none",)
ABLATION_DIR = Path("vw") / "ablation"


def smoother_label(smoother: str) -> str:
    return "no_kf" if smoother == "none" else "fixed_kf"


def default_run_name(smoother: str, obs_mode: str) -> str:
    return f"vw_tp_{smoother_label(smoother)}_{obs_mode}"


def default_model_path(save_dir: Path, smoother: str, obs_mode: str) -> Path:
    return save_dir / ABLATION_DIR / f"{default_run_name(smoother, obs_mode)}.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train two-phase v/w SAC with configurable no-KF/fixed-KF smoothing."
    )
    parser.add_argument("--smoother", type=str, default="none", choices=SMOOTHER_CHOICES)
    parser.add_argument("--obs-mode", type=str, default="prev_exec", choices=OBS_MODE_CHOICES)
    parser.add_argument("--kf-curriculum", type=str, default="none", choices=CURRICULUM_CHOICES)
    parser.add_argument("--total-steps", type=int, default=500_000)
    parser.add_argument("--apf-warmup-episodes", type=int, default=1_500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default="outputs")
    parser.add_argument("--tb-log-freq", type=int, default=100)
    parser.add_argument("--buffer-size", type=int, default=500_000)
    parser.add_argument("--show-progress", type=int, default=1, choices=[0, 1])
    parser.add_argument("--device", type=str, default="auto")
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


def two_phase_apf_warmup_action(env, action: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """Slow APF near the goal while preserving a low-speed crawl into the goal radius."""

    base = env.unwrapped
    distance = float(np.linalg.norm(base.goal - base.position))
    if distance > args.slowdown_radius:
        return action.astype(np.float32)

    command = base.normalized_action_to_command(action)
    factor = float(
        np.clip(
            (distance - base.goal_radius) / max(args.slowdown_radius - base.goal_radius, 1e-8),
            0.0,
            1.0,
        )
    )
    crawl_v = min(float(args.success_linear_threshold) * 0.8, float(base.max_linear_speed) * 0.2)
    crawl_w = min(float(args.success_angular_threshold) * 0.8, float(base.max_angular_speed) * 0.2)
    linear_limit = max(float(base.max_linear_speed) * factor, crawl_v)
    angular_limit = max(float(base.max_angular_speed) * factor, crawl_w)
    command[0] = min(float(command[0]), linear_limit)
    command[1] = float(np.clip(command[1], -angular_limit, angular_limit))
    return base.command_to_normalized_action(command).astype(np.float32)


def run_apf_warmup(
    model: SAC,
    episodes: int,
    seed: int,
    args: argparse.Namespace,
    writer: SummaryWriter,
    show_progress: bool,
) -> None:
    if episodes <= 0:
        return

    warmup_env = make_env(seed=seed, args=args)
    call_set_curriculum_progress(warmup_env, 1.0)
    apf_policy = DiffDriveAPFPolicy()
    iterator = range(episodes)
    if show_progress:
        iterator = tqdm(iterator, desc="Robot v/w two-phase APF warm-up", unit="episode")

    total_steps = 0
    successes = 0
    collisions = 0
    out_of_bounds = 0
    timeouts = 0
    docking_entries = 0
    mismatch_values: list[float] = []

    for episode_idx in iterator:
        apf_policy.reset()
        obs, _ = warmup_env.reset(seed=seed + 20_000 + episode_idx)
        terminated = False
        truncated = False
        episode_return = 0.0
        episode_steps = 0
        final_info = {}

        while not (terminated or truncated):
            apf_output = apf_policy.act(warmup_env.unwrapped)
            action = two_phase_apf_warmup_action(warmup_env, apf_output.action, args)
            next_obs, reward, terminated, truncated, info = warmup_env.step(action)
            add_transition_to_replay_buffer(
                model=model,
                obs=obs,
                next_obs=next_obs,
                action=action,
                reward=reward,
                done=terminated or truncated,
                info=info,
            )
            obs = next_obs
            episode_return += float(reward)
            episode_steps += 1
            total_steps += 1
            final_info = info
            docking_entries += int(bool(info.get("entered_docking_zone", False)))
            mismatch_values.append(float(info.get("filter_mismatch_norm", 0.0)))

        successes += int(bool(final_info.get("success", False)))
        collisions += int(bool(final_info.get("collision", False)))
        out_of_bounds += int(bool(final_info.get("out_of_bounds", False)))
        timeouts += int(bool(final_info.get("timeout", False)))
        writer.add_scalar("apf/episode_return", episode_return, episode_idx)
        writer.add_scalar("apf/episode_steps", episode_steps, episode_idx)
        writer.add_scalar("apf/success", float(final_info.get("success", False)), episode_idx)
        writer.add_scalar("apf/collision", float(final_info.get("collision", False)), episode_idx)
        writer.add_scalar("apf/out_of_bounds", float(final_info.get("out_of_bounds", False)), episode_idx)
        writer.add_scalar("apf/timeout", float(final_info.get("timeout", False)), episode_idx)
        writer.add_scalar("apf/filter_mismatch", float(final_info.get("filter_mismatch_norm", 0.0)), episode_idx)

    warmup_env.close()
    writer.add_scalar("apf/total_steps", total_steps, episodes)
    writer.add_scalar("apf/replay_buffer_size", replay_buffer_size(model), episodes)
    writer.add_scalar("apf/success_rate", successes / max(1, episodes), episodes)
    writer.add_scalar("apf/collision_rate", collisions / max(1, episodes), episodes)
    writer.add_scalar("apf/out_of_bounds_rate", out_of_bounds / max(1, episodes), episodes)
    writer.add_scalar("apf/timeout_rate", timeouts / max(1, episodes), episodes)
    writer.add_scalar("apf/docking_entries", docking_entries, episodes)
    writer.add_scalar(
        "apf/filter_mismatch_mean",
        float(np.mean(mismatch_values)) if mismatch_values else 0.0,
        episodes,
    )
    writer.flush()

    print(
        "Robot v/w two-phase APF warm-up finished: "
        f"episodes={episodes}, transitions={total_steps}, "
        f"smoother={args.smoother}, obs_mode={args.obs_mode}, "
        f"buffer_size={replay_buffer_size(model)}, "
        f"success_rate={successes / max(1, episodes):.3f}, "
        f"collision_rate={collisions / max(1, episodes):.3f}, "
        f"out_of_bounds_rate={out_of_bounds / max(1, episodes):.3f}, "
        f"timeout_rate={timeouts / max(1, episodes):.3f}"
    )


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    model_path = (
        Path(args.model_path)
        if args.model_path
        else default_model_path(save_dir, smoother=args.smoother, obs_mode=args.obs_mode)
    )
    run_name = args.run_name or model_path.stem or default_run_name(args.smoother, args.obs_mode)
    log_dir = save_dir / "logs" / "robot_diffdrive_complex" / run_name
    custom_tb_dir = log_dir / "custom"

    model_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    custom_tb_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    set_random_seed(args.seed)

    env = make_env(seed=args.seed, args=args)
    env = Monitor(env, filename=str(log_dir / "train"))
    writer = SummaryWriter(log_dir=str(custom_tb_dir))
    writer.add_text("robot/run_name", run_name, 0)
    writer.add_text("robot/model_path", str(model_path), 0)
    writer.add_text("robot/smoother", args.smoother, 0)
    writer.add_text("robot/obs_mode", args.obs_mode, 0)
    writer.add_text("robot/kf_curriculum", args.kf_curriculum, 0)
    writer.add_text("robot/reward", "two_phase_docking", 0)
    writer.add_scalar("robot/obs_dim", float(env.observation_space.shape[0]), 0)
    writer.add_scalar("robot/apf_warmup_episodes", float(args.apf_warmup_episodes), 0)
    writer.add_scalar("robot/buffer_size", float(args.buffer_size), 0)
    writer.add_scalar("two_phase/slowdown_radius", float(args.slowdown_radius), 0)
    writer.add_scalar("two_phase/docking_entry_bonus", float(args.docking_entry_bonus), 0)
    writer.add_scalar("two_phase/success_linear_threshold", float(args.success_linear_threshold), 0)
    writer.add_scalar("two_phase/success_angular_threshold", float(args.success_angular_threshold), 0)

    learning_starts = 0 if args.apf_warmup_episodes > 0 else 5_000
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=args.buffer_size,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        learning_starts=learning_starts,
        ent_coef="auto",
        policy_kwargs={"net_arch": [256, 256]},
        tensorboard_log=str(log_dir),
        seed=args.seed,
        verbose=1,
        device=args.device,
    )

    run_apf_warmup(
        model=model,
        episodes=args.apf_warmup_episodes,
        seed=args.seed,
        args=args,
        writer=writer,
        show_progress=bool(args.show_progress),
    )

    callbacks: list[BaseCallback] = [
        TensorboardMetricsCallback(writer=writer, log_freq=args.tb_log_freq),
        KFAblationMetricsCallback(
            writer=writer,
            total_timesteps=args.total_steps,
            kf_curriculum=args.kf_curriculum,
            log_freq=args.tb_log_freq,
        ),
        TwoPhaseMetricsCallback(writer=writer, log_freq=args.tb_log_freq),
    ]
    if args.show_progress:
        callbacks.insert(0, TqdmTrainingCallback(total_timesteps=args.total_steps))

    print(
        f"Training two-phase v/w SAC for {args.total_steps} steps, "
        f"smoother={args.smoother}, obs_mode={args.obs_mode}, "
        f"obs_dim={env.observation_space.shape[0]}, learning_starts={learning_starts}, "
        f"slowdown_radius={args.slowdown_radius}, "
        f"success_v<={args.success_linear_threshold}, "
        f"success_|w|<={args.success_angular_threshold}"
    )
    try:
        model.learn(
            total_timesteps=args.total_steps,
            callback=CallbackList(callbacks),
            log_interval=10,
            tb_log_name=f"sac_{run_name}",
            progress_bar=False,
        )
        model.save(str(model_path))
    finally:
        writer.flush()
        writer.close()
        env.close()

    print(f"Saved model to: {model_path}")
    print(f"TensorBoard logs: {log_dir}")
    print("Run evaluation with:")
    print(
        "  python -m evaluate.robot_diffdrive_two_phase_ablation "
        f"--model-path {model_path} --smoother {args.smoother} "
        f"--obs-mode {args.obs_mode} --kf-curriculum {args.kf_curriculum}"
    )


if __name__ == "__main__":
    main()
