"""Train fixed-KF v/w SAC with near-target slowdown reward shaping."""

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

from envs.robot_near_target_reward_wrapper import make_robot_near_target_env
from train.robot_diffdrive_complex import add_transition_to_replay_buffer, replay_buffer_size
from train.robot_diffdrive_kf_ablation import (
    KFAblationMetricsCallback,
    call_set_curriculum_progress,
)
from utils.apf_diffdrive import DiffDriveAPFPolicy
from utils.training_callbacks import TensorboardMetricsCallback, TqdmTrainingCallback


DEFAULT_RUN_NAME = "vw_apf_sac_fixed_kf_in_loop_kf_state_near_target"
DEFAULT_MODEL_PATH = Path("outputs/vw/ablation") / f"{DEFAULT_RUN_NAME}.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fixed-KF v/w SAC with near-target slowdown reward.")
    parser.add_argument("--total-steps", type=int, default=500_000)
    parser.add_argument("--apf-warmup-episodes", type=int, default=1_500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--run-name", type=str, default=DEFAULT_RUN_NAME)
    parser.add_argument("--save-dir", type=str, default="outputs")
    parser.add_argument("--tb-log-freq", type=int, default=100)
    parser.add_argument("--buffer-size", type=int, default=500_000)
    parser.add_argument("--show-progress", type=int, default=1, choices=[0, 1])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--slowdown-radius", type=float, default=8.0)
    parser.add_argument("--linear-weight", type=float, default=0.35)
    parser.add_argument("--angular-weight", type=float, default=0.15)
    return parser.parse_args()


def make_env(
    seed: int | None,
    slowdown_radius: float,
    linear_weight: float,
    angular_weight: float,
):
    return make_robot_near_target_env(
        smoother="fixed",
        obs_mode="kf_state",
        kf_curriculum="none",
        seed=seed,
        slowdown_radius=slowdown_radius,
        linear_weight=linear_weight,
        angular_weight=angular_weight,
    )


class NearTargetMetricsCallback(BaseCallback):
    """Write near-target slowdown reward terms to TensorBoard."""

    def __init__(self, writer: SummaryWriter, log_freq: int = 100, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.writer = writer
        self.log_freq = max(1, int(log_freq))

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq != 0:
            return True
        infos = self.locals.get("infos", [])
        if not infos:
            return True
        info = infos[0]
        step = int(self.num_timesteps)
        self.writer.add_scalar(
            "near_target/slowdown_penalty",
            float(info.get("near_target_slowdown_penalty", 0.0)),
            step,
        )
        self.writer.add_scalar(
            "near_target/slowdown_factor",
            float(info.get("near_target_slowdown_factor", 0.0)),
            step,
        )
        self.writer.add_scalar(
            "near_target/linear_penalty",
            float(info.get("near_target_linear_penalty", 0.0)),
            step,
        )
        self.writer.add_scalar(
            "near_target/angular_penalty",
            float(info.get("near_target_angular_penalty", 0.0)),
            step,
        )
        return True

    def _on_training_end(self) -> None:
        self.writer.flush()


def run_apf_warmup(
    model: SAC,
    episodes: int,
    seed: int,
    slowdown_radius: float,
    linear_weight: float,
    angular_weight: float,
    writer: SummaryWriter,
    show_progress: bool,
) -> None:
    if episodes <= 0:
        return

    warmup_env = make_env(
        seed=seed,
        slowdown_radius=slowdown_radius,
        linear_weight=linear_weight,
        angular_weight=angular_weight,
    )
    call_set_curriculum_progress(warmup_env, 1.0)
    apf_policy = DiffDriveAPFPolicy()
    iterator = range(episodes)
    if show_progress:
        iterator = tqdm(iterator, desc="Robot v/w APF warm-up", unit="episode")

    total_steps = 0
    successes = 0
    collisions = 0
    out_of_bounds = 0
    timeouts = 0
    slowdown_penalties: list[float] = []

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
            next_obs, reward, terminated, truncated, info = warmup_env.step(apf_output.action)
            done = terminated or truncated
            add_transition_to_replay_buffer(
                model=model,
                obs=obs,
                next_obs=next_obs,
                action=apf_output.action,
                reward=reward,
                done=done,
                info=info,
            )
            obs = next_obs
            episode_return += float(reward)
            episode_steps += 1
            total_steps += 1
            final_info = info
            slowdown_penalties.append(float(info.get("near_target_slowdown_penalty", 0.0)))

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
        writer.add_scalar(
            "apf/near_target_slowdown_penalty",
            float(final_info.get("near_target_slowdown_penalty", 0.0)),
            episode_idx,
        )

    warmup_env.close()
    writer.add_scalar("apf/total_steps", total_steps, episodes)
    writer.add_scalar("apf/replay_buffer_size", replay_buffer_size(model), episodes)
    writer.add_scalar("apf/success_rate", successes / max(1, episodes), episodes)
    writer.add_scalar("apf/collision_rate", collisions / max(1, episodes), episodes)
    writer.add_scalar("apf/out_of_bounds_rate", out_of_bounds / max(1, episodes), episodes)
    writer.add_scalar("apf/timeout_rate", timeouts / max(1, episodes), episodes)
    writer.add_scalar(
        "apf/near_target_slowdown_penalty_mean",
        float(np.mean(slowdown_penalties)) if slowdown_penalties else 0.0,
        episodes,
    )
    writer.flush()

    print(
        "Robot v/w APF warm-up finished: "
        f"episodes={episodes}, transitions={total_steps}, "
        f"buffer_size={replay_buffer_size(model)}, "
        f"success_rate={successes / max(1, episodes):.3f}, "
        f"collision_rate={collisions / max(1, episodes):.3f}, "
        f"out_of_bounds_rate={out_of_bounds / max(1, episodes):.3f}, "
        f"timeout_rate={timeouts / max(1, episodes):.3f}"
    )


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    save_dir = Path(args.save_dir)
    run_name = args.run_name or model_path.stem or DEFAULT_RUN_NAME
    log_dir = save_dir / "logs" / "robot_diffdrive_complex" / run_name
    custom_tb_dir = log_dir / "custom"

    model_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    custom_tb_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    set_random_seed(args.seed)

    env = make_env(
        seed=args.seed,
        slowdown_radius=args.slowdown_radius,
        linear_weight=args.linear_weight,
        angular_weight=args.angular_weight,
    )
    env = Monitor(env, filename=str(log_dir / "train"))
    writer = SummaryWriter(log_dir=str(custom_tb_dir))
    writer.add_text("robot/run_name", run_name, 0)
    writer.add_text("robot/model_path", str(model_path), 0)
    writer.add_text("robot/smoother", "fixed", 0)
    writer.add_text("robot/obs_mode", "kf_state", 0)
    writer.add_text("robot/kf_curriculum", "none", 0)
    writer.add_scalar("robot/obs_dim", float(env.observation_space.shape[0]), 0)
    writer.add_scalar("robot/apf_warmup_episodes", float(args.apf_warmup_episodes), 0)
    writer.add_scalar("robot/buffer_size", float(args.buffer_size), 0)
    writer.add_scalar("near_target/slowdown_radius", float(args.slowdown_radius), 0)
    writer.add_scalar("near_target/linear_weight", float(args.linear_weight), 0)
    writer.add_scalar("near_target/angular_weight", float(args.angular_weight), 0)

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
        slowdown_radius=args.slowdown_radius,
        linear_weight=args.linear_weight,
        angular_weight=args.angular_weight,
        writer=writer,
        show_progress=bool(args.show_progress),
    )

    callbacks: list[BaseCallback] = [
        TensorboardMetricsCallback(writer=writer, log_freq=args.tb_log_freq),
        KFAblationMetricsCallback(
            writer=writer,
            total_timesteps=args.total_steps,
            kf_curriculum="none",
            log_freq=args.tb_log_freq,
        ),
        NearTargetMetricsCallback(writer=writer, log_freq=args.tb_log_freq),
    ]
    if args.show_progress:
        callbacks.insert(0, TqdmTrainingCallback(total_timesteps=args.total_steps))

    print(
        f"Training fixed-KF near-target v/w SAC for {args.total_steps} steps, "
        f"obs_dim={env.observation_space.shape[0]}, learning_starts={learning_starts}, "
        f"slowdown_radius={args.slowdown_radius}, "
        f"linear_weight={args.linear_weight}, angular_weight={args.angular_weight}"
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
        "  python -m evaluate.robot_diffdrive_kf_near_target "
        f"--model-path {model_path}"
    )


if __name__ == "__main__":
    main()
