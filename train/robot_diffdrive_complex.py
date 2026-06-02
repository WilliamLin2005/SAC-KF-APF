"""Train SAC on the robotized differential-drive complex environment."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from envs.robot_diffdrive_complex_env import RobotDiffDriveComplexEnv
from utils.apf_diffdrive import DiffDriveAPFPolicy
from utils.training_callbacks import TensorboardMetricsCallback, TqdmTrainingCallback


ROBOT_VW_GROUP_DIR = Path("vw") / "group1_robot_vw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAC on RobotDiffDriveComplexEnv.")
    parser.add_argument("--use-kf", type=int, default=1, choices=[0, 1])
    parser.add_argument("--aug-prev-action", type=int, default=0, choices=[0, 1])
    parser.add_argument("--total-steps", type=int, default=300_000)
    parser.add_argument("--apf-warmup-episodes", type=int, default=1_500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default="outputs")
    parser.add_argument("--tb-log-freq", type=int, default=100)
    parser.add_argument("--buffer-size", type=int, default=500_000)
    parser.add_argument("--show-progress", type=int, default=1, choices=[0, 1])
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def run_name(use_kf: bool, aug_prev_action: bool) -> str:
    kf_part = "kf" if use_kf else "no_kf"
    aug_part = "aug" if aug_prev_action else "no_aug"
    return f"vw_{kf_part}_footprint_{aug_part}"


def training_run_name(
    requested_run_name: str | None,
    requested_model_path: str | None,
    model_path: Path,
    use_kf: bool,
    aug_prev_action: bool,
) -> str:
    if requested_run_name:
        return requested_run_name
    if requested_model_path:
        return model_path.stem
    return run_name(use_kf=use_kf, aug_prev_action=aug_prev_action)


def default_model_path(save_dir: Path, use_kf: bool, aug_prev_action: bool) -> Path:
    if use_kf and aug_prev_action:
        name = "B_vw_kf_footprint_aug.zip"
    elif use_kf:
        name = "A_vw_kf_footprint_no_aug.zip"
    elif aug_prev_action:
        name = "D_vw_no_kf_footprint_aug.zip"
    else:
        name = "C_vw_no_kf_footprint_no_aug.zip"
    return save_dir / ROBOT_VW_GROUP_DIR / name


def replay_buffer_size(model: SAC) -> int:
    replay_buffer = model.replay_buffer
    if replay_buffer is None:
        return 0
    return replay_buffer.buffer_size if replay_buffer.full else replay_buffer.pos


def add_transition_to_replay_buffer(
    model: SAC,
    obs: np.ndarray,
    next_obs: np.ndarray,
    action: np.ndarray,
    reward: float,
    done: bool,
    info: dict,
) -> None:
    if model.replay_buffer is None:
        raise RuntimeError("SAC replay buffer is not initialized.")

    model.replay_buffer.add(
        obs=np.asarray(obs, dtype=np.float32)[None, :],
        next_obs=np.asarray(next_obs, dtype=np.float32)[None, :],
        action=np.asarray(action, dtype=np.float32)[None, :],
        reward=np.asarray([reward], dtype=np.float32),
        done=np.asarray([done], dtype=np.float32),
        infos=[info],
    )


def make_env(use_kf: bool, aug_prev_action: bool, seed: int | None) -> RobotDiffDriveComplexEnv:
    return RobotDiffDriveComplexEnv(
        use_kf=use_kf,
        aug_prev_action=aug_prev_action,
        seed=seed,
    )


def run_apf_warmup(
    model: SAC,
    episodes: int,
    seed: int,
    use_kf: bool,
    aug_prev_action: bool,
    writer: SummaryWriter,
    show_progress: bool,
) -> None:
    if episodes <= 0:
        return

    warmup_env = make_env(use_kf=use_kf, aug_prev_action=aug_prev_action, seed=seed)
    apf_policy = DiffDriveAPFPolicy()
    iterator = range(episodes)
    if show_progress:
        iterator = tqdm(iterator, desc="Robot v/w APF warm-up", unit="episode")

    total_steps = 0
    successes = 0
    collisions = 0
    out_of_bounds = 0
    timeouts = 0

    for episode_idx in iterator:
        apf_policy.reset()
        obs, _ = warmup_env.reset(seed=seed + 20_000 + episode_idx)
        terminated = False
        truncated = False
        episode_return = 0.0
        episode_values: list[float] = []
        episode_potentials: list[float] = []
        episode_steps = 0
        final_info = {}

        while not (terminated or truncated):
            apf_output = apf_policy.act(warmup_env)
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
            episode_values.append(apf_output.state_value)
            episode_potentials.append(apf_output.potential)
            episode_steps += 1
            total_steps += 1
            final_info = info

        successes += int(bool(final_info.get("success", False)))
        collisions += int(bool(final_info.get("collision", False)))
        out_of_bounds += int(bool(final_info.get("out_of_bounds", False)))
        timeouts += int(bool(final_info.get("timeout", False)))
        writer.add_scalar("apf/episode_return", episode_return, episode_idx)
        writer.add_scalar("apf/episode_steps", episode_steps, episode_idx)
        writer.add_scalar("apf/state_value_mean", float(np.mean(episode_values)), episode_idx)
        writer.add_scalar("apf/potential_mean", float(np.mean(episode_potentials)), episode_idx)
        writer.add_scalar("apf/success", float(final_info.get("success", False)), episode_idx)
        writer.add_scalar("apf/collision", float(final_info.get("collision", False)), episode_idx)
        writer.add_scalar("apf/out_of_bounds", float(final_info.get("out_of_bounds", False)), episode_idx)
        writer.add_scalar("apf/timeout", float(final_info.get("timeout", False)), episode_idx)

    warmup_env.close()
    writer.add_scalar("apf/use_kf", float(use_kf), episodes)
    writer.add_scalar("apf/aug_prev_action", float(aug_prev_action), episodes)
    writer.add_scalar("apf/total_steps", total_steps, episodes)
    writer.add_scalar("apf/replay_buffer_size", replay_buffer_size(model), episodes)
    writer.add_scalar("apf/success_rate", successes / max(1, episodes), episodes)
    writer.add_scalar("apf/collision_rate", collisions / max(1, episodes), episodes)
    writer.add_scalar("apf/out_of_bounds_rate", out_of_bounds / max(1, episodes), episodes)
    writer.add_scalar("apf/timeout_rate", timeouts / max(1, episodes), episodes)
    writer.flush()

    print(
        "Robot v/w APF warm-up finished: "
        f"episodes={episodes}, transitions={total_steps}, "
        f"use_kf={int(use_kf)}, aug_prev_action={int(aug_prev_action)}, "
        f"buffer_size={replay_buffer_size(model)}, "
        f"success_rate={successes / max(1, episodes):.3f}, "
        f"collision_rate={collisions / max(1, episodes):.3f}, "
        f"out_of_bounds_rate={out_of_bounds / max(1, episodes):.3f}, "
        f"timeout_rate={timeouts / max(1, episodes):.3f}"
    )


def main() -> None:
    args = parse_args()
    use_kf = bool(args.use_kf)
    aug_prev_action = bool(args.aug_prev_action)
    save_dir = Path(args.save_dir)
    model_path = (
        Path(args.model_path)
        if args.model_path
        else default_model_path(save_dir, use_kf=use_kf, aug_prev_action=aug_prev_action)
    )
    name = training_run_name(
        requested_run_name=args.run_name,
        requested_model_path=args.model_path,
        model_path=model_path,
        use_kf=use_kf,
        aug_prev_action=aug_prev_action,
    )
    log_dir = save_dir / "logs" / "robot_diffdrive_complex" / name
    custom_tb_dir = log_dir / "custom"

    model_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    custom_tb_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    set_random_seed(args.seed)

    env = make_env(use_kf=use_kf, aug_prev_action=aug_prev_action, seed=args.seed)
    env = Monitor(env, filename=str(log_dir / "train"))
    writer = SummaryWriter(log_dir=str(custom_tb_dir))
    writer.add_scalar("robot/use_kf", float(use_kf), 0)
    writer.add_scalar("robot/aug_prev_action", float(aug_prev_action), 0)
    writer.add_scalar("robot/obs_dim", float(env.observation_space.shape[0]), 0)
    writer.add_scalar("robot/apf_warmup_episodes", float(args.apf_warmup_episodes), 0)
    writer.add_scalar("robot/buffer_size", float(args.buffer_size), 0)
    writer.add_text("robot/run_name", name, 0)
    writer.add_text("robot/model_path", str(model_path), 0)

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
        use_kf=use_kf,
        aug_prev_action=aug_prev_action,
        writer=writer,
        show_progress=bool(args.show_progress),
    )

    callbacks = [TensorboardMetricsCallback(writer=writer, log_freq=args.tb_log_freq)]
    if args.show_progress:
        callbacks.insert(0, TqdmTrainingCallback(total_timesteps=args.total_steps))

    print(
        f"Training robot v/w SAC for {args.total_steps} steps, "
        f"use_kf={int(use_kf)}, aug_prev_action={int(aug_prev_action)}, "
        f"obs_dim={env.observation_space.shape[0]}, learning_starts={learning_starts}"
    )
    try:
        model.learn(
            total_timesteps=args.total_steps,
            callback=CallbackList(callbacks),
            log_interval=10,
            tb_log_name=f"sac_{name}",
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
        "  python -m evaluate.robot_diffdrive_complex "
        f"--model-path {model_path} --use-kf {args.use_kf} --aug-prev-action {args.aug_prev_action}"
    )


if __name__ == "__main__":
    main()
