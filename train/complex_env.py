"""Train SAC + KF on the complex static maze navigation environment."""

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

from envs.complex_nav_env import ComplexNavEnv
from utils.apf_complex import ComplexAPFPolicy
from utils.training_callbacks import TensorboardMetricsCallback, TqdmTrainingCallback


SMOOTHER_TYPE_IDS = {
    "none": 0,
    "current_kf": 1,
    "rate_kf": 2,
    "singer_kf": 3,
    "ema": 4,
    "second_order_lowpass": 5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAC + KF on ComplexNavEnv.")
    parser.add_argument("--total-steps", type=int, default=500_000)
    parser.add_argument("--use-kf", type=int, default=1, choices=[0, 1])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default="outputs")
    parser.add_argument("--apf-warmup-episodes", type=int, default=2_000)
    parser.add_argument("--tb-log-freq", type=int, default=100)
    parser.add_argument("--buffer-size", type=int, default=500_000)
    parser.add_argument("--show-progress", type=int, default=1, choices=[0, 1])
    parser.add_argument("--smoother-type", type=str, default="current_kf", choices=list(SMOOTHER_TYPE_IDS))
    parser.add_argument("--smoother-beta", type=float, default=0.85)
    parser.add_argument("--singer-tau", type=float, default=3.0)
    parser.add_argument("--velocity-process-noise-std", type=float, default=0.05)
    parser.add_argument("--rate-process-noise-std", type=float, default=0.02)
    parser.add_argument("--measurement-noise-std", type=float, default=0.3)
    parser.add_argument("--max-linear-speed", type=float, default=1.2)
    parser.add_argument("--max-angular-speed", type=float, default=1.0)
    return parser.parse_args()


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


def run_apf_warmup(
    model: SAC,
    episodes: int,
    seed: int,
    use_kf: bool,
    smoother_type: str,
    smoother_kwargs: dict,
    max_linear_speed: float,
    max_angular_speed: float,
    writer: SummaryWriter,
    show_progress: bool,
) -> None:
    if episodes <= 0:
        return

    warmup_env = ComplexNavEnv(
        use_kf=use_kf,
        seed=seed,
        smoother_type=smoother_type,
        smoother_kwargs=smoother_kwargs,
        max_linear_speed=max_linear_speed,
        max_angular_speed=max_angular_speed,
    )
    apf_policy = ComplexAPFPolicy()
    iterator = range(episodes)
    if show_progress:
        iterator = tqdm(iterator, desc="Complex APF warm-up", unit="episode")

    total_steps = 0
    successes = 0
    collisions = 0
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
        timeouts += int(bool(final_info.get("timeout", False)))
        writer.add_scalar("apf/episode_return", episode_return, episode_idx)
        writer.add_scalar("apf/episode_steps", episode_steps, episode_idx)
        writer.add_scalar("apf/state_value_mean", float(np.mean(episode_values)), episode_idx)
        writer.add_scalar("apf/potential_mean", float(np.mean(episode_potentials)), episode_idx)
        writer.add_scalar("apf/success", float(final_info.get("success", False)), episode_idx)
        writer.add_scalar("apf/collision", float(final_info.get("collision", False)), episode_idx)
        writer.add_scalar("apf/timeout", float(final_info.get("timeout", False)), episode_idx)

    warmup_env.close()
    writer.add_scalar("apf/use_kf", float(use_kf), episodes)
    writer.add_scalar("apf/total_steps", total_steps, episodes)
    writer.add_scalar("apf/replay_buffer_size", replay_buffer_size(model), episodes)
    writer.add_scalar("apf/success_rate", successes / max(1, episodes), episodes)
    writer.add_scalar("apf/collision_rate", collisions / max(1, episodes), episodes)
    writer.add_scalar("apf/timeout_rate", timeouts / max(1, episodes), episodes)
    writer.flush()

    print(
        "Complex APF warm-up finished: "
        f"episodes={episodes}, transitions={total_steps}, "
        f"use_kf={use_kf}, buffer_size={replay_buffer_size(model)}, "
        f"success_rate={successes / max(1, episodes):.3f}, "
        f"collision_rate={collisions / max(1, episodes):.3f}, "
        f"timeout_rate={timeouts / max(1, episodes):.3f}"
    )


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    model_path = (
        Path(args.model_path)
        if args.model_path
        else save_dir / "models" / "complex" / f"sac_{args.smoother_type}_vw_complex_nav_seed{args.seed}.zip"
    )
    log_dir = save_dir / "logs" / "complex" / f"{args.smoother_type}_seed{args.seed}"
    custom_tb_dir = log_dir / "custom"

    model_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    custom_tb_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    set_random_seed(args.seed)

    use_kf = bool(args.use_kf)
    smoother_kwargs = {
        "beta": args.smoother_beta,
        "tau": args.singer_tau,
        "velocity_process_noise_std": args.velocity_process_noise_std,
        "rate_process_noise_std": args.rate_process_noise_std,
        "measurement_noise_std": args.measurement_noise_std,
    }
    env = ComplexNavEnv(
        use_kf=use_kf,
        seed=args.seed,
        smoother_type=args.smoother_type,
        smoother_kwargs=smoother_kwargs,
        max_linear_speed=args.max_linear_speed,
        max_angular_speed=args.max_angular_speed,
    )
    env = Monitor(env, filename=str(log_dir / "train_complex"))
    writer = SummaryWriter(log_dir=str(custom_tb_dir))
    writer.add_scalar("curriculum/smoother_type_id", SMOOTHER_TYPE_IDS[args.smoother_type], 0)
    writer.add_scalar("curriculum/use_kf", float(use_kf), 0)
    writer.add_scalar("env/max_linear_speed", float(args.max_linear_speed), 0)
    writer.add_scalar("env/max_angular_speed", float(args.max_angular_speed), 0)

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
    )

    run_apf_warmup(
        model=model,
        episodes=args.apf_warmup_episodes,
        seed=args.seed,
        use_kf=use_kf,
        smoother_type=args.smoother_type,
        smoother_kwargs=smoother_kwargs,
        max_linear_speed=args.max_linear_speed,
        max_angular_speed=args.max_angular_speed,
        writer=writer,
        show_progress=bool(args.show_progress),
    )

    callbacks = [TensorboardMetricsCallback(writer=writer, log_freq=args.tb_log_freq)]
    if args.show_progress:
        callbacks.insert(0, TqdmTrainingCallback(total_timesteps=args.total_steps))

    print(
        f"Training complex SAC for {args.total_steps} steps, "
        f"use_kf={use_kf}, smoother_type={args.smoother_type}, "
        f"max_linear_speed={args.max_linear_speed}, "
        f"max_angular_speed={args.max_angular_speed}, "
        f"obs_dim={env.unwrapped.obs_dim if hasattr(env, 'unwrapped') else 'unknown'}, "
        f"learning_starts={learning_starts}"
    )
    try:
        model.learn(
            total_timesteps=args.total_steps,
            callback=CallbackList(callbacks),
            log_interval=10,
            tb_log_name="sac_kf_complex",
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
        f"  python -m evaluate.complex_env --model-path {model_path} "
        f"--smoother-type {args.smoother_type} --use-kf {args.use_kf}"
    )


if __name__ == "__main__":
    main()
