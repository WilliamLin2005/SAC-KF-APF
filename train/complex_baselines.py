"""Train additional SAC baselines on ComplexNavEnv."""

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

from envs.baseline_wrappers import get_complex_base_env, make_complex_baseline_env
from utils.apf_complex import ComplexAPFPolicy
from utils.training_callbacks import TensorboardMetricsCallback, TqdmTrainingCallback


TRAIN_BASELINES = ("action_delta_penalty", "lowpass_in_loop", "gsde", "kf_no_aug")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train complex-env baseline variants.")
    parser.add_argument("--baseline", type=str, required=True, choices=TRAIN_BASELINES)
    parser.add_argument("--total-steps", type=int, default=300_000)
    parser.add_argument("--apf-warmup-episodes", type=int, default=1_500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default="outputs")
    parser.add_argument("--tb-log-freq", type=int, default=100)
    parser.add_argument("--buffer-size", type=int, default=500_000)
    parser.add_argument("--show-progress", type=int, default=1, choices=[0, 1])
    parser.add_argument("--action-penalty-weight", type=float, default=0.2)
    parser.add_argument("--lowpass-alpha", type=float, default=0.35)
    parser.add_argument("--sde-sample-freq", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
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
    baseline: str,
    episodes: int,
    seed: int,
    writer: SummaryWriter,
    show_progress: bool,
    action_penalty_weight: float,
    lowpass_alpha: float,
) -> None:
    if episodes <= 0:
        return

    warmup_env = make_complex_baseline_env(
        baseline=baseline,
        seed=seed,
        action_penalty_weight=action_penalty_weight,
        lowpass_alpha=lowpass_alpha,
    )
    base_env = get_complex_base_env(warmup_env)
    apf_policy = ComplexAPFPolicy()
    iterator = range(episodes)
    if show_progress:
        iterator = tqdm(iterator, desc=f"{baseline} APF warm-up", unit="episode")

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
            apf_output = apf_policy.act(base_env)
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
    writer.add_scalar("apf/use_kf", float(baseline == "kf_no_aug"), episodes)
    writer.add_scalar("apf/total_steps", total_steps, episodes)
    writer.add_scalar("apf/replay_buffer_size", replay_buffer_size(model), episodes)
    writer.add_scalar("apf/success_rate", successes / max(1, episodes), episodes)
    writer.add_scalar("apf/collision_rate", collisions / max(1, episodes), episodes)
    writer.add_scalar("apf/timeout_rate", timeouts / max(1, episodes), episodes)
    writer.flush()

    print(
        "Baseline APF warm-up finished: "
        f"baseline={baseline}, episodes={episodes}, transitions={total_steps}, "
        f"buffer_size={replay_buffer_size(model)}, "
        f"success_rate={successes / max(1, episodes):.3f}, "
        f"collision_rate={collisions / max(1, episodes):.3f}, "
        f"timeout_rate={timeouts / max(1, episodes):.3f}"
    )


def default_model_path(save_dir: Path, baseline: str) -> Path:
    names = {
        "action_delta_penalty": "D_action_penalty.zip",
        "lowpass_in_loop": "E_lowpass_in_loop.zip",
        "gsde": "F_gsde.zip",
        "kf_no_aug": "G_kf_no_aug.zip",
    }
    return save_dir / "ablations" / "group1_complex" / names[baseline]


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    model_path = Path(args.model_path) if args.model_path else default_model_path(save_dir, args.baseline)
    log_dir = save_dir / "logs" / "complex_baselines" / args.baseline
    custom_tb_dir = log_dir / "custom"

    model_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    custom_tb_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    set_random_seed(args.seed)

    env = make_complex_baseline_env(
        baseline=args.baseline,
        seed=args.seed,
        action_penalty_weight=args.action_penalty_weight,
        lowpass_alpha=args.lowpass_alpha,
    )
    env = Monitor(env, filename=str(log_dir / "train"))
    writer = SummaryWriter(log_dir=str(custom_tb_dir))
    writer.add_scalar("baseline/use_kf", float(args.baseline == "kf_no_aug"), 0)
    writer.add_scalar("baseline/obs_dim", float(env.observation_space.shape[0]), 0)
    writer.add_scalar("baseline/action_penalty_weight", float(args.action_penalty_weight), 0)
    writer.add_scalar("baseline/lowpass_alpha", float(args.lowpass_alpha), 0)

    learning_starts = 0 if args.apf_warmup_episodes > 0 else 5_000
    sac_kwargs = {}
    if args.baseline == "gsde":
        sac_kwargs.update(
            {
                "use_sde": True,
                "sde_sample_freq": args.sde_sample_freq,
                "use_sde_at_warmup": True,
            }
        )

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
        **sac_kwargs,
    )

    run_apf_warmup(
        model=model,
        baseline=args.baseline,
        episodes=args.apf_warmup_episodes,
        seed=args.seed,
        writer=writer,
        show_progress=bool(args.show_progress),
        action_penalty_weight=args.action_penalty_weight,
        lowpass_alpha=args.lowpass_alpha,
    )

    callbacks = [TensorboardMetricsCallback(writer=writer, log_freq=args.tb_log_freq)]
    if args.show_progress:
        callbacks.insert(0, TqdmTrainingCallback(total_timesteps=args.total_steps))

    print(
        f"Training complex baseline {args.baseline} for {args.total_steps} steps, "
        f"obs_dim={env.observation_space.shape[0]}, learning_starts={learning_starts}"
    )
    try:
        model.learn(
            total_timesteps=args.total_steps,
            callback=CallbackList(callbacks),
            log_interval=10,
            tb_log_name=f"sac_{args.baseline}",
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
    print(f"  python -m evaluate.complex_baselines --baseline {args.baseline} --model-path {model_path}")


if __name__ == "__main__":
    main()
