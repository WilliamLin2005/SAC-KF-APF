"""Train SAC with decoupled command smoothing wrappers for v/w ablations."""

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

from envs.robot_command_smoothing_wrappers import (
    CURRICULUM_CHOICES,
    OBS_MODE_CHOICES,
    SMOOTHER_CHOICES,
    make_robot_command_smoothing_env,
)
from train.robot_diffdrive_complex import add_transition_to_replay_buffer, replay_buffer_size
from utils.apf_diffdrive import DiffDriveAPFPolicy
from utils.training_callbacks import TensorboardMetricsCallback, TqdmTrainingCallback


ABLATION_DIR = Path("vw") / "ablation"
DEFAULT_RUN_NAME = "vw_apf_sac_adaptive_kf_in_loop_kf_state"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train v/w SAC with decoupled KF ablation wrappers.")
    parser.add_argument("--smoother", type=str, default="adaptive", choices=SMOOTHER_CHOICES)
    parser.add_argument("--obs-mode", type=str, default="kf_state", choices=OBS_MODE_CHOICES)
    parser.add_argument("--kf-curriculum", type=str, default="continuous", choices=CURRICULUM_CHOICES)
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
    return parser.parse_args()


def default_run_name(smoother: str, obs_mode: str) -> str:
    if smoother == "none":
        return f"vw_apf_sac_no_smoothing_{obs_mode}"
    return f"vw_apf_sac_{smoother}_kf_in_loop_{obs_mode}"


def default_model_path(save_dir: Path, smoother: str, obs_mode: str) -> Path:
    return save_dir / ABLATION_DIR / f"{default_run_name(smoother, obs_mode)}.zip"


def make_env(
    smoother: str,
    obs_mode: str,
    kf_curriculum: str,
    seed: int | None,
):
    return make_robot_command_smoothing_env(
        smoother=smoother,
        obs_mode=obs_mode,
        kf_curriculum=kf_curriculum,
        seed=seed,
    )


def call_set_curriculum_progress(env, progress: float) -> bool:
    current = env
    for _ in range(12):
        if "set_curriculum_progress" in type(current).__dict__:
            method = getattr(current, "set_curriculum_progress")
            method(progress)
            return True
        current = getattr(current, "env", None)
        if current is None:
            break
    return False


class KFAblationMetricsCallback(BaseCallback):
    """Advance KF curriculum and log wrapper-specific metrics."""

    def __init__(
        self,
        writer: SummaryWriter,
        total_timesteps: int,
        kf_curriculum: str,
        log_freq: int = 100,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.writer = writer
        self.total_timesteps = max(1, int(total_timesteps))
        self.kf_curriculum = kf_curriculum
        self.log_freq = max(1, int(log_freq))

    def _on_training_start(self) -> None:
        self._set_progress(0.0 if self.kf_curriculum == "continuous" else 1.0)

    def _on_step(self) -> bool:
        progress = min(1.0, float(self.num_timesteps) / float(self.total_timesteps))
        if self.kf_curriculum != "continuous":
            progress = 1.0
        self._set_progress(progress)

        if self.n_calls % self.log_freq != 0:
            return True

        infos = self.locals.get("infos", [])
        if not infos:
            return True
        info = infos[0]
        step = int(self.num_timesteps)
        self.writer.add_scalar("kf/filter_mismatch", float(info.get("filter_mismatch_norm", 0.0)), step)
        self.writer.add_scalar("kf/process_noise_v", float(info.get("kf_process_noise_v", 0.0)), step)
        self.writer.add_scalar("kf/process_noise_w", float(info.get("kf_process_noise_w", 0.0)), step)
        self.writer.add_scalar("kf/measurement_noise_v", float(info.get("kf_measurement_noise_v", 0.0)), step)
        self.writer.add_scalar("kf/measurement_noise_w", float(info.get("kf_measurement_noise_w", 0.0)), step)
        self.writer.add_scalar("kf/covariance_v", float(info.get("kf_covariance_v", 0.0)), step)
        self.writer.add_scalar("kf/covariance_w", float(info.get("kf_covariance_w", 0.0)), step)
        self.writer.add_scalar("kf/curriculum_progress", float(info.get("kf_curriculum_progress", progress)), step)
        self.writer.add_scalar("kf/responsive_factor", float(info.get("kf_responsive_factor", 0.0)), step)
        self.writer.add_scalar("kf/jitter_factor", float(info.get("kf_jitter_factor", 0.0)), step)
        return True

    def _on_training_end(self) -> None:
        self.writer.flush()

    def _set_progress(self, progress: float) -> None:
        envs = getattr(self.training_env, "envs", [])
        for env in envs:
            call_set_curriculum_progress(env, progress)


def run_apf_warmup(
    model: SAC,
    episodes: int,
    seed: int,
    smoother: str,
    obs_mode: str,
    kf_curriculum: str,
    writer: SummaryWriter,
    show_progress: bool,
) -> None:
    if episodes <= 0:
        return

    warmup_env = make_env(
        smoother=smoother,
        obs_mode=obs_mode,
        kf_curriculum=kf_curriculum,
        seed=seed,
    )
    call_set_curriculum_progress(warmup_env, 0.0 if kf_curriculum == "continuous" else 1.0)
    apf_policy = DiffDriveAPFPolicy()
    iterator = range(episodes)
    if show_progress:
        iterator = tqdm(iterator, desc="Robot v/w APF warm-up", unit="episode")

    total_steps = 0
    successes = 0
    collisions = 0
    out_of_bounds = 0
    timeouts = 0
    mismatch_values: list[float] = []

    for episode_idx in iterator:
        apf_policy.reset()
        obs, _ = warmup_env.reset(seed=seed + 20_000 + episode_idx)
        call_set_curriculum_progress(warmup_env, 0.0 if kf_curriculum == "continuous" else 1.0)
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
    writer.add_scalar(
        "apf/filter_mismatch_mean",
        float(np.mean(mismatch_values)) if mismatch_values else 0.0,
        episodes,
    )
    writer.flush()

    print(
        "Robot v/w APF warm-up finished: "
        f"episodes={episodes}, transitions={total_steps}, "
        f"smoother={smoother}, obs_mode={obs_mode}, "
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
    name = args.run_name or model_path.stem or DEFAULT_RUN_NAME
    log_dir = save_dir / "logs" / "robot_diffdrive_complex" / name
    custom_tb_dir = log_dir / "custom"

    model_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    custom_tb_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    set_random_seed(args.seed)

    env = make_env(
        smoother=args.smoother,
        obs_mode=args.obs_mode,
        kf_curriculum=args.kf_curriculum,
        seed=args.seed,
    )
    env = Monitor(env, filename=str(log_dir / "train"))
    writer = SummaryWriter(log_dir=str(custom_tb_dir))
    writer.add_text("robot/run_name", name, 0)
    writer.add_text("robot/model_path", str(model_path), 0)
    writer.add_text("robot/smoother", args.smoother, 0)
    writer.add_text("robot/obs_mode", args.obs_mode, 0)
    writer.add_text("robot/kf_curriculum", args.kf_curriculum, 0)
    writer.add_scalar("robot/obs_dim", float(env.observation_space.shape[0]), 0)
    writer.add_scalar("robot/apf_warmup_episodes", float(args.apf_warmup_episodes), 0)
    writer.add_scalar("robot/buffer_size", float(args.buffer_size), 0)

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
        smoother=args.smoother,
        obs_mode=args.obs_mode,
        kf_curriculum=args.kf_curriculum,
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
    ]
    if args.show_progress:
        callbacks.insert(0, TqdmTrainingCallback(total_timesteps=args.total_steps))

    print(
        f"Training robot v/w SAC for {args.total_steps} steps, "
        f"smoother={args.smoother}, obs_mode={args.obs_mode}, "
        f"kf_curriculum={args.kf_curriculum}, "
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
        "  python -m evaluate.robot_diffdrive_kf_ablation "
        f"--model-path {model_path} --smoother {args.smoother} "
        f"--obs-mode {args.obs_mode} --kf-curriculum {args.kf_curriculum}"
    )


if __name__ == "__main__":
    main()
