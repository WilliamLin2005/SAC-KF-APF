"""Train fixed-KF v/w SAC with two-phase approach/docking reward."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.utils import set_random_seed
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from envs.robot_two_phase_reward_wrapper import make_robot_two_phase_env
from train.robot_diffdrive_complex import add_transition_to_replay_buffer, replay_buffer_size
from train.robot_diffdrive_kf_ablation import (
    KFAblationMetricsCallback,
    call_set_curriculum_progress,
)
from utils.apf_diffdrive import DiffDriveAPFPolicy
from utils.training_callbacks import TensorboardMetricsCallback, TqdmTrainingCallback


DEFAULT_RUN_NAME = "vw_apf_sac_fixed_kf_in_loop_kf_state_two_phase"
DEFAULT_MODEL_PATH = Path("outputs/vw/ablation") / f"{DEFAULT_RUN_NAME}.zip"
APF_SUCCESS_REPLAY_RATIO = 0.25
APF_SUCCESS_REPLAY_DECAY_FRACTION = 0.5
APF_SUCCESS_BUFFER_SIZE = 500_000


class ApfSuccessReplayBuffer(ReplayBuffer):
    """Mix ordinary replay with a decaying sub-batch from successful APF episodes."""

    def __init__(
        self,
        *args,
        apf_success_ratio_initial: float = APF_SUCCESS_REPLAY_RATIO,
        apf_success_decay_fraction: float = APF_SUCCESS_REPLAY_DECAY_FRACTION,
        total_timesteps: int = 1,
        apf_success_buffer_size: int = APF_SUCCESS_BUFFER_SIZE,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.apf_success_ratio_initial = float(apf_success_ratio_initial)
        self.apf_success_decay_fraction = max(float(apf_success_decay_fraction), 1.0e-6)
        self.total_timesteps = max(int(total_timesteps), 1)
        self.apf_success_ratio_current = max(self.apf_success_ratio_initial, 0.0)
        self.apf_success_buffer_size = max(int(apf_success_buffer_size), 1)
        self.apf_success_pos = 0
        self.apf_success_full = False
        self.apf_success_episode_count = 0
        self.apf_success_transition_count = 0
        self.last_sample_apf_success_count = 0
        self.last_sample_apf_success_ratio = 0.0

        self.apf_success_observations = np.zeros(
            (self.apf_success_buffer_size, *self.obs_shape),
            dtype=self.observation_space.dtype,
        )
        self.apf_success_next_observations = np.zeros(
            (self.apf_success_buffer_size, *self.obs_shape),
            dtype=self.observation_space.dtype,
        )
        self.apf_success_actions = np.zeros(
            (self.apf_success_buffer_size, self.action_dim),
            dtype=self._maybe_cast_dtype(self.action_space.dtype),
        )
        self.apf_success_rewards = np.zeros(self.apf_success_buffer_size, dtype=np.float32)
        self.apf_success_dones = np.zeros(self.apf_success_buffer_size, dtype=np.float32)
        self.apf_success_timeouts = np.zeros(self.apf_success_buffer_size, dtype=np.float32)

    @property
    def apf_success_size(self) -> int:
        return self.apf_success_buffer_size if self.apf_success_full else self.apf_success_pos

    def set_training_progress(self, completed_timesteps: int, total_timesteps: int | None = None) -> None:
        if total_timesteps is not None:
            self.total_timesteps = max(int(total_timesteps), 1)
        decay_steps = max(
            int(round(self.total_timesteps * self.apf_success_decay_fraction)),
            1,
        )
        progress = min(max(int(completed_timesteps), 0) / decay_steps, 1.0)
        self.apf_success_ratio_current = self.apf_success_ratio_initial * (1.0 - progress)

    def add_apf_success_episode(self, transitions: Sequence[tuple]) -> None:
        if not transitions:
            return
        for transition in transitions:
            self.add_apf_success_transition(*transition)
        self.apf_success_episode_count += 1

    def add_apf_success_transition(
        self,
        obs,
        next_obs,
        action,
        reward,
        done,
        info,
    ) -> None:
        self.apf_success_observations[self.apf_success_pos] = np.asarray(obs)
        self.apf_success_next_observations[self.apf_success_pos] = np.asarray(next_obs)
        self.apf_success_actions[self.apf_success_pos] = np.asarray(action).reshape(self.action_dim)
        self.apf_success_rewards[self.apf_success_pos] = float(reward)
        self.apf_success_dones[self.apf_success_pos] = float(done)
        self.apf_success_timeouts[self.apf_success_pos] = float(
            bool(info.get("TimeLimit.truncated", False)) if isinstance(info, dict) else False
        )
        self.apf_success_transition_count += 1
        self.apf_success_pos += 1
        if self.apf_success_pos >= self.apf_success_buffer_size:
            self.apf_success_full = True
            self.apf_success_pos = 0

    def _sample_apf_success(self, batch_size: int, env=None) -> ReplayBufferSamples:
        if self.apf_success_size <= 0:
            raise RuntimeError("Cannot sample APF success replay because it is empty.")
        batch_inds = np.random.randint(0, self.apf_success_size, size=batch_size)
        data = (
            self._normalize_obs(self.apf_success_observations[batch_inds], env),
            self.apf_success_actions[batch_inds],
            self._normalize_obs(self.apf_success_next_observations[batch_inds], env),
            (
                self.apf_success_dones[batch_inds]
                * (1.0 - self.apf_success_timeouts[batch_inds])
            ).reshape(-1, 1),
            self._normalize_reward(self.apf_success_rewards[batch_inds].reshape(-1, 1), env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

    @staticmethod
    def _concat_samples(left: ReplayBufferSamples, right: ReplayBufferSamples) -> ReplayBufferSamples:
        values = []
        for field in left._fields:
            values.append(torch.cat((getattr(left, field), getattr(right, field)), dim=0))
        perm = torch.randperm(values[0].shape[0], device=values[0].device)
        return ReplayBufferSamples(*[value[perm] for value in values])

    def sample(self, batch_size: int, env=None) -> ReplayBufferSamples:
        success_count = int(round(batch_size * max(self.apf_success_ratio_current, 0.0)))
        success_count = min(max(success_count, 0), batch_size)
        if success_count <= 0 or self.apf_success_size <= 0:
            self.last_sample_apf_success_count = 0
            self.last_sample_apf_success_ratio = 0.0
            return super().sample(batch_size=batch_size, env=env)

        normal_count = batch_size - success_count
        success_samples = self._sample_apf_success(success_count, env=env)
        if normal_count <= 0:
            self.last_sample_apf_success_count = success_count
            self.last_sample_apf_success_ratio = 1.0
            return success_samples

        normal_samples = super().sample(batch_size=normal_count, env=env)
        self.last_sample_apf_success_count = success_count
        self.last_sample_apf_success_ratio = success_count / float(batch_size)
        return self._concat_samples(normal_samples, success_samples)

    def apf_success_stats(self) -> dict[str, float]:
        return {
            "ratio_current": float(self.apf_success_ratio_current),
            "ratio_initial": float(self.apf_success_ratio_initial),
            "decay_fraction": float(self.apf_success_decay_fraction),
            "buffer_size": float(self.apf_success_buffer_size),
            "stored_transitions": float(self.apf_success_size),
            "total_transitions": float(self.apf_success_transition_count),
            "episodes": float(self.apf_success_episode_count),
            "last_sample_count": float(self.last_sample_apf_success_count),
            "last_sample_ratio": float(self.last_sample_apf_success_ratio),
        }


class ApfSuccessReplayCallback(BaseCallback):
    """Advance APF-success decay and expose replay metrics in TensorBoard."""

    def __init__(self, writer: SummaryWriter, total_timesteps: int, log_freq: int = 100) -> None:
        super().__init__()
        self.writer = writer
        self.total_timesteps = max(int(total_timesteps), 1)
        self.log_freq = max(int(log_freq), 1)

    def _on_training_start(self) -> None:
        self._set_progress(0)

    def _on_step(self) -> bool:
        self._set_progress(int(self.num_timesteps))
        if self.n_calls % self.log_freq == 0:
            self._write_stats(int(self.num_timesteps))
        return True

    def _on_training_end(self) -> None:
        self._write_stats(int(self.num_timesteps))
        self.writer.flush()

    def _set_progress(self, completed_timesteps: int) -> None:
        replay_buffer = getattr(self.model, "replay_buffer", None)
        if isinstance(replay_buffer, ApfSuccessReplayBuffer):
            replay_buffer.set_training_progress(completed_timesteps, self.total_timesteps)

    def _write_stats(self, step: int) -> None:
        replay_buffer = getattr(self.model, "replay_buffer", None)
        if not isinstance(replay_buffer, ApfSuccessReplayBuffer):
            return
        stats = replay_buffer.apf_success_stats()
        self.writer.add_scalar("apf_success/ratio_current", stats["ratio_current"], step)
        self.writer.add_scalar("apf_success/stored_transitions", stats["stored_transitions"], step)
        self.writer.add_scalar("apf_success/episodes", stats["episodes"], step)
        self.writer.add_scalar("apf_success/last_sample_ratio", stats["last_sample_ratio"], step)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fixed-KF v/w SAC with two-phase docking reward.")
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
        smoother="fixed",
        obs_mode="kf_state",
        kf_curriculum="none",
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


class TwoPhaseMetricsCallback(BaseCallback):
    """Write two-phase reward terms to TensorBoard."""

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
        scalar_keys = {
            "two_phase/reward_phase_id": "reward_phase_id",
            "two_phase/entered_docking_zone": "entered_docking_zone",
            "two_phase/inside_goal_too_fast": "inside_goal_too_fast",
            "two_phase/approach_reward": "approach_reward",
            "two_phase/docking_reward": "docking_reward",
            "two_phase/docking_entry_bonus": "docking_entry_bonus",
            "two_phase/docking_distance_penalty": "docking_distance_penalty",
            "two_phase/terminal_linear_speed_penalty": "terminal_linear_speed_penalty",
            "two_phase/terminal_angular_speed_penalty": "terminal_angular_speed_penalty",
            "two_phase/terminal_heading_penalty": "terminal_heading_penalty",
            "two_phase/inside_goal_fast_penalty": "inside_goal_fast_penalty",
            "two_phase/low_speed_success_reward": "low_speed_success_reward",
            "two_phase/terminal_v_exec": "terminal_v_exec",
            "two_phase/terminal_abs_w_exec": "terminal_abs_w_exec",
        }
        for tag, key in scalar_keys.items():
            self.writer.add_scalar(tag, float(info.get(key, 0.0)), step)
        return True

    def _on_training_end(self) -> None:
        self.writer.flush()


def docking_slow_apf_action(env, action: np.ndarray, slowdown_radius: float) -> np.ndarray:
    """Reduce APF command inside the docking zone so warm-up matches low-speed success."""

    base = env.unwrapped
    distance = float(np.linalg.norm(base.goal - base.position))
    if distance > slowdown_radius:
        return action.astype(np.float32)

    command = base.normalized_action_to_command(action)
    factor = float(
        np.clip(
            (distance - base.goal_radius) / max(slowdown_radius - base.goal_radius, 1e-8),
            0.0,
            1.0,
        )
    )
    command[0] = min(float(command[0]), base.max_linear_speed * factor)
    angular_limit = base.max_angular_speed * factor
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

    for episode_idx in iterator:
        apf_policy.reset()
        obs, _ = warmup_env.reset(seed=seed + 20_000 + episode_idx)
        terminated = False
        truncated = False
        episode_return = 0.0
        episode_steps = 0
        episode_transitions: list[tuple] = []
        final_info = {}

        while not (terminated or truncated):
            apf_output = apf_policy.act(warmup_env.unwrapped)
            action = docking_slow_apf_action(
                warmup_env,
                apf_output.action,
                slowdown_radius=args.slowdown_radius,
            )
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
            episode_transitions.append(
                (
                    np.array(obs, copy=True),
                    np.array(next_obs, copy=True),
                    np.array(action, copy=True),
                    float(reward),
                    bool(terminated or truncated),
                    dict(info),
                )
            )
            obs = next_obs
            episode_return += float(reward)
            episode_steps += 1
            total_steps += 1
            final_info = info
            docking_entries += int(bool(info.get("entered_docking_zone", False)))

        if bool(final_info.get("success", False)) and isinstance(
            model.replay_buffer,
            ApfSuccessReplayBuffer,
        ):
            model.replay_buffer.add_apf_success_episode(episode_transitions)
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

    warmup_env.close()
    writer.add_scalar("apf/total_steps", total_steps, episodes)
    writer.add_scalar("apf/replay_buffer_size", replay_buffer_size(model), episodes)
    writer.add_scalar("apf/success_rate", successes / max(1, episodes), episodes)
    writer.add_scalar("apf/collision_rate", collisions / max(1, episodes), episodes)
    writer.add_scalar("apf/out_of_bounds_rate", out_of_bounds / max(1, episodes), episodes)
    writer.add_scalar("apf/timeout_rate", timeouts / max(1, episodes), episodes)
    writer.add_scalar("apf/docking_entries", docking_entries, episodes)
    if isinstance(model.replay_buffer, ApfSuccessReplayBuffer):
        stats = model.replay_buffer.apf_success_stats()
        writer.add_scalar("apf_success/ratio_current", stats["ratio_current"], episodes)
        writer.add_scalar("apf_success/stored_transitions", stats["stored_transitions"], episodes)
        writer.add_scalar("apf_success/episodes", stats["episodes"], episodes)
        writer.add_scalar("apf_success/last_sample_ratio", stats["last_sample_ratio"], episodes)
    writer.flush()

    print(
        "Robot v/w two-phase APF warm-up finished: "
        f"episodes={episodes}, transitions={total_steps}, "
        f"buffer_size={replay_buffer_size(model)}, "
        f"apf_success={model.replay_buffer.apf_success_stats()}, "
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

    env = make_env(seed=args.seed, args=args)
    env = Monitor(env, filename=str(log_dir / "train"))
    writer = SummaryWriter(log_dir=str(custom_tb_dir))
    writer.add_text("robot/run_name", run_name, 0)
    writer.add_text("robot/model_path", str(model_path), 0)
    writer.add_text("robot/smoother", "fixed", 0)
    writer.add_text("robot/obs_mode", "kf_state", 0)
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
        replay_buffer_class=ApfSuccessReplayBuffer,
        replay_buffer_kwargs={
            "apf_success_ratio_initial": APF_SUCCESS_REPLAY_RATIO,
            "apf_success_decay_fraction": APF_SUCCESS_REPLAY_DECAY_FRACTION,
            "total_timesteps": args.total_steps,
            "apf_success_buffer_size": APF_SUCCESS_BUFFER_SIZE,
        },
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
            kf_curriculum="none",
            log_freq=args.tb_log_freq,
        ),
        TwoPhaseMetricsCallback(writer=writer, log_freq=args.tb_log_freq),
        ApfSuccessReplayCallback(
            writer=writer,
            total_timesteps=args.total_steps,
            log_freq=args.tb_log_freq,
        ),
    ]
    if args.show_progress:
        callbacks.insert(0, TqdmTrainingCallback(total_timesteps=args.total_steps))

    print(
        f"Training fixed-KF two-phase v/w SAC for {args.total_steps} steps, "
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
        "  python -m evaluate.robot_diffdrive_kf_two_phase "
        f"--model-path {model_path}"
    )


if __name__ == "__main__":
    main()
