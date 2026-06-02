"""Training callbacks for tqdm progress and custom TensorBoard metrics."""

from __future__ import annotations

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm


class TqdmTrainingCallback(BaseCallback):
    """Progress bar for SB3 timesteps."""

    def __init__(self, total_timesteps: int, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.total_timesteps = int(total_timesteps)
        self._bar: tqdm | None = None
        self._last_step = 0

    def _on_training_start(self) -> None:
        self._bar = tqdm(total=self.total_timesteps, desc="SAC training", unit="step")
        self._last_step = 0

    def _on_step(self) -> bool:
        if self._bar is not None:
            current = min(int(self.num_timesteps), self.total_timesteps)
            delta = max(0, current - self._last_step)
            self._bar.update(delta)
            self._last_step = current
        return True

    def _on_training_end(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None


class TensorboardMetricsCallback(BaseCallback):
    """Write env info metrics to a SummaryWriter."""

    def __init__(
        self,
        writer: SummaryWriter,
        log_freq: int = 100,
        verbose: int = 0,
    ) -> None:
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
        raw_delta = float(info.get("raw_action_delta_norm", 0.0))
        exec_delta = float(info.get("exec_action_delta_norm", 0.0))
        smoothing_ratio = exec_delta / (raw_delta + 1e-8)

        step = int(self.num_timesteps)
        self.writer.add_scalar("env/distance_to_goal", float(info.get("distance_to_goal", 0.0)), step)
        self.writer.add_scalar("env/success", float(info.get("success", False)), step)
        self.writer.add_scalar("env/collision", float(info.get("collision", False)), step)
        self.writer.add_scalar("env/out_of_bounds", float(info.get("out_of_bounds", False)), step)
        self.writer.add_scalar("env/timeout", float(info.get("timeout", False)), step)
        self.writer.add_scalar("env/goal_progress", float(info.get("goal_progress", 0.0)), step)
        self.writer.add_scalar("action/raw_norm", float(info.get("raw_action_norm", 0.0)), step)
        self.writer.add_scalar("action/exec_norm", float(info.get("exec_action_norm", 0.0)), step)
        self.writer.add_scalar("action/raw_delta", raw_delta, step)
        self.writer.add_scalar("action/exec_delta", exec_delta, step)
        self.writer.add_scalar("action/smoothing_ratio", smoothing_ratio, step)
        raw_action = info.get("raw_action", None)
        exec_action = info.get("executed_action", None)
        if raw_action is not None:
            raw_action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
            if raw_action.shape[0] >= 2:
                self.writer.add_scalar("action/raw_channel0", float(raw_action[0]), step)
                self.writer.add_scalar("action/raw_channel1", float(raw_action[1]), step)
        if exec_action is not None:
            exec_action = np.asarray(exec_action, dtype=np.float32).reshape(-1)
            if exec_action.shape[0] >= 2:
                self.writer.add_scalar("action/exec_channel0", float(exec_action[0]), step)
                self.writer.add_scalar("action/exec_channel1", float(exec_action[1]), step)
        if raw_action is not None and "v_norm" in info:
            self.writer.add_scalar("action/raw_v_norm", float(raw_action[0]), step)
            self.writer.add_scalar("action/exec_v_norm", float(info.get("v_norm", 0.0)), step)
        if raw_action is not None and "omega_norm" in info:
            self.writer.add_scalar("action/raw_omega_norm", float(raw_action[1]), step)
            self.writer.add_scalar("action/exec_omega_norm", float(info.get("omega_norm", 0.0)), step)
        if "heading" in info:
            self.writer.add_scalar("env/heading", float(info.get("heading", 0.0)), step)
        if "linear_velocity" in info:
            self.writer.add_scalar("env/linear_velocity", float(info.get("linear_velocity", 0.0)), step)
        if "angular_velocity" in info:
            self.writer.add_scalar("env/angular_velocity", float(info.get("angular_velocity", 0.0)), step)
        smoother_features = info.get("smoother_features", None)
        if smoother_features is not None:
            smoother_features = np.asarray(smoother_features, dtype=np.float32).reshape(-1)
            for idx, value in enumerate(smoother_features):
                self.writer.add_scalar(f"smoother/feature_{idx}", float(value), step)
        self.writer.add_scalar("reward/progress_reward", float(info.get("progress_reward", 0.0)), step)
        self.writer.add_scalar("reward/heading_reward", float(info.get("heading_reward", 0.0)), step)
        self.writer.add_scalar("reward/obstacle_penalty", float(info.get("obstacle_penalty", 0.0)), step)
        self.writer.add_scalar("reward/distance_penalty", float(info.get("distance_penalty", 0.0)), step)
        self.writer.add_scalar("reward/stall_penalty", float(info.get("stall_penalty", 0.0)), step)
        self.writer.add_scalar(
            "reward/timeout_distance_penalty",
            float(info.get("timeout_distance_penalty", 0.0)),
            step,
        )
        self.writer.add_scalar(
            "env/min_obstacle_signed_distance",
            float(info.get("min_obstacle_signed_distance", 0.0)),
            step,
        )
        self.writer.add_scalar("curriculum/use_kf", float(info.get("use_kf", False)), step)

        dones = self.locals.get("dones")
        done_now = bool(np.asarray(dones).any()) if dones is not None else False
        if done_now:
            self.writer.add_scalar("env/episode_success", float(info.get("success", False)), step)
            self.writer.add_scalar("env/episode_collision", float(info.get("collision", False)), step)
            self.writer.add_scalar("env/episode_out_of_bounds", float(info.get("out_of_bounds", False)), step)
            self.writer.add_scalar("env/episode_timeout", float(info.get("timeout", False)), step)
            self.writer.add_scalar("env/episode_final_distance", float(info.get("distance_to_goal", 0.0)), step)
        return True

    def _on_training_end(self) -> None:
        self.writer.flush()
