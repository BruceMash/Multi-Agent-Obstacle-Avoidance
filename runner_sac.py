"""
SAC training entry for the single-agent DMP-RL environment.
"""

import csv
from collections import deque
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from Environment.single_agent_dmp_env import SingleAgentDMPEnv
from baseline.common.callbacks import BaseCallback, CallbackList
from baseline.sac import SAC
from experiment_config import EXPERIMENT_CONFIG, SACExperimentConfig


class _PolicyModeShim:
    """Compatibility shim for the legacy training loop."""

    def set_training_mode(self, mode: bool) -> None:
        _ = mode
        return None


def build_env(
    config: SACExperimentConfig = EXPERIMENT_CONFIG,
    action_guidance_enabled: bool | None = False,
    curriculum_state: dict[str, Any] | None = None,
) -> SingleAgentDMPEnv:
    """Build the training environment from a centralized config."""
    if curriculum_state is None and bool(config.training_scene_mixture_enabled):
        curriculum_state = config.build_curriculum_state()
    dynamics_config = config.build_dynamics_config()
    sensor_config = config.build_sensor_config()
    dmp_config = config.build_dmp_config()
    env_config = config.build_env_config()
    if action_guidance_enabled is not None:
        env_config.action_guidance_enabled = bool(action_guidance_enabled)
    fixed_box = config.build_fixed_box()
    start_goal_generator = config.build_start_goal_generator()
    static_obstacle_generator = config.build_static_obstacle_generator(fixed_box, curriculum_state=curriculum_state)
    dynamic_obstacle_generator = config.build_dynamic_obstacle_generator(curriculum_state=curriculum_state)
    static_obstacles = config.build_static_obstacles(fixed_box)

    env = SingleAgentDMPEnv(
        dynamics_config=dynamics_config,
        sensor_config=sensor_config,
        dmp_config=dmp_config,
        env_config=env_config,
        start_goal_generator=start_goal_generator,
        static_obstacles=static_obstacles,
        static_obstacle_generator=static_obstacle_generator,
        dynamic_obstacles=[],
        dynamic_obstacle_generator=dynamic_obstacle_generator,
    )
    env._default_start = np.asarray(config.default_start, dtype=float)
    env._default_goal = np.asarray(config.default_goal, dtype=float)
    env.goal = env._default_goal.copy()
    env.curriculum_state = curriculum_state
    return env


def build_model(
    env: SingleAgentDMPEnv,
    config: SACExperimentConfig = EXPERIMENT_CONFIG,
    tensorboard_log: str | None = None,
    verbose: int | None = None,
) -> SAC:
    """Build SAC using the centralized experiment config."""
    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=config.learning_rate,
        buffer_size=config.buffer_size,
        batch_size=config.batch_size,
        learning_starts=config.learning_starts,
        train_freq=config.train_freq,
        gradient_steps=config.gradient_steps,
        ent_coef=config.ent_coef,
        verbose=config.verbose if verbose is None else verbose,
        tensorboard_log=tensorboard_log,
        policy_kwargs=config.build_policy_kwargs(),
    )

    if model.policy is None:
        model.policy = _PolicyModeShim()
    return model


def save_checkpoint(model: SAC, model_path: str | Path, extra: dict[str, Any] | None = None) -> str:
    """Save a full training checkpoint."""
    save_path = Path(model_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint: dict[str, Any] = {
        "actor": model.actor.state_dict(),
        "critic": model.critic.state_dict(),
        "critic_target": model.critic_target.state_dict(),
        "actor_optimizer": model.actor.optimizer.state_dict(),
        "critic_optimizer": model.critic.optimizer.state_dict(),
        "num_timesteps": int(model.num_timesteps),
        "n_updates": int(getattr(model, "_n_updates", 0)),
    }

    if model.ent_coef_optimizer is not None and model.log_ent_coef is not None:
        checkpoint["log_ent_coef"] = model.log_ent_coef.detach().cpu()
        checkpoint["ent_coef_optimizer"] = model.ent_coef_optimizer.state_dict()
    elif hasattr(model, "ent_coef_tensor"):
        checkpoint["ent_coef_tensor"] = model.ent_coef_tensor.detach().cpu()

    if extra:
        checkpoint.update(extra)

    torch.save(checkpoint, str(save_path))
    return str(save_path)


def load_checkpoint(model: SAC, model_path: str | Path) -> SAC:
    """Restore model and optimizer states from a checkpoint."""
    checkpoint = torch.load(model_path, map_location=model.device)

    model.actor.load_state_dict(checkpoint["actor"])
    model.critic.load_state_dict(checkpoint["critic"])
    model.critic_target.load_state_dict(checkpoint["critic_target"])
    model.actor.optimizer.load_state_dict(checkpoint["actor_optimizer"])
    model.critic.optimizer.load_state_dict(checkpoint["critic_optimizer"])

    if model.ent_coef_optimizer is not None and "ent_coef_optimizer" in checkpoint and "log_ent_coef" in checkpoint:
        model.log_ent_coef = checkpoint["log_ent_coef"].to(model.device).requires_grad_(True)
        model.ent_coef_optimizer = torch.optim.Adam([model.log_ent_coef], lr=model.lr_schedule(1))
        model.ent_coef_optimizer.load_state_dict(checkpoint["ent_coef_optimizer"])
    elif "ent_coef_tensor" in checkpoint:
        model.ent_coef_tensor = checkpoint["ent_coef_tensor"].to(model.device)

    if "num_timesteps" in checkpoint:
        model.num_timesteps = int(checkpoint["num_timesteps"])
    if "n_updates" in checkpoint:
        model._n_updates = int(checkpoint["n_updates"])
    return model


class CheckpointAndBestCallback(BaseCallback):
    """Periodically save checkpoints and keep the best episode reward."""

    def __init__(self, run_dir: Path, save_freq: int = 500, verbose: int = 1):
        super().__init__(verbose=verbose)
        self.run_dir = run_dir
        self.ckpt_dir = run_dir / "checkpoints"
        self.save_freq = max(1, int(save_freq))
        self.best_model_path = run_dir / "best_model.pt"
        self.latest_checkpoint_path = run_dir / "checkpoint.pt"
        self.best_episode_reward = float("-inf")

    def _init_callback(self) -> None:
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _save_latest_checkpoint(self) -> None:
        save_checkpoint(
            model=self.model,
            model_path=self.latest_checkpoint_path,
            extra={"best_episode_reward": self.best_episode_reward},
        )

    def _save_step_checkpoint(self) -> None:
        step_ckpt = self.ckpt_dir / f"checkpoint_{self.num_timesteps:08d}.pt"
        save_checkpoint(
            model=self.model,
            model_path=step_ckpt,
            extra={"best_episode_reward": self.best_episode_reward},
        )

    def _maybe_save_best(self, infos: list[dict[str, Any]]) -> None:
        for info in infos:
            episode_info = info.get("episode")
            if episode_info is None:
                continue
            episode_reward = float(episode_info["r"])
            if episode_reward > self.best_episode_reward:
                self.best_episode_reward = episode_reward
                save_checkpoint(
                    model=self.model,
                    model_path=self.best_model_path,
                    extra={"best_episode_reward": self.best_episode_reward},
                )
                if self.verbose > 0:
                    print(
                        f"[best_model] timesteps={self.num_timesteps}, "
                        f"episode_reward={self.best_episode_reward:.3f}"
                    )

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            self._save_step_checkpoint()
            self._save_latest_checkpoint()

        infos = self.locals.get("infos", [])
        if isinstance(infos, list):
            self._maybe_save_best(infos)
        return True

    def _on_training_end(self) -> None:
        self._save_latest_checkpoint()


class TensorboardRewardCallback(BaseCallback):
    """Write reward-related metrics to TensorBoard."""

    def __init__(self, log_dir: Path, flush_freq: int = 100):
        super().__init__(verbose=0)
        self.log_dir = log_dir
        self.flush_freq = max(1, int(flush_freq))
        self.writer: SummaryWriter | None = None

    def _init_callback(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir), flush_secs=5)

    def _on_step(self) -> bool:
        if self.writer is None:
            return True

        rewards = self.locals.get("rewards")
        if rewards is not None:
            reward_value = float(rewards.mean()) if hasattr(rewards, "mean") else float(rewards)
            self.writer.add_scalar("step/reward_total", reward_value, self.num_timesteps)

        infos = self.locals.get("infos", [])
        if isinstance(infos, list) and infos:
            step_reward_values: list[float] = []
            obstacle_penalty_values: list[float] = []
            boundary_penalty_values: list[float] = []
            step_penalty_values: list[float] = []
            timeout_penalty_values: list[float] = []
            distance_values: list[float] = []
            boundary_distance_values: list[float] = []
            success_values: list[float] = []
            collision_values: list[float] = []
            timeout_values: list[float] = []
            action_guidance_values: list[float] = []
            episode_reward_values: list[float] = []

            for info in infos:
                if "reward_step_reward" in info:
                    step_reward_values.append(float(info["reward_step_reward"]))
                if "reward_obstacle_potential_penalty" in info:
                    obstacle_penalty_values.append(float(info["reward_obstacle_potential_penalty"]))
                if "reward_boundary_potential_penalty" in info:
                    boundary_penalty_values.append(float(info["reward_boundary_potential_penalty"]))
                if "reward_step_penalty" in info:
                    step_penalty_values.append(float(info["reward_step_penalty"]))
                if "reward_timeout_penalty" in info:
                    timeout_penalty_values.append(float(info["reward_timeout_penalty"]))
                if "distance_to_goal" in info:
                    distance_values.append(float(info["distance_to_goal"]))
                if "min_boundary_distance" in info:
                    boundary_distance_values.append(float(info["min_boundary_distance"]))
                if "success" in info:
                    success_values.append(float(info["success"]))
                if "collision" in info:
                    collision_values.append(float(info["collision"]))
                if "truncated" in info:
                    timeout_values.append(float(info["truncated"]))
                if "action_guidance_weight" in info:
                    action_guidance_values.append(float(info["action_guidance_weight"]))
                episode_info = info.get("episode")
                if episode_info is not None and "r" in episode_info:
                    episode_reward_values.append(float(episode_info["r"]))
                    if "success" in info:
                        self.writer.add_scalar("episode/success", float(info["success"]), self.num_timesteps)
                    if "collision" in info:
                        self.writer.add_scalar("episode/collision", float(info["collision"]), self.num_timesteps)
                    if "truncated" in info:
                        self.writer.add_scalar("episode/timeout", float(info["truncated"]), self.num_timesteps)
                    if "distance_to_goal" in info:
                        self.writer.add_scalar("episode/distance_to_goal", float(info["distance_to_goal"]), self.num_timesteps)

            if step_reward_values:
                self.writer.add_scalar(
                    "step/reward_step_reward",
                    sum(step_reward_values) / len(step_reward_values),
                    self.num_timesteps,
                )
            if obstacle_penalty_values:
                self.writer.add_scalar(
                    "step/reward_obstacle_potential_penalty",
                    sum(obstacle_penalty_values) / len(obstacle_penalty_values),
                    self.num_timesteps,
                )
            if boundary_penalty_values:
                self.writer.add_scalar(
                    "step/reward_boundary_potential_penalty",
                    sum(boundary_penalty_values) / len(boundary_penalty_values),
                    self.num_timesteps,
                )
            if step_penalty_values:
                self.writer.add_scalar(
                    "step/reward_step_penalty",
                    sum(step_penalty_values) / len(step_penalty_values),
                    self.num_timesteps,
                )
            if timeout_penalty_values:
                self.writer.add_scalar(
                    "step/reward_timeout_penalty",
                    sum(timeout_penalty_values) / len(timeout_penalty_values),
                    self.num_timesteps,
                )
            if distance_values:
                self.writer.add_scalar(
                    "step/distance_to_goal",
                    sum(distance_values) / len(distance_values),
                    self.num_timesteps,
                )
            if boundary_distance_values:
                self.writer.add_scalar(
                    "step/min_boundary_distance",
                    sum(boundary_distance_values) / len(boundary_distance_values),
                    self.num_timesteps,
                )
            if success_values:
                self.writer.add_scalar("step/success_rate", sum(success_values) / len(success_values), self.num_timesteps)
            if collision_values:
                self.writer.add_scalar("step/collision_rate", sum(collision_values) / len(collision_values), self.num_timesteps)
            if timeout_values:
                self.writer.add_scalar("step/timeout_rate", sum(timeout_values) / len(timeout_values), self.num_timesteps)
            if action_guidance_values:
                self.writer.add_scalar(
                    "step/action_guidance_weight",
                    sum(action_guidance_values) / len(action_guidance_values),
                    self.num_timesteps,
                )
            for episode_reward in episode_reward_values:
                self.writer.add_scalar("episode/reward", episode_reward, self.num_timesteps)

        if self.num_timesteps > 0 and self.num_timesteps % self.flush_freq == 0:
            self.writer.flush()
        return True

    def _on_training_end(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()


class SuccessRateCurriculumCallback(BaseCallback):
    """Adapt training scene difficulty from recent episode success rate."""

    def __init__(
        self,
        config: SACExperimentConfig,
        curriculum_state: dict[str, Any] | None,
        log_dir: Path,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.config = config
        self.curriculum_state = curriculum_state
        self.log_dir = log_dir
        self.writer: SummaryWriter | None = None
        self.success_window: deque[float] = deque(maxlen=max(1, int(config.curriculum_window_episodes)))
        self.episode_count = 0
        self.level_episode_count = 0
        self.episodes_since_check = 0
        self.last_logged_timestep = -1

    def _init_callback(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir), flush_secs=5)
        if self.curriculum_state is not None:
            self._apply_level(int(self.curriculum_state.get("level", 0)))
            self._write_tensorboard(success_rate=0.0)

    def _num_levels(self) -> int:
        dense_levels = tuple(float(value) for value in self.config.curriculum_dense_scene_probabilities)
        dynamic_levels = tuple(float(value) for value in self.config.curriculum_dense_dynamic_probabilities)
        if len(dense_levels) != len(dynamic_levels):
            raise ValueError("curriculum dense and dynamic probability schedules must have equal length")
        if len(dense_levels) < 1:
            raise ValueError("curriculum schedule must contain at least one level")
        return len(dense_levels)

    def _apply_level(self, level: int) -> None:
        if self.curriculum_state is None:
            return
        level = int(np.clip(level, 0, self._num_levels() - 1))
        dense_levels = tuple(float(value) for value in self.config.curriculum_dense_scene_probabilities)
        dynamic_levels = tuple(float(value) for value in self.config.curriculum_dense_dynamic_probabilities)
        self.curriculum_state["enabled"] = bool(self.config.curriculum_enabled)
        self.curriculum_state["level"] = level
        self.curriculum_state["dense_scene_probability"] = dense_levels[level]
        self.curriculum_state["dense_dynamic_probability"] = dynamic_levels[level]
        self.curriculum_state["level_episode_count"] = self.level_episode_count

    def _write_tensorboard(self, success_rate: float) -> None:
        if self.writer is None or self.curriculum_state is None:
            return
        self.writer.add_scalar("curriculum/level", int(self.curriculum_state.get("level", 0)), self.num_timesteps)
        self.writer.add_scalar(
            "curriculum/dense_scene_probability",
            float(self.curriculum_state.get("dense_scene_probability", 0.0)),
            self.num_timesteps,
        )
        self.writer.add_scalar(
            "curriculum/dense_dynamic_probability",
            float(self.curriculum_state.get("dense_dynamic_probability", 0.0)),
            self.num_timesteps,
        )
        self.writer.add_scalar("curriculum/recent_success_rate", float(success_rate), self.num_timesteps)
        self.writer.flush()

    def _maybe_update_level(self) -> None:
        if self.curriculum_state is None or not bool(self.config.curriculum_enabled):
            return
        if len(self.success_window) < int(self.config.curriculum_window_episodes):
            return
        if self.episodes_since_check < int(self.config.curriculum_check_interval_episodes):
            return
        if self.level_episode_count < int(self.config.curriculum_min_level_episodes):
            return

        self.episodes_since_check = 0
        success_rate = float(np.mean(self.success_window))
        level = int(self.curriculum_state.get("level", 0))
        next_level = level
        advance_thresholds = tuple(float(value) for value in self.config.curriculum_advance_success_thresholds)
        if len(advance_thresholds) != max(0, self._num_levels() - 1):
            raise ValueError("curriculum_advance_success_thresholds must have one value per level transition")

        if level < self._num_levels() - 1 and success_rate >= advance_thresholds[level]:
            next_level = level + 1
        elif level > 0 and success_rate < float(self.config.curriculum_rollback_success_threshold):
            next_level = level - 1

        self.curriculum_state["episode_count"] = self.episode_count
        self.curriculum_state["success_rate"] = success_rate
        if next_level != level:
            self.level_episode_count = 0
            self.success_window.clear()
            self._apply_level(next_level)
            if self.verbose > 0:
                print(
                    f"[curriculum] timesteps={self.num_timesteps}, "
                    f"level={next_level}, recent_success={success_rate:.3f}, "
                    f"dense_prob={float(self.curriculum_state['dense_scene_probability']):.2f}, "
                    f"dense_dynamic_prob={float(self.curriculum_state['dense_dynamic_probability']):.2f}"
                )
        self._write_tensorboard(success_rate)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if isinstance(infos, list):
            for info in infos:
                if info.get("episode") is None:
                    continue
                self.success_window.append(float(bool(info.get("success", False))))
                self.episode_count += 1
                self.level_episode_count += 1
                self.episodes_since_check += 1
                self._maybe_update_level()

        if (
            self.writer is not None
            and self.curriculum_state is not None
            and self.num_timesteps > 0
            and self.num_timesteps % 1000 == 0
            and self.last_logged_timestep != self.num_timesteps
        ):
            success_rate = float(np.mean(self.success_window)) if self.success_window else 0.0
            self.curriculum_state["episode_count"] = self.episode_count
            self.curriculum_state["level_episode_count"] = self.level_episode_count
            self.curriculum_state["success_rate"] = success_rate
            self._write_tensorboard(success_rate)
            self.last_logged_timestep = self.num_timesteps
        return True

    def _on_training_end(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()


class FixedSeedEvalCallback(BaseCallback):
    """Evaluate the policy on a fixed set of seeded benchmark scenarios."""

    CSV_FIELDS = [
        "timesteps",
        "seed",
        "reward",
        "length",
        "success",
        "collision",
        "timeout",
        "distance_to_goal",
    ]

    def __init__(
        self,
        eval_env: SingleAgentDMPEnv,
        run_dir: Path,
        log_dir: Path,
        eval_freq: int,
        eval_seeds: tuple[int, ...],
        deterministic: bool = True,
        save_visualizations: bool = True,
        visualization_dir: str | Path | None = None,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.run_dir = run_dir
        self.eval_dir = run_dir / "eval"
        self.log_dir = log_dir
        self.visualization_dir = Path(visualization_dir) if visualization_dir is not None else run_dir / "visualizations"
        self.eval_freq = max(1, int(eval_freq))
        self.eval_seeds = tuple(int(seed) for seed in eval_seeds)
        if not self.eval_seeds:
            raise ValueError("eval_seeds must contain at least one seed")
        self.deterministic = bool(deterministic)
        self.save_visualizations = bool(save_visualizations)
        self.csv_path = self.eval_dir / "fixed_seed_eval.csv"
        self.best_eval_model_path = run_dir / "best_eval_model.pt"
        self.best_mean_reward = float("-inf")
        self.writer: SummaryWriter | None = None

    def _init_callback(self) -> None:
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if self.save_visualizations:
            self.visualization_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir), flush_secs=5)
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            with self.csv_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=self.CSV_FIELDS)
                writer.writeheader()

    @staticmethod
    def _extract_obstacles(env: SingleAgentDMPEnv) -> list[dict[str, Any]]:
        obstacles: list[dict[str, Any]] = []
        for obstacle_kind, obstacle_list in (("static", env.static_obstacles), ("dynamic", env.dynamic_obstacles)):
            for obstacle in obstacle_list:
                item: dict[str, Any] = {
                    "kind": obstacle_kind,
                    "center": np.asarray(obstacle.center, dtype=float).copy(),
                }
                if hasattr(obstacle, "expanded_half_extents"):
                    item["type"] = "box"
                    item["half_extents"] = np.asarray(obstacle.expanded_half_extents, dtype=float).copy()
                elif hasattr(obstacle, "effective_radius"):
                    item["type"] = "sphere"
                    item["radius"] = float(obstacle.effective_radius)
                else:
                    item["type"] = "unsupported"
                obstacles.append(item)
        return obstacles

    @staticmethod
    def _set_equal_3d_axes(ax: Any, points: list[np.ndarray]) -> None:
        stacked = np.vstack(points)
        lower = stacked.min(axis=0)
        upper = stacked.max(axis=0)
        span = upper - lower
        min_span = 0.5
        for axis_index in range(3):
            if span[axis_index] < min_span:
                center = 0.5 * (lower[axis_index] + upper[axis_index])
                lower[axis_index] = center - 0.5 * min_span
                upper[axis_index] = center + 0.5 * min_span
                span[axis_index] = min_span

        padding = 0.08 * span
        lower = lower - padding
        upper = upper + padding
        span = upper - lower
        ax.set_xlim(lower[0], upper[0])
        ax.set_ylim(lower[1], upper[1])
        ax.set_zlim(lower[2], upper[2])
        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect(span)

    @staticmethod
    def _plot_sphere(ax: Any, center: np.ndarray, radius: float, color: str, label: str | None) -> None:
        u = np.linspace(0.0, 2.0 * np.pi, 24)
        v = np.linspace(0.0, np.pi, 14)
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(x, y, z, color=color, alpha=0.18, linewidth=0.0, shade=False)
        ax.scatter([center[0]], [center[1]], [center[2]], color=color, s=24, label=label)

    @staticmethod
    def _plot_box(ax: Any, center: np.ndarray, half_extents: np.ndarray, color: str, label: str | None) -> None:
        x0, y0, z0 = center - half_extents
        x1, y1, z1 = center + half_extents
        vertices = np.array(
            [
                [x0, y0, z0],
                [x1, y0, z0],
                [x1, y1, z0],
                [x0, y1, z0],
                [x0, y0, z1],
                [x1, y0, z1],
                [x1, y1, z1],
                [x0, y1, z1],
            ],
            dtype=float,
        )
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
        for start_index, end_index in edges:
            ax.plot(
                [vertices[start_index, 0], vertices[end_index, 0]],
                [vertices[start_index, 1], vertices[end_index, 1]],
                [vertices[start_index, 2], vertices[end_index, 2]],
                color=color,
                linewidth=1.2,
            )
        ax.scatter([center[0]], [center[1]], [center[2]], color=color, s=24, label=label)

    def _save_visualization(
        self,
        row: dict[str, float | int],
        trajectory: np.ndarray,
        start: np.ndarray,
        goal: np.ndarray,
        obstacles: list[dict[str, Any]],
    ) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        save_path = self.visualization_dir / f"eval_seed_{int(row['seed'])}_step_{int(row['timesteps'])}.png"
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(1, 1, 1, projection="3d")

        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color="tab:blue", linewidth=2.0, label="trajectory")
        ax.scatter([start[0]], [start[1]], [start[2]], color="tab:green", marker="o", s=70, label="start")
        ax.scatter([goal[0]], [goal[1]], [goal[2]], color="tab:orange", marker="*", s=130, label="goal")
        ax.scatter(
            [trajectory[-1, 0]],
            [trajectory[-1, 1]],
            [trajectory[-1, 2]],
            color="black",
            marker="x",
            s=70,
            label="final",
        )

        axis_points = [trajectory.min(axis=0), trajectory.max(axis=0), start, goal]
        used_labels: set[str] = set()
        for obstacle in obstacles:
            center = obstacle["center"]
            color = "tab:red" if obstacle["kind"] == "static" else "tab:purple"
            label_key = f"{obstacle['kind']} {obstacle['type']}"
            label = None if label_key in used_labels else label_key
            used_labels.add(label_key)
            if obstacle["type"] == "sphere":
                radius = float(obstacle["radius"])
                self._plot_sphere(ax, center, radius, color=color, label=label)
                axis_points.extend([center - radius, center + radius])
            elif obstacle["type"] == "box":
                half_extents = obstacle["half_extents"]
                self._plot_box(ax, center, half_extents, color=color, label=label)
                axis_points.extend([center - half_extents, center + half_extents])

        self._set_equal_3d_axes(ax, axis_points)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=32.0, azim=-58.0)
        title = (
            f"seed={int(row['seed'])} | reward={float(row['reward']):.2f} | length={int(row['length'])} | "
            f"success={int(row['success'])} collision={int(row['collision'])} timeout={int(row['timeout'])} | "
            f"distance={float(row['distance_to_goal']):.3f}"
        )
        ax.set_title(title, fontsize=10)
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _run_episode(self, seed: int) -> tuple[dict[str, float | int], dict[str, Any]]:
        observation, _ = self.eval_env.reset(seed=seed)
        start = self.eval_env.dynamics.p.copy()
        goal = self.eval_env.goal.copy()
        obstacles = self._extract_obstacles(self.eval_env)
        trajectory = [start.copy()]
        total_reward = 0.0
        length = 0
        info: dict[str, Any] = {}
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, _ = self.model.predict(observation, deterministic=self.deterministic)
            observation, reward, terminated, truncated, info = self.eval_env.step(action)
            total_reward += float(reward)
            length += 1
            trajectory.append(self.eval_env.dynamics.p.copy())

        row = {
            "timesteps": int(self.num_timesteps),
            "seed": int(seed),
            "reward": float(total_reward),
            "length": int(length),
            "success": int(bool(info.get("success", False))),
            "collision": int(bool(info.get("collision", False))),
            "timeout": int(bool(info.get("truncated", False))),
            "distance_to_goal": float(info.get("distance_to_goal", np.nan)),
        }
        visualization_data = {
            "trajectory": np.asarray(trajectory, dtype=float),
            "start": start,
            "goal": goal,
            "obstacles": obstacles,
        }
        return row, visualization_data

    def _write_csv_rows(self, rows: list[dict[str, float | int]]) -> None:
        with self.csv_path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.CSV_FIELDS)
            writer.writerows(rows)

    def _write_tensorboard(self, rows: list[dict[str, float | int]]) -> None:
        if self.writer is None:
            return

        rewards = [float(row["reward"]) for row in rows]
        lengths = [float(row["length"]) for row in rows]
        successes = [float(row["success"]) for row in rows]
        collisions = [float(row["collision"]) for row in rows]
        timeouts = [float(row["timeout"]) for row in rows]
        distances = [float(row["distance_to_goal"]) for row in rows]

        self.writer.add_scalar("eval/mean_reward", float(np.mean(rewards)), self.num_timesteps)
        self.writer.add_scalar("eval/std_reward", float(np.std(rewards)), self.num_timesteps)
        self.writer.add_scalar("eval/mean_length", float(np.mean(lengths)), self.num_timesteps)
        self.writer.add_scalar("eval/success_rate", float(np.mean(successes)), self.num_timesteps)
        self.writer.add_scalar("eval/collision_rate", float(np.mean(collisions)), self.num_timesteps)
        self.writer.add_scalar("eval/timeout_rate", float(np.mean(timeouts)), self.num_timesteps)
        self.writer.add_scalar("eval/mean_distance_to_goal", float(np.mean(distances)), self.num_timesteps)
        self.writer.flush()

    def _maybe_save_best_eval(self, mean_reward: float) -> None:
        if mean_reward <= self.best_mean_reward:
            return

        self.best_mean_reward = float(mean_reward)
        save_checkpoint(
            model=self.model,
            model_path=self.best_eval_model_path,
            extra={"best_eval_mean_reward": self.best_mean_reward},
        )
        if self.verbose > 0:
            print(
                f"[best_eval_model] timesteps={self.num_timesteps}, "
                f"mean_reward={self.best_mean_reward:.3f}"
            )

    def _evaluate(self) -> None:
        actor_was_training = bool(getattr(self.model.actor, "training", False))
        critic_was_training = bool(getattr(self.model.critic, "training", False))
        self.model.actor.train(False)
        self.model.critic.train(False)
        try:
            episode_results = [self._run_episode(seed) for seed in self.eval_seeds]
        finally:
            self.model.actor.train(actor_was_training)
            self.model.critic.train(critic_was_training)

        rows = [row for row, _ in episode_results]
        if self.save_visualizations:
            for row, visualization_data in episode_results:
                self._save_visualization(
                    row=row,
                    trajectory=visualization_data["trajectory"],
                    start=visualization_data["start"],
                    goal=visualization_data["goal"],
                    obstacles=visualization_data["obstacles"],
                )

        rewards = [float(row["reward"]) for row in rows]
        lengths = [float(row["length"]) for row in rows]
        successes = [float(row["success"]) for row in rows]
        collisions = [float(row["collision"]) for row in rows]
        timeouts = [float(row["timeout"]) for row in rows]
        distances = [float(row["distance_to_goal"]) for row in rows]
        mean_reward = float(np.mean(rewards))

        self._write_csv_rows(rows)
        self._write_tensorboard(rows)
        self._maybe_save_best_eval(mean_reward)

        if self.verbose > 0:
            print(
                f"[eval] timesteps={self.num_timesteps}, "
                f"reward={mean_reward:.3f}, "
                f"length={np.mean(lengths):.1f}, "
                f"success={np.mean(successes):.2f}, "
                f"collision={np.mean(collisions):.2f}, "
                f"timeout={np.mean(timeouts):.2f}, "
                f"distance={np.mean(distances):.3f}"
            )

    def _on_step(self) -> bool:
        if self.num_timesteps > 0 and self.num_timesteps % self.eval_freq == 0:
            self._evaluate()
        return True

    def _on_training_end(self) -> None:
        self.eval_env.close()
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()


def train(
    total_timesteps: int | None = None,
    output_root: str | None = None,
    save_freq: int | None = None,
    resume_from: str | None = None,
    config: SACExperimentConfig = EXPERIMENT_CONFIG,
) -> dict[str, str]:
    """Run training and return artifact paths."""
    total_timesteps = config.total_timesteps if total_timesteps is None else total_timesteps
    output_root = config.output_root if output_root is None else output_root
    save_freq = config.save_freq if save_freq is None else save_freq

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / timestamp
    tensorboard_dir = run_dir / "tensorboard"

    curriculum_state = config.build_curriculum_state() if bool(config.training_scene_mixture_enabled) else None
    env = build_env(
        config=config,
        action_guidance_enabled=config.action_guidance_enabled,
        curriculum_state=curriculum_state,
    )
    eval_config = replace(config, training_scene_mixture_enabled=False, curriculum_enabled=False)
    eval_env = build_env(config=eval_config, action_guidance_enabled=False)
    model = build_model(env, config=config, tensorboard_log=str(tensorboard_dir))
    if resume_from:
        model = load_checkpoint(model, resume_from)

    checkpoint_callback = CheckpointAndBestCallback(run_dir=run_dir, save_freq=save_freq, verbose=1)
    tensorboard_callback = TensorboardRewardCallback(log_dir=tensorboard_dir, flush_freq=100)
    curriculum_callback = SuccessRateCurriculumCallback(
        config=config,
        curriculum_state=curriculum_state,
        log_dir=tensorboard_dir,
        verbose=1,
    )
    eval_callback = FixedSeedEvalCallback(
        eval_env=eval_env,
        run_dir=run_dir,
        log_dir=tensorboard_dir,
        eval_freq=config.eval_freq,
        eval_seeds=config.eval_seeds,
        deterministic=config.eval_deterministic,
        save_visualizations=True,
        verbose=1,
    )
    callback = CallbackList([checkpoint_callback, tensorboard_callback, curriculum_callback, eval_callback])

    model.learn(total_timesteps=total_timesteps, callback=callback)

    final_model_path = run_dir / "final_model.pt"
    save_checkpoint(
        model=model,
        model_path=final_model_path,
        extra={
            "best_episode_reward": checkpoint_callback.best_episode_reward,
            "best_eval_mean_reward": eval_callback.best_mean_reward,
            "curriculum_state": dict(curriculum_state or {}),
        },
    )
    env.close()

    return {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_callback.latest_checkpoint_path),
        "best_model": str(checkpoint_callback.best_model_path),
        "best_eval_model": str(eval_callback.best_eval_model_path),
        "eval_csv": str(eval_callback.csv_path),
        "final_model": str(final_model_path),
        "tensorboard_dir": str(tensorboard_dir),
    }


if __name__ == "__main__":
    outputs = train()
    print("训练完成，已保存文件：")
    print(f"run_dir: {outputs['run_dir']}")
    print(f"checkpoint: {outputs['checkpoint']}")
    print(f"best_model: {outputs['best_model']}")
    print(f"best_eval_model: {outputs['best_eval_model']}")
    print(f"eval_csv: {outputs['eval_csv']}")
    print(f"final_model: {outputs['final_model']}")
    print(f"tensorboard_dir: {outputs['tensorboard_dir']}")
