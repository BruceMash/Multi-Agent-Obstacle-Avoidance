# coding: utf-8

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


SCRIPT_PATH = Path(__file__).resolve()
ALGO_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]
for path in (PROJECT_ROOT, ALGO_ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from Environment.multi_agent_dmp_env import MultiAgentDMPEnv
from MASAC.MASAC import MASAC
from MASAC.curriculum import (
    CurriculumStage,
    SuccessRateCurriculum,
    build_curriculum_stages,
    build_stage_env_kwargs,
)
from MASAC.config import MASAC_EXPERIMENT_CONFIG, MASACExperimentConfig, MASACNetworkConfig # 读取配置信息

REWARD_COMPONENT_KEYS = (
    "reward_progress",
    "reward_obstacle_potential_penalty",
    "reward_boundary_potential_penalty",
    "reward_inter_agent_potential_penalty",
    "reward_stagnation_penalty",
    "reward_individual_success_bonus",
    "reward_team_success_bonus",
    "reward_team_collision_penalty",
    "reward_local_collision_penalty",
    "reward_team_timeout_penalty",
)


def matrix_to_agent_dict(values: np.ndarray, agent_ids: list[str]) -> dict[str, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    return {
        agent_id: values[index].astype(np.float32, copy=True)
        for index, agent_id in enumerate(agent_ids)
    }


def vector_to_agent_dict(values: np.ndarray, agent_ids: list[str]) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float32)
    return {
        agent_id: float(values[index])
        for index, agent_id in enumerate(agent_ids)
    }


def agent_dict_to_matrix(values: dict[str, np.ndarray], agent_ids: list[str]) -> np.ndarray:
    return np.stack(
        [np.asarray(values[agent_id], dtype=np.float32) for agent_id in agent_ids],
        axis=0,
    )


def state_bound_tuple(value, state_dim: int) -> tuple[float, ...]:
    values = np.asarray(value, dtype=np.float32)
    if values.ndim == 0:
        values = np.full(state_dim, float(values), dtype=np.float32)
    values = values.reshape(-1)
    if values.shape != (state_dim,):
        raise ValueError(f"bound must have shape ({state_dim},), got {tuple(values.shape)}")
    return tuple(float(item) for item in values)


def build_critic_action_matrix(info: dict, env: MultiAgentDMPEnv) -> np.ndarray:
    applied_accelerations = np.asarray(
        info["applied_accelerations"],
        dtype=np.float32,
    )
    guided_action = np.asarray(info["guided_action"], dtype=np.float32)
    dmp_dims = int(env.dmp_config.dims)
    goal_offset = guided_action[:, dmp_dims : 2 * dmp_dims]
    critic_action = np.concatenate([applied_accelerations, goal_offset], axis=1)
    expected_shape = env.action_shape
    if critic_action.shape != expected_shape:
        raise ValueError(
            f"critic action must have shape {expected_shape}, "
            f"got {critic_action.shape}"
        )
    return critic_action.astype(np.float32, copy=False)


def build_env(
    config: MASACExperimentConfig,
    curriculum_stage: CurriculumStage | None = None,
) -> MultiAgentDMPEnv:
    kwargs = (
        config.build_core_env_kwargs()
        if curriculum_stage is None
        else build_stage_env_kwargs(config, curriculum_stage)
    )
    return MultiAgentDMPEnv(**kwargs)


def build_dim_info(env: MultiAgentDMPEnv, agent_ids: list[str]) -> dict[str, tuple[int, int]]:
    _, obs_dim = env.observation_shape
    _, action_dim = env.action_shape
    return {
        agent_id: (int(obs_dim), int(action_dim))
        for agent_id in agent_ids
    }


def build_network_config(
    env: MultiAgentDMPEnv,
    experiment_config: MASACExperimentConfig,
    args: argparse.Namespace,
) -> MASACNetworkConfig:
    sensor = env.sensors[0]
    state_dim = int(env.state_dim)
    action_low = tuple(float(value) for value in env.action_space.low[0])
    action_high = tuple(float(value) for value in env.action_space.high[0])
    acceleration_low = state_bound_tuple(env.dynamics[0].accelerate_min, state_dim)
    acceleration_high = state_bound_tuple(env.dynamics[0].accelerate_max, state_dim)
    ally_feature_dim = (
        int(env.single_pair_observation_dim)
        if int(env.nearest_agent_observation_count) > 0
        else 0
    )
    return MASACNetworkConfig(
        sensor_observation_dim=int(env.sensor_observation_dim),
        extra_observation_dim=int(env.extra_observation_dim),
        ally_feature_dim=ally_feature_dim,
        sensor_output_dim=int(args.sensor_output_dim),
        ally_output_dim=int(args.ally_output_dim),
        sensor_hidden_dim=int(args.sensor_hidden_dim),
        ally_hidden_dim=int(args.ally_hidden_dim),
        hidden_dim=int(args.hidden_dim),
        num_sensor_layers=int(args.num_sensor_layers),
        num_ally_layers=int(args.num_ally_layers),
        num_observation_layers=int(args.num_observation_layers),
        sensor_azimuth_bins=int(sensor.azimuth_bins),
        sensor_elevation_bins=int(sensor.elevation_bins),
        sensor_elevation_range_deg=tuple(float(v) for v in sensor.elevation_range_deg),
        sensor_include_previous_scan=bool(sensor.include_previous_scan),
        action_low=action_low,
        action_high=action_high,
        temporal_steps=int(args.temporal_steps),
        actor_log_std_min=float(args.actor_log_std_min),
        actor_log_std_max=float(args.actor_log_std_max),
        ally_pooling=str(experiment_config.ally_pooling),
        agent_pooling=str(experiment_config.agent_pooling),
        critic_encoder=str(args.critic_encoder),
        goal_distance_clip=float(sensor.goal_distance_clip),
        dmp_k_alpha=float(env.dmp_config.K_alpha),
        dmp_k_beta=float(env.dmp_config.K_beta),
        dmp_tau=float(env.dmp_config.tau),
        forcing_term_min=float(env.dmp_config.forcing_term_min),
        forcing_term_max=float(env.dmp_config.forcing_term_max),
        acceleration_low=acceleration_low,
        acceleration_high=acceleration_high,
    )


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return requested


def make_run_dir(output_root: str, seed: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / f"masac_multi_agent_dmp_seed_{seed}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False, default=str)


def flatten_hparams(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:   # 添加tensorboard记录
    flattened: dict[str, Any] = {}
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(flatten_hparams(value, name))
        elif isinstance(value, (str, int, float, bool)):
            flattened[name] = value
        elif value is None:
            flattened[name] = "null"
        else:
            flattened[name] = json.dumps(value, ensure_ascii=False, default=str)
    return flattened


def write_tensorboard_configuration(writer, payload: dict[str, Any]) -> None:   # tensorboard记录
    if writer is None:
        return
    for section, values in payload.items():
        writer.add_text(
            f"configuration/{section}",
            f"```json\n{json.dumps(values, indent=2, ensure_ascii=False, default=str)}\n```",
            0,
        )
    try:
        writer.add_hparams(
            flatten_hparams(payload),
            {"hparams/session_start": 0.0},
            run_name="hparams",
        )
    except (TypeError, ValueError):
        pass
    writer.flush()


def append_metrics(path: Path, row: dict[str, Any]) -> None:
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def mask_count(info: dict[str, Any], key: str) -> int:
    return int(np.sum(np.asarray(info.get(key, []), dtype=bool)))


def set_policy_train_mode(policy: MASAC, training: bool) -> None:
    for agent in policy.agents.values():
        agent.actor.train(training)
        agent.critic.train(training)
        agent.actor_target.train(training)
        agent.critic_target.train(training)


def evaluate_policy(
    *,
    policy: MASAC,
    experiment_config: MASACExperimentConfig,
    agent_ids: list[str],
    eval_seeds: list[int],
    curriculum_stage: CurriculumStage | None = None,
) -> dict[str, Any]:
    eval_env = build_env(experiment_config, curriculum_stage)
    try:
        previous_training_mode = next(iter(policy.agents.values())).actor.training
        set_policy_train_mode(policy, False)
        episode_rows = []
        for eval_index, eval_seed in enumerate(eval_seeds):
            obs_matrix, _ = eval_env.reset(seed=int(eval_seed))
            obs = matrix_to_agent_dict(obs_matrix, agent_ids)
            episode_reward = np.zeros(int(eval_env.num_agents), dtype=np.float64)
            episode_step = 0
            terminated = False
            truncated = False
            info: dict[str, Any] = {}
            inter_agent_collision_event = False
            obstacle_collision_event = False
            boundary_collision_event = False
            min_inter_agent_distance = float("inf")

            while not bool(terminated or truncated):
                action = policy.evaluate_action(obs)
                action_matrix = agent_dict_to_matrix(action, agent_ids)
                action_matrix = np.clip(
                    action_matrix,
                    eval_env.action_space.low,
                    eval_env.action_space.high,
                ).astype(np.float32)
                next_obs_matrix, rewards, terminated, truncated, info = eval_env.step(action_matrix)
                obs = matrix_to_agent_dict(next_obs_matrix, agent_ids)
                episode_reward += np.asarray(rewards, dtype=np.float64)
                episode_step += 1

                inter_agent_collision_event = inter_agent_collision_event or (
                    mask_count(info, "inter_agent_collision_mask") > 0
                )
                obstacle_collision_event = obstacle_collision_event or (
                    mask_count(info, "obstacle_collision_mask") > 0
                )
                boundary_collision_event = boundary_collision_event or (
                    mask_count(info, "boundary_collision_mask") > 0
                )
                min_inter_agent_distance = min(
                    min_inter_agent_distance,
                    float(info.get("min_inter_agent_distance", np.inf)),
                )

            success_mask = np.asarray(
                info.get("success_mask", np.zeros(int(eval_env.num_agents), dtype=bool)),
                dtype=bool,
            )
            success = bool(info.get("success", bool(np.all(success_mask))))
            reached_agent_count = int(np.sum(success_mask))
            episode_rows.append(
                {
                    "eval_index": eval_index,
                    "eval_seed": int(eval_seed),
                    "success": success,
                    "episode_step": int(episode_step),
                    "mean_reward": float(np.mean(episode_reward)),
                    "sum_reward": float(np.sum(episode_reward)),
                    "reached_agent_count": reached_agent_count,
                    "agent_success_rate": float(reached_agent_count / float(eval_env.num_agents)),
                    "inter_agent_collision": bool(inter_agent_collision_event),
                    "obstacle_collision": bool(obstacle_collision_event),
                    "boundary_collision": bool(boundary_collision_event),
                    "min_inter_agent_distance": float(min_inter_agent_distance),
                }
            )

        episode_count = max(1, len(episode_rows))
        return {
            "eval_episode_count": int(len(episode_rows)),
            "eval_success_rate": float(np.mean([row["success"] for row in episode_rows])),
            "eval_agent_success_rate": float(np.mean([row["agent_success_rate"] for row in episode_rows])),
            "eval_mean_reward": float(np.mean([row["mean_reward"] for row in episode_rows])),
            "eval_sum_reward": float(np.mean([row["sum_reward"] for row in episode_rows])),
            "eval_mean_len": float(np.mean([row["episode_step"] for row in episode_rows])),
            "eval_inter_agent_collision_rate": float(
                np.sum([row["inter_agent_collision"] for row in episode_rows]) / episode_count
            ),
            "eval_obstacle_collision_rate": float(
                np.sum([row["obstacle_collision"] for row in episode_rows]) / episode_count
            ),
            "eval_boundary_collision_rate": float(
                np.sum([row["boundary_collision"] for row in episode_rows]) / episode_count
            ),
            "eval_min_inter_agent_distance": float(
                np.min([row["min_inter_agent_distance"] for row in episode_rows])
            ),
        }
    finally:
        set_policy_train_mode(policy, previous_training_mode)
        eval_env.close()


class TerminalProgress:
    """Render training progress with tqdm, with a plain text fallback."""

    def __init__(self, *, total_steps: int, success_window: int, plain: bool = False):
        self.total_steps = max(1, int(total_steps))
        self.success_window = max(1, int(success_window))
        self.plain = bool(plain) or tqdm is None
        self.last_step = 0
        self.last_status: dict[str, Any] | None = None
        self.pbar = None
        if not self.plain:
            self.pbar = tqdm(
                total=self.total_steps,
                desc="MASAC",
                unit="step",
                dynamic_ncols=True,
                leave=True,
            )

    @staticmethod
    def _format_elapsed(seconds: float) -> str:
        seconds = max(0, int(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _plain_line(self, status: dict[str, Any]) -> str:
        return (
            f"MASAC | step {int(status['global_step']):,}/{self.total_steps:,} "
            f"| ep {int(status['episode_count'])} "
            f"| buf {int(status['buffer_size']):,} "
            f"| {status['curriculum_stage']} "
            f"| SR{self.success_window} {float(status['rolling_success_rate']):.3f} "
            f"| IA {int(status['inter_agent_collision_count'])} "
            f"| OBS {int(status['obstacle_collision_count'])} "
            f"| {self._format_elapsed(float(status['elapsed_sec']))}"
        )

    def update(self, **status: Any) -> None:
        self.last_status = dict(status)
        current_step = min(max(int(status["global_step"]), 0), self.total_steps)
        if self.pbar is None:
            print(self._plain_line(self.last_status), flush=True)
            self.last_step = current_step
            return

        delta = current_step - self.last_step
        if delta > 0:
            self.pbar.update(delta)
            self.last_step = current_step
        self.pbar.set_postfix(
            {
                "ep": int(status["episode_count"]),
                "buf": f"{int(status['buffer_size']):,}",
                "stage": status["curriculum_stage"],
                f"SR{self.success_window}": f"{float(status['rolling_success_rate']):.3f}",
                "IA": int(status["inter_agent_collision_count"]),
                "OBS": int(status["obstacle_collision_count"]),
                "elapsed": self._format_elapsed(float(status["elapsed_sec"])),
            },
            refresh=True,
        )

    def event(self, message: str) -> None:
        if self.pbar is None:
            print(message, flush=True)
            return
        tqdm.write(message)

    def close(self) -> None:
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None


def write_tensorboard_episode(writer, row: dict[str, Any]) -> None:
    if writer is None:
        return
    step = int(row["global_step"])
    writer.add_scalar("episode/mean_reward", row["mean_reward"], step)
    writer.add_scalar("episode/sum_reward", row["sum_reward"], step)
    writer.add_scalar("episode/length", row["episode_step"], step)
    writer.add_scalar("success/full_success", float(row["success"]), step)
    writer.add_scalar("success/agent_success_rate", row["agent_success_rate"], step)
    writer.add_scalar("success/rolling_success_rate", row["rolling_success_rate"], step)
    writer.add_scalar("curriculum/phase", row["curriculum_phase"], step)
    writer.add_scalar("curriculum/level", row["curriculum_level"], step)
    writer.add_scalar("curriculum/window_count", row["curriculum_window_count"], step)
    writer.add_scalar("curriculum/ground_box_count", row["ground_box_count"], step)
    writer.add_scalar("curriculum/aerial_sphere_count", row["aerial_sphere_count"], step)
    writer.add_scalar("curriculum/dynamic_sphere_count", row["dynamic_sphere_count"], step)
    writer.add_scalar("collision/inter_agent_event", float(row["inter_agent_collision"]), step)
    writer.add_scalar("collision/obstacle_event", float(row["obstacle_collision"]), step)
    writer.add_scalar("collision/boundary_event", float(row["boundary_collision"]), step)
    writer.add_scalar("collision/inter_agent_steps", row["inter_agent_collision_steps"], step)
    writer.add_scalar("collision/obstacle_steps", row["obstacle_collision_steps"], step)
    writer.add_scalar("collision/boundary_steps", row["boundary_collision_steps"], step)
    writer.add_scalar("collision/inter_agent_agent_count", row["inter_agent_collision_agent_count"], step)
    writer.add_scalar("collision/obstacle_agent_count", row["obstacle_collision_agent_count"], step)
    writer.add_scalar("collision/boundary_agent_count", row["boundary_collision_agent_count"], step)
    writer.add_scalar("safety/min_inter_agent_distance", row["min_inter_agent_distance"], step)
    for key in REWARD_COMPONENT_KEYS:
        writer.add_scalar(f"reward/{key.removeprefix('reward_')}", row[key], step)
    writer.add_scalar("stagnation/trigger_steps", row["stagnation_trigger_steps"], step)
    writer.add_scalar("stagnation/max_counter", row["stagnation_max_counter"], step)


def write_tensorboard_step(
    writer,
    *,
    global_step: int,
    buffer_size: int,
    episode_count: int,
    rolling_success_rate: float,
    inter_agent_collision_count: int,
    obstacle_collision_count: int,
    min_inter_agent_distance: float,
    curriculum_phase: int,
    curriculum_level: int,
) -> None:
    if writer is None:
        return
    writer.add_scalar("train/global_step", global_step, global_step)
    writer.add_scalar("train/buffer_size", buffer_size, global_step)
    writer.add_scalar("train/episode_count", episode_count, global_step)
    writer.add_scalar("success/rolling_success_rate_step", rolling_success_rate, global_step)
    writer.add_scalar("curriculum/phase_step", curriculum_phase, global_step)
    writer.add_scalar("curriculum/level_step", curriculum_level, global_step)
    writer.add_scalar("collision/inter_agent_agent_count_step", inter_agent_collision_count, global_step)
    writer.add_scalar("collision/obstacle_agent_count_step", obstacle_collision_count, global_step)
    writer.add_scalar("safety/min_inter_agent_distance_step", min_inter_agent_distance, global_step)


def write_tensorboard_eval(writer, row: dict[str, Any]) -> None:
    if writer is None:
        return
    step = int(row["global_step"])
    writer.add_scalar("eval/success_rate", row["eval_success_rate"], step)
    writer.add_scalar("eval/agent_success_rate", row["eval_agent_success_rate"], step)
    writer.add_scalar("eval/mean_reward", row["eval_mean_reward"], step)
    writer.add_scalar("eval/sum_reward", row["eval_sum_reward"], step)
    writer.add_scalar("eval/mean_len", row["eval_mean_len"], step)
    writer.add_scalar("eval/curriculum_phase", row["curriculum_phase"], step)
    writer.add_scalar("eval/curriculum_level", row["curriculum_level"], step)
    writer.add_scalar(
        "eval/inter_agent_collision_rate",
        row["eval_inter_agent_collision_rate"],
        step,
    )
    writer.add_scalar(
        "eval/obstacle_collision_rate",
        row["eval_obstacle_collision_rate"],
        step,
    )
    writer.add_scalar(
        "eval/boundary_collision_rate",
        row["eval_boundary_collision_rate"],
        step,
    )
    writer.add_scalar(
        "eval/min_inter_agent_distance",
        row["eval_min_inter_agent_distance"],
        step,
    )


def parse_int_tuple(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not parsed or any(item < 0 for item in parsed):
        raise argparse.ArgumentTypeError("counts must be non-negative integers")
    return parsed


def parse_float_range(value: str) -> tuple[float, float]:
    try:
        parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected two comma-separated numbers") from error
    if len(parsed) != 2 or parsed[0] < 0.0 or parsed[0] >= parsed[1]:
        raise argparse.ArgumentTypeError("range must satisfy 0 <= minimum < maximum")
    return parsed


def parse_args() -> argparse.Namespace:
    config = MASAC_EXPERIMENT_CONFIG
    parser = argparse.ArgumentParser(
        description="Train MASAC on the matrix-style MultiAgentDMPEnv."
    )
    parser.add_argument("--seed", type=int, default=int(config.seed))
    # 算法相关
    parser.add_argument("--total-steps", type=int, default=int(config.total_steps))
    parser.add_argument("--start-steps", type=int, default=int(config.start_steps))
    parser.add_argument("--batch-size", type=int, default=int(config.batch_size))
    parser.add_argument("--buffer-size", type=int, default=int(config.buffer_size))
    parser.add_argument("--actor-lr", type=float, default=float(config.actor_lr))
    parser.add_argument("--critic-lr", type=float, default=float(config.critic_lr))
    parser.add_argument("--gamma", type=float, default=float(config.gamma))
    parser.add_argument("--tau", type=float, default=float(config.soft_update_tau))
    parser.add_argument("--learn-interval", type=int, default=int(config.learn_interval))
    parser.add_argument("--updates-per-step", type=int, default=int(config.updates_per_step))
    parser.add_argument("--temporal-steps", type=int, default=int(config.temporal_steps))

    # 网络结构线管
    parser.add_argument("--hidden-dim", type=int, default=int(config.hidden_dim))
    parser.add_argument("--sensor-hidden-dim", type=int, default=int(config.sensor_hidden_dim))
    parser.add_argument("--ally-hidden-dim", type=int, default=int(config.ally_hidden_dim))
    parser.add_argument("--sensor-output-dim", type=int, default=int(config.sensor_output_dim))
    parser.add_argument("--ally-output-dim", type=int, default=int(config.ally_output_dim))
    parser.add_argument("--num-sensor-layers", type=int, default=int(config.num_sensor_layers))
    parser.add_argument("--num-ally-layers", type=int, default=int(config.num_ally_layers))
    parser.add_argument("--num-observation-layers", type=int, default=int(config.num_observation_layers))
    parser.add_argument("--actor-log-std-min", type=float, default=float(config.actor_log_std_min))
    parser.add_argument("--actor-log-std-max", type=float, default=float(config.actor_log_std_max))
    parser.add_argument(
        "--critic-encoder",
        type=str,
        choices=("attention", "mlp"),
        default=str(config.critic_encoder),
    )

    # Runtime and logging options
    parser.add_argument("--device", type=str, default=str(config.device))
    parser.add_argument("--output-root", type=str, default=str(config.output_root))
    parser.add_argument("--save-interval", type=int, default=int(config.save_interval))
    parser.add_argument("--eval-interval", type=int, default=int(config.save_interval))
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-seed-base", type=int, default=100_000)
    parser.add_argument("--log-interval", type=int, default=int(config.log_interval))
    parser.add_argument("--progress-interval", type=int, default=int(config.progress_interval))
    parser.add_argument("--success-window", type=int, default=int(config.success_window))
    parser.add_argument(
        "--curriculum-success-threshold",
        type=float,
        default=float(config.curriculum_success_threshold),
    )
    parser.add_argument(
        "--phase2-box-counts",
        type=parse_int_tuple,
        default=tuple(config.curriculum_phase2_box_counts),
    )
    parser.add_argument(
        "--phase2-sphere-counts",
        type=parse_int_tuple,
        default=tuple(config.curriculum_phase2_sphere_counts),
    )
    parser.add_argument(
        "--phase3-dynamic-counts",
        type=parse_int_tuple,
        default=tuple(config.curriculum_phase3_dynamic_counts),
    )
    parser.add_argument(
        "--box-half-extent-range",
        type=parse_float_range,
        default=tuple(config.curriculum_box_half_extent_range),
    )
    parser.add_argument(
        "--box-height-range",
        type=parse_float_range,
        default=tuple(config.curriculum_box_height_range),
    )
    parser.add_argument(
        "--aerial-sphere-radius-range",
        type=parse_float_range,
        default=tuple(config.curriculum_aerial_sphere_radius_range),
    )
    parser.add_argument(
        "--dynamic-sphere-radius-range",
        type=parse_float_range,
        default=tuple(config.curriculum_dynamic_sphere_radius_range),
    )
    parser.add_argument(
        "--dynamic-speed-range",
        type=parse_float_range,
        default=tuple(config.curriculum_dynamic_speed_range),
    )
    parser.add_argument(
        "--aerial-min-center-height",
        type=float,
        default=float(config.curriculum_aerial_min_center_height),
    )
    parser.add_argument(
        "--obstacle-safety-margin",
        type=float,
        default=float(config.curriculum_obstacle_safety_margin),
    )
    parser.add_argument(
        "--start-goal-clearance",
        type=float,
        default=float(config.curriculum_start_goal_clearance),
    )
    parser.add_argument(
        "--obstacle-separation",
        type=float,
        default=float(config.curriculum_obstacle_separation),
    )
    parser.add_argument(
        "--placement-attempts",
        type=int,
        default=int(config.curriculum_placement_attempts),
    )
    parser.add_argument(
        "--curved-turn-rate",
        type=float,
        default=float(config.curriculum_curved_turn_rate),
    )
    parser.add_argument(
        "--wandering-strength",
        type=float,
        default=float(config.curriculum_wandering_strength),
    )
    scenario_group = parser.add_mutually_exclusive_group()
    scenario_group.add_argument(
        "--disable-curriculum",
        action="store_true",
        default=not bool(config.curriculum_enabled),
    )
    scenario_group.add_argument(
        "--final-stage-only",
        action="store_true",
        help="Train directly and exclusively on the final curriculum stage.",
    )
    parser.add_argument("--disable-tensorboard", action="store_true", default=bool(config.disable_tensorboard))
    parser.add_argument("--plain-progress", action="store_true", default=bool(config.plain_progress))
    return parser.parse_args()


def train() -> dict[str, str]:  # 训练主循环
    args = parse_args() # 读args
    args.progress_interval = max(1, int(args.progress_interval))

    # 训练间隔
    args.log_interval = max(1, int(args.log_interval))
    args.eval_interval = max(1, int(args.eval_interval))
    args.eval_episodes = max(1, int(args.eval_episodes))
    args.success_window = max(1, int(args.success_window))  # 计算滚动成功率均值
    if not 0.0 <= float(args.curriculum_success_threshold) <= 1.0:
        raise ValueError("curriculum success threshold must be in [0, 1]")
    non_negative_values = (
        args.aerial_min_center_height,
        args.obstacle_safety_margin,
        args.start_goal_clearance,
        args.obstacle_separation,
        args.curved_turn_rate,
        args.wandering_strength,
    )
    if any(float(value) < 0.0 for value in non_negative_values):
        raise ValueError("curriculum geometry and motion parameters must be non-negative")
    if int(args.placement_attempts) <= 0:
        raise ValueError("placement attempts must be positive")
    
    # 固定随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    experiment_config = replace(    # 创建实验配置 这部分主要是课程，同时从arg中读取参数
        MASAC_EXPERIMENT_CONFIG,
        curriculum_enabled=not bool(args.disable_curriculum or args.final_stage_only),
        curriculum_success_threshold=float(args.curriculum_success_threshold),
        curriculum_phase2_box_counts=tuple(args.phase2_box_counts),
        curriculum_phase2_sphere_counts=tuple(args.phase2_sphere_counts),
        curriculum_phase3_dynamic_counts=tuple(args.phase3_dynamic_counts),
        curriculum_box_half_extent_range=tuple(args.box_half_extent_range),
        curriculum_box_height_range=tuple(args.box_height_range),
        curriculum_aerial_sphere_radius_range=tuple(args.aerial_sphere_radius_range),
        curriculum_dynamic_sphere_radius_range=tuple(args.dynamic_sphere_radius_range),
        curriculum_dynamic_speed_range=tuple(args.dynamic_speed_range),
        curriculum_aerial_min_center_height=float(args.aerial_min_center_height),
        curriculum_obstacle_safety_margin=float(args.obstacle_safety_margin),
        curriculum_start_goal_clearance=float(args.start_goal_clearance),
        curriculum_obstacle_separation=float(args.obstacle_separation),
        curriculum_placement_attempts=int(args.placement_attempts),
        curriculum_curved_turn_rate=float(args.curved_turn_rate),
        curriculum_wandering_strength=float(args.wandering_strength),
    )
    curriculum_stages = build_curriculum_stages(
        args.phase2_box_counts,
        args.phase2_sphere_counts,
        args.phase3_dynamic_counts,
    )
    curriculum = SuccessRateCurriculum(
        curriculum_stages,
        success_threshold=float(args.curriculum_success_threshold),
        success_window=int(args.success_window),
        enabled=not bool(args.disable_curriculum or args.final_stage_only),
        initial_stage_index=(len(curriculum_stages) - 1 if args.final_stage_only else 0),
    )
    active_stage = (
        curriculum.current_stage
        if curriculum.enabled or bool(args.final_stage_only)
        else None
    )
    env = build_env(experiment_config, active_stage)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(args.seed)

    agent_ids = [f"agent_{index}" for index in range(int(env.num_agents))]
    dim_info = build_dim_info(env, agent_ids)
    network_config = build_network_config(env, experiment_config, args)
    device = resolve_device(args.device)

    policy = MASAC(
        dim_info=dim_info,
        is_continue=True,
        actor_lr=float(args.actor_lr),
        critic_lr=float(args.critic_lr),
        buffer_size=int(args.buffer_size),
        device=device,
        network_config=network_config,
    )

    run_dir = make_run_dir(args.output_root, args.seed)
    model_dir = run_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    eval_metrics_path = run_dir / "eval_metrics.csv"
    eval_seeds = [
        int(args.eval_seed_base) + seed_offset
        for seed_offset in range(int(args.eval_episodes))
    ]
    progress = TerminalProgress(
        total_steps=int(args.total_steps),
        success_window=int(args.success_window),
        plain=bool(args.plain_progress),
    )
    writer = None
    if not bool(args.disable_tensorboard) and SummaryWriter is not None:
        writer = SummaryWriter(str(run_dir / "tensorboard"))
        progress.event(f"TensorBoard logging enabled: {run_dir / 'tensorboard'}")
    elif bool(args.disable_tensorboard):
        progress.event("TensorBoard logging disabled by command line argument.")
    else:
        progress.event("TensorBoard logging is unavailable because tensorboard is not installed.")

    run_config = {
        "script_args": vars(args),
        "experiment_config": asdict(experiment_config),
        "core_env_kwargs": experiment_config.build_core_env_kwargs(),
        "dim_info": dim_info,
        "network_config": asdict(network_config),
        "eval_seeds": eval_seeds,
        "curriculum": curriculum.to_dict(),
        "runtime": {
            "device": str(device),
            "torch_version": str(torch.__version__),
            "numpy_version": str(np.__version__),
            "python_version": str(sys.version),
            "tensorboard_enabled": writer is not None,
        },
    }
    write_json(run_dir / "config.json", run_config)
    write_tensorboard_configuration(writer, run_config)

    obs_matrix, _ = env.reset(seed=args.seed)
    obs = matrix_to_agent_dict(obs_matrix, agent_ids)
    episode_reward = np.zeros(int(env.num_agents), dtype=np.float64)
    episode_reward_components = {
        key: np.zeros(int(env.num_agents), dtype=np.float64)
        for key in REWARD_COMPONENT_KEYS
    }
    episode_stagnation_trigger_steps = 0
    episode_stagnation_max_counter = 0
    episode_count = 0
    episode_step = 0
    start_time = time.time()
    episode_inter_agent_collision_steps = 0
    episode_obstacle_collision_steps = 0
    episode_boundary_collision_steps = 0
    episode_inter_agent_collision_agent_count = 0
    episode_obstacle_collision_agent_count = 0
    episode_boundary_collision_agent_count = 0
    latest_inter_agent_collision_count = 0
    latest_obstacle_collision_count = 0
    latest_min_inter_agent_distance = float("inf")
    best_eval_success_rate = -1.0
    best_eval_mean_reward = -float("inf")
    best_eval_stage_index = -1

    min_learn_size = max(int(args.batch_size), int(args.temporal_steps))
    for global_step in range(1, int(args.total_steps) + 1):
        if global_step <= int(args.start_steps):
            action_matrix = env.action_space.sample().astype(np.float32)
            action = matrix_to_agent_dict(action_matrix, agent_ids)
        else:
            action = policy.select_action(obs)
            action_matrix = agent_dict_to_matrix(action, agent_ids)
            action_matrix = np.clip(
                action_matrix,
                env.action_space.low,
                env.action_space.high,
            ).astype(np.float32)
            action = matrix_to_agent_dict(action_matrix, agent_ids)

        next_obs_matrix, rewards, terminated, truncated, info = env.step(action_matrix)
        latest_inter_agent_collision_count = mask_count(info, "inter_agent_collision_mask")
        latest_obstacle_collision_count = mask_count(info, "obstacle_collision_mask")
        latest_boundary_collision_count = mask_count(info, "boundary_collision_mask")
        latest_min_inter_agent_distance = float(info.get("min_inter_agent_distance", np.inf))
        if latest_inter_agent_collision_count > 0:
            episode_inter_agent_collision_steps += 1
            episode_inter_agent_collision_agent_count += latest_inter_agent_collision_count
        if latest_obstacle_collision_count > 0:
            episode_obstacle_collision_steps += 1
            episode_obstacle_collision_agent_count += latest_obstacle_collision_count
        if latest_boundary_collision_count > 0:
            episode_boundary_collision_steps += 1
            episode_boundary_collision_agent_count += latest_boundary_collision_count
        for key in REWARD_COMPONENT_KEYS:
            episode_reward_components[key] += np.asarray(
                info.get(key, np.zeros(int(env.num_agents), dtype=np.float32)),
                dtype=np.float64,
            )
        stagnation_mask = np.asarray(
            info.get("stagnation_mask", np.zeros(int(env.num_agents), dtype=bool)),
            dtype=bool,
        )
        if np.any(stagnation_mask):
            episode_stagnation_trigger_steps += 1
        stagnation_counters = np.asarray(
            info.get("stagnation_counters", np.zeros(int(env.num_agents), dtype=np.int32)),
            dtype=np.int32,
        )
        if stagnation_counters.size:
            episode_stagnation_max_counter = max(
                episode_stagnation_max_counter,
                int(np.max(stagnation_counters)),
            )

        next_obs = matrix_to_agent_dict(next_obs_matrix, agent_ids)
        reward = vector_to_agent_dict(rewards, agent_ids)
        critic_action_matrix = build_critic_action_matrix(info, env)
        critic_action = matrix_to_agent_dict(critic_action_matrix, agent_ids)
        done_for_buffer = {
            agent_id: bool(terminated)
            for agent_id in agent_ids
        }
        policy.add(obs, critic_action, reward, next_obs, done_for_buffer)

        episode_reward += np.asarray(rewards, dtype=np.float64)
        episode_step += 1
        obs = next_obs

        can_learn = len(policy.buffers[policy.agent_x]) >= min_learn_size
        if (
            can_learn
            and global_step > int(args.start_steps)
            and global_step % int(args.learn_interval) == 0
        ):
            for _ in range(int(args.updates_per_step)):
                policy.learn(
                    batch_size=int(args.batch_size),
                    gamma=float(args.gamma),
                    tau=float(args.tau),
                )

        episode_done = bool(terminated or truncated)
        if episode_done:
            episode_count += 1
            success_mask = np.asarray(
                info.get("success_mask", np.zeros(int(env.num_agents), dtype=bool)),
                dtype=bool,
            )
            success = bool(info.get("success", False))
            curriculum_result = curriculum.record_episode(success)
            completed_stage = curriculum_result["completed_stage"]
            rolling_success_rate = float(curriculum_result["completed_success_rate"])
            reached_agent_count = int(np.sum(success_mask))
            agent_success_rate = float(reached_agent_count / float(env.num_agents))
            inter_agent_collision_event = episode_inter_agent_collision_steps > 0
            obstacle_collision_event = episode_obstacle_collision_steps > 0
            boundary_collision_event = episode_boundary_collision_steps > 0
            row = {
                "global_step": global_step,
                "episode": episode_count,
                "episode_step": episode_step,
                "mean_reward": float(np.mean(episode_reward)),
                "sum_reward": float(np.sum(episode_reward)),
                "min_reward": float(np.min(episode_reward)),
                "max_reward": float(np.max(episode_reward)),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "collision": bool(info.get("collision", False)),
                "success": success,
                "reached_agent_count": reached_agent_count,
                "agent_success_rate": agent_success_rate,
                "rolling_success_rate": rolling_success_rate,
                "curriculum_phase": completed_stage.phase,
                "curriculum_level": completed_stage.level,
                "curriculum_stage": completed_stage.name,
                "curriculum_window_count": curriculum_result["completed_window_count"],
                "curriculum_advanced": curriculum_result["advanced"],
                "next_curriculum_stage": curriculum_result["next_stage"].name,
                "ground_box_count": completed_stage.ground_box_count,
                "aerial_sphere_count": completed_stage.aerial_sphere_count,
                "dynamic_sphere_count": completed_stage.dynamic_sphere_count,
                "inter_agent_collision": inter_agent_collision_event,
                "obstacle_collision": obstacle_collision_event,
                "boundary_collision": boundary_collision_event,
                "inter_agent_collision_steps": episode_inter_agent_collision_steps,
                "obstacle_collision_steps": episode_obstacle_collision_steps,
                "boundary_collision_steps": episode_boundary_collision_steps,
                "inter_agent_collision_agent_count": episode_inter_agent_collision_agent_count,
                "obstacle_collision_agent_count": episode_obstacle_collision_agent_count,
                "boundary_collision_agent_count": episode_boundary_collision_agent_count,
                "min_inter_agent_distance": float(info.get("min_inter_agent_distance", np.inf)),
                **{
                    key: float(np.mean(values))
                    for key, values in episode_reward_components.items()
                },
                "stagnation_trigger_steps": int(episode_stagnation_trigger_steps),
                "stagnation_max_counter": int(episode_stagnation_max_counter),
                "elapsed_sec": float(time.time() - start_time),
            }
            append_metrics(metrics_path, row)
            write_tensorboard_episode(writer, row)
            progress.event(
                (
                    "episode={episode} step={global_step} len={episode_step} "
                    "reward={mean_reward:.3f} success={success} stage={curriculum_stage} "
                    "reached={reached_agent_count}/{num_agents} "
                    "SR{window}={rolling_success_rate:.3f} "
                    "IA_collision={inter_agent_collision} "
                    "OBS_collision={obstacle_collision} "
                    "BD_collision={boundary_collision}"
                ).format(
                    num_agents=int(env.num_agents),
                    window=int(args.success_window),
                    **row,
                )
            )

            if curriculum_result["advanced"]:
                next_stage = curriculum_result["next_stage"]
                progress.event(
                    f"curriculum advanced: {completed_stage.name} -> {next_stage.name} "
                    f"(SR={rolling_success_rate:.3f})"
                )
                env.close()
                env = build_env(experiment_config, next_stage)
                if hasattr(env.action_space, "seed"):
                    env.action_space.seed(args.seed + curriculum.stage_index)

            obs_matrix, _ = env.reset()
            obs = matrix_to_agent_dict(obs_matrix, agent_ids)
            episode_reward[:] = 0.0
            for values in episode_reward_components.values():
                values[:] = 0.0
            episode_stagnation_trigger_steps = 0
            episode_stagnation_max_counter = 0
            episode_step = 0
            episode_inter_agent_collision_steps = 0
            episode_obstacle_collision_steps = 0
            episode_boundary_collision_steps = 0
            episode_inter_agent_collision_agent_count = 0
            episode_obstacle_collision_agent_count = 0
            episode_boundary_collision_agent_count = 0

        rolling_success_rate = curriculum.success_rate
        if global_step % int(args.progress_interval) == 0 or global_step == 1:
            progress.update(
                global_step=global_step,
                episode_count=episode_count,
                buffer_size=len(policy.buffers[policy.agent_x]),
                curriculum_stage=curriculum.current_stage.name,
                rolling_success_rate=rolling_success_rate,
                inter_agent_collision_count=latest_inter_agent_collision_count,
                obstacle_collision_count=latest_obstacle_collision_count,
                elapsed_sec=float(time.time() - start_time),
            )

        if global_step % int(args.log_interval) == 0:
            progress.event(
                (
                    f"step={global_step} buffer={len(policy.buffers[policy.agent_x])} "
                    f"episodes={episode_count} stage={curriculum.current_stage.name} "
                    f"SR{int(args.success_window)}={rolling_success_rate:.3f} "
                    f"IA={latest_inter_agent_collision_count} OBS={latest_obstacle_collision_count} "
                    f"elapsed={time.time() - start_time:.1f}s"
                )
            )
            write_tensorboard_step(
                writer,
                global_step=global_step,
                buffer_size=len(policy.buffers[policy.agent_x]),
                episode_count=episode_count,
                rolling_success_rate=rolling_success_rate,
                inter_agent_collision_count=latest_inter_agent_collision_count,
                obstacle_collision_count=latest_obstacle_collision_count,
                min_inter_agent_distance=latest_min_inter_agent_distance,
                curriculum_phase=curriculum.current_stage.phase,
                curriculum_level=curriculum.current_stage.level,
            )

        if global_step % int(args.eval_interval) == 0:
            eval_row = evaluate_policy(
                policy=policy,
                experiment_config=experiment_config,
                agent_ids=agent_ids,
                eval_seeds=eval_seeds,
                curriculum_stage=(
                    curriculum.current_stage
                    if curriculum.enabled or bool(args.final_stage_only)
                    else None
                ),
            )
            eval_row.update(
                {
                    "global_step": int(global_step),
                    "episode_count": int(episode_count),
                    "elapsed_sec": float(time.time() - start_time),
                    "eval_seed_base": int(args.eval_seed_base),
                    "curriculum_phase": curriculum.current_stage.phase,
                    "curriculum_level": curriculum.current_stage.level,
                    "curriculum_stage": curriculum.current_stage.name,
                    "curriculum_stage_index": curriculum.stage_index,
                }
            )
            append_metrics(eval_metrics_path, eval_row)
            write_tensorboard_eval(writer, eval_row)
            progress.event(
                (
                    f"eval step={global_step} episodes={eval_row['eval_episode_count']} "
                    f"stage={eval_row['curriculum_stage']} eval_SR={eval_row['eval_success_rate']:.3f} "
                    f"agent_SR={eval_row['eval_agent_success_rate']:.3f} "
                    f"mean_reward={eval_row['eval_mean_reward']:.3f} "
                    f"mean_len={eval_row['eval_mean_len']:.1f} "
                    f"IA_rate={eval_row['eval_inter_agent_collision_rate']:.3f}"
                )
            )
            improved = (
                int(curriculum.stage_index) > best_eval_stage_index
                or (
                    int(curriculum.stage_index) == best_eval_stage_index
                    and (
                        float(eval_row["eval_success_rate"]) > best_eval_success_rate
                        or (
                            float(eval_row["eval_success_rate"]) == best_eval_success_rate
                            and float(eval_row["eval_mean_reward"]) > best_eval_mean_reward
                        )
                    )
                )
            )
            if improved:
                best_eval_stage_index = int(curriculum.stage_index)
                best_eval_success_rate = float(eval_row["eval_success_rate"])
                best_eval_mean_reward = float(eval_row["eval_mean_reward"])
                best_dir = model_dir / "best"
                best_dir.mkdir(parents=True, exist_ok=True)
                policy.save(best_dir)
                best_payload = dict(eval_row)
                best_payload["eval_seeds"] = list(eval_seeds)
                write_json(best_dir / "best_eval.json", best_payload)
                progress.event(
                    (
                        f"best model saved: {best_dir} "
                        f"(eval_SR={best_eval_success_rate:.3f}, "
                        f"mean_reward={best_eval_mean_reward:.3f})"
                    )
                )

        if global_step % int(args.save_interval) == 0:
            checkpoint_dir = model_dir / f"step_{global_step}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            policy.save(checkpoint_dir)
            progress.event(f"checkpoint saved: {checkpoint_dir}")

    final_dir = model_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    policy.save(final_dir)
    progress.event(f"final model saved: {final_dir}")
    env.close()
    if writer is not None:
        writer.close()
    progress.close()
    return {
        "run_dir": str(run_dir),
        "metrics": str(metrics_path),
        "eval_metrics": str(eval_metrics_path),
        "model_dir": str(final_dir),
    }


def main() -> None:
    outputs = train()
    print("MASAC training finished.")
    print(f"run_dir: {outputs['run_dir']}")
    print(f"metrics: {outputs['metrics']}")
    print(f"eval_metrics: {outputs['eval_metrics']}")
    print(f"model_dir: {outputs['model_dir']}")


if __name__ == "__main__":
    main()
