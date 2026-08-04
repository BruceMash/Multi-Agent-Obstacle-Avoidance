# coding: utf-8

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
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


SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_ROOT = SCRIPT_PATH.parent
ALGO_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]
for path in (PROJECT_ROOT, ALGO_ROOT, SCRIPTS_ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from Environment.multi_agent_continuous_env import MultiAgentContinuousEnv
from MASAC.config import (
    MASAC_EXPERIMENT_CONFIG,
    MASACExperimentConfig,
    MASACNetworkConfig,
)
from MASAC.curriculum import (
    CurriculumStage,
    SuccessRateCurriculum,
    build_curriculum_stages,
    build_stage_env_kwargs,
)
from MASAC.standard_masac import StandardMASAC
from train_masac_multi_agent_dmp import (
    TerminalProgress,
    agent_dict_to_matrix,
    append_metrics,
    flatten_hparams,
    mask_count,
    matrix_to_agent_dict,
    parse_int_tuple,
    resolve_device,
    set_policy_train_mode,
    vector_to_agent_dict,
    write_tensorboard_configuration,
    write_tensorboard_episode,
    write_tensorboard_eval,
    write_tensorboard_step,
)


REWARD_COMPONENT_KEYS = (
    "reward_progress",
    "reward_obstacle_potential_penalty",
    "reward_boundary_potential_penalty",
    "reward_inter_agent_potential_penalty",
    "reward_stagnation_penalty",
    "reward_acceleration_penalty",
    "reward_acceleration_clip_penalty",
    "reward_individual_success_bonus",
    "reward_team_success_bonus",
    "reward_team_collision_penalty",
    "reward_local_collision_penalty",
    "reward_team_timeout_penalty",
)

LEARNING_DIAGNOSTIC_KEYS = (
    "critic_loss",
    "actor_loss",
    "q_replay",
    "q_policy",
    "q_target",
    "entropy",
    "alpha",
    "alpha_loss",
)

DMP_CONFIG_FIELDS = {
    "dmp_dims",
    "dmp_k_alpha",
    "dmp_k_beta",
    "k_alpha",
    "k_beta",
    "alpha_s",
    "dmp_tau",
    "forcing_term_max",
    "forcing_term_min",
    "goal_offset_max",
}


class AgentObservationHistory:
    """维护在线执行所需的逐智能体有限时序观测。"""

    def __init__(self, temporal_steps: int, agent_ids: list[str]):
        self.temporal_steps = int(temporal_steps)
        if self.temporal_steps <= 0:
            raise ValueError("temporal_steps must be positive")
        self.agent_ids = tuple(agent_ids)
        if not self.agent_ids:
            raise ValueError("agent_ids cannot be empty")
        self._histories = {
            agent_id: deque(maxlen=self.temporal_steps)
            for agent_id in self.agent_ids
        }

    def reset(self, observations: dict[str, np.ndarray]) -> None:
        for agent_id in self.agent_ids:
            self._histories[agent_id].clear()
        self.append(observations)

    def append(self, observations: dict[str, np.ndarray]) -> None:
        for agent_id in self.agent_ids:
            self._histories[agent_id].append(
                np.asarray(observations[agent_id], dtype=np.float32).copy()
            )

    def policy_inputs(
        self,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        observations = {}
        masks = {}
        for agent_id in self.agent_ids:
            frames = list(self._histories[agent_id])
            if not frames:
                raise RuntimeError("observation history must be reset before use")
            observations[agent_id] = np.stack(frames, axis=0).astype(
                np.float32,
                copy=False,
            )
            masks[agent_id] = np.ones(len(frames), dtype=bool)
        return observations, masks


def _without_dmp_fields(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in DMP_CONFIG_FIELDS
    }


def build_standard_env_kwargs(
    experiment_config: MASACExperimentConfig,
    curriculum_stage: CurriculumStage | None = None,
) -> dict[str, Any]:
    if curriculum_stage is None:
        kwargs = experiment_config.build_core_env_kwargs()
    else:
        kwargs = build_stage_env_kwargs(
            experiment_config,
            curriculum_stage,
        )
    kwargs = dict(kwargs)
    kwargs.pop("dmp_config", None)
    return kwargs


def build_env(
    experiment_config: MASACExperimentConfig,
    curriculum_stage: CurriculumStage | None = None,
) -> MultiAgentContinuousEnv:
    return MultiAgentContinuousEnv(
        **build_standard_env_kwargs(
            experiment_config,
            curriculum_stage,
        )
    )


def build_dim_info(
    env: MultiAgentContinuousEnv,
    agent_ids: list[str],
) -> dict[str, tuple[int, int]]:
    _, obs_dim = env.observation_shape
    _, action_dim = env.action_shape
    return {
        agent_id: (int(obs_dim), int(action_dim))
        for agent_id in agent_ids
    }


def build_network_config(
    env: MultiAgentContinuousEnv,
    experiment_config: MASACExperimentConfig,
    args: argparse.Namespace,
) -> MASACNetworkConfig:
    sensor = env.sensors[0]
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
        sensor_elevation_range_deg=tuple(
            float(value) for value in sensor.elevation_range_deg
        ),
        sensor_include_previous_scan=bool(sensor.include_previous_scan),
        action_low=tuple(float(value) for value in env.action_space.low[0]),
        action_high=tuple(float(value) for value in env.action_space.high[0]),
        temporal_steps=int(args.temporal_steps),
        actor_log_std_min=float(args.actor_log_std_min),
        actor_log_std_max=float(args.actor_log_std_max),
        ally_pooling=str(experiment_config.ally_pooling),
        agent_pooling=str(experiment_config.agent_pooling),
        critic_encoder=str(args.critic_encoder),
    )


def build_replay_action_matrix(
    info: dict[str, Any],
    env: MultiAgentContinuousEnv,
) -> np.ndarray:
    actions = np.asarray(
        info["applied_accelerations"],
        dtype=np.float32,
    )
    if actions.shape != env.action_shape:
        raise ValueError(
            f"replay action must have shape {env.action_shape}, "
            f"got {actions.shape}"
        )
    return actions.copy()


def make_run_dir(output_root: str, seed: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = (
        Path(output_root)
        / f"standard_masac_multi_agent_seed_{seed}_{timestamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )


def evaluate_policy(
    *,
    policy: StandardMASAC,
    experiment_config: MASACExperimentConfig,
    agent_ids: list[str],
    eval_seeds: list[int],
    curriculum_stage: CurriculumStage | None = None,
) -> dict[str, Any]:
    eval_env = build_env(experiment_config, curriculum_stage)
    previous_training_mode = next(
        iter(policy.agents.values())
    ).actor.training
    set_policy_train_mode(policy, False)
    episode_rows = []
    try:
        for eval_index, eval_seed in enumerate(eval_seeds):
            obs_matrix, _ = eval_env.reset(seed=int(eval_seed))
            obs = matrix_to_agent_dict(obs_matrix, agent_ids)
            history = AgentObservationHistory(
                policy.temporal_steps,
                agent_ids,
            )
            history.reset(obs)
            episode_reward = np.zeros(
                int(eval_env.num_agents),
                dtype=np.float64,
            )
            terminated = False
            truncated = False
            episode_step = 0
            info: dict[str, Any] = {}
            collision_events = {
                "inter_agent": False,
                "obstacle": False,
                "boundary": False,
            }
            min_inter_agent_distance = float("inf")

            while not bool(terminated or truncated):
                policy_obs, temporal_masks = history.policy_inputs()
                action = policy.evaluate_action(
                    policy_obs,
                    temporal_masks=temporal_masks,
                )
                action_matrix = np.clip(
                    agent_dict_to_matrix(action, agent_ids),
                    eval_env.action_space.low,
                    eval_env.action_space.high,
                ).astype(np.float32)
                (
                    next_obs_matrix,
                    rewards,
                    terminated,
                    truncated,
                    info,
                ) = eval_env.step(action_matrix)
                obs = matrix_to_agent_dict(next_obs_matrix, agent_ids)
                history.append(obs)
                episode_reward += np.asarray(rewards, dtype=np.float64)
                episode_step += 1
                for name in collision_events:
                    collision_events[name] = collision_events[name] or (
                        mask_count(info, f"{name}_collision_mask") > 0
                    )
                min_inter_agent_distance = min(
                    min_inter_agent_distance,
                    float(
                        info.get(
                            "min_inter_agent_distance",
                            np.inf,
                        )
                    ),
                )

            success_mask = np.asarray(
                info.get(
                    "success_mask",
                    np.zeros(int(eval_env.num_agents), dtype=bool),
                ),
                dtype=bool,
            )
            reached_agent_count = int(np.sum(success_mask))
            episode_rows.append(
                {
                    "eval_index": int(eval_index),
                    "eval_seed": int(eval_seed),
                    "success": bool(info.get("success", False)),
                    "episode_step": int(episode_step),
                    "mean_reward": float(np.mean(episode_reward)),
                    "sum_reward": float(np.sum(episode_reward)),
                    "agent_success_rate": float(
                        reached_agent_count / float(eval_env.num_agents)
                    ),
                    "inter_agent_collision": collision_events[
                        "inter_agent"
                    ],
                    "obstacle_collision": collision_events["obstacle"],
                    "boundary_collision": collision_events["boundary"],
                    "min_inter_agent_distance": float(
                        min_inter_agent_distance
                    ),
                }
            )
    finally:
        set_policy_train_mode(policy, previous_training_mode)
        eval_env.close()

    episode_count = max(1, len(episode_rows))
    return {
        "eval_episode_count": len(episode_rows),
        "eval_success_rate": float(
            np.mean([row["success"] for row in episode_rows])
        ),
        "eval_agent_success_rate": float(
            np.mean([row["agent_success_rate"] for row in episode_rows])
        ),
        "eval_mean_reward": float(
            np.mean([row["mean_reward"] for row in episode_rows])
        ),
        "eval_sum_reward": float(
            np.mean([row["sum_reward"] for row in episode_rows])
        ),
        "eval_mean_len": float(
            np.mean([row["episode_step"] for row in episode_rows])
        ),
        "eval_inter_agent_collision_rate": float(
            np.sum(
                [row["inter_agent_collision"] for row in episode_rows]
            )
            / episode_count
        ),
        "eval_obstacle_collision_rate": float(
            np.sum([row["obstacle_collision"] for row in episode_rows])
            / episode_count
        ),
        "eval_boundary_collision_rate": float(
            np.sum([row["boundary_collision"] for row in episode_rows])
            / episode_count
        ),
        "eval_min_inter_agent_distance": float(
            np.min(
                [
                    row["min_inter_agent_distance"]
                    for row in episode_rows
                ]
            )
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    config = MASAC_EXPERIMENT_CONFIG
    parser = argparse.ArgumentParser(
        description=(
            "Train Standard MASAC on MultiAgentContinuousEnv using direct "
            "3D acceleration actions."
        )
    )
    parser.add_argument("--seed", type=int, default=int(config.seed))
    parser.add_argument(
        "--total-steps",
        type=int,
        default=int(config.total_steps),
    )
    parser.add_argument(
        "--start-steps",
        type=int,
        default=int(config.start_steps),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(config.batch_size),
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=int(config.buffer_size),
    )
    parser.add_argument(
        "--actor-lr",
        type=float,
        default=float(config.actor_lr),
    )
    parser.add_argument(
        "--critic-lr",
        type=float,
        default=float(config.critic_lr),
    )
    parser.add_argument("--gamma", type=float, default=float(config.gamma))
    parser.add_argument(
        "--tau",
        type=float,
        default=float(config.soft_update_tau),
    )
    parser.add_argument(
        "--learn-interval",
        type=int,
        default=int(config.learn_interval),
    )
    parser.add_argument(
        "--updates-per-step",
        type=int,
        default=int(config.updates_per_step),
    )
    parser.add_argument(
        "--temporal-steps",
        type=int,
        default=int(config.temporal_steps),
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=int(config.hidden_dim),
    )
    parser.add_argument(
        "--sensor-hidden-dim",
        type=int,
        default=int(config.sensor_hidden_dim),
    )
    parser.add_argument(
        "--ally-hidden-dim",
        type=int,
        default=int(config.ally_hidden_dim),
    )
    parser.add_argument(
        "--sensor-output-dim",
        type=int,
        default=int(config.sensor_output_dim),
    )
    parser.add_argument(
        "--ally-output-dim",
        type=int,
        default=int(config.ally_output_dim),
    )
    parser.add_argument(
        "--num-sensor-layers",
        type=int,
        default=int(config.num_sensor_layers),
    )
    parser.add_argument(
        "--num-ally-layers",
        type=int,
        default=int(config.num_ally_layers),
    )
    parser.add_argument(
        "--num-observation-layers",
        type=int,
        default=int(config.num_observation_layers),
    )
    parser.add_argument(
        "--actor-log-std-min",
        type=float,
        default=float(config.actor_log_std_min),
    )
    parser.add_argument(
        "--actor-log-std-max",
        type=float,
        default=float(config.actor_log_std_max),
    )
    parser.add_argument(
        "--critic-encoder",
        choices=("attention", "mlp"),
        default=str(config.critic_encoder),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=int(config.max_steps),
    )
    parser.add_argument(
        "--sensor-azimuth-bins",
        type=int,
        default=int(config.sensor_azimuth_bins),
    )
    parser.add_argument(
        "--sensor-elevation-bins",
        type=int,
        default=int(config.sensor_elevation_bins),
    )
    parser.add_argument("--device", default=str(config.device))
    parser.add_argument(
        "--output-root",
        default="artifacts/standard_masac",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=int(config.save_interval),
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=int(config.save_interval),
    )
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument(
        "--eval-seed-base",
        type=int,
        default=100_000,
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=int(config.log_interval),
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=int(config.progress_interval),
    )
    parser.add_argument(
        "--success-window",
        type=int,
        default=int(config.success_window),
    )
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
    scenario_group = parser.add_mutually_exclusive_group()
    scenario_group.add_argument(
        "--disable-curriculum",
        action="store_true",
        default=not bool(config.curriculum_enabled),
    )
    scenario_group.add_argument(
        "--final-stage-only",
        action="store_true",
    )
    parser.add_argument(
        "--disable-tensorboard",
        action="store_true",
        default=bool(config.disable_tensorboard),
    )
    parser.add_argument(
        "--plain-progress",
        action="store_true",
        default=bool(config.plain_progress),
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    positive_integer_names = (
        "total_steps",
        "batch_size",
        "buffer_size",
        "learn_interval",
        "updates_per_step",
        "temporal_steps",
        "max_steps",
        "sensor_azimuth_bins",
        "sensor_elevation_bins",
        "save_interval",
        "eval_interval",
        "eval_episodes",
        "log_interval",
        "progress_interval",
        "success_window",
    )
    for name in positive_integer_names:
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    if int(args.start_steps) < 0:
        raise ValueError("start_steps must be non-negative")
    if int(args.buffer_size) < int(args.batch_size):
        raise ValueError("buffer_size must be at least batch_size")
    if not 0.0 <= float(args.curriculum_success_threshold) <= 1.0:
        raise ValueError(
            "curriculum success threshold must be in [0, 1]"
        )


def _build_experiment_config(
    args: argparse.Namespace,
) -> MASACExperimentConfig:
    return replace(
        MASAC_EXPERIMENT_CONFIG,
        environment_name="multi_agent_continuous",
        hyperparam_source="standard_masac",
        max_steps=int(args.max_steps),
        sensor_azimuth_bins=int(args.sensor_azimuth_bins),
        sensor_elevation_bins=int(args.sensor_elevation_bins),
        curriculum_enabled=not bool(
            args.disable_curriculum or args.final_stage_only
        ),
        curriculum_success_threshold=float(
            args.curriculum_success_threshold
        ),
        curriculum_phase2_box_counts=tuple(args.phase2_box_counts),
        curriculum_phase2_sphere_counts=tuple(
            args.phase2_sphere_counts
        ),
        curriculum_phase3_dynamic_counts=tuple(
            args.phase3_dynamic_counts
        ),
        action_guidance_enabled=False,
    )


def _write_learning_diagnostics(
    writer,
    diagnostics: dict[str, float] | None,
    global_step: int,
) -> None:
    if writer is None or diagnostics is None:
        return
    for key, value in diagnostics.items():
        writer.add_scalar(f"learn/{key}", float(value), global_step)


def train(
    args: argparse.Namespace | None = None,
) -> dict[str, str]:
    args = parse_args() if args is None else args
    _validate_args(args)
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    experiment_config = _build_experiment_config(args)
    stages = build_curriculum_stages(
        args.phase2_box_counts,
        args.phase2_sphere_counts,
        args.phase3_dynamic_counts,
    )
    curriculum = SuccessRateCurriculum(
        stages,
        success_threshold=float(args.curriculum_success_threshold),
        success_window=int(args.success_window),
        enabled=not bool(
            args.disable_curriculum or args.final_stage_only
        ),
        initial_stage_index=(
            len(stages) - 1 if args.final_stage_only else 0
        ),
    )
    active_stage = (
        curriculum.current_stage
        if curriculum.enabled or bool(args.final_stage_only)
        else None
    )
    env = build_env(experiment_config, active_stage)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(int(args.seed))

    agent_ids = [
        f"agent_{index}" for index in range(int(env.num_agents))
    ]
    dim_info = build_dim_info(env, agent_ids)
    network_config = build_network_config(
        env,
        experiment_config,
        args,
    )
    device = resolve_device(args.device)
    policy = StandardMASAC(
        dim_info=dim_info,
        is_continue=True,
        actor_lr=float(args.actor_lr),
        critic_lr=float(args.critic_lr),
        buffer_size=int(args.buffer_size),
        device=device,
        network_config=network_config,
    )

    run_dir = make_run_dir(args.output_root, int(args.seed))
    model_dir = run_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    eval_metrics_path = run_dir / "eval_metrics.csv"
    eval_seeds = [
        int(args.eval_seed_base) + index
        for index in range(int(args.eval_episodes))
    ]
    progress = TerminalProgress(
        total_steps=int(args.total_steps),
        success_window=int(args.success_window),
        plain=bool(args.plain_progress),
    )
    writer = None
    if (
        not bool(args.disable_tensorboard)
        and SummaryWriter is not None
    ):
        writer = SummaryWriter(str(run_dir / "tensorboard"))

    experiment_payload = _without_dmp_fields(
        asdict(experiment_config)
    )
    network_payload = _without_dmp_fields(asdict(network_config))
    run_config = {
        "script_args": vars(args),
        "experiment_config": experiment_payload,
        "core_env_kwargs": build_standard_env_kwargs(
            experiment_config
        ),
        "dim_info": dim_info,
        "network_config": network_payload,
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

    obs_matrix, _ = env.reset(seed=int(args.seed))
    obs = matrix_to_agent_dict(obs_matrix, agent_ids)
    history = AgentObservationHistory(
        int(args.temporal_steps),
        agent_ids,
    )
    history.reset(obs)
    episode_reward = np.zeros(env.num_agents, dtype=np.float64)
    episode_reward_components = {
        key: np.zeros(env.num_agents, dtype=np.float64)
        for key in REWARD_COMPONENT_KEYS
    }
    episode_count = 0
    episode_step = 0
    collision_step_counts = {
        "inter_agent": 0,
        "obstacle": 0,
        "boundary": 0,
    }
    collision_agent_counts = {
        "inter_agent": 0,
        "obstacle": 0,
        "boundary": 0,
    }
    episode_acceleration_norm_sum = 0.0
    episode_acceleration_sample_count = 0
    episode_clip_count = 0
    episode_clip_sample_count = 0
    episode_stagnation_trigger_steps = 0
    episode_stagnation_max_counter = 0
    latest_diagnostics: dict[str, float] | None = None
    latest_min_inter_agent_distance = float("inf")
    best_eval_stage_index = -1
    best_eval_success_rate = -1.0
    best_eval_mean_reward = -float("inf")
    start_time = time.time()
    min_learn_size = max(
        int(args.batch_size),
        int(args.temporal_steps),
    )

    try:
        for global_step in range(1, int(args.total_steps) + 1):
            if global_step <= int(args.start_steps):
                action_matrix = env.action_space.sample().astype(
                    np.float32
                )
            else:
                policy_obs, temporal_masks = history.policy_inputs()
                action = policy.select_action(
                    policy_obs,
                    temporal_masks=temporal_masks,
                )
                action_matrix = np.clip(
                    agent_dict_to_matrix(action, agent_ids),
                    env.action_space.low,
                    env.action_space.high,
                ).astype(np.float32)

            (
                next_obs_matrix,
                rewards,
                terminated,
                truncated,
                info,
            ) = env.step(action_matrix)
            next_obs = matrix_to_agent_dict(
                next_obs_matrix,
                agent_ids,
            )
            reward = vector_to_agent_dict(rewards, agent_ids)
            replay_action = matrix_to_agent_dict(
                build_replay_action_matrix(info, env),
                agent_ids,
            )
            done_for_buffer = {
                agent_id: bool(terminated)
                for agent_id in agent_ids
            }
            episode_end_for_buffer = {
                agent_id: bool(terminated or truncated)
                for agent_id in agent_ids
            }
            policy.add(
                obs,
                replay_action,
                reward,
                next_obs,
                done_for_buffer,
                episode_end=episode_end_for_buffer,
            )
            history.append(next_obs)

            episode_reward += np.asarray(rewards, dtype=np.float64)
            episode_step += 1
            for key in REWARD_COMPONENT_KEYS:
                episode_reward_components[key] += np.asarray(
                    info.get(key, np.zeros(env.num_agents)),
                    dtype=np.float64,
                )
            for name in collision_step_counts:
                count = mask_count(info, f"{name}_collision_mask")
                collision_step_counts[name] += int(count > 0)
                collision_agent_counts[name] += count
            latest_min_inter_agent_distance = float(
                info.get("min_inter_agent_distance", np.inf)
            )
            applied_accelerations = np.asarray(
                info["applied_accelerations"],
                dtype=np.float64,
            )
            episode_acceleration_norm_sum += float(
                np.sum(np.linalg.norm(applied_accelerations, axis=1))
            )
            episode_acceleration_sample_count += int(env.num_agents)
            clip_mask = np.asarray(
                info["acceleration_clip_mask"],
                dtype=bool,
            )
            episode_clip_count += int(np.sum(clip_mask))
            episode_clip_sample_count += int(clip_mask.size)
            stagnation_mask = np.asarray(
                info.get(
                    "stagnation_mask",
                    np.zeros(env.num_agents, dtype=bool),
                ),
                dtype=bool,
            )
            episode_stagnation_trigger_steps += int(
                np.any(stagnation_mask)
            )
            stagnation_counters = np.asarray(
                info.get(
                    "stagnation_counters",
                    np.zeros(env.num_agents, dtype=np.int32),
                )
            )
            if stagnation_counters.size:
                episode_stagnation_max_counter = max(
                    episode_stagnation_max_counter,
                    int(np.max(stagnation_counters)),
                )

            can_learn = (
                len(policy.buffers[policy.agent_x]) >= min_learn_size
            )
            if (
                can_learn
                and global_step > int(args.start_steps)
                and global_step % int(args.learn_interval) == 0
            ):
                for _ in range(int(args.updates_per_step)):
                    latest_diagnostics = policy.learn(
                        batch_size=int(args.batch_size),
                        gamma=float(args.gamma),
                        tau=float(args.tau),
                    )
                _write_learning_diagnostics(
                    writer,
                    latest_diagnostics,
                    global_step,
                )

            obs = next_obs
            episode_done = bool(terminated or truncated)
            if episode_done:
                episode_count += 1
                success_mask = np.asarray(
                    info.get(
                        "success_mask",
                        np.zeros(env.num_agents, dtype=bool),
                    ),
                    dtype=bool,
                )
                success = bool(info.get("success", False))
                curriculum_result = curriculum.record_episode(success)
                completed_stage = curriculum_result["completed_stage"]
                rolling_success_rate = float(
                    curriculum_result["completed_success_rate"]
                )
                reached_agent_count = int(np.sum(success_mask))
                row = {
                    "global_step": int(global_step),
                    "episode": int(episode_count),
                    "episode_step": int(episode_step),
                    "mean_reward": float(np.mean(episode_reward)),
                    "sum_reward": float(np.sum(episode_reward)),
                    "min_reward": float(np.min(episode_reward)),
                    "max_reward": float(np.max(episode_reward)),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "collision": bool(info.get("collision", False)),
                    "success": success,
                    "reached_agent_count": reached_agent_count,
                    "agent_success_rate": float(
                        reached_agent_count / float(env.num_agents)
                    ),
                    "rolling_success_rate": rolling_success_rate,
                    "curriculum_phase": completed_stage.phase,
                    "curriculum_level": completed_stage.level,
                    "curriculum_stage": completed_stage.name,
                    "curriculum_window_count": (
                        curriculum_result["completed_window_count"]
                    ),
                    "curriculum_advanced": (
                        curriculum_result["advanced"]
                    ),
                    "next_curriculum_stage": (
                        curriculum_result["next_stage"].name
                    ),
                    "ground_box_count": (
                        completed_stage.ground_box_count
                    ),
                    "aerial_sphere_count": (
                        completed_stage.aerial_sphere_count
                    ),
                    "dynamic_sphere_count": (
                        completed_stage.dynamic_sphere_count
                    ),
                    "inter_agent_collision": (
                        collision_step_counts["inter_agent"] > 0
                    ),
                    "obstacle_collision": (
                        collision_step_counts["obstacle"] > 0
                    ),
                    "boundary_collision": (
                        collision_step_counts["boundary"] > 0
                    ),
                    "inter_agent_collision_steps": (
                        collision_step_counts["inter_agent"]
                    ),
                    "obstacle_collision_steps": (
                        collision_step_counts["obstacle"]
                    ),
                    "boundary_collision_steps": (
                        collision_step_counts["boundary"]
                    ),
                    "inter_agent_collision_agent_count": (
                        collision_agent_counts["inter_agent"]
                    ),
                    "obstacle_collision_agent_count": (
                        collision_agent_counts["obstacle"]
                    ),
                    "boundary_collision_agent_count": (
                        collision_agent_counts["boundary"]
                    ),
                    "min_inter_agent_distance": (
                        latest_min_inter_agent_distance
                    ),
                    "mean_applied_acceleration_norm": (
                        episode_acceleration_norm_sum
                        / max(episode_acceleration_sample_count, 1)
                    ),
                    "acceleration_clip_fraction": (
                        episode_clip_count
                        / max(episode_clip_sample_count, 1)
                    ),
                    **{
                        key: float(np.mean(values))
                        for key, values
                        in episode_reward_components.items()
                    },
                    "stagnation_trigger_steps": int(
                        episode_stagnation_trigger_steps
                    ),
                    "stagnation_max_counter": int(
                        episode_stagnation_max_counter
                    ),
                    "elapsed_sec": float(time.time() - start_time),
                }
                row.update(
                    {
                        f"learn_{key}": float(
                            latest_diagnostics[key]
                        )
                        if latest_diagnostics is not None
                        else float("nan")
                        for key in LEARNING_DIAGNOSTIC_KEYS
                    }
                )
                append_metrics(metrics_path, row)
                write_tensorboard_episode(writer, row)
                if writer is not None:
                    writer.add_scalar(
                        "action/mean_applied_acceleration_norm",
                        row["mean_applied_acceleration_norm"],
                        global_step,
                    )
                    writer.add_scalar(
                        "action/clip_fraction",
                        row["acceleration_clip_fraction"],
                        global_step,
                    )

                if curriculum_result["advanced"]:
                    env.close()
                    env = build_env(
                        experiment_config,
                        curriculum_result["next_stage"],
                    )
                    if hasattr(env.action_space, "seed"):
                        env.action_space.seed(
                            int(args.seed) + curriculum.stage_index
                        )

                obs_matrix, _ = env.reset()
                obs = matrix_to_agent_dict(obs_matrix, agent_ids)
                history.reset(obs)
                episode_reward[:] = 0.0
                for values in episode_reward_components.values():
                    values[:] = 0.0
                episode_step = 0
                collision_step_counts = {
                    key: 0 for key in collision_step_counts
                }
                collision_agent_counts = {
                    key: 0 for key in collision_agent_counts
                }
                episode_acceleration_norm_sum = 0.0
                episode_acceleration_sample_count = 0
                episode_clip_count = 0
                episode_clip_sample_count = 0
                episode_stagnation_trigger_steps = 0
                episode_stagnation_max_counter = 0

            if (
                global_step % int(args.progress_interval) == 0
                or global_step == 1
            ):
                progress.update(
                    global_step=global_step,
                    episode_count=episode_count,
                    buffer_size=len(
                        policy.buffers[policy.agent_x]
                    ),
                    curriculum_stage=(
                        curriculum.current_stage.name
                    ),
                    rolling_success_rate=curriculum.success_rate,
                    inter_agent_collision_count=mask_count(
                        info,
                        "inter_agent_collision_mask",
                    ),
                    obstacle_collision_count=mask_count(
                        info,
                        "obstacle_collision_mask",
                    ),
                    elapsed_sec=float(time.time() - start_time),
                )

            if global_step % int(args.log_interval) == 0:
                write_tensorboard_step(
                    writer,
                    global_step=global_step,
                    buffer_size=len(
                        policy.buffers[policy.agent_x]
                    ),
                    episode_count=episode_count,
                    rolling_success_rate=curriculum.success_rate,
                    inter_agent_collision_count=mask_count(
                        info,
                        "inter_agent_collision_mask",
                    ),
                    obstacle_collision_count=mask_count(
                        info,
                        "obstacle_collision_mask",
                    ),
                    min_inter_agent_distance=(
                        latest_min_inter_agent_distance
                    ),
                    curriculum_phase=(
                        curriculum.current_stage.phase
                    ),
                    curriculum_level=(
                        curriculum.current_stage.level
                    ),
                )

            if global_step % int(args.eval_interval) == 0:
                eval_row = evaluate_policy(
                    policy=policy,
                    experiment_config=experiment_config,
                    agent_ids=agent_ids,
                    eval_seeds=eval_seeds,
                    curriculum_stage=(
                        curriculum.current_stage
                        if curriculum.enabled
                        or bool(args.final_stage_only)
                        else None
                    ),
                )
                eval_row.update(
                    {
                        "global_step": int(global_step),
                        "episode_count": int(episode_count),
                        "elapsed_sec": float(
                            time.time() - start_time
                        ),
                        "eval_seed_base": int(args.eval_seed_base),
                        "curriculum_phase": (
                            curriculum.current_stage.phase
                        ),
                        "curriculum_level": (
                            curriculum.current_stage.level
                        ),
                        "curriculum_stage": (
                            curriculum.current_stage.name
                        ),
                        "curriculum_stage_index": (
                            curriculum.stage_index
                        ),
                    }
                )
                append_metrics(eval_metrics_path, eval_row)
                write_tensorboard_eval(writer, eval_row)
                improved = (
                    curriculum.stage_index > best_eval_stage_index
                    or (
                        curriculum.stage_index
                        == best_eval_stage_index
                        and (
                            eval_row["eval_success_rate"]
                            > best_eval_success_rate
                            or (
                                eval_row["eval_success_rate"]
                                == best_eval_success_rate
                                and eval_row["eval_mean_reward"]
                                > best_eval_mean_reward
                            )
                        )
                    )
                )
                if improved:
                    best_eval_stage_index = curriculum.stage_index
                    best_eval_success_rate = float(
                        eval_row["eval_success_rate"]
                    )
                    best_eval_mean_reward = float(
                        eval_row["eval_mean_reward"]
                    )
                    best_dir = model_dir / "best"
                    best_dir.mkdir(parents=True, exist_ok=True)
                    policy.save(best_dir)
                    best_payload = dict(eval_row)
                    best_payload["eval_seeds"] = list(eval_seeds)
                    write_json(
                        best_dir / "best_eval.json",
                        best_payload,
                    )

            if global_step % int(args.save_interval) == 0:
                checkpoint_dir = (
                    model_dir / f"step_{global_step}"
                )
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                policy.save(checkpoint_dir)

        final_dir = model_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        policy.save(final_dir)
    finally:
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
    print("Standard MASAC training finished.")
    print(f"run_dir: {outputs['run_dir']}")
    print(f"metrics: {outputs['metrics']}")
    print(f"eval_metrics: {outputs['eval_metrics']}")
    print(f"model_dir: {outputs['model_dir']}")


if __name__ == "__main__":
    main()
