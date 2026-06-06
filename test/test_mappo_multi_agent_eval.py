"""MAPPO multi-agent DMP checkpoint evaluation.

This script restores a MAPPO checkpoint, evaluates it on fixed and seed-defined
multi-agent scenes, and writes data-driven HTML trajectory animations.
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import math
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class Scenario:
    name: str
    starts: np.ndarray
    goals: np.ndarray
    source: str


@dataclass
class RolloutResult:
    name: str
    source: str
    steps: int
    terminal_status: str
    all_reached: bool
    reached_agent_count: int
    final_distance_mean: float
    final_distance_max: float
    min_distance_mean: float
    min_pairwise_distance: float
    min_boundary_distance: float
    first_inter_agent_collision_step: int | None
    first_boundary_collision_step: int | None
    first_obstacle_collision_step: int | None
    obstacle_collision_rate: float
    inter_agent_collision_rate: float
    boundary_collision_rate: float
    timeout: bool
    action_norm_mean: float
    forcing_norm_mean: float
    goal_offset_norm_mean: float
    applied_acceleration_norm_mean: float
    acceleration_clip_ratio: float
    total_reward: float
    starts: np.ndarray
    goals: np.ndarray
    positions: np.ndarray
    distances: np.ndarray
    pairwise_min_distances: np.ndarray
    boundary_min_distances: np.ndarray
    action_norms: np.ndarray
    forcing_norms: np.ndarray
    goal_offset_norms: np.ndarray
    applied_acceleration_norms: np.ndarray


def _repo_root() -> Path:
    return REPO_ROOT


def _load_mappo_config():
    try:
        from experiment_config import MAPPO_EXPERIMENT_CONFIG

        return MAPPO_EXPERIMENT_CONFIG, "import"
    except ModuleNotFoundError:
        config_path = _repo_root() / "experiment_config.py"
        tree = ast.parse(config_path.read_text(encoding="utf-8"))
        values: dict[str, Any] = {}
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "MAPPOExperimentConfig":
                for item in node.body:
                    if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name) and item.value is not None:
                        try:
                            values[item.target.id] = ast.literal_eval(item.value)
                        except ValueError:
                            continue
                break
        if not values:
            raise RuntimeError("failed to read MAPPOExperimentConfig defaults from experiment_config.py")
        return SimpleNamespace(**values), "static"


def _build_dynamics_config(config) -> dict[str, Any]:
    return {
        "velocity_clip": config.velocity_clip,
        "accelerate_clip": config.accelerate_clip,
        "time_step": config.time_step,
    }


def _build_sensor_config(config) -> dict[str, Any]:
    sensor_config: dict[str, Any] = {
        "sensing_radius": config.sensing_radius,
        "azimuth_bins": config.sensor_azimuth_bins,
        "elevation_bins": config.sensor_elevation_bins,
        "elevation_range_deg": config.sensor_elevation_range_deg,
    }
    if getattr(config, "sensor_goal_distance_clip", None) is not None:
        sensor_config["goal_distance_clip"] = config.sensor_goal_distance_clip
    return sensor_config


def _build_dmp_config(config) -> dict[str, Any]:
    return {
        "dt": config.time_step,
        "dims": config.dmp_dims,
        "K_alpha": config.k_alpha,
        "K_beta": config.k_beta,
        "alpha_s": config.alpha_s,
        "tau": config.tau,
        "forcing_term_max": config.forcing_term_max,
        "forcing_term_min": config.forcing_term_min,
        "goal_offset_max": config.goal_offset_max,
    }


def _build_env_config(config) -> dict[str, Any]:
    return {
        "num_agents": config.num_agents,
        "max_steps": config.max_steps,
        "goal_tolerance": config.goal_tolerance,
        "obstacle_potential_weight": config.obstacle_potential_weight,
        "obstacle_influence_distance": config.obstacle_influence_distance,
        "obstacle_potential_penalty_max": config.obstacle_potential_penalty_max,
        "step_reward_weight": config.step_reward_weight,
        "step_penalty": config.step_penalty,
        "collision_penalty": config.collision_penalty,
        "timeout_penalty": config.timeout_penalty,
        "success_bonus": config.success_bonus,
        "collision_margin": config.collision_margin,
        "workspace_bounds": config.workspace_bounds,
        "randomize_start_goal": config.randomize_start_goal,
        "start_position_bounds": config.start_position_bounds,
        "goal_position_bounds": config.goal_position_bounds,
        "min_start_distance": config.min_start_distance,
        "min_goal_distance": config.min_goal_distance,
        "min_start_goal_distance": config.min_start_goal_distance,
        "start_goal_max_attempts": config.start_goal_max_attempts,
        "boundary_influence_distance": config.boundary_influence_distance,
        "boundary_potential_weight": config.boundary_potential_weight,
        "boundary_potential_penalty_max": config.boundary_potential_penalty_max,
        "boundary_distance_epsilon": config.boundary_distance_epsilon,
        "action_guidance_enabled": config.action_guidance_enabled,
        "action_guidance_radius": config.action_guidance_radius,
        "action_guidance_initial_weight": config.action_guidance_initial_weight,
        "action_guidance_decay_steps": config.action_guidance_decay_steps,
        "near_goal_bonus_radius_1": config.near_goal_bonus_radius_1,
        "near_goal_bonus_1": config.near_goal_bonus_1,
        "near_goal_bonus_radius_2": config.near_goal_bonus_radius_2,
        "near_goal_bonus_2": config.near_goal_bonus_2,
        "inter_agent_safe_distance": config.inter_agent_safe_distance,
        "inter_agent_collision_penalty": config.inter_agent_collision_penalty,
        "inter_agent_potential_weight": config.inter_agent_potential_weight,
        "inter_agent_influence_distance": config.inter_agent_influence_distance,
        "nearest_agent_observation_count": config.nearest_agent_observation_count,
        "acceleration_penalty_weight": config.acceleration_penalty_weight,
        "acceleration_clip_penalty_weight": config.acceleration_clip_penalty_weight,
    }


def _build_core_env_kwargs(config) -> dict[str, Any]:
    if hasattr(config, "build_core_env_kwargs"):
        return copy.deepcopy(config.build_core_env_kwargs())
    return {
        "dynamics_config": _build_dynamics_config(config),
        "sensor_config": _build_sensor_config(config),
        "dmp_config": _build_dmp_config(config),
        "env_config": _build_env_config(config),
    }


def _build_fixed_eval_scenarios(config) -> list[Scenario]:
    if hasattr(config, "build_fixed_eval_scenarios"):
        rows = config.build_fixed_eval_scenarios()
        return [
            Scenario(
                name=str(row["name"]),
                starts=np.asarray(row["starts"], dtype=float),
                goals=np.asarray(row["goals"], dtype=float),
                source="fixed",
            )
            for row in rows
        ]

    bounds = np.asarray(config.workspace_bounds, dtype=float)
    lower, upper = bounds
    x_start = float(np.clip(0.0, lower[0] + 0.4, upper[0] - 0.4))
    x_goal = float(np.clip(8.0, lower[0] + 0.4, upper[0] - 0.4))
    y_margin = min(0.35, max(0.0, 0.2 * float(upper[1] - lower[1])))
    y_values = np.linspace(lower[1] + y_margin, upper[1] - y_margin, int(config.num_agents), dtype=float)
    z_margin = min(0.25, max(0.0, 0.2 * float(upper[2] - lower[2])))
    z_abs = max(0.0, min(0.55, 0.5 * float(upper[2] - lower[2]) - z_margin))
    z_values = np.array([(-1.0 if index % 2 == 0 else 1.0) * z_abs for index in range(int(config.num_agents))])
    starts = np.stack([np.full(int(config.num_agents), x_start), y_values, z_values], axis=1)
    goals = np.stack([np.full(int(config.num_agents), x_goal), y_values, z_values], axis=1)
    rows = [
        ("parallel_layered", starts, goals),
        ("height_swap", starts, np.column_stack([goals[:, 0], goals[:, 1], -goals[:, 2]])),
        ("lane_swap", starts, goals[::-1]),
    ]
    return [
        Scenario(name=name, starts=start_points, goals=goal_points, source="fixed")
        for name, start_points, goal_points in rows[: max(0, int(config.fixed_eval_num_scenarios))]
    ]


def _sample_spaced_points(
    rng: np.random.Generator,
    lower: np.ndarray,
    upper: np.ndarray,
    count: int,
    min_distance: float,
    max_attempts: int,
) -> np.ndarray | None:
    points: list[np.ndarray] = []
    for _ in range(count):
        accepted = None
        for _ in range(max_attempts):
            candidate = rng.uniform(lower, upper)
            if all(float(np.linalg.norm(candidate - point)) >= min_distance for point in points):
                accepted = candidate.copy()
                break
        if accepted is None:
            return None
        points.append(accepted)
    return np.stack(points, axis=0)


def _build_random_scenarios(config, count: int, seed: int, scenario_sample_attempts: int) -> list[Scenario]:
    rng = np.random.default_rng(seed)
    scenarios: list[Scenario] = []
    start_bounds = np.asarray(config.start_position_bounds, dtype=float)
    goal_bounds = np.asarray(config.goal_position_bounds, dtype=float)
    start_spacing = max(float(config.min_start_distance), float(config.inter_agent_safe_distance))
    goal_spacing = max(
        float(config.min_goal_distance),
        float(config.inter_agent_safe_distance) + 2.0 * float(config.goal_tolerance),
    )
    min_start_goal_distance = float(config.min_start_goal_distance)
    max_attempts = int(config.start_goal_max_attempts)

    attempts_per_group = max(max_attempts, int(scenario_sample_attempts))

    for scenario_index in range(max(0, int(count))):
        for _ in range(attempts_per_group):
            starts = _sample_spaced_points(
                rng,
                start_bounds[0],
                start_bounds[1],
                int(config.num_agents),
                start_spacing,
                max_attempts,
            )
            if starts is None:
                continue
            goals = _sample_spaced_points(
                rng,
                goal_bounds[0],
                goal_bounds[1],
                int(config.num_agents),
                goal_spacing,
                max_attempts,
            )
            if goals is None:
                continue
            if np.all(np.linalg.norm(goals - starts, axis=1) >= min_start_goal_distance):
                scenarios.append(
                    Scenario(
                        name=f"random_{scenario_index:03d}",
                        starts=starts,
                        goals=goals,
                        source="random",
                    )
                )
                break
        else:
            raise RuntimeError(
                f"failed to sample random scenario {scenario_index} after {attempts_per_group} scene attempts"
            )
    return scenarios


def _build_algo_args(config) -> dict[str, Any]:
    if hasattr(config, "build_algo_args"):
        return copy.deepcopy(config.build_algo_args())
    return {
        "use_gae": config.use_gae,
        "lambda": config.gae_lambda,
        "kl_coeff": config.kl_coeff,
        "batch_episode": config.batch_episode,
        "num_sgd_iter": config.num_sgd_iter,
        "vf_loss_coeff": config.vf_loss_coeff,
        "lr": config.learning_rate,
        "entropy_coeff": config.entropy_coeff,
        "clip_param": config.clip_param,
        "vf_clip_param": config.vf_clip_param,
        "batch_mode": config.batch_mode,
    }


def _build_model_preference(config) -> dict[str, Any]:
    if hasattr(config, "build_model_preference"):
        return copy.deepcopy(config.build_model_preference())
    return {
        "core_arch": config.model_core_arch,
        "hidden_dim": config.hidden_dim,
        "sensor_output_dim": config.sensor_output_dim,
        "num_sensor_layers": config.num_sensor_layers,
        "num_observation_layers": config.num_observation_layers,
        "actor_log_std_min": config.actor_log_std_min,
        "actor_log_std_max": config.actor_log_std_max,
    }


def _checkpoint_sort_key(path: Path) -> tuple[int, float, str]:
    text = path.name
    if path.is_dir():
        text = path.name
    number = -1
    for token in text.replace("-", "_").split("_"):
        if token.isdigit():
            number = max(number, int(token))
    return number, path.stat().st_mtime, str(path)


def _resolve_checkpoint_path(checkpoint: str | Path) -> Path:
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.is_dir():
        files = [path for path in checkpoint_path.iterdir() if path.name.startswith("checkpoint-")]
        files = [path for path in files if not path.name.endswith(".tune_metadata")]
        if files:
            return sorted(files, key=_checkpoint_sort_key)[-1]
    return checkpoint_path


def _find_latest_checkpoint(output_root: Path) -> Path:
    candidates = [path for path in output_root.rglob("checkpoint_*") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"no checkpoint_* directories found under {output_root}")
    return _resolve_checkpoint_path(sorted(candidates, key=_checkpoint_sort_key)[-1])


def _build_core_env(config):
    core_env_kwargs = _build_core_env_kwargs(config)
    from Controller.dmp_rl import DMPConfig
    from Environment.multi_agent_dmp_env import MultiAgentDMPEnv, MultiAgentEnvConfig

    core_kwargs = copy.deepcopy(core_env_kwargs)
    if isinstance(core_kwargs.get("env_config"), dict):
        core_kwargs["env_config"] = MultiAgentEnvConfig(**core_kwargs["env_config"])
    if isinstance(core_kwargs.get("dmp_config"), dict):
        core_kwargs["dmp_config"] = DMPConfig(**core_kwargs["dmp_config"])
    return MultiAgentDMPEnv(**core_kwargs)


def _build_run_config(exp_info: dict[str, Any], env_info: dict[str, Any], config, num_workers: int) -> dict[str, Any]:
    map_name = exp_info["env_args"]["map_name"]
    policy_mapping_info = env_info["policy_mapping_info"]
    if "all_scenario" in policy_mapping_info:
        policy_mapping_info = policy_mapping_info["all_scenario"]
    else:
        policy_mapping_info = policy_mapping_info[map_name]

    shared_policy_name = "default_policy" if exp_info["agent_level_batch_update"] else "shared_policy"
    if config.share_policy != "all":
        raise ValueError("this evaluation script currently expects share_policy='all'")
    if not policy_mapping_info["all_agents_one_policy"]:
        raise ValueError(f"policy can not be shared in map {map_name}")

    policies = {shared_policy_name}
    policy_mapping_fn = lambda agent_id, episode, **kwargs: shared_policy_name
    return {
        "seed": int(config.seed),
        "env": exp_info["env"] + "_" + map_name,
        "num_gpus_per_worker": 0,
        "num_gpus": 0,
        "num_workers": int(num_workers),
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn,
        },
        "framework": config.framework,
        "evaluation_interval": None,
        "simple_optimizer": False,
    }


def _build_trainer_config(config, num_workers: int) -> dict[str, Any]:
    import MARL as marl
    from MARL import _Algo
    from MARL.algos.utils.setup_utils import AlgVar
    from net.net_mappo import DMPMAPPOModel
    from ray.rllib.models import ModelCatalog
    from ray.tune.utils import merge_dicts

    env_tuple = marl.make_env(
        config.environment_name,
        config.map_name,
        core_env_kwargs=_build_core_env_kwargs(config),
    )
    env_instance, exp_info = env_tuple
    algo = _Algo("mappo")(config.hyperparam_source, **_build_algo_args(config))
    _, model_config = marl.build_model(env_tuple, algo, _build_model_preference(config))
    env_info = env_instance.get_env_info()
    env_info["agent_name_ls"] = list(env_instance.agents)
    env_instance.close()

    from marllib.marl.common import recursive_dict_update

    exp_info = recursive_dict_update(exp_info, model_config)
    exp_info = recursive_dict_update(exp_info, algo.algo_parameters)
    running_params = {
        "share_policy": config.share_policy,
        "evaluation_interval": config.evaluation_interval,
        "framework": config.framework,
        "local_mode": True,
        "num_gpus": 0,
        "num_gpus_per_worker": 0,
        "checkpoint_end": False,
        "checkpoint_freq": 0,
        "max_failures": 1,
        "restore_path": {"model_path": "", "params_path": ""},
        "stop_reward": config.stop_reward,
        "stop_timesteps": config.stop_timesteps,
        "stop_iters": config.training_iteration,
        "seed": config.seed,
        "local_dir": "",
    }
    if getattr(config, "fixed_batch_timesteps", None) is not None:
        running_params["fixed_batch_timesteps"] = int(config.fixed_batch_timesteps)
    exp_info = recursive_dict_update(exp_info, running_params)
    exp_info["algorithm"] = "mappo"

    run_config = _build_run_config(exp_info, env_info, config, num_workers)
    param = AlgVar(exp_info)
    train_batch_size = param["batch_episode"] * env_info["episode_limit"]
    if "fixed_batch_timesteps" in exp_info:
        train_batch_size = int(exp_info["fixed_batch_timesteps"])
    sgd_minibatch_size = train_batch_size
    episode_limit = int(env_info["episode_limit"])
    while sgd_minibatch_size < episode_limit:
        sgd_minibatch_size *= 2

    backup_config = merge_dicts(exp_info, env_info)
    backup_config.pop("algo_args", None)
    backup_config.pop("callbacks", None)
    trainer_config = {
        "batch_mode": param["batch_mode"],
        "train_batch_size": train_batch_size,
        "sgd_minibatch_size": sgd_minibatch_size,
        "lr": param["lr"],
        "entropy_coeff": param["entropy_coeff"],
        "num_sgd_iter": param["num_sgd_iter"],
        "clip_param": param["clip_param"],
        "use_gae": param["use_gae"],
        "lambda": param["lambda"],
        "vf_loss_coeff": param["vf_loss_coeff"],
        "kl_coeff": param["kl_coeff"],
        "vf_clip_param": param["vf_clip_param"],
        "model": {
            "custom_model": "Centralized_Critic_Model",
            "custom_model_config": backup_config,
        },
    }
    trainer_config.update(run_config)
    ModelCatalog.register_custom_model("Centralized_Critic_Model", DMPMAPPOModel)
    return trainer_config


@contextmanager
def _weights_only_restore_patch(enabled: bool):
    if not enabled:
        yield
        return

    from ray.rllib.policy.torch_policy import TorchPolicy

    original_set_state = TorchPolicy.set_state

    def set_state_without_optimizer(self, state):
        if isinstance(state, dict):
            state = dict(state)
            state["_optimizer_variables"] = []
            for key in (
                "optimizer_variables",
                "optimizer_state",
                "optim_state_dict",
            ):
                state.pop(key, None)
        return original_set_state(self, state)

    TorchPolicy.set_state = set_state_without_optimizer
    try:
        yield
    finally:
        TorchPolicy.set_state = original_set_state


def _build_trainer(config, checkpoint_path: Path, num_workers: int, weights_only_restore: bool):
    import ray
    from marllib.marl.algos.core.CC.mappo import MAPPOTrainer

    if not ray.is_initialized():
        ray.init(local_mode=True, num_gpus=0, ignore_reinit_error=True, include_dashboard=False)
    trainer = MAPPOTrainer(config=_build_trainer_config(config, num_workers=num_workers))
    with _weights_only_restore_patch(weights_only_restore):
        trainer.restore(str(checkpoint_path))
    return trainer


def _get_policy_id(trainer) -> str:
    try:
        policy_map = trainer.workers.local_worker().policy_map
        if "shared_policy" in policy_map:
            return "shared_policy"
        if "default_policy" in policy_map:
            return "default_policy"
        return next(iter(policy_map.keys()))
    except Exception:
        return "shared_policy"


def _compute_policy_action(trainer, observation: np.ndarray, policy_id: str, deterministic: bool) -> np.ndarray:
    action = trainer.compute_single_action(
        {"obs": observation.astype(np.float32, copy=True)},
        policy_id=policy_id,
        explore=not deterministic,
    )
    if isinstance(action, tuple):
        action = action[0]
    return np.asarray(action, dtype=np.float32)


def _min_pairwise_distance(positions: np.ndarray) -> float:
    if len(positions) <= 1:
        return float("inf")
    deltas = positions[:, None, :] - positions[None, :, :]
    distances = np.linalg.norm(deltas, axis=-1)
    upper = distances[np.triu_indices(len(positions), k=1)]
    return float(np.min(upper)) if upper.size else float("inf")


def _positions(env) -> np.ndarray:
    return np.asarray(env._positions(), dtype=float)


def _record_step(
    env,
    info: dict[str, Any],
    positions: list[np.ndarray],
    distances: list[np.ndarray],
    pairwise_min_distances: list[float],
    boundary_min_distances: list[float],
) -> None:
    current_positions = _positions(env)
    positions.append(current_positions.copy())
    distances.append(np.asarray(info.get("distance_to_goals", []), dtype=float).copy())
    pairwise_min_distances.append(_min_pairwise_distance(current_positions))
    boundary_distances = np.asarray(info.get("min_boundary_distances", []), dtype=float)
    boundary_min_distances.append(float(np.min(boundary_distances)) if boundary_distances.size else float("nan"))


def _rollout_scenario(
    *,
    config,
    trainer,
    policy_id: str,
    deterministic: bool,
    scenario: Scenario,
    seed: int,
) -> RolloutResult:
    env = _build_core_env(config)
    try:
        observation, info = env.reset(
            seed=seed,
            options={
                "starts": scenario.starts.copy(),
                "goals": scenario.goals.copy(),
            },
        )
        positions: list[np.ndarray] = []
        distances: list[np.ndarray] = []
        pairwise_min_distances: list[float] = []
        boundary_min_distances: list[float] = []
        action_norms: list[np.ndarray] = []
        forcing_norms: list[np.ndarray] = []
        goal_offset_norms: list[np.ndarray] = []
        applied_acceleration_norms: list[np.ndarray] = []
        _record_step(env, info, positions, distances, pairwise_min_distances, boundary_min_distances)

        first_inter_agent_collision_step = None
        first_boundary_collision_step = None
        first_obstacle_collision_step = None
        clip_count = 0
        clip_total = 0
        total_reward = 0.0
        terminal_status = "max_steps"

        initial_collision = bool(info.get("collision", False))
        if initial_collision:
            terminal_status = "initial_collision"

        for step in range(1, int(config.max_steps) + 1):
            if initial_collision:
                break

            action = np.stack(
                [
                    _compute_policy_action(trainer, observation[agent_index], policy_id, deterministic)
                    for agent_index in range(int(config.num_agents))
                ],
                axis=0,
            ).astype(np.float32)
            action_dims = action.shape[1] // 2
            action_norms.append(np.linalg.norm(action, axis=1))
            forcing_norms.append(np.linalg.norm(action[:, :action_dims], axis=1))
            goal_offset_norms.append(np.linalg.norm(action[:, action_dims:], axis=1))

            observation, rewards, terminated, truncated, info = env.step(action)
            total_reward += float(np.sum(rewards))
            done = bool(terminated or truncated)

            commanded = np.asarray(info.get("commanded_accelerations", []), dtype=float)
            applied = np.asarray(info.get("applied_accelerations", []), dtype=float)
            if commanded.shape == applied.shape and commanded.size:
                clipped = np.any(~np.isclose(commanded, applied, atol=1e-6), axis=1)
                clip_count += int(np.sum(clipped))
                clip_total += int(clipped.size)
                applied_acceleration_norms.append(np.linalg.norm(applied, axis=1))

            if first_inter_agent_collision_step is None and np.any(info.get("inter_agent_collision_mask", [])):
                first_inter_agent_collision_step = step
            if first_boundary_collision_step is None and np.any(info.get("boundary_collision_mask", [])):
                first_boundary_collision_step = step
            if first_obstacle_collision_step is None and np.any(info.get("obstacle_collision_mask", [])):
                first_obstacle_collision_step = step

            _record_step(env, info, positions, distances, pairwise_min_distances, boundary_min_distances)

            if done:
                if bool(info.get("success", False)):
                    terminal_status = "success"
                elif bool(info.get("collision", False)):
                    terminal_status = "collision"
                elif truncated:
                    terminal_status = "timeout"
                else:
                    terminal_status = "done"
                break
        else:
            if bool(info.get("success", False)):
                terminal_status = "success"
            elif bool(info.get("truncated", False)):
                terminal_status = "timeout"

        distance_array = np.asarray(distances, dtype=float)
        final_distances = distance_array[-1]
        min_distances = np.min(distance_array, axis=0)
        success_rewarded_mask = np.asarray(
            info.get("success_rewarded_mask", final_distances <= float(config.goal_tolerance)),
            dtype=bool,
        )
        success_mask = np.asarray(info.get("success_mask", final_distances <= float(config.goal_tolerance)), dtype=bool)
        all_reached = bool(np.all(success_mask))
        reached_agent_count = int(np.sum(success_rewarded_mask))
        action_norm_array = np.asarray(action_norms, dtype=float)
        forcing_norm_array = np.asarray(forcing_norms, dtype=float)
        goal_offset_norm_array = np.asarray(goal_offset_norms, dtype=float)
        applied_acceleration_norm_array = np.asarray(applied_acceleration_norms, dtype=float)
        obstacle_collision_mask = np.asarray(info.get("obstacle_collision_mask", []), dtype=bool)
        inter_agent_collision_mask = np.asarray(info.get("inter_agent_collision_mask", []), dtype=bool)
        boundary_collision_mask = np.asarray(info.get("boundary_collision_mask", []), dtype=bool)

        return RolloutResult(
            name=scenario.name,
            source=scenario.source,
            steps=len(positions) - 1,
            terminal_status=terminal_status,
            all_reached=all_reached,
            reached_agent_count=reached_agent_count,
            final_distance_mean=float(np.mean(final_distances)),
            final_distance_max=float(np.max(final_distances)),
            min_distance_mean=float(np.mean(min_distances)),
            min_pairwise_distance=float(np.nanmin(pairwise_min_distances)),
            min_boundary_distance=float(np.nanmin(boundary_min_distances)),
            first_inter_agent_collision_step=first_inter_agent_collision_step,
            first_boundary_collision_step=first_boundary_collision_step,
            first_obstacle_collision_step=first_obstacle_collision_step,
            obstacle_collision_rate=float(np.mean(obstacle_collision_mask)) if obstacle_collision_mask.size else 0.0,
            inter_agent_collision_rate=float(np.mean(inter_agent_collision_mask)) if inter_agent_collision_mask.size else 0.0,
            boundary_collision_rate=float(np.mean(boundary_collision_mask)) if boundary_collision_mask.size else 0.0,
            timeout=terminal_status == "timeout",
            action_norm_mean=float(np.mean(action_norm_array)) if action_norm_array.size else 0.0,
            forcing_norm_mean=float(np.mean(forcing_norm_array)) if forcing_norm_array.size else 0.0,
            goal_offset_norm_mean=float(np.mean(goal_offset_norm_array)) if goal_offset_norm_array.size else 0.0,
            applied_acceleration_norm_mean=(
                float(np.mean(applied_acceleration_norm_array)) if applied_acceleration_norm_array.size else 0.0
            ),
            acceleration_clip_ratio=float(clip_count / max(1, clip_total)),
            total_reward=total_reward,
            starts=scenario.starts.copy(),
            goals=scenario.goals.copy(),
            positions=np.asarray(positions, dtype=float),
            distances=distance_array,
            pairwise_min_distances=np.asarray(pairwise_min_distances, dtype=float),
            boundary_min_distances=np.asarray(boundary_min_distances, dtype=float),
            action_norms=action_norm_array,
            forcing_norms=forcing_norm_array,
            goal_offset_norms=goal_offset_norm_array,
            applied_acceleration_norms=applied_acceleration_norm_array,
        )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def _result_to_summary(result: RolloutResult) -> dict[str, Any]:
    return {
        "name": result.name,
        "source": result.source,
        "trace_path": f"scenario_traces/{result.name}.json",
        "steps": result.steps,
        "terminal_status": result.terminal_status,
        "all_reached": result.all_reached,
        "reached_agent_count": result.reached_agent_count,
        "final_distance_mean": result.final_distance_mean,
        "final_distance_max": result.final_distance_max,
        "min_distance_mean": result.min_distance_mean,
        "min_pairwise_distance": result.min_pairwise_distance,
        "min_boundary_distance": result.min_boundary_distance,
        "first_inter_agent_collision_step": result.first_inter_agent_collision_step,
        "first_boundary_collision_step": result.first_boundary_collision_step,
        "first_obstacle_collision_step": result.first_obstacle_collision_step,
        "obstacle_collision_rate": result.obstacle_collision_rate,
        "inter_agent_collision_rate": result.inter_agent_collision_rate,
        "boundary_collision_rate": result.boundary_collision_rate,
        "timeout": result.timeout,
        "action_norm_mean": result.action_norm_mean,
        "forcing_norm_mean": result.forcing_norm_mean,
        "goal_offset_norm_mean": result.goal_offset_norm_mean,
        "applied_acceleration_norm_mean": result.applied_acceleration_norm_mean,
        "acceleration_clip_ratio": result.acceleration_clip_ratio,
        "total_reward": result.total_reward,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _result_to_trace_payload(result: RolloutResult, config) -> dict[str, Any]:
    return {
        "metrics": _result_to_summary(result),
        "config": {
            "workspace_bounds": config.workspace_bounds,
            "goal_tolerance": config.goal_tolerance,
            "inter_agent_safe_distance": config.inter_agent_safe_distance,
            "num_agents": config.num_agents,
        },
        "starts": result.starts,
        "goals": result.goals,
        "positions": result.positions,
        "distances": result.distances,
        "pairwise_min_distances": result.pairwise_min_distances,
        "boundary_min_distances": result.boundary_min_distances,
        "action_norms": result.action_norms,
        "forcing_norms": result.forcing_norms,
        "goal_offset_norms": result.goal_offset_norms,
        "applied_acceleration_norms": result.applied_acceleration_norms,
    }


def _save_trace_json(result: RolloutResult, output_dir: Path, config) -> Path:
    trace_dir = output_dir / "scenario_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    output_path = trace_dir / f"{result.name}.json"
    payload = _jsonable(_result_to_trace_payload(result, config))
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def _print_result(result: RolloutResult, agent_count: int) -> None:
    print(
        f"{result.name:18s} | "
        f"status={result.terminal_status:9s} | "
        f"reached={result.reached_agent_count}/{agent_count} | "
        f"steps={result.steps:3d} | "
        f"final_mean={result.final_distance_mean:.3f} | "
        f"final_max={result.final_distance_max:.3f} | "
        f"min_pair={result.min_pairwise_distance:.3f} | "
        f"min_boundary={result.min_boundary_distance:.3f} | "
        f"clip={result.acceleration_clip_ratio:.3f}"
    )


def _write_summary(
    results: list[RolloutResult],
    output_dir: Path,
    config_source: str,
    checkpoint_path: Path,
    deterministic: bool,
    weights_only_restore: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [_result_to_summary(result) for result in results]
    payload = {
        "config_source": config_source,
        "checkpoint": str(checkpoint_path),
        "deterministic": deterministic,
        "weights_only_restore": weights_only_restore,
        "scenario_count": len(results),
        "all_reached_rate": float(np.mean([row["all_reached"] for row in rows])) if rows else 0.0,
        "mean_reached_agent_count": float(np.mean([row["reached_agent_count"] for row in rows])) if rows else 0.0,
        "mean_final_distance": float(np.mean([row["final_distance_mean"] for row in rows])) if rows else 0.0,
        "collision_rate": float(np.mean([row["terminal_status"] == "collision" for row in rows])) if rows else 0.0,
        "timeout_rate": float(np.mean([row["timeout"] for row in rows])) if rows else 0.0,
        "results": rows,
    }
    output_path = output_dir / "summary.json"
    output_path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def _select_visualized_results(
    results: list[RolloutResult],
    visualize_mode: str,
    max_visualized: int,
) -> list[RolloutResult]:
    if visualize_mode == "none" or max_visualized <= 0:
        return []
    if visualize_mode == "all":
        candidates = results
    elif visualize_mode == "fixed":
        candidates = [result for result in results if result.source == "fixed"]
    elif visualize_mode == "failures":
        candidates = [result for result in results if result.terminal_status != "success" or not result.all_reached]
    else:
        raise ValueError(f"unsupported visualize_mode: {visualize_mode}")
    return list(candidates[:max_visualized])


def _write_html(
    *,
    results: list[RolloutResult],
    visualized_results: list[RolloutResult],
    output_dir: Path,
    config,
    config_source: str,
    checkpoint_path: Path,
    deterministic: bool,
    weights_only_restore: bool,
    seed: int,
) -> Path | None:
    if not visualized_results:
        return None

    summary_rows = [_result_to_summary(result) for result in results]
    visualized_payload = {
        result.name: _jsonable(_result_to_trace_payload(result, config))
        for result in visualized_results
    }
    payload = {
        "meta": {
            "config_source": config_source,
            "checkpoint": str(checkpoint_path),
            "deterministic": deterministic,
            "weights_only_restore": weights_only_restore,
            "seed": seed,
            "scenario_count": len(results),
            "visualized_count": len(visualized_results),
            "num_agents": int(config.num_agents),
        },
        "config": {
            "workspace_bounds": config.workspace_bounds,
            "goal_tolerance": config.goal_tolerance,
            "inter_agent_safe_distance": config.inter_agent_safe_distance,
        },
        "summary": summary_rows,
        "traces": visualized_payload,
    }

    html_template = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MAPPO Multi-agent DMP Evaluation</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #111827;
      --panel: #1f2937;
      --panel-2: #243041;
      --text: #e5e7eb;
      --muted: #9ca3af;
      --line: #374151;
      --cyan: #22d3ee;
      --green: #34d399;
      --red: #fb7185;
      --amber: #fbbf24;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: "Segoe UI", Arial, sans-serif;
    }
    header {
      padding: 22px 28px 14px;
      border-bottom: 1px solid var(--line);
      background: #0f172a;
    }
    h1 {
      margin: 0 0 8px;
      font-size: 22px;
      font-weight: 650;
      letter-spacing: 0;
    }
    .subtitle {
      color: var(--muted);
      font-size: 13px;
    }
    main {
      max-width: 1440px;
      margin: 0 auto;
      padding: 20px 22px 32px;
    }
    .cards {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px 14px;
    }
    .card .label {
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
    }
    .card .value {
      font-size: 22px;
      font-weight: 650;
    }
    .layout {
      display: grid;
      grid-template-columns: minmax(0, 1.2fr) minmax(360px, 0.8fr);
      gap: 16px;
      align-items: start;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }
    .toolbar {
      display: grid;
      grid-template-columns: minmax(160px, 1fr) auto auto minmax(180px, 0.8fr);
      gap: 10px;
      align-items: center;
      margin-bottom: 12px;
    }
    select, button, input[type="range"] {
      min-height: 34px;
    }
    select, button {
      background: var(--panel-2);
      color: var(--text);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 6px 10px;
    }
    button {
      cursor: pointer;
      min-width: 76px;
    }
    canvas {
      width: 100%;
      background: #0b1220;
      border: 1px solid var(--line);
      border-radius: 8px;
      display: block;
    }
    #sceneCanvas { height: 520px; }
    .chart-grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 12px;
    }
    .chart {
      height: 168px;
    }
    .table-panel {
      margin-top: 16px;
    }
    .table-toolbar {
      display: flex;
      gap: 10px;
      align-items: center;
      margin-bottom: 10px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }
    th, td {
      border-bottom: 1px solid var(--line);
      padding: 8px 7px;
      text-align: left;
      white-space: nowrap;
    }
    th {
      color: #cbd5e1;
      background: #172033;
      position: sticky;
      top: 0;
    }
    tr[data-status="success"] td:first-child { color: var(--green); }
    tr[data-status="collision"] td:first-child,
    tr[data-status="timeout"] td:first-child,
    tr[data-status="initial_collision"] td:first-child { color: var(--red); }
    .scroll {
      max-height: 420px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .legend {
      color: var(--muted);
      font-size: 12px;
      margin-top: 8px;
      line-height: 1.5;
    }
    @media (max-width: 1040px) {
      .cards { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .layout { grid-template-columns: 1fr; }
      .toolbar { grid-template-columns: 1fr 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>MAPPO Multi-agent DMP Evaluation</h1>
    <div class="subtitle">通过 seed 构建确定性多机场景，并基于 MAPPO checkpoint rollout 数据进行 HTML 动画可视化。</div>
  </header>
  <main>
    <section class="cards" id="metricCards"></section>
    <section class="layout">
      <div class="panel">
        <div class="toolbar">
          <select id="scenarioSelect"></select>
          <button id="playButton">Play</button>
          <button id="resetButton">Reset</button>
          <input id="stepSlider" type="range" min="0" value="0">
        </div>
        <canvas id="sceneCanvas"></canvas>
        <div class="legend" id="sceneInfo"></div>
      </div>
      <div class="chart-grid">
        <div class="panel"><canvas class="chart" id="distanceCanvas"></canvas></div>
        <div class="panel"><canvas class="chart" id="pairwiseCanvas"></canvas></div>
        <div class="panel"><canvas class="chart" id="boundaryCanvas"></canvas></div>
      </div>
    </section>
    <section class="panel table-panel">
      <div class="table-toolbar">
        <label for="statusFilter">场景筛选</label>
        <select id="statusFilter">
          <option value="all">全部</option>
          <option value="success">成功</option>
          <option value="failure">失败</option>
          <option value="fixed">Fixed</option>
          <option value="random">Random</option>
        </select>
      </div>
      <div class="scroll">
        <table>
          <thead>
            <tr>
              <th>场景</th><th>来源</th><th>状态</th><th>到达</th><th>步数</th>
              <th>final mean</th><th>final max</th><th>min pair</th><th>min boundary</th>
              <th>inter rate</th><th>boundary rate</th><th>timeout</th>
              <th>action norm</th><th>offset norm</th><th>inter step</th><th>boundary step</th><th>trace</th>
            </tr>
          </thead>
          <tbody id="summaryTable"></tbody>
        </table>
      </div>
    </section>
  </main>
  <script>
    const DATA = __PAYLOAD_JSON__;
    const COLORS = ["#22d3ee", "#34d399", "#fbbf24", "#fb7185", "#a78bfa", "#60a5fa", "#f97316", "#14b8a6"];
    const traceNames = Object.keys(DATA.traces);
    let currentName = traceNames[0] || "";
    let currentStep = 0;
    let playing = false;
    let lastFrameTime = 0;

    const sceneCanvas = document.getElementById("sceneCanvas");
    const distanceCanvas = document.getElementById("distanceCanvas");
    const pairwiseCanvas = document.getElementById("pairwiseCanvas");
    const boundaryCanvas = document.getElementById("boundaryCanvas");
    const scenarioSelect = document.getElementById("scenarioSelect");
    const stepSlider = document.getElementById("stepSlider");
    const playButton = document.getElementById("playButton");
    const resetButton = document.getElementById("resetButton");
    const sceneInfo = document.getElementById("sceneInfo");

    function fmt(value, digits = 3) {
      return value === null || value === undefined || Number.isNaN(value) ? "-" : Number(value).toFixed(digits);
    }

    function resizeCanvas(canvas) {
      const rect = canvas.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      const width = Math.max(1, Math.round(rect.width * ratio));
      const height = Math.max(1, Math.round(rect.height * ratio));
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
      return { width, height, ratio };
    }

    function worldProject(point, width, height) {
      const bounds = DATA.config.workspace_bounds;
      const pad = 36;
      const xNorm = (point[0] - bounds[0][0]) / Math.max(1e-6, bounds[1][0] - bounds[0][0]);
      const yNorm = (point[1] - bounds[0][1]) / Math.max(1e-6, bounds[1][1] - bounds[0][1]);
      return [
        pad + xNorm * (width - 2 * pad),
        height - pad - yNorm * (height - 2 * pad)
      ];
    }

    function drawScene() {
      const trace = DATA.traces[currentName];
      const { width, height } = resizeCanvas(sceneCanvas);
      const ctx = sceneCanvas.getContext("2d");
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "#0b1220";
      ctx.fillRect(0, 0, width, height);

      if (!trace) return;
      const positions = trace.positions;
      const starts = trace.starts;
      const goals = trace.goals;
      const agentCount = starts.length;
      const step = Math.min(currentStep, positions.length - 1);

      ctx.strokeStyle = "#334155";
      ctx.lineWidth = 1;
      const bounds = DATA.config.workspace_bounds;
      const lower = worldProject([bounds[0][0], bounds[0][1], 0], width, height);
      const upper = worldProject([bounds[1][0], bounds[1][1], 0], width, height);
      ctx.strokeRect(lower[0], upper[1], upper[0] - lower[0], lower[1] - upper[1]);

      for (let agent = 0; agent < agentCount; agent += 1) {
        const color = COLORS[agent % COLORS.length];
        ctx.strokeStyle = color;
        ctx.globalAlpha = 0.30;
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (let i = 0; i < positions.length; i += 1) {
          const p = worldProject(positions[i][agent], width, height);
          if (i === 0) ctx.moveTo(p[0], p[1]);
          else ctx.lineTo(p[0], p[1]);
        }
        ctx.stroke();

        ctx.globalAlpha = 0.95;
        ctx.lineWidth = 3;
        ctx.beginPath();
        for (let i = 0; i <= step; i += 1) {
          const p = worldProject(positions[i][agent], width, height);
          if (i === 0) ctx.moveTo(p[0], p[1]);
          else ctx.lineTo(p[0], p[1]);
        }
        ctx.stroke();

        const start = worldProject(starts[agent], width, height);
        const goal = worldProject(goals[agent], width, height);
        const current = worldProject(positions[step][agent], width, height);

        ctx.fillStyle = color;
        ctx.globalAlpha = 0.75;
        ctx.beginPath();
        ctx.arc(start[0], start[1], 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.globalAlpha = 0.95;
        ctx.beginPath();
        ctx.moveTo(goal[0], goal[1] - 7);
        ctx.lineTo(goal[0] + 7, goal[1] + 7);
        ctx.lineTo(goal[0] - 7, goal[1] + 7);
        ctx.closePath();
        ctx.fill();
        ctx.beginPath();
        ctx.arc(current[0], current[1], 8, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = "#e5e7eb";
        ctx.font = "12px Segoe UI";
        ctx.fillText(`A${agent} z=${fmt(positions[step][agent][2], 2)}`, current[0] + 10, current[1] - 8);
      }
      ctx.globalAlpha = 1;

      const metrics = trace.metrics;
      sceneInfo.textContent =
        `${currentName} | step ${step}/${positions.length - 1} | status=${metrics.terminal_status} | ` +
        `reached=${metrics.reached_agent_count}/${DATA.meta.num_agents} | ` +
        `final_mean=${fmt(metrics.final_distance_mean)} | min_pair=${fmt(metrics.min_pairwise_distance)} | ` +
        `min_boundary=${fmt(metrics.min_boundary_distance)}`;
    }

    function drawChart(canvas, title, seriesList, thresholdList = []) {
      const { width, height } = resizeCanvas(canvas);
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "#0b1220";
      ctx.fillRect(0, 0, width, height);
      ctx.fillStyle = "#e5e7eb";
      ctx.font = "13px Segoe UI";
      ctx.fillText(title, 14, 22);

      const padL = 42, padR = 16, padT = 34, padB = 26;
      const values = [];
      for (const series of seriesList) for (const value of series.values) values.push(value);
      for (const item of thresholdList) values.push(item.value);
      let yMin = Math.min(...values, 0);
      let yMax = Math.max(...values, 1);
      if (!Number.isFinite(yMin) || !Number.isFinite(yMax) || Math.abs(yMax - yMin) < 1e-6) {
        yMin = 0; yMax = 1;
      }
      const xMax = Math.max(1, Math.max(...seriesList.map(s => s.values.length - 1)));
      const xScale = (x) => padL + (x / xMax) * (width - padL - padR);
      const yScale = (y) => height - padB - ((y - yMin) / (yMax - yMin)) * (height - padT - padB);

      ctx.strokeStyle = "#334155";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padL, padT);
      ctx.lineTo(padL, height - padB);
      ctx.lineTo(width - padR, height - padB);
      ctx.stroke();

      for (const threshold of thresholdList) {
        const y = yScale(threshold.value);
        ctx.strokeStyle = threshold.color;
        ctx.setLineDash([6, 5]);
        ctx.beginPath();
        ctx.moveTo(padL, y);
        ctx.lineTo(width - padR, y);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      for (const series of seriesList) {
        ctx.strokeStyle = series.color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        series.values.forEach((value, index) => {
          const x = xScale(index);
          const y = yScale(value);
          if (index === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();
      }

      const stepX = xScale(Math.min(currentStep, xMax));
      ctx.strokeStyle = "#f8fafc";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(stepX, padT);
      ctx.lineTo(stepX, height - padB);
      ctx.stroke();

      ctx.fillStyle = "#9ca3af";
      ctx.font = "11px Segoe UI";
      ctx.fillText(fmt(yMax), 8, padT + 4);
      ctx.fillText(fmt(yMin), 8, height - padB);
    }

    function drawCharts() {
      const trace = DATA.traces[currentName];
      if (!trace) return;
      const agentCount = trace.starts.length;
      const distanceSeries = [];
      for (let agent = 0; agent < agentCount; agent += 1) {
        distanceSeries.push({
          color: COLORS[agent % COLORS.length],
          values: trace.distances.map(row => row[agent])
        });
      }
      drawChart(distanceCanvas, "Distance to goals", distanceSeries, [
        { value: DATA.config.goal_tolerance, color: "#fb7185" }
      ]);
      drawChart(pairwiseCanvas, "Minimum inter-agent distance", [
        { color: "#22d3ee", values: trace.pairwise_min_distances }
      ], [{ value: DATA.config.inter_agent_safe_distance, color: "#fb7185" }]);
      drawChart(boundaryCanvas, "Minimum boundary distance", [
        { color: "#a78bfa", values: trace.boundary_min_distances }
      ], [{ value: 0, color: "#fb7185" }]);
    }

    function render() {
      const trace = DATA.traces[currentName];
      if (!trace) return;
      stepSlider.max = Math.max(0, trace.positions.length - 1);
      stepSlider.value = currentStep;
      drawScene();
      drawCharts();
    }

    function animationLoop(timestamp) {
      if (playing && timestamp - lastFrameTime > 90) {
        const trace = DATA.traces[currentName];
        const maxStep = trace ? trace.positions.length - 1 : 0;
        currentStep = currentStep >= maxStep ? 0 : currentStep + 1;
        lastFrameTime = timestamp;
        render();
      }
      requestAnimationFrame(animationLoop);
    }

    function buildControls() {
      for (const name of traceNames) {
        const option = document.createElement("option");
        option.value = name;
        option.textContent = name;
        scenarioSelect.appendChild(option);
      }
      scenarioSelect.addEventListener("change", () => {
        currentName = scenarioSelect.value;
        currentStep = 0;
        render();
      });
      playButton.addEventListener("click", () => {
        playing = !playing;
        playButton.textContent = playing ? "Pause" : "Play";
      });
      resetButton.addEventListener("click", () => {
        currentStep = 0;
        playing = false;
        playButton.textContent = "Play";
        render();
      });
      stepSlider.addEventListener("input", () => {
        currentStep = Number(stepSlider.value);
        render();
      });
      window.addEventListener("resize", render);
    }

    function buildCards() {
      const rows = DATA.summary;
      const allReachedRate = rows.length ? rows.filter(row => row.all_reached).length / rows.length : 0;
      const meanReached = rows.length ? rows.reduce((s, row) => s + row.reached_agent_count, 0) / rows.length : 0;
      const failures = rows.filter(row => row.terminal_status !== "success" || !row.all_reached).length;
      const meanFinal = rows.length ? rows.reduce((s, row) => s + row.final_distance_mean, 0) / rows.length : 0;
      const cards = [
        ["Scenarios", rows.length],
        ["All reached rate", fmt(allReachedRate, 3)],
        ["Mean reached", fmt(meanReached, 2)],
        ["Failures", failures],
        ["Mean final distance", fmt(meanFinal, 3)]
      ];
      const container = document.getElementById("metricCards");
      for (const [label, value] of cards) {
        const card = document.createElement("div");
        card.className = "card";
        card.innerHTML = `<div class="label">${label}</div><div class="value">${value}</div>`;
        container.appendChild(card);
      }
    }

    function buildTable() {
      const tbody = document.getElementById("summaryTable");
      function paint(filter) {
        tbody.innerHTML = "";
        for (const row of DATA.summary) {
          const success = row.terminal_status === "success" && row.all_reached;
          if (filter === "success" && !success) continue;
          if (filter === "failure" && success) continue;
          if (filter === "fixed" && row.source !== "fixed") continue;
          if (filter === "random" && row.source !== "random") continue;
          const tr = document.createElement("tr");
          tr.dataset.status = row.terminal_status;
          const visualized = Boolean(DATA.traces[row.name]);
          tr.innerHTML = `
            <td>${row.name}</td><td>${row.source}</td><td>${row.terminal_status}</td>
            <td>${row.reached_agent_count}/${DATA.meta.num_agents}</td><td>${row.steps}</td>
            <td>${fmt(row.final_distance_mean)}</td><td>${fmt(row.final_distance_max)}</td>
            <td>${fmt(row.min_pairwise_distance)}</td><td>${fmt(row.min_boundary_distance)}</td>
            <td>${fmt(row.inter_agent_collision_rate)}</td>
            <td>${fmt(row.boundary_collision_rate)}</td>
            <td>${row.timeout ? "1" : "0"}</td>
            <td>${fmt(row.action_norm_mean)}</td>
            <td>${fmt(row.goal_offset_norm_mean)}</td>
            <td>${row.first_inter_agent_collision_step ?? "-"}</td>
            <td>${row.first_boundary_collision_step ?? "-"}</td>
            <td>${visualized ? "HTML" : row.trace_path}</td>`;
          if (visualized) {
            tr.style.cursor = "pointer";
            tr.addEventListener("click", () => {
              currentName = row.name;
              scenarioSelect.value = row.name;
              currentStep = 0;
              render();
            });
          }
          tbody.appendChild(tr);
        }
      }
      document.getElementById("statusFilter").addEventListener("change", (event) => paint(event.target.value));
      paint("all");
    }

    buildCards();
    buildControls();
    buildTable();
    render();
    requestAnimationFrame(animationLoop);
  </script>
</body>
</html>
"""
    html = html_template.replace("__PAYLOAD_JSON__", json.dumps(_jsonable(payload), ensure_ascii=False))
    output_path = output_dir / "index.html"
    output_path.write_text(html, encoding="utf-8")
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a MAPPO checkpoint on multi-agent DMP scenarios.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--random-scenarios", type=int, default=50)
    parser.add_argument("--scenario-sample-attempts", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skip-fixed", action="store_true")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--restore-optimizer-state", action="store_true")
    parser.add_argument("--visualize-mode", choices=("failures", "all", "fixed", "none"), default="failures")
    parser.add_argument("--max-visualized", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "mappo_multi_agent_eval",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config, config_source = _load_mappo_config()
    seed = int(config.seed if args.seed is None else args.seed)
    output_dir = args.output_dir
    checkpoint_path = (
        _find_latest_checkpoint(Path(config.output_root))
        if args.checkpoint is None
        else _resolve_checkpoint_path(args.checkpoint)
    )
    deterministic = not bool(args.stochastic)
    weights_only_restore = not bool(args.restore_optimizer_state)

    scenarios = [] if args.skip_fixed else _build_fixed_eval_scenarios(config)
    scenarios.extend(
        _build_random_scenarios(
            config,
            args.random_scenarios,
            seed + 17,
            scenario_sample_attempts=args.scenario_sample_attempts,
        )
    )

    print("MAPPO multi-agent DMP evaluation")
    print(f"config_source: {config_source}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"deterministic: {deterministic}")
    print(f"weights_only_restore: {weights_only_restore}")
    print(f"seed: {seed}")
    print(f"num_agents: {config.num_agents}")
    print(f"scenario_count: {len(scenarios)}")
    print()

    trainer = _build_trainer(
        config,
        checkpoint_path=checkpoint_path,
        num_workers=args.num_workers,
        weights_only_restore=weights_only_restore,
    )
    policy_id = _get_policy_id(trainer)
    print(f"policy_id: {policy_id}")
    print()

    results: list[RolloutResult] = []
    try:
        for scenario_index, scenario in enumerate(scenarios):
            result = _rollout_scenario(
                config=config,
                trainer=trainer,
                policy_id=policy_id,
                deterministic=deterministic,
                scenario=scenario,
                seed=seed + scenario_index,
            )
            results.append(result)
            _print_result(result, int(config.num_agents))
            _save_trace_json(result, output_dir, config)
    finally:
        trainer.stop()
        try:
            import ray

            if ray.is_initialized():
                ray.shutdown()
        except Exception:
            pass

    summary_path = _write_summary(
        results,
        output_dir,
        config_source,
        checkpoint_path,
        deterministic,
        weights_only_restore,
    )
    visualized_results = _select_visualized_results(results, args.visualize_mode, args.max_visualized)
    html_path = _write_html(
        results=results,
        visualized_results=visualized_results,
        output_dir=output_dir,
        config=config,
        config_source=config_source,
        checkpoint_path=checkpoint_path,
        deterministic=deterministic,
        weights_only_restore=weights_only_restore,
        seed=seed,
    )
    print()
    print(f"summary: {summary_path}")
    print(f"traces: {output_dir / 'scenario_traces'}")
    if html_path is not None:
        print(f"html: {html_path}")


if __name__ == "__main__":
    main()
