"""Single-agent generalization evaluation entry point.

This script evaluates single-agent policy generalization, writes metric CSVs,
records policy outputs, and exports terminal-state figures plus HTML animations.
"""

from __future__ import annotations  # 启用延后解析，避免循环引用、前向引用带来的问题

import argparse
import csv
import html
import json
import re
from dataclasses import dataclass, fields, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from experiment_config import EXPERIMENT_CONFIG, SACExperimentConfig
from runner_sac import build_env, build_model, load_checkpoint


DEFAULT_MODEL_NAME = "best_eval_model.pt"
DEFAULT_OUTPUT_DIR_NAME = "generalization_eval"
DEFAULT_HTML_FPS = 10
ANIMATION_BG_COLOR = "#06111f"
ANIMATION_PANEL_COLOR = "#0b1b2b"
ANIMATION_GRID_COLOR = "#27445e"
ANIMATION_TRAJECTORY_COLOR = "#27e8ff"
ANIMATION_CURRENT_COLOR = "#fff2a8"
ANIMATION_DYNAMIC_COLOR = "#ff4fd8"
ANIMATION_STATIC_COLOR = "#ff5a5f"
ANIMATION_START_COLOR = "#33d17a"
ANIMATION_GOAL_COLOR = "#ffb347"
ANIMATION_TEXT_COLOR = "#d8ecff"
ANIMATION_ACCELERATION_COLOR = "#b6ff3b"


@dataclass(frozen=True)
class EvalPaths:    # 数据类
    """Resolved filesystem paths used by the evaluation script."""

    run_dir: Path
    model_path: Path
    output_root_dir: Path
    event_name: str
    output_dir: Path
    terminal_state_dir: Path
    trace_json_dir: Path
    trajectory_html_dir: Path
    policy_output_dir: Path
    episodes_csv: Path
    summary_csv: Path
    failed_cases_csv: Path
    visualization_manifest_csv: Path
    visualization_index_html: Path
    event_metadata_json: Path


@dataclass(frozen=True)
class EvalArguments:    # 验证参数
    """Validated command-line arguments for one evaluation run."""

    paths: EvalPaths
    episodes_per_scenario: int
    base_seed: int
    save_visualizations: bool


@dataclass(frozen=True)
class EvalRuntime:  # 读取验证过程的模型与环境pipeline
    """Loaded model and environment used by the evaluation pipeline."""

    config: SACExperimentConfig
    env: Any
    model: Any


@dataclass(frozen=True)
class EpisodeMetrics:   # 一个验证episode的统计量获取
    """Scalar metrics collected from one evaluation episode."""

    scenario: str
    seed: int
    reward: float
    episode_length: int
    success: int
    collision: int
    timeout: int
    distance_to_goal: float
    path_length: float
    straight_line_distance: float
    path_efficiency: float
    min_clearance: float
    min_boundary_distance: float
    mean_obstacle_potential_penalty: float
    mean_boundary_potential_penalty: float
    max_obstacle_potential_penalty: float
    max_boundary_potential_penalty: float


@dataclass(frozen=True)
class EpisodeTrace: # 收集路径数据
    """Trajectory data retained for later terminal-state figures and HTML animations."""

    scenario: str
    seed: int
    trajectory: np.ndarray
    acceleration_history: np.ndarray
    action_history: np.ndarray
    scaled_action_history: np.ndarray
    forcing_action_history: np.ndarray
    offside_action_history: np.ndarray
    forcing_log_prob_history: np.ndarray
    offside_log_prob_history: np.ndarray
    start: np.ndarray
    goal: np.ndarray
    obstacles: list[dict[str, Any]]
    obstacle_history: list[list[dict[str, Any]]]
    terminal_status: str


@dataclass(frozen=True)
class EpisodeResult:    # Episode结果
    """Complete result of one episode rollout."""

    metrics: EpisodeMetrics
    trace: EpisodeTrace


@dataclass(frozen=True)
class PolicyStepOutput:
    """Policy outputs recorded for one environment step."""

    action: np.ndarray
    scaled_action: np.ndarray
    forcing_action: np.ndarray
    offside_action: np.ndarray
    forcing_log_prob: float
    offside_log_prob: float


@dataclass(frozen=True)
class ScenarioSpec: # 从场景设置
    """Evaluation scenario generated from the base experiment config."""

    name: str
    description: str
    config: SACExperimentConfig


def parse_args() -> argparse.Namespace: # 构造parser参数类
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate single-agent policy generalization across obstacle and "
            "start/goal distributions."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Training artifact directory, for example artifacts/20260518_171240.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help=(
            "Checkpoint path to evaluate. If omitted, the script uses "
            "<run-dir>/best_eval_model.pt."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Root directory for evaluation events. If omitted, the script uses "
            "<run-dir>/generalization_eval."
        ),
    )
    parser.add_argument(
        "--event-name",
        type=str,
        default=None,
        help=(
            "Name of the current evaluation event directory. If omitted, the "
            "script uses eval_<timestamp>_seed_<base-seed>."
        ),
    )
    parser.add_argument(
        "--episodes-per-scenario",
        type=int,
        default=50,
        help="Number of seeded episodes to run for each scenario.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=202605190,
        help="Base seed used to derive deterministic evaluation seeds.",
    )
    parser.add_argument(
        "--no-visualizations",
        action="store_true",
        help="Disable terminal-state figures and HTML animation exports.",
    )
    return parser.parse_args()


def sanitize_event_name(event_name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", event_name.strip())
    cleaned = cleaned.strip("._-")
    if not cleaned:
        raise ValueError("--event-name must contain at least one valid filename character")
    return cleaned


def make_default_event_name(base_seed: int) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"eval_{timestamp}_seed_{int(base_seed)}"


def resolve_arguments(args: argparse.Namespace) -> EvalArguments:   # 读取参数类
    run_dir = args.run_dir.expanduser().resolve()
    model_path = (
        args.model_path.expanduser().resolve()
        if args.model_path is not None
        else run_dir / DEFAULT_MODEL_NAME
    )
    output_root_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run_dir / DEFAULT_OUTPUT_DIR_NAME
    )

    if args.episodes_per_scenario <= 0:
        raise ValueError("--episodes-per-scenario must be positive")

    base_seed = int(args.base_seed)
    event_name = sanitize_event_name(args.event_name) if args.event_name is not None else make_default_event_name(base_seed)
    output_dir = output_root_dir / event_name

    return EvalArguments(
        paths=EvalPaths(
            run_dir=run_dir,
            model_path=model_path,
            output_root_dir=output_root_dir,
            event_name=event_name,
            output_dir=output_dir,
            terminal_state_dir=output_dir / "terminal_states",
            trace_json_dir=output_dir / "trace_json",
            trajectory_html_dir=output_dir / "trajectory_html",
            policy_output_dir=output_dir / "policy_outputs",
            episodes_csv=output_dir / "episodes.csv",
            summary_csv=output_dir / "summary.csv",
            failed_cases_csv=output_dir / "failed_cases.csv",
            visualization_manifest_csv=output_dir / "visualization_manifest.csv",
            visualization_index_html=output_dir / "visualization_index.html",
            event_metadata_json=output_dir / "event_metadata.json",
        ),
        episodes_per_scenario=int(args.episodes_per_scenario),
        base_seed=base_seed,
        save_visualizations=not bool(args.no_visualizations),
    )


def prepare_output_directories(eval_args: EvalArguments) -> None:   # 组织输出字典
    eval_args.paths.output_dir.mkdir(parents=True, exist_ok=True)
    eval_args.paths.policy_output_dir.mkdir(parents=True, exist_ok=True)
    if eval_args.save_visualizations:
        eval_args.paths.terminal_state_dir.mkdir(parents=True, exist_ok=True)
        eval_args.paths.trace_json_dir.mkdir(parents=True, exist_ok=True)
        eval_args.paths.trajectory_html_dir.mkdir(parents=True, exist_ok=True)


def validate_input_paths(eval_args: EvalArguments) -> None: # 验证输入路径
    if not eval_args.paths.run_dir.exists():
        raise FileNotFoundError(f"run_dir does not exist: {eval_args.paths.run_dir}")
    if not eval_args.paths.model_path.exists():
        raise FileNotFoundError(f"model_path does not exist: {eval_args.paths.model_path}")


def load_evaluation_runtime(
    eval_args: EvalArguments,
    config: SACExperimentConfig = EXPERIMENT_CONFIG,
) -> EvalRuntime:   # 构建模型并恢复参数
    env = build_env(config=config, action_guidance_enabled=False)
    model = build_model(env=env, config=config, tensorboard_log=None, verbose=0)
    model = load_checkpoint(model=model, model_path=eval_args.paths.model_path)

    if hasattr(model, "actor"):
        model.actor.train(False)
    if hasattr(model, "critic"):
        model.critic.train(False)

    return EvalRuntime(config=config, env=env, model=model)


def close_runtime(runtime: EvalRuntime | None) -> None: # 当前场景有close事件，调用这个函数处理
    if runtime is None:
        return
    if hasattr(runtime.env, "close"):
        runtime.env.close()


def build_generalization_scenarios(
    base_config: SACExperimentConfig = EXPERIMENT_CONFIG,
) -> list[ScenarioSpec]:    # 构建主要场景
    base_config = replace(base_config, training_scene_mixture_enabled=False)
    scenarios = [
        ScenarioSpec(
            name="train_distribution",
            description="Base distribution used during training.",
            config=base_config,
        ),
        ScenarioSpec(
            name="more_static_obstacles",
            description="Increase static obstacle count while preserving the training start/goal distribution.",
            config=replace(
                base_config,
                static_obstacle_num=max(5, base_config.static_obstacle_num + 4),
            ),
        ),
        ScenarioSpec(
            name="shifted_static_positions",
            description="Move static obstacle sampling range to evaluate position-distribution generalization.",
            config=replace(
                base_config,
                static_obstacle_center=((2.0, -1.8, -0.5), (7.8, 1.8, 0.5)),
            ),
        ),
        ScenarioSpec(
            name="wide_start_goal",
            description="Expand start and goal sampling bounds while keeping the same workspace.",
            config=replace(
                base_config,
                start_position_bounds=((-0.2, -2.0, -0.8), (1.2, 1.4, 0.8)),
                goal_position_bounds=((6.8, -1.4, -0.8), (8.3, 1.6, 0.8)),
                min_start_goal_distance=5.5,
            ),
        ),
        ScenarioSpec(
            name="near_boundary_start_goal",
            description="Sample start and goal near workspace boundaries to test boundary-potential robustness.",
            config=replace(
                base_config,
                start_position_bounds=((-0.45, -2.35, -1.1), (0.3, -1.7, -0.6)),
                goal_position_bounds=((7.7, 1.2, 0.6), (8.45, 1.9, 1.1)),
                min_start_goal_distance=6.0,
            ),
        ),
        ScenarioSpec(
            name="dense_static_passage",
            description="Use many static obstacles near the nominal route to form a narrow passable corridor.",
            config=replace(
                base_config,
                static_obstacle_num=4,
                static_obstacle_radius=0.38,
                static_obstacle_safety_margin=0.08,
                static_obstacle_center=((1.0, -1.15, -0.45), (8.0, 1.15, 0.45)),
                static_cylinder_num=3,
                static_cylinder_radius=0.36,
                static_cylinder_half_height=0.62,
                static_cylinder_safety_margin=0.08,
                static_cylinder_center=((1.0, -1.15, 0.0), (8.0, 1.15, 0.0)),
                dynamic_obstacle_num=0,
            ),
        ),
        ScenarioSpec(
            name="dense_mixed_passage",
            description="Combine dense static and dynamic obstacles around a narrow passable corridor.",
            config=replace(
                base_config,
                static_obstacle_num=3,
                static_obstacle_radius=0.36,
                static_obstacle_safety_margin=0.08,
                static_obstacle_center=((1.0, -1.10, -0.45), (8.0, 1.10, 0.45)),
                static_cylinder_num=2,
                static_cylinder_radius=0.34,
                static_cylinder_half_height=0.62,
                static_cylinder_safety_margin=0.08,
                static_cylinder_center=((1.0, -1.10, 0.0), (8.0, 1.10, 0.0)),
                dynamic_obstacle_num=2,
                dynamic_obstacle_radius=0.22,
                dynamic_obstacle_safety_margin=0.06,
                dynamic_obstacle_center=((1.2, -1.35, -0.90), (8.0, 1.35, 0.70)),
                dynamic_obstacle_bounds=((0.8, -1.55, -1.00), (8.2, 1.55, 0.85)),
            ),
        ),
    ]
    return scenarios


def compute_path_length(trajectory: np.ndarray) -> float:
    if len(trajectory) < 2:
        return 0.0
    step_distances = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
    return float(np.sum(step_distances))


def safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))


def safe_max(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.max(values))


def finite_min(values: list[float]) -> float:
    finite_values = [value for value in values if np.isfinite(value)]
    if not finite_values:
        return float("nan")
    return float(np.min(finite_values))


def terminal_status_from_info(info: dict[str, Any]) -> str:
    if bool(info.get("success", False)):
        return "success"
    if bool(info.get("collision", False)):
        return "collision"
    if bool(info.get("truncated", False)):
        return "timeout"
    return "unknown"


def extract_obstacles(env: Any) -> list[dict[str, Any]]:
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
            elif hasattr(obstacle, "expanded_radius") and hasattr(obstacle, "expanded_half_height"):
                item["type"] = "cylinder"
                item["radius"] = float(obstacle.expanded_radius)
                item["half_height"] = float(obstacle.expanded_half_height)
                item["body_radius"] = float(obstacle.radius)
                item["body_half_height"] = float(obstacle.half_height)
                item["safety_margin"] = float(obstacle.safety_margin)
            elif hasattr(obstacle, "effective_radius"):
                item["type"] = "sphere"
                item["radius"] = float(obstacle.effective_radius)
            else:
                item["type"] = "unsupported"
            obstacles.append(item)
    return obstacles


def _first_scalar(value: np.ndarray, default: float = float("nan")) -> float:
    flat = np.asarray(value, dtype=float).reshape(-1)
    if flat.size == 0:
        return default
    return float(flat[0])


def _scale_action_if_available(model: Any, action: np.ndarray) -> np.ndarray:
    if not hasattr(model, "_scale_action"):
        return np.full_like(np.asarray(action, dtype=float), np.nan, dtype=float)
    try:
        return np.asarray(model._scale_action(np.asarray(action, dtype=float)), dtype=float)
    except Exception:
        return np.full_like(np.asarray(action, dtype=float), np.nan, dtype=float)


def _split_scaled_action(scaled_action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaled_action = np.asarray(scaled_action, dtype=float)
    if scaled_action.ndim != 1 or scaled_action.size % 2 != 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    split_index = scaled_action.size // 2
    return scaled_action[:split_index].copy(), scaled_action[split_index:].copy()


def predict_policy_step_output(
    model: Any,
    observation: np.ndarray,
    deterministic: bool,
) -> PolicyStepOutput:
    if all(hasattr(model, attr) for attr in ("actor", "_split_observations", "_unscale_action", "device")):
        try:
            import torch as th

            obs_array = np.asarray(observation, dtype=np.float32)
            vectorized = obs_array.ndim > 1
            if not vectorized:
                obs_array = obs_array[None, :]

            obs_tensor = th.as_tensor(obs_array, device=model.device, dtype=th.float32)
            with th.no_grad():
                sensor_observation, extra_observation = model._split_observations(obs_tensor)
                forcing_action, offside_action, forcing_log_prob, offside_log_prob = model.actor(
                    sensor_observation,
                    extra_observation,
                    deterministic=deterministic,
                )
                scaled_action_tensor = th.cat([forcing_action, offside_action], dim=-1)

            scaled_action = scaled_action_tensor.detach().cpu().numpy()
            action = np.asarray(model._unscale_action(scaled_action), dtype=float)
            forcing_action_np = forcing_action.detach().cpu().numpy()
            offside_action_np = offside_action.detach().cpu().numpy()
            forcing_log_prob_np = forcing_log_prob.detach().cpu().numpy()
            offside_log_prob_np = offside_log_prob.detach().cpu().numpy()

            if not vectorized:
                action = action.squeeze(axis=0)
                scaled_action = scaled_action.squeeze(axis=0)
                forcing_action_np = forcing_action_np.squeeze(axis=0)
                offside_action_np = offside_action_np.squeeze(axis=0)

            return PolicyStepOutput(
                action=np.asarray(action, dtype=float),
                scaled_action=np.asarray(scaled_action, dtype=float),
                forcing_action=np.asarray(forcing_action_np, dtype=float),
                offside_action=np.asarray(offside_action_np, dtype=float),
                forcing_log_prob=_first_scalar(forcing_log_prob_np),
                offside_log_prob=_first_scalar(offside_log_prob_np),
            )
        except Exception:
            pass

    action, _ = model.predict(observation, deterministic=deterministic)
    action_array = np.asarray(action, dtype=float)
    scaled_action = _scale_action_if_available(model, action_array)
    forcing_action, offside_action = _split_scaled_action(scaled_action)
    return PolicyStepOutput(
        action=action_array,
        scaled_action=scaled_action,
        forcing_action=forcing_action,
        offside_action=offside_action,
        forcing_log_prob=float("nan"),
        offside_log_prob=float("nan"),
    )


def run_single_episode(     # 运行单个场景
    model: Any,
    scenario: ScenarioSpec,
    seed: int,
    deterministic: bool = True,
) -> EpisodeResult:
    env = build_env(config=scenario.config, action_guidance_enabled=False)  # 构建环境
    try:
        observation, reset_info = env.reset(seed=seed)
        start = env.dynamics.p.copy()
        goal = env.goal.copy()
        trajectory = [start.copy()]
        acceleration_history = [np.zeros(3, dtype=float)]
        obstacle_history = [extract_obstacles(env)]
        action_history: list[np.ndarray] = []
        scaled_action_history: list[np.ndarray] = []
        forcing_action_history: list[np.ndarray] = []
        offside_action_history: list[np.ndarray] = []
        forcing_log_prob_history: list[float] = []
        offside_log_prob_history: list[float] = []
        total_reward = 0.0
        episode_length = 0

        # 拉几个图表，拿来记录信息
        info: dict[str, Any] = dict(reset_info) # 场景信息记录
        min_clearance_values: list[float] = []  # 最小的障碍物距离
        min_boundary_distance_values: list[float] = []  # 最小边界距离
        obstacle_penalty_values: list[float] = []   # 障碍物惩罚
        boundary_penalty_values: list[float] = []   # 边界惩罚只

        # 若有，则直接从info里解包
        if "min_clearance" in info:
            min_clearance_values.append(float(info["min_clearance"]))   
        if "min_boundary_distance" in info:
            min_boundary_distance_values.append(float(info["min_boundary_distance"]))

        terminated = False
        truncated = False

        while not (terminated or truncated):    # 推进环境
            policy_output = predict_policy_step_output(model, observation, deterministic=deterministic)
            action_history.append(policy_output.action.copy())
            scaled_action_history.append(policy_output.scaled_action.copy())
            forcing_action_history.append(policy_output.forcing_action.copy())
            offside_action_history.append(policy_output.offside_action.copy())
            forcing_log_prob_history.append(policy_output.forcing_log_prob)
            offside_log_prob_history.append(policy_output.offside_log_prob)

            observation, reward, terminated, truncated, info = env.step(policy_output.action) # 收集环境信息
            total_reward += float(reward)   # 累加reward
            episode_length += 1     # 长度+1
            trajectory.append(env.dynamics.p.copy())    # 记录动态过程的位置
            acceleration_history.append(np.asarray(info.get("applied_acceleration", np.zeros(3)), dtype=float).copy())
            obstacle_history.append(extract_obstacles(env))

            # 解包信息
            if "min_clearance" in info:
                min_clearance_values.append(float(info["min_clearance"]))
            if "min_boundary_distance" in info:
                min_boundary_distance_values.append(float(info["min_boundary_distance"]))
            if "reward_obstacle_potential_penalty" in info:
                obstacle_penalty_values.append(float(info["reward_obstacle_potential_penalty"]))
            if "reward_boundary_potential_penalty" in info:
                boundary_penalty_values.append(float(info["reward_boundary_potential_penalty"]))

        trajectory_array = np.asarray(trajectory, dtype=float)  # 记录轨迹向量
        path_length = compute_path_length(trajectory_array) # 记录路径长度
        straight_line_distance = float(np.linalg.norm(goal - start))    # 起点到中带你的直线距离
        path_efficiency = float(straight_line_distance / path_length) if path_length > 1e-8 else 0.0    # 路径效率
        terminal_status = terminal_status_from_info(info)   # 记录终止状态

        metrics = EpisodeMetrics(
            scenario=scenario.name,
            seed=int(seed),
            reward=float(total_reward),
            episode_length=int(episode_length),
            success=int(bool(info.get("success", False))),
            collision=int(bool(info.get("collision", False))),
            timeout=int(bool(info.get("truncated", False))),
            distance_to_goal=float(info.get("distance_to_goal", np.nan)),
            path_length=path_length,
            straight_line_distance=straight_line_distance,
            path_efficiency=path_efficiency,
            min_clearance=finite_min(min_clearance_values),
            min_boundary_distance=finite_min(min_boundary_distance_values),
            mean_obstacle_potential_penalty=safe_mean(obstacle_penalty_values),
            mean_boundary_potential_penalty=safe_mean(boundary_penalty_values),
            max_obstacle_potential_penalty=safe_max(obstacle_penalty_values),
            max_boundary_potential_penalty=safe_max(boundary_penalty_values),
        )
        trace = EpisodeTrace(
            scenario=scenario.name,
            seed=int(seed),
            trajectory=trajectory_array,
            acceleration_history=np.asarray(acceleration_history, dtype=float),
            action_history=np.asarray(action_history, dtype=float),
            scaled_action_history=np.asarray(scaled_action_history, dtype=float),
            forcing_action_history=np.asarray(forcing_action_history, dtype=float),
            offside_action_history=np.asarray(offside_action_history, dtype=float),
            forcing_log_prob_history=np.asarray(forcing_log_prob_history, dtype=float),
            offside_log_prob_history=np.asarray(offside_log_prob_history, dtype=float),
            start=start,
            goal=goal,
            obstacles=obstacle_history[-1],
            obstacle_history=obstacle_history,
            terminal_status=terminal_status,
        )
        return EpisodeResult(metrics=metrics, trace=trace)
    finally:
        env.close()


def print_episode_summary(result: EpisodeResult) -> None:
    metrics = result.metrics
    print(
        "smoke_episode: "
        f"scenario={metrics.scenario}, "
        f"seed={metrics.seed}, "
        f"status={result.trace.terminal_status}, "
        f"reward={metrics.reward:.3f}, "
        f"length={metrics.episode_length}, "
        f"distance_to_goal={metrics.distance_to_goal:.3f}, "
        f"path_length={metrics.path_length:.3f}, "
        f"path_efficiency={metrics.path_efficiency:.3f}"
    )


def make_episode_seed(base_seed: int, scenario_index: int, episode_index: int) -> int:
    return int(base_seed + scenario_index * 100_000 + episode_index)


def run_batch_evaluation(
    model: Any,
    scenarios: list[ScenarioSpec],
    episodes_per_scenario: int,
    base_seed: int,
    deterministic: bool = True,
) -> list[EpisodeResult]:
    results: list[EpisodeResult] = []
    for scenario_index, scenario in enumerate(scenarios):
        scenario_results: list[EpisodeResult] = []
        print(f"[scenario] {scenario.name}: start")
        for episode_index in range(episodes_per_scenario):
            seed = make_episode_seed(base_seed, scenario_index, episode_index)
            result = run_single_episode(
                model=model,
                scenario=scenario,
                seed=seed,
                deterministic=deterministic,
            )
            scenario_results.append(result)
            results.append(result)

        successes = [result.metrics.success for result in scenario_results]
        collisions = [result.metrics.collision for result in scenario_results]
        timeouts = [result.metrics.timeout for result in scenario_results]
        rewards = [result.metrics.reward for result in scenario_results]
        print(
            f"[scenario] {scenario.name}: done | "
            f"episodes={len(scenario_results)} | "
            f"success_rate={safe_mean(successes):.3f} | "
            f"collision_rate={safe_mean(collisions):.3f} | "
            f"timeout_rate={safe_mean(timeouts):.3f} | "
            f"mean_reward={safe_mean(rewards):.3f}"
        )
    return results


def episode_metric_fieldnames() -> list[str]:
    return [field.name for field in fields(EpisodeMetrics)]


def episode_metrics_to_row(metrics: EpisodeMetrics) -> dict[str, Any]:
    return {field_name: getattr(metrics, field_name) for field_name in episode_metric_fieldnames()}


def write_episode_csv(results: list[EpisodeResult], csv_path: Path) -> None:
    fieldnames = episode_metric_fieldnames()
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(episode_metrics_to_row(result.metrics))


def summarize_scenario_results(scenario: ScenarioSpec, results: list[EpisodeResult]) -> dict[str, Any]:
    metrics = [result.metrics for result in results]
    rewards = [metric.reward for metric in metrics]
    episode_lengths = [metric.episode_length for metric in metrics]
    successes = [metric.success for metric in metrics]
    collisions = [metric.collision for metric in metrics]
    timeouts = [metric.timeout for metric in metrics]
    distances = [metric.distance_to_goal for metric in metrics]
    path_lengths = [metric.path_length for metric in metrics]
    path_efficiencies = [metric.path_efficiency for metric in metrics]
    min_clearances = [metric.min_clearance for metric in metrics]
    min_boundary_distances = [metric.min_boundary_distance for metric in metrics]
    mean_obstacle_penalties = [metric.mean_obstacle_potential_penalty for metric in metrics]
    mean_boundary_penalties = [metric.mean_boundary_potential_penalty for metric in metrics]
    max_obstacle_penalties = [metric.max_obstacle_potential_penalty for metric in metrics]
    max_boundary_penalties = [metric.max_boundary_potential_penalty for metric in metrics]

    return {
        "scenario": scenario.name,
        "description": scenario.description,
        "episodes": len(metrics),
        "success_rate": safe_mean(successes),
        "collision_rate": safe_mean(collisions),
        "timeout_rate": safe_mean(timeouts),
        "mean_reward": safe_mean(rewards),
        "std_reward": float(np.std(rewards)) if rewards else 0.0,
        "mean_episode_length": safe_mean(episode_lengths),
        "mean_distance_to_goal": safe_mean(distances),
        "mean_path_length": safe_mean(path_lengths),
        "mean_path_efficiency": safe_mean(path_efficiencies),
        "mean_min_clearance": safe_mean(min_clearances),
        "mean_min_boundary_distance": safe_mean(min_boundary_distances),
        "mean_obstacle_potential_penalty": safe_mean(mean_obstacle_penalties),
        "mean_boundary_potential_penalty": safe_mean(mean_boundary_penalties),
        "max_obstacle_potential_penalty": safe_max(max_obstacle_penalties),
        "max_boundary_potential_penalty": safe_max(max_boundary_penalties),
    }


def write_summary_csv(results: list[EpisodeResult], scenarios: list[ScenarioSpec], csv_path: Path) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        scenario_results = [result for result in results if result.metrics.scenario == scenario.name]
        summary_rows.append(summarize_scenario_results(scenario, scenario_results))

    fieldnames = list(summary_rows[0].keys()) if summary_rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary_rows


def write_failed_cases_csv(results: list[EpisodeResult], csv_path: Path) -> None:
    metric_fieldnames = episode_metric_fieldnames()
    fieldnames = metric_fieldnames + ["terminal_status"]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            if result.metrics.success:
                continue
            row = episode_metrics_to_row(result.metrics)
            row["terminal_status"] = result.trace.terminal_status
            writer.writerow(row)


def write_evaluation_outputs(
    results: list[EpisodeResult],
    scenarios: list[ScenarioSpec],
    paths: EvalPaths,
) -> list[dict[str, Any]]:
    write_episode_csv(results, paths.episodes_csv)
    summary_rows = write_summary_csv(results, scenarios, paths.summary_csv)
    write_failed_cases_csv(results, paths.failed_cases_csv)
    return summary_rows


def write_event_metadata(eval_args: EvalArguments, scenarios: list[ScenarioSpec]) -> Path:
    metadata = {
        "event_name": eval_args.paths.event_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(eval_args.paths.run_dir),
        "model_path": str(eval_args.paths.model_path),
        "output_root_dir": str(eval_args.paths.output_root_dir),
        "event_dir": str(eval_args.paths.output_dir),
        "base_seed": eval_args.base_seed,
        "episodes_per_scenario": eval_args.episodes_per_scenario,
        "save_visualizations": eval_args.save_visualizations,
        "scenario_count": len(scenarios),
        "scenarios": [
            {
                "name": scenario.name,
                "description": scenario.description,
            }
            for scenario in scenarios
        ],
    }
    eval_args.paths.event_metadata_json.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return eval_args.paths.event_metadata_json


def set_equal_3d_axes(ax: Any, points: list[np.ndarray]) -> None:
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


def plot_sphere(ax: Any, center: np.ndarray, radius: float, color: str, label: str | None) -> None:
    u = np.linspace(0.0, 2.0 * np.pi, 28)
    v = np.linspace(0.0, np.pi, 16)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(
        x,
        y,
        z,
        color=color,
        alpha=0.24,
        linewidth=0.35,
        edgecolor="#ffd1d1",
        shade=True,
    )
    ax.plot_wireframe(x, y, z, color="#ffd1d1", linewidth=0.25, alpha=0.45)
    ax.scatter(
        [center[0]],
        [center[1]],
        [center[2]],
        color="#ffffff",
        edgecolor=color,
        linewidth=1.2,
        s=34,
        label=label,
    )


def plot_cylinder(
    ax: Any,
    center: np.ndarray,
    radius: float,
    half_height: float,
    color: str,
    label: str | None,
    safety_radius: float | None = None,
    safety_half_height: float | None = None,
) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 36)
    if safety_radius is not None and safety_half_height is not None:
        safety_radius = float(safety_radius)
        safety_half_height = float(safety_half_height)
        if safety_radius > radius + 1e-8 or safety_half_height > half_height + 1e-8:
            safety_z_values = np.linspace(center[2] - safety_half_height, center[2] + safety_half_height, 8)
            safety_theta_grid, safety_z_grid = np.meshgrid(theta, safety_z_values)
            safety_x = center[0] + safety_radius * np.cos(safety_theta_grid)
            safety_y = center[1] + safety_radius * np.sin(safety_theta_grid)
            ax.plot_surface(
                safety_x,
                safety_y,
                safety_z_grid,
                color=color,
                alpha=0.10,
                linewidth=0.25,
                edgecolor="#ffffff",
                shade=True,
            )
            ax.plot_wireframe(safety_x, safety_y, safety_z_grid, color="#ffffff", linewidth=0.30, alpha=0.45)

    z_values = np.linspace(center[2] - half_height, center[2] + half_height, 8)
    theta_grid, z_grid = np.meshgrid(theta, z_values)
    x = center[0] + radius * np.cos(theta_grid)
    y = center[1] + radius * np.sin(theta_grid)
    ax.plot_surface(
        x,
        y,
        z_grid,
        color=color,
        alpha=0.38,
        linewidth=0.35,
        edgecolor="#ffd1d1",
        shade=True,
    )
    ax.plot_wireframe(x, y, z_grid, color="#ffd1d1", linewidth=0.35, alpha=0.62)

    cap_x = center[0] + radius * np.cos(theta)
    cap_y = center[1] + radius * np.sin(theta)
    bottom_z = np.full_like(theta, center[2] - half_height)
    top_z = np.full_like(theta, center[2] + half_height)
    ax.plot(cap_x, cap_y, bottom_z, color="#ffffff", linewidth=1.4, alpha=0.82)
    ax.plot(cap_x, cap_y, top_z, color="#ffffff", linewidth=1.4, alpha=0.82)
    ax.scatter(
        [center[0]],
        [center[1]],
        [center[2]],
        color="#ffffff",
        edgecolor=color,
        linewidth=1.2,
        s=36,
        label=label,
    )


def plot_box(ax: Any, center: np.ndarray, half_extents: np.ndarray, color: str, label: str | None) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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
    faces = [
        [vertices[index] for index in (0, 1, 2, 3)],
        [vertices[index] for index in (4, 5, 6, 7)],
        [vertices[index] for index in (0, 1, 5, 4)],
        [vertices[index] for index in (2, 3, 7, 6)],
        [vertices[index] for index in (1, 2, 6, 5)],
        [vertices[index] for index in (0, 3, 7, 4)],
    ]
    collection = Poly3DCollection(
        faces,
        facecolors=color,
        edgecolors="#ffe0e0",
        linewidths=1.1,
        alpha=0.26,
    )
    ax.add_collection3d(collection)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    for start_index, end_index in edges:
        ax.plot(
            [vertices[start_index, 0], vertices[end_index, 0]],
            [vertices[start_index, 1], vertices[end_index, 1]],
            [vertices[start_index, 2], vertices[end_index, 2]],
            color="#ffe0e0",
            linewidth=1.7,
        )
    ax.scatter(
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2],
        color="#ffffff",
        edgecolor=color,
        linewidth=0.8,
        s=14,
    )
    ax.scatter([center[0]], [center[1]], [center[2]], color="#ffffff", edgecolor=color, linewidth=1.2, s=36, label=label)


def save_terminal_state_figure(result: EpisodeResult, output_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = result.metrics
    trace = result.trace
    save_path = output_dir / f"{trace.scenario}_seed_{trace.seed}_{trace.terminal_status}.png"

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    trajectory = trace.trajectory

    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color="tab:blue", linewidth=2.0, label="trajectory")
    ax.scatter([trace.start[0]], [trace.start[1]], [trace.start[2]], color="tab:green", marker="o", s=70, label="start")
    ax.scatter([trace.goal[0]], [trace.goal[1]], [trace.goal[2]], color="tab:orange", marker="*", s=130, label="goal")
    ax.scatter(
        [trajectory[-1, 0]],
        [trajectory[-1, 1]],
        [trajectory[-1, 2]],
        color="black",
        marker="x",
        s=80,
        label="terminal",
    )

    axis_points = [trajectory.min(axis=0), trajectory.max(axis=0), trace.start, trace.goal]
    used_labels: set[str] = set()
    for obstacle in trace.obstacles:
        center = obstacle["center"]
        color = "tab:red" if obstacle["kind"] == "static" else "tab:purple"
        label_key = f"{obstacle['kind']} {obstacle['type']}"
        label = None if label_key in used_labels else label_key
        used_labels.add(label_key)
        if obstacle["type"] == "sphere":
            radius = float(obstacle["radius"])
            plot_sphere(ax, center, radius, color=color, label=label)
            axis_points.extend([center - radius, center + radius])
        elif obstacle["type"] == "cylinder":
            radius = float(obstacle["radius"])
            half_height = float(obstacle["half_height"])
            body_radius = float(obstacle.get("body_radius", radius))
            body_half_height = float(obstacle.get("body_half_height", half_height))
            plot_cylinder(
                ax,
                center,
                body_radius,
                body_half_height,
                color=color,
                label=label,
                safety_radius=radius,
                safety_half_height=half_height,
            )
            axis_points.extend([
                center + np.array([-radius, -radius, -half_height], dtype=float),
                center + np.array([radius, radius, half_height], dtype=float),
            ])
        elif obstacle["type"] == "box":
            half_extents = obstacle["half_extents"]
            plot_box(ax, center, half_extents, color=color, label=label)
            axis_points.extend([center - half_extents, center + half_extents])

    set_equal_3d_axes(ax, axis_points)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=32.0, azim=-58.0)
    ax.set_title(
        f"{trace.scenario} | seed={trace.seed} | {trace.terminal_status} | "
        f"reward={metrics.reward:.2f} | length={metrics.episode_length} | "
        f"distance={metrics.distance_to_goal:.3f}",
        fontsize=10,
    )
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def save_terminal_state_figures(results: list[EpisodeResult], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for result in results:
        saved_paths.append(save_terminal_state_figure(result, output_dir))
    return saved_paths


def style_animation_3d_axis(ax: Any) -> None:
    ax.set_facecolor(ANIMATION_PANEL_COLOR)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(ANIMATION_PANEL_COLOR)
        axis.pane.set_edgecolor(ANIMATION_GRID_COLOR)
        axis._axinfo["grid"]["color"] = ANIMATION_GRID_COLOR
        axis._axinfo["grid"]["linewidth"] = 0.6
    ax.tick_params(colors=ANIMATION_TEXT_COLOR)
    ax.xaxis.label.set_color(ANIMATION_TEXT_COLOR)
    ax.yaxis.label.set_color(ANIMATION_TEXT_COLOR)
    ax.zaxis.label.set_color(ANIMATION_TEXT_COLOR)


def style_animation_2d_axis(ax: Any) -> None:
    ax.set_facecolor(ANIMATION_PANEL_COLOR)
    ax.grid(True, color=ANIMATION_GRID_COLOR, linewidth=0.7, alpha=0.8)
    ax.tick_params(colors=ANIMATION_TEXT_COLOR)
    ax.xaxis.label.set_color(ANIMATION_TEXT_COLOR)
    ax.yaxis.label.set_color(ANIMATION_TEXT_COLOR)
    ax.spines["bottom"].set_color(ANIMATION_GRID_COLOR)
    ax.spines["top"].set_color(ANIMATION_GRID_COLOR)
    ax.spines["left"].set_color(ANIMATION_GRID_COLOR)
    ax.spines["right"].set_color(ANIMATION_GRID_COLOR)


def set_equal_xy_axes(ax: Any, points: list[np.ndarray]) -> None:
    stacked = np.vstack([np.asarray(point, dtype=float)[:2] for point in points])
    lower = stacked.min(axis=0)
    upper = stacked.max(axis=0)
    span = upper - lower
    min_span = 0.5
    for axis_index in range(2):
        if span[axis_index] < min_span:
            center = 0.5 * (lower[axis_index] + upper[axis_index])
            lower[axis_index] = center - 0.5 * min_span
            upper[axis_index] = center + 0.5 * min_span
            span[axis_index] = min_span

    padding = 0.10 * span
    lower = lower - padding
    upper = upper + padding
    ax.set_xlim(lower[0], upper[0])
    ax.set_ylim(lower[1], upper[1])
    ax.set_aspect("equal", adjustable="box")


def draw_static_scene_for_animation(ax: Any, trace: EpisodeTrace) -> list[np.ndarray]:
    trajectory = trace.trajectory
    ax.scatter([trace.start[0]], [trace.start[1]], [trace.start[2]], color=ANIMATION_START_COLOR, marker="o", s=70, label="start")
    ax.scatter([trace.goal[0]], [trace.goal[1]], [trace.goal[2]], color=ANIMATION_GOAL_COLOR, marker="*", s=130, label="goal")

    axis_points = [trajectory.min(axis=0), trajectory.max(axis=0), trace.start, trace.goal]
    used_labels: set[str] = set()
    for obstacle in trace.obstacles:
        if obstacle["kind"] == "dynamic":
            continue
        center = obstacle["center"]
        color = ANIMATION_STATIC_COLOR
        label_key = f"{obstacle['kind']} {obstacle['type']}"
        label = None if label_key in used_labels else label_key
        used_labels.add(label_key)
        if obstacle["type"] == "sphere":
            radius = float(obstacle["radius"])
            plot_sphere(ax, center, radius, color=color, label=label)
            axis_points.extend([center - radius, center + radius])
        elif obstacle["type"] == "cylinder":
            radius = float(obstacle["radius"])
            half_height = float(obstacle["half_height"])
            body_radius = float(obstacle.get("body_radius", radius))
            body_half_height = float(obstacle.get("body_half_height", half_height))
            plot_cylinder(
                ax,
                center,
                body_radius,
                body_half_height,
                color=color,
                label=label,
                safety_radius=radius,
                safety_half_height=half_height,
            )
            axis_points.extend([
                center + np.array([-radius, -radius, -half_height], dtype=float),
                center + np.array([radius, radius, half_height], dtype=float),
            ])
        elif obstacle["type"] == "box":
            half_extents = obstacle["half_extents"]
            plot_box(ax, center, half_extents, color=color, label=label)
            axis_points.extend([center - half_extents, center + half_extents])
    for obstacle_frame in trace.obstacle_history:
        for obstacle in obstacle_frame:
            center = obstacle["center"]
            if obstacle["type"] == "sphere":
                radius = float(obstacle["radius"])
                axis_points.extend([center - radius, center + radius])
            elif obstacle["type"] == "cylinder":
                radius = float(obstacle["radius"])
                half_height = float(obstacle["half_height"])
                axis_points.extend([
                    center + np.array([-radius, -radius, -half_height], dtype=float),
                    center + np.array([radius, radius, half_height], dtype=float),
                ])
            elif obstacle["type"] == "box":
                half_extents = obstacle["half_extents"]
                axis_points.extend([center - half_extents, center + half_extents])
    return axis_points


def draw_top_down_scene_for_animation(ax: Any, trace: EpisodeTrace) -> list[np.ndarray]:
    from matplotlib.patches import Circle, Rectangle

    trajectory = trace.trajectory
    ax.scatter([trace.start[0]], [trace.start[1]], color=ANIMATION_START_COLOR, marker="o", s=70, label="start", zorder=5)
    ax.scatter([trace.goal[0]], [trace.goal[1]], color=ANIMATION_GOAL_COLOR, marker="*", s=140, label="goal", zorder=5)

    axis_points = [trajectory.min(axis=0), trajectory.max(axis=0), trace.start, trace.goal]
    used_labels: set[str] = set()
    for obstacle in trace.obstacles:
        if obstacle["kind"] == "dynamic":
            continue
        center = obstacle["center"]
        label_key = f"{obstacle['kind']} {obstacle['type']}"
        label = None if label_key in used_labels else label_key
        used_labels.add(label_key)
        if obstacle["type"] == "sphere":
            radius = float(obstacle["radius"])
            patch = Circle(
                (center[0], center[1]),
                radius,
                facecolor=ANIMATION_STATIC_COLOR,
                edgecolor="#ffd1d1",
                linewidth=1.8,
                alpha=0.40,
                label=label,
            )
            ax.add_patch(patch)
            ring = Circle(
                (center[0], center[1]),
                radius * 1.06,
                facecolor="none",
                edgecolor="#ffffff",
                linewidth=0.9,
                alpha=0.60,
            )
            ax.add_patch(ring)
            ax.scatter(
                [center[0]],
                [center[1]],
                color="#ffffff",
                edgecolor=ANIMATION_STATIC_COLOR,
                linewidth=1.0,
                s=18,
                zorder=5,
            )
            axis_points.extend([center - radius, center + radius])
        elif obstacle["type"] == "cylinder":
            radius = float(obstacle["radius"])
            half_height = float(obstacle["half_height"])
            body_radius = float(obstacle.get("body_radius", radius))
            body_half_height = float(obstacle.get("body_half_height", half_height))
            safety_patch = Circle(
                (center[0], center[1]),
                radius,
                facecolor=ANIMATION_STATIC_COLOR,
                edgecolor="#ffffff",
                linewidth=1.0,
                alpha=0.16,
                label=label,
            )
            ax.add_patch(safety_patch)
            patch = Circle(
                (center[0], center[1]),
                body_radius,
                facecolor=ANIMATION_STATIC_COLOR,
                edgecolor="#ffd1d1",
                linewidth=2.1,
                alpha=0.46,
            )
            ax.add_patch(patch)
            ring = Circle(
                (center[0], center[1]),
                body_radius * 1.06,
                facecolor="none",
                edgecolor="#ffffff",
                linewidth=1.1,
                alpha=0.78,
            )
            ax.add_patch(ring)
            ax.plot(
                [center[0] - body_radius, center[0] + body_radius],
                [center[1], center[1]],
                color="#ffffff",
                linewidth=1.0,
                alpha=0.68,
                zorder=5,
            )
            ax.plot(
                [center[0], center[0]],
                [center[1] - body_radius, center[1] + body_radius],
                color="#ffffff",
                linewidth=1.0,
                alpha=0.68,
                zorder=5,
            )
            ax.text(
                center[0],
                center[1],
                f"h={2.0 * body_half_height:.1f}",
                color="#ffffff",
                fontsize=7,
                ha="center",
                va="center",
                zorder=6,
            )
            axis_points.extend([
                center + np.array([-radius, -radius, -half_height], dtype=float),
                center + np.array([radius, radius, half_height], dtype=float),
            ])
        elif obstacle["type"] == "box":
            half_extents = obstacle["half_extents"]
            x0 = center[0] - half_extents[0]
            y0 = center[1] - half_extents[1]
            width = 2.0 * half_extents[0]
            height = 2.0 * half_extents[1]
            patch = Rectangle(
                (x0, y0),
                width,
                height,
                facecolor=ANIMATION_STATIC_COLOR,
                edgecolor="#ffd1d1",
                linewidth=2.0,
                alpha=0.42,
                label=label,
            )
            ax.add_patch(patch)
            outline = Rectangle(
                (x0, y0),
                width,
                height,
                facecolor="none",
                edgecolor="#ffffff",
                linewidth=1.0,
                alpha=0.75,
            )
            ax.add_patch(outline)
            ax.plot([x0, x0 + width], [y0, y0 + height], color="#ffffff", linewidth=0.8, alpha=0.55, zorder=5)
            ax.plot([x0, x0 + width], [y0 + height, y0], color="#ffffff", linewidth=0.8, alpha=0.55, zorder=5)
            ax.scatter(
                [center[0]],
                [center[1]],
                color="#ffffff",
                edgecolor=ANIMATION_STATIC_COLOR,
                linewidth=1.0,
                s=20,
                zorder=6,
            )
            axis_points.extend([center - half_extents, center + half_extents])

    for obstacle_frame in trace.obstacle_history:
        for obstacle in obstacle_frame:
            center = obstacle["center"]
            if obstacle["type"] == "sphere":
                radius = float(obstacle["radius"])
                axis_points.extend([center - radius, center + radius])
            elif obstacle["type"] == "cylinder":
                radius = float(obstacle["radius"])
                half_height = float(obstacle["half_height"])
                axis_points.extend([
                    center + np.array([-radius, -radius, -half_height], dtype=float),
                    center + np.array([radius, radius, half_height], dtype=float),
                ])
            elif obstacle["type"] == "box":
                half_extents = obstacle["half_extents"]
                axis_points.extend([center - half_extents, center + half_extents])
    return axis_points


def dynamic_obstacle_centers_at(trace: EpisodeTrace, point_index: int) -> np.ndarray:
    if not trace.obstacle_history:
        return np.empty((0, 3), dtype=float)
    history_index = min(max(int(point_index), 0), len(trace.obstacle_history) - 1)
    centers = [
        np.asarray(obstacle["center"], dtype=float)
        for obstacle in trace.obstacle_history[history_index]
        if obstacle["kind"] == "dynamic"
    ]
    if not centers:
        return np.empty((0, 3), dtype=float)
    return np.vstack(centers)


def acceleration_direction_at(trace: EpisodeTrace, point_index: int) -> np.ndarray | None:
    if len(trace.acceleration_history) == 0:
        return None
    history_index = min(max(int(point_index), 0), len(trace.acceleration_history) - 1)
    acceleration = np.asarray(trace.acceleration_history[history_index], dtype=float)
    acceleration_norm = float(np.linalg.norm(acceleration))
    if acceleration_norm < 1e-8:
        return None
    return acceleration / acceleration_norm


def dynamic_obstacles_at(trace: EpisodeTrace, point_index: int) -> list[dict[str, Any]]:
    if not trace.obstacle_history:
        return []
    history_index = min(max(int(point_index), 0), len(trace.obstacle_history) - 1)
    return [
        obstacle
        for obstacle in trace.obstacle_history[history_index]
        if obstacle["kind"] == "dynamic"
    ]


def obstacles_at(trace: EpisodeTrace, point_index: int) -> list[dict[str, Any]]:
    if not trace.obstacle_history:
        return trace.obstacles
    history_index = min(max(int(point_index), 0), len(trace.obstacle_history) - 1)
    return trace.obstacle_history[history_index]


def nearest_sphere_surface_point(point: np.ndarray, center: np.ndarray, radius: float) -> tuple[np.ndarray, float]:
    offset = point - center
    distance_to_center = float(np.linalg.norm(offset))
    if distance_to_center < 1e-8:
        return center.copy(), -float(radius)
    surface_point = center + offset / distance_to_center * radius
    return surface_point, float(distance_to_center - radius)


def nearest_box_surface_point(point: np.ndarray, center: np.ndarray, half_extents: np.ndarray) -> tuple[np.ndarray, float]:
    lower = center - half_extents
    upper = center + half_extents
    closest = np.clip(point, lower, upper)
    outside_delta = point - closest
    outside_distance = float(np.linalg.norm(outside_delta))
    if outside_distance > 1e-8:
        return closest, outside_distance

    distances_to_faces = np.array(
        [
            point[0] - lower[0],
            upper[0] - point[0],
            point[1] - lower[1],
            upper[1] - point[1],
            point[2] - lower[2],
            upper[2] - point[2],
        ],
        dtype=float,
    )
    face_index = int(np.argmin(distances_to_faces))
    surface_point = point.copy()
    if face_index == 0:
        surface_point[0] = lower[0]
    elif face_index == 1:
        surface_point[0] = upper[0]
    elif face_index == 2:
        surface_point[1] = lower[1]
    elif face_index == 3:
        surface_point[1] = upper[1]
    elif face_index == 4:
        surface_point[2] = lower[2]
    else:
        surface_point[2] = upper[2]
    return surface_point, -float(distances_to_faces[face_index])


def nearest_cylinder_surface_point(
    point: np.ndarray,
    center: np.ndarray,
    radius: float,
    half_height: float,
) -> tuple[np.ndarray, float]:
    offset = point - center
    xy_offset = offset[:2]
    z_offset = float(offset[2])
    xy_distance = float(np.linalg.norm(xy_offset))
    if xy_distance < 1e-8:
        xy_direction = np.array([1.0, 0.0], dtype=float)
    else:
        xy_direction = xy_offset / xy_distance

    radial_distance = xy_distance - radius
    vertical_distance = abs(z_offset) - half_height
    outside_radial = max(radial_distance, 0.0)
    outside_vertical = max(vertical_distance, 0.0)
    signed_distance = float(
        np.sqrt(outside_radial * outside_radial + outside_vertical * outside_vertical)
        + min(max(radial_distance, vertical_distance), 0.0)
    )

    if xy_distance <= radius and abs(z_offset) <= half_height:
        side_gap = radius - xy_distance
        top_gap = half_height - z_offset
        bottom_gap = half_height + z_offset
        if side_gap <= top_gap and side_gap <= bottom_gap:
            surface_point = np.array(
                [center[0] + xy_direction[0] * radius, center[1] + xy_direction[1] * radius, point[2]],
                dtype=float,
            )
        elif top_gap <= bottom_gap:
            surface_point = np.array([point[0], point[1], center[2] + half_height], dtype=float)
        else:
            surface_point = np.array([point[0], point[1], center[2] - half_height], dtype=float)
        return surface_point, signed_distance

    if xy_distance <= radius:
        z_sign = 1.0 if z_offset >= 0.0 else -1.0
        surface_point = np.array([point[0], point[1], center[2] + z_sign * half_height], dtype=float)
    elif abs(z_offset) <= half_height:
        surface_point = np.array(
            [center[0] + xy_direction[0] * radius, center[1] + xy_direction[1] * radius, point[2]],
            dtype=float,
        )
    else:
        z_sign = 1.0 if z_offset >= 0.0 else -1.0
        surface_point = np.array(
            [
                center[0] + xy_direction[0] * radius,
                center[1] + xy_direction[1] * radius,
                center[2] + z_sign * half_height,
            ],
            dtype=float,
        )
    return surface_point, signed_distance


def nearest_obstacle_surface(trace: EpisodeTrace, point_index: int) -> tuple[np.ndarray | None, float]:
    point = trace.trajectory[point_index]
    nearest_surface_point: np.ndarray | None = None
    nearest_surface_distance = float("inf")
    for obstacle in obstacles_at(trace, point_index):
        if obstacle["type"] == "sphere":
            center = np.asarray(obstacle["center"], dtype=float)
            radius = float(obstacle["radius"])
            surface_point, surface_distance = nearest_sphere_surface_point(point, center, radius)
        elif obstacle["type"] == "box":
            center = np.asarray(obstacle["center"], dtype=float)
            half_extents = np.asarray(obstacle["half_extents"], dtype=float)
            surface_point, surface_distance = nearest_box_surface_point(point, center, half_extents)
        elif obstacle["type"] == "cylinder":
            center = np.asarray(obstacle["center"], dtype=float)
            radius = float(obstacle["radius"])
            half_height = float(obstacle["half_height"])
            surface_point, surface_distance = nearest_cylinder_surface_point(point, center, radius, half_height)
        else:
            continue
        if surface_distance < nearest_surface_distance:
            nearest_surface_distance = surface_distance
            nearest_surface_point = surface_point
    return nearest_surface_point, nearest_surface_distance


def episode_artifact_stem(trace: EpisodeTrace) -> str:
    return f"{trace.scenario}_seed_{trace.seed}_{trace.terminal_status}"


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def trace_json_payload(result: EpisodeResult) -> dict[str, Any]:
    trace = result.trace
    return json_ready(
        {
            "schema_version": 1,
            "scenario": trace.scenario,
            "seed": trace.seed,
            "terminal_status": trace.terminal_status,
            "metrics": episode_metrics_to_row(result.metrics),
            "start": trace.start,
            "goal": trace.goal,
            "trajectory": trace.trajectory,
            "acceleration_history": trace.acceleration_history,
            "obstacles": trace.obstacles,
            "obstacle_history": trace.obstacle_history,
            "policy_outputs": {
                "action_history": trace.action_history,
                "scaled_action_history": trace.scaled_action_history,
                "forcing_action_history": trace.forcing_action_history,
                "offside_action_history": trace.offside_action_history,
                "forcing_log_prob_history": trace.forcing_log_prob_history,
                "offside_log_prob_history": trace.offside_log_prob_history,
            },
        }
    )


def save_trace_json(result: EpisodeResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"{episode_artifact_stem(result.trace)}.json"
    payload = trace_json_payload(result)
    save_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return save_path


def save_trace_jsons(results: list[EpisodeResult], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for result in results:
        saved_paths.append(save_trace_json(result, output_dir))
    return saved_paths


def build_data_driven_trajectory_html(payload: dict[str, Any], trace_json_href: str) -> str:
    json_text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    json_text = json_text.replace("</", "<\\/")
    template = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trajectory Viewer</title>
<style>
:root {
  color-scheme: dark;
  --bg: #06111f;
  --panel: #0b1b2b;
  --grid: #27445e;
  --text: #d8ecff;
  --muted: #8fb2cc;
  --cyan: #27e8ff;
  --yellow: #fff2a8;
  --pink: #ff4fd8;
  --red: #ff5a5f;
  --green: #33d17a;
  --orange: #ffb347;
  --lime: #b6ff3b;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: Arial, Helvetica, sans-serif;
}
main {
  max-width: 1480px;
  margin: 0 auto;
  padding: 22px;
}
header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 18px;
  margin-bottom: 16px;
}
h1 {
  margin: 0 0 6px;
  font-size: 22px;
}
.sub {
  color: var(--muted);
  font-size: 13px;
}
.grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  gap: 14px;
}
.panel {
  background: var(--panel);
  border: 1px solid var(--grid);
  border-radius: 8px;
  min-width: 0;
}
.panel h2 {
  margin: 0;
  padding: 10px 12px;
  border-bottom: 1px solid var(--grid);
  font-size: 15px;
}
canvas {
  display: block;
  width: 100%;
  height: 520px;
}
.controls {
  display: grid;
  grid-template-columns: auto minmax(180px, 1fr) auto auto auto;
  align-items: center;
  gap: 10px;
  padding: 12px;
  margin-bottom: 14px;
  background: var(--panel);
  border: 1px solid var(--grid);
  border-radius: 8px;
}
button {
  height: 34px;
  padding: 0 13px;
  border: 1px solid var(--grid);
  border-radius: 6px;
  background: #10253a;
  color: var(--text);
  cursor: pointer;
}
button:hover { border-color: var(--cyan); }
input[type="range"] {
  width: 100%;
}
.readout {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 10px;
  margin-top: 14px;
}
.card {
  background: var(--panel);
  border: 1px solid var(--grid);
  border-radius: 8px;
  padding: 10px 12px;
  min-height: 64px;
}
.label {
  color: var(--muted);
  font-size: 12px;
  margin-bottom: 5px;
}
.value {
  font-family: Consolas, Menlo, monospace;
  font-size: 13px;
  line-height: 1.35;
  overflow-wrap: anywhere;
}
.legend {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 10px;
  color: var(--muted);
  font-size: 12px;
}
.dot {
  display: inline-block;
  width: 9px;
  height: 9px;
  border-radius: 50%;
  margin-right: 5px;
}
@media (max-width: 980px) {
  .grid { grid-template-columns: 1fr; }
  .controls { grid-template-columns: 1fr; }
  .readout { grid-template-columns: 1fr 1fr; }
  canvas { height: 420px; }
}
</style>
</head>
<body>
<main>
<header>
  <div>
    <h1 id="title">Trajectory Viewer</h1>
    <div class="sub" id="subtitle"></div>
  </div>
  <div class="sub">Trace JSON: <a id="jsonLink" href="%%TRACE_JSON_HREF%%">%%TRACE_JSON_HREF%%</a></div>
</header>
<section class="controls">
  <button id="playButton">Play</button>
  <input id="stepRange" type="range" min="0" max="0" value="0">
  <span class="value" id="stepText">step 0/0</span>
  <label class="sub">yaw <input id="yawRange" type="range" min="-180" max="180" value="-45"></label>
  <label class="sub">pitch <input id="pitchRange" type="range" min="-65" max="65" value="28"></label>
</section>
<section class="grid">
  <div class="panel">
    <h2>Top-down x-y View</h2>
    <canvas id="view2d"></canvas>
  </div>
  <div class="panel">
    <h2>3D Projected View</h2>
    <canvas id="view3d"></canvas>
  </div>
</section>
<section class="legend">
  <span><span class="dot" style="background: var(--green)"></span>start</span>
  <span><span class="dot" style="background: var(--orange)"></span>goal</span>
  <span><span class="dot" style="background: var(--cyan)"></span>trajectory</span>
  <span><span class="dot" style="background: var(--yellow)"></span>current</span>
  <span><span class="dot" style="background: var(--red)"></span>static obstacle</span>
  <span><span class="dot" style="background: var(--pink)"></span>dynamic obstacle</span>
  <span><span class="dot" style="background: var(--lime)"></span>acceleration</span>
</section>
<section class="readout">
  <div class="card"><div class="label">state</div><div class="value" id="stateText"></div></div>
  <div class="card"><div class="label">position</div><div class="value" id="positionText"></div></div>
  <div class="card"><div class="label">action</div><div class="value" id="actionText"></div></div>
  <div class="card"><div class="label">scaled action</div><div class="value" id="scaledActionText"></div></div>
  <div class="card"><div class="label">forcing / offside</div><div class="value" id="branchText"></div></div>
</section>
</main>
<script id="trace-data" type="application/json">%%TRACE_JSON%%</script>
<script>
const TRACE = JSON.parse(document.getElementById("trace-data").textContent);
const FPS = %%FPS%%;
const COLORS = {
  bg: "#06111f",
  panel: "#0b1b2b",
  grid: "#27445e",
  text: "#d8ecff",
  muted: "#8fb2cc",
  trajectory: "#27e8ff",
  current: "#fff2a8",
  dynamic: "#ff4fd8",
  static: "#ff5a5f",
  start: "#33d17a",
  goal: "#ffb347",
  acceleration: "#b6ff3b"
};
const trajectory = TRACE.trajectory || [];
const obstacleHistory = TRACE.obstacle_history || [];
const policy = TRACE.policy_outputs || {};
const maxStep = Math.max(0, trajectory.length - 1);
const playButton = document.getElementById("playButton");
const stepRange = document.getElementById("stepRange");
const yawRange = document.getElementById("yawRange");
const pitchRange = document.getElementById("pitchRange");
stepRange.max = String(maxStep);
let playing = false;
let timer = null;

document.getElementById("title").textContent = `${TRACE.scenario} | seed=${TRACE.seed}`;
document.getElementById("subtitle").textContent = `${TRACE.terminal_status} | reward=${fmt(TRACE.metrics.reward)} | length=${TRACE.metrics.episode_length} | path_efficiency=${fmt(TRACE.metrics.path_efficiency)}`;

function fmt(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "nan";
  return Number(value).toFixed(3);
}

function vectorText(value) {
  if (!Array.isArray(value)) return "[]";
  return "[" + value.map(v => fmt(v)).join(", ") + "]";
}

function resizeCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(320, Math.floor(rect.width * ratio));
  const height = Math.max(260, Math.floor(rect.height * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  return { width, height, ratio };
}

function obstacleExtent(obstacle) {
  const c = obstacle.center || [0, 0, 0];
  if (obstacle.type === "box") {
    const h = obstacle.half_extents || [0, 0, 0];
    return [[c[0] - h[0], c[1] - h[1], c[2] - h[2]], [c[0] + h[0], c[1] + h[1], c[2] + h[2]]];
  }
  const r = Number(obstacle.radius || 0);
  const h = Number(obstacle.half_height || r);
  return [[c[0] - r, c[1] - r, c[2] - h], [c[0] + r, c[1] + r, c[2] + h]];
}

function computeBounds() {
  const points = [];
  for (const p of trajectory) points.push(p);
  points.push(TRACE.start || [0, 0, 0]);
  points.push(TRACE.goal || [1, 1, 1]);
  for (const frame of obstacleHistory) {
    for (const obstacle of frame || []) {
      const ext = obstacleExtent(obstacle);
      points.push(ext[0], ext[1]);
    }
  }
  const lower = [Infinity, Infinity, Infinity];
  const upper = [-Infinity, -Infinity, -Infinity];
  for (const p of points) {
    for (let i = 0; i < 3; i += 1) {
      const v = Number(p[i] || 0);
      lower[i] = Math.min(lower[i], v);
      upper[i] = Math.max(upper[i], v);
    }
  }
  for (let i = 0; i < 3; i += 1) {
    if (!Number.isFinite(lower[i]) || !Number.isFinite(upper[i]) || upper[i] - lower[i] < 0.5) {
      const center = Number.isFinite(lower[i]) ? 0.5 * (lower[i] + upper[i]) : 0;
      lower[i] = center - 0.25;
      upper[i] = center + 0.25;
    }
    const padding = 0.12 * (upper[i] - lower[i]);
    lower[i] -= padding;
    upper[i] += padding;
  }
  return { lower, upper, center: lower.map((v, i) => 0.5 * (v + upper[i])) };
}

const bounds = computeBounds();

function frameObstacles(step) {
  return obstacleHistory[Math.min(step, Math.max(0, obstacleHistory.length - 1))] || TRACE.obstacles || [];
}

function map2d(point, size) {
  const pad = 42 * size.ratio;
  const spanX = bounds.upper[0] - bounds.lower[0];
  const spanY = bounds.upper[1] - bounds.lower[1];
  const scale = Math.min((size.width - 2 * pad) / spanX, (size.height - 2 * pad) / spanY);
  const x = pad + (point[0] - bounds.lower[0]) * scale;
  const y = size.height - pad - (point[1] - bounds.lower[1]) * scale;
  return [x, y, scale];
}

function clear(ctx, size) {
  ctx.fillStyle = COLORS.panel;
  ctx.fillRect(0, 0, size.width, size.height);
}

function drawGrid2d(ctx, size) {
  ctx.strokeStyle = COLORS.grid;
  ctx.lineWidth = 1 * size.ratio;
  ctx.globalAlpha = 0.55;
  for (let i = 0; i <= 8; i += 1) {
    const x = (size.width * i) / 8;
    const y = (size.height * i) / 8;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, size.height);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(size.width, y);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
}

function drawCircle(ctx, x, y, radius, color, alpha = 1) {
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.fillStyle = color;
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.arc(x, y, Math.max(2, radius), 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

function drawObstacles2d(ctx, size, step) {
  for (const obstacle of frameObstacles(step)) {
    const c = obstacle.center || [0, 0, 0];
    const [x, y, scale] = map2d(c, size);
    const color = obstacle.kind === "dynamic" ? COLORS.dynamic : COLORS.static;
    if (obstacle.type === "box") {
      const h = obstacle.half_extents || [0.1, 0.1, 0.1];
      const p1 = map2d([c[0] - h[0], c[1] - h[1], c[2]], size);
      const p2 = map2d([c[0] + h[0], c[1] + h[1], c[2]], size);
      ctx.save();
      ctx.globalAlpha = obstacle.kind === "dynamic" ? 0.34 : 0.42;
      ctx.fillStyle = color;
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.2 * size.ratio;
      ctx.fillRect(p1[0], p2[1], p2[0] - p1[0], p1[1] - p2[1]);
      ctx.strokeRect(p1[0], p2[1], p2[0] - p1[0], p1[1] - p2[1]);
      ctx.restore();
    } else {
      const r = Number(obstacle.radius || 0.1) * scale;
      drawCircle(ctx, x, y, r, color, obstacle.kind === "dynamic" ? 0.34 : 0.42);
    }
  }
}

function drawPath2d(ctx, size, step) {
  function polyline(points, color, width, alpha) {
    if (points.length < 2) return;
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = color;
    ctx.lineWidth = width * size.ratio;
    ctx.beginPath();
    points.forEach((p, i) => {
      const [x, y] = map2d(p, size);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.restore();
  }
  polyline(trajectory, COLORS.trajectory, 1.4, 0.28);
  polyline(trajectory.slice(0, step + 1), COLORS.trajectory, 3.0, 0.95);
  const start = map2d(TRACE.start || trajectory[0] || [0, 0, 0], size);
  const goal = map2d(TRACE.goal || trajectory[trajectory.length - 1] || [0, 0, 0], size);
  drawCircle(ctx, start[0], start[1], 6 * size.ratio, COLORS.start, 0.95);
  drawCircle(ctx, goal[0], goal[1], 8 * size.ratio, COLORS.goal, 0.95);
  if (trajectory[step]) {
    const current = map2d(trajectory[step], size);
    drawCircle(ctx, current[0], current[1], 7 * size.ratio, COLORS.current, 1);
  }
}

function project3d(point, size) {
  const yaw = Number(yawRange.value) * Math.PI / 180;
  const pitch = Number(pitchRange.value) * Math.PI / 180;
  const dx = point[0] - bounds.center[0];
  const dy = point[1] - bounds.center[1];
  const dz = point[2] - bounds.center[2];
  const x1 = dx * Math.cos(yaw) - dy * Math.sin(yaw);
  const y1 = dx * Math.sin(yaw) + dy * Math.cos(yaw);
  const z1 = dz;
  const y2 = y1 * Math.cos(pitch) - z1 * Math.sin(pitch);
  const z2 = y1 * Math.sin(pitch) + z1 * Math.cos(pitch);
  const maxSpan = Math.max(bounds.upper[0] - bounds.lower[0], bounds.upper[1] - bounds.lower[1], bounds.upper[2] - bounds.lower[2]);
  const scale = 0.72 * Math.min(size.width, size.height) / maxSpan;
  return [size.width / 2 + x1 * scale, size.height / 2 - z2 * scale, y2, scale];
}

function boxCorners(obstacle) {
  const c = obstacle.center || [0, 0, 0];
  const h = obstacle.half_extents || [0.1, 0.1, 0.1];
  const points = [];
  for (const sx of [-1, 1]) for (const sy of [-1, 1]) for (const sz of [-1, 1]) {
    points.push([c[0] + sx * h[0], c[1] + sy * h[1], c[2] + sz * h[2]]);
  }
  return points;
}

function drawBox3d(ctx, size, obstacle, color) {
  const points = boxCorners(obstacle).map(p => project3d(p, size));
  const edges = [[0,1],[0,2],[0,4],[3,1],[3,2],[3,7],[5,1],[5,4],[5,7],[6,2],[6,4],[6,7]];
  ctx.save();
  ctx.strokeStyle = color;
  ctx.globalAlpha = 0.66;
  ctx.lineWidth = 1.2 * size.ratio;
  for (const [a, b] of edges) {
    ctx.beginPath();
    ctx.moveTo(points[a][0], points[a][1]);
    ctx.lineTo(points[b][0], points[b][1]);
    ctx.stroke();
  }
  ctx.restore();
}

function drawObstacles3d(ctx, size, step) {
  const items = frameObstacles(step).map(obstacle => {
    const projected = project3d(obstacle.center || [0, 0, 0], size);
    return { obstacle, projected, depth: projected[2] };
  }).sort((a, b) => b.depth - a.depth);
  for (const item of items) {
    const obstacle = item.obstacle;
    const color = obstacle.kind === "dynamic" ? COLORS.dynamic : COLORS.static;
    if (obstacle.type === "box") {
      drawBox3d(ctx, size, obstacle, color);
    } else {
      const radius = Number(obstacle.radius || 0.1) * item.projected[3];
      drawCircle(ctx, item.projected[0], item.projected[1], radius, color, obstacle.kind === "dynamic" ? 0.35 : 0.42);
    }
  }
}

function drawPath3d(ctx, size, step) {
  function polyline(points, color, width, alpha) {
    if (points.length < 2) return;
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = color;
    ctx.lineWidth = width * size.ratio;
    ctx.beginPath();
    points.forEach((p, i) => {
      const projected = project3d(p, size);
      if (i === 0) ctx.moveTo(projected[0], projected[1]);
      else ctx.lineTo(projected[0], projected[1]);
    });
    ctx.stroke();
    ctx.restore();
  }
  polyline(trajectory, COLORS.trajectory, 1.4, 0.24);
  polyline(trajectory.slice(0, step + 1), COLORS.trajectory, 3.0, 0.95);
  const start = project3d(TRACE.start || trajectory[0] || [0, 0, 0], size);
  const goal = project3d(TRACE.goal || trajectory[trajectory.length - 1] || [0, 0, 0], size);
  drawCircle(ctx, start[0], start[1], 6 * size.ratio, COLORS.start, 0.95);
  drawCircle(ctx, goal[0], goal[1], 8 * size.ratio, COLORS.goal, 0.95);
  if (trajectory[step]) {
    const current = project3d(trajectory[step], size);
    drawCircle(ctx, current[0], current[1], 7 * size.ratio, COLORS.current, 1);
  }
}

function actionAt(step, key) {
  const values = policy[key] || [];
  if (!values.length) return [];
  return values[Math.max(0, Math.min(values.length - 1, step - 1))] || [];
}

function updateReadout(step) {
  const point = trajectory[step] || [];
  document.getElementById("stepText").textContent = `step ${step}/${maxStep}`;
  document.getElementById("stateText").textContent = `${TRACE.terminal_status} | success=${TRACE.metrics.success} | collision=${TRACE.metrics.collision} | timeout=${TRACE.metrics.timeout}`;
  document.getElementById("positionText").textContent = vectorText(point);
  document.getElementById("actionText").textContent = vectorText(actionAt(step, "action_history"));
  document.getElementById("scaledActionText").textContent = vectorText(actionAt(step, "scaled_action_history"));
  document.getElementById("branchText").textContent = `forcing=${vectorText(actionAt(step, "forcing_action_history"))}\noffside=${vectorText(actionAt(step, "offside_action_history"))}`;
}

function render() {
  const step = Number(stepRange.value);
  const canvas2d = document.getElementById("view2d");
  const canvas3d = document.getElementById("view3d");
  const size2d = resizeCanvas(canvas2d);
  const size3d = resizeCanvas(canvas3d);
  const ctx2d = canvas2d.getContext("2d");
  const ctx3d = canvas3d.getContext("2d");
  clear(ctx2d, size2d);
  clear(ctx3d, size3d);
  drawGrid2d(ctx2d, size2d);
  drawGrid2d(ctx3d, size3d);
  drawObstacles2d(ctx2d, size2d, step);
  drawPath2d(ctx2d, size2d, step);
  drawObstacles3d(ctx3d, size3d, step);
  drawPath3d(ctx3d, size3d, step);
  updateReadout(step);
}

function setPlaying(enabled) {
  playing = enabled;
  playButton.textContent = playing ? "Pause" : "Play";
  if (timer) window.clearInterval(timer);
  timer = null;
  if (playing) {
    timer = window.setInterval(() => {
      const next = Number(stepRange.value) + 1;
      stepRange.value = String(next > maxStep ? 0 : next);
      render();
    }, 1000 / FPS);
  }
}

playButton.addEventListener("click", () => setPlaying(!playing));
stepRange.addEventListener("input", render);
yawRange.addEventListener("input", render);
pitchRange.addEventListener("input", render);
window.addEventListener("resize", render);
render();
</script>
</body>
</html>
"""
    return (
        template.replace("%%TRACE_JSON%%", json_text)
        .replace("%%TRACE_JSON_HREF%%", html.escape(trace_json_href, quote=True))
        .replace("%%FPS%%", str(DEFAULT_HTML_FPS))
    )


def save_trajectory_html(result: EpisodeResult, output_dir: Path, trace_json_href: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = trace_json_payload(result)
    save_path = output_dir / f"{episode_artifact_stem(result.trace)}.html"
    save_path.write_text(
        build_data_driven_trajectory_html(payload, trace_json_href),
        encoding="utf-8",
    )
    return save_path


def save_trajectory_htmls(results: list[EpisodeResult], paths: EvalPaths) -> list[Path]:
    paths.trajectory_html_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for result in results:
        trace_json_href = f"../trace_json/{episode_artifact_stem(result.trace)}.json"
        saved_paths.append(save_trajectory_html(result, paths.trajectory_html_dir, trace_json_href))
    return saved_paths


def terminal_state_path_for_result(result: EpisodeResult, paths: EvalPaths) -> Path:
    return paths.terminal_state_dir / f"{episode_artifact_stem(result.trace)}.png"


def trace_json_path_for_result(result: EpisodeResult, paths: EvalPaths) -> Path:
    return paths.trace_json_dir / f"{episode_artifact_stem(result.trace)}.json"


def trajectory_html_path_for_result(result: EpisodeResult, paths: EvalPaths) -> Path:
    return paths.trajectory_html_dir / f"{episode_artifact_stem(result.trace)}.html"


def policy_output_path_for_result(result: EpisodeResult, paths: EvalPaths) -> Path:
    return paths.policy_output_dir / f"{episode_artifact_stem(result.trace)}.npz"


def relative_artifact_path(path: Path, output_dir: Path) -> str:
    try:
        return path.relative_to(output_dir).as_posix()
    except ValueError:
        return str(path)


def save_policy_output_npz(result: EpisodeResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    trace = result.trace
    save_path = output_dir / f"{episode_artifact_stem(trace)}.npz"
    np.savez_compressed(
        save_path,
        scenario=np.asarray(trace.scenario),
        seed=np.asarray(trace.seed, dtype=np.int64),
        terminal_status=np.asarray(trace.terminal_status),
        trajectory=trace.trajectory,
        acceleration_history=trace.acceleration_history,
        action_history=trace.action_history,
        scaled_action_history=trace.scaled_action_history,
        forcing_action_history=trace.forcing_action_history,
        offside_action_history=trace.offside_action_history,
        forcing_log_prob_history=trace.forcing_log_prob_history,
        offside_log_prob_history=trace.offside_log_prob_history,
    )
    return save_path


def save_policy_output_npzs(results: list[EpisodeResult], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for result in results:
        saved_paths.append(save_policy_output_npz(result, output_dir))
    return saved_paths


def visualization_manifest_row(
    result: EpisodeResult,
    paths: EvalPaths,
    include_visualizations: bool,
) -> dict[str, Any]:
    metrics = result.metrics
    row = episode_metrics_to_row(metrics)
    row["terminal_status"] = result.trace.terminal_status
    row["policy_output_npz"] = relative_artifact_path(policy_output_path_for_result(result, paths), paths.output_dir)
    row["trace_json"] = (
        relative_artifact_path(trace_json_path_for_result(result, paths), paths.output_dir)
        if include_visualizations
        else ""
    )
    row["terminal_state_png"] = (
        relative_artifact_path(terminal_state_path_for_result(result, paths), paths.output_dir)
        if include_visualizations
        else ""
    )
    row["trajectory_html"] = (
        relative_artifact_path(trajectory_html_path_for_result(result, paths), paths.output_dir)
        if include_visualizations
        else ""
    )
    return row


def write_visualization_manifest(
    results: list[EpisodeResult],
    paths: EvalPaths,
    include_visualizations: bool,
) -> Path:
    artifact_fieldnames = [
        "terminal_status",
        "terminal_state_png",
        "trace_json",
        "trajectory_html",
        "policy_output_npz",
    ]
    fieldnames = episode_metric_fieldnames() + artifact_fieldnames
    with paths.visualization_manifest_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(visualization_manifest_row(result, paths, include_visualizations))
    return paths.visualization_manifest_csv


def _format_index_float(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.3f}"


def _artifact_link(relative_path: str, label: str) -> str:
    if not relative_path:
        return ""
    return f'<a href="{html.escape(relative_path, quote=True)}">{html.escape(label)}</a>'


def write_visualization_index(
    results: list[EpisodeResult],
    paths: EvalPaths,
    include_visualizations: bool,
) -> Path:
    rows = []
    for result in results:
        manifest_row = visualization_manifest_row(result, paths, include_visualizations)
        metrics = result.metrics
        rows.append(
            "<tr>"
            f"<td>{html.escape(metrics.scenario)}</td>"
            f"<td>{metrics.seed}</td>"
            f"<td>{html.escape(result.trace.terminal_status)}</td>"
            f"<td>{metrics.success}</td>"
            f"<td>{_format_index_float(metrics.reward)}</td>"
            f"<td>{metrics.episode_length}</td>"
            f"<td>{_format_index_float(metrics.distance_to_goal)}</td>"
            f"<td>{_format_index_float(metrics.path_efficiency)}</td>"
            f"<td>{_artifact_link(manifest_row['terminal_state_png'], 'png')}</td>"
            f"<td>{_artifact_link(manifest_row['trace_json'], 'json')}</td>"
            f"<td>{_artifact_link(manifest_row['trajectory_html'], 'trajectory')}</td>"
            f"<td>{_artifact_link(manifest_row['policy_output_npz'], 'npz')}</td>"
            "</tr>"
        )

    manifest_link = relative_artifact_path(paths.visualization_manifest_csv, paths.output_dir)
    metadata_link = relative_artifact_path(paths.event_metadata_json, paths.output_dir)
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Single-agent generalization artifacts</title>
<style>
:root {{
  color-scheme: dark;
  --bg: #06111f;
  --panel: #0b1b2b;
  --grid: #27445e;
  --text: #d8ecff;
  --muted: #8fb2cc;
  --accent: #27e8ff;
}}
body {{
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: Arial, Helvetica, sans-serif;
}}
main {{
  max-width: 1280px;
  margin: 0 auto;
  padding: 28px;
}}
h1 {{
  margin: 0 0 8px;
  font-size: 26px;
  font-weight: 700;
}}
p {{
  margin: 0 0 20px;
  color: var(--muted);
}}
table {{
  width: 100%;
  border-collapse: collapse;
  background: var(--panel);
  border: 1px solid var(--grid);
}}
th, td {{
  padding: 9px 10px;
  border-bottom: 1px solid var(--grid);
  font-size: 13px;
  text-align: left;
  white-space: nowrap;
}}
th {{
  color: #ffffff;
  background: #10253a;
  position: sticky;
  top: 0;
}}
a {{
  color: var(--accent);
  text-decoration: none;
}}
a:hover {{
  text-decoration: underline;
}}
.table-wrap {{
  overflow-x: auto;
}}
</style>
</head>
<body>
<main>
<h1>Single-agent generalization artifacts</h1>
<p>
Manifest: {_artifact_link(manifest_link, 'visualization_manifest.csv')} |
Metadata: {_artifact_link(metadata_link, 'event_metadata.json')}
</p>
<div class="table-wrap">
<table>
<thead>
<tr>
<th>scenario</th>
<th>seed</th>
<th>status</th>
<th>success</th>
<th>reward</th>
<th>length</th>
<th>distance</th>
<th>efficiency</th>
<th>terminal</th>
<th>trace</th>
<th>trajectory</th>
<th>policy</th>
</tr>
</thead>
<tbody>
{chr(10).join(rows)}
</tbody>
</table>
</div>
</main>
</body>
</html>
"""
    paths.visualization_index_html.write_text(page, encoding="utf-8")
    return paths.visualization_index_html


def print_output_summary(paths: EvalPaths, summary_rows: list[dict[str, Any]]) -> None:
    print(f"episodes_csv: {paths.episodes_csv}")
    print(f"summary_csv: {paths.summary_csv}")
    print(f"failed_cases_csv: {paths.failed_cases_csv}")
    print(f"visualization_manifest_csv: {paths.visualization_manifest_csv}")
    print(f"visualization_index_html: {paths.visualization_index_html}")
    print(f"event_metadata_json: {paths.event_metadata_json}")
    for row in summary_rows:
        print(
            f"[summary] {row['scenario']} | "
            f"success_rate={row['success_rate']:.3f} | "
            f"collision_rate={row['collision_rate']:.3f} | "
            f"timeout_rate={row['timeout_rate']:.3f} | "
            f"mean_path_length={row['mean_path_length']:.3f} | "
            f"mean_path_efficiency={row['mean_path_efficiency']:.3f}"
        )


def print_run_summary(eval_args: EvalArguments) -> None:    # 打印总结信息
    print("Single-agent generalization evaluation")
    print(f"run_dir: {eval_args.paths.run_dir}")
    print(f"model_path: {eval_args.paths.model_path}")
    print(f"output_root_dir: {eval_args.paths.output_root_dir}")
    print(f"event_name: {eval_args.paths.event_name}")
    print(f"event_dir: {eval_args.paths.output_dir}")
    print(f"episodes_csv: {eval_args.paths.episodes_csv}")
    print(f"summary_csv: {eval_args.paths.summary_csv}")
    print(f"failed_cases_csv: {eval_args.paths.failed_cases_csv}")
    print(f"visualization_manifest_csv: {eval_args.paths.visualization_manifest_csv}")
    print(f"visualization_index_html: {eval_args.paths.visualization_index_html}")
    print(f"event_metadata_json: {eval_args.paths.event_metadata_json}")
    print(f"policy_output_dir: {eval_args.paths.policy_output_dir}")
    print(f"episodes_per_scenario: {eval_args.episodes_per_scenario}")
    print(f"base_seed: {eval_args.base_seed}")
    print(f"save_visualizations: {eval_args.save_visualizations}")
    if eval_args.save_visualizations:
        print(f"terminal_state_dir: {eval_args.paths.terminal_state_dir}")
        print(f"trace_json_dir: {eval_args.paths.trace_json_dir}")
        print(f"trajectory_html_dir: {eval_args.paths.trajectory_html_dir}")


def print_runtime_summary(runtime: EvalRuntime) -> None:    # 运行读取场景总结
    print("runtime_status: loaded")
    print(f"observation_space: {runtime.env.observation_space}")
    print(f"action_space: {runtime.env.action_space}")
    print(f"action_guidance_enabled: {runtime.env.env_config.action_guidance_enabled}")


def print_scenario_summary(scenarios: list[ScenarioSpec]) -> None:
    print(f"scenario_count: {len(scenarios)}")
    for index, scenario in enumerate(scenarios, start=1):
        config = scenario.config
        print(
            f"scenario[{index}]: {scenario.name} | "
            f"static_obstacle_num={config.static_obstacle_num} | "
            f"static_cylinder_num={config.static_cylinder_num} | "
            f"dynamic_obstacle_num={config.dynamic_obstacle_num} | "
            f"start_bounds={config.start_position_bounds} | "
            f"goal_bounds={config.goal_position_bounds}"
        )


def main() -> None:
    args = parse_args()
    eval_args = resolve_arguments(args)
    validate_input_paths(eval_args)
    prepare_output_directories(eval_args)
    print_run_summary(eval_args)
    scenarios = build_generalization_scenarios()
    print_scenario_summary(scenarios)
    write_event_metadata(eval_args, scenarios)

    runtime: EvalRuntime | None = None
    try:
        runtime = load_evaluation_runtime(eval_args)
        print_runtime_summary(runtime)
        results = run_batch_evaluation(
            model=runtime.model,
            scenarios=scenarios,
            episodes_per_scenario=eval_args.episodes_per_scenario,
            base_seed=eval_args.base_seed,
            deterministic=runtime.config.eval_deterministic,
        )
        summary_rows = write_evaluation_outputs(results, scenarios, eval_args.paths)
        policy_output_paths = save_policy_output_npzs(results, eval_args.paths.policy_output_dir)
        print(f"policy_output_npzs: {len(policy_output_paths)}")
        if eval_args.save_visualizations:
            terminal_state_paths = save_terminal_state_figures(results, eval_args.paths.terminal_state_dir)
            print(f"terminal_state_figures: {len(terminal_state_paths)}")
            trace_json_paths = save_trace_jsons(results, eval_args.paths.trace_json_dir)
            print(f"trace_jsons: {len(trace_json_paths)}")
            trajectory_html_paths = save_trajectory_htmls(results, eval_args.paths)
            print(f"trajectory_htmls: {len(trajectory_html_paths)}")
            write_visualization_manifest(results, eval_args.paths, include_visualizations=True)
            write_visualization_index(results, eval_args.paths, include_visualizations=True)
            print("status: batch evaluation, CSV outputs, policy outputs, trace JSONs, terminal-state figures, and data-driven HTML visualizations completed")
        else:
            write_visualization_manifest(results, eval_args.paths, include_visualizations=False)
            write_visualization_index(results, eval_args.paths, include_visualizations=False)
            print("status: batch evaluation, CSV outputs, and policy outputs completed")
        print_output_summary(eval_args.paths, summary_rows)
    finally:    # 无论前面的try通不通过，都执行finally
        close_runtime(runtime)


if __name__ == "__main__":
    main()
