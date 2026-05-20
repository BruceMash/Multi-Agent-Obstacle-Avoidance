"""Single-agent generalization evaluation entry point.

This script is intentionally built in small modules. The current version only
parses command-line arguments, resolves artifact paths, and creates output
directories. Model loading, scenario construction, episode rollout, metrics, and
visual outputs are added in later reviewable steps.
"""

from __future__ import annotations  # 启用延后解析，避免循环引用、前向引用带来的问题

import argparse
import csv
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import numpy as np

from experiment_config import EXPERIMENT_CONFIG, SACExperimentConfig
from runner_sac import build_env, build_model, load_checkpoint


DEFAULT_MODEL_NAME = "best_eval_model.pt"
DEFAULT_OUTPUT_DIR_NAME = "generalization_eval"
DEFAULT_GIF_FPS = 10
DEFAULT_GIF_MAX_FRAMES = 40
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
    output_dir: Path
    terminal_state_dir: Path
    gif_dir: Path
    chase_gif_dir: Path
    episodes_csv: Path
    summary_csv: Path
    failed_cases_csv: Path


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
    """Trajectory data retained for later terminal-state figures and GIFs."""

    scenario: str
    seed: int
    trajectory: np.ndarray
    acceleration_history: np.ndarray
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
            "Directory for evaluation outputs. If omitted, the script uses "
            "<run-dir>/generalization_eval."
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
        help="Disable terminal-state figures and GIF exports.",
    )
    return parser.parse_args()


def resolve_arguments(args: argparse.Namespace) -> EvalArguments:   # 读取参数类
    run_dir = args.run_dir.expanduser().resolve()
    model_path = (
        args.model_path.expanduser().resolve()
        if args.model_path is not None
        else run_dir / DEFAULT_MODEL_NAME
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run_dir / DEFAULT_OUTPUT_DIR_NAME
    )

    if args.episodes_per_scenario <= 0:
        raise ValueError("--episodes-per-scenario must be positive")

    return EvalArguments(
        paths=EvalPaths(
            run_dir=run_dir,
            model_path=model_path,
            output_dir=output_dir,
            terminal_state_dir=output_dir / "terminal_states",
            gif_dir=output_dir / "gifs",
            chase_gif_dir=output_dir / "chase_gifs",
            episodes_csv=output_dir / "episodes.csv",
            summary_csv=output_dir / "summary.csv",
            failed_cases_csv=output_dir / "failed_cases.csv",
        ),
        episodes_per_scenario=int(args.episodes_per_scenario),
        base_seed=int(args.base_seed),
        save_visualizations=not bool(args.no_visualizations),
    )


def prepare_output_directories(eval_args: EvalArguments) -> None:   # 组织输出字典
    eval_args.paths.output_dir.mkdir(parents=True, exist_ok=True)
    if eval_args.save_visualizations:
        eval_args.paths.terminal_state_dir.mkdir(parents=True, exist_ok=True)
        eval_args.paths.gif_dir.mkdir(parents=True, exist_ok=True)
        eval_args.paths.chase_gif_dir.mkdir(parents=True, exist_ok=True)


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
            action, _ = model.predict(observation, deterministic=deterministic) # 模型决策网络输出
            observation, reward, terminated, truncated, info = env.step(action) # 收集环境信息
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


def make_animation_frame_indices(trajectory_length: int, max_frames: int = DEFAULT_GIF_MAX_FRAMES) -> np.ndarray:
    if trajectory_length <= 1:
        return np.array([0], dtype=int)
    frame_count = min(int(max_frames), int(trajectory_length))
    return np.unique(np.linspace(0, trajectory_length - 1, frame_count, dtype=int))


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


def plot_dynamic_sphere_range(ax: Any, center: np.ndarray, radius: float) -> list[Any]:
    u = np.linspace(0.0, 2.0 * np.pi, 18)
    v = np.linspace(0.0, np.pi, 10)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    surface = ax.plot_surface(
        x,
        y,
        z,
        color=ANIMATION_DYNAMIC_COLOR,
        alpha=0.16,
        linewidth=0.25,
        edgecolor="#ffd6ff",
        shade=True,
    )
    wire = ax.plot_wireframe(x, y, z, color="#ffd6ff", linewidth=0.25, alpha=0.55)
    return [surface, wire]


def save_trajectory_gif(result: EpisodeResult, output_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Circle

    metrics = result.metrics
    trace = result.trace
    trajectory = trace.trajectory
    frame_indices = make_animation_frame_indices(len(trajectory))
    save_path = output_dir / f"{trace.scenario}_seed_{trace.seed}_{trace.terminal_status}.gif"

    fig = plt.figure(figsize=(14, 7), facecolor=ANIMATION_BG_COLOR)
    ax_3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax_top = fig.add_subplot(1, 2, 2)

    axis_points_3d = draw_static_scene_for_animation(ax_3d, trace)
    set_equal_3d_axes(ax_3d, axis_points_3d)
    style_animation_3d_axis(ax_3d)
    ax_3d.set_xlabel("x")
    ax_3d.set_ylabel("y")
    ax_3d.set_zlabel("z")
    ax_3d.view_init(elev=34.0, azim=-56.0)

    axis_points_top = draw_top_down_scene_for_animation(ax_top, trace)
    set_equal_xy_axes(ax_top, axis_points_top)
    style_animation_2d_axis(ax_top)
    ax_top.set_xlabel("x")
    ax_top.set_ylabel("y")

    path_line_3d, = ax_3d.plot([], [], [], color=ANIMATION_TRAJECTORY_COLOR, linewidth=3.0, label="trajectory")
    current_point_3d = ax_3d.scatter(
        [],
        [],
        [],
        color=ANIMATION_CURRENT_COLOR,
        edgecolor="white",
        linewidth=0.8,
        marker="o",
        s=72,
        label="current",
    )
    dynamic_points_3d = ax_3d.scatter(
        [],
        [],
        [],
        color=ANIMATION_DYNAMIC_COLOR,
        edgecolor="white",
        linewidth=0.6,
        marker="o",
        s=70,
        label="dynamic obstacle",
    )

    path_line_top, = ax_top.plot([], [], color=ANIMATION_TRAJECTORY_COLOR, linewidth=3.2, label="trajectory", zorder=8)
    current_point_top = ax_top.scatter(
        [],
        [],
        color=ANIMATION_CURRENT_COLOR,
        edgecolor="white",
        linewidth=0.8,
        marker="o",
        s=82,
        label="current",
        zorder=6,
    )
    dynamic_points_top = ax_top.scatter(
        [],
        [],
        color=ANIMATION_DYNAMIC_COLOR,
        edgecolor="white",
        linewidth=0.6,
        marker="o",
        s=78,
        label="dynamic obstacle",
        zorder=6,
    )

    legend_3d = ax_3d.legend(loc="upper left", fontsize=8, facecolor=ANIMATION_PANEL_COLOR, edgecolor=ANIMATION_GRID_COLOR)
    legend_top = ax_top.legend(loc="upper left", fontsize=8, facecolor=ANIMATION_PANEL_COLOR, edgecolor=ANIMATION_GRID_COLOR)
    for legend in (legend_3d, legend_top):
        for text in legend.get_texts():
            text.set_color(ANIMATION_TEXT_COLOR)

    dynamic_range_artists_3d: list[Any] = []
    dynamic_range_patches_top: list[Any] = []
    clearance_line_3d, = ax_3d.plot([], [], [], color="#ffffff", linestyle="--", linewidth=1.2, alpha=0.80, label="nearest obstacle")
    clearance_line_top, = ax_top.plot([], [], color="#ffffff", linestyle="--", linewidth=1.2, alpha=0.80, zorder=7)
    acceleration_line_3d, = ax_3d.plot([], [], [], color=ANIMATION_ACCELERATION_COLOR, linewidth=2.4, alpha=0.95, label="acc direction")
    acceleration_line_top, = ax_top.plot([], [], color=ANIMATION_ACCELERATION_COLOR, linewidth=2.4, alpha=0.95, zorder=9)

    def update(frame_index: int) -> tuple[Any, Any, Any, Any, Any, Any]:
        nonlocal dynamic_range_artists_3d, dynamic_range_patches_top
        point_index = int(frame_indices[frame_index])
        partial = trajectory[: point_index + 1]
        path_line_3d.set_data(partial[:, 0], partial[:, 1])
        path_line_3d.set_3d_properties(partial[:, 2])
        path_line_top.set_data(partial[:, 0], partial[:, 1])

        current = trajectory[point_index]
        current_point_3d._offsets3d = ([current[0]], [current[1]], [current[2]])
        current_point_top.set_offsets([[current[0], current[1]]])

        dynamic_centers = dynamic_obstacle_centers_at(trace, point_index)
        if len(dynamic_centers) > 0:
            dynamic_points_3d._offsets3d = (
                dynamic_centers[:, 0],
                dynamic_centers[:, 1],
                dynamic_centers[:, 2],
            )
            dynamic_points_top.set_offsets(dynamic_centers[:, :2])
        else:
            dynamic_points_3d._offsets3d = ([], [], [])
            dynamic_points_top.set_offsets(np.empty((0, 2)))

        for artist in dynamic_range_artists_3d:
            artist.remove()
        dynamic_range_artists_3d = []
        for patch in dynamic_range_patches_top:
            patch.remove()
        dynamic_range_patches_top = []

        for obstacle in dynamic_obstacles_at(trace, point_index):
            if obstacle["type"] != "sphere":
                continue
            center = np.asarray(obstacle["center"], dtype=float)
            radius = float(obstacle["radius"])
            dynamic_range_artists_3d.extend(plot_dynamic_sphere_range(ax_3d, center, radius))
            range_patch = Circle(
                (center[0], center[1]),
                radius,
                facecolor=ANIMATION_DYNAMIC_COLOR,
                edgecolor="#ffd6ff",
                linewidth=1.4,
                alpha=0.22,
                zorder=3,
            )
            ax_top.add_patch(range_patch)
            dynamic_range_patches_top.append(range_patch)

        nearest_center, surface_distance = nearest_obstacle_surface(trace, point_index)
        if nearest_center is not None:
            clearance_line_3d.set_data([current[0], nearest_center[0]], [current[1], nearest_center[1]])
            clearance_line_3d.set_3d_properties([current[2], nearest_center[2]])
            clearance_line_top.set_data([current[0], nearest_center[0]], [current[1], nearest_center[1]])
            clearance_text = f"nearest clearance={surface_distance:.3f}"
        else:
            clearance_line_3d.set_data([], [])
            clearance_line_3d.set_3d_properties([])
            clearance_line_top.set_data([], [])
            clearance_text = "nearest clearance=n/a"

        acceleration_dir = acceleration_direction_at(trace, point_index)
        if acceleration_dir is not None:
            acceleration_scale = 0.65
            acceleration_end = current + acceleration_dir * acceleration_scale
            acceleration_line_3d.set_data([current[0], acceleration_end[0]], [current[1], acceleration_end[1]])
            acceleration_line_3d.set_3d_properties([current[2], acceleration_end[2]])
            acceleration_line_top.set_data([current[0], acceleration_end[0]], [current[1], acceleration_end[1]])
        else:
            acceleration_line_3d.set_data([], [])
            acceleration_line_3d.set_3d_properties([])
            acceleration_line_top.set_data([], [])

        title = (
            f"{trace.scenario} | seed={trace.seed} | {trace.terminal_status} | "
            f"step={point_index}/{len(trajectory) - 1} | reward={metrics.reward:.2f} | {clearance_text}"
        )
        ax_3d.set_title("3D trajectory\n" + title, fontsize=10, color=ANIMATION_TEXT_COLOR)
        ax_top.set_title("Top-down x-y view\n" + title, fontsize=10, color=ANIMATION_TEXT_COLOR)
        return (
            path_line_3d,
            current_point_3d,
            dynamic_points_3d,
            path_line_top,
            current_point_top,
            dynamic_points_top,
            clearance_line_3d,
            clearance_line_top,
            acceleration_line_3d,
            acceleration_line_top,
        )

    animation = FuncAnimation(fig, update, frames=len(frame_indices), interval=1000 / DEFAULT_GIF_FPS, blit=False)
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.08, top=0.88, wspace=0.36)
    animation.save(save_path, writer=PillowWriter(fps=DEFAULT_GIF_FPS))
    plt.close(fig)
    return save_path


def save_trajectory_gifs(results: list[EpisodeResult], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for result in results:
        saved_paths.append(save_trajectory_gif(result, output_dir))
    return saved_paths


def trajectory_direction_at(trajectory: np.ndarray, point_index: int) -> np.ndarray:
    if len(trajectory) < 2:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    if point_index > 0:
        direction = trajectory[point_index] - trajectory[point_index - 1]
    else:
        direction = trajectory[1] - trajectory[0]
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm < 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return direction / direction_norm


def set_chase_camera_axes(ax: Any, position: np.ndarray, direction: np.ndarray) -> None:
    horizontal = np.array([direction[0], direction[1], 0.0], dtype=float)
    horizontal_norm = float(np.linalg.norm(horizontal))
    if horizontal_norm < 1e-8:
        horizontal = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        horizontal = horizontal / horizontal_norm

    azimuth = float(np.degrees(np.arctan2(horizontal[1], horizontal[0])) - 180.0)
    ax.view_init(elev=18.0, azim=azimuth)

    forward_distance = 2.2
    backward_distance = 0.9
    side_distance = 1.15
    vertical_distance = 0.95
    center = position + horizontal * 0.55 + np.array([0.0, 0.0, 0.18], dtype=float)
    ax.set_xlim(center[0] - backward_distance, center[0] + forward_distance)
    ax.set_ylim(center[1] - side_distance, center[1] + side_distance)
    ax.set_zlim(center[2] - 0.45, center[2] + vertical_distance)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((forward_distance + backward_distance, 2.0 * side_distance, vertical_distance + 0.45))


def save_chase_view_gif(result: EpisodeResult, output_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    metrics = result.metrics
    trace = result.trace
    trajectory = trace.trajectory
    frame_indices = make_animation_frame_indices(len(trajectory))
    save_path = output_dir / f"{trace.scenario}_seed_{trace.seed}_{trace.terminal_status}_chase.gif"

    fig = plt.figure(figsize=(9, 6), facecolor=ANIMATION_BG_COLOR)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    axis_points = draw_static_scene_for_animation(ax, trace)
    set_equal_3d_axes(ax, axis_points)
    style_animation_3d_axis(ax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    path_line, = ax.plot([], [], [], color=ANIMATION_TRAJECTORY_COLOR, linewidth=3.0, label="trajectory")
    current_point = ax.scatter(
        [],
        [],
        [],
        color=ANIMATION_CURRENT_COLOR,
        edgecolor="white",
        linewidth=0.9,
        marker="o",
        s=90,
        label="agent",
    )
    dynamic_points = ax.scatter(
        [],
        [],
        [],
        color=ANIMATION_DYNAMIC_COLOR,
        edgecolor="white",
        linewidth=0.6,
        marker="o",
        s=78,
        label="dynamic obstacle",
    )
    clearance_line, = ax.plot([], [], [], color="#ffffff", linestyle="--", linewidth=1.4, alpha=0.85, label="nearest obstacle")
    acceleration_line, = ax.plot([], [], [], color=ANIMATION_ACCELERATION_COLOR, linewidth=2.6, alpha=0.95, label="acc direction")

    legend = ax.legend(loc="upper left", fontsize=8, facecolor=ANIMATION_PANEL_COLOR, edgecolor=ANIMATION_GRID_COLOR)
    for text in legend.get_texts():
        text.set_color(ANIMATION_TEXT_COLOR)

    dynamic_range_artists: list[Any] = []

    def update(frame_index: int) -> tuple[Any, ...]:
        nonlocal dynamic_range_artists
        point_index = int(frame_indices[frame_index])
        current = trajectory[point_index]
        direction = trajectory_direction_at(trajectory, point_index)

        partial_start = max(0, point_index - 18)
        partial = trajectory[partial_start: point_index + 1]
        path_line.set_data(partial[:, 0], partial[:, 1])
        path_line.set_3d_properties(partial[:, 2])
        current_point._offsets3d = ([current[0]], [current[1]], [current[2]])

        dynamic_centers = dynamic_obstacle_centers_at(trace, point_index)
        if len(dynamic_centers) > 0:
            dynamic_points._offsets3d = (
                dynamic_centers[:, 0],
                dynamic_centers[:, 1],
                dynamic_centers[:, 2],
            )
        else:
            dynamic_points._offsets3d = ([], [], [])

        for artist in dynamic_range_artists:
            artist.remove()
        dynamic_range_artists = []
        for obstacle in dynamic_obstacles_at(trace, point_index):
            if obstacle["type"] != "sphere":
                continue
            center = np.asarray(obstacle["center"], dtype=float)
            radius = float(obstacle["radius"])
            dynamic_range_artists.extend(plot_dynamic_sphere_range(ax, center, radius))

        nearest_surface_point, surface_distance = nearest_obstacle_surface(trace, point_index)
        if nearest_surface_point is not None:
            clearance_line.set_data([current[0], nearest_surface_point[0]], [current[1], nearest_surface_point[1]])
            clearance_line.set_3d_properties([current[2], nearest_surface_point[2]])
            clearance_text = f"nearest clearance={surface_distance:.3f}"
        else:
            clearance_line.set_data([], [])
            clearance_line.set_3d_properties([])
            clearance_text = "nearest clearance=n/a"

        acceleration_dir = acceleration_direction_at(trace, point_index)
        if acceleration_dir is not None:
            acceleration_scale = 0.65
            acceleration_end = current + acceleration_dir * acceleration_scale
            acceleration_line.set_data([current[0], acceleration_end[0]], [current[1], acceleration_end[1]])
            acceleration_line.set_3d_properties([current[2], acceleration_end[2]])
        else:
            acceleration_line.set_data([], [])
            acceleration_line.set_3d_properties([])

        set_chase_camera_axes(ax, current, direction)
        ax.set_title(
            "Chase view\n"
            f"{trace.scenario} | seed={trace.seed} | {trace.terminal_status} | "
            f"step={point_index}/{len(trajectory) - 1} | {clearance_text}",
            fontsize=10,
            color=ANIMATION_TEXT_COLOR,
        )
        return path_line, current_point, dynamic_points, clearance_line, acceleration_line, *dynamic_range_artists

    animation = FuncAnimation(fig, update, frames=len(frame_indices), interval=1000 / DEFAULT_GIF_FPS, blit=False)
    fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.88)
    animation.save(save_path, writer=PillowWriter(fps=DEFAULT_GIF_FPS))
    plt.close(fig)
    return save_path


def save_chase_view_gifs(results: list[EpisodeResult], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for result in results:
        saved_paths.append(save_chase_view_gif(result, output_dir))
    return saved_paths


def print_output_summary(paths: EvalPaths, summary_rows: list[dict[str, Any]]) -> None:
    print(f"episodes_csv: {paths.episodes_csv}")
    print(f"summary_csv: {paths.summary_csv}")
    print(f"failed_cases_csv: {paths.failed_cases_csv}")
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
    print(f"output_dir: {eval_args.paths.output_dir}")
    print(f"episodes_csv: {eval_args.paths.episodes_csv}")
    print(f"summary_csv: {eval_args.paths.summary_csv}")
    print(f"failed_cases_csv: {eval_args.paths.failed_cases_csv}")
    print(f"episodes_per_scenario: {eval_args.episodes_per_scenario}")
    print(f"base_seed: {eval_args.base_seed}")
    print(f"save_visualizations: {eval_args.save_visualizations}")
    if eval_args.save_visualizations:
        print(f"terminal_state_dir: {eval_args.paths.terminal_state_dir}")
        print(f"gif_dir: {eval_args.paths.gif_dir}")
        print(f"chase_gif_dir: {eval_args.paths.chase_gif_dir}")


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
        if eval_args.save_visualizations:
            terminal_state_paths = save_terminal_state_figures(results, eval_args.paths.terminal_state_dir)
            print(f"terminal_state_figures: {len(terminal_state_paths)}")
            gif_paths = save_trajectory_gifs(results, eval_args.paths.gif_dir)
            print(f"trajectory_gifs: {len(gif_paths)}")
            chase_gif_paths = save_chase_view_gifs(results, eval_args.paths.chase_gif_dir)
            print(f"chase_view_gifs: {len(chase_gif_paths)}")
            print("status: batch evaluation, CSV outputs, terminal-state figures, GIFs, and chase-view GIFs completed")
        else:
            print("status: batch evaluation and CSV outputs completed")
        print_output_summary(eval_args.paths, summary_rows)
    finally:    # 无论前面的try通不通过，都执行finally
        close_runtime(runtime)


if __name__ == "__main__":
    main()
