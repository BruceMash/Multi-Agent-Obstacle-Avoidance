"""Visualize aggregate and representative trajectory results for frozen policy Guidance."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = Path(__file__).resolve().parent
for search_path in (REPO_ROOT, ALGO_ROOT, SCRIPTS_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from Entity.static_obstacles import AxisAlignedBoxObstacle  # noqa: E402
from experiment_config import EXPERIMENT_CONFIG as SINGLE_AGENT_CONFIG  # noqa: E402
from runner_sac import build_env as build_single_env  # noqa: E402
from runner_sac import build_model as build_single_model  # noqa: E402
from runner_sac import load_checkpoint  # noqa: E402
from scripts.evaluate_frozen_policy_waypoint_guidance import (  # noqa: E402
    ActiveWaypointState,
    build_active_goal_observations,
    generate_waypoint_sequence,
    predict_actions_without_postprocessing,
    select_safe_lookahead,
    set_dmp_active_goal_preserve_phase,
    update_stagnation,
    active_waypoint_invalid,
)
from scripts.evaluate_single_policy_aligned_multi_agent import (  # noqa: E402
    STAGE_SPECS,
    build_single_distribution_multi_config,
    build_stage_scenario,
)
from scripts.evaluate_single_policy_multi_agent import (  # noqa: E402
    _build_environment,
    build_policy_observations,
)
from Guidance.reference_point_proposal_demo import ProposalConfig  # noqa: E402


DEFAULT_ARTIFACT_DIR = (
    REPO_ROOT / "artifacts" / "frozen_policy_waypoint_guidance_20260805_104148"
)
DEFAULT_CONFIG_PATH = (
    REPO_ROOT / "configs" / "evaluation" / "frozen_policy_waypoint_guidance.json"
)
SCENARIO_ORDER = (
    "C_parallel_peer_spheres",
    "D_permuted_peer_spheres",
    "E_head_on_narrow_peer_spheres",
)
SCENARIO_SHORT = {
    "C_parallel_peer_spheres": "C 平行航路",
    "D_permuted_peer_spheres": "D 目标置换",
    "E_head_on_narrow_peer_spheres": "E 对向窄通道",
}
CONTROLLER_ORDER = ("baseline", "w1", "w2", "w3")
CONTROLLER_LABELS = {
    "baseline": "Baseline",
    "w1": "W1",
    "w2": "W2",
    "w3": "W3",
}
CONTROLLER_COLORS = {
    "baseline": "#465B7A",
    "w1": "#2A9D8F",
    "w2": "#E9C46A",
    "w3": "#E76F51",
}
AGENT_COLORS = ("#247BA0", "#F25F5C", "#70C1B3")


@dataclass
class ReplayResult:
    scenario: str
    controller: str
    seed: int
    status: str
    positions: np.ndarray
    starts: np.ndarray
    goals: np.ndarray
    waypoint_histories: list[np.ndarray]
    static_obstacles: list[Any]
    dynamic_paths: list[np.ndarray]
    collision_points: np.ndarray
    workspace_bounds: tuple[tuple[float, float, float], tuple[float, float, float]]


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#6B7280",
            "axes.labelcolor": "#374151",
            "xtick.color": "#4B5563",
            "ytick.color": "#4B5563",
            "figure.facecolor": "#F4F6F8",
            "axes.facecolor": "#FFFFFF",
            "savefig.facecolor": "#F4F6F8",
        }
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def select_representative_seeds(rows: list[dict[str, str]]) -> dict[str, int]:
    lookup = {
        (row["scenario"], int(row["seed"]), row["controller"]): row for row in rows
    }
    selections: dict[str, int] = {}
    for scenario in SCENARIO_ORDER:
        seeds = sorted({int(row["seed"]) for row in rows if row["scenario"] == scenario})
        chosen: int | None = None
        for seed in seeds:
            baseline = lookup[(scenario, seed, "baseline")]
            w3 = lookup[(scenario, seed, "w3")]
            if scenario.startswith("C_"):
                matches = baseline["team_success"] == "True" and w3["timeout"] == "True"
            elif scenario.startswith("D_"):
                matches = baseline["team_success"] == "True" and w3["team_success"] == "False"
            else:
                matches = (
                    baseline["obstacle_collision"] == "True"
                    and w3["obstacle_collision"] == "True"
                )
            if matches:
                chosen = seed
                break
        if chosen is None:
            chosen = seeds[0]
        selections[scenario] = int(chosen)
    return selections


def _annotate_bars(ax: Any, bars: Any, *, percent: bool) -> None:
    for bar in bars:
        value = float(bar.get_height())
        label = f"{value:.0%}" if percent else f"{value:.2f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + (0.018 if percent else max(0.02, 0.02 * max(value, 1.0))),
            label,
            ha="center",
            va="bottom",
            fontsize=7,
            color="#374151",
        )


def grouped_metric_plot(
    ax: Any,
    aggregates: list[dict[str, Any]],
    metric: str,
    title: str,
    *,
    percent: bool,
) -> None:
    lookup = {
        (row["scenario"], row["controller"]): float(row[metric]) for row in aggregates
    }
    x = np.arange(len(SCENARIO_ORDER), dtype=float)
    width = 0.19
    for index, controller in enumerate(CONTROLLER_ORDER):
        values = [lookup[(scenario, controller)] for scenario in SCENARIO_ORDER]
        bars = ax.bar(
            x + (index - 1.5) * width,
            values,
            width,
            label=CONTROLLER_LABELS[controller],
            color=CONTROLLER_COLORS[controller],
            edgecolor="white",
            linewidth=0.6,
        )
        _annotate_bars(ax, bars, percent=percent)
    ax.set_xticks(x, [SCENARIO_SHORT[scenario] for scenario in SCENARIO_ORDER])
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    if percent:
        ax.set_ylim(0.0, 1.14)
        ax.yaxis.set_major_formatter(lambda value, position: f"{value:.0%}")


def save_figure(fig: Any, output_base: Path) -> None:
    fig.savefig(output_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_performance(aggregates: list[dict[str, Any]], figures_dir: Path) -> None:
    metrics = (
        ("team_success_rate", "团队成功率", True),
        ("agent_success_rate_mean", "单机到达率", True),
        ("inter_agent_collision_rate", "机间碰撞率", True),
        ("obstacle_collision_rate", "障碍物碰撞率", True),
        ("timeout_rate", "超时率", True),
        ("boundary_excursion_rate", "实际越界回合率", True),
    )
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.8))
    for ax, (metric, title, percent) in zip(axes.flat, metrics):
        grouped_metric_plot(ax, aggregates, metric, title, percent=percent)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.915),
        ncol=4,
        frameon=False,
        fontsize=10,
    )
    fig.suptitle(
        "冻结单机策略接入真实 Guidance 路径点：总体性能",
        fontsize=16,
        fontweight="bold",
        y=0.988,
    )
    fig.text(
        0.5,
        0.947,
        "网络 action 原样进入 DMP；W1/W2/W3 仅改变上层 active waypoint 深度",
        ha="center",
        fontsize=10,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.875))
    save_figure(fig, figures_dir / "01_overall_performance")


def plot_diagnostics(aggregates: list[dict[str, Any]], figures_dir: Path) -> None:
    metrics = (
        ("waypoint_active_distance_mean_mean", "平均 active waypoint 距离 / m", False),
        ("waypoint_dead_end_count_mean", "每回合候选生成失败次数", False),
        ("waypoint_boundary_rejection_count_mean", "每回合边界拒绝候选数", False),
        ("final_goal_distance_mean_mean", "回合结束时目标剩余距离 / m", False),
        ("path_length_mean_mean", "平均实际路径长度 / m", False),
        ("forcing_axis_saturation_rate_mean", "forcing 轴饱和率", True),
    )
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.8))
    for ax, (metric, title, percent) in zip(axes.flat, metrics):
        grouped_metric_plot(ax, aggregates, metric, title, percent=percent)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.915),
        ncol=4,
        frameon=False,
        fontsize=10,
    )
    fig.suptitle(
        "真实路径点 Guidance：控制接口与候选可行性诊断",
        fontsize=16,
        fontweight="bold",
        y=0.988,
    )
    fig.text(
        0.5,
        0.947,
        "短 active goal 降低 DMP 推进能力；窄通道中候选生成失败显著增加",
        ha="center",
        fontsize=10,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.875))
    save_figure(fig, figures_dir / "02_guidance_diagnostics")


def replay_episode(
    *,
    model: Any,
    config: Any,
    stage: dict[str, Any],
    seed: int,
    controller: str,
    peer_radius: float,
    proposal_config: ProposalConfig,
    settings: dict[str, Any],
) -> ReplayResult:
    env = _build_environment(
        config,
        observation_mode="peer_spheres",
        peer_radius=peer_radius,
        training_distribution=False,
        include_boundaries_in_sensor=False,
        terminate_on_boundary_collision=False,
    )
    try:
        options = build_stage_scenario(config, stage, seed=seed)
        _, info = env.reset(seed=seed, options=copy.deepcopy(options))
        task_goals = np.asarray(env.goals, dtype=float).copy()
        states = [ActiveWaypointState() for _ in range(int(env.num_agents))]
        position_frames = [np.asarray(env._positions(), dtype=float).copy()]
        waypoint_histories: list[list[np.ndarray]] = [[] for _ in range(int(env.num_agents))]
        dynamic_frames: list[list[np.ndarray]] = [
            [np.asarray(obstacle.center, dtype=float).copy() for obstacle in env.dynamic_obstacles]
        ]
        terminated = False
        truncated = False
        collision_points = np.zeros((0, 3), dtype=float)
        while not (terminated or truncated):
            active_mask = np.logical_not(env.success_rewarded_mask.copy())
            positions = np.asarray(env._positions(), dtype=float)
            if controller == "baseline":
                observations = build_policy_observations(env)
            else:
                active_goals = task_goals.copy()
                for agent_index in range(int(env.num_agents)):
                    if not bool(active_mask[agent_index]):
                        set_dmp_active_goal_preserve_phase(
                            env.dmps[agent_index], task_goals[agent_index]
                        )
                        continue
                    state = states[agent_index]
                    task_distance = float(
                        np.linalg.norm(task_goals[agent_index] - positions[agent_index])
                    )
                    update_stagnation(
                        state,
                        task_distance,
                        progress_epsilon=float(settings["stagnation_progress_epsilon"]),
                    )
                    reason = None
                    if state.point is None:
                        reason = "initial"
                    elif active_waypoint_invalid(
                        env,
                        agent_index,
                        state.point,
                        boundary_margin=float(settings["boundary_margin"]),
                        collision_clearance=float(settings["collision_clearance"]),
                        segment_samples=int(settings["segment_samples"]),
                    ):
                        reason = "invalid"
                    elif float(np.linalg.norm(state.point - positions[agent_index])) <= float(
                        settings["reached_tolerance"]
                    ):
                        reason = "reached"
                    elif state.stagnation_steps >= int(settings["stagnation_steps"]):
                        reason = "stagnation"
                    if reason is not None:
                        sequence, _ = generate_waypoint_sequence(
                            env,
                            agent_index,
                            task_goals[agent_index],
                            depth=3,
                            proposal_config=proposal_config,
                            boundary_margin=float(settings["boundary_margin"]),
                            collision_clearance=float(settings["collision_clearance"]),
                        )
                        point, selected_depth, _ = select_safe_lookahead(
                            env,
                            agent_index,
                            sequence,
                            requested_depth=3,
                            boundary_margin=float(settings["boundary_margin"]),
                            collision_clearance=float(settings["collision_clearance"]),
                            segment_samples=int(settings["segment_samples"]),
                        )
                        state.point = None if point is None else point.copy()
                        state.selected_depth = int(selected_depth)
                        state.stagnation_steps = 0
                        if point is not None:
                            history = waypoint_histories[agent_index]
                            if not history or float(np.linalg.norm(history[-1] - point)) > 1.0e-6:
                                history.append(point.copy())
                    if state.point is not None:
                        active_goals[agent_index] = state.point.copy()
                    set_dmp_active_goal_preserve_phase(
                        env.dmps[agent_index], active_goals[agent_index]
                    )
                observations = build_active_goal_observations(env, active_goals)

            actions = predict_actions_without_postprocessing(
                model,
                observations,
                env.action_shape,
            )
            _, _, terminated, truncated, info = env.step(actions)
            position_frames.append(np.asarray(env._positions(), dtype=float).copy())
            dynamic_frames.append(
                [np.asarray(obstacle.center, dtype=float).copy() for obstacle in env.dynamic_obstacles]
            )
            if bool(info["collision"]):
                masks = (
                    np.asarray(info["obstacle_collision_mask"], dtype=bool)
                    | np.asarray(info["inter_agent_collision_mask"], dtype=bool)
                    | np.asarray(info["boundary_collision_mask"], dtype=bool)
                )
                collision_points = np.asarray(env._positions(), dtype=float)[masks].copy()

        status = "success" if bool(info.get("success", False)) else "collision" if bool(info["collision"]) else "timeout"
        dynamic_paths: list[np.ndarray] = []
        if env.dynamic_obstacles:
            for obstacle_index in range(len(env.dynamic_obstacles)):
                dynamic_paths.append(
                    np.stack([frame[obstacle_index] for frame in dynamic_frames], axis=0)
                )
        return ReplayResult(
            scenario=str(stage["name"]),
            controller=controller,
            seed=int(seed),
            status=status,
            positions=np.stack(position_frames, axis=0),
            starts=np.asarray(env.starts, dtype=float).copy(),
            goals=task_goals.copy(),
            waypoint_histories=[
                np.stack(history, axis=0) if history else np.zeros((0, 3), dtype=float)
                for history in waypoint_histories
            ],
            static_obstacles=copy.deepcopy(env.static_obstacles),
            dynamic_paths=dynamic_paths,
            collision_points=collision_points,
            workspace_bounds=env.env_config.workspace_bounds,
        )
    finally:
        env.close()


def _draw_obstacles(ax: Any, result: ReplayResult, dims: tuple[int, int]) -> None:
    for obstacle in result.static_obstacles:
        if isinstance(obstacle, AxisAlignedBoxObstacle):
            center = np.asarray(obstacle.center, dtype=float)
            half = np.asarray(obstacle.expanded_half_extents, dtype=float)
            ax.add_patch(
                Rectangle(
                    (center[dims[0]] - half[dims[0]], center[dims[1]] - half[dims[1]]),
                    2.0 * half[dims[0]],
                    2.0 * half[dims[1]],
                    facecolor="#9CA3AF",
                    edgecolor="#4B5563",
                    alpha=0.42,
                    linewidth=0.8,
                    zorder=1,
                )
            )
        elif hasattr(obstacle, "center"):
            center = np.asarray(obstacle.center, dtype=float)
            radius = float(
                getattr(
                    obstacle,
                    "expanded_radius",
                    getattr(obstacle, "effective_radius", getattr(obstacle, "radius", 0.2)),
                )
            )
            ax.add_patch(
                Circle(
                    (center[dims[0]], center[dims[1]]),
                    radius,
                    facecolor="#9CA3AF",
                    edgecolor="#4B5563",
                    alpha=0.42,
                    zorder=1,
                )
            )
    for path in result.dynamic_paths:
        ax.plot(
            path[:, dims[0]],
            path[:, dims[1]],
            color="#7C3AED",
            linestyle=":",
            linewidth=1.2,
            alpha=0.8,
            zorder=2,
        )
        ax.scatter(
            path[0, dims[0]],
            path[0, dims[1]],
            s=26,
            color="#7C3AED",
            marker="D",
            zorder=3,
        )


def _draw_projection(
    ax: Any,
    result: ReplayResult,
    dims: tuple[int, int],
    axis_labels: tuple[str, str],
    title: str,
) -> None:
    _draw_obstacles(ax, result, dims)
    for agent_index in range(result.positions.shape[1]):
        color = AGENT_COLORS[agent_index]
        path = result.positions[:, agent_index]
        ax.plot(
            path[:, dims[0]],
            path[:, dims[1]],
            color=color,
            linewidth=2.1,
            label=f"UAV {agent_index + 1}",
            zorder=4,
        )
        ax.scatter(
            result.starts[agent_index, dims[0]],
            result.starts[agent_index, dims[1]],
            s=52,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            marker="o",
            zorder=5,
        )
        ax.scatter(
            result.goals[agent_index, dims[0]],
            result.goals[agent_index, dims[1]],
            s=88,
            color=color,
            edgecolor="#1F2937",
            linewidth=0.6,
            marker="*",
            zorder=5,
        )
        waypoints = result.waypoint_histories[agent_index]
        if len(waypoints):
            ax.scatter(
                waypoints[:, dims[0]],
                waypoints[:, dims[1]],
                s=28,
                facecolor="none",
                edgecolor=color,
                linewidth=1.1,
                marker="s",
                zorder=5,
            )
    if len(result.collision_points):
        ax.scatter(
            result.collision_points[:, dims[0]],
            result.collision_points[:, dims[1]],
            s=95,
            color="#B91C1C",
            marker="X",
            edgecolor="white",
            linewidth=0.8,
            label="碰撞点",
            zorder=7,
        )
    lower, upper = np.asarray(result.workspace_bounds, dtype=float)
    ax.set_xlim(lower[dims[0]], upper[dims[0]])
    ax.set_ylim(lower[dims[1]], upper[dims[1]])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_title(title, fontsize=10)
    ax.grid(linestyle="--", alpha=0.22)


def plot_trajectory_comparison(
    baseline: ReplayResult,
    w3: ReplayResult,
    figures_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.8))
    _draw_projection(
        axes[0, 0], baseline, (0, 1), ("x / m", "y / m"), f"Baseline · XY · {baseline.status}"
    )
    _draw_projection(
        axes[1, 0], baseline, (0, 2), ("x / m", "z / m"), f"Baseline · XZ · {baseline.status}"
    )
    _draw_projection(
        axes[0, 1], w3, (0, 1), ("x / m", "y / m"), f"W3 · XY · {w3.status}"
    )
    _draw_projection(
        axes[1, 1], w3, (0, 2), ("x / m", "z / m"), f"W3 · XZ · {w3.status}"
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.905),
        ncol=4,
        frameon=False,
    )
    fig.suptitle(
        f"{SCENARIO_SHORT[baseline.scenario]}：典型轨迹对比（seed={baseline.seed}）",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.945,
        "圆点：起点　星形：任务终点　空心方块：W3 active waypoints　红叉：碰撞点",
        ha="center",
        fontsize=9.5,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.855))
    scenario_prefix = baseline.scenario.split("_", 1)[0].lower()
    save_figure(fig, figures_dir / f"03_{scenario_prefix}_trajectory_baseline_vs_w3")


def write_visualization_summary(
    path: Path,
    selections: dict[str, int],
    replays: dict[tuple[str, str], ReplayResult],
) -> None:
    lines = [
        "# 冻结策略 Guidance 实验可视化",
        "",
        "## 图表文件",
        "",
        "- `01_overall_performance`：成功率、碰撞率、超时率和越界率。",
        "- `02_guidance_diagnostics`：active waypoint 距离、候选失败和控制指标。",
        "- `03_c/d/e_trajectory_baseline_vs_w3`：代表性种子的 XY/XZ 轨迹。",
        "",
        "## 代表性种子",
        "",
        "| 场景 | seed | Baseline | W3 |",
        "|---|---:|---|---|",
    ]
    for scenario in SCENARIO_ORDER:
        baseline = replays[(scenario, "baseline")]
        w3 = replays[(scenario, "w3")]
        lines.append(
            f"| {SCENARIO_SHORT[scenario]} | {selections[scenario]} | {baseline.status} | {w3.status} |"
        )
    lines.extend(
        [
            "",
            "轨迹重放保持 checkpoint 冻结，策略网络 action 原样进入 DMP；可视化记录不参与控制。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def main() -> Path:
    args = parse_args()
    artifact_dir = args.artifact_dir.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    figures_dir = artifact_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    summary = json.loads((artifact_dir / "summary.json").read_text(encoding="utf-8"))
    rows = read_rows(artifact_dir / "episodes.csv")
    selections = select_representative_seeds(rows)
    plot_performance(summary["aggregates"], figures_dir)
    plot_diagnostics(summary["aggregates"], figures_dir)

    settings = json.loads(config_path.read_text(encoding="utf-8"))
    config = build_single_distribution_multi_config(
        num_agents=int(settings.get("num_agents", 3)),
        max_steps=int(settings.get("max_steps", 50)),
    )
    checkpoint = Path(settings["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = REPO_ROOT / checkpoint
    reference_env = build_single_env(
        config=SINGLE_AGENT_CONFIG,
        action_guidance_enabled=False,
    )
    model = build_single_model(reference_env, config=SINGLE_AGENT_CONFIG, verbose=0)
    load_checkpoint(model, checkpoint.resolve())
    model.actor.train(False)
    proposal_config = ProposalConfig(**dict(settings.get("proposal", {})))
    stage_lookup = {str(stage["name"]): stage for stage in STAGE_SPECS}
    replays: dict[tuple[str, str], ReplayResult] = {}
    for scenario in SCENARIO_ORDER:
        for controller in ("baseline", "w3"):
            replays[(scenario, controller)] = replay_episode(
                model=model,
                config=config,
                stage=stage_lookup[scenario],
                seed=selections[scenario],
                controller=controller,
                peer_radius=float(settings.get("peer_radius", 0.3)),
                proposal_config=proposal_config,
                settings=settings,
            )
        plot_trajectory_comparison(
            replays[(scenario, "baseline")],
            replays[(scenario, "w3")],
            figures_dir,
        )
    reference_env.close()
    write_visualization_summary(
        figures_dir / "README.md",
        selections,
        replays,
    )
    print(f"Visualizations written to: {figures_dir}")
    return figures_dir


if __name__ == "__main__":
    main()
