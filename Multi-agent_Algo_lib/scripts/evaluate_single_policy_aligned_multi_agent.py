"""Run an A-E multi-agent evaluation aligned with the historical SAC domain."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


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
from scripts.evaluate_single_policy_multi_agent import (  # noqa: E402
    MODE_LABELS,
    _jsonable,
    _sha256,
    _write_csv,
    _write_json,
    aggregate_rows,
    build_aligned_config,
    run_episode,
)


DEFAULT_CONFIG_PATH = (
    REPO_ROOT / "configs" / "evaluation" / "single_policy_aligned_multi_agent.json"
)
HISTORICAL_SINGLE_SUCCESS_RATE = 341.0 / 350.0
STAGE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "name": "A_parallel_open",
        "label": "A 平行航路·空旷",
        "observation_mode": "blind",
        "assignment": "matched",
        "obstacles": "none",
    },
    {
        "name": "B_parallel_training_obstacles",
        "label": "B 平行航路·训练障碍物",
        "observation_mode": "blind",
        "assignment": "matched",
        "obstacles": "training",
    },
    {
        "name": "C_parallel_peer_spheres",
        "label": "C 平行航路·邻机动态球",
        "observation_mode": "peer_spheres",
        "assignment": "matched",
        "obstacles": "training",
    },
    {
        "name": "D_permuted_peer_spheres",
        "label": "D 目标置换·邻机动态球",
        "observation_mode": "peer_spheres",
        "assignment": "permuted",
        "obstacles": "training",
    },
    {
        "name": "E_head_on_narrow_peer_spheres",
        "label": "E 对向窄通道·邻机动态球",
        "observation_mode": "peer_spheres",
        "assignment": "head_on",
        "obstacles": "corridor",
    },
)


def build_single_distribution_multi_config(
    *,
    num_agents: int = 3,
    max_steps: int = 50,
):
    """Restore the task ranges and reward semantics used by single-agent SAC."""
    single = SINGLE_AGENT_CONFIG
    return replace(
        build_aligned_config(num_agents=num_agents, max_steps=max_steps),
        workspace_bounds=tuple(tuple(value for value in point) for point in single.workspace_bounds),
        start_position_bounds=tuple(
            tuple(value for value in point) for point in single.start_position_bounds
        ),
        goal_position_bounds=tuple(
            tuple(value for value in point) for point in single.goal_position_bounds
        ),
        min_start_goal_distance=float(single.min_start_goal_distance),
        goal_tolerance=float(single.goal_tolerance),
        obstacle_potential_weight=float(single.obstacle_potential_weight),
        obstacle_influence_distance=float(single.obstacle_influence_distance),
        obstacle_potential_penalty_max=float(single.obstacle_potential_penalty_max),
        boundary_influence_distance=float(single.boundary_influence_distance),
        boundary_potential_weight=float(single.boundary_potential_weight),
        boundary_potential_penalty_max=float(single.boundary_potential_penalty_max),
        boundary_distance_epsilon=float(single.boundary_distance_epsilon),
        step_reward_weight=float(single.step_reward_weight),
        step_penalty=float(single.step_penalty),
        collision_margin=float(single.collision_margin),
    )


def _pairwise_min_distance(points: np.ndarray) -> float:
    if len(points) < 2:
        return float("inf")
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    return float(np.min(distances[np.triu_indices(len(points), k=1)]))


def _sample_spaced_points(
    rng: np.random.Generator,
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
    count: int,
    minimum_distance: float,
) -> np.ndarray:
    lower, upper = np.asarray(bounds, dtype=float)
    for _ in range(200):
        points: list[np.ndarray] = []
        for _ in range(int(count)):
            accepted = None
            for _ in range(5000):
                candidate = rng.uniform(lower, upper)
                if all(
                    float(np.linalg.norm(candidate - point)) >= float(minimum_distance)
                    for point in points
                ):
                    accepted = candidate
                    break
            if accepted is None:
                break
            points.append(accepted)
        if len(points) == int(count):
            return np.stack(points, axis=0).astype(float)
    raise RuntimeError("failed to sample spaced points in the historical bounds")


def _matched_goal_order(starts: np.ndarray, goals: np.ndarray) -> np.ndarray:
    """Choose the assignment with the smallest normalized lateral displacement."""
    lateral_scale = np.asarray([2.0, 0.8], dtype=float)
    best_order: tuple[int, ...] | None = None
    best_cost = float("inf")
    for order in itertools.permutations(range(len(goals))):
        ordered = goals[np.asarray(order)]
        cost = float(
            np.sum(np.linalg.norm((ordered[:, 1:] - starts[:, 1:]) / lateral_scale, axis=1))
        )
        if cost < best_cost:
            best_cost = cost
            best_order = order
    if best_order is None:
        raise RuntimeError("failed to match goals")
    return goals[np.asarray(best_order)].copy()


def sample_historical_starts_goals(
    config: Any,
    *,
    seed: int,
    assignment: str,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    starts = _sample_spaced_points(
        rng,
        config.start_position_bounds,
        int(config.num_agents),
        float(config.inter_agent_safe_distance),
    )
    goals = _sample_spaced_points(
        rng,
        config.goal_position_bounds,
        int(config.num_agents),
        float(config.inter_agent_safe_distance) + 2.0 * float(config.goal_tolerance),
    )
    goals = _matched_goal_order(starts, goals)
    if assignment == "permuted":
        shift = 1 + int(seed) % max(1, int(config.num_agents) - 1)
        goals = np.roll(goals, shift=shift, axis=0)
    distances = np.linalg.norm(goals - starts, axis=1)
    if np.any(distances < float(config.min_start_goal_distance)):
        raise RuntimeError("historical start/goal sample violates minimum route length")
    return starts, goals


def _obstacles_clear_points(
    obstacles: list[Any],
    protected_points: np.ndarray,
    clearance: float = 0.3,
) -> bool:
    return all(
        float(obstacle.signed_distance(point)) > float(clearance)
        for obstacle in obstacles
        for point in protected_points
    )


def sample_training_obstacles(
    starts: np.ndarray,
    goals: np.ndarray,
    *,
    seed: int,
) -> tuple[list[Any], list[Any]]:
    """Sample one static and one dynamic obstacle from the base SAC distribution."""
    single = replace(
        SINGLE_AGENT_CONFIG,
        training_scene_mixture_enabled=False,
        curriculum_enabled=False,
    )
    static_generator = single.build_static_obstacle_generator(single.build_fixed_box())
    dynamic_generator = single.build_dynamic_obstacle_generator()
    protected = np.concatenate([starts, goals], axis=0)
    for attempt in range(1000):
        candidate_seed = int(seed) + attempt * 104729
        static_obstacles = static_generator(starts[0], goals[0], candidate_seed)
        if not _obstacles_clear_points(static_obstacles, protected):
            continue
        dynamic_obstacles = dynamic_generator(
            starts[0],
            goals[0],
            candidate_seed + 30011,
            static_obstacles,
        )
        if not _obstacles_clear_points(dynamic_obstacles, protected):
            continue
        return copy.deepcopy(static_obstacles), copy.deepcopy(dynamic_obstacles)
    raise RuntimeError("failed to sample training obstacles clear of all UAV endpoints")


def build_head_on_corridor(
    config: Any,
    *,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    y_jitter = float(rng.uniform(-0.035, 0.035))
    z_jitter = float(rng.uniform(-0.12, 0.12))
    starts = np.asarray(
        [
            [0.2, -0.18 + y_jitter, z_jitter],
            [7.8, 0.18 - y_jitter, -z_jitter],
            [0.2, -1.80, -0.62],
        ],
        dtype=float,
    )
    goals = np.asarray(
        [
            [7.8, -0.18 + y_jitter, z_jitter],
            [0.2, 0.18 - y_jitter, -z_jitter],
            [7.8, -1.80, -0.62],
        ],
        dtype=float,
    )
    wall_half_width = 0.34
    corridor_half_width = 0.48
    static_obstacles = [
        AxisAlignedBoxObstacle(
            center=np.asarray([4.0, sign * (corridor_half_width + wall_half_width), 0.0]),
            half_extents=np.asarray([3.45, wall_half_width, 1.05]),
            safety_margin=0.02,
        )
        for sign in (-1.0, 1.0)
    ]
    return {
        "starts": starts,
        "goals": goals,
        "static_obstacles": static_obstacles,
        "dynamic_obstacles": [],
    }


def build_stage_scenario(
    config: Any,
    stage: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    if stage["assignment"] == "head_on":
        return build_head_on_corridor(config, seed=seed)
    starts, matched_goals = sample_historical_starts_goals(
        config,
        seed=seed,
        assignment="matched",
    )
    goals = matched_goals.copy()
    if stage["assignment"] == "permuted":
        shift = 1 + int(seed) % max(1, int(config.num_agents) - 1)
        goals = np.roll(goals, shift=shift, axis=0)
    static_obstacles: list[Any] = []
    dynamic_obstacles: list[Any] = []
    if stage["obstacles"] == "training":
        static_obstacles, dynamic_obstacles = sample_training_obstacles(
            starts,
            matched_goals,
            seed=seed + 50021,
        )
    return {
        "starts": starts,
        "goals": goals,
        "static_obstacles": static_obstacles,
        "dynamic_obstacles": dynamic_obstacles,
    }


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows])) if rows else 0.0


def build_paired_transition_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    transitions = (
        ("A_parallel_open", "B_parallel_training_obstacles", "加入训练障碍物"),
        ("B_parallel_training_obstacles", "C_parallel_peer_spheres", "加入邻机动态球"),
        ("C_parallel_peer_spheres", "D_permuted_peer_spheres", "置换目标分配"),
    )
    by_stage_seed = {
        (str(row["scenario"]), int(row["seed"])): row
        for row in rows
    }
    results: list[dict[str, Any]] = []
    for source, target, factor in transitions:
        seeds = sorted(
            int(row["seed"])
            for row in rows
            if row["scenario"] == source
        )
        counts = {
            "both_success": 0,
            "source_success_only": 0,
            "target_success_only": 0,
            "both_fail": 0,
            "inter_agent_collision_resolved": 0,
            "inter_agent_collision_introduced": 0,
        }
        for seed in seeds:
            source_row = by_stage_seed[(source, seed)]
            target_row = by_stage_seed[(target, seed)]
            source_success = bool(source_row["team_success"])
            target_success = bool(target_row["team_success"])
            if source_success and target_success:
                counts["both_success"] += 1
            elif source_success:
                counts["source_success_only"] += 1
            elif target_success:
                counts["target_success_only"] += 1
            else:
                counts["both_fail"] += 1
            source_collision = bool(source_row["inter_agent_collision"])
            target_collision = bool(target_row["inter_agent_collision"])
            if source_collision and not target_collision:
                counts["inter_agent_collision_resolved"] += 1
            elif target_collision and not source_collision:
                counts["inter_agent_collision_introduced"] += 1
        results.append(
            {
                "source_stage": source,
                "target_stage": target,
                "changed_factor": factor,
                "paired_episodes": len(seeds),
                **counts,
                "net_success_change": (
                    counts["target_success_only"] - counts["source_success_only"]
                ),
            }
        )
    return results


def write_stage_report(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
    checkpoint: Path,
) -> None:
    theoretical_team_rate = HISTORICAL_SINGLE_SUCCESS_RATE ** 3
    lines = [
        "# 单机分布对齐的多机分层实验",
        "",
        f"- Checkpoint：`{checkpoint}`",
        f"- 历史单机成功率：{HISTORICAL_SINGLE_SUCCESS_RATE:.2%}",
        f"- 三机独立成功率参考值：{theoretical_team_rate:.2%}",
        "- 单机语义：边界不进入 LiDAR，越界不触发 episode 终止",
        "- 所有阶段：冻结共享 SAC Actor，不使用上层调度、通信或显式邻机状态",
        "",
        "## 分层结果",
        "",
        "| 阶段 | 回合 | 团队成功率 | 单机到达率 | 机间碰撞率 | 障碍碰撞率 | 超时率 | 平均路径/m | 最小机距/m |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for aggregate in aggregates:
        lines.append(
            "| {label} | {episodes} | {team:.2%} | {agent:.2%} | {inter:.2%} | "
            "{obstacle:.2%} | {timeout:.2%} | {path:.3f} | {distance:.3f} |".format(
                label=aggregate["scenario_label"],
                episodes=aggregate["episodes"],
                team=aggregate["team_success_rate"],
                agent=aggregate["agent_success_rate_mean"],
                inter=aggregate["inter_agent_collision_rate"],
                obstacle=aggregate["obstacle_collision_rate"],
                timeout=aggregate["timeout_rate"],
                path=aggregate["path_length_mean_mean"],
                distance=aggregate["min_pairwise_distance_mean"],
            )
        )
    by_stage = {row["scenario"]: row for row in aggregates}
    transitions = build_paired_transition_rows(rows)
    stage_a = by_stage.get("A_parallel_open")
    stage_b = by_stage.get("B_parallel_training_obstacles")
    stage_c = by_stage.get("C_parallel_peer_spheres")
    stage_d = by_stage.get("D_permuted_peer_spheres")
    stage_e = by_stage.get("E_head_on_narrow_peer_spheres")
    lines.extend(["", "## 分解指标", ""])
    if stage_a:
        lines.append(
            f"- 基础三机复用差距：{theoretical_team_rate - stage_a['team_success_rate']:+.2%} "
            "（包含无障碍条件下的机间轨迹耦合）。"
        )
    if stage_a and stage_b:
        lines.append(
            f"- 训练障碍物附加损失：{stage_a['team_success_rate'] - stage_b['team_success_rate']:+.2%}。"
        )
    if stage_b and stage_c:
        lines.append(
            f"- 邻机动态球净变化：{stage_c['team_success_rate'] - stage_b['team_success_rate']:+.2%}。"
        )
    if stage_c and stage_d:
        lines.append(
            f"- 目标置换直接损失：{stage_c['team_success_rate'] - stage_d['team_success_rate']:+.2%}。"
        )
    if stage_d and stage_e:
        lines.append(
            f"- 对向窄通道附加损失：{stage_d['team_success_rate'] - stage_e['team_success_rate']:+.2%}。"
        )
    lines.extend(
        [
            "",
            "## 配对回合转移",
            "",
            "| 单一变化因素 | 两阶段均成功 | 原阶段独有成功 | 新阶段独有成功 | 两阶段均失败 | 净成功变化 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for transition in transitions:
        lines.append(
            "| {factor} | {both_success} | {source_only} | {target_only} | {both_fail} | {net:+d} |".format(
                factor=transition["changed_factor"],
                both_success=transition["both_success"],
                source_only=transition["source_success_only"],
                target_only=transition["target_success_only"],
                both_fail=transition["both_fail"],
                net=transition["net_success_change"],
            )
        )
    lines.extend(
        [
            "",
            "上述分解用于区分底层导航器复用能力与多机冲突协调能力，不把局部动态球感知等同于协调策略。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--episodes-per-stage", type=int, default=None)
    parser.add_argument("--seed-base", type=int, default=None)
    return parser.parse_args()


def main() -> Path:
    args = _parse_args()
    config_path = args.config.expanduser().resolve()
    settings = json.loads(config_path.read_text(encoding="utf-8"))
    checkpoint = _resolve_path(args.checkpoint or settings["checkpoint"]).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    episodes_per_stage = int(
        args.episodes_per_stage
        if args.episodes_per_stage is not None
        else settings.get("episodes_per_stage", 50)
    )
    if episodes_per_stage <= 0:
        raise ValueError("episodes_per_stage must be positive")
    seed_base = int(args.seed_base or settings.get("seed_base", 202608050))
    peer_radius = float(settings.get("peer_radius", 0.3))
    num_agents = int(settings.get("num_agents", 3))
    max_steps = int(settings.get("max_steps", 50))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else REPO_ROOT / "artifacts" / f"single_policy_aligned_multi_agent_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    config = build_single_distribution_multi_config(
        num_agents=num_agents,
        max_steps=max_steps,
    )
    reference_env = build_single_env(
        config=SINGLE_AGENT_CONFIG,
        action_guidance_enabled=False,
    )
    model = build_single_model(reference_env, config=SINGLE_AGENT_CONFIG, verbose=0)
    load_checkpoint(model, checkpoint)
    model.actor.train(False)

    snapshot = {
        "created_at": datetime.now().astimezone().isoformat(),
        "evaluation_config_path": config_path,
        "checkpoint": checkpoint,
        "checkpoint_sha256": _sha256(checkpoint),
        "episodes_per_stage": episodes_per_stage,
        "seed_base": seed_base,
        "peer_radius": peer_radius,
        "historical_single_success_rate": HISTORICAL_SINGLE_SUCCESS_RATE,
        "theoretical_independent_team_success_rate": HISTORICAL_SINGLE_SUCCESS_RATE ** num_agents,
        "include_boundaries_in_sensor": False,
        "terminate_on_boundary_collision": False,
        "paired_stage_design": "A-D share identical seeds and endpoint sets; B-D share obstacles",
        "stages": STAGE_SPECS,
        "aligned_config": asdict(config),
        "historical_single_agent_config": asdict(SINGLE_AGENT_CONFIG),
    }
    _write_json(output_dir / "config_snapshot.json", snapshot)

    rows: list[dict[str, Any]] = []
    total = len(STAGE_SPECS) * episodes_per_stage
    completed = 0
    for stage_index, stage in enumerate(STAGE_SPECS):
        for episode_index in range(episodes_per_stage):
            seed = seed_base + episode_index
            options = build_stage_scenario(config, stage, seed=seed)
            row = run_episode(
                model=model,
                config=config,
                suite="single_aligned_stages",
                scenario_name=str(stage["name"]),
                seed=seed,
                episode_index=episode_index,
                observation_mode=str(stage["observation_mode"]),
                peer_radius=peer_radius,
                scenario_options_override=options,
                scenario_label_override=str(stage["label"]),
                include_boundaries_in_sensor=False,
                terminate_on_boundary_collision=False,
            )
            row["stage_index"] = stage_index
            row["assignment"] = stage["assignment"]
            row["obstacle_profile"] = stage["obstacles"]
            rows.append(row)
            completed += 1
            if completed % 10 == 0 or completed == total:
                print(f"[{completed}/{total}] {stage['name']}")

    reference_env.close()
    aggregates = aggregate_rows(rows)
    aggregates.sort(key=lambda row: next(
        index for index, stage in enumerate(STAGE_SPECS) if stage["name"] == row["scenario"]
    ))
    _write_csv(output_dir / "episodes.csv", rows)
    _write_csv(output_dir / "stage_summary.csv", aggregates)
    _write_csv(output_dir / "paired_transitions.csv", build_paired_transition_rows(rows))
    _write_json(output_dir / "summary.json", aggregates)
    write_stage_report(
        output_dir / "report.md",
        rows=rows,
        aggregates=aggregates,
        checkpoint=checkpoint,
    )
    print(f"Artifacts written to: {output_dir}")
    return output_dir


if __name__ == "__main__":
    main()
