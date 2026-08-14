"""Validate FP-SHEP candidate discriminability against controlled real rollouts."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
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

from Guidance.reference_point_proposal_demo import (  # noqa: E402
    ProposalConfig,
    propose_reference_points,
)
from Environment.frozen_sac_dmp_execution import (  # noqa: E402
    freeze_policy,
    predict_frozen_actions,
)
from experiment_config import EXPERIMENT_CONFIG as SINGLE_AGENT_CONFIG  # noqa: E402
from planning.candidate_execution_benchmark import (  # noqa: E402
    FEATURE_PAIRS,
    classify_cases,
    correlation_rows,
    environment_state_fingerprint,
    evaluate_candidate_set,
    jsonable,
    score_and_summarize_candidate_set,
)
from runner_sac import build_env as build_single_env  # noqa: E402
from runner_sac import build_model as build_single_model  # noqa: E402
from runner_sac import load_checkpoint  # noqa: E402
from scripts.evaluate_single_policy_aligned_multi_agent import (  # noqa: E402
    STAGE_SPECS,
    build_single_distribution_multi_config,
    build_stage_scenario,
)
from scripts.evaluate_single_policy_multi_agent import (  # noqa: E402
    _build_environment,
    _sha256,
)


DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "evaluation" / "fp_shep_validation.json"
SCENE_ADAPTERS: dict[str, dict[str, Any]] = {
    "open": {
        "stage": "A_parallel_open", "observation_mode": "blind",
        "include_boundaries_in_sensor": False, "terminate_on_boundary_collision": False,
    },
    "bounded": {
        "stage": "A_parallel_open", "observation_mode": "blind",
        "include_boundaries_in_sensor": True, "terminate_on_boundary_collision": True,
    },
    "sparse_static": {
        "stage": "B_parallel_training_obstacles", "observation_mode": "blind",
        "include_boundaries_in_sensor": False, "terminate_on_boundary_collision": False,
        "drop_dynamic_obstacles": True,
    },
    "dense_static": {
        "stage": "E_head_on_narrow_peer_spheres", "observation_mode": "blind",
        "include_boundaries_in_sensor": False, "terminate_on_boundary_collision": False,
    },
    "dynamic_obstacle": {
        "stage": "B_parallel_training_obstacles", "observation_mode": "blind",
        "include_boundaries_in_sensor": False, "terminate_on_boundary_collision": False,
    },
    "multi_agent": {
        "stage": "D_permuted_peer_spheres", "observation_mode": "peer_spheres",
        "include_boundaries_in_sensor": False, "terminate_on_boundary_collision": False,
    },
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(jsonable(value), ensure_ascii=False)
                if isinstance(value, (list, tuple, dict, np.ndarray)) else value
                for key, value in row.items()
            })


def _finite_mean(values: list[Any]) -> float | None:
    array = np.asarray([value for value in values if value is not None], dtype=float)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else None


def _finite_median(values: list[Any]) -> float | None:
    array = np.asarray([value for value in values if value is not None], dtype=float)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if array.size else None


def _stage(name: str) -> dict[str, Any]:
    return next(stage for stage in STAGE_SPECS if stage["name"] == name)


def build_validation_environment(
    *,
    config: Any,
    scene_type: str,
    seed: int,
    peer_radius: float,
) -> tuple[Any, dict[str, Any]]:
    if scene_type not in SCENE_ADAPTERS:
        raise ValueError(f"unknown validation scene type: {scene_type}")
    adapter = SCENE_ADAPTERS[scene_type]
    stage = _stage(str(adapter["stage"]))
    options = build_stage_scenario(config, stage, seed=int(seed))
    if adapter.get("drop_dynamic_obstacles", False):
        options["dynamic_obstacles"] = []
    env = _build_environment(
        config,
        observation_mode=str(adapter["observation_mode"]),
        peer_radius=float(peer_radius),
        training_distribution=False,
        include_boundaries_in_sensor=bool(adapter["include_boundaries_in_sensor"]),
        terminate_on_boundary_collision=bool(adapter["terminate_on_boundary_collision"]),
    )
    env.reset(seed=int(seed), options=copy.deepcopy(options))
    metadata = {
        "scene_type": scene_type,
        "source_stage": stage["name"],
        "observation_mode": adapter["observation_mode"],
        "include_boundaries_in_sensor": adapter["include_boundaries_in_sensor"],
        "terminate_on_boundary_collision": adapter["terminate_on_boundary_collision"],
        "static_obstacle_count": len(options["static_obstacles"]),
        "dynamic_obstacle_count": len(options["dynamic_obstacles"]),
    }
    return env, metadata


def _policy_observations(env: Any) -> np.ndarray:
    from planning.candidate_execution_benchmark import _active_goal_observations

    goals = np.stack([np.asarray(dmp.goal, dtype=float) for dmp in env.dmps])
    return _active_goal_observations(env, goals)


def advance_to_evaluation_state(env: Any, policy: Any, steps: int) -> bool:
    for _ in range(int(steps)):
        observations = _policy_observations(env)
        actions = predict_frozen_actions(policy, observations, expected_shape=tuple(env.action_shape))
        _, _, terminated, truncated, _ = env.step(actions)
        if terminated or truncated:
            return False
    return True


def _proposals(env: Any, agent_index: int, config: ProposalConfig) -> list[Any]:
    packet = env.latest_sensor_packets[agent_index]
    if packet is None:
        raise RuntimeError("environment must be reset before proposal generation")
    return propose_reference_points(
        env.dynamics[agent_index].p,
        env.goals[agent_index],
        env.dynamics[agent_index].v,
        packet,
        env.sensors[agent_index],
        config,
        env.env_config.goal_tolerance,
    )


def _scene_snapshot(env: Any) -> dict[str, Any]:
    def obstacle_record(obstacle: Any) -> dict[str, Any]:
        record: dict[str, Any] = {"type": type(obstacle).__name__}
        for key, value in vars(obstacle).items():
            if isinstance(value, (str, int, float, bool, type(None), np.ndarray, list, tuple)):
                record[key] = copy.deepcopy(value)
        return record

    return {
        "workspace_bounds": copy.deepcopy(env.env_config.workspace_bounds),
        "starts": np.asarray(env.starts, dtype=float).copy(),
        "goals": np.asarray(env.goals, dtype=float).copy(),
        "positions": np.stack([np.asarray(item.p, dtype=float) for item in env.dynamics]),
        "velocities": np.stack([np.asarray(item.v, dtype=float) for item in env.dynamics]),
        "static_obstacles": [obstacle_record(item) for item in env.static_obstacles],
        "dynamic_obstacles": [obstacle_record(item) for item in env.dynamic_obstacles],
        "diagnostic_only_not_preview_input": True,
    }


def aggregate_summaries(
    candidate_rows: list[dict[str, Any]],
    scenario_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_h: list[dict[str, Any]] = []
    by_scene: list[dict[str, Any]] = []
    for field, values in (
        ("H", sorted({int(row["H"]) for row in scenario_rows})),
        ("scenario_type", sorted({str(row["scenario_type"]) for row in scenario_rows})),
    ):
        destination = by_h if field == "H" else by_scene
        for value in values:
            summaries = [row for row in scenario_rows if row[field] == value]
            candidates = [row for row in candidate_rows if row[field] == value]
            destination.append({
                field: value,
                "state_count": len(summaries),
                "candidate_count": len(candidates),
                "mean_preview_real_spearman": _finite_mean([row["preview_real_spearman"] for row in summaries]),
                "median_preview_real_spearman": _finite_median([row["preview_real_spearman"] for row in summaries]),
                "mean_geometry_real_spearman": _finite_mean([row["geometry_real_spearman"] for row in summaries]),
                "preview_top1_accuracy": _finite_mean([row["preview_top1_match"] for row in summaries]),
                "geometry_top1_accuracy": _finite_mean([row["geometry_top1_match"] for row in summaries]),
                "preview_top3_hit_rate": _finite_mean([row["preview_top3_hit"] for row in summaries]),
                "geometry_top3_hit_rate": _finite_mean([row["geometry_top3_hit"] for row in summaries]),
                "preview_pairwise_accuracy": _finite_mean([row["preview_pairwise_accuracy"] for row in summaries]),
                "geometry_pairwise_accuracy": _finite_mean([row["geometry_pairwise_accuracy"] for row in summaries]),
                "mean_preview_runtime_ms": _finite_mean([row["preview_runtime_ms"] for row in candidates]),
                "total_preview_runtime_ms": _finite_mean([row["total_preview_runtime_ms"] for row in summaries]),
                "mean_trajectory_error": _finite_mean([row["mean_trajectory_error"] for row in candidates]),
                "terminal_position_error": _finite_mean([row["terminal_position_error"] for row in candidates]),
                "mean_trajectory_error_at_effective_horizon": _finite_mean([
                    row["mean_trajectory_error_at_effective_horizon"] for row in candidates
                ]),
                "terminal_position_error_at_effective_horizon": _finite_mean([
                    row["terminal_position_error_at_effective_horizon"] for row in candidates
                ]),
                "full_horizon_trajectory_fraction": _finite_mean([
                    row["trajectory_error_is_full_horizon"] for row in candidates
                ]),
                "collision_rate": _finite_mean([row["collision"] for row in candidates]),
                "candidate_discriminability": {
                    name: {
                        "mean_preview_std": _finite_mean([
                            row[f"{name}_preview_std"] for row in summaries
                        ]),
                        "mean_preview_normalized_range": _finite_mean([
                            row[f"{name}_preview_normalized_range"] for row in summaries
                        ]),
                    }
                    for name, _, _ in FEATURE_PAIRS
                },
                "feature_correlations": correlation_rows(candidates),
            })
    return {
        "overall": {
            "scenario_state_count": len(scenario_rows),
            "candidate_count": len(candidate_rows),
            "mean_preview_real_spearman": _finite_mean([row["preview_real_spearman"] for row in scenario_rows]),
            "median_preview_real_spearman": _finite_median([row["preview_real_spearman"] for row in scenario_rows]),
            "mean_geometry_real_spearman": _finite_mean([row["geometry_real_spearman"] for row in scenario_rows]),
            "median_geometry_real_spearman": _finite_median([row["geometry_real_spearman"] for row in scenario_rows]),
            "preview_top1_accuracy": _finite_mean([row["preview_top1_match"] for row in scenario_rows]),
            "geometry_top1_accuracy": _finite_mean([row["geometry_top1_match"] for row in scenario_rows]),
            "preview_top3_hit_rate": _finite_mean([row["preview_top3_hit"] for row in scenario_rows]),
            "geometry_top3_hit_rate": _finite_mean([row["geometry_top3_hit"] for row in scenario_rows]),
            "preview_pairwise_accuracy": _finite_mean([row["preview_pairwise_accuracy"] for row in scenario_rows]),
            "geometry_pairwise_accuracy": _finite_mean([row["geometry_pairwise_accuracy"] for row in scenario_rows]),
            "mean_preview_runtime_ms": _finite_mean([row["preview_runtime_ms"] for row in candidate_rows]),
            "total_preview_runtime_ms": float(np.sum([
                float(row["preview_runtime_ms"]) for row in candidate_rows
            ])) if candidate_rows else None,
            "collision_rate": _finite_mean([row["collision"] for row in candidate_rows]),
            "full_horizon_trajectory_fraction": _finite_mean([
                row["trajectory_error_is_full_horizon"] for row in candidate_rows
            ]),
            "mean_trajectory_error": _finite_mean([
                row["mean_trajectory_error"] for row in candidate_rows
            ]),
            "terminal_position_error": _finite_mean([
                row["terminal_position_error"] for row in candidate_rows
            ]),
            "mean_trajectory_error_at_effective_horizon": _finite_mean([
                row["mean_trajectory_error_at_effective_horizon"] for row in candidate_rows
            ]),
            "terminal_position_error_at_effective_horizon": _finite_mean([
                row["terminal_position_error_at_effective_horizon"] for row in candidate_rows
            ]),
            "candidate_discriminability": {
                name: {
                    "mean_preview_std": _finite_mean([
                        row[f"{name}_preview_std"] for row in scenario_rows
                    ]),
                    "mean_preview_normalized_range": _finite_mean([
                        row[f"{name}_preview_normalized_range"] for row in scenario_rows
                    ]),
                }
                for name, _, _ in FEATURE_PAIRS
            },
            "feature_correlations": correlation_rows(candidate_rows),
        },
        "per_H": by_h,
        "per_scene": by_scene,
    }


def _save_trajectory(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        candidate_xyz=np.asarray(payload["candidate_xyz"], dtype=float),
        preview_positions=np.asarray(payload["preview_positions"], dtype=float),
        real_positions=np.asarray(payload["real_positions"], dtype=float),
        real_velocities=np.asarray(payload["real_velocities"], dtype=float),
        current_position=np.asarray(payload["current_position"], dtype=float),
        task_goal=np.asarray(payload["task_goal"], dtype=float),
        visible_surface_points=np.asarray(payload["visible_surface_points"], dtype=float),
        initial_current_scan=np.asarray(payload["initial_current_scan"], dtype=float),
        initial_previous_scan=np.asarray(payload["initial_previous_scan"], dtype=float),
        neighbor_positions=np.asarray(payload["neighbor_positions"], dtype=float),
        neighbor_velocities=np.asarray(payload["neighbor_velocities"], dtype=float),
        initial_sensor_observation=np.asarray(payload["initial_sensor_observation"], dtype=float),
    )


def _save_case_bundles(
    output_dir: Path,
    *,
    case_kind: str,
    cases: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    trajectories: list[dict[str, Any]],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for index, case in enumerate(cases):
        case_dir = output_dir / "cases" / f"{case_kind}_{index:03d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        key = (case["scenario_id"], case["seed"], case["agent_id"], case["H"])
        rows = [
            row for row in candidate_rows
            if (row["scenario_id"], row["seed"], row["agent_id"], row["H"]) == key
        ]
        paths = [
            item for item in trajectories
            if (item["scenario_id"], item["seed"], item["agent_id"], item["H"]) == key
        ]
        _write_json(case_dir / "case.json", case)
        _write_csv(case_dir / "candidate_metrics.csv", rows)
        if paths:
            _write_json(case_dir / "initial_scene.json", paths[0]["scene_snapshot"])
        for payload in paths:
            _save_trajectory(
                case_dir / f"candidate_{payload['candidate_id']}.npz",
                payload,
            )
        if not paths:
            continue
        real_best = int(case["real_best_candidate"])
        geometry_best = int(case["geometry_best_candidate"])
        preview_best = int(case["preview_best_candidate"])
        emphasized = {real_best, geometry_best, preview_best}
        fig = plt.figure(figsize=(9.0, 6.8))
        axis = fig.add_subplot(111, projection="3d")
        for payload in paths:
            candidate_id = int(payload["candidate_id"])
            preview_path = np.asarray(payload["preview_positions"], dtype=float)
            real_path = np.asarray(payload["real_positions"], dtype=float)
            alpha = 0.95 if candidate_id in emphasized else 0.18
            width = 2.0 if candidate_id in emphasized else 0.7
            axis.plot(*preview_path.T, alpha=alpha, linewidth=width)
            if candidate_id in emphasized:
                axis.plot(*real_path.T, linestyle="--", alpha=0.95, linewidth=width)
                candidate = np.asarray(payload["candidate_xyz"], dtype=float)
                axis.text(*candidate, f"c{candidate_id}", fontsize=8)
        sample = paths[0]
        surfaces = np.asarray(sample["visible_surface_points"], dtype=float)
        if surfaces.size:
            axis.scatter(*surfaces.T, s=4, alpha=0.18, color="gray", label="visible surfaces")
        axis.scatter(*sample["current_position"], marker="o", s=60, label="current UAV")
        axis.scatter(*sample["task_goal"], marker="*", s=100, label="task goal")
        axis.set_title(
            f"{case_kind}: geometry c{geometry_best}, preview c{preview_best}, real c{real_best}"
        )
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        axis.set_zlabel("z (m)")
        axis.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(case_dir / "trajectory_comparison.png", dpi=180)
        plt.close(fig)

        ordered = sorted(rows, key=lambda row: int(row["candidate_id"]))
        ids = np.arange(len(ordered))
        width = 0.25
        fig, axis = plt.subplots(figsize=(8.0, 4.6))
        axis.bar(ids - width, [row["geometry_rank"] for row in ordered], width, label="Geometry rank")
        axis.bar(ids, [row["preview_rank"] for row in ordered], width, label="Preview rank")
        axis.bar(ids + width, [row["real_rank"] for row in ordered], width, label="Real rank")
        axis.set_xlabel("Candidate ID")
        axis.set_ylabel("Rank (1 is best)")
        axis.set_xticks(ids)
        axis.invert_yaxis()
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(case_dir / "ranking_comparison.png", dpi=180)
        plt.close(fig)


def _plot_results(
    output_dir: Path,
    candidate_rows: list[dict[str, Any]],
    h_rows: list[dict[str, Any]],
    trajectories: list[dict[str, Any]],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    if candidate_rows:
        preview = np.asarray([row["J_preview"] for row in candidate_rows], dtype=float)
        real = np.asarray([row["J_real"] for row in candidate_rows], dtype=float)
        geometry = np.asarray([row["proposal_score"] for row in candidate_rows], dtype=float)
        for name, x, label in (
            ("j_preview_vs_real.png", preview, "J preview"),
            ("geometry_vs_real.png", geometry, "Proposal geometry score"),
        ):
            fig, axis = plt.subplots(figsize=(6.2, 5.0))
            axis.scatter(x, real, alpha=0.55, s=18)
            axis.set_xlabel(label)
            axis.set_ylabel("J real")
            axis.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(figures / name, dpi=180)
            plt.close(fig)
    if h_rows:
        h = [row["H"] for row in h_rows]
        correlation = [np.nan if row["mean_preview_real_spearman"] is None else row["mean_preview_real_spearman"] for row in h_rows]
        runtime = [np.nan if row["mean_preview_runtime_ms"] is None else row["mean_preview_runtime_ms"] for row in h_rows]
        fig, left = plt.subplots(figsize=(7.0, 5.0))
        right = left.twinx()
        left.plot(h, correlation, marker="o", color="#1261A0", label="Spearman")
        right.plot(h, runtime, marker="s", color="#D1495B", label="Runtime")
        left.set_xlabel("Preview horizon H")
        left.set_ylabel("Mean Spearman", color="#1261A0")
        right.set_ylabel("Mean runtime per candidate (ms)", color="#D1495B")
        left.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(figures / "horizon_quality_runtime.png", dpi=180)
        plt.close(fig)
    if trajectories:
        first_key = (trajectories[0]["scenario_id"], trajectories[0]["seed"], trajectories[0]["agent_id"], trajectories[0]["H"])
        selected = [item for item in trajectories if (item["scenario_id"], item["seed"], item["agent_id"], item["H"]) == first_key]
        fig = plt.figure(figsize=(8.0, 6.2))
        axis = fig.add_subplot(111, projection="3d")
        for item in selected:
            path = np.asarray(item["preview_positions"], dtype=float)
            axis.plot(path[:, 0], path[:, 1], path[:, 2], alpha=0.8)
            candidate = np.asarray(item["candidate_xyz"], dtype=float)
            axis.scatter(*candidate, s=25)
        axis.scatter(*selected[0]["current_position"], marker="o", s=70, label="current UAV")
        axis.scatter(*selected[0]["task_goal"], marker="*", s=100, label="task goal")
        axis.set_title("Candidate FP-SHEP trajectories")
        axis.legend()
        fig.tight_layout()
        fig.savefig(figures / "candidate_preview_trajectories.png", dpi=180)
        plt.close(fig)
        sample = selected[0]
        preview_path = np.asarray(sample["preview_positions"], dtype=float)
        real_path = np.asarray(sample["real_positions"], dtype=float)
        fig = plt.figure(figsize=(8.0, 6.2))
        axis = fig.add_subplot(111, projection="3d")
        axis.plot(*preview_path.T, label="FP-SHEP")
        axis.plot(*real_path.T, linestyle="--", label="Real rollout")
        axis.scatter(*sample["candidate_xyz"], marker="x", s=70, label="candidate")
        axis.scatter(*sample["task_goal"], marker="*", s=100, label="task goal")
        axis.set_title("Preview vs real trajectory")
        axis.legend()
        fig.tight_layout()
        fig.savefig(figures / "preview_vs_real_trajectory.png", dpi=180)
        plt.close(fig)


def run_validation(settings: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    checkpoint = Path(settings["checkpoint"])
    checkpoint = checkpoint if checkpoint.is_absolute() else REPO_ROOT / checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if settings.get("deterministic_policy") is not True:
        raise ValueError("FP-SHEP validation requires deterministic_policy=true")
    scene_types = [str(value) for value in settings["scenario_types"]]
    seeds = [int(value) for value in settings["seeds"]]
    expected_count = len(scene_types) * len(seeds)
    configured_count = int(settings.get("number_of_scenarios", expected_count))
    if configured_count != expected_count:
        raise ValueError("number_of_scenarios must equal len(scenario_types) * len(seeds)")
    horizons = sorted({int(value) for value in settings["H_values"]})
    if not horizons or any(value <= 0 for value in horizons):
        raise ValueError("H_values must contain at least one positive horizon")
    proposal_config = ProposalConfig(**settings.get("proposal_config", {}))
    k_requested = int(settings["K"])
    if k_requested <= 0:
        raise ValueError("K must be positive")
    score_weights = {key: float(value) for key, value in settings["score_weights"].items()}
    clearance_cap = float(settings["clearance_normalization_cap"])

    multi_config = build_single_distribution_multi_config(
        num_agents=int(settings.get("num_agents", 3)),
        max_steps=int(settings.get("max_steps", max(horizons) + 1)),
    )
    if not np.isclose(float(multi_config.time_step), float(settings["dt"])):
        raise ValueError("configured dt differs from checkpoint-aligned environment dt")
    reference_env = build_single_env(config=SINGLE_AGENT_CONFIG, action_guidance_enabled=False)
    model = build_single_model(reference_env, config=SINGLE_AGENT_CONFIG, verbose=0)
    load_checkpoint(model, checkpoint)
    freeze_policy(model)

    output_dir.mkdir(parents=True, exist_ok=False)
    resolved_config = copy.deepcopy(settings)
    resolved_config.update({
        "created_at": datetime.now().astimezone().isoformat(),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "output_dir": str(output_dir.resolve()),
        "actual_number_of_scenarios": expected_count,
        "score_normalization": "candidate-set min-max; constant feature maps to zero",
        "clearance_analysis_note": "preview and real clearance use different sources; correlation is diagnostic",
    })
    _write_json(output_dir / "config.json", resolved_config)

    candidate_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    skipped_states: list[dict[str, Any]] = []
    completed = 0
    total = expected_count * len(settings.get("agent_indices", [0])) * len(horizons)
    try:
        for scene_type in scene_types:
            for seed in seeds:
                env, scene_metadata = build_validation_environment(
                    config=multi_config,
                    scene_type=scene_type,
                    seed=seed,
                    peer_radius=float(settings.get("peer_radius", 0.3)),
                )
                try:
                    if not advance_to_evaluation_state(
                        env,
                        model,
                        int(settings.get("evaluation_state_warmup_steps", 0)),
                    ):
                        skipped_states.append({"scenario_type": scene_type, "seed": seed, "reason": "warmup terminated"})
                        continue
                    state_fingerprint = environment_state_fingerprint(env)
                    for agent_index in [int(value) for value in settings.get("agent_indices", [0])]:
                        proposals = _proposals(env, agent_index, proposal_config)
                        if not proposals:
                            skipped_states.append({"scenario_type": scene_type, "seed": seed, "agent_id": agent_index, "reason": "no valid candidate"})
                            continue
                        for horizon in horizons:
                            scenario_id = f"{scene_type}_seed{seed}"
                            state_id = f"{scenario_id}_agent{agent_index}"
                            record_id = f"{state_id}_H{horizon}"
                            rows, paths = evaluate_candidate_set(
                                env=env,
                                agent_index=agent_index,
                                proposals=proposals,
                                policy=model,
                                horizon=horizon,
                                consumer_top_k=k_requested,
                            )
                            for row in rows:
                                row.update(
                                    scenario_id=scenario_id,
                                    state_id=state_id,
                                    scenario_type=scene_type,
                                    seed=seed,
                                    evaluation_state_fingerprint=state_fingerprint,
                                    proposal_count=len(proposals),
                                    **scene_metadata,
                                )
                            summary = score_and_summarize_candidate_set(
                                rows,
                                weights=score_weights,
                                clearance_cap=clearance_cap,
                            )
                            summary.update(
                                scenario_id=scenario_id,
                                state_id=state_id,
                                scenario_type=scene_type,
                                seed=seed,
                                agent_id=agent_index,
                                H=horizon,
                                K_requested=k_requested,
                                proposal_count=len(proposals),
                                **scene_metadata,
                            )
                            candidate_rows.extend(rows)
                            scenario_rows.append(summary)
                            from planning.policy_preview import build_preview_inputs_from_env
                            _, context = build_preview_inputs_from_env(env, agent_index)
                            for payload in paths:
                                payload.update(
                                    scenario_id=scenario_id,
                                    state_id=state_id,
                                    scenario_type=scene_type,
                                    seed=seed,
                                    agent_id=agent_index,
                                    H=horizon,
                                    current_position=np.asarray(env.dynamics[agent_index].p, dtype=float).copy(),
                                    task_goal=np.asarray(env.goals[agent_index], dtype=float).copy(),
                                    visible_surface_points=context.visible_surface_points.copy(),
                                    initial_current_scan=context.current_scan.copy(),
                                    initial_previous_scan=context.previous_scan.copy(),
                                    initial_sensor_observation=np.asarray(
                                        env.latest_sensor_packets[agent_index].observation,
                                        dtype=float,
                                    ).copy(),
                                    neighbor_positions=np.stack([
                                        np.asarray(env.dynamics[index].p, dtype=float)
                                        for index in range(int(env.num_agents)) if index != agent_index
                                    ]),
                                    neighbor_velocities=np.stack([
                                        np.asarray(env.dynamics[index].v, dtype=float)
                                        for index in range(int(env.num_agents)) if index != agent_index
                                    ]),
                                    scene_snapshot=_scene_snapshot(env),
                                )
                                trajectories.append(payload)
                                _save_trajectory(
                                    output_dir / "trajectories" / f"{record_id}_candidate{payload['candidate_id']}.npz",
                                    payload,
                                )
                            completed += 1
                            print(f"[{completed}/{total}] {record_id}: K_t={len(rows)}")
                finally:
                    env.close()
    finally:
        reference_env.close()

    aggregate = aggregate_summaries(candidate_rows, scenario_rows)
    success_cases, failure_cases = classify_cases(
        candidate_rows,
        scenario_rows,
        score_gap_threshold=float(settings.get("case_score_gap_threshold", 0.1)),
    )
    aggregate["success_case_count"] = len(success_cases)
    aggregate["failure_case_count"] = len(failure_cases)
    aggregate["skipped_states"] = skipped_states
    correlations = correlation_rows(candidate_rows)
    _write_csv(output_dir / "candidate_level_results.csv", candidate_rows)
    _write_csv(output_dir / "scenario_summary.csv", scenario_rows)
    _write_csv(output_dir / "correlation_summary.csv", correlations)
    _write_csv(output_dir / "horizon_summary.csv", aggregate["per_H"])
    _write_csv(output_dir / "scene_type_summary.csv", aggregate["per_scene"])
    _write_json(output_dir / "summary.json", aggregate)
    _write_json(output_dir / "success_cases.json", success_cases)
    _write_json(output_dir / "failure_cases.json", failure_cases)
    _write_json(output_dir / "skipped_states.json", skipped_states)
    _save_case_bundles(
        output_dir,
        case_kind="geometry_failure_preview_success",
        cases=success_cases,
        candidate_rows=candidate_rows,
        trajectories=trajectories,
    )
    _save_case_bundles(
        output_dir,
        case_kind="preview_failure",
        cases=failure_cases,
        candidate_rows=candidate_rows,
        trajectories=trajectories,
    )
    _plot_results(output_dir, candidate_rows, aggregate["per_H"], trajectories)
    return aggregate


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--horizons", type=int, nargs="+", default=None)
    parser.add_argument("--scenes", type=str, nargs="+", default=None)
    parser.add_argument("--k", type=int, default=None)
    return parser.parse_args()


def main() -> Path:
    args = _parse_args()
    config_path = args.config.expanduser().resolve()
    settings = json.loads(config_path.read_text(encoding="utf-8"))
    if args.checkpoint is not None:
        settings["checkpoint"] = str(args.checkpoint)
    if args.seeds is not None:
        settings["seeds"] = args.seeds
    if args.horizons is not None:
        settings["H_values"] = args.horizons
    if args.scenes is not None:
        settings["scenario_types"] = args.scenes
    if args.k is not None:
        settings["K"] = args.k
    settings["number_of_scenarios"] = len(settings["scenario_types"]) * len(settings["seeds"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is not None:
        output_dir = args.output_dir.expanduser().resolve()
    else:
        base = Path(settings["output_dir"])
        base = base if base.is_absolute() else REPO_ROOT / base
        output_dir = base / timestamp
    run_validation(settings, output_dir)
    print(f"Artifacts written to: {output_dir}")
    return output_dir


if __name__ == "__main__":
    main()
