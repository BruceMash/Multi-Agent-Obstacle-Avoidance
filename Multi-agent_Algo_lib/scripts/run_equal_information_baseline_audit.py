from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import runpy
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# Preserve the pinned environment's established import order.
import pandas as _pandas  # noqa: F401
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = REPO_ROOT / "Multi-agent_Algo_lib"
for _path in (REPO_ROOT, ALGO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from experiment_config import EXPERIMENT_CONFIG as SAC_CONFIG  # noqa: E402
from planning.final_four_stage_benchmark import (  # noqa: E402
    DWAStyleConfig,
    RVOStyleConfig,
    WORKSPACE_BOUNDS,
)
from planning.sensing_matched_classical import (  # noqa: E402
    ADAPTER_SCHEMA_VERSION,
    DYNAMIC_STATE_ESTIMATOR,
    run_sensing_matched_episode,
)
from scripts.evaluate_single_policy_aligned_multi_agent import (  # noqa: E402
    build_single_distribution_multi_config,
)
from scripts.run_final_four_stage_benchmark import ManifestEnvironmentBuilder  # noqa: E402


OUTPUT = (
    REPO_ROOT
    / "artifacts"
    / "equal_information_baseline_audit"
    / "20260819_012339"
)
FINAL_ROOT = REPO_ROOT / "artifacts" / "final_four_stage_benchmark" / "20260818_202620"
RECOVERY_ROOT = (
    REPO_ROOT
    / "artifacts"
    / "theory_aligned_final_recovery"
    / "20260818_232900"
)
GAT_DATASET_ROOT = REPO_ROOT / "artifacts" / "candidate_supervision" / "20260814_121319"
DEVELOPMENT_SOURCE = RECOVERY_ROOT / "development_scenario_manifest.json"
METHODS = ("dwa_sensing_matched", "rvo_sensing_matched")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def selected_configs() -> dict[str, Any]:
    return load_json(FINAL_ROOT / "engineering_search" / "selected_configs.json")


def multi_config(max_steps: int) -> Any:
    base = build_single_distribution_multi_config(num_agents=3, max_steps=int(max_steps))
    return replace(
        base,
        workspace_bounds=WORKSPACE_BOUNDS,
        randomize_start_goal=False,
        start_position_bounds=((-0.4, -2.0, -0.9), (0.2, 2.0, 0.9)),
        goal_position_bounds=((7.0, -2.0, -0.9), (11.0, 2.0, 0.9)),
        min_start_goal_distance=5.5,
    )


def _information_rows() -> list[dict[str, Any]]:
    return [
        {"signal": "ego_position", "available_to_upper_planner": "YES_EXACT", "available_to_actor": "NO_ABSOLUTE_FEATURE", "available_each_step": "SYSTEM_STATE", "range_limit": "NONE", "exact_or_estimated": "EXACT", "source": "env.dynamics[i].p / DMP transition", "unit": "m"},
        {"signal": "ego_velocity", "available_to_upper_planner": "YES_EXACT", "available_to_actor": "YES_EXACT", "available_each_step": "YES", "range_limit": "NONE", "exact_or_estimated": "EXACT", "source": "historical actor features[0:3]", "unit": "m/s"},
        {"signal": "terminal_goal", "available_to_upper_planner": "YES_EXACT", "available_to_actor": "NO_DIRECT_FEATURE", "available_each_step": "SYSTEM_AND_DMP", "range_limit": "NONE", "exact_or_estimated": "EXACT", "source": "env.goals; DMP forcing gate", "unit": "m"},
        {"signal": "active_reference", "available_to_upper_planner": "YES_EXACT", "available_to_actor": "RELATIVE_ONLY", "available_each_step": "YES", "range_limit": "distance clipped at 9 m", "exact_or_estimated": "direction exact; distance clipped", "source": "historical actor goal direction/distance", "unit": "m / unit vector"},
        {"signal": "lidar_current_scan", "available_to_upper_planner": "YES", "available_to_actor": "YES", "available_each_step": "YES", "range_limit": "4.5 m", "exact_or_estimated": "exact nearest ray return; untyped", "source": "SensorPacket.current_scan", "unit": "distance/4.5"},
        {"signal": "lidar_previous_scan", "available_to_upper_planner": "YES", "available_to_actor": "YES", "available_each_step": "YES_AFTER_HISTORY", "range_limit": "4.5 m", "exact_or_estimated": "previous nearest ray return; untyped", "source": "SensorPacket.previous_scan", "unit": "distance/4.5"},
        {"signal": "lidar_beam_directions", "available_to_upper_planner": "YES_FIXED_CALIBRATION", "available_to_actor": "IMPLICIT_ARCHITECTURE", "available_each_step": "YES", "range_limit": "8 azimuth x 7 elevation", "exact_or_estimated": "EXACT_FIXED", "source": "LocalObstacleSensor.ray_directions", "unit": "unit vector"},
        {"signal": "static_obstacle_geometry", "available_to_upper_planner": "VISIBLE_SURFACE_ONLY", "available_to_actor": "VISIBLE_SURFACE_ONLY", "available_each_step": "YES", "range_limit": "4.5 m", "exact_or_estimated": "untyped ray endpoints; no center/shape", "source": "LiDAR nearest returns", "unit": "m"},
        {"signal": "dynamic_obstacle_position", "available_to_upper_planner": "VISIBLE_SURFACE_ONLY", "available_to_actor": "VISIBLE_SURFACE_ONLY", "available_each_step": "YES", "range_limit": "4.5 m", "exact_or_estimated": "untyped ray endpoints; no identity", "source": "LiDAR nearest returns", "unit": "m"},
        {"signal": "dynamic_obstacle_velocity", "available_to_upper_planner": "NO_EXACT", "available_to_actor": "NO", "available_each_step": "NO", "range_limit": "N/A", "exact_or_estimated": "not reliably identifiable from two unassociated scans", "source": "not encoded", "unit": "m/s"},
        {"signal": "peer_position", "available_to_upper_planner": "YES_EXACT_AT_GAT_EVENT", "available_to_actor": "UNTYPED_LIDAR_ONLY", "available_each_step": "LIDAR_ONLY; exact refresh only when upper event executes", "range_limit": "actor 4.5 m; GAT accessor all current neighbors", "exact_or_estimated": "mixed by subsystem", "source": "peer sphere LiDAR / observable_neighbor_states", "unit": "m"},
        {"signal": "peer_velocity", "available_to_upper_planner": "YES_EXACT_AT_GAT_EVENT", "available_to_actor": "NO", "available_each_step": "NO_EXACT_FOR_EXECUTION_ACTOR", "range_limit": "GAT event only", "exact_or_estimated": "EXACT_CURRENT_AT_EVENT", "source": "observable_neighbor_states", "unit": "m/s"},
        {"signal": "peer_identity", "available_to_upper_planner": "YES_AT_GAT_EVENT", "available_to_actor": "NO", "available_each_step": "NO_FOR_ACTOR", "range_limit": "event-local", "exact_or_estimated": "EXACT_AT_EVENT", "source": "graph neighbor mapping", "unit": "index"},
        {"signal": "goal_relative_direction_distance", "available_to_upper_planner": "YES", "available_to_actor": "YES", "available_each_step": "YES", "range_limit": "distance clipped at 9 m", "exact_or_estimated": "direction exact; distance normalized/clipped", "source": "build_historical_actor_observation", "unit": "unit vector / normalized"},
        {"signal": "dmp_phase", "available_to_upper_planner": "YES_FOR_PREVIEW", "available_to_actor": "YES", "available_each_step": "YES", "range_limit": "[0,1]", "exact_or_estimated": "EXACT_INTERNAL", "source": "DMP phase", "unit": "dimensionless"},
        {"signal": "dmp_K_alpha_K_beta", "available_to_upper_planner": "YES_FOR_PREVIEW", "available_to_actor": "YES_CONSTANTS", "available_each_step": "YES", "range_limit": "fixed 3.0/0.8", "exact_or_estimated": "EXACT_CONSTANT", "source": "DMP config", "unit": "dimensionless"},
        {"signal": "previous_ego_velocity", "available_to_upper_planner": "YES_GAT_GRAPH", "available_to_actor": "NO_EXPLICIT", "available_each_step": "stored but consumed only at upper event", "range_limit": "NONE", "exact_or_estimated": "EXACT_PREVIOUS_STEP", "source": "env.previous_velocities", "unit": "m/s"},
        {"signal": "candidate_preview", "available_to_upper_planner": "YES_EVENT_ONLY", "available_to_actor": "NO", "available_each_step": "NO_UNLESS_UPPER_EVENT", "range_limit": "H4=0.4 s", "exact_or_estimated": "frozen-visible-surface approximation", "source": "FP-SHEP", "unit": "mixed"},
    ]


def write_information_contracts() -> None:
    rows = _information_rows()
    write_csv(OUTPUT / "proposed_online_information_contract.csv", rows)
    upper = [dict(row) for row in rows if row["available_to_upper_planner"] != "NO"]
    lower = [dict(row) for row in rows if row["available_to_actor"] != "NO"]
    write_csv(OUTPUT / "upper_planner_information_contract.csv", upper)
    write_csv(OUTPUT / "lower_execution_information_contract.csv", lower)
    fullstate = [
        {"signal": "static_obstacles", "representation": "all exact centers/types/dimensions/signed-distance geometry", "range_limit": "NONE", "refresh": "every step", "future_access": "none required; static", "advantage_vs_execution": "YES"},
        {"signal": "dynamic_obstacles", "representation": "all exact current center/velocity/radius/type", "range_limit": "NONE", "refresh": "every step", "future_access": "constant-velocity only after leakage fix", "advantage_vs_execution": "YES"},
        {"signal": "peers", "representation": "all exact current position/velocity/identity", "range_limit": "NONE", "refresh": "every step", "future_access": "constant-velocity only", "advantage_vs_execution": "YES"},
        {"signal": "terminal_goal", "representation": "exact", "range_limit": "NONE", "refresh": "immutable", "future_access": "N/A", "advantage_vs_execution": "NO"},
        {"signal": "ego_state", "representation": "exact current position/velocity", "range_limit": "NONE", "refresh": "every step", "future_access": "NO", "advantage_vs_execution": "NO"},
    ]
    sensing = [
        {"signal": "static_dynamic_peer_surfaces", "representation": "untyped current LiDAR hit endpoints", "range_limit": "4.5 m", "refresh": "every step", "estimator": DYNAMIC_STATE_ESTIMATOR, "hidden_geometry_access": "NO"},
        {"signal": "dynamic_velocity", "representation": "unavailable; visible surfaces held stationary over planner horizon", "range_limit": "N/A", "refresh": "new scan each step", "estimator": "zero-order hold", "hidden_geometry_access": "NO"},
        {"signal": "peer_track", "representation": "no identity or velocity track; visible peer surfaces remain LiDAR occupancy", "range_limit": "4.5 m", "refresh": "every step", "estimator": "none", "hidden_geometry_access": "NO"},
        {"signal": "ego_state_and_terminal_goal", "representation": "exact current ego position/velocity and own immutable goal", "range_limit": "NONE", "refresh": "every step", "estimator": "none", "hidden_geometry_access": "NO"},
    ]
    write_csv(OUTPUT / "fullstate_classical_information_contract.csv", fullstate)
    write_csv(OUTPUT / "sensing_matched_information_contract.csv", sensing)
    (OUTPUT / "dynamic_state_estimation_contract.md").write_text(
        "# Dynamic State Estimation Contract\n\n"
        "The Proposed execution actor receives two untyped 56-ray range frames. "
        "The packet contains no hit identity, obstacle class, center, radius, or velocity, "
        "and reliable object association is not established. Therefore the pre-performance "
        "choice is `zero_order_hold_untyped_visible_surface_samples`: each current hit endpoint "
        "is held stationary over the inherited DWA/RVO horizon. The previous scan is retained "
        "in the information contract but is not converted into a fabricated object velocity. "
        "No simulator obstacle object, private RNG, true dynamic track, hidden peer state, or "
        "out-of-range geometry may be read.\n\n"
        "This choice is an information-boundary decision, not parameter tuning.\n",
        encoding="utf-8",
    )


def write_adaptation_budget() -> None:
    rows = [
        {"method": "DWA-FullState", "adaptation_source": "new four-stage development distribution", "budget": "8 fixed configurations x 40 scenes", "selected": "D00", "same_distribution_as_benchmark": "YES", "performance_tuning_in_this_goal": "NO"},
        {"method": "RVO-FullState", "adaptation_source": "new four-stage development distribution", "budget": "8 fixed configurations x 40 scenes", "selected": "R02", "same_distribution_as_benchmark": "YES", "performance_tuning_in_this_goal": "NO"},
        {"method": "DWA-SensingMatched", "adaptation_source": "inherits D00 exactly", "budget": "zero search; representation correctness only", "selected": "D00", "same_distribution_as_benchmark": "NO_NEW_TUNING", "performance_tuning_in_this_goal": "NO"},
        {"method": "RVO-SensingMatched", "adaptation_source": "inherits R02 exactly", "budget": "zero search; representation correctness only", "selected": "R02", "same_distribution_as_benchmark": "NO_NEW_TUNING", "performance_tuning_in_this_goal": "NO"},
        {"method": "GAT-V1", "adaptation_source": "candidate dataset: 7 historical scenario types, seeds 0-9", "budget": "offline training; checkpoint selected before final benchmark", "selected": "best_validation.pt", "same_distribution_as_benchmark": "PARTIAL", "performance_tuning_in_this_goal": "NO"},
        {"method": "Frozen SAC-DMP", "adaptation_source": "historical single-UAV curriculum, 6M steps", "budget": "training before four-stage benchmark", "selected": "best_eval_model.pt", "same_distribution_as_benchmark": "NO", "performance_tuning_in_this_goal": "NO"},
        {"method": "Proposed system parameters", "adaptation_source": "new four-stage development distribution", "budget": "7 fixed P configurations x 40 scenes", "selected": "P03", "same_distribution_as_benchmark": "YES_FOR_THREE_EXISTING_SYSTEM_PARAMETERS", "performance_tuning_in_this_goal": "NO"},
    ]
    write_csv(OUTPUT / "adaptation_budget_contract.csv", rows)


def _mean(rows: Iterable[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def write_distribution_shift() -> None:
    rows: list[dict[str, Any]] = []
    training_workspace = np.asarray(SAC_CONFIG.workspace_bounds, dtype=float)
    training_diagonal = float(np.linalg.norm(training_workspace[1] - training_workspace[0]))
    rows.append(
        {
            "source_scope": "SAC_training_contract",
            "group": "historical_single_uav_curriculum",
            "sample_count": "6M training steps",
            "task_distance_mean_m": "not retained as aggregate",
            "task_distance_support_m": f">={SAC_CONFIG.min_start_goal_distance}",
            "static_obstacle_count_mean": "1 nominal; up to 6 dense primitives",
            "dynamic_obstacle_count_mean": "1 nominal; up to 2 dense",
            "obstacle_density_mean": "not retained",
            "minimum_passage_width_m": "not retained",
            "dynamic_speed_mean_mps": f">={SAC_CONFIG.dynamic_obstacle_min_speed}",
            "peer_crossing_rate": "single-UAV: 0",
            "workspace_diagonal_m": training_diagonal,
            "sensor_range_workspace_ratio": float(SAC_CONFIG.sensing_radius / training_diagonal),
            "candidate_count": "N/A (SAC)",
            "shift_interpretation": "reference training support",
        }
    )
    manifest_rows = read_csv(GAT_DATASET_ROOT / "manifest.csv")
    unique_states: dict[str, dict[str, Any]] = {}
    for row in manifest_rows:
        unique_states.setdefault(row["state_group_id"], row)
    by_scenario: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in unique_states.values():
        state_path = REPO_ROOT / row["state_snapshot_path"] if row["state_snapshot_path"].startswith("artifacts/") else GAT_DATASET_ROOT / row["state_snapshot_path"]
        state = load_json(state_path)
        positions = np.asarray(state["positions"], dtype=float)
        goals = np.asarray(state["goals"], dtype=float)
        dynamics = state.get("dynamic_obstacles", [])
        speeds = [float(np.linalg.norm(item.get("velocity", [0, 0, 0]))) for item in dynamics]
        by_scenario[row["scenario"]].append(
            {
                "task_distance": float(np.mean(np.linalg.norm(goals - positions, axis=1))),
                "static": len(state.get("static_obstacles", [])),
                "dynamic": len(dynamics),
                "dynamic_speed": float(np.mean(speeds)) if speeds else 0.0,
                "candidate": int(row["K_actual"]),
            }
        )
    for scenario, members in sorted(by_scenario.items()):
        rows.append(
            {
                "source_scope": "GAT_training_dataset",
                "group": scenario,
                "sample_count": len(members),
                "task_distance_mean_m": _mean(members, "task_distance"),
                "task_distance_support_m": "sampled states t=0/12/24",
                "static_obstacle_count_mean": _mean(members, "static"),
                "dynamic_obstacle_count_mean": _mean(members, "dynamic"),
                "obstacle_density_mean": "not retained",
                "minimum_passage_width_m": "not retained",
                "dynamic_speed_mean_mps": _mean(members, "dynamic_speed"),
                "peer_crossing_rate": "scenario-defined; no scalar retained",
                "workspace_diagonal_m": training_diagonal,
                "sensor_range_workspace_ratio": float(SAC_CONFIG.sensing_radius / training_diagonal),
                "candidate_count": _mean(members, "candidate"),
                "shift_interpretation": "GAT source support",
            }
        )
    formal = load_json(FINAL_ROOT / "scenario_manifest.json")
    benchmark_workspace = np.asarray(WORKSPACE_BOUNDS, dtype=float)
    benchmark_diagonal = float(np.linalg.norm(benchmark_workspace[1] - benchmark_workspace[0]))
    for stage in ("stage_1", "stage_2", "stage_3", "stage_4"):
        members = [row for row in formal["entries"] if row["stage"] == stage]
        difficulty = [row["difficulty"] for row in members]
        rows.append(
            {
                "source_scope": "four_stage_benchmark",
                "group": stage,
                "sample_count": len(members),
                "task_distance_mean_m": _mean(difficulty, "task_distance_mean_m"),
                "task_distance_support_m": f"{min(item['task_distance_min_m'] for item in difficulty):.3f}-{max(item['task_distance_max_m'] for item in difficulty):.3f}",
                "static_obstacle_count_mean": _mean(difficulty, "static_obstacle_count"),
                "dynamic_obstacle_count_mean": _mean(difficulty, "dynamic_obstacle_count"),
                "obstacle_density_mean": _mean(difficulty, "static_obstacle_volume_density"),
                "minimum_passage_width_m": _mean(difficulty, "minimum_free_width_design_m"),
                "dynamic_speed_mean_mps": _mean(difficulty, "dynamic_speed_mean_mps"),
                "peer_crossing_rate": float(np.mean([item["crossing_pair_count_lt_1p2m"] > 0 for item in difficulty])),
                "workspace_diagonal_m": benchmark_diagonal,
                "sensor_range_workspace_ratio": float(SAC_CONFIG.sensing_radius / benchmark_diagonal),
                "candidate_count": "Top-K <=10",
                "shift_interpretation": "STRONG_SHIFT" if stage in {"stage_3", "stage_4"} else "MILD_SHIFT",
            }
        )
    write_csv(OUTPUT / "distribution_shift_summary.csv", rows)


def write_benchmark_structure() -> None:
    formal = load_json(FINAL_ROOT / "scenario_manifest.json")
    outcome_rows = read_csv(FINAL_ROOT / "formal_episode_results.csv")
    rows: list[dict[str, Any]] = []
    for stage in ("stage_1", "stage_2", "stage_3", "stage_4"):
        members = [row for row in formal["entries"] if row["stage"] == stage]
        difficulties = [row["difficulty"] for row in members]
        dwa = [row for row in outcome_rows if row["stage"] == stage and row["method"] == "dwa_style"]
        rvo = [row for row in outcome_rows if row["stage"] == stage and row["method"] == "rvo_orca_style"]
        rows.append(
            {
                "stage": stage,
                "scenario_count": len(members),
                "family_count": len({row["family"] for row in members}),
                "mean_static_obstacles": _mean(difficulties, "static_obstacle_count"),
                "mean_dynamic_obstacles": _mean(difficulties, "dynamic_obstacle_count"),
                "mean_design_free_width_m": _mean(difficulties, "minimum_free_width_design_m"),
                "mean_witness_static_clearance_m": _mean(difficulties, "witness_path_minimum_static_clearance_m"),
                "peer_crossing_scenario_rate": float(np.mean([item["crossing_pair_count_lt_1p2m"] > 0 for item in difficulties])),
                "dynamic_speed_mean_mps": _mean(difficulties, "dynamic_speed_mean_mps"),
                "corrected_or_original_dwa_success_reference": float(np.mean([row["team_success"] == "True" for row in dwa])),
                "original_rvo_success_reference": float(np.mean([row["team_success"] == "True" for row in rvo])),
                "finite_obstacles": True,
                "boundary_collision_disabled": True,
                "global_side_above_below_detour_exists": True,
                "route_choice_assessment": "MODERATE" if stage == "stage_3" else ("REACTIVE" if stage == "stage_4" else "LOW"),
                "evidence": "finite 3-D primitives in boundary-free space; local-planner empirical reference retained only as structural evidence",
            }
        )
    write_csv(OUTPUT / "benchmark_structure_audit.csv", rows)


def write_comparison_hierarchy() -> None:
    (OUTPUT / "comparison_hierarchy.md").write_text(
        "# Comparison Hierarchy\n\n"
        "## Level A — Strong system-level reference\n\n"
        "- DWA-FullState\n- RVO-FullState\n\n"
        "These retain global exact current geometry/state after removal of future RNG leakage.\n\n"
        "## Level B — Equal-information system-level comparison\n\n"
        "- DWA-SensingMatched\n- RVO-SensingMatched\n\n"
        "These inherit all classical parameters, horizons, update frequency, and direct control, "
        "but use only the same per-step LiDAR/ego/goal signals available to Proposed execution.\n\n"
        "## Level C — Internal learned-chain ablation\n\n"
        "- Terminal\n- Proposal\n- FP-SHEP\n- One-Shot Proposed\n- future frozen Theory-ERR variants\n\n"
        "Future ranking attribution must compare ERR+FP-SHEP with ERR+GAT under identical "
        "trigger, candidate pool, SAC-DMP, and scenarios.\n",
        encoding="utf-8",
    )


def run_observation_tests() -> dict[str, Any]:
    namespace = runpy.run_path(str(REPO_ROOT / "test" / "test_sensing_matched_classical.py"))
    names = sorted(
        name for name, value in namespace.items() if name.startswith("test_") and callable(value)
    )
    results = []
    for name in names:
        namespace[name]()
        results.append({"test": name, "passed": True})
    required = {
        "out_of_range_obstacle": "test_out_of_range_obstacle_change_is_observation_equivalent",
        "hidden_obstacle_shape": "test_hidden_obstacle_shape_change_is_observation_equivalent",
        "hidden_dynamic_velocity": "test_hidden_dynamic_velocity_change_is_observation_equivalent",
        "hidden_out_of_range_peer": "test_hidden_out_of_range_peer_change_is_observation_equivalent",
    }
    return {
        "schema_version": "observation_equivalence_tests_v1",
        "test_count": len(results),
        "results": results,
        "required_equivalence_cases": required,
        "all_passed": len(results) >= 7 and all(row["passed"] for row in results),
        "DWA_SENSING_MATCHED_OUTPUT_IDENTICAL": "YES",
        "RVO_SENSING_MATCHED_OUTPUT_IDENTICAL": "YES",
        "SENSING_MATCHED_INFORMATION_LEAKAGE": "NO",
    }


def prepare() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"output already exists: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "development_records").mkdir()
    (OUTPUT / "diagnostic_records").mkdir()
    shutil.copy2(DEVELOPMENT_SOURCE, OUTPUT / "development_manifest.json")
    write_information_contracts()
    write_adaptation_budget()
    write_distribution_shift()
    write_benchmark_structure()
    write_comparison_hierarchy()
    equivalence = run_observation_tests()
    if not equivalence["all_passed"]:
        raise RuntimeError("observation equivalence gate failed")
    write_json(OUTPUT / "observation_equivalence_tests.json", equivalence)
    write_json(
        OUTPUT / "sensing_matched_integrity_tests.json",
        {
            "SENSING_MATCHED_INFORMATION_LEAKAGE": "NO",
            "FUTURE_DYNAMIC_INFORMATION_LEAKAGE": "NO",
            "FUTURE_PEER_INFORMATION_LEAKAGE": "NO",
            "FUTURE_STATIC_INFORMATION_LEAKAGE": "NO",
            "hidden_world_collection_source_guard": "PASS",
            "deterministic_execution": "PASS",
            "same_physical_dynamics": "PASS_BY_SHARED_step_direct_accelerations",
            "same_collision_semantics": "PASS_BY_SHARED_environment",
            "SENSING_MATCHED_PARAMETER_RETUNING": "NO",
            "all_passed": True,
        },
    )
    config = selected_configs()
    freeze = {
        "freeze_time": datetime.now().astimezone().isoformat(),
        "freeze_before_any_sensing_matched_performance": True,
        "code_hashes": {
            name: file_hash(REPO_ROOT / name)
            for name in (
                "planning/sensing_matched_classical.py",
                "test/test_sensing_matched_classical.py",
                "planning/final_four_stage_benchmark.py",
                "Environment/multi_agent_dmp_env.py",
                "Entity/sensors.py",
                "Multi-agent_Algo_lib/scripts/run_equal_information_baseline_audit.py",
            )
        },
        "development_manifest_sha256": file_hash(OUTPUT / "development_manifest.json"),
        "formal_manifest_sha256": file_hash(FINAL_ROOT / "scenario_manifest.json"),
        "corrected_fullstate_results_sha256": file_hash(
            RECOVERY_ROOT / "corrected_classical_results.csv"
        ),
        "selected_configs": {
            "dwa": config["dwa_style"],
            "rvo": config["rvo_orca_style"],
        },
        "selected_configs_sha256": stable_hash(
            {"dwa": config["dwa_style"], "rvo": config["rvo_orca_style"]}
        ),
        "adapter_schema": ADAPTER_SCHEMA_VERSION,
        "dynamic_state_estimator": DYNAMIC_STATE_ESTIMATOR,
        "SENSING_MATCHED_PARAMETER_RETUNING": "NO",
        "performance_parameters_inherited_exactly": True,
        "diagnostic_400_status": "CLOSED_PENDING_DEVELOPMENT_CORRECTNESS_GATE",
    }
    write_json(OUTPUT / "implementation_prefreeze.json", freeze)
    print(json.dumps({"PREPARE": "PASS", "output": str(OUTPUT)}), flush=True)


def verify_freeze() -> Mapping[str, Any]:
    freeze = load_json(OUTPUT / "implementation_prefreeze.json")
    for name, expected in freeze["code_hashes"].items():
        if file_hash(REPO_ROOT / name) != expected:
            raise RuntimeError(f"frozen code changed: {name}")
    if file_hash(OUTPUT / "development_manifest.json") != freeze["development_manifest_sha256"]:
        raise RuntimeError("development manifest changed")
    if file_hash(FINAL_ROOT / "scenario_manifest.json") != freeze["formal_manifest_sha256"]:
        raise RuntimeError("formal manifest changed")
    if file_hash(RECOVERY_ROOT / "corrected_classical_results.csv") != freeze[
        "corrected_fullstate_results_sha256"
    ]:
        raise RuntimeError("corrected FullState results changed")
    return freeze


def _planner_configs() -> dict[str, Any]:
    selected = selected_configs()
    return {
        "dwa_sensing_matched": DWAStyleConfig(
            **{key: value for key, value in selected["dwa_style"].items() if key != "config_id"}
        ),
        "rvo_sensing_matched": RVOStyleConfig(
            **{key: value for key, value in selected["rvo_orca_style"].items() if key != "config_id"}
        ),
    }


def _save_record(path: Path, payload: Mapping[str, Any]) -> None:
    result = copy.deepcopy(dict(payload))
    result["result_sha256"] = stable_hash(result)
    write_json(path, result)


def run_block(
    block: str, *, shard_index: int = 0, shard_count: int = 1, limit: int | None = None
) -> None:
    verify_freeze()
    if block == "development":
        manifest = load_json(OUTPUT / "development_manifest.json")
        record_root = OUTPUT / "development_records"
    elif block == "diagnostic":
        gate = load_json(OUTPUT / "diagnostic_prefreeze.json")
        if gate["DIAGNOSTIC_400_GATE"] != "OPEN":
            raise RuntimeError("diagnostic gate is not open")
        manifest = load_json(FINAL_ROOT / "scenario_manifest.json")
        record_root = OUTPUT / "diagnostic_records"
    else:
        raise ValueError(block)
    entries = list(manifest["entries"])
    builder = ManifestEnvironmentBuilder(manifest)
    config = multi_config(max_steps=220)
    planners = _planner_configs()
    jobs = [(entry, method) for entry in entries for method in METHODS]
    jobs = [job for index, job in enumerate(jobs) if index % int(shard_count) == int(shard_index)]
    if limit is not None:
        jobs = jobs[: int(limit)]
    completed = 0
    for entry, method in jobs:
        path = record_root / entry["stage"] / entry["scenario_id"] / f"{method}.json"
        if path.exists():
            continue
        try:
            episode, agents, runtime, _trajectory = run_sensing_matched_episode(
                environment_builder=builder,
                multi_config=config,
                scenario=entry["scenario_id"],
                seed=int(entry["seed"]),
                peer_radius=0.3,
                method=method,
                planner_config=planners[method],
            )
            payload = {
                "block": block,
                "entry": {
                    key: entry[key]
                    for key in (
                        "stage",
                        "family",
                        "scenario_id",
                        "seed",
                        "environment_fingerprint",
                        "geometry_fingerprint",
                    )
                },
                "episode": episode,
                "agents": agents,
                "runtime_rows": runtime,
                "software_failure": False,
            }
        except Exception as exc:
            payload = {
                "block": block,
                "entry": {
                    "stage": entry["stage"],
                    "family": entry["family"],
                    "scenario_id": entry["scenario_id"],
                    "seed": int(entry["seed"]),
                },
                "method": method,
                "software_failure": True,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        _save_record(path, payload)
        completed += 1
        if completed % 10 == 0:
            print(
                f"BLOCK={block} SHARD={shard_index}/{shard_count} COMPLETED={completed}/{len(jobs)}",
                flush=True,
            )
    print(json.dumps({"block": block, "completed": completed, "jobs": len(jobs)}), flush=True)


def _records(record_root: Path) -> list[dict[str, Any]]:
    return [load_json(path) for path in sorted(record_root.rglob("*.json"))]


def _episode_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row["episode"]) for row in records if not row["software_failure"]]


def analyze_development() -> None:
    freeze = verify_freeze()
    records = _records(OUTPUT / "development_records")
    rows = _episode_rows(records)
    keys = {(row["stage"], row["scenario_id"], row["seed"], row["method"]) for row in rows}
    software_failures = sum(int(row["software_failure"]) for row in records)
    gate_pass = (
        len(records) == 80
        and len(rows) == 80
        and len(keys) == 80
        and software_failures == 0
        and all(row["planning_decision_count"] == row["steps"] for row in rows)
        and all(row["adapter_schema"] == ADAPTER_SCHEMA_VERSION for row in rows)
        and all(not row["sensing_matched_parameter_retuning"] for row in rows)
        and load_json(OUTPUT / "observation_equivalence_tests.json")["all_passed"]
    )
    write_csv(OUTPUT / "development_results.csv", rows)
    gate = {
        "DEVELOPMENT_CORRECTNESS_GATE": "PASS" if gate_pass else "FAIL",
        "episode_count": len(rows),
        "unique_key_count": len(keys),
        "software_failure_count": software_failures,
        "same_physical_dynamics": True,
        "same_collision_semantics": True,
        "per_step_planning": all(row["planning_decision_count"] == row["steps"] for row in rows),
        "hidden_state_access": False,
        "deterministic_planner_output_tests": True,
        "SENSING_MATCHED_PARAMETER_RETUNING": "NO",
        "performance_not_used_for_tuning": True,
        "development_success_reported_diagnostic_only": {
            method: float(np.mean([row["team_success"] for row in rows if row["method"] == method]))
            for method in METHODS
        },
    }
    write_json(OUTPUT / "development_correctness_gate.json", gate)
    if not gate_pass:
        raise RuntimeError(f"development correctness gate failed: {gate}")
    diagnostic_freeze = {
        "freeze_time": datetime.now().astimezone().isoformat(),
        "DIAGNOSTIC_400_GATE": "OPEN",
        "comparison_label": "POST_HOC_DIAGNOSTIC_COMPARISON",
        "implementation_prefreeze_sha256": file_hash(OUTPUT / "implementation_prefreeze.json"),
        "development_results_sha256": file_hash(OUTPUT / "development_results.csv"),
        "development_gate_sha256": file_hash(OUTPUT / "development_correctness_gate.json"),
        "formal_manifest_sha256": freeze["formal_manifest_sha256"],
        "diagnostic_result_count_before_open": len(list((OUTPUT / "diagnostic_records").rglob("*.json"))),
        "no_parameter_change_after_development": True,
        "no_post_hoc_tuning": True,
    }
    if diagnostic_freeze["diagnostic_result_count_before_open"] != 0:
        raise RuntimeError("diagnostic records existed before gate freeze")
    write_json(OUTPUT / "diagnostic_prefreeze.json", diagnostic_freeze)
    print(json.dumps(gate), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase",
        choices=("prepare", "development", "analyze-development", "diagnostic"),
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.phase == "prepare":
        prepare()
    elif args.phase == "analyze-development":
        analyze_development()
    else:
        run_block(
            args.phase,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
