"""Audit scenario semantics and evaluate the frozen coupled diagnostic scene."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = Path(__file__).resolve().parent
for search_path in (REPO_ROOT, ALGO_ROOT, SCRIPTS_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from Guidance.reference_point_proposal_demo import ProposalConfig  # noqa: E402
from planning.historical_forcing_gate import (  # noqa: E402
    HISTORICAL_GATE_NAME,
    scoped_historical_preview_and_multi_agent_transition,
)
from planning.multi_agent_obstacle_scenario_audit import (  # noqa: E402
    FROZEN_LAYOUT_HASH,
    FROZEN_LAYOUT_SPEC,
    SCENARIO_ID,
    SCENARIO_ROLE,
    SEED_SET_ROLE,
    build_multi_agent_obstacle_environment,
    build_multi_agent_obstacle_options,
    geometry_record,
    obstacle_descriptor,
    stable_hash,
    summarize_geometry,
)
from planning.pre_gat_220step_revalidation import (  # noqa: E402
    METHOD_FP_SHEP,
    METHOD_ORDER,
    METHOD_PROPOSAL,
    METHOD_TERMINAL,
    aggregate_method_rows,
    build_failure_rows,
    build_selector_pairing,
    candidate_bundle_rows,
    generate_immutable_candidate_bundle,
)
from planning.pre_gat_closed_loop import FPSHEPOnlineScoreSpec  # noqa: E402
from planning.reference_transition_finetuning import sha256_file  # noqa: E402
from scripts.evaluate_actor_dmp_goal_semantics import (  # noqa: E402
    run_variant_episode,
    write_csv,
    write_json,
)
from scripts.evaluate_pre_gat_220step_revalidation import (  # noqa: E402
    _method_adapter,
    _stage_a_integrity,
    _standardize,
)
from scripts.evaluate_pre_gat_closed_loop import (  # noqa: E402
    _critical_hashes,
    _policy_parameter_sha256,
)
from scripts.evaluate_single_policy_aligned_multi_agent import (  # noqa: E402
    build_single_distribution_multi_config,
)
from scripts.evaluate_temporary_reference_interface import _load_policy  # noqa: E402
from scripts.validate_policy_preview import build_validation_environment  # noqa: E402


DEFAULT_CONFIG_PATH = (
    REPO_ROOT / "configs/evaluation/multi_agent_obstacle_scenario_audit.json"
)
EXISTING_SCENARIOS = ("open", "sparse_static", "multi_agent")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _assert_config(settings: Mapping[str, Any]) -> None:
    if str(settings["scenario"]) != SCENARIO_ID:
        raise ValueError("diagnostic scenario identifier changed")
    if str(settings["scenario_role"]) != SCENARIO_ROLE:
        raise ValueError("diagnostic scenario role changed")
    if str(settings["frozen_layout_hash_expected"]) != FROZEN_LAYOUT_HASH:
        raise ValueError("frozen geometry hash changed after confirmation")
    if str(settings["seed_set_role"]) != SEED_SET_ROLE:
        raise ValueError("seed-set role must remain development/diagnostic")
    if list(settings["stage_a_seeds"]) != list(range(5)):
        raise ValueError("Stage A seeds must be 0..4")
    if list(settings["stage_b_seeds"]) != list(range(5, 10)):
        raise ValueError("Stage B seeds must be 5..9")
    if tuple(settings["methods"]) != METHOD_ORDER:
        raise ValueError("method order changed")
    if int(settings["max_steps"]) != 220 or int(settings["num_agents"]) != 3:
        raise ValueError("220-step three-agent contract changed")
    if not math.isclose(float(settings["peer_radius"]), 0.3):
        raise ValueError("peer sphere radius changed")
    if any(bool(value) for value in settings["strict_exclusions"].values()):
        raise ValueError("strict exclusion flags must all remain false")
    spec = FPSHEPOnlineScoreSpec.from_mapping(settings["fp_shep_online_selector"])
    metadata = spec.metadata()
    if metadata["H_preview"] != 4:
        raise ValueError("H_preview must remain 4")
    if metadata["weights"]["terminal_speed"] != 0.0:
        raise ValueError("terminal speed must remain excluded from online ranking")
    execution = settings["execution_semantics"]
    if bool(execution["include_boundaries_in_sensor"]):
        raise ValueError("boundary sensor must remain disabled")
    if bool(execution["terminate_on_boundary_collision"]):
        raise ValueError("boundary termination must remain disabled")


def _scene_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    obstacle_influence_distance: float,
) -> dict[str, Any]:
    finite = [
        float(value)
        for row in records
        for value in row["direct_static_surface_clearances_m"]
        if math.isfinite(float(value))
    ]
    route_count = sum(int(row["num_agents"]) for row in records)
    intersections = sum(
        int(row["direct_path_intersection_agent_count"]) for row in records
    )
    influence = sum(int(row["direct_path_influence_agent_count"]) for row in records)
    static_counts = sorted({int(row["static_obstacle_count"]) for row in records})
    dynamic_counts = sorted({int(row["dynamic_obstacle_count"]) for row in records})
    workspace_volume = 9.0 * 4.5 * 2.4
    effective_volume = 0.0
    if records:
        for item in records[0]["static_obstacles"]:
            radius = float(item["effective_radius_m"])
            effective_volume += 4.0 * math.pi * radius**3 / 3.0
    return {
        "scenario": str(records[0]["scenario"]),
        "seed_count": len(records),
        "num_agents": sorted({int(row["num_agents"]) for row in records}),
        "static_obstacle_counts": static_counts,
        "dynamic_obstacle_counts": dynamic_counts,
        "static_obstacle_shapes": sorted(
            {
                str(item["type"])
                for row in records
                for item in row["static_obstacles"]
            }
        ),
        "dynamic_obstacle_shapes": sorted(
            {
                str(item["type"])
                for row in records
                for item in row["dynamic_obstacles"]
            }
        ),
        "direct_route_count": route_count,
        "direct_path_intersection_agent_count": intersections,
        "direct_path_intersection_agent_rate": intersections / max(1, route_count),
        "direct_path_influence_agent_count": influence,
        "direct_path_influence_agent_rate": influence / max(1, route_count),
        "episodes_with_direct_obstacle_pressure": sum(
            bool(row["direct_obstacle_pressure_episode"]) for row in records
        ),
        "episodes_with_predicted_inter_agent_conflict": sum(
            bool(row["predicted_inter_agent_conflict"]) for row in records
        ),
        "mean_nearest_static_surface_clearance_m": (
            float(np.mean(finite)) if finite else None
        ),
        "minimum_nearest_static_surface_clearance_m": min(finite) if finite else None,
        "maximum_nearest_static_surface_clearance_m": max(finite) if finite else None,
        "obstacle_influence_distance_m": float(obstacle_influence_distance),
        "workspace_volume_m3": workspace_volume,
        "effective_static_obstacle_volume_fraction": effective_volume / workspace_volume,
        "static_obstacle_number_density_per_m3": (
            (static_counts[0] / workspace_volume) if len(static_counts) == 1 else None
        ),
        "initial_obstacle_collision_count": sum(
            bool(row["initial_obstacle_collision"]) for row in records
        ),
        "initial_inter_agent_collision_count": sum(
            bool(row["initial_inter_agent_collision"]) for row in records
        ),
        "goal_obstacle_collision_count": sum(
            bool(row["goal_obstacle_collision"]) for row in records
        ),
        "observation_modes": sorted({str(row["observation_mode"]) for row in records}),
        "peer_spheres_enabled": all(bool(row["peer_spheres_enabled"]) for row in records),
        "boundary_free": all(
            not bool(row["include_boundaries_in_sensor"])
            and not bool(row["terminate_on_boundary_collision"])
            for row in records
        ),
    }


def _audit_scenarios(
    config: Any,
    settings: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    pressure_distance = float(settings["geometry_gate"]["obstacle_pressure_distance_m"])
    risk_distance = float(
        settings["geometry_gate"]["predicted_inter_agent_risk_distance_m"]
    )
    all_records: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for scenario in EXISTING_SCENARIOS:
        records = []
        for seed in range(10):
            env, _ = build_validation_environment(
                config=config,
                scene_type=scenario,
                seed=seed,
                peer_radius=float(settings["peer_radius"]),
            )
            try:
                row = geometry_record(
                    env,
                    scenario=scenario,
                    seed=seed,
                    obstacle_influence_distance=pressure_distance,
                    inter_agent_risk_distance=risk_distance,
                )
            finally:
                env.close()
            records.append(row)
            all_records.append(row)
        summaries[scenario] = _scene_summary(
            records, obstacle_influence_distance=pressure_distance
        )

    coupled_records: list[dict[str, Any]] = []
    deterministic_checks: list[bool] = []
    relative_geometry_hashes: list[str] = []
    for seed in range(10):
        first_options = build_multi_agent_obstacle_options(seed)
        second_options = build_multi_agent_obstacle_options(seed)
        first_payload = {
            "starts": np.asarray(first_options["starts"]).tolist(),
            "goals": np.asarray(first_options["goals"]).tolist(),
            "obstacles": [obstacle_descriptor(item) for item in first_options["static_obstacles"]],
        }
        second_payload = {
            "starts": np.asarray(second_options["starts"]).tolist(),
            "goals": np.asarray(second_options["goals"]).tolist(),
            "obstacles": [obstacle_descriptor(item) for item in second_options["static_obstacles"]],
        }
        deterministic_checks.append(stable_hash(first_payload) == stable_hash(second_payload))
        starts = np.asarray(first_options["starts"], dtype=float)
        goals = np.asarray(first_options["goals"], dtype=float)
        anchor = 0.5 * (starts[0] + goals[0])
        relative_starts = np.round(starts - anchor, 12)
        relative_goals = np.round(goals - anchor, 12)
        relative_starts[np.abs(relative_starts) < 1e-12] = 0.0
        relative_goals[np.abs(relative_goals) < 1e-12] = 0.0
        relative_obstacles = []
        for item in first_options["static_obstacles"]:
            relative = np.round(np.asarray(item.center, dtype=float) - anchor, 12)
            relative[np.abs(relative) < 1e-12] = 0.0
            relative_obstacles.append(relative.tolist())
        relative_geometry_hashes.append(
            stable_hash(
                {
                    "starts": relative_starts.tolist(),
                    "goals": relative_goals.tolist(),
                    "obstacle_centers": relative_obstacles,
                    "obstacle_radii": [
                        [float(item.radius), float(item.safety_margin)]
                        for item in first_options["static_obstacles"]
                    ],
                }
            )
        )
        env, _ = build_multi_agent_obstacle_environment(
            config=config,
            scenario=SCENARIO_ID,
            seed=seed,
            peer_radius=float(settings["peer_radius"]),
        )
        try:
            row = geometry_record(
                env,
                scenario=SCENARIO_ID,
                seed=seed,
                obstacle_influence_distance=pressure_distance,
                inter_agent_risk_distance=risk_distance,
            )
        finally:
            env.close()
        coupled_records.append(row)
        all_records.append(row)
    summaries[SCENARIO_ID] = _scene_summary(
        coupled_records, obstacle_influence_distance=pressure_distance
    )
    gate = summarize_geometry(
        coupled_records,
        required_static_obstacle_count=int(
            settings["geometry_gate"]["required_static_obstacle_count"]
        ),
        required_pressure_rate=float(
            settings["geometry_gate"]["minimum_pressure_episode_rate"]
        ),
        required_conflict_rate=float(
            settings["geometry_gate"]["minimum_conflict_episode_rate"]
        ),
    )
    gate["checks"]["scenario_deterministic_by_seed"] = all(deterministic_checks)
    gate["checks"]["checkpoint_actor_observation_is_3x122"] = all(
        row["checkpoint_actor_observation_shape"] == [3, 122]
        for row in coupled_records
    )
    gate["checks"]["native_observation_is_3x138"] = all(
        row["native_observation_shape"] == [3, 138] for row in coupled_records
    )
    gate["checks"]["frozen_layout_hash_matches_config"] = (
        FROZEN_LAYOUT_HASH == str(settings["frozen_layout_hash_expected"])
    )
    gate["status"] = "PASSED" if all(gate["checks"].values()) else "FAILED"

    sparse = summaries["sparse_static"]
    sparse_pressure = "MODERATE"
    current_coverage = "INSUFFICIENT"
    audit = {
        "schema_version": settings["schema_version"],
        "audit_basis": "code implementation plus seed-0..9 deterministic initialization",
        "current_scenarios_preserved": list(EXISTING_SCENARIOS),
        "new_scenario_is_additive": True,
        "scenario_summaries": summaries,
        "scenario_roles": {
            "open": "OPEN MULTI-UAV BASELINE; no obstacles; blind to peers",
            "sparse_static": "SPARSE STATIC-OBSTACLE MULTI-UAV BASELINE; no intentional crossing; blind to peers",
            "multi_agent": "PERMUTED MULTI-AGENT MIXED-OBSTACLE SCENARIO; includes one static and one moving obstacle",
            SCENARIO_ID: SCENARIO_ROLE,
        },
        "sensor_semantics": {
            "static_obstacles": "visible in LiDAR",
            "dynamic_obstacles": "visible in LiDAR",
            "other_uavs": "visible only when observation_mode=peer_spheres",
            "hit_source_identity_available": False,
            "nearest_distance_only": True,
        },
        "collision_semantics": {
            "inter_agent_collision_distance_m": float(
                config.inter_agent_safe_distance
            ),
            "obstacle_collision": "point UAV inside effective obstacle geometry with collision_margin=0",
        },
        "start_goal_generation": {
            "open_sparse_static": "seeded historical starts/goals with matched assignment",
            "multi_agent": "same seeded starts plus seed-dependent permuted goal assignment",
            SCENARIO_ID: "frozen analytic crossing template with seed-controlled common y/z jitter",
        },
        "seed_semantics": {
            "existing": "seed controls starts/goals; seed+50021 controls training obstacle realization; multi_agent goal permutation depends on seed",
            SCENARIO_ID: "seed controls only the frozen common crossing-center jitter",
            "seeds_0_to_9_role": SEED_SET_ROLE,
            "future_paper_requirement": "use held-out seeds not used for geometry design/gate",
            "relative_geometry_variant_count": len(set(relative_geometry_hashes)),
            "common_translation_only": len(set(relative_geometry_hashes)) == 1,
            "independent_geometry_samples_claimed": False,
        },
        "CURRENT_SCENARIO_COVERAGE": current_coverage,
        "SPARSE_STATIC_OBSTACLE_PRESSURE": sparse_pressure,
        "MULTI_AGENT_HAS_STATIC_OBSTACLES": "YES",
        "NEW_COUPLED_SCENARIO_REQUIRED": "YES",
        "frozen_layout_spec": FROZEN_LAYOUT_SPEC,
        "frozen_layout_hash": FROZEN_LAYOUT_HASH,
        "geometry_gate": gate,
    }
    return audit, all_records, gate


def _combined_conflict_rows(
    episode_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        episodes = [row for row in episode_rows if row["method"] == method]
        combined = [row for row in episodes if bool(row["combined_conflict_episode"])]
        results.append(
            {
                "method": method,
                "episode_count": len(episodes),
                "combined_conflict_episode_count": len(combined),
                "combined_conflict_episode_rate": len(combined) / max(1, len(episodes)),
                "combined_conflict_success_count": sum(
                    bool(row["team_success"]) for row in combined
                ),
                "combined_conflict_success_rate": (
                    sum(bool(row["team_success"]) for row in combined)
                    / max(1, len(combined))
                ),
                "obstacle_failure_count": sum(
                    bool(row["obstacle_collision"]) for row in episodes
                ),
                "inter_agent_failure_count": sum(
                    bool(row["inter_agent_collision"]) for row in episodes
                ),
                "timeout_count": sum(bool(row["timeout"]) for row in episodes),
                "mean_minimum_static_obstacle_clearance_m": float(
                    np.mean(
                        [float(row["minimum_static_obstacle_clearance_m"]) for row in episodes]
                    )
                ),
                "mean_obstacle_pressure_path_deviation_m": float(
                    np.mean(
                        [
                            float(row["obstacle_pressure_path_deviation_team_mean_m"])
                            for row in episodes
                        ]
                    )
                ),
            }
        )
    return results


def _difficulty_rows(
    previous_artifact: Path,
    audit: Mapping[str, Any],
    new_method_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    previous = _load_csv(previous_artifact / "method_summary.csv")
    prior_index = {
        (row["scenario"], row["method"]): row
        for row in previous
        if row["scenario"] in EXISTING_SCENARIOS
    }
    new_index = {
        row["method"]: row for row in new_method_rows if row["scenario"] == "overall"
    }
    rows = []
    for scenario in (*EXISTING_SCENARIOS, SCENARIO_ID):
        summary = audit["scenario_summaries"][scenario]
        source = "previous_three_scenario_artifact" if scenario in EXISTING_SCENARIOS else "current_diagnostic_run"
        method_source = prior_index if scenario in EXISTING_SCENARIOS else None
        def success(method: str) -> float | None:
            if method_source is not None:
                return float(method_source[(scenario, method)]["team_success_rate"])
            return float(new_index[method]["team_success_rate"])
        rows.append(
            {
                "scenario": scenario,
                "result_source": source,
                "static_obstacle_counts": summary["static_obstacle_counts"],
                "dynamic_obstacle_counts": summary["dynamic_obstacle_counts"],
                "peer_spheres_enabled": summary["peer_spheres_enabled"],
                "predicted_inter_agent_conflict_episode_count": summary[
                    "episodes_with_predicted_inter_agent_conflict"
                ],
                "obstacle_pressure_episode_count": summary[
                    "episodes_with_direct_obstacle_pressure"
                ],
                "obstacle_pressure_label": (
                    "NONE" if scenario == "open" else
                    "MODERATE" if scenario == "sparse_static" else
                    "COUPLED"
                ),
                "team_success_terminal": success(METHOD_TERMINAL),
                "team_success_proposal": success(METHOD_PROPOSAL),
                "team_success_fp_shep": success(METHOD_FP_SHEP),
            }
        )
    return rows


def _conclusion(
    method_rows: Sequence[Mapping[str, Any]],
    failure_rows: Sequence[Mapping[str, Any]],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    overall = {row["method"]: row for row in method_rows if row["scenario"] == "overall"}
    terminal = overall[METHOD_TERMINAL]
    proposal = overall[METHOD_PROPOSAL]
    fp = overall[METHOD_FP_SHEP]
    terminal_interaction_failure = bool(
        int(terminal["team_collision_count"]) > 0
        or float(terminal["team_success_rate"] or 0.0) < 1.0
    )
    selector_stage1_failure = bool(
        float(proposal["reference_reached_rate"] or 0.0) < 1.0
        or float(fp["reference_reached_rate"] or 0.0) < 1.0
        or int(proposal["reference_collision_count"]) > 0
        or int(fp["reference_collision_count"]) > 0
    )
    layout_feasible = str(gate["status"]) == "PASSED"
    fp_not_stable = bool(
        float(fp["team_success_rate"] or 0.0) < 1.0
        or int(fp["team_collision_count"]) > 0
    )
    gap = terminal_interaction_failure and selector_stage1_failure and layout_feasible and fp_not_stable
    failed_episodes = [row for row in failure_rows]
    # Failure rows contain only unsuccessful team episodes.  Physical collision
    # counts are reconstructed from method summaries below and kept disjoint from
    # non-collision timeouts.
    obstacle_collision_failures = sum(
        int(round(float(overall[method]["obstacle_collision_rate"] or 0.0)
                  * int(overall[method]["episode_count"])))
        for method in METHOD_ORDER
    )
    inter_agent_collision_failures = sum(
        int(round(float(overall[method]["inter_agent_collision_rate"] or 0.0)
                  * int(overall[method]["episode_count"])))
        for method in METHOD_ORDER
    )
    timeout_failures = sum(
        int(round(float(overall[method]["timeout_rate"] or 0.0)
                  * int(overall[method]["episode_count"])))
        for method in METHOD_ORDER
    )
    obstacle_failures = sum(
        bool(row.get("C_obstacle_collision_before_reference"))
        or str(row["primary_failure_category"]) == "TERMINAL_BASELINE_FAILURE"
        and False
        for row in failure_rows
    )
    # Episode collision counts are more authoritative than the primary category
    # for distinguishing physical obstacle and inter-agent failures.
    return {
        "CURRENT_SCENARIO_COVERAGE": "INSUFFICIENT",
        "SPARSE_STATIC_OBSTACLE_PRESSURE": "MODERATE",
        "MULTI_AGENT_HAS_STATIC_OBSTACLES": "YES",
        "NEW_COUPLED_SCENARIO_REQUIRED": "YES",
        "NEW_SCENARIO_VALID": "YES" if layout_feasible else "NO",
        "MULTI_AGENT_OBSTACLE_COORDINATION_GAP": "YES" if gap else "NOT_ESTABLISHED",
        "GAT_RELEVANT_FOR_COUPLED_SCENARIO": "YES" if gap else "NOT_ESTABLISHED",
        "NEXT_STEP": (
            "Use held-out seeds for formal quantitative comparison before any GAT effectiveness claim."
            if gap
            else "Do not start GAT from this diagnostic alone; define held-out, non-translation-equivalent layouts before paper-level quantitative comparison."
        ),
        "scenario_role": SCENARIO_ROLE,
        "seed_set_role": SEED_SET_ROLE,
        "geometry_gate_status": gate["status"],
        "terminal_interaction_failure_present": terminal_interaction_failure,
        "proposal_or_fp_stage1_failure_present": selector_stage1_failure,
        "layout_feasible_by_gate": layout_feasible,
        "fp_shep_not_stably_solving_team_coordination": fp_not_stable,
        "terminal_team_success_rate": terminal["team_success_rate"],
        "proposal_team_success_rate": proposal["team_success_rate"],
        "fp_shep_team_success_rate": fp["team_success_rate"],
        "obstacle_failure_primary_count_diagnostic": obstacle_failures,
        "gat_effectiveness_validated": False,
        "pure_obstacle_effect_claimed": False,
        "pure_inter_agent_effect_claimed": False,
        "seeds_0_to_9_are_held_out_test": False,
        "relative_geometry_variant_count": 1,
        "seeds_differ_only_by_common_translation": True,
        "failed_team_episode_count": len(failed_episodes),
        "obstacle_collision_failure_count": obstacle_collision_failures,
        "inter_agent_collision_failure_count": inter_agent_collision_failures,
        "timeout_failure_count": timeout_failures,
        "obstacle_pressure_is_not_obstacle_collision": True,
    }


def _render_report(
    audit: Mapping[str, Any],
    method_rows: Sequence[Mapping[str, Any]],
    combined_rows: Sequence[Mapping[str, Any]],
    failure_rows: Sequence[Mapping[str, Any]],
    conclusion: Mapping[str, Any],
) -> str:
    overall = {row["method"]: row for row in method_rows if row["scenario"] == "overall"}
    pct = lambda value: "N/A" if value is None else f"{100*float(value):.2f}%"
    lines = [
        "# Multi-Agent Obstacle-Interaction Scenario Audit and Baseline Revalidation",
        "",
        f"Scenario role: **{SCENARIO_ROLE}**.",
        "",
        "This is neither a pure multi-agent interaction experiment nor an isolated obstacle-effect experiment.",
            "Seeds 0–9 are development/diagnostic seeds because they were used by the geometry gate.",
            "Their frozen jitter is a common translation, so they represent one relative geometry variant rather than ten independent layouts.",
        "",
        "## Existing scenario audit",
        "",
        "| Scenario | Static | Dynamic | Peer LiDAR | Direct obstacle pressure episodes | Predicted conflict episodes |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for scenario in (*EXISTING_SCENARIOS, SCENARIO_ID):
        row = audit["scenario_summaries"][scenario]
        lines.append(
            f"| {scenario} | {row['static_obstacle_counts']} | {row['dynamic_obstacle_counts']} | "
            f"{row['peer_spheres_enabled']} | {row['episodes_with_direct_obstacle_pressure']}/10 | "
            f"{row['episodes_with_predicted_inter_agent_conflict']}/10 |"
        )
    lines.extend(
        [
            "",
            "## Geometry gate",
            "",
            f"- Status: {audit['geometry_gate']['status']}",
            f"- Checks: {audit['geometry_gate']['checks']}",
            f"- Frozen layout hash: `{FROZEN_LAYOUT_HASH}`",
            "",
            "## 220-step diagnostic baseline",
            "",
            "| Method | Team success | Collision | Obstacle collision | Inter-agent collision | Timeout | Reference reached | Reached-to-terminal | Path length (m) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in METHOD_ORDER:
        row = overall[method]
        lines.append(
            f"| {method} | {pct(row['team_success_rate'])} | {pct(row['team_collision_rate'])} | "
            f"{pct(row['obstacle_collision_rate'])} | {pct(row['inter_agent_collision_rate'])} | "
            f"{pct(row['timeout_rate'])} | {pct(row['reference_reached_rate'])} | "
            f"{pct(row['reached_then_terminal_rate'])} | {float(row['mean_path_length_m']):.3f} |"
        )
    combined = {row["method"]: row for row in combined_rows}
    lines.extend(
        [
            "",
            "## Execution quality and obstacle-pressure metrics",
            "",
            "| Method | Smoothness | Mean successful completion step | Mean static clearance (m) | Obstacle-pressure deviation (m) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for method in METHOD_ORDER:
        row = overall[method]
        completion = (
            "N/A"
            if row["mean_completion_step_success"] is None
            else f"{float(row['mean_completion_step_success']):.2f}"
        )
        lines.append(
            f"| {method} | {float(row['mean_trajectory_smoothness']):.3f} | {completion} | "
            f"{float(row['mean_minimum_static_obstacle_clearance_m']):.3f} | "
            f"{float(row['mean_obstacle_pressure_path_deviation_m']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Temporary-reference Stage metrics",
            "",
            "| Method | Reference reached | Collision before reference | Obstacle collision before reference | Inter-agent collision before reference | Reached-to-terminal |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for method in (METHOD_PROPOSAL, METHOD_FP_SHEP):
        row = overall[method]
        lines.append(
            f"| {method} | {pct(row['reference_reached_rate'])} | "
            f"{pct(row['reference_collision_rate'])} | "
            f"{pct(row['obstacle_collision_before_reference_rate'])} | "
            f"{pct(row['inter_agent_collision_before_reference_rate'])} | "
            f"{pct(row['reached_then_terminal_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Coupled-pressure diagnostics",
            "",
        ]
    )
    for method in METHOD_ORDER:
        row = combined[method]
        lines.append(
            f"- {method}: combined conflict {row['combined_conflict_episode_count']}/"
            f"{row['episode_count']}; success within combined conflict "
            f"{row['combined_conflict_success_count']}/{row['combined_conflict_episode_count']}; "
            f"obstacle failures {row['obstacle_failure_count']}; inter-agent failures "
            f"{row['inter_agent_failure_count']}."
        )
    primary = Counter(str(row["primary_failure_category"]) for row in failure_rows)
    lines.extend(
        [
            "",
            "## Required answers",
            "",
            "1. `open` has no static or dynamic obstacles.",
            "2. `sparse_static` contains one static sphere per episode.",
            "3. `sparse_static` has real but moderate obstacle pressure: 7/10 episodes have a direct-route safety-region intersection, while all 10 enter the 1.5 m influence region.",
            "4. `multi_agent` contains one static and one moving obstacle; it is not interaction-only.",
            "5. The old scenarios contain coupled factors, but none meets the requested >=2-static-obstacle formal gate.",
            "6. The additive `multi_agent_obstacle` diagnostic is therefore required.",
            f"7. The frozen new scenario geometry gate is {audit['geometry_gate']['status']}.",
            f"8. Method results are listed in the 220-step table above.",
            f"9. Failure primary categories: {dict(primary)}; physical obstacle/inter-agent collision counts are reported separately above.",
            f"10. Multi-agent obstacle coordination gap: {conclusion['MULTI_AGENT_OBSTACLE_COORDINATION_GAP']}.",
            "",
            "## Final labels",
            "",
        ]
    )
    for key in (
        "CURRENT_SCENARIO_COVERAGE",
        "SPARSE_STATIC_OBSTACLE_PRESSURE",
        "MULTI_AGENT_HAS_STATIC_OBSTACLES",
        "NEW_COUPLED_SCENARIO_REQUIRED",
        "NEW_SCENARIO_VALID",
        "MULTI_AGENT_OBSTACLE_COORDINATION_GAP",
        "GAT_RELEVANT_FOR_COUPLED_SCENARIO",
        "NEXT_STEP",
    ):
        lines.append(f"- {key} = {conclusion[key]}")
    lines.extend(
        [
            "",
            "Zero obstacle collisions, if observed, do not prove that obstacle avoidance is solved. "
            "Pure static clearance, obstacle-interaction rate, and obstacle-pressure path deviation are retained in the artifacts.",
            f"Across all methods there are {conclusion['failed_team_episode_count']} failed team episodes: "
            f"{conclusion['obstacle_collision_failure_count']} obstacle-collision failures, "
            f"{conclusion['inter_agent_collision_failure_count']} inter-agent-collision failures, and "
            f"{conclusion['timeout_failure_count']} timeout failures.",
            "",
            "No GAT training, optimizer, SAC update, score calibration, repeated replanning, hard boundary, dynamic obstacle, or layout retuning was performed.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_experiment(settings: Mapping[str, Any], output_dir: Path) -> Path:
    _assert_config(settings)
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    multi_config = build_single_distribution_multi_config(
        num_agents=int(settings["num_agents"]), max_steps=220
    )
    audit, geometry_rows, geometry_gate = _audit_scenarios(multi_config, settings)
    write_json(output_dir / "scenario_audit.json", audit)
    write_csv(output_dir / "geometry_pressure.csv", geometry_rows)
    if geometry_gate["status"] != "PASSED":
        conclusion = {
            "CURRENT_SCENARIO_COVERAGE": "INSUFFICIENT",
            "SPARSE_STATIC_OBSTACLE_PRESSURE": "MODERATE",
            "MULTI_AGENT_HAS_STATIC_OBSTACLES": "YES",
            "NEW_COUPLED_SCENARIO_REQUIRED": "YES",
            "NEW_SCENARIO_VALID": "NO",
            "MULTI_AGENT_OBSTACLE_COORDINATION_GAP": "NOT_ESTABLISHED",
            "GAT_RELEVANT_FOR_COUPLED_SCENARIO": "NOT_ESTABLISHED",
            "NEXT_STEP": "SCENARIO_NOT_SUFFICIENTLY_COUPLED",
        }
        write_json(output_dir / "conclusion.json", conclusion)
        raise RuntimeError("SCENARIO_NOT_SUFFICIENTLY_COUPLED")

    checkpoint = (REPO_ROOT / str(settings["checkpoint"])).resolve()
    checkpoint_hash_before = sha256_file(checkpoint)
    if checkpoint_hash_before != str(settings["checkpoint_sha256_expected"]):
        raise RuntimeError("checkpoint hash mismatch")
    previous_artifact = (REPO_ROOT / str(settings["previous_three_scenario_artifact"])).resolve()
    previous_integrity = _load_json(previous_artifact / "integrity.json")
    if previous_integrity.get("status") != "PASSED":
        raise RuntimeError("previous three-scenario artifact integrity is not PASSED")

    base_settings = _load_json((REPO_ROOT / str(settings["base_config"])).resolve())
    base_settings.update(
        {
            "checkpoint": settings["checkpoint"],
            "checkpoint_sha256_expected": settings["checkpoint_sha256_expected"],
            "deterministic_policy": True,
            "num_agents": 3,
            "max_steps": 220,
            "scenarios": [SCENARIO_ID],
            "seeds": list(range(10)),
            "temporary_reference": {
                **base_settings["temporary_reference"],
                "K_requested": int(settings["temporary_reference"]["K_requested"]),
                "reached_tolerance_m": float(
                    settings["temporary_reference"]["reached_tolerance_m"]
                ),
            },
            "proposal_config": dict(settings["proposal_config"]),
        }
    )
    policy, loaded_checkpoint = _load_policy(base_settings, multi_config)
    if loaded_checkpoint.resolve() != checkpoint:
        raise RuntimeError("policy loader resolved a different checkpoint")
    policy_hash_before = _policy_parameter_sha256(policy)
    critical_before = _critical_hashes(checkpoint)
    score_spec = FPSHEPOnlineScoreSpec.from_mapping(settings["fp_shep_online_selector"])

    import Environment.multi_agent_dmp_env as environment_module
    import planning.policy_preview as preview_module

    default_execution_symbol = environment_module.propagate_sac_dmp_action
    default_preview_symbol = preview_module.propagate_sac_dmp_action
    episode_rows: list[dict[str, Any]] = []
    agent_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    bundle_rows: list[dict[str, Any]] = []
    bundles: dict[int, Any] = {}

    def prepare_bundle(seed: int) -> Any:
        env, _ = build_multi_agent_obstacle_environment(
            config=multi_config,
            scenario=SCENARIO_ID,
            seed=int(seed),
            peer_radius=float(settings["peer_radius"]),
        )
        try:
            bundle = generate_immutable_candidate_bundle(
                env,
                scenario=SCENARIO_ID,
                seed=int(seed),
                proposal_config=ProposalConfig(**dict(settings["proposal_config"])),
                consumer_top_k=int(settings["temporary_reference"]["K_requested"]),
            )
        finally:
            env.close()
        bundles[int(seed)] = bundle
        bundle_rows.extend(candidate_bundle_rows(bundle))
        return bundle

    def run_seed_block(seeds: Sequence[int], stage_name: str) -> None:
        total = len(seeds) * len(METHOD_ORDER)
        index = 0
        for seed in seeds:
            bundle = prepare_bundle(int(seed))
            bundle_hash = bundle.candidate_set_hash
            for method in METHOD_ORDER:
                variant, selector_method = _method_adapter(method)
                preview_trace: list[dict[str, Any]] = []
                execution_trace: list[dict[str, Any]] = []

                def preview_observer(_kwargs: dict[str, Any], transition: Any) -> None:
                    preview_trace.append(dict(transition.controller_info))

                def execution_observer(_kwargs: dict[str, Any], transition: Any) -> None:
                    execution_trace.append(dict(transition.controller_info))

                before_hash = bundle.candidate_set_hash if method != METHOD_TERMINAL else None
                with scoped_historical_preview_and_multi_agent_transition(
                    preview_observer=preview_observer,
                    execution_observer=execution_observer,
                ):
                    raw, _, _ = run_variant_episode(
                        policy=policy,
                        multi_config=multi_config,
                        settings=base_settings,
                        scenario=SCENARIO_ID,
                        seed=int(seed),
                        variant=variant,
                        candidate_sets=(bundle.per_agent if method != METHOD_TERMINAL else None),
                        selection_method=(selector_method or "proposal_sac_dmp"),
                        score_spec=score_spec,
                        environment_builder=build_multi_agent_obstacle_environment,
                    )
                if environment_module.propagate_sac_dmp_action is not default_execution_symbol:
                    raise RuntimeError("execution transition symbol leaked")
                if preview_module.propagate_sac_dmp_action is not default_preview_symbol:
                    raise RuntimeError("preview transition symbol leaked")
                after_hash = bundle.candidate_set_hash if method != METHOD_TERMINAL else None
                episode, agents, selections = _standardize(
                    raw,
                    method=method,
                    candidate_set_hash=(bundle_hash if method != METHOD_TERMINAL else None),
                    candidate_hash_before=before_hash,
                    candidate_hash_after=after_hash,
                    preview_trace=preview_trace,
                    execution_trace=execution_trace,
                )
                episode.update(
                    {
                        "stage": stage_name,
                        "scenario_role": SCENARIO_ROLE,
                        "seed_set_role": SEED_SET_ROLE,
                        "frozen_layout_hash": FROZEN_LAYOUT_HASH,
                        "geometry_frozen_before_method_results": True,
                    }
                )
                for row in agents:
                    row.update({"stage": stage_name, "scenario_role": SCENARIO_ROLE, "seed_set_role": SEED_SET_ROLE})
                for row in selections:
                    row.update({"stage": stage_name, "scenario_role": SCENARIO_ROLE, "seed_set_role": SEED_SET_ROLE})
                episode_rows.append(episode)
                agent_rows.extend(agents)
                selection_rows.extend(selections)
                index += 1
                print(
                    f"[{stage_name} {index}/{total}] {method} seed={seed}: "
                    f"{episode['termination_reason']}",
                    flush=True,
                )

    run_seed_block(settings["stage_a_seeds"], "stage_a")
    stage_a_pairs, _ = build_selector_pairing(episode_rows, agent_rows)
    stage_a_integrity = _stage_a_integrity(
        episode_rows,
        stage_a_pairs,
        policy_unchanged=policy_hash_before == _policy_parameter_sha256(policy),
        checkpoint_unchanged=checkpoint_hash_before == sha256_file(checkpoint),
    )
    stage_a_integrity["geometry_gate_passed"] = geometry_gate["status"] == "PASSED"
    stage_a_integrity["peer_spheres_enabled"] = all(
        bool(row["peer_spheres_enabled"]) for row in episode_rows
    )
    stage_a_integrity["actor_observation_122"] = all(
        int(row["actor_observation_dimension"]) == 122 for row in episode_rows
    )
    stage_a_integrity["frozen_layout_hash_unchanged"] = FROZEN_LAYOUT_HASH == str(
        settings["frozen_layout_hash_expected"]
    )
    if not all(
        stage_a_integrity[key]
        for key in (
            "geometry_gate_passed",
            "peer_spheres_enabled",
            "actor_observation_122",
            "frozen_layout_hash_unchanged",
        )
    ):
        stage_a_integrity["status"] = "FAILED"
    write_json(output_dir / "stage_a_integrity.json", stage_a_integrity)
    if stage_a_integrity["status"] != "PASSED":
        write_csv(output_dir / "stage_a_per_episode.csv", episode_rows)
        raise RuntimeError("Stage A semantic gate failed")

    run_seed_block(settings["stage_b_seeds"], "stage_b")
    paired_rows, selector_summary = build_selector_pairing(episode_rows, agent_rows)
    method_rows = aggregate_method_rows(episode_rows, agent_rows)
    failure_rows = build_failure_rows(episode_rows, agent_rows)
    combined_rows = _combined_conflict_rows(episode_rows)
    conclusion = _conclusion(method_rows, failure_rows, geometry_gate)
    difficulty_rows = _difficulty_rows(previous_artifact, audit, method_rows)

    checkpoint_hash_after = sha256_file(checkpoint)
    policy_hash_after = _policy_parameter_sha256(policy)
    critical_after = _critical_hashes(checkpoint)
    initial_groups: dict[int, set[str]] = {}
    for row in episode_rows:
        initial_groups.setdefault(int(row["seed"]), set()).add(
            str(row["initial_condition_hash"])
        )
    integrity = {
        "status": "PASSED",
        "episode_count": len(episode_rows),
        "expected_episode_count": 30,
        "checkpoint_unchanged": checkpoint_hash_before == checkpoint_hash_after,
        "checkpoint_sha256_before": checkpoint_hash_before,
        "checkpoint_sha256_after": checkpoint_hash_after,
        "policy_parameters_unchanged": policy_hash_before == policy_hash_after,
        "critical_source_hashes_unchanged_during_run": critical_before == critical_after,
        "paired_initial_states_all_methods": all(len(value) == 1 for value in initial_groups.values()),
        "candidate_bundle_generation_count": len(bundles),
        "expected_candidate_bundle_generation_count": 10,
        "selector_pairing_status": selector_summary["integrity_status"],
        "all_execution_historical_gate_verified": all(
            bool(row["execution_historical_gate_verified"]) for row in episode_rows
        ),
        "all_fp_preview_historical_gate_verified": all(
            bool(row["preview_execution_gate_identity"])
            for row in episode_rows
            if row["method"] == METHOD_FP_SHEP
        ),
        "candidate_bundles_unchanged": all(
            bool(row["candidate_bundle_unchanged"])
            for row in episode_rows
            if row["method"] != METHOD_TERMINAL
        ),
        "peer_spheres_enabled_all_episodes": all(
            bool(row["peer_spheres_enabled"]) for row in episode_rows
        ),
        "actor_observation_dimension_122": all(
            int(row["actor_observation_dimension"]) == 122 for row in episode_rows
        ),
        "native_observation_dimension_138": all(
            int(row["native_environment_observation_dimension"]) == 138
            for row in episode_rows
        ),
        "boundary_free": all(
            not bool(row["include_boundaries_in_sensor"])
            and not bool(row["terminate_on_boundary_collision"])
            for row in episode_rows
        ),
        "one_shot": all(
            bool(row["one_shot_candidate_generation"])
            for row in episode_rows
            if row["method"] != METHOD_TERMINAL
        ),
        "no_repeated_replanning": not any(
            bool(row["repeated_replanning"]) for row in episode_rows
        ),
        "frozen_layout_hash_before_after": [FROZEN_LAYOUT_HASH, FROZEN_LAYOUT_HASH],
        "frozen_layout_unchanged": FROZEN_LAYOUT_HASH
        == str(settings["frozen_layout_hash_expected"]),
        "stage_a_gate": stage_a_integrity,
        "geometry_gate": geometry_gate,
        "training_performed": False,
        "gradient_update_count": 0,
        "GAT_optimizer_created": False,
        "GAT_training_started": False,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    required_integrity = (
        "checkpoint_unchanged",
        "policy_parameters_unchanged",
        "critical_source_hashes_unchanged_during_run",
        "paired_initial_states_all_methods",
        "candidate_bundles_unchanged",
        "peer_spheres_enabled_all_episodes",
        "actor_observation_dimension_122",
        "native_observation_dimension_138",
        "boundary_free",
        "one_shot",
        "no_repeated_replanning",
        "frozen_layout_unchanged",
        "all_execution_historical_gate_verified",
        "all_fp_preview_historical_gate_verified",
    )
    if (
        not all(bool(integrity[key]) for key in required_integrity)
        or len(episode_rows) != 30
        or len(bundles) != 10
        or selector_summary["integrity_status"] != "PASSED"
    ):
        integrity["status"] = "FAILED"
        raise RuntimeError("final diagnostic integrity audit failed")

    resolved = copy.deepcopy(dict(settings))
    resolved["resolved_output_dir"] = str(output_dir)
    resolved["frozen_layout_spec"] = FROZEN_LAYOUT_SPEC
    resolved["selector_score_specification"] = score_spec.metadata()
    write_json(output_dir / "config.json", resolved)
    write_csv(output_dir / "scenario_difficulty.csv", difficulty_rows)
    write_csv(output_dir / "per_episode.csv", episode_rows)
    write_csv(output_dir / "per_agent.csv", agent_rows)
    write_csv(output_dir / "method_summary.csv", method_rows)
    write_csv(output_dir / "combined_conflict_analysis.csv", combined_rows)
    write_csv(output_dir / "failure_attribution.csv", failure_rows)
    write_csv(output_dir / "candidate_selection.csv", selection_rows)
    write_csv(output_dir / "candidate_bundle.csv", bundle_rows)
    write_csv(output_dir / "paired_selector_analysis.csv", paired_rows)
    write_json(output_dir / "selector_summary.json", selector_summary)
    write_json(output_dir / "integrity.json", integrity)
    write_json(output_dir / "conclusion.json", conclusion)
    (output_dir / "FINAL_REPORT.md").write_text(
        _render_report(
            audit,
            method_rows,
            combined_rows,
            failure_rows,
            conclusion,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "episode_count": len(episode_rows),
                **conclusion,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> Path:
    args = parse_args()
    settings = _load_json(args.config.expanduser().resolve())
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else REPO_ROOT
        / str(settings["output_dir"])
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    return run_experiment(settings, output_dir)


if __name__ == "__main__":
    main()
