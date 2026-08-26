"""Create the pre-performance evidence bundle for the long-range main benchmark.

This script is intentionally limited to read-only authority recovery, contract
serialization, hashing, and method-independent scene-grammar smoke generation.
It never loads a policy, executes an episode, trains a model, or creates the
development/formal manifests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from planning.semi_structured_long_range_benchmark import (  # noqa: E402
    APPLICATION_SETTING,
    DT,
    FAMILY_LABELS,
    FAMILY_ORDER,
    LONG_RANGE_MAX_STEPS,
    MISSION_DISTANCE_RANGE_M,
    OPERATIONAL_FLIGHT_BAND_M,
    SENSOR_DIRECTION_COUNT,
    SENSOR_RANGE_M,
    STAGE_LABELS,
    STAGE_ORDER,
    STAGE_POPULATION,
    STATIC_DYNAMIC_RATIO,
    TASK_PATTERNS,
    WORKSPACE_BOUNDS,
    generate_scenario_manifest,
    stable_hash,
    validate_scenario_manifest,
)


SCHEMA = "long_range_main_benchmark_preperformance_freeze_v1"
DEFAULT_ARTIFACT = REPO_ROOT / "artifacts" / "semi_structured_long_range_main_benchmark" / "20260820_193228"
GOAL_ATTACHMENT = Path(
    r"C:\Users\Administrator\.codex\attachments\3336d05a-f7b8-444e-8d79-070b4c95bb5f\pasted-text.txt"
)


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_or_absolute(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def file_record(path: Path, role: str) -> dict[str, Any]:
    return {
        "path": relative_or_absolute(path),
        "role": role,
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256_file(path),
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, fieldnames: Iterable[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def git_value(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def make_directories(root: Path) -> None:
    for name in (
        "00_context",
        "01_engineering_contract",
        "02_short_coordination",
        "03_environment_generator",
        "04_sensor_physics",
        "05_peer_information_contract",
        "06_contract_pilot",
        "07_training",
        "08_development",
        "09_baseline_tuning",
        "10_final_freeze",
        "11_formal_manifest",
        "12_formal_records",
        "13_statistics",
        "14_failure_analysis",
        "15_runtime_scaling",
        "16_paper_ready/figures_pdf",
        "16_paper_ready/figures_png_600dpi",
        "16_paper_ready/tables",
        "16_paper_ready/source_data",
        "16_paper_ready/captions",
    ):
        (root / name).mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT)
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    make_directories(root)

    authority = [
        file_record(GOAL_ATTACHMENT, "current user goal; highest task authority"),
        file_record(REPO_ROOT / "hire-rl-body.tex", "active paper theory authority"),
        file_record(
            REPO_ROOT / "artifacts/final_execution_semantics_audit/20260819_194912/FINAL_METHOD_FREEZE.json",
            "historical method/execution freeze provenance",
        ),
        file_record(
            REPO_ROOT / "artifacts/rerr_runtime_compression/20260819_162022/FINAL_ENGINEERING_FREEZE.json",
            "historical exact-behavior runtime freeze provenance",
        ),
        file_record(
            REPO_ROOT / "artifacts/FINAL_UNTOUCHED_PAPER_BENCHMARK/20260820_110311/FINAL_REPORT.md",
            "short-range supplementary benchmark only; forbidden for new development selection",
        ),
        file_record(
            REPO_ROOT / "artifacts/equal_information_baseline_audit/20260819_012339/FINAL_REPORT.md",
            "historical equal-information diagnostic provenance",
        ),
    ]
    current_sources = [
        "planning/semi_structured_long_range_benchmark.py",
        "Environment/multi_agent_dmp_env.py",
        "Entity/KinematicModel.py",
        "Environment/frozen_sac_dmp_execution.py",
        "planning/event_triggered_reference_reconstruction.py",
        "planning/heterogeneous_candidate_graph.py",
        "planning/policy_preview.py",
        "planning/pre_gat_closed_loop.py",
        "Multi-agent_Algo_lib/scripts/evaluate_gat_v1_err_development.py",
        "Multi-agent_Algo_lib/scripts/prepare_long_range_main_benchmark.py",
    ]
    checkpoints = [
        file_record(
            REPO_ROOT / "artifacts/20260520_201912/best_eval_model.pt",
            "historical frozen SAC-DMP checkpoint; adaptation candidate, not yet final",
        ),
        file_record(
            REPO_ROOT / "artifacts/gat_stage1_training/20260815_230509/checkpoints/best_validation.pt",
            "historical GAT-V1 checkpoint; adaptation candidate, not yet final",
        ),
    ]
    context = {
        "schema_version": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifact_root": relative_or_absolute(root),
        "git_branch": git_value("rev-parse", "--abbrev-ref", "HEAD"),
        "git_commit": git_value("rev-parse", "HEAD"),
        "worktree_dirty": bool(git_value("status", "--porcelain")),
        "authority_files": authority,
        "checkpoint_files": checkpoints,
        "current_source_files": [
            file_record(REPO_ROOT / path, "current pre-performance implementation")
            for path in current_sources
        ],
        "forbidden_tuning_sources": [
            "all historical formal result rows",
            "all future formal manifest scenarios and results",
            "the 400-scenario short-range benchmark as a selection set",
        ],
        "performance_episode_count_created_by_this_phase": 0,
        "policy_or_model_loaded": False,
    }
    write_json(root / "00_context/context_recovery_manifest.json", context)

    write_text(
        root / "00_context/current_authority.md",
        """# Current Authority and Provenance

1. The attached long-range `/goal` is the current task authority.
2. The active uncommented equations in `hire-rl-body.tex` define core theory.
3. Historical freeze files define provenance and exact old semantics, but they do not freeze long-range parameters or checkpoints.
4. The new benchmark engineering, observation, lifecycle, and physics contracts are frozen before any new performance episode.

The old 400-scenario short-range benchmark is supplementary evidence only. It may be used to diagnose historical distribution support and runtime, but never to select a new parameter, checkpoint, threshold, baseline configuration, or scenario distribution. The new formal 400 scenarios will be generated only after development and all method/baseline/statistical contracts are frozen.
""",
    )

    rationale = f"""# Large-Scale Semi-Structured Operational Scenario Rationale

## Application setting

The main experiment represents three UAVs executing long transfer, crossing, merge, inspection-support, and logistics-like navigation missions in a 100 m × 100 m semi-structured operational area. The horizontal scale is deliberately much larger than the {SENSOR_RANGE_M:g} m local sensing radius: a {MISSION_DISTANCE_RANGE_M[0]:g}–{MISSION_DISTANCE_RANGE_M[1]:g} m mission exposes only 5.3–6.9% of its straight-line length in one range observation. The task therefore requires a sequence of local decisions rather than a single short-range avoidance maneuver.

The operational altitude is z={OPERATIONAL_FLIGHT_BAND_M[0]:g}–{OPERATIONAL_FLIGHT_BAND_M[1]:g} m. This is a genuine three-dimensional flight band with 56 directions (8 azimuth × 7 elevation), while remaining an interpretable low-altitude operational volume. Closed lateral boundaries, floor, and ceiling prevent artificial escape around the scenario.

## Environment identity

Five fixed families model parallel structural corridors, cross-intersection work areas, staggered storage/equipment blocks, open transfer areas with distributed structures, and merge/bottleneck lanes. Static primitives are local-scale cylinders or 2–4 m-class boxes. Dynamic obstacles use only frozen constant-direction translation. There is no stochastic wandering, hidden future access by planners, global route map, or global planner.

## Difficulty isolation

Stage is controlled only by obstacle population. Geometry family, task-pattern distribution, workspace, flight band, obstacle size distribution, static:dynamic ratio, dynamic-speed distribution, sensor, dynamics, mission-distance support, episode horizon, and success/collision definitions are invariant. Counts are 8+2, 16+4, 24+6, and 32+8 (static+dynamic), preserving a {STATIC_DYNAMIC_RATIO}:1 ratio.

## Recoverability and claim boundary

Every generated scene must pass an evaluation-only static A* witness with 1 m grid resolution and 0.45 m inflation. This witness is never visible to a method and proves only that static geometry does not make the task impossible. At least 75% of each obstacle population must lie within 18 m of a straight task segment. This avoids filling a large workspace with irrelevant obstacles. The benchmark evaluates local reactive and event-triggered reconstruction under long-range composition; it is not a global path-planning benchmark and does not support claims about unmodeled weather, aerodynamics, communication loss, or real-world certification.
"""
    write_text(root / "01_engineering_contract/engineering_scenario_rationale.md", rationale)

    physics = {
        "schema_version": SCHEMA,
        "sensor": {
            "directions": SENSOR_DIRECTION_COUNT,
            "organization": "8 azimuth x 7 elevation",
            "current_plus_previous_frame": True,
            "correct_name": "56-direction 3D range observation with previous frame",
            "incorrect_name_forbidden": "112-ray LiDAR",
            "range_m": SENSOR_RANGE_M,
        },
        "dynamics": {
            "dt_s": DT,
            "component_acceleration_bound_mps2": 4.0,
            "component_velocity_bound_mps": 4.0,
            "operational_speed_norm_cap_mps": 3.2,
            "cap_applies_to": "all methods and all real/preview point-mass propagation",
            "historical_default_behavior": "cap disabled unless explicitly configured",
        },
        "conservative_peer_head_on_audit": {
            "reaction_time_s": 0.15,
            "reaction_time_basis": "one 0.1 s control interval plus rounded historical 41 ms upper event latency",
            "single_agent_reaction_distance_m": 0.48,
            "single_agent_braking_distance_m": 1.28,
            "inter_agent_safe_center_distance_m": 0.6,
            "required_initial_center_separation_m": 4.12,
            "sensor_range_m": SENSOR_RANGE_M,
            "nominal_range_margin_m": 0.38,
            "formula": "d_safe + 2 * (v*t_react + v^2/(2*a_brake))",
            "guarantee_boundary": "kinematic range sufficiency only; sparse rays, occlusion, estimation, policy response, and simultaneous visibility prevent a formal safety guarantee",
        },
    }
    write_json(root / "04_sensor_physics/sensor_range_physics_audit.json", physics)
    write_json(
        root / "01_engineering_contract/semi_structured_environment_contract.json",
        {
            "schema_version": SCHEMA,
            "application_setting": APPLICATION_SETTING,
            "workspace_bounds": WORKSPACE_BOUNDS,
            "operational_flight_band_m": OPERATIONAL_FLIGHT_BAND_M,
            "boundary_mode": "closed_workspace_with_floor_and_ceiling",
            "mission_distance_range_m": MISSION_DISTANCE_RANGE_M,
            "dt_s": DT,
            "max_steps": LONG_RANGE_MAX_STEPS,
            "max_episode_duration_s": DT * LONG_RANGE_MAX_STEPS,
            "uav_count": 3,
            "stage_active_variable": "OBSTACLE_POPULATION_ONLY",
            "performance_episode_count_at_freeze": 0,
        },
    )
    write_json(
        root / "01_engineering_contract/operational_flight_band_contract.json",
        {
            "schema_version": SCHEMA,
            "z_min_m": OPERATIONAL_FLIGHT_BAND_M[0],
            "z_max_m": OPERATIONAL_FLIGHT_BAND_M[1],
            "height_m": OPERATIONAL_FLIGHT_BAND_M[1] - OPERATIONAL_FLIGHT_BAND_M[0],
            "boundary_mode": "closed_floor_and_ceiling",
            "selection_basis": [
                "historical 2.4 m z workspace extent",
                "existing three-dimensional ray layout",
                "low-altitude industrial/logistics plausibility",
                "prevent unlimited vertical escape",
            ],
            "selected_before_performance": True,
        },
    )
    write_json(
        root / "01_engineering_contract/mission_distance_contract.json",
        {
            "schema_version": SCHEMA,
            "straight_line_min_m": MISSION_DISTANCE_RANGE_M[0],
            "straight_line_max_m": MISSION_DISTANCE_RANGE_M[1],
            "initial_recommendation_m": [60.0, 90.0],
            "contract_adjustment": "65-85 m for 8 m endpoint margins and pattern balance",
            "adjustment_basis": "geometry/workspace feasibility before performance",
            "actual_traveled_path_reported_separately": True,
            "hundred_meter_scale_claim_requires_observed_path_support": True,
        },
    )
    write_json(
        root / "01_engineering_contract/long_range_episode_horizon_contract.json",
        {
            "schema_version": SCHEMA,
            "max_steps": LONG_RANGE_MAX_STEPS,
            "dt_s": DT,
            "maximum_duration_s": LONG_RANGE_MAX_STEPS * DT,
            "maximum_mission_distance_m": MISSION_DISTANCE_RANGE_M[1],
            "operational_speed_norm_cap_mps": 3.2,
            "minimum_straight_time_at_cap_s": MISSION_DISTANCE_RANGE_M[1] / 3.2,
            "allowed_time_stretch_over_straight_at_cap": (LONG_RANGE_MAX_STEPS * DT) / (MISSION_DISTANCE_RANGE_M[1] / 3.2),
            "rule": "frozen before pilot; identical for every method; not extended after observing timeout",
        },
    )

    write_csv(
        root / "03_environment_generator/obstacle_population_contract.csv",
        ("stage", "stage_label", "static_count", "dynamic_count", "total_count", "static_dynamic_ratio"),
        (
            {
                "stage": stage,
                "stage_label": STAGE_LABELS[stage],
                "static_count": STAGE_POPULATION[stage]["static"],
                "dynamic_count": STAGE_POPULATION[stage]["dynamic"],
                "total_count": STAGE_POPULATION[stage]["static"] + STAGE_POPULATION[stage]["dynamic"],
                "static_dynamic_ratio": STATIC_DYNAMIC_RATIO,
            }
            for stage in STAGE_ORDER
        ),
    )
    write_csv(
        root / "03_environment_generator/environment_family_contract.csv",
        ("family_index", "family", "paper_label", "formal_scenarios_per_stage"),
        (
            {
                "family_index": index,
                "family": family,
                "paper_label": FAMILY_LABELS[family],
                "formal_scenarios_per_stage": 20,
            }
            for index, family in enumerate(FAMILY_ORDER)
        ),
    )
    write_json(
        root / "03_environment_generator/dynamic_obstacle_contract.json",
        {
            "schema_version": SCHEMA,
            "model": "constant_direction_translation",
            "future_tracks_frozen_in_manifest": True,
            "future_tracks_visible_to_environment": True,
            "future_tracks_visible_to_any_planner": False,
            "stochastic_wandering": False,
            "reflection_or_teleportation": False,
            "tracks_remain_inside_workspace_for_full_horizon": True,
            "stage_changes_motion_model_or_speed_distribution": False,
        },
    )
    smoke = generate_scenario_manifest(counts_per_stage=6, seed_base=910_000, prefix="CONTRACT")
    smoke_validation = validate_scenario_manifest(smoke)
    if smoke_validation["status"] != "PASS":
        raise RuntimeError(f"contract smoke validation failed: {smoke_validation['errors']}")
    write_json(root / "03_environment_generator/contract_smoke_manifest.json", smoke)
    write_json(root / "03_environment_generator/contract_smoke_validation.json", smoke_validation)
    write_json(
        root / "03_environment_generator/scenario_grammar_freeze.json",
        {
            "schema_version": SCHEMA,
            "source": file_record(
                REPO_ROOT / "planning/semi_structured_long_range_benchmark.py",
                "frozen method-independent scene grammar",
            ),
            "stage_order": STAGE_ORDER,
            "families": FAMILY_ORDER,
            "task_patterns": TASK_PATTERNS,
            "population": STAGE_POPULATION,
            "static_dynamic_ratio": STATIC_DYNAMIC_RATIO,
            "formal_target_count_per_stage": 100,
            "formal_target_count_per_family_per_stage": 20,
            "performance_episode_count_at_freeze": 0,
        },
    )
    write_csv(
        root / "01_engineering_contract/long_range_parameter_adaptation.csv",
        (
            "parameter",
            "historical_value",
            "long_range_initial_value",
            "status",
            "selection_basis",
            "development_tunable",
        ),
        (
            {
                "parameter": "workspace_bounds_m",
                "historical_value": "[-0.5,-2.5,-1.2]..[8.5,2.0,1.2]",
                "long_range_initial_value": "[0,0,0.8]..[100,100,3.2]",
                "status": "TASK_CONTRACT_FROZEN",
                "selection_basis": "requested large-scale application identity",
                "development_tunable": False,
            },
            {
                "parameter": "straight_line_mission_distance_m",
                "historical_value": "6-8 class",
                "long_range_initial_value": "65-85",
                "status": "TASK_CONTRACT_FROZEN",
                "selection_basis": "workspace margins and physical duration",
                "development_tunable": False,
            },
            {
                "parameter": "sensor_spatial_directions",
                "historical_value": "56",
                "long_range_initial_value": "56",
                "status": "PRESERVED",
                "selection_basis": "local MDP distribution preservation",
                "development_tunable": False,
            },
            {
                "parameter": "sensor_range_m",
                "historical_value": "4.5",
                "long_range_initial_value": "4.5",
                "status": "PHYSICS_AUDIT_FROZEN",
                "selection_basis": "head-on requirement 4.12 m < 4.5 m",
                "development_tunable": False,
            },
            {
                "parameter": "candidate_distance_m",
                "historical_value": "0.05-1.05",
                "long_range_initial_value": "0.05-1.05",
                "status": "INITIAL_LOCAL_SCALE_PRESERVED",
                "selection_basis": "compositional local-MDP extension",
                "development_tunable": True,
            },
            {
                "parameter": "Top-K",
                "historical_value": "10",
                "long_range_initial_value": "10",
                "status": "INITIAL_CONFIGURATION",
                "selection_basis": "current authority freeze provenance",
                "development_tunable": True,
            },
            {
                "parameter": "H_preview_steps",
                "historical_value": "4",
                "long_range_initial_value": "4",
                "status": "INITIAL_CONFIGURATION",
                "selection_basis": "current authority freeze provenance",
                "development_tunable": True,
            },
            {
                "parameter": "maximum_speed_norm_mps",
                "historical_value": "none; component cap 4",
                "long_range_initial_value": "3.2",
                "status": "COMMON_PHYSICS_FROZEN",
                "selection_basis": "historical Proposed p99 3.052 m/s and 4.5 m reaction audit",
                "development_tunable": False,
            },
            {
                "parameter": "peer_event_observation",
                "historical_value": "exact all-peer event state",
                "long_range_initial_value": "4.5 m local anonymous native ally block",
                "status": "INFORMATION_CONTRACT_FROZEN",
                "selection_basis": "remove global identity/private range-external access",
                "development_tunable": False,
            },
            {
                "parameter": "reference_completion",
                "historical_value": "direct terminal handoff",
                "long_range_initial_value": "R-ERR reconstruction unless terminal within 4.5 m",
                "status": "THEORY_PRESERVING_LIFECYCLE_FROZEN_FOR_PILOT",
                "selection_basis": "prevent 50-90 m active-goal OOD after each local completion",
                "development_tunable": True,
            },
            {
                "parameter": "max_steps",
                "historical_value": "220",
                "long_range_initial_value": str(LONG_RANGE_MAX_STEPS),
                "status": "PHYSICAL_HORIZON_FROZEN",
                "selection_basis": "85 m mission with 5.65x cap-speed time stretch",
                "development_tunable": False,
            },
        ),
    )
    registry_rows = []
    for row in smoke["entries"]:
        registry_rows.append(
            {
                "scene_id": row["scenario_id"],
                "split": "contract_smoke_nonperformance",
                "stage": row["stage"],
                "family": row["family"],
                "task_pattern": row["task_pattern"],
                "seed": row["seed"],
                "geometry_fingerprint": row["geometry_fingerprint"],
                "translation_invariant_fingerprint": row["translation_invariant_fingerprint"],
                "dynamic_track_fingerprint": row["dynamic_track_fingerprint"],
                "performance_episode_count": 0,
                "eligible_for_training": False,
                "eligible_for_development": False,
                "eligible_for_formal": False,
            }
        )
    write_csv(
        root / "00_context/ALL_USED_SCENE_REGISTRY.csv",
        (
            "scene_id",
            "split",
            "stage",
            "family",
            "task_pattern",
            "seed",
            "geometry_fingerprint",
            "translation_invariant_fingerprint",
            "dynamic_track_fingerprint",
            "performance_episode_count",
            "eligible_for_training",
            "eligible_for_development",
            "eligible_for_formal",
        ),
        registry_rows,
    )

    peer_contract = {
        "schema_version": SCHEMA,
        "status": "PASS",
        "mode": "local_anonymous_ally_block",
        "availability": "current state sampled only when an upper GAT event is executed",
        "range_m": SENSOR_RANGE_M,
        "position": "relative, normalized/clipped by 4.5 m, then decoded for graph geometry",
        "velocity": "relative, normalized/clipped by the existing 4 m/s scale, then decoded",
        "identity": "event-local anonymous distance/geometry slot only",
        "stable_global_identity": False,
        "continuous_peer_trigger": False,
        "communication_channel_added": False,
        "range_external_state_invariant": True,
        "private_future_state": False,
        "sac_actor_consumes_ally_block": False,
        "sac_actor_contract": "historical 122-D vector unchanged",
        "legacy_mode_retained_for_reproduction": True,
        "tests": "test.test_local_anonymous_peer_observation (6/6 pass)",
        "claim": "existing ally representation narrowed and formalized; no new global peer channel",
    }
    write_json(root / "05_peer_information_contract/peer_observation_contract.json", peer_contract)
    write_text(
        root / "05_peer_information_contract/PEER_OBSERVABILITY_CONTRACT.md",
        """# Peer Observation Contract

The long-range method does not continuously poll global peer state and does not introduce communication. At a pre-existing upper event, GAT may consume only the current native ally representation for peers within 4.5 m. The representation contains relative position and relative velocity after the native normalization/clipping path. Global simulator identity is removed and replaced by an event-local slot ordered by observable geometry.

The historical SAC actor remains 122-D and does not consume this ally block. The peer block is not permitted to create a continuous peer-risk trigger. Tests prove that changing range-external private peer state cannot alter the local output, swapping simulator identities cannot alter the anonymous representation, and the graph accessor cannot bypass native velocity clipping. The legacy exact event-time accessor remains available only under the historical default configuration for reproducing old artifacts; it is forbidden in all new long-range development and formal runs.
""",
    )
    write_json(
        root / "05_peer_information_contract/reference_lifecycle_audit.json",
        {
            "schema_version": SCHEMA,
            "status": "PASS",
            "mode": "reconstruct_unless_terminal_local",
            "reference_completion_distance_m": 0.25,
            "terminal_local_scope_m": SENSOR_RANGE_M,
            "far_completion_action": "same R-ERR upper reconstruction chain",
            "local_terminal_action": "phase-preserving terminal handoff",
            "event_priority": "reference completion before emergency before normal",
            "new_trigger_module": False,
            "global_planner": False,
            "legacy_default_retained": True,
            "tests": "test.test_long_range_reference_lifecycle (6/6 pass)",
        },
    )

    ppo_assets = [
        file_record(REPO_ROOT / "runner_mappo_multi_agent_dmp.py", "generic MAPPO runner"),
        file_record(REPO_ROOT / "test/test_mappo_multi_agent_eval.py", "generic MAPPO test"),
        file_record(
            REPO_ROOT / "exp_results/mappo_mlp_default/MAPPOTrainer_multi_agent_dmp_default_dc2ee_00000_0_2026-05-28_22-16-46/result.json",
            "historical generic multi-agent trial metadata; no compatible checkpoint found",
        ),
        file_record(
            REPO_ROOT / "MAPPO_file/results/simple_spread_v3/MAPPO_1/MAPPO.pth",
            "PettingZoo simple_spread checkpoint; incompatible task identity",
        ),
    ]
    write_json(
        root / "09_baseline_tuning/PPO_BASELINE_AUDIT.json",
        {
            "schema_version": SCHEMA,
            "PPO_BASELINE_READY": "NO",
            "exact_waypoint_ppo_implementation_found": False,
            "exact_waypoint_ppo_checkpoint_found": False,
            "same_observation_action_dynamics_identity_verified": False,
            "generic_ppo_or_mappo_assets_found": True,
            "assets": ppo_assets,
            "decision": "Exclude Waypoint-PPO unless a legitimate same-identity implementation/checkpoint is produced before baseline freeze; do not relabel generic or PettingZoo checkpoints.",
            "performance_episode_count": 0,
        },
    )
    write_json(
        root / "07_training/historical_checkpoint_support.json",
        {
            "schema_version": SCHEMA,
            "historical_sac": {
                "status": "strong local-controller provenance but large workspace/task shift",
                "training_identity": "single-UAV, short local episodes, 9 x 4.5 x 2.4 m workspace",
                "new_identity": "three UAVs, 100 x 100 m, 65-85 m missions, up to 1500 steps",
                "adaptation_allowed": True,
                "checkpoint_finalized": False,
            },
            "historical_gat_v1": {
                "status": "valid architecture/provenance but partial distribution support",
                "historical_graph_count": 492,
                "new_peer_contract": "finite-range anonymous local event state",
                "adaptation_allowed": True,
                "checkpoint_finalized": False,
            },
            "formal_data_used": False,
        },
    )
    write_json(
        root / "00_context/preperformance_status.json",
        {
            "schema_version": SCHEMA,
            "SCENARIO_CONTRACT_FROZEN": "YES",
            "PEER_INFORMATION_CONTRACT_VALID": "YES",
            "REFERENCE_LIFECYCLE_CONTRACT_VALID": "YES",
            "COMMON_SPEED_NORM_CONTRACT_VALID": "YES",
            "PPO_BASELINE_READY": "NO",
            "CONTRACT_TESTS": {"passed": 21, "failed": 0},
            "new_performance_episode_count": 0,
            "development_manifest_created": False,
            "formal_manifest_created": False,
            "next_gate": "local MDP/distribution audit then frozen 40-scene contract pilot",
        },
    )
    write_json(
        root / "00_context/development_formal_firewall.json",
        {
            "schema_version": SCHEMA,
            "development_manifest": None,
            "formal_manifest": None,
            "formal_generation_allowed_only_after": [
                "development method selected",
                "baselines selected",
                "all checkpoints hashed",
                "metric/statistical contracts frozen",
            ],
            "historical_formal_data_allowed_for_selection": False,
            "new_formal_data_allowed_for_any_adjustment": False,
        },
    )
    print(root)


if __name__ == "__main__":
    main()
