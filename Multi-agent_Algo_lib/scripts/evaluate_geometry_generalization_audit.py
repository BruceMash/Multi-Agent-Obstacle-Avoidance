"""Run the frozen non-translation-equivalent geometry generalization audit."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = Path(__file__).resolve().parent
for search_path in (REPO_ROOT, ALGO_ROOT, SCRIPTS_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from Guidance.reference_point_proposal_demo import ProposalConfig  # noqa: E402
from planning.geometry_generalization_analysis import (  # noqa: E402
    aggregate_method_summary,
    build_conclusion,
    build_fp_shep_failure_analysis,
    build_paired_method_analysis,
    build_selected_tuple_compatibility,
)
from planning.geometry_generalization_scenarios import (  # noqa: E402
    FROZEN_HELD_OUT_LAYOUTS,
    FROZEN_LAYOUT_TABLE_SHA256,
    LAYOUT_SET_ROLE,
    NUM_AGENTS,
    PEER_RADIUS_M,
    SCENARIO_ID,
    SCENARIO_ROLE,
    analytic_geometry_gate,
    build_environment_options,
    build_scenario_manifest,
    layout_by_id,
    relative_geometry_record,
    stable_hash,
    validate_manifest_geometry,
)
from planning.historical_forcing_gate import (  # noqa: E402
    HISTORICAL_GATE_NAME,
    scoped_historical_preview_and_multi_agent_transition,
)
from planning.multi_agent_obstacle_scenario_audit import (  # noqa: E402
    build_multi_agent_obstacle_options,
)
from planning.pre_gat_220step_revalidation import (  # noqa: E402
    METHOD_DISPLAY_NAMES,
    METHOD_FP_SHEP,
    METHOD_ORDER,
    METHOD_PROPOSAL,
    METHOD_TERMINAL,
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
    _standardize,
)
from scripts.evaluate_pre_gat_closed_loop import (  # noqa: E402
    _critical_hashes,
    _policy_parameter_sha256,
)
from scripts.evaluate_single_policy_aligned_multi_agent import (  # noqa: E402
    build_single_distribution_multi_config,
)
from scripts.evaluate_single_policy_multi_agent import _build_environment  # noqa: E402
from scripts.evaluate_temporary_reference_interface import _load_policy  # noqa: E402


DEFAULT_CONFIG_PATH = REPO_ROOT / "configs/evaluation/geometry_generalization_audit.json"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_config(settings: Mapping[str, Any]) -> None:
    if str(settings["scenario"]) != SCENARIO_ID:
        raise ValueError("scenario contract changed")
    if str(settings["scenario_role"]) != SCENARIO_ROLE:
        raise ValueError("scenario role changed")
    if str(settings["layout_set_role"]) != LAYOUT_SET_ROLE:
        raise ValueError("layout-set role changed")
    if int(settings["layout_count"]) != 24 or len(FROZEN_HELD_OUT_LAYOUTS) != 24:
        raise ValueError("formal layout count must remain 24")
    if list(settings["families"]) != list("ABCDEF"):
        raise ValueError("geometry families must remain A..F")
    if tuple(settings["methods"]) != METHOD_ORDER:
        raise ValueError("method order changed")
    if int(settings["num_agents"]) != NUM_AGENTS:
        raise ValueError("audit requires exactly three UAVs")
    if int(settings["max_steps"]) != 220:
        raise ValueError("max_steps must remain 220")
    if not np.isclose(float(settings["peer_radius"]), PEER_RADIUS_M):
        raise ValueError("peer radius changed")
    if str(settings["frozen_layout_table_sha256_expected"]) != FROZEN_LAYOUT_TABLE_SHA256:
        raise ValueError("frozen layout table hash mismatch")
    if len(settings["stage_2_integrity_layouts"]) > 4:
        raise ValueError("Stage 2 may use at most four frozen layouts")
    if any(bool(value) for value in settings["strict_exclusions"].values()):
        raise ValueError("strict exclusion flags must remain false")
    spec = FPSHEPOnlineScoreSpec.from_mapping(settings["fp_shep_online_selector"])
    metadata = spec.metadata()
    if int(metadata["H_preview"]) != 4:
        raise ValueError("H_preview must remain 4")
    if float(metadata["weights"]["terminal_speed"]) != 0.0:
        raise ValueError("terminal speed must remain excluded from online ranking")
    semantics = settings["execution_semantics"]
    if str(semantics["historical_gate"]) != HISTORICAL_GATE_NAME:
        raise ValueError("historical gate contract changed")
    if not bool(semantics["one_shot"]) or bool(semantics["repeated_replanning"]):
        raise ValueError("one-shot execution contract changed")
    if bool(semantics["include_boundaries_in_sensor"]) or bool(
        semantics["terminate_on_boundary_collision"]
    ):
        raise ValueError("audit must remain boundary-free")


def _manifest_file_sha256(path: Path) -> str:
    return sha256_file(path)


def _assert_manifest_frozen(path: Path, expected_sha256: str, expected_bytes: bytes) -> None:
    current_bytes = path.read_bytes()
    if current_bytes != expected_bytes:
        raise RuntimeError("frozen scenario manifest bytes changed")
    if _manifest_file_sha256(path) != expected_sha256:
        raise RuntimeError("frozen scenario manifest SHA256 changed")


def _layout_from_evaluation_seed(seed: int):
    index = int(seed) - 1000
    if index < 0 or index >= len(FROZEN_HELD_OUT_LAYOUTS):
        raise KeyError(f"unknown geometry evaluation seed: {seed}")
    return FROZEN_HELD_OUT_LAYOUTS[index]


def build_generalization_environment(
    *,
    config: Any,
    scenario: str,
    seed: int,
    peer_radius: float,
) -> tuple[Any, dict[str, Any]]:
    """Build one frozen layout without registering a training scenario."""

    if str(scenario) != SCENARIO_ID:
        raise ValueError(f"expected scenario {SCENARIO_ID!r}, got {scenario!r}")
    if int(config.num_agents) != NUM_AGENTS:
        raise ValueError("generalization audit requires exactly three agents")
    if not np.isclose(float(peer_radius), PEER_RADIUS_M):
        raise ValueError("peer radius differs from frozen manifest")
    layout = _layout_from_evaluation_seed(int(seed))
    options = build_environment_options(layout)
    env = _build_environment(
        config,
        observation_mode="peer_spheres",
        peer_radius=float(peer_radius),
        training_distribution=False,
        include_boundaries_in_sensor=False,
        terminate_on_boundary_collision=False,
    )
    env.reset(seed=int(seed), options=copy.deepcopy(options))
    starts, goals, obstacles = layout.geometry()
    return env, {
        "scene_type": SCENARIO_ID,
        "scenario_role": SCENARIO_ROLE,
        "scenario_geometry_version": "frozen_non_translation_equivalent_v1",
        "scenario_geometry_hash": stable_hash(
            {
                "starts": starts,
                "goals": goals,
                "obstacle_centers": [item.center for item in obstacles],
            }
        ),
        "seed_set_role": LAYOUT_SET_ROLE,
        "layout_id": layout.layout_id,
        "family": layout.family,
        "observation_mode": "peer_spheres",
        "peer_spheres_enabled": True,
        "include_boundaries_in_sensor": False,
        "terminate_on_boundary_collision": False,
        "static_obstacle_count": 2,
        "dynamic_obstacle_count": 0,
        "geometry_frozen_before_method_results": True,
    }


def _legacy_relative_hash() -> str:
    options = build_multi_agent_obstacle_options(0)
    return stable_hash(
        relative_geometry_record(
            np.asarray(options["starts"], dtype=float),
            np.asarray(options["goals"], dtype=float),
            options["static_obstacles"],
        )
    )


def _geometry_audit_rows(
    manifest: Mapping[str, Any], settings: Mapping[str, Any]
) -> list[dict[str, Any]]:
    gates = settings["geometry_gate"]
    manifest_by_id = {str(row["layout_id"]): row for row in manifest["layouts"]}
    rows: list[dict[str, Any]] = []
    for layout in FROZEN_HELD_OUT_LAYOUTS:
        audit = analytic_geometry_gate(
            layout,
            obstacle_influence_distance_m=float(gates["obstacle_influence_distance_m"]),
            predicted_inter_agent_risk_distance_m=float(
                gates["predicted_inter_agent_risk_distance_m"]
            ),
            near_orthogonal_tolerance_deg=float(
                gates["near_orthogonal_3d_tolerance_deg"]
            ),
            near_orthogonal_time_difference_max=float(
                gates["near_orthogonal_time_difference_max_fraction"]
            ),
        )
        descriptor = audit["descriptor"]
        record = manifest_by_id[layout.layout_id]
        rows.append(
            {
                "layout_id": layout.layout_id,
                "evaluation_seed": int(record["evaluation_seed"]),
                "family": layout.family,
                "status": audit["status"],
                "failed_checks": audit["failed_checks"],
                "layout_hash": record["layout_hash"],
                "relative_geometry_hash": record["relative_geometry_hash"],
                "minimum_initial_inter_agent_distance_m": descriptor[
                    "minimum_initial_inter_agent_distance_m"
                ],
                "minimum_terminal_inter_agent_distance_m": descriptor[
                    "minimum_terminal_inter_agent_distance_m"
                ],
                "route_lengths_m": descriptor["route_lengths_m"],
                "near_orthogonal_route_angle_3d_deg": descriptor[
                    "near_orthogonal_route_angle_3d_deg"
                ],
                "near_orthogonal_route_angle_xy_deg": descriptor[
                    "near_orthogonal_route_angle_xy_deg"
                ],
                "direct_path_predicted_minimum_inter_agent_distance_m": descriptor[
                    "direct_path_predicted_minimum_inter_agent_distance_m"
                ],
                "predicted_closest_approach_time_fraction": descriptor[
                    "predicted_closest_approach_time_fraction"
                ],
                "predicted_closest_approach_time_difference_fraction": descriptor[
                    "predicted_closest_approach_time_difference_fraction"
                ],
                "near_orthogonal_pair_predicted_minimum_distance_m": descriptor[
                    "near_orthogonal_pair_predicted_minimum_distance_m"
                ],
                "near_orthogonal_pair_time_difference_fraction": descriptor[
                    "near_orthogonal_pair_time_difference_fraction"
                ],
                "near_orthogonal_pair_vertical_separation_m": descriptor[
                    "near_orthogonal_pair_vertical_separation_m"
                ],
                "number_of_direct_obstacle_intersecting_paths": descriptor[
                    "number_of_direct_obstacle_intersecting_paths"
                ],
                "number_of_direct_obstacle_influence_paths": descriptor[
                    "number_of_direct_obstacle_influence_paths"
                ],
                "minimum_direct_path_obstacle_clearance_m": descriptor[
                    "minimum_direct_path_obstacle_clearance_m"
                ],
                "minimum_obstacle_to_crossing_distance_m": descriptor[
                    "minimum_obstacle_to_crossing_distance_m"
                ],
                "third_agent_proximity_m": descriptor["third_agent_proximity_m"],
                "method_outcome_used": audit["method_outcome_used"],
                "solvability_statement": audit["solvability_statement"],
                **{f"gate__{key}": value for key, value in audit["checks"].items()},
            }
        )
    return rows


def _stage_integrity(
    episode_rows: Sequence[Mapping[str, Any]],
    selection_rows: Sequence[Mapping[str, Any]],
    *,
    expected_layout_ids: Sequence[str],
    checkpoint_unchanged: bool,
    policy_unchanged: bool,
    manifest_unchanged: bool,
) -> dict[str, Any]:
    expected_episodes = len(expected_layout_ids) * len(METHOD_ORDER)
    initial: dict[str, set[str]] = defaultdict(set)
    candidate_hashes: dict[str, set[str]] = defaultdict(set)
    order_pairs: dict[tuple[str, int], dict[str, str]] = defaultdict(dict)
    for row in episode_rows:
        initial[str(row["layout_id"])].add(str(row["initial_condition_hash"]))
        if row["method"] in {METHOD_PROPOSAL, METHOD_FP_SHEP}:
            candidate_hashes[str(row["layout_id"])].add(str(row["candidate_set_hash"]))
    for row in selection_rows:
        order_pairs[(str(row["layout_id"]), int(row["agent_id"]))][
            str(row["method"])
        ] = str(row["candidate_order_hash"])
    checks = {
        "expected_episode_count": len(episode_rows) == expected_episodes,
        "expected_layout_set": {str(row["layout_id"]) for row in episode_rows}
        == set(expected_layout_ids),
        "paired_initial_state": all(len(values) == 1 for values in initial.values()),
        "candidate_hash_match": all(len(values) == 1 for values in candidate_hashes.values()),
        "candidate_order_match": all(
            values.get(METHOD_PROPOSAL) == values.get(METHOD_FP_SHEP)
            for values in order_pairs.values()
        ),
        "candidate_bundle_unchanged": all(
            bool(row["candidate_bundle_unchanged"])
            for row in episode_rows
            if row["method"] != METHOD_TERMINAL
        ),
        "preview_execution_gate_identity": all(
            bool(row["preview_execution_gate_identity"])
            for row in episode_rows
            if row["method"] == METHOD_FP_SHEP
        ),
        "execution_historical_gate": all(
            bool(row["execution_historical_gate_verified"]) for row in episode_rows
        ),
        "H_preview_4": all(
            int(row["H_preview"]) == 4
            for row in episode_rows
            if row["method"] == METHOD_FP_SHEP
        ),
        "one_shot": all(
            not bool(row["repeated_replanning"]) for row in episode_rows
        ),
        "max_steps_220": all(int(row["max_steps"]) == 220 for row in episode_rows),
        "actor_observation_122": all(
            int(row["actor_observation_dimension"]) == 122 for row in episode_rows
        ),
        "native_observation_138": all(
            int(row["native_environment_observation_dimension"]) == 138
            for row in episode_rows
        ),
        "peer_spheres_enabled": all(bool(row["peer_spheres_enabled"]) for row in episode_rows),
        "boundary_free": all(
            not bool(row["include_boundaries_in_sensor"])
            and not bool(row["terminate_on_boundary_collision"])
            for row in episode_rows
        ),
        "checkpoint_unchanged": checkpoint_unchanged,
        "policy_unchanged": policy_unchanged,
        "manifest_unchanged": manifest_unchanged,
        "outcome_not_used_for_integrity_gate": True,
        "no_training": all(not bool(row["training_performed"]) for row in episode_rows),
        "no_GAT": all(not bool(row["GAT_used"]) for row in episode_rows),
    }
    return {
        "status": "PASSED" if all(checks.values()) else "FAILED",
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "layout_ids": list(expected_layout_ids),
        "method_outcomes_used_to_modify_geometry": False,
        "method_outcomes_used_in_integrity_gate": False,
    }


def _render_report(
    *,
    manifest_sha256: str,
    geometry_audit: Mapping[str, Any],
    method_summary: Sequence[Mapping[str, Any]],
    family_summary: Sequence[Mapping[str, Any]],
    paired_summary: Mapping[str, Any],
    failure_rows: Sequence[Mapping[str, Any]],
    conclusion: Mapping[str, Any],
    integrity: Mapping[str, Any],
) -> str:
    methods = {str(row["method"]): row for row in method_summary}

    def pct(value: Any) -> str:
        return "N/A" if value is None else f"{100.0 * float(value):.1f}%"

    lines = [
        "# Non-Translation-Equivalent Multi-Agent Geometry Generalization Audit",
        "",
        "## Frozen geometry contract",
        "",
        f"- Scenario manifest SHA256: `{manifest_sha256}`",
        f"- Layouts: {geometry_audit['layout_count']} across Families A-F (4 per family).",
        f"- Relative geometry variants: {geometry_audit['relative_geometry_variant_count']}.",
        f"- Geometry gate: {geometry_audit['status']}.",
        "- Family C uses actual 3D synchronized closest approach; XY angle is auxiliary only.",
        "- Feasibility is analytic only: finite spheres in boundary-free R3 do not seal free space.",
        "- No Proposal, FP-SHEP, SAC rollout, planner, GAT, or method outcome entered geometry acceptance.",
        "",
        "## Overall method results",
        "",
        "| Method | Team success | Obstacle collision | Inter-agent collision | Timeout | Reference reached | Reached-to-terminal | Path length (m) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHOD_ORDER:
        row = methods[method]
        lines.append(
            f"| {METHOD_DISPLAY_NAMES[method]} | {row['team_success_count']}/{row['episode_count']} "
            f"({pct(row['team_success_rate'])}) | {row['obstacle_collision_count']}/{row['episode_count']} | "
            f"{row['inter_agent_collision_count']}/{row['episode_count']} | "
            f"{row['timeout_count']}/{row['episode_count']} | "
            f"{pct(row['reference_reached_rate'])} | {pct(row['reached_then_terminal_rate'])} | "
            f"{float(row['mean_path_length_m']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Family-stratified results",
            "",
            "| Family | Method | Team success | Inter-agent collision | Timeout |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in family_summary:
        lines.append(
            f"| {row['family']} | {METHOD_DISPLAY_NAMES[str(row['method'])]} | "
            f"{row['team_success_count']}/{row['episode_count']} ({pct(row['team_success_rate'])}) | "
            f"{row['inter_agent_collision_count']}/{row['episode_count']} | "
            f"{row['timeout_count']}/{row['episode_count']} |"
        )
    primary = Counter(str(row["primary_failure_category"]) for row in failure_rows)
    residual_by_family = Counter(
        str(row["family"])
        for row in failure_rows
        if bool(row["residual_multi_agent_coordination_signature"])
    )
    lines.extend(
        [
            "",
            "## Paired Proposal vs FP-SHEP",
            "",
            f"- Both success: {paired_summary['both_success_count']}",
            f"- Proposal only success: {paired_summary['proposal_only_success_count']}",
            f"- FP-SHEP only success: {paired_summary['fp_shep_only_success_count']}",
            f"- Both fail: {paired_summary['both_fail_count']}",
            f"- Selector disagreement: {paired_summary['selector_disagreement_count']}/"
            f"{paired_summary['selector_comparable_agent_count']} "
            f"({pct(paired_summary['selector_disagreement_rate'])}); K=0 pairs excluded.",
            "",
            "## FP-SHEP residual failures",
            "",
            f"- Failure count: {len(failure_rows)}",
            f"- Primary categories: {dict(primary)}",
            f"- Residual coordination signatures: {conclusion['residual_coordination_failure_count']}",
            f"- Individual-executable but joint-conflict cases: {conclusion['individual_vs_joint_case_count']}",
            "",
            "## Required answers",
            "",
            f"1. Non-translation-equivalent layouts: {'YES' if geometry_audit['relative_geometry_variant_count'] == 24 else 'NO'}.",
            f"2. Terminal / Proposal / FP-SHEP success: {pct(methods[METHOD_TERMINAL]['team_success_rate'])} / "
            f"{pct(methods[METHOD_PROPOSAL]['team_success_rate'])} / {pct(methods[METHOD_FP_SHEP]['team_success_rate'])}.",
            f"3. FP-SHEP advantage over Proposal: {100.0 * float(conclusion['fp_shep_minus_proposal_team_success_rate']):+.1f} percentage points.",
            f"4. FP-SHEP residual failure types: {dict(primary)}.",
            f"5. Residual failures with explicit coordination signature: {conclusion['residual_coordination_failure_count']}.",
            f"6. Stable individual-vs-joint cases: {conclusion['INDIVIDUAL_VS_JOINT_COMPATIBILITY_GAP']}.",
            f"7. Residual coordination failures by family: {dict(residual_by_family)}; family-level FP-SHEP rates: {conclusion['fp_shep_family_success_rates']}.",
            f"8. FP-SHEP sufficiently solves the defined problem: {'YES' if float(conclusion['fp_shep_team_success_rate']) >= 0.9 else 'NO'}.",
            f"9. Evidence that individual executability differs from joint compatibility: {conclusion['INDIVIDUAL_VS_JOINT_COMPATIBILITY_GAP']}.",
            f"10. GAT core necessity: {conclusion['GAT_CORE_NECESSITY']}.",
            "",
            "## Final labels",
            "",
        ]
    )
    for key in (
        "GEOMETRY_GENERALIZATION_VALID",
        "FP_SHEP_GENERALIZATION_SIGNAL",
        "RESIDUAL_MULTI_AGENT_COORDINATION_GAP",
        "INDIVIDUAL_VS_JOINT_COMPATIBILITY_GAP",
        "GAT_CORE_NECESSITY",
        "PROCEED_TO_GAT_STAGE_I",
        "PRIMARY_LIMITATION",
        "NEXT_STEP",
    ):
        lines.append(f"- {key} = {conclusion[key]}")
    lines.extend(
        [
            "",
            f"Integrity status: {integrity['status']}.",
            "Counts, rates, rate differences, family consistency, and paired outcomes are descriptive; no statistical significance is claimed.",
            "No GAT training, GAT dataset creation, SAC fine-tuning, Proposal/FP-SHEP modification, repeated replanning, dynamic obstacle, hard boundary, or hyperparameter search was performed.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_experiment(settings: Mapping[str, Any], output_dir: Path) -> Path:
    _assert_config(settings)
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()

    # Stage 1: generate, audit, write, and freeze the complete manifest before
    # checkpoint loading or any method evaluation.
    manifest = build_scenario_manifest()
    legacy_hash = _legacy_relative_hash()
    geometry_audit = validate_manifest_geometry(
        manifest, legacy_relative_hashes=[legacy_hash]
    )
    geometry_rows = _geometry_audit_rows(manifest, settings)
    manifest_path = output_dir / "scenario_manifest.json"
    write_json(manifest_path, manifest)
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha256 = _manifest_file_sha256(manifest_path)
    (output_dir / "SCENARIO_MANIFEST_SHA256.txt").write_text(
        manifest_sha256 + "\n", encoding="ascii"
    )
    write_csv(output_dir / "geometry_audit.csv", geometry_rows)
    stage_1 = {
        "status": geometry_audit["status"],
        "scenario_manifest_sha256": manifest_sha256,
        "frozen_layout_table_sha256": FROZEN_LAYOUT_TABLE_SHA256,
        "legacy_relative_geometry_hash": legacy_hash,
        "geometry_audit": geometry_audit,
        "checkpoint_loaded": False,
        "method_evaluation_started": False,
        "manifest_frozen": geometry_audit["status"] == "PASSED",
    }
    write_json(output_dir / "stage_1_geometry_freeze.json", stage_1)
    resolved_config = copy.deepcopy(dict(settings))
    resolved_config["resolved_output_dir"] = str(output_dir)
    resolved_config["SCENARIO_MANIFEST_SHA256"] = manifest_sha256
    resolved_config["manifest_frozen_before_checkpoint_load"] = True
    write_json(output_dir / "config.json", resolved_config)
    if geometry_audit["status"] != "PASSED":
        conclusion = {
            "GEOMETRY_GENERALIZATION_VALID": "NO",
            "FP_SHEP_GENERALIZATION_SIGNAL": "WEAK",
            "RESIDUAL_MULTI_AGENT_COORDINATION_GAP": "NO",
            "INDIVIDUAL_VS_JOINT_COMPATIBILITY_GAP": "NOT_ESTABLISHED",
            "GAT_CORE_NECESSITY": "NOT_ESTABLISHED",
            "PROCEED_TO_GAT_STAGE_I": "NO",
            "PRIMARY_LIMITATION": "GEOMETRY_MANIFEST_OR_ACCEPTANCE_GATE_INVALID",
            "NEXT_STEP": "Fix geometry code/manifest semantics only; no method evaluation was run.",
        }
        write_json(output_dir / "conclusion.json", conclusion)
        raise RuntimeError(f"geometry gate failed: {geometry_audit['failed_checks']}")
    _assert_manifest_frozen(manifest_path, manifest_sha256, manifest_bytes)

    checkpoint = (REPO_ROOT / str(settings["checkpoint"])).resolve()
    checkpoint_hash_before = sha256_file(checkpoint)
    if checkpoint_hash_before != str(settings["checkpoint_sha256_expected"]):
        raise RuntimeError("checkpoint hash mismatch")
    base_settings = _load_json((REPO_ROOT / str(settings["base_config"])).resolve())
    base_settings.update(
        {
            "checkpoint": settings["checkpoint"],
            "checkpoint_sha256_expected": settings["checkpoint_sha256_expected"],
            "deterministic_policy": True,
            "num_agents": NUM_AGENTS,
            "max_steps": 220,
            "scenarios": [SCENARIO_ID],
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
    multi_config = build_single_distribution_multi_config(
        num_agents=NUM_AGENTS, max_steps=220
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
    bundles: dict[str, Any] = {}
    manifest_index = {str(row["layout_id"]): row for row in manifest["layouts"]}

    def prepare_bundle(layout_id: str) -> Any:
        record = manifest_index[layout_id]
        evaluation_seed = int(record["evaluation_seed"])
        env, _ = build_generalization_environment(
            config=multi_config,
            scenario=SCENARIO_ID,
            seed=evaluation_seed,
            peer_radius=float(settings["peer_radius"]),
        )
        try:
            bundle = generate_immutable_candidate_bundle(
                env,
                scenario=SCENARIO_ID,
                seed=evaluation_seed,
                proposal_config=ProposalConfig(**dict(settings["proposal_config"])),
                consumer_top_k=int(settings["temporary_reference"]["K_requested"]),
            )
        finally:
            env.close()
        bundles[layout_id] = bundle
        for row in candidate_bundle_rows(bundle):
            row.update(
                {
                    "layout_id": layout_id,
                    "family": record["family"],
                    "layout_hash": record["layout_hash"],
                }
            )
            bundle_rows.append(row)
        return bundle

    def run_layout_block(layout_ids: Sequence[str], stage_name: str) -> None:
        total = len(layout_ids) * len(METHOD_ORDER)
        counter = 0
        for layout_id in layout_ids:
            record = manifest_index[layout_id]
            evaluation_seed = int(record["evaluation_seed"])
            bundle = prepare_bundle(layout_id)
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
                        seed=evaluation_seed,
                        variant=variant,
                        candidate_sets=(
                            bundle.per_agent if method != METHOD_TERMINAL else None
                        ),
                        selection_method=selector_method or "proposal_sac_dmp",
                        score_spec=score_spec,
                        environment_builder=build_generalization_environment,
                    )
                if environment_module.propagate_sac_dmp_action is not default_execution_symbol:
                    raise RuntimeError("execution transition symbol leaked after scope")
                if preview_module.propagate_sac_dmp_action is not default_preview_symbol:
                    raise RuntimeError("preview transition symbol leaked after scope")
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
                common = {
                    "layout_id": layout_id,
                    "family": record["family"],
                    "layout_hash": record["layout_hash"],
                    "relative_geometry_hash": record["relative_geometry_hash"],
                    "evaluation_stage": stage_name,
                    "scenario_manifest_sha256": manifest_sha256,
                }
                episode.update(common)
                for row in agents:
                    row.update(common)
                for row in selections:
                    row.update(common)
                episode_rows.append(episode)
                agent_rows.extend(agents)
                selection_rows.extend(selections)
                counter += 1
                print(
                    f"[{stage_name} {counter}/{total}] {method} {layout_id}: "
                    f"{episode['termination_reason']}",
                    flush=True,
                )

    stage_2_ids = [str(value) for value in settings["stage_2_integrity_layouts"]]
    run_layout_block(stage_2_ids, "stage_2_integrity_subset")
    _assert_manifest_frozen(manifest_path, manifest_sha256, manifest_bytes)
    stage_2_integrity = _stage_integrity(
        episode_rows,
        selection_rows,
        expected_layout_ids=stage_2_ids,
        checkpoint_unchanged=checkpoint_hash_before == sha256_file(checkpoint),
        policy_unchanged=policy_hash_before == _policy_parameter_sha256(policy),
        manifest_unchanged=True,
    )
    stage_2_integrity["method_outcome_summary_diagnostic_only"] = {
        method: {
            "success_count": sum(
                bool(row["team_success"])
                for row in episode_rows
                if row["method"] == method
            ),
            "episode_count": sum(row["method"] == method for row in episode_rows),
        }
        for method in METHOD_ORDER
    }
    write_json(output_dir / "stage_2_integrity.json", stage_2_integrity)
    if stage_2_integrity["status"] != "PASSED":
        write_csv(output_dir / "stage_2_per_episode.csv", episode_rows)
        raise RuntimeError(
            f"Stage 2 semantic integrity failed: {stage_2_integrity['failed_checks']}"
        )

    remaining_ids = [
        layout.layout_id
        for layout in FROZEN_HELD_OUT_LAYOUTS
        if layout.layout_id not in set(stage_2_ids)
    ]
    run_layout_block(remaining_ids, "stage_3_full_frozen_set")
    _assert_manifest_frozen(manifest_path, manifest_sha256, manifest_bytes)

    paired_selector_rows, paired_selector_summary = build_selector_pairing(
        episode_rows, agent_rows
    )
    if paired_selector_summary["integrity_status"] != "PASSED":
        raise RuntimeError(
            f"B/C candidate fairness failed: {paired_selector_summary['integrity_errors']}"
        )
    method_summary = aggregate_method_summary(episode_rows, agent_rows)
    family_summary = aggregate_method_summary(
        episode_rows, agent_rows, group_fields=("family", "method")
    )
    compatibility_rows = build_selected_tuple_compatibility(
        episode_rows, agent_rows, selection_rows, manifest
    )
    failure_rows = build_fp_shep_failure_analysis(
        episode_rows,
        agent_rows,
        compatibility_rows,
        inter_agent_pressure_distance_m=float(
            settings["compatibility_diagnostic"]["inter_agent_pressure_distance_m"]
        ),
    )
    paired_rows, paired_summary = build_paired_method_analysis(
        episode_rows, selection_rows
    )
    conclusion = build_conclusion(
        method_summary,
        family_summary,
        paired_summary,
        failure_rows,
        geometry_valid=geometry_audit["status"] == "PASSED",
        config=settings,
    )

    checkpoint_hash_after = sha256_file(checkpoint)
    policy_hash_after = _policy_parameter_sha256(policy)
    critical_after = _critical_hashes(checkpoint)
    _assert_manifest_frozen(manifest_path, manifest_sha256, manifest_bytes)
    full_integrity = _stage_integrity(
        episode_rows,
        selection_rows,
        expected_layout_ids=[layout.layout_id for layout in FROZEN_HELD_OUT_LAYOUTS],
        checkpoint_unchanged=checkpoint_hash_before == checkpoint_hash_after,
        policy_unchanged=policy_hash_before == policy_hash_after,
        manifest_unchanged=True,
    )
    full_integrity.update(
        {
            "episode_count": len(episode_rows),
            "expected_episode_count": 72,
            "agent_record_count": len(agent_rows),
            "expected_agent_record_count": 216,
            "candidate_bundle_count": len(bundles),
            "expected_candidate_bundle_count": 24,
            "checkpoint_sha256_before": checkpoint_hash_before,
            "checkpoint_sha256_after": checkpoint_hash_after,
            "policy_parameter_sha256_before": policy_hash_before,
            "policy_parameter_sha256_after": policy_hash_after,
            "critical_source_hashes_unchanged": critical_before == critical_after,
            "scenario_manifest_sha256_before_after": [
                manifest_sha256,
                _manifest_file_sha256(manifest_path),
            ],
            "frozen_layout_table_sha256": FROZEN_LAYOUT_TABLE_SHA256,
            "default_execution_symbol_restored": (
                environment_module.propagate_sac_dmp_action is default_execution_symbol
            ),
            "default_preview_symbol_restored": (
                preview_module.propagate_sac_dmp_action is default_preview_symbol
            ),
            "selector_pairing_integrity": paired_selector_summary,
            "gradient_update_count": 0,
            "GAT_inference_count": 0,
            "GAT_optimizer_created": False,
            "SAC_fine_tuning_performed": False,
            "layout_modified_after_manifest_freeze": False,
            "outcome_based_layout_rejection_count": 0,
            "runtime_seconds": float(time.perf_counter() - started),
        }
    )
    additional_checks = {
        "episode_and_agent_counts": len(episode_rows) == 72 and len(agent_rows) == 216,
        "candidate_bundle_count": len(bundles) == 24,
        "critical_source_hashes_unchanged": critical_before == critical_after,
        "default_execution_symbol_restored": (
            environment_module.propagate_sac_dmp_action is default_execution_symbol
        ),
        "default_preview_symbol_restored": (
            preview_module.propagate_sac_dmp_action is default_preview_symbol
        ),
        "selector_pairing_integrity": paired_selector_summary["integrity_status"]
        == "PASSED",
    }
    full_integrity["checks"].update(additional_checks)
    full_integrity["failed_checks"] = [
        key for key, value in full_integrity["checks"].items() if not value
    ]
    full_integrity["status"] = (
        "PASSED" if not full_integrity["failed_checks"] else "FAILED"
    )
    if full_integrity["status"] != "PASSED":
        raise RuntimeError(f"final integrity failed: {full_integrity['failed_checks']}")

    resolved_config["selector_score_specification"] = score_spec.metadata()
    resolved_config["runtime_seconds"] = float(time.perf_counter() - started)
    write_json(output_dir / "config.json", resolved_config)
    write_csv(output_dir / "per_episode.csv", episode_rows)
    write_csv(output_dir / "per_agent.csv", agent_rows)
    write_csv(output_dir / "candidate_selection.csv", selection_rows)
    write_csv(output_dir / "candidate_bundle.csv", bundle_rows)
    write_csv(output_dir / "method_summary.csv", method_summary)
    write_csv(output_dir / "family_summary.csv", family_summary)
    write_csv(output_dir / "paired_method_analysis.csv", paired_rows)
    write_csv(output_dir / "paired_selector_integrity.csv", paired_selector_rows)
    write_csv(output_dir / "fp_shep_failure_analysis.csv", failure_rows)
    write_csv(output_dir / "compatibility_diagnostics.csv", compatibility_rows)
    write_json(output_dir / "paired_summary.json", paired_summary)
    write_json(output_dir / "integrity.json", full_integrity)
    write_json(output_dir / "conclusion.json", conclusion)
    (output_dir / "FINAL_REPORT.md").write_text(
        _render_report(
            manifest_sha256=manifest_sha256,
            geometry_audit=geometry_audit,
            method_summary=method_summary,
            family_summary=family_summary,
            paired_summary=paired_summary,
            failure_rows=failure_rows,
            conclusion=conclusion,
            integrity=full_integrity,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "episode_count": len(episode_rows),
                "manifest_sha256": manifest_sha256,
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
    settings = _load_json(args.config.resolve())
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = REPO_ROOT / str(settings["output_dir"]) / stamp
    else:
        output_dir = args.output_dir.resolve()
    return run_experiment(settings, output_dir)


if __name__ == "__main__":
    main()
