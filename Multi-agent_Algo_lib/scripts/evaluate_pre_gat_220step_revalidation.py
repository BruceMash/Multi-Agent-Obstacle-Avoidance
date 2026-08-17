"""Run the 220-step one-shot Proposal vs FP-SHEP Pre-GAT revalidation."""

from __future__ import annotations

import argparse
import copy
import json
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
from planning.goal_semantics_diagnosis import VARIANT_A, VARIANT_D  # noqa: E402
from planning.historical_forcing_gate import (  # noqa: E402
    HISTORICAL_GATE_NAME,
    scoped_historical_preview_and_multi_agent_transition,
)
from planning.pre_gat_220step_revalidation import (  # noqa: E402
    METHOD_DISPLAY_NAMES,
    METHOD_FP_SHEP,
    METHOD_ORDER,
    METHOD_PROPOSAL,
    METHOD_TERMINAL,
    SCHEMA_VERSION,
    aggregate_method_rows,
    build_conclusion,
    build_failure_rows,
    build_selector_pairing,
    candidate_bundle_rows,
    generate_immutable_candidate_bundle,
    stable_hash,
)
from planning.pre_gat_closed_loop import FPSHEPOnlineScoreSpec  # noqa: E402
from planning.reference_transition_finetuning import sha256_file  # noqa: E402
from scripts.evaluate_actor_dmp_goal_semantics import (  # noqa: E402
    run_variant_episode,
    write_csv,
    write_json,
)
from scripts.evaluate_pre_gat_closed_loop import (  # noqa: E402
    _critical_hashes,
    _policy_parameter_sha256,
    _scenario_hash,
    _scene_snapshot,
    build_closed_loop_environment,
)
from scripts.evaluate_single_policy_aligned_multi_agent import (  # noqa: E402
    build_single_distribution_multi_config,
)
from scripts.evaluate_temporary_reference_interface import _load_policy  # noqa: E402


DEFAULT_CONFIG_PATH = (
    REPO_ROOT / "configs/evaluation/pre_gat_220step_revalidation.json"
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_config(settings: Mapping[str, Any]) -> None:
    if int(settings["max_steps"]) != 220:
        raise ValueError("revalidation max_steps must be 220")
    if list(settings["scenarios"]) != ["open", "sparse_static", "multi_agent"]:
        raise ValueError("scenario contract changed")
    if list(settings["stage_a_seeds"]) != list(range(5)):
        raise ValueError("Stage A seeds must be 0..4")
    if list(settings["stage_b_seeds"]) != list(range(5, 10)):
        raise ValueError("Stage B seeds must be 5..9")
    if tuple(settings["methods"]) != METHOD_ORDER:
        raise ValueError("method order changed")
    if any(bool(value) for value in settings["strict_exclusions"].values()):
        raise ValueError("strict exclusion flags must remain false")
    spec = FPSHEPOnlineScoreSpec.from_mapping(settings["fp_shep_online_selector"])
    metadata = spec.metadata()
    if metadata["H_preview"] != 4:
        raise ValueError("FP-SHEP H_preview must remain 4")
    if metadata["weights"]["terminal_speed"] != 0.0:
        raise ValueError("terminal speed must not enter online ranking")


def _method_adapter(method: str) -> tuple[str, str | None]:
    if method == METHOD_TERMINAL:
        return VARIANT_A, None
    if method == METHOD_PROPOSAL:
        return VARIANT_D, "proposal_sac_dmp"
    if method == METHOD_FP_SHEP:
        return VARIANT_D, "fp_shep_sac_dmp"
    raise ValueError(f"unknown method: {method}")


def _candidate_order_hash(candidates: Sequence[Any]) -> str:
    return stable_hash(
        [
            {
                "index": int(item.original_index),
                "point": list(item.world_point),
                "score": float(item.proposal_score),
                "metadata_json": item.metadata_json,
            }
            for item in candidates
        ]
    )


def _standardize(
    raw: Mapping[str, Any],
    *,
    method: str,
    candidate_set_hash: str | None,
    candidate_hash_before: str | None,
    candidate_hash_after: str | None,
    preview_trace: Sequence[Mapping[str, Any]],
    execution_trace: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    episode = {
        "schema_version": SCHEMA_VERSION,
        "method": method,
        "method_display_name": METHOD_DISPLAY_NAMES[method],
        "scenario": raw["scenario"],
        "seed": int(raw["seed"]),
        "max_steps": 220,
        "team_success": bool(raw["success"]),
        "collision": bool(raw["collision"]),
        "obstacle_collision": bool(raw["obstacle_collision"]),
        "inter_agent_collision": bool(raw["inter_agent_collision"]),
        "timeout": bool(raw["truncated"]),
        "termination_reason": raw["termination_reason"],
        "steps": int(raw["steps"]),
        "completion_time_s": float(raw["steps"]) * 0.1,
        "path_length_m": float(raw["path_length_team_mean_m"]),
        "path_length_team_sum_m": float(raw["path_length_team_sum_m"]),
        "trajectory_smoothness": float(raw["trajectory_smoothness"]),
        "terminal_progress_m": float(raw["terminal_progress_team_mean_m"]),
        "minimum_obstacle_clearance_m": float(raw["minimum_obstacle_clearance_m"]),
        "minimum_obstacle_clearance_source": raw.get(
            "minimum_obstacle_clearance_source", "composite_lidar_nearest_hit"
        ),
        "minimum_static_obstacle_clearance_m": float(
            raw.get("minimum_static_obstacle_clearance_m", float("inf"))
        ),
        "minimum_static_obstacle_clearance_source": raw.get(
            "minimum_static_obstacle_clearance_source",
            "ground_truth_static_geometry_evaluation_only",
        ),
        "obstacle_interaction_agent_count": int(
            raw.get("obstacle_interaction_agent_count", 0)
        ),
        "obstacle_interaction_episode": bool(
            raw.get("obstacle_interaction_episode", False)
        ),
        "minimum_inter_agent_distance_m": float(raw["minimum_inter_agent_distance_m"]),
        "inter_agent_interaction_episode": bool(
            raw.get("inter_agent_interaction_episode", False)
        ),
        "combined_conflict_episode": bool(raw.get("combined_conflict_episode", False)),
        "maximum_terminal_line_deviation_team_mean_m": float(
            raw.get("maximum_terminal_line_deviation_team_mean_m", 0.0)
        ),
        "maximum_terminal_line_deviation_team_max_m": float(
            raw.get("maximum_terminal_line_deviation_team_max_m", 0.0)
        ),
        "obstacle_pressure_path_deviation_team_mean_m": float(
            raw.get("obstacle_pressure_path_deviation_team_mean_m", 0.0)
        ),
        "obstacle_pressure_path_deviation_team_max_m": float(
            raw.get("obstacle_pressure_path_deviation_team_max_m", 0.0)
        ),
        "temporary_reference_available_count": int(raw["temporary_reference_count"]),
        "temporary_reference_reached_count": int(raw["temporary_reference_reached_count"]),
        "reference_handoff_count": int(raw["temporary_reference_reached_count"]),
        "candidate_bundle_generation_count_per_state": (
            0 if method == METHOD_TERMINAL else 1
        ),
        "reached_then_terminal_completion_count": int(
            raw["reached_then_terminal_completion_count"]
        ),
        "candidate_set_hash": candidate_set_hash,
        "candidate_hash_before_selector": candidate_hash_before,
        "candidate_hash_after_selector": candidate_hash_after,
        "candidate_bundle_unchanged": candidate_hash_before == candidate_hash_after,
        "initial_condition_hash": raw["initial_condition_hash"],
        "terminal_task_goals_unchanged": bool(raw["terminal_task_goals_unchanged"]),
        "phase_reset_on_switch": bool(raw["phase_reset_on_switch"]),
        "maximum_phase_switch_delta": float(raw["maximum_phase_switch_delta"]),
        "preview_transition_call_count": len(preview_trace),
        "execution_transition_call_count": len(execution_trace),
        "preview_forcing_gate_semantics": (
            HISTORICAL_GATE_NAME if preview_trace else None
        ),
        "execution_forcing_gate_semantics": HISTORICAL_GATE_NAME,
        "execution_historical_gate_verified": bool(
            execution_trace
            and all(
                row["forcing_gate_semantics"] == HISTORICAL_GATE_NAME
                for row in execution_trace
            )
        ),
        "preview_execution_gate_identity": bool(
            method != METHOD_FP_SHEP
            or (
                preview_trace
                and execution_trace
                and all(
                    row["forcing_gate_semantics"] == HISTORICAL_GATE_NAME
                    for row in (*preview_trace, *execution_trace)
                )
            )
        ),
        "H_preview": 4 if method == METHOD_FP_SHEP else None,
        "one_shot_candidate_generation": method != METHOD_TERMINAL,
        "repeated_replanning": False,
        "historical_gate": True,
        "training_performed": False,
        "GAT_used": False,
        "supervision_label_used_online": False,
        "hard_boundary_used": False,
        "actor_observation_dimension": int(raw["actor_observation_dimension"]),
        "native_environment_observation_dimension": int(
            raw["native_environment_observation_dimension"]
        ),
        "scenario_role": raw.get("scenario_role"),
        "scenario_geometry_version": raw.get("scenario_geometry_version"),
        "scenario_geometry_hash": raw.get("scenario_geometry_hash"),
        "seed_set_role": raw.get("seed_set_role"),
        "observation_mode": raw.get("observation_mode"),
        "peer_spheres_enabled": bool(raw.get("peer_spheres_enabled", False)),
        "include_boundaries_in_sensor": bool(
            raw.get("include_boundaries_in_sensor", False)
        ),
        "terminate_on_boundary_collision": bool(
            raw.get("terminate_on_boundary_collision", False)
        ),
    }
    agent_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for raw_agent in raw["agent_stage_records"]:
        agent = dict(raw_agent)
        agent_id = int(agent["agent_id"])
        candidates = (
            {} if method == METHOD_TERMINAL else raw["candidate_records"][agent_id]
        )
        selection = {
            "schema_version": SCHEMA_VERSION,
            "method": method,
            "scenario": raw["scenario"],
            "seed": int(raw["seed"]),
            "agent_id": agent_id,
            "candidate_set_hash": candidate_set_hash,
            "candidate_order_hash": (
                None
                if method == METHOD_TERMINAL
                else stable_hash(
                    {
                        "points": candidates["candidate_world_points"],
                        "scores": candidates["proposal_scores"],
                        "metadata": candidates["candidate_metadata"],
                    }
                )
            ),
            "K_t": int(candidates.get("K_t", 0)) if candidates else 0,
            "candidate_world_points": candidates.get("candidate_world_points", []),
            "proposal_scores": candidates.get("proposal_scores", []),
            "candidate_metadata": candidates.get("candidate_metadata", []),
            "selected_candidate_index": candidates.get("selected_candidate_id"),
            "selected_original_proposal_rank": candidates.get("selected_proposal_rank"),
            "selected_proposal_score": candidates.get("selected_proposal_score"),
            "selected_fp_shep_score": candidates.get("selected_fp_shep_score"),
            "selected_world_reference": candidates.get("temporary_reference"),
            "goal_jump_m": (
                float(
                    np.linalg.norm(
                        np.asarray(candidates["temporary_reference"], dtype=float)
                        - np.asarray(raw["terminal_task_goals"][agent_id], dtype=float)
                    )
                )
                if candidates
                else 0.0
            ),
            "no_candidate_fallback": bool(
                candidates.get("no_candidate_fallback", False)
            )
            if candidates
            else False,
            "selection_kind": candidates.get("selection_source") if candidates else None,
            "fp_shep_candidate_records": candidates.get(
                "fp_shep_candidate_records", []
            ),
            "terminal_speed_used_for_online_ranking": False,
            "preview_forcing_gate_semantics": (
                HISTORICAL_GATE_NAME if method == METHOD_FP_SHEP else None
            ),
            "execution_forcing_gate_semantics": HISTORICAL_GATE_NAME,
            "selector_score_specification": (
                "+progress +clearance -deviation; terminal_speed_weight=0"
                if method == METHOD_FP_SHEP
                else None
            ),
        }
        selected_preview = None
        if method == METHOD_FP_SHEP and selection["selected_candidate_index"] is not None:
            selected_preview = selection["fp_shep_candidate_records"][
                int(selection["selected_candidate_index"])
            ]
        agent.update(
            {
                "schema_version": SCHEMA_VERSION,
                "method": method,
                "method_display_name": METHOD_DISPLAY_NAMES[method],
                "scenario": raw["scenario"],
                "seed": int(raw["seed"]),
                "candidate_set_hash": candidate_set_hash,
                "candidate_order_hash": selection["candidate_order_hash"],
                "selected_candidate_index": selection["selected_candidate_index"],
                "selected_world_reference": selection["selected_world_reference"],
                "selected_preview_task_progress": (
                    selected_preview.get("preview_task_progress")
                    if selected_preview else None
                ),
                "selected_preview_min_clearance": (
                    selected_preview.get("preview_min_clearance")
                    if selected_preview else None
                ),
                "selected_preview_max_execution_deviation": (
                    selected_preview.get("preview_max_execution_deviation")
                    if selected_preview else None
                ),
                "selected_preview_terminal_speed": (
                    selected_preview.get("preview_terminal_speed")
                    if selected_preview else None
                ),
                "preview_real_comparison_scope": (
                    "directional_sanity_only_non_equal_horizon"
                    if selected_preview else None
                ),
                "prediction_error_reported": False,
            }
        )
        agent_rows.append(agent)
        if method != METHOD_TERMINAL:
            selection_rows.append(selection)
    return episode, agent_rows, selection_rows


def _stage_a_integrity(
    episodes: Sequence[Mapping[str, Any]],
    selector_pairs: Sequence[Mapping[str, Any]],
    *,
    policy_unchanged: bool,
    checkpoint_unchanged: bool,
) -> dict[str, Any]:
    stage_a = [row for row in episodes if int(row["seed"]) < 5]
    initialization_groups: dict[tuple[str, int], set[str]] = {}
    for row in stage_a:
        initialization_groups.setdefault(
            (str(row["scenario"]), int(row["seed"])), set()
        ).add(str(row["initial_condition_hash"]))
    checks = {
        "paired_initial_state": all(
            len(hashes) == 1 for hashes in initialization_groups.values()
        ),
        "candidate_hash_match": all(bool(row["candidate_hash_match"]) for row in selector_pairs),
        "candidate_order_match": all(bool(row["candidate_order_match"]) for row in selector_pairs),
        "candidate_bundle_unchanged": all(
            bool(row["candidate_bundle_unchanged"])
            for row in stage_a
            if row["method"] != METHOD_TERMINAL
        ),
        "preview_execution_gate_identity": all(
            bool(row["preview_execution_gate_identity"])
            for row in stage_a
            if row["method"] == METHOD_FP_SHEP
        ),
        "H_preview_4": all(
            int(row["H_preview"]) == 4
            for row in stage_a
            if row["method"] == METHOD_FP_SHEP
        ),
        "one_shot": all(
            bool(row["one_shot_candidate_generation"])
            and int(row["candidate_bundle_generation_count_per_state"]) == 1
            and int(row["reference_handoff_count"]) <= 3
            for row in stage_a
            if row["method"] != METHOD_TERMINAL
        ),
        "max_steps_220": all(int(row["max_steps"]) == 220 for row in stage_a),
        "historical_gate": all(
            bool(row["historical_gate"])
            and bool(row["execution_historical_gate_verified"])
            for row in stage_a
        ),
        "no_training": all(not bool(row["training_performed"]) for row in stage_a),
        "no_repeated_replanning": all(
            not bool(row["repeated_replanning"]) for row in stage_a
        ),
        "phase_preserved": all(not bool(row["phase_reset_on_switch"]) for row in stage_a),
        "policy_unchanged": bool(policy_unchanged),
        "checkpoint_unchanged": bool(checkpoint_unchanged),
    }
    return {
        "status": "PASSED" if all(checks.values()) else "FAILED",
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
    }


def _render_report(
    method_rows: Sequence[Mapping[str, Any]],
    selector_summary: Mapping[str, Any],
    failure_rows: Sequence[Mapping[str, Any]],
    conclusion: Mapping[str, Any],
) -> str:
    overall = {row["method"]: row for row in method_rows if row["scenario"] == "overall"}
    pct = lambda value: "N/A" if value is None else f"{100*float(value):.2f}%"
    lines = [
        "# 220-Step Pre-GAT Closed-Loop Revalidation",
        "",
        "All methods use the frozen checkpoint, historical vector goal-eff gate, "
        "220 steps, boundary-free execution, and deterministic inference.",
        "",
        "| Method | Team success | Reference reached | Collision before reference | Reached-to-terminal | Path length (m) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHOD_ORDER:
        row = overall[method]
        lines.append(
            f"| {METHOD_DISPLAY_NAMES[method]} | {pct(row['team_success_rate'])} | "
            f"{pct(row['reference_reached_rate'])} | {pct(row['reference_collision_rate'])} | "
            f"{pct(row['reached_then_terminal_rate'])} | {float(row['mean_path_length_m']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Scenario breakdown",
            "",
            "| Method | Scenario | Team success | Reference reached | Collision before reference | Inter-agent collision | Timeout |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    lookup = {(row["method"], row["scenario"]): row for row in method_rows}
    for method in METHOD_ORDER:
        for scenario in ("open", "sparse_static", "multi_agent"):
            row = lookup[(method, scenario)]
            lines.append(
                f"| {METHOD_DISPLAY_NAMES[method]} | {scenario} | "
                f"{pct(row['team_success_rate'])} | {pct(row['reference_reached_rate'])} | "
                f"{pct(row['reference_collision_rate'])} | "
                f"{pct(row['inter_agent_collision_rate'])} | {pct(row['timeout_rate'])} |"
            )
    selector_failures = [
        row for row in failure_rows if row["method"] in {METHOD_PROPOSAL, METHOD_FP_SHEP}
    ]
    failure_counts = Counter(str(row["primary_failure_category"]) for row in selector_failures)
    lines.extend(
        [
            "",
            "## Selector pairing",
            "",
            f"- Eligible pairs: {selector_summary['eligible_selector_pair_count']}",
            f"- No-candidate pairs: {selector_summary['no_candidate_pair_count']}",
            f"- Disagreement rate: {pct(selector_summary['selector_disagreement_rate'])}",
            f"- Reference outcomes: {selector_summary['reference_pair_class_counts']}",
            f"- Disagreement-only reference outcomes: {selector_summary['disagreement_reference_pair_class_counts']}",
            f"- Team outcomes: {selector_summary['team_pair_class_counts']}",
            f"- Disagreement-episode team outcomes: {selector_summary['disagreement_episode_team_class_counts']}",
            "",
            "## Failure attribution",
            "",
            f"- Primary categories: {dict(failure_counts)}",
            f"- Failures with Stage-1 flags: {conclusion['failure_before_reference_count']}",
            f"- Failures with Stage-2 flags: {conclusion['failure_after_reference_count']}",
            f"- Interaction-related failures: {conclusion['interaction_related_failure_count']}",
            "",
            "FP-SHEP predicted progress is an H=4 short-horizon feature. It is not "
            "reported as a numerical prediction error against the longer Stage-1 execution.",
            "",
            "## Conclusion",
            "",
            f"- FP_SHEP_CLOSED_LOOP_EXECUTABILITY_SIGNAL = {conclusion['FP_SHEP_CLOSED_LOOP_EXECUTABILITY_SIGNAL']}",
            f"- STAGE1_IS_PRIMARY_BOTTLENECK = {conclusion['STAGE1_IS_PRIMARY_BOTTLENECK']}",
            f"- MULTI_AGENT_COORDINATION_GAP = {conclusion['MULTI_AGENT_COORDINATION_GAP']}",
            f"- PROCEED_TO_GAT_STAGE_I = {conclusion['PROCEED_TO_GAT_STAGE_I']}",
            f"- PRIMARY_LIMITATION = {conclusion['PRIMARY_LIMITATION']}",
            f"- NEXT_STEP = {conclusion['NEXT_STEP']}",
            "",
            "## Required questions",
            "",
            f"1. Proposal Top-1: team success {overall[METHOD_PROPOSAL]['team_success_count']}/"
            f"{overall[METHOD_PROPOSAL]['episode_count']} ({pct(overall[METHOD_PROPOSAL]['team_success_rate'])}); "
            f"reference reached {overall[METHOD_PROPOSAL]['reference_reached_count']}/"
            f"{overall[METHOD_PROPOSAL]['reference_available_count']} ({pct(overall[METHOD_PROPOSAL]['reference_reached_rate'])}); "
            f"collision before reference {overall[METHOD_PROPOSAL]['reference_collision_count']}/"
            f"{overall[METHOD_PROPOSAL]['reference_available_count']} ({pct(overall[METHOD_PROPOSAL]['reference_collision_rate'])}).",
            f"2. FP-SHEP changes reference reach by {100*float(conclusion['delta_reference_reached_rate']):+.2f} pp.",
            f"3. FP-SHEP changes collision-before-reference by "
            f"{-100*float(conclusion['collision_before_reference_reduction']):+.2f} pp "
            "(negative means fewer collisions).",
            f"4. FP-SHEP changes team success by {100*float(conclusion['delta_team_success_rate']):+.2f} pp.",
            f"5. Selectors disagree on {selector_summary['selector_disagreement_count']}/"
            f"{selector_summary['eligible_selector_pair_count']} eligible agent pairs "
            f"({pct(selector_summary['selector_disagreement_rate'])}).",
            f"6. On disagreement pairs, reference outcomes are "
            f"{selector_summary['disagreement_reference_pair_class_counts']}.",
            "7. Scenario-specific performance is reported in the Scenario breakdown table; "
            "the aggregate rate is not used to hide scenario composition.",
            f"8. Selector failures with Stage-1 flags: {conclusion['failure_before_reference_count']}; "
            f"with Stage-2 flags: {conclusion['failure_after_reference_count']}.",
            f"9. Multi-agent coordination gap: {conclusion['MULTI_AGENT_COORDINATION_GAP']}; "
            f"interaction-related selector failures: {conclusion['interaction_related_failure_count']}.",
            f"10. Proceed to GAT Stage-I: {conclusion['PROCEED_TO_GAT_STAGE_I']}. "
            "This run provides evaluation evidence only and does not start training.",
            "",
            "No GAT, optimizer, fine-tuning, repeated replanning, hard boundary, "
            "oracle, or supervision label was used.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_experiment(settings: Mapping[str, Any], output_dir: Path) -> Path:
    _assert_config(settings)
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = (REPO_ROOT / str(settings["checkpoint"])).resolve()
    checkpoint_hash_before = sha256_file(checkpoint)
    if checkpoint_hash_before != settings["checkpoint_sha256_expected"]:
        raise RuntimeError("checkpoint hash mismatch")

    base_settings = _load_json((REPO_ROOT / settings["base_config"]).resolve())
    base_settings.update(
        {
            "checkpoint": settings["checkpoint"],
            "checkpoint_sha256_expected": settings["checkpoint_sha256_expected"],
            "deterministic_policy": True,
            "num_agents": int(settings["num_agents"]),
            "max_steps": 220,
            "scenarios": list(settings["scenarios"]),
            "seeds": list(settings["stage_a_seeds"]) + list(settings["stage_b_seeds"]),
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
        num_agents=int(settings["num_agents"]), max_steps=220
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
    bundles: dict[tuple[str, int], Any] = {}
    started = time.perf_counter()

    def prepare_bundle(scenario: str, seed: int) -> Any:
        env, _ = build_closed_loop_environment(
            config=multi_config,
            scenario=scenario,
            seed=int(seed),
            peer_radius=float(settings["peer_radius"]),
        )
        try:
            bundle = generate_immutable_candidate_bundle(
                env,
                scenario=scenario,
                seed=int(seed),
                proposal_config=ProposalConfig(**dict(settings["proposal_config"])),
                consumer_top_k=int(settings["temporary_reference"]["K_requested"]),
            )
        finally:
            env.close()
        bundles[(scenario, int(seed))] = bundle
        bundle_rows.extend(candidate_bundle_rows(bundle))
        return bundle

    def run_seed_block(seeds: Sequence[int], stage_name: str) -> None:
        total = len(seeds) * len(settings["scenarios"]) * len(METHOD_ORDER)
        index = 0
        for scenario in settings["scenarios"]:
            for seed in seeds:
                bundle = prepare_bundle(str(scenario), int(seed))
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
                            scenario=str(scenario),
                            seed=int(seed),
                            variant=variant,
                            candidate_sets=(
                                bundle.per_agent if method != METHOD_TERMINAL else None
                            ),
                            selection_method=(
                                selector_method or "proposal_sac_dmp"
                            ),
                            score_spec=score_spec,
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
                    episode["stage"] = stage_name
                    for row in agents:
                        row["stage"] = stage_name
                    for row in selections:
                        row["stage"] = stage_name
                    episode_rows.append(episode)
                    agent_rows.extend(agents)
                    selection_rows.extend(selections)
                    index += 1
                    print(
                        f"[{stage_name} {index}/{total}] {method} {scenario} seed={seed}: "
                        f"{episode['termination_reason']}",
                        flush=True,
                    )

    run_seed_block(settings["stage_a_seeds"], "stage_a")
    stage_a_paired, stage_a_selector_summary = build_selector_pairing(
        episode_rows, agent_rows
    )
    policy_unchanged_a = policy_hash_before == _policy_parameter_sha256(policy)
    checkpoint_unchanged_a = checkpoint_hash_before == sha256_file(checkpoint)
    stage_a_integrity = _stage_a_integrity(
        episode_rows,
        stage_a_paired,
        policy_unchanged=policy_unchanged_a,
        checkpoint_unchanged=checkpoint_unchanged_a,
    )
    write_json(output_dir / "stage_a_integrity.json", stage_a_integrity)
    if stage_a_integrity["status"] != "PASSED":
        write_csv(output_dir / "stage_a_per_episode.csv", episode_rows)
        raise RuntimeError(f"Stage A gate failed: {stage_a_integrity['failed_checks']}")

    run_seed_block(settings["stage_b_seeds"], "stage_b")
    paired_rows, selector_summary = build_selector_pairing(episode_rows, agent_rows)
    if selector_summary["integrity_status"] != "PASSED":
        raise RuntimeError(f"selector pairing failed: {selector_summary['integrity_errors']}")
    method_rows = aggregate_method_rows(episode_rows, agent_rows)
    failure_rows = build_failure_rows(episode_rows, agent_rows)
    conclusion = build_conclusion(method_rows, selector_summary, failure_rows)
    sparse_rows = [row for row in method_rows if row["scenario"] == "sparse_static"]

    checkpoint_hash_after = sha256_file(checkpoint)
    policy_hash_after = _policy_parameter_sha256(policy)
    critical_after = _critical_hashes(checkpoint)
    initial_groups: dict[tuple[str, int], set[str]] = {}
    for row in episode_rows:
        initial_groups.setdefault(
            (str(row["scenario"]), int(row["seed"])), set()
        ).add(str(row["initial_condition_hash"]))
    integrity = {
        "status": "PASSED",
        "episode_count": len(episode_rows),
        "expected_episode_count": 90,
        "checkpoint_sha256_before": checkpoint_hash_before,
        "checkpoint_sha256_after": checkpoint_hash_after,
        "checkpoint_unchanged": checkpoint_hash_before == checkpoint_hash_after,
        "policy_parameter_sha256_before": policy_hash_before,
        "policy_parameter_sha256_after": policy_hash_after,
        "policy_parameters_unchanged": policy_hash_before == policy_hash_after,
        "critical_source_hashes_unchanged": critical_before == critical_after,
        "paired_initial_states_all_methods": all(
            len(hashes) == 1 for hashes in initial_groups.values()
        ),
        "candidate_bundle_generation_count": len(bundles),
        "expected_candidate_bundle_generation_count": 30,
        "all_execution_historical_gate_verified": all(
            bool(row["execution_historical_gate_verified"]) for row in episode_rows
        ),
        "all_fp_shep_preview_historical_gate_verified": all(
            bool(row["preview_execution_gate_identity"])
            for row in episode_rows
            if row["method"] == METHOD_FP_SHEP
        ),
        "default_execution_symbol_restored": (
            environment_module.propagate_sac_dmp_action is default_execution_symbol
        ),
        "default_preview_symbol_restored": (
            preview_module.propagate_sac_dmp_action is default_preview_symbol
        ),
        "stage_a_gate": stage_a_integrity,
        "selector_pairing": selector_summary,
        "training_performed": False,
        "gradient_update_count": 0,
        "GAT_optimizer_created": False,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    if not all(
        integrity[key]
        for key in (
            "checkpoint_unchanged",
            "policy_parameters_unchanged",
            "critical_source_hashes_unchanged",
            "paired_initial_states_all_methods",
            "all_execution_historical_gate_verified",
            "all_fp_shep_preview_historical_gate_verified",
            "default_execution_symbol_restored",
            "default_preview_symbol_restored",
        )
    ) or len(episode_rows) != 90 or len(bundles) != 30:
        integrity["status"] = "FAILED"
        raise RuntimeError("final integrity audit failed")

    resolved = copy.deepcopy(dict(settings))
    resolved["resolved_output_dir"] = str(output_dir)
    resolved["selector_score_specification"] = score_spec.metadata()
    write_json(output_dir / "config.json", resolved)
    write_csv(output_dir / "per_episode.csv", episode_rows)
    write_csv(output_dir / "per_agent.csv", agent_rows)
    write_csv(output_dir / "candidate_selection.csv", selection_rows)
    write_csv(output_dir / "candidate_bundle.csv", bundle_rows)
    write_csv(output_dir / "stage_summary.csv", method_rows)
    write_csv(output_dir / "method_summary.csv", method_rows)
    write_csv(output_dir / "paired_selector_analysis.csv", paired_rows)
    write_csv(output_dir / "failure_attribution.csv", failure_rows)
    write_csv(output_dir / "sparse_static_analysis.csv", sparse_rows)
    write_json(output_dir / "selector_summary.json", selector_summary)
    write_json(output_dir / "integrity.json", integrity)
    write_json(output_dir / "conclusion.json", conclusion)
    (output_dir / "FINAL_REPORT.md").write_text(
        _render_report(method_rows, selector_summary, failure_rows, conclusion),
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
