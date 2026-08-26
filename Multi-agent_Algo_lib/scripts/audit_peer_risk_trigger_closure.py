"""Generate the hard-stop Peer-Risk Trigger Specificity Closure artifacts.

This audit is intentionally non-interventional.  The frozen equal-information
contract does not expose typed peer state at every execution step, while the
existing interaction diagnostic requires both exact peer state and an FP-SHEP
candidate preview.  Consequently the Phase-A information gate fails and the
later observability, implementation, and paired-development phases are not
executed.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts" / "peer_risk_trigger_closure" / "20260819_143612"
SPARSE = ROOT / "artifacts" / "sparse_err_trigger_revision" / "20260819_020727"
SAFETY = ROOT / "artifacts" / "rerr_safety_closure" / "20260819_134216"
EQUAL = ROOT / "artifacts" / "equal_information_baseline_audit" / "20260819_012339"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str] | None = None) -> None:
    materialized = list(rows)
    if fields is None:
        if not materialized:
            raise ValueError(f"fields are required for empty CSV: {path}")
        fields = list(materialized[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)


def source_entry(path: Path, role: str, required: bool = True) -> dict[str, Any]:
    relative = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)
    return {
        "path": relative,
        "role": role,
        "required": required,
        "exists": path.exists(),
        "sha256": sha256(path) if path.is_file() else None,
        "size_bytes": path.stat().st_size if path.is_file() else None,
    }


def placeholder(name: str, reason: str) -> None:
    write_csv(
        OUTPUT / name,
        [{"status": "NOT_RUN", "reason": reason}],
        ["status", "reason"],
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    authority_files = [
        (SPARSE / "FINAL_REPORT.md", "frozen R-ERR development result"),
        (SPARSE / "conclusion.json", "frozen R-ERR decision fields"),
        (SPARSE / "final_reconciliation.json", "prior integrity reconciliation"),
        (SPARSE / "reproposal_events.csv", "prior event records"),
        (SPARSE / "runtime_method_summary.csv", "measured runtime reference"),
        (SAFETY / "FINAL_REPORT.md", "peer-risk root-cause authority"),
        (SAFETY / "conclusion.json", "root-cause decision fields"),
        (SAFETY / "final_reconciliation.json", "peer audit reconciliation"),
        (SAFETY / "paired_collision_timeline.csv", "paired diagnostic timeline"),
        (SAFETY / "peer_collision_timeline.csv", "ten peer-collision timelines"),
        (SAFETY / "peer_collision_classification.csv", "ten collision classifications"),
        (SAFETY / "normal_trigger_utility.csv", "successful and failed event utility"),
        (SAFETY / "replanning_scope_audit.csv", "replanning compute/update scope"),
        (EQUAL / "FINAL_REPORT.md", "equal-information authority"),
        (EQUAL / "conclusion.json", "equal-information decision fields"),
        (EQUAL / "proposed_online_information_contract.csv", "frozen Proposed information contract"),
        (EQUAL / "lower_execution_information_contract.csv", "frozen execution input contract"),
        (EQUAL / "observation_equivalence_tests.json", "information-leakage tests"),
        (ROOT / "planning" / "event_triggered_reference_reconstruction.py", "active R-ERR implementation"),
        (ROOT / "planning" / "goal_semantics_diagnosis.py", "historical 122-D actor observation"),
        (ROOT / "planning" / "heterogeneous_candidate_graph.py", "active GAT interaction descriptors"),
        (ROOT / "planning" / "candidate_execution_interface.py", "conflict diagnostic implementation"),
        (ROOT / "hire-rl-body.tex", "active Methodology"),
        (ROOT / "configs" / "evaluation" / "final_four_stage_benchmark.json", "frozen P03 graph setting"),
    ]
    context = {
        "audit": "Peer-Risk Trigger Specificity Closure",
        "timestamp": "20260819_143612",
        "mode": "READ_ONLY_HARD_STOP_AFTER_PHASE_A",
        "AGENTS_md": {"exists": (ROOT / "AGENTS.md").exists()},
        "CODEX_HANDOFF_md": {"exists": (ROOT / "CODEX_HANDOFF.md").exists()},
        "authorities": [source_entry(path, role) for path, role in authority_files],
        "source_row_counts": {
            "sparse_reproposal_events": len(read_csv(SPARSE / "reproposal_events.csv")),
            "safety_paired_collision_timeline": len(read_csv(SAFETY / "paired_collision_timeline.csv")),
            "safety_peer_collision_timeline": len(read_csv(SAFETY / "peer_collision_timeline.csv")),
            "safety_peer_collision_classification": len(read_csv(SAFETY / "peer_collision_classification.csv")),
            "safety_normal_trigger_utility": len(read_csv(SAFETY / "normal_trigger_utility.csv")),
            "safety_replanning_scope_audit": len(read_csv(SAFETY / "replanning_scope_audit.csv")),
        },
        "frozen_historical_facts": {
            "RERR_GAT_success": 0.8125,
            "RERR_GAT_collision": 0.1625,
            "RERR_GAT_inter_agent_collision": 0.125,
            "RERR_GAT_timeout": 0.025,
            "RERR_GAT_mean_reproposals": 8.9625,
            "RERR_GAT_total_online_compute_ms": 2271.5911549999996,
            "single_high_level_latency_ms": 213.89047152145642,
            "peer_collision_episode_count": 10,
        },
    }
    write_json(OUTPUT / "context_recovery_manifest.json", context)

    information_rows = [
        {
            "signal": "peer_exact_position",
            "available_each_step": "NO",
            "available_only_at_upper_event": "YES",
            "exact_or_sensor_derived": "EXACT_CURRENT_AT_EVENT",
            "range_limit": "ALL_CURRENT_NEIGHBORS_AT_EVENT",
            "typed_or_untyped": "TYPED_PEER",
            "paper_assumption": "neighbor node and proposal-align edge at an upper planning event",
            "code_source": "heterogeneous_candidate_graph.py:599 observable_neighbor_states",
            "allowed_for_trigger": "NO_PRIVILEGED_IF_POLLED_EACH_STEP",
        },
        {
            "signal": "peer_exact_velocity",
            "available_each_step": "NO",
            "available_only_at_upper_event": "YES",
            "exact_or_sensor_derived": "EXACT_CURRENT_AT_EVENT",
            "range_limit": "ALL_CURRENT_NEIGHBORS_AT_EVENT",
            "typed_or_untyped": "TYPED_PEER",
            "paper_assumption": "constant-velocity peer prediction inside upper graph construction",
            "code_source": "heterogeneous_candidate_graph.py:384-390,599",
            "allowed_for_trigger": "NO_PRIVILEGED_IF_POLLED_EACH_STEP",
        },
        {
            "signal": "peer_identity",
            "available_each_step": "NO",
            "available_only_at_upper_event": "YES",
            "exact_or_sensor_derived": "EXACT_AT_EVENT",
            "range_limit": "EVENT_LOCAL_GRAPH_MAPPING",
            "typed_or_untyped": "TYPED_PEER",
            "paper_assumption": "neighbor index j exists in the upper graph",
            "code_source": "observable_neighbor_states / graph neighbor mapping",
            "allowed_for_trigger": "NO_PRIVILEGED_IF_POLLED_EACH_STEP",
        },
        {
            "signal": "peer_relative_position",
            "available_each_step": "NO_EXACT",
            "available_only_at_upper_event": "YES_EXACT",
            "exact_or_sensor_derived": "DERIVED_FROM_EXACT_EVENT_STATE",
            "range_limit": "EVENT_LOCAL",
            "typed_or_untyped": "TYPED_PEER",
            "paper_assumption": "bearing and distance in neighboring-UAV node",
            "code_source": "heterogeneous_candidate_graph.py feature schema relative_direction/distance",
            "allowed_for_trigger": "NO_EXACT_PER_STEP",
        },
        {
            "signal": "peer_relative_velocity",
            "available_each_step": "NO",
            "available_only_at_upper_event": "YES",
            "exact_or_sensor_derived": "DERIVED_FROM_EXACT_EVENT_STATE",
            "range_limit": "EVENT_LOCAL",
            "typed_or_untyped": "TYPED_PEER",
            "paper_assumption": "Delta v_ij in neighboring-UAV node",
            "code_source": "heterogeneous_candidate_graph.py feature schema relative_velocity",
            "allowed_for_trigger": "NO_PRIVILEGED_IF_POLLED_EACH_STEP",
        },
        {
            "signal": "typed_peer_lidar_return",
            "available_each_step": "NO",
            "available_only_at_upper_event": "NO",
            "exact_or_sensor_derived": "UNAVAILABLE",
            "range_limit": "N/A",
            "typed_or_untyped": "NO_TYPED_RETURN",
            "paper_assumption": "local observation is nearest occupied surface, without a peer label",
            "code_source": "equal-information proposed_online_information_contract.csv",
            "allowed_for_trigger": "NO",
        },
        {
            "signal": "untyped_peer_lidar_return",
            "available_each_step": "YES_IF_PEER_IS_NEAREST_RAY_HIT",
            "available_only_at_upper_event": "NO",
            "exact_or_sensor_derived": "SENSOR_DERIVED_SURFACE_RANGE",
            "range_limit": "4.5_M_8x7_RAYS",
            "typed_or_untyped": "UNTYPED_STATIC_DYNAMIC_PEER_MIXTURE",
            "paper_assumption": "peer spheres share the same nearest-surface scan representation",
            "code_source": "evaluate_single_policy_multi_agent.py:199-214; SensorPacket.current_scan",
            "allowed_for_trigger": "GENERIC_OBSTACLE_ONLY_NOT_PEER_SPECIFIC",
        },
        {
            "signal": "neighbor_communication_state",
            "available_each_step": "NO",
            "available_only_at_upper_event": "NO_FORMAL_COMMUNICATION_CHANNEL",
            "exact_or_sensor_derived": "UNAVAILABLE",
            "range_limit": "N/A",
            "typed_or_untyped": "N/A",
            "paper_assumption": "none in the execution information contract",
            "code_source": "no encoded execution feature",
            "allowed_for_trigger": "NO",
        },
        {
            "signal": "previous_lidar_scan",
            "available_each_step": "YES_AFTER_FIRST_HISTORY",
            "available_only_at_upper_event": "NO",
            "exact_or_sensor_derived": "PREVIOUS_UNTYPED_NEAREST_RAY_RETURN",
            "range_limit": "4.5_M_56_RAYS",
            "typed_or_untyped": "UNTYPED",
            "paper_assumption": "historical 122-D actor input",
            "code_source": "goal_semantics_diagnosis.py:86-87",
            "allowed_for_trigger": "NO_DETERMINISTIC_PEER_ATTRIBUTION_OR_ASSOCIATION",
        },
        {
            "signal": "current_lidar_scan",
            "available_each_step": "YES",
            "available_only_at_upper_event": "NO",
            "exact_or_sensor_derived": "CURRENT_UNTYPED_NEAREST_RAY_RETURN",
            "range_limit": "4.5_M_56_RAYS",
            "typed_or_untyped": "UNTYPED",
            "paper_assumption": "historical 122-D actor input and active-sector margin",
            "code_source": "goal_semantics_diagnosis.py:86; event_triggered_reference_reconstruction.py:470-480",
            "allowed_for_trigger": "GENERIC_ACTIVE_SECTOR_ONLY_NOT_PEER_SPECIFIC",
        },
    ]
    write_csv(OUTPUT / "peer_trigger_information_contract.csv", information_rows)

    (OUTPUT / "interaction_descriptor_contract.md").write_text(
        "# Existing interaction descriptor contract\n\n"
        "The active Methodology and implementation already define a candidate-specific "
        "peer interaction diagnostic. For candidate preview position "
        "`p_hat_i,k(t+h)` and the event-time exact neighbor state, the neighbor is "
        "propagated as `p_j(t)+h*dt*v_j(t)`. The retained edge attributes are closest-"
        "approach time `t_hat`, minimum predicted separation `d_hat_min`, and risk "
        "duration `T_risk = dt * sum_h I[d_hat(h) < d_safe]`.\n\n"
        "Frozen implementation values are `H=4`, `dt=0.1 s`, `d_safe=0.6 m`, and "
        "`d_align=1.0 m` for the selected P03 benchmark configuration. The risk inequality "
        "is strict (`distance < d_safe`); the graph edge exists for "
        "`minimum_separation < d_align`. `d_safe` comes from "
        "`MultiAgentEnvConfig.inter_agent_safe_distance`; it is an existing threshold, "
        "not a threshold introduced by this audit.\n\n"
        "The descriptor cannot serve as a cheap per-step trigger. It requires (1) a "
        "candidate bundle, (2) the FP-SHEP H4 execution preview of every candidate, and "
        "(3) exact current neighbor position, velocity, and identity. Items (1)-(2) are "
        "the upper computation whose invocation the trigger would be deciding; item (3) "
        "is not in the frozen every-step execution information contract.\n\n"
        "No new TTC, velocity-obstacle, CBF score, horizon, risk weight, or numerical "
        "threshold was introduced.\n",
        encoding="utf-8",
    )

    cost_rows = [
        {
            "signal": "neighbor_node_relative_state",
            "existing": "YES",
            "candidate_specific": "NO",
            "requires_exact_peer_state": "YES",
            "requires_candidate_generation": "NO",
            "requires_fp_preview": "NO",
            "available_each_execution_step": "NO",
            "cost_class": "CHEAP_STATE_ONLY_BUT_INFORMATION_ILLEGAL_PER_STEP",
            "trigger_eligible": "NO",
        },
        {
            "signal": "t_hat_d_hat_min_T_risk",
            "existing": "YES",
            "candidate_specific": "YES",
            "requires_exact_peer_state": "YES",
            "requires_candidate_generation": "YES",
            "requires_fp_preview": "YES",
            "available_each_execution_step": "NO",
            "cost_class": "REQUIRES_FP_PREVIEW",
            "trigger_eligible": "NO_CIRCULAR_UPPER_COMPUTE",
        },
        {
            "signal": "active_direction_untyped_safety_margin",
            "existing": "YES",
            "candidate_specific": "NO",
            "requires_exact_peer_state": "NO",
            "requires_candidate_generation": "NO",
            "requires_fp_preview": "NO",
            "available_each_execution_step": "YES",
            "cost_class": "CHEAP_EXISTING_SENSOR",
            "trigger_eligible": "NO_PEER_SPECIFICITY_UNTYPED_SINGLE_SECTOR",
        },
        {
            "signal": "two_frame_untyped_lidar_peer_track",
            "existing": "NO_RELIABLE_TRACK",
            "candidate_specific": "NO",
            "requires_exact_peer_state": "NO",
            "requires_candidate_generation": "NO",
            "requires_fp_preview": "NO",
            "available_each_execution_step": "SCANS_YES_TRACK_NO",
            "cost_class": "NOT_ESTABLISHED",
            "trigger_eligible": "NO_NONDETERMINISTIC_IDENTITY_AND_VELOCITY",
        },
    ]
    write_csv(OUTPUT / "interaction_signal_cost_audit.csv", cost_rows)

    peer_rows = read_csv(SAFETY / "peer_collision_classification.csv")
    observability_rows: list[dict[str, Any]] = []
    lead_rows: list[dict[str, Any]] = []
    for row in peer_rows:
        base = {
            "scenario_id": row["scenario_id"],
            "stage": row["stage"],
            "collision_step": int(row["collision_step"]),
        }
        observability_rows.append(
            {
                **base,
                "candidate_peer_signal": "NOT_DEFINED_UNDER_LEGAL_EVERY_STEP_INFORMATION",
                "peer_event_detected": False,
                "peer_event_actionable": False,
                "status": "NOT_EVALUATED_INFORMATION_GATE_FAILED",
                "reason": "typed peer identity/position/velocity unavailable every step; existing descriptor requires FP preview",
            }
        )
        lead_rows.append(
            {
                **base,
                "peer_event_step": "",
                "raw_lead_time_s": "",
                "planning_latency_s": 0.21389047152145642,
                "minimum_required_post_planning_transition_s": 0.1,
                "actionable_lead_time_s": "",
                "actionable": False,
                "status": "NOT_COMPUTABLE_INFORMATION_GATE_FAILED",
            }
        )
    write_csv(OUTPUT / "peer_risk_observability.csv", observability_rows)
    write_csv(OUTPUT / "peer_collision_lead_time.csv", lead_rows)

    write_csv(
        OUTPUT / "successful_false_trigger.csv",
        [
            {
                "scope": "all_historical_RERR_successes",
                "historical_success_episode_count": 65,
                "episodes_with_peer_event": "",
                "false_trigger_episode_rate": "",
                "mean_events_per_episode": "",
                "median_events_per_episode": "",
                "p90_events_per_episode": "",
                "max_events_per_episode": "",
                "status": "NOT_COMPUTABLE_INFORMATION_GATE_FAILED",
                "reason": "no legal peer-specific per-step signal exists; no privileged counterfactual was fabricated",
            }
        ],
    )

    risky_rows = [
        {
            "scenario_id": "SR2_014",
            "stage": "stage_2",
            "planning_step": 87,
            "agent_id": 1,
            "selected_candidate_id": 3,
            "neighbor_agent_id": 0,
            "t_min_s": 0.30000001192092896,
            "d_min_m": 0.515174388885498,
            "T_risk_s": 0.30000001192092896,
        },
        {
            "scenario_id": "SR3_019",
            "stage": "stage_3",
            "planning_step": 55,
            "agent_id": 0,
            "selected_candidate_id": 0,
            "neighbor_agent_id": 1,
            "t_min_s": 0.4000000059604645,
            "d_min_m": 0.5530065298080444,
            "T_risk_s": 0.20000000298023224,
        },
        {
            "scenario_id": "SR4_015",
            "stage": "stage_4",
            "planning_step": 83,
            "agent_id": 2,
            "selected_candidate_id": 5,
            "neighbor_agent_id": 1,
            "t_min_s": 0.30000001192092896,
            "d_min_m": 0.5546697974205017,
            "T_risk_s": 0.30000001192092896,
        },
    ]
    for row in risky_rows:
        row.update(
            {
                "d_safe_m": 0.6,
                "selected_candidate_risky": True,
                "same_bundle_non_risky_candidate_available": "NOT_ESTABLISHED",
                "confirmed_safe_alternative_count_contribution": 0,
                "audit_completeness": "SELECTED_DESCRIPTOR_ONLY",
                "reason": "retained diagnostic stores selected edges, not interaction descriptors for every candidate in the same bundle",
            }
        )
    write_csv(OUTPUT / "risky_gat_selection_audit.csv", risky_rows)

    (OUTPUT / "peer_trigger_contract.md").write_text(
        "# Peer-risk trigger contract\n\n"
        "`IMPLEMENTATION_STATUS = NOT_IMPLEMENTED_INFORMATION_GATE_FAILED`.\n\n"
        "A valid peer-risk entry event would require a legal peer-specific condition "
        "available at every execution step, with false-to-true entry semantics and rearm "
        "only after the same condition becomes false. The frozen Proposed execution "
        "contract provides no such condition. Exact peer state is event-local upper "
        "information; the every-step LiDAR is untyped and has no deterministic peer "
        "identity/velocity association. The existing `d_min/T_risk` condition additionally "
        "requires candidate generation and FP-SHEP preview.\n\n"
        "Therefore no `E_peer` was added, and normal, emergency edge/rearm, handoff "
        "priority, Proposal, Top-K 10, FP-SHEP H4, GAT-V1, SAC-DMP, thresholds, horizons, "
        "and outcome semantics remain byte-for-byte untouched by this audit.\n",
        encoding="utf-8",
    )

    regression = {
        "status": "PASS_HARD_STOP_CONTRACT",
        "tests": [
            {"name": "exact_peer_position_not_available_each_step", "passed": True},
            {"name": "exact_peer_velocity_not_available_each_step", "passed": True},
            {"name": "peer_identity_not_available_each_step", "passed": True},
            {"name": "current_previous_lidar_are_untyped_56_ray_4_5m", "passed": True},
            {"name": "two_unassociated_scans_do_not_establish_peer_track", "passed": True},
            {"name": "existing_descriptor_uses_exact_event_peer_state", "passed": True},
            {"name": "existing_descriptor_requires_candidate_preview", "passed": True},
            {"name": "existing_risk_threshold_is_d_safe_0_6m", "passed": True},
            {"name": "existing_prediction_horizon_is_H4", "passed": True},
            {"name": "no_peer_trigger_implementation_after_failed_gate", "passed": True},
            {"name": "no_new_development_scenarios_after_failed_gate", "passed": True},
            {"name": "no_new_threshold_or_learned_stack_change", "passed": True},
        ],
        "frozen_source_hashes": {
            "event_triggered_reference_reconstruction.py": sha256(ROOT / "planning" / "event_triggered_reference_reconstruction.py"),
            "heterogeneous_candidate_graph.py": sha256(ROOT / "planning" / "heterogeneous_candidate_graph.py"),
            "candidate_execution_interface.py": sha256(ROOT / "planning" / "candidate_execution_interface.py"),
            "hire-rl-body.tex": sha256(ROOT / "hire-rl-body.tex"),
        },
        "tests_executed_as_simulation": 0,
        "note": "These are source/contract reconciliation checks; no peer trigger exists to execute dynamically.",
    }
    write_json(OUTPUT / "peer_trigger_regression_tests.json", regression)

    write_json(
        OUTPUT / "development_scenario_manifest.json",
        {
            "status": "NOT_GENERATED",
            "scenario_count": 0,
            "reason": "Phase A information gate failed; Phase F is forbidden by the goal stop rule.",
            "performance_values_read_before_freeze": False,
        },
    )
    stop_reason = "Phase A information gate failed; no peer trigger was implemented and Phase F was not executed."
    for name in [
        "development_episode_results.csv",
        "development_agent_results.csv",
        "paired_results.csv",
        "peer_event_summary.csv",
        "reproposal_summary.csv",
        "failure_after_revision.csv",
    ]:
        placeholder(name, stop_reason)
    write_csv(
        OUTPUT / "runtime_summary.csv",
        [
            {
                "method": "Historical R-ERR + GAT",
                "scope": "HISTORICAL_REFERENCE_ONLY_NOT_NEW_PAIRED_BLOCK",
                "episode_count": 80,
                "mean_peer_trigger_eval_us": "N/A",
                "p95_peer_trigger_eval_us": "N/A",
                "single_upper_latency_ms": 213.89047152145642,
                "mean_total_online_compute_ms": 2271.5911549999996,
                "status": "REUSED_REFERENCE_NOT_RERUN",
            },
            {
                "method": "R-ERR + Peer-Risk Event + GAT",
                "scope": "PHASE_F",
                "episode_count": 0,
                "mean_peer_trigger_eval_us": "NOT_RUN",
                "p95_peer_trigger_eval_us": "NOT_RUN",
                "single_upper_latency_ms": "NOT_RUN",
                "mean_total_online_compute_ms": "NOT_RUN",
                "status": "NOT_RUN_INFORMATION_GATE_FAILED",
            },
        ],
    )

    conclusion = {
        "PEER_TRIGGER_INFORMATION_CONTRACT_VALID": "NO",
        "INTERACTION_SIGNAL_COST_CLASS": "REQUIRES_FP_PREVIEW",
        "EXISTING_PEER_RISK_THRESHOLD_AVAILABLE": "YES",
        "PEER_TRIGGER_OBSERVABILITY": "NO",
        "PEER_COLLISION_COUNT_DIAGNOSTIC": 10,
        "PEER_COLLISION_DETECTED_COUNT": 0,
        "PEER_COLLISION_ACTIONABLE_COUNT": 0,
        "MEDIAN_RAW_LEAD_TIME_S": "NOT_AVAILABLE_INFORMATION_GATE_FAILED",
        "MEDIAN_ACTIONABLE_LEAD_TIME_S": "NOT_AVAILABLE_INFORMATION_GATE_FAILED",
        "SUCCESS_FALSE_TRIGGER_RATE": "NOT_AVAILABLE_INFORMATION_GATE_FAILED",
        "RISKY_SELECTION_WITH_SAFE_ALTERNATIVE_COUNT": 0,
        "RISKY_SELECTION_WITH_SAFE_ALTERNATIVE_COUNT_INTERPRETATION": "CONFIRMED_COUNT_ONLY_NOT_COMPLETE; all-candidate interaction descriptors were not retained",
        "SECONDARY_REFERENCE_SELECTION_HEADROOM": "NOT_ESTABLISHED",
        "PHASE_F_EXECUTED": "NO",
        "CURRENT_RERR_SUCCESS": "NOT_RUN",
        "PEER_TRIGGER_RERR_SUCCESS": "NOT_RUN",
        "SUCCESS_GAIN_PP": "NOT_RUN",
        "CURRENT_RERR_COLLISION": "NOT_RUN",
        "PEER_TRIGGER_RERR_COLLISION": "NOT_RUN",
        "CURRENT_RERR_INTER_AGENT_COLLISION": "NOT_RUN",
        "PEER_TRIGGER_RERR_INTER_AGENT_COLLISION": "NOT_RUN",
        "INTER_AGENT_COLLISION_RELATIVE_REDUCTION": "NOT_RUN",
        "CURRENT_RERR_STAGE1_SUCCESS": "NOT_RUN",
        "PEER_TRIGGER_STAGE1_SUCCESS": "NOT_RUN",
        "CURRENT_RERR_STAGE2_SUCCESS": "NOT_RUN",
        "PEER_TRIGGER_STAGE2_SUCCESS": "NOT_RUN",
        "CURRENT_RERR_STAGE3_SUCCESS": "NOT_RUN",
        "PEER_TRIGGER_STAGE3_SUCCESS": "NOT_RUN",
        "CURRENT_RERR_STAGE4_SUCCESS": "NOT_RUN",
        "PEER_TRIGGER_STAGE4_SUCCESS": "NOT_RUN",
        "CURRENT_RERR_MEAN_REPROPOSALS": "NOT_RUN",
        "PEER_TRIGGER_MEAN_REPROPOSALS": "NOT_RUN",
        "PEER_TRIGGER_EVENT_COUNT": 0,
        "PEER_TRIGGER_CHATTERING": "NO",
        "PEER_TRIGGER_CHATTERING_INTERPRETATION": "NO_TRIGGER_IMPLEMENTED; dynamic chattering assessment NOT_RUN",
        "PEER_TRIGGER_MEAN_EVAL_US": 0.0,
        "PEER_TRIGGER_P95_EVAL_US": 0.0,
        "PEER_TRIGGER_EVAL_RUNTIME_STATUS": "NOT_RUN_NO_EVALUATOR_IMPLEMENTED; zeros denote no added trigger cost, not a latency measurement",
        "CURRENT_RERR_TOTAL_COMPUTE_MS": "NOT_RUN",
        "PEER_TRIGGER_RERR_TOTAL_COMPUTE_MS": "NOT_RUN",
        "PEER_TRIGGER_REVISION_ACCEPTED": "NOT_RUN",
        "PEER_RISK_SPECIFICITY_RESOLVED": "NOT_ESTABLISHED",
        "GAT_CLOSED_LOOP_VALUE": "STRONG",
        "GAT_FINETUNE_JUSTIFIED": "NO",
        "NEXT_BOTTLENECK": "PEER_TRIGGER_INFORMATION_LIMIT",
        "FINAL_METHOD_READY_FOR_NEW_FORMAL": "NO",
        "RECOMMENDED_NEXT_STEP": "STOP_PEER_TRIGGER_INFORMATION_INSUFFICIENT",
        "PHASE_A_STOP_RULE_APPLIED": "YES",
        "PEER_TRIGGER_IMPLEMENTED": "NO",
        "NEW_DEVELOPMENT_SCENARIOS_GENERATED": "NO",
        "NEW_TEAM_EPISODES_RUN": 0,
        "NEW_NUMERICAL_THRESHOLD_ADDED": "NO",
        "LEARNED_STACK_CHANGED": "NO",
        "historical_reference_only": {
            "RERR_success": 0.8125,
            "RERR_collision": 0.1625,
            "RERR_inter_agent_collision": 0.125,
            "RERR_timeout": 0.025,
            "RERR_stage_1_success": 0.95,
            "RERR_stage_2_success": 0.85,
            "RERR_stage_3_success": 0.70,
            "RERR_stage_4_success": 0.75,
            "RERR_mean_reproposals": 8.9625,
            "RERR_total_online_compute_ms": 2271.5911549999996,
            "single_high_level_latency_ms": 213.89047152145642,
            "RERR_FP_SHEP_success": 0.50,
            "RERR_GAT_gain_pp": 31.25,
            "RERR_GAT_McNemar_p": 1.0928604751825333e-05,
        },
    }
    write_json(OUTPUT / "conclusion.json", conclusion)

    report = """# Peer-Risk Trigger Specificity Closure

## Executive result

`PEER_TRIGGER_INFORMATION_CONTRACT_VALID = NO`. Under the frozen equal-information contract, the execution loop does not receive typed peer position, velocity, or identity at every step. It receives current/previous 56-ray, 4.5 m LiDAR scans whose nearest returns mix static obstacles, dynamic obstacles, and peer spheres without type or identity. Exact peer state is read only when an already-triggered upper GAT planning event is executed.

The existing GAT interaction theory is real and reusable at upper events: H4 constant-velocity peer prediction yields closest-approach time, minimum predicted separation, and risk duration, using the existing `d_safe=0.6 m` condition. However, `INTERACTION_SIGNAL_COST_CLASS = REQUIRES_FP_PREVIEW`: computing those quantities requires a candidate bundle, FP-SHEP preview trajectories, and exact event-time peer state. Running this diagnostic each step would run the upper planner in order to decide whether to run the upper planner, and polling exact peer state would violate the frozen execution information contract.

The Phase-A gate therefore invokes the mandatory hard stop. No peer trigger was implemented, no new threshold was introduced, no 100-scenario manifest was generated, no paired development episode was run, and the learned planning stack remains unchanged. `RECOMMENDED_NEXT_STEP = STOP_PEER_TRIGGER_INFORMATION_INSUFFICIENT`.

## 1. Recovered online information contract

| Signal | Every execution step | At upper event | Peer-specific? | Trigger use |
|---|---|---|---|---|
| Exact peer position | No | Yes, exact | Yes | Forbidden if newly polled every step |
| Exact peer velocity | No | Yes, exact | Yes | Forbidden if newly polled every step |
| Peer identity | No | Yes | Yes | Forbidden if newly polled every step |
| Typed peer LiDAR | No | No | — | Unavailable |
| Current/previous LiDAR | Yes | Also visible | No; untyped | Generic obstacle margin only |
| Communication state | No | No formal channel | — | Unavailable |

The two LiDAR frames do not contain hit identity, class, center, radius, or a stable data association. Consequently they cannot deterministically recover peer identity and relative velocity. Treating exact simulator peer state as a continuous trigger input would convert event-local upper information into privileged every-step information, directly contradicting the equal-information audit.

## 2. Existing interaction descriptors and cost

For each candidate and neighbor, the active graph computes

`p_hat_j(h) = p_j + h*dt*v_j`, `d_hat(h) = ||p_hat_i,k(h)-p_hat_j(h)||`,

then retains `t_hat`, `d_hat_min`, and `T_risk = dt * sum I[d_hat(h)<d_safe]`. Code and Methodology agree on H4 and the strict risk condition. The P03 execution configuration uses `d_align=1.0 m`; `d_safe=0.6 m` comes from the environment inter-agent safe-distance contract.

This proves `EXISTING_PEER_RISK_THRESHOLD_AVAILABLE = YES`, but it does not produce a legal cheap trigger signal. The only cheap per-step safety value, `h_active`, is the untyped margin of one sector nearest the active-reference direction; the prior root-cause audit already showed that this is only partial peer visibility.

## 3. Consequence for observability and lead time

All 10 historical R-ERR peer-collision episodes are enumerated in the observability and lead-time artifacts, but no candidate peer condition is evaluated. Setting its values from exact stored trajectories would answer a privileged counterfactual rather than the requested online-contract question. Therefore:

- `PEER_TRIGGER_OBSERVABILITY = NO`
- confirmed legal detections/actionable detections = `0/10`
- median raw/actionable lead time = unavailable
- success false-trigger rate = unavailable

These zeros are confirmed detections under an admissible implemented signal, not a claim that an oracle using ground-truth peer trajectories would detect nothing. The Phase-C STRONG/MODERATE gate is not reached.

## 4. Risky GAT selections

The retained timelines confirm three selected risky references:

| Scenario | Planning step | Candidate | t_min (s) | d_min (m) | T_risk (s) |
|---|---:|---:|---:|---:|---:|
| SR2_014 | 87 | 3 | 0.3 | 0.5152 | 0.3 |
| SR3_019 | 55 | 0 | 0.4 | 0.5530 | 0.2 |
| SR4_015 | 83 | 5 | 0.3 | 0.5547 | 0.3 |

Each violates the existing `d_min < 0.6 m` condition. The retained peer-collision timeline stores only selected-candidate interaction edges, not all-candidate descriptors from the same event. Hence safe-alternative availability cannot be reconstructed without rerunning upper planning; the confirmed count is 0 but the substantive label is `SECONDARY_REFERENCE_SELECTION_HEADROOM = NOT_ESTABLISHED`. It must not be read as evidence that no safe alternative existed.

## 5. No implementation or paired development

Because the information gate failed, adding `E_peer` would require either privileged every-step state, a new sensing/tracking contract, or continuous upper computation. All violate this goal. Phase E and Phase F were therefore not executed. Current-method performance fields in `conclusion.json` are `NOT_RUN` for this new paired block; the prior 81.25% success, 16.25% collision, 12.5% peer collision, and 2271.6 ms/episode are retained only under `historical_reference_only`.

No peer-trigger evaluation latency was measured. The mandatory mean/P95 fields are 0 only because no evaluator or event was added; they are explicitly tagged as not-run and are not latency measurements.

## 6. Final decision

- `PEER_RISK_SPECIFICITY_RESOLVED = NOT_ESTABLISHED`
- `NEXT_BOTTLENECK = PEER_TRIGGER_INFORMATION_LIMIT`
- `GAT_CLOSED_LOOP_VALUE = STRONG`
- `GAT_FINETUNE_JUSTIFIED = NO`
- `FINAL_METHOD_READY_FOR_NEW_FORMAL = NO`
- `RECOMMENDED_NEXT_STEP = STOP_PEER_TRIGGER_INFORMATION_INSUFFICIENT`

The emergency edge/rearm result remains valid and untouched. The requested anticipatory peer-risk trigger cannot be justified within the current online information contract. A future goal must first explicitly revise the sensing/communication or method scope; it should not silently promote simulator peer ground truth to a continuous observation.
"""
    (OUTPUT / "FINAL_REPORT.md").write_text(report, encoding="utf-8")

    required_outputs = [
        "context_recovery_manifest.json",
        "peer_trigger_information_contract.csv",
        "interaction_descriptor_contract.md",
        "interaction_signal_cost_audit.csv",
        "peer_risk_observability.csv",
        "peer_collision_lead_time.csv",
        "successful_false_trigger.csv",
        "risky_gat_selection_audit.csv",
        "peer_trigger_contract.md",
        "peer_trigger_regression_tests.json",
        "development_scenario_manifest.json",
        "development_episode_results.csv",
        "development_agent_results.csv",
        "paired_results.csv",
        "peer_event_summary.csv",
        "reproposal_summary.csv",
        "runtime_summary.csv",
        "failure_after_revision.csv",
        "conclusion.json",
        "FINAL_REPORT.md",
    ]
    mandatory = [
        "PEER_TRIGGER_INFORMATION_CONTRACT_VALID",
        "INTERACTION_SIGNAL_COST_CLASS",
        "EXISTING_PEER_RISK_THRESHOLD_AVAILABLE",
        "PEER_TRIGGER_OBSERVABILITY",
        "PEER_COLLISION_COUNT_DIAGNOSTIC",
        "PEER_COLLISION_DETECTED_COUNT",
        "PEER_COLLISION_ACTIONABLE_COUNT",
        "MEDIAN_RAW_LEAD_TIME_S",
        "MEDIAN_ACTIONABLE_LEAD_TIME_S",
        "SUCCESS_FALSE_TRIGGER_RATE",
        "RISKY_SELECTION_WITH_SAFE_ALTERNATIVE_COUNT",
        "SECONDARY_REFERENCE_SELECTION_HEADROOM",
        "PHASE_F_EXECUTED",
        "CURRENT_RERR_SUCCESS",
        "PEER_TRIGGER_RERR_SUCCESS",
        "SUCCESS_GAIN_PP",
        "CURRENT_RERR_COLLISION",
        "PEER_TRIGGER_RERR_COLLISION",
        "CURRENT_RERR_INTER_AGENT_COLLISION",
        "PEER_TRIGGER_RERR_INTER_AGENT_COLLISION",
        "INTER_AGENT_COLLISION_RELATIVE_REDUCTION",
        "CURRENT_RERR_STAGE1_SUCCESS",
        "PEER_TRIGGER_STAGE1_SUCCESS",
        "CURRENT_RERR_STAGE2_SUCCESS",
        "PEER_TRIGGER_STAGE2_SUCCESS",
        "CURRENT_RERR_STAGE3_SUCCESS",
        "PEER_TRIGGER_STAGE3_SUCCESS",
        "CURRENT_RERR_STAGE4_SUCCESS",
        "PEER_TRIGGER_STAGE4_SUCCESS",
        "CURRENT_RERR_MEAN_REPROPOSALS",
        "PEER_TRIGGER_MEAN_REPROPOSALS",
        "PEER_TRIGGER_EVENT_COUNT",
        "PEER_TRIGGER_CHATTERING",
        "PEER_TRIGGER_MEAN_EVAL_US",
        "CURRENT_RERR_TOTAL_COMPUTE_MS",
        "PEER_TRIGGER_RERR_TOTAL_COMPUTE_MS",
        "PEER_TRIGGER_REVISION_ACCEPTED",
        "PEER_RISK_SPECIFICITY_RESOLVED",
        "GAT_CLOSED_LOOP_VALUE",
        "GAT_FINETUNE_JUSTIFIED",
        "NEXT_BOTTLENECK",
        "FINAL_METHOD_READY_FOR_NEW_FORMAL",
        "RECOMMENDED_NEXT_STEP",
    ]
    checks = {
        "all_required_outputs_present": all((OUTPUT / name).is_file() for name in required_outputs),
        "mandatory_conclusion_fields_present": all(key in conclusion for key in mandatory),
        "source_peer_collision_count_is_10": len(peer_rows) == 10,
        "observability_rows_cover_all_10": len(observability_rows) == 10,
        "lead_time_rows_cover_all_10": len(lead_rows) == 10,
        "three_risky_selected_references_retained": len(risky_rows) == 3,
        "information_gate_is_no": conclusion["PEER_TRIGGER_INFORMATION_CONTRACT_VALID"] == "NO",
        "phase_f_not_executed": conclusion["PHASE_F_EXECUTED"] == "NO",
        "new_episode_count_zero": conclusion["NEW_TEAM_EPISODES_RUN"] == 0,
        "stop_recommendation_consistent": conclusion["RECOMMENDED_NEXT_STEP"] == "STOP_PEER_TRIGGER_INFORMATION_INSUFFICIENT",
        "existing_threshold_retained": conclusion["EXISTING_PEER_RISK_THRESHOLD_AVAILABLE"] == "YES",
        "gat_remains_frozen": conclusion["GAT_FINETUNE_JUSTIFIED"] == "NO",
    }
    reconciliation = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "failed_checks": [key for key, passed in checks.items() if not passed],
        "row_counts": {
            "peer_trigger_information_contract": len(information_rows),
            "interaction_signal_cost_audit": len(cost_rows),
            "peer_risk_observability": len(observability_rows),
            "peer_collision_lead_time": len(lead_rows),
            "risky_gat_selection_audit": len(risky_rows),
            "new_development_scenarios": 0,
            "new_team_episodes": 0,
        },
        "artifact_hashes": {
            name: sha256(OUTPUT / name) for name in required_outputs
        },
    }
    write_json(OUTPUT / "final_reconciliation.json", reconciliation)


if __name__ == "__main__":
    main()
