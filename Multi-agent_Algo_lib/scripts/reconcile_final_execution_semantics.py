from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts/final_execution_semantics_audit/20260819_194912"
ENGINEERING = ROOT / "artifacts/rerr_runtime_compression/20260819_162022/FINAL_ENGINEERING_FREEZE.json"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(name: str) -> list[dict[str, str]]:
    with (OUTPUT / name).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def truth(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    required = (
        "context_recovery_manifest.json",
        "theory_execution_order.md",
        "code_execution_order.md",
        "execution_order_audit.csv",
        "same_goal_state_reset_audit.csv",
        "same_goal_consequence_audit.csv",
        "multi_agent_atomicity_audit.csv",
        "agent_order_invariance.csv",
        "recurrent_state_timestamp_audit.csv",
        "lidar_history_alignment.csv",
        "candidate_index_contract.csv",
        "stale_candidate_audit.csv",
        "diagnostic_replay_equivalence.csv",
        "root_cause_decision.json",
        "conclusion.json",
        "regression_tests.json",
        "runtime_summary.csv",
        "FINAL_METHOD_FREEZE.json",
        "FINAL_REPORT.md",
    )
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, observed: Any, expected: Any) -> None:
        checks.append(
            {
                "check": name,
                "pass": bool(condition),
                "observed": observed,
                "expected": expected,
            }
        )

    missing = [name for name in required if not (OUTPUT / name).is_file()]
    check("required_artifacts", not missing, missing, [])

    conclusion = read_json(OUTPUT / "conclusion.json")
    decision = read_json(OUTPUT / "root_cause_decision.json")
    freeze = read_json(OUTPUT / "FINAL_METHOD_FREEZE.json")
    engineering = read_json(ENGINEERING)
    regression = read_json(OUTPUT / "regression_tests.json")
    context = read_json(OUTPUT / "context_recovery_manifest.json")

    check("decision_conclusion_exact", conclusion == decision, conclusion == decision, True)
    mismatch_fields = (
        "TRIGGER_ACTION_MISMATCH",
        "SAME_GOAL_RESET_THEORY_CODE_MISMATCH",
        "MULTI_AGENT_ATOMICITY_MISMATCH",
        "RECURRENT_STATE_FRESHNESS_MISMATCH",
        "CANDIDATE_INDEX_MAPPING_MISMATCH",
    )
    observed_mismatches = {key: conclusion.get(key) for key in mismatch_fields}
    check(
        "five_mismatch_fields",
        all(value == "NO" for value in observed_mismatches.values()),
        observed_mismatches,
        {key: "NO" for key in mismatch_fields},
    )
    final_fields = {
        "PRIMARY_THEORY_CODE_MISMATCH": "NONE",
        "ONE_MINIMAL_FIX_APPLIED": "NONE",
        "SECOND_CORRECTNESS_ISSUE_REMAINS": "NO",
        "FINAL_NON_THEORETICAL_HEADROOM_EXHAUSTED": "YES",
        "PHASE_H_EXECUTED": "NO",
        "FINAL_METHOD_READY_FOR_NEW_FORMAL": "YES",
        "RECOMMENDED_NEXT_STEP": "FREEZE_CURRENT_METHOD",
    }
    check(
        "stop_rule_and_freeze_decision",
        all(conclusion.get(key) == value for key, value in final_fields.items()),
        {key: conclusion.get(key) for key in final_fields},
        final_fields,
    )

    order = read_csv("execution_order_audit.csv")
    aligned_count = sum(row["alignment"] == "YES" for row in order)
    theory_undefined_count = sum(
        row["alignment"] == "CODE_DEFINED_THEORY_ORDER_UNDEFINED" for row in order
    )
    check(
        "execution_order",
        len(order) == 9 and aligned_count == 7 and theory_undefined_count == 2,
        {
            "rows": len(order),
            "aligned": aligned_count,
            "code_defined_theory_order_undefined": theory_undefined_count,
        },
        {
            "rows": 9,
            "aligned": 7,
            "code_defined_theory_order_undefined": 2,
        },
    )

    same = read_csv("same_goal_state_reset_audit.csv")
    same_fields = (
        "execution_age_reset",
        "progress_window_reset",
        "normal_dwell_reset",
        "last_reference_update_step_reset",
        "progress_history_cleared_and_seeded",
        "goal_version_incremented",
        "emergency_latch_resynchronized_to_new_active_direction",
        "dmp_goal_setter_called",
        "dmp_phase_preserved",
    )
    same_ok = len(same) == 115 and all(
        all(truth(row[field]) for field in same_fields)
        and not truth(row["dmp_phase_reset"])
        and float(row["goal_distance_change_m"]) <= float(row["existing_goal_tolerance_m"])
        for row in same
    )
    check("same_goal_computation_event_resets", same_ok, len(same), 115)
    consequences = read_csv("same_goal_consequence_audit.csv")
    legal_suppressed = sum(int(row["legal_trigger_suppressed_by_incorrect_reset"]) for row in consequences)
    dwell_blocked = sum(int(row["surface_positive_but_theoretical_dwell_unsatisfied"]) for row in consequences)
    check("same_goal_consequence_windows", len(consequences) == 345 and legal_suppressed == 0 and dwell_blocked == 977, {"rows": len(consequences), "legal_suppressed": legal_suppressed, "dwell_blocked": dwell_blocked}, {"rows": 345, "legal_suppressed": 0, "dwell_blocked": 977})

    atomic = read_csv("multi_agent_atomicity_audit.csv")
    atomic_fields = (
        "all_trigger_decisions_collected_before_commit",
        "single_precommit_upper_call",
        "selection_computed_before_any_reference_commit",
        "upper_and_order_replay_state_immutable",
        "semantic_commit_set_atomic",
    )
    check("multi_agent_atomicity", len(atomic) == 28 and all(all(truth(row[field]) for field in atomic_fields) and not truth(row["sequential_python_writes_visible_to_planner"]) for row in atomic), len(atomic), 28)
    invariant = read_csv("agent_order_invariance.csv")
    check("agent_order_invariance", len(invariant) == 28 and all(truth(row["agent_order_invariant"]) for row in invariant), len(invariant), 28)

    recurrent = read_csv("recurrent_state_timestamp_audit.csv")
    timestamp_fields = (
        "trigger_state_timestamp",
        "ego_position_timestamp",
        "ego_velocity_timestamp",
        "lidar_current_timestamp",
        "peer_state_timestamp",
        "dynamic_obstacle_timestamp",
        "candidate_state_timestamp",
        "fp_shep_initial_state_timestamp",
        "gat_graph_timestamp",
        "dmp_phase_timestamp",
        "new_goal_actor_first_use_timestamp",
    )
    recurrent_ok = len(recurrent) == 769 and all(
        truth(row["upper_state_immutable"])
        and all(int(row[field]) == int(row["event_step"]) for field in timestamp_fields)
        and int(row["lidar_previous_timestamp"]) == max(int(row["event_step"]) - 1, 0)
        for row in recurrent
    )
    check("recurrent_timestamp_alignment", recurrent_ok, len(recurrent), 769)

    lidar = read_csv("lidar_history_alignment.csv")
    lidar_fields = (
        "step_increment_one",
        "all_previous_equal_pre_current",
        "all_current_equal_post_state_scan",
        "all_sensor_internal_equal_current",
        "all_previous_velocity_aligned",
    )
    lidar_ok = len(lidar) == 10023 and all(
        all(truth(row[field]) for field in lidar_fields)
        and float(row["maximum_current_scan_abs_error"]) == 0.0
        for row in lidar
    )
    check("lidar_history", lidar_ok, len(lidar), 10023)

    candidates = read_csv("candidate_index_contract.csv")
    candidate_fields = (
        "alignment_pass",
        "candidate_descriptor_length_alignment",
        "class_index_mapping_valid",
        "selected_goal_decode_valid",
        "candidate_points_finite",
        "proposal_scores_finite",
        "nonclearance_descriptors_finite",
        "normalized_descriptors_finite_and_aligned",
        "open_space_sentinel_contract_valid",
        "descriptor_numeric_contract_valid",
        "logits_finite",
        "probabilities_finite",
        "correct_ego_ownership",
        "proposal_fp_graph_order_preserved",
    )
    sentinel_count = sum(int(row["nonfinite_raw_min_clearance_candidate_count"]) for row in candidates)
    candidate_ok = len(candidates) == 957 and all(all(truth(row[field]) for field in candidate_fields) for row in candidates)
    check("candidate_descriptor_index_contract", candidate_ok and sentinel_count == 430, {"rows": len(candidates), "open_space_sentinels": sentinel_count}, {"rows": 957, "open_space_sentinels": 430})

    stale = read_csv("stale_candidate_audit.csv")
    stale_ok = len(stale) == 769 and all(
        truth(row["fresh_proposal_generation_call"])
        and truth(row["fresh_fp_shep_call"])
        and truth(row["fresh_graph_build_call"])
        and not truth(row["cross_event_cache_enabled"])
        and not truth(row["stale_recurrent_candidate_usage"])
        for row in stale
    )
    check("no_stale_candidate_use", stale_ok, len(stale), 769)

    behavior = read_csv("diagnostic_replay_equivalence.csv")
    behavior_fields = (
        "event_sequence_match",
        "trigger_sequence_match",
        "trajectory_hash_match",
        "outcome_match",
        "all_exact",
    )
    check("diagnostic_replay", len(behavior) == 80 and all(all(truth(row[field]) for field in behavior_fields) for row in behavior), len(behavior), 80)

    check("legacy_regressions", regression.get("status") == "PASS" and regression.get("test_count") == 25 and len(regression.get("tests", [])) == 25 and all(row.get("status") == "PASS" for row in regression.get("tests", [])) and all(regression.get("semantic_audit_checks", {}).values()), {"status": regression.get("status"), "count": regression.get("test_count")}, {"status": "PASS", "count": 25})

    source_hashes = freeze["source_sha256"]
    actual_sources = {relative: sha256(ROOT / relative) for relative in source_hashes}
    check("frozen_source_hashes", actual_sources == source_hashes, actual_sources, source_hashes)
    engineering_hash = sha256(ENGINEERING)
    check("engineering_freeze_hash", engineering_hash == freeze["engineering_freeze_sha256"], engineering_hash, freeze["engineering_freeze_sha256"])
    check("checkpoint_hash_inheritance", freeze["checkpoint_sha256"] == engineering["checkpoint_sha256"], freeze["checkpoint_sha256"], engineering["checkpoint_sha256"])
    check("context_freeze_verified", context.get("accepted_engineering_freeze_verified") is True and all(value is False for value in context.get("forbidden_scope_changes", {}).values()), {"accepted": context.get("accepted_engineering_freeze_verified"), "scope": context.get("forbidden_scope_changes")}, {"accepted": True, "scope_all_false": True})
    check("freeze_status", freeze.get("status") == "READY_FOR_NEW_FORMAL" and freeze.get("method_code_modified_by_this_audit") == [] and freeze.get("execution_semantics", {}).get("new_goal_first_use_delay_steps") == 0, {"status": freeze.get("status"), "method_code_modified": freeze.get("method_code_modified_by_this_audit"), "delay": freeze.get("execution_semantics", {}).get("new_goal_first_use_delay_steps")}, {"status": "READY_FOR_NEW_FORMAL", "method_code_modified": [], "delay": 0})

    runtime = read_csv("runtime_summary.csv")
    current_runtime = next(row for row in runtime if row["behavior_match"] == "YES")
    fix_runtime = next(row for row in runtime if row["method"] == "Fix arm")
    runtime_ok = abs(float(current_runtime["single_upper_latency_ms"]) - 41.075544473342) < 1.0e-12 and abs(float(current_runtime["total_online_compute_ms_per_episode"]) - 613.6181124999999) < 1.0e-9 and fix_runtime["behavior_match"] == "NOT_RUN"
    check("runtime_freeze", runtime_ok, current_runtime, {"single_upper_latency_ms": 41.075544473342, "total_online_compute_ms_per_episode": 613.6181124999999, "fix": "NOT_RUN"})

    forbidden_new_outputs = (
        "minimal_fix_contract.md",
        "development_scenario_manifest.json",
        "paired_development_results.csv",
        "paired_transition_table.csv",
    )
    present_forbidden = [name for name in forbidden_new_outputs if (OUTPUT / name).exists()]
    check("no_fix_or_phase_h_outputs", not present_forbidden, present_forbidden, [])

    status = "PASS" if all(row["pass"] for row in checks) else "FAIL"
    artifact_hashes = {
        name: sha256(OUTPUT / name)
        for name in required
        if (OUTPUT / name).is_file()
    }
    payload = {
        "schema_version": "final_execution_semantics_independent_reconciliation_v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "status": status,
        "check_count": len(checks),
        "passed_check_count": sum(bool(row["pass"]) for row in checks),
        "checks": checks,
        "recomputed_counts": {
            "episodes": len(behavior),
            "upper_calls": len(recurrent),
            "candidate_selection_events": len(candidates),
            "multi_agent_events": len(atomic),
            "environment_transitions": len(lidar),
            "same_goal_events": len(same),
            "open_space_clearance_sentinels": sentinel_count,
            "legacy_regression_tests": regression.get("test_count"),
        },
        "artifact_sha256": artifact_hashes,
    }
    (OUTPUT / "final_reconciliation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if status != "PASS":
        failed = [row["check"] for row in checks if not row["pass"]]
        raise RuntimeError(f"independent reconciliation failed: {failed}")
    print(json.dumps({"status": status, "checks": len(checks), "output": str(OUTPUT / "final_reconciliation.json")}))


if __name__ == "__main__":
    main()
