"""Independent raw-artifact reconciliation for the R-ERR safety audit."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts/rerr_safety_closure/20260819_134216"


def load_json(name: str) -> Any:
    return json.loads((OUTPUT / name).read_text(encoding="utf-8"))


def read_csv(name: str) -> list[dict[str, str]]:
    with (OUTPUT / name).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    required = (
        "context_recovery_manifest.json",
        "paired_collision_timeline.csv",
        "collision_conversion_classification.csv",
        "replanning_scope_contract.md",
        "replanning_scope_audit.csv",
        "cross_agent_reference_updates.csv",
        "normal_trigger_event_table.csv",
        "normal_trigger_utility.csv",
        "normal_chattering_cases.md",
        "counterfactual_normal_edge_rearm.csv",
        "peer_collision_timeline.csv",
        "peer_collision_classification.csv",
        "root_cause_decision.json",
        "minimal_revision_contract.md",
        "regression_tests.json",
        "development_scenario_manifest.json",
        "development_episode_results.csv",
        "development_agent_results.csv",
        "paired_revision_results.csv",
        "reproposal_summary.csv",
        "runtime_summary.csv",
        "conclusion.json",
        "FINAL_REPORT.md",
        "diagnostic_replay_reconciliation.json",
    )
    missing = [name for name in required if not (OUTPUT / name).exists()]
    if missing:
        raise RuntimeError(f"missing required artifacts: {missing}")

    conclusion = load_json("conclusion.json")
    replay = load_json("diagnostic_replay_reconciliation.json")
    conversions = read_csv("collision_conversion_classification.csv")
    paired_timeline = read_csv("paired_collision_timeline.csv")
    scope = read_csv("replanning_scope_audit.csv")
    cross = read_csv("cross_agent_reference_updates.csv")
    normal = read_csv("normal_trigger_event_table.csv")
    normal_utility = read_csv("normal_trigger_utility.csv")
    counterfactual = read_csv("counterfactual_normal_edge_rearm.csv")
    peer_timeline = read_csv("peer_collision_timeline.csv")
    peer = read_csv("peer_collision_classification.csv")

    checks: dict[str, bool] = {}
    checks["diagnostic_replay_88_pass"] = (
        replay.get("status") == "PASS"
        and int(replay.get("record_count", -1)) == 88
        and int(replay.get("M3_record_count", -1)) == 80
        and int(replay.get("M2_adverse_record_count", -1)) == 8
    )
    checks["paired_timeline_rows"] = len(paired_timeline) == 6606
    checks["conversion_rows"] = len(conversions) == 8
    checks["conversion_counts"] = (
        sum(row["M2_outcome"] == "success" for row in conversions) == 5
        and sum(row["M2_outcome"] == "timeout" for row in conversions) == 3
        and all(
            "EMERGENCY_REARM_TRAJECTORY_EFFECT" in row["contributing_factors"]
            for row in conversions
        )
    )
    checks["first_divergence_order"] = all(
        int(row["FIRST_TRIGGER_DIVERGENCE_STEP"])
        <= int(row["FIRST_REFERENCE_DIVERGENCE_STEP"])
        <= int(row["FIRST_CONTROL_DIVERGENCE_STEP"])
        < int(row["FIRST_PEER_GEOMETRY_DIVERGENCE_STEP"])
        for row in conversions
    )
    checks["scope_rows"] = len(scope) == 689
    checks["scope_contract"] = all(
        row["compute_scope"] == "TEAM_WIDE"
        and row["update_scope"] == "AGENT_LOCAL"
        and int(row["cross_agent_update_count"]) == 0
        for row in scope
    )
    checks["cross_agent_zero"] = (
        len(cross) == 1
        and cross[0]["status"] == "NO_CROSS_AGENT_REFERENCE_UPDATE_OBSERVED"
        and conclusion["CROSS_AGENT_REFERENCE_UPDATE_COUNT"] == 0
    )
    utility_counts = Counter(row["utility"] for row in normal)
    checks["normal_rows"] = len(normal) == 243 and normal == normal_utility
    checks["normal_counts_reproduced"] = (
        utility_counts["HELPFUL"] == conclusion["NORMAL_EVENT_HELPFUL_COUNT"]
        and utility_counts["REDUNDANT"] == conclusion["NORMAL_EVENT_REDUNDANT_COUNT"]
        and utility_counts["HARMFUL"] == conclusion["NORMAL_EVENT_HARMFUL_COUNT"]
        and utility_counts["UNRESOLVED"] == conclusion["NORMAL_EVENT_UNRESOLVED_COUNT"]
    )
    cf_counts = Counter(row["counterfactual_action"] for row in counterfactual)
    removed_utility = Counter(
        row["utility"]
        for row in counterfactual
        if row["counterfactual_action"] == "REMOVE"
    )
    checks["counterfactual_reproduced"] = (
        len(counterfactual) == 243
        and cf_counts["RETAIN"] == 228
        and cf_counts["REMOVE"] == 15
        and removed_utility["HELPFUL"] == 8
        and removed_utility["REDUNDANT"] == 7
        and abs(
            conclusion["COUNTERFACTUAL_NORMAL_EVENT_REDUCTION"] - 15 / 243
        )
        < 1e-15
    )
    checks["peer_rows"] = len(peer) == 10 and len(peer_timeline) == 630
    peer_factors = Counter()
    for row in peer:
        peer_factors.update(json.loads(row["contributing_factors"]))
    checks["peer_counts_reproduced"] = (
        peer_factors["TRIGGER_TOO_LATE"]
        == conclusion["PEER_COLLISION_TRIGGER_TOO_LATE"]
        and peer_factors["REFERENCE_SWITCH_CREATED_CONFLICT"]
        == conclusion["PEER_COLLISION_REFERENCE_SWITCH_CREATED_CONFLICT"]
        and conclusion["PEER_COLLISION_CROSS_AGENT_UPDATE_CREATED_CONFLICT"] == 0
    )
    required_conclusion_fields = (
        "REPLANNING_COMPUTE_SCOPE",
        "REPLANNING_UPDATE_SCOPE",
        "THEORY_REPLANNING_UPDATE_SCOPE",
        "REPLANNING_SCOPE_THEORY_CODE_MATCH",
        "CROSS_AGENT_REFERENCE_UPDATE_COUNT",
        "M2_SUCCESS_TO_M3_COLLISION",
        "M2_TIMEOUT_TO_M3_COLLISION",
        "NORMAL_LEVEL_TRIGGER_PERSISTENCE",
        "NORMAL_EVENT_HELPFUL_COUNT",
        "NORMAL_EVENT_REDUNDANT_COUNT",
        "NORMAL_EVENT_HARMFUL_COUNT",
        "NORMAL_EDGE_REARM_SUPPORTED",
        "COUNTERFACTUAL_NORMAL_EVENT_REDUCTION",
        "PEER_RISK_VISIBLE_TO_TRIGGER",
        "PEER_COLLISION_TRIGGER_TOO_LATE",
        "PEER_COLLISION_REFERENCE_SWITCH_CREATED_CONFLICT",
        "PEER_COLLISION_CROSS_AGENT_UPDATE_CREATED_CONFLICT",
        "PRIMARY_REMAINING_FAILURE_SOURCE",
        "ONE_MINIMAL_CHANGE_APPLIED",
        "PHASE_F_EXECUTED",
        "CURRENT_RERR_SUCCESS",
        "REVISION_SUCCESS",
        "CURRENT_RERR_COLLISION",
        "REVISION_COLLISION",
        "CURRENT_RERR_INTER_AGENT_COLLISION",
        "REVISION_INTER_AGENT_COLLISION",
        "CURRENT_RERR_STAGE2_SUCCESS",
        "REVISION_STAGE2_SUCCESS",
        "CURRENT_RERR_STAGE3_SUCCESS",
        "REVISION_STAGE3_SUCCESS",
        "CURRENT_RERR_STAGE4_SUCCESS",
        "REVISION_STAGE4_SUCCESS",
        "CURRENT_RERR_MEAN_REPROPOSALS",
        "REVISION_MEAN_REPROPOSALS",
        "CURRENT_RERR_TOTAL_COMPUTE_MS",
        "REVISION_TOTAL_COMPUTE_MS",
        "CHATTERING_PRESENT",
        "CHATTERING_SOURCE",
        "REVISION_ACCEPTED",
        "GAT_CLOSED_LOOP_VALUE",
        "GAT_FINETUNE_JUSTIFIED",
        "FINAL_METHOD_READY_FOR_NEW_FORMAL",
        "RECOMMENDED_NEXT_STEP",
    )
    checks["mandatory_conclusion_fields"] = all(
        key in conclusion for key in required_conclusion_fields
    )
    checks["stop_rule"] = (
        conclusion["PRIMARY_REMAINING_FAILURE_SOURCE"] == "PEER_RISK_SPECIFICITY"
        and conclusion["ONE_MINIMAL_CHANGE_APPLIED"] == "NONE"
        and conclusion["PHASE_F_EXECUTED"] == "NO"
        and conclusion["REVISION_ACCEPTED"] == "NOT_RUN"
        and conclusion["FINAL_METHOD_READY_FOR_NEW_FORMAL"] == "NO"
        and conclusion["RECOMMENDED_NEXT_STEP"] == "AUDIT_PEER_RISK_TRIGGER"
        and not (OUTPUT / "FINAL_METHOD_FREEZE.json").exists()
    )
    checks["phase_f_placeholders"] = (
        load_json("development_scenario_manifest.json")["status"] == "NOT_RUN"
        and all(
            read_csv(name)[0]["status"] == "NOT_RUN"
            for name in (
                "development_episode_results.csv",
                "development_agent_results.csv",
                "paired_revision_results.csv",
                "reproposal_summary.csv",
                "runtime_summary.csv",
            )
        )
    )
    checks["regression_tests"] = (
        load_json("regression_tests.json")["status"] == "PASS"
        and load_json("regression_tests.json")["test_count"] == 25
    )

    failed = [name for name, passed in checks.items() if not passed]
    payload = {
        "status": "PASS" if not failed else "FAIL",
        "checks": checks,
        "failed_checks": failed,
        "row_counts": {
            "paired_collision_timeline": len(paired_timeline),
            "collision_conversions": len(conversions),
            "replanning_scope": len(scope),
            "normal_events": len(normal),
            "counterfactual_normal_events": len(counterfactual),
            "peer_collision_timeline": len(peer_timeline),
            "peer_collision_episodes": len(peer),
        },
        "artifact_hashes": {
            name: sha256(OUTPUT / name) for name in required
        },
    }
    temporary = OUTPUT / "final_reconciliation.json.tmp"
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(OUTPUT / "final_reconciliation.json")
    if failed:
        raise RuntimeError(f"reconciliation failed: {failed}")
    print("FINAL_RECONCILIATION=PASS")


if __name__ == "__main__":
    main()
