"""Independent reconciliation for the read-only classical-baseline audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


REQUIRED_OUTPUTS = (
    "config.json",
    "context_recovery_manifest.json",
    "integrity_manifest.json",
    "aggregate_reproduction.csv",
    "trajectory_collision_recheck.csv",
    "swept_collision_audit.csv",
    "dynamic_time_alignment.csv",
    "obstacle_geometry_contract.csv",
    "radius_contract.csv",
    "speed_acceleration_audit.csv",
    "dynamics_interface_contract.csv",
    "method_information_contract.csv",
    "future_information_audit.csv",
    "termination_order_audit.csv",
    "success_contract_audit.csv",
    "post_success_semantics.csv",
    "trajectory_geometry_audit.csv",
    "position_jump_audit.csv",
    "dwa_semantic_audit.md",
    "rvo_semantic_audit.md",
    "prediction_horizon_comparison.csv",
    "planning_frequency.csv",
    "runtime_reconciliation.csv",
    "path_efficiency_audit.csv",
    "comparison_contract.csv",
    "issue_impact_assessment.csv",
    "conclusion.json",
    "FINAL_REPORT.md",
)

FORCED_FIELDS = (
    "BASELINE_AGGREGATE_REPRODUCTION",
    "INDEPENDENT_COLLISION_RECHECK_MATCH",
    "DWA_SWEEP_ONLY_COLLISIONS",
    "RVO_SWEEP_ONLY_COLLISIONS",
    "PROPOSED_SWEEP_ONLY_COLLISIONS",
    "DISCRETE_TIME_TUNNELING_ADVANTAGE",
    "DYNAMIC_OBSTACLE_TIME_ALIGNMENT",
    "CLASSICAL_OBSTACLE_MODEL_MISMATCH",
    "CLASSICAL_UNDER_INFLATED_GEOMETRY",
    "SPEED_CONTRACT_EQUAL",
    "CLASSICAL_ACCELERATION_BYPASS",
    "CONTROL_AUTHORITY_EQUAL",
    "FULL_STATE_GEOMETRY_ADVANTAGE",
    "PEER_INFORMATION_ADVANTAGE",
    "FUTURE_DYNAMIC_INFORMATION_LEAKAGE",
    "FUTURE_PEER_INFORMATION_LEAKAGE",
    "TERMINATION_EVENT_ORDER_EQUAL",
    "TEAM_SUCCESS_CONTRACT_EQUAL",
    "POST_SUCCESS_AGENT_SEMANTIC_MISMATCH",
    "TRAJECTORY_GEOMETRY_VALID",
    "POSITION_JUMP_VIOLATION_COUNT_DWA",
    "POSITION_JUMP_VIOLATION_COUNT_RVO",
    "RVO_IMPLEMENTATION_SCOPE",
    "DWA_IMPLEMENTATION_SCOPE",
    "PREDICTION_HORIZON_ADVANTAGE",
    "CLASSICAL_CLOSED_LOOP_UPDATE_ADVANTAGE",
    "PATH_EFFICIENCY_GT1_CAUSE",
    "AGENT_SUCCESS_RECONSTRUCTION_MATCH",
    "BASELINE_COMPARISON_FAIRNESS",
    "DWA_FORMAL_RESULT_VALID",
    "RVO_FORMAL_RESULT_VALID",
    "INVALIDATING_ISSUE_COUNT",
    "BASELINE_RERUN_REQUIRED",
    "PRIMARY_CLASSICAL_ADVANTAGE_SOURCE",
    "RECOMMENDED_NEXT_STEP",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-artifact", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def truth(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_digest(root: Path, pattern: str) -> dict[str, Any]:
    paths = sorted(path for path in root.glob(pattern) if path.is_file())
    digest = hashlib.sha256()
    total = 0
    for path in paths:
        rel = path.relative_to(root).as_posix()
        payload_hash = sha256(path)
        size = path.stat().st_size
        total += size
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload_hash.encode("ascii"))
        digest.update(b"\n")
    return {"file_count": len(paths), "total_bytes": total, "tree_sha256": digest.hexdigest()}


def main() -> None:
    args = parse_args()
    source = args.source_artifact.resolve()
    audit = args.audit_dir.resolve()
    failures: list[str] = []

    missing = [name for name in REQUIRED_OUTPUTS if not (audit / name).is_file()]
    if missing:
        failures.append(f"missing outputs: {missing}")

    conclusion = read_json(audit / "conclusion.json")
    missing_fields = [field for field in FORCED_FIELDS if field not in conclusion]
    if missing_fields:
        failures.append(f"missing conclusion fields: {missing_fields}")

    raw_counts: dict[str, Counter[str]] = {}
    for method in ("dwa_style", "rvo_orca_style"):
        counter: Counter[str] = Counter()
        files = sorted((source / "formal_records").glob(f"**/{method}.json"))
        for path in files:
            row = read_json(path)["episode"]
            counter["records"] += 1
            counter["success"] += int(bool(row["team_success"]))
            counter["collision"] += int(bool(row["any_collision"]))
            counter["timeout"] += int(bool(row["timeout"]))
            counter["planner_infeasible"] += int(row["termination_reason"] == "planner_infeasible")
        raw_counts[method] = counter
    expected = {
        "dwa_style": Counter(records=400, success=393, collision=1, timeout=6, planner_infeasible=1),
        "rvo_orca_style": Counter(records=400, success=395, collision=4, timeout=1, planner_infeasible=0),
    }
    if raw_counts != expected:
        failures.append(f"raw aggregates differ: {raw_counts}")

    collision = read_csv(audit / "trajectory_collision_recheck.csv")
    collision_methods = Counter(row["method"] for row in collision)
    if len(collision) != 1200 or collision_methods != Counter(dwa_style=400, rvo_orca_style=400, gat_v1=400):
        failures.append(f"collision row contract differs: n={len(collision)}, methods={collision_methods}")
    if any(not truth(row["all_collision_flags_match"]) for row in collision):
        failures.append("at least one independently reconstructed collision flag differs")

    swept = read_csv(audit / "swept_collision_audit.csv")
    swept_counts = Counter(
        row["method"] for row in swept if truth(row["sweep_only_collision_episode"])
    )
    expected_swept = {
        "dwa_style": int(conclusion["DWA_SWEEP_ONLY_COLLISIONS"]),
        "rvo_orca_style": int(conclusion["RVO_SWEEP_ONLY_COLLISIONS"]),
        "gat_v1": int(conclusion["PROPOSED_SWEEP_ONLY_COLLISIONS"]),
    }
    if any(swept_counts[method] != count for method, count in expected_swept.items()):
        failures.append(f"sweep counts differ: {swept_counts} vs {expected_swept}")

    dynamic = read_csv(audit / "dynamic_time_alignment.csv")
    if any(row["alignment"] != "YES" for row in dynamic):
        failures.append("dynamic-obstacle time alignment did not pass")

    speeds = read_csv(audit / "speed_acceleration_audit.csv")
    classic_overall = [row for row in speeds if row["scope"] == "overall" and row["method"] in {"dwa_style", "rvo_orca_style"}]
    if any(int(row["applied_acceleration_component_violation_count"]) for row in classic_overall):
        failures.append("persisted classic applied acceleration violates the component limit")
    if any(int(row["unexplained_velocity_delta_acceleration_violation_count"]) for row in classic_overall):
        failures.append("classic velocity delta contains a non-freeze acceleration violation")
    if any(int(row["velocity_component_violation_count"]) for row in classic_overall):
        failures.append("classic velocity violates the component limit")

    future = read_csv(audit / "future_information_audit.csv")
    classic_future = [row for row in future if row["information_type"] == "dynamic_obstacle_future" and row["method"] in {"dwa_style", "rvo_orca_style"}]
    if len(classic_future) != 2 or any(not truth(row["future_information_leakage"]) for row in classic_future):
        failures.append("classic future-information finding is not represented twice")
    if any(int(row["audited_prediction_comparisons"]) <= 0 or float(row["max_future_position_error_m"]) > 1.0e-12 for row in classic_future):
        failures.append("future prediction reproduction evidence differs")

    path_rows = read_csv(audit / "path_efficiency_audit.csv")
    if any(float(row["path_length_abs_error_m"]) > 1.0e-9 for row in path_rows):
        failures.append("path-length reconstruction differs")
    gt_one = [row for row in path_rows if truth(row["raw_efficiency_gt_1"])]
    if not gt_one or any(not truth(row["adjusted_efficiency_le_1"]) for row in gt_one):
        failures.append("success-radius explanation for efficiency above one does not reconcile")

    integrity = read_json(audit / "integrity_manifest.json")
    for name, expected_hash in integrity["output_hashes"].items():
        if sha256(audit / name) != expected_hash:
            failures.append(f"audit output hash differs: {name}")
    for name, expected_hash in integrity["source_hashes"].items():
        if sha256(source / name) != expected_hash:
            failures.append(f"source artifact changed: {name}")
    source_tree_specs = {
        "formal_records": (source / "formal_records", "**/*.json"),
        "trajectories": (source / "trajectories", "**/*.npz"),
        "method_configs": (source / "method_configs", "*.json"),
    }
    for name, (root, pattern) in source_tree_specs.items():
        if tree_digest(root, pattern) != integrity["source_trees"][name]:
            failures.append(f"source tree changed: {name}")

    report = (audit / "FINAL_REPORT.md").read_text(encoding="utf-8")
    section_count = sum(line.startswith("## ") for line in report.splitlines())
    if section_count != 19:  # executive + 17 numbered sections + stop rule
        failures.append(f"report section count differs: {section_count}")

    expected_conclusions = {
        "BASELINE_AGGREGATE_REPRODUCTION": "YES",
        "INDEPENDENT_COLLISION_RECHECK_MATCH": "YES",
        "CLASSICAL_ACCELERATION_BYPASS": "NO",
        "FUTURE_DYNAMIC_INFORMATION_LEAKAGE": "YES",
        "BASELINE_COMPARISON_FAIRNESS": "INVALID",
        "DWA_FORMAL_RESULT_VALID": "CONDITIONAL",
        "RVO_FORMAL_RESULT_VALID": "CONDITIONAL",
        "INVALIDATING_ISSUE_COUNT": 1,
        "BASELINE_RERUN_REQUIRED": "YES",
    }
    for field, expected_value in expected_conclusions.items():
        if conclusion.get(field) != expected_value:
            failures.append(f"conclusion differs: {field}={conclusion.get(field)!r}, expected {expected_value!r}")

    payload = {
        "schema_version": "classical_baseline_validity_audit_reconciliation_v1",
        "status": "PASSED" if not failures else "FAILED",
        "checks": {
            "required_output_count": len(REQUIRED_OUTPUTS),
            "forced_conclusion_field_count": len(FORCED_FIELDS),
            "raw_aggregates": {method: dict(counts) for method, counts in raw_counts.items()},
            "collision_rows": len(collision),
            "collision_method_counts": dict(collision_methods),
            "sweep_only_episode_counts": dict(swept_counts),
            "path_efficiency_rows": len(path_rows),
            "raw_efficiency_gt_one_rows": len(gt_one),
            "report_section_count": section_count,
            "source_hashes_unchanged": not any(item.startswith("source") for item in failures),
            "formal_episode_rerun": False,
        },
        "audit_integrity_manifest_sha256": sha256(audit / "integrity_manifest.json"),
        "failure_count": len(failures),
        "failure_examples": failures[:20],
    }
    target = audit / "independent_reconciliation.json"
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
