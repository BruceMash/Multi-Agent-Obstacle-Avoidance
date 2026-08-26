from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = REPO_ROOT / "artifacts" / "rerr_runtime_compression" / "20260819_162022"
SPARSE = REPO_ROOT / "artifacts" / "sparse_err_trigger_revision" / "20260819_020727"
SAFETY = REPO_ROOT / "artifacts" / "rerr_safety_closure" / "20260819_134216"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def canonical(value: Any) -> Any:
    """Match the authority writer's representation of unavailable numerics."""

    if isinstance(value, dict):
        return {str(key): canonical(item) for key, item in value.items()}
    if isinstance(value, list):
        return [canonical(item) for item in value]
    if value in {"NaN", "Infinity", "-Infinity"}:
        return None
    return value


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            canonical(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def main() -> None:
    rows: list[dict[str, Any]] = []
    event_fields = (
        "step",
        "agent_id",
        "event",
        "selected_candidate_id",
        "selected_null",
        "new_active_goal",
        "new_active_goal_type",
        "goal_changed",
        "phase_before",
        "phase_after",
    )
    trigger_fields = (
        "step",
        "agent_id",
        "event",
        "normal_condition",
        "emergency_condition",
    )
    for path in sorted((OUTPUT / "optimized_replay_records").rglob("*.json")):
        optimized = load_json(path)
        entry = optimized["entry"]
        sparse = load_json(
            SPARSE
            / "development_records"
            / entry["stage"]
            / entry["scenario_id"]
            / "gat_v1_rerr.json"
        )
        safety = load_json(
            SAFETY
            / "diagnostic_records"
            / entry["stage"]
            / entry["scenario_id"]
            / "gat_v1_rerr.json"
        )
        new_events = [
            {key: row.get(key) for key in event_fields}
            for row in optimized["events"]
        ]
        old_events = [
            {key: row.get(key) for key in event_fields} for row in sparse["events"]
        ]
        new_triggers = [
            {key: row.get(key) for key in trigger_fields}
            for row in optimized["triggers"]
        ]
        old_triggers = [
            {key: row.get(key) for key in trigger_fields}
            for row in sparse["triggers"]
        ]
        episode = optimized["episode"]
        outcome_match = all(
            episode[key] == sparse["episode"][key]
            for key in (
                "team_success",
                "collision",
                "obstacle_collision",
                "inter_agent_collision",
                "timeout",
                "termination_reason",
                "steps",
                "replanning_count",
                "planning_decision_count",
            )
        )
        new_hash = stable_hash(optimized["path_rows"])
        old_hash = stable_hash(safety["path_rows"])
        rows.append(
            {
                "stage": entry["stage"],
                "scenario_id": entry["scenario_id"],
                "seed": int(entry["seed"]),
                "outcome_match": outcome_match,
                "event_sequence_match": canonical(new_events) == canonical(old_events),
                "trigger_sequence_match": canonical(new_triggers) == canonical(old_triggers),
                "trajectory_hash_match": new_hash == old_hash,
                "new_trajectory_hash": new_hash,
                "source_trajectory_hash": old_hash,
                "event_count": len(new_events),
                "trigger_count": len(new_triggers),
                "canonicalization": "nonfinite_unavailable_numeric_to_null",
            }
        )
    write_csv(OUTPUT / "closed_loop_replay_events.csv", rows)
    status = all(
        row["outcome_match"]
        and row["event_sequence_match"]
        and row["trigger_sequence_match"]
        and row["trajectory_hash_match"]
        for row in rows
    )
    result = {
        "status": "PASS" if status else "FAIL",
        "episode_count": len(rows),
        "outcome_match_count": sum(row["outcome_match"] for row in rows),
        "event_sequence_match_count": sum(row["event_sequence_match"] for row in rows),
        "trigger_sequence_match_count": sum(
            row["trigger_sequence_match"] for row in rows
        ),
        "trajectory_hash_match_count": sum(
            row["trajectory_hash_match"] for row in rows
        ),
        "OPTIMIZED_BEHAVIOR_MATCH": "YES" if status else "NO",
        "reconciliation_correction": {
            "type": "representation_only",
            "original_false_mismatch_cause": (
                "optimized artifact writer serialized non-finite unavailable initial-step "
                "values as string tokens while the authority writer serialized them as null"
            ),
            "trajectory_numeric_or_categorical_value_changed": False,
            "canonical_rule": "NaN/Infinity/-Infinity string tokens and authority null are equivalent unavailable values",
        },
    }
    write_json(OUTPUT / "closed_loop_replay_reconciliation.json", result)
    if not status:
        raise RuntimeError("canonical independent reconciliation failed")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
