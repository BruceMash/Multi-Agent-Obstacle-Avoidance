#!/usr/bin/env python3
"""One-time trace-only amendment after the pre-result serializer failure."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "artifacts/turn_sign_persistence_zigzag_repair/20260825_222130"
FREEZE = ROOT / "TURN_PERSISTENCE_RUNTIME_FREEZE.json"
AMENDMENT = ROOT / "TURN_PERSISTENCE_TRACE_SERIALIZATION_AMENDMENT.json"
PERFORMANCE_AMENDMENT = ROOT / "TURN_PERSISTENCE_TRACE_PERFORMANCE_AMENDMENT.json"
SOURCE = REPO_ROOT / "planning/turn_sign_persistence.py"
REPAIRED = ROOT / "development/repaired/episode_records"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    if AMENDMENT.exists():
        if PERFORMANCE_AMENDMENT.exists():
            raise RuntimeError("trace performance amendment already exists")
        successful = sorted(
            path.name for path in REPAIRED.glob("TSP_DEV_*.json")
            if not path.name.endswith("_SOFTWARE_ERROR.json")
        )
        if successful:
            raise RuntimeError(f"performance amendment requires zero saved repaired results: {successful}")
        freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
        source_name = SOURCE.relative_to(REPO_ROOT).as_posix()
        previous = freeze["source_sha256"][source_name]
        current = sha256(SOURCE)
        if previous == current:
            raise RuntimeError("wrapper source hash did not change")
        amendment = {
            "schema_version": "turn_persistence_trace_performance_amendment_v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "DIAGNOSTIC_COMPLEXITY_FIX_BEFORE_ANY_REPAIRED_RESULT_WAS_SAVED",
            "interrupted_scenario_ids": ["TSP_DEV_1_000", "TSP_DEV_1_001", "TSP_DEV_1_002", "TSP_DEV_1_003"],
            "saved_repaired_performance_records_before_fix": 0,
            "control_semantics_changed": False,
            "fix": "read only the last finalized wrapped Strong trace row instead of serializing the entire trace at every control step",
            "complexity_change": "diagnostic O(T^2) to O(T); acceleration output unchanged",
            "source": source_name,
            "old_sha256": previous,
            "new_sha256": current,
        }
        write(PERFORMANCE_AMENDMENT, amendment)
        freeze["source_sha256"][source_name] = current
        freeze["post_freeze_trace_performance_amendment"] = {
            "artifact": PERFORMANCE_AMENDMENT.relative_to(REPO_ROOT).as_posix(),
            "artifact_sha256": sha256(PERFORMANCE_AMENDMENT),
            "saved_repaired_performance_records_before_fix": 0,
            "control_semantics_changed": False,
        }
        write(FREEZE, freeze)
        print(json.dumps(amendment, indent=2))
        return
    successful = sorted(
        path.name for path in REPAIRED.glob("TSP_DEV_*.json")
        if not path.name.endswith("_SOFTWARE_ERROR.json")
    )
    errors = sorted(REPAIRED.glob("*_SOFTWARE_ERROR.json"))
    if successful or len(errors) != 4:
        raise RuntimeError(f"amendment requires 0 saved results and exactly 4 serializer errors: {successful}, {errors}")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in errors]
    if any(row.get("error_type") != "KeyError" or row.get("error_message") != "'executed_jerk_norm_mps3'" for row in payloads):
        raise RuntimeError("failure signature is not the known post-execution trace serializer defect")
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    source_name = SOURCE.relative_to(REPO_ROOT).as_posix()
    previous = freeze["source_sha256"][source_name]
    current = sha256(SOURCE)
    if previous == current:
        raise RuntimeError("wrapper source hash did not change")
    amendment = {
        "schema_version": "turn_persistence_trace_serialization_amendment_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "TRACE_ONLY_FIX_BEFORE_ANY_REPAIRED_RESULT_WAS_SAVED",
        "excluded_scenario_ids": [row["scenario_id"] for row in payloads],
        "saved_repaired_performance_records_before_fix": 0,
        "failure_point": "save_episode_record after run_episode returned",
        "observed_or_retained_repaired_outcomes_before_fix": 0,
        "control_semantics_changed": False,
        "fix": "merge the completed wrapped Strong diagnostic row after observe_executed so jerk fields are serializable",
        "source": source_name,
        "old_sha256": previous,
        "new_sha256": current,
    }
    write(AMENDMENT, amendment)
    freeze["source_sha256"][source_name] = current
    freeze["post_freeze_trace_only_amendment"] = {
        "artifact": AMENDMENT.relative_to(REPO_ROOT).as_posix(),
        "artifact_sha256": sha256(AMENDMENT),
        "saved_repaired_performance_records_before_fix": 0,
        "control_semantics_changed": False,
    }
    write(FREEZE, freeze)
    print(json.dumps(amendment, indent=2))


if __name__ == "__main__":
    main()
