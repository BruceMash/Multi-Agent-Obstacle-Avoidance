"""Independent raw-record reconciliation for the long-range formal benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ALGO = Path(__file__).resolve().parents[1]
for search_path in (ROOT, ALGO):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from planning.long_range_collision_recheck import audit_trajectory_collisions  # noqa: E402
from scripts.run_long_range_formal_benchmark import (  # noqa: E402
    ARTIFACT_ROOT,
    EXPECTED_SCENARIOS,
    FORMAL_MANIFEST,
    FORMAL_RUN_FREEZE,
    METHOD_ORDER,
    RECORD_DIR,
    SCHEMA,
    content_hash,
    load_json,
    sha256_file,
    verify_formal_freeze,
    write_csv,
)


STATISTICS_DIR = ARTIFACT_ROOT / "13_statistics"
FAILURE_DIR = ARTIFACT_ROOT / "14_failure_analysis"


def array_content_hash(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(arrays):
        array = np.ascontiguousarray(np.asarray(arrays[key]))
        digest.update(key.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def reconcile() -> dict[str, Any]:
    freeze = verify_formal_freeze()
    manifest = load_json(FORMAL_MANIFEST)
    entries = {str(row["scenario_id"]): row for row in manifest["entries"]}
    expected_keys = {
        (scenario_id, method_id)
        for scenario_id in entries
        for method_id in METHOD_ORDER
    }
    team_csv = read_csv(RECORD_DIR / "formal_team_results.csv")
    agent_csv = read_csv(RECORD_DIR / "formal_agent_results.csv")
    observed_team_keys = {(row["scenario_id"], row["method_id"]) for row in team_csv}
    observed_agent_keys = {
        (row["scenario_id"], row["method_id"], int(row["agent_id"]))
        for row in agent_csv
    }

    failures: dict[str, list[Any]] = {
        "missing_records": [],
        "software_errors": [],
        "result_hash": [],
        "trajectory_file_hash": [],
        "trajectory_content_hash": [],
        "metadata": [],
        "trajectory_shape": [],
        "nonfinite": [],
        "initial_state": [],
        "collision_replay": [],
    }
    collision_rows: list[dict[str, Any]] = []
    record_count = 0
    agent_count = 0
    for scenario_id, method_id in sorted(expected_keys):
        directory = RECORD_DIR / method_id
        record_path = directory / f"{scenario_id}.json"
        trajectory_path = directory / f"{scenario_id}_trajectory.npz"
        error_path = directory / f"{scenario_id}_SOFTWARE_ERROR.json"
        if error_path.is_file():
            failures["software_errors"].append(f"{scenario_id}:{method_id}")
        if not record_path.is_file() or not trajectory_path.is_file():
            failures["missing_records"].append(f"{scenario_id}:{method_id}")
            continue
        record = load_json(record_path)
        record_count += 1
        agent_count += len(record.get("agents", []))
        raw_without_hash = {key: value for key, value in record.items() if key != "result_hash"}
        if content_hash(raw_without_hash) != record.get("result_hash"):
            failures["result_hash"].append(f"{scenario_id}:{method_id}")
        if sha256_file(trajectory_path) != record.get("trajectory_file_sha256"):
            failures["trajectory_file_hash"].append(f"{scenario_id}:{method_id}")
        with np.load(trajectory_path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        if array_content_hash(arrays) != record.get("trajectory_content_hash"):
            failures["trajectory_content_hash"].append(f"{scenario_id}:{method_id}")
        entry = entries[scenario_id]
        metadata_ok = all(
            (
                record.get("schema_version") == SCHEMA,
                record.get("method_id") == method_id,
                record.get("scenario_id") == scenario_id,
                record.get("scenario_environment_fingerprint") == entry["environment_fingerprint"],
                record.get("scenario_geometry_fingerprint") == entry["geometry_fingerprint"],
                record.get("manifest_semantic_sha256") == freeze["manifest_semantic_sha256"],
                record.get("method_config_sha256") == freeze["method_config_sha256"][method_id],
                record.get("checkpoint_sha256") == freeze["checkpoint_sha256"],
                len(record.get("agents", [])) == 3,
            )
        )
        if not metadata_ok:
            failures["metadata"].append(f"{scenario_id}:{method_id}")
        positions = np.asarray(arrays.get("positions"), dtype=float)
        episode = record["episode"]
        if positions.ndim != 3 or positions.shape[1:] != (3, 3) or len(positions) != int(episode["steps"]) + 1:
            failures["trajectory_shape"].append(
                {"key": f"{scenario_id}:{method_id}", "shape": positions.shape, "steps": episode["steps"]}
            )
            continue
        if not all(np.all(np.isfinite(np.asarray(value))) for value in arrays.values()):
            failures["nonfinite"].append(f"{scenario_id}:{method_id}")
        if not np.allclose(
            positions[0], np.asarray(entry["starts"], dtype=float), rtol=0.0, atol=1.0e-7
        ):
            failures["initial_state"].append(f"{scenario_id}:{method_id}")
        replay = audit_trajectory_collisions(positions, entry)
        keys = (
            "static_obstacle_collision",
            "dynamic_obstacle_collision",
            "obstacle_collision",
            "inter_agent_collision",
            "boundary_collision",
            "any_collision",
        )
        mismatch = {
            key: bool(episode.get(key, False)) != bool(replay[key]) for key in keys
        }
        if any(mismatch.values()):
            failures["collision_replay"].append(
                {"key": f"{scenario_id}:{method_id}", "mismatch": mismatch}
            )
        collision_rows.append(
            {
                "scenario_id": scenario_id,
                "method_id": method_id,
                **{key: replay[key] for key in replay if not key.startswith("agent_")},
                "collision_labels_exact": not any(mismatch.values()),
            }
        )

    expected_agent_keys = {
        (scenario_id, method_id, agent_id)
        for scenario_id, method_id in expected_keys
        for agent_id in range(3)
    }
    row_count_checks = {
        "manifest_scenarios": len(entries) == EXPECTED_SCENARIOS,
        "record_count": record_count == EXPECTED_SCENARIOS * len(METHOD_ORDER),
        "record_agent_count": agent_count == EXPECTED_SCENARIOS * len(METHOD_ORDER) * 3,
        "team_csv_count": len(team_csv) == EXPECTED_SCENARIOS * len(METHOD_ORDER),
        "team_csv_keys": observed_team_keys == expected_keys,
        "agent_csv_count": len(agent_csv) == EXPECTED_SCENARIOS * len(METHOD_ORDER) * 3,
        "agent_csv_keys": observed_agent_keys == expected_agent_keys,
        "method_balance": set(Counter(row["method_id"] for row in team_csv).values()) == {EXPECTED_SCENARIOS},
    }
    failure_counts = {key: len(value) for key, value in failures.items()}
    passed = all(row_count_checks.values()) and not any(failure_counts.values())
    result = {
        "schema_version": SCHEMA,
        "FINAL_RECONCILIATION": "PASS" if passed else "FAIL",
        "FINAL_INFORMATION_INTEGRITY": "PASS" if passed else "FAIL",
        "row_count_checks": row_count_checks,
        "failure_counts": failure_counts,
        "failure_details": failures,
        "source_and_freeze_hashes_verified": True,
        "all_trajectory_files_reopened": True,
        "all_collision_labels_independently_recomputed": True,
        "formal_performance_used_during_reconciliation": False,
    }
    STATISTICS_DIR.mkdir(parents=True, exist_ok=True)
    FAILURE_DIR.mkdir(parents=True, exist_ok=True)
    with (STATISTICS_DIR / "final_reconciliation.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    write_csv(FAILURE_DIR / "collision_recheck.csv", collision_rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    result = reconcile()
    print(json.dumps({"reconciliation": result["FINAL_RECONCILIATION"], "failure_counts": result["failure_counts"]}), flush=True)
    if result["FINAL_RECONCILIATION"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
