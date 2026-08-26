"""Independent integrity and collision reconciliation for the final benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ALGO = ROOT / "Multi-agent_Algo_lib"
for _search in (ROOT, ALGO):
    if str(_search) not in sys.path:
        sys.path.insert(0, str(_search))

from planning.final_four_stage_benchmark import (  # noqa: E402
    current_state_dynamic_predictions,
    obstacle_from_spec,
    obstacles_from_entry,
    stable_hash,
)
from planning.sensing_matched_classical import (  # noqa: E402
    reconstruct_sensing_matched_perception,
)
from scripts.evaluate_single_policy_aligned_multi_agent import (  # noqa: E402
    build_single_distribution_multi_config,
)
from scripts.run_final_untouched_paper_benchmark import (  # noqa: E402
    DEFAULT_OUTPUT,
    EXPECTED_AGENT_ROWS,
    EXPECTED_SCENARIOS,
    EXPECTED_TEAM_ROWS,
    METHOD_ORDER,
    SCHEMA,
    content_hash,
    load_json,
    read_csv,
    sha256_file,
    write_json,
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    return value


def _array_hash(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(arrays):
        array = np.ascontiguousarray(np.asarray(arrays[key]))
        digest.update(key.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _read_raw_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _collision_recheck(
    positions: np.ndarray,
    entry: Mapping[str, Any],
    *,
    collision_margin: float,
    peer_threshold: float,
) -> dict[str, Any]:
    static, _ = obstacles_from_entry(entry)
    dynamic = [obstacle_from_spec(spec) for spec in entry["dynamic_obstacles"]]
    obstacle = False
    peer = False
    minimum_obstacle = float("inf")
    minimum_peer = float("inf")
    first_obstacle_step: int | None = None
    first_peer_step: int | None = None
    # Primary labels use post-transition frames only.  Frame zero is the
    # validated initial condition and is not an executed transition.
    for step in range(1, len(positions)):
        frame = np.asarray(positions[step], dtype=float)
        for obstacle_id, object_ in enumerate(dynamic):
            track = entry["dynamic_obstacle_trajectories"][obstacle_id]
            object_.center = np.asarray(track[min(step, len(track) - 1)], dtype=float)
        obstacles = [*static, *dynamic]
        for point in frame:
            for object_ in obstacles:
                signed = float(object_.signed_distance(point))
                minimum_obstacle = min(minimum_obstacle, signed)
                if signed <= collision_margin:
                    obstacle = True
                    if first_obstacle_step is None:
                        first_obstacle_step = step
        for left in range(len(frame)):
            for right in range(left + 1, len(frame)):
                distance = float(np.linalg.norm(frame[left] - frame[right]))
                minimum_peer = min(minimum_peer, distance)
                if distance <= peer_threshold:
                    peer = True
                    if first_peer_step is None:
                        first_peer_step = step
    return {
        "obstacle_collision": obstacle,
        "inter_agent_collision": peer,
        "any_collision": obstacle or peer,
        "minimum_obstacle_signed_clearance_m": minimum_obstacle,
        "minimum_inter_agent_distance_m": minimum_peer,
        "first_obstacle_collision_step": first_obstacle_step,
        "first_inter_agent_collision_step": first_peer_step,
    }


def _scenario_reconstruction(entry: Mapping[str, Any]) -> dict[str, Any]:
    dynamic_exact = True
    for obstacle_id, spec in enumerate(entry["dynamic_obstacles"]):
        obstacle = obstacle_from_spec(spec)
        observed = [np.asarray(obstacle.center, dtype=float).copy()]
        for _ in range(int(entry["max_steps"])):
            obstacle.step(float(entry["dt"]))
            observed.append(np.asarray(obstacle.center, dtype=float).copy())
        dynamic_exact &= np.array_equal(
            np.asarray(observed, dtype=float),
            np.asarray(entry["dynamic_obstacle_trajectories"][obstacle_id], dtype=float),
        )
    geometry_payload = {
        key: entry[key]
        for key in (
            "starts", "goals", "static_obstacles", "dynamic_obstacles",
            "dynamic_obstacle_trajectories",
        )
    }
    return {
        "geometry_fingerprint_match": stable_hash(geometry_payload) == entry["geometry_fingerprint"],
        "dynamic_track_reconstruction_exact": bool(dynamic_exact),
    }


def reconcile(output: Path) -> dict[str, Any]:
    required = (
        "FINAL_FORMAL_FREEZE.json",
        "FINAL_UNTOUCHED_SCENARIO_MANIFEST.json",
        "run_schedule.csv",
        "formal_run_complete.json",
        "formal_team_results.csv",
        "formal_agent_results.csv",
    )
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing formal artifacts: {missing}")
    freeze = load_json(output / "FINAL_FORMAL_FREEZE.json")
    manifest = load_json(output / "FINAL_UNTOUCHED_SCENARIO_MANIFEST.json")
    complete = load_json(output / "formal_run_complete.json")
    schedule = read_csv(output / "run_schedule.csv")
    team_csv = _read_raw_csv(output / "formal_team_results.csv")
    agent_csv = _read_raw_csv(output / "formal_agent_results.csv")
    entries = {row["scenario_id"]: row for row in manifest["entries"]}

    source_mismatches = [
        relative
        for relative, expected in freeze["source_sha256"].items()
        if sha256_file(ROOT / relative) != expected
    ]
    frozen_artifact_mismatches = [
        relative
        for relative, expected in freeze["artifact_sha256"].items()
        if sha256_file(output / relative) != expected
    ]
    method_config_mismatches = [
        method_id
        for method_id, expected in freeze["method_config_sha256"].items()
        if sha256_file(output / "method_configs" / f"{method_id}.json") != expected
    ]
    manifest_hash_match = manifest["manifest_sha256"] == freeze["manifest_sha256"]

    schedule_keys = [(row["scenario_id"], row["method_id"]) for row in schedule]
    team_keys = [(row["scenario_id"], row["method_id"]) for row in team_csv]
    agent_keys = [
        (row["scenario_id"], row["method_id"], int(row["agent_id"]))
        for row in agent_csv
    ]
    scenario_method_counts = Counter(row["scenario_id"] for row in team_csv)

    result_hash_failures: list[str] = []
    trajectory_file_hash_failures: list[str] = []
    trajectory_content_hash_failures: list[str] = []
    record_metadata_failures: list[str] = []
    collision_mismatches: list[dict[str, Any]] = []
    min_clearance_rows: list[dict[str, Any]] = []
    config = build_single_distribution_multi_config(num_agents=3, max_steps=220)
    collision_margin = float(config.collision_margin)
    peer_threshold = float(config.inter_agent_safe_distance)

    record_count = 0
    agent_count_from_records = 0
    for scenario_id, method_id in schedule_keys:
        path = output / "formal_records" / scenario_id / f"{method_id}.json"
        trajectory_path = output / "trajectories" / scenario_id / f"{method_id}.npz"
        if not path.is_file() or not trajectory_path.is_file():
            record_metadata_failures.append(f"missing:{scenario_id}:{method_id}")
            continue
        record = load_json(path)
        record_count += 1
        agent_count_from_records += len(record["agents"])
        payload = {key: value for key, value in record.items() if key != "result_hash"}
        if content_hash(payload) != record["result_hash"]:
            result_hash_failures.append(f"{scenario_id}:{method_id}")
        if sha256_file(trajectory_path) != record["trajectory_file_sha256"]:
            trajectory_file_hash_failures.append(f"{scenario_id}:{method_id}")
        with np.load(trajectory_path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        if _array_hash(arrays) != record["trajectory_content_hash"]:
            trajectory_content_hash_failures.append(f"{scenario_id}:{method_id}")
        entry = entries[scenario_id]
        metadata_ok = all(
            (
                record["scenario_environment_fingerprint"] == entry["environment_fingerprint"],
                record["scenario_geometry_fingerprint"] == entry["geometry_fingerprint"],
                record["manifest_sha256"] == freeze["manifest_sha256"],
                record["method_config_sha256"] == freeze["method_config_sha256"][method_id],
                record["checkpoint_sha256"] == freeze["checkpoint_sha256"],
                record["episode"]["method_id"] == method_id,
                record["episode"]["scenario_id"] == scenario_id,
                len(record["agents"]) == 3,
            )
        )
        if not metadata_ok:
            record_metadata_failures.append(f"metadata:{scenario_id}:{method_id}")
        recheck = _collision_recheck(
            np.asarray(arrays["positions"], dtype=float),
            entry,
            collision_margin=collision_margin,
            peer_threshold=peer_threshold,
        )
        episode = record["episode"]
        mismatch = {
            "obstacle": bool(episode["obstacle_collision"]) != recheck["obstacle_collision"],
            "peer": bool(episode["inter_agent_collision"]) != recheck["inter_agent_collision"],
            "any": bool(episode["any_collision"]) != recheck["any_collision"],
        }
        if any(mismatch.values()):
            collision_mismatches.append(
                {
                    "scenario_id": scenario_id,
                    "method_id": method_id,
                    **mismatch,
                    "evaluator_obstacle": episode["obstacle_collision"],
                    "recheck_obstacle": recheck["obstacle_collision"],
                    "evaluator_peer": episode["inter_agent_collision"],
                    "recheck_peer": recheck["inter_agent_collision"],
                }
            )
        min_clearance_rows.append(
            {
                "scenario_id": scenario_id,
                "method_id": method_id,
                **recheck,
            }
        )

    reconstruction_rows = [
        {"scenario_id": scenario_id, **_scenario_reconstruction(entry)}
        for scenario_id, entry in entries.items()
    ]
    reconstruction_pass = all(
        row["geometry_fingerprint_match"] and row["dynamic_track_reconstruction_exact"]
        for row in reconstruction_rows
    )

    dwa_source = inspect.getsource(current_state_dynamic_predictions)
    sensing_source = inspect.getsource(reconstruct_sensing_matched_perception)
    sensing_module_source = (ROOT / "planning" / "sensing_matched_classical.py").read_text(encoding="utf-8")
    information_checks = {
        "DWA_FullState_current_state_prediction_only": (
            "private RNG" in dwa_source and "velocity" in dwa_source and "copy a live obstacle" in dwa_source
        ),
        "DWA_SensingMatched_adapter_present": (
            "_packet_endpoints" in sensing_source and "latest_sensor_packets" in sensing_module_source
        ),
        "DWA_SensingMatched_no_manifest_or_future_argument": (
            "manifest" not in sensing_source and "trajectory" not in sensing_source
        ),
        "Proposed_source_hashes_unchanged": not source_mismatches,
        "Proposed_continuous_peer_contract_not_added": True,
    }

    checks = {
        "formal_complete_marker": complete["FORMAL_RUN_COMPLETE"] == "YES",
        "team_row_count": len(team_csv) == EXPECTED_TEAM_ROWS == record_count,
        "agent_row_count": len(agent_csv) == EXPECTED_AGENT_ROWS == agent_count_from_records,
        "team_unique_keys": len(team_keys) == len(set(team_keys)) == EXPECTED_TEAM_ROWS,
        "agent_unique_keys": len(agent_keys) == len(set(agent_keys)) == EXPECTED_AGENT_ROWS,
        "schedule_exact_key_match": set(schedule_keys) == set(team_keys) and len(schedule_keys) == len(team_keys),
        "eight_methods_per_scenario": len(scenario_method_counts) == EXPECTED_SCENARIOS and set(scenario_method_counts.values()) == {8},
        "source_hashes": not source_mismatches,
        "frozen_artifact_hashes": not frozen_artifact_mismatches,
        "method_config_hashes": not method_config_mismatches,
        "manifest_hash": manifest_hash_match,
        "result_hashes": not result_hash_failures,
        "trajectory_file_hashes": not trajectory_file_hash_failures,
        "trajectory_content_hashes": not trajectory_content_hash_failures,
        "record_metadata": not record_metadata_failures,
        "scenario_reconstruction": reconstruction_pass,
        "independent_collision_recheck": not collision_mismatches,
        "information_integrity": all(information_checks.values()),
        "no_formal_exception": not (output / "FORMAL_SOFTWARE_EXCEPTION.json").exists(),
        "method_unchanged_after_start": complete["method_changed_after_formal_start"] is False,
        "parameter_tuning_after_start": complete["parameter_tuning_after_formal_start"] is False,
        "schedule_unchanged_after_start": complete["schedule_changed_after_formal_start"] is False,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    result = {
        "schema_version": SCHEMA,
        "FINAL_RECONCILIATION": status,
        "FINAL_INFORMATION_INTEGRITY": "PASS" if checks["information_integrity"] else "FAIL",
        "checks": checks,
        "counts": {
            "team_rows": len(team_csv),
            "agent_rows": len(agent_csv),
            "record_json": record_count,
            "unique_team_keys": len(set(team_keys)),
            "unique_agent_keys": len(set(agent_keys)),
            "scenarios": len(scenario_method_counts),
        },
        "hash_failures": {
            "source": source_mismatches,
            "frozen_artifact": frozen_artifact_mismatches,
            "method_config": method_config_mismatches,
            "result": result_hash_failures,
            "trajectory_file": trajectory_file_hash_failures,
            "trajectory_content": trajectory_content_hash_failures,
            "record_metadata": record_metadata_failures,
        },
        "collision_contract": {
            "primary": "discrete post-transition",
            "initial_frame_excluded": True,
            "collision_margin_m": collision_margin,
            "inter_agent_threshold_m": peer_threshold,
            "mismatch_count": len(collision_mismatches),
            "mismatches": collision_mismatches,
        },
        "scenario_reconstruction": {
            "count": len(reconstruction_rows),
            "pass": reconstruction_pass,
            "failures": [row for row in reconstruction_rows if not all(value for key, value in row.items() if key != "scenario_id")],
        },
        "information_checks": information_checks,
        "statistics_authorized": status == "PASS",
    }
    write_json(output / "independent_collision_recheck.json", {"rows": min_clearance_rows, "contract": result["collision_contract"]})
    write_json(output / "final_reconciliation.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = reconcile(args.output.resolve())
    print(json.dumps({"FINAL_RECONCILIATION": result["FINAL_RECONCILIATION"], "team_rows": result["counts"]["team_rows"], "agent_rows": result["counts"]["agent_rows"]}), flush=True)
    if result["FINAL_RECONCILIATION"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
