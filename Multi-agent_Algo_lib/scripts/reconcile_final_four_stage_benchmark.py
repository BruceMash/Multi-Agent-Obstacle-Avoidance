"""Independent integrity reconciliation for the final four-stage benchmark.

This script intentionally does not import benchmark or evaluator modules.  It
recomputes hashes and cross-method scenario identity directly from persisted
JSON/NPZ artifacts so that the audit is independent of the execution path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np


METHODS = (
    "dwa_style",
    "rvo_orca_style",
    "terminal",
    "proposal",
    "fp_shep",
    "gat_v1",
)
STAGES = ("stage_1", "stage_2", "stage_3", "stage_4")


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def stable_hash(value: Any) -> str:
    raw = json.dumps(
        json_ready(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    workspace = Path(__file__).resolve().parents[2]

    manifest = load_json(out / "scenario_manifest.json")
    entries = manifest["entries"]
    by_scenario = {entry["scenario_id"]: entry for entry in entries}
    expected_keys = {
        (stage, f"S{stage[-1]}_{index:03d}", method)
        for stage in STAGES
        for index in range(100)
        for method in METHODS
    }

    paths = sorted((out / "formal_records").glob("stage_*/*/*.json"))
    observed_keys: list[tuple[str, str, str]] = []
    agent_keys: list[tuple[str, str, str, int]] = []
    failures: list[str] = []
    result_hash_failures: list[str] = []
    trajectory_hash_failures: list[str] = []
    scenario_state_failures: list[str] = []
    dynamic_trajectory_failures: list[str] = []
    status_counts: Counter[str] = Counter()
    per_scenario_methods: dict[str, set[str]] = defaultdict(set)
    stored_initial_hashes: dict[str, set[str]] = defaultdict(set)

    for path in paths:
        record = load_json(path)
        key = (record.get("stage"), record.get("scenario_id"), record.get("method"))
        observed_keys.append(key)
        stage, scenario_id, method = key
        status_counts[str(record.get("status"))] += 1
        per_scenario_methods[scenario_id].add(method)

        payload = dict(record)
        observed_result_hash = payload.pop("result_hash", None)
        if observed_result_hash != stable_hash(payload):
            result_hash_failures.append(str(path.relative_to(out)))

        trajectory_path = out / Path(record["trajectory_path"])
        if not trajectory_path.is_file():
            trajectory_hash_failures.append(f"missing:{trajectory_path.relative_to(out)}")
            continue
        if file_sha256(trajectory_path) != record.get("trajectory_sha256"):
            trajectory_hash_failures.append(str(trajectory_path.relative_to(out)))

        scene = by_scenario[scenario_id]
        stored_initial_hashes[scenario_id].add(
            str(record.get("episode", {}).get("initial_condition_hash"))
        )
        try:
            with np.load(trajectory_path, allow_pickle=False) as trajectory:
                starts = np.asarray(trajectory["starts"], dtype=np.float64)
                goals = np.asarray(trajectory["goals"], dtype=np.float64)
                positions = np.asarray(trajectory["positions"], dtype=np.float64)
                dynamic = np.asarray(
                    trajectory["dynamic_obstacle_positions"], dtype=np.float64
                )
                if not np.allclose(starts, scene["starts"], rtol=0.0, atol=1e-12):
                    scenario_state_failures.append(f"{scenario_id}/{method}:starts")
                if not np.allclose(goals, scene["goals"], rtol=0.0, atol=1e-12):
                    scenario_state_failures.append(f"{scenario_id}/{method}:goals")
                if positions.shape[0] < 1 or not np.allclose(
                    positions[0], starts, rtol=0.0, atol=1e-12
                ):
                    scenario_state_failures.append(f"{scenario_id}/{method}:initial_positions")
                frozen_dynamic = np.asarray(
                    scene["dynamic_obstacle_trajectories"], dtype=np.float64
                )
                if frozen_dynamic.size == 0:
                    frozen_dynamic = np.empty((0, 0, 3), dtype=np.float64)
                elif frozen_dynamic.ndim == 3:
                    frozen_dynamic = np.transpose(frozen_dynamic, (1, 0, 2))
                n_steps = min(dynamic.shape[0], frozen_dynamic.shape[0])
                if dynamic.shape[1] != len(scene["dynamic_obstacles"]):
                    dynamic_trajectory_failures.append(
                        f"{scenario_id}/{method}:dynamic_count"
                    )
                elif n_steps and not np.allclose(
                    dynamic[:n_steps], frozen_dynamic[:n_steps], rtol=0.0, atol=1e-10
                ):
                    dynamic_trajectory_failures.append(
                        f"{scenario_id}/{method}:dynamic_positions"
                    )
        except Exception as exc:  # pragma: no cover - audit trail only
            failures.append(f"{scenario_id}/{method}:npz:{type(exc).__name__}:{exc}")

        for agent in record.get("agents", []):
            agent_keys.append((stage, scenario_id, method, int(agent["agent_id"])))

    observed_set = set(observed_keys)
    duplicate_record_count = len(observed_keys) - len(observed_set)
    duplicate_agent_count = len(agent_keys) - len(set(agent_keys))
    missing_keys = sorted(expected_keys - observed_set)
    unexpected_keys = sorted(observed_set - expected_keys)

    core_hash_failures: list[dict[str, str]] = []
    phase = load_json(out / "formal_phase_state.json")
    for relative, expected_hash in phase["core_hashes"].items():
        current_hash = file_sha256(workspace / relative)
        if current_hash != expected_hash:
            core_hash_failures.append(
                {"path": relative, "expected": expected_hash, "actual": current_hash}
            )

    checkpoint_paths = {
        "gat_checkpoint": workspace
        / "artifacts/gat_stage1_training/20260815_230509/checkpoints/best_validation.pt",
        "sac_checkpoint": workspace / "artifacts/20260520_201912/best_eval_model.pt",
    }
    expected_checkpoint_hashes = load_json(out / "preformal_integrity.json")[
        "checkpoint_hashes"
    ]
    checkpoint_hashes = {key: file_sha256(path) for key, path in checkpoint_paths.items()}
    checkpoint_hash_failures = {
        key: {"expected": expected_checkpoint_hashes[key], "actual": digest}
        for key, digest in checkpoint_hashes.items()
        if digest != expected_checkpoint_hashes[key]
    }

    manifest_payload = dict(manifest)
    embedded_manifest_hash = manifest_payload.pop("manifest_sha256")
    recomputed_manifest_hash = stable_hash(manifest_payload)
    manifest_hash_valid = (
        embedded_manifest_hash
        == recomputed_manifest_hash
        == phase["formal_manifest_sha256"]
    )

    method_coverage_failures = {
        scenario_id: sorted(set(METHODS) - methods)
        for scenario_id, methods in per_scenario_methods.items()
        if methods != set(METHODS)
    }
    state_identity_valid = not scenario_state_failures and not dynamic_trajectory_failures
    # The classical and SAC evaluators serialize different internal hash payloads.
    # Raw persisted starts, goals, initial positions, and frozen obstacle tracks are
    # therefore the authoritative cross-method identity check.
    stored_hash_cardinality = Counter(len(v) for v in stored_initial_hashes.values())

    checks = {
        "team_record_count_exact": len(paths) == 2400,
        "team_keys_unique": duplicate_record_count == 0,
        "team_keys_complete": not missing_keys and not unexpected_keys,
        "agent_record_count_exact": len(agent_keys) == 7200,
        "agent_keys_unique": duplicate_agent_count == 0,
        "all_status_complete": status_counts == Counter({"COMPLETE": 2400}),
        "result_hashes_valid": not result_hash_failures,
        "trajectory_files_and_hashes_valid": not trajectory_hash_failures,
        "raw_scenario_state_identical_across_methods": state_identity_valid,
        "all_scenarios_have_all_methods": not method_coverage_failures,
        "manifest_hash_valid": manifest_hash_valid,
        "core_hashes_unchanged": not core_hash_failures,
        "checkpoint_hashes_unchanged": not checkpoint_hash_failures,
        "no_reconciliation_exceptions": not failures,
    }
    audit = {
        "schema_version": "final_four_stage_reconciliation_v1",
        "status": "PASSED" if all(checks.values()) else "FAILED",
        "checks": checks,
        "counts": {
            "team_records": len(paths),
            "unique_team_keys": len(observed_set),
            "agent_records": len(agent_keys),
            "unique_agent_keys": len(set(agent_keys)),
            "scenario_count": len(per_scenario_methods),
            "methods_per_scenario": dict(
                sorted(Counter(len(v) for v in per_scenario_methods.values()).items())
            ),
            "stored_initial_hash_cardinality_per_scenario": dict(
                sorted(stored_hash_cardinality.items())
            ),
        },
        "identity_note": (
            "Cross-method fairness is verified from raw NPZ starts/goals/initial "
            "positions and frozen dynamic tracks. Stored evaluator hashes are not "
            "compared across evaluator families because their payload schemas differ."
        ),
        "manifest_hash": recomputed_manifest_hash,
        "checkpoint_hashes": checkpoint_hashes,
        "failure_counts": {
            "missing_team_keys": len(missing_keys),
            "unexpected_team_keys": len(unexpected_keys),
            "duplicate_team_keys": duplicate_record_count,
            "duplicate_agent_keys": duplicate_agent_count,
            "result_hash_failures": len(result_hash_failures),
            "trajectory_hash_failures": len(trajectory_hash_failures),
            "scenario_state_failures": len(scenario_state_failures),
            "dynamic_trajectory_failures": len(dynamic_trajectory_failures),
            "method_coverage_failures": len(method_coverage_failures),
            "core_hash_failures": len(core_hash_failures),
            "checkpoint_hash_failures": len(checkpoint_hash_failures),
            "exceptions": len(failures),
        },
        "failure_examples": {
            "missing_team_keys": missing_keys[:10],
            "unexpected_team_keys": unexpected_keys[:10],
            "result_hash_failures": result_hash_failures[:10],
            "trajectory_hash_failures": trajectory_hash_failures[:10],
            "scenario_state_failures": scenario_state_failures[:10],
            "dynamic_trajectory_failures": dynamic_trajectory_failures[:10],
            "method_coverage_failures": dict(list(method_coverage_failures.items())[:10]),
            "core_hash_failures": core_hash_failures[:10],
            "checkpoint_hash_failures": checkpoint_hash_failures,
            "exceptions": failures[:10],
        },
    }
    destination = out / "independent_reconciliation.json"
    destination.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": audit["status"], **audit["counts"]}, indent=2))
    if audit["status"] != "PASSED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
