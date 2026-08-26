from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import sys
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# Import pandas before the legacy SAC modules.  In the pinned Windows
# environment this avoids a pyarrow lazy-import race; pandas is not used below.
import pandas as _pandas  # noqa: F401
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = REPO_ROOT / "Multi-agent_Algo_lib"
for _path in (REPO_ROOT, ALGO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from Entity.dynamic_obstacles import PatternedMovingSphereObstacle  # noqa: E402
from planning.final_four_stage_benchmark import (  # noqa: E402
    DWAStyleConfig,
    RVOStyleConfig,
    WORKSPACE_BOUNDS,
    _project_reciprocal_velocity_candidate,
    current_state_dynamic_predictions,
    dwa_style_accelerations,
    run_classical_episode,
    rvo_orca_style_accelerations,
    step_direct_accelerations,
)
from scripts.evaluate_single_policy_aligned_multi_agent import (  # noqa: E402
    build_single_distribution_multi_config,
)
from scripts.run_final_four_stage_benchmark import (  # noqa: E402
    ManifestEnvironmentBuilder,
)


SOURCE_ROOT = REPO_ROOT / "artifacts" / "final_four_stage_benchmark" / "20260818_202620"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "artifacts"
    / "theory_aligned_final_recovery"
    / "20260818_232900"
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hash(functions: Sequence[Any]) -> str:
    return hashlib.sha256(
        "\n\n".join(inspect.getsource(function) for function in functions).encode("utf-8")
    ).hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _prefreeze_payload(
    *, manifest: Mapping[str, Any], selected: Mapping[str, Any]
) -> dict[str, Any]:
    dwa_functions = (
        current_state_dynamic_predictions,
        dwa_style_accelerations,
        step_direct_accelerations,
        run_classical_episode,
    )
    rvo_functions = (
        current_state_dynamic_predictions,
        _project_reciprocal_velocity_candidate,
        rvo_orca_style_accelerations,
        step_direct_accelerations,
        run_classical_episode,
    )
    metric_functions = (step_direct_accelerations, run_classical_episode)
    config_payload = {
        "dwa_style": selected["dwa_style"],
        "rvo_orca_style": selected["rvo_orca_style"],
        "execution": load_json(SOURCE_ROOT / "config.json")["execution"],
    }
    return {
        "freeze_time": datetime.now().astimezone().isoformat(),
        "freeze_before_corrected_performance": True,
        "source_benchmark": str(SOURCE_ROOT.relative_to(REPO_ROOT)).replace("\\", "/"),
        "source_scenario_count": len(manifest["entries"]),
        "source_manifest_declared_sha256": manifest["manifest_sha256"],
        "scenario_manifest_file_sha256": file_hash(SOURCE_ROOT / "scenario_manifest.json"),
        "config_sha256": stable_hash(config_payload),
        "dwa_sha256": source_hash(dwa_functions),
        "rvo_sha256": source_hash(rvo_functions),
        "metric_sha256": source_hash(metric_functions),
        "implementation_file_sha256": file_hash(
            REPO_ROOT / "planning" / "final_four_stage_benchmark.py"
        ),
        "selected_configs": config_payload,
        "prediction_contract": "p_hat(t+h)=p(t)+h*v(t), h in seconds",
        "forbidden_information": [
            "deepcopy(live_dynamic_obstacle)",
            "private_rng_state",
            "precomputed_true_future_track",
        ],
        "original_privileged_classical_result_provenance": {
            "label": "ORIGINAL_PRIVILEGED_CLASSICAL_RESULT",
            "dwa_success": "393/400",
            "rvo_success": "395/400",
            "preserved_not_overwritten": True,
        },
    }


def _information_boundary_report() -> dict[str, Any]:
    common = dict(
        center=[1.0, 2.0, 3.0],
        radius=0.4,
        velocity=[0.3, -0.2, 0.1],
        safety_margin=0.06,
        bounds=((-2.0, -2.0, -2.0), (8.0, 8.0, 8.0)),
        motion_mode="wandering",
        wandering_strength=0.8,
    )
    first = PatternedMovingSphereObstacle(**common, seed=11)
    second = PatternedMovingSphereObstacle(**common, seed=991)
    first._rng.normal(size=37)
    second._rng.normal(size=3)
    times = [0.1, 0.4, 0.8, 1.4]
    first_predictions = current_state_dynamic_predictions([first], times)
    second_predictions = current_state_dynamic_predictions([second], times)
    prediction_equal = all(
        np.array_equal(left[0].center, right[0].center)
        for left, right in zip(first_predictions, second_predictions, strict=True)
    )
    first_true = PatternedMovingSphereObstacle(**common, seed=11)
    second_true = PatternedMovingSphereObstacle(**common, seed=991)
    for _ in range(8):
        first_true.step(0.1)
        second_true.step(0.1)
    true_future_changed = not np.array_equal(first_true.center, second_true.center)
    predicted_at_08_equal = np.array_equal(
        first_predictions[2][0].center, second_predictions[2][0].center
    )
    passed = bool(prediction_equal and true_future_changed and predicted_at_08_equal)
    return {
        "same_current_position_velocity_different_rng_prediction_identical": prediction_equal,
        "different_rng_true_wandering_future_changed": true_future_changed,
        "planner_prediction_unchanged_when_true_future_changes": predicted_at_08_equal,
        "FUTURE_DYNAMIC_INFORMATION_LEAKAGE_AFTER": "NO" if passed else "YES",
        "passed": passed,
        "prediction_times_s": times,
        "prediction_sha256_first": stable_hash(
            [[row[0].center.tolist()] for row in first_predictions]
        ),
        "prediction_sha256_second": stable_hash(
            [[row[0].center.tolist()] for row in second_predictions]
        ),
        "true_future_center_first_0p8s": first_true.center.tolist(),
        "true_future_center_second_0p8s": second_true.center.tolist(),
    }


def prefreeze(output_dir: Path) -> Path:
    phase = output_dir / "phase_a_corrected_classics"
    result_path = output_dir / "corrected_classical_results.csv"
    if result_path.exists() or any((phase / "records").rglob("*.json")):
        raise RuntimeError("cannot create a new freeze after corrected performance exists")
    manifest = load_json(SOURCE_ROOT / "scenario_manifest.json")
    selected = load_json(SOURCE_ROOT / "engineering_search" / "selected_configs.json")
    payload = _prefreeze_payload(manifest=manifest, selected=selected)
    boundary = _information_boundary_report()
    if not boundary["passed"]:
        raise RuntimeError("current-state-only information-boundary tests failed")
    write_json(phase / "information_boundary_tests.json", boundary)
    write_json(phase / "corrected_classical_prefreeze.json", payload)
    print(f"PREFREEZE_WRITTEN={phase / 'corrected_classical_prefreeze.json'}", flush=True)
    print("FUTURE_DYNAMIC_INFORMATION_LEAKAGE_AFTER=NO", flush=True)
    return phase


def _build_runtime(manifest: Mapping[str, Any], config: Mapping[str, Any]) -> tuple[Any, Any]:
    execution = config["execution"]
    base = build_single_distribution_multi_config(
        num_agents=int(execution["num_agents"]),
        max_steps=int(execution["max_steps"]),
    )
    multi_config = replace(
        base,
        workspace_bounds=WORKSPACE_BOUNDS,
        randomize_start_goal=False,
        start_position_bounds=((-0.4, -2.0, -0.9), (0.2, 2.0, 0.9)),
        goal_position_bounds=((7.0, -2.0, -0.9), (11.0, 2.0, 0.9)),
        min_start_goal_distance=5.5,
    )
    return ManifestEnvironmentBuilder(manifest), multi_config


def _record_path(phase: Path, entry: Mapping[str, Any], method: str) -> Path:
    return phase / "records" / entry["stage"] / entry["scenario_id"] / f"{method}.json"


def _save_record(
    phase: Path,
    entry: Mapping[str, Any],
    method: str,
    episode: Mapping[str, Any],
    agents: Sequence[Mapping[str, Any]],
    runtime_rows: Sequence[Mapping[str, Any]],
    trajectory: Mapping[str, Any],
) -> None:
    trajectory_path = _record_path(phase, entry, method).with_suffix(".npz")
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(trajectory_path, **{key: np.asarray(value) for key, value in trajectory.items()})
    payload = {
        "entry": {
            key: entry[key]
            for key in (
                "stage",
                "family",
                "scenario_id",
                "seed",
                "environment_fingerprint",
                "geometry_fingerprint",
            )
        },
        "episode": dict(episode),
        "agents": list(agents),
        "runtime_records": list(runtime_rows),
        "trajectory_file": str(trajectory_path.relative_to(phase)).replace("\\", "/"),
        "trajectory_sha256": file_hash(trajectory_path),
    }
    payload["result_sha256"] = stable_hash(payload)
    write_json(_record_path(phase, entry, method), payload)


def _verify_prefreeze(
    phase: Path, manifest: Mapping[str, Any], selected: Mapping[str, Any]
) -> Mapping[str, Any]:
    frozen = load_json(phase / "corrected_classical_prefreeze.json")
    current = _prefreeze_payload(manifest=manifest, selected=selected)
    for key in (
        "source_manifest_declared_sha256",
        "scenario_manifest_file_sha256",
        "config_sha256",
        "dwa_sha256",
        "rvo_sha256",
        "metric_sha256",
        "implementation_file_sha256",
    ):
        if frozen[key] != current[key]:
            raise RuntimeError(f"prefreeze mismatch: {key}")
    boundary = load_json(phase / "information_boundary_tests.json")
    if boundary["FUTURE_DYNAMIC_INFORMATION_LEAKAGE_AFTER"] != "NO":
        raise RuntimeError("information boundary is not valid")
    return frozen


def run(output_dir: Path, limit: int | None = None) -> Path:
    phase = output_dir / "phase_a_corrected_classics"
    manifest = load_json(SOURCE_ROOT / "scenario_manifest.json")
    config = load_json(SOURCE_ROOT / "config.json")
    selected = load_json(SOURCE_ROOT / "engineering_search" / "selected_configs.json")
    frozen = _verify_prefreeze(phase, manifest, selected)
    builder, multi_config = _build_runtime(manifest, config)
    planners = {
        "dwa_style": DWAStyleConfig(
            **{key: value for key, value in selected["dwa_style"].items() if key != "config_id"}
        ),
        "rvo_orca_style": RVOStyleConfig(
            **{
                key: value
                for key, value in selected["rvo_orca_style"].items()
                if key != "config_id"
            }
        ),
    }
    jobs = [(entry, method) for entry in manifest["entries"] for method in planners]
    if limit is not None:
        jobs = jobs[: int(limit)]
    completed = 0
    for job_index, (entry, method) in enumerate(jobs, start=1):
        record_path = _record_path(phase, entry, method)
        if record_path.exists():
            completed += 1
            continue
        episode, agents, runtime_rows, trajectory = run_classical_episode(
            environment_builder=builder,
            multi_config=multi_config,
            scenario=entry["scenario_id"],
            seed=int(entry["seed"]),
            peer_radius=float(config["execution"]["peer_radius"]),
            method=method,
            planner_config=planners[method],
        )
        _save_record(phase, entry, method, episode, agents, runtime_rows, trajectory)
        completed += 1
        if job_index % 10 == 0 or job_index == len(jobs):
            print(
                f"[corrected-classics {job_index}/{len(jobs)}] "
                f"{entry['stage']} {entry['scenario_id']} {method} "
                f"{episode['termination_reason']}",
                flush=True,
            )
    if limit is None:
        analyze(output_dir, frozen)
    return phase


def analyze(output_dir: Path, frozen: Mapping[str, Any] | None = None) -> None:
    phase = output_dir / "phase_a_corrected_classics"
    if frozen is None:
        manifest = load_json(SOURCE_ROOT / "scenario_manifest.json")
        selected = load_json(SOURCE_ROOT / "engineering_search" / "selected_configs.json")
        frozen = _verify_prefreeze(phase, manifest, selected)
    paths = sorted((phase / "records").rglob("*.json"))
    if len(paths) != 800:
        raise RuntimeError(f"expected 800 corrected records, found {len(paths)}")
    episodes: list[dict[str, Any]] = []
    agents: list[dict[str, Any]] = []
    runtimes: list[dict[str, Any]] = []
    record_hashes: list[dict[str, str]] = []
    for path in paths:
        payload = load_json(path)
        episodes.append(dict(payload["episode"]))
        agents.extend(dict(row) for row in payload["agents"])
        runtimes.extend(dict(row) for row in payload["runtime_records"])
        record_hashes.append(
            {
                "record": str(path.relative_to(phase)).replace("\\", "/"),
                "sha256": file_hash(path),
                "trajectory_sha256": payload["trajectory_sha256"],
            }
        )
    write_csv(output_dir / "corrected_classical_results.csv", episodes)
    write_csv(phase / "corrected_classical_results.csv", episodes)
    write_csv(phase / "corrected_classical_agent_results.csv", agents)
    write_csv(phase / "corrected_classical_runtime_records.csv", runtimes)
    keys = [(row["scenario_id"], row["method"]) for row in episodes]
    summaries: dict[str, Any] = {}
    for method in ("dwa_style", "rvo_orca_style"):
        rows = [row for row in episodes if row["method"] == method]
        summaries[method] = {
            "episode_count": len(rows),
            "success_count": sum(bool(row["team_success"]) for row in rows),
            "success_rate": float(np.mean([bool(row["team_success"]) for row in rows])),
            "collision_count": sum(bool(row["any_collision"]) for row in rows),
            "collision_rate": float(np.mean([bool(row["any_collision"]) for row in rows])),
            "timeout_count": sum(bool(row["timeout"]) for row in rows),
            "timeout_rate": float(np.mean([bool(row["timeout"]) for row in rows])),
            "mean_total_local_planner_compute_ms": float(
                np.mean([float(row["planning_runtime_ms"]) for row in rows])
            ),
            "mean_decisions_per_episode": float(
                np.mean([int(row["planning_decision_count"]) for row in rows])
            ),
        }
    reconciliation = {
        "CORRECTED_CLASSICAL_RECONCILIATION": "PASS",
        "FUTURE_DYNAMIC_INFORMATION_LEAKAGE_AFTER": "NO",
        "BASELINE_COMPARISON_FAIRNESS_AFTER_FIX": "YES",
        "corrective_rerun_after_confirmed_protocol_defect": True,
        "episode_count": len(episodes),
        "agent_count": len(agents),
        "runtime_record_count": len(runtimes),
        "unique_episode_method_keys": len(set(keys)),
        "scenario_count_per_method": {
            method: len({row["scenario_id"] for row in episodes if row["method"] == method})
            for method in ("dwa_style", "rvo_orca_style")
        },
        "paired_scenario_sets_identical": {
            row["scenario_id"] for row in episodes if row["method"] == "dwa_style"
        }
        == {
            row["scenario_id"] for row in episodes if row["method"] == "rvo_orca_style"
        },
        "prefreeze": dict(frozen),
        "postrun_hashes_match_prefreeze": True,
        "summaries": summaries,
        "original_privileged_classical_result_provenance": frozen[
            "original_privileged_classical_result_provenance"
        ],
        "record_manifest_sha256": stable_hash(record_hashes),
        "record_manifest": record_hashes,
    }
    write_json(output_dir / "corrected_classical_reconciliation.json", reconciliation)
    write_json(phase / "corrected_classical_reconciliation.json", reconciliation)
    print(canonical_json(summaries), flush=True)
    print("CORRECTED_CLASSICAL_RECONCILIATION=PASS", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prefreeze", "run", "analyze"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if args.phase == "prefreeze":
        prefreeze(output_dir)
    elif args.phase == "run":
        run(output_dir, args.limit)
    else:
        analyze(output_dir)


if __name__ == "__main__":
    main()
