from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

# Stabilize the pinned Windows import stack before legacy SAC imports.
import pandas as _pandas  # noqa: F401
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = REPO_ROOT / "Multi-agent_Algo_lib"
for _path in (REPO_ROOT, ALGO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from planning.online_runtime_instrumentation import (  # noqa: E402
    OnlineRuntimeRecorder,
    TimedPolicyProxy,
)
from planning.pre_gat_220step_revalidation import stable_hash  # noqa: E402
from scripts.evaluate_gat_closed_loop import (  # noqa: E402
    METHOD_FP_SHEP,
    METHOD_GAT,
    METHOD_PROPOSAL,
    METHOD_TERMINAL,
    build_shared_selection_bundle,
    run_method_episode,
)
from scripts.run_final_four_stage_benchmark import (  # noqa: E402
    FrozenRuntime,
    proposed_eval_config,
)


SOURCE_ROOT = REPO_ROOT / "artifacts" / "final_four_stage_benchmark" / "20260818_202620"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "artifacts"
    / "theory_aligned_final_recovery"
    / "20260818_232900"
)
METHODS = {
    "terminal": METHOD_TERMINAL,
    "proposal": METHOD_PROPOSAL,
    "fp_shep": METHOD_FP_SHEP,
    "gat_v1_one_shot": METHOD_GAT,
}
PREFREEZE_NAME = "runtime_prefreeze_v2.json"
RECORD_DIRECTORY = "records_v2"
ORIGINAL_NAMES = {
    "terminal": "terminal",
    "proposal": "proposal",
    "fp_shep": "fp_shep",
    "gat_v1_one_shot": "gat_v1",
}


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


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _prefreeze_payload() -> dict[str, Any]:
    files = (
        "Guidance/reference_point_proposal_demo.py",
        "planning/online_runtime_instrumentation.py",
        "planning/pre_gat_closed_loop.py",
        "planning/pre_gat_220step_revalidation.py",
        "planning/heterogeneous_candidate_graph.py",
        "planning/gat/candidate_selector.py",
        "planning/gat/edge_enhanced_gat.py",
        "Controller/dmp_rl.py",
        "Environment/frozen_sac_dmp_execution.py",
        "Environment/multi_agent_dmp_env.py",
        "Multi-agent_Algo_lib/scripts/evaluate_gat_closed_loop.py",
        "Multi-agent_Algo_lib/scripts/run_final_four_stage_benchmark.py",
        "Multi-agent_Algo_lib/scripts/run_theory_recovery_runtime_replay.py",
        "configs/evaluation/final_four_stage_benchmark.json",
    )
    return {
        "freeze_time": datetime.now().astimezone().isoformat(),
        "freeze_before_timing_replay": True,
        "source_manifest_sha256": file_hash(SOURCE_ROOT / "scenario_manifest.json"),
        "selected_configs_sha256": file_hash(
            SOURCE_ROOT / "engineering_search" / "selected_configs.json"
        ),
        "metric_source_sha256": file_hash(SOURCE_ROOT / "config.json"),
        "code_hashes": {name: file_hash(REPO_ROOT / name) for name in files},
        "timing_contract": {
            "cuda_actor_and_gat_synchronized": True,
            "execution_actor_only_in_execution_actor_component": True,
            "preview_actor_in_fp_shep_total_only": True,
            "execution_dmp_scope": "compute_dmp_transition_only",
            "environment_step_excluded": True,
            "collision_detection_excluded": True,
            "io_excluded": True,
            "total_online_algorithm_compute": "upper_planning + execution_actor + execution_dmp",
        },
    }


def prefreeze(output_dir: Path) -> None:
    phase = output_dir / "phase_b_runtime"
    if any((phase / RECORD_DIRECTORY).rglob("*.json")):
        raise RuntimeError("cannot freeze runtime instrumentation after replay records exist")
    write_json(phase / PREFREEZE_NAME, _prefreeze_payload())
    print(f"RUNTIME_PREFREEZE={phase / PREFREEZE_NAME}", flush=True)


def verify_prefreeze(output_dir: Path) -> Mapping[str, Any]:
    path = output_dir / "phase_b_runtime" / PREFREEZE_NAME
    frozen = load_json(path)
    current = _prefreeze_payload()
    for key in (
        "source_manifest_sha256",
        "selected_configs_sha256",
        "metric_source_sha256",
        "code_hashes",
        "timing_contract",
    ):
        if frozen[key] != current[key]:
            raise RuntimeError(f"runtime prefreeze mismatch: {key}")
    return frozen


def _original_record(entry: Mapping[str, Any], method: str) -> Mapping[str, Any]:
    return load_json(
        SOURCE_ROOT
        / "formal_records"
        / entry["stage"]
        / entry["scenario_id"]
        / f"{ORIGINAL_NAMES[method]}.json"
    )


def _trajectory_matches(
    original_record: Mapping[str, Any], trajectory: Mapping[str, Any]
) -> tuple[bool, dict[str, bool]]:
    source = SOURCE_ROOT / str(original_record["trajectory_path"])
    checks: dict[str, bool] = {}
    with np.load(source) as frozen:
        for key in ("positions", "velocities", "accelerations"):
            checks[key] = key in trajectory and np.array_equal(
                np.asarray(trajectory[key]), np.asarray(frozen[key])
            )
    return all(checks.values()), checks


def _behavior_match(
    *,
    original: Mapping[str, Any],
    episode: Mapping[str, Any],
    trajectory: Mapping[str, Any],
) -> dict[str, Any]:
    previous = original["episode"]
    categorical_fields = (
        "team_success",
        "collision",
        "obstacle_collision",
        "inter_agent_collision",
        "timeout",
        "termination_reason",
        "steps",
    )
    categorical = {
        field: previous[field] == episode[field] for field in categorical_fields
    }
    selection_hash_match = previous.get("selection_plan_hash") == episode.get(
        "selection_plan_hash"
    )
    candidate_hash_match = previous.get("candidate_bundle_hash") == episode.get(
        "candidate_bundle_hash"
    )
    trajectory_match, trajectory_checks = _trajectory_matches(original, trajectory)
    return {
        "categorical": categorical,
        "selection_plan_hash_match": selection_hash_match,
        "selection_plan_hash_gate": False,
        "selection_plan_hash_note": (
            "serialized diagnostic hashes may differ after audit-only metadata changes; "
            "candidate identity and exact executed trajectory are the behavior gates"
        ),
        "candidate_bundle_hash_match": candidate_hash_match,
        "trajectory_exact_match": trajectory_match,
        "trajectory_components": trajectory_checks,
        "all_match": bool(
            all(categorical.values()) and candidate_hash_match and trajectory_match
        ),
    }


def _method_components(method: str, shared: Mapping[str, Any]) -> dict[str, float]:
    full = shared["runtime_components"]
    zero = 0.0
    proposal = float(full["proposal_generation_ms"])
    coarse = float(full["coarse_ranking_ms"])
    fp = float(full["fp_shep_total_ms"])
    preview_actor = float(full["fp_shep_preview_actor_ms"] or 0.0)
    graph = float(full["graph_build_ms"])
    gat = float(full["gat_forward_ms"])
    if method == "terminal":
        return {
            "proposal_generation_ms": zero,
            "coarse_ranking_ms": zero,
            "fp_shep_total_ms": zero,
            "fp_shep_preview_actor_ms": zero,
            "graph_build_ms": zero,
            "gat_forward_ms": zero,
            "upper_planning_total_ms": zero,
        }
    if method == "proposal":
        fp = preview_actor = graph = gat = zero
        upper = proposal + coarse
    elif method == "fp_shep":
        graph = gat = zero
        upper = proposal + coarse + fp
    else:
        upper = float(full["upper_planning_total_ms"])
    return {
        "proposal_generation_ms": proposal,
        "coarse_ranking_ms": coarse,
        "fp_shep_total_ms": fp,
        "fp_shep_preview_actor_ms": preview_actor,
        "graph_build_ms": graph,
        "gat_forward_ms": gat,
        "upper_planning_total_ms": upper,
    }


def _record_path(phase: Path, entry: Mapping[str, Any]) -> Path:
    return phase / RECORD_DIRECTORY / entry["stage"] / f"{entry['scenario_id']}.json"


def _warmup(runtime: FrozenRuntime, eval_config: Mapping[str, Any], entry: Mapping[str, Any]) -> None:
    shared = build_shared_selection_bundle(
        config=eval_config,
        execution_settings=runtime.execution_settings,
        multi_config=runtime.multi_config,
        policy=runtime.policy,
        gat_model=runtime.gat_model,
        gat_device=runtime.gat_device,
        scenario=entry["scenario_id"],
        seed=int(entry["seed"]),
        environment_builder=runtime.builder,
    )
    sink: dict[str, Any] = {}
    run_method_episode(
        config=eval_config,
        execution_settings=runtime.execution_settings,
        multi_config=runtime.multi_config,
        policy=runtime.policy,
        shared=shared,
        method=METHOD_GAT,
        environment_builder=runtime.builder,
        trajectory_sink=sink,
    )


def _run_scenario(
    runtime: FrozenRuntime,
    eval_config: Mapping[str, Any],
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    recorder = OnlineRuntimeRecorder()
    policy = TimedPolicyProxy(runtime.policy, recorder)
    with recorder.instrument_dmp(), recorder.scoped_context(
        stage=entry["stage"],
        scenario_id=entry["scenario_id"],
        seed=int(entry["seed"]),
        method="shared_upper",
        planning_decision_index=0,
        event_step=0,
    ):
        shared = build_shared_selection_bundle(
            config=eval_config,
            execution_settings=runtime.execution_settings,
            multi_config=runtime.multi_config,
            policy=policy,
            gat_model=runtime.gat_model,
            gat_device=runtime.gat_device,
            scenario=entry["scenario_id"],
            seed=int(entry["seed"]),
            environment_builder=runtime.builder,
            runtime_recorder=recorder,
        )
        upper_actor_rows = [dict(row) for row in recorder.actor_rows]
        methods: dict[str, Any] = {}
        for method, internal in METHODS.items():
            actor_start = len(recorder.actor_rows)
            dmp_start = len(recorder.dmp_rows)
            sink: dict[str, Any] = {}
            with recorder.scoped_context(method=method):
                episode, agents = run_method_episode(
                    config=eval_config,
                    execution_settings=runtime.execution_settings,
                    multi_config=runtime.multi_config,
                    policy=policy,
                    shared=shared,
                    method=internal,
                    environment_builder=runtime.builder,
                    trajectory_sink=sink,
                    runtime_recorder=recorder,
                )
            actor_rows = [dict(row) for row in recorder.actor_rows[actor_start:]]
            dmp_rows = [dict(row) for row in recorder.dmp_rows[dmp_start:]]
            components = _method_components(method, shared)
            execution_actor_ms = float(sum(row["runtime_ms"] for row in actor_rows))
            execution_dmp_ms = float(sum(row["runtime_ms"] for row in dmp_rows))
            components.update(
                {
                    "execution_actor_forward_ms": execution_actor_ms,
                    "execution_actor_call_count": len(actor_rows),
                    "execution_dmp_ms": execution_dmp_ms,
                    "execution_dmp_call_count": len(dmp_rows),
                    "planning_decision_count": 0 if method == "terminal" else 1,
                    "replanning_count": 0,
                }
            )
            components["total_online_algorithm_compute_ms"] = (
                components["upper_planning_total_ms"]
                + execution_actor_ms
                + execution_dmp_ms
            )
            match = _behavior_match(
                original=_original_record(entry, method),
                episode=episode,
                trajectory=sink,
            )
            methods[method] = {
                "episode": episode,
                "agent_count": len(agents),
                "timing": components,
                "behavior_match": match,
                "execution_actor_rows": actor_rows,
                "execution_dmp_rows": dmp_rows,
            }
    return {
        "stage": entry["stage"],
        "family": entry["family"],
        "scenario_id": entry["scenario_id"],
        "seed": int(entry["seed"]),
        "shared_runtime_components": shared["runtime_components"],
        "upper_actor_rows": upper_actor_rows,
        "upper_rows": recorder.upper_rows,
        "methods": methods,
    }


def run(output_dir: Path, limit: int | None = None) -> None:
    verify_prefreeze(output_dir)
    phase = output_dir / "phase_b_runtime"
    config = load_json(SOURCE_ROOT / "config.json")
    manifest = load_json(SOURCE_ROOT / "scenario_manifest.json")
    selected = load_json(SOURCE_ROOT / "engineering_search" / "selected_configs.json")
    runtime = FrozenRuntime(config, manifest)
    eval_config = proposed_eval_config(runtime.base_eval_config, selected["gat_v1"])
    entries = list(manifest["entries"])
    _warmup(runtime, eval_config, entries[0])
    if limit is not None:
        entries = entries[: int(limit)]
    for index, entry in enumerate(entries, start=1):
        path = _record_path(phase, entry)
        if path.exists():
            continue
        payload = _run_scenario(runtime, eval_config, entry)
        write_json(path, payload)
        if index % 5 == 0 or index == len(entries):
            print(
                f"[runtime-replay {index}/{len(entries)}] "
                f"{entry['stage']} {entry['scenario_id']}",
                flush=True,
            )
    if limit is None:
        analyze(output_dir)


def _stats(values: Sequence[float]) -> dict[str, float | None]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"mean": None, "median": None, "p90": None, "max": None}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(np.max(array)),
    }


def _summary_rows(episodes: Sequence[Mapping[str, Any]], scope_key: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scopes = sorted({str(row[scope_key]) for row in episodes}) if scope_key != "overall" else ["overall"]
    methods = sorted({str(row["method"]) for row in episodes})
    for scope in scopes:
        for method in methods:
            members = [
                row
                for row in episodes
                if row["method"] == method
                and (scope == "overall" or str(row[scope_key]) == scope)
            ]
            if not members:
                continue
            decision = _stats([float(row["planning_decision_count"]) for row in members])
            upper = _stats([float(row["upper_planning_total_ms"]) for row in members])
            total = _stats([float(row["total_online_algorithm_compute_ms"]) for row in members])
            latency_values = [
                float(row["upper_planning_total_ms"]) / float(row["planning_decision_count"])
                for row in members
                if float(row["planning_decision_count"]) > 0
            ]
            latency = _stats(latency_values)
            rows.append(
                {
                    "scope": scope,
                    "method": method,
                    "episode_count": len(members),
                    "mean_decisions_per_episode": decision["mean"],
                    "median_decisions_per_episode": decision["median"],
                    "p90_decisions_per_episode": decision["p90"],
                    "mean_decision_latency_ms": latency["mean"],
                    "mean_decision_latency_over_100ms": (
                        float(latency["mean"]) / 100.0 if latency["mean"] is not None else None
                    ),
                    "mean_cumulative_planning_ms": upper["mean"],
                    "p90_cumulative_planning_ms": upper["p90"],
                    "mean_execution_actor_ms": float(
                        np.mean([float(row["execution_actor_forward_ms"]) for row in members])
                    ),
                    "mean_execution_dmp_ms": float(
                        np.mean([float(row["execution_dmp_ms"]) for row in members])
                    ),
                    "mean_total_online_algorithm_compute_ms": total["mean"],
                    "median_total_online_algorithm_compute_ms": total["median"],
                    "p90_total_online_algorithm_compute_ms": total["p90"],
                }
            )
    return rows


def analyze(output_dir: Path) -> None:
    verify_prefreeze(output_dir)
    phase = output_dir / "phase_b_runtime"
    paths = sorted((phase / RECORD_DIRECTORY).rglob("*.json"))
    if len(paths) != 400:
        raise RuntimeError(f"expected 400 runtime replay records, found {len(paths)}")
    timing_rows: list[dict[str, Any]] = []
    actor_rows: list[dict[str, Any]] = []
    dmp_rows: list[dict[str, Any]] = []
    matches: list[bool] = []
    for path in paths:
        payload = load_json(path)
        upper_actor = payload["upper_actor_rows"]
        for method, data in payload["methods"].items():
            row = {
                "stage": payload["stage"],
                "family": payload["family"],
                "scenario_id": payload["scenario_id"],
                "seed": payload["seed"],
                "method": method,
                **data["timing"],
            }
            timing_rows.append(row)
            matches.append(bool(data["behavior_match"]["all_match"]))
            actor_rows.extend(data["execution_actor_rows"])
            dmp_rows.extend(data["execution_dmp_rows"])
            if method in {"fp_shep", "gat_v1_one_shot"}:
                for source_row in upper_actor:
                    copied = dict(source_row)
                    copied["method"] = method
                    actor_rows.append(copied)
    corrected_path = output_dir / "corrected_classical_results.csv"
    with corrected_path.open("r", encoding="utf-8-sig", newline="") as handle:
        corrected = list(csv.DictReader(handle))
    for row in corrected:
        timing_rows.append(
            {
                "stage": row["stage"],
                "family": row["family"],
                "scenario_id": row["scenario_id"],
                "seed": int(row["seed"]),
                "method": row["method"],
                "proposal_generation_ms": 0.0,
                "coarse_ranking_ms": 0.0,
                "fp_shep_total_ms": 0.0,
                "fp_shep_preview_actor_ms": 0.0,
                "graph_build_ms": 0.0,
                "gat_forward_ms": 0.0,
                "upper_planning_total_ms": float(row["planning_runtime_ms"]),
                "execution_actor_forward_ms": 0.0,
                "execution_actor_call_count": 0,
                "execution_dmp_ms": 0.0,
                "execution_dmp_call_count": 0,
                "planning_decision_count": int(row["planning_decision_count"]),
                "replanning_count": 0,
                "total_online_algorithm_compute_ms": float(row["planning_runtime_ms"]),
            }
        )
    method_summary = _summary_rows(timing_rows, "overall")
    stage_summary = _summary_rows(timing_rows, "stage")
    for path, rows in (
        (output_dir / "runtime_method_summary.csv", method_summary),
        (output_dir / "runtime_stage_summary.csv", stage_summary),
        (output_dir / "actor_forward_timing.csv", actor_rows),
        (output_dir / "dmp_timing.csv", dmp_rows),
        (phase / "runtime_episode_results.csv", timing_rows),
    ):
        write_csv(path, rows)
        write_csv(phase / path.name, rows)
    reconciliation = {
        "TIMING_REPLAY_BEHAVIOR_MATCH": "YES" if all(matches) else "NO",
        "matched_episode_count": sum(matches),
        "expected_episode_count": 1600,
        "runtime_scenario_record_count": len(paths),
        "runtime_episode_count_including_corrected_classics": len(timing_rows),
        "actor_timing_row_count": len(actor_rows),
        "dmp_timing_row_count": len(dmp_rows),
        "all_cuda_actor_calls_synchronized": all(
            bool(row["cuda_synchronized"]) for row in actor_rows
        ) if actor_rows else True,
        "trajectory_exact_match_required": True,
        "timing_contract": load_json(phase / PREFREEZE_NAME)["timing_contract"],
    }
    write_json(phase / "runtime_reconciliation.json", reconciliation)
    print(json.dumps(reconciliation, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prefreeze", "run", "analyze"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if args.phase == "prefreeze":
        prefreeze(output)
    elif args.phase == "run":
        run(output, args.limit)
    else:
        analyze(output)


if __name__ == "__main__":
    main()
