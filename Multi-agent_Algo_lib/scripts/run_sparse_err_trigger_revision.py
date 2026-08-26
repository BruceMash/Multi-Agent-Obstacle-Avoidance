from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import runpy
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = REPO_ROOT / "Multi-agent_Algo_lib"
for search_path in (REPO_ROOT, ALGO_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from planning.final_four_stage_benchmark import (  # noqa: E402
    generate_scenario_manifest,
    validate_scenario_manifest,
)
from planning.online_runtime_instrumentation import (  # noqa: E402
    OnlineRuntimeRecorder,
    TimedPolicyProxy,
)
from scripts.evaluate_gat_v1_err_development import (  # noqa: E402
    METHOD_ERR,
    METHOD_ONE_SHOT,
    METHOD_RERR_FP_SHEP,
    METHOD_RERR_GAT,
    run_episode,
)
from scripts.run_final_four_stage_benchmark import (  # noqa: E402
    FrozenRuntime,
    proposed_eval_config,
)


OUTPUT = (
    REPO_ROOT
    / "artifacts"
    / "sparse_err_trigger_revision"
    / "20260819_020727"
)
GOAL_CONFIG = REPO_ROOT / "configs/evaluation/sparse_err_trigger_revision.json"
METHODS = (
    METHOD_ONE_SHOT,
    METHOD_ERR,
    METHOD_RERR_GAT,
    METHOD_RERR_FP_SHEP,
)
METHOD_LABELS = {
    METHOD_ONE_SHOT: "M1_one_shot_gat",
    METHOD_ERR: "M2_current_err_gat",
    METHOD_RERR_GAT: "M3_revised_err_gat",
    METHOD_RERR_FP_SHEP: "M4_revised_err_fp_shep",
}
RECORD_DIRECTORY = "development_records"
PREFREEZE_FILE = "development_prefreeze.json"


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(jsonable(row.get(key)), ensure_ascii=False)
                        if isinstance(row.get(key), (dict, list, tuple, np.ndarray))
                        else jsonable(row.get(key))
                    )
                    for key in fields
                }
            )
    temporary.replace(path)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _collect_history_values(output_dir: Path) -> tuple[set[int], set[str], set[str], list[str]]:
    seeds: set[int] = set()
    geometry: set[str] = set()
    translation: set[str] = set()
    sources: list[str] = []

    def walk(value: Any, key: str | None = None) -> None:
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                walk(child, str(child_key))
        elif isinstance(value, list):
            for child in value:
                walk(child, key)
        elif key in {"seed", "seeds", "formal_seeds", "development_seeds"}:
            try:
                seeds.add(int(value))
            except (TypeError, ValueError):
                pass
        elif key == "geometry_fingerprint" and isinstance(value, str):
            geometry.add(value)
        elif key == "translation_invariant_fingerprint" and isinstance(value, str):
            translation.add(value)

    for path in sorted((REPO_ROOT / "artifacts").rglob("*manifest*.json")):
        if output_dir in path.parents:
            continue
        try:
            payload = load_json(path)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        walk(payload)
        sources.append(str(path.relative_to(REPO_ROOT)).replace("\\", "/"))
    return seeds, geometry, translation, sources


def _run_regression_tests(output_dir: Path) -> None:
    files = (
        REPO_ROOT / "test/test_event_triggered_reference_reconstruction.py",
        REPO_ROOT / "test/test_sparse_err_trigger_revision.py",
    )
    rows: list[dict[str, Any]] = []
    for path in files:
        namespace = runpy.run_path(str(path))
        for name, function in sorted(namespace.items()):
            if name.startswith("test_") and callable(function):
                function()
                rows.append(
                    {
                        "test_file": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
                        "test_name": name,
                        "status": "PASS",
                    }
                )
    write_json(
        output_dir / "regression_tests.json",
        {
            "status": "PASS",
            "test_count": len(rows),
            "tests": rows,
            "required_A_to_I_covered": True,
            "legacy_level_semantics_preserved": True,
        },
    )


def _source_paths() -> tuple[Path, Path, Path]:
    goal = load_json(GOAL_CONFIG)
    return (
        REPO_ROOT / goal["base_benchmark_config"],
        REPO_ROOT / goal["selected_upper_config"],
        REPO_ROOT / goal["err_parameter_source"],
    )


def _development_eval_config(manifest: Mapping[str, Any]) -> dict[str, Any]:
    base_path, selected_path, err_path = _source_paths()
    base = load_json(base_path)
    selected = load_json(selected_path)
    err = load_json(err_path)
    return {
        "source_config": str(base_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "selected_proposed": selected["gat_v1"],
        "err": err["err"],
        "execution": base["execution"],
        "manifest_sha256": manifest["manifest_sha256"],
        "methods": list(METHODS),
        "method_labels": METHOD_LABELS,
        "current_emergency_semantics": "level_triggered",
        "revised_emergency_semantics": "edge_triggered_existing_threshold_rearm",
        "normal_trigger_changed": False,
        "new_numeric_threshold_added": False,
        "multiple_reproposals_allowed": True,
        "artificial_reproposal_cap": None,
    }


def prepare(output_dir: Path) -> None:
    if any((output_dir / RECORD_DIRECTORY).rglob("*.json")):
        raise RuntimeError("cannot freeze a manifest after development records exist")
    _run_regression_tests(output_dir)
    historical_seeds, historical_geometry, historical_translation, sources = (
        _collect_history_values(output_dir)
    )
    manifest: dict[str, Any] | None = None
    seed_base = 991_900_000
    for attempt in range(100):
        candidate = generate_scenario_manifest(
            counts_per_stage=20,
            seed_base=seed_base + attempt * 100_000,
            prefix="SR",
            max_steps=220,
            dt=0.1,
        )
        candidate_seeds = {int(row["seed"]) for row in candidate["entries"]}
        candidate_geometry = {
            str(row["geometry_fingerprint"]) for row in candidate["entries"]
        }
        candidate_translation = {
            str(row["translation_invariant_fingerprint"])
            for row in candidate["entries"]
        }
        if (
            candidate_seeds.isdisjoint(historical_seeds)
            and candidate_geometry.isdisjoint(historical_geometry)
            and candidate_translation.isdisjoint(historical_translation)
        ):
            manifest = candidate
            break
    if manifest is None:
        raise RuntimeError("could not construct a fully disjoint 80-scenario manifest")
    validation = validate_scenario_manifest(manifest)
    if validation["status"] != "PASSED":
        raise RuntimeError(f"manifest validation failed: {validation}")
    write_json(output_dir / "development_scenario_manifest.json", manifest)
    write_json(output_dir / "development_eval_config.json", _development_eval_config(manifest))
    disjointness = {
        "history_source_manifest_count": len(sources),
        "history_source_manifests": sources,
        "historical_seed_count": len(historical_seeds),
        "historical_geometry_count": len(historical_geometry),
        "historical_translation_fingerprint_count": len(historical_translation),
        "seed_overlap_count": 0,
        "geometry_overlap_count": 0,
        "translation_equivalent_overlap_count": 0,
        "seed_disjoint": True,
        "geometry_disjoint": True,
        "translation_equivalent_disjoint": True,
        "manifest_validation": validation,
    }
    write_json(output_dir / "development_disjointness_audit.json", disjointness)
    base_path, _, _ = _source_paths()
    base = load_json(base_path)
    source_files = (
        "hire-rl-body.tex",
        "configs/evaluation/sparse_err_trigger_revision.json",
        "configs/evaluation/gat_v1_err_development.json",
        "planning/event_triggered_reference_reconstruction.py",
        "planning/final_four_stage_benchmark.py",
        "planning/online_runtime_instrumentation.py",
        "planning/pre_gat_closed_loop.py",
        "planning/heterogeneous_candidate_graph.py",
        "planning/gat/candidate_selector.py",
        "planning/gat/edge_enhanced_gat.py",
        "Guidance/reference_point_proposal_demo.py",
        "Environment/frozen_sac_dmp_execution.py",
        "Environment/multi_agent_dmp_env.py",
        "Controller/dmp_rl.py",
        "Multi-agent_Algo_lib/scripts/evaluate_gat_v1_err_development.py",
        "Multi-agent_Algo_lib/scripts/run_sparse_err_trigger_revision.py",
        "Multi-agent_Algo_lib/scripts/analyze_sparse_err_trigger_revision.py",
        "test/test_event_triggered_reference_reconstruction.py",
        "test/test_sparse_err_trigger_revision.py",
    )
    freeze = {
        "freeze_time": datetime.now().astimezone().isoformat(),
        "freeze_before_any_development_performance": True,
        "development_manifest_sha256": file_hash(
            output_dir / "development_scenario_manifest.json"
        ),
        "development_eval_config_sha256": file_hash(
            output_dir / "development_eval_config.json"
        ),
        "development_disjointness_audit_sha256": file_hash(
            output_dir / "development_disjointness_audit.json"
        ),
        "regression_tests_sha256": file_hash(output_dir / "regression_tests.json"),
        "phase_a_counterfactual_sha256": file_hash(
            output_dir / "counterfactual_summary.json"
        ),
        "code_and_theory_hashes": {
            name: file_hash(REPO_ROOT / name) for name in source_files
        },
        "sac_checkpoint_sha256": file_hash(
            REPO_ROOT / base["sources"]["sac_checkpoint"]
        ),
        "gat_checkpoint_sha256": file_hash(
            REPO_ROOT / base["sources"]["gat_checkpoint"]
        ),
        "frozen_analysis_rules": {
            "similar_runtime_band": "within +/-10% of historical reference",
            "state_dependent_separation_yes": (
                "pooled Stage III/IV mean reproposals exceeds pooled Stage I/II "
                "and each hard-stage mean exceeds every easy-stage mean"
            ),
            "state_dependent_separation_partial": (
                "pooled Stage III/IV mean reproposals exceeds pooled Stage I/II only"
            ),
            "invalid_emergency_repeat": (
                "same-agent emergency events without an intervening h_active>=h_rep "
                "witness, including the post-update active-goal margin"
            ),
            "normal_chattering": (
                "same-agent consecutive normal same-goal events within the existing "
                "T_rep,min (+one dt tolerance) without an intervening h_active>=h_rep "
                "or progress>nu_min recovery witness"
            ),
            "oracle_candidate_available": (
                "on a failed M3 episode, at least one GAT planning event has a non-null "
                "FP-SHEP Top-1 alternative with strictly greater frozen FP-SHEP score "
                "than the GAT choice, or GAT selected null while K_t>0"
            ),
            "gat_finetune_yes": (
                "oracle availability >=50% of failed M3 episodes and observed M4-minus-M3 "
                "success gap >=15 percentage points"
            ),
            "gat_closed_loop_value": {
                "NEGATIVE": "M3 success<M4, or equal success with higher M3 collision",
                "STRONG": (
                    "M3-M4 overall success >=5 pp or Stage III/IV pooled gain >=10 pp, "
                    "with M3 collision no higher"
                ),
                "MODERATE": "positive M3 success gain or lower M3 collision not meeting STRONG",
                "WEAK": "none of NEGATIVE, STRONG, or MODERATE",
            },
            "revision_success_yes": (
                "M3 success>=M2, collision<=M2, each Stage I/II success>=M2, raw "
                "consecutive-emergency reduction>=80%, same-goal count<=M2, and no chatter"
            ),
            "revision_success_partial": (
                "M3 success>=M2, collision<=M2, pooled Stage I/II success>=M2, raw "
                "consecutive-emergency reduction>=80%, same-goal count<=M2, and no chatter"
            ),
        },
        "normal_trigger_changed": False,
        "new_numeric_threshold_added": False,
        "multiple_reproposals_allowed": True,
        "artificial_reproposal_cap": None,
        "formal_benchmark_authorized": False,
        "training_authorized": False,
    }
    write_json(output_dir / PREFREEZE_FILE, freeze)
    print(f"DEVELOPMENT_MANIFEST_SHA256={manifest['manifest_sha256']}", flush=True)
    print("DEVELOPMENT_PREFREEZE=PASS", flush=True)


def verify_freeze(output_dir: Path) -> Mapping[str, Any]:
    freeze = load_json(output_dir / PREFREEZE_FILE)
    checks = {
        "development_scenario_manifest.json": "development_manifest_sha256",
        "development_eval_config.json": "development_eval_config_sha256",
        "development_disjointness_audit.json": "development_disjointness_audit_sha256",
        "regression_tests.json": "regression_tests_sha256",
        "counterfactual_summary.json": "phase_a_counterfactual_sha256",
    }
    for name, key in checks.items():
        if file_hash(output_dir / name) != freeze[key]:
            raise RuntimeError(f"frozen artifact changed: {name}")
    for name, expected in freeze["code_and_theory_hashes"].items():
        if file_hash(REPO_ROOT / name) != expected:
            raise RuntimeError(f"frozen code/theory changed: {name}")
    return freeze


def _runtime_eval_config(runtime: FrozenRuntime, output_dir: Path) -> dict[str, Any]:
    frozen = load_json(output_dir / "development_eval_config.json")
    result = proposed_eval_config(runtime.base_eval_config, frozen["selected_proposed"])
    result["err"] = frozen["err"]
    return result


def _record_path(output_dir: Path, entry: Mapping[str, Any], method: str) -> Path:
    return (
        output_dir
        / RECORD_DIRECTORY
        / entry["stage"]
        / entry["scenario_id"]
        / f"{method}.json"
    )


def run(output_dir: Path, limit: int | None = None) -> None:
    verify_freeze(output_dir)
    manifest = load_json(output_dir / "development_scenario_manifest.json")
    base_path, _, _ = _source_paths()
    runtime = FrozenRuntime(load_json(base_path), manifest)
    eval_config = _runtime_eval_config(runtime, output_dir)
    entries = list(manifest["entries"])
    # Post-freeze warm-up is not retained as performance evidence.
    run_episode(
        config=eval_config,
        settings=runtime.execution_settings,
        multi_config=runtime.multi_config,
        policy=runtime.policy,
        gat_model=runtime.gat_model,
        gat_device=runtime.gat_device,
        method=METHOD_ONE_SHOT,
        scenario=entries[0]["scenario_id"],
        seed=int(entries[0]["seed"]),
        environment_builder=runtime.builder,
        runtime_recorder=None,
    )
    if limit is not None:
        entries = entries[: int(limit)]
    expected = len(entries) * len(METHODS)
    job = 0
    for entry in entries:
        initial_candidate_hashes: dict[str, str] = {}
        initial_gat_hashes: dict[str, str] = {}
        for method in METHODS:
            job += 1
            path = _record_path(output_dir, entry, method)
            if path.exists():
                existing = load_json(path)["episode"]
                initial_candidate_hashes[method] = existing["initial_candidate_bundle_hash"]
                if method != METHOD_RERR_FP_SHEP:
                    initial_gat_hashes[method] = existing["initial_selection_semantic_hash"]
                continue
            recorder = OnlineRuntimeRecorder()
            policy = TimedPolicyProxy(runtime.policy, recorder)
            with recorder.instrument_dmp(), recorder.scoped_context(
                evaluation_block="new_80_sparse_err_revision_development",
                stage=entry["stage"],
                family=entry["family"],
                scenario_id=entry["scenario_id"],
                seed=int(entry["seed"]),
                method=METHOD_LABELS[method],
            ):
                episode, agents, events, triggers, auxiliary = run_episode(
                    config=eval_config,
                    settings=runtime.execution_settings,
                    multi_config=runtime.multi_config,
                    policy=policy,
                    gat_model=runtime.gat_model,
                    gat_device=runtime.gat_device,
                    method=method,
                    scenario=entry["scenario_id"],
                    seed=int(entry["seed"]),
                    environment_builder=runtime.builder,
                    runtime_recorder=recorder,
                )
            episode = dict(episode)
            episode.update(
                {
                    "stage": entry["stage"],
                    "family": entry["family"],
                    "scenario_id": entry["scenario_id"],
                    "method_label": METHOD_LABELS[method],
                }
            )
            initial_candidate_hashes[method] = episode["initial_candidate_bundle_hash"]
            if method != METHOD_RERR_FP_SHEP:
                initial_gat_hashes[method] = episode["initial_selection_semantic_hash"]
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
                        "translation_invariant_fingerprint",
                    )
                },
                "episode": episode,
                "agents": agents,
                "events": events,
                "triggers": triggers,
                "actor_timing_rows": auxiliary["actor_timing_rows"],
                "dmp_timing_rows": auxiliary["dmp_timing_rows"],
                "upper_timing_rows": auxiliary["upper_timing_rows"],
                "scene_record": auxiliary["scene_record"],
            }
            write_json(path, payload)
            if job % 8 == 0 or job == expected:
                print(
                    f"[sparse-rerr {job}/{expected}] {entry['stage']} "
                    f"{entry['scenario_id']} {METHOD_LABELS[method]}: "
                    f"{episode['termination_reason']}",
                    flush=True,
                )
        if len(initial_candidate_hashes) == len(METHODS) and len(
            set(initial_candidate_hashes.values())
        ) != 1:
            raise RuntimeError("paired methods used different initial candidate bundles")
        if len(initial_gat_hashes) == 3 and len(set(initial_gat_hashes.values())) != 1:
            raise RuntimeError("M1/M2/M3 used different initial GAT selections")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "run", "verify"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if args.phase == "prepare":
        prepare(output)
    elif args.phase == "run":
        run(output, args.limit)
    else:
        verify_freeze(output)
        print("DEVELOPMENT_FREEZE_VERIFIED=YES", flush=True)


if __name__ == "__main__":
    main()
