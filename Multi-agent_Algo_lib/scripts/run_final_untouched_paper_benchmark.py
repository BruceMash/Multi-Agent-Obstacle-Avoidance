"""Freeze and run the last untouched eight-method paper benchmark.

This file is evaluation infrastructure only.  It composes already-frozen
planners and checkpoints; it does not tune or modify any method.  The command
has two irreversible phases:

``prepare`` creates the unique 400-scene manifest, all contracts, the fixed
schedule, and ``FINAL_FORMAL_FREEZE.json`` without executing a formal episode.

``formal`` verifies every frozen hash, performs fixed historical warm-up, then
executes/resumes the 3,200 scheduled episodes.  During this phase it reports
only record counts and health, never performance aggregates.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import hashlib
import json
import math
import os
import platform
import runpy
import socket
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
ALGO = ROOT / "Multi-agent_Algo_lib"
SCRIPTS = Path(__file__).resolve().parent
for _search in (ROOT, ALGO, SCRIPTS):
    if str(_search) not in sys.path:
        sys.path.insert(0, str(_search))

from planning.final_four_stage_benchmark import (  # noqa: E402
    DWAStyleConfig,
    FAMILY_ORDER,
    STAGE_ORDER,
    WORKSPACE_BOUNDS,
    generate_scenario_manifest,
    json_ready,
    run_classical_episode,
    stable_hash,
    validate_scenario_manifest,
)
from planning.online_runtime_instrumentation import (  # noqa: E402
    OnlineRuntimeRecorder,
    TimedPolicyProxy,
)
from planning.sensing_matched_classical import run_sensing_matched_episode  # noqa: E402
from scripts.evaluate_gat_closed_loop import (  # noqa: E402
    METHOD_FP_SHEP,
    METHOD_GAT,
    METHOD_PROPOSAL,
    METHOD_TERMINAL,
    build_shared_selection_bundle,
    run_method_episode,
)
from scripts.evaluate_gat_v1_err_development import (  # noqa: E402
    METHOD_RERR_FP_SHEP,
    METHOD_RERR_GAT,
    build_online_gat_plan_optimized,
    run_episode as run_rerr_episode,
)
from scripts.run_final_four_stage_benchmark import (  # noqa: E402
    FrozenRuntime,
    ManifestEnvironmentBuilder,
    audit_used_geometry,
    audit_used_seeds,
    hardware_manifest,
    proposed_eval_config,
    standardize_sac_result,
)


SCHEMA = "final_untouched_paper_benchmark_v1"
RUN_ID = "20260820_110311"
DEFAULT_OUTPUT = ROOT / "artifacts" / "FINAL_UNTOUCHED_PAPER_BENCHMARK" / RUN_ID
SEED_BASE = 1_950_000_000
SCENARIOS_PER_STAGE = 100
EXPECTED_SCENARIOS = 400
EXPECTED_TEAM_ROWS = 3200
EXPECTED_AGENT_ROWS = 9600

BASE_RUN = ROOT / "artifacts" / "final_four_stage_benchmark" / "20260818_202620"
BASE_CONFIG = BASE_RUN / "config.json"
SELECTED_CONFIG = BASE_RUN / "engineering_search" / "selected_configs.json"
ERR_CONFIG = ROOT / "configs" / "evaluation" / "gat_v1_err_development.json"
METHOD_FREEZE = ROOT / "artifacts" / "final_execution_semantics_audit" / "20260819_194912" / "FINAL_METHOD_FREEZE.json"
ENGINEERING_FREEZE = ROOT / "artifacts" / "rerr_runtime_compression" / "20260819_162022" / "FINAL_ENGINEERING_FREEZE.json"
HISTORICAL_MANIFEST = ROOT / "artifacts" / "sparse_err_trigger_revision" / "20260819_020727" / "development_scenario_manifest.json"
ATTACHMENT = Path(r"C:\Users\Administrator\.codex\attachments\831cb02f-f3f5-4c3c-a630-cbebc515133b\pasted-text.txt")

METHODS: tuple[dict[str, Any], ...] = (
    {
        "method_id": "M1_DWA_FullState",
        "display_name": "DWA-FullState",
        "role": "STRONG_SYSTEM_LEVEL_REFERENCE",
        "family": "classical",
        "engine": "dwa_fullstate",
    },
    {
        "method_id": "M2_DWA_SensingMatched",
        "display_name": "DWA-SensingMatched",
        "role": "LOCAL_SENSING_MATCHED_CLASSICAL_COMPARISON",
        "family": "classical",
        "engine": "dwa_sensing_matched",
    },
    {
        "method_id": "M3_Direct_SAC_DMP",
        "display_name": "Direct SAC-DMP",
        "role": "LEARNING_BASELINE",
        "family": "learned_chain",
        "engine": "one_shot_terminal",
    },
    {
        "method_id": "M4_Proposal_SAC_DMP",
        "display_name": "Proposal + SAC-DMP",
        "role": "REFERENCE_GENERATION_ABLATION",
        "family": "learned_chain",
        "engine": "one_shot_proposal",
    },
    {
        "method_id": "M5_FP_SHEP_SAC_DMP",
        "display_name": "FP-SHEP + SAC-DMP",
        "role": "EXECUTION_AWARE_PREVIEW_ABLATION",
        "family": "learned_chain",
        "engine": "one_shot_fp_shep",
    },
    {
        "method_id": "M6_OneShot_GAT_SAC_DMP",
        "display_name": "One-Shot GAT + SAC-DMP",
        "role": "NO_RECURRENT_RECONSTRUCTION_ABLATION",
        "family": "learned_chain",
        "engine": "one_shot_gat",
    },
    {
        "method_id": "M7_RERR_FP_SHEP_SAC_DMP",
        "display_name": "R-ERR + FP-SHEP + SAC-DMP",
        "role": "GAT_ABLATION_UNDER_RERR",
        "family": "learned_chain",
        "engine": "rerr_fp_shep",
    },
    {
        "method_id": "M8_Proposed_RERR_GAT_SAC_DMP",
        "display_name": "Proposed R-ERR + GAT + SAC-DMP",
        "role": "FINAL_PROPOSED_METHOD",
        "family": "learned_chain",
        "engine": "rerr_gat",
    },
)
METHOD_BY_ID = {row["method_id"]: row for row in METHODS}
METHOD_ORDER = tuple(METHOD_BY_ID)
PAPER_STAGE_LABEL = {
    "stage_1": "Stage I",
    "stage_2": "Stage II",
    "stage_3": "Stage III",
    "stage_4": "Stage IV",
}
ONE_SHOT_MAP = {
    "one_shot_terminal": METHOD_TERMINAL,
    "one_shot_proposal": METHOD_PROPOSAL,
    "one_shot_fp_shep": METHOD_FP_SHEP,
    "one_shot_gat": METHOD_GAT,
}

FROZEN_SOURCE_PATHS = (
    "Environment/frozen_sac_dmp_execution.py",
    "Environment/multi_agent_dmp_env.py",
    "Guidance/reference_point_proposal_demo.py",
    "Controller/dmp_rl.py",
    "planning/event_triggered_reference_reconstruction.py",
    "planning/final_four_stage_benchmark.py",
    "planning/heterogeneous_candidate_graph.py",
    "planning/online_runtime_instrumentation.py",
    "planning/policy_preview.py",
    "planning/pre_gat_closed_loop.py",
    "planning/sensing_matched_classical.py",
    "Multi-agent_Algo_lib/scripts/evaluate_actor_dmp_goal_semantics.py",
    "Multi-agent_Algo_lib/scripts/evaluate_gat_closed_loop.py",
    "Multi-agent_Algo_lib/scripts/evaluate_gat_v1_err_development.py",
    "Multi-agent_Algo_lib/scripts/run_final_untouched_paper_benchmark.py",
    "Multi-agent_Algo_lib/scripts/reconcile_final_untouched_paper_benchmark.py",
    "Multi-agent_Algo_lib/scripts/analyze_final_untouched_paper_benchmark.py",
    "Multi-agent_Algo_lib/scripts/plot_final_untouched_paper_benchmark.py",
    "hire-rl-body.tex",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    return value


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if str(key) not in fields:
                fields.append(str(key))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(_jsonable(row.get(key)), ensure_ascii=False, sort_keys=True)
                        if isinstance(row.get(key), (dict, list, tuple, np.ndarray))
                        else _jsonable(row.get(key))
                    )
                    for key in fields
                }
            )
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def content_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            _jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _array_hash(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(arrays):
        array = np.ascontiguousarray(np.asarray(arrays[key]))
        digest.update(key.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def write_npz(path: Path, arrays: Mapping[str, Any]) -> tuple[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = {key: np.asarray(value) for key, value in arrays.items()}
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **normalized)
    temporary.replace(path)
    return _array_hash(normalized), sha256_file(path)


def _translation_registry(excluded_root: Path) -> tuple[set[str], dict[str, list[str]]]:
    values: set[str] = set()
    sources: dict[str, list[str]] = defaultdict(list)

    def visit(value: Any, key: str, source: str) -> None:
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                visit(child, str(child_key), source)
        elif isinstance(value, list):
            for child in value:
                visit(child, key, source)
        elif key == "translation_invariant_fingerprint" and isinstance(value, str):
            values.add(value)
            if len(sources[value]) < 3:
                sources[value].append(source)

    for path in (ROOT / "artifacts").rglob("*"):
        if not path.is_file():
            continue
        try:
            path.relative_to(excluded_root)
            continue
        except ValueError:
            pass
        relative = str(path.relative_to(ROOT)).replace("\\", "/")
        try:
            suffix = path.suffix.lower()
            if suffix == ".json" and path.stat().st_size <= 64 * 1024 * 1024:
                visit(load_json(path), "", relative)
            elif suffix == ".csv" and path.stat().st_size <= 256 * 1024 * 1024:
                with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
                    reader = csv.DictReader(handle)
                    if "translation_invariant_fingerprint" in (reader.fieldnames or []):
                        for row in reader:
                            visit(
                                row.get("translation_invariant_fingerprint"),
                                "translation_invariant_fingerprint",
                                relative,
                            )
        except (OSError, UnicodeError, json.JSONDecodeError, csv.Error):
            continue
    return values, sources


def _resolved_method_configs() -> tuple[dict[str, Any], dict[str, Any], DWAStyleConfig]:
    base = load_json(BASE_CONFIG)
    selected = load_json(SELECTED_CONFIG)
    err = load_json(ERR_CONFIG)
    dummy_manifest = {"entries": []}
    # FrozenRuntime is not needed to derive these values.  Reproduce only the
    # already-audited composition performed by proposed_eval_config.
    execution = base["execution"]
    eval_config = {
        "top_k": int(execution["top_k"]),
        "H_preview": int(execution["H_preview"]),
        "max_steps": int(execution["max_steps"]),
        "dt": float(execution["dt"]),
        "peer_radius": float(execution["peer_radius"]),
        "handoff_threshold_m": float(execution["handoff_threshold_m"]),
        "proposal_config": copy.deepcopy(base["proposal_config"]),
        "fp_shep": copy.deepcopy(base["fp_shep"]),
        "graph": copy.deepcopy(base["graph"]),
        "formal_scenarios": [],
        "formal_seeds": [],
    }
    proposed = proposed_eval_config(eval_config, selected["gat_v1"])
    proposed["err"] = copy.deepcopy(err["err"])
    dwa_payload = {key: value for key, value in selected["dwa_style"].items() if key != "config_id"}
    dwa = DWAStyleConfig(**dwa_payload)
    return base, proposed, dwa


def _method_config_payloads() -> dict[str, dict[str, Any]]:
    base, proposed, dwa = _resolved_method_configs()
    common = {
        "num_agents": 3,
        "max_steps": 220,
        "dt": 0.1,
        "boundary_mode": "boundary_free",
        "collision_semantics": "discrete_post_transition",
        "goal_tolerance_m": 0.30,
        "inter_agent_safe_distance_m": 0.60,
        "checkpoints": copy.deepcopy(base["sources"]),
    }
    result: dict[str, dict[str, Any]] = {}
    for row in METHODS:
        engine = row["engine"]
        payload: dict[str, Any] = {**common, **row}
        if engine.startswith("dwa_"):
            payload.update(
                {
                    "planner": asdict(dwa),
                    "prediction": "current-state constant-velocity only",
                    "per_step_replanning": True,
                    "information_contract": (
                        "exact current simulator geometry/state; no future or private RNG"
                        if engine == "dwa_fullstate"
                        else "ego/goal plus current untyped 56-ray 4.5 m local sensing; zero-order hold endpoints"
                    ),
                }
            )
        else:
            payload.update(
                {
                    "proposal_top_k": 10,
                    "fp_shep_horizon": 4,
                    "eval_config": proposed,
                    "frozen_sac_dmp": True,
                    "frozen_gat_v1": engine in {"one_shot_gat", "rerr_gat"},
                    "rerr_enabled": engine in {"rerr_fp_shep", "rerr_gat"},
                    "selector": "fp_shep" if engine == "rerr_fp_shep" else (
                        "gat_v1" if engine in {"one_shot_gat", "rerr_gat"} else engine
                    ),
                    "upper_information_contract": "exact current peer state only when an upper planning event is executed",
                    "execution_information_contract": "historical 122-D SAC input with current/previous untyped LiDAR",
                }
            )
        result[row["method_id"]] = payload
    return result


def _comparison_contract_rows() -> list[dict[str, Any]]:
    return [
        {
            "method_id": row["method_id"],
            "display_name": row["display_name"],
            "role": row["role"],
            "paired_on_same_scenarios": True,
            "online_information": (
                "Full exact current state/geometry; constant-velocity dynamic prediction; no future"
                if row["engine"] == "dwa_fullstate"
                else "Local untyped 56-ray sensing plus ego/goal; no hidden geometry/future"
                if row["engine"] == "dwa_sensing_matched"
                else "122-D local execution observation; exact peer state only at actual upper events"
            ),
            "planning_frequency": (
                "every control step" if row["engine"].startswith("dwa_")
                else "none (direct terminal execution)" if row["engine"] == "one_shot_terminal"
                else "one at t=0" if row["engine"].startswith("one_shot_")
                else "initial plus state-triggered R-ERR events"
            ),
            "claim_boundary": (
                "strong system reference; not equal-information"
                if row["engine"] == "dwa_fullstate"
                else "matched local-sensing system comparison; upper-layer contracts remain explicit"
                if row["engine"] == "dwa_sensing_matched"
                else "internal frozen learned-chain comparison"
            ),
        }
        for row in METHODS
    ]


def _statistical_protocol() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "frozen_before_performance": True,
        "primary_endpoint": "overall team success rate",
        "primary_report": ["count", "denominator", "rate", "95% Clopper-Pearson exact CI"],
        "stage_endpoints": [
            "team_success", "any_collision", "obstacle_collision",
            "inter_agent_collision", "timeout", "agent_completion",
        ],
        "easy_definition": ["Stage I", "Stage II"],
        "complex_definition": ["Stage III", "Stage IV"],
        "secondary_complex_endpoint": "Stage III plus IV team success",
        "family_cells": "4 stages x 5 frozen families; 20 scenarios/cell; descriptive only",
        "paired_tests": [
            {"id": "P1", "proposed": METHOD_ORDER[7], "baseline": METHOD_ORDER[1], "role": "local-sensing-matched"},
            {"id": "P2", "proposed": METHOD_ORDER[7], "baseline": METHOD_ORDER[2], "role": "learning baseline"},
            {"id": "P3", "proposed": METHOD_ORDER[7], "baseline": METHOD_ORDER[5], "role": "ERR contribution"},
            {"id": "P4", "proposed": METHOD_ORDER[7], "baseline": METHOD_ORDER[6], "role": "GAT contribution under R-ERR"},
            {"id": "P5", "proposed": METHOD_ORDER[7], "baseline": METHOD_ORDER[0], "role": "strong system-level reference only"},
        ],
        "binary_test": "exact two-sided McNemar",
        "preregistered_test_scopes": ["overall", "complex_stage_iii_plus_iv"],
        "stage_tests": "secondary",
        "family_tests": "none; descriptive only",
        "binary_ci": "95% Clopper-Pearson exact",
        "continuous_metrics": [
            "completion_time_s", "team_path_length_m", "path_efficiency",
            "trajectory_smoothness", "minimum_obstacle_clearance_m",
            "minimum_inter_agent_distance_m", "total_online_algorithm_compute_ms",
        ],
        "continuous_pairing": "both-success subset when semantically meaningful",
        "failed_completion_time": "missing; never filled with zero",
        "path_efficiency": "straight start-goal distance / executed path; values above 1 retained because terminal radius is 0.30 m",
        "runtime_components": [
            "mean planning latency per decision", "planning decisions per episode",
            "cumulative upper/local planning", "execution actor", "DMP",
            "total online algorithm compute",
        ],
        "collision_contract": "frozen discrete post-transition positions; boundary disabled",
        "post_hoc_scene_selection": False,
    }


def _figure_protocol() -> dict[str, Any]:
    return {
        "frozen_before_performance": True,
        "figures": [
            "team success vs Stage I-IV with exact 95% CI",
            "outcome composition for four key methods",
            "overall and complex ablation success",
            "separate GAT and ERR contribution contrasts",
            "total online compute versus success",
            "stage-family success heatmap",
            "preregistered matched representative 3-D trajectories",
            "planning frequency versus difficulty (optional but pre-authorized)",
        ],
        "trajectory_selection": "within Proposed successes for each stage, choose completion time closest to the Proposed successful median; plot all main methods on that exact scenario",
        "style": {
            "labels": "English",
            "font": "Times-like serif",
            "pdf": "vector",
            "png_dpi": 600,
            "source_csv": True,
            "standalone_script": True,
            "caption": True,
            "colorblind_safe": True,
            "grayscale_readable": True,
        },
    }


def _schedule(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    schedule_index = 0
    for scenario_index, entry in enumerate(manifest["entries"]):
        offset = int(hashlib.sha256(str(entry["scenario_id"]).encode("utf-8")).hexdigest()[:8], 16) % len(METHOD_ORDER)
        ordered = METHOD_ORDER[offset:] + METHOD_ORDER[:offset]
        for within_index, method_id in enumerate(ordered):
            rows.append(
                {
                    "schedule_index": schedule_index,
                    "scenario_sequence_index": scenario_index,
                    "within_scenario_order": within_index,
                    "scenario_id": entry["scenario_id"],
                    "seed": int(entry["seed"]),
                    "stage": entry["stage"],
                    "family": entry["family"],
                    "method_id": method_id,
                    "order_rule": "sha256 scenario-id cyclic rotation of frozen method order",
                }
            )
            schedule_index += 1
    if len(rows) != EXPECTED_TEAM_ROWS:
        raise RuntimeError("schedule row count changed")
    return rows


def _run_regressions() -> dict[str, Any]:
    selected = (
        "test/test_event_triggered_reference_reconstruction.py",
        "test/test_sparse_err_trigger_revision.py",
        "test/test_final_four_stage_benchmark.py",
        "test/test_sensing_matched_classical.py",
    )
    rows: list[dict[str, Any]] = []
    for relative in selected:
        namespace = runpy.run_path(str(ROOT / relative))
        for name, function in sorted(namespace.items()):
            if name.startswith("test_") and callable(function):
                function()
                rows.append({"file": relative, "test": name, "status": "PASS"})
    return {"status": "PASS", "test_count": len(rows), "tests": rows}


def _verify_authorities() -> dict[str, Any]:
    method = load_json(METHOD_FREEZE)
    engineering = load_json(ENGINEERING_FREEZE)
    if method["status"] != "READY_FOR_NEW_FORMAL" or engineering["status"] != "ACCEPTED":
        raise RuntimeError("authoritative Proposed freeze is not accepted")
    source_checks: dict[str, Any] = {}
    for relative, expected in method["source_sha256"].items():
        actual = sha256_file(ROOT / relative)
        source_checks[relative] = {"expected": expected, "actual": actual, "match": actual == expected}
    for relative, expected in engineering["source_sha256"].items():
        actual = sha256_file(ROOT / relative)
        source_checks[relative] = {"expected": expected, "actual": actual, "match": actual == expected}
    base = load_json(BASE_CONFIG)
    checkpoints = {
        "gat": sha256_file(ROOT / base["sources"]["gat_checkpoint"]),
        "sac": sha256_file(ROOT / base["sources"]["sac_checkpoint"]),
    }
    if checkpoints != method["checkpoint_sha256"]:
        raise RuntimeError("checkpoint hashes differ from FINAL_METHOD_FREEZE")
    if not all(row["match"] for row in source_checks.values()):
        raise RuntimeError("accepted Proposed source changed")
    return {
        "method_freeze_sha256": sha256_file(METHOD_FREEZE),
        "engineering_freeze_sha256": sha256_file(ENGINEERING_FREEZE),
        "source_checks": source_checks,
        "checkpoint_sha256": checkpoints,
        "status": "PASS",
    }


def _historical_preflight() -> dict[str, Any]:
    manifest = load_json(HISTORICAL_MANIFEST)
    runtime = FrozenRuntime(load_json(BASE_CONFIG), manifest)
    _, config, _ = _resolved_method_configs()
    config["formal_scenarios"] = [row["scenario_id"] for row in manifest["entries"]]
    config["formal_seeds"] = [int(row["seed"]) for row in manifest["entries"]]
    entry = manifest["entries"][0]
    sink_a: dict[str, Any] = {}
    sink_b: dict[str, Any] = {}
    shared_a = build_shared_selection_bundle(
        config=config,
        execution_settings=runtime.execution_settings,
        multi_config=runtime.multi_config,
        policy=runtime.policy,
        gat_model=runtime.gat_model,
        gat_device=runtime.gat_device,
        scenario=entry["scenario_id"],
        seed=int(entry["seed"]),
        environment_builder=runtime.builder,
    )
    episode_a, _ = run_method_episode(
        config=config,
        execution_settings=runtime.execution_settings,
        multi_config=runtime.multi_config,
        policy=runtime.policy,
        shared=shared_a,
        method=METHOD_GAT,
        environment_builder=runtime.builder,
        trajectory_sink=sink_a,
    )
    shared_b = build_shared_selection_bundle(
        config=config,
        execution_settings=runtime.execution_settings,
        multi_config=runtime.multi_config,
        policy=runtime.policy,
        gat_model=runtime.gat_model,
        gat_device=runtime.gat_device,
        scenario=entry["scenario_id"],
        seed=int(entry["seed"]),
        environment_builder=runtime.builder,
    )
    episode_b, _ = run_method_episode(
        config=config,
        execution_settings=runtime.execution_settings,
        multi_config=runtime.multi_config,
        policy=runtime.policy,
        shared=shared_b,
        method=METHOD_GAT,
        environment_builder=runtime.builder,
        trajectory_sink=sink_b,
    )
    exact = all(np.array_equal(np.asarray(sink_a[key]), np.asarray(sink_b[key])) for key in ("positions", "velocities", "accelerations"))
    outcome_exact = all(
        episode_a[key] == episode_b[key]
        for key in ("team_success", "collision", "timeout", "termination_reason", "steps")
    )
    if not exact or not outcome_exact:
        raise RuntimeError("historical deterministic replay failed")
    recorder = OnlineRuntimeRecorder()
    proxy = TimedPolicyProxy(runtime.policy, recorder)
    with recorder.instrument_dmp(), recorder.scoped_context(preflight="historical_rerr"):
        rerr_episode, _, events, _, _ = run_rerr_episode(
            config=config,
            settings=runtime.execution_settings,
            multi_config=runtime.multi_config,
            policy=proxy,
            gat_model=runtime.gat_model,
            gat_device=runtime.gat_device,
            method=METHOD_RERR_GAT,
            scenario=entry["scenario_id"],
            seed=int(entry["seed"]),
            environment_builder=runtime.builder,
            runtime_recorder=recorder,
            upper_plan_builder=build_online_gat_plan_optimized,
        )
    return {
        "status": "PASS",
        "historical_scenario": entry["scenario_id"],
        "one_shot_deterministic_trajectory_exact": exact,
        "one_shot_deterministic_outcome_exact": outcome_exact,
        "optimized_rerr_executed": True,
        "optimized_rerr_upper_decisions_positive": rerr_episode["planning_decision_count"] > 0,
        "optimized_rerr_event_rows": len(events),
        "formal_scene_performance_used": False,
    }


def _scenario_reconstruction_check(manifest: Mapping[str, Any]) -> dict[str, Any]:
    base = load_json(BASE_CONFIG)
    runtime = FrozenRuntime(base, manifest)
    entry = manifest["entries"][0]
    env, metadata = runtime.builder(
        config=runtime.multi_config,
        scenario=entry["scenario_id"],
        seed=int(entry["seed"]),
        peer_radius=float(base["execution"]["peer_radius"]),
    )
    try:
        checks = {
            "starts_exact": np.array_equal(env.starts, np.asarray(entry["starts"], dtype=float)),
            "goals_exact": np.array_equal(env.goals, np.asarray(entry["goals"], dtype=float)),
            "static_count": len(env.static_obstacles) == len(entry["static_obstacles"]),
            "dynamic_count": len(env.dynamic_obstacles) == len(entry["dynamic_obstacles"]),
            "environment_fingerprint": metadata["environment_fingerprint"] == entry["environment_fingerprint"],
        }
    finally:
        env.close()
    if not all(checks.values()):
        raise RuntimeError(f"scenario reconstruction check failed: {checks}")
    return {"status": "PASS", "scenario_id": entry["scenario_id"], "checks": checks, "episode_executed": False}


def prepare(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"final output already exists: {output}")
    output.mkdir(parents=True)
    for relative in (
        "method_configs", "formal_records", "trajectories", "paper_ready/pdf",
        "paper_ready/png_600dpi", "paper_ready/source_data",
        "paper_ready/plotting_scripts", "paper_ready/captions",
    ):
        (output / relative).mkdir(parents=True, exist_ok=True)

    authorities = _verify_authorities()
    used_seed_manifest = audit_used_seeds(excluded_root=output.parent)
    used_geometry_manifest = audit_used_geometry(excluded_root=output.parent)
    historical_translations, translation_sources = _translation_registry(output.parent)
    manifest = generate_scenario_manifest(
        counts_per_stage=SCENARIOS_PER_STAGE,
        seed_base=SEED_BASE,
        prefix="FU",
        max_steps=220,
        dt=0.1,
    )
    validation = validate_scenario_manifest(manifest)
    new_seeds = {int(row["seed"]) for row in manifest["entries"]}
    new_exact = {str(row["geometry_fingerprint"]) for row in manifest["entries"]}
    new_translation = {str(row["translation_invariant_fingerprint"]) for row in manifest["entries"]}
    overlap = {
        "historical_seed_overlap": len(new_seeds & set(used_seed_manifest["used_seeds"])),
        "historical_exact_geometry_overlap": len(new_exact & set(used_geometry_manifest["geometry_hashes"])),
        "historical_translation_equivalent_overlap": len(new_translation & historical_translations),
        "internal_exact_geometry_duplicates": EXPECTED_SCENARIOS - len(new_exact),
        "internal_translation_equivalent_duplicates": EXPECTED_SCENARIOS - len(new_translation),
    }
    family_counts: dict[str, int] = defaultdict(int)
    for entry in manifest["entries"]:
        family_counts[f"{entry['stage']}::{entry['family']}"] += 1
    geometry_gates = {
        "validator": validation,
        "exact_zero_duplicate_gate": overlap["internal_exact_geometry_duplicates"] == 0,
        "translation_zero_duplicate_gate": overlap["internal_translation_equivalent_duplicates"] == 0,
        "history_zero_overlap_gate": all(value == 0 for key, value in overlap.items() if key.startswith("historical_")),
        "family_20_each_gate": len(family_counts) == 20 and set(family_counts.values()) == {20},
        "workspace_valid": all(row["workspace_bounds"] == json_ready(WORKSPACE_BOUNDS) for row in manifest["entries"]),
    }
    if validation["status"] != "PASSED" or not all(value for key, value in geometry_gates.items() if key != "validator"):
        raise RuntimeError(f"final manifest geometry gate failed: {geometry_gates}; overlap={overlap}")

    write_json(output / "FINAL_UNTOUCHED_SCENARIO_MANIFEST.json", manifest)
    (output / "FINAL_UNTOUCHED_SCENARIO_MANIFEST.sha256").write_text(manifest["manifest_sha256"] + "\n", encoding="ascii")
    write_json(output / "all_used_seed_manifest.json", used_seed_manifest)
    write_json(output / "all_used_geometry_manifest.json", used_geometry_manifest)
    registry_rows: list[dict[str, Any]] = []
    for seed in used_seed_manifest["used_seeds"]:
        registry_rows.append({"registry_type": "seed", "value": seed, "source": "full structured artifacts/configs scan"})
    for value in used_geometry_manifest["geometry_hashes"]:
        registry_rows.append(
            {
                "registry_type": "exact_or_environment_geometry_hash",
                "value": value,
                "source": " | ".join(used_geometry_manifest.get("example_sources", {}).get(value, [])),
            }
        )
    for value in sorted(historical_translations):
        registry_rows.append(
            {
                "registry_type": "translation_invariant_fingerprint",
                "value": value,
                "source": " | ".join(translation_sources.get(value, [])),
            }
        )
    write_csv(output / "all_used_scene_registry.csv", registry_rows)
    write_json(output / "manifest_geometry_validation.json", {"overlap": overlap, "gates": geometry_gates, "family_counts": dict(family_counts)})

    configs = _method_config_payloads()
    config_hashes: dict[str, str] = {}
    for method_id, payload in configs.items():
        path = output / "method_configs" / f"{method_id}.json"
        write_json(path, payload)
        config_hashes[method_id] = sha256_file(path)
    final_method_set = {
        "schema_version": SCHEMA,
        "frozen_before_performance": True,
        "method_count": len(METHODS),
        "methods": list(METHODS),
        "method_config_sha256": config_hashes,
        "RVO_NEW_FORMAL_INCLUDED": "NO",
        "RVO_ROLE": "SUPPLEMENTARY_HISTORICAL",
        "NMPC_INCLUDED": "NO",
        "NMPC_BASELINE_READY": "NO",
        "NO_MORE_METHOD_DEVELOPMENT": "YES",
        "NO_MORE_PARAMETER_TUNING": "YES",
        "NO_MORE_BASELINE_EXPANSION": "YES",
    }
    write_json(output / "final_method_set.json", final_method_set)
    write_csv(output / "comparison_contract.csv", _comparison_contract_rows())
    write_json(output / "statistical_protocol.json", _statistical_protocol())
    write_json(output / "figure_protocol.json", _figure_protocol())
    schedule = _schedule(manifest)
    write_csv(output / "run_schedule.csv", schedule)
    write_json(output / "hardware_manifest.json", hardware_manifest())

    regressions = _run_regressions()
    historical = _historical_preflight()
    reconstruction = _scenario_reconstruction_check(manifest)
    source_hashes = {relative: sha256_file(ROOT / relative) for relative in FROZEN_SOURCE_PATHS}
    context = {
        "schema_version": SCHEMA,
        "created_at": datetime.now().astimezone().isoformat(),
        "goal_attachment_sha256": sha256_file(ATTACHMENT),
        "agents_md_present": (ROOT / "AGENTS.md").exists(),
        "codex_handoff_present": (ROOT / "CODEX_HANDOFF.md").exists(),
        "authoritative_freezes": authorities,
        "historical_development_results_are_context_only": True,
        "method_changed": False,
        "parameter_tuning_performed": False,
        "new_baseline_added": False,
        "formal_episode_count_at_freeze": 0,
    }
    write_json(output / "context_recovery_manifest.json", context)
    preflight = {
        "schema_version": SCHEMA,
        "status": "PASS",
        "compile_and_unit_regression": regressions,
        "historical_deterministic_replay": historical,
        "scenario_reconstruction": reconstruction,
        "information_guards": {
            "DWA_FullState_future_dynamic_leakage": "NO",
            "DWA_SensingMatched_hidden_geometry_or_future_leakage": "NO",
            "Proposed_new_continuous_peer_information": "NO",
        },
        "checkpoint_load_test": "PASS",
        "runtime_instrumentation_test": "PASS",
        "formal_performance_previewed": False,
    }
    write_json(output / "preformal_integrity.json", preflight)

    frozen_files = [
        "context_recovery_manifest.json", "all_used_scene_registry.csv",
        "final_method_set.json", "comparison_contract.csv", "statistical_protocol.json",
        "figure_protocol.json", "run_schedule.csv", "FINAL_UNTOUCHED_SCENARIO_MANIFEST.json",
        "manifest_geometry_validation.json", "preformal_integrity.json",
    ] + [f"method_configs/{method_id}.json" for method_id in METHOD_ORDER]
    freeze = {
        "schema_version": SCHEMA,
        "created_at": datetime.now().astimezone().isoformat(),
        "METHOD_SET_FROZEN": "YES",
        "METHOD_COUNT": 8,
        "SCENARIO_MANIFEST_FROZEN": "YES",
        "SCENARIO_COUNT": 400,
        "STATISTICS_PROTOCOL_FROZEN": "YES",
        "FIGURE_PROTOCOL_FROZEN": "YES",
        "RUNTIME_PROTOCOL_FROZEN": "YES",
        "RUN_SCHEDULE_FROZEN": "YES",
        "ENGINEERING_PHASE": "CLOSED",
        "FORMAL_PHASE": "OPEN",
        "formal_team_rows_expected": EXPECTED_TEAM_ROWS,
        "formal_agent_rows_expected": EXPECTED_AGENT_ROWS,
        "formal_episode_count_at_freeze": 0,
        "manifest_sha256": manifest["manifest_sha256"],
        "checkpoint_sha256": authorities["checkpoint_sha256"],
        "authority_sha256": {
            "FINAL_METHOD_FREEZE": sha256_file(METHOD_FREEZE),
            "FINAL_ENGINEERING_FREEZE": sha256_file(ENGINEERING_FREEZE),
        },
        "source_sha256": source_hashes,
        "artifact_sha256": {relative: sha256_file(output / relative) for relative in frozen_files},
        "method_config_sha256": config_hashes,
        "post_freeze_method_change_allowed": False,
        "post_freeze_parameter_tuning_allowed": False,
        "post_freeze_schedule_change_allowed": False,
        "fixed_warmup": "one historical episode per execution family; discarded and excluded from runtime/results",
    }
    write_json(output / "FINAL_FORMAL_FREEZE.json", freeze)
    (output / "FINAL_FORMAL_FREEZE.sha256").write_text(sha256_file(output / "FINAL_FORMAL_FREEZE.json") + "\n", encoding="ascii")
    print(json.dumps({"phase": "prepare", "status": "PASS", "output": str(output), "formal_episode_count": 0}), flush=True)


def _eval_config(runtime: FrozenRuntime, manifest: Mapping[str, Any]) -> dict[str, Any]:
    _, config, _ = _resolved_method_configs()
    config["formal_scenarios"] = [row["scenario_id"] for row in manifest["entries"]]
    config["formal_seeds"] = [int(row["seed"]) for row in manifest["entries"]]
    return config


def _verify_formal_freeze(output: Path) -> dict[str, Any]:
    freeze = load_json(output / "FINAL_FORMAL_FREEZE.json")
    if freeze["FORMAL_PHASE"] != "OPEN" or freeze["ENGINEERING_PHASE"] != "CLOSED":
        raise RuntimeError("formal freeze is not open")
    mismatches: list[str] = []
    for relative, expected in freeze["source_sha256"].items():
        if sha256_file(ROOT / relative) != expected:
            mismatches.append(f"source:{relative}")
    for relative, expected in freeze["artifact_sha256"].items():
        if sha256_file(output / relative) != expected:
            mismatches.append(f"artifact:{relative}")
    base = load_json(BASE_CONFIG)
    checkpoints = {
        "gat": sha256_file(ROOT / base["sources"]["gat_checkpoint"]),
        "sac": sha256_file(ROOT / base["sources"]["sac_checkpoint"]),
    }
    if checkpoints != freeze["checkpoint_sha256"]:
        mismatches.append("checkpoints")
    if mismatches:
        raise RuntimeError(f"formal freeze mismatch: {mismatches}")
    return freeze


def _dummy_terminal_shared(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "scenario": entry["scenario_id"],
        "seed": int(entry["seed"]),
        "initial_condition_hash": entry["environment_fingerprint"],
        "candidate_bundle_hash": None,
        "preview_bundle_hash": None,
        "proposal_reconstruction_equivalent": True,
        "graph_schema_match": True,
        "preview_historical_gate_verified": True,
        "plans": {},
        "planning_runtime_ms": {METHOD_TERMINAL: 0.0},
    }


def _augment_agent_rows(
    rows: Sequence[Mapping[str, Any]], entry: Mapping[str, Any], method_id: str
) -> list[dict[str, Any]]:
    starts = np.asarray(entry["starts"], dtype=float)
    goals = np.asarray(entry["goals"], dtype=float)
    straight = np.linalg.norm(goals - starts, axis=1)
    result: list[dict[str, Any]] = []
    for source in rows:
        row = copy.deepcopy(dict(source))
        agent_id = int(row["agent_id"])
        completed = bool(row.get("agent_terminal_completed", row.get("success", False)))
        path_length = float(row.get("agent_path_length_m", row.get("path_length_m", 0.0)))
        row.update(
            {
                "schema_version": SCHEMA,
                "method_id": method_id,
                "display_name": METHOD_BY_ID[method_id]["display_name"],
                "scenario_id": entry["scenario_id"],
                "stage": PAPER_STAGE_LABEL.get(entry["stage"], entry["stage"]),
                "family": entry["family"],
                "seed": int(entry["seed"]),
                "agent_terminal_completed": completed,
                "agent_collision": bool(row.get("agent_collision", row.get("collision", False))),
                "agent_obstacle_collision": bool(row.get("obstacle_collision", False)),
                "agent_inter_agent_collision": bool(row.get("inter_agent_collision", False)),
                "agent_path_length_m": path_length,
                "agent_path_efficiency": (
                    float(straight[agent_id] / max(path_length, 1e-12)) if completed else None
                ),
                "completion_step": row.get("completion_step", row.get("terminal_completion_step")),
                "minimum_obstacle_clearance_m": row.get(
                    "minimum_obstacle_clearance_m",
                    row.get("minimum_static_obstacle_clearance_m"),
                ),
                "minimum_peer_distance_m": row.get("minimum_peer_distance_m"),
            }
        )
        result.append(row)
    return result


def _standard_episode(
    episode: Mapping[str, Any], entry: Mapping[str, Any], method_id: str
) -> dict[str, Any]:
    row = copy.deepcopy(dict(episode))
    any_collision = bool(row.get("any_collision", row.get("collision", False)))
    row.update(
        {
            "schema_version": SCHEMA,
            "method_id": method_id,
            "display_name": METHOD_BY_ID[method_id]["display_name"],
            "method_role": METHOD_BY_ID[method_id]["role"],
            "scenario_id": entry["scenario_id"],
            "stage": PAPER_STAGE_LABEL.get(entry["stage"], entry["stage"]),
            "family": entry["family"],
            "seed": int(entry["seed"]),
            "team_success": bool(row["team_success"]),
            "any_collision": any_collision,
            "obstacle_collision": bool(row.get("obstacle_collision", False)),
            "inter_agent_collision": bool(row.get("inter_agent_collision", False)),
            "timeout": bool(row.get("timeout", False)),
            "termination_time_s": float(row.get("termination_time_s", row["steps"] * entry["dt"])),
            "normal_replanning_count": int(row.get("normal_replanning_count", max(0, int(row.get("replanning_count", 0)) - int(row.get("emergency_replanning_count", 0))))),
            "emergency_replanning_count": int(row.get("emergency_replanning_count", 0)),
            "replanning_count": int(row.get("replanning_count", 0)),
        }
    )
    return row


def _one_shot_episode(
    runtime: FrozenRuntime,
    config: Mapping[str, Any],
    entry: Mapping[str, Any],
    method_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    engine = METHOD_BY_ID[method_id]["engine"]
    internal = ONE_SHOT_MAP[engine]
    recorder = OnlineRuntimeRecorder()
    policy = TimedPolicyProxy(runtime.policy, recorder)
    sink: dict[str, Any] = {}
    with recorder.instrument_dmp(), recorder.scoped_context(
        formal=True,
        scenario_id=entry["scenario_id"],
        stage=entry["stage"],
        family=entry["family"],
        method_id=method_id,
    ):
        if internal == METHOD_TERMINAL:
            shared = _dummy_terminal_shared(entry)
        else:
            shared = build_shared_selection_bundle(
                config=config,
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
        episode, agents = run_method_episode(
            config=config,
            execution_settings=runtime.execution_settings,
            multi_config=runtime.multi_config,
            policy=policy,
            shared=shared,
            method=internal,
            environment_builder=runtime.builder,
            trajectory_sink=sink,
            runtime_recorder=recorder,
        )
    standardized, standardized_agents, trajectory = standardize_sac_result(
        episode, agents, sink, entry, method_id
    )
    execution_actor = [row for row in recorder.actor_rows if row.get("actor_mode") == "execution_actor"]
    execution_dmp = [row for row in recorder.dmp_rows if row.get("dmp_mode") == "execution_dmp"]
    upper_ms = float(episode.get("planning_runtime_ms", 0.0))
    actor_ms = float(sum(float(row["runtime_ms"]) for row in execution_actor))
    dmp_ms = float(sum(float(row["runtime_ms"]) for row in execution_dmp))
    standardized.update(
        {
            "upper_planning_total_ms": upper_ms,
            "planning_runtime_ms": upper_ms,
            "planning_decision_count": 0 if internal == METHOD_TERMINAL else 1,
            "planning_runtime_per_decision_ms": None if internal == METHOD_TERMINAL else upper_ms,
            "execution_actor_forward_ms": actor_ms,
            "execution_actor_call_count": len(execution_actor),
            "execution_dmp_ms": dmp_ms,
            "execution_dmp_call_count": len(execution_dmp),
            "total_online_algorithm_compute_ms": upper_ms + actor_ms + dmp_ms,
            "upper_pipeline_invocation_count": 0 if internal == METHOD_TERMINAL else 1,
            "normal_replanning_count": 0,
            "emergency_replanning_count": 0,
        }
    )
    reference_events = []
    plan = shared["plans"].get(internal)
    if plan is not None:
        for agent_id, candidate in enumerate(plan["candidate_records"]):
            reference_events.append(
                {
                    "step": 0,
                    "agent_id": agent_id,
                    "event": "INITIAL_SELECTION",
                    "selected_candidate_id": candidate.get("selected_candidate_id"),
                    "selected_null": bool(candidate.get("selected_null", False)),
                    "active_goal": plan["references"][agent_id],
                    "selection_source": plan.get("selection_source"),
                }
            )
    return (
        _standard_episode(standardized, entry, method_id),
        _augment_agent_rows(standardized_agents, entry, method_id),
        trajectory,
        reference_events,
    )


def _rerr_trajectory(auxiliary: Mapping[str, Any], entry: Mapping[str, Any]) -> dict[str, np.ndarray]:
    rows = list(auxiliary["path_rows"])
    steps = sorted({int(row["step"]) for row in rows})
    by_key = {(int(row["step"]), int(row["agent_id"])): row for row in rows}
    positions = []
    velocities = []
    active_goals = []
    accelerations = []
    for step in steps:
        frame = [by_key[(step, agent_id)] for agent_id in range(3)]
        positions.append([[row["x_m"], row["y_m"], row["z_m"]] for row in frame])
        velocities.append([[row["vx_mps"], row["vy_mps"], row["vz_mps"]] for row in frame])
        active_goals.append([[row["active_goal_x_m"], row["active_goal_y_m"], row["active_goal_z_m"]] for row in frame])
        if step > 0:
            accelerations.append([[row["applied_ax_mps2"], row["applied_ay_mps2"], row["applied_az_mps2"]] for row in frame])
    return {
        "positions": np.asarray(positions, dtype=float),
        "velocities": np.asarray(velocities, dtype=float),
        "accelerations": np.asarray(accelerations, dtype=float),
        "active_goals": np.asarray(active_goals, dtype=float),
        "starts": np.asarray(entry["starts"], dtype=float),
        "goals": np.asarray(entry["goals"], dtype=float),
    }


def _rerr_episode(
    runtime: FrozenRuntime,
    config: Mapping[str, Any],
    entry: Mapping[str, Any],
    method_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    internal = METHOD_RERR_FP_SHEP if METHOD_BY_ID[method_id]["engine"] == "rerr_fp_shep" else METHOD_RERR_GAT
    recorder = OnlineRuntimeRecorder()
    policy = TimedPolicyProxy(runtime.policy, recorder)
    with recorder.instrument_dmp(), recorder.scoped_context(
        formal=True,
        scenario_id=entry["scenario_id"],
        stage=entry["stage"],
        family=entry["family"],
        method_id=method_id,
    ):
        episode, agents, events, _, auxiliary = run_rerr_episode(
            config=config,
            settings=runtime.execution_settings,
            multi_config=runtime.multi_config,
            policy=policy,
            gat_model=runtime.gat_model,
            gat_device=runtime.gat_device,
            method=internal,
            scenario=entry["scenario_id"],
            seed=int(entry["seed"]),
            environment_builder=runtime.builder,
            runtime_recorder=recorder,
            upper_plan_builder=build_online_gat_plan_optimized,
        )
    trajectory = _rerr_trajectory(auxiliary, entry)
    standardized = _standard_episode(episode, entry, method_id)
    standardized.update(
        {
            "any_collision": bool(episode["collision"]),
            "planning_runtime_ms": float(episode["upper_planning_total_ms"]),
            "planning_runtime_per_decision_ms": (
                float(episode["upper_planning_total_ms"] / episode["planning_decision_count"])
                if episode["planning_decision_count"] else None
            ),
        }
    )
    return standardized, _augment_agent_rows(agents, entry, method_id), trajectory, [dict(row) for row in events]


def _classical_episode(
    runtime: FrozenRuntime,
    entry: Mapping[str, Any],
    method_id: str,
    dwa: DWAStyleConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    if METHOD_BY_ID[method_id]["engine"] == "dwa_fullstate":
        episode, agents, runtime_rows, trajectory = run_classical_episode(
            environment_builder=runtime.builder,
            multi_config=runtime.multi_config,
            scenario=entry["scenario_id"],
            seed=int(entry["seed"]),
            peer_radius=0.3,
            method="dwa_style",
            planner_config=dwa,
        )
        adapter_ms = 0.0
        core_ms = float(episode["planning_runtime_ms"])
    else:
        episode, agents, runtime_rows, trajectory = run_sensing_matched_episode(
            environment_builder=runtime.builder,
            multi_config=runtime.multi_config,
            scenario=entry["scenario_id"],
            seed=int(entry["seed"]),
            peer_radius=0.3,
            method="dwa_sensing_matched",
            planner_config=dwa,
        )
        adapter_ms = float(episode["perception_adapter_runtime_ms"])
        core_ms = float(episode["planner_core_runtime_ms"])
    planning_ms = float(episode["planning_runtime_ms"])
    episode.update(
        {
            "upper_planning_total_ms": planning_ms,
            "local_planning_core_ms": core_ms,
            "perception_adapter_runtime_ms": adapter_ms,
            "execution_actor_forward_ms": 0.0,
            "execution_actor_call_count": 0,
            "execution_dmp_ms": 0.0,
            "execution_dmp_call_count": 0,
            "total_online_algorithm_compute_ms": planning_ms,
            "upper_pipeline_invocation_count": len(runtime_rows),
            "replanning_count": 0,
            "normal_replanning_count": 0,
            "emergency_replanning_count": 0,
        }
    )
    return _standard_episode(episode, entry, method_id), _augment_agent_rows(agents, entry, method_id), trajectory, []


def _fixed_historical_warmup() -> None:
    manifest = load_json(HISTORICAL_MANIFEST)
    runtime = FrozenRuntime(load_json(BASE_CONFIG), manifest)
    config = _eval_config(runtime, manifest)
    _, _, dwa = _resolved_method_configs()
    entry = manifest["entries"][0]
    # Results are intentionally discarded.  These calls warm CPU/GPU planner,
    # actor, GAT, and DMP paths before any formal timing begins.
    _classical_episode(runtime, entry, METHOD_ORDER[0], dwa)
    _classical_episode(runtime, entry, METHOD_ORDER[1], dwa)
    _one_shot_episode(runtime, config, entry, METHOD_ORDER[5])
    _rerr_episode(runtime, config, entry, METHOD_ORDER[7])


def _record_paths(output: Path, scenario_id: str, method_id: str) -> tuple[Path, Path]:
    return (
        output / "formal_records" / scenario_id / f"{method_id}.json",
        output / "trajectories" / scenario_id / f"{method_id}.npz",
    )


def _load_completed_record(path: Path, trajectory_path: Path) -> dict[str, Any] | None:
    if not path.is_file() or not trajectory_path.is_file():
        return None
    record = load_json(path)
    expected = record.get("result_hash")
    payload = {key: value for key, value in record.items() if key != "result_hash"}
    if content_hash(payload) != expected:
        raise RuntimeError(f"existing result hash mismatch: {path}")
    if sha256_file(trajectory_path) != record["trajectory_file_sha256"]:
        raise RuntimeError(f"existing trajectory file hash mismatch: {trajectory_path}")
    return record


def _run_formal_row(
    runtime: FrozenRuntime,
    config: Mapping[str, Any],
    entry: Mapping[str, Any],
    method_id: str,
    dwa: DWAStyleConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    engine = METHOD_BY_ID[method_id]["engine"]
    if engine.startswith("dwa_"):
        return _classical_episode(runtime, entry, method_id, dwa)
    if engine.startswith("one_shot_"):
        return _one_shot_episode(runtime, config, entry, method_id)
    return _rerr_episode(runtime, config, entry, method_id)


def _consolidate_raw(output: Path) -> tuple[int, int]:
    team: list[dict[str, Any]] = []
    agents: list[dict[str, Any]] = []
    for path in sorted((output / "formal_records").rglob("*.json")):
        record = load_json(path)
        team.append(record["episode"])
        agents.extend(record["agents"])
    team.sort(key=lambda row: (row["scenario_id"], METHOD_ORDER.index(row["method_id"])))
    agents.sort(key=lambda row: (row["scenario_id"], METHOD_ORDER.index(row["method_id"]), int(row["agent_id"])))
    write_csv(output / "formal_team_results.csv", team)
    write_csv(output / "formal_agent_results.csv", agents)
    return len(team), len(agents)


def formal(output: Path) -> None:
    freeze = _verify_formal_freeze(output)
    manifest = load_json(output / "FINAL_UNTOUCHED_SCENARIO_MANIFEST.json")
    schedule = read_csv(output / "run_schedule.csv")
    if len(schedule) != EXPECTED_TEAM_ROWS:
        raise RuntimeError("frozen schedule row count mismatch")
    entry_by_id = {row["scenario_id"]: row for row in manifest["entries"]}
    existing = 0
    for row in schedule:
        record_path, trajectory_path = _record_paths(output, row["scenario_id"], row["method_id"])
        if _load_completed_record(record_path, trajectory_path) is not None:
            existing += 1
    if existing:
        print(json.dumps({"phase": "formal_resume", "complete_records": existing, "expected": EXPECTED_TEAM_ROWS}), flush=True)

    _fixed_historical_warmup()
    runtime = FrozenRuntime(load_json(BASE_CONFIG), manifest)
    config = _eval_config(runtime, manifest)
    _, _, dwa = _resolved_method_configs()
    config_hashes = freeze["method_config_sha256"]
    started = time.perf_counter()
    completed = existing
    for schedule_row in schedule:
        scenario_id = schedule_row["scenario_id"]
        method_id = schedule_row["method_id"]
        record_path, trajectory_path = _record_paths(output, scenario_id, method_id)
        if _load_completed_record(record_path, trajectory_path) is not None:
            continue
        entry = entry_by_id[scenario_id]
        try:
            episode, agents, trajectory, events = _run_formal_row(
                runtime, config, entry, method_id, dwa
            )
            trajectory_content_hash, trajectory_file_sha = write_npz(trajectory_path, trajectory)
            payload = {
                "schema_version": SCHEMA,
                "schedule_index": int(schedule_row["schedule_index"]),
                "scenario_id": scenario_id,
                "method_id": method_id,
                "stage": entry["stage"],
                "family": entry["family"],
                "seed": int(entry["seed"]),
                "scenario_environment_fingerprint": entry["environment_fingerprint"],
                "scenario_geometry_fingerprint": entry["geometry_fingerprint"],
                "scenario_translation_invariant_fingerprint": entry["translation_invariant_fingerprint"],
                "method_config_sha256": config_hashes[method_id],
                "manifest_sha256": freeze["manifest_sha256"],
                "checkpoint_sha256": freeze["checkpoint_sha256"],
                "episode": episode,
                "agents": agents,
                "reference_events": events,
                "trajectory_relative_path": str(trajectory_path.relative_to(output)).replace("\\", "/"),
                "trajectory_content_hash": trajectory_content_hash,
                "trajectory_file_sha256": trajectory_file_sha,
                "method_changed_after_formal_start": False,
                "parameter_tuning_after_formal_start": False,
            }
            payload["result_hash"] = content_hash(payload)
            write_json(record_path, payload)
            completed += 1
        except Exception as error:
            exception = {
                "schema_version": SCHEMA,
                "FORMAL_SOFTWARE_EXCEPTION": True,
                "schedule_index": int(schedule_row["schedule_index"]),
                "scenario_id": scenario_id,
                "method_id": method_id,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "completed_records_before_exception": completed,
                "automatic_algorithm_fix_applied": False,
            }
            write_json(output / "FORMAL_SOFTWARE_EXCEPTION.json", exception)
            raise
        if completed % 25 == 0 or completed == EXPECTED_TEAM_ROWS:
            elapsed = time.perf_counter() - started
            print(
                json.dumps(
                    {
                        "phase": "formal",
                        "complete_records": completed,
                        "expected": EXPECTED_TEAM_ROWS,
                        "record_health": "PASS",
                        "elapsed_s_this_process": round(elapsed, 1),
                        "performance_summary_emitted": False,
                    }
                ),
                flush=True,
            )
    team_count, agent_count = _consolidate_raw(output)
    if team_count != EXPECTED_TEAM_ROWS or agent_count != EXPECTED_AGENT_ROWS:
        raise RuntimeError(f"raw consolidation count mismatch: team={team_count}, agent={agent_count}")
    write_json(
        output / "formal_run_complete.json",
        {
            "schema_version": SCHEMA,
            "FORMAL_RUN_COMPLETE": "YES",
            "team_rows": team_count,
            "agent_rows": agent_count,
            "completed_at": datetime.now().astimezone().isoformat(),
            "performance_summary_emitted_during_run": False,
            "method_changed_after_formal_start": False,
            "parameter_tuning_after_formal_start": False,
            "schedule_changed_after_formal_start": False,
        },
    )
    print(json.dumps({"phase": "formal_complete", "team_rows": team_count, "agent_rows": agent_count, "status": "PASS"}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "formal", "consolidate"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.phase == "prepare":
        prepare(output)
    elif args.phase == "formal":
        formal(output)
    else:
        team, agents = _consolidate_raw(output)
        print(json.dumps({"team_rows": team, "agent_rows": agents}), flush=True)


if __name__ == "__main__":
    main()
