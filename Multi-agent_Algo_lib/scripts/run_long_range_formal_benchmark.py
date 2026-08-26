"""Freeze and execute the untouched long-range eight-method benchmark.

``prepare`` closes development, writes the final method freeze, generates the
new 400-scene manifest, proves disjointness from prior manifests, and freezes
all executable identities before any formal episode.  ``run`` is resumable and
never computes performance aggregates.  ``finalize`` only materializes raw
team/agent CSV files after all 3,200 records exist; statistical analysis lives
in a separate post-formal script.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as _pandas  # noqa: F401  # initialize Windows pyarrow before torch


ROOT = Path(__file__).resolve().parents[2]
ALGO = Path(__file__).resolve().parents[1]
for search_path in (ROOT, ALGO):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from planning.long_range_collision_recheck import audit_trajectory_collisions  # noqa: E402
from planning.online_runtime_instrumentation import (  # noqa: E402
    OnlineRuntimeRecorder,
    TimedPolicyProxy,
)
from planning.semi_structured_long_range_benchmark import (  # noqa: E402
    FAMILY_ORDER,
    STAGE_ORDER,
    generate_scenario_manifest,
    json_ready,
    stable_hash,
    validate_scenario_manifest,
)
from planning.final_four_stage_benchmark import run_classical_episode  # noqa: E402
from planning.sensing_matched_classical import run_sensing_matched_episode  # noqa: E402
from scripts.evaluate_gat_v1_err_development import (  # noqa: E402
    METHOD_RERR_FP_SHEP,
    METHOD_RERR_GAT,
    build_online_gat_plan_optimized,
    run_episode as run_rerr_episode,
)
from scripts.run_long_range_ablation_development import (  # noqa: E402
    DevelopmentRuntime,
    evaluate_episode as evaluate_ablation_episode,
    method_contract,
    resolved_config as resolve_ablation_config,
)
from scripts.run_long_range_classical_development import (  # noqa: E402
    build_runtime as build_classical_runtime,
    load_config as load_classical_config,
    planner_config,
)


SCHEMA = "semi_structured_long_range_formal_benchmark_v1"
ARTIFACT_ROOT = ROOT / "artifacts/semi_structured_long_range_main_benchmark/20260820_193228"
FREEZE_DIR = ARTIFACT_ROOT / "10_final_freeze"
MANIFEST_DIR = ARTIFACT_ROOT / "11_formal_manifest"
RECORD_DIR = ARTIFACT_ROOT / "12_formal_records"
FINAL_METHOD_FREEZE = FREEZE_DIR / "FINAL_LONG_RANGE_METHOD_FREEZE.json"
FORMAL_RUN_FREEZE = FREEZE_DIR / "FORMAL_RUN_FREEZE.json"
FORMAL_MANIFEST = MANIFEST_DIR / "FINAL_LONG_RANGE_MANIFEST.json"
FORMAL_REGISTRY = MANIFEST_DIR / "ALL_USED_SCENE_REGISTRY.csv"
DEVELOPMENT_SELECTION_FREEZE = FREEZE_DIR / "DEVELOPMENT_SELECTION_FREEZE.json"
SHORT_COORDINATION_RECONCILIATION = (
    ARTIFACT_ROOT / "02_short_coordination/short_coordination_reconciliation.json"
)
FORMAL_ENGINE_PREFLIGHT = FREEZE_DIR / "formal_engine_preflight.json"
SEED_BASE = 2_600_000_000
SCENARIOS_PER_STAGE = 100
EXPECTED_SCENARIOS = 400


METHODS: tuple[dict[str, str], ...] = (
    {
        "method_id": "M1_DWA_FullState",
        "display_name": "DWA-FullState",
        "engine": "dwa_fullstate",
        "role": "STRONG_SYSTEM_REFERENCE",
        "config": "configs/evaluation/long_range_dwa_fs_full_s00.json",
    },
    {
        "method_id": "M2_DWA_SensingMatched",
        "display_name": "DWA-SensingMatched",
        "engine": "dwa_sensing_matched",
        "role": "EQUAL_INFORMATION_CLASSICAL_BASELINE",
        "config": "configs/evaluation/long_range_dwa_sm_full_s05.json",
    },
    {
        "method_id": "M4_Direct_SAC_DMP",
        "display_name": "Direct SAC-DMP",
        "engine": "one_shot_ablation",
        "role": "LEARNING_BASELINE",
        "config": "configs/evaluation/semi_structured_long_range_ablation_direct_sac.json",
    },
    {
        "method_id": "M5_Proposal_SAC_DMP",
        "display_name": "Proposal + SAC-DMP",
        "engine": "one_shot_ablation",
        "role": "REFERENCE_GENERATION_ABLATION",
        "config": "configs/evaluation/semi_structured_long_range_ablation_proposal.json",
    },
    {
        "method_id": "M6_FP_SHEP_SAC_DMP",
        "display_name": "FP-SHEP + SAC-DMP",
        "engine": "one_shot_ablation",
        "role": "EXECUTION_AWARE_PREVIEW_ABLATION",
        "config": "configs/evaluation/semi_structured_long_range_ablation_fp_shep.json",
    },
    {
        "method_id": "M7_OneShot_GAT_SAC_DMP",
        "display_name": "One-Shot GAT + SAC-DMP",
        "engine": "one_shot_ablation",
        "role": "NO_RECURRENT_RECONSTRUCTION_ABLATION",
        "config": "configs/evaluation/semi_structured_long_range_ablation_one_shot_gat.json",
    },
    {
        "method_id": "M8_RERR_FP_SHEP_SAC_DMP",
        "display_name": "R-ERR + FP-SHEP + SAC-DMP",
        "engine": "rerr_fp_shep",
        "role": "GAT_ABLATION_UNDER_IDENTICAL_RERR",
        "config": "configs/evaluation/semi_structured_long_range_rerr_fp_f00_full200.json",
    },
    {
        "method_id": "M9_Proposed_RERR_GAT_SAC_DMP",
        "display_name": "Proposed R-ERR + GAT + SAC-DMP",
        "engine": "rerr_gat",
        "role": "FINAL_PROPOSED_METHOD",
        "config": "configs/evaluation/semi_structured_long_range_development_d05.json",
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

DEVELOPMENT_EVIDENCE = {
    "M1_DWA_FullState": "09_baseline_tuning/DWA_FS_FULL_S00/development_reconciliation.json",
    "M2_DWA_SensingMatched": "09_baseline_tuning/DWA_SM_FULL_S05/development_reconciliation.json",
    "M4_Direct_SAC_DMP": "08_development/ABL_M4_Direct_SAC_DMP_full200/development_reconciliation.json",
    "M5_Proposal_SAC_DMP": "08_development/ABL_M5_Proposal_SAC_DMP_full200/development_reconciliation.json",
    "M6_FP_SHEP_SAC_DMP": "08_development/ABL_M6_FP_SHEP_SAC_DMP_full200/development_reconciliation.json",
    "M7_OneShot_GAT_SAC_DMP": "08_development/ABL_M7_OneShot_GAT_SAC_DMP_full200/development_reconciliation.json",
    "M8_RERR_FP_SHEP_SAC_DMP": "08_development/RERR_FP_F00_base_common_rerr_full200/development_reconciliation.json",
    "M9_Proposed_RERR_GAT_SAC_DMP": "08_development/D05_interaction_feasibility_mask_full200/development_reconciliation.json",
}

SOURCE_PATHS = (
    "Environment/frozen_sac_dmp_execution.py",
    "Environment/multi_agent_dmp_env.py",
    "Environment/single_agent_dmp_env.py",
    "Entity/KinematicModel.py",
    "planning/event_triggered_reference_reconstruction.py",
    "planning/final_four_stage_benchmark.py",
    "planning/heterogeneous_candidate_graph.py",
    "planning/historical_forcing_gate.py",
    "planning/long_range_collision_recheck.py",
    "planning/online_runtime_instrumentation.py",
    "planning/policy_preview.py",
    "planning/pre_gat_closed_loop.py",
    "planning/semi_structured_long_range_benchmark.py",
    "planning/sensing_matched_classical.py",
    "Multi-agent_Algo_lib/scripts/evaluate_gat_closed_loop.py",
    "Multi-agent_Algo_lib/scripts/evaluate_gat_v1_err_development.py",
    "Multi-agent_Algo_lib/scripts/finalize_long_range_development.py",
    "Multi-agent_Algo_lib/scripts/run_short_horizon_coordination_final.py",
    "Multi-agent_Algo_lib/scripts/run_long_range_ablation_development.py",
    "Multi-agent_Algo_lib/scripts/run_long_range_classical_development.py",
    "Multi-agent_Algo_lib/scripts/run_long_range_contract_pilot.py",
    "Multi-agent_Algo_lib/scripts/run_long_range_development.py",
    "Multi-agent_Algo_lib/scripts/run_long_range_formal_benchmark.py",
    "Multi-agent_Algo_lib/scripts/reconcile_long_range_formal_benchmark.py",
    "Multi-agent_Algo_lib/scripts/analyze_long_range_formal_benchmark.py",
    "Multi-agent_Algo_lib/scripts/plot_long_range_formal_benchmark.py",
    "configs/evaluation/short_horizon_coordination_final.json",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_hash(value: Any) -> str:
    payload = json.dumps(
        json_ready(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_ready(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if str(key) not in fields:
                fields.append(str(key))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(json_ready(row.get(key)), ensure_ascii=False, separators=(",", ":"))
                    if isinstance(row.get(key), (dict, list, tuple, np.ndarray))
                    else row.get(key)
                    for key in fields
                }
            )
    temporary.replace(path)


def write_npz(path: Path, arrays: Mapping[str, Any]) -> tuple[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    normalized = {key: np.asarray(value) for key, value in arrays.items()}
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **normalized)
    temporary.replace(path)
    digest = hashlib.sha256()
    for key in sorted(normalized):
        array = np.ascontiguousarray(normalized[key])
        digest.update(key.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest(), sha256_file(path)


def resolved_method_config(method: Mapping[str, str]) -> dict[str, Any]:
    path = ROOT / method["config"]
    if method["engine"].startswith("dwa_"):
        return load_classical_config(path)
    if method["engine"] == "one_shot_ablation":
        return resolve_ablation_config(path)
    return load_json(path)


def _start_goal_fingerprint(entry: Mapping[str, Any]) -> str:
    return stable_hash({"starts": entry["starts"], "goals": entry["goals"]})


def _extract_scene_entries(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Mapping):
        return []
    entries = value.get("entries")
    if not isinstance(entries, list):
        return []
    return [row for row in entries if isinstance(row, Mapping) and "scenario_id" in row]


def historical_scene_registry() -> tuple[list[dict[str, Any]], dict[str, set[Any]]]:
    """Read all prior manifest-like JSONs without opening episode outcomes."""

    sets: dict[str, set[Any]] = {
        "seed": set(),
        "geometry_fingerprint": set(),
        "dynamic_track_fingerprint": set(),
        "translation_invariant_fingerprint": set(),
        "start_goal_fingerprint": set(),
    }
    rows: list[dict[str, Any]] = []
    candidates = sorted(
        path
        for path in (ROOT / "artifacts").rglob("*.json")
        if "manifest" in path.name.lower()
        and path.resolve() != FORMAL_MANIFEST.resolve()
    )
    for path in candidates:
        try:
            payload = load_json(path)
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        for entry in _extract_scene_entries(payload):
            seed = entry.get("seed")
            geometry = entry.get("geometry_fingerprint")
            dynamic = entry.get("dynamic_track_fingerprint")
            translation = entry.get("translation_invariant_fingerprint")
            start_goal = (
                _start_goal_fingerprint(entry)
                if "starts" in entry and "goals" in entry
                else None
            )
            values = {
                "seed": int(seed) if seed is not None else None,
                "geometry_fingerprint": geometry,
                "dynamic_track_fingerprint": dynamic,
                "translation_invariant_fingerprint": translation,
                "start_goal_fingerprint": start_goal,
            }
            for key, item in values.items():
                if item is not None:
                    sets[key].add(item)
            rows.append(
                {
                    "scene_id": entry.get("scenario_id"),
                    "split": "historical",
                    "source": str(path.relative_to(ROOT)).replace("\\", "/"),
                    **values,
                }
            )
    return rows, sets


def development_gate() -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    if not DEVELOPMENT_SELECTION_FREEZE.is_file():
        raise FileNotFoundError("development selection freeze is missing")
    selection = load_json(DEVELOPMENT_SELECTION_FREEZE)
    if selection.get("status") != "DEVELOPMENT_CLOSED_BEFORE_SHORT_AND_FORMAL":
        raise RuntimeError("development selection freeze is not closed")
    if bool(selection.get("formal_data_used_for_training_tuning_or_selection", True)):
        raise RuntimeError("development selection freeze reports formal-data contamination")
    evidence["DEVELOPMENT_SELECTION_FREEZE"] = {
        "path": str(DEVELOPMENT_SELECTION_FREEZE.relative_to(ARTIFACT_ROOT)).replace("\\", "/"),
        "sha256": sha256_file(DEVELOPMENT_SELECTION_FREEZE),
        "development_90_percent_target_met": selection["development_90_percent_target_met"],
        "matched_baseline_target_met": selection["proposed_matches_or_exceeds_strongest_matched_baseline"],
    }
    for method_id, relative in DEVELOPMENT_EVIDENCE.items():
        path = ARTIFACT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing development reconciliation: {relative}")
        payload = load_json(path)
        if payload.get("status") != "PASS" or int(payload.get("completed_scenarios", 0)) != 200:
            raise RuntimeError(f"development arm is not complete: {method_id}: {payload}")
        if int(payload.get("software_error_count", 0)) != 0:
            raise RuntimeError(f"development arm has software errors: {method_id}")
        evidence[method_id] = {
            "path": relative,
            "sha256": sha256_file(path),
            "overall": payload.get("overall"),
        }
    ppo_path = ARTIFACT_ROOT / "09_baseline_tuning/PPO_BASELINE_AUDIT.json"
    ppo = load_json(ppo_path)
    if ppo.get("PPO_BASELINE_READY") != "NO":
        raise RuntimeError("PPO gate changed; frozen method set must be reconsidered")
    evidence["M3_PPO_EXCLUSION"] = {
        "path": str(ppo_path.relative_to(ARTIFACT_ROOT)).replace("\\", "/"),
        "sha256": sha256_file(ppo_path),
        "PPO_BASELINE_READY": "NO",
    }
    if not SHORT_COORDINATION_RECONCILIATION.is_file():
        raise FileNotFoundError("final-method short-horizon coordination revalidation is missing")
    short = load_json(SHORT_COORDINATION_RECONCILIATION)
    if short.get("status") != "PASS" or int(short.get("completed_scenario_count", 0)) != 100:
        raise RuntimeError("short-horizon coordination revalidation is incomplete")
    if not bool(short.get("all_external_obstacle_counts_zero", False)):
        raise RuntimeError("short-horizon coordination contract contains external obstacles")
    if bool(short.get("used_for_long_range_method_selection", True)):
        raise RuntimeError("short-horizon revalidation was used for method selection")
    if bool(short.get("long_range_formal_data_used", True)):
        raise RuntimeError("short-horizon revalidation reports formal-data contamination")
    evidence["SHORT_COORDINATION_REVALIDATION"] = {
        "path": str(SHORT_COORDINATION_RECONCILIATION.relative_to(ARTIFACT_ROOT)).replace("\\", "/"),
        "sha256": sha256_file(SHORT_COORDINATION_RECONCILIATION),
        "overall": short.get("overall"),
    }
    return evidence


def prepare() -> None:
    existing_records = list(RECORD_DIR.rglob("*.json"))
    if existing_records:
        raise RuntimeError("formal records already exist; prepare is irreversible")
    FREEZE_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    RECORD_DIR.mkdir(parents=True, exist_ok=True)

    evidence = development_gate()
    if not FORMAL_ENGINE_PREFLIGHT.is_file():
        raise FileNotFoundError("formal engine preflight is missing")
    preflight = load_json(FORMAL_ENGINE_PREFLIGHT)
    if preflight.get("status") != "PASS" or int(preflight.get("method_count", 0)) != len(METHODS):
        raise RuntimeError("formal engine preflight did not pass for every frozen method")
    if bool(preflight.get("formal_data_used", True)):
        raise RuntimeError("formal engine preflight reports formal-data contamination")
    evidence["FORMAL_ENGINE_PREFLIGHT"] = {
        "path": str(FORMAL_ENGINE_PREFLIGHT.relative_to(ARTIFACT_ROOT)).replace("\\", "/"),
        "sha256": sha256_file(FORMAL_ENGINE_PREFLIGHT),
        "method_count": preflight["method_count"],
    }
    method_configs: dict[str, dict[str, Any]] = {}
    method_config_hashes: dict[str, str] = {}
    (FREEZE_DIR / "method_configs").mkdir(parents=True, exist_ok=True)
    for method in METHODS:
        payload = resolved_method_config(method)
        path = FREEZE_DIR / "method_configs" / f"{method['method_id']}.json"
        atomic_json(path, payload)
        method_configs[method["method_id"]] = payload
        method_config_hashes[method["method_id"]] = sha256_file(path)

    checkpoint_paths = {
        "sac": method_configs["M9_Proposed_RERR_GAT_SAC_DMP"]["sac_checkpoint"],
        "gat": method_configs["M9_Proposed_RERR_GAT_SAC_DMP"]["gat_checkpoint"],
    }
    checkpoint_hashes = {
        key: sha256_file(ROOT / relative) for key, relative in checkpoint_paths.items()
    }
    proposed = method_configs["M9_Proposed_RERR_GAT_SAC_DMP"]
    if checkpoint_hashes["sac"] != proposed["sac_checkpoint_sha256_expected"]:
        raise RuntimeError("SAC checkpoint identity mismatch")
    if checkpoint_hashes["gat"] != proposed["gat_checkpoint_sha256_expected"]:
        raise RuntimeError("GAT checkpoint identity mismatch")

    final_method = {
        "schema_version": SCHEMA,
        "status": "FROZEN_BEFORE_FORMAL_MANIFEST",
        "created_at": datetime.now().astimezone().isoformat(),
        "final_method": proposed.get(
            "final_method_description",
            "Proposal + FP-SHEP H4 + GAT-V1 + R-ERR + adapted 256-ray SAC-DMP",
        ),
        "method_set": list(METHODS),
        "method_count": len(METHODS),
        "PPO_INCLUDED": "NO_NOT_READY",
        "NMPC_INCLUDED": "NO_DEFAULT_EXCLUSION",
        "RVO_INCLUDED": "NO_SUPPLEMENTARY_ONLY",
        "proposed_configuration_id": proposed["configuration_id"],
        "development_evidence": evidence,
        "method_config_sha256": method_config_hashes,
        "checkpoint_paths": checkpoint_paths,
        "checkpoint_sha256": checkpoint_hashes,
        "formal_manifest_created_at_freeze": False,
        "formal_episode_count_at_freeze": 0,
        "development_closed": True,
        "post_freeze_tuning_allowed": False,
    }
    atomic_json(FINAL_METHOD_FREEZE, final_method)

    historical_rows, used = historical_scene_registry()
    manifest = generate_scenario_manifest(
        counts_per_stage=SCENARIOS_PER_STAGE,
        seed_base=SEED_BASE,
        prefix="FORMAL_LR_",
    )
    validation = validate_scenario_manifest(manifest)
    if validation.get("status") != "PASS":
        raise RuntimeError(f"formal manifest validation failed: {validation}")
    formal_sets = {
        "seed": {int(row["seed"]) for row in manifest["entries"]},
        "geometry_fingerprint": {str(row["geometry_fingerprint"]) for row in manifest["entries"]},
        "dynamic_track_fingerprint": {str(row["dynamic_track_fingerprint"]) for row in manifest["entries"]},
        "translation_invariant_fingerprint": {
            str(row["translation_invariant_fingerprint"]) for row in manifest["entries"]
        },
        "start_goal_fingerprint": {_start_goal_fingerprint(row) for row in manifest["entries"]},
    }
    overlap = {key: len(formal_sets[key] & used[key]) for key in formal_sets}
    internal_duplicates = {
        key: EXPECTED_SCENARIOS - len(values) for key, values in formal_sets.items()
    }
    if any(overlap.values()) or any(internal_duplicates.values()):
        raise RuntimeError(
            f"formal manifest disjointness gate failed: overlap={overlap}, internal={internal_duplicates}"
        )
    atomic_json(FORMAL_MANIFEST, manifest)
    atomic_json(
        MANIFEST_DIR / "formal_manifest_validation.json",
        {
            "status": "PASS",
            "validator": validation,
            "historical_manifest_file_count": len(
                {row["source"] for row in historical_rows}
            ),
            "historical_scene_row_count": len(historical_rows),
            "historical_overlap": overlap,
            "internal_duplicates": internal_duplicates,
            "formal_performance_previewed": False,
        },
    )
    formal_rows = [
        {
            "scene_id": row["scenario_id"],
            "split": "formal_untouched",
            "source": str(FORMAL_MANIFEST.relative_to(ROOT)).replace("\\", "/"),
            "seed": int(row["seed"]),
            "geometry_fingerprint": row["geometry_fingerprint"],
            "dynamic_track_fingerprint": row["dynamic_track_fingerprint"],
            "translation_invariant_fingerprint": row["translation_invariant_fingerprint"],
            "start_goal_fingerprint": _start_goal_fingerprint(row),
        }
        for row in manifest["entries"]
    ]
    write_csv(FORMAL_REGISTRY, [*historical_rows, *formal_rows])

    schedule: list[dict[str, Any]] = []
    for scenario_index, entry in enumerate(manifest["entries"]):
        offset = int(hashlib.sha256(entry["scenario_id"].encode("utf-8")).hexdigest()[:8], 16) % len(METHOD_ORDER)
        rotated = METHOD_ORDER[offset:] + METHOD_ORDER[:offset]
        for within, method_id in enumerate(rotated):
            schedule.append(
                {
                    "schedule_index": len(schedule),
                    "scenario_sequence_index": scenario_index,
                    "within_scenario_order": within,
                    "scenario_id": entry["scenario_id"],
                    "seed": int(entry["seed"]),
                    "stage": entry["stage"],
                    "family": entry["family"],
                    "method_id": method_id,
                }
            )
    write_csv(MANIFEST_DIR / "formal_schedule.csv", schedule)

    source_hashes = {relative: sha256_file(ROOT / relative) for relative in SOURCE_PATHS}
    frozen_artifacts = {
        str(path.relative_to(ARTIFACT_ROOT)).replace("\\", "/"): sha256_file(path)
        for path in (
            FINAL_METHOD_FREEZE,
            FORMAL_MANIFEST,
            MANIFEST_DIR / "formal_manifest_validation.json",
            MANIFEST_DIR / "formal_schedule.csv",
            FORMAL_REGISTRY,
        )
    }
    formal_freeze = {
        "schema_version": SCHEMA,
        "status": "OPEN_UNTOUCHED_FORMAL",
        "created_at": datetime.now().astimezone().isoformat(),
        "scenario_count": EXPECTED_SCENARIOS,
        "method_count": len(METHODS),
        "expected_team_rows": EXPECTED_SCENARIOS * len(METHODS),
        "expected_agent_rows": EXPECTED_SCENARIOS * len(METHODS) * 3,
        "manifest_semantic_sha256": manifest["manifest_sha256"],
        "method_config_sha256": method_config_hashes,
        "checkpoint_sha256": checkpoint_hashes,
        "source_sha256": source_hashes,
        "frozen_artifact_sha256": frozen_artifacts,
        "formal_episode_count_at_freeze": 0,
        "performance_aggregates_allowed_during_run": False,
        "post_freeze_tuning_allowed": False,
    }
    atomic_json(FORMAL_RUN_FREEZE, formal_freeze)
    print(
        json.dumps(
            {
                "phase": "prepare",
                "status": "PASS",
                "scenario_count": EXPECTED_SCENARIOS,
                "method_count": len(METHODS),
                "historical_overlap": overlap,
                "formal_episode_count": 0,
            }
        ),
        flush=True,
    )


def verify_formal_freeze() -> dict[str, Any]:
    freeze = load_json(FORMAL_RUN_FREEZE)
    if freeze.get("status") != "OPEN_UNTOUCHED_FORMAL":
        raise RuntimeError("formal freeze is not open")
    mismatches: list[str] = []
    for relative, expected in freeze["source_sha256"].items():
        if sha256_file(ROOT / relative) != expected:
            mismatches.append(f"source:{relative}")
    for relative, expected in freeze["frozen_artifact_sha256"].items():
        if sha256_file(ARTIFACT_ROOT / relative) != expected:
            mismatches.append(f"artifact:{relative}")
    for method_id, expected in freeze["method_config_sha256"].items():
        path = FREEZE_DIR / "method_configs" / f"{method_id}.json"
        if sha256_file(path) != expected:
            mismatches.append(f"method_config:{method_id}")
    if mismatches:
        raise RuntimeError(f"formal freeze mismatch: {mismatches}")
    return freeze


def _rerr_trajectory(extra: Mapping[str, Any], entry: Mapping[str, Any]) -> dict[str, np.ndarray]:
    rows = list(extra["path_rows"])
    steps = sorted({int(row["step"]) for row in rows})
    by_key = {(int(row["step"]), int(row["agent_id"])): row for row in rows}
    positions: list[list[list[float]]] = []
    velocities: list[list[list[float]]] = []
    active_goals: list[list[list[float]]] = []
    accelerations: list[list[list[float]]] = []
    for step in steps:
        frame = [by_key[(step, agent_id)] for agent_id in range(3)]
        positions.append([[row["x_m"], row["y_m"], row["z_m"]] for row in frame])
        velocities.append([[row["vx_mps"], row["vy_mps"], row["vz_mps"]] for row in frame])
        active_goals.append(
            [[row["active_goal_x_m"], row["active_goal_y_m"], row["active_goal_z_m"]] for row in frame]
        )
        if step > 0:
            accelerations.append(
                [[row["applied_ax_mps2"], row["applied_ay_mps2"], row["applied_az_mps2"]] for row in frame]
            )
    return {
        "positions": np.asarray(positions, dtype=float),
        "velocities": np.asarray(velocities, dtype=float),
        "accelerations": np.asarray(accelerations, dtype=float),
        "active_goals": np.asarray(active_goals, dtype=float),
        "starts": np.asarray(entry["starts"], dtype=float),
        "goals": np.asarray(entry["goals"], dtype=float),
    }


def _standardize(
    episode: Mapping[str, Any],
    agents: Sequence[Mapping[str, Any]],
    trajectory: Mapping[str, Any],
    entry: Mapping[str, Any],
    method: Mapping[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    row = copy.deepcopy(dict(episode))
    row.update(
        {
            "schema_version": SCHEMA,
            "method_id": method["method_id"],
            "display_name": method["display_name"],
            "method_role": method["role"],
            "scenario_id": entry["scenario_id"],
            "scenario": entry["scenario_id"],
            "seed": int(entry["seed"]),
            "stage": PAPER_STAGE_LABEL.get(entry["stage"], entry["stage"]),
            "family": entry["family"],
            "task_pattern": entry["task_pattern"],
            "environment_fingerprint": entry["environment_fingerprint"],
            "geometry_fingerprint": entry["geometry_fingerprint"],
            "dynamic_track_fingerprint": entry["dynamic_track_fingerprint"],
            "team_success": bool(row.get("team_success", row.get("success", False))),
            "any_collision": bool(row.get("any_collision", row.get("collision", False))),
            "obstacle_collision": bool(row.get("obstacle_collision", False)),
            "inter_agent_collision": bool(row.get("inter_agent_collision", False)),
            "timeout": bool(row.get("timeout", row.get("truncated", False))),
            "steps": int(row["steps"]),
            "termination_time_s": float(row.get("completion_time_s") or int(row["steps"]) * float(entry["dt"])),
        }
    )
    collision = audit_trajectory_collisions(np.asarray(trajectory["positions"]), entry)
    online_triplet = (
        row["obstacle_collision"],
        row["inter_agent_collision"],
        row["any_collision"],
    )
    replay_triplet = (
        collision["obstacle_collision"],
        collision["inter_agent_collision"],
        collision["any_collision"],
    )
    if online_triplet != replay_triplet:
        raise RuntimeError(
            f"collision replay mismatch {entry['scenario_id']} {method['method_id']}: online={online_triplet}, replay={replay_triplet}"
        )
    row.update(collision)
    standardized_agents: list[dict[str, Any]] = []
    starts = np.asarray(entry["starts"], dtype=float)
    goals = np.asarray(entry["goals"], dtype=float)
    for source in agents:
        agent = copy.deepcopy(dict(source))
        agent_id = int(agent["agent_id"])
        completed = bool(agent.get("agent_terminal_completed", agent.get("success", False)))
        path_length = float(agent.get("agent_path_length_m", agent.get("path_length_m", 0.0)))
        agent.update(
            {
                "schema_version": SCHEMA,
                "method_id": method["method_id"],
                "display_name": method["display_name"],
                "scenario_id": entry["scenario_id"],
                "seed": int(entry["seed"]),
                "stage": PAPER_STAGE_LABEL.get(entry["stage"], entry["stage"]),
                "family": entry["family"],
                "agent_terminal_completed": completed,
                "agent_path_length_m": path_length,
                "agent_path_efficiency": (
                    float(np.linalg.norm(goals[agent_id] - starts[agent_id]) / max(path_length, 1.0e-12))
                    if completed
                    else None
                ),
                "agent_static_obstacle_collision": collision["agent_static_obstacle_collision"][agent_id],
                "agent_dynamic_obstacle_collision": collision["agent_dynamic_obstacle_collision"][agent_id],
                "agent_obstacle_collision": collision["agent_obstacle_collision"][agent_id],
                "agent_inter_agent_collision": collision["agent_inter_agent_collision"][agent_id],
                "agent_boundary_collision": collision["agent_boundary_collision"][agent_id],
                "agent_any_collision": collision["agent_any_collision"][agent_id],
            }
        )
        standardized_agents.append(agent)
    completed_efficiencies = [
        float(agent["agent_path_efficiency"])
        for agent in standardized_agents
        if agent["agent_path_efficiency"] is not None
    ]
    row["team_path_efficiency"] = (
        float(np.mean(completed_efficiencies)) if completed_efficiencies else None
    )
    return row, standardized_agents


def build_method_runtime(method: Mapping[str, str], manifest: Mapping[str, Any]) -> Any:
    config = resolved_method_config(method)
    if method["engine"].startswith("dwa_"):
        builder, multi_config = build_classical_runtime(config, manifest)
        return {"config": config, "builder": builder, "multi_config": multi_config, "planner": planner_config(config)}
    return {"config": config, "runtime": DevelopmentRuntime(config, manifest)}


def evaluate_formal_episode(
    method: Mapping[str, str], runtime_bundle: Mapping[str, Any], entry: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray], list[dict[str, Any]]]:
    engine = method["engine"]
    config = runtime_bundle["config"]
    if engine.startswith("dwa_"):
        kwargs = dict(
            environment_builder=runtime_bundle["builder"],
            multi_config=runtime_bundle["multi_config"],
            scenario=entry["scenario_id"],
            seed=int(entry["seed"]),
            peer_radius=float(config["peer_radius"]),
            planner_config=runtime_bundle["planner"],
        )
        if engine == "dwa_sensing_matched":
            episode, agents, runtime_rows, trajectory = run_sensing_matched_episode(
                method="dwa_sensing_matched", **kwargs
            )
        else:
            episode, agents, runtime_rows, trajectory = run_classical_episode(
                method="dwa_style", **kwargs
            )
        planning_ms = float(episode["planning_runtime_ms"])
        episode.update(
            {
                "upper_planning_total_ms": planning_ms,
                "planning_decision_count": len(runtime_rows),
                "planning_runtime_per_decision_ms": planning_ms / len(runtime_rows) if runtime_rows else None,
                "upper_pipeline_invocation_count": len(runtime_rows),
                "execution_actor_forward_ms": 0.0,
                "execution_dmp_ms": 0.0,
                "total_online_algorithm_compute_ms": planning_ms,
                "replanning_count": 0,
                "normal_replanning_count": 0,
                "emergency_replanning_count": 0,
            }
        )
        row, agent_rows = _standardize(episode, agents, trajectory, entry, method)
        return row, agent_rows, trajectory, []

    runtime: DevelopmentRuntime = runtime_bundle["runtime"]
    if engine == "one_shot_ablation":
        contract = method_contract(config)
        episode, agents, trajectory, events, _ = evaluate_ablation_episode(
            runtime, config, entry, contract
        )
        row, agent_rows = _standardize(episode, agents, trajectory, entry, method)
        return row, agent_rows, trajectory, events

    internal = METHOD_RERR_FP_SHEP if engine == "rerr_fp_shep" else METHOD_RERR_GAT
    recorder = OnlineRuntimeRecorder()
    policy = TimedPolicyProxy(runtime.policy, recorder)
    with recorder.instrument_dmp(), recorder.scoped_context(
        evaluation_block="long_range_untouched_formal",
        method_id=method["method_id"],
        scenario_id=entry["scenario_id"],
        stage=entry["stage"],
        family=entry["family"],
    ):
        episode, agents, events, _, extra = run_rerr_episode(
            config=runtime.eval_config,
            settings=runtime.settings,
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
    trajectory = _rerr_trajectory(extra, entry)
    episode = dict(episode)
    episode["planning_runtime_ms"] = float(episode["upper_planning_total_ms"])
    episode["normal_replanning_count"] = max(
        0,
        int(episode.get("replanning_count", 0))
        - int(episode.get("emergency_replanning_count", 0)),
    )
    episode["planning_runtime_per_decision_ms"] = (
        float(episode["upper_planning_total_ms"] / episode["planning_decision_count"])
        if episode["planning_decision_count"]
        else None
    )
    row, agent_rows = _standardize(episode, agents, trajectory, entry, method)
    return row, agent_rows, trajectory, [dict(item) for item in events]


def preflight() -> None:
    """Exercise every formal engine on one development scene before freezing."""

    if FORMAL_MANIFEST.exists() or (RECORD_DIR.exists() and any(RECORD_DIR.rglob("*.json"))):
        raise RuntimeError("formal engine preflight is forbidden after formal opening")
    development_gate()
    manifest_path = ARTIFACT_ROOT / "08_development/development_manifest.json"
    manifest = load_json(manifest_path)
    entry = dict(manifest["entries"][0])
    checks: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for method in METHODS:
        try:
            runtime = build_method_runtime(method, manifest)
            row, agents, trajectory, events = evaluate_formal_episode(method, runtime, entry)
            positions = np.asarray(trajectory["positions"], dtype=float)
            if row.get("method_id") != method["method_id"]:
                raise RuntimeError("standardized method identity mismatch")
            if row.get("scenario_id") != entry["scenario_id"]:
                raise RuntimeError("standardized scenario identity mismatch")
            if len(agents) != 3 or {int(agent["agent_id"]) for agent in agents} != {0, 1, 2}:
                raise RuntimeError("standardized agent rows are incomplete")
            if positions.ndim != 3 or positions.shape[1:] != (3, 3) or positions.shape[0] < 2:
                raise RuntimeError(f"invalid trajectory shape: {positions.shape}")
            if not np.isfinite(positions).all():
                raise RuntimeError("trajectory contains non-finite values")
            required = {
                "team_success",
                "any_collision",
                "obstacle_collision",
                "inter_agent_collision",
                "timeout",
                "steps",
                "total_online_algorithm_compute_ms",
            }
            missing = sorted(required - set(row))
            if missing:
                raise RuntimeError(f"standardized row missing fields: {missing}")
            checks.append(
                {
                    "method_id": method["method_id"],
                    "engine": method["engine"],
                    "status": "PASS",
                    "agent_row_count": len(agents),
                    "trajectory_shape": list(positions.shape),
                    "event_schema_exercised": isinstance(events, list),
                    "collision_replay_gate_exercised": True,
                    "performance_values_retained": False,
                }
            )
        except Exception as exc:  # pragma: no cover - exercised only by failed integration
            errors.append(
                {
                    "method_id": method["method_id"],
                    "engine": method["engine"],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    payload = {
        "schema_version": "long_range_formal_engine_preflight_v1",
        "status": "PASS" if len(checks) == len(METHODS) and not errors else "FAIL",
        "created_at": datetime.now().astimezone().isoformat(),
        "source_split": "development",
        "source_manifest": str(manifest_path.relative_to(ARTIFACT_ROOT)).replace("\\", "/"),
        "source_manifest_sha256": sha256_file(manifest_path),
        "scenario_id": entry["scenario_id"],
        "method_count": len(checks),
        "expected_method_count": len(METHODS),
        "checks": checks,
        "errors": errors,
        "performance_selection_allowed": False,
        "performance_values_retained": False,
        "formal_manifest_exists": False,
        "formal_episode_count": 0,
        "formal_data_used": False,
    }
    atomic_json(FORMAL_ENGINE_PREFLIGHT, payload)
    print(json.dumps({"phase": "preflight", "status": payload["status"], "method_count": len(checks), "errors": len(errors)}), flush=True)
    if payload["status"] != "PASS":
        raise RuntimeError(f"formal engine preflight failed: {errors}")


def _record_paths(method_id: str, scenario_id: str) -> tuple[Path, Path]:
    directory = RECORD_DIR / method_id
    return directory / f"{scenario_id}.json", directory / f"{scenario_id}_trajectory.npz"


def run(method_id: str, shard_index: int, shard_count: int, limit: int | None) -> None:
    freeze = verify_formal_freeze()
    if method_id not in METHOD_BY_ID:
        raise ValueError(f"unknown formal method: {method_id}")
    if not 0 <= int(shard_index) < int(shard_count):
        raise ValueError("shard_index must be in [0, shard_count)")
    manifest = load_json(FORMAL_MANIFEST)
    method = METHOD_BY_ID[method_id]
    runtime = build_method_runtime(method, manifest)
    entries = [
        entry
        for index, entry in enumerate(manifest["entries"])
        if index % int(shard_count) == int(shard_index)
    ]
    if limit is not None:
        entries = entries[: int(limit)]
    completed = 0
    for entry in entries:
        record_path, trajectory_path = _record_paths(method_id, entry["scenario_id"])
        if record_path.is_file() and trajectory_path.is_file():
            continue
        print(f"[formal] start {method_id} {entry['scenario_id']}", flush=True)
        try:
            episode, agents, trajectory, events = evaluate_formal_episode(method, runtime, entry)
            trajectory_content_hash, trajectory_file_hash = write_npz(trajectory_path, trajectory)
            payload = {
                "schema_version": SCHEMA,
                "method_id": method_id,
                "scenario_id": entry["scenario_id"],
                "scenario_environment_fingerprint": entry["environment_fingerprint"],
                "scenario_geometry_fingerprint": entry["geometry_fingerprint"],
                "manifest_semantic_sha256": freeze["manifest_semantic_sha256"],
                "method_config_sha256": freeze["method_config_sha256"][method_id],
                "checkpoint_sha256": freeze["checkpoint_sha256"],
                "episode": episode,
                "agents": agents,
                "events": events,
                "trajectory_file": trajectory_path.name,
                "trajectory_content_hash": trajectory_content_hash,
                "trajectory_file_sha256": trajectory_file_hash,
            }
            payload["result_hash"] = content_hash(payload)
            atomic_json(record_path, payload)
            completed += 1
            print(f"[formal] complete {method_id} {entry['scenario_id']} count={completed}", flush=True)
        except Exception as error:
            atomic_json(
                record_path.with_name(record_path.stem + "_SOFTWARE_ERROR.json"),
                {
                    "schema_version": SCHEMA,
                    "method_id": method_id,
                    "scenario_id": entry["scenario_id"],
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
            raise
    print(json.dumps({"phase": "run", "method_id": method_id, "new_records": completed, "performance_hidden": True}), flush=True)


def collect_records() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    team: list[dict[str, Any]] = []
    agents: list[dict[str, Any]] = []
    errors: list[str] = []
    for method_id in METHOD_ORDER:
        directory = RECORD_DIR / method_id
        errors.extend(str(path.relative_to(ARTIFACT_ROOT)) for path in directory.glob("*_SOFTWARE_ERROR.json"))
        for path in sorted(directory.glob("FORMAL_LR_*.json")):
            if path.name.endswith("_SOFTWARE_ERROR.json"):
                continue
            payload = load_json(path)
            team.append(dict(payload["episode"]))
            agents.extend(dict(row) for row in payload["agents"])
    team.sort(key=lambda row: (str(row["scenario_id"]), METHOD_ORDER.index(row["method_id"])))
    agents.sort(key=lambda row: (str(row["scenario_id"]), METHOD_ORDER.index(row["method_id"]), int(row["agent_id"])))
    return team, agents, errors


def finalize() -> None:
    freeze = verify_formal_freeze()
    team, agents, errors = collect_records()
    expected_team = int(freeze["expected_team_rows"])
    expected_agents = int(freeze["expected_agent_rows"])
    team_keys = [(row["scenario_id"], row["method_id"]) for row in team]
    agent_keys = [(row["scenario_id"], row["method_id"], int(row["agent_id"])) for row in agents]
    complete = (
        len(team) == expected_team
        and len(set(team_keys)) == expected_team
        and len(agents) == expected_agents
        and len(set(agent_keys)) == expected_agents
        and not errors
    )
    reconciliation = {
        "schema_version": SCHEMA,
        "status": "PASS" if complete else "IN_PROGRESS",
        "team_row_count": len(team),
        "expected_team_row_count": expected_team,
        "agent_row_count": len(agents),
        "expected_agent_row_count": expected_agents,
        "unique_team_key_count": len(set(team_keys)),
        "unique_agent_key_count": len(set(agent_keys)),
        "software_errors": errors,
        "method_counts": dict(Counter(row["method_id"] for row in team)),
        "formal_performance_aggregated": False,
    }
    atomic_json(RECORD_DIR / "formal_record_reconciliation.json", reconciliation)
    if not complete:
        print(json.dumps(reconciliation), flush=True)
        return
    write_csv(RECORD_DIR / "formal_team_results.csv", team)
    write_csv(RECORD_DIR / "formal_agent_results.csv", agents)
    atomic_json(
        RECORD_DIR / "formal_run_complete.json",
        {
            **reconciliation,
            "completed_at": datetime.now().astimezone().isoformat(),
            "formal_performance_aggregated": False,
            "authorized_next_phase": "POST_FORMAL_ANALYSIS",
        },
    )
    print(json.dumps({"phase": "finalize", "status": "PASS", "team_rows": len(team), "performance_hidden": True}), flush=True)


def status() -> None:
    team, _, errors = collect_records()
    counts = Counter(row["method_id"] for row in team)
    print(json.dumps({"completed": dict(counts), "total": len(team), "errors": len(errors), "performance_hidden": True}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("preflight", "prepare", "run", "status", "finalize"))
    parser.add_argument("--method-id", choices=METHOD_ORDER)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.phase == "preflight":
        preflight()
    elif args.phase == "prepare":
        prepare()
    elif args.phase == "run":
        if args.method_id is None:
            raise ValueError("--method-id is required for run")
        run(args.method_id, args.shard_index, args.shard_count, args.limit)
    elif args.phase == "status":
        status()
    else:
        finalize()


if __name__ == "__main__":
    main()
