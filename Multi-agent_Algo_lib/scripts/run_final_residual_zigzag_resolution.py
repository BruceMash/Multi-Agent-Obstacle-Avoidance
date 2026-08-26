#!/usr/bin/env python3
"""Frozen Dev/Holdout runner for the final residual-zigzag resolution.

This runner evaluates exactly two paired arms: frozen Strong, and Strong with
the source-supported Early Safety Bypass plus the single selected DCTB branch.
It never trains a network and never runs Formal V2.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import traceback
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as _pandas  # noqa: F401 -- stable Windows torch import order


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
for search_path in (REPO_ROOT, ALGO_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from planning.direction_continuity_tiebreak import (  # noqa: E402
    DirectionContinuityTieBreak,
    DirectionContinuityTieBreakConfig,
)
from planning.online_runtime_instrumentation import (  # noqa: E402
    OnlineRuntimeRecorder,
    TimedPolicyProxy,
)
from planning.pre_gat_220step_revalidation import stable_hash  # noqa: E402
from planning.semi_structured_long_range_benchmark import (  # noqa: E402
    FAMILY_ORDER,
    STAGE_ORDER,
    generate_scenario_manifest,
    json_ready,
    validate_scenario_manifest,
)
from scripts.evaluate_gat_v1_err_development import (  # noqa: E402
    METHOD_RERR_GAT,
    build_online_gat_plan_optimized,
    run_episode,
)
from scripts.prepare_gat_recurrent_rescue_splits import (  # noqa: E402
    compact_scene_index,
    start_goal_fingerprint,
)
from scripts.run_gat_recurrent_r_development import FileBackedBuilder  # noqa: E402
from scripts.run_long_range_development import DevelopmentRuntime  # noqa: E402
from scripts.run_safety_adaptive_jerk_limiter import (  # noqa: E402
    FINAL_METHOD_CONFIG,
    limiter_for,
    save_episode_record,
)


ARTIFACT_RELATIVE = Path("artifacts/final_residual_zigzag_resolution/20260825_183146")
ARTIFACT_ROOT = REPO_ROOT / ARTIFACT_RELATIVE
EARLY_CONTRACT = ARTIFACT_ROOT / "STRONG_EARLY_BYPASS_CONTRACT.json"
EARLY_MICROTEST = ARTIFACT_ROOT / "STRONG_EARLY_BYPASS_MICROTEST.json"
BRANCH_DECISION = ARTIFACT_ROOT / "RESIDUAL_ZIGZAG_BRANCH_DECISION.json"
COUPLING_SUMMARY = ARTIFACT_ROOT / "LAGGED_COUPLING_SUMMARY.json"
RUNTIME_CONTRACT = ARTIFACT_ROOT / "FINAL_ZIGZAG_RUNTIME_CONTRACT.json"
MANIFEST_PATHS = {
    "development": ARTIFACT_ROOT / "FINAL_ZIGZAG_DEV100_MANIFEST.json",
    "holdout": ARTIFACT_ROOT / "FINAL_ZIGZAG_HOLDOUT100_MANIFEST.json",
}
SCENE_DIRS = {
    "development": ARTIFACT_ROOT / "scenes/development",
    "holdout": ARTIFACT_ROOT / "scenes/holdout",
}
OUTPUT_DIRS = {
    "development": {
        "strong": ARTIFACT_ROOT / "development/strong/episode_records",
        "repaired": ARTIFACT_ROOT / "development/repaired/episode_records",
    },
    "holdout": {
        "strong": ARTIFACT_ROOT / "holdout/strong/episode_records",
        "repaired": ARTIFACT_ROOT / "holdout/repaired/episode_records",
    },
}
SEED_BASE = {"development": 4_100_000_000, "holdout": 4_200_000_000}
PREFIX = {"development": "FZR_DEV_", "holdout": "FZR_HOLDOUT_"}
NEAR_TIE_SOURCE = (
    REPO_ROOT
    / "artifacts/safety_adaptive_jerk_limiter/20260825_115704/dev_records/strong/episode_records"
)
HISTORICAL_MANIFESTS = (
    REPO_ROOT
    / "artifacts/gat_recurrent_smoothness_rescue/20260821_180244/02_recurrent_dataset/GAT_RS_TRAIN_SCENE_MANIFEST.json",
    REPO_ROOT
    / "artifacts/gat_recurrent_smoothness_rescue/20260821_180244/07_development/GAT_RS_DEV_SCENE_MANIFEST.json",
    REPO_ROOT
    / "artifacts/gat_recurrent_smoothness_rescue/20260821_180244/08_holdout/GAT_RS_HOLDOUT_SCENE_MANIFEST.json",
    REPO_ROOT
    / "artifacts/gat_recurrent_smoothness_rescue/20260821_180244/10_formal_v2/FORMAL_V2_MANIFEST.json",
    REPO_ROOT
    / "artifacts/recurrent_selector_ablation_confirmation/20260822_143118/02_manifest/SELECTOR_ABLATION_MANIFEST.json",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_ready(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _historical_values() -> dict[str, set[Any]]:
    values: dict[str, set[Any]] = {
        "seed": set(),
        "geometry_fingerprint": set(),
        "dynamic_track_fingerprint": set(),
        "translation_invariant_fingerprint": set(),
        "start_goal_fingerprint": set(),
    }
    for path in HISTORICAL_MANIFESTS:
        if not path.exists():
            raise FileNotFoundError(path)
        for row in load_json(path).get("entries", []):
            for key in values:
                value = row.get(key)
                if value is not None and value != "":
                    values[key].add(int(value) if key == "seed" else str(value))
    return values


def _validate_disjoint(entries: Sequence[Mapping[str, Any]], *, other: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    historical = _historical_values()
    for row in other:
        for key in historical:
            value = row.get(key)
            if value is not None and value != "":
                historical[key].add(int(value) if key == "seed" else str(value))
    overlaps: dict[str, list[Any]] = {}
    for key, old in historical.items():
        current = {
            int(row[key]) if key == "seed" else str(row[key])
            for row in entries
        }
        overlaps[key] = sorted(current & old)
    result = {
        "historical_manifest_count": len(HISTORICAL_MANIFESTS),
        "overlap_counts": {key: len(value) for key, value in overlaps.items()},
        "PASS": all(not value for value in overlaps.values()),
    }
    if not result["PASS"]:
        raise RuntimeError(f"new split overlaps historical registry: {overlaps}")
    return result


def _compact_and_materialize(block: str, generated: Mapping[str, Any]) -> dict[str, Any]:
    scene_dir = SCENE_DIRS[block]
    scene_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for scene in generated["entries"]:
        full_scene = copy.deepcopy(dict(scene))
        full_scene["start_goal_fingerprint"] = start_goal_fingerprint(full_scene)
        scene_path = scene_dir / f"{scene['scenario_id']}.json"
        atomic_json(scene_path, full_scene)
        entries.append(
            compact_scene_index(
                full_scene,
                relative(scene_path.relative_to(REPO_ROOT)),
                sha256_file(scene_path),
            )
        )
    manifest = {
        "schema_version": "final_residual_zigzag_scene_manifest_v1",
        "split": block,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed_base": SEED_BASE[block],
        "counts_per_stage": 25,
        "unique_scenario_count": len(entries),
        "stage_order": list(STAGE_ORDER),
        "family_order": list(FAMILY_ORDER),
        "materialization_contract": "each scenario_file is one complete frozen long-range scene",
        "entries": entries,
    }
    manifest["manifest_semantic_sha256"] = stable_hash(
        {key: value for key, value in manifest.items() if key != "created_at"}
    )
    atomic_json(MANIFEST_PATHS[block], manifest)
    return manifest


def _generate_manifest(block: str) -> dict[str, Any]:
    if MANIFEST_PATHS[block].exists():
        return load_json(MANIFEST_PATHS[block])
    generated = generate_scenario_manifest(
        counts_per_stage=25,
        seed_base=SEED_BASE[block],
        prefix=PREFIX[block],
    )
    validation = validate_scenario_manifest(generated)
    if validation.get("status") != "PASS":
        raise RuntimeError(f"generated manifest failed validation: {validation}")
    other_entries: list[Mapping[str, Any]] = []
    if block == "holdout":
        if not MANIFEST_PATHS["development"].exists():
            raise RuntimeError("Development manifest must be frozen before Holdout")
        other_entries = load_json(MANIFEST_PATHS["development"])["entries"]
    full_entries = []
    for row in generated["entries"]:
        full = dict(row)
        full["start_goal_fingerprint"] = start_goal_fingerprint(full)
        full_entries.append(full)
    _validate_disjoint(full_entries, other=other_entries)
    return _compact_and_materialize(block, generated)


def _near_tie_threshold() -> tuple[float, int]:
    values: list[float] = []
    files = sorted(NEAR_TIE_SOURCE.glob("*.json"))
    for path in files:
        if path.name.endswith("_SOFTWARE_ERROR.json"):
            continue
        payload = load_json(path)
        for row in payload.get("events", []):
            if row.get("event") == "INITIAL_SELECTION" or row.get("selected_null"):
                continue
            value = row.get("top1_top2_probability_margin")
            if value is not None and np.isfinite(float(value)):
                values.append(float(value))
    if len(files) != 100 or len(values) < 1000:
        raise RuntimeError("frozen Strong confidence source is incomplete")
    return float(np.percentile(np.asarray(values, dtype=float), 25)), len(values)


def _runtime_source_paths() -> list[Path]:
    return [
        Path(__file__).resolve(),
        REPO_ROOT / "planning/direction_continuity_tiebreak.py",
        REPO_ROOT / "planning/safety_adaptive_jerk_limiter.py",
        REPO_ROOT / "planning/event_triggered_reference_reconstruction.py",
        REPO_ROOT / "Multi-agent_Algo_lib/scripts/evaluate_gat_v1_err_development.py",
        REPO_ROOT / "Multi-agent_Algo_lib/scripts/run_safety_adaptive_jerk_limiter.py",
    ]


def prepare_development() -> None:
    if load_json(EARLY_MICROTEST).get("status") != "PASS":
        raise RuntimeError("Early Bypass microtest did not pass")
    if load_json(BRANCH_DECISION).get("SELECTED_BRANCH") != "DCTB":
        raise RuntimeError("frozen lag audit did not select DCTB")
    if RUNTIME_CONTRACT.exists():
        raise RuntimeError("runtime contract already exists; refusing to overwrite freeze")
    if any(path.exists() and any(path.glob("*.json")) for path in OUTPUT_DIRS["development"].values()):
        raise RuntimeError("Development result records already exist before freeze")
    manifest = _generate_manifest("development")
    cells = Counter((row["stage"], row["family"]) for row in manifest["entries"])
    if len(manifest["entries"]) != 100 or len(cells) != 20 or set(cells.values()) != {5}:
        raise RuntimeError(f"Development is not 4x5x5: {cells}")
    threshold, sample_count = _near_tie_threshold()
    visual_ids = [f"FZR_DEV_{index}_000" for index in range(1, 5)]
    if not set(visual_ids).issubset({row["scenario_id"] for row in manifest["entries"]}):
        raise RuntimeError("predeclared visual scene is absent")
    config = load_json(FINAL_METHOD_CONFIG)
    contract = {
        "schema_version": "final_residual_zigzag_runtime_contract_v1",
        "status": "FROZEN_BEFORE_NEW_DEVELOPMENT_PERFORMANCE",
        "selected_branch": "DCTB",
        "exactly_one_branch": True,
        "early_safety_bypass": {
            "contract": relative(EARLY_CONTRACT),
            "contract_sha256": sha256_file(EARLY_CONTRACT),
            "warning_margin_m": float(config["err"]["h_rep_m"]),
            "hard_margin_m": float(config["err"]["h_emg_m"]),
            "strong_jerk_threshold_changed": False,
        },
        "dctb": {
            "position": "post-GAT candidate commitment tie-break at accepted upper events",
            "initial_selection_excluded": True,
            "history_reset_on_terminal_handoff_or_null": True,
            "activation": "clear signed yaw or pitch reversal AND GAT near-tie AND not safety-critical",
            "near_tie_probability_margin_max": threshold,
            "near_tie_source": relative(NEAR_TIE_SOURCE),
            "near_tie_source_statistic": "P25 of existing Strong Development noninitial non-null top1-top2 probability margins",
            "near_tie_source_sample_count": sample_count,
            "alternative_scope": "existing GAT candidate ranks 2-3 only",
            "admissibility": [
                "FP preview valid", "positive preview task progress",
                "minimum FP clearance no lower than GAT top-1",
                "interaction risk class no worse; risky alternatives also no worse in duration/separation",
            ],
            "selection_rule": "first rank-ordered admissible continuation minimizing the triggered-axis second difference; no weighted objective",
            "new_threshold_grid": False,
        },
        "development": {
            "manifest": relative(MANIFEST_PATHS["development"]),
            "manifest_sha256": sha256_file(MANIFEST_PATHS["development"]),
            "scenario_count": 100,
            "balance": "4 stages x 5 families x 5 scenes",
            "arms": ["Frozen Strong", "Strong + Early Bypass + DCTB"],
            "fixed_visual_scene_ids_before_outcomes": visual_ids,
        },
        "development_gates_frozen_before_results": {
            "maximum_team_success_loss_pp": 1.0,
            "maximum_peer_collision_increase_pp": 1.0,
            "maximum_obstacle_collision_increase_pp": 1.0,
            "minimum_executed_yaw_or_pitch_reversal_reduction_percent": 25.0,
            "minimum_same_axis_directional_total_variation_reduction_percent": 10.0,
            "maximum_smoothness_cost_increase_percent_on_both_success": 5.0,
            "visual_gate": "at least 3/4 predeclared raw scenes reduce combined executed reversals and combined directional TV, with visibly longer same-direction arcs; no signal smoothing",
            "all_gates_required": True,
        },
        "holdout_rule": "generate and run new independent Holdout100 only if every Development gate passes",
        "formal_v2_execution_authorized": False,
        "frozen_method": {
            "config": relative(FINAL_METHOD_CONFIG),
            "config_sha256": sha256_file(FINAL_METHOD_CONFIG),
            "gat_checkpoint": config["gat_checkpoint"],
            "gat_checkpoint_sha256": sha256_file(REPO_ROOT / config["gat_checkpoint"]),
            "sac_checkpoint": config["sac_checkpoint"],
            "sac_checkpoint_sha256": sha256_file(REPO_ROOT / config["sac_checkpoint"]),
            "strong_j_smooth_mps3": load_json(EARLY_CONTRACT)["frozen_strong"]["j_smooth_mps3"],
        },
        "source_sha256": {
            relative(path): sha256_file(path) for path in _runtime_source_paths()
        },
        "performance_episodes_observed_before_freeze": 0,
    }
    atomic_json(RUNTIME_CONTRACT, contract)
    print(json.dumps({"status": "PASS", "near_tie_threshold": threshold, "visual_ids": visual_ids}, indent=2))


def _verify_runtime_freeze(block: str) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = load_json(RUNTIME_CONTRACT)
    for name, expected in contract["source_sha256"].items():
        if sha256_file(REPO_ROOT / name) != expected:
            raise RuntimeError(f"runtime source changed after freeze: {name}")
    if sha256_file(EARLY_CONTRACT) != contract["early_safety_bypass"]["contract_sha256"]:
        raise RuntimeError("Early Bypass contract changed after runtime freeze")
    manifest = load_json(MANIFEST_PATHS[block])
    for row in manifest["entries"]:
        path = REPO_ROOT / row["scenario_file"]
        if sha256_file(path) != row["scenario_file_sha256"]:
            raise RuntimeError(f"scene file changed after freeze: {row['scenario_id']}")
    return contract, manifest


def _runtime(manifest: Mapping[str, Any]) -> tuple[DevelopmentRuntime, dict[str, Any]]:
    config = load_json(FINAL_METHOD_CONFIG)
    runtime = DevelopmentRuntime(config, {"entries": []})
    runtime.builder = FileBackedBuilder(manifest, config, REPO_ROOT)
    return runtime, config


def _strong_limiter(*, early_bypass: bool) -> Any:
    limiter = limiter_for("strong")
    if early_bypass:
        limiter.config = replace(
            limiter.config,
            early_bypass_enabled=True,
            warning_margin_m=float(limiter.config.comfortable_margin_m),
        )
    return limiter


def _dctb(contract: Mapping[str, Any], config: Mapping[str, Any]) -> DirectionContinuityTieBreak:
    return DirectionContinuityTieBreak(
        DirectionContinuityTieBreakConfig(
            near_tie_probability_margin=float(
                contract["dctb"]["near_tie_probability_margin_max"]
            ),
            warning_safety_margin_m=float(config["err"]["h_rep_m"]),
        ),
        num_agents=int(config["num_agents"]),
    )


def run_block(block: str, arm: str, shard_index: int, shard_count: int) -> None:
    if block == "holdout":
        decision = ARTIFACT_ROOT / "FINAL_ZIGZAG_DEV_GO_NO_GO.json"
        if not decision.exists() or load_json(decision).get("DEV_GATE") != "PASS":
            raise RuntimeError("Development did not authorize Holdout")
        if not MANIFEST_PATHS["holdout"].exists():
            raise RuntimeError("Holdout manifest must be frozen by prepare-holdout first")
    contract, manifest = _verify_runtime_freeze(block)
    runtime, config = _runtime(manifest)
    output = OUTPUT_DIRS[block][arm]
    output.mkdir(parents=True, exist_ok=True)
    completed = {
        path.stem for path in output.glob("*.json")
        if not path.name.endswith("_SOFTWARE_ERROR.json")
    }
    entries = [
        row for index, row in enumerate(manifest["entries"])
        if index % int(shard_count) == int(shard_index)
    ]
    for local_index, entry in enumerate(entries, start=1):
        sid = str(entry["scenario_id"])
        if sid in completed:
            continue
        print(f"[{block}:{arm}:{shard_index}/{shard_count}] {local_index:03d}/{len(entries):03d} start {sid}", flush=True)
        recorder = OnlineRuntimeRecorder()
        proxy = TimedPolicyProxy(runtime.policy, recorder)
        repaired = arm == "repaired"
        limiter = _strong_limiter(early_bypass=repaired)
        tiebreak = _dctb(contract, config) if repaired else None
        try:
            with recorder.instrument_dmp(), recorder.scoped_context(
                evaluation_block=f"final_residual_zigzag_{block}",
                configuration_id=("STRONG_EARLY_BYPASS_DCTB" if repaired else "FROZEN_STRONG"),
                stage=entry["stage"], family=entry["family"], scenario_id=sid,
                seed=int(entry["seed"]), method=METHOD_RERR_GAT,
            ):
                episode, agents, events, triggers, extra = run_episode(
                    config=runtime.eval_config,
                    settings=runtime.settings,
                    multi_config=runtime.multi_config,
                    policy=proxy,
                    gat_model=runtime.gat_model,
                    gat_device=runtime.gat_device,
                    method=METHOD_RERR_GAT,
                    scenario=sid,
                    seed=int(entry["seed"]),
                    environment_builder=runtime.builder,
                    runtime_recorder=recorder,
                    upper_plan_builder=build_online_gat_plan_optimized,
                    execution_acceleration_limiter=limiter,
                    candidate_commitment_tiebreak=tiebreak,
                )
            save_episode_record(output, entry, episode, agents, events, triggers, extra)
            print(
                f"[{block}:{arm}:{shard_index}/{shard_count}] complete {sid} success={int(episode['team_success'])} collision={int(episode['collision'])}",
                flush=True,
            )
        except Exception as error:
            atomic_json(
                output / f"{sid}_SOFTWARE_ERROR.json",
                {
                    "scenario_id": sid,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
            raise


def prepare_holdout() -> None:
    decision = ARTIFACT_ROOT / "FINAL_ZIGZAG_DEV_GO_NO_GO.json"
    if not decision.exists() or load_json(decision).get("DEV_GATE") != "PASS":
        raise RuntimeError("Development did not pass; Holdout generation is forbidden")
    if MANIFEST_PATHS["holdout"].exists():
        raise RuntimeError("Holdout manifest already exists; refusing overwrite")
    manifest = _generate_manifest("holdout")
    contract = load_json(RUNTIME_CONTRACT)
    contract["holdout"] = {
        "manifest": relative(MANIFEST_PATHS["holdout"]),
        "manifest_sha256": sha256_file(MANIFEST_PATHS["holdout"]),
        "scenario_count": len(manifest["entries"]),
        "fixed_visual_scene_ids_before_outcomes": [
            f"FZR_HOLDOUT_{index}_000" for index in range(1, 5)
        ],
        "generated_only_after_development_gate_pass": True,
    }
    # Preserve the original pre-Development freeze and append only an
    # authorized Holdout block. Runtime source hashes stay unchanged.
    atomic_json(RUNTIME_CONTRACT, contract)
    print(json.dumps({"status": "PASS", "holdout_count": len(manifest["entries"])}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare-dev")
    sub.add_parser("prepare-holdout")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--block", choices=("development", "holdout"), required=True)
    run_parser.add_argument("--arm", choices=("strong", "repaired"), required=True)
    run_parser.add_argument("--shard-index", type=int, default=0)
    run_parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if args.command == "prepare-dev":
        prepare_development()
    elif args.command == "prepare-holdout":
        prepare_holdout()
    else:
        run_block(args.block, args.arm, args.shard_index, args.shard_count)


if __name__ == "__main__":
    main()
