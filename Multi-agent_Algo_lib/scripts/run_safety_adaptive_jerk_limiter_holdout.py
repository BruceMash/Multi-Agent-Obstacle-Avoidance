#!/usr/bin/env python3
"""Freeze and run the selected safety-adaptive jerk limiter on Holdout100."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_safety_adaptive_jerk_limiter as base


METHOD_RERR_GAT = base.METHOD_RERR_GAT
run_episode = base.run_episode
OnlineRuntimeRecorder = base.OnlineRuntimeRecorder
TimedPolicyProxy = base.TimedPolicyProxy
build_online_gat_plan_optimized = base.build_online_gat_plan_optimized


ARTIFACT_ROOT = REPO_ROOT / "artifacts/safety_adaptive_jerk_limiter/20260825_115704"
SOURCE_STUDY_ROOT = REPO_ROOT / "artifacts/gat_recurrent_smoothness_rescue/20260821_180244"
SOURCE_MANIFEST = SOURCE_STUDY_ROOT / "08_holdout/GAT_RS_HOLDOUT_SCENE_MANIFEST.json"
SOURCE_ORIGINAL = SOURCE_STUDY_ROOT / "13_objective_revision/08_holdout/GAT_R_FP_ANCHOR_HOLDOUT400/episode_records"
FREEZE_PATH = ARTIFACT_ROOT / "FINAL_JERK_LIMITER_FREEZE.json"
OUTPUT_ROOT = ARTIFACT_ROOT / "holdout_records/strong/episode_records"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def selected_entries() -> list[dict[str, Any]]:
    entries = list(load_json(SOURCE_MANIFEST)["entries"])
    cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in entries:
        cells[(str(row["stage"]), str(row["family"]))].append(dict(row))
    if len(cells) != 20:
        raise RuntimeError(f"expected 20 holdout cells, got {len(cells)}")
    selected: list[dict[str, Any]] = []
    for key in sorted(cells):
        ordered = sorted(cells[key], key=lambda row: str(row["scenario_id"]))
        if len(ordered) < 5:
            raise RuntimeError(f"cell {key} has fewer than 5 scenes")
        selected.extend(ordered[:5])
    selected.sort(key=lambda row: str(row["scenario_id"]))
    counts = Counter((str(row["stage"]), str(row["family"])) for row in selected)
    if len(selected) != 100 or set(counts.values()) != {5}:
        raise RuntimeError(f"invalid balanced Holdout100: {counts}")
    return selected


def holdout_manifest(entries: list[Mapping[str, Any]]) -> dict[str, Any]:
    source = load_json(SOURCE_MANIFEST)
    return {**source, "entries": [dict(row) for row in entries], "unique_scenario_count": len(entries)}


def freeze() -> None:
    decision_path = ARTIFACT_ROOT / "development_decision.json"
    decision = load_json(decision_path)
    if decision.get("selected_variant") != "strong" or not decision.get("holdout_authorized"):
        raise RuntimeError("Development decision did not authorize Strong Holdout")
    entries = selected_entries()
    ids = [str(row["scenario_id"]) for row in entries]
    original_hashes: dict[str, dict[str, str]] = {}
    scene_hashes: dict[str, str] = {}
    for row in entries:
        sid = str(row["scenario_id"])
        scene_path = SOURCE_STUDY_ROOT / str(row["scenario_file"])
        if sha256_file(scene_path) != str(row["scenario_file_sha256"]):
            raise RuntimeError(f"scene hash mismatch: {sid}")
        scene_hashes[sid] = sha256_file(scene_path)
        json_path = SOURCE_ORIGINAL / f"{sid}.json"
        npz_path = SOURCE_ORIGINAL / f"{sid}_trajectory.npz"
        if not json_path.exists() or not npz_path.exists():
            raise RuntimeError(f"missing frozen Original Holdout record: {sid}")
        original_hashes[sid] = {"record_sha256": sha256_file(json_path), "trajectory_sha256": sha256_file(npz_path)}
    tracked = [
        Path(__file__).resolve(),
        REPO_ROOT / "planning/safety_adaptive_jerk_limiter.py",
        REPO_ROOT / "planning/historical_forcing_gate.py",
        REPO_ROOT / "Environment/frozen_sac_dmp_execution.py",
        REPO_ROOT / "Environment/multi_agent_dmp_env.py",
        REPO_ROOT / "Multi-agent_Algo_lib/scripts/evaluate_gat_v1_err_development.py",
        REPO_ROOT / "Multi-agent_Algo_lib/scripts/run_safety_adaptive_jerk_limiter.py",
    ]
    source_hashes = {path.resolve().relative_to(REPO_ROOT.resolve()).as_posix(): sha256_file(path) for path in tracked}
    config = load_json(base.FINAL_METHOD_CONFIG)
    payload = {
        "schema_version": "final_safety_adaptive_jerk_limiter_holdout_freeze_v1",
        "status": "FROZEN_BEFORE_SELECTED_HOLDOUT_PERFORMANCE",
        "created_unix_s": time.time(),
        "selected_variant": "STRONG",
        "j_smooth_mps3": float(load_json(base.THRESHOLD_PATH)["variants"]["strong"]["j_smooth_mps3"]),
        "safety_mapping": load_json(base.CONTRACT_PATH)["safety_adaptation"],
        "development_decision_sha256": sha256_file(decision_path),
        "control_contract_sha256": sha256_file(base.CONTRACT_PATH),
        "thresholds_sha256": sha256_file(base.THRESHOLD_PATH),
        "microtest_sha256": sha256_file(base.MICROTEST_PATH),
        "holdout_selection_rule": "lexicographically first five scenario IDs in each of 4 stage x 5 family cells; outcome-blind",
        "holdout_manifest_source": SOURCE_MANIFEST.resolve().relative_to(REPO_ROOT.resolve()).as_posix(),
        "holdout_manifest_sha256": sha256_file(SOURCE_MANIFEST),
        "scenario_count": len(entries),
        "balance": "4 stages x 5 families x 5 scenes",
        "scenario_ids": ids,
        "scene_sha256": scene_hashes,
        "original_record_source": SOURCE_ORIGINAL.resolve().relative_to(REPO_ROOT.resolve()).as_posix(),
        "original_record_reuse_basis": "frozen prior GAT-R Holdout trajectories; disabled-limiter microtest established exact execution equivalence",
        "original_record_sha256": original_hashes,
        "gat_checkpoint": config["gat_checkpoint"],
        "gat_checkpoint_sha256": sha256_file(REPO_ROOT / config["gat_checkpoint"]),
        "sac_checkpoint": config["sac_checkpoint"],
        "sac_checkpoint_sha256": sha256_file(REPO_ROOT / config["sac_checkpoint"]),
        "source_sha256": source_hashes,
        "performance_episodes_executed_by_freeze_phase": 0,
        "formal_v2_authorized": False,
    }
    atomic_json(FREEZE_PATH, payload)
    print(json.dumps({"status": "PASS", "scenario_count": len(entries), "artifact": str(FREEZE_PATH)}, indent=2))


def verify_freeze() -> dict[str, Any]:
    freeze_payload = load_json(FREEZE_PATH)
    if freeze_payload.get("status") != "FROZEN_BEFORE_SELECTED_HOLDOUT_PERFORMANCE":
        raise RuntimeError("invalid holdout freeze")
    for relative, expected in freeze_payload["source_sha256"].items():
        if sha256_file(REPO_ROOT / relative) != expected:
            raise RuntimeError(f"post-freeze source changed: {relative}")
    return freeze_payload


def run_holdout(shard_index: int, shard_count: int) -> None:
    freeze_payload = verify_freeze()
    entries = selected_entries()
    if [str(row["scenario_id"]) for row in entries] != list(freeze_payload["scenario_ids"]):
        raise RuntimeError("selected Holdout IDs changed")
    manifest = holdout_manifest(entries)
    runtime, _ = base.runtime_for(manifest)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    completed = {path.stem for path in OUTPUT_ROOT.glob("GATRS_HOLDOUT_*.json") if not path.name.endswith("_SOFTWARE_ERROR.json")}
    indexed = [row for index, row in enumerate(entries) if index % int(shard_count) == int(shard_index)]
    for local_index, entry in enumerate(indexed, start=1):
        sid = str(entry["scenario_id"])
        if sid in completed:
            continue
        print(f"[holdout:strong:{shard_index}/{shard_count}] {local_index:03d}/{len(indexed):03d} start {sid}", flush=True)
        recorder = OnlineRuntimeRecorder()
        proxy = TimedPolicyProxy(runtime.policy, recorder)
        limiter = base.limiter_for("strong")
        try:
            with recorder.instrument_dmp(), recorder.scoped_context(
                evaluation_block="safety_adaptive_jerk_limiter_holdout",
                configuration_id="JERK_LIMITER_STRONG_FROZEN",
                stage=entry["stage"], family=entry["family"], scenario_id=sid,
                seed=int(entry["seed"]), method=METHOD_RERR_GAT,
            ):
                episode, agents, events, triggers, extra = run_episode(
                    config=runtime.eval_config, settings=runtime.settings,
                    multi_config=runtime.multi_config, policy=proxy,
                    gat_model=runtime.gat_model, gat_device=runtime.gat_device,
                    method=METHOD_RERR_GAT, scenario=sid, seed=int(entry["seed"]),
                    environment_builder=runtime.builder, runtime_recorder=recorder,
                    upper_plan_builder=build_online_gat_plan_optimized,
                    execution_acceleration_limiter=limiter,
                )
            base.save_episode_record(OUTPUT_ROOT, entry, episode, agents, events, triggers, extra)
            print(f"[holdout:strong:{shard_index}/{shard_count}] complete {sid} success={int(episode['team_success'])} collision={int(episode['collision'])}", flush=True)
        except Exception as error:
            atomic_json(OUTPUT_ROOT / f"{sid}_SOFTWARE_ERROR.json", {
                "scenario_id": sid,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
            })
            raise


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--shard-index", type=int, default=0)
    run_parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze()
    else:
        run_holdout(args.shard_index, args.shard_count)


if __name__ == "__main__":
    main()
