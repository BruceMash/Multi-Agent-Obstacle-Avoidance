#!/usr/bin/env python3
"""Frozen two-arm Dev/Holdout runner for turn-sign persistence.

The repaired arm is exactly Frozen Strong with the already accepted early
safety bypass, followed by N=2 lateral/vertical turn-sign persistence.  The
script cannot launch Formal V2.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import pandas as _pandas  # noqa: F401 -- stable Windows torch import order


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
for search_path in (REPO_ROOT, ALGO_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from planning.online_runtime_instrumentation import OnlineRuntimeRecorder, TimedPolicyProxy  # noqa: E402
from planning.turn_sign_persistence import (  # noqa: E402
    TurnSignPersistenceConfig,
    TurnSignPersistenceExecutionFilter,
)
from scripts import run_final_residual_zigzag_resolution as split_tools  # noqa: E402
from scripts.evaluate_gat_v1_err_development import (  # noqa: E402
    METHOD_RERR_GAT,
    build_online_gat_plan_optimized,
    run_episode,
)
from scripts.run_safety_adaptive_jerk_limiter import (  # noqa: E402
    FINAL_METHOD_CONFIG,
    limiter_for,
    save_episode_record,
)


ARTIFACT_ROOT = REPO_ROOT / "artifacts/turn_sign_persistence_zigzag_repair/20260825_222130"
THRESHOLDS = ARTIFACT_ROOT / "TURN_PERSISTENCE_THRESHOLDS.json"
MICROTEST = ARTIFACT_ROOT / "TURN_PERSISTENCE_MICROTEST.json"
CONTROL_CONTRACT = ARTIFACT_ROOT / "TURN_PERSISTENCE_CONTROL_CONTRACT.json"
RUNTIME_CONTRACT = ARTIFACT_ROOT / "TURN_PERSISTENCE_RUNTIME_FREEZE.json"
MANIFEST_PATHS = {
    "development": ARTIFACT_ROOT / "TURN_PERSISTENCE_DEV100_MANIFEST.json",
    "holdout": ARTIFACT_ROOT / "TURN_PERSISTENCE_HOLDOUT100_MANIFEST.json",
}
SCENE_DIRS = {
    "development": ARTIFACT_ROOT / "scenes/development",
    "holdout": ARTIFACT_ROOT / "scenes/holdout",
}
OUTPUT_DIRS = {
    block: {
        arm: ARTIFACT_ROOT / block / arm / "episode_records"
        for arm in ("strong", "repaired")
    }
    for block in ("development", "holdout")
}
SEED_BASE = {"development": 4_300_000_000, "holdout": 4_400_000_000}
PREFIX = {"development": "TSP_DEV_", "holdout": "TSP_HOLDOUT_"}
DEV_DECISION = ARTIFACT_ROOT / "TURN_PERSISTENCE_DEV_GO_NO_GO.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def configure_split_tools() -> None:
    """Point the already-audited split helpers at this isolated artifact."""
    split_tools.ARTIFACT_RELATIVE = ARTIFACT_ROOT.relative_to(REPO_ROOT)
    split_tools.ARTIFACT_ROOT = ARTIFACT_ROOT
    split_tools.MANIFEST_PATHS = MANIFEST_PATHS
    split_tools.SCENE_DIRS = SCENE_DIRS
    split_tools.SEED_BASE = SEED_BASE
    split_tools.PREFIX = PREFIX
    prior = (
        REPO_ROOT
        / "artifacts/final_residual_zigzag_resolution/20260825_183146/FINAL_ZIGZAG_DEV100_MANIFEST.json"
    )
    split_tools.HISTORICAL_MANIFESTS = tuple(split_tools.HISTORICAL_MANIFESTS) + (prior,)


def source_paths() -> list[Path]:
    return [
        Path(__file__).resolve(),
        REPO_ROOT / "planning/turn_sign_persistence.py",
        REPO_ROOT / "planning/safety_adaptive_jerk_limiter.py",
        REPO_ROOT / "planning/event_triggered_reference_reconstruction.py",
        REPO_ROOT / "Multi-agent_Algo_lib/scripts/evaluate_gat_v1_err_development.py",
        REPO_ROOT / "Multi-agent_Algo_lib/scripts/run_safety_adaptive_jerk_limiter.py",
    ]


def prepare_development() -> None:
    configure_split_tools()
    if load_json(MICROTEST).get("status") != "PASS":
        raise RuntimeError("turn-persistence microtest did not pass")
    if RUNTIME_CONTRACT.exists():
        raise RuntimeError("runtime freeze already exists; refusing overwrite")
    if any(path.exists() and any(path.glob("*.json")) for path in OUTPUT_DIRS["development"].values()):
        raise RuntimeError("Development records exist before runtime freeze")
    manifest = split_tools._generate_manifest("development")
    cells = Counter((row["stage"], row["family"]) for row in manifest["entries"])
    if len(manifest["entries"]) != 100 or len(cells) != 20 or set(cells.values()) != {5}:
        raise RuntimeError(f"Development is not balanced 4x5x5: {cells}")
    visual_ids = [f"TSP_DEV_{stage}_000" for stage in range(1, 5)]
    if not set(visual_ids).issubset({row["scenario_id"] for row in manifest["entries"]}):
        raise RuntimeError("a predeclared raw visual scene is absent")
    config = load_json(FINAL_METHOD_CONFIG)
    thresholds = load_json(THRESHOLDS)
    contract = {
        "schema_version": "turn_persistence_runtime_freeze_v1",
        "status": "FROZEN_BEFORE_NEW_DEVELOPMENT_PERFORMANCE",
        "arms": ["Frozen Strong", "Frozen Strong + Early Safety Bypass + Turn-Sign Persistence"],
        "execution_order": [
            "raw SAC-DMP acceleration",
            "Frozen Strong jerk limiter",
            "turn-sign persistence",
            "original physical acceleration clipping",
            "dynamics",
        ],
        "safety_bypass": {
            "warning": "m_t <= h_rep=0.35 m bypasses Strong and persistence",
            "hard": "m_t <= h_emg=0.0 m",
            "output": "raw SAC-DMP acceleration subject only to original physical clipping",
        },
        "persistence": {
            "n_persist": 2,
            "horizontal_override_mps2": thresholds["A_REV_LAT"],
            "vertical_override_mps2": thresholds["A_REV_VERT"],
            "threshold_statistic": "P75 of existing safe-state Strong reversal requests",
            "threshold_grid": False,
            "independent_axis_state": True,
            "horizontal_tangent_preserved_exactly_before_physical_clipping": True,
        },
        "development": {
            "manifest": relative(MANIFEST_PATHS["development"]),
            "manifest_sha256": split_tools.sha256_file(MANIFEST_PATHS["development"]),
            "scenario_count": 100,
            "balance": "4 stages x 5 families x 5 scenes",
            "fixed_visual_scene_ids_before_outcomes": visual_ids,
        },
        "development_gates_frozen_before_results": {
            "maximum_team_success_loss_pp": 1.0,
            "maximum_peer_collision_increase_pp": 1.0,
            "maximum_obstacle_collision_increase_pp": 1.0,
            "minimum_executed_yaw_reversal_reduction_percent": 25.0,
            "minimum_executed_pitch_reversal_reduction_percent": 25.0,
            "minimum_yaw_rate_total_variation_reduction_percent": 10.0,
            "minimum_pitch_rate_total_variation_reduction_percent": 10.0,
            "maximum_smoothness_cost_increase_percent_on_both_success": 5.0,
            "minimum_visual_scenes_with_longer_arcs": 3,
            "all_gates_required": True,
        },
        "holdout_rule": "freeze and run new independent Holdout100 only if every Development gate passes",
        "formal_v2_execution_authorized": False,
        "frozen_method": {
            "config": relative(FINAL_METHOD_CONFIG),
            "config_sha256": split_tools.sha256_file(FINAL_METHOD_CONFIG),
            "gat_checkpoint": config["gat_checkpoint"],
            "gat_checkpoint_sha256": split_tools.sha256_file(REPO_ROOT / config["gat_checkpoint"]),
            "sac_checkpoint": config["sac_checkpoint"],
            "sac_checkpoint_sha256": split_tools.sha256_file(REPO_ROOT / config["sac_checkpoint"]),
            "strong_j_smooth_mps3": 22.869440564651327,
        },
        "frozen_artifact_sha256": {
            relative(path): split_tools.sha256_file(path)
            for path in (THRESHOLDS, MICROTEST, CONTROL_CONTRACT)
        },
        "source_sha256": {relative(path): split_tools.sha256_file(path) for path in source_paths()},
        "performance_episodes_observed_before_freeze": 0,
    }
    split_tools.atomic_json(RUNTIME_CONTRACT, contract)
    print(json.dumps({"status": "PASS", "visual_ids": visual_ids}, indent=2))


def verify_freeze(block: str) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = load_json(RUNTIME_CONTRACT)
    for name, expected in contract["source_sha256"].items():
        if split_tools.sha256_file(REPO_ROOT / name) != expected:
            raise RuntimeError(f"runtime source changed after freeze: {name}")
    for name, expected in contract["frozen_artifact_sha256"].items():
        if split_tools.sha256_file(REPO_ROOT / name) != expected:
            raise RuntimeError(f"frozen artifact changed after freeze: {name}")
    manifest = load_json(MANIFEST_PATHS[block])
    for row in manifest["entries"]:
        if split_tools.sha256_file(REPO_ROOT / row["scenario_file"]) != row["scenario_file_sha256"]:
            raise RuntimeError(f"scene file changed after freeze: {row['scenario_id']}")
    return contract, manifest


def strong_limiter(*, early_bypass: bool) -> Any:
    limiter = limiter_for("strong")
    if early_bypass:
        limiter.config = replace(
            limiter.config,
            early_bypass_enabled=True,
            warning_margin_m=float(limiter.config.comfortable_margin_m),
        )
    return limiter


def repaired_limiter(contract: Mapping[str, Any]) -> TurnSignPersistenceExecutionFilter:
    persistence = contract["persistence"]
    return TurnSignPersistenceExecutionFilter(
        strong_limiter(early_bypass=True),
        TurnSignPersistenceConfig(
            n_persist=2,
            a_rev_lat_mps2=float(persistence["horizontal_override_mps2"]),
            a_rev_vert_mps2=float(persistence["vertical_override_mps2"]),
            warning_margin_m=0.35,
        ),
    )


def run_block(block: str, arm: str, shard_index: int, shard_count: int) -> None:
    configure_split_tools()
    if block == "holdout":
        if not DEV_DECISION.exists() or load_json(DEV_DECISION).get("DEV_GATE") != "PASS":
            raise RuntimeError("Development did not authorize Holdout")
        if not MANIFEST_PATHS["holdout"].exists():
            raise RuntimeError("Holdout manifest has not been frozen")
    contract, manifest = verify_freeze(block)
    runtime, _ = split_tools._runtime(manifest)
    output = OUTPUT_DIRS[block][arm]
    output.mkdir(parents=True, exist_ok=True)
    completed = {path.stem for path in output.glob("*.json") if not path.name.endswith("_SOFTWARE_ERROR.json")}
    entries = [row for index, row in enumerate(manifest["entries"]) if index % shard_count == shard_index]
    for local_index, entry in enumerate(entries, start=1):
        sid = str(entry["scenario_id"])
        if sid in completed:
            continue
        print(f"[{block}:{arm}:{shard_index}/{shard_count}] {local_index:03d}/{len(entries):03d} start {sid}", flush=True)
        recorder = OnlineRuntimeRecorder()
        proxy = TimedPolicyProxy(runtime.policy, recorder)
        limiter = repaired_limiter(contract) if arm == "repaired" else strong_limiter(early_bypass=False)
        try:
            with recorder.instrument_dmp(), recorder.scoped_context(
                evaluation_block=f"turn_sign_persistence_{block}",
                configuration_id=("STRONG_EARLY_BYPASS_TURN_PERSISTENCE" if arm == "repaired" else "FROZEN_STRONG"),
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
                )
            save_episode_record(output, entry, episode, agents, events, triggers, extra)
            print(f"[{block}:{arm}] complete {sid} success={int(episode['team_success'])} collision={int(episode['collision'])}", flush=True)
        except Exception as error:
            split_tools.atomic_json(
                output / f"{sid}_SOFTWARE_ERROR.json",
                {"scenario_id": sid, "error_type": type(error).__name__, "error_message": str(error), "traceback": traceback.format_exc()},
            )
            raise


def prepare_holdout() -> None:
    configure_split_tools()
    if not DEV_DECISION.exists() or load_json(DEV_DECISION).get("DEV_GATE") != "PASS":
        raise RuntimeError("Development did not pass; Holdout generation is forbidden")
    if MANIFEST_PATHS["holdout"].exists():
        raise RuntimeError("Holdout manifest already exists; refusing overwrite")
    manifest = split_tools._generate_manifest("holdout")
    contract = load_json(RUNTIME_CONTRACT)
    contract["holdout"] = {
        "manifest": relative(MANIFEST_PATHS["holdout"]),
        "manifest_sha256": split_tools.sha256_file(MANIFEST_PATHS["holdout"]),
        "scenario_count": len(manifest["entries"]),
        "fixed_visual_scene_ids_before_outcomes": [f"TSP_HOLDOUT_{stage}_000" for stage in range(1, 5)],
        "generated_only_after_development_gate_pass": True,
    }
    split_tools.atomic_json(RUNTIME_CONTRACT, contract)
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
