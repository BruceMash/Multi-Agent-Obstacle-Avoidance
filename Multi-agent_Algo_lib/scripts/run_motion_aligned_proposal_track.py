#!/usr/bin/env python3
"""Track A: frozen-threshold Motion-Aligned Proposal experiment.

This script is intentionally unable to run Formal V2.  It derives one P90
threshold from existing successful Frozen Strong Development records, freezes a
new balanced Development block, and runs exactly A0/A1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import traceback
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pandas as _pandas  # noqa: F401 -- stable Windows torch import order


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
for search_path in (REPO_ROOT, ALGO_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import planning.motion_aligned_proposal as motion_module  # noqa: E402
from planning.motion_aligned_proposal import (  # noqa: E402
    MotionAlignedProposalConfig,
    MotionAlignedProposalEligibility,
    wrap_plan_builder,
)
from planning.online_runtime_instrumentation import OnlineRuntimeRecorder, TimedPolicyProxy  # noqa: E402
from scripts import evaluate_gat_v1_err_development as evaluator  # noqa: E402
from scripts import run_final_residual_zigzag_resolution as split_tools  # noqa: E402
from scripts.evaluate_gat_v1_err_development import METHOD_RERR_GAT, run_episode  # noqa: E402
from scripts.run_safety_adaptive_jerk_limiter import limiter_for, save_episode_record  # noqa: E402


ARTIFACT_ROOT = REPO_ROOT / "artifacts/parallel_zigzag_resolution/20260826_091334/track_A_motion_proposal"
COMMON_FREEZE = ARTIFACT_ROOT.parent / "00_context/COMMON_EXPERIMENT_FREEZE.json"
SOURCE_STRONG_DEV = REPO_ROOT / "artifacts/safety_adaptive_jerk_limiter/20260825_115704/dev_records/strong/episode_records"
METHOD_CONFIG = REPO_ROOT / "artifacts/gat_recurrent_smoothness_rescue/20260821_180244/09_final_freeze/method_configs/M9_Proposed_RERR_GAT_SAC_DMP.json"
THRESHOLD_PATH = ARTIFACT_ROOT / "MOTION_ALIGNED_PROPOSAL_THRESHOLD.json"
CONTRACT_PATH = ARTIFACT_ROOT / "MOTION_PROPOSAL_CONTRACT.json"
MICROTEST_PATH = ARTIFACT_ROOT / "MOTION_PROPOSAL_MICROTEST.json"
MANIFEST_PATHS = {
    "development": ARTIFACT_ROOT / "MOTION_PROPOSAL_DEV100_MANIFEST.json",
    "holdout": ARTIFACT_ROOT / "MOTION_PROPOSAL_HOLDOUT100_MANIFEST.json",
}
SCENE_DIRS = {
    "development": ARTIFACT_ROOT / "scenes/development",
    "holdout": ARTIFACT_ROOT / "scenes/holdout",
}
OUTPUT_DIRS = {
    block: {arm: ARTIFACT_ROOT / block / arm / "episode_records" for arm in ("a0_strong", "a1_motion")}
    for block in ("development", "holdout")
}
SEED_BASE = {"development": 4_500_000_000, "holdout": 4_600_000_000}
PREFIX = {"development": "MAP_DEV_", "holdout": "MAP_HOLDOUT_"}
DEV_DECISION = ARTIFACT_ROOT / "MOTION_PROPOSAL_DEV_GO_NO_GO.json"
VELOCITY_EPS = 1.0e-6


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def configure_split_tools() -> None:
    split_tools.ARTIFACT_RELATIVE = ARTIFACT_ROOT.relative_to(REPO_ROOT)
    split_tools.ARTIFACT_ROOT = ARTIFACT_ROOT
    split_tools.MANIFEST_PATHS = MANIFEST_PATHS
    split_tools.SCENE_DIRS = SCENE_DIRS
    split_tools.SEED_BASE = SEED_BASE
    split_tools.PREFIX = PREFIX
    additions = (
        REPO_ROOT / "artifacts/final_residual_zigzag_resolution/20260825_183146/FINAL_ZIGZAG_DEV100_MANIFEST.json",
        REPO_ROOT / "artifacts/turn_sign_persistence_zigzag_repair/20260825_222130/TURN_PERSISTENCE_DEV100_MANIFEST.json",
        REPO_ROOT / "artifacts/final_goal_aligned_sac_repair/20260825_003105/development_manifest_100.json",
    )
    split_tools.HISTORICAL_MANIFESTS = tuple(dict.fromkeys((*split_tools.HISTORICAL_MANIFESTS, *additions)))


def derive_threshold() -> None:
    if THRESHOLD_PATH.exists():
        raise RuntimeError("threshold is already frozen; refusing overwrite")
    common = load_json(COMMON_FREEZE)
    h_rep = float(common["paper_method"]["h_rep_m"])
    files = sorted(path for path in SOURCE_STRONG_DEV.glob("*.json") if not path.name.endswith("_SOFTWARE_ERROR.json"))
    if len(files) != 100:
        raise RuntimeError(f"expected 100 Strong Development records, found {len(files)}")
    angles: list[float] = []
    successful_records = 0
    eligible_events = 0
    excluded = Counter()
    source_hashes: dict[str, dict[str, str]] = {}
    for path in files:
        payload = load_json(path)
        sid = str(payload["entry_identity"]["scenario_id"])
        trajectory_path = path.with_name(str(payload["trajectory_file"]))
        trace_path = path.with_name(str(payload["limiter_trace_file"]))
        source_hashes[sid] = {
            "record": sha256_file(path),
            "trajectory": sha256_file(trajectory_path),
            "limiter_trace": sha256_file(trace_path),
        }
        if not bool(payload["episode"]["team_success"]):
            excluded["failed_episode"] += 1
            continue
        successful_records += 1
        arrays = np.load(trajectory_path)
        positions = np.asarray(arrays["positions"], dtype=float)
        velocities = np.asarray(arrays["velocities"], dtype=float)
        for event in payload.get("events", []):
            if event.get("selected_candidate_id") is None or event.get("new_active_goal_type") != "reference":
                excluded["null_or_nonproposal_event"] += 1
                continue
            margin = event.get("active_safety_margin_m")
            if margin is None or not float(margin) > h_rep:
                excluded["not_comfortable_safe"] += 1
                continue
            step, agent = int(event["step"]), int(event["agent_id"])
            if step >= len(positions):
                excluded["step_outside_trajectory"] += 1
                continue
            velocity = velocities[step, agent]
            speed = float(np.linalg.norm(velocity))
            if speed <= VELOCITY_EPS:
                excluded["low_speed"] += 1
                continue
            selected = int(event["selected_candidate_id"])
            points = event.get("candidate_world_points", [])
            if not 0 <= selected < len(points):
                excluded["selected_point_unavailable"] += 1
                continue
            direction = np.asarray(points[selected], dtype=float) - positions[step, agent]
            norm = float(np.linalg.norm(direction))
            if norm <= 1.0e-12:
                excluded["degenerate_reference"] += 1
                continue
            angle = math.acos(float(np.clip(np.dot(velocity / speed, direction / norm), -1.0, 1.0)))
            angles.append(float(angle))
            eligible_events += 1
    if not angles:
        raise RuntimeError("comfortable safe selected-reference angle distribution is empty")
    values = np.asarray(angles, dtype=float)
    percentiles = {str(key): float(np.percentile(values, key)) for key in (50, 75, 90, 95)}
    payload = {
        "schema_version": "motion_aligned_proposal_threshold_v1",
        "status": "FROZEN_BEFORE_TRACK_A_PERFORMANCE",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "record_root": relative(SOURCE_STRONG_DEV),
            "record_count": len(files),
            "successful_record_count": successful_records,
            "comfortable_safe_selected_proposal_event_count": eligible_events,
            "record_hashes_semantic_sha256": hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest(),
        },
        "sample_contract": {
            "episode_condition": "Frozen Strong Development team_success = true",
            "event_condition": "non-null actually selected Proposal/reference event",
            "comfortable_safe_condition": "pre-update active-direction margin m_t > h_rep = 0.35 m",
            "angle": "acos(clip(dot(current executed 3-D velocity direction, selected Proposal direction),-1,1))",
            "low_speed_exclusion_mps": VELOCITY_EPS,
            "low_speed_epsilon_source": "existing residual-zigzag direction metric numerical speed epsilon",
            "excluded_counts": dict(excluded),
        },
        "percentiles_rad": percentiles,
        "percentiles_deg": {key: float(np.degrees(value)) for key, value in percentiles.items()},
        "theta_max_statistic": "P90",
        "theta_max_rad": percentiles["90"],
        "theta_max_deg": float(np.degrees(percentiles["90"])),
        "threshold_grid": False,
        "performance_episodes_observed": 0,
    }
    atomic_json(THRESHOLD_PATH, payload)
    print(json.dumps({"status": "PASS", "theta_max_deg": payload["theta_max_deg"], "n": eligible_events}, indent=2))


@dataclass
class _DummyProposal:
    direction: np.ndarray
    distance_progress: float = 1.0


def microtest() -> None:
    threshold = load_json(THRESHOLD_PATH)
    original_safety = motion_module.active_direction_safety_margin
    motion_module.active_direction_safety_margin = lambda *args, **kwargs: SimpleNamespace(value_m=1.0)
    try:
        # Dense enough that the frozen ~P90 cone retains more than Top-K while
        # still excluding the outer synthetic directions.
        directions = [
            np.asarray([math.cos(angle), math.sin(angle), 0.0])
            for angle in np.linspace(-0.6, 0.6, 25)
        ]
        source = [_DummyProposal(direction=value) for value in directions]
        sensor = SimpleNamespace(ray_directions=np.zeros((16, 16, 3)))
        env = SimpleNamespace(
            sensors=[sensor], dmps=[SimpleNamespace(goal=np.asarray([1.0, 0.0, 0.0]))], steps=4
        )
        proposal_config = SimpleNamespace(positive_progress_epsilon=1.0e-9)
        original = lambda *args, **kwargs: source
        enabled = MotionAlignedProposalEligibility(
            MotionAlignedProposalConfig(theta_max_rad=float(threshold["theta_max_rad"]), enabled=True), original
        )
        enabled.reset_episode("micro")
        enabled.begin_plan(env)
        selected = enabled(np.zeros(3), np.ones(3), np.asarray([1.0, 0.0, 0.0]), None, sensor, proposal_config, 0.1)
        disabled = MotionAlignedProposalEligibility(
            MotionAlignedProposalConfig(theta_max_rad=float(threshold["theta_max_rad"]), enabled=False), original
        )
        disabled.reset_episode("micro")
        disabled.begin_plan(env)
        exact = disabled(np.zeros(3), np.ones(3), np.asarray([1.0, 0.0, 0.0]), None, sensor, proposal_config, 0.1)
        low = MotionAlignedProposalEligibility(
            MotionAlignedProposalConfig(theta_max_rad=float(threshold["theta_max_rad"]), enabled=True), original
        )
        low.reset_episode("micro")
        low.begin_plan(env)
        low_result = low(np.zeros(3), np.ones(3), np.zeros(3), None, sensor, proposal_config, 0.1)
        checks = {
            "a0_returns_same_list_objects_and_order": len(exact) == len(source) and all(a is b for a, b in zip(exact, source)),
            "ordinary_safe_cone_filters_without_padding": 10 <= len(selected) < len(source) and all(any(item is source_item for source_item in source) for item in selected),
            "low_speed_full_sphere_fallback_exact": len(low_result) == len(source) and all(a is b for a, b in zip(low_result, source)),
            "raw_sensor_direction_count_remains_256": low.rows[0]["raw_direction_count"] == 256,
            "top_k_interface_preserved": enabled.rows[0]["top_k_construction_success"],
        }
    finally:
        motion_module.active_direction_safety_margin = original_safety
    payload = {"schema_version": "motion_proposal_microtest_v1", "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}
    atomic_json(MICROTEST_PATH, payload)
    if payload["status"] != "PASS":
        raise RuntimeError(payload)
    print(json.dumps(payload, indent=2))


def prepare_development() -> None:
    configure_split_tools()
    if load_json(MICROTEST_PATH).get("status") != "PASS":
        raise RuntimeError("microtest must pass before Development freeze")
    if CONTRACT_PATH.exists():
        raise RuntimeError("contract already frozen; refusing overwrite")
    manifest = split_tools._generate_manifest("development")
    cells = Counter((row["stage"], row["family"]) for row in manifest["entries"])
    if len(manifest["entries"]) != 100 or len(cells) != 20 or set(cells.values()) != {5}:
        raise RuntimeError(f"Development is not balanced 4x5x5: {cells}")
    visual_ids = [f"MAP_DEV_{stage}_000" for stage in range(1, 5)]
    threshold = load_json(THRESHOLD_PATH)
    common = load_json(COMMON_FREEZE)
    config = load_json(METHOD_CONFIG)
    contract = {
        "schema_version": "motion_proposal_contract_v1",
        "status": "FROZEN_BEFORE_MOTION_PROPOSAL_DEV100_PERFORMANCE",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "arms": {"A0": "Frozen Strong", "A1": "Motion-Aligned Proposal + Frozen Strong"},
        "eligibility": {
            "ordinary_safe": "theta_k <= theta_max relative to current executed 3-D velocity",
            "theta_max_rad": threshold["theta_max_rad"],
            "theta_max_deg": threshold["theta_max_deg"],
            "threshold_statistic": "P90 exactly; no grid",
            "velocity_epsilon_mps": VELOCITY_EPS,
            "fallback_A": "m_t <= h_rep=0.35 m",
            "fallback_B": "3-D speed <= 1e-6 m/s",
            "fallback_C": "fewer than 10 feasible cone candidates",
            "fallback_D": "restricted set cannot construct Top-K 10",
            "fallback_E": "m_t <= h_emg=0 or frozen Proposal has no positive-progress escape",
            "full_sphere_return_is_original_list": True,
            "invented_padding": False,
        },
        "frozen_interfaces": common["paper_method"],
        "frozen_checkpoints": common["checkpoints"],
        "frozen_strong": common["common_execution_baseline"],
        "development": {
            "manifest": relative(MANIFEST_PATHS["development"]),
            "manifest_sha256": sha256_file(MANIFEST_PATHS["development"]),
            "scenario_count": 100,
            "balance": "4 stages x 5 families x 5 scenes",
            "fixed_visual_scene_ids_before_outcomes": visual_ids,
            "result_based_scene_replacement_forbidden": True,
        },
        "development_gates": {
            "maximum_success_loss_pp": 1.0,
            "maximum_peer_collision_increase_pp": 1.0,
            "maximum_obstacle_collision_increase_pp": 1.0,
            "minimum_yaw_or_pitch_reversal_reduction_percent": 20.0,
            "minimum_long_arc_scenes": 3,
            "long_arc_objective_proxy": "combined yaw+pitch reversal rate decreases AND combined yaw+pitch directional TV decreases",
            "maximum_smoothness_cost_increase_percent": 5.0,
            "all_required": True,
        },
        "holdout_rule": "generate independent Holdout100 only after Development PASS; no tuning",
        "formal_execution_authorized": False,
        "source_sha256": {
            relative(Path(__file__)): sha256_file(Path(__file__)),
            "planning/motion_aligned_proposal.py": sha256_file(REPO_ROOT / "planning/motion_aligned_proposal.py"),
            "Guidance/reference_point_proposal_demo.py": sha256_file(REPO_ROOT / "Guidance/reference_point_proposal_demo.py"),
            relative(METHOD_CONFIG): sha256_file(METHOD_CONFIG),
            relative(COMMON_FREEZE): sha256_file(COMMON_FREEZE),
        },
        "config_fairness": {
            "top_k": int(config["top_k"]),
            "H_preview": int(config["H_preview"]),
            "sensor_direction_count": int(config["sensor"]["direction_count"]),
            "gat_projection_direction_count": int(config["sensor"]["gat_canonical_projection_directions"]),
        },
        "performance_episodes_observed": 0,
    }
    atomic_json(CONTRACT_PATH, contract)
    print(json.dumps({"status": "PASS", "visual_ids": visual_ids}, indent=2))


def verify_freeze(block: str) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = load_json(CONTRACT_PATH)
    for name, expected in contract["source_sha256"].items():
        if sha256_file(REPO_ROOT / name) != expected:
            raise RuntimeError(f"Track A frozen source changed: {name}")
    manifest = load_json(MANIFEST_PATHS[block])
    for row in manifest["entries"]:
        if sha256_file(REPO_ROOT / row["scenario_file"]) != row["scenario_file_sha256"]:
            raise RuntimeError(f"scene changed: {row['scenario_id']}")
    return contract, manifest


def run_block(block: str, arm: str, shard_index: int, shard_count: int) -> None:
    if block == "holdout":
        if not DEV_DECISION.exists() or load_json(DEV_DECISION).get("DEV_GATE") != "PASS":
            raise RuntimeError("Development did not authorize Holdout")
    configure_split_tools()
    contract, manifest = verify_freeze(block)
    runtime, _ = split_tools._runtime(manifest)
    output = OUTPUT_DIRS[block][arm]
    output.mkdir(parents=True, exist_ok=True)
    entries = [row for index, row in enumerate(manifest["entries"]) if index % shard_count == shard_index]
    completed = {path.stem for path in output.glob("*.json") if "_motion_proposal_trace" not in path.stem and not path.name.endswith("_SOFTWARE_ERROR.json")}
    original_propose = evaluator.propose_reference_points
    eligibility = MotionAlignedProposalEligibility(
        MotionAlignedProposalConfig(
            theta_max_rad=float(contract["eligibility"]["theta_max_rad"]),
            top_k=10,
            warning_margin_m=0.35,
            emergency_margin_m=0.0,
            velocity_epsilon_mps=VELOCITY_EPS,
            enabled=(arm == "a1_motion"),
        ),
        original_propose,
    )
    evaluator.propose_reference_points = eligibility
    plan_builder = wrap_plan_builder(evaluator.build_online_gat_plan_optimized, eligibility)
    try:
        for local_index, entry in enumerate(entries, start=1):
            sid = str(entry["scenario_id"])
            if sid in completed:
                continue
            print(f"[{block}:{arm}:{shard_index}/{shard_count}] {local_index}/{len(entries)} {sid}", flush=True)
            eligibility.reset_episode(sid)
            recorder = OnlineRuntimeRecorder()
            proxy = TimedPolicyProxy(runtime.policy, recorder)
            limiter = limiter_for("strong")
            try:
                with recorder.instrument_dmp(), recorder.scoped_context(
                    evaluation_block=f"motion_proposal_{block}", configuration_id=arm,
                    stage=entry["stage"], family=entry["family"], scenario_id=sid,
                    seed=int(entry["seed"]), method=METHOD_RERR_GAT,
                ):
                    episode, agents, events, triggers, extra = run_episode(
                        config=runtime.eval_config, settings=runtime.settings,
                        multi_config=runtime.multi_config, policy=proxy,
                        gat_model=runtime.gat_model, gat_device=runtime.gat_device,
                        method=METHOD_RERR_GAT, scenario=sid, seed=int(entry["seed"]),
                        environment_builder=runtime.builder, runtime_recorder=recorder,
                        upper_plan_builder=plan_builder,
                        execution_acceleration_limiter=limiter,
                    )
                save_episode_record(output, entry, episode, agents, events, triggers, extra)
                trace_path = output / f"{sid}_motion_proposal_trace.json"
                atomic_json(trace_path, {"schema_version": "motion_proposal_trace_v1", "rows": eligibility.rows})
                record_path = output / f"{sid}.json"
                record = load_json(record_path)
                record["motion_proposal_summary"] = eligibility.summary()
                record["motion_proposal_trace_file"] = trace_path.name
                record["motion_proposal_trace_sha256"] = sha256_file(trace_path)
                atomic_json(record_path, record)
                print(f"[{block}:{arm}] complete {sid} success={int(episode['team_success'])}", flush=True)
            except Exception as error:
                atomic_json(output / f"{sid}_SOFTWARE_ERROR.json", {
                    "scenario_id": sid, "error_type": type(error).__name__,
                    "error_message": str(error), "traceback": traceback.format_exc(),
                })
                raise
    finally:
        evaluator.propose_reference_points = original_propose


def prepare_holdout() -> None:
    configure_split_tools()
    if not DEV_DECISION.exists() or load_json(DEV_DECISION).get("DEV_GATE") != "PASS":
        raise RuntimeError("Development failed; Holdout generation forbidden")
    manifest = split_tools._generate_manifest("holdout")
    contract = load_json(CONTRACT_PATH)
    contract["holdout"] = {
        "manifest": relative(MANIFEST_PATHS["holdout"]),
        "manifest_sha256": sha256_file(MANIFEST_PATHS["holdout"]),
        "scenario_count": 100,
        "fixed_visual_scene_ids_before_outcomes": [f"MAP_HOLDOUT_{stage}_000" for stage in range(1, 5)],
        "generated_after_development_pass": True,
    }
    atomic_json(CONTRACT_PATH, contract)
    print(json.dumps({"status": "PASS", "n": len(manifest["entries"])}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("derive-threshold")
    sub.add_parser("microtest")
    sub.add_parser("prepare-dev")
    sub.add_parser("prepare-holdout")
    run = sub.add_parser("run")
    run.add_argument("--block", choices=("development", "holdout"), required=True)
    run.add_argument("--arm", choices=("a0_strong", "a1_motion"), required=True)
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if args.command == "derive-threshold":
        derive_threshold()
    elif args.command == "microtest":
        microtest()
    elif args.command == "prepare-dev":
        prepare_development()
    elif args.command == "prepare-holdout":
        prepare_holdout()
    else:
        run_block(args.block, args.arm, args.shard_index, args.shard_count)


if __name__ == "__main__":
    main()
