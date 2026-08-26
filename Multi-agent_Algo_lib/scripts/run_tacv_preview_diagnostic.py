#!/usr/bin/env python3
"""Exact-behavior FP-SHEP transient diagnostic on frozen ERR episodes.

This runner does not change Proposal, FP-SHEP, GAT-R, ERR, SAC-DMP, or the
environment.  It wraps the existing batched FP-SHEP call only to retain the
already-computed H=4 candidate trajectories, then verifies that the executed
episode exactly reproduces the previously frozen Original-CRT control record.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as _pandas  # noqa: F401  # Windows torch/pyarrow import order


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
for search_path in (REPO_ROOT, ALGO_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import scripts.evaluate_gat_v1_err_development as evalmod  # noqa: E402
from planning.online_runtime_instrumentation import (  # noqa: E402
    OnlineRuntimeRecorder,
    TimedPolicyProxy,
)
from planning.semi_structured_long_range_benchmark import json_ready  # noqa: E402
from scripts.run_continuous_reference_transition import trajectory_arrays  # noqa: E402
from scripts.run_gat_recurrent_r_development import (  # noqa: E402
    FileBackedBuilder,
    load_json,
    resolve_runtime_config,
)
from scripts.run_long_range_development import DevelopmentRuntime  # noqa: E402


ARTIFACT_ROOT = REPO_ROOT / "artifacts/transient_aware_candidate_veto/20260824_184551"
CRT_ROOT = REPO_ROOT / "artifacts/continuous_reference_transition/20260824_132552"
SOURCE_ROOT = REPO_ROOT / "artifacts/gat_recurrent_smoothness_rescue/20260821_180244"
DEV_CONFIG = REPO_ROOT / "configs/evaluation/gat_recurrent_r_fp_anchor_dev.json"
HOLDOUT_CONFIG = REPO_ROOT / "configs/evaluation/gat_recurrent_r_fp_anchor_holdout.json"
BLOCK_CONTRACT = {
    "development": {
        "config": DEV_CONFIG,
        "manifest": CRT_ROOT / "00_context/CRT_DEVELOPMENT_MANIFEST.json",
        "source_records": CRT_ROOT / "04_development/records/original/episode_records",
    },
    "holdout": {
        "config": HOLDOUT_CONFIG,
        "manifest": CRT_ROOT / "00_context/CRT_HOLDOUT_MANIFEST.json",
        "source_records": CRT_ROOT / "07_holdout/records/original/episode_records",
    },
}
FORMAL_MANIFEST = SOURCE_ROOT / "10_formal_v2/FORMAL_V2_MANIFEST.json"
DT = 0.1
HORIZON = 4

SOURCE_PATHS = (
    "planning/policy_preview.py",
    "planning/pre_gat_closed_loop.py",
    "planning/candidate_execution_interface.py",
    "planning/heterogeneous_candidate_graph.py",
    "planning/event_triggered_reference_reconstruction.py",
    "Environment/frozen_sac_dmp_execution.py",
    "Multi-agent_Algo_lib/scripts/evaluate_gat_v1_err_development.py",
    "Multi-agent_Algo_lib/scripts/run_tacv_preview_diagnostic.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    data = list(rows)
    fields: list[str] = []
    for row in data:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["status"])
        writer.writeheader()
        writer.writerows(data)
    temporary.replace(path)


def source_line(path: Path, token: str) -> int:
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if token in line:
            return index
    raise RuntimeError(f"source token not found: {path}: {token}")


def prepare() -> None:
    for name in (
        "00_context",
        "01_preview_contract",
        "02_predictability",
        "03_replaceability",
        "04_gate_decision",
        "05_tacv_implementation",
        "06_development",
        "07_failure_audit",
        "08_holdout",
        "09_runtime",
        "10_figures",
        "11_freeze",
        "12_paper_ready",
    ):
        (ARTIFACT_ROOT / name).mkdir(parents=True, exist_ok=True)

    dev_manifest = load_json(BLOCK_CONTRACT["development"]["manifest"])
    holdout_manifest = load_json(BLOCK_CONTRACT["holdout"]["manifest"])
    if len(dev_manifest["entries"]) != 100 or len(holdout_manifest["entries"]) != 100:
        raise RuntimeError("diagnostic blocks must each contain exactly 100 scenarios")

    runtime_config = resolve_runtime_config(load_json(DEV_CONFIG), "gat_r")
    if int(runtime_config["H_preview"]) != HORIZON or not np.isclose(
        float(runtime_config["dt"]), DT
    ):
        raise RuntimeError("frozen H/dt contract mismatch")
    gat_checkpoint = REPO_ROOT / str(runtime_config["gat_checkpoint"])
    sac_checkpoint = REPO_ROOT / str(runtime_config["sac_checkpoint"])

    freeze = {
        "schema_version": "tacv_preview_diagnostic_freeze_v1",
        "status": "FROZEN_BEFORE_DIAGNOSTIC_EXECUTION",
        "artifact_root": str(ARTIFACT_ROOT.relative_to(REPO_ROOT).as_posix()),
        "development_scenario_count": 100,
        "holdout_scenario_count": 100,
        "development_manifest_sha256": sha256_file(BLOCK_CONTRACT["development"]["manifest"]),
        "holdout_manifest_sha256": sha256_file(BLOCK_CONTRACT["holdout"]["manifest"]),
        "formal_v2_manifest_sha256": sha256_file(FORMAL_MANIFEST),
        "formal_v2_execution_authorized": False,
        "diagnostic_behavior_reference": "Original arm of rejected CRT study; exact trajectory reproduction required",
        "source_sha256": {path: sha256_file(REPO_ROOT / path) for path in SOURCE_PATHS},
        "checkpoint_sha256": {
            "gat_r": sha256_file(gat_checkpoint),
            "sac_dmp": sha256_file(sac_checkpoint),
        },
        "closed_branches": {
            "CRT_BRANCH_STATUS": "CLOSED_REJECTED",
            "SECTOR_DENSIFICATION_STATUS": "CLOSED_INCOMPATIBLE",
        },
        "preexisting_test_audit": {
            "command": "pytest test/test_policy_preview.py test/test_pre_gat_closed_loop.py -q",
            "passed": 36,
            "failed": 1,
            "failure": "stale test expects FPSHEPOnlineScoreSpec(horizon!=4) to raise, while current diagnostic-capable source permits non-H4 specs",
            "impact_on_this_goal": "NONE; runtime config is independently asserted to H=4 before execution",
            "source_modified_to_satisfy_stale_test": False,
        },
        "forbidden_changes": [
            "training",
            "network checkpoint change",
            "Proposal/Top-K/FP-SHEP/GAT/ERR/SAC-DMP semantic change",
            "Formal V2 run",
        ],
    }
    atomic_json(ARTIFACT_ROOT / "00_context/PRE_DIAGNOSTIC_FREEZE.json", freeze)

    preview_source = REPO_ROOT / "planning/policy_preview.py"
    metric_source = REPO_ROOT / "planning/pre_gat_closed_loop.py"
    contract = {
        "schema_version": "fp_preview_transient_contract_v1",
        "FP_PREVIEW_ACCELERATION_AVAILABLE": "YES",
        "preview_horizon_steps": HORIZON,
        "dt_s": DT,
        "covered_time_s": HORIZON * DT,
        "initial_state_cloning": {
            "position": "copy of current real dynamics.p",
            "velocity": "copy of current real dynamics.v",
            "current_executed_acceleration": "latest_controller_info.applied_acceleration; zero only before first execution",
            "phase": "copy of current DMP phase",
            "active_candidate_goal": "independent candidate branch",
            "terminal_task_goal": "unchanged episode terminal goal",
            "scan_history": "independent current/previous scan copies",
            "branch_independence": "position, velocity, phase, scans, DMP transition, and clearance history are independent per candidate",
        },
        "sac_calls": {
            "calls_per_candidate": HORIZON,
            "batched_calls_per_agent_event": HORIZON,
            "policy": "frozen deterministic SAC actor",
        },
        "dmp_state_cloning": "phase-only historical DMP controller; no separate z state; each branch starts at the current real phase",
        "predicted_control_variables": [
            "positions",
            "velocities",
            "applied_accelerations",
            "commanded_accelerations",
            "SAC_actions",
            "DMP_phases",
            "reconstructed_current_and_previous_scans",
            "approximate_clearances",
        ],
        "available_acceleration_sequence": "PreviewTrajectory.accelerations, shape (H,3), applied acceleration after native dynamics clipping",
        "current_executed_acceleration_available": True,
        "clearance": {
            "source": "frozen current LiDAR visible-surface samples",
            "typed": False,
            "approximate": True,
            "minimum_clearance": "already computed",
            "time_to_minimum_clearance": "recoverable as dt*(argmin(clearances)+1) without changing rollout",
        },
        "peer_risk": {
            "source": "existing H4 graph constant-velocity interaction edges",
            "minimum_predicted_separation": "available",
            "risk_duration": "available",
            "d_safe_m": 0.6,
            "strict_condition": "distance < d_safe",
        },
        "PREVIEW_TRANSIENT_METRIC": {
            "acceleration_sequence": "a_hat[0]=current real applied acceleration; a_hat[h]=preview applied acceleration for h=1..H",
            "jerk_sequence": "j_hat[h]=(a_hat[h]-a_hat[h-1])/dt",
            "J_preview": "mean_h(sum_xyz(j_hat[h]^2)); identical discrete squared-jerk convention to trajectory_metrics",
            "J_preview_peak": "max_h(||j_hat[h]||_2)",
            "J_preview_mean": "mean_h(||j_hat[h]||_2)",
            "J_preview_vertical": "mean_h(j_hat_z[h]^2)",
            "J_preview_lateral": "mean_h(j_hat_x[h]^2+j_hat_y[h]^2)",
        },
        "source_locations": {
            "PreviewInitialState": {"file": "planning/policy_preview.py", "line": source_line(preview_source, "class PreviewInitialState")},
            "PreviewTrajectory": {"file": "planning/policy_preview.py", "line": source_line(preview_source, "class PreviewTrajectory")},
            "environment_state_clone": {"file": "planning/policy_preview.py", "line": source_line(preview_source, "def build_preview_inputs_from_env")},
            "serial_preview": {"file": "planning/policy_preview.py", "line": source_line(preview_source, "def preview_candidate")},
            "batched_exact_semantics_preview": {"file": "planning/policy_preview.py", "line": source_line(preview_source, "def preview_candidates_batched")},
            "existing_smoothness": {"file": "planning/pre_gat_closed_loop.py", "line": source_line(metric_source, "def trajectory_metrics")},
        },
        "new_predictor_or_rollout_semantics": False,
    }
    atomic_json(
        ARTIFACT_ROOT / "01_preview_contract/FP_PREVIEW_TRANSIENT_CONTRACT.json",
        contract,
    )

    gate_contract = {
        "schema_version": "tacv_diagnostic_gate_contract_v1",
        "frozen_before_diagnostic_results": True,
        "primary_target": "J_real_mean_H0p4 using the same four jerk samples as H4 preview",
        "clean_window": "no later accepted active-reference command for the same agent at steps s+1 through s+4 inclusive",
        "high_realized_jerk": "within-block P90 of J_real_peak_0p5",
        "high_preview_tail": "Development CLEAN-WINDOW P90 of J_preview; same numeric threshold applied to Holdout",
        "predictability": {
            "STRONG": "both blocks clean rho>=0.35, bootstrap CI lower>0, Q4/Q1 realized target ratio>=1.25, AUROC>=0.65",
            "MODERATE": "both blocks clean rho>=0.15, bootstrap CI lower>0, Q4/Q1 ratio>=1.10, AUROC>=0.58",
            "WEAK": "positive but either magnitude, uncertainty, calibration, or replication gate fails",
            "NONE": "nonpositive or directionally inconsistent association",
            "authorized_classes": ["STRONG", "MODERATE"],
        },
        "replaceability": {
            "strict_lower_transient": "J_preview_alt < J_preview_selected - 1e-12",
            "HIGH": "replaceable fraction >=0.35",
            "MODERATE": "replaceable fraction >=0.15 and <0.35",
            "LOW": "replaceable fraction >=0.05 and <0.15",
            "NONE": "replaceable fraction <0.05",
            "authorized_classes": ["HIGH", "MODERATE"],
        },
        "bootstrap": {
            "unit": "episode/scenario cluster",
            "replicates": 2000,
            "seed": 20260824,
        },
        "no_single_p_value_authorization": True,
    }
    atomic_json(ARTIFACT_ROOT / "00_context/DIAGNOSTIC_GATE_CONTRACT.json", gate_contract)


class PreviewCapture:
    def __init__(self) -> None:
        self.context: dict[str, Any] = {}
        self.rows: list[dict[str, Any]] = []
        self._original = evalmod.score_fp_shep_candidates_batched

    def install(self) -> None:
        capture = self

        def wrapped_score(*, env: Any, agent_index: int, proposals: Sequence[Any], policy: Any, spec: Any = None, timing_sink: dict[str, float] | None = None) -> Any:
            agent_index = int(agent_index)
            info = env.latest_controller_infos[agent_index] or {}
            a0 = np.asarray(info.get("applied_acceleration", np.zeros(3)), dtype=float)
            if a0.shape != (3,) or not np.all(np.isfinite(a0)):
                a0 = np.zeros(3, dtype=float)
            position = np.asarray(env.dynamics[agent_index].p, dtype=float).copy()
            velocity = np.asarray(env.dynamics[agent_index].v, dtype=float).copy()
            phase = float(env.dmps[agent_index].phase)
            records = capture._original(
                env=env,
                agent_index=agent_index,
                proposals=proposals,
                policy=policy,
                spec=spec,
                timing_sink=timing_sink,
            )
            dt = float(capture.context.get("dt_s", DT))
            for record in records:
                trajectory = record.preview.trajectory
                acceleration_sequence = np.vstack(
                    [a0[None, :], np.asarray(trajectory.accelerations, dtype=float)]
                )
                jerk = np.diff(acceleration_sequence, axis=0) / dt
                jerk_norm = np.linalg.norm(jerk, axis=1)
                clearances = np.asarray(trajectory.clearances, dtype=float)
                capture.rows.append(
                    {
                        **capture.context,
                        "agent_id": agent_index,
                        "candidate_id": int(record.candidate_id),
                        "candidate_world_position": np.asarray(record.candidate_world_position, dtype=float).tolist(),
                        "current_position": position.tolist(),
                        "current_velocity": velocity.tolist(),
                        "current_applied_acceleration": a0.tolist(),
                        "current_dmp_phase": phase,
                        "fp_shep_online_score": float(record.score),
                        "preview_task_progress": float(record.preview_task_progress),
                        "preview_min_clearance": float(record.preview_min_clearance),
                        "preview_time_to_min_clearance_s": float((int(np.argmin(clearances)) + 1) * dt),
                        "preview_max_execution_deviation": float(record.preview_max_execution_deviation),
                        "preview_terminal_speed": float(record.preview_terminal_speed),
                        "J_preview": float(np.mean(np.sum(jerk ** 2, axis=1))),
                        "J_preview_peak": float(np.max(jerk_norm)),
                        "J_preview_mean": float(np.mean(jerk_norm)),
                        "J_preview_vertical": float(np.mean(jerk[:, 2] ** 2)),
                        "J_preview_lateral": float(np.mean(np.sum(jerk[:, :2] ** 2, axis=1))),
                        "preview_positions": np.asarray(trajectory.positions, dtype=float).tolist(),
                        "preview_velocities": np.asarray(trajectory.velocities, dtype=float).tolist(),
                        "preview_applied_accelerations": np.asarray(trajectory.accelerations, dtype=float).tolist(),
                        "preview_commanded_accelerations": np.asarray(trajectory.commanded_accelerations, dtype=float).tolist(),
                        "preview_actions": np.asarray(trajectory.actions, dtype=float).tolist(),
                        "preview_clearances": clearances.tolist(),
                    }
                )
            return records

        evalmod.score_fp_shep_candidates_batched = wrapped_score

    def uninstall(self) -> None:
        evalmod.score_fp_shep_candidates_batched = self._original

    def builder(self, **kwargs: Any) -> dict[str, Any]:
        recorder = kwargs.get("runtime_recorder")
        context = dict(recorder.context) if recorder is not None else {}
        self.context = {
            "scenario_id": str(kwargs["scenario"]),
            "seed": int(kwargs["seed"]),
            "planning_decision_index": int(context.get("planning_decision_index", -1)),
            "step": int(context.get("event_step", -1)),
            "event_type": str(context.get("event_type", "UNKNOWN")),
            "dt_s": float(kwargs["config"]["dt"]),
        }
        return evalmod.build_online_gat_plan_optimized(**kwargs)


def jsonl_gzip(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_ready(row), ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    temporary.replace(path)


def event_signature(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        int(row["step"]),
        int(row["agent_id"]),
        str(row["event"]),
        bool(row.get("goal_changed", False)),
        row.get("selected_candidate_id"),
        bool(row.get("selected_null", False)),
    )


def compare_source(
    *,
    block: str,
    scenario_id: str,
    episode: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    source_json = Path(BLOCK_CONTRACT[block]["source_records"]) / f"{scenario_id}.json"
    source = load_json(source_json)
    source_npz = source_json.with_name(str(source["trajectory_file"]))
    with np.load(source_npz, allow_pickle=False) as frozen:
        array_checks = {
            name: bool(np.array_equal(np.asarray(arrays[name]), np.asarray(frozen[name])))
            for name in ("positions", "velocities", "commanded_accelerations", "applied_accelerations")
        }
    source_signatures = [event_signature(row) for row in source["events"]]
    rerun_signatures = [event_signature(row) for row in events]
    checks = {
        "team_success": bool(episode["team_success"]) == bool(source["episode"]["team_success"]),
        "collision": bool(episode["collision"]) == bool(source["episode"]["collision"]),
        "timeout": bool(episode["timeout"]) == bool(source["episode"]["timeout"]),
        "event_signatures_exact": rerun_signatures == source_signatures,
        **{f"array_{name}_exact": value for name, value in array_checks.items()},
    }
    return {
        "schema_version": "tacv_preview_episode_reproduction_v1",
        "block": block,
        "scenario_id": scenario_id,
        "source_record": str(source_json.relative_to(REPO_ROOT).as_posix()),
        "source_record_sha256": sha256_file(source_json),
        "source_trajectory_sha256": sha256_file(source_npz),
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def output_dir(block: str) -> Path:
    return ARTIFACT_ROOT / "02_predictability/diagnostic_runs" / block


def run(block: str, shard_index: int | None, shard_count: int | None, limit: int | None) -> None:
    contract = BLOCK_CONTRACT[block]
    config_payload = load_json(contract["config"])
    manifest = load_json(contract["manifest"])
    runtime_config = resolve_runtime_config(config_payload, "gat_r")
    runtime = DevelopmentRuntime(runtime_config, {"entries": []})
    runtime.builder = FileBackedBuilder(manifest, runtime_config, SOURCE_ROOT)
    target = output_dir(block)
    completed = {path.name.removesuffix("_reproduction.json") for path in (target / "episode_records").glob("*_reproduction.json")}
    indexed = list(enumerate(manifest["entries"]))
    if shard_count is not None:
        if shard_index is None or not 0 <= shard_index < shard_count:
            raise ValueError("invalid shard index/count")
        indexed = [item for item in indexed if item[0] % shard_count == shard_index]
    pending = [entry for _, entry in indexed if str(entry["scenario_id"]) not in completed]
    if limit is not None:
        pending = pending[:limit]
    for entry in pending:
        sid = str(entry["scenario_id"])
        print(f"[TACV-preview:{block}] start {sid}", flush=True)
        recorder = OnlineRuntimeRecorder()
        proxy = TimedPolicyProxy(runtime.policy, recorder)
        capture = PreviewCapture()
        capture.install()
        try:
            with recorder.instrument_dmp(), recorder.scoped_context(
                evaluation_block=f"tacv_preview_{block}",
                configuration_id="TACV_PREVIEW_DIAGNOSTIC_ORIGINAL",
                stage=entry["stage"],
                family=entry["family"],
                scenario_id=sid,
                seed=int(entry["seed"]),
                method="original_proposed_diagnostic_replay",
            ):
                episode, agents, events, triggers, extra = evalmod.run_episode(
                    config=runtime.eval_config,
                    settings=runtime.settings,
                    multi_config=runtime.multi_config,
                    policy=proxy,
                    gat_model=runtime.gat_model,
                    gat_device=runtime.gat_device,
                    method=evalmod.METHOD_RERR_GAT,
                    scenario=sid,
                    seed=int(entry["seed"]),
                    environment_builder=runtime.builder,
                    runtime_recorder=recorder,
                    upper_plan_builder=capture.builder,
                    crt=None,
                )
            arrays = trajectory_arrays(extra["path_rows"], int(episode["num_agents"]))
            reproduction = compare_source(
                block=block,
                scenario_id=sid,
                episode=episode,
                events=events,
                arrays=arrays,
            )
            if reproduction["status"] != "PASS":
                raise RuntimeError(f"frozen Original reproduction failed for {sid}: {reproduction}")
            records = target / "episode_records"
            jsonl_path = records / f"{sid}_candidate_previews.jsonl.gz"
            jsonl_gzip(jsonl_path, capture.rows)
            reproduction.update(
                {
                    "candidate_preview_row_count": len(capture.rows),
                    "candidate_preview_file": jsonl_path.name,
                    "candidate_preview_sha256": sha256_file(jsonl_path),
                    "upper_event_agent_count": len({(row["step"], row["agent_id"]) for row in capture.rows}),
                    "software_error": False,
                }
            )
            atomic_json(records / f"{sid}_reproduction.json", reproduction)
            print(
                f"[TACV-preview:{block}] complete {sid} previews={len(capture.rows)}",
                flush=True,
            )
        except Exception as error:
            atomic_json(
                target / "episode_records" / f"{sid}_SOFTWARE_ERROR.json",
                {
                    "scenario_id": sid,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
            raise
        finally:
            capture.uninstall()


def finalize(block: str) -> None:
    target = output_dir(block)
    manifest = load_json(BLOCK_CONTRACT[block]["manifest"])
    rows = [load_json(path) for path in sorted((target / "episode_records").glob("*_reproduction.json"))]
    errors = list((target / "episode_records").glob("*_SOFTWARE_ERROR.json"))
    summary = [
        {
            "block": block,
            "scenario_id": row["scenario_id"],
            "status": row["status"],
            "candidate_preview_row_count": row["candidate_preview_row_count"],
            "upper_event_agent_count": row["upper_event_agent_count"],
            "source_record_sha256": row["source_record_sha256"],
            "source_trajectory_sha256": row["source_trajectory_sha256"],
            "candidate_preview_sha256": row["candidate_preview_sha256"],
        }
        for row in rows
    ]
    write_csv(target / "DIAGNOSTIC_REPRODUCTION_SUMMARY.csv", summary)
    checks = {
        "episode_count": len(rows) == len(manifest["entries"]) == 100,
        "unique_scenarios": len({row["scenario_id"] for row in rows}) == 100,
        "all_exact_reproduction": all(row["status"] == "PASS" for row in rows),
        "no_software_errors": not errors,
        "all_preview_files_exist": all(
            (target / "episode_records" / row["candidate_preview_file"]).exists()
            for row in rows
        ),
    }
    reconciliation = {
        "schema_version": "tacv_preview_block_reconciliation_v1",
        "block": block,
        "checks": checks,
        "candidate_preview_row_count": sum(int(row["candidate_preview_row_count"]) for row in rows),
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    atomic_json(target / "RECONCILIATION.json", reconciliation)
    print(json.dumps(reconciliation, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "run", "finalize"))
    parser.add_argument("--block", choices=tuple(BLOCK_CONTRACT))
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare()
        return
    if args.block is None:
        raise ValueError("--block is required")
    if args.phase == "run":
        run(args.block, args.shard_index, args.shard_count, args.limit)
    else:
        finalize(args.block)


if __name__ == "__main__":
    main()
