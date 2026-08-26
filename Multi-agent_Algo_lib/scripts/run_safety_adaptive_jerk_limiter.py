#!/usr/bin/env python3
"""Final bounded no-retraining safety-adaptive jerk-limiter experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
import traceback
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as _pandas  # noqa: F401 -- stable Windows torch import order


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
for search_path in (REPO_ROOT, ALGO_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from Entity.KinematicModel import propagate_point_mass  # noqa: E402
from Guidance.reference_point_proposal_demo import ProposalConfig  # noqa: E402
from planning.event_triggered_reference_reconstruction import (  # noqa: E402
    active_direction_safety_margin,
)
from planning.online_runtime_instrumentation import (  # noqa: E402
    OnlineRuntimeRecorder,
    TimedPolicyProxy,
)
from planning.safety_adaptive_jerk_limiter import (  # noqa: E402
    SafetyAdaptiveJerkLimiterConfig,
    SafetyAdaptiveVectorJerkLimiter,
)
from scripts.evaluate_gat_v1_err_development import (  # noqa: E402
    METHOD_RERR_GAT,
    build_online_gat_plan_optimized,
    run_episode,
)
from scripts.run_gat_recurrent_r_development import FileBackedBuilder  # noqa: E402
from scripts.run_long_range_development import DevelopmentRuntime  # noqa: E402


ARTIFACT_RELATIVE = Path("artifacts/safety_adaptive_jerk_limiter/20260825_115704")
ARTIFACT_ROOT = REPO_ROOT / ARTIFACT_RELATIVE
SOURCE_STUDY_ROOT = REPO_ROOT / "artifacts/gat_recurrent_smoothness_rescue/20260821_180244"
SOURCE_DEV_MANIFEST = SOURCE_STUDY_ROOT / "07_development/GAT_RS_DEV_SCENE_MANIFEST.json"
SOURCE_HOLDOUT_MANIFEST = SOURCE_STUDY_ROOT / "08_holdout/GAT_RS_HOLDOUT_SCENE_MANIFEST.json"
FINAL_METHOD_CONFIG = SOURCE_STUDY_ROOT / "09_final_freeze/method_configs/M9_Proposed_RERR_GAT_SAC_DMP.json"
SOURCE_ORIGINAL_DEV = REPO_ROOT / "artifacts/continuous_reference_transition/20260824_132552/04_development/records/original/episode_records"
DEV_RECORD_ROOT = ARTIFACT_ROOT / "dev_records"
CONTRACT_PATH = ARTIFACT_ROOT / "JERK_LIMITER_CONTROL_CONTRACT.json"
THRESHOLD_PATH = ARTIFACT_ROOT / "JERK_LIMITER_THRESHOLDS.json"
MICROTEST_PATH = ARTIFACT_ROOT / "JERK_LIMITER_MICROTEST.json"
DT = 0.1
ACCELERATION_BOUND = 4.0
MAX_SPEED_NORM = 3.2
J_FREE = 8.0 * math.sqrt(3.0) / DT
VARIANT_PERCENTILES = {"mild": 90, "medium": 75, "strong": 60}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_ready(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    rows = list(rows)
    names = list(fields or [])
    for row in rows:
        for key in row:
            if key not in names:
                names.append(str(key))
    if not names:
        names = ["status"]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_ready(row.get(key)) for key in names})
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def distribution(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not array.size:
        return {"count": 0}
    percentiles = (5, 25, 50, 60, 75, 90, 95, 99)
    result = {f"P{value:02d}": float(np.percentile(array, value)) for value in percentiles}
    result.update(
        count=int(array.size),
        mean=float(np.mean(array)),
        maximum=float(np.max(array)),
    )
    return result


def source_ids() -> list[str]:
    ids = sorted(
        path.stem
        for path in SOURCE_ORIGINAL_DEV.glob("GATRS_DEV_*.json")
        if not path.name.endswith("_SOFTWARE_ERROR.json")
    )
    if len(ids) != 100:
        raise RuntimeError(f"expected frozen Development100 source, found {len(ids)}")
    return ids


def filtered_manifest(source_path: Path = SOURCE_DEV_MANIFEST, selected_ids: Sequence[str] | None = None) -> dict[str, Any]:
    source = load_json(source_path)
    selected = set(source_ids() if selected_ids is None else map(str, selected_ids))
    entries = [row for row in source["entries"] if str(row["scenario_id"]) in selected]
    if len(entries) != len(selected):
        raise RuntimeError("manifest is missing one or more selected scenes")
    return {**source, "entries": entries, "unique_scenario_count": len(entries)}


def validate_development_balance(manifest: Mapping[str, Any]) -> None:
    entries = list(manifest["entries"])
    cells = Counter((str(row["stage"]), str(row["family"])) for row in entries)
    if len(entries) != 100 or len(cells) != 20 or set(cells.values()) != {5}:
        raise RuntimeError(f"Development block is not frozen 4x5x5: {cells}")


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def write_contract() -> None:
    config = load_json(FINAL_METHOD_CONFIG)
    if not np.isclose(config["dt"], DT, rtol=0.0, atol=0.0):
        raise RuntimeError("frozen dt changed")
    if not np.isclose(config["maximum_speed_norm_mps"], MAX_SPEED_NORM, rtol=0.0, atol=0.0):
        raise RuntimeError("frozen speed norm changed")
    manifest = filtered_manifest()
    validate_development_balance(manifest)
    tracked = [
        REPO_ROOT / "planning/safety_adaptive_jerk_limiter.py",
        REPO_ROOT / "planning/historical_forcing_gate.py",
        REPO_ROOT / "Environment/frozen_sac_dmp_execution.py",
        REPO_ROOT / "Environment/multi_agent_dmp_env.py",
        REPO_ROOT / "Entity/KinematicModel.py",
        REPO_ROOT / "planning/event_triggered_reference_reconstruction.py",
        REPO_ROOT / "Multi-agent_Algo_lib/scripts/evaluate_gat_v1_err_development.py",
        Path(__file__).resolve(),
    ]
    source_hashes = {relative(path): sha256_file(path) for path in tracked}
    payload = {
        "schema_version": "safety_adaptive_vector_jerk_limiter_contract_v1",
        "status": "FROZEN_BEFORE_THRESHOLD_RECOVERY_AND_PERFORMANCE",
        "scope": "execution-local acceleration handoff only",
        "exact_handoff": {
            "raw_quantity": "historical SAC-DMP transition commanded_acceleration in m/s^2",
            "previous_quantity": "previous actual physically applied acceleration for the same agent",
            "vector_rule": "project raw-minus-previous onto an L2 ball of radius j_max*dt",
            "order": [
                "frozen SAC actor",
                "frozen historical DMP acceleration",
                "safety-adaptive vector jerk projection",
                "existing componentwise acceleration saturation",
                "existing componentwise velocity saturation",
                "existing 3.2 m/s velocity-norm cap",
                "existing state propagation",
            ],
            "dt_s": DT,
            "existing_acceleration_bounds_mps2": [[-ACCELERATION_BOUND] * 3, [ACCELERATION_BOUND] * 3],
            "existing_maximum_speed_norm_mps": MAX_SPEED_NORM,
            "initial_previous_executed_acceleration_mps2": [0.0, 0.0, 0.0],
            "first_control_jerk_recorded_but_excluded_from_diff-based tail summaries": False,
        },
        "safety_adaptation": {
            "legal_runtime_quantity": "existing R-ERR active_direction_safety_margin.value_m",
            "quantity_source": "current 256-ray Proposal safety field nearest the active-reference direction",
            "no_new_prediction_or_future_state": True,
            "emergency_margin_m": float(config["err"]["h_emg_m"]),
            "comfortable_margin_m": float(config["err"]["h_rep_m"]),
            "q_safe_formula": "clip((h_active-h_emg)/(h_rep-h_emg),0,1)",
            "j_max_formula": "j_smooth + (1-q_safe)*(j_free-j_smooth)",
            "mandatory_hard_bypass": "h_active <= h_emg passes raw DMP acceleration directly to original physical saturation",
            "hard_bypass_is_level_condition_each_control_step": True,
            "j_free_mps3": J_FREE,
            "j_free_basis": "maximum possible change between opposite corners of the frozen +/-4 m/s^2 acceleration box in one 0.1 s tick",
        },
        "threshold_contract": {
            "source": relative(SOURCE_ORIGINAL_DEV),
            "source_arm": "Original frozen Development100",
            "state_definition": "safe iff reconstructed existing h_active >= frozen h_rep=0.35 m",
            "mild": "safe-state actual executed vector jerk P90",
            "medium": "safe-state actual executed vector jerk P75",
            "strong": "safe-state actual executed vector jerk P60",
            "threshold_tuning_allowed": False,
        },
        "development_contract": {
            "manifest": relative(SOURCE_DEV_MANIFEST),
            "manifest_sha256": sha256_file(SOURCE_DEV_MANIFEST),
            "scenario_count": 100,
            "balance": "4 stages x 5 families x 5 scenes",
            "arms": ["Original", "Mild", "Medium", "Strong"],
            "paired_scenes": True,
            "predeclared_fixed_visual_scenes": [f"GATRS_DEV_{stage}_000" for stage in range(1, 5)],
        },
        "frozen_method": {
            "config": relative(FINAL_METHOD_CONFIG),
            "config_sha256": sha256_file(FINAL_METHOD_CONFIG),
            "gat_checkpoint": config["gat_checkpoint"],
            "gat_checkpoint_sha256": sha256_file(REPO_ROOT / config["gat_checkpoint"]),
            "sac_checkpoint": config["sac_checkpoint"],
            "sac_checkpoint_sha256": sha256_file(REPO_ROOT / config["sac_checkpoint"]),
            "proposal_top_k": int(config["top_k"]),
            "fp_shep_horizon": int(config["H_preview"]),
            "sensor_directions": int(config["sensor"]["direction_count"]),
            "gat_projection_directions": int(config["sensor"]["gat_canonical_projection_directions"]),
            "unchanged_components": [
                "Proposal", "Top-K 10", "FP-SHEP H4", "GAT-R", "R-ERR",
                "active references", "SAC checkpoint/observation/reward", "DMP",
                "dynamics", "collision and success definitions",
            ],
        },
        "development_gates_frozen_before_results": {
            "maximum_team_success_loss_pp": 1.0,
            "maximum_peer_collision_increase_pp": 1.0,
            "minimum_smoothness_reduction_percent": 15.0,
            "preferred_smoothness_reduction_percent": 20.0,
            "jerk_tail_gate": "pooled P90 and P95 must both decrease; paired episode-bootstrap 95% CI for mean per-episode P90 and P95 change must exclude zero",
            "visual_gate": "at least 3 of 4 fixed scenes reduce both episode smoothness and P95 jerk by >=15%, with no fixed scene worsening either by >10%",
            "selection_order_when_comparable": ["Mild", "Medium", "Strong"],
            "systematic_peer_increase_immediate_reject": True,
        },
        "formal_v2_original_result_preserved": {
            "success": "381/400 (95.25%)",
            "new_formal_not_authorized_by_this_phase": True,
        },
        "source_sha256": source_hashes,
        "performance_episodes_executed_by_contract_phase": 0,
    }
    atomic_json(CONTRACT_PATH, payload)
    print(json.dumps({"status": "PASS", "artifact": relative(CONTRACT_PATH)}, indent=2))


def runtime_for(manifest: Mapping[str, Any]) -> tuple[DevelopmentRuntime, dict[str, Any]]:
    config = load_json(FINAL_METHOD_CONFIG)
    runtime = DevelopmentRuntime(config, {"entries": []})
    runtime.builder = FileBackedBuilder(manifest, config, SOURCE_STUDY_ROOT)
    return runtime, config


def _set_recorded_state(env: Any, scene: Mapping[str, Any], arrays: Any, state_step: int) -> None:
    positions = np.asarray(arrays["positions"][state_step], dtype=float)
    velocities = np.asarray(arrays["velocities"][state_step], dtype=float)
    for agent_id in range(int(env.num_agents)):
        env.dynamics[agent_id].p = positions[agent_id].copy()
        env.dynamics[agent_id].v = velocities[agent_id].copy()
    tracks = scene.get("dynamic_obstacle_trajectories", [])
    for obstacle_id, obstacle in enumerate(env.dynamic_obstacles):
        track = np.asarray(tracks[obstacle_id], dtype=float)
        obstacle.center = track[min(int(state_step), len(track) - 1)].copy()
    env.steps = int(state_step)
    for agent_id in range(int(env.num_agents)):
        env.latest_sensor_packets[agent_id] = env.sensors[agent_id].sense(
            env.dynamics[agent_id].p,
            env.dynamics[agent_id].v,
            env.goals[agent_id],
            env._sensor_static_obstacles(),
            env._sensor_dynamic_obstacles(agent_id),
        )


def recover_thresholds() -> None:
    if not CONTRACT_PATH.exists():
        raise RuntimeError("control contract must be frozen first")
    manifest = filtered_manifest()
    validate_development_balance(manifest)
    runtime, config = runtime_for(manifest)
    proposal = ProposalConfig(**dict(config["proposal_config"]))
    safe: list[float] = []
    intermediate: list[float] = []
    critical: list[float] = []
    all_values: list[float] = []
    margins: list[float] = []
    entry_by_id = {str(row["scenario_id"]): row for row in manifest["entries"]}
    started = time.perf_counter()
    for episode_index, sid in enumerate(source_ids(), start=1):
        entry = entry_by_id[sid]
        scene = load_json(SOURCE_STUDY_ROOT / str(entry["scenario_file"]))
        record = load_json(SOURCE_ORIGINAL_DEV / f"{sid}.json")
        arrays = np.load(SOURCE_ORIGINAL_DEV / f"{sid}_trajectory.npz")
        env, _ = runtime.builder(
            config=runtime.multi_config,
            scenario=sid,
            seed=int(entry["seed"]),
            peer_radius=float(config["peer_radius"]),
        )
        try:
            acceleration = np.asarray(
                arrays["applied_accelerations_full"], dtype=float
            ).copy()
            # Stored trajectory row zero precedes the first control action and
            # therefore carries NaN by construction.  The actual previous
            # applied acceleration at reset is the frozen zero vector.
            if not np.all(np.isfinite(acceleration[0])):
                acceleration[0] = 0.0
            goals = np.asarray(arrays["g_cmd"], dtype=float)
            completion = {
                int(row["agent_id"]): (
                    int(row["terminal_completion_step"])
                    if row.get("terminal_completion_step") is not None
                    else acceleration.shape[0] - 1
                )
                for row in record["agents"]
            }
            for step in range(acceleration.shape[0] - 1):
                _set_recorded_state(env, scene, arrays, step)
                for agent_id in range(int(env.num_agents)):
                    if step >= completion[agent_id]:
                        continue
                    margin = float(
                        active_direction_safety_margin(
                            env, agent_id, goals[step, agent_id], proposal
                        ).value_m
                    )
                    jerk = float(
                        np.linalg.norm(
                            acceleration[step + 1, agent_id]
                            - acceleration[step, agent_id]
                        )
                        / DT
                    )
                    if not np.isfinite(margin) or not np.isfinite(jerk):
                        raise FloatingPointError("non-finite reconstructed threshold sample")
                    margins.append(margin)
                    all_values.append(jerk)
                    if margin <= float(config["err"]["h_emg_m"]):
                        critical.append(jerk)
                    elif margin >= float(config["err"]["h_rep_m"]):
                        safe.append(jerk)
                    else:
                        intermediate.append(jerk)
        finally:
            env.close()
        print(
            f"[thresholds] {episode_index:03d}/100 {sid} safe_samples={len(safe)}",
            flush=True,
        )
    safe_array = np.asarray(safe, dtype=float)
    if safe_array.size < 1000:
        raise RuntimeError("insufficient frozen safe-state samples")
    percentiles = {value: float(np.percentile(safe_array, value)) for value in (60, 75, 90)}
    payload = {
        "schema_version": "safety_adaptive_jerk_limiter_thresholds_v1",
        "status": "FROZEN_FROM_ORIGINAL_DEVELOPMENT_BEFORE_VARIANT_PERFORMANCE",
        "source_record_root": relative(SOURCE_ORIGINAL_DEV),
        "source_scenario_count": 100,
        "threshold_recovery_is_read_only_noninterventional": True,
        "state_alignment": "margin at stored state t paired with ||a_applied[t+1]-a_applied[t]||/dt",
        "completed_agent_post-completion_ticks_excluded": True,
        "safe_state_definition": f"h_active >= {float(config['err']['h_rep_m']):.6f} m",
        "critical_state_definition": f"h_active <= {float(config['err']['h_emg_m']):.6f} m",
        "jerk_distributions_mps3": {
            "all_active_agent_steps": distribution(all_values),
            "safe": distribution(safe),
            "intermediate": distribution(intermediate),
            "critical": distribution(critical),
        },
        "margin_distribution_m": distribution(margins),
        "variants": {
            "mild": {"source_percentile": 90, "j_smooth_mps3": percentiles[90]},
            "medium": {"source_percentile": 75, "j_smooth_mps3": percentiles[75]},
            "strong": {"source_percentile": 60, "j_smooth_mps3": percentiles[60]},
        },
        "j_free_mps3": J_FREE,
        "threshold_tuning_after_performance": False,
        "elapsed_s": float(time.perf_counter() - started),
        "contract_sha256": sha256_file(CONTRACT_PATH),
    }
    atomic_json(THRESHOLD_PATH, payload)
    print(json.dumps({"status": "PASS", "variants": payload["variants"]}, indent=2))


def limiter_for(variant: str, *, enabled: bool = True) -> SafetyAdaptiveVectorJerkLimiter:
    thresholds = load_json(THRESHOLD_PATH)
    config = load_json(FINAL_METHOD_CONFIG)
    return SafetyAdaptiveVectorJerkLimiter(
        SafetyAdaptiveJerkLimiterConfig(
            dt=DT,
            j_smooth_mps3=float(thresholds["variants"][variant]["j_smooth_mps3"]),
            j_free_mps3=float(thresholds["j_free_mps3"]),
            emergency_margin_m=float(config["err"]["h_emg_m"]),
            comfortable_margin_m=float(config["err"]["h_rep_m"]),
            enabled=bool(enabled),
        ),
        num_agents=int(config["num_agents"]),
    )


def _transition_microtests() -> dict[str, Any]:
    limiter = limiter_for("mild")
    limiter.reset()
    limiter.set_context(0, step=0, safety_margin_m=1.0, time_since_reference_change_s=0.0)
    unchanged = limiter.limit(0, np.zeros(3))
    limiter.observe_executed(0, unchanged)
    unchanged_pass = bool(np.array_equal(unchanged, np.zeros(3)))
    limiter.set_context(0, step=1, safety_margin_m=1.0, time_since_reference_change_s=0.1)
    raw = np.asarray([4.0, -4.0, 4.0])
    limited = limiter.limit(0, raw)
    maximum = float(limiter.config.j_smooth_mps3 * DT)
    projection_pass = bool(
        np.isclose(np.linalg.norm(limited), maximum, rtol=1.0e-12, atol=1.0e-12)
        and np.allclose(limited / np.linalg.norm(limited), raw / np.linalg.norm(raw), rtol=1.0e-12, atol=1.0e-12)
    )
    limiter.observe_executed(0, limited)
    limiter.set_context(1, step=0, safety_margin_m=0.0, time_since_reference_change_s=0.0)
    bypass_raw = np.asarray([9.0, -9.0, 2.0])
    bypass = limiter.limit(1, bypass_raw)
    bypass_pass = bool(np.array_equal(bypass, bypass_raw))
    motion = propagate_point_mass(
        position=np.zeros(3), velocity=np.zeros(3), acceleration=bypass,
        dt=DT, acceleration_min=np.full(3, -4.0), acceleration_max=np.full(3, 4.0),
        velocity_min=np.full(3, -4.0), velocity_max=np.full(3, 4.0),
        maximum_speed_norm=MAX_SPEED_NORM,
    )
    limiter.observe_executed(1, motion["applied_acceleration"])
    physical_pass = bool(np.array_equal(motion["applied_acceleration"], np.asarray([4.0, -4.0, 2.0])))
    invalid_rejected = False
    try:
        limiter.set_context(2, step=0, safety_margin_m=1.0, time_since_reference_change_s=0.0)
        limiter.limit(2, np.asarray([np.nan, 0.0, 0.0]))
    except ValueError:
        invalid_rejected = True
    deterministic_a = limiter_for("medium")
    deterministic_b = limiter_for("medium")
    deterministic_rows = []
    for step in range(25):
        command = np.asarray([math.sin(step), math.cos(0.7 * step), math.sin(0.3 * step)]) * 4.0
        margin = 0.1 + 0.04 * step
        outputs = []
        for candidate in (deterministic_a, deterministic_b):
            candidate.set_context(0, step=step, safety_margin_m=margin, time_since_reference_change_s=step * DT)
            value = candidate.limit(0, command)
            candidate.observe_executed(0, value)
            outputs.append(value)
        deterministic_rows.append(np.array_equal(outputs[0], outputs[1]))
    checks = {
        "constant_acceleration_unchanged": unchanged_pass,
        "safe_state_vector_projection_exact": projection_pass,
        "hard_critical_bypass_exact": bypass_pass,
        "existing_physical_acceleration_saturation_after_bypass": physical_pass,
        "nonfinite_input_rejected": invalid_rejected,
        "deterministic_replay_exact": bool(all(deterministic_rows)),
    }
    return {"checks": checks, "PASS": bool(all(checks.values()))}


def microtest() -> None:
    if not THRESHOLD_PATH.exists():
        raise RuntimeError("thresholds must be frozen first")
    transition = _transition_microtests()
    manifest = filtered_manifest(selected_ids=[source_ids()[0]])
    runtime, config = runtime_for(manifest)
    entry = manifest["entries"][0]
    limiter = limiter_for("mild", enabled=False)
    recorder = OnlineRuntimeRecorder()
    proxy = TimedPolicyProxy(runtime.policy, recorder)
    with recorder.instrument_dmp(), recorder.scoped_context(
        evaluation_block="jerk_limiter_disabled_equivalence_microtest",
        configuration_id="DISABLED_EXACT_EQUIVALENCE",
        stage=entry["stage"], family=entry["family"], scenario_id=entry["scenario_id"],
        seed=int(entry["seed"]), method=METHOD_RERR_GAT,
    ):
        episode, _, _, _, extra = run_episode(
            config=runtime.eval_config, settings=runtime.settings,
            multi_config=runtime.multi_config, policy=proxy,
            gat_model=runtime.gat_model, gat_device=runtime.gat_device,
            method=METHOD_RERR_GAT, scenario=str(entry["scenario_id"]), seed=int(entry["seed"]),
            environment_builder=runtime.builder, runtime_recorder=recorder,
            upper_plan_builder=build_online_gat_plan_optimized,
            execution_acceleration_limiter=limiter,
        )
    source_payload = load_json(SOURCE_ORIGINAL_DEV / f"{entry['scenario_id']}.json")
    source_arrays = np.load(SOURCE_ORIGINAL_DEV / f"{entry['scenario_id']}_trajectory.npz")
    rows = list(extra["path_rows"])
    step_count = max(int(row["step"]) for row in rows) + 1
    positions = np.full((step_count, int(config["num_agents"]), 3), np.nan)
    velocities = np.full_like(positions, np.nan)
    applied = np.full_like(positions, np.nan)
    for row in rows:
        step, agent = int(row["step"]), int(row["agent_id"])
        positions[step, agent] = [row["x_m"], row["y_m"], row["z_m"]]
        velocities[step, agent] = [row["vx_mps"], row["vy_mps"], row["vz_mps"]]
        applied[step, agent] = [row["applied_ax_mps2"], row["applied_ay_mps2"], row["applied_az_mps2"]]
    source_position = np.asarray(source_arrays["positions"], dtype=float)
    source_velocity = np.asarray(source_arrays["velocities"], dtype=float)
    source_applied = np.asarray(source_arrays["applied_accelerations_full"], dtype=float)
    exact = {
        "position_array_equal": bool(np.array_equal(positions, source_position)),
        "velocity_array_equal": bool(np.array_equal(velocities, source_velocity)),
        "applied_acceleration_array_equal": bool(
            np.array_equal(applied, source_applied, equal_nan=True)
        ),
        "team_success_equal": bool(episode["team_success"] == source_payload["episode"]["team_success"]),
        "termination_reason_equal": bool(episode["termination_reason"] == source_payload["episode"]["termination_reason"]),
        "steps_equal": bool(episode["steps"] == source_payload["episode"]["steps"]),
        "max_position_absolute_difference": float(np.max(np.abs(positions - source_position))),
        "max_velocity_absolute_difference": float(np.max(np.abs(velocities - source_velocity))),
        "max_applied_acceleration_absolute_difference": float(
            np.nanmax(np.abs(applied - source_applied))
        ),
    }
    equivalence_pass = bool(all(exact[key] for key in (
        "position_array_equal", "velocity_array_equal", "applied_acceleration_array_equal",
        "team_success_equal", "termination_reason_equal", "steps_equal",
    )))
    payload = {
        "schema_version": "safety_adaptive_jerk_limiter_microtest_v1",
        "status": "PASS" if transition["PASS"] and equivalence_pass else "FAIL",
        "transition_tests": transition,
        "disabled_full_episode_exact_equivalence": {
            "scenario_id": entry["scenario_id"],
            "source_record": relative(SOURCE_ORIGINAL_DEV / f"{entry['scenario_id']}.json"),
            **exact,
            "PASS": equivalence_pass,
        },
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "thresholds_sha256": sha256_file(THRESHOLD_PATH),
        "hard_stop_on_failure": True,
    }
    atomic_json(MICROTEST_PATH, payload)
    if payload["status"] != "PASS":
        raise RuntimeError(f"jerk-limiter microtest failed: {payload}")
    print(json.dumps({"status": "PASS", "equivalence": exact}, indent=2))


def save_episode_record(
    root: Path,
    entry: Mapping[str, Any],
    episode: Mapping[str, Any],
    agents: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    triggers: Sequence[Mapping[str, Any]],
    extra: Mapping[str, Any],
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    sid = str(entry["scenario_id"])
    rows = list(extra["path_rows"])
    agent_count = int(episode["num_agents"])
    step_count = max(int(row["step"]) for row in rows) + 1
    shape = (step_count, agent_count, 3)
    positions = np.full(shape, np.nan, dtype=np.float64)
    velocities = np.full(shape, np.nan, dtype=np.float64)
    commanded = np.full(shape, np.nan, dtype=np.float64)
    applied = np.full(shape, np.nan, dtype=np.float64)
    active_goals = np.full(shape, np.nan, dtype=np.float64)
    terminal_goals = np.full(shape, np.nan, dtype=np.float64)
    for row in rows:
        step, agent = int(row["step"]), int(row["agent_id"])
        positions[step, agent] = [row["x_m"], row["y_m"], row["z_m"]]
        velocities[step, agent] = [row["vx_mps"], row["vy_mps"], row["vz_mps"]]
        commanded[step, agent] = [row["commanded_ax_mps2"], row["commanded_ay_mps2"], row["commanded_az_mps2"]]
        applied[step, agent] = [row["applied_ax_mps2"], row["applied_ay_mps2"], row["applied_az_mps2"]]
        active_goals[step, agent] = [row["active_goal_x_m"], row["active_goal_y_m"], row["active_goal_z_m"]]
        terminal_goals[step, agent] = [row["terminal_goal_x_m"], row["terminal_goal_y_m"], row["terminal_goal_z_m"]]
    trace = list(extra["jerk_limiter_trace_rows"])
    trace_fields = {
        "step": np.asarray([row["step"] for row in trace], dtype=np.int32),
        "agent_id": np.asarray([row["agent_id"] for row in trace], dtype=np.int8),
        "safety_margin_m": np.asarray([row["safety_margin_m"] for row in trace], dtype=np.float64),
        "q_safe": np.asarray([row["q_safe"] for row in trace], dtype=np.float64),
        "hard_bypass": np.asarray([row["hard_bypass"] for row in trace], dtype=bool),
        "early_bypass": np.asarray(
            [bool(row.get("early_bypass", False)) for row in trace], dtype=bool
        ),
        "bypass_active": np.asarray(
            [bool(row.get("bypass_active", row["hard_bypass"])) for row in trace],
            dtype=bool,
        ),
        "previous_safety_margin_m": np.asarray(
            [
                np.nan
                if row.get("previous_safety_margin_m") is None
                else float(row["previous_safety_margin_m"])
                for row in trace
            ],
            dtype=np.float64,
        ),
        "safety_margin_delta_m": np.asarray(
            [
                np.nan
                if row.get("safety_margin_delta_m") is None
                else float(row["safety_margin_delta_m"])
                for row in trace
            ],
            dtype=np.float64,
        ),
        "warning_low_margin": np.asarray(
            [bool(row.get("warning_low_margin", False)) for row in trace], dtype=bool
        ),
        "deteriorating_warning": np.asarray(
            [bool(row.get("deteriorating_warning", False)) for row in trace],
            dtype=bool,
        ),
        "limiter_active": np.asarray([row["limiter_active"] for row in trace], dtype=bool),
        "j_max_mps3": np.asarray([row["j_max_mps3"] for row in trace], dtype=np.float64),
        "raw_acceleration": np.asarray([row["raw_acceleration"] for row in trace], dtype=np.float64),
        "limited_acceleration": np.asarray([row["limited_acceleration"] for row in trace], dtype=np.float64),
        "previous_executed_acceleration": np.asarray([row["previous_executed_acceleration"] for row in trace], dtype=np.float64),
        "executed_acceleration": np.asarray([row["executed_acceleration"] for row in trace], dtype=np.float64),
        "executed_jerk_norm_mps3": np.asarray([row["executed_jerk_norm_mps3"] for row in trace], dtype=np.float64),
        "limiter_modification_norm_mps2": np.asarray([row["limiter_modification_norm_mps2"] for row in trace], dtype=np.float64),
        "limiter_runtime_ns": np.asarray([row["limiter_runtime_ns"] for row in trace], dtype=np.int64),
    }
    if trace and "persistence_output_acceleration" in trace[0]:
        horizontal = [row.get("horizontal", {}) for row in trace]
        vertical = [row.get("vertical", {}) for row in trace]
        trace_fields.update(
            {
                "execution_velocity": np.asarray(
                    [row["execution_velocity"] for row in trace], dtype=np.float64
                ),
                "strong_limited_acceleration": np.asarray(
                    [row["strong_limited_acceleration"] for row in trace], dtype=np.float64
                ),
                "persistence_output_acceleration": np.asarray(
                    [row["persistence_output_acceleration"] for row in trace], dtype=np.float64
                ),
                "horizontal_valid_heading": np.asarray(
                    [bool(row.get("valid_heading", False)) for row in horizontal], dtype=bool
                ),
                "horizontal_requested_mps2": np.asarray(
                    [np.nan if row.get("requested") is None else float(row["requested"]) for row in horizontal],
                    dtype=np.float64,
                ),
                "horizontal_executed_mps2": np.asarray(
                    [np.nan if row.get("executed") is None else float(row["executed"]) for row in horizontal],
                    dtype=np.float64,
                ),
                "vertical_requested_mps2": np.asarray(
                    [float(row.get("requested", np.nan)) for row in vertical], dtype=np.float64
                ),
                "vertical_executed_mps2": np.asarray(
                    [float(row.get("executed", np.nan)) for row in vertical], dtype=np.float64
                ),
                "horizontal_persistence_activation": np.asarray(
                    [bool(row.get("activation", False)) for row in horizontal], dtype=bool
                ),
                "vertical_persistence_activation": np.asarray(
                    [bool(row.get("activation", False)) for row in vertical], dtype=bool
                ),
                "horizontal_confirmed_reversal": np.asarray(
                    [bool(row.get("confirmed_reversal", False)) for row in horizontal], dtype=bool
                ),
                "vertical_confirmed_reversal": np.asarray(
                    [bool(row.get("confirmed_reversal", False)) for row in vertical], dtype=bool
                ),
                "horizontal_magnitude_override": np.asarray(
                    [bool(row.get("magnitude_override", False)) for row in horizontal], dtype=bool
                ),
                "vertical_magnitude_override": np.asarray(
                    [bool(row.get("magnitude_override", False)) for row in vertical], dtype=bool
                ),
                "horizontal_accepted_sign": np.asarray(
                    [int(row["horizontal_accepted_sign"]) for row in trace], dtype=np.int8
                ),
                "vertical_accepted_sign": np.asarray(
                    [int(row["vertical_accepted_sign"]) for row in trace], dtype=np.int8
                ),
                "horizontal_pending_sign": np.asarray(
                    [int(row["horizontal_pending_sign"]) for row in trace], dtype=np.int8
                ),
                "vertical_pending_sign": np.asarray(
                    [int(row["vertical_pending_sign"]) for row in trace], dtype=np.int8
                ),
                "horizontal_pending_count": np.asarray(
                    [int(row["horizontal_pending_count"]) for row in trace], dtype=np.int8
                ),
                "vertical_pending_count": np.asarray(
                    [int(row["vertical_pending_count"]) for row in trace], dtype=np.int8
                ),
                "persistence_bypassed_for_safety": np.asarray(
                    [bool(row["persistence_bypassed_for_safety"]) for row in trace], dtype=bool
                ),
                "persistence_modified": np.asarray(
                    [bool(row["persistence_modified"]) for row in trace], dtype=bool
                ),
                "persistence_unchanged": np.asarray(
                    [bool(row["persistence_unchanged_fraction_flag"]) for row in trace], dtype=bool
                ),
                "physical_acceleration_clip_active": np.asarray(
                    [bool(row["physical_acceleration_clip_active"]) for row in trace], dtype=bool
                ),
            }
        )
    trajectory = root / f"{sid}_trajectory.npz"
    np.savez_compressed(
        trajectory, positions=positions, velocities=velocities,
        commanded_accelerations_full=commanded, applied_accelerations_full=applied,
        active_goals=active_goals, terminal_goals=terminal_goals, dt=np.asarray(float(episode["dt"])),
    )
    trace_path = root / f"{sid}_limiter_trace.npz"
    np.savez_compressed(trace_path, **trace_fields)
    dctb_trace = list(extra.get("direction_continuity_trace_rows", []))
    atomic_json(
        root / f"{sid}.json",
        {
            "entry_identity": {key: entry[key] for key in ("scenario_id", "seed", "stage", "family", "task_pattern")},
            "episode": episode,
            "agents": list(agents),
            "events": list(events),
            "trigger_summary": {"row_count": len(triggers), "event_counts": dict(Counter(str(row["event"]) for row in triggers))},
            "limiter_summary": {
                "trace_count": len(trace),
                "activation_count": int(sum(bool(row["limiter_active"]) for row in trace)),
                "hard_bypass_count": int(sum(bool(row["hard_bypass"]) for row in trace)),
                "early_bypass_count": int(
                    sum(bool(row.get("early_bypass", False)) for row in trace)
                ),
                "combined_bypass_count": int(
                    sum(
                        bool(row.get("bypass_active", row["hard_bypass"]))
                        for row in trace
                    )
                ),
                "runtime_ms": float(sum(int(row["limiter_runtime_ns"]) for row in trace) / 1.0e6),
                "mean_modification_norm_mps2": float(np.mean([row["limiter_modification_norm_mps2"] for row in trace])) if trace else 0.0,
                "persistence_activation_count": int(
                    sum(
                        bool(row.get("horizontal", {}).get("activation", False))
                        or bool(row.get("vertical", {}).get("activation", False))
                        for row in trace
                    )
                ),
                "horizontal_persistence_activation_count": int(
                    sum(bool(row.get("horizontal", {}).get("activation", False)) for row in trace)
                ),
                "vertical_persistence_activation_count": int(
                    sum(bool(row.get("vertical", {}).get("activation", False)) for row in trace)
                ),
                "horizontal_confirmed_reversal_count": int(
                    sum(bool(row.get("horizontal", {}).get("confirmed_reversal", False)) for row in trace)
                ),
                "vertical_confirmed_reversal_count": int(
                    sum(bool(row.get("vertical", {}).get("confirmed_reversal", False)) for row in trace)
                ),
                "horizontal_magnitude_override_count": int(
                    sum(bool(row.get("horizontal", {}).get("magnitude_override", False)) for row in trace)
                ),
                "vertical_magnitude_override_count": int(
                    sum(bool(row.get("vertical", {}).get("magnitude_override", False)) for row in trace)
                ),
                "persistence_safety_bypass_count": int(
                    sum(bool(row.get("persistence_bypassed_for_safety", False)) for row in trace)
                ),
                "persistence_modified_count": int(
                    sum(bool(row.get("persistence_modified", False)) for row in trace)
                ),
                "persistence_unchanged_fraction": float(
                    np.mean([bool(row.get("persistence_unchanged_fraction_flag", True)) for row in trace])
                ) if trace else 1.0,
            },
            "direction_continuity_summary": {
                "trace_count": len(dctb_trace),
                "activation_count": int(
                    sum(bool(row.get("activated", False)) for row in dctb_trace)
                ),
                "replacement_count": int(
                    sum(bool(row.get("replaced", False)) for row in dctb_trace)
                ),
                "history_reset_count": int(
                    sum(bool(row.get("history_reset", False)) for row in dctb_trace)
                ),
            },
            "direction_continuity_trace": dctb_trace,
            "trajectory_file": trajectory.name,
            "trajectory_sha256": sha256_file(trajectory),
            "limiter_trace_file": trace_path.name,
            "limiter_trace_sha256": sha256_file(trace_path),
        },
    )


def run_development(variant: str, shard_index: int, shard_count: int) -> None:
    if load_json(MICROTEST_PATH)["status"] != "PASS":
        raise RuntimeError("microtest did not pass")
    if variant not in VARIANT_PERCENTILES:
        raise ValueError("invalid variant")
    manifest = filtered_manifest()
    validate_development_balance(manifest)
    runtime, _ = runtime_for(manifest)
    output = DEV_RECORD_ROOT / variant / "episode_records"
    completed = {path.stem for path in output.glob("GATRS_DEV_*.json") if not path.name.endswith("_SOFTWARE_ERROR.json")}
    indexed = [
        entry for index, entry in enumerate(manifest["entries"])
        if index % int(shard_count) == int(shard_index)
    ]
    for local_index, entry in enumerate(indexed, start=1):
        sid = str(entry["scenario_id"])
        if sid in completed:
            continue
        print(f"[dev:{variant}:{shard_index}/{shard_count}] {local_index:03d}/{len(indexed):03d} start {sid}", flush=True)
        recorder = OnlineRuntimeRecorder()
        proxy = TimedPolicyProxy(runtime.policy, recorder)
        limiter = limiter_for(variant)
        try:
            with recorder.instrument_dmp(), recorder.scoped_context(
                evaluation_block="safety_adaptive_jerk_limiter_development",
                configuration_id=f"JERK_LIMITER_{variant.upper()}",
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
            save_episode_record(output, entry, episode, agents, events, triggers, extra)
            print(
                f"[dev:{variant}:{shard_index}/{shard_count}] complete {sid} success={int(episode['team_success'])} collision={int(episode['collision'])}",
                flush=True,
            )
        except Exception as error:
            atomic_json(
                output / f"{sid}_SOFTWARE_ERROR.json",
                {"scenario_id": sid, "error_type": type(error).__name__, "error_message": str(error), "traceback": traceback.format_exc()},
            )
            raise


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("contract")
    sub.add_parser("thresholds")
    sub.add_parser("microtest")
    run_parser = sub.add_parser("run-dev")
    run_parser.add_argument("--variant", choices=sorted(VARIANT_PERCENTILES), required=True)
    run_parser.add_argument("--shard-index", type=int, default=0)
    run_parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if args.command == "contract":
        write_contract()
    elif args.command == "thresholds":
        recover_thresholds()
    elif args.command == "microtest":
        microtest()
    elif args.command == "run-dev":
        run_development(args.variant, args.shard_index, args.shard_count)


if __name__ == "__main__":
    main()
