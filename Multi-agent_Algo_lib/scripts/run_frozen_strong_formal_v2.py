#!/usr/bin/env python3
"""Freeze, preflight, and run the one-shot Frozen-Strong Formal V2 arm.

This runner never regenerates the Formal V2 manifest and never executes another
method.  Formal performance is deliberately not aggregated during execution.
"""

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
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for search in (REPO_ROOT, SCRIPT_DIR):
    if str(search) not in sys.path:
        sys.path.insert(0, str(search))

import run_safety_adaptive_jerk_limiter as strong_base  # noqa: E402
import run_gat_recurrent_formal_v2 as formal_v2  # noqa: E402
from planning.safety_adaptive_jerk_limiter import (  # noqa: E402
    SafetyAdaptiveJerkLimiterConfig,
    SafetyAdaptiveVectorJerkLimiter,
)
from Entity.KinematicModel import propagate_point_mass  # noqa: E402


ROOT = REPO_ROOT / "artifacts/frozen_strong_formal_v2/20260826_110337"
IDENTITY_DIR = ROOT / "00_identity"
PREFREEZE_DIR = ROOT / "01_prefreeze"
MICROTEST_DIR = ROOT / "02_microtests"
RECORD_DIR = ROOT / "03_formal_run/episode_records"

STRONG_STUDY = REPO_ROOT / "artifacts/safety_adaptive_jerk_limiter/20260825_115704"
PARALLEL_STUDY = REPO_ROOT / "artifacts/parallel_zigzag_resolution/20260826_091334"
RESIDUAL_STUDY = REPO_ROOT / "artifacts/final_residual_zigzag_resolution/20260825_183146"
FORMAL_STUDY = REPO_ROOT / "artifacts/gat_recurrent_smoothness_rescue/20260821_180244"
FORMAL_MANIFEST = FORMAL_STUDY / "10_formal_v2/FORMAL_V2_MANIFEST.json"
ORIGINAL_RECORD_DIR = FORMAL_STUDY / "10_formal_v2/formal_records/M9_Proposed_RERR_GAT_SAC_DMP"

METHOD_CONFIG = FORMAL_STUDY / "09_final_freeze/method_configs/M9_Proposed_RERR_GAT_SAC_DMP.json"
SAC_CHECKPOINT = REPO_ROOT / "artifacts/semi_structured_long_range_main_benchmark/20260820_193228/07_training/sac_long_range_256_encoder_adapt_run1/best_validation.pt"
GAT_CHECKPOINT = FORMAL_STUDY / "13_objective_revision/05_gat_r_fp_anchor_training/checkpoints/best_validation.pt"
THRESHOLDS = STRONG_STUDY / "JERK_LIMITER_THRESHOLDS.json"
HOLDOUT_FREEZE = STRONG_STUDY / "FINAL_JERK_LIMITER_FREEZE.json"

IDENTITY_AUDIT = IDENTITY_DIR / "FINAL_STRONG_IDENTITY_AUDIT.json"
RUNTIME_CONTRACT = IDENTITY_DIR / "FINAL_FROZEN_STRONG_RUNTIME_CONTRACT.json"
PREFREEZE = PREFREEZE_DIR / "FINAL_FROZEN_STRONG_PREFREEZE.json"
ACCEPTANCE_RULE = PREFREEZE_DIR / "FINAL_STRONG_FORMAL_ACCEPTANCE_RULE.json"
SCENE_FREEZE = PREFREEZE_DIR / "FORMAL_REPRESENTATIVE_SCENE_FREEZE.json"
MICROTEST = MICROTEST_DIR / "FINAL_STRONG_PREFORMAL_MICROTEST.json"
MANIFEST_VERIFICATION = PREFREEZE_DIR / "FORMAL_V2_MANIFEST_VERIFICATION.json"
RUN_AUTHORIZATION = PREFREEZE_DIR / "FORMAL_STRONG_RUN_AUTHORIZATION.json"
FORMAL_SCHEDULE = PREFREEZE_DIR / "formal_strong_schedule.csv"

EXPECTED_SAC_SHA = "56a358ae9c342a8f0431c5659ce9933c58dc24efacca8910cec66d7b13dd5d86"
EXPECTED_GAT_SHA = "df8a9a8c13ba8f29fc1b8a64626fe336ccde68cab189587062beb6b93510baef"
EXPECTED_THRESHOLD = 22.869440564651327
EXPECTED_ORIGINAL_SUCCESS = 381
SHARD_COUNT = 4

RUNTIME_SOURCES = (
    "planning/safety_adaptive_jerk_limiter.py",
    "planning/historical_forcing_gate.py",
    "planning/event_triggered_reference_reconstruction.py",
    "Environment/frozen_sac_dmp_execution.py",
    "Environment/multi_agent_dmp_env.py",
    "Environment/single_agent_dmp_env.py",
    "Entity/KinematicModel.py",
    "Multi-agent_Algo_lib/scripts/evaluate_gat_v1_err_development.py",
    "Multi-agent_Algo_lib/scripts/run_safety_adaptive_jerk_limiter.py",
    "Multi-agent_Algo_lib/scripts/run_long_range_contract_pilot.py",
    "Multi-agent_Algo_lib/scripts/run_long_range_development.py",
    "Multi-agent_Algo_lib/scripts/run_long_range_ablation_development.py",
    "Multi-agent_Algo_lib/scripts/run_long_range_formal_benchmark.py",
    "Multi-agent_Algo_lib/scripts/run_gat_recurrent_formal_v2.py",
    "Multi-agent_Algo_lib/scripts/run_frozen_strong_formal_v2.py",
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            default=lambda item: item.item() if isinstance(item, np.generic) else str(item),
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def now() -> str:
    return datetime.now().astimezone().isoformat()


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def source_hashes() -> dict[str, str]:
    return {name: sha256_file(REPO_ROOT / name) for name in RUNTIME_SOURCES}


def verify_frozen_sources() -> dict[str, Any]:
    frozen = load_json(PREFREEZE)
    observed = source_hashes()
    expected = dict(frozen["runtime_source_sha256"])
    if observed != expected:
        changed = {
            name: {"expected": expected.get(name), "observed": observed.get(name)}
            for name in sorted(set(expected) | set(observed))
            if expected.get(name) != observed.get(name)
        }
        raise RuntimeError(f"post-freeze runtime source drift: {changed}")
    if sha256_file(SAC_CHECKPOINT) != EXPECTED_SAC_SHA:
        raise RuntimeError("SAC checkpoint hash mismatch")
    if sha256_file(GAT_CHECKPOINT) != EXPECTED_GAT_SHA:
        raise RuntimeError("GAT checkpoint hash mismatch")
    if sha256_file(THRESHOLDS) != frozen["thresholds_sha256"]:
        raise RuntimeError("Strong threshold artifact drift")
    return frozen


def freeze_identity() -> None:
    if any(RECORD_DIR.glob("*.json")) or any(RECORD_DIR.glob("*.npz")):
        raise RuntimeError("Formal output directory is not empty before identity freeze")

    selection = load_json(PARALLEL_STUDY / "final_selection/conclusion.json")
    strong_conclusion = load_json(STRONG_STUDY / "conclusion.json")
    holdout_freeze = load_json(HOLDOUT_FREEZE)
    thresholds = load_json(THRESHOLDS)
    residual = load_json(RESIDUAL_STUDY / "conclusion.json")
    method = load_json(METHOD_CONFIG)

    checks = {
        "parallel_selection_is_frozen_strong": selection.get("SELECTED_FINAL_CANDIDATE") == "FROZEN_STRONG",
        "strong_study_selected_original_strong": strong_conclusion.get("SELECTED_VARIANT") == "STRONG",
        "strong_holdout_passed": strong_conclusion.get("HOLDOUT_PASSED") == "YES",
        "residual_repair_rejected": residual.get("FINAL_ZIGZAG_REPAIR") == "REJECTED",
        "residual_recommends_frozen_strong": str(residual.get("FINAL_RECOMMENDATION", "")).lower().startswith("keep frozen strong"),
        "strong_threshold_exact": float(thresholds["variants"]["strong"]["j_smooth_mps3"]) == EXPECTED_THRESHOLD,
        "sac_checkpoint_exact": sha256_file(SAC_CHECKPOINT) == EXPECTED_SAC_SHA,
        "gat_checkpoint_exact": sha256_file(GAT_CHECKPOINT) == EXPECTED_GAT_SHA,
        "method_top_k": int(method["top_k"]) == 10,
        "method_h4": int(method["H_preview"]) == 4,
        "method_sensor_256": int(method["sensor"]["direction_count"]) == 256,
        "method_gat_projection_56": int(method["sensor"]["gat_canonical_projection_directions"]) == 56,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        payload = {
            "selected_name": "FROZEN_STRONG",
            "Formal_ready": "NO",
            "ambiguities": failures,
            "checks": checks,
        }
        atomic_json(IDENTITY_AUDIT, payload)
        raise RuntimeError(f"Frozen Strong identity hard gate failed: {failures}")

    current_source = REPO_ROOT / "planning/safety_adaptive_jerk_limiter.py"
    safety = dict(holdout_freeze["safety_mapping"])
    identity = {
        "schema_version": "final_frozen_strong_identity_audit_v1",
        "created_at": now(),
        "selected_name": "FROZEN_STRONG",
        "exact_identity": "Safety-Adaptive Vector Jerk Limiter, Strong P60, early_bypass_enabled=False",
        "identity_class": "A_ORIGINAL_SAFETY_ADAPTIVE_STRONG",
        "source_file": relative(current_source),
        "source_sha256": sha256_file(current_source),
        "holdout_source_sha256_provenance": holdout_freeze["source_sha256"]["planning/safety_adaptive_jerk_limiter.py"],
        "current_source_contains_later_optional_guard_support": True,
        "later_optional_guard_enabled_for_selected_identity": False,
        "SAC_checkpoint": relative(SAC_CHECKPOINT),
        "SAC_checkpoint_sha256": sha256_file(SAC_CHECKPOINT),
        "GAT_checkpoint": relative(GAT_CHECKPOINT),
        "GAT_checkpoint_sha256": sha256_file(GAT_CHECKPOINT),
        "Strong_threshold": EXPECTED_THRESHOLD,
        "threshold_source": relative(THRESHOLDS),
        "threshold_source_sha256": sha256_file(THRESHOLDS),
        "safety_mapping": safety,
        "hard_bypass_rule": "h_active <= h_emg = 0.0 m passes raw DMP acceleration to original physical saturation",
        "early_bypass_present": "NO",
        "early_bypass_rule": "NOT_ACTIVE; config early_bypass_enabled=False",
        "execution_order": [
            "Proposal 256 directions",
            "Top-K 10",
            "FP-SHEP H4",
            "GAT-R canonical 56-direction graph",
            "R-ERR",
            "frozen SAC-DMP",
            "Frozen Strong vector jerk projection (hard bypass only)",
            "original physical saturation",
            "dynamics",
        ],
        "physical_clip_order": [
            "componentwise acceleration saturation +/-4 m/s^2",
            "componentwise velocity saturation",
            "velocity-norm cap 3.2 m/s",
            "state propagation",
        ],
        "sensor_contract": {
            "Proposal_direction_count": 256,
            "SAC_sensor_directions": 256,
            "SAC_temporal_frames": 2,
            "SAC_observation_dim": 522,
            "GAT_canonical_projection": 56,
        },
        "dt": 0.1,
        "speed_limit": {"norm_mps": 3.2},
        "acceleration_limit": {"componentwise_mps2": [-4.0, 4.0]},
        "DMP_identity": {
            "source_file": "Environment/frozen_sac_dmp_execution.py",
            "source_sha256": sha256_file(REPO_ROOT / "Environment/frozen_sac_dmp_execution.py"),
            "forcing_gate": "historical_vector_goal_eff_gate",
            "changed": False,
        },
        "Formal_ready": "YES",
        "ambiguities": [],
        "checks": checks,
    }
    atomic_json(IDENTITY_AUDIT, identity)

    runtime_contract = {
        "schema_version": "final_frozen_strong_runtime_contract_v1",
        "created_at": now(),
        "identity_audit_sha256": sha256_file(IDENTITY_AUDIT),
        "method_config": relative(METHOD_CONFIG),
        "method_config_sha256": sha256_file(METHOD_CONFIG),
        "chain": identity["execution_order"],
        "early_bypass_included": "NO",
        "limiter_config": {
            "enabled": True,
            "early_bypass_enabled": False,
            "dt_s": 0.1,
            "j_smooth_mps3": EXPECTED_THRESHOLD,
            "j_free_mps3": float(thresholds["j_free_mps3"]),
            "h_emg_m": 0.0,
            "h_rep_m": 0.35,
            "q_safe": "clip((h_active-h_emg)/(h_rep-h_emg),0,1)",
            "j_max": "j_smooth+(1-q_safe)*(j_free-j_smooth)",
        },
        "UPPER_FRAMEWORK_CHANGED": "NO",
        "LEARNED_CHECKPOINT_CHANGED": "NO",
        "DMP_CHANGED": "NO",
        "VEHICLE_PHYSICS_CHANGED": "NO",
        "TASK_BENCHMARK_CONTRACT_CHANGED": "NO",
    }
    atomic_json(RUNTIME_CONTRACT, runtime_contract)

    acceptance = {
        "schema_version": "final_strong_formal_acceptance_rule_v1",
        "created_at": now(),
        "frozen_before_strong_formal_performance": True,
        "comparison": "existing Original Proposed Formal V2 versus one new Frozen Strong arm on identical 400 scenes",
        "original_success_count": EXPECTED_ORIGINAL_SUCCESS,
        "original_success_rate": 0.9525,
        "minimum_strong_success_rate": 0.9425,
        "minimum_integer_success_count": 377,
        "reliability": {
            "maximum_success_loss_pp": 1.0,
            "maximum_total_collision_increase_pp": 1.0,
            "maximum_peer_collision_increase_pp": 1.0,
            "maximum_obstacle_collision_increase_pp": 1.0,
            "software_error_count": 0,
            "no_new_failure_subtype_rate_at_least": 0.02,
            "p_value_is_not_the_acceptance_rule": True,
        },
        "trajectory_quality": {
            "minimum_overall_smoothness_reduction_percent": 30.0,
            "minimum_vertical_jerk_reduction_percent": 20.0,
            "minimum_lateral_jerk_reduction_percent": 20.0,
            "minimum_p95_jerk_reduction_percent": 20.0,
            "paired_subset": "both-success only",
        },
        "runtime": {
            "maximum_limiter_ms_per_agent_step": 0.1,
            "existing_online_accounting_boundary_unchanged": True,
        },
        "raw_trajectory": {
            "required": True,
            "dt_s": 0.1,
            "smoothing": "FORBIDDEN",
            "post_processing": "FORBIDDEN",
        },
        "all_required": True,
        "no_post_formal_tuning": True,
    }
    atomic_json(ACCEPTANCE_RULE, acceptance)

    manifest = load_json(FORMAL_MANIFEST)
    stage_first: dict[int, str] = {}
    for entry in manifest["entries"]:
        stage_first.setdefault(int(entry["stage_index"]), str(entry["scenario_id"]))
    scene_freeze = {
        "schema_version": "formal_strong_representative_scene_freeze_v1",
        "created_at": now(),
        "selection_timing": "before Frozen Strong Formal execution",
        "selection_rule": "lexicographically first scenario ID in each stage; method- and outcome-independent",
        "stage_scene_ids": {f"stage_{key}": stage_first[key] for key in sorted(stage_first)},
        "primary_comparison_scene": stage_first[2],
        "high_density_scene": stage_first[4],
        "strong_outcomes_read": False,
    }
    atomic_json(SCENE_FREEZE, scene_freeze)

    prefreeze = {
        "schema_version": "final_frozen_strong_prefreeze_v1",
        "created_at": now(),
        "identity_audit_sha256": sha256_file(IDENTITY_AUDIT),
        "runtime_contract_sha256": sha256_file(RUNTIME_CONTRACT),
        "acceptance_rule_sha256": sha256_file(ACCEPTANCE_RULE),
        "representative_scene_freeze_sha256": sha256_file(SCENE_FREEZE),
        "runtime_source_sha256": source_hashes(),
        "thresholds_sha256": sha256_file(THRESHOLDS),
        "SAC_checkpoint_sha256": sha256_file(SAC_CHECKPOINT),
        "GAT_checkpoint_sha256": sha256_file(GAT_CHECKPOINT),
        "formal_record_count": 0,
        "performance_read_or_generated": False,
    }
    atomic_json(PREFREEZE, prefreeze)
    print(json.dumps({"phase": "freeze", "status": "PASS", "identity": identity["exact_identity"]}, indent=2))


def arrays_from_path_rows(rows: list[Mapping[str, Any]], num_agents: int) -> dict[str, np.ndarray]:
    step_count = max(int(row["step"]) for row in rows) + 1
    shape = (step_count, int(num_agents), 3)
    result = {name: np.full(shape, np.nan, dtype=float) for name in ("positions", "velocities", "applied")}
    for row in rows:
        step, agent = int(row["step"]), int(row["agent_id"])
        result["positions"][step, agent] = [row["x_m"], row["y_m"], row["z_m"]]
        result["velocities"][step, agent] = [row["vx_mps"], row["vy_mps"], row["vz_mps"]]
        result["applied"][step, agent] = [row["applied_ax_mps2"], row["applied_ay_mps2"], row["applied_az_mps2"]]
    return result


def execute_dev_microtest(runtime: Any, entry: Mapping[str, Any], limiter: Any, label: str) -> tuple[Any, Any, Any, Any, Any]:
    recorder = strong_base.OnlineRuntimeRecorder()
    proxy = strong_base.TimedPolicyProxy(runtime.policy, recorder)
    with recorder.instrument_dmp(), recorder.scoped_context(
        evaluation_block="final_frozen_strong_preformal_microtest",
        configuration_id=label,
        stage=entry["stage"], family=entry["family"], scenario_id=entry["scenario_id"],
        seed=int(entry["seed"]), method=strong_base.METHOD_RERR_GAT,
    ):
        return strong_base.run_episode(
            config=runtime.eval_config, settings=runtime.settings,
            multi_config=runtime.multi_config, policy=proxy,
            gat_model=runtime.gat_model, gat_device=runtime.gat_device,
            method=strong_base.METHOD_RERR_GAT, scenario=str(entry["scenario_id"]), seed=int(entry["seed"]),
            environment_builder=runtime.builder, runtime_recorder=recorder,
            upper_plan_builder=strong_base.build_online_gat_plan_optimized,
            execution_acceleration_limiter=limiter,
        )


def frozen_formal_runtime(manifest: Mapping[str, Any]) -> Any:
    """Build the exact retained Formal-V2 GAT-R runtime for embedded scenes."""

    formal_v2.configure(frozen=True)
    method = next(
        row for row in formal_v2.METHODS
        if row["method_id"] == "M9_Proposed_RERR_GAT_SAC_DMP"
    )
    bundle = formal_v2.base.build_method_runtime(method, manifest)
    runtime = bundle.get("runtime")
    if runtime is None:
        raise RuntimeError("Formal V2 GAT-R runtime bundle is missing the learned runtime")
    return runtime


def run_microtests() -> None:
    verify_frozen_sources()
    identity = load_json(IDENTITY_AUDIT)
    if identity.get("Formal_ready") != "YES" or identity.get("early_bypass_present") != "NO":
        raise RuntimeError("identity gate is not ready or early bypass identity drifted")

    deterministic_a = strong_base.limiter_for("strong")
    deterministic_b = strong_base.limiter_for("strong")
    deterministic = True
    early_seen = False
    finite = True
    threshold_values: list[float] = []
    for step in range(40):
        command = 4.0 * np.asarray([math.sin(0.7 * step), math.cos(0.4 * step), math.sin(0.2 * step)], dtype=float)
        margin = -0.05 if step == 17 else 0.02 + 0.11 * (step % 9)
        outputs = []
        for limiter in (deterministic_a, deterministic_b):
            limiter.set_context(0, step=step, safety_margin_m=margin, time_since_reference_change_s=0.1 * step)
            value = limiter.limit(0, command)
            limiter.observe_executed(0, np.clip(value, -4.0, 4.0))
            outputs.append(value)
            row = limiter.trace_rows()[-1]
            early_seen = early_seen or bool(row.get("early_bypass", False))
            finite = finite and bool(np.isfinite(value).all())
            threshold_values.append(float(row["j_max_mps3"]))
        deterministic = deterministic and bool(np.array_equal(outputs[0], outputs[1]))

    disabled = SafetyAdaptiveVectorJerkLimiter(
        SafetyAdaptiveJerkLimiterConfig(
            dt=0.1, j_smooth_mps3=EXPECTED_THRESHOLD,
            j_free_mps3=float(load_json(THRESHOLDS)["j_free_mps3"]),
            emergency_margin_m=0.0, comfortable_margin_m=0.35,
            enabled=False, early_bypass_enabled=False,
        ), num_agents=3,
    )
    disabled.set_context(0, step=0, safety_margin_m=1.0, time_since_reference_change_s=0.0)
    disabled_raw = np.asarray([3.25, -2.5, 1.75])
    disabled_output = disabled.limit(0, disabled_raw)
    disabled.observe_executed(0, disabled_output)

    hard = strong_base.limiter_for("strong")
    hard.set_context(0, step=0, safety_margin_m=0.0, time_since_reference_change_s=0.0)
    hard_raw = np.asarray([8.0, -9.0, 2.0])
    hard_output = hard.limit(0, hard_raw)
    propagated = propagate_point_mass(
        position=np.zeros(3), velocity=np.asarray([3.1, 0.0, 0.0]), acceleration=hard_output,
        dt=0.1, acceleration_min=np.full(3, -4.0), acceleration_max=np.full(3, 4.0),
        velocity_min=np.full(3, -4.0), velocity_max=np.full(3, 4.0), maximum_speed_norm=3.2,
    )
    hard.observe_executed(0, propagated["applied_acceleration"])

    manifest = strong_base.filtered_manifest(selected_ids=["GATRS_DEV_1_000"])
    runtime, config = strong_base.runtime_for(manifest)
    entry = dict(manifest["entries"][0])
    original_run = execute_dev_microtest(runtime, entry, strong_base.limiter_for("strong", enabled=False), "DISABLED_STRONG")
    strong_run = execute_dev_microtest(runtime, entry, strong_base.limiter_for("strong"), "FROZEN_STRONG")
    original_episode, _, _, _, original_extra = original_run
    strong_episode, strong_agents, strong_events, strong_triggers, strong_extra = strong_run
    original_arrays = arrays_from_path_rows(list(original_extra["path_rows"]), int(config["num_agents"]))
    strong_arrays = arrays_from_path_rows(list(strong_extra["path_rows"]), int(config["num_agents"]))

    source_original_path = strong_base.SOURCE_ORIGINAL_DEV / f"{entry['scenario_id']}_trajectory.npz"
    source_strong_path = STRONG_STUDY / f"dev_records/strong/episode_records/{entry['scenario_id']}_trajectory.npz"
    with np.load(source_original_path) as source:
        original_exact = {
            "positions": bool(np.array_equal(original_arrays["positions"], source["positions"])),
            "velocities": bool(np.array_equal(original_arrays["velocities"], source["velocities"])),
            "applied_accelerations": bool(np.array_equal(original_arrays["applied"], source["applied_accelerations_full"], equal_nan=True)),
        }
    with np.load(source_strong_path) as source:
        strong_exact = {
            "positions": bool(np.array_equal(strong_arrays["positions"], source["positions"])),
            "velocities": bool(np.array_equal(strong_arrays["velocities"], source["velocities"])),
            "applied_accelerations": bool(np.array_equal(strong_arrays["applied"], source["applied_accelerations_full"], equal_nan=True)),
        }

    integration_dir = MICROTEST_DIR / "integration_record"
    strong_base.save_episode_record(
        integration_dir, entry, strong_episode, strong_agents, strong_events, strong_triggers, strong_extra
    )
    saved_json = integration_dir / f"{entry['scenario_id']}.json"
    saved_npz = integration_dir / f"{entry['scenario_id']}_trajectory.npz"
    trace = list(strong_extra["jerk_limiter_trace_rows"])
    trace_runtime = [int(row["limiter_runtime_ns"]) for row in trace]

    # Exercise the corrected embedded-scene adapter on an old non-Formal
    # development scene.  No Formal V2 outcome is opened or generated here.
    embedded_manifest_path = (
        REPO_ROOT
        / "artifacts/semi_structured_long_range_main_benchmark/20260820_193228/08_development/development_manifest.json"
    )
    embedded_manifest = load_json(embedded_manifest_path)
    embedded_entry = dict(embedded_manifest["entries"][0])
    embedded_runtime = frozen_formal_runtime(embedded_manifest)
    embedded_run = execute_dev_microtest(
        embedded_runtime,
        embedded_entry,
        strong_base.limiter_for("strong"),
        "FROZEN_STRONG_EMBEDDED_SCENE_ADAPTER_PREFLIGHT",
    )
    embedded_episode, _, _, _, embedded_extra = embedded_run
    embedded_arrays = arrays_from_path_rows(
        list(embedded_extra["path_rows"]), int(embedded_runtime.settings["num_agents"])
    )
    embedded_builder_ok = (
        str(embedded_episode.get("scenario")) == str(embedded_entry["scenario_id"])
        and bool(np.isfinite(embedded_arrays["positions"]).all())
        and bool(np.isfinite(embedded_arrays["velocities"]).all())
    )

    checks = {
        "deterministic_fixed_input_exact": deterministic,
        "disabled_limiter_returns_raw_exact": bool(np.array_equal(disabled_output, disabled_raw)),
        "disabled_full_episode_reproduces_original": bool(all(original_exact.values())),
        "current_strong_full_episode_reproduces_frozen_strong": bool(all(strong_exact.values())),
        "strong_threshold_frozen": float(load_json(THRESHOLDS)["variants"]["strong"]["j_smooth_mps3"]) == EXPECTED_THRESHOLD,
        "hard_bypass_exact": bool(np.array_equal(hard_output, hard_raw)),
        "early_bypass_never_active": not early_seen and not any(bool(row.get("early_bypass", False)) for row in trace),
        "acceleration_bounds_hold": bool(np.max(np.abs(propagated["applied_acceleration"])) <= 4.0 + 1e-12),
        "speed_cap_holds": bool(np.linalg.norm(propagated["velocity"]) <= 3.2 + 1e-12),
        "no_nan_inf": finite and bool(np.isfinite(strong_arrays["positions"]).all()) and bool(np.isfinite(strong_arrays["velocities"]).all()),
        "serialization_works": saved_json.is_file() and saved_npz.is_file() and sha256_file(saved_npz) == load_json(saved_json)["trajectory_sha256"],
        "timer_accounting_works": bool(trace_runtime) and min(trace_runtime) >= 0 and np.isfinite(float(strong_episode["total_online_algorithm_compute_ms"])),
        "embedded_scene_formal_adapter_preflight": embedded_builder_ok,
        "source_and_checkpoint_hashes_match": True,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    payload = {
        "schema_version": "final_strong_preformal_microtest_v1",
        "created_at": now(),
        "status": status,
        "scenario_id": entry["scenario_id"],
        "checks": checks,
        "disabled_original_exact_arrays": original_exact,
        "enabled_strong_holdout_semantics_exact_arrays": strong_exact,
        "original_episode_identity": {
            "team_success": original_episode["team_success"],
            "termination_reason": original_episode["termination_reason"],
            "steps": original_episode["steps"],
        },
        "strong_episode_identity": {
            "team_success": strong_episode["team_success"],
            "termination_reason": strong_episode["termination_reason"],
            "steps": strong_episode["steps"],
        },
        "limiter_trace_count": len(trace),
        "threshold_range_mps3": [min(threshold_values), max(threshold_values)],
        "integration_record_sha256": sha256_file(saved_json),
        "integration_trajectory_sha256": sha256_file(saved_npz),
        "formal_scenarios_used": 0,
        "embedded_adapter_preflight_source": relative(embedded_manifest_path),
        "embedded_adapter_preflight_scenario_id": embedded_entry["scenario_id"],
        "embedded_adapter_preflight_performance_retained": False,
        "hard_stop_on_failure": True,
    }
    atomic_json(MICROTEST, payload)
    print(
        json.dumps(
            {"phase": "microtest", "status": status, "checks": checks},
            indent=2,
            default=lambda item: item.item() if isinstance(item, np.generic) else str(item),
        )
    )
    if status != "PASS":
        raise RuntimeError("Frozen Strong preformal microtest failed")


def verify_manifest() -> None:
    verify_frozen_sources()
    micro = load_json(MICROTEST)
    if micro.get("status") != "PASS":
        raise RuntimeError("preformal microtest did not pass")
    if any(RECORD_DIR.glob("*.json")) or any(RECORD_DIR.glob("*.npz")):
        raise RuntimeError("Formal record directory is not empty at authorization")

    manifest = load_json(FORMAL_MANIFEST)
    entries = list(manifest["entries"])
    ids = [str(entry["scenario_id"]) for entry in entries]
    original_json = sorted(ORIGINAL_RECORD_DIR.glob("FORMAL_LR_*.json"))
    original_npz = sorted(ORIGINAL_RECORD_DIR.glob("FORMAL_LR_*_trajectory.npz"))
    original_ids = {path.stem for path in original_json}
    counts = Counter(int(entry["stage_index"]) for entry in entries)
    family_counts = Counter((int(entry["stage_index"]), str(entry["family"])) for entry in entries)
    obstacle_contract = {
        int(stage): {
            "static": sorted({len(entry["static_obstacles"]) for entry in entries if int(entry["stage_index"]) == int(stage)}),
            "dynamic": sorted({len(entry["dynamic_obstacles"]) for entry in entries if int(entry["stage_index"]) == int(stage)}),
        }
        for stage in range(1, 5)
    }
    checks = {
        "manifest_file_sha256_matches_formal_freeze": sha256_file(FORMAL_MANIFEST) == "4ba0987819940820f914b46a9c0205d4f5d17955b8472ded545e2a66613db282",
        "manifest_semantic_sha256_matches": manifest.get("manifest_sha256") == "20c9ed45c66e2a2ddbcfbd73fbb3841ec6620f90949eaab8afbb663e08e437a5",
        "scenario_count_400": len(entries) == 400,
        "unique_scenario_count_400": len(set(ids)) == 400,
        "stage_count_100_each": counts == Counter({1: 100, 2: 100, 3: 100, 4: 100}),
        "five_families_20_each_per_stage": len(family_counts) == 20 and set(family_counts.values()) == {20},
        "stage_obstacle_contract": obstacle_contract == {
            1: {"static": [8], "dynamic": [2]},
            2: {"static": [16], "dynamic": [4]},
            3: {"static": [24], "dynamic": [6]},
            4: {"static": [32], "dynamic": [8]},
        },
        "original_record_count_400": len(original_json) == 400,
        "original_trajectory_count_400": len(original_npz) == 400,
        "original_scenario_ids_exact": original_ids == set(ids),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    verification = {
        "schema_version": "formal_v2_manifest_verification_v1",
        "created_at": now(),
        "status": status,
        "manifest": relative(FORMAL_MANIFEST),
        "manifest_sha256": sha256_file(FORMAL_MANIFEST),
        "manifest_semantic_sha256": manifest.get("manifest_sha256"),
        "scenario_count": len(entries),
        "stage_counts": dict(sorted(counts.items())),
        "family_cell_counts": {f"stage_{stage}:{family}": count for (stage, family), count in sorted(family_counts.items())},
        "obstacle_contract": obstacle_contract,
        "checks": checks,
        "original_formal_success_preserved": "381/400 (95.25%)",
        "strong_formal_episode_count": 0,
    }
    atomic_json(MANIFEST_VERIFICATION, verification)
    if status != "PASS":
        raise RuntimeError(f"Formal manifest verification failed: {checks}")

    with FORMAL_SCHEDULE.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("schedule_index", "scenario_id", "stage", "family", "seed", "shard_index", "shard_count"))
        writer.writeheader()
        for index, entry in enumerate(entries):
            writer.writerow({
                "schedule_index": index,
                "scenario_id": entry["scenario_id"],
                "stage": entry["stage"],
                "family": entry["family"],
                "seed": entry["seed"],
                "shard_index": index % SHARD_COUNT,
                "shard_count": SHARD_COUNT,
            })
    authorization = {
        "schema_version": "formal_strong_run_authorization_v1",
        "created_at": now(),
        "status": "AUTHORIZED_ONCE",
        "identity_audit_sha256": sha256_file(IDENTITY_AUDIT),
        "runtime_contract_sha256": sha256_file(RUNTIME_CONTRACT),
        "prefreeze_sha256": sha256_file(PREFREEZE),
        "microtest_sha256": sha256_file(MICROTEST),
        "acceptance_rule_sha256": sha256_file(ACCEPTANCE_RULE),
        "manifest_verification_sha256": sha256_file(MANIFEST_VERIFICATION),
        "schedule_sha256": sha256_file(FORMAL_SCHEDULE),
        "scenario_count": 400,
        "method_count": 1,
        "method_identity": "FROZEN_STRONG",
        "shard_count": SHARD_COUNT,
        "rerun_other_methods": False,
        "post_authorization_tuning_allowed": False,
    }
    atomic_json(RUN_AUTHORIZATION, authorization)
    print(json.dumps({"phase": "verify-manifest", "status": "PASS", "scenario_count": 400}, indent=2))


def formal_entries() -> list[dict[str, Any]]:
    return [dict(entry) for entry in load_json(FORMAL_MANIFEST)["entries"]]


def run_formal(shard_index: int, shard_count: int) -> None:
    if int(shard_count) != SHARD_COUNT or not 0 <= int(shard_index) < SHARD_COUNT:
        raise ValueError(f"Formal schedule is frozen to {SHARD_COUNT} shards")
    verify_frozen_sources()
    authorization = load_json(RUN_AUTHORIZATION)
    if authorization.get("status") != "AUTHORIZED_ONCE" or int(authorization.get("scenario_count", 0)) != 400:
        raise RuntimeError("Formal run is not authorized")
    if sha256_file(FORMAL_SCHEDULE) != authorization["schedule_sha256"]:
        raise RuntimeError("Formal schedule drift")

    manifest = load_json(FORMAL_MANIFEST)
    runtime = frozen_formal_runtime(manifest)
    entries = [entry for index, entry in enumerate(formal_entries()) if index % SHARD_COUNT == int(shard_index)]
    health_path = ROOT / f"03_formal_run/health_shard_{shard_index}.json"
    started = time.time()
    completed = 0
    health = {
        "schema_version": "frozen_strong_formal_shard_health_v1",
        "shard_index": int(shard_index),
        "shard_count": SHARD_COUNT,
        "assigned_count": len(entries),
        "completed_count": 0,
        "software_error_count": 0,
        "nan_inf_count": 0,
        "status": "RUNNING",
        "started_at": now(),
    }
    atomic_json(health_path, health)
    for local_index, entry in enumerate(entries, start=1):
        sid = str(entry["scenario_id"])
        targets = (
            RECORD_DIR / f"{sid}.json",
            RECORD_DIR / f"{sid}_trajectory.npz",
            RECORD_DIR / f"{sid}_limiter_trace.npz",
        )
        if any(path.exists() for path in targets):
            raise RuntimeError(f"refusing duplicate Formal execution for {sid}")
        print(f"[formal-strong:{shard_index}/{SHARD_COUNT}] {local_index:03d}/{len(entries):03d} start {sid}", flush=True)
        recorder = strong_base.OnlineRuntimeRecorder()
        proxy = strong_base.TimedPolicyProxy(runtime.policy, recorder)
        limiter = strong_base.limiter_for("strong")
        try:
            with recorder.instrument_dmp(), recorder.scoped_context(
                evaluation_block="frozen_strong_formal_v2",
                configuration_id="FORMAL_V2_FROZEN_STRONG_P60_NO_EARLY_BYPASS",
                stage=entry["stage"], family=entry["family"], scenario_id=sid,
                seed=int(entry["seed"]), method=strong_base.METHOD_RERR_GAT,
            ):
                episode, agents, events, triggers, extra = strong_base.run_episode(
                    config=runtime.eval_config, settings=runtime.settings,
                    multi_config=runtime.multi_config, policy=proxy,
                    gat_model=runtime.gat_model, gat_device=runtime.gat_device,
                    method=strong_base.METHOD_RERR_GAT, scenario=sid, seed=int(entry["seed"]),
                    environment_builder=runtime.builder, runtime_recorder=recorder,
                    upper_plan_builder=strong_base.build_online_gat_plan_optimized,
                    execution_acceleration_limiter=limiter,
                )
            strong_base.save_episode_record(RECORD_DIR, entry, episode, agents, events, triggers, extra)
            with np.load(RECORD_DIR / f"{sid}_trajectory.npz") as arrays:
                if not all(np.isfinite(np.asarray(arrays[key], dtype=float)).all() for key in ("positions", "velocities")):
                    raise FloatingPointError("serialized trajectory contains NaN/Inf")
            completed += 1
            health.update({"completed_count": completed, "last_scenario_id": sid, "updated_at": now()})
            atomic_json(health_path, health)
            print(f"[formal-strong:{shard_index}/{SHARD_COUNT}] complete {sid} records={completed}/{len(entries)}", flush=True)
        except Exception as error:
            atomic_json(RECORD_DIR / f"{sid}_SOFTWARE_ERROR.json", {
                "scenario_id": sid,
                "shard_index": int(shard_index),
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
                "control_semantics_changed": False,
            })
            health.update({
                "status": "FAILED",
                "completed_count": completed,
                "software_error_count": 1,
                "failed_scenario_id": sid,
                "finished_at": now(),
            })
            atomic_json(health_path, health)
            raise
    health.update({
        "status": "PASS",
        "completed_count": completed,
        "software_error_count": 0,
        "elapsed_s": time.time() - started,
        "finished_at": now(),
    })
    atomic_json(health_path, health)


def status() -> None:
    json_records = [path for path in RECORD_DIR.glob("FORMAL_LR_*.json") if "SOFTWARE_ERROR" not in path.name]
    trajectories = list(RECORD_DIR.glob("FORMAL_LR_*_trajectory.npz"))
    traces = list(RECORD_DIR.glob("FORMAL_LR_*_limiter_trace.npz"))
    errors = list(RECORD_DIR.glob("*_SOFTWARE_ERROR.json"))
    payload = {
        "record_count": len(json_records),
        "trajectory_count": len(trajectories),
        "limiter_trace_count": len(traces),
        "software_error_count": len(errors),
        "complete": len(json_records) == len(trajectories) == len(traces) == 400 and not errors,
    }
    print(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("freeze", "microtest", "verify-manifest", "run", "status"))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=SHARD_COUNT)
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze_identity()
    elif args.phase == "microtest":
        run_microtests()
    elif args.phase == "verify-manifest":
        verify_manifest()
    elif args.phase == "run":
        run_formal(args.shard_index, args.shard_count)
    else:
        status()


if __name__ == "__main__":
    main()
