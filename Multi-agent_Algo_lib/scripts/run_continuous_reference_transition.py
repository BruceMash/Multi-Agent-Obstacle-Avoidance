#!/usr/bin/env python3
"""Paired Development and sealed Holdout runner for Proposed-CRT.

The runner reuses frozen GAT-R, R-ERR, SAC-DMP, and file-backed long-range
scenes.  Its only optional runtime change is the Continuous Reference
Transition passed into the established episode engine.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as _pandas  # noqa: F401  # Windows pyarrow/torch initialization order


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
for search_path in (REPO_ROOT, ALGO_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from planning.long_range_collision_recheck import audit_trajectory_collisions  # noqa: E402
from planning.online_runtime_instrumentation import OnlineRuntimeRecorder, TimedPolicyProxy  # noqa: E402
from planning.semi_structured_long_range_benchmark import json_ready  # noqa: E402
from scripts.evaluate_gat_v1_err_development import (  # noqa: E402
    METHOD_RERR_GAT,
    build_online_gat_plan_optimized,
    run_episode,
)
from scripts.run_gat_recurrent_r_development import (  # noqa: E402
    FileBackedBuilder,
    load_json,
    resolve_runtime_config,
)
from scripts.run_long_range_contract_pilot import summarize_record  # noqa: E402
from scripts.run_long_range_development import DevelopmentRuntime  # noqa: E402


ARTIFACT_ROOT = REPO_ROOT / "artifacts/continuous_reference_transition/20260824_132552"
SOURCE_ROOT = REPO_ROOT / "artifacts/gat_recurrent_smoothness_rescue/20260821_180244"
DEV_CONFIG = REPO_ROOT / "configs/evaluation/gat_recurrent_r_fp_anchor_dev.json"
HOLDOUT_CONFIG = REPO_ROOT / "configs/evaluation/gat_recurrent_r_fp_anchor_holdout.json"
FORMAL_MANIFEST = SOURCE_ROOT / "10_formal_v2/FORMAL_V2_MANIFEST.json"
DEV_SUBSET_MANIFEST = ARTIFACT_ROOT / "00_context/CRT_DEVELOPMENT_MANIFEST.json"
HOLDOUT_SUBSET_MANIFEST = ARTIFACT_ROOT / "00_context/CRT_HOLDOUT_MANIFEST.json"

VARIANT_T_REF = {
    "original": None,
    "crt_0p2": 0.2,
    "crt_0p3": 0.3,
    "crt_0p5": 0.5,
}

SOURCE_PATHS = (
    "planning/continuous_reference_transition.py",
    "planning/event_triggered_reference_reconstruction.py",
    "planning/goal_semantics_diagnosis.py",
    "Environment/frozen_sac_dmp_execution.py",
    "Environment/multi_agent_dmp_env.py",
    "Controller/dmp_rl.py",
    "Entity/KinematicModel.py",
    "Multi-agent_Algo_lib/scripts/evaluate_gat_v1_err_development.py",
    "Multi-agent_Algo_lib/scripts/run_continuous_reference_transition.py",
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
    rows = list(rows)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields or ["status"], extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(json_ready(row.get(key)), ensure_ascii=False)
                    if isinstance(row.get(key), (dict, list, tuple, np.ndarray))
                    else row.get(key)
                    for key in fields
                }
            )
    temporary.replace(path)


def balanced_subset(index: Mapping[str, Any], per_family: int = 5) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for raw in index["entries"]:
        row = dict(raw)
        groups.setdefault((str(row["stage"]), str(row["family"])), []).append(row)
    if len(groups) != 20:
        raise RuntimeError(f"expected 20 stage-family groups, got {len(groups)}")
    selected: list[dict[str, Any]] = []
    for key in sorted(groups):
        members = sorted(groups[key], key=lambda row: str(row["scenario_id"]))
        if len(members) < per_family:
            raise RuntimeError(f"insufficient scenes in group {key}")
        selected.extend(members[:per_family])
    if len(selected) != 100:
        raise RuntimeError("balanced CRT subset must contain exactly 100 scenes")
    return selected


def identity_sets(entries: Sequence[Mapping[str, Any]]) -> dict[str, set[Any]]:
    keys = (
        "seed",
        "geometry_fingerprint",
        "dynamic_track_fingerprint",
        "translation_invariant_fingerprint",
        "start_goal_fingerprint",
    )
    return {
        key: {
            row[key]
            for row in entries
            if key in row and row[key] not in (None, "")
        }
        for key in keys
    }


def prepare() -> None:
    for name in (
        "00_context",
        "01_handoff_contract",
        "02_implementation",
        "03_microtests",
        "04_development",
        "05_event_alignment",
        "06_selection",
        "07_holdout",
        "08_runtime",
        "09_figures",
        "10_freeze",
        "11_paper_ready",
    ):
        (ARTIFACT_ROOT / name).mkdir(parents=True, exist_ok=True)

    dev_config = load_json(DEV_CONFIG)
    holdout_config = load_json(HOLDOUT_CONFIG)
    dev_source_manifest = SOURCE_ROOT / str(dev_config["development_manifest"])
    holdout_source_manifest = SOURCE_ROOT / str(
        holdout_config["development_manifest"]
    )
    dev_entries = balanced_subset(load_json(dev_source_manifest))
    holdout_entries = balanced_subset(load_json(holdout_source_manifest))
    formal_entries = load_json(FORMAL_MANIFEST)["entries"]
    dev_ids = identity_sets(dev_entries)
    holdout_ids = identity_sets(holdout_entries)
    formal_ids = identity_sets(formal_entries)
    overlap = {
        "development_vs_holdout": {
            key: len(dev_ids[key] & holdout_ids[key]) for key in dev_ids
        },
        "development_vs_formal_v2": {
            key: len(dev_ids[key] & formal_ids[key]) for key in dev_ids
        },
        "holdout_vs_formal_v2": {
            key: len(holdout_ids[key] & formal_ids[key]) for key in dev_ids
        },
    }
    if any(value for section in overlap.values() for value in section.values()):
        raise RuntimeError(f"CRT split identity overlap detected: {overlap}")

    atomic_json(
        DEV_SUBSET_MANIFEST,
        {
            "schema_version": "crt_paired_development_manifest_v1",
            "source_manifest": str(dev_source_manifest.relative_to(REPO_ROOT).as_posix()),
            "source_manifest_sha256": sha256_file(dev_source_manifest),
            "selection_rule": "first five lexicographic scenario ids per stage-family cell",
            "scenario_count": len(dev_entries),
            "entries": dev_entries,
        },
    )
    atomic_json(
        HOLDOUT_SUBSET_MANIFEST,
        {
            "schema_version": "crt_sealed_holdout_manifest_v1",
            "status": "SEALED_BEFORE_DEVELOPMENT_PERFORMANCE",
            "source_manifest": str(
                holdout_source_manifest.relative_to(REPO_ROOT).as_posix()
            ),
            "source_manifest_sha256": sha256_file(holdout_source_manifest),
            "selection_rule": "first five lexicographic scenario ids per stage-family cell",
            "scenario_count": len(holdout_entries),
            "entries": holdout_entries,
        },
    )

    runtime_config = resolve_runtime_config(dev_config, "gat_r")
    checkpoint_paths = {
        "gat_r": REPO_ROOT / str(runtime_config["gat_checkpoint"]),
        "sac_dmp": REPO_ROOT / str(runtime_config["sac_checkpoint"]),
    }
    freeze = {
        "schema_version": "crt_predevelopment_freeze_v1",
        "status": "FROZEN_BEFORE_DEVELOPMENT",
        "artifact_root": str(ARTIFACT_ROOT.relative_to(REPO_ROOT).as_posix()),
        "development_scenario_count": len(dev_entries),
        "holdout_scenario_count": len(holdout_entries),
        "stage_family_balance": "4 stages x 5 families x 5 scenes = 100 per block",
        "development_manifest_sha256": sha256_file(DEV_SUBSET_MANIFEST),
        "holdout_manifest_sha256": sha256_file(HOLDOUT_SUBSET_MANIFEST),
        "formal_v2_manifest_sha256": sha256_file(FORMAL_MANIFEST),
        "identity_overlap": overlap,
        "variants": {
            key: {
                "settling_time_s": value,
                "omega_rad_s": None if value is None else 5.83392170191739 / value,
            }
            for key, value in VARIANT_T_REF.items()
        },
        "parameter_mapping": "T_ref is the 2% critical settling time; omega=5.83392170191739/T_ref",
        "fixed_safety_bandwidth_multiplier": 2.0,
        "source_sha256": {
            path: sha256_file(REPO_ROOT / path) for path in SOURCE_PATHS
        },
        "checkpoint_sha256": {
            key: sha256_file(path) for key, path in checkpoint_paths.items()
        },
        "previous_audit": {
            "conclusion": "artifacts/sector_resolution_oscillation_audit/20260824_110611/conclusion.json",
            "conclusion_sha256": sha256_file(
                REPO_ROOT
                / "artifacts/sector_resolution_oscillation_audit/20260824_110611/conclusion.json"
            ),
            "independent_reconciliation_sha256": sha256_file(
                REPO_ROOT
                / "artifacts/sector_resolution_oscillation_audit/20260824_110611/INDEPENDENT_RECONCILIATION.json"
            ),
        },
        "formal_v2_execution_authorized": False,
    }
    atomic_json(ARTIFACT_ROOT / "00_context/CRT_PREDEVELOPMENT_FREEZE.json", freeze)


def block_contract(block: str) -> tuple[Path, Path, dict[str, Any]]:
    if block == "development":
        config_path = DEV_CONFIG
        subset_path = DEV_SUBSET_MANIFEST
    elif block == "holdout":
        config_path = HOLDOUT_CONFIG
        subset_path = HOLDOUT_SUBSET_MANIFEST
    elif block == "microtest":
        config_path = DEV_CONFIG
        subset_path = DEV_SUBSET_MANIFEST
    else:
        raise ValueError(f"unsupported block {block}")
    return config_path, subset_path, load_json(config_path)


def output_dir(block: str, variant: str) -> Path:
    if block == "development":
        return ARTIFACT_ROOT / "04_development/records" / variant
    if block == "holdout":
        return ARTIFACT_ROOT / "07_holdout/records" / variant
    return ARTIFACT_ROOT / "03_microtests/integration_records" / variant


def verify_holdout_freeze(variant: str) -> None:
    freeze_path = ARTIFACT_ROOT / "10_freeze/FINAL_CRT_FREEZE.json"
    selection_path = ARTIFACT_ROOT / "06_selection/CRT_DEVELOPMENT_SELECTION.json"
    if not freeze_path.exists() or not selection_path.exists():
        raise RuntimeError("Holdout requires a completed Development selection and freeze")
    freeze = load_json(freeze_path)
    selection = load_json(selection_path)
    selected = str(selection["selected_variant"])
    if freeze["status"] != "FROZEN_BEFORE_HOLDOUT" or not selection["holdout_authorized"]:
        raise RuntimeError("Development did not authorize Holdout")
    if variant not in {"original", selected}:
        raise RuntimeError(
            f"Holdout permits only original and the frozen selected variant {selected}"
        )
    if sha256_file(selection_path) != freeze["development_selection_sha256"]:
        raise RuntimeError("Development selection changed after freeze")
    if sha256_file(HOLDOUT_SUBSET_MANIFEST) != freeze["holdout_manifest_sha256"]:
        raise RuntimeError("Holdout manifest changed after freeze")
    for relative, expected in freeze["source_sha256"].items():
        if sha256_file(REPO_ROOT / relative) != expected:
            raise RuntimeError(f"frozen CRT source changed before Holdout: {relative}")


def trajectory_arrays(path_rows: Sequence[Mapping[str, Any]], num_agents: int) -> dict[str, np.ndarray]:
    if not path_rows:
        raise ValueError("path_rows must be non-empty")
    steps = sorted({int(row["step"]) for row in path_rows})
    if steps != list(range(max(steps) + 1)):
        raise RuntimeError("trajectory steps must be contiguous from zero")
    shape = (len(steps), int(num_agents), 3)
    arrays = {
        "positions": np.full(shape, np.nan, dtype=np.float64),
        "velocities": np.full(shape, np.nan, dtype=np.float64),
        "commanded_accelerations_full": np.full(shape, np.nan, dtype=np.float64),
        "applied_accelerations_full": np.full(shape, np.nan, dtype=np.float64),
        "g_cmd": np.full(shape, np.nan, dtype=np.float64),
        "g_exec": np.full(shape, np.nan, dtype=np.float64),
        "gdot_exec": np.full(shape, np.nan, dtype=np.float64),
        "command_error_m": np.full(shape[:2], np.nan, dtype=np.float64),
        "filter_speed_mps": np.full(shape[:2], np.nan, dtype=np.float64),
    }
    for row in path_rows:
        step = int(row["step"])
        agent = int(row["agent_id"])
        arrays["positions"][step, agent] = [row["x_m"], row["y_m"], row["z_m"]]
        arrays["velocities"][step, agent] = [row["vx_mps"], row["vy_mps"], row["vz_mps"]]
        arrays["commanded_accelerations_full"][step, agent] = [
            row["commanded_ax_mps2"], row["commanded_ay_mps2"], row["commanded_az_mps2"]
        ]
        arrays["applied_accelerations_full"][step, agent] = [
            row["applied_ax_mps2"], row["applied_ay_mps2"], row["applied_az_mps2"]
        ]
        arrays["g_cmd"][step, agent] = [
            row["commanded_reference_x_m"], row["commanded_reference_y_m"], row["commanded_reference_z_m"]
        ]
        arrays["g_exec"][step, agent] = [
            row["executed_reference_x_m"], row["executed_reference_y_m"], row["executed_reference_z_m"]
        ]
        arrays["gdot_exec"][step, agent] = [
            row["executed_reference_vx_mps"], row["executed_reference_vy_mps"], row["executed_reference_vz_mps"]
        ]
        arrays["command_error_m"][step, agent] = row["command_execution_error_m"]
        arrays["filter_speed_mps"][step, agent] = row["reference_filter_speed_mps"]
    if any(not np.all(np.isfinite(value)) for key, value in arrays.items() if key not in {"commanded_accelerations_full", "applied_accelerations_full"}):
        raise RuntimeError("non-finite trajectory/reference state")
    arrays["commanded_accelerations"] = arrays["commanded_accelerations_full"][1:]
    arrays["applied_accelerations"] = arrays["applied_accelerations_full"][1:]
    return arrays


def extended_metrics(arrays: Mapping[str, np.ndarray], dt: float, starts: np.ndarray, goals: np.ndarray) -> dict[str, Any]:
    accelerations = np.asarray(arrays["applied_accelerations"], dtype=float)
    jerk = np.diff(accelerations, axis=0) / float(dt)
    positions = np.asarray(arrays["positions"], dtype=float)
    step_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=2)
    path_lengths = np.sum(step_lengths, axis=0)
    straight = np.linalg.norm(np.asarray(goals) - np.asarray(starts), axis=1)
    efficiency = np.divide(
        straight,
        path_lengths,
        out=np.full_like(straight, np.nan),
        where=path_lengths > 0.0,
    )
    return {
        "vertical_jerk_mean_squared_m2_s6": float(np.mean(jerk[:, :, 2] ** 2)) if jerk.size else None,
        "lateral_jerk_mean_squared_m2_s6": float(np.mean(np.sum(jerk[:, :, :2] ** 2, axis=2))) if jerk.size else None,
        "jerk_norm_peak_mps3": float(np.max(np.linalg.norm(jerk, axis=2))) if jerk.size else None,
        "vertical_jerk_peak_mps3": float(np.max(np.abs(jerk[:, :, 2]))) if jerk.size else None,
        "path_efficiency_mean": float(np.nanmean(efficiency)),
        "path_efficiency_per_agent": efficiency.tolist(),
    }


def save_episode(
    target: Path,
    entry: Mapping[str, Any],
    episode: dict[str, Any],
    agents: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    triggers: Sequence[Mapping[str, Any]],
    extra: Mapping[str, Any],
    summary: dict[str, Any],
    source_scene: Mapping[str, Any],
) -> None:
    records = target / "episode_records"
    records.mkdir(parents=True, exist_ok=True)
    sid = str(entry["scenario_id"])
    arrays = trajectory_arrays(extra["path_rows"], int(episode["num_agents"]))
    collision = audit_trajectory_collisions(
        arrays["positions"],
        source_scene,
        collision_margin_m=0.0,
        peer_threshold_m=0.6,
    )
    if bool(collision["any_collision"]) != bool(episode["collision"]):
        raise RuntimeError(
            f"independent collision recheck disagrees for {sid}: "
            f"{collision['any_collision']} != {episode['collision']}"
        )
    episode.update(
        {
            "static_obstacle_collision": bool(collision["static_obstacle_collision"]),
            "dynamic_obstacle_collision": bool(collision["dynamic_obstacle_collision"]),
            "minimum_static_obstacle_signed_clearance_m": float(collision["minimum_static_obstacle_signed_clearance_m"]),
            "minimum_dynamic_obstacle_signed_clearance_m": float(collision["minimum_dynamic_obstacle_signed_clearance_m"]),
            "minimum_obstacle_signed_clearance_rechecked_m": float(collision["minimum_obstacle_signed_clearance_m"]),
            **extended_metrics(
                arrays,
                float(episode["dt"]),
                np.asarray(source_scene["starts"], dtype=float),
                np.asarray(source_scene["goals"], dtype=float),
            ),
        }
    )
    summary.update(
        {
            "static_obstacle_collision": episode["static_obstacle_collision"],
            "dynamic_obstacle_collision": episode["dynamic_obstacle_collision"],
            "vertical_jerk_mean_squared_m2_s6": episode["vertical_jerk_mean_squared_m2_s6"],
            "lateral_jerk_mean_squared_m2_s6": episode["lateral_jerk_mean_squared_m2_s6"],
            "path_efficiency_mean": episode["path_efficiency_mean"],
            "crt_enabled": episode["crt_enabled"],
            "crt_settling_time_s": episode["crt_settling_time_s"],
            "crt_omega_rad_s": episode["crt_omega_rad_s"],
            "crt_bandwidth_override_count": episode["crt_bandwidth_override_count"],
            "crt_instant_replacement_count": episode["crt_instant_replacement_count"],
            "crt_total_runtime_ms": episode["crt_total_runtime_ms"],
        }
    )
    npz_path = records / f"{sid}_trajectory.npz"
    np.savez_compressed(
        npz_path,
        **arrays,
        dt=np.asarray(float(episode["dt"]), dtype=np.float64),
    )
    trajectory_sha = sha256_file(npz_path)
    atomic_json(
        records / f"{sid}.json",
        {
            "entry_identity": {
                key: entry[key]
                for key in (
                    "scenario_id",
                    "seed",
                    "stage",
                    "family",
                    "task_pattern",
                    "geometry_fingerprint",
                    "dynamic_track_fingerprint",
                )
            },
            "summary": summary,
            "episode": episode,
            "agents": list(agents),
            "events": list(events),
            "trigger_summary": {
                "row_count": len(triggers),
                "event_counts": {
                    event: sum(row["event"] == event for row in triggers)
                    for event in sorted({row["event"] for row in triggers})
                },
            },
            "trajectory_file": npz_path.name,
            "trajectory_sha256": trajectory_sha,
            "path_row_count": len(extra["path_rows"]),
            "timing": {
                "actor_rows": extra["actor_timing_rows"],
                "dmp_rows": extra["dmp_timing_rows"],
                "upper_rows": extra["upper_timing_rows"],
            },
        },
    )


def run(block: str, variant: str, shard_index: int | None, shard_count: int | None, limit: int | None) -> None:
    if variant not in VARIANT_T_REF:
        raise ValueError(f"unknown variant {variant}")
    if block == "holdout":
        verify_holdout_freeze(variant)
    config_path, subset_path, source_config = block_contract(block)
    if not subset_path.exists():
        raise RuntimeError("run prepare before episode execution")
    subset = load_json(subset_path)
    runtime_config = resolve_runtime_config(source_config, "gat_r")
    runtime = DevelopmentRuntime(runtime_config, {"entries": []})
    runtime.builder = FileBackedBuilder(subset, runtime_config, SOURCE_ROOT)
    target = output_dir(block, variant)
    completed = {
        path.stem
        for path in (target / "episode_records").glob("*.json")
        if not path.stem.endswith("_SOFTWARE_ERROR")
    }
    indexed = list(enumerate(subset["entries"]))
    if block == "microtest":
        indexed = indexed[:1]
    if shard_count is not None:
        if shard_index is None or not 0 <= int(shard_index) < int(shard_count):
            raise ValueError("invalid shard index/count")
        indexed = [
            pair for pair in indexed if pair[0] % int(shard_count) == int(shard_index)
        ]
    pending = [entry for _, entry in indexed if str(entry["scenario_id"]) not in completed]
    if limit is not None:
        pending = pending[: int(limit)]
    t_ref = VARIANT_T_REF[variant]
    crt = (
        None
        if t_ref is None
        else {
            "settling_time_s": float(t_ref),
            "safety_bandwidth_multiplier": 2.0,
            "hard_safety_margin_m": float(runtime_config["err"]["h_emg_m"]),
            "settled_absolute_tolerance_m": 1.0e-6,
        }
    )
    for entry in pending:
        sid = str(entry["scenario_id"])
        print(f"[CRT:{block}:{variant}] start {sid}", flush=True)
        recorder = OnlineRuntimeRecorder()
        proxy = TimedPolicyProxy(runtime.policy, recorder)
        scene_path = SOURCE_ROOT / str(entry["scenario_file"])
        source_scene = load_json(scene_path)
        try:
            with recorder.instrument_dmp(), recorder.scoped_context(
                evaluation_block=f"crt_{block}",
                configuration_id=f"CRT_{block.upper()}_{variant.upper()}",
                stage=entry["stage"],
                family=entry["family"],
                scenario_id=sid,
                seed=int(entry["seed"]),
                method=variant,
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
                    crt=crt,
                )
            summary = summarize_record(
                entry, episode, events, triggers, extra["path_rows"]
            )
            summary.update(
                {
                    "configuration_id": f"CRT_{block.upper()}_{variant.upper()}",
                    "variant": variant,
                    "method": "Original Proposed" if variant == "original" else "Proposed-CRT",
                    "team_success": bool(episode["team_success"]),
                    "collision": bool(episode["collision"]),
                    "timeout": bool(episode["timeout"]),
                    "performance_used_for_selection": block == "development",
                }
            )
            save_episode(
                target,
                entry,
                episode,
                agents,
                events,
                triggers,
                extra,
                summary,
                source_scene,
            )
            print(
                f"[CRT:{block}:{variant}] complete {sid} "
                f"success={int(summary['team_success'])} collision={int(summary['collision'])}",
                flush=True,
            )
        except Exception as error:
            atomic_json(
                target / "episode_records" / f"{sid}_SOFTWARE_ERROR.json",
                {
                    "scenario_id": sid,
                    "seed": int(entry["seed"]),
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
            raise


def collect(block: str, variant: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((output_dir(block, variant) / "episode_records").glob("*.json")):
        if path.stem.endswith("_SOFTWARE_ERROR"):
            continue
        rows.append(load_json(path)["summary"])
    return rows


def finalize(block: str, variant: str) -> None:
    _, subset_path, _ = block_contract(block)
    expected = 1 if block == "microtest" else len(load_json(subset_path)["entries"])
    rows = collect(block, variant)
    errors = list((output_dir(block, variant) / "episode_records").glob("*_SOFTWARE_ERROR.json"))
    write_csv(output_dir(block, variant) / "episode_summary.csv", rows)
    reconciliation = {
        "schema_version": "crt_variant_reconciliation_v1",
        "block": block,
        "variant": variant,
        "expected_episode_count": expected,
        "completed_episode_count": len(rows),
        "software_error_count": len(errors),
        "unique_scenario_count": len({row["scenario_id"] for row in rows}),
        "status": "PASS" if len(rows) == expected and not errors else "INCOMPLETE",
    }
    atomic_json(output_dir(block, variant) / "reconciliation.json", reconciliation)
    print(json.dumps(reconciliation), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "run", "finalize"))
    parser.add_argument("--block", choices=("microtest", "development", "holdout"))
    parser.add_argument("--variant", choices=tuple(VARIANT_T_REF))
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare()
        return
    if args.block is None or args.variant is None:
        raise ValueError("--block and --variant are required")
    if args.phase == "run":
        run(args.block, args.variant, args.shard_index, args.shard_count, args.limit)
    else:
        finalize(args.block, args.variant)


if __name__ == "__main__":
    main()
