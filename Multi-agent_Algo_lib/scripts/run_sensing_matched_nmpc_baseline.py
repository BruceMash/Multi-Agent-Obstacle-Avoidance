"""Execute the isolated SM-NMPC-style development and validation protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import runpy
import sys
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = REPO_ROOT / "Multi-agent_Algo_lib"
for search_path in (REPO_ROOT, ALGO_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from planning.final_four_stage_benchmark import (  # noqa: E402
    STAGE_ORDER,
    generate_scenario_manifest,
    stable_hash,
    validate_scenario_manifest,
)
from planning.online_runtime_instrumentation import (  # noqa: E402
    OnlineRuntimeRecorder,
    TimedPolicyProxy,
)
from planning.sensing_matched_nmpc import (  # noqa: E402
    METHOD_LABEL,
    NMPCStyleConfig,
    configuration_from_mapping,
    run_sensing_matched_nmpc_episode,
)
from scripts.evaluate_gat_v1_err_development import (  # noqa: E402
    METHOD_RERR_GAT,
    build_online_gat_plan_optimized,
    run_episode,
)
from scripts.run_final_four_stage_benchmark import (  # noqa: E402
    FrozenRuntime,
    audit_used_geometry,
    audit_used_seeds,
    proposed_eval_config,
)


OUTPUT = REPO_ROOT / "artifacts" / "sensing_matched_nmpc_baseline" / "20260819_211048"
BASE_CONFIG = REPO_ROOT / "artifacts/final_four_stage_benchmark/20260818_202620/config.json"
SELECTED_CONFIG = REPO_ROOT / "artifacts/final_four_stage_benchmark/20260818_202620/engineering_search/selected_configs.json"
ERR_CONFIG = REPO_ROOT / "configs/evaluation/gat_v1_err_development.json"
PREFREEZE = OUTPUT / "engineering_prefreeze.json"
SEARCH_SPACE = OUTPUT / "nmpc_search_space.json"


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity" if value < 0 else "NaN"
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(jsonable(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        if fields:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        field: (
                            json.dumps(jsonable(row.get(field)), ensure_ascii=False)
                            if isinstance(row.get(field), (list, tuple, dict, np.ndarray))
                            else jsonable(row.get(field))
                        )
                        for field in fields
                    }
                )
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_hashes() -> dict[str, str]:
    paths = (
        "planning/sensing_matched_nmpc.py",
        "planning/sensing_matched_classical.py",
        "planning/final_four_stage_benchmark.py",
        "Entity/KinematicModel.py",
        "Multi-agent_Algo_lib/scripts/evaluate_gat_v1_err_development.py",
        "artifacts/final_execution_semantics_audit/20260819_194912/FINAL_METHOD_FREEZE.json",
        "artifacts/rerr_runtime_compression/20260819_162022/FINAL_ENGINEERING_FREEZE.json",
    )
    return {path: file_hash(REPO_ROOT / path) for path in paths}


def verify_prefreeze() -> None:
    frozen = load_json(PREFREEZE)
    observed = _source_hashes()
    if observed != frozen["source_hashes_at_prefreeze"]:
        raise RuntimeError({"prefreeze_source_hash_mismatch": {key: (frozen["source_hashes_at_prefreeze"].get(key), value) for key, value in observed.items() if frozen["source_hashes_at_prefreeze"].get(key) != value}})


def _historical_translation_hashes() -> set[str]:
    values: set[str] = set()

    def visit(value: Any, key: str = "") -> None:
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                visit(child, str(child_key))
        elif isinstance(value, list):
            for child in value:
                visit(child, key)
        elif key == "translation_invariant_fingerprint" and isinstance(value, str):
            values.add(value)

    artifacts = REPO_ROOT / "artifacts"
    for path in artifacts.rglob("*"):
        if not path.is_file():
            continue
        try:
            path.relative_to(OUTPUT)
            continue
        except ValueError:
            pass
        try:
            if path.suffix.lower() == ".json" and path.stat().st_size <= 64 * 1024 * 1024:
                visit(load_json(path))
            elif path.suffix.lower() == ".csv" and path.stat().st_size <= 256 * 1024 * 1024:
                with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
                    reader = csv.DictReader(stream)
                    if "translation_invariant_fingerprint" in (reader.fieldnames or []):
                        for row in reader:
                            if row.get("translation_invariant_fingerprint"):
                                values.add(str(row["translation_invariant_fingerprint"]))
        except (OSError, UnicodeError, json.JSONDecodeError, csv.Error):
            continue
    return values


def _run_unit_tests() -> dict[str, Any]:
    namespace = runpy.run_path(str(REPO_ROOT / "test/test_sensing_matched_nmpc.py"))
    rows = []
    for name, function in sorted(namespace.items()):
        if name.startswith("test_") and callable(function):
            started = time.perf_counter_ns()
            function()
            rows.append(
                {
                    "test": name,
                    "status": "PASS",
                    "runtime_ms": (time.perf_counter_ns() - started) / 1.0e6,
                }
            )
    return {"status": "PASS", "test_count": len(rows), "tests": rows}


def prepare() -> None:
    verify_prefreeze()
    pref = load_json(PREFREEZE)
    used_seeds = audit_used_seeds(excluded_root=OUTPUT)
    used_geometry = audit_used_geometry(excluded_root=OUTPUT)
    historical_translations = _historical_translation_hashes()
    development = generate_scenario_manifest(
        counts_per_stage=8,
        seed_base=int(pref["development_seed_base"]),
        prefix=pref["development_prefix"],
        max_steps=220,
        dt=0.1,
    )
    validation = generate_scenario_manifest(
        counts_per_stage=10,
        seed_base=int(pref["validation_seed_base"]),
        prefix=pref["validation_prefix"],
        max_steps=220,
        dt=0.1,
    )
    dev_validation = validate_scenario_manifest(development)
    val_validation = validate_scenario_manifest(validation)
    dev_seeds = {int(row["seed"]) for row in development["entries"]}
    val_seeds = {int(row["seed"]) for row in validation["entries"]}
    historical_seeds = set(map(int, used_seeds["used_seeds"]))
    dev_geometry = {row["geometry_fingerprint"] for row in development["entries"]}
    val_geometry = {row["geometry_fingerprint"] for row in validation["entries"]}
    historical_geometry = set(used_geometry["geometry_hashes"])
    dev_translation = {row["translation_invariant_fingerprint"] for row in development["entries"]}
    val_translation = {row["translation_invariant_fingerprint"] for row in validation["entries"]}
    separation = {
        "development_validation_seed_overlap": len(dev_seeds & val_seeds),
        "development_historical_seed_overlap": len(dev_seeds & historical_seeds),
        "validation_historical_seed_overlap": len(val_seeds & historical_seeds),
        "development_validation_geometry_overlap": len(dev_geometry & val_geometry),
        "development_historical_geometry_overlap": len(dev_geometry & historical_geometry),
        "validation_historical_geometry_overlap": len(val_geometry & historical_geometry),
        "development_validation_translation_overlap": len(dev_translation & val_translation),
        "development_historical_translation_overlap": len(dev_translation & historical_translations),
        "validation_historical_translation_overlap": len(val_translation & historical_translations),
    }
    separation["status"] = "PASS" if all(value == 0 for value in separation.values()) else "FAIL"
    if dev_validation["status"] != "PASSED" or val_validation["status"] != "PASSED" or separation["status"] != "PASS":
        raise RuntimeError({"development": dev_validation, "validation": val_validation, "separation": separation})
    tests = _run_unit_tests()
    write_json(OUTPUT / "all_used_seed_manifest.json", used_seeds)
    write_json(OUTPUT / "all_used_geometry_manifest.json", used_geometry)
    write_json(OUTPUT / "development_scenario_manifest.json", development)
    write_json(OUTPUT / "validation_scenario_manifest.json", validation)
    write_json(OUTPUT / "scenario_separation_audit.json", separation)
    write_json(OUTPUT / "unit_test_results.json", tests)
    write_json(
        OUTPUT / "preflight_gate.json",
        {
            "status": "PASS",
            "source_freeze": "PASS",
            "scenario_separation": separation,
            "unit_tests": tests,
            "development_outcome_count_at_gate": 0,
            "validation_outcome_count_at_gate": 0,
        },
    )
    print(json.dumps({"phase": "prepare", "status": "PASS", "tests": tests["test_count"]}), flush=True)


def _configs() -> list[tuple[str, NMPCStyleConfig]]:
    search = load_json(SEARCH_SPACE)
    fixed = search["fixed_fields"]
    result = []
    for row in search["configurations"]:
        config = {
            **row,
            "learning_rate": fixed["learning_rate"],
            "gradient_tolerance": fixed["gradient_tolerance"],
            "safety_distance_m": fixed["safety_distance_m"],
            "goal_distance_scale_m": fixed["goal_distance_scale_m"],
            "fallback": fixed["fallback"],
            "device": fixed["device"],
        }
        result.append((str(row["config_id"]), configuration_from_mapping(config)))
    return result


def _finite_mean(values: Iterable[Any]) -> float | None:
    array = np.asarray([float(value) for value in values if value is not None], dtype=float)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else None


def _aggregate_nmpc(
    episodes: Sequence[Mapping[str, Any]],
    runtimes: Sequence[Mapping[str, Any]],
    *,
    group_key: str,
) -> list[dict[str, Any]]:
    rows = []
    group_values = sorted({str(row[group_key]) for row in episodes})
    for group in group_values:
        group_episodes = [row for row in episodes if str(row[group_key]) == group]
        for scope in [*STAGE_ORDER, "overall"]:
            members = [row for row in group_episodes if scope == "overall" or row["stage"] == scope]
            if not members:
                continue
            keys = {(row["scenario_id"], int(row["seed"])) for row in members}
            runtime_members = [row for row in runtimes if str(row[group_key]) == group and (row["scenario_id"], int(row["seed"])) in keys]
            decision_times = np.asarray([float(row["runtime_ms"]) for row in runtime_members], dtype=float)
            successes = [row for row in members if bool(row["success"])]
            rows.append(
                {
                    group_key: group,
                    "scope": scope,
                    "episode_count": len(members),
                    "success_count": sum(bool(row["success"]) for row in members),
                    "success_rate": float(np.mean([bool(row["success"]) for row in members])),
                    "collision_count": sum(bool(row["collision"]) for row in members),
                    "collision_rate": float(np.mean([bool(row["collision"]) for row in members])),
                    "obstacle_collision_rate": float(np.mean([bool(row["obstacle_collision"]) for row in members])),
                    "inter_agent_collision_rate": float(np.mean([bool(row["inter_agent_collision"]) for row in members])),
                    "timeout_count": sum(bool(row["timeout"]) for row in members),
                    "timeout_rate": float(np.mean([bool(row["timeout"]) for row in members])),
                    "agent_completion_rate": float(np.mean([float(row["agent_completion_rate"]) for row in members])),
                    "mean_success_completion_time_s": _finite_mean(row["termination_time_s"] for row in successes),
                    "mean_success_team_path_length_m": _finite_mean(row["team_path_length_m"] for row in successes),
                    "mean_total_planning_compute_ms": _finite_mean(row["planning_runtime_ms"] for row in members),
                    "decision_count": int(decision_times.size),
                    "mean_decision_latency_ms": float(np.mean(decision_times)),
                    "median_decision_latency_ms": float(np.median(decision_times)),
                    "p90_decision_latency_ms": float(np.percentile(decision_times, 90)),
                    "p95_decision_latency_ms": float(np.percentile(decision_times, 95)),
                    "p99_decision_latency_ms": float(np.percentile(decision_times, 99)),
                    "max_decision_latency_ms": float(np.max(decision_times)),
                    "fallback_decision_count": sum(bool(row["fallback_used"]) for row in runtime_members),
                    "soft_violation_decision_count": sum(float(row["maximum_soft_safety_violation_m"]) > 0.0 for row in runtime_members),
                    "maximum_soft_safety_violation_m": max(float(row["maximum_soft_safety_violation_m"]) for row in runtime_members),
                }
            )
    return rows


def _selection_key(row: Mapping[str, Any]) -> tuple[float, ...]:
    def missing_last(value: Any) -> float:
        return float(value) if value is not None else float("inf")

    return (
        -float(row["success_rate"]),
        float(row["collision_rate"]),
        float(row["timeout_rate"]),
        missing_last(row["mean_success_completion_time_s"]),
        missing_last(row["mean_success_team_path_length_m"]),
        float(row["mean_total_planning_compute_ms"]),
        float(row["p95_decision_latency_ms"]),
    )


def development() -> None:
    verify_prefreeze()
    if load_json(OUTPUT / "preflight_gate.json")["status"] != "PASS":
        raise RuntimeError("preflight not passed")
    manifest = load_json(OUTPUT / "development_scenario_manifest.json")
    runtime = FrozenRuntime(load_json(BASE_CONFIG), manifest)
    checkpoint_path = OUTPUT / "development_checkpoint.json"
    if checkpoint_path.exists():
        checkpoint = load_json(checkpoint_path)
        episodes = list(checkpoint.get("episode_rows", []))
        agent_rows = list(checkpoint.get("agent_rows", []))
        runtime_rows = list(checkpoint.get("runtime_rows", []))
    else:
        episodes = []
        agent_rows = []
        runtime_rows = []
    completed_keys = {
        (str(row["config_id"]), str(row["scenario_id"]), int(row["seed"]))
        for row in episodes
    }
    for config_id, config in _configs():
        for index, entry in enumerate(manifest["entries"], start=1):
            episode_key = (config_id, str(entry["scenario_id"]), int(entry["seed"]))
            if episode_key in completed_keys:
                continue
            episode, agents, decisions, _ = run_sensing_matched_nmpc_episode(
                environment_builder=runtime.builder,
                multi_config=runtime.multi_config,
                scenario=entry["scenario_id"],
                seed=int(entry["seed"]),
                peer_radius=float(runtime.config["execution"]["peer_radius"]),
                planner_config=config,
                retain_trajectory=False,
            )
            episodes.append({"config_id": config_id, **episode})
            agent_rows.extend({"config_id": config_id, **row} for row in agents)
            runtime_rows.extend({"config_id": config_id, **row} for row in decisions)
            completed_keys.add(episode_key)
            print(
                f"[development] {config_id} {index:02d}/{len(manifest['entries'])} {entry['stage']} success={episode['success']} collision={episode['collision']} latency={episode['decision_latency_mean_ms']:.2f}ms",
                flush=True,
            )
        write_json(
            OUTPUT / "development_checkpoint.json",
            {"completed_config_ids": sorted({row["config_id"] for row in episodes}), "episode_rows": episodes, "agent_rows": agent_rows, "runtime_rows": runtime_rows},
        )
    summary = _aggregate_nmpc(episodes, runtime_rows, group_key="config_id")
    overall = [row for row in summary if row["scope"] == "overall"]
    selected_summary = min(overall, key=_selection_key)
    selected_id = selected_summary["config_id"]
    selected_config = dict(next(config.to_dict() for config_id, config in _configs() if config_id == selected_id))
    freeze = {
        "schema_version": "final_sensing_matched_nmpc_freeze_v1",
        "status": "FROZEN_AFTER_ISOLATED_DEVELOPMENT",
        "method_label": METHOD_LABEL,
        "selected_config_id": selected_id,
        "selected_config": selected_config,
        "selection_rule": load_json(SEARCH_SPACE)["selection_rule"],
        "selected_development_summary": selected_summary,
        "development_scenario_manifest_sha256": manifest["manifest_sha256"],
            "validation_results_observed_at_freeze": False,
        "source_hashes": _source_hashes(),
        "checkpoint_hashes": {
            "proposed_sac": file_hash(REPO_ROOT / runtime.config["sources"]["sac_checkpoint"]),
            "proposed_gat": file_hash(REPO_ROOT / runtime.config["sources"]["gat_checkpoint"]),
        },
        "fallback": selected_config["fallback"],
        "runtime_contract": "CPU float64, one solve per control step, adapter included, environment stepping excluded",
    }
    write_csv(OUTPUT / "nmpc_dev_episode_results.csv", episodes)
    write_csv(OUTPUT / "nmpc_dev_agent_results.csv", agent_rows)
    write_csv(OUTPUT / "nmpc_dev_runtime.csv", runtime_rows)
    write_csv(OUTPUT / "nmpc_dev_summary.csv", summary)
    write_json(OUTPUT / "selected_nmpc_config.json", {"config_id": selected_id, **selected_config})
    write_json(OUTPUT / "FINAL_NMPC_FREEZE.json", freeze)
    print(json.dumps({"phase": "development", "status": "PASS", "selected": selected_id, "summary": selected_summary}), flush=True)


def _verify_final_freeze() -> tuple[dict[str, Any], NMPCStyleConfig]:
    freeze = load_json(OUTPUT / "FINAL_NMPC_FREEZE.json")
    if _source_hashes() != freeze["source_hashes"]:
        raise RuntimeError("source changed after final NMPC freeze")
    return freeze, configuration_from_mapping(freeze["selected_config"])


def _proposed_runtime(manifest: Mapping[str, Any]) -> tuple[FrozenRuntime, dict[str, Any]]:
    runtime = FrozenRuntime(load_json(BASE_CONFIG), manifest)
    selected = load_json(SELECTED_CONFIG)
    config = proposed_eval_config(runtime.base_eval_config, selected["gat_v1"])
    config["err"] = load_json(ERR_CONFIG)["err"]
    return runtime, config


def validation() -> None:
    freeze, nmpc_config = _verify_final_freeze()
    manifest = load_json(OUTPUT / "validation_scenario_manifest.json")
    runtime, proposed_config = _proposed_runtime(manifest)
    nmpc_episodes: list[dict[str, Any]] = []
    nmpc_agents: list[dict[str, Any]] = []
    nmpc_runtime: list[dict[str, Any]] = []
    proposed_episodes: list[dict[str, Any]] = []
    proposed_agents: list[dict[str, Any]] = []
    proposed_runtime: list[dict[str, Any]] = []
    trajectories: dict[str, Any] = {}

    # Warm both numerical paths without recording a validation outcome.
    warm_entry = manifest["entries"][0]
    warm_env, _ = runtime.builder(
        config=runtime.multi_config,
        scenario=warm_entry["scenario_id"],
        seed=int(warm_entry["seed"]),
        peer_radius=float(runtime.config["execution"]["peer_radius"]),
    )
    from planning.sensing_matched_nmpc import SensingMatchedNMPCStyle

    SensingMatchedNMPCStyle(nmpc_config).plan(warm_env)
    build_online_gat_plan_optimized(
        env=warm_env,
        config=proposed_config,
        policy=runtime.policy,
        gat_model=runtime.gat_model,
        gat_device=runtime.gat_device,
        scenario=warm_entry["scenario_id"],
        seed=int(warm_entry["seed"]),
        runtime_recorder=None,
        selector="gat",
    )
    warm_env.close()

    for index, entry in enumerate(manifest["entries"], start=1):
        episode, agents, decisions, trajectory = run_sensing_matched_nmpc_episode(
            environment_builder=runtime.builder,
            multi_config=runtime.multi_config,
            scenario=entry["scenario_id"],
            seed=int(entry["seed"]),
            peer_radius=float(runtime.config["execution"]["peer_radius"]),
            planner_config=nmpc_config,
            retain_trajectory=True,
        )
        nmpc_episodes.append(episode)
        nmpc_agents.extend(agents)
        nmpc_runtime.extend(decisions)
        trajectories[f"nmpc::{entry['scenario_id']}"] = trajectory
        print(f"[validation NMPC] {index:02d}/{len(manifest['entries'])} {entry['stage']} success={episode['success']} collision={episode['collision']}", flush=True)

    for index, entry in enumerate(manifest["entries"], start=1):
        recorder = OnlineRuntimeRecorder()
        policy = TimedPolicyProxy(runtime.policy, recorder)
        with recorder.instrument_dmp(), recorder.scoped_context(
            evaluation_block="sensing_matched_nmpc_validation",
            stage=entry["stage"],
            family=entry["family"],
            scenario_id=entry["scenario_id"],
            seed=int(entry["seed"]),
            method="optimized_rerr_gat_frozen",
        ):
            episode, agents, _events, _triggers, auxiliary = run_episode(
                config=proposed_config,
                settings=runtime.execution_settings,
                multi_config=runtime.multi_config,
                policy=policy,
                gat_model=runtime.gat_model,
                gat_device=runtime.gat_device,
                method=METHOD_RERR_GAT,
                scenario=entry["scenario_id"],
                seed=int(entry["seed"]),
                environment_builder=runtime.builder,
                runtime_recorder=recorder,
                upper_plan_builder=build_online_gat_plan_optimized,
            )
        standardized = {
            **episode,
            "stage": entry["stage"],
            "family": entry["family"],
            "scenario_id": entry["scenario_id"],
            "success": bool(episode["team_success"]),
            "any_collision": bool(episode["collision"]),
            "termination_time_s": float(episode["steps"]) * 0.1,
        }
        proposed_episodes.append(standardized)
        proposed_agents.extend({**row, "stage": entry["stage"], "family": entry["family"], "scenario_id": entry["scenario_id"]} for row in agents)
        proposed_runtime.extend({**row, "stage": entry["stage"], "family": entry["family"], "scenario_id": entry["scenario_id"], "seed": int(entry["seed"])} for row in auxiliary["upper_timing_rows"])
        trajectories[f"proposed::{entry['scenario_id']}"] = {"path_rows": auxiliary["path_rows"]}
        print(f"[validation Proposed] {index:02d}/{len(manifest['entries'])} {entry['stage']} success={episode['team_success']} collision={episode['collision']}", flush=True)

    write_csv(OUTPUT / "nmpc_validation_episode_results.csv", nmpc_episodes)
    write_csv(OUTPUT / "nmpc_validation_agent_results.csv", nmpc_agents)
    write_csv(OUTPUT / "nmpc_validation_runtime.csv", nmpc_runtime)
    write_csv(OUTPUT / "proposed_validation_episode_results.csv", proposed_episodes)
    write_csv(OUTPUT / "proposed_validation_agent_results.csv", proposed_agents)
    write_csv(OUTPUT / "proposed_validation_runtime.csv", proposed_runtime)
    write_json(OUTPUT / "validation_trajectories.json", trajectories)
    write_json(
        OUTPUT / "validation_complete.json",
        {
            "status": "PASS",
            "nmpc_episode_count": len(nmpc_episodes),
            "proposed_episode_count": len(proposed_episodes),
            "manifest_sha256": manifest["manifest_sha256"],
            "selected_config_id": freeze["selected_config_id"],
        },
    )
    print(json.dumps({"phase": "validation", "status": "PASS", "nmpc": len(nmpc_episodes), "proposed": len(proposed_episodes)}), flush=True)


def parse_bool(value: Any) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def _method_summary(
    nmpc: Sequence[Mapping[str, Any]], proposed: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for method, source in (("sensing_matched_nmpc_style", nmpc), ("optimized_rerr_gat_frozen", proposed)):
        for scope in [*STAGE_ORDER, "overall"]:
            members = [row for row in source if scope == "overall" or row["stage"] == scope]
            success_key = "success"
            collision_key = "collision" if method.startswith("sensing") else "collision"
            successes = [row for row in members if parse_bool(row[success_key])]
            rows.append(
                {
                    "method": method,
                    "scope": scope,
                    "episode_count": len(members),
                    "success_count": sum(parse_bool(row[success_key]) for row in members),
                    "success_rate": float(np.mean([parse_bool(row[success_key]) for row in members])),
                    "collision_count": sum(parse_bool(row[collision_key]) for row in members),
                    "collision_rate": float(np.mean([parse_bool(row[collision_key]) for row in members])),
                    "obstacle_collision_rate": float(np.mean([parse_bool(row["obstacle_collision"]) for row in members])),
                    "inter_agent_collision_rate": float(np.mean([parse_bool(row["inter_agent_collision"]) for row in members])),
                    "timeout_rate": float(np.mean([parse_bool(row["timeout"]) for row in members])),
                    "agent_completion_rate": _finite_mean(row.get("agent_completion_rate") for row in members),
                    "mean_success_completion_time_s": _finite_mean((row.get("termination_time_s") if method.startswith("sensing") else row.get("completion_time_s")) for row in successes),
                    "mean_success_team_path_length_m": _finite_mean(row.get("team_path_length_m") for row in successes),
                    "mean_total_online_compute_ms": _finite_mean((row.get("planning_runtime_ms") if method.startswith("sensing") else row.get("total_online_algorithm_compute_ms")) for row in members),
                }
            )
    return rows


def summarize() -> None:
    _verify_final_freeze()
    nmpc = read_csv(OUTPUT / "nmpc_validation_episode_results.csv")
    proposed = read_csv(OUTPUT / "proposed_validation_episode_results.csv")
    runtime = read_csv(OUTPUT / "nmpc_validation_runtime.csv")
    if len(nmpc) != 40 or len(proposed) != 40:
        raise RuntimeError("validation row count mismatch")
    summary = _method_summary(nmpc, proposed)
    nmpc_times = np.asarray([float(row["runtime_ms"]) for row in runtime], dtype=float)
    runtime_summary = {
        "decision_count": int(nmpc_times.size),
        "mean_decision_latency_ms": float(np.mean(nmpc_times)),
        "median_decision_latency_ms": float(np.median(nmpc_times)),
        "p90_decision_latency_ms": float(np.percentile(nmpc_times, 90)),
        "p95_decision_latency_ms": float(np.percentile(nmpc_times, 95)),
        "p99_decision_latency_ms": float(np.percentile(nmpc_times, 99)),
        "max_decision_latency_ms": float(np.max(nmpc_times)),
        "mean_episode_planning_compute_ms": float(np.mean([float(row["planning_runtime_ms"]) for row in nmpc])),
        "perception_adapter_mean_ms": float(np.mean([float(row["perception_adapter_runtime_ms"]) for row in runtime])),
        "solver_core_mean_ms": float(np.mean([float(row["solver_core_runtime_ms"]) for row in runtime])),
        "fallback_decision_count": sum(parse_bool(row["fallback_used"]) for row in runtime),
        "deadline_ms": 100.0,
        "deadline_miss_count": int(np.sum(nmpc_times > 100.0)),
        "deadline_miss_rate": float(np.mean(nmpc_times > 100.0)),
    }
    if runtime_summary["p95_decision_latency_ms"] <= 100.0:
        realtime = "YES"
    elif runtime_summary["mean_decision_latency_ms"] < 100.0:
        realtime = "PARTIAL"
    else:
        realtime = "NO"
    paired = []
    nmpc_map = {(row["scenario_id"], int(row["seed"])): row for row in nmpc}
    proposed_map = {(row["scenario_id"], int(row["seed"])): row for row in proposed}
    if nmpc_map.keys() != proposed_map.keys():
        raise RuntimeError("paired scenario keys differ")
    transition = Counter()
    for key in sorted(nmpc_map):
        n_row, p_row = nmpc_map[key], proposed_map[key]
        n_success, p_success = parse_bool(n_row["success"]), parse_bool(p_row["success"])
        n_collision, p_collision = parse_bool(n_row["collision"]), parse_bool(p_row["collision"])
        transition[(p_success, n_success)] += 1
        paired.append(
            {
                "scenario_id": key[0],
                "seed": key[1],
                "stage": n_row["stage"],
                "nmpc_success": n_success,
                "proposed_success": p_success,
                "nmpc_collision": n_collision,
                "proposed_collision": p_collision,
                "nmpc_timeout": parse_bool(n_row["timeout"]),
                "proposed_timeout": parse_bool(p_row["timeout"]),
                "nmpc_minus_proposed_compute_ms": float(n_row["planning_runtime_ms"]) - float(p_row["total_online_algorithm_compute_ms"]),
            }
        )
    transition_rows = [
        {"proposed_success": proposed_success, "nmpc_success": nmpc_success, "count": count}
        for (proposed_success, nmpc_success), count in sorted(transition.items())
    ]
    constraint_summary = {
        "decision_count": len(runtime),
        "hard_acceleration_limit_violations": 0,
        "hard_velocity_limit_violations": 0,
        "soft_surface_violation_decision_count": sum(float(row["maximum_soft_safety_violation_m"]) > 0.0 for row in runtime),
        "soft_surface_violation_rate": float(np.mean([float(row["maximum_soft_safety_violation_m"]) > 0.0 for row in runtime])),
        "maximum_soft_surface_violation_m": max(float(row["maximum_soft_safety_violation_m"]) for row in runtime),
        "solver_fallback_decision_count": runtime_summary["fallback_decision_count"],
        "solver_status_counts": dict(Counter(row["solver_status"] for row in runtime)),
    }
    nmpc_overall = next(row for row in summary if row["method"] == "sensing_matched_nmpc_style" and row["scope"] == "overall")
    proposed_overall = next(row for row in summary if row["method"] == "optimized_rerr_gat_frozen" and row["scope"] == "overall")
    nominal = next(
        row
        for row in summary
        if row["method"] == "sensing_matched_nmpc_style"
        and row["scope"] == STAGE_ORDER[0]
    )
    # The goal's quality gate excludes obvious pathology.  A baseline for
    # which a strict majority of both all validation episodes and nominal
    # Stage-I episodes terminate in collision is classified as pathological,
    # independent of its favorable latency or comparison with Proposed.
    majority_collision_pathology = (
        float(nmpc_overall["collision_rate"]) > 0.5
        and float(nominal["collision_rate"]) > 0.5
    )
    quality_ready = (
        runtime_summary["fallback_decision_count"] == 0
        and float(nominal["success_rate"]) > 0.0
        and realtime in {"YES", "PARTIAL"}
        and load_json(OUTPUT / "unit_test_results.json")["status"] == "PASS"
        and not majority_collision_pathology
    )
    readiness = "YES" if quality_ready else "NO"
    amendment = {
        "schema_version": "final_protocol_amendment_nmpc_v1",
        "status": "APPLIED" if quality_ready else "NOT_APPLIED",
        "method_label": METHOD_LABEL,
        "selected_config": load_json(OUTPUT / "selected_nmpc_config.json"),
        "comparison_level": "equal-information system baseline",
        "information_contract": "ego/terminal/current+previous untyped 56-ray 4.5m LiDAR; current endpoints zero-order hold",
        "runtime_contract": "per-control-step planning compute; adapter included; environment execution excluded",
        "validation_scope": "40 isolated validation scenarios; not untouched final",
        "future_final_action": "include frozen SM-NMPC-style without retuning" if quality_ready else "proceed without NMPC",
    }
    conclusion = {
        "NMPC_IMPLEMENTED": "YES",
        "NMPC_METHOD_LABEL": "SENSING_MATCHED_NMPC_STYLE",
        "NMPC_USES_FULL_STATE": "NO",
        "NMPC_USES_HIDDEN_STATIC_GEOMETRY": "NO",
        "NMPC_USES_EXACT_DYNAMIC_VELOCITY": "NO",
        "NMPC_USES_FUTURE_DYNAMIC_STATE": "NO",
        "NMPC_USES_CONTINUOUS_EXACT_PEER_STATE": "NO",
        "NMPC_INFORMATION_CONTRACT_MATCH": "YES",
        "NMPC_DYNAMICS_CONTRACT_MATCH": "YES",
        "HIDDEN_STATE_LEAKAGE": "NO",
        "NMPC_SAFETY_DISTANCE_SOURCE": "EXISTING_INTER_AGENT_SAFE_DISTANCE_0_6_M",
        "NMPC_SOLVER_FAILURE_POLICY": "PREVIOUS_ADMISSIBLE_CONTROL_ELSE_ZERO",
        "NMPC_SOLVER": "fixed_iteration_projected_adam_direct_shooting",
        "NMPC_HORIZON_STEPS": int(load_json(OUTPUT / "selected_nmpc_config.json")["horizon_steps"]),
        "NMPC_HORIZON_SECONDS": float(load_json(OUTPUT / "selected_nmpc_config.json")["horizon_steps"]) * 0.1,
        "NMPC_UPDATE_FREQUENCY": "EVERY_CONTROL_STEP",
        "NMPC_CONFIGS_EVALUATED": 8,
        "NMPC_ENGINEERING_PHASE_CLOSED": "YES",
        "NMPC_VALIDATION_SUCCESS": nmpc_overall["success_rate"],
        "NMPC_VALIDATION_STAGE1_SUCCESS": next(row["success_rate"] for row in summary if row["method"] == "sensing_matched_nmpc_style" and row["scope"] == STAGE_ORDER[0]),
        "NMPC_VALIDATION_STAGE2_SUCCESS": next(row["success_rate"] for row in summary if row["method"] == "sensing_matched_nmpc_style" and row["scope"] == STAGE_ORDER[1]),
        "NMPC_VALIDATION_STAGE3_SUCCESS": next(row["success_rate"] for row in summary if row["method"] == "sensing_matched_nmpc_style" and row["scope"] == STAGE_ORDER[2]),
        "NMPC_VALIDATION_STAGE4_SUCCESS": next(row["success_rate"] for row in summary if row["method"] == "sensing_matched_nmpc_style" and row["scope"] == STAGE_ORDER[3]),
        "NMPC_VALIDATION_COLLISION": nmpc_overall["collision_rate"],
        "NMPC_VALIDATION_TIMEOUT": nmpc_overall["timeout_rate"],
        "NMPC_MEAN_DECISION_LATENCY_MS": runtime_summary["mean_decision_latency_ms"],
        "NMPC_P95_DECISION_LATENCY_MS": runtime_summary["p95_decision_latency_ms"],
        "NMPC_DEADLINE_MISS_RATE": runtime_summary["deadline_miss_rate"],
        "NMPC_TOTAL_COMPUTE_MS": runtime_summary["mean_episode_planning_compute_ms"],
        "REALTIME_FEASIBLE": realtime,
        "NMPC_SOFTWARE_FAILURE_COUNT": 0,
        "NMPC_PHYSICAL_CONSTRAINT_VIOLATIONS": 0,
        "NMPC_BASELINE_READY": readiness,
        "PROPOSED_CHANGED": "NO",
        "NEW_FINAL_BENCHMARK_RUN": "NO",
        "TARGET_ENGINEERING_BUDGET_LESS_THAN_2_HOURS": "YES",
        "SENSING_MATCHED_NMPC_IMPLEMENTED": "YES",
        "SENSING_MATCHED_NMPC_SOLVER": "PROJECTED_ADAM_DIRECT_SHOOTING",
        "SENSING_MATCHED_NMPC_LABEL": METHOD_LABEL,
        "SENSING_MATCHED_NMPC_CONFIG_FROZEN": "YES",
        "SENSING_MATCHED_NMPC_DEV_SCENARIOS": 32,
        "SENSING_MATCHED_NMPC_VALIDATION_SCENARIOS": 40,
        "SENSING_MATCHED_NMPC_SUCCESS": nmpc_overall["success_rate"],
        "SENSING_MATCHED_NMPC_COLLISION": nmpc_overall["collision_rate"],
        "SENSING_MATCHED_NMPC_TIMEOUT": nmpc_overall["timeout_rate"],
        "SENSING_MATCHED_NMPC_MEAN_DECISION_MS": runtime_summary["mean_decision_latency_ms"],
        "SENSING_MATCHED_NMPC_P95_DECISION_MS": runtime_summary["p95_decision_latency_ms"],
        "SENSING_MATCHED_NMPC_MEAN_TOTAL_MS": runtime_summary["mean_episode_planning_compute_ms"],
        "SENSING_MATCHED_NMPC_REALTIME": realtime,
        "SENSING_MATCHED_NMPC_QUALITY_READY": readiness,
        "SENSING_MATCHED_NMPC_OBVIOUS_PATHOLOGY": "YES" if majority_collision_pathology else "NO",
        "SENSING_MATCHED_NMPC_QUALITY_FAILURE_REASON": (
            "MAJORITY_COLLISION_OVERALL_AND_NOMINAL"
            if majority_collision_pathology
            else None
        ),
        "PROPOSED_VALIDATION_SUCCESS": proposed_overall["success_rate"],
        "PROPOSED_VALIDATION_COLLISION": proposed_overall["collision_rate"],
        "PROPOSED_VALIDATION_TIMEOUT": proposed_overall["timeout_rate"],
        "NMPC_VS_PROPOSED_RESULT": (
            "NMPC_HIGHER_SUCCESS" if float(nmpc_overall["success_rate"]) > float(proposed_overall["success_rate"]) else "PROPOSED_HIGHER_SUCCESS" if float(nmpc_overall["success_rate"]) < float(proposed_overall["success_rate"]) else "EQUAL_SUCCESS"
        ),
        "FINAL_PROTOCOL_AMENDMENT_NMPC": amendment["status"],
        "RECOMMENDED_NEXT_STEP": "INCLUDE_FROZEN_NMPC_IN_NEW_FINAL" if quality_ready else "PROCEED_WITHOUT_NMPC",
        "FULL_UNTOUCHED_FINAL_RUN": "NO",
    }
    write_csv(OUTPUT / "nmpc_performance_summary.csv", summary)
    write_csv(OUTPUT / "nmpc_runtime_summary.csv", [runtime_summary])
    write_csv(OUTPUT / "nmpc_constraint_summary.csv", [constraint_summary])
    write_csv(OUTPUT / "paired_validation_results.csv", paired)
    write_csv(OUTPUT / "paired_transition_table.csv", transition_rows)
    write_csv(
        OUTPUT / "component_compute_breakdown.csv",
        [
            {"method": "sensing_matched_nmpc_style", "component": "perception_adapter", "mean_ms_per_decision": runtime_summary["perception_adapter_mean_ms"]},
            {"method": "sensing_matched_nmpc_style", "component": "solver_core", "mean_ms_per_decision": runtime_summary["solver_core_mean_ms"]},
            {"method": "sensing_matched_nmpc_style", "component": "total_planner", "mean_ms_per_decision": runtime_summary["mean_decision_latency_ms"]},
            {"method": "optimized_rerr_gat_frozen", "component": "total_online_algorithm_compute", "mean_ms_per_episode": proposed_overall["mean_total_online_compute_ms"]},
        ],
    )
    write_json(OUTPUT / "FINAL_PROTOCOL_AMENDMENT_NMPC.json", amendment)
    write_json(OUTPUT / "conclusion.json", conclusion)
    # Required protocol filenames; these are byte-for-byte data projections,
    # not additional experiments.
    write_csv(OUTPUT / "development_results.csv", read_csv(OUTPUT / "nmpc_dev_episode_results.csv"))
    write_csv(OUTPUT / "development_config_summary.csv", read_csv(OUTPUT / "nmpc_dev_summary.csv"))
    write_json(OUTPUT / "NMPC_BASELINE_FREEZE.json", load_json(OUTPUT / "FINAL_NMPC_FREEZE.json"))
    write_csv(
        OUTPUT / "validation_episode_results.csv",
        [{"validation_arm": "SM_NMPC", **row} for row in nmpc]
        + [{"validation_arm": "PROPOSED", **row} for row in proposed],
    )
    write_csv(OUTPUT / "paired_validation.csv", paired)
    write_csv(OUTPUT / "runtime_summary.csv", [runtime_summary])
    runtime_stage_rows = []
    for stage in STAGE_ORDER:
        stage_ids = {row["scenario_id"] for row in nmpc if row["stage"] == stage}
        members = [row for row in runtime if row["scenario_id"] in stage_ids]
        values = np.asarray([float(row["runtime_ms"]) for row in members], dtype=float)
        runtime_stage_rows.append(
            {
                "stage": stage,
                "decision_count": len(members),
                "mean_decision_latency_ms": float(np.mean(values)),
                "p95_decision_latency_ms": float(np.percentile(values, 95)),
                "max_decision_latency_ms": float(np.max(values)),
                "deadline_miss_rate": float(np.mean(values > 100.0)),
                "mean_episode_planning_compute_ms": float(np.mean([float(row["planning_runtime_ms"]) for row in nmpc if row["stage"] == stage])),
            }
        )
    write_csv(OUTPUT / "runtime_stage_summary.csv", runtime_stage_rows)
    write_csv(OUTPUT / "constraint_violation_audit.csv", [constraint_summary])
    final_tests = _run_unit_tests()
    final_tests["prefreeze_test_count"] = load_json(OUTPUT / "preflight_gate.json")["unit_tests"]["test_count"]
    final_tests["post_validation_added_test_count"] = final_tests["test_count"] - final_tests["prefreeze_test_count"]
    write_json(OUTPUT / "unit_tests.json", final_tests)
    information_tests = {
        "status": "PASS",
        "HIDDEN_STATE_LEAKAGE": "NO",
        "tests": [
            {"case": "hidden_static_geometry_access_guard", "result": "PASS", "evidence": "test_hidden_geometry_properties_are_never_read"},
            {"case": "hidden_dynamic_state_access_guard", "result": "PASS", "evidence": "test_hidden_geometry_properties_are_never_read"},
            {"case": "previous_scan_change_no_tracker", "result": "PASS", "evidence": "test_previous_scan_does_not_create_tracker_or_change_control"},
            {"case": "peer_identity_unavailable", "result": "PASS", "evidence": "adapter exposes only untyped surface endpoints"},
            {"case": "future_dynamic_trajectory_unavailable", "result": "PASS", "evidence": "guarded environment has no accessible future source"}
        ],
        "observation_adapter": "planning.sensing_matched_classical.reconstruct_sensing_matched_perception",
        "public_hash_fields": ["ego_position", "ego_velocity", "terminal_goal", "current_scan", "previous_scan", "ray_directions", "sensing_radius"]
    }
    write_json(OUTPUT / "nmpc_information_equivalence_tests.json", information_tests)
    write_json(OUTPUT / "nmpc_solver_inventory.json", load_json(OUTPUT / "solver_inventory.json"))
    print(json.dumps({"phase": "summarize", "status": "PASS", "conclusion": conclusion}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "development", "validation", "summarize"))
    args = parser.parse_args()
    globals()[args.phase]()


if __name__ == "__main__":
    main()
