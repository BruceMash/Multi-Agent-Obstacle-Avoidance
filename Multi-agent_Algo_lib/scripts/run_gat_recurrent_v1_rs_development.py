#!/usr/bin/env python3
"""Run the remaining frozen GAT-V1 and GAT-RS Dev selector arms."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping

import pandas as _pandas  # noqa: F401  # Windows torch initialization order


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
for search_path in (REPO_ROOT, ALGO_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from planning.online_runtime_instrumentation import OnlineRuntimeRecorder, TimedPolicyProxy  # noqa: E402
from scripts.evaluate_gat_v1_err_development import (  # noqa: E402
    METHOD_RERR_GAT,
    build_online_gat_plan_optimized,
    run_episode,
)
from scripts.run_gat_recurrent_r_development import (  # noqa: E402
    FileBackedBuilder,
    collect,
    finalize,
    load_json,
    save_record,
    sha256_file,
    summarize_record,
    write_json,
)
from scripts.run_long_range_development import DevelopmentRuntime  # noqa: E402


SCHEMA_VERSION = "gat_recurrent_v1_rs_development_v1"
DEFAULT_CONFIG = REPO_ROOT / "configs/evaluation/gat_recurrent_v1_rs_dev.json"
SOURCE_PATHS = (
    "Environment/frozen_sac_dmp_execution.py",
    "Environment/multi_agent_dmp_env.py",
    "planning/event_triggered_reference_reconstruction.py",
    "planning/heterogeneous_candidate_graph.py",
    "planning/policy_preview.py",
    "planning/pre_gat_closed_loop.py",
    "Multi-agent_Algo_lib/scripts/evaluate_gat_v1_err_development.py",
    "Multi-agent_Algo_lib/scripts/run_long_range_development.py",
    "Multi-agent_Algo_lib/scripts/run_gat_recurrent_r_development.py",
    "Multi-agent_Algo_lib/scripts/run_gat_recurrent_v1_rs_development.py",
)


def artifact_root(config: Mapping[str, Any]) -> Path:
    return (REPO_ROOT / str(config["artifact_root"])).resolve()


def manifest_path(config: Mapping[str, Any]) -> Path:
    return artifact_root(config) / str(config["development_manifest"])


def output_dir(config: Mapping[str, Any], selector: str) -> Path:
    return artifact_root(config) / str(config["selectors"][selector]["output_subdir"])


def resolve_runtime_config(config: Mapping[str, Any], selector: str) -> dict[str, Any]:
    base = load_json(REPO_ROOT / str(config["base_method_config"]))
    selected = config["selectors"][selector]
    base.update(
        {
            "artifact_root": str(config["artifact_root"]),
            "development_manifest": str(config["development_manifest"]),
            "output_subdir": str(selected["output_subdir"]),
            "stage1_config": str(selected["stage1_config"]),
            "gat_checkpoint": str(selected["checkpoint"]),
            "gat_checkpoint_sha256_expected": str(selected["checkpoint_sha256_expected"]),
            "method": "gat_v1_rerr",
            "configuration_id": f"GATRS_DEV_{selector.upper()}",
            "development_scenario_ids": None,
        }
    )
    base["graph"] = copy.deepcopy(base["graph"])
    base["graph"]["task_goal_distance_scale_m"] = float(selected["goal_distance_scale_m"])
    return base


def prepare(config_path: Path) -> None:
    config = load_json(config_path)
    if config["schema_version"] != SCHEMA_VERSION:
        raise RuntimeError("unexpected V1/RS development config schema")
    root = artifact_root(config)
    index_path = manifest_path(config)
    index = load_json(index_path)
    if len(index["entries"]) != int(config["expected_scenario_count"]):
        raise RuntimeError("Dev index must contain exactly 400 scenarios")
    scene_checks = []
    for row in index["entries"]:
        scene_path = root / str(row["scenario_file"])
        scene_checks.append(sha256_file(scene_path) == str(row["scenario_file_sha256"]))
    if not all(scene_checks):
        raise RuntimeError("one or more Dev scene hashes failed")
    completed_arms: dict[str, Any] = {}
    for selector, relative in config["completed_arms"].items():
        reconciliation_path = root / str(relative)
        reconciliation = load_json(reconciliation_path)
        if reconciliation["status"] != "PASS" or int(reconciliation["completed_scenarios"]) != 400:
            raise RuntimeError(f"completed arm did not reconcile: {selector}")
        completed_arms[selector] = {
            "reconciliation": str(relative),
            "reconciliation_sha256": sha256_file(reconciliation_path),
            "status": "PASS",
        }
    selectors: dict[str, Any] = {}
    for selector, selected in config["selectors"].items():
        checkpoint = REPO_ROOT / str(selected["checkpoint"])
        observed_hash = sha256_file(checkpoint)
        if observed_hash != str(selected["checkpoint_sha256_expected"]):
            raise RuntimeError(f"checkpoint hash mismatch: {selector}")
        runtime_config = resolve_runtime_config(config, selector)
        output = output_dir(config, selector)
        output.mkdir(parents=True, exist_ok=True)
        selectors[selector] = {
            "display_name": selected["display_name"],
            "checkpoint": str(selected["checkpoint"]),
            "checkpoint_sha256": observed_hash,
            "goal_distance_scale_m": float(selected["goal_distance_scale_m"]),
            "output": str(output.relative_to(REPO_ROOT).as_posix()),
            "resolved_runtime_config": runtime_config,
        }
    gat_v1_runtime = selectors["gat_v1"]["resolved_runtime_config"]
    gat_rs_runtime = selectors["gat_rs"]["resolved_runtime_config"]
    fairness_exclusions = {"stage1_config", "gat_checkpoint", "gat_checkpoint_sha256_expected", "configuration_id", "output_subdir"}
    v1_fair = {key: value for key, value in gat_v1_runtime.items() if key not in fairness_exclusions}
    rs_fair = {key: value for key, value in gat_rs_runtime.items() if key not in fairness_exclusions}
    v1_graph = copy.deepcopy(v1_fair["graph"])
    rs_graph = copy.deepcopy(rs_fair["graph"])
    v1_scale = v1_graph.pop("task_goal_distance_scale_m")
    rs_scale = rs_graph.pop("task_goal_distance_scale_m")
    v1_fair["graph"] = v1_graph
    rs_fair["graph"] = rs_graph
    if v1_fair != rs_fair:
        raise RuntimeError("V1/RS runtime configs differ beyond selector contract")
    freeze = {
        "schema_version": SCHEMA_VERSION,
        "status": "FROZEN_BEFORE_REMAINING_DEV_PERFORMANCE",
        "config_sha256": sha256_file(config_path),
        "development_index_sha256": sha256_file(index_path),
        "development_index_semantic_sha256": index["manifest_semantic_sha256"],
        "scene_count": len(index["entries"]),
        "scene_file_hashes_verified": sum(scene_checks),
        "source_sha256": {path: sha256_file(REPO_ROOT / path) for path in SOURCE_PATHS},
        "selectors": selectors,
        "completed_arms": completed_arms,
        "paired_scene_contract": True,
        "runtime_contract_equal_except_checkpoint_and_frozen_goal_scale": True,
        "gat_v1_goal_distance_scale_m": v1_scale,
        "gat_rs_goal_distance_scale_m": rs_scale,
        "canonical_gat_projection_directions": 56,
        "runtime_future_information_added": False,
        "formal_v1_used": False,
        "holdout_opened": False,
        "formal_v2_generated": False,
    }
    write_json(root / "07_development/GAT_V1_RS_DEV_PREFLIGHT_FREEZE.json", freeze)
    print(json.dumps({"status": "PASS", "scene_count": len(scene_checks)}, indent=2))


def verify(config_path: Path, config: Mapping[str, Any]) -> None:
    freeze = load_json(artifact_root(config) / "07_development/GAT_V1_RS_DEV_PREFLIGHT_FREEZE.json")
    checks = {
        "config": sha256_file(config_path) == freeze["config_sha256"],
        "index": sha256_file(manifest_path(config)) == freeze["development_index_sha256"],
        "sources": all(
            sha256_file(REPO_ROOT / path) == expected
            for path, expected in freeze["source_sha256"].items()
        ),
        "checkpoints": all(
            sha256_file(REPO_ROOT / selected["checkpoint"]) == selected["checkpoint_sha256"]
            for selected in freeze["selectors"].values()
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"V1/RS Dev freeze mismatch: {checks}")


def run(
    config_path: Path,
    selector: str,
    *,
    shard_index: int | None,
    shard_count: int | None,
    limit: int | None,
    defer_finalize: bool,
) -> None:
    config = load_json(config_path)
    verify(config_path, config)
    index = load_json(manifest_path(config))
    runtime_config = resolve_runtime_config(config, selector)
    runtime = DevelopmentRuntime(runtime_config, {"entries": []})
    runtime.builder = FileBackedBuilder(index, runtime_config, artifact_root(config))
    output = output_dir(config, selector)
    completed = {str(row["scenario_id"]) for row in collect(output)}
    indexed = list(enumerate(index["entries"]))
    if shard_count is not None:
        if shard_index is None or not 0 <= int(shard_index) < int(shard_count):
            raise ValueError("invalid shard index/count")
        indexed = [(index_value, row) for index_value, row in indexed if index_value % int(shard_count) == int(shard_index)]
    pending = [row for _, row in indexed if str(row["scenario_id"]) not in completed]
    if limit is not None:
        pending = pending[: int(limit)]
    for entry in pending:
        scenario_id = str(entry["scenario_id"])
        print(f"[GAT-V1-RS-Dev:{selector}] start {scenario_id}", flush=True)
        recorder = OnlineRuntimeRecorder()
        policy = TimedPolicyProxy(runtime.policy, recorder)
        try:
            with recorder.instrument_dmp(), recorder.scoped_context(
                evaluation_block="gat_recurrent_four_selector_development",
                configuration_id=runtime_config["configuration_id"],
                stage=entry["stage"],
                family=entry["family"],
                scenario_id=scenario_id,
                seed=int(entry["seed"]),
                method=selector,
            ):
                episode, agents, events, triggers, extra = run_episode(
                    config=runtime.eval_config,
                    settings=runtime.settings,
                    multi_config=runtime.multi_config,
                    policy=policy,
                    gat_model=runtime.gat_model,
                    gat_device=runtime.gat_device,
                    method=METHOD_RERR_GAT,
                    scenario=scenario_id,
                    seed=int(entry["seed"]),
                    environment_builder=runtime.builder,
                    runtime_recorder=recorder,
                    upper_plan_builder=build_online_gat_plan_optimized,
                )
            summary = summarize_record(entry, episode, events, triggers, extra["path_rows"])
            summary.update(
                {
                    "configuration_id": runtime_config["configuration_id"],
                    "selector": selector,
                    "method": config["selectors"][selector]["display_name"],
                    "team_success": bool(episode["team_success"]),
                    "collision": bool(episode["collision"]),
                    "timeout": bool(episode["timeout"]),
                    "performance_used_for_selection": True,
                }
            )
            save_record(output, entry, episode, agents, events, triggers, extra, summary)
            print(
                f"[GAT-V1-RS-Dev:{selector}] complete {scenario_id} "
                f"success={int(summary['team_success'])} collision={int(summary['collision'])}",
                flush=True,
            )
        except Exception as error:
            write_json(
                output / "episode_records" / f"{scenario_id}_SOFTWARE_ERROR.json",
                {
                    "scenario_id": scenario_id,
                    "seed": int(entry["seed"]),
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
            raise
    if not defer_finalize:
        finalize(config, selector)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "run", "finalize"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--selector", choices=("gat_v1", "gat_rs"))
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--defer-finalize", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve() if args.config.is_absolute() else (REPO_ROOT / args.config).resolve()
    if args.phase == "prepare":
        prepare(config_path)
    elif args.phase == "run":
        if args.selector is None:
            raise ValueError("--selector is required for run")
        run(
            config_path,
            args.selector,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            limit=args.limit,
            defer_finalize=args.defer_finalize,
        )
    else:
        if args.selector is None:
            raise ValueError("--selector is required for finalize")
        finalize(load_json(config_path), args.selector)


if __name__ == "__main__":
    main()
