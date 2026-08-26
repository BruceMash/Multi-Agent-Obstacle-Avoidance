#!/usr/bin/env python3
"""Paired file-backed evaluation for R-ERR+FP-SHEP and R-ERR+GAT-R.

The default block is development.  A frozen configuration may set
``evaluation_block`` to ``holdout`` so that the same execution contract is
used without relabelling holdout performance as development performance.
"""

from __future__ import annotations

import argparse
import copy
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

from planning.online_runtime_instrumentation import OnlineRuntimeRecorder, TimedPolicyProxy  # noqa: E402
from planning.semi_structured_long_range_benchmark import STAGE_ORDER, json_ready  # noqa: E402
from scripts.evaluate_gat_v1_err_development import (  # noqa: E402
    METHOD_RERR_FP_SHEP,
    METHOD_RERR_GAT,
    build_online_gat_plan_optimized,
    run_episode,
)
from scripts.run_long_range_contract_pilot import (  # noqa: E402
    LongRangeManifestEnvironmentBuilder,
    save_record,
    summarize_record,
)
from scripts.run_long_range_development import DevelopmentRuntime  # noqa: E402


SCHEMA_VERSION = "gat_recurrent_r_development_v1"
DEFAULT_CONFIG = REPO_ROOT / "configs/evaluation/gat_recurrent_r_dev.json"
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
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
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
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["status"], extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(json_ready(value), ensure_ascii=False)
                    if isinstance(value, (dict, list, tuple, np.ndarray))
                    else value
                    for key, value in row.items()
                }
            )


def artifact_root(config: Mapping[str, Any]) -> Path:
    return (REPO_ROOT / str(config["artifact_root"])).resolve()


def manifest_path(config: Mapping[str, Any]) -> Path:
    return artifact_root(config) / str(config["development_manifest"])


def output_dir(config: Mapping[str, Any], selector: str) -> Path:
    return artifact_root(config) / str(config["selectors"][selector]["output_subdir"])


def preflight_path(config: Mapping[str, Any]) -> Path:
    return artifact_root(config) / str(
        config.get(
            "preflight_path",
            "07_development/GAT_R_VS_FP_PREFLIGHT_FREEZE.json",
        )
    )


def evaluation_block(config: Mapping[str, Any]) -> str:
    block = str(config.get("evaluation_block", "development"))
    if block not in {"development", "holdout", "formal_v2"}:
        raise ValueError(f"unsupported evaluation block: {block}")
    return block


def result_prefix(config: Mapping[str, Any]) -> str:
    return {
        "development": "development",
        "holdout": "holdout",
        "formal_v2": "formal_v2",
    }[evaluation_block(config)]


def resolve_runtime_config(config: Mapping[str, Any], selector: str) -> dict[str, Any]:
    base = load_json(REPO_ROOT / str(config["base_method_config"]))
    selected = config["selectors"][selector]
    base.update(
        {
            "artifact_root": str(config["artifact_root"]),
            "development_manifest": str(config["development_manifest"]),
            "output_subdir": str(selected["output_subdir"]),
            "stage1_config": str(config["gat_r_training_config"]),
            "gat_checkpoint": str(config["gat_r_checkpoint"]),
            "gat_checkpoint_sha256_expected": str(config["gat_r_checkpoint_sha256_expected"]),
            "method": str(selected["runtime_method"]),
            "configuration_id": f"GATRS_{evaluation_block(config).upper()}_{selector.upper()}",
            "development_scenario_ids": None,
        }
    )
    base["graph"] = copy.deepcopy(base["graph"])
    base["graph"]["task_goal_distance_scale_m"] = float(
        config["goal_distance_normalization"]["gat_r_scale_m"]
    )
    return base


class FileBackedBuilder:
    def __init__(self, index: Mapping[str, Any], runtime_config: Mapping[str, Any], root: Path) -> None:
        self.entries = {str(row["scenario_id"]): row for row in index["entries"]}
        self.runtime_config = runtime_config
        self.root = root

    def __call__(self, *, config: Any, scenario: str, seed: int, peer_radius: float) -> tuple[Any, dict[str, Any]]:
        index_row = self.entries[str(scenario)]
        scene_path = self.root / str(index_row["scenario_file"])
        if sha256_file(scene_path) != str(index_row["scenario_file_sha256"]):
            raise RuntimeError(f"scene file SHA mismatch: {scenario}")
        scene = load_json(scene_path)
        if str(scene["scenario_id"]) != str(scenario) or int(scene["seed"]) != int(seed):
            raise RuntimeError("file-backed scene identity mismatch")
        builder = LongRangeManifestEnvironmentBuilder({"entries": [scene]}, self.runtime_config)
        return builder(config=config, scenario=scenario, seed=seed, peer_radius=peer_radius)


def collect(output: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((output / "episode_records").glob("*.json")):
        if path.name.endswith("_SOFTWARE_ERROR.json"):
            continue
        payload = load_json(path)
        if "summary" not in payload:
            continue
        row = dict(payload["summary"])
        episode = payload.get("episode", {})
        for name in (
            "agent_completion_rate",
            "obstacle_collision",
            "inter_agent_collision",
            "minimum_obstacle_clearance_m",
            "minimum_static_obstacle_clearance_m",
            "minimum_inter_agent_distance_m",
            "team_path_length_m",
            "team_path_length_mean_agent_m",
            "completion_time_s",
            "trajectory_smoothness",
            "terminal_progress_team_mean_m",
            "action_saturation_rate",
        ):
            row[name] = episode.get(name)
        events = payload.get("events", [])
        row["selection_event_count"] = len(events)
        row["effective_null_selection_count"] = sum(bool(event.get("selected_null")) for event in events)
        row["raw_null_selection_count"] = sum(bool(event.get("raw_selected_null", event.get("selected_null"))) for event in events)
        row["interaction_feasibility_mask_count"] = sum(bool(event.get("interaction_feasibility_mask_applied")) for event in events)
        rows.append(row)
    return sorted(rows, key=lambda row: str(row["scenario_id"]))


def prepare(config_path: Path) -> None:
    config = load_json(config_path)
    if config["schema_version"] != SCHEMA_VERSION:
        raise RuntimeError("unexpected development config schema")
    root = artifact_root(config)
    index_path = manifest_path(config)
    index = load_json(index_path)
    block = evaluation_block(config)
    if len(index["entries"]) != int(config["expected_scenario_count"]):
        raise RuntimeError(f"{block} index must contain the expected scenario count")
    checkpoint = REPO_ROOT / str(config["gat_r_checkpoint"])
    if sha256_file(checkpoint) != str(config["gat_r_checkpoint_sha256_expected"]):
        raise RuntimeError("GAT-R checkpoint hash mismatch")
    scene_checks = []
    for row in index["entries"]:
        scene = root / str(row["scenario_file"])
        scene_checks.append(sha256_file(scene) == str(row["scenario_file_sha256"]))
    if not all(scene_checks):
        raise RuntimeError(f"one or more {block} scene hashes failed")
    selectors = {}
    for selector in config["selectors"]:
        resolved = resolve_runtime_config(config, selector)
        out = output_dir(config, selector)
        out.mkdir(parents=True, exist_ok=True)
        selectors[selector] = {
            "runtime_method": resolved["method"],
            "display_name": config["selectors"][selector]["display_name"],
            "output": str(out.relative_to(REPO_ROOT).as_posix()),
            "resolved_runtime_config": resolved,
            "only_selector_difference": True,
        }
    freeze = {
        "schema_version": SCHEMA_VERSION,
        "status": f"FROZEN_BEFORE_{block.upper()}_PERFORMANCE",
        "evaluation_block": block,
        "config_sha256": sha256_file(config_path),
        "development_index_sha256": sha256_file(index_path),
        "development_index_semantic_sha256": index["manifest_semantic_sha256"],
        "scene_count": len(index["entries"]),
        "scene_file_hashes_verified": sum(scene_checks),
        "gat_r_checkpoint_sha256": sha256_file(checkpoint),
        "sac_checkpoint_sha256": sha256_file(REPO_ROOT / selectors["gat_r"]["resolved_runtime_config"]["sac_checkpoint"]),
        "source_sha256": {path: sha256_file(REPO_ROOT / path) for path in SOURCE_PATHS},
        "selectors": selectors,
        "paired_scene_contract": True,
        "gat_r_goal_distance_scale_m": 100.0,
        "gat_v1_default_scale_unchanged_m": 9.0,
        "canonical_gat_projection_directions": 56,
        "runtime_future_information_added": False,
        "formal_v1_used": False,
        "holdout_opened": block == "holdout",
        "formal_v2_generated": False,
    }
    write_json(preflight_path(config), freeze)
    print(json.dumps({"status": "PASS", "scene_count": len(scene_checks)}, indent=2))


def verify(config_path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    freeze_path = preflight_path(config)
    freeze = load_json(freeze_path)
    checks = {
        "config": sha256_file(config_path) == freeze["config_sha256"],
        "index": sha256_file(manifest_path(config)) == freeze["development_index_sha256"],
        "checkpoint": sha256_file(REPO_ROOT / config["gat_r_checkpoint"]) == freeze["gat_r_checkpoint_sha256"],
        "sources": all(sha256_file(REPO_ROOT / path) == expected for path, expected in freeze["source_sha256"].items()),
    }
    if not all(checks.values()):
        raise RuntimeError(f"{evaluation_block(config)} freeze mismatch: {checks}")
    return freeze


def finalize(config: Mapping[str, Any], selector: str) -> None:
    index = load_json(manifest_path(config))
    expected = {str(row["scenario_id"]) for row in index["entries"]}
    output = output_dir(config, selector)
    rows = collect(output)
    prefix = result_prefix(config)
    write_csv(output / f"{prefix}_team_results.csv", rows)
    summaries: list[dict[str, Any]] = []
    for scope in ("overall", *STAGE_ORDER):
        members = rows if scope == "overall" else [row for row in rows if row["stage"] == scope]
        if not members:
            continue
        successful = [row for row in members if bool(row["team_success"])]
        summaries.append(
            {
                "schema_version": SCHEMA_VERSION,
                "selector": selector,
                "scope": scope,
                "n": len(members),
                "team_success_rate": float(np.mean([bool(row["team_success"]) for row in members])),
                "collision_rate": float(np.mean([bool(row["collision"]) for row in members])),
                "obstacle_collision_rate": float(np.mean([bool(row["obstacle_collision"]) for row in members])),
                "inter_agent_collision_rate": float(np.mean([bool(row["inter_agent_collision"]) for row in members])),
                "timeout_rate": float(np.mean([bool(row["timeout"]) for row in members])),
                "agent_completion_rate": float(np.mean([float(row["agent_completion_rate"]) for row in members])),
                "successful_episode_count": len(successful),
                "success_smoothness_mean": float(np.mean([float(row["trajectory_smoothness"]) for row in successful])) if successful else None,
                "mean_upper_invocations": float(np.mean([float(row["upper_pipeline_invocation_count"]) for row in members])),
                "mean_total_compute_ms": float(np.mean([float(row["total_online_algorithm_compute_ms"]) for row in members])),
            }
        )
    write_csv(output / f"{prefix}_summary.csv", summaries)
    observed = {str(row["scenario_id"]) for row in rows}
    errors = list((output / "episode_records").glob("*_SOFTWARE_ERROR.json"))
    write_json(
        output / f"{prefix}_reconciliation.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS" if observed == expected and not errors else "IN_PROGRESS",
            "selector": selector,
            "completed_scenarios": len(observed),
            "expected_scenarios": len(expected),
            "missing_scenario_ids": sorted(expected - observed),
            "unexpected_scenario_ids": sorted(observed - expected),
            "software_error_count": len(errors),
            "overall": next((row for row in summaries if row["scope"] == "overall"), None),
            "evaluation_block": evaluation_block(config),
            "formal_data_used": evaluation_block(config) == "formal_v2",
        },
    )


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
    block = evaluation_block(config)
    indexed = list(enumerate(index["entries"]))
    if shard_count is not None:
        if shard_index is None or not 0 <= int(shard_index) < int(shard_count):
            raise ValueError("invalid shard index/count")
        indexed = [(i, row) for i, row in indexed if i % int(shard_count) == int(shard_index)]
    pending = [row for _, row in indexed if str(row["scenario_id"]) not in completed]
    if limit is not None:
        pending = pending[: int(limit)]
    runtime_method = METHOD_RERR_FP_SHEP if selector == "fp_shep" else METHOD_RERR_GAT
    for entry in pending:
        sid = str(entry["scenario_id"])
        print(f"[GAT-R-{block}:{selector}] start {sid}", flush=True)
        recorder = OnlineRuntimeRecorder()
        proxy = TimedPolicyProxy(runtime.policy, recorder)
        try:
            with recorder.instrument_dmp(), recorder.scoped_context(
                evaluation_block=f"gat_recurrent_r_{block}",
                configuration_id=runtime_config["configuration_id"],
                stage=entry["stage"],
                family=entry["family"],
                scenario_id=sid,
                seed=int(entry["seed"]),
                method=selector,
            ):
                episode, agents, events, triggers, extra = run_episode(
                    config=runtime.eval_config,
                    settings=runtime.settings,
                    multi_config=runtime.multi_config,
                    policy=proxy,
                    gat_model=runtime.gat_model,
                    gat_device=runtime.gat_device,
                    method=runtime_method,
                    scenario=sid,
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
                    "performance_used_for_selection": block == "development",
                }
            )
            save_record(output, entry, episode, agents, events, triggers, extra, summary)
            print(f"[GAT-R-{block}:{selector}] complete {sid} success={int(summary['team_success'])} collision={int(summary['collision'])}", flush=True)
        except Exception as error:
            write_json(
                output / "episode_records" / f"{sid}_SOFTWARE_ERROR.json",
                {
                    "scenario_id": sid,
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
    parser.add_argument("--selector", choices=("fp_shep", "gat_r"))
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
