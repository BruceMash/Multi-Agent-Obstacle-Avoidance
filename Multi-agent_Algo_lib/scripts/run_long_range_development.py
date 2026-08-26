"""Run resumable long-range development evaluations on the frozen 200-scene split."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import sys
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as _pandas  # noqa: F401  # initialize Windows pyarrow before torch
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
for search_path in (REPO_ROOT, ALGO_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from Environment.frozen_sac_dmp_execution import freeze_policy  # noqa: E402
from experiment_config import EXPERIMENT_CONFIG  # noqa: E402
from planning.gat.stage1_training import load_model_checkpoint, resolve_device  # noqa: E402
from planning.online_runtime_instrumentation import OnlineRuntimeRecorder, TimedPolicyProxy  # noqa: E402
from planning.semi_structured_long_range_benchmark import STAGE_ORDER, WORKSPACE_BOUNDS, json_ready  # noqa: E402
from runner_sac import build_model, load_checkpoint  # noqa: E402
from scripts.evaluate_gat_closed_loop import _model_hash, _policy_parameter_sha256  # noqa: E402
from scripts.evaluate_gat_v1_err_development import (  # noqa: E402
    METHOD_RERR_GAT,
    METHOD_RERR_FP_SHEP,
    build_online_gat_plan_optimized,
    run_episode,
)
from scripts.evaluate_single_policy_aligned_multi_agent import build_single_distribution_multi_config  # noqa: E402
from scripts.run_long_range_contract_pilot import (  # noqa: E402
    LongRangeManifestEnvironmentBuilder,
    save_record,
    summarize_record,
)
from scripts.train_long_range_local_sac import LongRangeLocalReferenceEnv  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs/evaluation/semi_structured_long_range_development_d01.json"
SOURCE_PATHS = (
    "Environment/frozen_sac_dmp_execution.py",
    "Environment/multi_agent_dmp_env.py",
    "Entity/KinematicModel.py",
    "planning/semi_structured_long_range_benchmark.py",
    "planning/event_triggered_reference_reconstruction.py",
    "planning/historical_forcing_gate.py",
    "planning/heterogeneous_candidate_graph.py",
    "planning/policy_preview.py",
    "planning/pre_gat_closed_loop.py",
    "planning/goal_semantics_diagnosis.py",
    "Multi-agent_Algo_lib/scripts/evaluate_gat_v1_err_development.py",
    "Multi-agent_Algo_lib/scripts/run_long_range_contract_pilot.py",
    "Multi-agent_Algo_lib/scripts/run_long_range_development.py",
    "configs/evaluation/semi_structured_long_range_development_d01.json",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_ready(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(json_ready(row.get(key)), ensure_ascii=False, separators=(",", ":"))
                    if isinstance(row.get(key), (dict, list, tuple, np.ndarray))
                    else row.get(key)
                    for key in fields
                }
            )
    temporary.replace(path)


class DevelopmentRuntime:
    def __init__(self, config: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
        sensor = config["sensor"]
        base = build_single_distribution_multi_config(
            num_agents=int(config["num_agents"]), max_steps=int(config["max_steps"])
        )
        self.multi_config = replace(
            base,
            workspace_bounds=tuple(tuple(row) for row in WORKSPACE_BOUNDS),
            randomize_start_goal=False,
            start_position_bounds=None,
            goal_position_bounds=None,
            min_start_goal_distance=0.0,
            sensing_radius=float(sensor["range_m"]),
            sensor_azimuth_bins=int(sensor["azimuth_bins"]),
            sensor_elevation_bins=int(sensor["elevation_bins"]),
            sensor_goal_distance_clip=2.0 * float(sensor["range_m"]),
        )
        self.builder = LongRangeManifestEnvironmentBuilder(manifest, config)
        self.eval_config = {
            "top_k": int(config["top_k"]),
            "H_preview": int(config["H_preview"]),
            "max_steps": int(config["max_steps"]),
            "dt": float(config["dt"]),
            "peer_radius": float(config["peer_radius"]),
            "handoff_threshold_m": float(config["handoff_threshold_m"]),
            "proposal_config": copy.deepcopy(config["proposal_config"]),
            "fp_shep": copy.deepcopy(config["fp_shep"]),
            "graph": copy.deepcopy(config["graph"]),
            "err": copy.deepcopy(config["err"]),
        }
        self.settings = load_json(REPO_ROOT / config["base_execution_config"])
        self.settings.update(
            {
                "checkpoint": config["sac_checkpoint"],
                "checkpoint_sha256_expected": config["sac_checkpoint_sha256_expected"],
                "deterministic_policy": True,
                "num_agents": int(config["num_agents"]),
                "max_steps": int(config["max_steps"]),
                "dt": float(config["dt"]),
                "peer_radius": float(config["peer_radius"]),
                "temporary_reference": {
                    **self.settings["temporary_reference"],
                    "K_requested": int(config["top_k"]),
                    "reached_tolerance_m": float(config["handoff_threshold_m"]),
                    "replanning_enabled": True,
                },
                "proposal_config": copy.deepcopy(config["proposal_config"]),
            }
        )

        training_config = load_json(REPO_ROOT / config["training_config"])
        training_root = REPO_ROOT / training_config["artifact_root"]
        training_manifest = load_json(training_root / training_config["training_manifest"])
        reference_env = LongRangeLocalReferenceEnv(
            training_manifest, training_config, seed=int(training_config["seed"]), training=False
        )
        model_config = replace(
            EXPERIMENT_CONFIG,
            learning_rate=float(training_config["learning_rate"]),
            buffer_size=int(training_config["buffer_size"]),
            batch_size=int(training_config["batch_size"]),
            learning_starts=int(training_config["learning_starts"]),
            train_freq=int(training_config["train_freq"]),
            gradient_steps=int(training_config["gradient_steps"]),
            ent_coef=training_config["ent_coef"],
            verbose=0,
        )
        self.policy = build_model(reference_env, config=model_config, verbose=0)
        checkpoint = torch.load(REPO_ROOT / config["sac_checkpoint"], map_location=self.policy.device)
        self.policy.actor.load_state_dict(checkpoint["actor"], strict=True)
        self.policy.critic.load_state_dict(checkpoint["critic"], strict=True)
        self.policy.critic_target.load_state_dict(checkpoint["critic_target"], strict=True)
        if "log_ent_coef" in checkpoint:
            self.policy.log_ent_coef = checkpoint["log_ent_coef"].to(self.policy.device).detach()
        freeze_policy(self.policy)

        stage1 = load_json(REPO_ROOT / config["stage1_config"])
        self.gat_device = resolve_device(stage1["training"]["device"])
        self.gat_model = load_model_checkpoint(
            REPO_ROOT / config["gat_checkpoint"], stage1, self.gat_device
        )
        for parameter in self.gat_model.parameters():
            parameter.requires_grad_(False)
        self.gat_model.eval()
        self.policy_hash = _policy_parameter_sha256(self.policy)
        self.gat_hash = _model_hash(self.gat_model)


def paths(config: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    root = (REPO_ROOT / config["artifact_root"]).resolve()
    manifest = root / config["development_manifest"]
    output = root / config["output_subdir"]
    return root, manifest, output


def evaluation_scenario_ids(
    config: Mapping[str, Any], manifest: Mapping[str, Any]
) -> list[str]:
    available = [str(entry["scenario_id"]) for entry in manifest["entries"]]
    requested = config.get("development_scenario_ids")
    selected = available if requested is None else [str(value) for value in requested]
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("development scenario ids must be non-empty and unique")
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"development scenario ids are absent from manifest: {unknown}")
    return selected


def evaluation_method(config: Mapping[str, Any]) -> str:
    method = str(config["method"])
    if method not in {METHOD_RERR_GAT, METHOD_RERR_FP_SHEP}:
        raise ValueError("long-range development supports gat_v1_rerr or fp_shep_rerr")
    return method


def prepare(config_path: Path) -> None:
    config = load_json(config_path)
    root, manifest_path, output = paths(config)
    output.mkdir(parents=True, exist_ok=True)
    manifest = load_json(manifest_path)
    if len(manifest["entries"]) != 200:
        raise RuntimeError("development manifest must contain exactly 200 scenarios")
    selected_ids = evaluation_scenario_ids(config, manifest)
    method = evaluation_method(config)
    checkpoint_hashes = {
        "sac": sha256_file(REPO_ROOT / config["sac_checkpoint"]),
        "gat": sha256_file(REPO_ROOT / config["gat_checkpoint"]),
    }
    if checkpoint_hashes["sac"] != config["sac_checkpoint_sha256_expected"]:
        raise RuntimeError("SAC checkpoint hash mismatch")
    if checkpoint_hashes["gat"] != config["gat_checkpoint_sha256_expected"]:
        raise RuntimeError("GAT checkpoint hash mismatch")
    freeze = {
        "schema_version": "long_range_development_freeze_v1",
        "status": "FROZEN_BEFORE_DEVELOPMENT_EVALUATION",
        "configuration_sha256": sha256_file(config_path),
        "development_manifest_sha256": sha256_file(manifest_path),
        "development_manifest_semantic_sha256": manifest["manifest_sha256"],
        "scenario_count": len(selected_ids),
        "development_scenario_ids": selected_ids,
        "method": method,
        "checkpoint_sha256": checkpoint_hashes,
        "source_sha256": {path: sha256_file(REPO_ROOT / path) for path in SOURCE_PATHS},
        "formal_manifest_exists": False,
        "formal_result_count": 0,
        "performance_selection_authorized": True,
    }
    atomic_json(output / "development_freeze.json", freeze)
    atomic_json(
        output / "information_contract.json",
        {
            "actor_spatial_directions": int(config["sensor"]["direction_count"]),
            "actor_temporal_frames": 2,
            "actor_observation_dim": int(config["sensor"]["actor_observation_dim"]),
            "sensor_range_m": float(config["sensor"]["range_m"]),
            "gat_sector_projection_directions": int(
                config["sensor"]["gat_canonical_projection_directions"]
            ),
            "gat_checkpoint_retrained": False,
            "gat_checkpoint_used": method == METHOD_RERR_GAT,
            "continuous_global_peer_state_used": False,
            "global_dynamic_map_used": False,
        },
    )
    print(
        json.dumps(
            {"phase": "prepare", "status": "PASS", "scenario_count": len(selected_ids)}
        ),
        flush=True,
    )


def verify(config_path: Path, config: Mapping[str, Any], manifest_path: Path, output: Path) -> None:
    freeze = load_json(output / "development_freeze.json")
    if sha256_file(config_path) != freeze["configuration_sha256"]:
        raise RuntimeError("development config changed after freeze")
    if sha256_file(manifest_path) != freeze["development_manifest_sha256"]:
        raise RuntimeError("development manifest changed after freeze")
    for source, expected in freeze["source_sha256"].items():
        if sha256_file(REPO_ROOT / source) != expected:
            raise RuntimeError(f"development source changed after freeze: {source}")


def collect(output: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((output / "episode_records").glob("DEV_LR_*.json")):
        payload = load_json(path)
        if "summary" in payload:
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
            row["effective_null_selection_count"] = sum(
                bool(event.get("selected_null")) for event in events
            )
            row["raw_null_selection_count"] = sum(
                bool(event.get("raw_selected_null", event.get("selected_null")))
                for event in events
            )
            row["far_terminal_null_mask_count"] = sum(
                bool(event.get("far_terminal_null_mask_applied")) for event in events
            )
            row["far_terminal_null_mask_unavailable_count"] = sum(
                bool(event.get("far_terminal_null_mask_unavailable_no_candidate"))
                for event in events
            )
            row["interaction_feasibility_mask_count"] = sum(
                bool(event.get("interaction_feasibility_mask_applied"))
                for event in events
            )
            rows.append(row)
    return sorted(rows, key=lambda row: str(row["scenario_id"]))


def finalize(config: Mapping[str, Any], output: Path) -> None:
    root, manifest_path, _ = paths(config)
    del root
    manifest = load_json(manifest_path)
    expected_ids = evaluation_scenario_ids(config, manifest)
    rows = collect(output)
    write_csv(output / "development_team_results.csv", rows)
    scopes: list[dict[str, Any]] = []
    for scope in ["overall", *STAGE_ORDER]:
        members = rows if scope == "overall" else [row for row in rows if row["stage"] == scope]
        if not members:
            continue
        scopes.append(
            {
                "scope": scope,
                "n": len(members),
                "team_success_rate": float(np.mean([row["team_success"] for row in members])),
                "collision_rate": float(np.mean([row["collision"] for row in members])),
                "obstacle_collision_rate": float(
                    np.mean([row["obstacle_collision"] for row in members])
                ),
                "inter_agent_collision_rate": float(
                    np.mean([row["inter_agent_collision"] for row in members])
                ),
                "timeout_rate": float(np.mean([row["timeout"] for row in members])),
                "mean_upper_invocations": float(np.mean([row["upper_pipeline_invocation_count"] for row in members])),
                "mean_total_compute_ms": float(np.mean([row["total_online_algorithm_compute_ms"] for row in members])),
            }
        )
    write_csv(output / "development_summary.csv", scopes)
    observed_ids = [str(row["scenario_id"]) for row in rows]
    orphan_software_errors = [
        path
        for path in (output / "episode_records").glob("*_SOFTWARE_ERROR.json")
        if path.name.removesuffix("_SOFTWARE_ERROR.json") not in set(observed_ids)
    ]
    complete = set(observed_ids) == set(expected_ids) and len(observed_ids) == len(
        expected_ids
    )
    overall = next((row for row in scopes if row["scope"] == "overall"), None)
    atomic_json(
        output / "development_reconciliation.json",
        {
            "status": "PASS" if complete else "IN_PROGRESS",
            "completed_scenarios": len(rows),
            "expected_scenarios": len(expected_ids),
            "missing_scenario_ids": sorted(set(expected_ids) - set(observed_ids)),
            "unexpected_scenario_ids": sorted(set(observed_ids) - set(expected_ids)),
            "software_error_count": len(orphan_software_errors),
            "overall": overall,
            "formal_data_used": False,
        },
    )


def run(
    config_path: Path,
    limit: int | None,
    *,
    shard_index: int | None = None,
    shard_count: int | None = None,
    defer_finalize: bool = False,
) -> None:
    config = load_json(config_path)
    _, manifest_path, output = paths(config)
    verify(config_path, config, manifest_path, output)
    manifest = load_json(manifest_path)
    runtime = DevelopmentRuntime(config, manifest)
    method = evaluation_method(config)
    selected_ids = set(evaluation_scenario_ids(config, manifest))
    completed = {row["scenario_id"] for row in collect(output)}
    indexed_entries = [
        (index, entry)
        for index, entry in enumerate(manifest["entries"])
        if str(entry["scenario_id"]) in selected_ids
    ]
    if shard_count is not None:
        if shard_index is None or not 0 <= int(shard_index) < int(shard_count):
            raise ValueError("shard_index must be in [0, shard_count)")
        indexed_entries = [
            (index, entry)
            for index, entry in indexed_entries
            if index % int(shard_count) == int(shard_index)
        ]
    pending = [entry for _, entry in indexed_entries if entry["scenario_id"] not in completed]
    if limit is not None:
        pending = pending[: int(limit)]
    for entry in pending:
        sid = str(entry["scenario_id"])
        print(f"[long-range-dev] start {sid} {entry['stage']} {entry['family']}", flush=True)
        recorder = OnlineRuntimeRecorder()
        proxy = TimedPolicyProxy(runtime.policy, recorder)
        try:
            with recorder.instrument_dmp(), recorder.scoped_context(
                evaluation_block="long_range_development",
                configuration_id=config["configuration_id"],
                stage=entry["stage"],
                family=entry["family"],
                scenario_id=sid,
                seed=int(entry["seed"]),
                method=method,
            ):
                episode, agents, events, triggers, extra = run_episode(
                    config=runtime.eval_config,
                    settings=runtime.settings,
                    multi_config=runtime.multi_config,
                    policy=proxy,
                    gat_model=runtime.gat_model,
                    gat_device=runtime.gat_device,
                    method=method,
                    scenario=sid,
                    seed=int(entry["seed"]),
                    environment_builder=runtime.builder,
                    runtime_recorder=recorder,
                    upper_plan_builder=build_online_gat_plan_optimized,
                )
            summary = summarize_record(entry, episode, events, triggers, extra["path_rows"])
            summary.update(
                {
                    "configuration_id": config["configuration_id"],
                    "method": method,
                    "team_success": bool(episode["team_success"]),
                    "collision": bool(episode["collision"]),
                    "timeout": bool(episode["timeout"]),
                    "performance_used_for_selection": True,
                }
            )
            save_record(output, entry, episode, agents, events, triggers, extra, summary)
            print(
                f"[long-range-dev] complete {sid} success={int(summary['team_success'])} collision={int(summary['collision'])} steps={summary['steps']}",
                flush=True,
            )
        except Exception as error:
            atomic_json(
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
            finalize(config, output)
    if not defer_finalize:
        finalize(config, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "run", "finalize"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    parser.add_argument("--defer-finalize", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    if args.phase == "prepare":
        prepare(config_path)
    elif args.phase == "run":
        run(
            config_path,
            args.limit,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            defer_finalize=args.defer_finalize,
        )
    else:
        config = load_json(config_path)
        _, _, output = paths(config)
        finalize(config, output)


if __name__ == "__main__":
    main()
