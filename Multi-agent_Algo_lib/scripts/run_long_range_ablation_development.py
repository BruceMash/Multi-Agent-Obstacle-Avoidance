"""Evaluate frozen one-shot SAC-DMP ablations on the 200-scene long-range development split.

This runner is development-only.  It preserves the same scenario manifest,
adapted 256-direction actor, candidate generator, FP-SHEP preview, GAT-V1,
execution contract, and collision rules used by the selected recurrent method.
Only the one-shot selector is changed between configurations.
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
import pandas as _pandas  # noqa: F401  # initialize Windows pyarrow before torch


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
for search_path in (REPO_ROOT, ALGO_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from planning.online_runtime_instrumentation import (  # noqa: E402
    OnlineRuntimeRecorder,
    TimedPolicyProxy,
)
from planning.long_range_collision_recheck import audit_trajectory_collisions  # noqa: E402
from planning.semi_structured_long_range_benchmark import (  # noqa: E402
    STAGE_ORDER,
    json_ready,
)
from scripts.evaluate_gat_closed_loop import (  # noqa: E402
    METHOD_FP_SHEP,
    METHOD_GAT,
    METHOD_PROPOSAL,
    METHOD_TERMINAL,
    build_shared_selection_bundle,
    run_method_episode,
)
from scripts.evaluate_gat_v1_err_development import (  # noqa: E402
    build_online_gat_plan_optimized,
)
from scripts.evaluate_pre_gat_closed_loop import (  # noqa: E402
    _scenario_hash,
    _scene_snapshot,
)
from scripts.run_long_range_development import DevelopmentRuntime  # noqa: E402
from planning.pre_gat_220step_revalidation import stable_hash  # noqa: E402


SCHEMA = "semi_structured_long_range_ablation_development_v1"
DEFAULT_CONFIG = REPO_ROOT / "configs/evaluation/semi_structured_long_range_ablation_direct_sac.json"
METHODS = {
    "direct_sac": {
        "internal": METHOD_TERMINAL,
        "display_name": "Direct SAC-DMP",
        "role": "LEARNING_BASELINE",
    },
    "proposal": {
        "internal": METHOD_PROPOSAL,
        "display_name": "Proposal + SAC-DMP",
        "role": "REFERENCE_GENERATION_ABLATION",
    },
    "fp_shep": {
        "internal": METHOD_FP_SHEP,
        "display_name": "FP-SHEP + SAC-DMP",
        "role": "EXECUTION_AWARE_PREVIEW_ABLATION",
    },
    "one_shot_gat": {
        "internal": METHOD_GAT,
        "display_name": "One-Shot GAT + SAC-DMP",
        "role": "NO_RECURRENT_RECONSTRUCTION_ABLATION",
    },
}
SOURCE_PATHS = (
    "Environment/frozen_sac_dmp_execution.py",
    "Environment/multi_agent_dmp_env.py",
    "Entity/KinematicModel.py",
    "planning/semi_structured_long_range_benchmark.py",
    "planning/historical_forcing_gate.py",
    "planning/heterogeneous_candidate_graph.py",
    "planning/online_runtime_instrumentation.py",
    "planning/policy_preview.py",
    "planning/pre_gat_closed_loop.py",
    "Multi-agent_Algo_lib/scripts/evaluate_gat_closed_loop.py",
    "Multi-agent_Algo_lib/scripts/run_final_four_stage_benchmark.py",
    "Multi-agent_Algo_lib/scripts/run_long_range_contract_pilot.py",
    "Multi-agent_Algo_lib/scripts/run_long_range_development.py",
    "Multi-agent_Algo_lib/scripts/run_long_range_ablation_development.py",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolved_config(config_path: Path) -> dict[str, Any]:
    raw = load_json(config_path)
    base_relative = raw.get("base_config")
    if base_relative is None:
        return raw
    base_path = (REPO_ROOT / str(base_relative)).resolve()
    merged = copy.deepcopy(load_json(base_path))
    merged.update({key: value for key, value in raw.items() if key != "base_config"})
    merged["_base_config"] = str(base_relative)
    merged["_base_config_sha256"] = sha256_file(base_path)
    return merged


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
        json.dumps(json_ready(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
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
            if str(key) not in fields:
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


def artifact_paths(config: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    root = (REPO_ROOT / str(config["artifact_root"])).resolve()
    manifest = root / str(config["development_manifest"])
    output = root / str(config["output_subdir"])
    return root, manifest, output


def selected_entries(config: Mapping[str, Any], manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries = [dict(row) for row in manifest["entries"]]
    requested = config.get("development_scenario_ids")
    if requested is None:
        return entries
    requested_ids = [str(value) for value in requested]
    if not requested_ids or len(requested_ids) != len(set(requested_ids)):
        raise ValueError("development_scenario_ids must be non-empty and unique")
    by_id = {str(row["scenario_id"]): row for row in entries}
    unknown = sorted(set(requested_ids) - set(by_id))
    if unknown:
        raise ValueError(f"unknown development scenarios: {unknown}")
    return [by_id[scenario_id] for scenario_id in requested_ids]


def method_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    method = str(config["method"])
    if method not in METHODS:
        raise ValueError(f"unsupported one-shot ablation: {method}")
    return {"method": method, **METHODS[method]}


def prepare(config_path: Path) -> None:
    config = resolved_config(config_path)
    _, manifest_path, output = artifact_paths(config)
    output.mkdir(parents=True, exist_ok=True)
    manifest = load_json(manifest_path)
    if len(manifest["entries"]) != 200:
        raise RuntimeError("development manifest must contain exactly 200 scenarios")
    entries = selected_entries(config, manifest)
    method = method_contract(config)
    checkpoint_hashes = {
        "sac": sha256_file(REPO_ROOT / str(config["sac_checkpoint"])),
        "gat": sha256_file(REPO_ROOT / str(config["gat_checkpoint"])),
    }
    if checkpoint_hashes["sac"] != str(config["sac_checkpoint_sha256_expected"]):
        raise RuntimeError("SAC checkpoint hash mismatch")
    if checkpoint_hashes["gat"] != str(config["gat_checkpoint_sha256_expected"]):
        raise RuntimeError("GAT checkpoint hash mismatch")
    freeze = {
        "schema_version": SCHEMA,
        "status": "FROZEN_BEFORE_DEVELOPMENT_EVALUATION",
        "configuration_sha256": sha256_file(config_path),
        "base_configuration": config.get("_base_config"),
        "base_configuration_sha256": config.get("_base_config_sha256"),
        "development_manifest_sha256": sha256_file(manifest_path),
        "development_manifest_semantic_sha256": manifest["manifest_sha256"],
        "scenario_count": len(entries),
        "development_scenario_ids": [row["scenario_id"] for row in entries],
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
            "continuous_global_peer_state_used": False,
            "global_dynamic_map_used": False,
            "upper_event_peer_state_used": method["internal"] == METHOD_GAT,
            "recurrent_reconstruction_enabled": False,
        },
    )
    print(json.dumps({"phase": "prepare", "status": "PASS", "method": method["method"], "scenario_count": len(entries)}), flush=True)


def verify(config_path: Path, config: Mapping[str, Any], manifest_path: Path, output: Path) -> dict[str, Any]:
    freeze = load_json(output / "development_freeze.json")
    if sha256_file(config_path) != freeze["configuration_sha256"]:
        raise RuntimeError("development config changed after freeze")
    if sha256_file(manifest_path) != freeze["development_manifest_sha256"]:
        raise RuntimeError("development manifest changed after freeze")
    if freeze.get("base_configuration") is not None:
        base_path = REPO_ROOT / str(freeze["base_configuration"])
        if sha256_file(base_path) != freeze["base_configuration_sha256"]:
            raise RuntimeError("base development configuration changed after freeze")
    for relative, expected in freeze["source_sha256"].items():
        if sha256_file(REPO_ROOT / relative) != expected:
            raise RuntimeError(f"development source changed after freeze: {relative}")
    return freeze


def terminal_shared(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "scenario": entry["scenario_id"],
        "seed": int(entry["seed"]),
        "initial_condition_hash": entry["environment_fingerprint"],
        "candidate_bundle_hash": None,
        "preview_bundle_hash": None,
        "proposal_reconstruction_equivalent": True,
        "graph_schema_match": True,
        "preview_historical_gate_verified": True,
        "plans": {},
        "planning_runtime_ms": {METHOD_TERMINAL: 0.0},
    }


def optimized_gat_shared(
    runtime: DevelopmentRuntime,
    config: Mapping[str, Any],
    entry: Mapping[str, Any],
    policy: Any,
    recorder: OnlineRuntimeRecorder,
) -> dict[str, Any]:
    """Build the selected D05 one-shot GAT plan, including both accepted masks."""

    env, scene_metadata = runtime.builder(
        config=runtime.multi_config,
        scenario=str(entry["scenario_id"]),
        seed=int(entry["seed"]),
        peer_radius=float(config["peer_radius"]),
    )
    try:
        snapshot = _scene_snapshot(env)
        initial_hash = _scenario_hash(snapshot)
        selection = build_online_gat_plan_optimized(
            env=env,
            config=runtime.eval_config,
            policy=policy,
            gat_model=runtime.gat_model,
            gat_device=runtime.gat_device,
            scenario=str(entry["scenario_id"]),
            seed=int(entry["seed"]),
            runtime_recorder=recorder,
            selector="gat",
        )
        graphs = selection.get("_audit_graphs_by_agent", {})
        preview_identity: list[dict[str, Any]] = []
        for agent_id in sorted(graphs):
            graph = graphs[agent_id]
            for candidate_id, proposal in enumerate(graph.original_proposals):
                preview_identity.append(
                    {
                        "agent_id": int(agent_id),
                        "candidate_id": int(candidate_id),
                        "candidate_world_position": np.asarray(proposal.point, dtype=float),
                        "preview_positions": np.asarray(
                            graph.candidate_preview_positions[candidate_id], dtype=float
                        ),
                        "preview_velocities": np.asarray(
                            graph.candidate_preview_velocities[candidate_id], dtype=float
                        ),
                    }
                )
        upper_ms = float(selection["runtime_components"]["upper_planning_total_ms"])
        return {
            "scenario": str(entry["scenario_id"]),
            "seed": int(entry["seed"]),
            "initial_condition_hash": initial_hash,
            "scenario_snapshot": json_ready(snapshot),
            "scene_metadata": json_ready(scene_metadata),
            "candidate_bundle_hash": selection["candidate_bundle_hash"],
            "preview_bundle_hash": stable_hash(preview_identity),
            "proposal_reconstruction_equivalent": bool(
                selection["proposal_reconstruction_equivalent"]
            ),
            "graph_schema_match": bool(selection["graph_schema_match"]),
            "preview_historical_gate_verified": bool(
                selection["preview_historical_gate_verified"]
            ),
            "plans": {METHOD_GAT: selection["plan"]},
            "planning_runtime_ms": {
                METHOD_GAT: upper_ms,
                "upper_planning_total_actual": upper_ms,
            },
            "candidate_count_per_agent": selection["candidate_count_per_agent"],
            "runtime_components": selection["runtime_components"],
        }
    finally:
        env.close()


def standardize_long_range_sac_result(
    episode: Mapping[str, Any],
    agents: Sequence[Mapping[str, Any]],
    trajectory: Mapping[str, Any],
    entry: Mapping[str, Any],
    method: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    """Attach long-range identities and independently recomputed clearances."""

    positions = np.asarray(trajectory["positions"], dtype=float)
    collision = audit_trajectory_collisions(positions, entry)
    online = (
        bool(episode["obstacle_collision"]),
        bool(episode["inter_agent_collision"]),
        bool(episode["collision"]),
    )
    replay = (
        bool(collision["obstacle_collision"]),
        bool(collision["inter_agent_collision"]),
        bool(collision["any_collision"]),
    )
    if online != replay:
        raise RuntimeError(f"one-shot collision replay mismatch: online={online}, replay={replay}")
    result = copy.deepcopy(dict(episode))
    result.update(
        {
            "stage": entry["stage"],
            "family": entry["family"],
            "task_pattern": entry["task_pattern"],
            "scenario_id": entry["scenario_id"],
            "seed": int(entry["seed"]),
            "method": method,
            "any_collision": bool(episode["collision"]),
            "termination_time_s": float(episode["steps"] * entry["dt"]),
            "end_to_end_runtime_ms": float(episode["episode_runtime_ms"])
            + float(episode.get("planning_runtime_ms", 0.0)),
            **{key: value for key, value in collision.items() if not key.startswith("agent_")},
            "minimum_obstacle_clearance_m": collision[
                "minimum_obstacle_signed_clearance_m"
            ],
            "minimum_inter_agent_distance_m": collision[
                "minimum_inter_agent_distance_m"
            ],
        }
    )
    starts = np.asarray(entry["starts"], dtype=float)
    goals = np.asarray(entry["goals"], dtype=float)
    straight = np.linalg.norm(goals - starts, axis=1)
    path_lengths = np.asarray(episode["per_agent_path_length_m"], dtype=float)
    standardized_agents: list[dict[str, Any]] = []
    for source in agents:
        row = copy.deepcopy(dict(source))
        agent_id = int(row["agent_id"])
        completion_step = row.get("terminal_completed_step")
        completed = completion_step is not None
        row.update(
            {
                "stage": entry["stage"],
                "family": entry["family"],
                "scenario_id": entry["scenario_id"],
                "seed": int(entry["seed"]),
                "method": method,
                "agent_terminal_completed": bool(completed),
                "agent_collision": bool(collision["agent_any_collision"][agent_id]),
                "agent_obstacle_collision": bool(
                    collision["agent_obstacle_collision"][agent_id]
                ),
                "agent_inter_agent_collision": bool(
                    collision["agent_inter_agent_collision"][agent_id]
                ),
                "agent_boundary_collision": bool(
                    collision["agent_boundary_collision"][agent_id]
                ),
                "agent_path_length_m": float(path_lengths[agent_id]),
                "agent_path_efficiency": (
                    float(straight[agent_id] / max(path_lengths[agent_id], 1.0e-9))
                    if completed
                    else None
                ),
                "completion_step": completion_step,
                "minimum_obstacle_clearance_m": collision[
                    "agent_minimum_obstacle_signed_clearance_m"
                ][agent_id],
                "minimum_peer_distance_m": collision[
                    "agent_minimum_inter_agent_distance_m"
                ][agent_id],
            }
        )
        standardized_agents.append(row)
    dynamic_tracks = entry.get("dynamic_obstacle_trajectories", [])
    trajectory_payload = {
        **{key: np.asarray(value) for key, value in trajectory.items()},
        "dynamic_obstacle_positions": (
            np.asarray(dynamic_tracks, dtype=float)[:, : len(positions), :].transpose(1, 0, 2)
            if dynamic_tracks
            else np.empty((len(positions), 0, 3), dtype=float)
        ),
    }
    return result, standardized_agents, trajectory_payload


def evaluate_episode(
    runtime: DevelopmentRuntime,
    config: Mapping[str, Any],
    entry: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    recorder = OnlineRuntimeRecorder()
    policy = TimedPolicyProxy(runtime.policy, recorder)
    trajectory_sink: dict[str, Any] = {}
    internal = str(contract["internal"])
    execution_settings = copy.deepcopy(runtime.settings)
    nearest_peer_count = int(
        config.get("peer_information_contract", {}).get(
            "nearest_agent_count", max(int(config["num_agents"]) - 1, 0)
        )
    )
    native_observation_dim = (
        7
        + 2 * int(config["sensor"]["direction_count"])
        + 5
        + 7 * nearest_peer_count
    )
    execution_settings["checkpoint_observation"] = {
        **execution_settings["checkpoint_observation"],
        "dimension": int(config["sensor"]["actor_observation_dim"]),
        "environment_native_observation_dimension": native_observation_dim,
        "native_environment_observation_used_for_actor": False,
    }
    execution_settings["temporary_reference"] = {
        **execution_settings["temporary_reference"],
        "replanning_enabled": False,
    }
    with recorder.instrument_dmp(), recorder.scoped_context(
        evaluation_block="long_range_ablation_development",
        configuration_id=config["configuration_id"],
        stage=entry["stage"],
        family=entry["family"],
        scenario_id=entry["scenario_id"],
        seed=int(entry["seed"]),
        method=contract["method"],
    ):
        if internal == METHOD_TERMINAL:
            shared = terminal_shared(entry)
        elif internal == METHOD_GAT:
            shared = optimized_gat_shared(runtime, config, entry, policy, recorder)
        else:
            shared = build_shared_selection_bundle(
                config=runtime.eval_config,
                execution_settings=execution_settings,
                multi_config=runtime.multi_config,
                policy=policy,
                gat_model=runtime.gat_model,
                gat_device=runtime.gat_device,
                scenario=entry["scenario_id"],
                seed=int(entry["seed"]),
                environment_builder=runtime.builder,
                runtime_recorder=recorder,
            )
        episode, agents = run_method_episode(
            config=runtime.eval_config,
            execution_settings=execution_settings,
            multi_config=runtime.multi_config,
            policy=policy,
            shared=shared,
            method=internal,
            environment_builder=runtime.builder,
            trajectory_sink=trajectory_sink,
            runtime_recorder=recorder,
        )
    standardized, standardized_agents, trajectory = standardize_long_range_sac_result(
        episode, agents, trajectory_sink, entry, str(contract["method"])
    )
    execution_actor = [row for row in recorder.actor_rows if row.get("actor_mode") == "execution_actor"]
    execution_dmp = [row for row in recorder.dmp_rows if row.get("dmp_mode") == "execution_dmp"]
    upper_ms = float(episode.get("planning_runtime_ms", 0.0))
    actor_ms = float(sum(float(row["runtime_ms"]) for row in execution_actor))
    dmp_ms = float(sum(float(row["runtime_ms"]) for row in execution_dmp))
    standardized.update(
        {
            "schema_version": SCHEMA,
            "configuration_id": config["configuration_id"],
            "display_name": contract["display_name"],
            "method_role": contract["role"],
            "team_success": bool(episode["team_success"]),
            "collision": bool(episode["collision"]),
            "any_collision": bool(episode["collision"]),
            "timeout": bool(episode["timeout"]),
            "upper_planning_total_ms": upper_ms,
            "planning_runtime_ms": upper_ms,
            "planning_decision_count": 0 if internal == METHOD_TERMINAL else 1,
            "planning_runtime_per_decision_ms": None if internal == METHOD_TERMINAL else upper_ms,
            "execution_actor_forward_ms": actor_ms,
            "execution_actor_call_count": len(execution_actor),
            "execution_dmp_ms": dmp_ms,
            "execution_dmp_call_count": len(execution_dmp),
            "total_online_algorithm_compute_ms": upper_ms + actor_ms + dmp_ms,
            "upper_pipeline_invocation_count": 0 if internal == METHOD_TERMINAL else 1,
            "replanning_count": 0,
            "normal_replanning_count": 0,
            "emergency_replanning_count": 0,
            "performance_used_for_selection": True,
            "max_steps": int(config["max_steps"]),
        }
    )
    reference_events: list[dict[str, Any]] = []
    plan = shared["plans"].get(internal)
    if plan is not None:
        for agent_id, candidate in enumerate(plan["candidate_records"]):
            reference_events.append(
                {
                    "step": 0,
                    "agent_id": int(agent_id),
                    "event": "INITIAL_SELECTION",
                    "selected_candidate_id": candidate.get("selected_candidate_id"),
                    "selected_null": bool(candidate.get("selected_null", False)),
                    "active_goal": plan["references"][agent_id],
                    "selection_source": plan.get("selection_source"),
                }
            )
    shared_identity = {
        key: shared.get(key)
        for key in (
            "initial_condition_hash",
            "candidate_bundle_hash",
            "preview_bundle_hash",
            "proposal_reconstruction_equivalent",
            "graph_schema_match",
            "preview_historical_gate_verified",
        )
    }
    return standardized, standardized_agents, trajectory, reference_events, shared_identity


def save_episode(
    output: Path,
    entry: Mapping[str, Any],
    episode: Mapping[str, Any],
    agents: Sequence[Mapping[str, Any]],
    trajectory: Mapping[str, np.ndarray],
    events: Sequence[Mapping[str, Any]],
    shared_identity: Mapping[str, Any],
) -> None:
    records = output / "episode_records"
    records.mkdir(parents=True, exist_ok=True)
    scenario_id = str(entry["scenario_id"])
    trajectory_path = records / f"{scenario_id}_trajectory.npz"
    temporary = trajectory_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **{key: np.asarray(value) for key, value in trajectory.items()})
    temporary.replace(trajectory_path)
    record = {
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
        "summary": dict(episode),
        "episode": dict(episode),
        "agents": [dict(row) for row in agents],
        "events": [dict(row) for row in events],
        "shared_identity": dict(shared_identity),
        "trajectory_file": trajectory_path.name,
        "trajectory_sha256": sha256_file(trajectory_path),
    }
    atomic_json(records / f"{scenario_id}.json", record)


def collect(output: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    team: list[dict[str, Any]] = []
    agents: list[dict[str, Any]] = []
    records = output / "episode_records"
    for path in sorted(records.glob("DEV_LR_*.json")):
        if path.name.endswith("_SOFTWARE_ERROR.json"):
            continue
        payload = load_json(path)
        if "summary" not in payload:
            continue
        team.append(dict(payload["summary"]))
        agents.extend(dict(row) for row in payload.get("agents", []))
    team.sort(key=lambda row: str(row["scenario_id"]))
    agents.sort(key=lambda row: (str(row["scenario_id"]), int(row["agent_id"])))
    return team, agents


def summarize_scope(scope: str, members: Sequence[Mapping[str, Any]], agents: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scenario_ids = {str(row["scenario_id"]) for row in members}
    scoped_agents = [row for row in agents if str(row["scenario_id"]) in scenario_ids]
    return {
        "scope": scope,
        "n": len(members),
        "team_success_rate": float(np.mean([bool(row["team_success"]) for row in members])),
        "collision_rate": float(np.mean([bool(row["collision"]) for row in members])),
        "timeout_rate": float(np.mean([bool(row["timeout"]) for row in members])),
        "agent_completion_rate": float(np.mean([bool(row["agent_terminal_completed"]) for row in scoped_agents])),
        "mean_upper_invocations": float(np.mean([float(row["upper_pipeline_invocation_count"]) for row in members])),
        "mean_total_compute_ms": float(np.mean([float(row["total_online_algorithm_compute_ms"]) for row in members])),
    }


def finalize(config: Mapping[str, Any], output: Path) -> None:
    _, manifest_path, _ = artifact_paths(config)
    manifest = load_json(manifest_path)
    expected = selected_entries(config, manifest)
    expected_ids = [str(row["scenario_id"]) for row in expected]
    team, agents = collect(output)
    write_csv(output / "development_team_results.csv", team)
    write_csv(output / "development_agent_results.csv", agents)
    summaries = [summarize_scope("overall", team, agents)] if team else []
    for stage in STAGE_ORDER:
        members = [row for row in team if row["stage"] == stage]
        if members:
            summaries.append(summarize_scope(stage, members, agents))
    families = sorted({str(row["family"]) for row in team})
    for family in families:
        members = [row for row in team if row["family"] == family]
        summaries.append(summarize_scope(f"family:{family}", members, agents))
    write_csv(output / "development_summary.csv", summaries)
    observed_ids = [str(row["scenario_id"]) for row in team]
    errors = list((output / "episode_records").glob("*_SOFTWARE_ERROR.json"))
    complete = len(observed_ids) == len(expected_ids) and set(observed_ids) == set(expected_ids)
    atomic_json(
        output / "development_reconciliation.json",
        {
            "status": "PASS" if complete and not errors else "IN_PROGRESS",
            "completed_scenarios": len(observed_ids),
            "expected_scenarios": len(expected_ids),
            "missing_scenario_ids": sorted(set(expected_ids) - set(observed_ids)),
            "unexpected_scenario_ids": sorted(set(observed_ids) - set(expected_ids)),
            "software_error_count": len(errors),
            "overall": next((row for row in summaries if row["scope"] == "overall"), None),
            "formal_data_used": False,
        },
    )


def smoke(config_path: Path) -> None:
    """Exercise one development scene without persisting performance output."""

    config = resolved_config(config_path)
    _, manifest_path, _ = artifact_paths(config)
    manifest = load_json(manifest_path)
    runtime = DevelopmentRuntime(config, manifest)
    contract = method_contract(config)
    entry = selected_entries(config, manifest)[0]
    episode, agents, trajectory, events, identity = evaluate_episode(
        runtime, config, entry, contract
    )
    positions = np.asarray(trajectory["positions"])
    checks = {
        "three_agents": len(agents) == 3,
        "trajectory_shape": positions.ndim == 3 and positions.shape[1:] == (3, 3),
        "trajectory_step_count": len(positions) == int(episode["steps"]) + 1,
        "replanning_disabled": int(episode["replanning_count"]) == 0,
        "finite_compute": np.isfinite(float(episode["total_online_algorithm_compute_ms"])),
        "identity_fields_present": set(identity) == {
            "initial_condition_hash",
            "candidate_bundle_hash",
            "preview_bundle_hash",
            "proposal_reconstruction_equivalent",
            "graph_schema_match",
            "preview_historical_gate_verified",
        },
        "event_count_valid": len(events) in {0, 3},
    }
    if not all(checks.values()):
        raise RuntimeError(f"ablation smoke failed: {checks}")
    print(
        json.dumps(
            json_ready(
            {
                "phase": "smoke",
                "status": "PASS",
                "method": contract["method"],
                "scenario_id": entry["scenario_id"],
                "checks": checks,
                "performance_persisted": False,
            }
            )
        ),
        flush=True,
    )


def run(
    config_path: Path,
    limit: int | None,
    shard_index: int | None,
    shard_count: int | None,
    defer_finalize: bool,
) -> None:
    config = resolved_config(config_path)
    _, manifest_path, output = artifact_paths(config)
    verify(config_path, config, manifest_path, output)
    manifest = load_json(manifest_path)
    runtime = DevelopmentRuntime(config, manifest)
    contract = method_contract(config)
    entries = selected_entries(config, manifest)
    completed = {str(row["scenario_id"]) for row in collect(output)[0]}
    indexed = list(enumerate(entries))
    if shard_count is not None:
        if shard_index is None or not 0 <= int(shard_index) < int(shard_count):
            raise ValueError("shard_index must be in [0, shard_count)")
        indexed = [(index, entry) for index, entry in indexed if index % int(shard_count) == int(shard_index)]
    pending = [entry for _, entry in indexed if str(entry["scenario_id"]) not in completed]
    if limit is not None:
        pending = pending[: int(limit)]
    for entry in pending:
        scenario_id = str(entry["scenario_id"])
        print(f"[long-range-ablation] start {contract['method']} {scenario_id}", flush=True)
        try:
            episode, agents, trajectory, events, shared_identity = evaluate_episode(
                runtime, config, entry, contract
            )
            save_episode(output, entry, episode, agents, trajectory, events, shared_identity)
            print(
                f"[long-range-ablation] complete {contract['method']} {scenario_id} "
                f"success={int(episode['team_success'])} collision={int(episode['collision'])} steps={episode['steps']}",
                flush=True,
            )
        except Exception as error:
            atomic_json(
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
            finalize(config, output)
    if not defer_finalize:
        finalize(config, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("smoke", "prepare", "run", "finalize"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    parser.add_argument("--defer-finalize", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    if args.phase == "smoke":
        smoke(config_path)
    elif args.phase == "prepare":
        prepare(config_path)
    elif args.phase == "run":
        run(config_path, args.limit, args.shard_index, args.shard_count, args.defer_finalize)
    else:
        config = resolved_config(config_path)
        _, _, output = artifact_paths(config)
        finalize(config, output)


if __name__ == "__main__":
    main()
