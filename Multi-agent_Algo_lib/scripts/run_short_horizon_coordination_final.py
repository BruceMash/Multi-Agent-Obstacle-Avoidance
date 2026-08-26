"""Re-run the frozen final method on 100 short-range obstacle-free 3-UAV tasks.

This is the paper's foundational local-coordination test.  It is deliberately
separate from the untouched long-range formal benchmark and cannot be used for
method selection.  The three task families are crossing, merge, and geometric
conflict; no static or dynamic external obstacle is present.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
import traceback
from collections import Counter
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT, ALGO_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from planning.online_runtime_instrumentation import (  # noqa: E402
    OnlineRuntimeRecorder,
    TimedPolicyProxy,
)
from scripts.evaluate_gat_v1_err_development import (  # noqa: E402
    METHOD_RERR_GAT,
    build_online_gat_plan_optimized,
    run_episode,
)
from scripts.evaluate_single_policy_multi_agent import (  # noqa: E402
    SinglePolicyMultiAgentEnv,
    build_policy_observations,
)
from scripts.run_long_range_contract_pilot import (  # noqa: E402
    save_record,
    summarize_record,
)
from scripts.run_long_range_development import DevelopmentRuntime  # noqa: E402
from planning.semi_structured_long_range_benchmark import stable_hash  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/evaluation/short_horizon_coordination_final.json"
PATTERNS = ("crossing", "merge", "conflict")
SOURCE_PATHS = (
    "Environment/frozen_sac_dmp_execution.py",
    "Environment/multi_agent_dmp_env.py",
    "Entity/KinematicModel.py",
    "Guidance/reference_point_proposal_demo.py",
    "planning/event_triggered_reference_reconstruction.py",
    "planning/heterogeneous_candidate_graph.py",
    "planning/historical_forcing_gate.py",
    "planning/online_runtime_instrumentation.py",
    "planning/policy_preview.py",
    "planning/pre_gat_closed_loop.py",
    "Multi-agent_Algo_lib/scripts/evaluate_gat_closed_loop.py",
    "Multi-agent_Algo_lib/scripts/evaluate_gat_v1_err_development.py",
    "Multi-agent_Algo_lib/scripts/finalize_long_range_development.py",
    "Multi-agent_Algo_lib/scripts/run_long_range_development.py",
    "Multi-agent_Algo_lib/scripts/run_short_horizon_coordination_final.py",
    "configs/evaluation/short_horizon_coordination_final.json",
)
DEVELOPMENT_SELECTION_FREEZE = "10_final_freeze/DEVELOPMENT_SELECTION_FREEZE.json"
DEVELOPMENT_EVIDENCE = (
    "08_development/D05_interaction_feasibility_mask_full200/development_reconciliation.json",
    "08_development/RERR_FP_F00_base_common_rerr_full200/development_reconciliation.json",
    "08_development/ABL_M4_Direct_SAC_DMP_full200/development_reconciliation.json",
    "08_development/ABL_M5_Proposal_SAC_DMP_full200/development_reconciliation.json",
    "08_development/ABL_M6_FP_SHEP_SAC_DMP_full200/development_reconciliation.json",
    "08_development/ABL_M7_OneShot_GAT_SAC_DMP_full200/development_reconciliation.json",
    "09_baseline_tuning/DWA_FS_FULL_S00/development_reconciliation.json",
    "09_baseline_tuning/DWA_SM_FULL_S05/development_reconciliation.json",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def paths(config: Mapping[str, Any]) -> tuple[Path, Path, Path, Path]:
    output = ROOT / config["artifact_root"] / config["output_subdir"]
    return (
        output,
        output / "SHORT_COORDINATION_MANIFEST.json",
        output / "SHORT_COORDINATION_FREEZE.json",
        ROOT / config["base_final_method_config"],
    )


def _base_geometry(pattern: str) -> tuple[np.ndarray, np.ndarray]:
    if pattern == "crossing":
        starts = np.asarray(
            [[0.55, 0.65, 1.05], [0.55, 2.25, 1.35], [0.55, 3.85, 1.15]],
            dtype=float,
        )
        goals = np.asarray(
            [[8.45, 3.85, 1.15], [8.45, 2.25, 1.35], [8.45, 0.65, 1.05]],
            dtype=float,
        )
    elif pattern == "merge":
        starts = np.asarray(
            [[0.55, 0.50, 1.00], [0.55, 2.25, 1.35], [0.55, 4.00, 1.10]],
            dtype=float,
        )
        goals = np.asarray(
            [[8.45, 0.80, 1.10], [8.45, 2.25, 1.35], [8.45, 3.70, 1.00]],
            dtype=float,
        )
    elif pattern == "conflict":
        starts = np.asarray(
            [[0.55, 2.05, 1.20], [8.45, 2.50, 1.20], [4.50, 0.40, 1.20]],
            dtype=float,
        )
        goals = np.asarray(
            [[8.45, 2.05, 1.20], [0.55, 2.50, 1.20], [4.50, 4.10, 1.20]],
            dtype=float,
        )
    else:
        raise ValueError(pattern)
    return starts, goals


def generate_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    count = int(config["scenario_count"])
    lower, upper = np.asarray(config["workspace_bounds"], dtype=float)
    entries: list[dict[str, Any]] = []
    for index in range(count):
        pattern = PATTERNS[index % len(PATTERNS)]
        seed = int(config["seed_base"]) + index
        rng = np.random.default_rng(seed)
        starts, goals = _base_geometry(pattern)
        starts = starts.copy()
        goals = goals.copy()
        starts[:, 0] += rng.uniform(-0.10, 0.10, 3)
        goals[:, 0] += rng.uniform(-0.10, 0.10, 3)
        starts[:, 1] += rng.uniform(-0.08, 0.08, 3)
        goals[:, 1] += rng.uniform(-0.08, 0.08, 3)
        shared_z = rng.uniform(-0.06, 0.06, 3)
        starts[:, 2] += shared_z
        goals[:, 2] += shared_z
        if np.any(starts <= lower) or np.any(starts >= upper) or np.any(goals <= lower) or np.any(goals >= upper):
            raise RuntimeError("short coordination endpoint outside workspace")
        start_min = min(
            float(np.linalg.norm(starts[a] - starts[b])) for a in range(3) for b in range(a + 1, 3)
        )
        goal_min = min(
            float(np.linalg.norm(goals[a] - goals[b])) for a in range(3) for b in range(a + 1, 3)
        )
        if min(start_min, goal_min) <= 0.6:
            raise RuntimeError("short coordination endpoint separation invalid")
        relative = np.vstack([starts, goals])
        relative = relative - np.mean(relative, axis=0, keepdims=True)
        identity = {
            "starts": np.round(starts, 9).tolist(),
            "goals": np.round(goals, 9).tolist(),
            "static_obstacles": [],
            "dynamic_obstacles": [],
        }
        entry = {
            "schema_version": "short_horizon_coordination_scene_v1",
            "scenario_id": f"SHORT_COORD_{index:03d}",
            "scenario_index": index,
            "seed": seed,
            "stage": "short_horizon_local_coordination",
            "family": pattern,
            "task_pattern": pattern,
            "starts": starts.tolist(),
            "initial_velocities": np.zeros((3, 3), dtype=float).tolist(),
            "goals": goals.tolist(),
            "workspace_bounds": copy.deepcopy(config["workspace_bounds"]),
            "boundary_mode": "boundary_free",
            "dt": float(config["dt"]),
            "max_steps": int(config["max_steps"]),
            "static_obstacles": [],
            "dynamic_obstacles": [],
            "dynamic_obstacle_trajectories": [],
            "task_distances_m": np.linalg.norm(goals - starts, axis=1).tolist(),
            "minimum_start_separation_m": start_min,
            "minimum_goal_separation_m": goal_min,
            "geometry_fingerprint": stable_hash(identity),
            "dynamic_track_fingerprint": stable_hash([]),
            "translation_invariant_fingerprint": stable_hash(
                {"relative_endpoints": np.round(relative, 9).tolist(), "pattern": pattern}
            ),
        }
        entry["environment_fingerprint"] = stable_hash(entry)
        entries.append(entry)
    manifest = {
        "schema_version": "short_horizon_coordination_manifest_v1",
        "scenario_count": count,
        "pattern_counts": dict(Counter(row["task_pattern"] for row in entries)),
        "entries": entries,
    }
    manifest["manifest_sha256"] = stable_hash(manifest)
    return manifest


def resolved_method_config(config: Mapping[str, Any]) -> dict[str, Any]:
    base = load_json(ROOT / config["base_final_method_config"])
    resolved = copy.deepcopy(base)
    resolved.update(
        {
            "configuration_id": "FINAL_D05_SHORT_COORDINATION_RECHECK",
            "max_steps": int(config["max_steps"]),
            "workspace_bounds": copy.deepcopy(config["workspace_bounds"]),
        }
    )
    return resolved


def _require_closed_development(config: Mapping[str, Any]) -> dict[str, Any]:
    root = ROOT / config["artifact_root"]
    evidence: dict[str, Any] = {}
    selection_path = root / DEVELOPMENT_SELECTION_FREEZE
    if not selection_path.is_file():
        raise FileNotFoundError("development selection freeze is missing")
    selection = load_json(selection_path)
    if selection.get("status") != "DEVELOPMENT_CLOSED_BEFORE_SHORT_AND_FORMAL":
        raise RuntimeError("development selection freeze is not closed")
    if bool(selection.get("formal_data_used_for_training_tuning_or_selection", True)):
        raise RuntimeError("development selection freeze reports formal-data contamination")
    evidence[DEVELOPMENT_SELECTION_FREEZE] = {
        "sha256": sha256_file(selection_path),
        "status": selection["status"],
        "development_90_percent_target_met": selection["development_90_percent_target_met"],
        "matched_baseline_target_met": selection["proposed_matches_or_exceeds_strongest_matched_baseline"],
    }
    for relative in DEVELOPMENT_EVIDENCE:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"development is not closed: {relative}")
        payload = load_json(path)
        if payload.get("status") != "PASS" or int(payload.get("completed_scenarios", 0)) != 200:
            raise RuntimeError(f"development evidence is incomplete: {relative}")
        if int(payload.get("software_error_count", 0)) != 0:
            raise RuntimeError(f"development evidence contains software errors: {relative}")
        evidence[relative] = {
            "sha256": sha256_file(path),
            "overall": payload.get("overall"),
        }
    return evidence


def prepare(config_path: Path) -> None:
    config = load_json(config_path)
    output, manifest_path, freeze_path, method_path = paths(config)
    records = output / "episode_records"
    if list(records.glob("SHORT_COORD_*.json")):
        raise RuntimeError("short coordination records already exist")
    if (ROOT / config["artifact_root"] / "11_formal_manifest/FINAL_LONG_RANGE_MANIFEST.json").exists():
        raise RuntimeError("short coordination must be frozen and run before the untouched formal manifest")
    evidence = _require_closed_development(config)
    output.mkdir(parents=True, exist_ok=True)
    manifest = generate_manifest(config)
    atomic_json(manifest_path, manifest)
    method = resolved_method_config(config)
    atomic_json(output / "resolved_final_method_config.json", method)
    checkpoint_hashes = {
        "sac": sha256_file(ROOT / method["sac_checkpoint"]),
        "gat": sha256_file(ROOT / method["gat_checkpoint"]),
    }
    if checkpoint_hashes["sac"] != method["sac_checkpoint_sha256_expected"]:
        raise RuntimeError("SAC checkpoint mismatch")
    if checkpoint_hashes["gat"] != method["gat_checkpoint_sha256_expected"]:
        raise RuntimeError("GAT checkpoint mismatch")
    historical = {
        "historical_source": "artifacts/gat_v1_ia_formal_closed_loop/20260818_161930/FINAL_REPORT.md",
        "historical_open_success": "19/20 (95.0%)",
        "historical_method": "one-shot GAT-V1 + frozen 56-ray SAC-DMP",
        "historical_gat_sha256": "bf224c91731f0dc3f3d43cfb7b4f9da70b58a789b8b4c409226a087ff92b36d8",
        "historical_sac_sha256": "0c3595f738b2f2f2b7e88fc90479e6d525c986a216ab2d1b2eecc139833ad3d5",
        "historical_actor_observation_dim": 122,
        "historical_sensor_directions": 56,
        "historical_execution": "one_shot_phase_preserving_handoff_no_ERR",
        "historical_collision_contract": "discrete obstacle/inter-agent collision; boundary-free",
        "historical_success_contract": "all three UAVs reach terminal goals within 0.3 m",
        "historical_uav_peer_radius_m": 0.3,
        "historical_inter_agent_collision_threshold_m": 0.6,
        "historical_horizon_steps": 220,
        "current_method_differs": True,
        "reuse_historical_95_percent_as_current_result": False,
        "rerun_required": True,
    }
    atomic_json(output / "historical_short_coordination_audit.json", historical)
    freeze = {
        "schema_version": "short_horizon_coordination_freeze_v1",
        "status": "FROZEN_AFTER_LONG_RANGE_DEVELOPMENT_BEFORE_SHORT_RECHECK",
        "created_at": datetime.now().astimezone().isoformat(),
        "configuration_sha256": sha256_file(config_path),
        "base_method_configuration_sha256": sha256_file(method_path),
        "resolved_method_configuration_sha256": sha256_file(output / "resolved_final_method_config.json"),
        "manifest_file_sha256": sha256_file(manifest_path),
        "manifest_semantic_sha256": manifest["manifest_sha256"],
        "development_evidence": evidence,
        "checkpoint_sha256": checkpoint_hashes,
        "source_sha256": {relative: sha256_file(ROOT / relative) for relative in SOURCE_PATHS},
        "scenario_count": int(config["scenario_count"]),
        "performance_selection_allowed": False,
        "long_range_formal_data_used": False,
        "long_range_formal_manifest_exists_at_freeze": False,
    }
    atomic_json(freeze_path, freeze)
    print(json.dumps({"phase": "prepare", "status": "PASS", "scenario_count": len(manifest["entries"])}), flush=True)


def verify(config_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    config = load_json(config_path)
    output, manifest_path, freeze_path, method_path = paths(config)
    freeze = load_json(freeze_path)
    mismatches: list[str] = []
    if sha256_file(config_path) != freeze["configuration_sha256"]:
        mismatches.append("config")
    if sha256_file(method_path) != freeze["base_method_configuration_sha256"]:
        mismatches.append("base_method")
    if sha256_file(output / "resolved_final_method_config.json") != freeze["resolved_method_configuration_sha256"]:
        mismatches.append("resolved_method")
    if sha256_file(manifest_path) != freeze["manifest_file_sha256"]:
        mismatches.append("manifest")
    for relative, expected in freeze["source_sha256"].items():
        if sha256_file(ROOT / relative) != expected:
            mismatches.append(relative)
    if mismatches:
        raise RuntimeError(f"short coordination freeze mismatch: {mismatches}")
    return config, load_json(manifest_path), load_json(output / "resolved_final_method_config.json"), output


class ShortCoordinationEnvironmentBuilder:
    def __init__(self, manifest: Mapping[str, Any], method_config: Mapping[str, Any]) -> None:
        self.entries = {str(row["scenario_id"]): row for row in manifest["entries"]}
        self.method_config = method_config

    def __call__(self, *, config: Any, scenario: str, seed: int, peer_radius: float) -> tuple[Any, dict[str, Any]]:
        entry = self.entries[str(scenario)]
        if int(seed) != int(entry["seed"]):
            raise ValueError("short coordination seed mismatch")
        kwargs = copy.deepcopy(config.build_core_env_kwargs())
        kwargs["dynamics_config"]["maximum_speed_norm"] = float(
            self.method_config["maximum_speed_norm_mps"]
        )
        kwargs["env_config"].update(
            {
                "workspace_bounds": tuple(tuple(row) for row in entry["workspace_bounds"]),
                "max_steps": int(entry["max_steps"]),
                "randomize_start_goal": False,
                "start_position_bounds": None,
                "goal_position_bounds": None,
                "min_start_goal_distance": 0.0,
                "inter_agent_influence_distance": float(self.method_config["sensor"]["range_m"]),
                "nearest_agent_observation_count": 2,
                "peer_state_observation_mode": "local_anonymous_ally_block",
                "peer_state_observation_range": float(self.method_config["sensor"]["range_m"]),
            }
        )
        env = SinglePolicyMultiAgentEnv(
            **kwargs,
            observation_mode="peer_spheres",
            peer_radius=float(peer_radius),
            include_boundaries_in_sensor=False,
            terminate_on_boundary_collision=False,
        )
        env.reset(
            seed=int(seed),
            options={
                "starts": np.asarray(entry["starts"], dtype=float),
                "goals": np.asarray(entry["goals"], dtype=float),
                "static_obstacles": [],
                "dynamic_obstacles": [],
            },
        )
        observed = build_policy_observations(env)
        if observed.shape != (3, int(self.method_config["sensor"]["actor_observation_dim"])):
            raise RuntimeError(f"short actor observation mismatch: {observed.shape}")
        return env, {
            "stage": entry["stage"],
            "family": entry["family"],
            "task_pattern": entry["task_pattern"],
            "scenario_id": entry["scenario_id"],
            "environment_fingerprint": entry["environment_fingerprint"],
            "geometry_fingerprint": entry["geometry_fingerprint"],
            "dynamic_track_fingerprint": entry["dynamic_track_fingerprint"],
            "boundary_mode": entry["boundary_mode"],
            "static_obstacle_count": 0,
            "dynamic_obstacle_count": 0,
            "scene_reconstruction_match": True,
            "policy_observation_shape_at_reset": list(observed.shape),
            "sensor_direction_count_at_reset": int(env.sensors[0].ray_directions.shape[0]),
            "peer_information_mode": env.env_config.peer_state_observation_mode,
            "peer_information_range_m": env.env_config.peer_state_observation_range,
        }


def collect(output: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((output / "episode_records").glob("SHORT_COORD_*.json")):
        if path.name.endswith("_SOFTWARE_ERROR.json"):
            continue
        payload = load_json(path)
        row = dict(payload["summary"])
        episode = payload["episode"]
        row.update(
            {
                "team_success": bool(episode["team_success"]),
                "collision": bool(episode["collision"]),
                "obstacle_collision": bool(episode["obstacle_collision"]),
                "inter_agent_collision": bool(episode["inter_agent_collision"]),
                "timeout": bool(episode["timeout"]),
                "agent_completion_rate": float(episode["agent_completion_rate"]),
                "completion_time_s": episode.get("completion_time_s"),
                "team_path_length_m": float(episode["team_path_length_m"]),
                "minimum_inter_agent_distance_m": float(episode["minimum_inter_agent_distance_m"]),
                "total_online_algorithm_compute_ms": float(episode["total_online_algorithm_compute_ms"]),
            }
        )
        rows.append(row)
    return rows


def run(config_path: Path, *, shard_index: int, shard_count: int, limit: int | None) -> None:
    config, manifest, method_config, output = verify(config_path)
    runtime = DevelopmentRuntime(method_config, manifest)
    workspace = tuple(tuple(float(value) for value in row) for row in config["workspace_bounds"])
    multi_config = replace(
        runtime.multi_config,
        workspace_bounds=workspace,
        max_steps=int(config["max_steps"]),
        randomize_start_goal=False,
        start_position_bounds=None,
        goal_position_bounds=None,
        min_start_goal_distance=0.0,
    )
    builder = ShortCoordinationEnvironmentBuilder(manifest, method_config)
    completed = {str(row["scenario_id"]) for row in collect(output)}
    entries = [
        entry for index, entry in enumerate(manifest["entries"])
        if index % int(shard_count) == int(shard_index) and str(entry["scenario_id"]) not in completed
    ]
    if limit is not None:
        entries = entries[: int(limit)]
    for entry in entries:
        sid = str(entry["scenario_id"])
        print(f"[short-coordination] start {sid} {entry['task_pattern']}", flush=True)
        recorder = OnlineRuntimeRecorder()
        policy = TimedPolicyProxy(runtime.policy, recorder)
        try:
            with recorder.instrument_dmp(), recorder.scoped_context(
                evaluation_block="short_horizon_local_coordination",
                configuration_id=method_config["configuration_id"],
                scenario_id=sid,
                seed=int(entry["seed"]),
                family=entry["family"],
                stage=entry["stage"],
                method=METHOD_RERR_GAT,
            ):
                episode, agents, events, triggers, extra = run_episode(
                    config=runtime.eval_config,
                    settings=runtime.settings,
                    multi_config=multi_config,
                    policy=policy,
                    gat_model=runtime.gat_model,
                    gat_device=runtime.gat_device,
                    method=METHOD_RERR_GAT,
                    scenario=sid,
                    seed=int(entry["seed"]),
                    environment_builder=builder,
                    runtime_recorder=recorder,
                    upper_plan_builder=build_online_gat_plan_optimized,
                )
            summary = summarize_record(entry, episode, events, triggers, extra["path_rows"])
            summary.update(
                {
                    "configuration_id": method_config["configuration_id"],
                    "method": METHOD_RERR_GAT,
                    "team_success": bool(episode["team_success"]),
                    "collision": bool(episode["collision"]),
                    "timeout": bool(episode["timeout"]),
                    "performance_used_for_selection": False,
                }
            )
            save_record(output, entry, episode, agents, events, triggers, extra, summary)
            print(
                f"[short-coordination] complete {sid} success={int(episode['team_success'])} collision={int(episode['collision'])} steps={episode['steps']}",
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


def _load_trajectory(output: Path, scenario_id: str) -> np.ndarray:
    path = output / "episode_records" / f"{scenario_id}_trajectory.npz"
    with np.load(path, allow_pickle=False) as arrays:
        flat = np.asarray(arrays["positions"], dtype=float)
        steps = np.asarray(arrays["steps"], dtype=int)
        agents = np.asarray(arrays["agent_ids"], dtype=int)
    unique_steps = np.unique(steps)
    positions = np.empty((len(unique_steps), 3, 3), dtype=float)
    for frame_index, step in enumerate(unique_steps):
        for agent_id in range(3):
            matches = np.flatnonzero((steps == step) & (agents == agent_id))
            if len(matches) != 1:
                raise RuntimeError("trajectory is not a complete step-agent grid")
            positions[frame_index, agent_id] = flat[int(matches[0])]
    return positions


def _peer_collision_recheck(positions: np.ndarray, threshold: float = 0.6) -> tuple[bool, float]:
    minimum = float("inf")
    collision = False
    for frame in positions[1:]:
        for left in range(3):
            for right in range(left + 1, 3):
                distance = float(np.linalg.norm(frame[left] - frame[right]))
                minimum = min(minimum, distance)
                collision = collision or distance <= float(threshold)
    return collision, minimum


def _plot_representatives(output: Path, rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]) -> None:
    entry_by_id = {str(row["scenario_id"]): row for row in manifest["entries"]}
    representatives: list[dict[str, Any]] = []
    for pattern in PATTERNS:
        successes = [row for row in rows if row["task_pattern"] == pattern and bool(row["team_success"])]
        if not successes:
            continue
        median = float(np.median([int(row["steps"]) for row in successes]))
        chosen = min(successes, key=lambda row: (abs(int(row["steps"]) - median), str(row["scenario_id"])))
        representatives.append({"task_pattern": pattern, "scenario_id": chosen["scenario_id"], "steps": chosen["steps"]})
    write_csv(output / "representative_trajectory_manifest.csv", representatives)
    if not representatives:
        return
    figure = plt.figure(figsize=(4.9 * len(representatives), 4.4), constrained_layout=True)
    colors = ("#0072B2", "#D55E00", "#009E73")
    labels = {"crossing": "Crossing", "merge": "Merge", "conflict": "Conflict"}
    for panel, representative in enumerate(representatives, start=1):
        axis = figure.add_subplot(1, len(representatives), panel, projection="3d")
        sid = str(representative["scenario_id"])
        entry = entry_by_id[sid]
        positions = _load_trajectory(output, sid)
        starts = np.asarray(entry["starts"], dtype=float)
        goals = np.asarray(entry["goals"], dtype=float)
        for agent_id, color in enumerate(colors):
            axis.plot(
                positions[:, agent_id, 0], positions[:, agent_id, 1], positions[:, agent_id, 2],
                color=color, linewidth=1.8, label=f"UAV {agent_id + 1}",
            )
            axis.scatter(*starts[agent_id], color=color, marker="o", s=34, depthshade=False)
            axis.scatter(*goals[agent_id], color=color, marker="*", s=90, depthshade=False)
        axis.set_title(labels[str(representative["task_pattern"])])
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        axis.set_zlabel("z (m)")
        axis.set_xlim(0.0, 9.0)
        axis.set_ylim(0.0, 4.5)
        axis.set_zlim(0.0, 2.4)
        axis.view_init(elev=24, azim=-62)
        axis.grid(True, alpha=0.25)
        if panel == 1:
            axis.legend(loc="upper left", frameon=False)
    png = output / "short_horizon_local_coordination_3d.png"
    pdf = output / "short_horizon_local_coordination_3d.pdf"
    figure.savefig(png, dpi=600, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)


def finalize(config_path: Path) -> None:
    config, manifest, _, output = verify(config_path)
    rows = collect(output)
    expected_ids = {str(entry["scenario_id"]) for entry in manifest["entries"]}
    observed_ids = {str(row["scenario_id"]) for row in rows}
    error_paths = list((output / "episode_records").glob("*_SOFTWARE_ERROR.json"))
    collision_mismatches: list[str] = []
    recheck_rows: list[dict[str, Any]] = []
    for row in rows:
        positions = _load_trajectory(output, str(row["scenario_id"]))
        peer, minimum = _peer_collision_recheck(positions)
        if peer != bool(row["inter_agent_collision"]):
            collision_mismatches.append(str(row["scenario_id"]))
        recheck_rows.append(
            {
                "scenario_id": row["scenario_id"],
                "online_inter_agent_collision": bool(row["inter_agent_collision"]),
                "recomputed_inter_agent_collision": peer,
                "recomputed_minimum_inter_agent_distance_m": minimum,
                "match": peer == bool(row["inter_agent_collision"]),
            }
        )
    write_csv(output / "short_coordination_results.csv", rows)
    write_csv(output / "short_coordination_collision_recheck.csv", recheck_rows)
    summary_rows: list[dict[str, Any]] = []
    for scope in ("overall", *PATTERNS):
        members = rows if scope == "overall" else [row for row in rows if row["task_pattern"] == scope]
        if not members:
            continue
        success_rows = [row for row in members if bool(row["team_success"])]
        summary_rows.append(
            {
                "scope": scope,
                "n": len(members),
                "team_success_count": sum(bool(row["team_success"]) for row in members),
                "team_success_rate": float(np.mean([bool(row["team_success"]) for row in members])),
                "collision_rate": float(np.mean([bool(row["collision"]) for row in members])),
                "inter_agent_collision_rate": float(np.mean([bool(row["inter_agent_collision"]) for row in members])),
                "obstacle_collision_rate": float(np.mean([bool(row["obstacle_collision"]) for row in members])),
                "timeout_rate": float(np.mean([bool(row["timeout"]) for row in members])),
                "agent_completion_rate": float(np.mean([float(row["agent_completion_rate"]) for row in members])),
                "mean_success_completion_time_s": float(np.mean([float(row["completion_time_s"]) for row in success_rows])) if success_rows else None,
                "mean_total_compute_ms": float(np.mean([float(row["total_online_algorithm_compute_ms"]) for row in members])),
            }
        )
    write_csv(output / "short_coordination_summary.csv", summary_rows)
    complete = (
        len(rows) == int(config["scenario_count"])
        and observed_ids == expected_ids
        and not error_paths
        and not collision_mismatches
    )
    reconciliation = {
        "schema_version": "short_horizon_coordination_reconciliation_v1",
        "status": "PASS" if complete else "IN_PROGRESS",
        "expected_scenario_count": int(config["scenario_count"]),
        "completed_scenario_count": len(rows),
        "missing_scenario_ids": sorted(expected_ids - observed_ids),
        "unexpected_scenario_ids": sorted(observed_ids - expected_ids),
        "software_error_files": [path.name for path in error_paths],
        "collision_recheck_mismatches": collision_mismatches,
        "all_external_obstacle_counts_zero": all(
            not entry["static_obstacles"] and not entry["dynamic_obstacles"] for entry in manifest["entries"]
        ),
        "pattern_counts": dict(Counter(row["task_pattern"] for row in rows)),
        "overall": next((row for row in summary_rows if row["scope"] == "overall"), None),
        "used_for_long_range_method_selection": False,
        "long_range_formal_data_used": False,
    }
    atomic_json(output / "short_coordination_reconciliation.json", reconciliation)
    _plot_representatives(output, rows, manifest)
    if complete:
        atomic_json(
            output / "short_coordination_complete.json",
            {
                **reconciliation,
                "completed_at": datetime.now().astimezone().isoformat(),
                "paper_role": config["performance_role"],
            },
        )
    print(json.dumps({"phase": "finalize", "status": reconciliation["status"], "completed": len(rows)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "run", "finalize"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config_path = args.config.resolve()
    if args.phase == "prepare":
        prepare(config_path)
    elif args.phase == "run":
        if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
            raise ValueError("invalid shard selection")
        run(
            config_path,
            shard_index=int(args.shard_index),
            shard_count=int(args.shard_count),
            limit=args.limit,
        )
    else:
        finalize(config_path)


if __name__ == "__main__":
    main()
