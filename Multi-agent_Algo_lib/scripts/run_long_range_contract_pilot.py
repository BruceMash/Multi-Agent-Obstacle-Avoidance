"""Freeze and run the non-tuning 40-scene long-range contract pilot.

The pilot is deliberately separated from training, development, and formal
evaluation.  Categorical outcomes are retained for provenance but cannot be
used to alter the scene grammar or select a parameter/checkpoint.
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
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = Path(__file__).resolve().parent
for search_path in (REPO_ROOT, ALGO_ROOT, SCRIPTS_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from Entity.dynamic_obstacles import MovingSphereObstacle  # noqa: E402
from Entity.static_obstacles import (  # noqa: E402
    AxisAlignedBoxObstacle,
    StaticCylinderObstacle,
)
from planning.gat.stage1_training import load_model_checkpoint, resolve_device  # noqa: E402
from planning.online_runtime_instrumentation import (  # noqa: E402
    OnlineRuntimeRecorder,
    TimedPolicyProxy,
)
from planning.semi_structured_long_range_benchmark import (  # noqa: E402
    LONG_RANGE_MAX_STEPS,
    SENSOR_DIRECTION_COUNT,
    SENSOR_RANGE_M,
    STAGE_ORDER,
    WORKSPACE_BOUNDS,
    generate_scenario_manifest,
    json_ready,
    stable_hash,
    validate_scenario_manifest,
)
from scripts.evaluate_gat_closed_loop import (  # noqa: E402
    _model_hash,
    _policy_parameter_sha256,
)
from scripts.evaluate_gat_v1_err_development import (  # noqa: E402
    METHOD_RERR_GAT,
    build_online_gat_plan_optimized,
    run_episode,
)
from planning.event_triggered_reference_reconstruction import (  # noqa: E402
    EVENT_EMERGENCY_REPROPOSAL,
    EVENT_NORMAL_REPROPOSAL,
    EVENT_REFERENCE_COMPLETION_REPROPOSAL,
)
from scripts.evaluate_single_policy_aligned_multi_agent import (  # noqa: E402
    build_single_distribution_multi_config,
)
from scripts.evaluate_single_policy_multi_agent import (  # noqa: E402
    SinglePolicyMultiAgentEnv,
    build_policy_observations,
)
from scripts.evaluate_temporary_reference_interface import _load_policy  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs/evaluation/semi_structured_long_range_contract_pilot.json"
SOURCE_PATHS = (
    "planning/semi_structured_long_range_benchmark.py",
    "Environment/multi_agent_dmp_env.py",
    "Entity/KinematicModel.py",
    "Environment/frozen_sac_dmp_execution.py",
    "planning/event_triggered_reference_reconstruction.py",
    "planning/historical_forcing_gate.py",
    "planning/heterogeneous_candidate_graph.py",
    "planning/policy_preview.py",
    "planning/pre_gat_closed_loop.py",
    "Multi-agent_Algo_lib/scripts/evaluate_gat_v1_err_development.py",
    "Multi-agent_Algo_lib/scripts/run_long_range_contract_pilot.py",
    "configs/evaluation/semi_structured_long_range_contract_pilot.json",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_ready(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(json_ready(value), ensure_ascii=False, separators=(",", ":"))


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
            writer.writerow({key: csv_value(row.get(key)) for key in fields})
    temporary.replace(path)


def obstacle_from_spec(spec: Mapping[str, Any]) -> Any:
    kind = str(spec["type"])
    if kind == "box":
        return AxisAlignedBoxObstacle(
            center=np.asarray(spec["center"], dtype=float),
            half_extents=np.asarray(spec["half_extents"], dtype=float),
            safety_margin=float(spec.get("safety_margin", 0.0)),
        )
    if kind == "cylinder":
        return StaticCylinderObstacle(
            center=np.asarray(spec["center"], dtype=float),
            radius=float(spec["radius"]),
            half_height=float(spec["half_height"]),
            safety_margin=float(spec.get("safety_margin", 0.0)),
        )
    if kind == "moving_sphere_constant_translation":
        return MovingSphereObstacle(
            center=np.asarray(spec["center"], dtype=float),
            radius=float(spec["radius"]),
            velocity=np.asarray(spec["velocity"], dtype=float),
            safety_margin=float(spec.get("safety_margin", 0.0)),
            bounds=None,
        )
    raise ValueError(f"unsupported long-range obstacle type: {kind}")


def obstacles_from_entry(entry: Mapping[str, Any]) -> tuple[list[Any], list[Any]]:
    return (
        [obstacle_from_spec(row) for row in entry["static_obstacles"]],
        [obstacle_from_spec(row) for row in entry["dynamic_obstacles"]],
    )


class LongRangeManifestEnvironmentBuilder:
    def __init__(self, manifest: Mapping[str, Any], pilot_config: Mapping[str, Any]) -> None:
        self.entries = {str(row["scenario_id"]): row for row in manifest["entries"]}
        self.pilot_config = pilot_config

    def __call__(
        self, *, config: Any, scenario: str, seed: int, peer_radius: float
    ) -> tuple[Any, dict[str, Any]]:
        entry = self.entries[str(scenario)]
        if int(seed) != int(entry["seed"]):
            raise ValueError("scenario seed does not match frozen pilot manifest")
        static, dynamic = obstacles_from_entry(entry)
        kwargs = copy.deepcopy(config.build_core_env_kwargs())
        kwargs["dynamics_config"]["maximum_speed_norm"] = float(
            self.pilot_config["maximum_speed_norm_mps"]
        )
        kwargs["env_config"].update(
            {
                "workspace_bounds": tuple(tuple(row) for row in WORKSPACE_BOUNDS),
                "max_steps": int(self.pilot_config["max_steps"]),
                "randomize_start_goal": False,
                "start_position_bounds": None,
                "goal_position_bounds": None,
                "min_start_goal_distance": 0.0,
                "inter_agent_influence_distance": float(SENSOR_RANGE_M),
                "nearest_agent_observation_count": 2,
                "peer_state_observation_mode": "local_anonymous_ally_block",
                "peer_state_observation_range": float(SENSOR_RANGE_M),
            }
        )
        env = SinglePolicyMultiAgentEnv(
            **kwargs,
            observation_mode="peer_spheres",
            peer_radius=float(peer_radius),
            include_boundaries_in_sensor=True,
            terminate_on_boundary_collision=True,
        )
        env.reset(
            seed=int(seed),
            options={
                "starts": np.asarray(entry["starts"], dtype=float),
                "goals": np.asarray(entry["goals"], dtype=float),
                "static_obstacles": static,
                "dynamic_obstacles": dynamic,
            },
        )
        self._validate_reconstruction(env, entry)
        policy_observation_shape = list(build_policy_observations(env).shape)
        metadata = {
            "stage": entry["stage"],
            "family": entry["family"],
            "task_pattern": entry["task_pattern"],
            "scenario_id": entry["scenario_id"],
            "environment_fingerprint": entry["environment_fingerprint"],
            "geometry_fingerprint": entry["geometry_fingerprint"],
            "dynamic_track_fingerprint": entry["dynamic_track_fingerprint"],
            "boundary_mode": entry["boundary_mode"],
            "static_obstacle_count": len(static),
            "dynamic_obstacle_count": len(dynamic),
            "scene_reconstruction_match": True,
            "policy_observation_shape_at_reset": policy_observation_shape,
            "sensor_direction_count_at_reset": int(
                np.asarray(env.latest_sensor_packets[0].current_scan).size
            ),
            "peer_information_mode": env.env_config.peer_state_observation_mode,
            "peer_information_range_m": env.env_config.peer_state_observation_range,
        }
        return env, metadata

    @staticmethod
    def _validate_reconstruction(env: Any, entry: Mapping[str, Any]) -> None:
        if not np.array_equal(np.asarray(env.starts), np.asarray(entry["starts"])):
            raise RuntimeError("start reconstruction mismatch")
        if not np.array_equal(np.asarray(env.goals), np.asarray(entry["goals"])):
            raise RuntimeError("goal reconstruction mismatch")
        if len(env.static_obstacles) != len(entry["static_obstacles"]):
            raise RuntimeError("static obstacle reconstruction mismatch")
        if len(env.dynamic_obstacles) != len(entry["dynamic_obstacles"]):
            raise RuntimeError("dynamic obstacle reconstruction mismatch")
        for obstacle, spec, track in zip(
            env.dynamic_obstacles,
            entry["dynamic_obstacles"],
            entry["dynamic_obstacle_trajectories"],
            strict=True,
        ):
            center = np.asarray(spec["center"], dtype=float)
            velocity = np.asarray(spec["velocity"], dtype=float)
            if not np.array_equal(obstacle.center, center):
                raise RuntimeError("dynamic initial center mismatch")
            if not np.array_equal(obstacle.velocity, velocity):
                raise RuntimeError("dynamic velocity mismatch")
            check_steps = (0, 1, 149, LONG_RANGE_MAX_STEPS)
            for step in check_steps:
                predicted = center + step * float(entry["dt"]) * velocity
                if not np.allclose(predicted, np.asarray(track[step]), atol=1.0e-10, rtol=0.0):
                    raise RuntimeError("dynamic track reconstruction mismatch")


class PilotRuntime:
    def __init__(self, config: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
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
        settings = load_json(REPO_ROOT / config["base_execution_config"])
        settings.update(
            {
                "checkpoint": config["sac_checkpoint"],
                "checkpoint_sha256_expected": config["sac_checkpoint_sha256_expected"],
                "deterministic_policy": True,
                "num_agents": int(config["num_agents"]),
                "max_steps": int(config["max_steps"]),
                "dt": float(config["dt"]),
                "peer_radius": float(config["peer_radius"]),
                "temporary_reference": {
                    **settings["temporary_reference"],
                    "K_requested": int(config["top_k"]),
                    "reached_tolerance_m": float(config["handoff_threshold_m"]),
                    "replanning_enabled": True,
                },
                "proposal_config": copy.deepcopy(config["proposal_config"]),
            }
        )
        self.settings = settings
        self.policy, loaded = _load_policy(settings, self.multi_config)
        if loaded.resolve() != (REPO_ROOT / config["sac_checkpoint"]).resolve():
            raise RuntimeError("SAC loader resolved an unexpected checkpoint")
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


def artifact_paths(config: Mapping[str, Any]) -> tuple[Path, Path]:
    root = (REPO_ROOT / config["artifact_root"]).resolve()
    return root, root / "06_contract_pilot"


def prepare(config_path: Path) -> None:
    config = load_json(config_path)
    root, pilot_dir = artifact_paths(config)
    manifest_path = pilot_dir / "contract_pilot_manifest.json"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
    else:
        manifest = generate_scenario_manifest(
            counts_per_stage=int(config["pilot_scenarios_per_stage"]),
            seed_base=int(config["pilot_seed_base"]),
            prefix="PILOT_LR_",
        )
        atomic_json(manifest_path, manifest)
    validation = validate_scenario_manifest(manifest)
    if validation["status"] != "PASS":
        raise RuntimeError(f"pilot manifest validation failed: {validation}")
    atomic_json(pilot_dir / "contract_pilot_manifest_validation.json", validation)

    source_hashes = {path: sha256_file(REPO_ROOT / path) for path in SOURCE_PATHS}
    freeze = {
        "schema_version": "long_range_contract_pilot_freeze_v1",
        "purpose": "software/physics/lifecycle/observation/logging/runtime contract only",
        "performance_selection_authorized": False,
        "scenario_manifest_sha256": sha256_file(manifest_path),
        "scenario_semantic_sha256": manifest["manifest_sha256"],
        "configuration_sha256": sha256_file(config_path),
        "source_sha256": source_hashes,
        "checkpoint_sha256": {
            "sac": sha256_file(REPO_ROOT / config["sac_checkpoint"]),
            "gat": sha256_file(REPO_ROOT / config["gat_checkpoint"]),
        },
        "pilot_scenario_count": len(manifest["entries"]),
        "development_eligibility": False,
        "formal_eligibility": False,
        "result_count_at_freeze": 0,
    }
    if freeze["checkpoint_sha256"]["sac"] != config["sac_checkpoint_sha256_expected"]:
        raise RuntimeError("SAC checkpoint hash mismatch")
    if freeze["checkpoint_sha256"]["gat"] != config["gat_checkpoint_sha256_expected"]:
        raise RuntimeError("GAT checkpoint hash mismatch")
    atomic_json(pilot_dir / "contract_pilot_freeze.json", freeze)
    update_registry(root, manifest)
    print(json.dumps({"phase": "prepare", "scenario_count": len(manifest["entries"]), "status": "PASS"}), flush=True)


def update_registry(root: Path, manifest: Mapping[str, Any]) -> None:
    path = root / "00_context/ALL_USED_SCENE_REGISTRY.csv"
    existing: list[dict[str, Any]] = []
    if path.is_file() and path.stat().st_size:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            existing = list(csv.DictReader(handle))
    seen = {str(row["scene_id"]) for row in existing}
    for entry in manifest["entries"]:
        if entry["scenario_id"] in seen:
            continue
        existing.append(
            {
                "scene_id": entry["scenario_id"],
                "split": "contract_pilot",
                "stage": entry["stage"],
                "family": entry["family"],
                "task_pattern": entry["task_pattern"],
                "seed": entry["seed"],
                "geometry_fingerprint": entry["geometry_fingerprint"],
                "translation_invariant_fingerprint": entry["translation_invariant_fingerprint"],
                "dynamic_track_fingerprint": entry["dynamic_track_fingerprint"],
                "performance_episode_count": 1,
                "eligible_for_training": False,
                "eligible_for_development": False,
                "eligible_for_formal": False,
            }
        )
    write_csv(path, existing)


def summarize_record(
    entry: Mapping[str, Any], episode: Mapping[str, Any], events: Sequence[Mapping[str, Any]],
    triggers: Sequence[Mapping[str, Any]], path_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    speeds = np.asarray(
        [math.sqrt(row["vx_mps"] ** 2 + row["vy_mps"] ** 2 + row["vz_mps"] ** 2) for row in path_rows],
        dtype=float,
    )
    active_distances = np.asarray(
        [row["active_goal_distance_m"] for row in path_rows if row["active_goal_type"] == "reference"],
        dtype=float,
    )
    completion_events = [
        row for row in events
        if row["event"] == EVENT_REFERENCE_COMPLETION_REPROPOSAL
    ]
    values_to_check = [
        episode.get("minimum_obstacle_clearance_m"),
        episode.get("minimum_inter_agent_distance_m"),
        episode.get("maximum_phase_switch_delta"),
        *speeds.tolist(),
    ]
    nonfinite = sum(
        value is not None and not np.isfinite(float(value)) for value in values_to_check
    )
    return {
        "scenario_id": entry["scenario_id"],
        "seed": int(entry["seed"]),
        "stage": entry["stage"],
        "family": entry["family"],
        "task_pattern": entry["task_pattern"],
        "software_error": False,
        "team_success_descriptive_only": bool(episode["team_success"]),
        "collision_descriptive_only": bool(episode["collision"]),
        "timeout_descriptive_only": bool(episode["timeout"]),
        "termination_reason": episode["termination_reason"],
        "steps": int(episode["steps"]),
        "scene_reconstruction_match": bool(episode["scene_reconstruction_match"]),
        "actor_observation_dim": int(episode["policy_observation_shape_at_reset"][1]),
        "sensor_direction_count": int(episode["sensor_direction_count_at_reset"]),
        "maximum_speed_norm_mps": float(np.max(speeds)),
        "maximum_phase_switch_delta": float(episode["maximum_phase_switch_delta"]),
        "upper_pipeline_invocation_count": int(episode["upper_pipeline_invocation_count"]),
        "reference_selection_count": int(episode["reference_selection_count"]),
        "reference_reached_count": int(episode["reference_reached_count"]),
        "reference_completion_reproposal_count": len(completion_events),
        "normal_reproposal_count": sum(
            row["event"] == EVENT_NORMAL_REPROPOSAL for row in events
        ),
        "emergency_reproposal_count": sum(
            row["event"] == EVENT_EMERGENCY_REPROPOSAL for row in events
        ),
        "reference_active_distance_mean_m": float(np.mean(active_distances)) if active_distances.size else None,
        "reference_active_distance_max_m": float(np.max(active_distances)) if active_distances.size else None,
        "trigger_check_count": len(triggers),
        "path_record_count": len(path_rows),
        "nonfinite_record_count": int(nonfinite),
        "upper_planning_total_ms": float(episode["upper_planning_total_ms"]),
        "total_online_algorithm_compute_ms": float(episode["total_online_algorithm_compute_ms"]),
        "wall_episode_runtime_ms": float(episode["episode_runtime_ms"]),
        "performance_used_for_selection": False,
    }


def save_record(
    pilot_dir: Path, entry: Mapping[str, Any], episode: Mapping[str, Any],
    agents: Sequence[Mapping[str, Any]], events: Sequence[Mapping[str, Any]],
    triggers: Sequence[Mapping[str, Any]], extra: Mapping[str, Any], summary: Mapping[str, Any],
) -> None:
    records_dir = pilot_dir / "episode_records"
    records_dir.mkdir(parents=True, exist_ok=True)
    sid = str(entry["scenario_id"])
    path_rows = extra["path_rows"]
    positions = np.asarray([[row["x_m"], row["y_m"], row["z_m"]] for row in path_rows], dtype=np.float32)
    velocities = np.asarray([[row["vx_mps"], row["vy_mps"], row["vz_mps"]] for row in path_rows], dtype=np.float32)
    steps = np.asarray([row["step"] for row in path_rows], dtype=np.int32)
    agent_ids = np.asarray([row["agent_id"] for row in path_rows], dtype=np.int8)
    np.savez_compressed(records_dir / f"{sid}_trajectory.npz", positions=positions, velocities=velocities, steps=steps, agent_ids=agent_ids)
    atomic_json(
        records_dir / f"{sid}.json",
        {
            "entry_identity": {key: entry[key] for key in ("scenario_id", "seed", "stage", "family", "task_pattern", "geometry_fingerprint", "dynamic_track_fingerprint")},
            "summary": summary,
            "episode": episode,
            "agents": list(agents),
            "events": list(events),
            "trigger_summary": {
                "row_count": len(triggers),
                "event_counts": {event: sum(row["event"] == event for row in triggers) for event in sorted({row["event"] for row in triggers})},
            },
            "timing": {
                "actor_rows": extra["actor_timing_rows"],
                "dmp_rows": extra["dmp_timing_rows"],
                "upper_rows": extra["upper_timing_rows"],
            },
            "trajectory_file": f"{sid}_trajectory.npz",
            "path_row_count": len(path_rows),
        },
    )


def collect_summaries(pilot_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((pilot_dir / "episode_records").glob("PILOT_LR_*.json")):
        rows.append(load_json(path)["summary"])
    return rows


def verify_freeze(config_path: Path, pilot_dir: Path) -> None:
    freeze = load_json(pilot_dir / "contract_pilot_freeze.json")
    if sha256_file(config_path) != freeze["configuration_sha256"]:
        raise RuntimeError("pilot config changed after freeze")
    for relative, expected in freeze["source_sha256"].items():
        if sha256_file(REPO_ROOT / relative) != expected:
            raise RuntimeError(f"pilot source changed after freeze: {relative}")
    if sha256_file(pilot_dir / "contract_pilot_manifest.json") != freeze["scenario_manifest_sha256"]:
        raise RuntimeError("pilot manifest changed after freeze")


def run(config_path: Path, limit: int | None) -> None:
    config = load_json(config_path)
    _, pilot_dir = artifact_paths(config)
    verify_freeze(config_path, pilot_dir)
    manifest = load_json(pilot_dir / "contract_pilot_manifest.json")
    runtime = PilotRuntime(config, manifest)
    if runtime.policy_hash != config["sac_checkpoint_sha256_expected"]:
        # State-dict semantic hashes differ from file hashes; record both but
        # checkpoint file identity above is the binding integrity check.
        pass
    completed = {row["scenario_id"] for row in collect_summaries(pilot_dir)}
    pending = [entry for entry in manifest["entries"] if entry["scenario_id"] not in completed]
    if limit is not None:
        pending = pending[: int(limit)]
    for entry in pending:
        sid = str(entry["scenario_id"])
        print(f"[contract-pilot] start {sid} {entry['stage']} {entry['family']}", flush=True)
        recorder = OnlineRuntimeRecorder()
        proxy = TimedPolicyProxy(runtime.policy, recorder)
        try:
            with recorder.instrument_dmp(), recorder.scoped_context(
                evaluation_block="long_range_contract_pilot",
                stage=entry["stage"],
                family=entry["family"],
                scenario_id=sid,
                seed=int(entry["seed"]),
                method="RERR_GAT_SAC_DMP_INITIAL_CHECKPOINTS",
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
                )
            summary = summarize_record(entry, episode, events, triggers, extra["path_rows"])
            save_record(pilot_dir, entry, episode, agents, events, triggers, extra, summary)
        except Exception as error:
            failure = {
                "scenario_id": sid,
                "seed": int(entry["seed"]),
                "stage": entry["stage"],
                "family": entry["family"],
                "task_pattern": entry["task_pattern"],
                "software_error": True,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
                "performance_used_for_selection": False,
            }
            atomic_json(pilot_dir / "episode_records" / f"{sid}_SOFTWARE_ERROR.json", failure)
            raise
        summaries = collect_summaries(pilot_dir)
        write_csv(pilot_dir / "contract_pilot_results.csv", summaries)
        print(f"[contract-pilot] complete {sid} rows={len(summaries)}/40", flush=True)
    finalize(config_path, require_complete=False)


def finalize(config_path: Path, require_complete: bool = True) -> None:
    config = load_json(config_path)
    _, pilot_dir = artifact_paths(config)
    manifest = load_json(pilot_dir / "contract_pilot_manifest.json")
    summaries = collect_summaries(pilot_dir)
    write_csv(pilot_dir / "contract_pilot_results.csv", summaries)
    failures = sorted((pilot_dir / "episode_records").glob("*_SOFTWARE_ERROR.json"))
    rules = config["pilot_decision_rule"]
    stage_counts = {stage: sum(row["stage"] == stage for row in summaries) for stage in STAGE_ORDER}
    complete = len(summaries) == len(manifest["entries"])
    checks = {
        "all_40_episodes_complete": complete,
        "software_error_count_zero": len(failures) == int(rules["required_software_error_count"]),
        "nonfinite_record_count_zero": sum(int(row["nonfinite_record_count"]) for row in summaries) == int(rules["required_nonfinite_record_count"]),
        "scene_reconstruction_exact": bool(summaries) and all(bool(row["scene_reconstruction_match"]) for row in summaries),
        "actor_observation_dim_122": bool(summaries) and all(int(row["actor_observation_dim"]) == int(rules["required_actor_observation_dim"]) for row in summaries),
        "sensor_direction_count_56": bool(summaries) and all(int(row["sensor_direction_count"]) == int(rules["required_sensor_direction_count"]) for row in summaries),
        "phase_continuity": bool(summaries) and max(float(row["maximum_phase_switch_delta"]) for row in summaries) <= float(rules["required_phase_continuity_max_abs_delta"]),
        "operational_speed_norm": bool(summaries) and max(float(row["maximum_speed_norm_mps"]) for row in summaries) <= float(rules["required_speed_norm_upper_bound_mps"]),
        "reference_completion_lifecycle_exercised": bool(summaries) and sum(int(row["reference_completion_reproposal_count"]) for row in summaries) > 0,
        "balanced_stage_completion": complete and all(stage_counts[stage] == int(config["pilot_scenarios_per_stage"]) for stage in STAGE_ORDER),
    }
    status = "PASS" if complete and all(checks.values()) else ("FAIL" if complete else "IN_PROGRESS")
    aggregate = {
        "schema_version": "long_range_contract_pilot_reconciliation_v1",
        "status": status,
        "completed_episode_count": len(summaries),
        "expected_episode_count": len(manifest["entries"]),
        "stage_counts": stage_counts,
        "software_error_count": len(failures),
        "checks": checks,
        "performance_metrics_are_descriptive_only": True,
        "descriptive_team_success_rate": float(np.mean([row["team_success_descriptive_only"] for row in summaries])) if summaries else None,
        "descriptive_collision_rate": float(np.mean([row["collision_descriptive_only"] for row in summaries])) if summaries else None,
        "descriptive_timeout_rate": float(np.mean([row["timeout_descriptive_only"] for row in summaries])) if summaries else None,
        "mean_upper_invocations": float(np.mean([row["upper_pipeline_invocation_count"] for row in summaries])) if summaries else None,
        "reference_completion_reproposal_total": sum(int(row["reference_completion_reproposal_count"]) for row in summaries),
        "maximum_observed_speed_norm_mps": max((float(row["maximum_speed_norm_mps"]) for row in summaries), default=None),
        "selection_or_tuning_from_pilot_authorized": False,
        "next_phase_if_pass": "independent_training_and_development_manifest_freeze",
    }
    atomic_json(pilot_dir / "contract_pilot_reconciliation.json", aggregate)
    print(json.dumps({"phase": "finalize", "status": status, "completed": len(summaries)}), flush=True)
    if require_complete and status != "PASS":
        raise RuntimeError(f"contract pilot did not pass: {status}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "run", "finalize"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config_path = args.config.resolve()
    if args.phase == "prepare":
        prepare(config_path)
    elif args.phase == "run":
        run(config_path, args.limit)
    else:
        finalize(config_path)


if __name__ == "__main__":
    main()
