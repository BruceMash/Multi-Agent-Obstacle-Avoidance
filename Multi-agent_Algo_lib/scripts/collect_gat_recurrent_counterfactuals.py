"""Collect recurrent GAT states and training-only candidate counterfactuals.

Behavior states come from the frozen R-ERR + FP-SHEP closed loop on the new
training split.  Candidate branches clone the current real environment and use
only the frozen adapted SAC-DMP executor.  Future branch outcomes are written
as supervision artifacts and never enter runtime graph features.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as _pandas  # noqa: F401  # Windows torch/pyarrow initialization order
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
for search_path in (REPO_ROOT, ALGO_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from Guidance.reference_point_proposal_demo import ProposalConfig  # noqa: E402
from planning.event_triggered_reference_reconstruction import (  # noqa: E402
    ACTIVE_GOAL_REFERENCE,
    ACTIVE_GOAL_TERMINAL,
    EMERGENCY_SEMANTICS_EDGE_REARM,
    ERRConfig,
    EVENT_NO_UPDATE,
    EventTriggeredReferenceSupervisor,
    active_direction_safety_margin,
    set_active_goal_preserve_dmp_phase,
)
from planning.goal_semantics_diagnosis import (  # noqa: E402
    temporary_checkpoint_observations,
)
from planning.historical_forcing_gate import (  # noqa: E402
    scoped_historical_preview_and_multi_agent_transition,
)
from planning.online_runtime_instrumentation import OnlineRuntimeRecorder  # noqa: E402
from planning.pre_gat_220step_revalidation import stable_hash  # noqa: E402
from scripts.evaluate_gat_v1_err_development import (  # noqa: E402
    METHOD_RERR_FP_SHEP,
    build_online_gat_plan_optimized,
    run_episode,
)
from scripts.run_long_range_development import (  # noqa: E402
    DevelopmentRuntime,
)


ARTIFACT_ROOT = (
    REPO_ROOT
    / "artifacts/gat_recurrent_smoothness_rescue/20260821_180244"
)
TRAIN_INDEX = ARTIFACT_ROOT / "02_recurrent_dataset/GAT_RS_TRAIN_SCENE_MANIFEST.json"
BASE_METHOD_CONFIG = (
    REPO_ROOT
    / "artifacts/semi_structured_long_range_main_benchmark/20260820_193228"
    / "10_final_freeze/method_configs/M8_RERR_FP_SHEP_SAC_DMP.json"
)
OUTPUT = ARTIFACT_ROOT / "02_recurrent_dataset/recurrent_collection"
FREEZE_PATH = ARTIFACT_ROOT / "02_recurrent_dataset/recurrent_collection_freeze.json"
RESOLVED_CONFIG_PATH = ARTIFACT_ROOT / "02_recurrent_dataset/recurrent_collection_config.json"

PROGRESS_BINS = (
    ("early_0_25", 0.0, 0.25),
    ("early_mid_25_50", 0.25, 0.50),
    ("mid_late_50_75", 0.50, 0.75),
    ("late_75_100", 0.75, math.inf),
)
REPLAN_BINS = (
    ("replan_1_10", 1, 10),
    ("replan_11_30", 11, 30),
    ("replan_31_60", 31, 60),
    ("replan_61_100", 61, 100),
    ("replan_101_150", 101, 150),
    ("replan_151_250", 151, 250),
    ("replan_gt_250", 251, 10**9),
)
FIXED_HORIZON_STEPS = (10, 20, 30, 50)
NEXT_EVENT_MAX_STEPS = 150
TRAIN_SCENES_PER_STAGE = 50
AMBIGUOUS_FP_SCORE_GAP = 0.10
MAX_UNIQUE_STATES_PER_EPISODE = 7
TASK_GOAL_DISTANCE_SCALE_M = 100.0
SOURCE_PATHS = (
    "Environment/frozen_sac_dmp_execution.py",
    "Environment/multi_agent_dmp_env.py",
    "planning/event_triggered_reference_reconstruction.py",
    "planning/heterogeneous_candidate_graph.py",
    "planning/goal_semantics_diagnosis.py",
    "Multi-agent_Algo_lib/scripts/evaluate_gat_v1_err_development.py",
    "Multi-agent_Algo_lib/scripts/run_long_range_development.py",
    "Multi-agent_Algo_lib/scripts/collect_gat_recurrent_counterfactuals.py",
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
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


def progress_bin(value: float) -> str:
    value = float(value)
    for name, lower, upper in PROGRESS_BINS:
        if lower <= value < upper:
            return name
    raise RuntimeError(f"invalid mission progress: {value}")


def replan_bin(value: int) -> str:
    value = int(value)
    for name, lower, upper in REPLAN_BINS:
        if lower <= value <= upper:
            return name
    raise RuntimeError(f"invalid replan index: {value}")


def selected_training_index() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = load_json(TRAIN_INDEX)
    entries = [
        row
        for row in manifest["entries"]
        if int(str(row["scenario_id"]).rsplit("_", 1)[1])
        < TRAIN_SCENES_PER_STAGE
    ]
    if len(entries) != 4 * TRAIN_SCENES_PER_STAGE:
        raise RuntimeError("training collection panel must contain 200 scenes")
    return manifest, entries


def materialize_entries(index_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    entries = []
    for row in index_rows:
        path = ARTIFACT_ROOT / str(row["scenario_file"])
        if sha256_file(path) != row["scenario_file_sha256"]:
            raise RuntimeError(f"scene file changed after split freeze: {path}")
        entries.append(load_json(path))
    return entries


def prepare() -> None:
    if FREEZE_PATH.exists() or OUTPUT.exists():
        raise RuntimeError("recurrent collection was already prepared")
    train_manifest, selected = selected_training_index()
    config = load_json(BASE_METHOD_CONFIG)
    resolved = {
        "schema_version": "gat_recurrent_counterfactual_collection_v1",
        "behavior_method": "R-ERR + FP-SHEP H4 + frozen adapted SAC-DMP",
        "base_method_config": str(BASE_METHOD_CONFIG.relative_to(REPO_ROOT)).replace(
            "\\", "/"
        ),
        "base_method_config_sha256": sha256_file(BASE_METHOD_CONFIG),
        "training_index": str(TRAIN_INDEX.relative_to(REPO_ROOT)).replace("\\", "/"),
        "training_index_sha256": sha256_file(TRAIN_INDEX),
        "panel_selection": "scenario_index 0..49 within every stage",
        "panel_scenario_count": len(selected),
        "panel_scenario_ids": [row["scenario_id"] for row in selected],
        "panel_balance": "4 stages x 5 families x 10 scenes",
        "fixed_horizon_steps": list(FIXED_HORIZON_STEPS),
        "fixed_horizon_seconds": [step * float(config["dt"]) for step in FIXED_HORIZON_STEPS],
        "until_next_rerr_event_max_steps": NEXT_EVENT_MAX_STEPS,
        "progress_bins": [list(row) for row in PROGRESS_BINS],
        "replan_bins": [list(row) for row in REPLAN_BINS],
        "ambiguous_fp_top1_top2_gap_threshold": AMBIGUOUS_FP_SCORE_GAP,
        "max_unique_counterfactual_states_per_episode": MAX_UNIQUE_STATES_PER_EPISODE,
        "gat_r_task_goal_distance_scale_m": TASK_GOAL_DISTANCE_SCALE_M,
        "gat_rs_task_goal_distance_scale_m": TASK_GOAL_DISTANCE_SCALE_M,
        "gat_canonical_sector_count": 56,
        "candidate_count_max": int(config["top_k"]),
        "null_policy": config["err"]["far_terminal_null_policy"],
        "null_rollout_eligibility": "terminal goal distance <= terminal_local_scope_m",
        "runtime_future_information_added": False,
        "offline_rollout_training_only": True,
        "formal_v1_episode_or_state_data_used": False,
        "formal_v2_generated": False,
    }
    atomic_json(RESOLVED_CONFIG_PATH, resolved)
    freeze = {
        "schema_version": "gat_recurrent_counterfactual_collection_freeze_v1",
        "status": "FROZEN_BEFORE_RECURRENT_EPISODE_0",
        "resolved_config_sha256": sha256_file(RESOLVED_CONFIG_PATH),
        "base_method_config_sha256": sha256_file(BASE_METHOD_CONFIG),
        "training_index_sha256": sha256_file(TRAIN_INDEX),
        "training_index_semantic_sha256": train_manifest["manifest_semantic_sha256"],
        "sac_checkpoint": config["sac_checkpoint"],
        "sac_checkpoint_sha256": sha256_file(REPO_ROOT / config["sac_checkpoint"]),
        "gat_v1_checkpoint_used_only_to_materialize_frozen_architecture_graph_forward": config[
            "gat_checkpoint"
        ],
        "gat_v1_checkpoint_sha256": sha256_file(REPO_ROOT / config["gat_checkpoint"]),
        "behavior_selection_uses_gat_v1_logits": False,
        "behavior_selection": "FP_SHEP_TOP1",
        "source_sha256": {
            name: sha256_file(REPO_ROOT / name) for name in SOURCE_PATHS
        },
        "expected_episode_count": len(selected),
        "completed_episode_count_at_freeze": 0,
        "counterfactual_state_count_at_freeze": 0,
        "formal_v1_result_rows_read": 0,
        "formal_v2_generated": False,
    }
    atomic_json(FREEZE_PATH, freeze)
    OUTPUT.mkdir(parents=True)
    print(json.dumps({"phase": "prepare", "status": "PASS", "episodes": len(selected)}), flush=True)


def verify_freeze() -> dict[str, Any]:
    freeze = load_json(FREEZE_PATH)
    if sha256_file(RESOLVED_CONFIG_PATH) != freeze["resolved_config_sha256"]:
        raise RuntimeError("recurrent collection config changed after freeze")
    if sha256_file(BASE_METHOD_CONFIG) != freeze["base_method_config_sha256"]:
        raise RuntimeError("base R-ERR FP config changed after freeze")
    if sha256_file(TRAIN_INDEX) != freeze["training_index_sha256"]:
        raise RuntimeError("training scene index changed after freeze")
    for name, expected in freeze["source_sha256"].items():
        if sha256_file(REPO_ROOT / name) != expected:
            raise RuntimeError(f"collection source changed after freeze: {name}")
    return freeze


def _minimum_signed(obstacles: Sequence[Any], point: np.ndarray) -> float:
    if not obstacles:
        return float("inf")
    return min(float(obstacle.signed_distance(point)) for obstacle in obstacles)


def _peer_minimum(env: Any, focal: int) -> float:
    point = np.asarray(env.dynamics[focal].p, dtype=float)
    values = [
        float(np.linalg.norm(point - env.dynamics[other].p))
        for other in range(int(env.num_agents))
        if other != focal
    ]
    return min(values) if values else float("inf")


def _line_deviation(points: np.ndarray, start: np.ndarray, goal: np.ndarray) -> float:
    direction = np.asarray(goal, dtype=float) - np.asarray(start, dtype=float)
    denominator = float(np.dot(direction, direction))
    if denominator <= 1.0e-12:
        return float(np.max(np.linalg.norm(points - start, axis=1)))
    tau = np.clip(((points - start) @ direction) / denominator, 0.0, 1.0)
    closest = start[None, :] + tau[:, None] * direction[None, :]
    return float(np.max(np.linalg.norm(points - closest, axis=1)))


def _motion_metrics(
    positions: Sequence[np.ndarray], accelerations: Sequence[np.ndarray], dt: float
) -> dict[str, Any]:
    pos = np.asarray(positions, dtype=float)
    acc = np.asarray(accelerations, dtype=float)
    path = (
        np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=2), axis=0)
        if len(pos) >= 2
        else np.zeros(pos.shape[1], dtype=float)
    )
    if len(acc) >= 2:
        delta = np.diff(acc, axis=0)
        acceleration_variation = np.mean(np.sum(delta**2, axis=2), axis=0)
        jerk = np.mean(np.sum((delta / float(dt)) ** 2, axis=2), axis=0)
    else:
        acceleration_variation = np.full(pos.shape[1], np.nan)
        jerk = np.full(pos.shape[1], np.nan)
    return {
        "path_length_by_agent_m": path.tolist(),
        "acceleration_variation_by_agent_m2_s4": acceleration_variation.tolist(),
        "trajectory_smoothness_by_agent_m2_s6": jerk.tolist(),
        "trajectory_smoothness_team_mean_m2_s6": (
            float(np.nanmean(jerk)) if np.any(np.isfinite(jerk)) else None
        ),
    }


def _branch_snapshot(branch: dict[str, Any], step: int, *, reached: bool) -> dict[str, Any]:
    env = branch["env"]
    focal = int(branch["focal"])
    terminal_distance = float(
        np.linalg.norm(env.goals[focal] - env.dynamics[focal].p)
    )
    motion = _motion_metrics(branch["positions"], branch["accelerations"], branch["dt"])
    focal_points = np.asarray(branch["positions"], dtype=float)[:, focal, :]
    return {
        "step": int(step),
        "time_s": float(step) * float(branch["dt"]),
        "horizon_reached": True,
        "terminal_distance_m": terminal_distance,
        "terminal_progress_m": float(branch["initial_terminal_distance"] - terminal_distance),
        "progress_per_second_mps": (
            float((branch["initial_terminal_distance"] - terminal_distance) / (step * branch["dt"]))
            if step > 0
            else 0.0
        ),
        "reference_reached": bool(reached),
        "minimum_sensor_clearance_m": float(branch["minimum_sensor_clearance"]),
        "minimum_static_clearance_m": float(branch["minimum_static_clearance"]),
        "minimum_dynamic_clearance_m": float(branch["minimum_dynamic_clearance"]),
        "minimum_peer_center_distance_m": float(branch["minimum_peer_distance"]),
        "minimum_peer_margin_m": float(branch["minimum_peer_distance"] - branch["d_safe"]),
        "peer_risk_duration_s": float(branch["peer_risk_steps"] * branch["dt"]),
        "collision": bool(branch["team_collision"]),
        "focal_collision": bool(branch["focal_collision"]),
        "static_collision": bool(branch["static_collision"]),
        "dynamic_collision": bool(branch["dynamic_collision"]),
        "peer_collision": bool(branch["peer_collision"]),
        "boundary_collision": bool(branch["boundary_collision"]),
        "travel_distance_focal_m": float(motion["path_length_by_agent_m"][focal]),
        "execution_deviation_m": _line_deviation(
            focal_points, branch["branch_start_position"], branch["candidate_goal"]
        ),
        "acceleration_variation_focal_m2_s4": motion[
            "acceleration_variation_by_agent_m2_s4"
        ][focal],
        "trajectory_smoothness_focal_m2_s6": motion[
            "trajectory_smoothness_by_agent_m2_s6"
        ][focal],
        "trajectory_smoothness_team_mean_m2_s6": motion[
            "trajectory_smoothness_team_mean_m2_s6"
        ],
        "control_discontinuity_mps2": branch["control_discontinuity_mps2"],
    }


def _init_branch(
    env: Any,
    *,
    focal: int,
    candidate_id: int | None,
    candidate_goal: np.ndarray,
    config: Mapping[str, Any],
    null_eligible: bool,
) -> dict[str, Any]:
    clone = copy.deepcopy(env)
    focal = int(focal)
    phase_before = float(clone.dmps[focal].phase)
    set_active_goal_preserve_dmp_phase(clone.dmps[focal], candidate_goal)
    if float(clone.dmps[focal].phase) != phase_before:
        raise RuntimeError("counterfactual goal update changed DMP phase")
    active_goals = np.stack([dmp.goal.copy() for dmp in clone.dmps])
    goal_types = [
        ACTIVE_GOAL_TERMINAL
        if np.array_equal(active_goals[index], clone.goals[index])
        else ACTIVE_GOAL_REFERENCE
        for index in range(int(clone.num_agents))
    ]
    goal_types[focal] = (
        ACTIVE_GOAL_TERMINAL if candidate_id is None else ACTIVE_GOAL_REFERENCE
    )
    err_config = ERRConfig.from_mapping({"dt": config["dt"], **config["err"]})
    supervisor = EventTriggeredReferenceSupervisor(
        err_config,
        terminal_goals=np.asarray(clone.goals, dtype=float),
        active_goals=active_goals,
        active_goal_types=goal_types,
        initial_positions=clone._positions().copy(),
        emergency_event_semantics=EMERGENCY_SEMANTICS_EDGE_REARM,
    )
    proposal_config = ProposalConfig(**dict(config["proposal_config"]))
    initial_safety = active_direction_safety_margin(
        clone, focal, candidate_goal, proposal_config
    )
    supervisor.synchronize_emergency_latch(
        focal, active_safety_margin_m=initial_safety.value_m
    )
    position = np.asarray(clone.dynamics[focal].p, dtype=float)
    previous_acceleration = (
        np.asarray(clone.dynamics[focal].v, dtype=float)
        - np.asarray(clone.previous_velocities[focal], dtype=float)
    ) / float(config["dt"])
    static = _minimum_signed(clone.static_obstacles, position)
    dynamic = _minimum_signed(clone.dynamic_obstacles, position)
    peer = _peer_minimum(clone, focal)
    sensor = float(clone.latest_sensor_packets[focal].min_clearance)
    return {
        "env": clone,
        "focal": focal,
        "candidate_id": candidate_id,
        "candidate_goal": np.asarray(candidate_goal, dtype=float).copy(),
        "null_eligible": bool(null_eligible),
        "supervisor": supervisor,
        "proposal_config": proposal_config,
        "dt": float(config["dt"]),
        "d_safe": float(clone.env_config.inter_agent_safe_distance),
        "initial_terminal_distance": float(np.linalg.norm(clone.goals[focal] - position)),
        "branch_start_position": position.copy(),
        "previous_applied_acceleration": previous_acceleration,
        "positions": [clone._positions().copy()],
        "accelerations": [],
        "minimum_sensor_clearance": sensor,
        "minimum_static_clearance": static,
        "minimum_dynamic_clearance": dynamic,
        "minimum_peer_distance": peer,
        "peer_risk_steps": int(peer < float(clone.env_config.inter_agent_safe_distance)),
        "team_collision": False,
        "focal_collision": False,
        "static_collision": static <= 0.0,
        "dynamic_collision": dynamic <= 0.0,
        "peer_collision": False,
        "boundary_collision": False,
        "control_discontinuity_mps2": None,
        "reference_reached": float(np.linalg.norm(candidate_goal - position)) <= float(config["handoff_threshold_m"]),
        "next_event": None,
        "fixed_horizons": {},
        "terminated": False,
        "truncated": False,
        "done": False,
    }


def _update_branch(branch: dict[str, Any], info: Mapping[str, Any]) -> None:
    env = branch["env"]
    focal = int(branch["focal"])
    position = np.asarray(env.dynamics[focal].p, dtype=float)
    static = _minimum_signed(env.static_obstacles, position)
    dynamic = _minimum_signed(env.dynamic_obstacles, position)
    peer = _peer_minimum(env, focal)
    branch["minimum_sensor_clearance"] = min(
        branch["minimum_sensor_clearance"],
        float(env.latest_sensor_packets[focal].min_clearance),
    )
    branch["minimum_static_clearance"] = min(branch["minimum_static_clearance"], static)
    branch["minimum_dynamic_clearance"] = min(branch["minimum_dynamic_clearance"], dynamic)
    branch["minimum_peer_distance"] = min(branch["minimum_peer_distance"], peer)
    branch["peer_risk_steps"] += int(peer < branch["d_safe"])
    branch["team_collision"] |= bool(info.get("collision", False))
    obstacle_mask = np.asarray(info.get("obstacle_collision_mask", []), dtype=bool)
    peer_mask = np.asarray(info.get("inter_agent_collision_mask", []), dtype=bool)
    boundary_mask = np.asarray(info.get("boundary_collision_mask", []), dtype=bool)
    focal_obstacle = bool(obstacle_mask[focal]) if obstacle_mask.size else False
    focal_peer = bool(peer_mask[focal]) if peer_mask.size else False
    focal_boundary = bool(boundary_mask[focal]) if boundary_mask.size else False
    branch["focal_collision"] |= focal_obstacle or focal_peer or focal_boundary
    branch["static_collision"] |= static <= 0.0
    branch["dynamic_collision"] |= dynamic <= 0.0
    branch["peer_collision"] |= focal_peer
    branch["boundary_collision"] |= focal_boundary
    branch["reference_reached"] |= float(
        np.linalg.norm(branch["candidate_goal"] - position)
    ) <= 0.25


def rollout_candidates(
    env: Any,
    *,
    focal: int,
    candidate_points: Sequence[Sequence[float]],
    config: Mapping[str, Any],
    policy: Any,
) -> list[dict[str, Any]]:
    focal = int(focal)
    terminal_distance = float(
        np.linalg.norm(env.goals[focal] - env.dynamics[focal].p)
    )
    terminal_scope = float(config["err"]["terminal_local_scope_m"])
    null_eligible = terminal_distance <= terminal_scope
    specifications: list[tuple[int | None, np.ndarray, bool]] = [
        (index, np.asarray(point, dtype=float), True)
        for index, point in enumerate(candidate_points)
    ]
    if null_eligible:
        specifications.append((None, np.asarray(env.goals[focal], dtype=float), True))
    branches = [
        _init_branch(
            env,
            focal=focal,
            candidate_id=candidate_id,
            candidate_goal=goal,
            config=config,
            null_eligible=eligible,
        )
        for candidate_id, goal, eligible in specifications
    ]
    err_dt = float(config["dt"])
    with scoped_historical_preview_and_multi_agent_transition():
        for local_step in range(0, NEXT_EVENT_MAX_STEPS + 1):
            active = [branch for branch in branches if not branch["done"]]
            if not active:
                break
            for branch in active:
                if branch["next_event"] is None and local_step > 0:
                    focal_id = int(branch["focal"])
                    safety = active_direction_safety_margin(
                        branch["env"],
                        focal_id,
                        branch["candidate_goal"],
                        branch["proposal_config"],
                    )
                    decision = branch["supervisor"].evaluate(
                        focal_id,
                        current_step=local_step,
                        position=branch["env"].dynamics[focal_id].p,
                        active_safety_margin_m=safety.value_m,
                    )
                    if decision.event != EVENT_NO_UPDATE:
                        snapshot = _branch_snapshot(
                            branch, local_step, reached=branch["reference_reached"]
                        )
                        branch["next_event"] = {
                            **snapshot,
                            "event": decision.event,
                            "trigger_reasons": list(decision.trigger_reasons),
                            "subsequent_emergency_trigger": bool(
                                decision.emergency_trigger
                            ),
                        }
                if (
                    local_step >= max(FIXED_HORIZON_STEPS)
                    and branch["next_event"] is not None
                ):
                    branch["done"] = True
            active = [branch for branch in branches if not branch["done"]]
            if not active or local_step >= NEXT_EVENT_MAX_STEPS:
                break

            observations = np.concatenate(
                [
                    temporary_checkpoint_observations(
                        branch["env"],
                        np.stack([dmp.goal for dmp in branch["env"].dmps]),
                    )
                    for branch in active
                ],
                axis=0,
            )
            actions, _ = policy.predict(observations, deterministic=True)
            actions = np.asarray(actions, dtype=np.float32).reshape(
                len(active), int(env.num_agents), -1
            )
            for branch, action in zip(active, actions, strict=True):
                _, _, terminated, truncated, info = branch["env"].step(action)
                applied = np.asarray(info["applied_accelerations"], dtype=float)
                if not branch["accelerations"]:
                    branch["control_discontinuity_mps2"] = float(
                        np.linalg.norm(
                            applied[int(branch["focal"])]
                            - branch["previous_applied_acceleration"]
                        )
                    )
                branch["accelerations"].append(applied.copy())
                branch["positions"].append(branch["env"]._positions().copy())
                _update_branch(branch, info)
                completed_step = local_step + 1
                if completed_step in FIXED_HORIZON_STEPS:
                    branch["fixed_horizons"][str(completed_step)] = _branch_snapshot(
                        branch,
                        completed_step,
                        reached=branch["reference_reached"],
                    )
                branch["terminated"] = bool(terminated)
                branch["truncated"] = bool(truncated)
                if terminated or truncated:
                    branch["done"] = True

    rows: list[dict[str, Any]] = []
    for branch in branches:
        final_step = len(branch["accelerations"])
        fixed = {}
        for horizon in FIXED_HORIZON_STEPS:
            key = str(horizon)
            if key in branch["fixed_horizons"]:
                fixed[f"horizon_{horizon}"] = branch["fixed_horizons"][key]
            else:
                fixed[f"horizon_{horizon}"] = {
                    **_branch_snapshot(
                        branch, final_step, reached=branch["reference_reached"]
                    ),
                    "requested_step": horizon,
                    "horizon_reached": False,
                }
        rows.append(
            {
                "class_index": 0 if branch["candidate_id"] is None else int(branch["candidate_id"]) + 1,
                "candidate_id": branch["candidate_id"],
                "null_branch": branch["candidate_id"] is None,
                "null_eligible": branch["null_eligible"],
                "candidate_goal": branch["candidate_goal"].tolist(),
                "executed_steps": final_step,
                "terminated": branch["terminated"],
                "truncated": branch["truncated"],
                "next_rerr_event": branch["next_event"],
                "next_rerr_event_observed": branch["next_event"] is not None,
                "next_rerr_event_censored_at_s": (
                    None
                    if branch["next_event"] is not None
                    else NEXT_EVENT_MAX_STEPS * err_dt
                ),
                **fixed,
            }
        )
        branch["env"].close()
    if not null_eligible:
        rows.append(
            {
                "class_index": 0,
                "candidate_id": None,
                "null_branch": True,
                "null_eligible": False,
                "candidate_goal": np.asarray(env.goals[focal], dtype=float).tolist(),
                "not_run_reason": "far_terminal_null_mask",
            }
        )
    return sorted(rows, key=lambda row: int(row["class_index"]))


def graph_with_recurrent_scale(graph: Any, metadata: Mapping[str, Any]) -> Any:
    result = copy.deepcopy(graph).cpu()
    for node_type in ("null", "agent"):
        raw_distance = float(result[node_type].x_raw[0, 3])
        normalized = float(np.clip(raw_distance / TASK_GOAL_DISTANCE_SCALE_M, 0.0, 1.0))
        result[node_type].x_normalized[0, 3] = normalized
        result[node_type].x[0, 3] = normalized
    graph_metadata = copy.deepcopy(dict(result.graph_metadata))
    graph_metadata["normalization_scales"]["task_goal_distance"] = TASK_GOAL_DISTANCE_SCALE_M
    graph_metadata["task_goal_distance_scale_source"] = (
        "gat_recurrent_preperformance_100m_workspace_axis_freeze"
    )
    graph_metadata["offline_future_information_in_graph"] = False
    result.graph_metadata = graph_metadata
    result.recurrent_state_metadata = dict(metadata)
    return result


def fp_behavior_selection(selection: Mapping[str, Any], env: Any) -> dict[str, Any]:
    result = dict(selection)
    source_plan = selection["plan"]
    records = copy.deepcopy(source_plan["candidate_records"])
    references = np.asarray(env.goals, dtype=float).copy()
    available = np.zeros(int(env.num_agents), dtype=bool)
    for agent_id, record in enumerate(records):
        preview_records = record.get("fp_shep_candidate_records", [])
        if preview_records:
            scores = np.asarray(
                [float(item["fp_shep_online_score"]) for item in preview_records]
            )
            selected = int(np.argmax(scores))
            references[agent_id] = np.asarray(
                record["candidate_world_points"][selected], dtype=float
            )
            available[agent_id] = True
            record.update(
                {
                    "candidate_available": True,
                    "temporary_reference": references[agent_id].tolist(),
                    "selected_candidate_id": selected,
                    "selected_null": False,
                    "selection_source": "fp_shep_h4_training_behavior",
                    "selected_proposal_rank": selected,
                    "selected_proposal_rank_1based": selected + 1,
                    "selected_proposal_score": float(record["proposal_scores"][selected]),
                    "selected_fp_shep_score": float(scores[selected]),
                    "selected_fp_shep_rank_1based": 1,
                }
            )
        else:
            record.update(
                {
                    "candidate_available": False,
                    "temporary_reference": references[agent_id].tolist(),
                    "selected_candidate_id": None,
                    "selected_null": False,
                    "selection_source": "untriggered_agent_no_update",
                }
            )
    plan = {
        "selection_source": "fp_shep_h4_training_behavior",
        "references": references.tolist(),
        "available": available.tolist(),
        "candidate_records": records,
    }
    plan["selection_plan_hash"] = stable_hash(plan)
    result["plan"] = plan
    result["selector"] = "fp_shep"
    result["behavior_selection_uses_gat_logits"] = False
    result["graphs_built_for_training_only"] = True
    return result


class CollectionBuilder:
    def __init__(
        self,
        *,
        scenario_entry: Mapping[str, Any],
        episode_dir: Path,
        policy: Any,
    ) -> None:
        self.entry = dict(scenario_entry)
        self.episode_dir = episode_dir
        self.policy = policy
        self.initial_distances: np.ndarray | None = None
        self.universe: list[dict[str, Any]] = []
        self.sampled: list[dict[str, Any]] = []
        self.sample_keys: set[tuple[int, int]] = set()
        self.sampled_roles: set[str] = set()
        self.call_index = 0

    def _state_roles(
        self,
        *,
        agent_id: int,
        progress_name: str,
        replan_name: str,
        ambiguous: bool,
        peer_rich: bool,
    ) -> list[str]:
        scenario_index = int(self.entry["scenario_index"])
        roles: list[str] = []
        progress_names = [row[0] for row in PROGRESS_BINS]
        progress_role = f"progress:{progress_name}"
        progress_agent = (scenario_index + progress_names.index(progress_name)) % 3
        if agent_id == progress_agent and progress_role not in self.sampled_roles:
            roles.append(progress_role)
        replan_names = [row[0] for row in REPLAN_BINS]
        target_replan = replan_names[scenario_index % len(replan_names)]
        replan_agent = (scenario_index // len(replan_names)) % 3
        replan_role = f"replan:{target_replan}"
        if (
            replan_name == target_replan
            and agent_id == replan_agent
            and replan_role not in self.sampled_roles
        ):
            roles.append(replan_role)
        if ambiguous and "oversample:ambiguous" not in self.sampled_roles:
            roles.append("oversample:ambiguous")
        if peer_rich and "oversample:peer_conflict" not in self.sampled_roles:
            roles.append("oversample:peer_conflict")
        return roles

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        env = kwargs["env"]
        runtime_recorder = kwargs.get("runtime_recorder")
        context = runtime_recorder.context if runtime_recorder is not None else {}
        planning_index = int(context.get("planning_decision_index", self.call_index)) + 1
        event_step = int(context.get("event_step", env.steps))
        self.call_index += 1
        if self.initial_distances is None:
            self.initial_distances = np.linalg.norm(
                np.asarray(env.goals, dtype=float) - np.asarray(env.starts, dtype=float),
                axis=1,
            )
        selection = build_online_gat_plan_optimized(
            **{**kwargs, "selector": "gat"}
        )
        graphs = selection.get("_audit_graphs_by_agent", {})
        computed = [int(value) for value in selection.get("computed_agent_ids", [])]
        source_records = selection["plan"]["candidate_records"]

        for agent_id in computed:
            graph = graphs[agent_id]
            remaining = float(
                np.linalg.norm(env.goals[agent_id] - env.dynamics[agent_id].p)
            )
            progress = float(
                np.clip(1.0 - remaining / self.initial_distances[agent_id], 0.0, 1.5)
            )
            progress_name = progress_bin(progress)
            replan_name = replan_bin(planning_index)
            record = source_records[agent_id]
            preview = record.get("fp_shep_candidate_records", [])
            scores = sorted(
                [float(item["fp_shep_online_score"]) for item in preview],
                reverse=True,
            )
            score_gap = scores[0] - scores[1] if len(scores) >= 2 else None
            interactions = record.get("all_candidate_interaction_records", [])
            safe_count = sum(not bool(item.get("risky")) for item in interactions)
            risky_count = sum(bool(item.get("risky")) for item in interactions)
            ambiguous = bool(
                len(scores) >= 2
                and safe_count >= 2
                and score_gap is not None
                and score_gap <= AMBIGUOUS_FP_SCORE_GAP
            )
            peer_rich = bool(risky_count > 0 or graph["align"].num_nodes > 0)
            roles = self._state_roles(
                agent_id=agent_id,
                progress_name=progress_name,
                replan_name=replan_name,
                ambiguous=ambiguous,
                peer_rich=peer_rich,
            )
            universe_row = {
                "scenario_id": self.entry["scenario_id"],
                "stage": self.entry["stage"],
                "family": self.entry["family"],
                "task_pattern": self.entry["task_pattern"],
                "event_step": event_step,
                "planning_decision_index": planning_index,
                "agent_id": agent_id,
                "mission_progress": progress,
                "progress_bin": progress_name,
                "replan_bin": replan_name,
                "candidate_count": len(preview),
                "fp_top1_top2_gap": score_gap,
                "safe_candidate_count_h4": safe_count,
                "risky_candidate_count_h4": risky_count,
                "align_neighbor_count": int(graph["align"].num_nodes),
                "ambiguous_pre_rollout": ambiguous,
                "peer_rich_pre_rollout": peer_rich,
                "sampling_roles": roles,
            }
            self.universe.append(universe_row)
            state_key = (event_step, agent_id)
            if (
                roles
                and state_key not in self.sample_keys
                and len(self.sampled) < MAX_UNIQUE_STATES_PER_EPISODE
            ):
                state_id = f"{self.entry['scenario_id']}__t{event_step:04d}__a{agent_id}"
                # Keep the on-disk suffix short enough for legacy Windows path
                # limits; the full collision-free state_id remains in metadata.
                state_dir = self.episode_dir / "s" / f"t{event_step}a{agent_id}"
                state_dir.mkdir(parents=True, exist_ok=False)
                state_metadata = {
                    **universe_row,
                    "state_id": state_id,
                    "sampling_roles": roles,
                    "terminal_distance_m": remaining,
                    "goal_distance_normalization_scale_m": TASK_GOAL_DISTANCE_SCALE_M,
                    "offline_future_information_in_runtime_graph": False,
                }
                graph_to_save = graph_with_recurrent_scale(graph, state_metadata)
                graph_path = state_dir / "graph.pt"
                torch.save(graph_to_save, graph_path)
                rollouts = rollout_candidates(
                    env,
                    focal=agent_id,
                    candidate_points=record["candidate_world_points"],
                    config=kwargs["config"],
                    policy=self.policy,
                )
                payload = {
                    "schema_version": "gat_recurrent_counterfactual_state_v1",
                    "state": state_metadata,
                    "graph_file": "graph.pt",
                    "graph_file_sha256": sha256_file(graph_path),
                    "candidate_world_points": record["candidate_world_points"],
                    "proposal_scores": record["proposal_scores"],
                    "fp_shep_candidate_records": preview,
                    "candidate_interaction_records": interactions,
                    "behavior_fp_selected_candidate_id": record.get(
                        "fp_shep_selected_candidate_id"
                    ),
                    "counterfactual_rollouts": rollouts,
                    "counterfactual_executor": "frozen adapted SAC-DMP",
                    "counterfactual_runtime_use": False,
                }
                atomic_json(state_dir / "state.json", payload)
                final_episode_dir = Path(str(self.episode_dir).removesuffix(".partial"))
                final_state_dir = final_episode_dir / state_dir.relative_to(
                    self.episode_dir
                )
                self.sampled.append(
                    {
                        **state_metadata,
                        "state_file": str((final_state_dir / "state.json").relative_to(OUTPUT)).replace("\\", "/"),
                        "graph_file": str((final_state_dir / "graph.pt").relative_to(OUTPUT)).replace("\\", "/"),
                        "candidate_branch_count": len(rollouts),
                    }
                )
                self.sample_keys.add(state_key)
                self.sampled_roles.update(roles)
                print(
                    f"[recurrent-state] {state_id} roles={','.join(roles)} branches={len(rollouts)}",
                    flush=True,
                )
        return fp_behavior_selection(selection, env)


def run_shard(shard_index: int, shard_count: int, limit: int | None) -> None:
    verify_freeze()
    _, selected = selected_training_index()
    selected = [
        row for ordinal, row in enumerate(selected) if ordinal % shard_count == shard_index
    ]
    if limit is not None:
        selected = selected[: int(limit)]
    shard_dir = OUTPUT / f"shard_{shard_index:02d}_of_{shard_count:02d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    completed = {
        path.parent.name
        for path in shard_dir.glob("episodes/*/episode.json")
    }
    pending_index = [row for row in selected if row["scenario_id"] not in completed]
    entries = materialize_entries(pending_index)
    if not entries:
        print(json.dumps({"status": "NO_PENDING", "shard": shard_index}), flush=True)
        return
    base_config = load_json(BASE_METHOD_CONFIG)
    runtime = DevelopmentRuntime(base_config, {"entries": entries})
    for entry in entries:
        sid = str(entry["scenario_id"])
        final_dir = shard_dir / "episodes" / sid
        partial_dir = shard_dir / "episodes" / f"{sid}.partial"
        if final_dir.exists():
            continue
        if partial_dir.exists():
            raise RuntimeError(f"partial episode requires audit before resume: {partial_dir}")
        partial_dir.mkdir(parents=True, exist_ok=False)
        collector = CollectionBuilder(
            scenario_entry=entry,
            episode_dir=partial_dir,
            policy=runtime.policy,
        )
        recorder = OnlineRuntimeRecorder()
        started = time.time()
        print(f"[recurrent-episode] start {sid}", flush=True)
        try:
            episode, agents, events, triggers, _ = run_episode(
                config=runtime.eval_config,
                settings=runtime.settings,
                multi_config=runtime.multi_config,
                policy=runtime.policy,
                gat_model=runtime.gat_model,
                gat_device=runtime.gat_device,
                method=METHOD_RERR_FP_SHEP,
                scenario=sid,
                seed=int(entry["seed"]),
                environment_builder=runtime.builder,
                runtime_recorder=recorder,
                upper_plan_builder=collector,
            )
            atomic_json(partial_dir / "state_universe.json", collector.universe)
            atomic_json(
                partial_dir / "episode.json",
                {
                    "schema_version": "gat_recurrent_collection_episode_v1",
                    "scenario_id": sid,
                    "seed": int(entry["seed"]),
                    "stage": entry["stage"],
                    "family": entry["family"],
                    "task_pattern": entry["task_pattern"],
                    "behavior_method": "fp_shep_rerr",
                    "episode_outcome": episode,
                    "agent_outcomes": agents,
                    "selection_event_count": len(events),
                    "trigger_row_count": len(triggers),
                    "upper_state_universe_count": len(collector.universe),
                    "sampled_state_count": len(collector.sampled),
                    "sampled_states": collector.sampled,
                    "wall_time_s": time.time() - started,
                    "formal_v1_data_used": False,
                    "future_information_used_by_behavior": False,
                },
            )
            partial_dir.replace(final_dir)
            print(
                f"[recurrent-episode] complete {sid} success={int(episode['team_success'])} "
                f"steps={episode['steps']} universe={len(collector.universe)} sampled={len(collector.sampled)} "
                f"wall_s={time.time()-started:.1f}",
                flush=True,
            )
        except Exception as error:
            atomic_json(
                partial_dir / "SOFTWARE_ERROR.json",
                {
                    "scenario_id": sid,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
            raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare()
        return
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")
    run_shard(args.shard_index, args.shard_count, args.limit)


if __name__ == "__main__":
    main()
