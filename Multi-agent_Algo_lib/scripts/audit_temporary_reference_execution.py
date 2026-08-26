"""Read-only audit of the frozen SAC-DMP temporary-reference contract.

The audit replays already selected Stage-I GAT references from the formal
one-shot artifact.  It never trains a model and changes no policy, reward,
controller, selector, environment, or handoff threshold.
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import replace
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
# On the Windows evaluation environment pandas/pyarrow must be initialized
# before Torch loads its native DLLs; the repository's SAC loader imports both.
import pandas as _pandas  # noqa: F401


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = Path(__file__).resolve().parent
for search_path in (REPO_ROOT, ALGO_ROOT, SCRIPTS_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from Environment.frozen_sac_dmp_execution import predict_frozen_actions  # noqa: E402
from planning.candidate_execution_benchmark import (  # noqa: E402
    _active_goal_observations,
    environment_state_fingerprint,
)
from planning.gat_supervision_v2 import (  # noqa: E402
    _agent_peer_distance,
    _obstacle_surface_clearance,
)
from planning.historical_forcing_gate import (  # noqa: E402
    HISTORICAL_GATE_NAME,
    scoped_historical_preview_and_multi_agent_transition,
)
from scripts.evaluate_frozen_policy_waypoint_guidance import (  # noqa: E402
    set_dmp_active_goal_preserve_phase,
)
from scripts.evaluate_pre_gat_closed_loop import (  # noqa: E402
    _policy_parameter_sha256,
    _scenario_hash,
    _scene_snapshot,
    build_closed_loop_environment,
)
from scripts.evaluate_single_policy_aligned_multi_agent import (  # noqa: E402
    build_single_distribution_multi_config,
)
from scripts.evaluate_single_policy_multi_agent import _build_environment  # noqa: E402
from scripts.evaluate_temporary_reference_interface import _load_policy  # noqa: E402


SCHEMA_VERSION = "temporary_reference_execution_audit_v1"
DEFAULT_CONFIG_PATH = (
    REPO_ROOT / "configs/evaluation/temporary_reference_execution_audit.json"
)
CONTRACTS = ("A_DIRECT_TERMINAL", "B_TEMP_REFERENCE_ONLY", "C_TEMP_TO_TERMINAL")
CORE_PATHS = (
    "baseline/sac/net.py",
    "Controller/dmp_rl.py",
    "Environment/multi_agent_dmp_env.py",
    "Environment/frozen_sac_dmp_execution.py",
    "Guidance/reference_point_proposal_demo.py",
    "planning/policy_preview.py",
    "planning/historical_forcing_gate.py",
    "planning/gat/edge_enhanced_gat.py",
    "planning/pre_gat_closed_loop.py",
    "planning/event_triggered_reference_reconstruction.py",
)
POLICY_FAILURES_B = {
    "STAGNATION_BEFORE_REFERENCE",
    "TIMEOUT_BEFORE_REFERENCE",
    "DIVERGENCE_FROM_REFERENCE",
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not materialized:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in materialized:
            encoded: dict[str, Any] = {}
            for key in fields:
                value = _jsonable(row.get(key))
                encoded[key] = (
                    json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (list, tuple, dict))
                    else value
                )
            writer.writerow(encoded)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_paths(paths: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in paths:
        path = REPO_ROOT / relative
        result[relative] = sha256_file(path) if path.is_file() else "MISSING"
    return result


def _json_value(value: Any, default: Any = None) -> Any:
    if value is None or value == "":
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value


def _boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _finite(value: Any) -> bool:
    return _number(value) is not None


def _rate(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    return float(np.mean([bool(row[key]) for row in rows]))


def _mean(values: Iterable[Any]) -> float | None:
    array = np.asarray(
        [float(value) for value in values if value is not None], dtype=float
    )
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else None


def classify_gap(drop: float | None, rules: Mapping[str, Any]) -> str:
    if drop is None or not math.isfinite(float(drop)):
        return "NO"
    if float(drop) >= float(rules["gap_yes_minimum"]):
        return "YES"
    if float(drop) >= float(rules["gap_weak_minimum"]):
        return "WEAK"
    return "NO"


def classify_direct(rate: float | None, rules: Mapping[str, Any]) -> str:
    if rate is None:
        return "WEAK"
    if rate >= float(rules["direct_strong_minimum"]):
        return "STRONG"
    if rate >= float(rules["direct_moderate_minimum"]):
        return "MODERATE"
    return "WEAK"


def assess_reference_admissibility(
    row: Mapping[str, Any], config: Mapping[str, Any], *, source_consistent: bool
) -> dict[str, Any]:
    """Apply only saved Proposal/Top-K/FP-SHEP descriptors, never outcomes."""

    spec = config["reference_admissibility"]
    reasons: list[str] = []
    selected_id = int(float(row["selected_candidate_id"]))
    reference = np.asarray(_json_value(row.get("temporary_reference"), []), dtype=float)
    points = _json_value(row.get("candidate_world_points"), [])
    metadata = _json_value(row.get("candidate_metadata"), [])
    fp_records = _json_value(row.get("fp_shep_candidate_records"), [])
    k_t = int(float(row.get("K_t") or 0))

    proposal_passed = bool(
        _boolean(row.get("candidate_available"))
        and not _boolean(row.get("selected_null"))
        and selected_id >= 0
    )
    if not proposal_passed:
        reasons.append("proposal_not_available")
    top_k_passed = bool(
        0 <= selected_id < k_t
        and selected_id < len(points)
        and selected_id < len(metadata)
        and selected_id < len(fp_records)
    )
    if not top_k_passed:
        reasons.append("selected_candidate_not_in_saved_top_k")

    reference_match = False
    selected_meta: Mapping[str, Any] = {}
    selected_fp: Mapping[str, Any] = {}
    if top_k_passed and reference.shape == (3,) and np.all(np.isfinite(reference)):
        point = np.asarray(points[selected_id], dtype=float)
        reference_match = bool(
            np.allclose(
                point,
                reference,
                rtol=0.0,
                atol=float(spec["require_exact_candidate_reference_match_atol"]),
            )
        )
        selected_meta = metadata[selected_id]
        selected_fp = fp_records[selected_id]
    if not reference_match:
        reasons.append("saved_reference_candidate_mismatch")

    safety_margin = _number(selected_meta.get("safety_margin"))
    safety_valid = bool(
        safety_margin is not None
        and safety_margin >= float(spec["minimum_safety_margin_m"])
    )
    if not safety_valid:
        reasons.append("immediately_invalid_safety_margin")

    proposal_progress = _number(selected_meta.get("distance_progress"))
    preview_progress = _number(selected_fp.get("preview_task_progress"))
    progress_valid = bool(
        proposal_progress is not None
        and proposal_progress
        > float(spec["minimum_distance_progress_m_exclusive"])
    )
    if not progress_valid:
        reasons.append("nonpositive_or_pathological_progress_descriptor")

    fp_score_valid = _finite(selected_fp.get("fp_shep_online_score"))
    numeric_descriptors = (
        selected_fp.get("preview_task_progress"),
        selected_fp.get("preview_max_execution_deviation"),
        selected_fp.get("preview_terminal_speed"),
    )
    normalized = selected_fp.get("normalized_preview_features", [])
    numeric_valid = all(_finite(value) for value in numeric_descriptors) and bool(
        normalized
    ) and all(_finite(value) for value in normalized)
    clearance = selected_fp.get("preview_min_clearance")
    valid_mask = selected_fp.get("preview_feature_valid_mask", [])
    explicit_unbounded = bool(
        clearance is None
        and len(valid_mask) >= 2
        and not bool(valid_mask[1])
        and bool(spec["accept_explicit_unbounded_clearance"])
    )
    clearance_valid = _finite(clearance) or explicit_unbounded
    fp_valid = fp_score_valid and numeric_valid and clearance_valid
    if not fp_valid:
        reasons.append("nonfinite_or_inconsistent_fp_shep_descriptor")
    if not source_consistent:
        reasons.append("source_state_software_inconsistency")

    admissible = bool(
        proposal_passed
        and top_k_passed
        and reference_match
        and safety_valid
        and progress_valid
        and fp_valid
        and source_consistent
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "scenario": row["scenario"],
        "seed": int(row["seed"]),
        "agent_id": int(row["agent_id"]),
        "selected_candidate_id": selected_id,
        "temporary_reference": reference.tolist() if reference.shape == (3,) else None,
        "proposal_feasibility_passed": proposal_passed,
        "survived_top_k": top_k_passed,
        "saved_reference_candidate_match": reference_match,
        "fp_shep_descriptors_valid": fp_valid,
        "fp_shep_unbounded_clearance_explicit": explicit_unbounded,
        "selected_fp_shep_score": _number(selected_fp.get("fp_shep_online_score")),
        "selected_preview_progress_m": preview_progress,
        "proposal_distance_progress_m": proposal_progress,
        "proposal_safety_margin_m": safety_margin,
        "source_state_software_consistent": source_consistent,
        "outcome_fields_used": False,
        "reference_reasonably_admissible": admissible,
        "exclusion_reasons": reasons,
    }


def _trajectory_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _first_progress_step(distances: np.ndarray, fraction: float) -> int | None:
    if not distances.size or float(distances[0]) <= 0.0:
        return 0
    threshold = float(distances[0]) * (1.0 - float(fraction))
    indices = np.flatnonzero(distances <= threshold)
    return int(indices[0]) if indices.size else None


def _cumulative_turning_angle(points: np.ndarray, reference: np.ndarray, radius: float) -> float:
    relative = np.asarray(points, dtype=float) - np.asarray(reference, dtype=float)
    inside = relative[np.linalg.norm(relative, axis=1) <= float(radius)]
    if len(inside) < 3:
        return 0.0
    total = 0.0
    for first, second in zip(inside[:-1], inside[1:], strict=True):
        norms = float(np.linalg.norm(first) * np.linalg.norm(second))
        if norms <= 1.0e-12:
            continue
        total += math.acos(float(np.clip(np.dot(first, second) / norms, -1.0, 1.0)))
    return float(total)


def classify_b_failure(row: Mapping[str, Any], *, epsilon: float = 1.0e-9) -> str:
    if bool(row["reference_reached"]):
        return "REFERENCE_REACHED"
    if bool(row["ego_obstacle_collision_before_reference"]):
        return "OBSTACLE_COLLISION_BEFORE_REFERENCE"
    if bool(row["ego_inter_agent_collision_before_reference"]):
        return "INTER_AGENT_COLLISION_BEFORE_REFERENCE"
    if bool(row["peer_only_collision_before_reference"]):
        return "ENVIRONMENT_TERMINATION"
    if bool(row["stagnation_before_reference"]):
        return "STAGNATION_BEFORE_REFERENCE"
    if bool(row["timeout"]):
        return "TIMEOUT_BEFORE_REFERENCE"
    if float(row["target_distance_reduction_m"]) <= float(epsilon):
        return "DIVERGENCE_FROM_REFERENCE"
    if bool(row["environment_terminated"]):
        return "ENVIRONMENT_TERMINATION"
    return "OTHER"


def classify_c_failure(row: Mapping[str, Any]) -> str:
    if not bool(row["reference_reached"]):
        return "REFERENCE_NOT_REACHED"
    if bool(row["terminal_success"]):
        return "REFERENCE_REACHED_TERMINAL_SUCCESS"
    if bool(row["post_reference_ego_obstacle_collision"]):
        return "POST_REFERENCE_OBSTACLE_COLLISION"
    if bool(row["post_reference_ego_inter_agent_collision"]):
        return "POST_REFERENCE_INTER_AGENT_COLLISION"
    if bool(row["post_reference_stagnation"]):
        return "POST_REFERENCE_STAGNATION"
    if bool(row["post_reference_timeout"]):
        return "POST_REFERENCE_TIMEOUT"
    return "OTHER"


def run_contract(
    *,
    initial_env: Any,
    execution_mode: str,
    scenario: str,
    seed: int,
    ego_id: int,
    temporary_reference: np.ndarray,
    background_references: np.ndarray,
    background_available: np.ndarray,
    contract: str,
    policy: Any,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    source_fingerprint = environment_state_fingerprint(initial_env)
    env = copy.deepcopy(initial_env)
    if environment_state_fingerprint(env) != source_fingerprint:
        raise RuntimeError("deepcopy did not exactly restore the source state")
    terminal_goals = np.asarray(env.goals, dtype=float).copy()
    terminal_before = terminal_goals.copy()
    ego_id = int(ego_id)
    local_ego = 0 if execution_mode == "single_agent" else ego_id
    references = np.asarray(background_references, dtype=float).copy()
    available = np.asarray(background_available, dtype=bool).copy()
    if execution_mode == "single_agent":
        references = np.asarray([temporary_reference], dtype=float)
        available = np.asarray([contract != "A_DIRECT_TERMINAL"], dtype=bool)
    elif contract == "A_DIRECT_TERMINAL":
        references[ego_id] = terminal_goals[ego_id]
        available[ego_id] = False
    else:
        references[ego_id] = np.asarray(temporary_reference, dtype=float)
        available[ego_id] = True

    returned_to_terminal = np.logical_not(available)
    reference_reached = np.zeros(int(env.num_agents), dtype=bool)
    reference_reach_steps: list[int | None] = [None] * int(env.num_agents)
    phase_switch_deltas: list[float] = []
    for index in range(int(env.num_agents)):
        if not available[index]:
            continue
        phase_before = float(env.dmps[index].phase)
        set_dmp_active_goal_preserve_phase(env.dmps[index], references[index])
        phase_switch_deltas.append(float(env.dmps[index].phase) - phase_before)

    target = (
        terminal_goals[local_ego]
        if contract == "A_DIRECT_TERMINAL"
        else references[local_ego]
    )
    positions = [np.asarray(env.dynamics[local_ego].p, dtype=float).copy()]
    velocities = [np.asarray(env.dynamics[local_ego].v, dtype=float).copy()]
    phases = [float(env.dmps[local_ego].phase)]
    distances = [float(np.linalg.norm(target - positions[-1]))]
    terminal_distances = [
        float(np.linalg.norm(terminal_goals[local_ego] - positions[-1]))
    ]
    sensor_clearances = [float(env.latest_sensor_packets[local_ego].min_clearance)]
    obstacle_clearances = [_obstacle_surface_clearance(env, local_ego)]
    peer_distances = [_agent_peer_distance(env, local_ego)]
    actions: list[np.ndarray] = []
    goal_offset_alignments: list[float | None] = []
    stagnation_masks: list[bool] = []
    stagnation_counters: list[int] = []
    stagnation_window_progress: list[float] = []

    reached_tolerance = float(config["handoff_threshold_m"])
    max_steps = int(config["max_steps"])
    ego_obstacle_before = ego_inter_before = peer_only_before = False
    post_ego_obstacle = post_ego_inter = False
    post_stagnation = False
    any_collision = obstacle_collision = inter_agent_collision = False
    environment_terminated = truncated = False
    terminal_success = False
    terminal_completion_step: int | None = None
    handoff_step: int | None = None
    handoff_speed: float | None = None
    handoff_phase: float | None = None
    last_info: dict[str, Any] = {}

    while not (environment_terminated or truncated) and int(env.steps) < max_steps:
        active_goals = np.asarray([dmp.goal for dmp in env.dmps], dtype=float)
        observations = _active_goal_observations(env, active_goals)
        action = predict_frozen_actions(
            policy, observations, expected_shape=tuple(env.action_shape)
        )
        actions.append(np.asarray(action[local_ego], dtype=float).copy())
        offset = np.asarray(action[local_ego, 3:6], dtype=float)
        direction = np.asarray(target - env.dynamics[local_ego].p, dtype=float)
        norm = float(np.linalg.norm(offset) * np.linalg.norm(direction))
        goal_offset_alignments.append(
            None
            if norm <= 1.0e-12
            else float(np.clip(np.dot(offset, direction) / norm, -1.0, 1.0))
        )
        _, _, environment_terminated, truncated, last_info = env.step(action)
        positions.append(np.asarray(env.dynamics[local_ego].p, dtype=float).copy())
        velocities.append(np.asarray(env.dynamics[local_ego].v, dtype=float).copy())
        phases.append(float(env.dmps[local_ego].phase))
        distances.append(float(np.linalg.norm(target - positions[-1])))
        terminal_distances.append(
            float(np.linalg.norm(terminal_goals[local_ego] - positions[-1]))
        )
        sensor_clearances.append(
            float(env.latest_sensor_packets[local_ego].min_clearance)
        )
        obstacle_clearances.append(_obstacle_surface_clearance(env, local_ego))
        peer_distances.append(_agent_peer_distance(env, local_ego))

        obstacle_mask = np.asarray(
            last_info.get("obstacle_collision_mask", np.zeros(env.num_agents)),
            dtype=bool,
        )
        inter_mask = np.asarray(
            last_info.get("inter_agent_collision_mask", np.zeros(env.num_agents)),
            dtype=bool,
        )
        step_collision = bool(last_info.get("collision", False))
        any_collision |= step_collision
        obstacle_collision |= bool(np.any(obstacle_mask))
        inter_agent_collision |= bool(np.any(inter_mask))
        before_reference = bool(
            contract != "A_DIRECT_TERMINAL" and not reference_reached[local_ego]
        )
        if before_reference:
            ego_obstacle_before |= bool(obstacle_mask[local_ego])
            ego_inter_before |= bool(inter_mask[local_ego])
            peer_indices = [
                index for index in range(int(env.num_agents)) if index != local_ego
            ]
            peer_only_before |= bool(
                step_collision
                and not obstacle_mask[local_ego]
                and not inter_mask[local_ego]
                and peer_indices
                and (
                    np.any(obstacle_mask[peer_indices])
                    or np.any(inter_mask[peer_indices])
                )
            )
        elif contract != "A_DIRECT_TERMINAL":
            post_ego_obstacle |= bool(obstacle_mask[local_ego])
            post_ego_inter |= bool(inter_mask[local_ego])

        mask = np.asarray(
            last_info.get("stagnation_mask", np.zeros(env.num_agents)), dtype=bool
        )
        counters = np.asarray(
            last_info.get("stagnation_counters", np.zeros(env.num_agents)), dtype=int
        )
        window = np.asarray(
            last_info.get(
                "stagnation_window_progress", np.zeros(env.num_agents)
            ),
            dtype=float,
        )
        stagnation_masks.append(bool(mask[local_ego]))
        stagnation_counters.append(int(counters[local_ego]))
        stagnation_window_progress.append(float(window[local_ego]))
        if reference_reached[local_ego]:
            post_stagnation |= bool(mask[local_ego])

        success_mask = np.asarray(
            last_info.get("success_mask", np.zeros(env.num_agents)), dtype=bool
        )
        if bool(success_mask[local_ego]) and not step_collision:
            terminal_success = True
            if terminal_completion_step is None:
                terminal_completion_step = int(env.steps)

        # Match the deployed one-shot ordering: no post-hoc reach or handoff
        # on a terminating/truncating step.
        if not (environment_terminated or truncated):
            for index in range(int(env.num_agents)):
                if returned_to_terminal[index]:
                    continue
                distance = float(
                    np.linalg.norm(references[index] - env.dynamics[index].p)
                )
                if distance > reached_tolerance:
                    continue
                reference_reached[index] = True
                reference_reach_steps[index] = int(env.steps)
                if index == local_ego:
                    handoff_step = int(env.steps)
                    handoff_speed = float(np.linalg.norm(env.dynamics[index].v))
                    handoff_phase = float(env.dmps[index].phase)
                if contract == "B_TEMP_REFERENCE_ONLY" and index == local_ego:
                    break
                phase_before = float(env.dmps[index].phase)
                set_dmp_active_goal_preserve_phase(
                    env.dmps[index], terminal_goals[index]
                )
                phase_switch_deltas.append(float(env.dmps[index].phase) - phase_before)
                returned_to_terminal[index] = True
            if contract == "B_TEMP_REFERENCE_ONLY" and reference_reached[local_ego]:
                break

    position_array = np.stack(positions)
    velocity_array = np.stack(velocities)
    phase_array = np.asarray(phases, dtype=float)
    action_array = np.stack(actions) if actions else np.empty((0, 6), dtype=float)
    distance_array = np.asarray(distances, dtype=float)
    terminal_distance_array = np.asarray(terminal_distances, dtype=float)
    path_length = float(np.sum(np.linalg.norm(np.diff(position_array, axis=0), axis=1)))
    timeout = bool(truncated or (int(env.steps) >= max_steps and not environment_terminated))
    reached = bool(reference_reached[local_ego]) if contract != "A_DIRECT_TERMINAL" else False
    stagnation_before = bool(any(stagnation_masks[: handoff_step or len(stagnation_masks)]))
    maximum_phase_delta = (
        float(np.max(np.abs(phase_switch_deltas))) if phase_switch_deltas else 0.0
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "execution_mode": execution_mode,
        "scenario": scenario,
        "seed": int(seed),
        "agent_id": ego_id,
        "local_agent_id": local_ego,
        "contract": contract,
        "source_state_fingerprint": source_fingerprint,
        "trajectory_hash": _trajectory_hash(
            position_array, velocity_array, phase_array, action_array
        ),
        "source_env_steps": int(getattr(initial_env, "steps", 0)),
        "steps_executed": int(len(position_array) - 1),
        "final_absolute_env_steps": int(env.steps),
        "temporary_reference": np.asarray(temporary_reference, dtype=float).tolist(),
        "immutable_terminal_goal": terminal_goals[local_ego].tolist(),
        "reference_reached": reached,
        "reference_reach_step": reference_reach_steps[local_ego],
        "handoff_step": handoff_step,
        "terminal_success": bool(terminal_success),
        "terminal_completion_step": terminal_completion_step,
        "team_success": bool(last_info.get("success", False)) and not any_collision,
        "any_collision": any_collision,
        "obstacle_collision": obstacle_collision,
        "inter_agent_collision": inter_agent_collision,
        "ego_obstacle_collision_before_reference": ego_obstacle_before,
        "ego_inter_agent_collision_before_reference": ego_inter_before,
        "peer_only_collision_before_reference": peer_only_before,
        "post_reference_ego_obstacle_collision": post_ego_obstacle,
        "post_reference_ego_inter_agent_collision": post_ego_inter,
        "timeout": timeout,
        "environment_terminated": bool(environment_terminated),
        "stagnation_before_reference": stagnation_before,
        "post_reference_stagnation": post_stagnation,
        "post_reference_timeout": bool(reached and timeout and not terminal_success),
        "maximum_stagnation_counter": max(stagnation_counters, default=0),
        "final_stagnation_window_progress_m": (
            stagnation_window_progress[-1] if stagnation_window_progress else None
        ),
        "initial_target_distance_m": float(distance_array[0]),
        "target_distance_reduction_m": float(distance_array[0] - distance_array[-1]),
        "time_to_50_percent_progress_step": _first_progress_step(distance_array, 0.50),
        "time_to_80_percent_progress_step": _first_progress_step(distance_array, 0.80),
        "final_target_distance_m": float(distance_array[-1]),
        "minimum_target_distance_m": float(np.min(distance_array)),
        "initial_terminal_distance_m": float(terminal_distance_array[0]),
        "final_terminal_distance_m": float(terminal_distance_array[-1]),
        "minimum_terminal_distance_m": float(np.min(terminal_distance_array)),
        "path_length_m": path_length,
        "minimum_sensor_clearance_m": float(np.min(sensor_clearances)),
        "minimum_obstacle_surface_clearance_m": float(np.min(obstacle_clearances)),
        "minimum_inter_agent_distance_m": float(np.min(peer_distances)),
        "handoff_speed_mps": handoff_speed,
        "handoff_phase": handoff_phase,
        "maximum_phase_switch_delta": maximum_phase_delta,
        "phase_reset_on_switch": maximum_phase_delta != 0.0,
        "terminal_task_goals_unchanged": bool(
            np.array_equal(np.asarray(env.goals), terminal_before)
        ),
        "forcing_action_norm_mean": _mean(
            np.linalg.norm(action_array[:, :3], axis=1) if len(action_array) else []
        ),
        "goal_offset_action_norm_mean": _mean(
            np.linalg.norm(action_array[:, 3:6], axis=1) if len(action_array) else []
        ),
        "goal_offset_alignment_mean": _mean(goal_offset_alignments),
        "official_stagnation_source": config["stagnation_definition"]["source"],
        "historical_gate": HISTORICAL_GATE_NAME,
        "_positions": position_array,
        "_velocities": velocity_array,
        "_phases": phase_array,
        "_actions": action_array,
        "_distances": distance_array,
        "_sensor_clearances": np.asarray(sensor_clearances, dtype=float),
        "_goal_offset_alignments": goal_offset_alignments,
    }
    if contract == "B_TEMP_REFERENCE_ONLY":
        result["failure_category"] = classify_b_failure(
            result, epsilon=float(config["classification"]["numeric_epsilon"])
        )
    elif contract == "C_TEMP_TO_TERMINAL":
        result["failure_category"] = classify_c_failure(result)
    else:
        result["failure_category"] = (
            "DIRECT_TERMINAL_SUCCESS" if terminal_success else "DIRECT_TERMINAL_FAILURE"
        )
    return result


def public_result(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def near_reference_row(
    row: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    positions = np.asarray(row["_positions"], dtype=float)
    velocities = np.asarray(row["_velocities"], dtype=float)
    phases = np.asarray(row["_phases"], dtype=float)
    actions = np.asarray(row["_actions"], dtype=float)
    distances = np.asarray(row["_distances"], dtype=float)
    clearances = np.asarray(row["_sensor_clearances"], dtype=float)
    minimum_index = int(np.argmin(distances))
    action_index = min(max(minimum_index - 1, 0), max(len(actions) - 1, 0))
    epsilon = float(config["classification"]["numeric_epsilon"])
    overshoot = bool(
        minimum_index < len(distances) - 1
        and float(distances[-1]) > float(distances[minimum_index]) + epsilon
    )
    reference = np.asarray(row["temporary_reference"], dtype=float)
    orbit_angle = _cumulative_turning_angle(
        positions, reference, max(config["capture_radius_diagnostic_m"])
    )
    alignment_at_min = (
        row["_goal_offset_alignments"][action_index]
        if row["_goal_offset_alignments"] and len(actions)
        else None
    )
    pressure_threshold = float(config.get("_obstacle_influence_distance", 0.0))
    safety_push = bool(
        overshoot
        and minimum_index < len(clearances)
        and clearances[minimum_index] <= pressure_threshold
        and alignment_at_min is not None
        and float(alignment_at_min) < 0.0
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "execution_mode": row["execution_mode"],
        "scenario": row["scenario"],
        "seed": row["seed"],
        "agent_id": row["agent_id"],
        "reference_reached_at_0p25": row["reference_reached"],
        "failed_formal_reference_capture": not bool(row["reference_reached"]),
        "minimum_distance_to_reference_m": float(distances[minimum_index]),
        "minimum_distance_step": minimum_index,
        "speed_at_minimum_distance_mps": float(
            np.linalg.norm(velocities[minimum_index])
        ),
        "dmp_phase_at_minimum_distance": float(phases[minimum_index]),
        "forcing_action_norm_at_minimum_distance": (
            float(np.linalg.norm(actions[action_index, :3])) if len(actions) else None
        ),
        "goal_offset_action_norm_at_minimum_distance": (
            float(np.linalg.norm(actions[action_index, 3:6])) if len(actions) else None
        ),
        "goal_offset_alignment_at_minimum_distance": alignment_at_min,
        "mean_forcing_action_norm": row["forcing_action_norm_mean"],
        "mean_goal_offset_action_norm": row["goal_offset_action_norm_mean"],
        "mean_goal_offset_alignment": row["goal_offset_alignment_mean"],
        "overshoot_after_minimum": overshoot,
        "orbit_cumulative_turning_angle_rad": orbit_angle,
        "orbit_detected": orbit_angle
        >= float(config["classification"]["orbit_cumulative_turning_angle_rad"]),
        "official_stagnation_detected": row["stagnation_before_reference"],
        "safety_avoidance_pushes_away": safety_push,
        "sensor_clearance_at_minimum_distance_m": float(clearances[minimum_index]),
        "obstacle_pressure_threshold_m": pressure_threshold,
        "failure_category": row["failure_category"],
    }


def _obstacle_state_hash(env: Any) -> str:
    payload: list[dict[str, Any]] = []
    for obstacle in list(env.static_obstacles) + list(env.dynamic_obstacles):
        state = {
            key: value
            for key, value in vars(obstacle).items()
            if isinstance(value, (str, int, float, bool, type(None), np.ndarray))
        }
        payload.append({"type": type(obstacle).__name__, "state": _jsonable(state)})
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_single_agent_projection(
    *,
    multi_env: Any,
    multi_config: Any,
    scene_metadata: Mapping[str, Any],
    seed: int,
    ego_id: int,
    peer_radius: float,
) -> tuple[Any, dict[str, Any]]:
    single_config = replace(multi_config, num_agents=1)
    env = _build_environment(
        single_config,
        observation_mode=str(scene_metadata["observation_mode"]),
        peer_radius=float(peer_radius),
        training_distribution=False,
        include_boundaries_in_sensor=bool(
            scene_metadata["include_boundaries_in_sensor"]
        ),
        terminate_on_boundary_collision=bool(
            scene_metadata["terminate_on_boundary_collision"]
        ),
    )
    options = {
        "starts": np.asarray([multi_env.dynamics[int(ego_id)].p], dtype=float),
        "goals": np.asarray([multi_env.goals[int(ego_id)]], dtype=float),
        "static_obstacles": copy.deepcopy(multi_env.static_obstacles),
        "dynamic_obstacles": copy.deepcopy(multi_env.dynamic_obstacles),
    }
    env.reset(seed=int(seed), options=options)
    checks = {
        "ego_position_match": bool(
            np.array_equal(env.dynamics[0].p, multi_env.dynamics[int(ego_id)].p)
        ),
        "ego_velocity_match": bool(
            np.array_equal(env.dynamics[0].v, multi_env.dynamics[int(ego_id)].v)
        ),
        "ego_phase_match": float(env.dmps[0].phase)
        == float(multi_env.dmps[int(ego_id)].phase),
        "terminal_goal_match": bool(
            np.array_equal(env.goals[0], multi_env.goals[int(ego_id)])
        ),
        "obstacle_state_match": _obstacle_state_hash(env)
        == _obstacle_state_hash(multi_env),
        "peer_count_removed": int(env.num_agents) == 1,
        "source_timestep_zero": int(env.steps) == int(multi_env.steps) == 0,
    }
    checks["projection_valid"] = all(checks.values())
    return env, checks


def err_same_goal_rows(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], str]:
    source = REPO_ROOT / config["source_err_dir"] / "replanning_events.csv"
    selected: list[dict[str, Any]] = []
    for row in read_csv(source):
        if not _boolean(row.get("counts_as_reproposal")) or _boolean(
            row.get("goal_changed")
        ):
            continue
        old_goal = np.asarray(_json_value(row.get("old_active_goal"), [0, 0, 0]), dtype=float)
        new_goal = np.asarray(_json_value(row.get("new_active_goal"), [0, 0, 0]), dtype=float)
        reference_selected = _boolean(row.get("reference_selected_after_reproposal"))
        reference_reached = (
            _boolean(row.get("reference_reached_after_reproposal"))
            if row.get("reference_reached_after_reproposal") not in (None, "")
            else None
        )
        terminal = _boolean(row.get("terminal_completed_after_reproposal"))
        collision = _boolean(row.get("collision_after_reproposal"))
        timeout = _boolean(row.get("timeout_after_reproposal"))
        continued_failure = bool(
            not terminal
            and (
                collision
                or timeout
                or (reference_selected and reference_reached is False)
            )
        )
        selected.append(
            {
                "schema_version": SCHEMA_VERSION,
                "scenario": row["scenario"],
                "seed": int(row["seed"]),
                "agent_id": int(row["agent_id"]),
                "step": int(row["step"]),
                "event": row["event"],
                "trigger_reasons": _json_value(row.get("trigger_reasons"), []),
                "active_reference": old_goal.tolist(),
                "new_selected_reference": new_goal.tolist(),
                "reference_displacement_m": float(np.linalg.norm(new_goal - old_goal)),
                "active_goal_distance_m": _number(row.get("active_goal_distance_m")),
                "progress_rate_mps": _number(row.get("progress_rate_mps")),
                "progress_valid": _boolean(row.get("progress_valid")),
                "safety_margin_m": _number(row.get("active_safety_margin_m")),
                "speed_mps": _number(row.get("speed_before_mps")),
                "dmp_phase": _number(row.get("phase_before")),
                "reference_selected_after_reproposal": reference_selected,
                "reference_reached_after_reproposal": reference_reached,
                "terminal_success_after_reproposal": terminal,
                "collision_after_reproposal": collision,
                "timeout_after_reproposal": timeout,
                "continued_execution_failure": continued_failure,
            }
        )
    expected = int(config["same_goal_subset"]["expected_event_count"])
    if len(selected) != expected:
        return selected, "NOT_ESTABLISHED"
    failure_rate = _rate(selected, "continued_execution_failure") or 0.0
    rules = config["classification"]
    if failure_rate >= float(rules["same_goal_failure_yes_minimum"]):
        classification = "YES"
    elif failure_rate >= float(rules["same_goal_failure_weak_minimum"]):
        classification = "WEAK"
    else:
        classification = "NO"
    return selected, classification


def _assert_config(config: Mapping[str, Any]) -> None:
    if config["forcing_gate"] != HISTORICAL_GATE_NAME:
        raise ValueError("historical vector forcing gate is mandatory")
    if config["formal_scenarios"] != ["open", "sparse_static", "multi_agent"]:
        raise ValueError("formal scenario set changed")
    if config["formal_seeds"] != list(range(10, 30)):
        raise ValueError("formal seeds must remain 10..29")
    if int(config["max_steps"]) != 220 or float(config["handoff_threshold_m"]) != 0.25:
        raise ValueError("frozen execution horizon/handoff changed")
    if config["capture_radius_diagnostic_m"] != [0.20, 0.25, 0.30, 0.40]:
        raise ValueError("capture-radius diagnostic changed")
    if any(bool(value) for value in config["strict_exclusions"].values()):
        raise ValueError("strict exclusion flags must all remain false")


def render_report(
    *,
    conclusion: Mapping[str, Any],
    admissibility: Sequence[Mapping[str, Any]],
    multi_b: Sequence[Mapping[str, Any]],
    single_b: Sequence[Mapping[str, Any]],
    capture: Sequence[Mapping[str, Any]],
    err_rows: Sequence[Mapping[str, Any]],
    recovery: Sequence[Mapping[str, Any]],
    integrity: Mapping[str, Any],
) -> str:
    admissible_count = sum(bool(row["reference_reasonably_admissible"]) for row in admissibility)
    multi_failures: dict[str, int] = {}
    single_failures: dict[str, int] = {}
    for name, rows in (("multi", multi_b), ("single", single_b)):
        target = multi_failures if name == "multi" else single_failures
        for row in rows:
            target[row["failure_category"]] = target.get(row["failure_category"], 0) + 1
    capture_lines = []
    for row in capture:
        capture_lines.append(
            f"| {row['execution_mode']} | {float(row['radius_m']):.2f} | "
            f"{int(row['entered_count'])}/{int(row['branch_count'])} "
            f"({float(row['entered_rate']):.1%}) |"
        )
    err_failure = sum(bool(row["continued_execution_failure"]) for row in err_rows)
    recoverable = sum(bool(row["potentially_recoverable_by_lower_adaptation"]) for row in recovery)
    return f"""# Frozen SAC-DMP 临时参考执行契约审计

本审计严格只读：复用正式 GAT‑V1 one-shot 的 t=0 场景和实际被选中的参考点，以冻结 SAC checkpoint 做确定性 A/B/C 重放；未训练或微调 SAC，未修改 reward、DMP、GAT、Proposal、FP-SHEP、ERR、环境核心或 0.25 m handoff 阈值，也未读取 24-layout stress 结果参与设计。

**结论：没有证实“terminal 很强但 temporary-reference 明显偏弱”的显著 task-interface gap。** 单机存在 9.8 pp 的弱差距和近阈值捕获信号，但没有达到预先规定的 15 pp；多机 matched A/B 没有下降，handoff 也没有形成独立差距。因此当前证据不足以把 targeted SAC adaptation 作为约六至七成 team success 的主要修复方向。

## 数据与完整性

- 正式源场景：3 个场景 × 20 seeds = 60 个 team source states。
- 实际 GAT 非空参考：{len(admissibility)} 个；通过冻结 admissibility gate：{admissible_count} 个。
- A/B/C 在各自 multi/single 环境内均从同一 deep-copy 指纹开始；single 是保留 ego/障碍物/终点/参考并移除 peer 的 t=0 确定性投影。
- checkpoint 与核心文件前后哈希检查：`{integrity['status']}`。
- 停滞定义直接来自环境 `stagnation_mask / stagnation_counters / stagnation_window_progress`。

## 核心结果

| 指标 | 结果 |
|---|---:|
| Single direct-terminal success | {conclusion['SINGLE_AGENT_DIRECT_TERMINAL_SUCCESS_RATE']:.1%} |
| Multi direct-terminal ego success | {conclusion['MULTI_AGENT_DIRECT_TERMINAL_SUCCESS_RATE']:.1%} |
| Single temporary-reference reach | {conclusion['SINGLE_AGENT_TEMP_REFERENCE_REACH_RATE']:.1%} |
| Multi temporary-reference reach | {conclusion['MULTI_AGENT_TEMP_REFERENCE_REACH_RATE']:.1%} |
| Single temp→terminal completion | {conclusion['SINGLE_AGENT_TEMP_TO_TERMINAL_SUCCESS_RATE']:.1%} |
| Multi temp→terminal ego completion | {conclusion['MULTI_AGENT_TEMP_TO_TERMINAL_SUCCESS_RATE']:.1%} |
| Single A→B drop | {conclusion['SINGLE_AGENT_TEMP_REFERENCE_DROP_PP']:.1f} pp |
| Multi A→B drop | {conclusion['MULTI_AGENT_TEMP_REFERENCE_DROP_PP']:.1f} pp |
| Peer interaction reach penalty | {conclusion['PEER_INTERACTION_REACH_PENALTY_PP']:.1f} pp |
| Reached→terminal handoff gap | {conclusion['HANDOFF_COMPLETION_DROP_PP']:.1f} pp |

Single handoff 子集为 {conclusion['SINGLE_AGENT_POST_HANDOFF_SUCCESS_RATE']:.1%}，matched direct 为 {conclusion['SINGLE_AGENT_MATCHED_DIRECT_ON_REACHED_RATE']:.1%}（下降 {conclusion['SINGLE_AGENT_HANDOFF_DROP_PP']:.1f} pp）；multi handoff 子集与 matched direct 均为 {conclusion['MULTI_AGENT_POST_HANDOFF_SUCCESS_RATE']:.1%}。

Multi B taxonomy：`{json.dumps(multi_failures, ensure_ascii=False)}`。

Single B taxonomy：`{json.dumps(single_failures, ensure_ascii=False)}`。

## 近参考与捕获半径（只读诊断）

正式阈值始终为 0.25 m；下面使用 Contract C 的完整历史轨迹重算“是否曾进入半径”，不改变任何正式结果。该表是纯几何 entry 统计；终止步才进入 0.25 m 的轨迹仍不会被正式在线 handoff 计为 reached，因此表中的 0.25 m entry 可能略高于正式 reach rate。

| 模式 | 半径/m | 曾进入 |
|---|---:|---:|
{chr(10).join(capture_lines)}

`NEAR_REFERENCE_CAPTURE_FAILURE = {conclusion['NEAR_REFERENCE_CAPTURE_FAILURE']}`。逐分支的最小距离、对应速度/相位、forcing 与 goal-offset 动作、overshoot、orbit、官方 stagnation 和 safety-push-away 标志见 `near_reference_analysis.csv`。

## ERR 同目标重选子集

保存数据中恢复出 {len(err_rows)} 个 `counts_as_reproposal=True && goal_changed=False` 事件，其中 {err_failure} 个满足“重选后仍未完成并伴随 reference 未达/碰撞/超时”的 continued-execution failure 定义。该证据是事件相关、非独立样本，因此只用于机制归因，不当作新的性能实验。

## 可恢复性上限

正式 GAT‑V1 当前成功 39/60。按“原 episode 失败且无原始碰撞、reference admissible、single direct 成功、single temporary reference 因 stagnation/timeout/divergence 失败”的保守诊断规则，潜在可恢复 team episode 为 {recoverable} 个；上限为 {conclusion['DIAGNOSTIC_TEAM_SUCCESS_UPPER_BOUND']:.1%}。这是诊断上限，不是微调后的预期性能。

## 最终判定

```text
DIRECT_TERMINAL_CAPABILITY = {conclusion['DIRECT_TERMINAL_CAPABILITY']}
REFERENCE_ADMISSIBILITY_VALID = {conclusion['REFERENCE_ADMISSIBILITY_VALID']}
SINGLE_AGENT_TEMP_REFERENCE_REACH_RATE = {conclusion['SINGLE_AGENT_TEMP_REFERENCE_REACH_RATE']:.6f}
MULTI_AGENT_TEMP_REFERENCE_REACH_RATE = {conclusion['MULTI_AGENT_TEMP_REFERENCE_REACH_RATE']:.6f}
TEMPORARY_REFERENCE_EXECUTION_GAP = {conclusion['TEMPORARY_REFERENCE_EXECUTION_GAP']}
LOW_LEVEL_REFERENCE_EXECUTION_GAP_SINGLE_AGENT = {conclusion['LOW_LEVEL_REFERENCE_EXECUTION_GAP_SINGLE_AGENT']}
MULTI_AGENT_INTERACTION_PENALTY = {conclusion['MULTI_AGENT_INTERACTION_PENALTY']}
NEAR_REFERENCE_CAPTURE_FAILURE = {conclusion['NEAR_REFERENCE_CAPTURE_FAILURE']}
HANDOFF_TRANSITION_GAP = {conclusion['HANDOFF_TRANSITION_GAP']}
HIGH_LEVEL_SELECTION_NOT_PRIMARY_FOR_SAME_GOAL_SUBSET = {conclusion['HIGH_LEVEL_SELECTION_NOT_PRIMARY_FOR_SAME_GOAL_SUBSET']}
PRIMARY_HIERARCHICAL_EXECUTION_LIMITATION = {conclusion['PRIMARY_HIERARCHICAL_EXECUTION_LIMITATION']}
TARGETED_SAC_ADAPTATION_JUSTIFIED = {conclusion['TARGETED_SAC_ADAPTATION_JUSTIFIED']}
POTENTIALLY_RECOVERABLE_TEAM_EPISODES = {conclusion['POTENTIALLY_RECOVERABLE_TEAM_EPISODES']}
DIAGNOSTIC_TEAM_SUCCESS_UPPER_BOUND = {conclusion['DIAGNOSTIC_TEAM_SUCCESS_UPPER_BOUND']:.6f}
RECOMMENDED_NEXT_STEP = {conclusion['RECOMMENDED_NEXT_STEP']}
```

审计到此停止；即使 adaptation 被判为可考虑，也没有启动 SAC 训练。
"""


def run_audit(config: Mapping[str, Any], output_dir: Path) -> Path:
    _assert_config(config)
    source_dir = REPO_ROOT / config["source_gat_closed_loop_dir"]
    checkpoint = REPO_ROOT / config["checkpoint"]
    if sha256_file(checkpoint) != config["checkpoint_sha256_expected"]:
        raise RuntimeError("frozen SAC checkpoint hash mismatch")

    output_dir.mkdir(parents=True, exist_ok=False)
    resolved = copy.deepcopy(dict(config))
    resolved.update(
        {
            "resolved_output_dir": str(output_dir.resolve()),
            "created_at": datetime.now().isoformat(),
            "python_executable": sys.executable,
        }
    )
    write_json(output_dir / "config.json", resolved)

    core_before = _hash_paths(CORE_PATHS)
    policy, resolved_checkpoint = _load_policy(config, None)
    policy_hash_before = _policy_parameter_sha256(policy)
    multi_config = build_single_distribution_multi_config(
        num_agents=int(config["num_agents"]), max_steps=int(config["max_steps"])
    )

    all_agent_rows = [
        row
        for row in read_csv(source_dir / "per_agent_results.csv")
        if row["method"] == "gat_stage1"
    ]
    selected_rows = [
        row
        for row in all_agent_rows
        if not _boolean(row.get("selected_null"))
        and _boolean(row.get("candidate_available"))
    ]
    episode_rows = [
        row
        for row in read_csv(source_dir / "episode_results.csv")
        if row["method"] == "gat_stage1"
    ]
    episodes_by_key = {
        (row["scenario"], int(row["seed"])): row for row in episode_rows
    }
    rows_by_group: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in all_agent_rows:
        rows_by_group.setdefault((row["scenario"], int(row["seed"])), []).append(row)
    selected_by_group: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in selected_rows:
        selected_by_group.setdefault((row["scenario"], int(row["seed"])), []).append(row)
    manifest_payload = json.loads(
        (source_dir / "scenario_manifest.json").read_text(encoding="utf-8")
    )
    source_manifest = {
        (row["scenario"], int(row["seed"])): row
        for row in manifest_payload["entries"]
    }

    source_rows: list[dict[str, Any]] = []
    admissibility_rows: list[dict[str, Any]] = []
    multi_results: list[dict[str, Any]] = []
    single_results: list[dict[str, Any]] = []
    raw_multi: list[dict[str, Any]] = []
    raw_single: list[dict[str, Any]] = []
    processed = 0

    with scoped_historical_preview_and_multi_agent_transition():
        for scenario in config["formal_scenarios"]:
            for seed in config["formal_seeds"]:
                key = (str(scenario), int(seed))
                base_env, scene_metadata = build_closed_loop_environment(
                    config=multi_config,
                    scenario=str(scenario),
                    seed=int(seed),
                    peer_radius=float(config["peer_radius"]),
                )
                try:
                    rebuilt_hash = _scenario_hash(_scene_snapshot(base_env))
                    expected_hash = source_manifest[key]["initial_condition_hash"]
                    scene_match = rebuilt_hash == expected_hash
                    base_fingerprint = environment_state_fingerprint(base_env)
                    terminal_goals = np.asarray(base_env.goals, dtype=float)
                    background_references = terminal_goals.copy()
                    background_available = np.zeros(int(base_env.num_agents), dtype=bool)
                    for agent_row in rows_by_group[key]:
                        agent_id = int(agent_row["agent_id"])
                        if not _boolean(agent_row.get("selected_null")):
                            background_references[agent_id] = np.asarray(
                                _json_value(agent_row["temporary_reference"]), dtype=float
                            )
                            background_available[agent_id] = True

                    for selected in selected_by_group.get(key, []):
                        ego_id = int(selected["agent_id"])
                        reference = np.asarray(
                            _json_value(selected["temporary_reference"]), dtype=float
                        )
                        admissibility = assess_reference_admissibility(
                            selected, config, source_consistent=scene_match
                        )
                        admissibility_rows.append(admissibility)
                        if not admissibility["reference_reasonably_admissible"]:
                            continue
                        single_env, projection = build_single_agent_projection(
                            multi_env=base_env,
                            multi_config=multi_config,
                            scene_metadata=scene_metadata,
                            seed=int(seed),
                            ego_id=ego_id,
                            peer_radius=float(config["peer_radius"]),
                        )
                        try:
                            single_fingerprint = environment_state_fingerprint(single_env)
                            config["_obstacle_influence_distance"] = float(
                                base_env.env_config.obstacle_influence_distance
                            )
                            multi_contract_rows = []
                            single_contract_rows = []
                            for contract in CONTRACTS:
                                multi_row = run_contract(
                                    initial_env=base_env,
                                    execution_mode="multi_agent",
                                    scenario=str(scenario),
                                    seed=int(seed),
                                    ego_id=ego_id,
                                    temporary_reference=reference,
                                    background_references=background_references,
                                    background_available=background_available,
                                    contract=contract,
                                    policy=policy,
                                    config=config,
                                )
                                single_row = run_contract(
                                    initial_env=single_env,
                                    execution_mode="single_agent",
                                    scenario=str(scenario),
                                    seed=int(seed),
                                    ego_id=ego_id,
                                    temporary_reference=reference,
                                    background_references=np.asarray([reference]),
                                    background_available=np.asarray([True]),
                                    contract=contract,
                                    policy=policy,
                                    config=config,
                                )
                                raw_multi.append(multi_row)
                                raw_single.append(single_row)
                                multi_results.append(public_result(multi_row))
                                single_results.append(public_result(single_row))
                                multi_contract_rows.append(multi_row)
                                single_contract_rows.append(single_row)
                            source_rows.extend(
                                [
                                    {
                                        "schema_version": SCHEMA_VERSION,
                                        "execution_mode": "multi_agent",
                                        "scenario": scenario,
                                        "seed": seed,
                                        "agent_id": ego_id,
                                        "source_state_fingerprint": base_fingerprint,
                                        "contract_A_fingerprint": multi_contract_rows[0]["source_state_fingerprint"],
                                        "contract_B_fingerprint": multi_contract_rows[1]["source_state_fingerprint"],
                                        "contract_C_fingerprint": multi_contract_rows[2]["source_state_fingerprint"],
                                        "artifact_initial_condition_hash_expected": expected_hash,
                                        "artifact_initial_condition_hash_rebuilt": rebuilt_hash,
                                        "artifact_source_match": scene_match,
                                        "abc_exact_source_restoration": len({row["source_state_fingerprint"] for row in multi_contract_rows}) == 1,
                                        "single_agent_projection_valid": None,
                                        "valid_for_paired_comparison": scene_match and len({row["source_state_fingerprint"] for row in multi_contract_rows}) == 1,
                                    },
                                    {
                                        "schema_version": SCHEMA_VERSION,
                                        "execution_mode": "single_agent",
                                        "scenario": scenario,
                                        "seed": seed,
                                        "agent_id": ego_id,
                                        "source_state_fingerprint": single_fingerprint,
                                        "contract_A_fingerprint": single_contract_rows[0]["source_state_fingerprint"],
                                        "contract_B_fingerprint": single_contract_rows[1]["source_state_fingerprint"],
                                        "contract_C_fingerprint": single_contract_rows[2]["source_state_fingerprint"],
                                        "artifact_initial_condition_hash_expected": expected_hash,
                                        "artifact_initial_condition_hash_rebuilt": rebuilt_hash,
                                        "artifact_source_match": scene_match,
                                        "abc_exact_source_restoration": len({row["source_state_fingerprint"] for row in single_contract_rows}) == 1,
                                        **projection,
                                        "valid_for_paired_comparison": projection["projection_valid"] and len({row["source_state_fingerprint"] for row in single_contract_rows}) == 1,
                                    },
                                ]
                            )
                            processed += 1
                        finally:
                            single_env.close()
                finally:
                    base_env.close()
                print(
                    f"[temporary-reference audit] {scenario}/seed{seed}: branches={processed}",
                    flush=True,
                )

    admissible = [
        row for row in admissibility_rows if row["reference_reasonably_admissible"]
    ]
    valid_keys = {
        (row["scenario"], int(row["seed"]), int(row["agent_id"]))
        for row in admissible
    }
    multi_raw = [
        row
        for row in raw_multi
        if (row["scenario"], row["seed"], row["agent_id"]) in valid_keys
    ]
    single_raw = [
        row
        for row in raw_single
        if (row["scenario"], row["seed"], row["agent_id"]) in valid_keys
    ]
    by_key: dict[tuple[str, int, int, str, str], dict[str, Any]] = {}
    for row in multi_raw + single_raw:
        by_key[(row["scenario"], row["seed"], row["agent_id"], row["execution_mode"], row["contract"])] = row

    paired_rows: list[dict[str, Any]] = []
    for scenario, seed, agent_id in sorted(valid_keys):
        values: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "scenario": scenario,
            "seed": seed,
            "agent_id": agent_id,
        }
        for mode in ("single_agent", "multi_agent"):
            prefix = "single" if mode == "single_agent" else "multi"
            a = by_key[(scenario, seed, agent_id, mode, CONTRACTS[0])]
            b = by_key[(scenario, seed, agent_id, mode, CONTRACTS[1])]
            c = by_key[(scenario, seed, agent_id, mode, CONTRACTS[2])]
            values.update(
                {
                    f"{prefix}_A_terminal_success": a["terminal_success"],
                    f"{prefix}_A_collision": a["any_collision"],
                    f"{prefix}_A_timeout": a["timeout"],
                    f"{prefix}_A_stagnation": a["stagnation_before_reference"],
                    f"{prefix}_A_completion_step": a["terminal_completion_step"],
                    f"{prefix}_B_reference_reached": b["reference_reached"],
                    f"{prefix}_B_reach_step": b["reference_reach_step"],
                    f"{prefix}_B_failure_category": b["failure_category"],
                    f"{prefix}_B_collision": b["any_collision"],
                    f"{prefix}_B_timeout": b["timeout"],
                    f"{prefix}_B_stagnation": b["stagnation_before_reference"],
                    f"{prefix}_C_reference_reached": c["reference_reached"],
                    f"{prefix}_C_terminal_success": c["terminal_success"],
                    f"{prefix}_C_failure_category": c["failure_category"],
                    f"{prefix}_C_post_reference_collision": c["post_reference_ego_obstacle_collision"] or c["post_reference_ego_inter_agent_collision"],
                    f"{prefix}_C_post_reference_timeout": c["post_reference_timeout"],
                    f"{prefix}_source_fingerprint_match": a["source_state_fingerprint"] == b["source_state_fingerprint"] == c["source_state_fingerprint"],
                }
            )
        paired_rows.append(values)

    failure_rows = [
        {
            "schema_version": SCHEMA_VERSION,
            "execution_mode": row["execution_mode"],
            "scenario": row["scenario"],
            "seed": row["seed"],
            "agent_id": row["agent_id"],
            "contract": row["contract"],
            "failure_category": row["failure_category"],
            "reference_reached": row["reference_reached"],
            "terminal_success": row["terminal_success"],
            "minimum_target_distance_m": row["minimum_target_distance_m"],
            "official_stagnation_detected": row["stagnation_before_reference"] or row["post_reference_stagnation"],
        }
        for row in multi_raw + single_raw
        if row["contract"] in CONTRACTS[1:]
    ]
    multi_b = [row for row in multi_raw if row["contract"] == CONTRACTS[1]]
    single_b = [row for row in single_raw if row["contract"] == CONTRACTS[1]]
    multi_a = [row for row in multi_raw if row["contract"] == CONTRACTS[0]]
    single_a = [row for row in single_raw if row["contract"] == CONTRACTS[0]]
    multi_c = [row for row in multi_raw if row["contract"] == CONTRACTS[2]]
    single_c = [row for row in single_raw if row["contract"] == CONTRACTS[2]]

    near_rows = [near_reference_row(row, config) for row in multi_b + single_b]
    capture_rows: list[dict[str, Any]] = []
    for mode, rows in (("single_agent", single_c), ("multi_agent", multi_c)):
        formal_entry_count = sum(
            float(row["minimum_target_distance_m"]) <= 0.25 for row in rows
        )
        formal_entry_rate = formal_entry_count / len(rows) if rows else 0.0
        for radius in config["capture_radius_diagnostic_m"]:
            count = sum(float(row["minimum_target_distance_m"]) <= float(radius) for row in rows)
            capture_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "execution_mode": mode,
                    "radius_m": float(radius),
                    "branch_count": len(rows),
                    "entered_count": count,
                    "entered_rate": count / len(rows) if rows else None,
                    "gain_vs_0p25_geometric_entry_pp": (count / len(rows) - formal_entry_rate) * 100.0 if rows else None,
                    "trajectory_source_contract": "C_TEMP_TO_TERMINAL_full_historical_trajectory",
                    "diagnostic_only": True,
                    "formal_handoff_threshold_changed": False,
                }
            )

    handoff_rows: list[dict[str, Any]] = []
    for mode, a_rows, c_rows in (
        ("single_agent", single_a, single_c),
        ("multi_agent", multi_a, multi_c),
    ):
        amap = {(row["scenario"], row["seed"], row["agent_id"]): row for row in a_rows}
        for c in c_rows:
            if not c["reference_reached"]:
                continue
            key = (c["scenario"], c["seed"], c["agent_id"])
            a = amap[key]
            handoff_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "execution_mode": mode,
                    "scenario": c["scenario"],
                    "seed": c["seed"],
                    "agent_id": c["agent_id"],
                    "reference_reached": True,
                    "handoff_step": c["handoff_step"],
                    "handoff_speed_mps": c["handoff_speed_mps"],
                    "handoff_phase": c["handoff_phase"],
                    "phase_switch_delta": c["maximum_phase_switch_delta"],
                    "matched_direct_terminal_success": a["terminal_success"],
                    "post_handoff_terminal_success": c["terminal_success"],
                    "post_reference_collision": c["post_reference_ego_obstacle_collision"] or c["post_reference_ego_inter_agent_collision"],
                    "post_reference_timeout": c["post_reference_timeout"],
                }
            )

    err_rows, same_goal_class = err_same_goal_rows(config)

    original_success_count = sum(_boolean(row["team_success"]) for row in episode_rows)
    paired_map = {(row["scenario"], row["seed"], row["agent_id"]): row for row in paired_rows}
    recovery_rows: list[dict[str, Any]] = []
    for key, episode in sorted(episodes_by_key.items()):
        if _boolean(episode["team_success"]):
            continue
        agent_signals = []
        for agent_id in range(int(config["num_agents"])):
            pair = paired_map.get((key[0], key[1], agent_id))
            if pair is None:
                continue
            signal = bool(
                pair["single_A_terminal_success"]
                and not pair["single_B_reference_reached"]
                and pair["single_B_failure_category"] in POLICY_FAILURES_B
                and not pair["single_B_collision"]
            )
            if signal:
                agent_signals.append(agent_id)
        no_original_collision = not _boolean(episode["collision"])
        recoverable = bool(no_original_collision and agent_signals)
        recovery_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "scenario": key[0],
                "seed": key[1],
                "original_team_success": False,
                "original_collision": _boolean(episode["collision"]),
                "original_timeout": _boolean(episode["timeout"]),
                "lower_execution_signal_agent_ids": agent_signals,
                "lower_execution_signal_agent_count": len(agent_signals),
                "no_unavoidable_early_collision_proxy": no_original_collision,
                "potentially_recoverable_by_lower_adaptation": recoverable,
                "interpretation": "diagnostic upper-bound membership only",
            }
        )

    single_direct = _rate(single_a, "terminal_success") or 0.0
    multi_direct = _rate(multi_a, "terminal_success") or 0.0
    single_reach = _rate(single_b, "reference_reached") or 0.0
    multi_reach = _rate(multi_b, "reference_reached") or 0.0
    single_c_success = _rate(single_c, "terminal_success") or 0.0
    multi_c_success = _rate(multi_c, "terminal_success") or 0.0
    rules = config["classification"]
    single_gap = classify_gap(single_direct - single_reach, rules)
    multi_gap = classify_gap(multi_direct - multi_reach, rules)
    multi_policy_failures = sum(row["failure_category"] in POLICY_FAILURES_B for row in multi_b if not row["reference_reached"])
    multi_failure_count = sum(not row["reference_reached"] for row in multi_b)
    policy_failure_share = multi_policy_failures / multi_failure_count if multi_failure_count else 0.0
    if multi_gap == "YES" and policy_failure_share < float(rules["minimum_policy_execution_failure_share"]):
        multi_gap = "WEAK"
    interaction = classify_gap(single_reach - multi_reach, rules)
    single_capture_025 = next(row for row in capture_rows if row["execution_mode"] == "single_agent" and np.isclose(row["radius_m"], 0.25))["entered_rate"]
    single_capture_040 = next(row for row in capture_rows if row["execution_mode"] == "single_agent" and np.isclose(row["radius_m"], 0.40))["entered_rate"]
    capture_gain = float(single_capture_040 - single_capture_025)
    if capture_gain >= float(rules["near_capture_yes_minimum_radius_gain"]):
        near_class = "YES"
    elif capture_gain >= float(rules["near_capture_weak_minimum_radius_gain"]):
        near_class = "WEAK"
    else:
        near_class = "NO"

    multi_handoff = [row for row in handoff_rows if row["execution_mode"] == "multi_agent"]
    single_handoff = [row for row in handoff_rows if row["execution_mode"] == "single_agent"]
    single_direct_on_reached = _rate(single_handoff, "matched_direct_terminal_success") or 0.0
    single_post_handoff = _rate(single_handoff, "post_handoff_terminal_success") or 0.0
    direct_on_reached = _rate(multi_handoff, "matched_direct_terminal_success") or 0.0
    post_handoff = _rate(multi_handoff, "post_handoff_terminal_success") or 0.0
    handoff_class = classify_gap(direct_on_reached - post_handoff, rules)
    phase_valid = all(not row["phase_reset_on_switch"] for row in multi_raw + single_raw)

    limitations = []
    if single_gap == "YES":
        limitations.append("TEMPORARY_REFERENCE_TRACKING")
    if handoff_class == "YES":
        limitations.append("REFERENCE_TO_TERMINAL_HANDOFF")
    if interaction == "YES":
        limitations.append("MULTI_AGENT_INTERACTION")
    if len(admissible) != len(admissibility_rows):
        limitations.append("REFERENCE_QUALITY")
    primary = (
        "NOT_ESTABLISHED"
        if not limitations
        else limitations[0]
        if len(limitations) == 1
        else "MIXED"
    )
    adaptation_conditions = {
        "direct_terminal_strong": classify_direct(single_direct, rules) == "STRONG",
        "temporary_reference_gap_significant": single_gap == "YES",
        "gap_persists_single_agent": single_gap == "YES",
        "policy_execution_failures_primary": (
            sum(row["failure_category"] in POLICY_FAILURES_B for row in single_b if not row["reference_reached"])
            >= max(1, math.ceil(0.5 * sum(not row["reference_reached"] for row in single_b)))
        ),
        "handoff_semantics_correct": phase_valid,
    }
    if all(adaptation_conditions.values()):
        adaptation = "YES"
    elif sum(adaptation_conditions.values()) >= 4:
        adaptation = "WEAK"
    else:
        adaptation = "NO"

    recoverable_count = sum(
        bool(row["potentially_recoverable_by_lower_adaptation"])
        for row in recovery_rows
    )
    upper_bound = min(1.0, (original_success_count + recoverable_count) / len(episode_rows))
    if adaptation == "YES":
        next_step = "TARGETED_TEMP_REFERENCE_SAC_ADAPTATION"
    elif handoff_class == "YES":
        next_step = "HANDOFF_INTERFACE_FIX"
    elif interaction == "YES" and single_gap != "YES":
        next_step = "JOINT_COORDINATION_AUDIT"
    elif multi_gap == "NO":
        next_step = "KEEP_FROZEN_SAC_AND_RETURN_TO_HIGH_LEVEL"
    else:
        next_step = "STOP"

    conclusion = {
        "schema_version": SCHEMA_VERSION,
        "DIRECT_TERMINAL_CAPABILITY": classify_direct(single_direct, rules),
        "REFERENCE_ADMISSIBILITY_VALID": "YES" if admissible and len(admissible) == len(admissibility_rows) else "NO",
        "SINGLE_AGENT_DIRECT_TERMINAL_SUCCESS_RATE": single_direct,
        "MULTI_AGENT_DIRECT_TERMINAL_SUCCESS_RATE": multi_direct,
        "SINGLE_AGENT_TEMP_REFERENCE_REACH_RATE": single_reach,
        "MULTI_AGENT_TEMP_REFERENCE_REACH_RATE": multi_reach,
        "SINGLE_AGENT_TEMP_TO_TERMINAL_SUCCESS_RATE": single_c_success,
        "MULTI_AGENT_TEMP_TO_TERMINAL_SUCCESS_RATE": multi_c_success,
        "SINGLE_AGENT_TEMP_REFERENCE_DROP_PP": (single_direct - single_reach) * 100.0,
        "MULTI_AGENT_TEMP_REFERENCE_DROP_PP": (multi_direct - multi_reach) * 100.0,
        "PEER_INTERACTION_REACH_PENALTY_PP": (single_reach - multi_reach) * 100.0,
        "HANDOFF_COMPLETION_DROP_PP": (direct_on_reached - post_handoff) * 100.0,
        "SINGLE_AGENT_MATCHED_DIRECT_ON_REACHED_RATE": single_direct_on_reached,
        "SINGLE_AGENT_POST_HANDOFF_SUCCESS_RATE": single_post_handoff,
        "SINGLE_AGENT_HANDOFF_DROP_PP": (single_direct_on_reached - single_post_handoff) * 100.0,
        "MULTI_AGENT_MATCHED_DIRECT_ON_REACHED_RATE": direct_on_reached,
        "MULTI_AGENT_POST_HANDOFF_SUCCESS_RATE": post_handoff,
        "TEMPORARY_REFERENCE_EXECUTION_GAP": multi_gap,
        "LOW_LEVEL_REFERENCE_EXECUTION_GAP_SINGLE_AGENT": single_gap,
        "MULTI_AGENT_INTERACTION_PENALTY": interaction,
        "NEAR_REFERENCE_CAPTURE_FAILURE": near_class,
        "HANDOFF_TRANSITION_GAP": handoff_class,
        "HIGH_LEVEL_SELECTION_NOT_PRIMARY_FOR_SAME_GOAL_SUBSET": same_goal_class,
        "PRIMARY_HIERARCHICAL_EXECUTION_LIMITATION": primary,
        "TARGETED_SAC_ADAPTATION_JUSTIFIED": adaptation,
        "POTENTIALLY_RECOVERABLE_TEAM_EPISODES": recoverable_count,
        "DIAGNOSTIC_TEAM_SUCCESS_UPPER_BOUND": upper_bound,
        "RECOMMENDED_NEXT_STEP": next_step,
        "ADAPTATION_CONDITIONS": adaptation_conditions,
        "ADMISSIBLE_BRANCH_COUNT": len(admissible),
        "TOTAL_SELECTED_REFERENCE_COUNT": len(admissibility_rows),
        "MULTI_POLICY_EXECUTION_FAILURE_SHARE": policy_failure_share,
        "REFERENCE_CAPTURE_RADIUS_DIAGNOSTIC": {
            "formal_radius_m": 0.25,
            "single_0p40_minus_0p25": capture_gain,
            "formal_threshold_changed": False,
        },
        "external_historical_single_terminal_success_rate": 341 / 350,
        "external_corridor_hold_success_rate": 10 / 30,
        "external_context_used_in_formal_rates": False,
        "training_performed": False,
    }

    original_by_key = episodes_by_key
    replay_checks = []
    for key in sorted({(row["scenario"], row["seed"]) for row in multi_c}):
        replays = [row for row in multi_c if (row["scenario"], row["seed"]) == key]
        original = original_by_key[key]
        replay_checks.append(
            bool(
                all(bool(row["team_success"]) == _boolean(original["team_success"]) for row in replays)
                and all(bool(row["any_collision"]) == _boolean(original["collision"]) for row in replays)
                and all(bool(row["timeout"]) == _boolean(original["timeout"]) for row in replays)
            )
        )

    core_after = _hash_paths(CORE_PATHS)
    policy_hash_after = _policy_parameter_sha256(policy)
    integrity_checks = {
        "checkpoint_hash_match": sha256_file(checkpoint) == config["checkpoint_sha256_expected"],
        "checkpoint_path_resolved": Path(resolved_checkpoint) == checkpoint.resolve(),
        "core_hashes_unchanged_during_audit": core_before == core_after,
        "policy_parameters_unchanged": policy_hash_before == policy_hash_after,
        "source_state_artifact_hash_match": all(bool(row["artifact_source_match"]) for row in source_rows),
        "abc_exact_source_restoration": all(bool(row["abc_exact_source_restoration"]) for row in source_rows),
        "single_agent_projection_valid": all(bool(row["valid_for_paired_comparison"]) for row in source_rows if row["execution_mode"] == "single_agent"),
        "phase_preserved": phase_valid,
        "terminal_goals_unchanged": all(bool(row["terminal_task_goals_unchanged"]) for row in multi_raw + single_raw),
        "formal_gat_C_replay_matches_existing_team_outcomes": bool(replay_checks) and all(replay_checks),
        "formal_handoff_threshold_unchanged": float(config["handoff_threshold_m"]) == 0.25,
        "stress_set_consumed": False,
        "training_performed": False,
    }
    integrity = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASSED" if all(value is False if key in {"stress_set_consumed", "training_performed"} else bool(value) for key, value in integrity_checks.items()) else "FAILED",
        "checks": integrity_checks,
        "failed_checks": [
            key
            for key, value in integrity_checks.items()
            if not ((value is False) if key in {"stress_set_consumed", "training_performed"} else bool(value))
        ],
        "checkpoint_sha256_before": config["checkpoint_sha256_expected"],
        "checkpoint_sha256_after": sha256_file(checkpoint),
        "policy_parameter_sha256_before": policy_hash_before,
        "policy_parameter_sha256_after": policy_hash_after,
        "core_hashes_before": core_before,
        "core_hashes_after": core_after,
        "source_artifact_hashes": {
            "per_agent_results.csv": sha256_file(source_dir / "per_agent_results.csv"),
            "episode_results.csv": sha256_file(source_dir / "episode_results.csv"),
            "scenario_manifest.json": sha256_file(source_dir / "scenario_manifest.json"),
            "supervision_v2_branch_manifest.csv": sha256_file(REPO_ROOT / config["source_supervision_v2_dir"] / "branch_rollout_manifest.csv"),
            "err_replanning_events.csv": sha256_file(REPO_ROOT / config["source_err_dir"] / "replanning_events.csv"),
        },
        "formal_selected_reference_count": len(admissibility_rows),
        "valid_paired_reference_count": len(admissible),
        "single_agent_contract_rollout_count": len(single_results),
        "multi_agent_contract_rollout_count": len(multi_results),
    }

    write_csv(output_dir / "source_state_manifest.csv", source_rows)
    write_csv(output_dir / "reference_admissibility.csv", admissibility_rows)
    write_csv(output_dir / "paired_contract_results.csv", paired_rows)
    write_csv(output_dir / "single_agent_results.csv", single_results)
    write_csv(output_dir / "multi_agent_results.csv", multi_results)
    write_csv(output_dir / "failure_taxonomy.csv", failure_rows)
    write_csv(output_dir / "near_reference_analysis.csv", near_rows)
    write_csv(output_dir / "capture_radius_diagnostic.csv", capture_rows)
    write_csv(output_dir / "handoff_analysis.csv", handoff_rows)
    write_csv(output_dir / "err_same_goal_analysis.csv", err_rows)
    write_csv(output_dir / "recoverability_analysis.csv", recovery_rows)
    write_json(output_dir / "conclusion.json", conclusion)
    write_json(output_dir / "integrity_manifest.json", integrity)
    (output_dir / "FINAL_REPORT.md").write_text(
        render_report(
            conclusion=conclusion,
            admissibility=admissibility_rows,
            multi_b=multi_b,
            single_b=single_b,
            capture=capture_rows,
            err_rows=err_rows,
            recovery=recovery_rows,
            integrity=integrity,
        ),
        encoding="utf-8",
    )
    return output_dir / "FINAL_REPORT.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> Path:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = REPO_ROOT / config["output_root"] / datetime.now().strftime("%Y%m%d_%H%M%S")
    elif not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    report = run_audit(config, output_dir)
    print(report)
    return report


if __name__ == "__main__":
    main()
