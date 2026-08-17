"""Deployment-aligned long-horizon supervision primitives.

This module is evaluation/data-generation only.  It does not alter Proposal,
FP-SHEP, the graph schema, the GAT, SAC-DMP, or the environment.  Every real
branch advances an isolated deep copy through ``MultiAgentDMPEnv.step``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
import time
from typing import Any, Mapping, Sequence

import numpy as np

from Environment.frozen_sac_dmp_execution import predict_frozen_actions
from planning.candidate_execution_benchmark import (
    _active_goal_observations,
    environment_state_fingerprint,
)
from planning.pre_gat_closed_loop import FPSHEPOnlineScoreSpec, score_fp_shep_candidates
from scripts.evaluate_frozen_policy_waypoint_guidance import (
    set_dmp_active_goal_preserve_phase,
)


SCHEMA_VERSION = "gat_supervision_v2"
HISTORICAL_GATE_NAME = "historical_vector_goal_eff_gate"
BACKGROUND_FP_SHEP = "FP_SHEP_TOP1"
BACKGROUND_NULL = "NULL_TERMINAL"


def _readonly(value: Any, *, dtype: Any = float) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _portable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, Mapping):
        return {str(key): _portable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable(item) for item in value]
    return value


def stable_hash(value: Any) -> str:
    payload = json.dumps(
        _portable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class FrozenBackgroundPlan:
    selector: str
    references: np.ndarray
    available: np.ndarray
    selected_candidate_ids: tuple[int | None, ...]
    scores: tuple[tuple[float, ...], ...]
    plan_hash: str

    def __post_init__(self) -> None:
        references = _readonly(self.references)
        available = _readonly(self.available, dtype=bool)
        if references.ndim != 2 or references.shape[1] != 3:
            raise ValueError("background references must have shape (N,3)")
        if available.shape != (references.shape[0],):
            raise ValueError("background availability must have shape (N,)")
        if len(self.selected_candidate_ids) != references.shape[0]:
            raise ValueError("background candidate ids must have one entry per agent")
        if len(self.scores) != references.shape[0]:
            raise ValueError("background scores must have one entry per agent")
        object.__setattr__(self, "references", references)
        object.__setattr__(self, "available", available)


@dataclass(frozen=True)
class BranchRollout:
    outcome: dict[str, Any]
    actions: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    phases: np.ndarray
    dmp_goals: np.ndarray
    trajectory_hash: str
    runtime_ms: float

    def __post_init__(self) -> None:
        for name in ("actions", "positions", "velocities", "phases", "dmp_goals"):
            object.__setattr__(self, name, _readonly(getattr(self, name)))


@dataclass(frozen=True)
class TieredTarget:
    tiers: np.ndarray
    tie_break_keys: tuple[tuple[float, ...], ...]
    utilities: np.ndarray
    probabilities: np.ndarray
    ranks_zero_based: np.ndarray
    hard_target: int
    top1_tied: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "tiers", _readonly(self.tiers, dtype=int))
        object.__setattr__(self, "utilities", _readonly(self.utilities))
        object.__setattr__(self, "probabilities", _readonly(self.probabilities))
        object.__setattr__(self, "ranks_zero_based", _readonly(self.ranks_zero_based, dtype=int))


def build_background_plan(
    *,
    env: Any,
    proposals_by_agent: Sequence[Sequence[Any]],
    policy: Any,
    score_spec: FPSHEPOnlineScoreSpec,
    selector: str = BACKGROUND_FP_SHEP,
) -> FrozenBackgroundPlan:
    """Choose each non-ego plan once; caller supplies the historical gate scope."""

    terminal_goals = np.asarray(env.goals, dtype=float)
    references = terminal_goals.copy()
    available = np.zeros(int(env.num_agents), dtype=bool)
    selected: list[int | None] = []
    all_scores: list[tuple[float, ...]] = []
    for agent_id in range(int(env.num_agents)):
        proposals = tuple(proposals_by_agent[agent_id])
        if selector == BACKGROUND_NULL or not proposals:
            selected.append(None)
            all_scores.append(())
            continue
        if selector != BACKGROUND_FP_SHEP:
            raise ValueError(f"unknown background selector: {selector}")
        records = score_fp_shep_candidates(
            env=env,
            agent_index=agent_id,
            proposals=proposals,
            policy=policy,
            spec=score_spec,
        )
        scores = tuple(float(item.score) for item in records)
        candidate_id = int(np.argmax(np.asarray(scores, dtype=float)))
        references[agent_id] = np.asarray(proposals[candidate_id].point, dtype=float)
        available[agent_id] = True
        selected.append(candidate_id)
        all_scores.append(scores)
    payload = {
        "selector": selector,
        "references": references,
        "available": available,
        "selected_candidate_ids": selected,
        "scores": all_scores,
    }
    return FrozenBackgroundPlan(
        selector=selector,
        references=references,
        available=available,
        selected_candidate_ids=tuple(selected),
        scores=tuple(all_scores),
        plan_hash=stable_hash(payload),
    )


def _obstacle_surface_clearance(env: Any, agent_id: int) -> float:
    position = np.asarray(env.dynamics[int(agent_id)].p, dtype=float)
    obstacles = list(getattr(env, "static_obstacles", ())) + list(
        getattr(env, "dynamic_obstacles", ())
    )
    values = [float(item.signed_distance(position)) for item in obstacles]
    return float(min(values)) if values else float("inf")


def _agent_peer_distance(env: Any, agent_id: int) -> float:
    if int(env.num_agents) <= 1:
        return float("inf")
    pairwise = np.asarray(env._compute_pairwise_distances(), dtype=float)
    return float(np.min(np.delete(pairwise[int(agent_id)], int(agent_id))))


def _direction_alignment(velocity: np.ndarray, direction: np.ndarray) -> float | None:
    velocity = np.asarray(velocity, dtype=float)
    direction = np.asarray(direction, dtype=float)
    first = float(np.linalg.norm(velocity))
    second = float(np.linalg.norm(direction))
    if first <= 1.0e-12 or second <= 1.0e-12:
        return None
    return float(np.clip(np.dot(velocity, direction) / (first * second), -1.0, 1.0))


def _trajectory_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.asarray(value, dtype=np.float64)
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def run_long_horizon_branch(
    *,
    initial_env: Any,
    ego_agent_id: int,
    ego_candidate_goal: np.ndarray | None,
    background_plan: FrozenBackgroundPlan,
    policy: Any,
    max_absolute_episode_steps: int,
    reference_reached_tolerance_m: float,
    class_index: int,
    candidate_id: int | None,
) -> BranchRollout:
    """Run one immutable intervention to the environment's absolute horizon.

    The caller must enter ``scoped_historical_preview_and_multi_agent_transition``.
    No transition implementation is copied here; propagation, sensing,
    collision, success, and timeout all flow through ``env.step``.
    """

    ego_agent_id = int(ego_agent_id)
    source_fingerprint = environment_state_fingerprint(initial_env)
    env = copy.deepcopy(initial_env)
    if environment_state_fingerprint(env) != source_fingerprint:
        raise RuntimeError("deep-copied long-horizon branch changed source state")
    if int(env.env_config.max_steps) != int(max_absolute_episode_steps):
        raise ValueError("source environment must already use absolute max_steps=220")
    source_step = int(env.steps)
    if source_step > int(max_absolute_episode_steps):
        raise ValueError("source state is beyond the absolute episode horizon")

    terminal_goals = np.asarray(env.goals, dtype=float).copy()
    terminal_goals_before = terminal_goals.copy()
    branch_start_positions = np.asarray(env._positions(), dtype=float).copy()
    references = np.asarray(background_plan.references, dtype=float).copy()
    available = np.asarray(background_plan.available, dtype=bool).copy()
    reference_applicable = ego_candidate_goal is not None
    if reference_applicable:
        references[ego_agent_id] = np.asarray(ego_candidate_goal, dtype=float)
        available[ego_agent_id] = True
    else:
        references[ego_agent_id] = terminal_goals[ego_agent_id]
        available[ego_agent_id] = False

    returned_to_terminal = np.logical_not(available)
    reference_reached = np.zeros(int(env.num_agents), dtype=bool)
    reference_reached_steps: list[int | None] = [None] * int(env.num_agents)
    terminal_reached_steps: list[int | None] = [None] * int(env.num_agents)
    phase_switch_deltas: list[float] = []
    for agent_id in range(int(env.num_agents)):
        if available[agent_id]:
            phase_before = float(env.dmps[agent_id].phase)
            set_dmp_active_goal_preserve_phase(env.dmps[agent_id], references[agent_id])
            phase_switch_deltas.append(float(env.dmps[agent_id].phase) - phase_before)

    initial_terminal_distances = np.linalg.norm(
        terminal_goals - branch_start_positions, axis=1
    )
    initial_success = initial_terminal_distances <= float(env.env_config.goal_tolerance)
    for agent_id in np.flatnonzero(initial_success):
        terminal_reached_steps[int(agent_id)] = source_step

    actions: list[np.ndarray] = []
    positions = [np.asarray(env._positions(), dtype=float).copy()]
    velocities = [np.asarray(env._velocities(), dtype=float).copy()]
    phases = [np.asarray([dmp.phase for dmp in env.dmps], dtype=float)]
    dmp_goals = [np.asarray([dmp.goal for dmp in env.dmps], dtype=float)]
    ego_obstacle_clearances = [_obstacle_surface_clearance(env, ego_agent_id)]
    ego_peer_distances = [_agent_peer_distance(env, ego_agent_id)]
    sensor_clearances = [
        float(env.latest_sensor_packets[ego_agent_id].min_clearance)
    ]
    ego_handoff_terminal_distance: float | None = None
    ego_handoff_speed: float | None = None
    ego_handoff_alignment: float | None = None
    ego_handoff_min_obstacle: float | None = None
    ego_handoff_min_peer: float | None = None
    collision = obstacle_collision = inter_agent_collision = False
    ego_collision = peer_collision = False
    collision_step: int | None = None
    terminated = truncated = False
    last_info: dict[str, Any] = {}
    started = time.perf_counter_ns()
    try:
        while not (terminated or truncated) and int(env.steps) < int(
            max_absolute_episode_steps
        ):
            active_goals = np.asarray([dmp.goal for dmp in env.dmps], dtype=float)
            observations = _active_goal_observations(env, active_goals)
            action = predict_frozen_actions(
                policy, observations, expected_shape=tuple(env.action_shape)
            )
            actions.append(np.asarray(action, dtype=float).copy())
            _, _, terminated, truncated, last_info = env.step(action)
            positions.append(np.asarray(env._positions(), dtype=float).copy())
            velocities.append(np.asarray(env._velocities(), dtype=float).copy())
            phases.append(np.asarray([dmp.phase for dmp in env.dmps], dtype=float))
            dmp_goals.append(np.asarray([dmp.goal for dmp in env.dmps], dtype=float))
            ego_obstacle_clearances.append(_obstacle_surface_clearance(env, ego_agent_id))
            ego_peer_distances.append(_agent_peer_distance(env, ego_agent_id))
            sensor_clearances.append(
                float(env.latest_sensor_packets[ego_agent_id].min_clearance)
            )

            obstacle_mask = np.asarray(
                last_info.get("obstacle_collision_mask", np.zeros(env.num_agents)),
                dtype=bool,
            )
            inter_mask = np.asarray(
                last_info.get("inter_agent_collision_mask", np.zeros(env.num_agents)),
                dtype=bool,
            )
            step_collision = bool(last_info.get("collision", False))
            collision |= step_collision
            obstacle_collision |= bool(np.any(obstacle_mask))
            inter_agent_collision |= bool(np.any(inter_mask))
            ego_collision |= bool(obstacle_mask[ego_agent_id] or inter_mask[ego_agent_id])
            peer_indices = [index for index in range(int(env.num_agents)) if index != ego_agent_id]
            peer_collision |= bool(
                np.any(obstacle_mask[peer_indices]) or np.any(inter_mask[peer_indices])
            )
            if step_collision and collision_step is None:
                collision_step = int(env.steps)

            success_mask = np.asarray(
                last_info.get("success_mask", np.zeros(env.num_agents)), dtype=bool
            )
            for agent_id in np.flatnonzero(success_mask):
                if terminal_reached_steps[int(agent_id)] is None:
                    terminal_reached_steps[int(agent_id)] = int(env.steps)

            # This ordering matches the existing one-shot deployment helper:
            # a terminated/truncated step cannot create a post-hoc handoff.
            if not (terminated or truncated):
                for agent_id in range(int(env.num_agents)):
                    if returned_to_terminal[agent_id]:
                        continue
                    distance = float(
                        np.linalg.norm(references[agent_id] - env.dynamics[agent_id].p)
                    )
                    if distance > float(reference_reached_tolerance_m):
                        continue
                    reference_reached[agent_id] = True
                    reference_reached_steps[agent_id] = int(env.steps)
                    if agent_id == ego_agent_id:
                        ego_handoff_terminal_distance = float(
                            np.linalg.norm(
                                terminal_goals[agent_id] - env.dynamics[agent_id].p
                            )
                        )
                        ego_handoff_speed = float(
                            np.linalg.norm(env.dynamics[agent_id].v)
                        )
                        ego_handoff_alignment = _direction_alignment(
                            env.dynamics[agent_id].v,
                            terminal_goals[agent_id] - env.dynamics[agent_id].p,
                        )
                        ego_handoff_min_obstacle = float(
                            np.min(ego_obstacle_clearances)
                        )
                        ego_handoff_min_peer = float(np.min(ego_peer_distances))
                    phase_before = float(env.dmps[agent_id].phase)
                    set_dmp_active_goal_preserve_phase(
                        env.dmps[agent_id], terminal_goals[agent_id]
                    )
                    phase_switch_deltas.append(
                        float(env.dmps[agent_id].phase) - phase_before
                    )
                    returned_to_terminal[agent_id] = True
    finally:
        runtime_ms = (time.perf_counter_ns() - started) / 1.0e6

    position_array = np.stack(positions)
    velocity_array = np.stack(velocities)
    action_array = (
        np.stack(actions)
        if actions
        else np.empty((0, int(env.num_agents), int(env.action_shape[-1])), dtype=float)
    )
    phase_array = np.stack(phases)
    goal_array = np.stack(dmp_goals)
    final_positions = position_array[-1]
    final_terminal_distances = np.linalg.norm(terminal_goals - final_positions, axis=1)
    terminal_progress = initial_terminal_distances - final_terminal_distances
    path_steps = np.linalg.norm(np.diff(position_array, axis=0), axis=2)
    path_lengths = np.sum(path_steps, axis=0) if len(path_steps) else np.zeros(env.num_agents)
    final_success_mask = final_terminal_distances <= float(env.env_config.goal_tolerance)
    ego_reference_step = reference_reached_steps[ego_agent_id]
    ego_terminal_step = terminal_reached_steps[ego_agent_id]
    team_success = bool(last_info.get("success", False)) or bool(
        np.all(final_success_mask) and not collision
    )
    timeout = bool(truncated) or bool(
        not terminated and int(env.steps) >= int(max_absolute_episode_steps)
    )
    post_reference_obstacle_collision = bool(
        ego_reference_step is not None
        and collision_step is not None
        and int(collision_step) >= int(ego_reference_step)
        and obstacle_collision
    )
    post_reference_inter_agent_collision = bool(
        ego_reference_step is not None
        and collision_step is not None
        and int(collision_step) >= int(ego_reference_step)
        and inter_agent_collision
    )
    post_reference_timeout = bool(
        ego_reference_step is not None and timeout and not final_success_mask[ego_agent_id]
    )
    reference_to_terminal_progress = (
        None
        if ego_handoff_terminal_distance is None
        else float(ego_handoff_terminal_distance - final_terminal_distances[ego_agent_id])
    )
    termination_type = (
        "team_success"
        if team_success
        else "collision"
        if collision
        else "timeout"
        if timeout
        else "other_termination"
        if terminated
        else "no_remaining_budget"
    )
    maximum_phase_switch_delta = (
        float(np.max(np.abs(phase_switch_deltas))) if phase_switch_deltas else 0.0
    )
    terminal_goals_unchanged = bool(np.array_equal(env.goals, terminal_goals_before))
    trace_hash = _trajectory_hash(
        action_array, position_array, velocity_array, phase_array, goal_array
    )
    outcome = {
        "schema_version": SCHEMA_VERSION,
        "class_index": int(class_index),
        "candidate_id": candidate_id,
        "class_kind": "null" if not reference_applicable else "proposal",
        "ego_agent_id": ego_agent_id,
        "source_env_steps": source_step,
        "max_absolute_episode_steps": int(max_absolute_episode_steps),
        "remaining_budget_at_source": max(0, int(max_absolute_episode_steps) - source_step),
        "effective_transition_count": int(env.steps) - source_step,
        "final_absolute_env_steps": int(env.steps),
        "ego_reference_applicable": bool(reference_applicable),
        "ego_reference_reached": bool(reference_reached[ego_agent_id]),
        "reference_reach_step": ego_reference_step,
        "ego_terminal_reached": bool(final_success_mask[ego_agent_id]),
        "ego_terminal_distance_final": float(final_terminal_distances[ego_agent_id]),
        "ego_terminal_progress": float(terminal_progress[ego_agent_id]),
        "reference_to_terminal_progress": reference_to_terminal_progress,
        "team_success": team_success,
        "all_agents_terminal_reached": bool(np.all(final_success_mask)),
        "obstacle_collision": bool(obstacle_collision),
        "inter_agent_collision": bool(inter_agent_collision),
        "any_collision": bool(collision),
        "timeout": timeout,
        "ego_collision": bool(ego_collision),
        "peer_collision": bool(peer_collision),
        "collision_step": collision_step,
        "min_obstacle_clearance": float(np.min(ego_obstacle_clearances)),
        "min_obstacle_clearance_source": (
            "ground_truth_static_dynamic_obstacle_surface_evaluation_only"
        ),
        "min_sensor_clearance": float(np.min(sensor_clearances)),
        "min_sensor_clearance_source": "real_environment_composite_lidar_nearest_hit",
        "min_inter_agent_distance": float(np.min(ego_peer_distances)),
        "min_inter_agent_distance_source": "ground_truth_center_distance_evaluation_only",
        "path_length": float(path_lengths[ego_agent_id]),
        "path_length_per_agent": path_lengths.astype(float).tolist(),
        "completion_step": ego_terminal_step,
        "team_completion_step": int(env.steps) if team_success else None,
        "reference_handoff_terminal_distance": ego_handoff_terminal_distance,
        "reference_handoff_speed": ego_handoff_speed,
        "reference_handoff_direction_alignment": ego_handoff_alignment,
        "reference_handoff_min_obstacle_clearance": ego_handoff_min_obstacle,
        "reference_handoff_min_inter_agent_distance": ego_handoff_min_peer,
        "post_reference_obstacle_collision": post_reference_obstacle_collision,
        "post_reference_inter_agent_collision": post_reference_inter_agent_collision,
        "post_reference_timeout": post_reference_timeout,
        "termination_type": termination_type,
        "background_selector": background_plan.selector,
        "background_plan_hash": background_plan.plan_hash,
        "terminal_task_goals_unchanged": terminal_goals_unchanged,
        "maximum_phase_switch_delta": maximum_phase_switch_delta,
        "phase_reset_on_switch": bool(maximum_phase_switch_delta != 0.0),
        "trajectory_hash": trace_hash,
        "source_state_fingerprint": source_fingerprint,
        "forcing_gate": HISTORICAL_GATE_NAME,
        "one_shot": True,
        "high_level_replanning": False,
    }
    close = getattr(env, "close", None)
    if callable(close):
        close()
    if environment_state_fingerprint(initial_env) != source_fingerprint:
        raise RuntimeError("long-horizon branch mutated source environment")
    if not terminal_goals_unchanged:
        raise RuntimeError("long-horizon branch changed terminal task goals")
    if maximum_phase_switch_delta != 0.0:
        raise RuntimeError("long-horizon handoff changed DMP phase")
    return BranchRollout(
        outcome=outcome,
        actions=action_array,
        positions=position_array,
        velocities=velocity_array,
        phases=phase_array,
        dmp_goals=goal_array,
        trajectory_hash=trace_hash,
        runtime_ms=float(runtime_ms),
    )


def outcome_tier(outcome: Mapping[str, Any], *, progress_epsilon: float) -> int:
    if bool(outcome["any_collision"]):
        return 0
    if bool(outcome["team_success"]):
        return 5
    if bool(outcome["ego_terminal_reached"]):
        return 4
    if bool(outcome["ego_reference_applicable"]) and bool(
        outcome["ego_reference_reached"]
    ):
        return 3
    if float(outcome["ego_terminal_progress"]) > float(progress_epsilon):
        return 2
    return 1


def _metric(outcome: Mapping[str, Any], name: str, *, default: float) -> float:
    value = outcome.get(name)
    if value is None:
        return float(default)
    value = float(value)
    return value if not math.isnan(value) else float(default)


def outcome_tie_break_key(
    outcome: Mapping[str, Any], *, tier: int, round_decimals: int
) -> tuple[float, ...]:
    rd = int(round_decimals)
    if tier in {4, 5}:
        raw = (
            -_metric(outcome, "completion_step", default=float("inf")),
            -_metric(outcome, "path_length", default=float("inf")),
            _metric(outcome, "min_obstacle_clearance", default=-float("inf")),
            _metric(outcome, "min_inter_agent_distance", default=-float("inf")),
        )
    elif tier == 3:
        raw = (
            _metric(outcome, "reference_to_terminal_progress", default=-float("inf")),
            -_metric(outcome, "ego_terminal_distance_final", default=float("inf")),
            _metric(outcome, "min_obstacle_clearance", default=-float("inf")),
            _metric(outcome, "min_inter_agent_distance", default=-float("inf")),
            _metric(outcome, "reference_handoff_direction_alignment", default=-1.0),
            -_metric(outcome, "reference_handoff_speed", default=float("inf")),
        )
    elif tier in {1, 2}:
        raw = (
            _metric(outcome, "ego_terminal_progress", default=-float("inf")),
            -_metric(outcome, "ego_terminal_distance_final", default=float("inf")),
            _metric(outcome, "min_obstacle_clearance", default=-float("inf")),
            _metric(outcome, "min_inter_agent_distance", default=-float("inf")),
            -_metric(outcome, "path_length", default=float("inf")),
        )
    else:
        raw = (
            _metric(outcome, "collision_step", default=-float("inf")),
            _metric(outcome, "min_obstacle_clearance", default=-float("inf")),
            _metric(outcome, "min_inter_agent_distance", default=-float("inf")),
            _metric(outcome, "ego_terminal_progress", default=-float("inf")),
        )
    return tuple(round(value, rd) if math.isfinite(value) else value for value in raw)


def build_tiered_target(
    outcomes: Sequence[Mapping[str, Any]],
    *,
    progress_epsilon: float,
    within_tier_scale: float,
    round_decimals: int,
    temperature: float,
) -> TieredTarget:
    if not outcomes:
        raise ValueError("target requires at least the real null class")
    if not 0.0 <= float(within_tier_scale) < 1.0:
        raise ValueError("within-tier scale must preserve a strict tier gap")
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    tiers = np.asarray(
        [outcome_tier(item, progress_epsilon=progress_epsilon) for item in outcomes],
        dtype=int,
    )
    keys = tuple(
        outcome_tie_break_key(item, tier=int(tier), round_decimals=round_decimals)
        for item, tier in zip(outcomes, tiers, strict=True)
    )
    within = np.zeros(len(outcomes), dtype=float)
    for tier in sorted(set(tiers.tolist())):
        indices = np.flatnonzero(tiers == tier).astype(int).tolist()
        unique_keys = sorted({keys[index] for index in indices})
        key_rank = {key: rank for rank, key in enumerate(unique_keys)}
        denominator = max(1, len(unique_keys) - 1)
        for index in indices:
            within[index] = float(key_rank[keys[index]] / denominator)
    utilities = tiers.astype(float) + float(within_tier_scale) * within
    order = np.argsort(-utilities, kind="mergesort")
    ranks = np.empty(len(order), dtype=int)
    ranks[order] = np.arange(len(order), dtype=int)
    shifted = (utilities - float(np.max(utilities))) / float(temperature)
    exp_values = np.exp(shifted)
    probabilities = exp_values / float(np.sum(exp_values))
    maximum = float(np.max(utilities))
    top_indices = np.flatnonzero(np.isclose(utilities, maximum, rtol=0.0, atol=0.0))
    return TieredTarget(
        tiers=tiers,
        tie_break_keys=keys,
        utilities=utilities,
        probabilities=probabilities,
        ranks_zero_based=ranks,
        hard_target=int(order[0]),
        top1_tied=len(top_indices) > 1,
    )


def information_density(
    targets: Sequence[TieredTarget], thresholds: Mapping[str, Any]
) -> dict[str, Any]:
    if not targets:
        return {"classification": "FAILED", "reason": "no_targets"}
    valid = all(
        np.all(np.isfinite(item.probabilities))
        and np.isclose(float(np.sum(item.probabilities)), 1.0, rtol=0.0, atol=1.0e-10)
        for item in targets
    )
    unique_top1_rate = float(np.mean([not item.top1_tied for item in targets]))
    same_tier_rate = float(np.mean([len(set(item.tiers.tolist())) == 1 for item in targets]))
    mean_distinct_tiers = float(np.mean([len(set(item.tiers.tolist())) for item in targets]))
    margins = []
    entropies = []
    effective_counts = []
    null_top1 = []
    top1_classes = []
    single_class_count = 0
    for item in targets:
        ordered = np.sort(item.utilities)[::-1]
        if len(ordered) > 1:
            margins.append(float(ordered[0] - ordered[1]))
        else:
            single_class_count += 1
        p = item.probabilities
        entropy = float(-np.sum(p * np.log(np.clip(p, 1.0e-15, 1.0))))
        entropies.append(entropy)
        effective_counts.append(float(np.exp(entropy)))
        null_top1.append(item.hard_target == 0)
        top1_classes.append(item.hard_target)
    dominant_class_rate = max(
        top1_classes.count(value) / len(top1_classes) for value in set(top1_classes)
    )
    failed = thresholds["failed"]
    if (
        not valid
        or unique_top1_rate < float(failed["maximum_unique_top1_rate_exclusive"])
        or same_tier_rate > float(failed["minimum_all_same_tier_rate_exclusive"])
        or dominant_class_rate >= 0.999999
    ):
        classification = "FAILED"
    else:
        strong = thresholds["strong"]
        adequate = thresholds["adequate"]
        if (
            unique_top1_rate >= float(strong["minimum_unique_top1_rate"])
            and same_tier_rate <= float(strong["maximum_all_same_tier_rate"])
            and mean_distinct_tiers >= float(strong["minimum_mean_distinct_tiers"])
        ):
            classification = "STRONG"
        elif (
            unique_top1_rate >= float(adequate["minimum_unique_top1_rate"])
            and same_tier_rate <= float(adequate["maximum_all_same_tier_rate"])
            and mean_distinct_tiers >= float(adequate["minimum_mean_distinct_tiers"])
        ):
            classification = "ADEQUATE"
        else:
            classification = "WEAK"
    return {
        "classification": classification,
        "graph_count": len(targets),
        "probabilities_valid": valid,
        "unique_top1_rate": unique_top1_rate,
        "tied_top1_rate": 1.0 - unique_top1_rate,
        "all_candidates_same_tier_rate": same_tier_rate,
        "mean_distinct_tiers_per_graph": mean_distinct_tiers,
        "single_class_graph_count": single_class_count,
        "single_class_graph_rate": float(single_class_count / len(targets)),
        "top1_top2_margin_graph_count": len(margins),
        "mean_top1_top2_utility_margin": (
            float(np.mean(margins)) if margins else None
        ),
        "median_top1_top2_utility_margin": (
            float(np.median(margins)) if margins else None
        ),
        "mean_soft_target_entropy": float(np.mean(entropies)),
        "mean_effective_class_count": float(np.mean(effective_counts)),
        "null_top1_rate": float(np.mean(null_top1)),
        "dominant_class_top1_rate": float(dominant_class_rate),
    }


def classify_attribution_stability(
    metrics: Mapping[str, float], thresholds: Mapping[str, Any]
) -> str:
    weak = thresholds["weak_if_any_below"]
    strong = thresholds["strong_if_all_at_least"]
    if any(float(metrics[name]) < float(limit) for name, limit in weak.items()):
        return "WEAK"
    if all(float(metrics[name]) >= float(limit) for name, limit in strong.items()):
        return "STRONG"
    return "ADEQUATE"


def pairwise_ordering_agreement(first: Sequence[float], second: Sequence[float]) -> float:
    first_array = np.asarray(first, dtype=float)
    second_array = np.asarray(second, dtype=float)
    if first_array.shape != second_array.shape:
        raise ValueError("ranking vectors must have equal shapes")
    correct = total = 0
    for left in range(len(first_array)):
        for right in range(left + 1, len(first_array)):
            first_sign = int(np.sign(first_array[left] - first_array[right]))
            second_sign = int(np.sign(second_array[left] - second_array[right]))
            if first_sign == 0 and second_sign == 0:
                correct += 1
            elif first_sign != 0 and first_sign == second_sign:
                correct += 1
            total += 1
    return float(correct / total) if total else 1.0
