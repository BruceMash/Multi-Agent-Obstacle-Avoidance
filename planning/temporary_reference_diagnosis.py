"""Pure diagnostics for temporary-reference execution interfaces.

The helpers in this module never mutate the environment, policy, DMP, or
candidate generator.  They expose the distribution shift and controller
discontinuities already present when ``dmp.goal`` is changed while
``env.goals`` remains the terminal task goal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from Controller.dmp_rl import compute_dmp_transition
from Environment.frozen_sac_dmp_execution import (
    build_actor_observation,
    predict_policy_action,
)


PROTOCOL_TERMINAL = "terminal_goal_baseline"
PROTOCOL_ONE_SHOT = "one_shot_temporary_reference"
PROTOCOL_EXISTING_BOUNDARY_FREE = "existing_waypoint_handoff_boundary_free"
PROTOCOL_EXISTING_LEGACY = "existing_full_waypoint_interface_legacy"
PROTOCOL_FIXED_PERIOD = "fixed_period_hard_switching"

PROTOCOL_DISPLAY_NAMES = {
    PROTOCOL_TERMINAL: "Terminal-Goal Baseline",
    PROTOCOL_ONE_SHOT: "One-Shot Temporary Reference",
    PROTOCOL_EXISTING_BOUNDARY_FREE: "Existing Waypoint Handoff, Boundary-Free",
    PROTOCOL_EXISTING_LEGACY: "Existing Full Waypoint Interface (Legacy)",
    PROTOCOL_FIXED_PERIOD: "Fixed-Period Hard Switching",
}

OBSERVATION_SLICES = {
    "velocity": slice(0, 3),
    "goal_direction": slice(3, 6),
    "goal_distance": slice(6, 7),
    "current_scan": slice(7, 63),
    "previous_scan": slice(63, 119),
    "phase": slice(119, 120),
    "K_alpha": slice(120, 121),
    "K_beta": slice(121, 122),
}


def _vector3(value: Any, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite 3-vector")
    return vector.copy()


def angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1.0e-12:
        return 0.0
    cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def actor_observation_for_goal(env: Any, agent_index: int, active_goal: np.ndarray) -> np.ndarray:
    """Build the active policy input from one unchanged environment state."""

    agent_index = int(agent_index)
    packet = env.latest_sensor_packets[agent_index]
    if packet is None:
        raise RuntimeError("environment must be reset before observation diagnosis")
    dmp = env.dmps[agent_index]
    return build_actor_observation(
        velocity=env.dynamics[agent_index].v,
        active_goal=_vector3(active_goal, "active_goal"),
        position=env.dynamics[agent_index].p,
        current_scan=packet.current_scan,
        previous_scan=packet.previous_scan,
        goal_distance_clip=env.sensors[agent_index].goal_distance_clip,
        phase=dmp.phase,
        k_alpha=dmp.config.K_alpha,
        k_beta=dmp.config.K_beta,
    )


def observation_switch_diagnostic(
    env: Any,
    agent_index: int,
    *,
    previous_active_goal: np.ndarray,
    new_active_goal: np.ndarray,
) -> dict[str, Any]:
    """Compare two active-goal inputs on exactly the same physical/sensor state."""

    previous = actor_observation_for_goal(env, agent_index, previous_active_goal)
    current = actor_observation_for_goal(env, agent_index, new_active_goal)
    delta = current.astype(float) - previous.astype(float)
    direction_delta = delta[OBSERVATION_SLICES["goal_direction"]]
    distance_delta = delta[OBSERVATION_SLICES["goal_distance"]]
    non_goal = delta.copy()
    non_goal[OBSERVATION_SLICES["goal_direction"]] = 0.0
    non_goal[OBSERVATION_SLICES["goal_distance"]] = 0.0
    return {
        "previous_goal_direction": previous[3:6].astype(float).tolist(),
        "new_goal_direction": current[3:6].astype(float).tolist(),
        "goal_direction_delta": direction_delta.tolist(),
        "goal_direction_delta_l2": float(np.linalg.norm(direction_delta)),
        "previous_normalized_goal_distance": float(previous[6]),
        "new_normalized_goal_distance": float(current[6]),
        "normalized_goal_distance_delta": float(distance_delta[0]),
        "phase": float(current[119]),
        "K_alpha": float(current[120]),
        "K_beta": float(current[121]),
        "observation_l2_change": float(np.linalg.norm(delta)),
        "non_goal_feature_l2_change": float(np.linalg.norm(non_goal)),
        "observation_dimension": int(current.size),
        "same_state_snapshot": True,
    }


def _controller_acceleration(
    env: Any,
    agent_index: int,
    *,
    active_goal: np.ndarray,
    terminal_goal: np.ndarray,
    action: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    dmp = env.dmps[int(agent_index)]
    acceleration, _, info = compute_dmp_transition(
        config=dmp.config,
        position=env.dynamics[int(agent_index)].p,
        velocity=env.dynamics[int(agent_index)].v,
        rl_action=np.asarray(action, dtype=float),
        active_goal=_vector3(active_goal, "active_goal"),
        terminal_goal=_vector3(terminal_goal, "terminal_goal"),
        phase=float(dmp.phase),
    )
    return np.asarray(acceleration, dtype=float), info


def dmp_switch_diagnostic(
    env: Any,
    agent_index: int,
    *,
    previous_active_goal: np.ndarray,
    new_active_goal: np.ndarray,
    terminal_goal: np.ndarray,
    policy: Any | None = None,
) -> dict[str, Any]:
    """Measure attractor-only and policy-conditioned command discontinuities."""

    previous_active_goal = _vector3(previous_active_goal, "previous_active_goal")
    new_active_goal = _vector3(new_active_goal, "new_active_goal")
    terminal_goal = _vector3(terminal_goal, "terminal_goal")
    zero_action = np.zeros(6, dtype=np.float32)
    nominal_before, nominal_info_before = _controller_acceleration(
        env,
        agent_index,
        active_goal=previous_active_goal,
        terminal_goal=terminal_goal,
        action=zero_action,
    )
    nominal_after, nominal_info_after = _controller_acceleration(
        env,
        agent_index,
        active_goal=new_active_goal,
        terminal_goal=terminal_goal,
        action=zero_action,
    )
    result: dict[str, Any] = {
        "previous_active_goal": previous_active_goal.tolist(),
        "new_active_goal": new_active_goal.tolist(),
        "terminal_goal": terminal_goal.tolist(),
        "goal_jump_m": float(np.linalg.norm(new_active_goal - previous_active_goal)),
        "position": np.asarray(env.dynamics[int(agent_index)].p, dtype=float).tolist(),
        "velocity": np.asarray(env.dynamics[int(agent_index)].v, dtype=float).tolist(),
        "phase": float(env.dmps[int(agent_index)].phase),
        "zero_action_nominal_acceleration_before": nominal_before.tolist(),
        "zero_action_nominal_acceleration_after": nominal_after.tolist(),
        "zero_action_nominal_acceleration_jump_mps2": float(
            np.linalg.norm(nominal_after - nominal_before)
        ),
        "zero_action_nominal_direction_change_deg": angle_degrees(
            nominal_before, nominal_after
        ),
        "forcing_gate_before": float(nominal_info_before["forcing_gate_scalar"]),
        "forcing_gate_after": float(nominal_info_after["forcing_gate_scalar"]),
        "state_mutated": False,
    }
    if policy is not None:
        previous_observation = actor_observation_for_goal(
            env, agent_index, previous_active_goal
        )
        new_observation = actor_observation_for_goal(env, agent_index, new_active_goal)
        previous_action = predict_policy_action(policy, previous_observation)
        new_action = predict_policy_action(policy, new_observation)
        closed_before, closed_info_before = _controller_acceleration(
            env,
            agent_index,
            active_goal=previous_active_goal,
            terminal_goal=terminal_goal,
            action=previous_action,
        )
        closed_after, closed_info_after = _controller_acceleration(
            env,
            agent_index,
            active_goal=new_active_goal,
            terminal_goal=terminal_goal,
            action=new_action,
        )
        result.update(
            {
                "policy_action_before": previous_action.astype(float).tolist(),
                "policy_action_after": new_action.astype(float).tolist(),
                "policy_action_jump_l2": float(
                    np.linalg.norm(new_action.astype(float) - previous_action.astype(float))
                ),
                "closed_loop_commanded_acceleration_before": closed_before.tolist(),
                "closed_loop_commanded_acceleration_after": closed_after.tolist(),
                "closed_loop_commanded_acceleration_jump_mps2": float(
                    np.linalg.norm(closed_after - closed_before)
                ),
                "closed_loop_direction_change_deg": angle_degrees(
                    closed_before, closed_after
                ),
                "residual_drive_norm_before": float(
                    np.linalg.norm(closed_info_before["residual_drive"])
                ),
                "residual_drive_norm_after": float(
                    np.linalg.norm(closed_info_after["residual_drive"])
                ),
            }
        )
    return result


def candidate_safety_diagnostic(
    env: Any,
    agent_index: int,
    *,
    candidate: np.ndarray,
    boundary_margin: float,
    collision_clearance: float,
    segment_samples: int,
) -> dict[str, Any]:
    """Measure point/segment safety; boundary validity is diagnostic only."""

    candidate = _vector3(candidate, "candidate")
    start = np.asarray(env.dynamics[int(agent_index)].p, dtype=float)
    obstacles = list(env._sensor_static_obstacles())
    obstacles.extend(env._sensor_dynamic_obstacles(int(agent_index)))
    point_clearance = min(
        (float(obstacle.signed_distance(candidate)) for obstacle in obstacles),
        default=float("inf"),
    )
    sampled_clearances: list[float] = []
    for ratio in np.linspace(0.0, 1.0, int(segment_samples) + 1, dtype=float)[1:]:
        point = (1.0 - ratio) * start + ratio * candidate
        sampled_clearances.append(
            min(
                (float(obstacle.signed_distance(point)) for obstacle in obstacles),
                default=float("inf"),
            )
        )
    segment_clearance = min(sampled_clearances, default=float("inf"))
    lower, upper = np.asarray(env.env_config.workspace_bounds, dtype=float)
    boundary_validity = bool(
        np.all(candidate >= lower + float(boundary_margin))
        and np.all(candidate <= upper - float(boundary_margin))
    )
    return {
        "point_clearance_m": float(point_clearance),
        "point_safety": bool(point_clearance > float(collision_clearance)),
        "segment_clearance_m": float(segment_clearance),
        "segment_validity": bool(segment_clearance > float(collision_clearance)),
        "boundary_validity": boundary_validity,
        "boundary_validity_used_for_selection": False,
        "segment_samples": int(segment_samples),
    }


@dataclass
class OneShotReferenceState:
    """Strict one-shot lifecycle with one optional terminal return."""

    temporary_reference: np.ndarray | None = None
    candidate_created_count: int = 0
    activated_step: int | None = None
    reached_step: int | None = None
    released_step: int | None = None
    returned_to_terminal: bool = False

    def activate(self, reference: np.ndarray, timestep: int) -> None:
        if self.candidate_created_count != 0:
            raise RuntimeError("one-shot protocol may create only one candidate")
        self.temporary_reference = _vector3(reference, "reference")
        self.candidate_created_count = 1
        self.activated_step = int(timestep)

    def active_goal(self, terminal_goal: np.ndarray) -> np.ndarray:
        if self.temporary_reference is None or self.returned_to_terminal:
            return _vector3(terminal_goal, "terminal_goal")
        return self.temporary_reference.copy()

    def update_reached(
        self,
        *,
        position: np.ndarray,
        terminal_goal: np.ndarray,
        completed_step: int,
        reached_tolerance: float,
    ) -> tuple[np.ndarray, bool]:
        if self.temporary_reference is None or self.returned_to_terminal:
            return _vector3(terminal_goal, "terminal_goal"), False
        distance = float(
            np.linalg.norm(_vector3(position, "position") - self.temporary_reference)
        )
        if distance > float(reached_tolerance):
            return self.temporary_reference.copy(), False
        self.reached_step = int(completed_step)
        self.released_step = int(completed_step)
        self.returned_to_terminal = True
        return _vector3(terminal_goal, "terminal_goal"), True


def failure_attribution(
    *,
    episode: Mapping[str, Any],
    step_rows: Sequence[Mapping[str, Any]],
    reference_events: Sequence[Mapping[str, Any]],
    shortly_after_switch_steps: int,
) -> str:
    """Assign one deterministic, pre-ordered diagnostic failure category."""

    if bool(episode.get("success", episode.get("team_success", False))):
        return "success"
    collision_rows = [row for row in step_rows if bool(row.get("collision", False))]
    first_collision_step = min(
        (int(row["completed_step"]) for row in collision_rows), default=None
    )
    reached_steps = [
        int(row["reference_reached_step"])
        for row in reference_events
        if row.get("reference_reached_step") is not None
    ]
    first_reached_step = min(reached_steps, default=None)
    switch_steps = sorted(
        int(row["reference_activated_step"])
        for row in reference_events
        if row.get("reference_activated_step") is not None
    )
    if first_collision_step is not None:
        if first_reached_step is None or first_collision_step < first_reached_step:
            return "collision_before_first_temporary_reference_reached"
        if any(
            bool(row.get("fixed_period_hold_after_reached", False))
            and bool(row.get("collision", False))
            for row in step_rows
        ):
            return "collision_during_fixed_period_hold_after_reference_reached"
        if any(
            0 <= first_collision_step - switch <= int(shortly_after_switch_steps)
            for switch in switch_steps
        ):
            return "collision_shortly_after_reference_switch"
        if any(bool(row.get("terminal_return_active", False)) for row in collision_rows):
            return "collision_during_terminal_goal_return"
        if any(bool(row.get("inter_agent_collision", False)) for row in collision_rows):
            return "inter_agent_collision_unrelated_to_recent_switch"
        return "collision_unknown"
    if bool(episode.get("truncated", False)) or bool(episode.get("timeout", False)):
        return "timeout_or_stagnation"
    return "unknown"


def lifecycle_summary(events: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    rows = list(events)
    created = len(rows)
    reached = sum(bool(row.get("reference_reached", False)) for row in rows)
    return {
        "reference_count": int(created),
        "reference_reached_count": int(reached),
        "reference_reached_rate": float(reached / max(1, created)),
        "mean_goal_jump_m": float(
            np.mean([float(row.get("goal_jump_m", 0.0)) for row in rows])
        )
        if rows
        else 0.0,
        "mean_dmp_acceleration_jump_mps2": float(
            np.mean(
                [
                    float(row.get("closed_loop_commanded_acceleration_jump_mps2", 0.0))
                    for row in rows
                ]
            )
        )
        if rows
        else 0.0,
    }
