"""Pure helpers for Actor--DMP goal-semantics decoupling diagnostics.

These helpers only construct checkpoint-compatible observations, run the
already-frozen Actor, and evaluate the existing DMP transition equation on an
unchanged state.  They never mutate the environment, DMP, policy, or candidate
generator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from Controller.dmp_rl import compute_dmp_transition
from Environment.frozen_sac_dmp_execution import (
    actor_observation_dim,
    build_actor_observation,
    predict_policy_action,
)
from planning.temporary_reference_diagnosis import OBSERVATION_SLICES, angle_degrees


VARIANT_A = "A_terminal_actor_terminal_dmp"
VARIANT_B = "B_temporary_actor_terminal_dmp"
VARIANT_C = "C_terminal_actor_temporary_dmp"
VARIANT_D = "D_temporary_actor_temporary_dmp"

VARIANT_ORDER = (VARIANT_A, VARIANT_B, VARIANT_C, VARIANT_D)
VARIANT_DISPLAY_NAMES = {
    VARIANT_A: "A: Terminal Actor / Terminal DMP",
    VARIANT_B: "B: Temporary Actor / Terminal DMP",
    VARIANT_C: "C: Terminal Actor / Temporary DMP",
    VARIANT_D: "D: Temporary Actor / Temporary DMP",
}


@dataclass(frozen=True)
class GoalSemantics:
    actor_goal: str
    dmp_goal: str

    def __post_init__(self) -> None:
        allowed = {"terminal", "temporary"}
        if self.actor_goal not in allowed or self.dmp_goal not in allowed:
            raise ValueError("goal semantics must be 'terminal' or 'temporary'")


VARIANT_SEMANTICS: Mapping[str, GoalSemantics] = {
    VARIANT_A: GoalSemantics(actor_goal="terminal", dmp_goal="terminal"),
    VARIANT_B: GoalSemantics(actor_goal="temporary", dmp_goal="terminal"),
    VARIANT_C: GoalSemantics(actor_goal="terminal", dmp_goal="temporary"),
    VARIANT_D: GoalSemantics(actor_goal="temporary", dmp_goal="temporary"),
}


def _vector3(value: Any, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite 3-vector")
    return vector.copy()


def checkpoint_observation_for_goal(
    env: Any,
    agent_index: int,
    goal: np.ndarray,
) -> np.ndarray:
    """Build the active SAC-DMP checkpoint input for an explicit goal.

    This intentionally does not use ``env.get_observation()``: the current
    multi-agent environment exposes a 138-D native observation, whereas the
    frozen historical checkpoint requires exactly 122 dimensions.
    """

    agent_index = int(agent_index)
    packet = env.latest_sensor_packets[agent_index]
    if packet is None:
        raise RuntimeError("environment must be reset before building observations")
    dmp = env.dmps[agent_index]
    observation = build_actor_observation(
        velocity=env.dynamics[agent_index].v,
        active_goal=_vector3(goal, "goal"),
        position=env.dynamics[agent_index].p,
        current_scan=packet.current_scan,
        previous_scan=packet.previous_scan,
        goal_distance_clip=env.sensors[agent_index].goal_distance_clip,
        phase=dmp.phase,
        k_alpha=dmp.config.K_alpha,
        k_beta=dmp.config.K_beta,
    )
    expected_dim = actor_observation_dim(np.asarray(packet.current_scan).size)
    if observation.shape != (expected_dim,):
        raise RuntimeError("checkpoint observation does not match the active ray contract")
    return observation


def terminal_checkpoint_observations(env: Any) -> np.ndarray:
    """Return terminal-conditioned 122-D observations for every agent."""

    return np.stack(
        [
            checkpoint_observation_for_goal(env, agent_index, env.goals[agent_index])
            for agent_index in range(int(env.num_agents))
        ]
    ).astype(np.float32)


def temporary_checkpoint_observations(
    env: Any,
    temporary_references: np.ndarray,
) -> np.ndarray:
    """Return temporary-reference-conditioned 122-D observations."""

    references = np.asarray(temporary_references, dtype=float)
    if references.shape != (int(env.num_agents), 3):
        raise ValueError("temporary_references must have shape (num_agents, 3)")
    return np.stack(
        [
            checkpoint_observation_for_goal(env, agent_index, references[agent_index])
            for agent_index in range(int(env.num_agents))
        ]
    ).astype(np.float32)


def task_aware_checkpoint_observations(
    env: Any,
    temporary_references: np.ndarray,
    *,
    switch_ages_s: np.ndarray,
    previous_applied_accelerations: np.ndarray,
    task_distance_scale_m: float = 100.0,
    switch_age_scale_s: float = 5.0,
    acceleration_scale_mps2: float = 4.0,
) -> np.ndarray:
    """Append the nine preregistered task/switch/control-history features.

    The original sensor block and three DMP fields are left byte-for-byte in
    their historical order.  The added context is:
    final-goal direction (3), normalized final-goal distance (1), the existing
    proposal safety quantity in that direction (1), normalized switch age (1),
    and previous actually applied acceleration (3).
    """

    from Guidance.reference_point_proposal_demo import (  # local import avoids legacy import cycles
        ProposalConfig,
        compute_sector_safety_field,
    )

    references = np.asarray(temporary_references, dtype=float)
    switch_ages = np.asarray(switch_ages_s, dtype=float)
    previous_accelerations = np.asarray(previous_applied_accelerations, dtype=float)
    expected_agents = int(env.num_agents)
    if references.shape != (expected_agents, 3):
        raise ValueError("temporary_references must have shape (num_agents, 3)")
    if switch_ages.shape != (expected_agents,):
        raise ValueError("switch_ages_s must have shape (num_agents,)")
    if previous_accelerations.shape != (expected_agents, 3):
        raise ValueError("previous_applied_accelerations must have shape (num_agents, 3)")
    if min(task_distance_scale_m, switch_age_scale_s, acceleration_scale_mps2) <= 0.0:
        raise ValueError("task/switch/acceleration normalization scales must be positive")

    base = temporary_checkpoint_observations(env, references)
    contexts: list[np.ndarray] = []
    proposal_config = ProposalConfig()
    goal_tolerance = float(env.env_config.goal_tolerance)
    for agent_index in range(expected_agents):
        position = np.asarray(env.dynamics[agent_index].p, dtype=float)
        velocity = np.asarray(env.dynamics[agent_index].v, dtype=float)
        terminal_goal = np.asarray(env.goals[agent_index], dtype=float)
        delta = terminal_goal - position
        distance = float(np.linalg.norm(delta))
        task_direction = np.zeros(3, dtype=float) if distance < 1.0e-8 else delta / distance
        safety = compute_sector_safety_field(
            position,
            terminal_goal,
            velocity,
            env.latest_sensor_packets[agent_index],
            env.sensors[agent_index],
            proposal_config,
            goal_tolerance,
        )
        directions = np.asarray(env.sensors[agent_index].ray_directions, dtype=float).reshape(-1, 3)
        direction_index = int(np.argmax(directions @ task_direction))
        task_safety = float(safety.normalized_margin.reshape(-1)[direction_index])
        contexts.append(
            np.concatenate(
                [
                    task_direction.astype(np.float32),
                    np.asarray(
                        [
                            np.clip(distance / task_distance_scale_m, 0.0, 1.0),
                            task_safety,
                            np.clip(switch_ages[agent_index] / switch_age_scale_s, 0.0, 1.0),
                        ],
                        dtype=np.float32,
                    ),
                    np.clip(
                        previous_accelerations[agent_index] / acceleration_scale_mps2,
                        -1.0,
                        1.0,
                    ).astype(np.float32),
                ]
            )
        )
    expanded = np.concatenate([base, np.stack(contexts).astype(np.float32)], axis=1)
    if expanded.shape != (expected_agents, base.shape[1] + 9):
        raise RuntimeError("task-aware checkpoint observation has an unexpected shape")
    return expanded.astype(np.float32)


def action_saturation_mask(
    action: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    *,
    relative_tolerance: float,
) -> np.ndarray:
    """Return dimensions within a configurable fraction of either bound."""

    action = np.asarray(action, dtype=float)
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    if action.shape != low.shape or action.shape != high.shape:
        raise ValueError("action and bounds must have identical shapes")
    tolerance = float(relative_tolerance)
    if not 0.0 <= tolerance < 0.5:
        raise ValueError("relative_tolerance must be in [0, 0.5)")
    span = high - low
    if np.any(span <= 0.0):
        raise ValueError("action bounds must have positive span")
    margin = tolerance * span
    return np.logical_or(action - low <= margin, high - action <= margin)


def actor_goal_shift_diagnostic(
    env: Any,
    agent_index: int,
    *,
    temporary_reference: np.ndarray,
    policy: Any,
    saturation_relative_tolerance: float,
) -> dict[str, Any]:
    """Measure Actor-only goal-conditioning shift on one unchanged state."""

    agent_index = int(agent_index)
    terminal_observation = checkpoint_observation_for_goal(
        env, agent_index, env.goals[agent_index]
    )
    temporary_observation = checkpoint_observation_for_goal(
        env, agent_index, temporary_reference
    )
    terminal_action = predict_policy_action(policy, terminal_observation)
    temporary_action = predict_policy_action(policy, temporary_observation)
    observation_delta = temporary_observation.astype(float) - terminal_observation.astype(float)
    action_delta = temporary_action.astype(float) - terminal_action.astype(float)
    low = np.asarray(env.action_space.low[agent_index], dtype=float)
    high = np.asarray(env.action_space.high[agent_index], dtype=float)
    terminal_saturation = action_saturation_mask(
        terminal_action,
        low,
        high,
        relative_tolerance=saturation_relative_tolerance,
    )
    temporary_saturation = action_saturation_mask(
        temporary_action,
        low,
        high,
        relative_tolerance=saturation_relative_tolerance,
    )
    non_goal_delta = observation_delta.copy()
    non_goal_delta[OBSERVATION_SLICES["goal_direction"]] = 0.0
    non_goal_delta[OBSERVATION_SLICES["goal_distance"]] = 0.0
    return {
        "terminal_observation": terminal_observation.astype(float).tolist(),
        "temporary_observation": temporary_observation.astype(float).tolist(),
        "observation_l2_delta": float(np.linalg.norm(observation_delta)),
        "goal_direction_l2_delta": float(
            np.linalg.norm(observation_delta[OBSERVATION_SLICES["goal_direction"]])
        ),
        "goal_distance_delta": float(
            observation_delta[OBSERVATION_SLICES["goal_distance"]][0]
        ),
        "non_goal_feature_l2_delta": float(np.linalg.norm(non_goal_delta)),
        "terminal_action": terminal_action.astype(float).tolist(),
        "temporary_action": temporary_action.astype(float).tolist(),
        "action_l2_delta": float(np.linalg.norm(action_delta)),
        "forcing_action_l2_delta": float(np.linalg.norm(action_delta[:3])),
        "goal_offset_action_l2_delta": float(np.linalg.norm(action_delta[3:])),
        "terminal_action_saturation_mask": terminal_saturation.tolist(),
        "temporary_action_saturation_mask": temporary_saturation.tolist(),
        "terminal_action_saturation_rate": float(np.mean(terminal_saturation)),
        "temporary_action_saturation_rate": float(np.mean(temporary_saturation)),
        "same_state_snapshot": True,
        "state_mutated": False,
    }


def _dmp_acceleration(
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


def dmp_attractor_shift_diagnostic(
    env: Any,
    agent_index: int,
    *,
    temporary_reference: np.ndarray,
    shared_terminal_action: np.ndarray,
) -> dict[str, Any]:
    """Measure attractor-only change using zero and identical policy actions."""

    agent_index = int(agent_index)
    terminal_goal = _vector3(env.goals[agent_index], "terminal_goal")
    temporary_reference = _vector3(temporary_reference, "temporary_reference")
    position_before = np.asarray(env.dynamics[agent_index].p, dtype=float).copy()
    velocity_before = np.asarray(env.dynamics[agent_index].v, dtype=float).copy()
    dmp_goal_before = np.asarray(env.dmps[agent_index].goal, dtype=float).copy()
    phase_before = float(env.dmps[agent_index].phase)
    zero_action = np.zeros(6, dtype=np.float32)
    nominal_terminal, nominal_terminal_info = _dmp_acceleration(
        env,
        agent_index,
        active_goal=terminal_goal,
        terminal_goal=terminal_goal,
        action=zero_action,
    )
    nominal_temporary, nominal_temporary_info = _dmp_acceleration(
        env,
        agent_index,
        active_goal=temporary_reference,
        terminal_goal=terminal_goal,
        action=zero_action,
    )
    commanded_terminal, commanded_terminal_info = _dmp_acceleration(
        env,
        agent_index,
        active_goal=terminal_goal,
        terminal_goal=terminal_goal,
        action=shared_terminal_action,
    )
    commanded_temporary, commanded_temporary_info = _dmp_acceleration(
        env,
        agent_index,
        active_goal=temporary_reference,
        terminal_goal=terminal_goal,
        action=shared_terminal_action,
    )
    state_unchanged = (
        np.array_equal(position_before, env.dynamics[agent_index].p)
        and np.array_equal(velocity_before, env.dynamics[agent_index].v)
        and np.array_equal(dmp_goal_before, env.dmps[agent_index].goal)
        and phase_before == float(env.dmps[agent_index].phase)
    )
    if not state_unchanged:
        raise RuntimeError("DMP attractor diagnostic mutated execution state")
    return {
        "terminal_goal": terminal_goal.tolist(),
        "temporary_reference": temporary_reference.tolist(),
        "shared_terminal_action": np.asarray(shared_terminal_action, dtype=float).tolist(),
        "zero_action_terminal_acceleration": nominal_terminal.tolist(),
        "zero_action_temporary_acceleration": nominal_temporary.tolist(),
        "zero_action_nominal_acceleration_delta": float(
            np.linalg.norm(nominal_temporary - nominal_terminal)
        ),
        "zero_action_direction_change_deg": angle_degrees(
            nominal_terminal, nominal_temporary
        ),
        "same_action_terminal_acceleration": commanded_terminal.tolist(),
        "same_action_temporary_acceleration": commanded_temporary.tolist(),
        "same_action_commanded_acceleration_delta": float(
            np.linalg.norm(commanded_temporary - commanded_terminal)
        ),
        "same_action_direction_change_deg": angle_degrees(
            commanded_terminal, commanded_temporary
        ),
        "forcing_gate_terminal_nominal": float(
            nominal_terminal_info["forcing_gate_scalar"]
        ),
        "forcing_gate_temporary_nominal": float(
            nominal_temporary_info["forcing_gate_scalar"]
        ),
        "forcing_gate_terminal_commanded": float(
            commanded_terminal_info["forcing_gate_scalar"]
        ),
        "forcing_gate_temporary_commanded": float(
            commanded_temporary_info["forcing_gate_scalar"]
        ),
        "same_state_snapshot": True,
        "state_mutated": False,
    }


def factorial_interaction(
    value_a: float,
    value_b: float,
    value_c: float,
    value_d: float,
) -> float:
    """Return the 2x2 interaction D - C - B + A."""

    return float(value_d - value_c - value_b + value_a)
