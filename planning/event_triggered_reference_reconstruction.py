"""Event-triggered active-reference reconstruction supervisor.

This module contains only the lightweight execution-state and trigger logic.
Proposal generation, FP-SHEP, GAT inference, SAC inference, DMP dynamics,
reward, and environment semantics remain external and frozen.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from Guidance.reference_point_proposal_demo import (
    ProposalConfig,
    compute_sector_safety_field,
)


EVENT_REFERENCE_HANDOFF = "REFERENCE_COMPLETION_HANDOFF"
EVENT_REFERENCE_COMPLETION_REPROPOSAL = "REFERENCE_COMPLETION_REPROPOSAL"
EVENT_EMERGENCY_REPROPOSAL = "EMERGENCY_REPROPOSAL"
EVENT_NORMAL_REPROPOSAL = "NORMAL_REPROPOSAL"
EVENT_NO_UPDATE = "NO_UPDATE"
ACTIVE_GOAL_TERMINAL = "terminal"
ACTIVE_GOAL_REFERENCE = "reference"
EMERGENCY_SEMANTICS_LEVEL = "level_triggered"
EMERGENCY_SEMANTICS_EDGE_REARM = "edge_triggered_existing_threshold_rearm"
REFERENCE_COMPLETION_TERMINAL_HANDOFF = "terminal_handoff"
REFERENCE_COMPLETION_RECONSTRUCT_LONG_RANGE = "reconstruct_unless_terminal_local"
FAR_TERMINAL_NULL_ALLOW = "allow_far_terminal"
FAR_TERMINAL_NULL_MASK_TO_GAT_CANDIDATE = "mask_to_best_gat_non_null"
INTERACTION_FEASIBILITY_ALLOW_RISKY = "allow_risky"
INTERACTION_FEASIBILITY_MASK_WHEN_SAFE = "mask_risky_when_safe_alternative_exists"
ACTIVE_SAFETY_MARGIN_SOURCE = (
    "Guidance.reference_point_proposal_demo.compute_sector_safety_field."
    "safety_margin[nearest_active_direction_sector]"
)


@dataclass(frozen=True)
class CandidateSelectionLifecycleDecision:
    """Decode the existing GAT classes under the long-range goal lifecycle.

    Class zero remains the historical terminal/null class.  The long-range
    adaptation only changes whether that class is eligible while the terminal
    goal is still outside the already configured local terminal scope.  It
    does not add a candidate, score, prediction head, or planner.
    """

    raw_class_index: int
    effective_class_index: int
    raw_selected_candidate_id: int | None
    effective_selected_candidate_id: int | None
    terminal_goal_distance_m: float
    terminal_null_eligible: bool
    far_terminal_null_mask_applied: bool
    far_terminal_null_mask_unavailable_no_candidate: bool
    policy: str


@dataclass(frozen=True)
class InteractionFeasibilityDecision:
    selected_candidate_id_before_mask: int | None
    effective_selected_candidate_id: int | None
    interaction_mask_applied: bool
    selected_candidate_was_risky: bool | None
    safe_candidate_count: int
    risky_candidate_count: int
    policy: str


def apply_far_terminal_null_policy(
    *,
    raw_class_index: int,
    class_logits: Sequence[float] | np.ndarray,
    proposal_count: int,
    terminal_goal_distance_m: float,
    terminal_local_scope_m: float | None,
    policy: str,
) -> CandidateSelectionLifecycleDecision:
    """Apply a terminal-class eligibility mask without changing GAT logits."""

    logits = np.asarray(class_logits, dtype=float)
    proposal_count = int(proposal_count)
    raw_class_index = int(raw_class_index)
    terminal_goal_distance_m = float(terminal_goal_distance_m)
    valid_policies = {
        FAR_TERMINAL_NULL_ALLOW,
        FAR_TERMINAL_NULL_MASK_TO_GAT_CANDIDATE,
    }
    if policy not in valid_policies:
        raise ValueError(f"far-terminal null policy must be one of {sorted(valid_policies)}")
    if proposal_count < 0 or logits.shape != (proposal_count + 1,):
        raise ValueError("class logits must contain one null class plus all proposals")
    if not np.all(np.isfinite(logits)):
        raise ValueError("class logits must be finite")
    if not 0 <= raw_class_index <= proposal_count:
        raise ValueError("raw class index is outside the null/proposal class range")
    if not np.isfinite(terminal_goal_distance_m) or terminal_goal_distance_m < 0.0:
        raise ValueError("terminal goal distance must be finite and non-negative")
    if policy == FAR_TERMINAL_NULL_MASK_TO_GAT_CANDIDATE:
        if terminal_local_scope_m is None or float(terminal_local_scope_m) <= 0.0:
            raise ValueError("masked far-terminal null policy requires terminal_local_scope_m")

    raw_selected = None if raw_class_index == 0 else raw_class_index - 1
    terminal_null_eligible = (
        terminal_local_scope_m is None
        or terminal_goal_distance_m <= float(terminal_local_scope_m)
    )
    mask_requested = (
        policy == FAR_TERMINAL_NULL_MASK_TO_GAT_CANDIDATE
        and raw_class_index == 0
        and not terminal_null_eligible
    )
    mask_applied = bool(mask_requested and proposal_count > 0)
    mask_unavailable = bool(mask_requested and proposal_count == 0)
    if mask_applied:
        effective_class_index = 1 + int(np.argmax(logits[1:]))
    else:
        effective_class_index = raw_class_index
    effective_selected = (
        None if effective_class_index == 0 else effective_class_index - 1
    )
    return CandidateSelectionLifecycleDecision(
        raw_class_index=raw_class_index,
        effective_class_index=effective_class_index,
        raw_selected_candidate_id=raw_selected,
        effective_selected_candidate_id=effective_selected,
        terminal_goal_distance_m=terminal_goal_distance_m,
        terminal_null_eligible=bool(terminal_null_eligible),
        far_terminal_null_mask_applied=mask_applied,
        far_terminal_null_mask_unavailable_no_candidate=mask_unavailable,
        policy=policy,
    )


def apply_interaction_feasibility_policy(
    *,
    selected_candidate_id: int | None,
    class_logits: Sequence[float] | np.ndarray,
    candidate_risky: Sequence[bool],
    policy: str,
) -> InteractionFeasibilityDecision:
    """Mask an already identified risky proposal only when a safe one exists.

    The input risk flags must come from the existing graph interaction
    descriptor and its existing ``d_safe`` condition.  This function does not
    create a new risk model or numerical threshold.
    """

    risky = tuple(bool(value) for value in candidate_risky)
    logits = np.asarray(class_logits, dtype=float)
    valid_policies = {
        INTERACTION_FEASIBILITY_ALLOW_RISKY,
        INTERACTION_FEASIBILITY_MASK_WHEN_SAFE,
    }
    if policy not in valid_policies:
        raise ValueError(
            f"interaction feasibility policy must be one of {sorted(valid_policies)}"
        )
    if logits.shape != (len(risky) + 1,) or not np.all(np.isfinite(logits)):
        raise ValueError("class logits must contain finite null plus proposal values")
    if selected_candidate_id is not None and not 0 <= int(selected_candidate_id) < len(risky):
        raise ValueError("selected candidate id is outside the proposal range")

    safe_ids = [index for index, value in enumerate(risky) if not value]
    selected_was_risky = (
        None
        if selected_candidate_id is None
        else bool(risky[int(selected_candidate_id)])
    )
    should_mask = bool(
        policy == INTERACTION_FEASIBILITY_MASK_WHEN_SAFE
        and selected_was_risky
        and safe_ids
    )
    effective = selected_candidate_id
    if should_mask:
        effective = max(safe_ids, key=lambda index: float(logits[index + 1]))
    return InteractionFeasibilityDecision(
        selected_candidate_id_before_mask=selected_candidate_id,
        effective_selected_candidate_id=effective,
        interaction_mask_applied=should_mask,
        selected_candidate_was_risky=selected_was_risky,
        safe_candidate_count=len(safe_ids),
        risky_candidate_count=len(risky) - len(safe_ids),
        policy=policy,
    )


def _vector3(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite 3-vector")
    return array.copy()


@dataclass(frozen=True)
class ERRConfig:
    dt: float
    progress_window_steps: int
    reference_age_threshold_s: float
    minimum_reconstruction_interval_s: float
    minimum_progress_rate_mps: float
    progress_scale_mps: float
    safety_reproposal_margin_m: float
    safety_scale_m: float
    emergency_safety_margin_m: float
    handoff_distance_m: float
    goal_equality_tolerance_m: float = 1.0e-9
    reference_completion_mode: str = REFERENCE_COMPLETION_TERMINAL_HANDOFF
    terminal_local_scope_m: float | None = None
    far_terminal_null_policy: str = FAR_TERMINAL_NULL_ALLOW

    def __post_init__(self) -> None:
        positive = {
            "dt": self.dt,
            "progress_window_steps": self.progress_window_steps,
            "reference_age_threshold_s": self.reference_age_threshold_s,
            "minimum_reconstruction_interval_s": self.minimum_reconstruction_interval_s,
            "progress_scale_mps": self.progress_scale_mps,
            "safety_scale_m": self.safety_scale_m,
            "handoff_distance_m": self.handoff_distance_m,
            "goal_equality_tolerance_m": self.goal_equality_tolerance_m,
        }
        if any(float(value) <= 0.0 for value in positive.values()):
            raise ValueError(f"ERR positive settings invalid: {positive}")
        if int(self.progress_window_steps) != self.progress_window_steps:
            raise ValueError("progress_window_steps must be an integer")
        if self.emergency_safety_margin_m >= self.safety_reproposal_margin_m:
            raise ValueError("emergency margin must be below normal safety margin")
        completion_modes = {
            REFERENCE_COMPLETION_TERMINAL_HANDOFF,
            REFERENCE_COMPLETION_RECONSTRUCT_LONG_RANGE,
        }
        if self.reference_completion_mode not in completion_modes:
            raise ValueError(
                f"reference_completion_mode must be one of {sorted(completion_modes)}"
            )
        if self.reference_completion_mode == REFERENCE_COMPLETION_RECONSTRUCT_LONG_RANGE:
            if self.terminal_local_scope_m is None or float(self.terminal_local_scope_m) <= 0.0:
                raise ValueError("long-range completion reconstruction requires terminal_local_scope_m")
        valid_null_policies = {
            FAR_TERMINAL_NULL_ALLOW,
            FAR_TERMINAL_NULL_MASK_TO_GAT_CANDIDATE,
        }
        if self.far_terminal_null_policy not in valid_null_policies:
            raise ValueError(
                f"far_terminal_null_policy must be one of {sorted(valid_null_policies)}"
            )
        if (
            self.far_terminal_null_policy
            == FAR_TERMINAL_NULL_MASK_TO_GAT_CANDIDATE
            and self.terminal_local_scope_m is None
        ):
            raise ValueError("masked far-terminal null policy requires terminal_local_scope_m")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ERRConfig":
        return cls(
            dt=float(values["dt"]),
            progress_window_steps=int(values["W_p"]),
            reference_age_threshold_s=float(values["T_rep_s"]),
            minimum_reconstruction_interval_s=float(values["T_dwell_s"]),
            minimum_progress_rate_mps=float(values["p_min_mps"]),
            progress_scale_mps=float(values["p_scale_mps"]),
            safety_reproposal_margin_m=float(values["h_rep_m"]),
            safety_scale_m=float(values["h_scale_m"]),
            emergency_safety_margin_m=float(values["h_emg_m"]),
            handoff_distance_m=float(values["d_hand_m"]),
            goal_equality_tolerance_m=float(
                values.get("goal_equality_tolerance_m", 1.0e-9)
            ),
            reference_completion_mode=str(
                values.get(
                    "reference_completion_mode",
                    REFERENCE_COMPLETION_TERMINAL_HANDOFF,
                )
            ),
            terminal_local_scope_m=(
                None
                if values.get("terminal_local_scope_m") is None
                else float(values["terminal_local_scope_m"])
            ),
            far_terminal_null_policy=str(
                values.get("far_terminal_null_policy", FAR_TERMINAL_NULL_ALLOW)
            ),
        )


@dataclass
class AgentReferenceState:
    active_goal: np.ndarray
    terminal_goal: np.ndarray
    active_goal_type: str
    last_reference_update_step: int
    progress_distance_history: deque[tuple[int, float]]
    goal_version: int = 0
    number_of_trigger_checks: int = 0
    number_of_reproposal_events: int = 0
    number_of_emergency_events: int = 0
    number_of_reference_handoffs: int = 0
    number_of_reference_completion_reproposals: int = 0
    number_of_goal_changes: int = 0
    emergency_armed: bool = False
    previous_active_safety_margin_m: float | None = None

    def __post_init__(self) -> None:
        self.active_goal = _vector3(self.active_goal, "active_goal")
        terminal = _vector3(self.terminal_goal, "terminal_goal")
        terminal.setflags(write=False)
        self.terminal_goal = terminal
        if self.active_goal_type not in {
            ACTIVE_GOAL_TERMINAL,
            ACTIVE_GOAL_REFERENCE,
        }:
            raise ValueError("active_goal_type must be terminal or reference")


@dataclass(frozen=True)
class TriggerDecision:
    event: str
    current_step: int
    active_goal_distance_m: float
    active_age_s: float
    progress_rate_mps: float | None
    progress_valid: bool
    active_safety_margin_m: float
    phi_tau: float
    phi_p: float | None
    phi_h: float
    S_rep: float
    dwell_satisfied: bool
    normal_trigger: bool
    emergency_trigger: bool
    handoff_trigger: bool
    reference_completion_trigger: bool
    terminal_goal_distance_m: float
    reference_completion_action: str | None
    trigger_reasons: tuple[str, ...]
    emergency_event_semantics: str
    emergency_armed_before: bool | None
    emergency_armed_after: bool | None
    emergency_rearmed: bool
    previous_active_safety_margin_m: float | None


@dataclass(frozen=True)
class SafetyMarginObservation:
    value_m: float
    normalized_value: float
    source: str
    unit: str
    sector_index_flat: int
    azimuth_index: int
    elevation_index: int
    raw_obstacle_distance_m: float
    guarded_obstacle_distance_m: float
    effective_safe_radius_m: float
    braking_distance_m: float
    normalization: str


class EventTriggeredReferenceSupervisor:
    """Maintain independent ERR state for each agent."""

    def __init__(
        self,
        config: ERRConfig,
        terminal_goals: Sequence[Sequence[float]],
        active_goals: Sequence[Sequence[float]] | None = None,
        active_goal_types: Sequence[str] | None = None,
        *,
        initial_step: int = 0,
        initial_positions: Sequence[Sequence[float]] | None = None,
        emergency_event_semantics: str = EMERGENCY_SEMANTICS_LEVEL,
    ) -> None:
        self.config = config
        if emergency_event_semantics not in {
            EMERGENCY_SEMANTICS_LEVEL,
            EMERGENCY_SEMANTICS_EDGE_REARM,
        }:
            raise ValueError("unsupported emergency_event_semantics")
        self.emergency_event_semantics = str(emergency_event_semantics)
        terminals = np.asarray(terminal_goals, dtype=float)
        if terminals.ndim != 2 or terminals.shape[1] != 3:
            raise ValueError("terminal_goals must have shape (N, 3)")
        active = terminals.copy() if active_goals is None else np.asarray(active_goals, dtype=float)
        if active.shape != terminals.shape or not np.all(np.isfinite(active)):
            raise ValueError("active_goals must match terminal_goals")
        types = (
            [ACTIVE_GOAL_TERMINAL] * len(terminals)
            if active_goal_types is None
            else list(active_goal_types)
        )
        if len(types) != len(terminals):
            raise ValueError("active_goal_types must have one item per agent")
        positions = None if initial_positions is None else np.asarray(initial_positions, dtype=float)
        if positions is not None and positions.shape != terminals.shape:
            raise ValueError("initial_positions must match terminal_goals")
        self.states: list[AgentReferenceState] = []
        for index, terminal in enumerate(terminals):
            history: deque[tuple[int, float]] = deque(
                maxlen=int(config.progress_window_steps) + 1
            )
            if positions is not None:
                history.append(
                    (
                        int(initial_step),
                        float(np.linalg.norm(positions[index] - active[index])),
                    )
                )
            self.states.append(
                AgentReferenceState(
                    active_goal=active[index],
                    terminal_goal=terminal,
                    active_goal_type=types[index],
                    last_reference_update_step=int(initial_step),
                    progress_distance_history=history,
                )
            )

    def synchronize_emergency_latch(
        self,
        agent_id: int,
        *,
        active_safety_margin_m: float,
    ) -> None:
        """Synchronize the edge latch to a newly active goal observation.

        Initial planning, a reproposal, and a reference handoff can all change
        the meaning of the active-direction margin.  They never rearm the
        emergency latch by virtue of being updates.  The latch is armed only
        when the newly active direction is already in the existing normal
        safety region h_active >= h_rep.
        """

        if self.emergency_event_semantics != EMERGENCY_SEMANTICS_EDGE_REARM:
            return
        margin = float(active_safety_margin_m)
        if not np.isfinite(margin):
            raise ValueError("active_safety_margin_m must be finite")
        state = self.states[int(agent_id)]
        state.emergency_armed = bool(
            margin >= self.config.safety_reproposal_margin_m
        )
        state.previous_active_safety_margin_m = margin

    def _evaluate_emergency_edge(
        self,
        state: AgentReferenceState,
        margin: float,
    ) -> tuple[bool, bool, bool, float | None]:
        armed_before = bool(state.emergency_armed)
        previous = state.previous_active_safety_margin_m
        rearmed = False
        if previous is None:
            # First observation after construction or an unsynchronized goal
            # update initializes the latch without creating a second same-tick
            # upper decision.
            state.emergency_armed = bool(
                margin >= self.config.safety_reproposal_margin_m
            )
            state.previous_active_safety_margin_m = margin
            return False, armed_before, False, previous
        if (
            not state.emergency_armed
            and margin >= self.config.safety_reproposal_margin_m
        ):
            state.emergency_armed = True
            rearmed = True
        emergency = bool(
            state.emergency_armed
            and previous > self.config.emergency_safety_margin_m
            and margin <= self.config.emergency_safety_margin_m
        )
        if emergency:
            state.emergency_armed = False
        state.previous_active_safety_margin_m = margin
        return emergency, armed_before, rearmed, previous

    def evaluate(
        self,
        agent_id: int,
        *,
        current_step: int,
        position: Sequence[float],
        active_safety_margin_m: float,
    ) -> TriggerDecision:
        state = self.states[int(agent_id)]
        step = int(current_step)
        point = _vector3(position, "position")
        margin = float(active_safety_margin_m)
        if not np.isfinite(margin):
            raise ValueError("active_safety_margin_m must be finite")
        distance = float(np.linalg.norm(point - state.active_goal))
        terminal_distance = float(np.linalg.norm(point - state.terminal_goal))
        history = state.progress_distance_history
        if not history or history[-1][0] != step:
            history.append((step, distance))
        else:
            history[-1] = (step, distance)
        window = int(self.config.progress_window_steps)
        progress_valid = bool(
            len(history) == window + 1 and step - history[0][0] == window
        )
        progress = (
            float((history[0][1] - history[-1][1]) / (window * self.config.dt))
            if progress_valid
            else None
        )
        age = float((step - state.last_reference_update_step) * self.config.dt)
        phi_tau = float(
            (age - self.config.reference_age_threshold_s)
            / self.config.reference_age_threshold_s
        )
        phi_p = (
            float(
                (self.config.minimum_progress_rate_mps - progress)
                / self.config.progress_scale_mps
            )
            if progress is not None
            else None
        )
        phi_h = float(
            (self.config.safety_reproposal_margin_m - margin)
            / self.config.safety_scale_m
        )
        valid_phi = [phi_tau, phi_h]
        if phi_p is not None:
            valid_phi.append(phi_p)
        score = float(max(valid_phi))
        dwell = age >= self.config.minimum_reconstruction_interval_s
        normal = bool(score >= 0.0 and dwell)
        if self.emergency_event_semantics == EMERGENCY_SEMANTICS_EDGE_REARM:
            emergency, emergency_armed_before, emergency_rearmed, previous_margin = (
                self._evaluate_emergency_edge(state, margin)
            )
            emergency_armed_after: bool | None = bool(state.emergency_armed)
        else:
            emergency = bool(margin <= self.config.emergency_safety_margin_m)
            emergency_armed_before = None
            emergency_armed_after = None
            emergency_rearmed = False
            previous_margin = None
        reference_completion = bool(
            state.active_goal_type == ACTIVE_GOAL_REFERENCE
            and distance <= self.config.handoff_distance_m
        )
        completion_action: str | None = None
        if reference_completion:
            if (
                self.config.reference_completion_mode
                == REFERENCE_COMPLETION_RECONSTRUCT_LONG_RANGE
                and terminal_distance > float(self.config.terminal_local_scope_m)
            ):
                completion_action = EVENT_REFERENCE_COMPLETION_REPROPOSAL
            else:
                completion_action = EVENT_REFERENCE_HANDOFF
        handoff = completion_action == EVENT_REFERENCE_HANDOFF
        reasons: list[str] = []
        if phi_tau >= 0.0:
            reasons.append("reference_age")
        if phi_p is not None and phi_p >= 0.0:
            reasons.append("progress_degradation")
        if phi_h >= 0.0:
            reasons.append("safety_degradation")
        if reference_completion:
            reasons.append("reference_completion")
        if completion_action is not None:
            event = completion_action
        elif emergency:
            event = EVENT_EMERGENCY_REPROPOSAL
        elif normal:
            event = EVENT_NORMAL_REPROPOSAL
        else:
            event = EVENT_NO_UPDATE
        state.number_of_trigger_checks += 1
        return TriggerDecision(
            event=event,
            current_step=step,
            active_goal_distance_m=distance,
            active_age_s=age,
            progress_rate_mps=progress,
            progress_valid=progress_valid,
            active_safety_margin_m=margin,
            phi_tau=phi_tau,
            phi_p=phi_p,
            phi_h=phi_h,
            S_rep=score,
            dwell_satisfied=dwell,
            normal_trigger=normal,
            emergency_trigger=emergency,
            handoff_trigger=handoff,
            reference_completion_trigger=reference_completion,
            terminal_goal_distance_m=terminal_distance,
            reference_completion_action=completion_action,
            trigger_reasons=tuple(reasons),
            emergency_event_semantics=self.emergency_event_semantics,
            emergency_armed_before=emergency_armed_before,
            emergency_armed_after=emergency_armed_after,
            emergency_rearmed=emergency_rearmed,
            previous_active_safety_margin_m=previous_margin,
        )

    def update_goal(
        self,
        agent_id: int,
        *,
        new_goal: Sequence[float],
        new_goal_type: str,
        current_step: int,
        position: Sequence[float],
        event: str,
    ) -> bool:
        state = self.states[int(agent_id)]
        goal = _vector3(new_goal, "new_goal")
        if new_goal_type not in {ACTIVE_GOAL_TERMINAL, ACTIVE_GOAL_REFERENCE}:
            raise ValueError("new_goal_type must be terminal or reference")
        changed = bool(
            np.linalg.norm(goal - state.active_goal)
            > self.config.goal_equality_tolerance_m
            or new_goal_type != state.active_goal_type
        )
        state.active_goal = goal
        state.active_goal_type = new_goal_type
        state.last_reference_update_step = int(current_step)
        state.goal_version += 1
        state.number_of_goal_changes += int(changed)
        state.progress_distance_history.clear()
        point = _vector3(position, "position")
        state.progress_distance_history.append(
            (int(current_step), float(np.linalg.norm(point - goal)))
        )
        if self.emergency_event_semantics == EMERGENCY_SEMANTICS_EDGE_REARM:
            # The caller must synchronize this state using the observable
            # safety margin of the newly active goal.  An update itself never
            # rearms the emergency latch.
            state.emergency_armed = False
            state.previous_active_safety_margin_m = None
        if event == EVENT_REFERENCE_HANDOFF:
            state.number_of_reference_handoffs += 1
        elif event in {
            EVENT_NORMAL_REPROPOSAL,
            EVENT_EMERGENCY_REPROPOSAL,
            EVENT_REFERENCE_COMPLETION_REPROPOSAL,
        }:
            state.number_of_reproposal_events += 1
            state.number_of_reference_completion_reproposals += int(
                event == EVENT_REFERENCE_COMPLETION_REPROPOSAL
            )
            state.number_of_emergency_events += int(
                event == EVENT_EMERGENCY_REPROPOSAL
            )
        return changed

    def handoff_to_terminal(
        self,
        agent_id: int,
        *,
        current_step: int,
        position: Sequence[float],
    ) -> bool:
        state = self.states[int(agent_id)]
        return self.update_goal(
            int(agent_id),
            new_goal=state.terminal_goal,
            new_goal_type=ACTIVE_GOAL_TERMINAL,
            current_step=int(current_step),
            position=position,
            event=EVENT_REFERENCE_HANDOFF,
        )

    def assert_terminal_goals_immutable(
        self, terminal_goals_after: Sequence[Sequence[float]]
    ) -> None:
        after = np.asarray(terminal_goals_after, dtype=float)
        expected = np.stack([state.terminal_goal for state in self.states])
        if not np.array_equal(after, expected):
            raise RuntimeError("terminal task goal changed during ERR execution")


def active_direction_safety_margin(
    env: Any,
    agent_id: int,
    active_goal: Sequence[float],
    proposal_config: ProposalConfig,
) -> SafetyMarginObservation:
    """Reuse Proposal's observable sector margin in the active-goal direction."""

    index = int(agent_id)
    position = np.asarray(env.dynamics[index].p, dtype=float)
    velocity = np.asarray(env.dynamics[index].v, dtype=float)
    goal = _vector3(active_goal, "active_goal")
    packet = env.latest_sensor_packets[index]
    sensor = env.sensors[index]
    if packet is None:
        raise RuntimeError("environment must be reset before safety evaluation")
    field = compute_sector_safety_field(
        position,
        goal,
        velocity,
        packet,
        sensor,
        proposal_config,
        float(env.env_config.goal_tolerance),
    )
    direction = goal - position
    norm = float(np.linalg.norm(direction))
    rays = np.asarray(sensor.ray_directions, dtype=float)
    flat = rays.reshape(-1, 3)
    if norm > 1.0e-12:
        sector_flat = int(np.argmax(flat @ (direction / norm)))
    else:
        sector_flat = int(np.argmax(field.normalized_margin.reshape(-1)))
    azimuth, elevation = np.unravel_index(sector_flat, field.safety_margin.shape)
    value = float(field.safety_margin[azimuth, elevation])
    normalized = float(field.normalized_margin[azimuth, elevation])
    return SafetyMarginObservation(
        value_m=value,
        normalized_value=normalized,
        source=ACTIVE_SAFETY_MARGIN_SOURCE,
        unit="m",
        sector_index_flat=sector_flat,
        azimuth_index=int(azimuth),
        elevation_index=int(elevation),
        raw_obstacle_distance_m=float(
            field.raw_obstacle_distance[azimuth, elevation]
        ),
        guarded_obstacle_distance_m=float(
            field.obstacle_distance[azimuth, elevation]
        ),
        effective_safe_radius_m=float(field.effective_safe_radius),
        braking_distance_m=float(field.braking_distance),
        normalization=(
            f"clip(safety_margin / {float(proposal_config.h_max):g} m, 0, 1)"
        ),
    )


def selected_goal_or_terminal(
    selected_candidate_id: int | None,
    proposals: Sequence[Any],
    terminal_goal: Sequence[float],
) -> tuple[np.ndarray, str]:
    """Map GAT null or K=0 safely to the immutable terminal task goal."""

    if selected_candidate_id is None or len(proposals) == 0:
        return _vector3(terminal_goal, "terminal_goal"), ACTIVE_GOAL_TERMINAL
    candidate_id = int(selected_candidate_id)
    if not 0 <= candidate_id < len(proposals):
        raise IndexError("selected candidate index is outside current K_t")
    return _vector3(proposals[candidate_id].point, "proposal.point"), ACTIVE_GOAL_REFERENCE


def set_active_goal_preserve_dmp_phase(dmp: Any, active_goal: Sequence[float]) -> None:
    """Change only the DMP goal and assert that its phase is untouched."""

    phase_before = float(dmp.phase)
    dmp.goal = _vector3(active_goal, "active_goal")
    if float(dmp.phase) != phase_before:
        raise RuntimeError("updating an active goal must preserve DMP phase")


__all__ = [
    "ACTIVE_GOAL_REFERENCE",
    "ACTIVE_GOAL_TERMINAL",
    "ACTIVE_SAFETY_MARGIN_SOURCE",
    "AgentReferenceState",
    "EMERGENCY_SEMANTICS_EDGE_REARM",
    "EMERGENCY_SEMANTICS_LEVEL",
    "ERRConfig",
    "EVENT_EMERGENCY_REPROPOSAL",
    "EVENT_NO_UPDATE",
    "EVENT_NORMAL_REPROPOSAL",
    "EVENT_REFERENCE_COMPLETION_REPROPOSAL",
    "EVENT_REFERENCE_HANDOFF",
    "EventTriggeredReferenceSupervisor",
    "SafetyMarginObservation",
    "TriggerDecision",
    "active_direction_safety_margin",
    "set_active_goal_preserve_dmp_phase",
    "selected_goal_or_terminal",
    "REFERENCE_COMPLETION_RECONSTRUCT_LONG_RANGE",
    "REFERENCE_COMPLETION_TERMINAL_HANDOFF",
]
