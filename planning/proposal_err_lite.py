"""Minimal event triggers for sequential Proposal execution.

The state machine deliberately contains no safety/emergency trigger and no
selector logic.  It only decides when the unchanged Proposal generator should
be invoked again.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


TRIGGER_REFERENCE_REACHED = "REFERENCE_REACHED"
TRIGGER_REFERENCE_STAGNATION = "REFERENCE_STAGNATION"
TRIGGER_TERMINAL_RETRY = "TERMINAL_FALLBACK_RETRY"


def _vector3(value: Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite 3-vector")
    return result.copy()


@dataclass(frozen=True)
class ProposalERRLiteConfig:
    dt: float = 0.1
    progress_window_steps: int = 30
    minimum_progress_rate_mps: float = 0.08 / 3.0
    minimum_replan_interval_s: float = 1.0
    terminal_retry_age_s: float = 5.0
    reference_reached_distance_m: float = 0.25
    goal_equality_tolerance_m: float = 1.0e-9

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.progress_window_steps <= 0:
            raise ValueError("progress_window_steps must be positive")
        if self.minimum_progress_rate_mps < 0.0:
            raise ValueError("minimum_progress_rate_mps must be non-negative")
        for value in (
            self.minimum_replan_interval_s,
            self.terminal_retry_age_s,
            self.reference_reached_distance_m,
            self.goal_equality_tolerance_m,
        ):
            if value <= 0.0:
                raise ValueError("ERR-lite time and distance settings must be positive")


@dataclass
class ProposalERRLiteState:
    active_goal: np.ndarray
    is_reference: bool
    terminal_retry_enabled: bool = False
    last_update_step: int = 0
    distance_history: deque[tuple[int, float]] = field(default_factory=deque)
    replan_count: int = 0
    goal_change_count: int = 0
    unchanged_replan_count: int = 0

    def __post_init__(self) -> None:
        self.active_goal = _vector3(self.active_goal, "active_goal")


@dataclass(frozen=True)
class ProposalERRLiteDecision:
    trigger: str | None
    active_goal_distance_m: float
    active_goal_age_s: float
    progress_rate_mps: float | None
    dwell_satisfied: bool


class ProposalERRLiteSupervisor:
    """Per-agent trigger state for Proposal-only reference reconstruction."""

    def __init__(
        self,
        config: ProposalERRLiteConfig,
        active_goals: Sequence[Sequence[float]],
        is_reference: Sequence[bool],
        terminal_retry_enabled: Sequence[bool] | None,
        positions: Sequence[Sequence[float]],
    ) -> None:
        self.config = config
        goals = np.asarray(active_goals, dtype=float)
        points = np.asarray(positions, dtype=float)
        flags = np.asarray(is_reference, dtype=bool)
        retry_flags = (
            np.logical_not(flags)
            if terminal_retry_enabled is None
            else np.asarray(terminal_retry_enabled, dtype=bool)
        )
        if goals.ndim != 2 or goals.shape[1] != 3 or points.shape != goals.shape:
            raise ValueError("active_goals and positions must have shape (N, 3)")
        if flags.shape != (len(goals),):
            raise ValueError("is_reference must have one value per agent")
        if retry_flags.shape != (len(goals),):
            raise ValueError("terminal_retry_enabled must have one value per agent")
        self.states: list[ProposalERRLiteState] = []
        for goal, flag, retry, point in zip(
            goals, flags, retry_flags, points, strict=True
        ):
            history: deque[tuple[int, float]] = deque(
                maxlen=int(config.progress_window_steps) + 1
            )
            history.append((0, float(np.linalg.norm(goal - point))))
            self.states.append(
                ProposalERRLiteState(
                    active_goal=goal,
                    is_reference=bool(flag),
                    terminal_retry_enabled=bool(retry),
                    distance_history=history,
                )
            )

    def evaluate(
        self,
        agent_id: int,
        *,
        current_step: int,
        position: Sequence[float],
    ) -> ProposalERRLiteDecision:
        state = self.states[int(agent_id)]
        point = _vector3(position, "position")
        step = int(current_step)
        distance = float(np.linalg.norm(state.active_goal - point))
        history = state.distance_history
        if not history or history[-1][0] != step:
            history.append((step, distance))
        else:
            history[-1] = (step, distance)
        window = int(self.config.progress_window_steps)
        valid = len(history) == window + 1 and step - history[0][0] == window
        progress = (
            float((history[0][1] - history[-1][1]) / (window * self.config.dt))
            if valid
            else None
        )
        age = float((step - state.last_update_step) * self.config.dt)
        dwell = age >= self.config.minimum_replan_interval_s
        trigger: str | None = None
        if (
            state.is_reference
            and dwell
            and distance <= self.config.reference_reached_distance_m
        ):
            trigger = TRIGGER_REFERENCE_REACHED
        elif (
            state.is_reference
            and dwell
            and progress is not None
            and progress < self.config.minimum_progress_rate_mps
        ):
            trigger = TRIGGER_REFERENCE_STAGNATION
        elif (
            not state.is_reference
            and state.terminal_retry_enabled
            and dwell
            and age >= self.config.terminal_retry_age_s
        ):
            trigger = TRIGGER_TERMINAL_RETRY
        return ProposalERRLiteDecision(
            trigger=trigger,
            active_goal_distance_m=distance,
            active_goal_age_s=age,
            progress_rate_mps=progress,
            dwell_satisfied=dwell,
        )

    def update(
        self,
        agent_id: int,
        *,
        new_goal: Sequence[float],
        is_reference: bool,
        terminal_retry_enabled: bool,
        current_step: int,
        position: Sequence[float],
        count_as_replan: bool = True,
    ) -> bool:
        state = self.states[int(agent_id)]
        goal = _vector3(new_goal, "new_goal")
        changed = bool(
            np.linalg.norm(goal - state.active_goal)
            > self.config.goal_equality_tolerance_m
            or bool(is_reference) != state.is_reference
            or bool(terminal_retry_enabled) != state.terminal_retry_enabled
        )
        state.active_goal = goal
        state.is_reference = bool(is_reference)
        state.terminal_retry_enabled = bool(terminal_retry_enabled)
        state.last_update_step = int(current_step)
        state.replan_count += int(count_as_replan)
        state.goal_change_count += int(changed)
        state.unchanged_replan_count += int(not changed)
        state.distance_history.clear()
        point = _vector3(position, "position")
        state.distance_history.append(
            (int(current_step), float(np.linalg.norm(goal - point)))
        )
        return changed


__all__ = [
    "ProposalERRLiteConfig",
    "ProposalERRLiteDecision",
    "ProposalERRLiteState",
    "ProposalERRLiteSupervisor",
    "TRIGGER_REFERENCE_REACHED",
    "TRIGGER_REFERENCE_STAGNATION",
    "TRIGGER_TERMINAL_RETRY",
]
