from __future__ import annotations

import numpy as np

from planning.proposal_err_lite import (
    ProposalERRLiteConfig,
    ProposalERRLiteSupervisor,
    TRIGGER_REFERENCE_REACHED,
    TRIGGER_REFERENCE_STAGNATION,
    TRIGGER_TERMINAL_RETRY,
)


def _supervisor(*, reference: bool = True) -> ProposalERRLiteSupervisor:
    return ProposalERRLiteSupervisor(
        ProposalERRLiteConfig(),
        active_goals=[[1.0, 0.0, 0.0]],
        is_reference=[reference],
        terminal_retry_enabled=[not reference],
        positions=[[0.0, 0.0, 0.0]],
    )


def test_reference_reached_has_priority() -> None:
    supervisor = _supervisor()
    decision = supervisor.evaluate(
        0, current_step=10, position=[0.8, 0.0, 0.0]
    )
    assert decision.trigger == TRIGGER_REFERENCE_REACHED


def test_stagnation_requires_full_window_and_dwell() -> None:
    supervisor = _supervisor()
    for step in range(1, 30):
        decision = supervisor.evaluate(
            0, current_step=step, position=[0.0, 0.0, 0.0]
        )
        assert decision.trigger is None
    decision = supervisor.evaluate(
        0, current_step=30, position=[0.0, 0.0, 0.0]
    )
    assert decision.trigger == TRIGGER_REFERENCE_STAGNATION
    assert decision.progress_rate_mps == 0.0


def test_terminal_fallback_retries_only_at_five_seconds() -> None:
    supervisor = _supervisor(reference=False)
    assert (
        supervisor.evaluate(0, current_step=49, position=[0.0, 0.0, 0.0]).trigger
        is None
    )
    assert (
        supervisor.evaluate(0, current_step=50, position=[0.0, 0.0, 0.0]).trigger
        == TRIGGER_TERMINAL_RETRY
    )


def test_update_resets_window_and_counts_unchanged_goal() -> None:
    supervisor = _supervisor()
    state = supervisor.states[0]
    changed = supervisor.update(
        0,
        new_goal=np.array([1.0, 0.0, 0.0]),
        is_reference=True,
        terminal_retry_enabled=False,
        current_step=30,
        position=[0.0, 0.0, 0.0],
    )
    assert not changed
    assert state.replan_count == 1
    assert state.unchanged_replan_count == 1
    assert list(state.distance_history) == [(30, 1.0)]


def test_safe_terminal_handoff_does_not_retry() -> None:
    supervisor = _supervisor()
    supervisor.update(
        0,
        new_goal=[2.0, 0.0, 0.0],
        is_reference=False,
        terminal_retry_enabled=False,
        current_step=10,
        position=[0.8, 0.0, 0.0],
        count_as_replan=False,
    )
    decision = supervisor.evaluate(
        0, current_step=100, position=[0.8, 0.0, 0.0]
    )
    assert decision.trigger is None
    assert supervisor.states[0].replan_count == 0
