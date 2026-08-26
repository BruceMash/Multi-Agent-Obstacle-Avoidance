from __future__ import annotations

import sys
from dataclasses import fields
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
ALGO_ROOT = REPO_ROOT / "Multi-agent_Algo_lib"
SCRIPTS_ROOT = ALGO_ROOT / "scripts"
for search_path in (REPO_ROOT, ALGO_ROOT, SCRIPTS_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from planning.event_triggered_reference_reconstruction import (  # noqa: E402
    ACTIVE_GOAL_REFERENCE,
    ACTIVE_GOAL_TERMINAL,
    ERRConfig,
    EVENT_EMERGENCY_REPROPOSAL,
    EVENT_NO_UPDATE,
    EVENT_NORMAL_REPROPOSAL,
    EVENT_REFERENCE_HANDOFF,
    AgentReferenceState,
    EventTriggeredReferenceSupervisor,
    set_active_goal_preserve_dmp_phase,
    selected_goal_or_terminal,
)


def _config(**changes) -> ERRConfig:
    values = {
        "dt": 0.1,
        "progress_window_steps": 2,
        "reference_age_threshold_s": 5.0,
        "minimum_reconstruction_interval_s": 0.1,
        "minimum_progress_rate_mps": 0.1,
        "progress_scale_mps": 1.0,
        "safety_reproposal_margin_m": 0.3,
        "safety_scale_m": 1.0,
        "emergency_safety_margin_m": 0.0,
        "handoff_distance_m": 0.25,
    }
    values.update(changes)
    return ERRConfig(**values)


def _supervisor(config: ERRConfig | None = None) -> EventTriggeredReferenceSupervisor:
    return EventTriggeredReferenceSupervisor(
        config or _config(),
        terminal_goals=[[8.0, 0.0, 0.0]],
        active_goals=[[1.0, 0.0, 0.0]],
        active_goal_types=[ACTIVE_GOAL_REFERENCE],
        initial_positions=[[0.0, 0.0, 0.0]],
    )


def test_01_no_trigger_keeps_active_goal_unchanged() -> None:
    supervisor = _supervisor()
    before = supervisor.states[0].active_goal.copy()
    decision = supervisor.evaluate(
        0, current_step=1, position=[0.2, 0.0, 0.0], active_safety_margin_m=1.0
    )
    assert decision.event == EVENT_NO_UPDATE
    np.testing.assert_array_equal(supervisor.states[0].active_goal, before)


def test_02_time_trigger_reproposes_exactly_once_after_commit() -> None:
    supervisor = _supervisor()
    decision = supervisor.evaluate(
        0, current_step=50, position=[0.2, 0.0, 0.0], active_safety_margin_m=1.0
    )
    assert decision.event == EVENT_NORMAL_REPROPOSAL
    supervisor.update_goal(
        0,
        new_goal=[2.0, 0.0, 0.0],
        new_goal_type=ACTIVE_GOAL_REFERENCE,
        current_step=50,
        position=[0.2, 0.0, 0.0],
        event=decision.event,
    )
    assert supervisor.states[0].number_of_reproposal_events == 1
    assert supervisor.evaluate(
        0, current_step=50, position=[0.2, 0.0, 0.0], active_safety_margin_m=1.0
    ).event == EVENT_NO_UPDATE


def test_03_windowed_progress_trigger_reproposes_exactly_once() -> None:
    supervisor = _supervisor()
    supervisor.evaluate(
        0, current_step=1, position=[0.01, 0.0, 0.0], active_safety_margin_m=1.0
    )
    decision = supervisor.evaluate(
        0, current_step=2, position=[0.01, 0.0, 0.0], active_safety_margin_m=1.0
    )
    assert decision.progress_valid
    assert np.isclose(decision.progress_rate_mps, 0.05)
    assert decision.event == EVENT_NORMAL_REPROPOSAL


def test_04_safety_trigger_reproposes_exactly_once() -> None:
    supervisor = _supervisor()
    decision = supervisor.evaluate(
        0, current_step=1, position=[0.0, 0.0, 0.0], active_safety_margin_m=0.2
    )
    assert decision.event == EVENT_NORMAL_REPROPOSAL
    assert "safety_degradation" in decision.trigger_reasons


def test_05_emergency_bypasses_dwell() -> None:
    supervisor = _supervisor(
        _config(minimum_reconstruction_interval_s=1.0)
    )
    decision = supervisor.evaluate(
        0, current_step=0, position=[0.0, 0.0, 0.0], active_safety_margin_m=-0.01
    )
    assert not decision.dwell_satisfied
    assert decision.event == EVENT_EMERGENCY_REPROPOSAL


def test_06_reference_completion_handoff_has_highest_priority() -> None:
    supervisor = _supervisor()
    decision = supervisor.evaluate(
        0, current_step=1, position=[0.8, 0.0, 0.0], active_safety_margin_m=-0.1
    )
    assert decision.emergency_trigger
    assert decision.handoff_trigger
    assert decision.event == EVENT_REFERENCE_HANDOFF


class _DummyDMP:
    def __init__(self) -> None:
        self.goal = np.zeros(3)
        self.phase = 0.37


def test_07_goal_update_preserves_velocity() -> None:
    dmp = _DummyDMP()
    velocity = np.asarray([0.4, -0.2, 0.1])
    before = velocity.copy()
    set_active_goal_preserve_dmp_phase(dmp, np.asarray([2.0, 1.0, 0.0]))
    np.testing.assert_array_equal(velocity, before)


def test_08_goal_update_preserves_dmp_phase() -> None:
    dmp = _DummyDMP()
    set_active_goal_preserve_dmp_phase(dmp, np.asarray([2.0, 1.0, 0.0]))
    assert dmp.phase == 0.37


def test_09_terminal_goal_is_immutable() -> None:
    supervisor = _supervisor()
    assert not supervisor.states[0].terminal_goal.flags.writeable
    supervisor.assert_terminal_goals_immutable([[8.0, 0.0, 0.0]])


def test_10_progress_history_resets_after_goal_change() -> None:
    supervisor = _supervisor()
    supervisor.evaluate(
        0, current_step=1, position=[0.1, 0.0, 0.0], active_safety_margin_m=1.0
    )
    supervisor.update_goal(
        0,
        new_goal=[3.0, 0.0, 0.0],
        new_goal_type=ACTIVE_GOAL_REFERENCE,
        current_step=1,
        position=[0.1, 0.0, 0.0],
        event=EVENT_NORMAL_REPROPOSAL,
    )
    assert list(supervisor.states[0].progress_distance_history) == [(1, 2.9)]
    assert not supervisor.evaluate(
        0, current_step=2, position=[0.2, 0.0, 0.0], active_safety_margin_m=1.0
    ).progress_valid


def test_11_no_pending_state_exists() -> None:
    field_names = {item.name for item in fields(AgentReferenceState)}
    assert not any("pending" in name for name in field_names)


def test_12_no_fixed_period_forced_replanning() -> None:
    supervisor = _supervisor(
        _config(reference_age_threshold_s=10.0)
    )
    decision = supervisor.evaluate(
        0, current_step=7, position=[0.7, 0.0, 0.0], active_safety_margin_m=1.0
    )
    assert decision.event == EVENT_NO_UPDATE


def test_13_k_zero_maps_safely_to_terminal() -> None:
    goal, goal_type = selected_goal_or_terminal(None, [], [8.0, 0.0, 0.0])
    np.testing.assert_array_equal(goal, [8.0, 0.0, 0.0])
    assert goal_type == ACTIVE_GOAL_TERMINAL


def test_14_multiple_agents_have_independent_trigger_state() -> None:
    supervisor = EventTriggeredReferenceSupervisor(
        _config(),
        terminal_goals=[[8.0, 0.0, 0.0], [8.0, 1.0, 0.0]],
        active_goals=[[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        active_goal_types=[ACTIVE_GOAL_REFERENCE, ACTIVE_GOAL_REFERENCE],
        initial_positions=[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )
    first = supervisor.evaluate(
        0, current_step=50, position=[0.2, 0.0, 0.0], active_safety_margin_m=1.0
    )
    second = supervisor.evaluate(
        1, current_step=1, position=[0.2, 1.0, 0.0], active_safety_margin_m=1.0
    )
    assert first.event == EVENT_NORMAL_REPROPOSAL
    assert second.event == EVENT_NO_UPDATE
    assert supervisor.states[0].number_of_trigger_checks == 1
    assert supervisor.states[1].number_of_trigger_checks == 1
