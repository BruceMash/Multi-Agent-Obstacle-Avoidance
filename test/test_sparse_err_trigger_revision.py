from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from planning.event_triggered_reference_reconstruction import (  # noqa: E402
    ACTIVE_GOAL_REFERENCE,
    EMERGENCY_SEMANTICS_EDGE_REARM,
    EMERGENCY_SEMANTICS_LEVEL,
    ERRConfig,
    EVENT_EMERGENCY_REPROPOSAL,
    EVENT_NO_UPDATE,
    EVENT_NORMAL_REPROPOSAL,
    EVENT_REFERENCE_HANDOFF,
    EventTriggeredReferenceSupervisor,
    set_active_goal_preserve_dmp_phase,
)


def _config(**changes: float | int) -> ERRConfig:
    values: dict[str, float | int] = {
        "dt": 0.1,
        "progress_window_steps": 2,
        "reference_age_threshold_s": 5.0,
        "minimum_reconstruction_interval_s": 1.0,
        "minimum_progress_rate_mps": 0.1,
        "progress_scale_mps": 1.0,
        "safety_reproposal_margin_m": 0.3,
        "safety_scale_m": 1.0,
        "emergency_safety_margin_m": 0.0,
        "handoff_distance_m": 0.25,
    }
    values.update(changes)
    return ERRConfig(**values)


def _supervisor(
    *,
    semantics: str = EMERGENCY_SEMANTICS_EDGE_REARM,
    config: ERRConfig | None = None,
) -> EventTriggeredReferenceSupervisor:
    return EventTriggeredReferenceSupervisor(
        config or _config(),
        terminal_goals=[[8.0, 0.0, 0.0]],
        active_goals=[[1.0, 0.0, 0.0]],
        active_goal_types=[ACTIVE_GOAL_REFERENCE],
        initial_positions=[[0.0, 0.0, 0.0]],
        emergency_event_semantics=semantics,
    )


def _evaluate(
    supervisor: EventTriggeredReferenceSupervisor,
    step: int,
    margin: float,
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
):
    return supervisor.evaluate(
        0,
        current_step=step,
        position=position,
        active_safety_margin_m=margin,
    )


def test_a_persistent_emergency_region_triggers_once() -> None:
    supervisor = _supervisor()
    supervisor.synchronize_emergency_latch(0, active_safety_margin_m=0.4)
    events = [_evaluate(supervisor, 1, -0.1).event]
    events.extend(_evaluate(supervisor, step, -0.1).event for step in range(2, 11))
    assert events.count(EVENT_EMERGENCY_REPROPOSAL) == 1


def test_b_hysteresis_band_does_not_rearm() -> None:
    supervisor = _supervisor()
    supervisor.synchronize_emergency_latch(0, active_safety_margin_m=0.4)
    assert _evaluate(supervisor, 1, -0.1).event == EVENT_EMERGENCY_REPROPOSAL
    for step, margin in enumerate((0.05, 0.2, 0.29, -0.1), start=2):
        assert _evaluate(supervisor, step, margin).event != EVENT_EMERGENCY_REPROPOSAL
    assert not supervisor.states[0].emergency_armed


def test_c_existing_h_rep_rearms_latch() -> None:
    supervisor = _supervisor()
    supervisor.synchronize_emergency_latch(0, active_safety_margin_m=0.4)
    _evaluate(supervisor, 1, -0.1)
    decision = _evaluate(supervisor, 2, 0.3)
    assert decision.emergency_rearmed
    assert decision.emergency_armed_after


def test_d_second_downcross_after_recovery_triggers_again() -> None:
    supervisor = _supervisor()
    supervisor.synchronize_emergency_latch(0, active_safety_margin_m=0.4)
    first = _evaluate(supervisor, 1, -0.1)
    _evaluate(supervisor, 2, 0.3)
    second = _evaluate(supervisor, 3, -0.1)
    assert first.event == EVENT_EMERGENCY_REPROPOSAL
    assert second.event == EVENT_EMERGENCY_REPROPOSAL


def test_e_arbitrary_number_of_independent_emergency_entries_is_allowed() -> None:
    supervisor = _supervisor()
    supervisor.synchronize_emergency_latch(0, active_safety_margin_m=0.4)
    emergency_count = 0
    for cycle in range(7):
        emergency_count += int(
            _evaluate(supervisor, 2 * cycle + 1, -0.1).event
            == EVENT_EMERGENCY_REPROPOSAL
        )
        _evaluate(supervisor, 2 * cycle + 2, 0.4)
    assert emergency_count == 7


def test_f_no_artificial_event_cap_exists() -> None:
    source = inspect.getsource(EventTriggeredReferenceSupervisor)
    forbidden = ("N_replan_max", "max_reproposal_count", "event_cap")
    assert all(token not in source for token in forbidden)


def test_g_normal_branch_is_unchanged_between_semantics() -> None:
    level = _supervisor(semantics=EMERGENCY_SEMANTICS_LEVEL)
    edge = _supervisor(semantics=EMERGENCY_SEMANTICS_EDGE_REARM)
    edge.synchronize_emergency_latch(0, active_safety_margin_m=1.0)
    for step, position, margin in (
        (1, (0.1, 0.0, 0.0), 1.0),
        (2, (0.1, 0.0, 0.0), 1.0),
        (10, (0.1, 0.0, 0.0), 0.2),
    ):
        level_decision = _evaluate(level, step, margin, position)
        edge_decision = _evaluate(edge, step, margin, position)
        assert level_decision.normal_trigger == edge_decision.normal_trigger
        assert level_decision.S_rep == edge_decision.S_rep
        assert level_decision.phi_tau == edge_decision.phi_tau
        assert level_decision.phi_p == edge_decision.phi_p
        assert level_decision.phi_h == edge_decision.phi_h
    assert edge_decision.event == EVENT_NORMAL_REPROPOSAL


def test_h_handoff_keeps_same_tick_priority_over_emergency() -> None:
    supervisor = _supervisor()
    supervisor.synchronize_emergency_latch(0, active_safety_margin_m=0.4)
    decision = _evaluate(supervisor, 1, -0.1, (0.8, 0.0, 0.0))
    assert decision.handoff_trigger and decision.emergency_trigger
    assert decision.event == EVENT_REFERENCE_HANDOFF


class _DummyDMP:
    def __init__(self) -> None:
        self.goal = np.zeros(3)
        self.phase = 0.37


def test_i_goal_update_preserves_external_position_velocity_and_dmp_phase() -> None:
    supervisor = _supervisor()
    position = np.asarray([0.2, -0.1, 0.3])
    velocity = np.asarray([0.4, -0.2, 0.1])
    position_before = position.copy()
    velocity_before = velocity.copy()
    dmp = _DummyDMP()
    supervisor.update_goal(
        0,
        new_goal=[2.0, 0.0, 0.0],
        new_goal_type=ACTIVE_GOAL_REFERENCE,
        current_step=1,
        position=position,
        event=EVENT_NORMAL_REPROPOSAL,
    )
    set_active_goal_preserve_dmp_phase(dmp, supervisor.states[0].active_goal)
    np.testing.assert_array_equal(position, position_before)
    np.testing.assert_array_equal(velocity, velocity_before)
    assert dmp.phase == 0.37


def test_post_update_synchronization_uses_new_active_margin_only() -> None:
    supervisor = _supervisor()
    supervisor.synchronize_emergency_latch(0, active_safety_margin_m=0.4)
    _evaluate(supervisor, 1, -0.1)
    supervisor.update_goal(
        0,
        new_goal=[2.0, 0.0, 0.0],
        new_goal_type=ACTIVE_GOAL_REFERENCE,
        current_step=1,
        position=[0.0, 0.0, 0.0],
        event=EVENT_EMERGENCY_REPROPOSAL,
    )
    assert not supervisor.states[0].emergency_armed
    supervisor.synchronize_emergency_latch(0, active_safety_margin_m=0.2)
    assert not supervisor.states[0].emergency_armed
    assert _evaluate(supervisor, 2, -0.1).event != EVENT_EMERGENCY_REPROPOSAL


def test_default_semantics_remains_legacy_level_for_frozen_m2() -> None:
    default = _supervisor(semantics=EMERGENCY_SEMANTICS_LEVEL)
    assert _evaluate(default, 0, -0.1).event == EVENT_EMERGENCY_REPROPOSAL
    assert _evaluate(default, 1, -0.1).event == EVENT_EMERGENCY_REPROPOSAL
