from __future__ import annotations

import unittest

import numpy as np

from planning.event_triggered_reference_reconstruction import (
    ACTIVE_GOAL_REFERENCE,
    ACTIVE_GOAL_TERMINAL,
    ERRConfig,
    EVENT_REFERENCE_COMPLETION_REPROPOSAL,
    EVENT_REFERENCE_HANDOFF,
    FAR_TERMINAL_NULL_ALLOW,
    FAR_TERMINAL_NULL_MASK_TO_GAT_CANDIDATE,
    INTERACTION_FEASIBILITY_ALLOW_RISKY,
    INTERACTION_FEASIBILITY_MASK_WHEN_SAFE,
    EventTriggeredReferenceSupervisor,
    REFERENCE_COMPLETION_RECONSTRUCT_LONG_RANGE,
    apply_far_terminal_null_policy,
    apply_interaction_feasibility_policy,
)


def _config(*, long_range: bool) -> ERRConfig:
    return ERRConfig(
        dt=0.1,
        progress_window_steps=5,
        reference_age_threshold_s=2.0,
        minimum_reconstruction_interval_s=0.5,
        minimum_progress_rate_mps=0.1,
        progress_scale_mps=0.5,
        safety_reproposal_margin_m=0.4,
        safety_scale_m=0.4,
        emergency_safety_margin_m=0.0,
        handoff_distance_m=0.25,
        reference_completion_mode=(
            REFERENCE_COMPLETION_RECONSTRUCT_LONG_RANGE if long_range else "terminal_handoff"
        ),
        terminal_local_scope_m=4.5 if long_range else None,
    )


def _supervisor(*, terminal, active, long_range: bool) -> EventTriggeredReferenceSupervisor:
    return EventTriggeredReferenceSupervisor(
        _config(long_range=long_range),
        terminal_goals=[terminal],
        active_goals=[active],
        active_goal_types=[ACTIVE_GOAL_REFERENCE],
        initial_positions=[[0.0, 0.0, 2.0]],
    )


class LongRangeReferenceLifecycleTest(unittest.TestCase):
    def test_default_mode_preserves_historical_terminal_handoff(self) -> None:
        supervisor = _supervisor(terminal=[80, 0, 2], active=[4, 0, 2], long_range=False)
        decision = supervisor.evaluate(
            0, current_step=10, position=[4, 0, 2], active_safety_margin_m=1.0
        )
        self.assertEqual(decision.event, EVENT_REFERENCE_HANDOFF)
        self.assertTrue(decision.handoff_trigger)
        self.assertTrue(decision.reference_completion_trigger)

    def test_far_terminal_converts_completion_into_existing_upper_reconstruction(self) -> None:
        supervisor = _supervisor(terminal=[80, 0, 2], active=[4, 0, 2], long_range=True)
        decision = supervisor.evaluate(
            0, current_step=10, position=[4, 0, 2], active_safety_margin_m=1.0
        )
        self.assertEqual(decision.event, EVENT_REFERENCE_COMPLETION_REPROPOSAL)
        self.assertFalse(decision.handoff_trigger)
        self.assertTrue(decision.reference_completion_trigger)
        self.assertGreater(decision.terminal_goal_distance_m, 4.5)
        self.assertIn("reference_completion", decision.trigger_reasons)

    def test_true_local_terminal_still_hands_off_directly(self) -> None:
        supervisor = _supervisor(terminal=[5, 0, 2], active=[4.8, 0, 2], long_range=True)
        decision = supervisor.evaluate(
            0, current_step=10, position=[4.8, 0, 2], active_safety_margin_m=1.0
        )
        self.assertEqual(decision.event, EVENT_REFERENCE_HANDOFF)
        self.assertTrue(decision.handoff_trigger)
        self.assertLessEqual(decision.terminal_goal_distance_m, 4.5)

    def test_reference_completion_keeps_priority_over_emergency(self) -> None:
        supervisor = _supervisor(terminal=[80, 0, 2], active=[4, 0, 2], long_range=True)
        decision = supervisor.evaluate(
            0, current_step=10, position=[4, 0, 2], active_safety_margin_m=-0.1
        )
        self.assertTrue(decision.emergency_trigger)
        self.assertEqual(decision.event, EVENT_REFERENCE_COMPLETION_REPROPOSAL)

    def test_completion_reproposal_uses_normal_atomic_goal_update_accounting(self) -> None:
        supervisor = _supervisor(terminal=[80, 0, 2], active=[4, 0, 2], long_range=True)
        changed = supervisor.update_goal(
            0,
            new_goal=[8, 1, 2],
            new_goal_type=ACTIVE_GOAL_REFERENCE,
            current_step=10,
            position=[4, 0, 2],
            event=EVENT_REFERENCE_COMPLETION_REPROPOSAL,
        )
        state = supervisor.states[0]
        self.assertTrue(changed)
        self.assertEqual(state.active_goal_type, ACTIVE_GOAL_REFERENCE)
        np.testing.assert_allclose(state.active_goal, [8, 1, 2])
        self.assertEqual(state.number_of_reproposal_events, 1)
        self.assertEqual(state.number_of_reference_completion_reproposals, 1)
        self.assertEqual(state.number_of_reference_handoffs, 0)

    def test_invalid_long_range_scope_is_rejected(self) -> None:
        values = vars(_config(long_range=True)).copy()
        values["terminal_local_scope_m"] = None
        with self.assertRaisesRegex(ValueError, "terminal_local_scope_m"):
            ERRConfig(**values)

    def test_far_terminal_null_is_masked_to_highest_non_null_gat_logit(self) -> None:
        decision = apply_far_terminal_null_policy(
            raw_class_index=0,
            class_logits=[4.0, 1.0, 3.5, 2.0],
            proposal_count=3,
            terminal_goal_distance_m=70.0,
            terminal_local_scope_m=4.5,
            policy=FAR_TERMINAL_NULL_MASK_TO_GAT_CANDIDATE,
        )
        self.assertEqual(decision.raw_selected_candidate_id, None)
        self.assertEqual(decision.effective_class_index, 2)
        self.assertEqual(decision.effective_selected_candidate_id, 1)
        self.assertTrue(decision.far_terminal_null_mask_applied)
        self.assertFalse(decision.terminal_null_eligible)

    def test_terminal_null_remains_eligible_inside_local_scope(self) -> None:
        decision = apply_far_terminal_null_policy(
            raw_class_index=0,
            class_logits=[4.0, 1.0, 3.5],
            proposal_count=2,
            terminal_goal_distance_m=4.0,
            terminal_local_scope_m=4.5,
            policy=FAR_TERMINAL_NULL_MASK_TO_GAT_CANDIDATE,
        )
        self.assertIsNone(decision.effective_selected_candidate_id)
        self.assertFalse(decision.far_terminal_null_mask_applied)
        self.assertTrue(decision.terminal_null_eligible)

    def test_legacy_null_policy_is_unchanged(self) -> None:
        decision = apply_far_terminal_null_policy(
            raw_class_index=0,
            class_logits=[4.0, 1.0, 3.5],
            proposal_count=2,
            terminal_goal_distance_m=70.0,
            terminal_local_scope_m=4.5,
            policy=FAR_TERMINAL_NULL_ALLOW,
        )
        self.assertIsNone(decision.effective_selected_candidate_id)
        self.assertFalse(decision.far_terminal_null_mask_applied)

    def test_far_terminal_null_without_candidates_is_explicitly_unavailable(self) -> None:
        decision = apply_far_terminal_null_policy(
            raw_class_index=0,
            class_logits=[1.0],
            proposal_count=0,
            terminal_goal_distance_m=70.0,
            terminal_local_scope_m=4.5,
            policy=FAR_TERMINAL_NULL_MASK_TO_GAT_CANDIDATE,
        )
        self.assertIsNone(decision.effective_selected_candidate_id)
        self.assertFalse(decision.far_terminal_null_mask_applied)
        self.assertTrue(decision.far_terminal_null_mask_unavailable_no_candidate)

    def test_risky_candidate_is_masked_to_highest_logit_safe_alternative(self) -> None:
        decision = apply_interaction_feasibility_policy(
            selected_candidate_id=0,
            class_logits=[0.0, 5.0, 2.0, 3.0],
            candidate_risky=[True, False, False],
            policy=INTERACTION_FEASIBILITY_MASK_WHEN_SAFE,
        )
        self.assertEqual(decision.selected_candidate_id_before_mask, 0)
        self.assertEqual(decision.effective_selected_candidate_id, 2)
        self.assertTrue(decision.interaction_mask_applied)
        self.assertEqual(decision.safe_candidate_count, 2)

    def test_interaction_mask_does_not_invent_safe_candidate(self) -> None:
        decision = apply_interaction_feasibility_policy(
            selected_candidate_id=1,
            class_logits=[0.0, 2.0, 5.0],
            candidate_risky=[True, True],
            policy=INTERACTION_FEASIBILITY_MASK_WHEN_SAFE,
        )
        self.assertEqual(decision.effective_selected_candidate_id, 1)
        self.assertFalse(decision.interaction_mask_applied)
        self.assertEqual(decision.safe_candidate_count, 0)

    def test_legacy_interaction_policy_preserves_risky_selection(self) -> None:
        decision = apply_interaction_feasibility_policy(
            selected_candidate_id=0,
            class_logits=[0.0, 5.0, 3.0],
            candidate_risky=[True, False],
            policy=INTERACTION_FEASIBILITY_ALLOW_RISKY,
        )
        self.assertEqual(decision.effective_selected_candidate_id, 0)
        self.assertFalse(decision.interaction_mask_applied)


if __name__ == "__main__":
    unittest.main()
