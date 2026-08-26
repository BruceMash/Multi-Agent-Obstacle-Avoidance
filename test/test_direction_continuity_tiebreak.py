from __future__ import annotations

import math
from types import SimpleNamespace

from planning.direction_continuity_tiebreak import (
    DirectionContinuityTieBreak,
    DirectionContinuityTieBreakConfig,
)


def _point(alpha: float) -> list[float]:
    return [5.0 * math.cos(alpha), 5.0 * math.sin(alpha), 0.0]


def _selection(selected: int, angles: list[float], margin: float) -> dict:
    points = [_point(value) for value in angles]
    probabilities = [0.01, 0.55, 0.08, 0.36]
    if selected == 1:
        probabilities = [0.01, 0.08, 0.55, 0.36]
    record = {
        "selected_candidate_id": selected,
        "selected_class": selected + 1,
        "selected_null": False,
        "selection_source": "gat_stage1",
        "candidate_world_points": points,
        "proposal_scores": [3.0, 2.0, 1.0],
        "class_probabilities": probabilities,
        "class_logits": probabilities,
        "top1_top2_probability_margin": margin,
        "fp_shep_candidate_records": [
            {
                "candidate_id": candidate_id,
                "fp_shep_online_score": 3.0 - candidate_id,
                "preview_task_progress": 1.0,
                "preview_min_clearance": 1.0,
                "preview_feature_valid_mask": [True, True, True],
            }
            for candidate_id in range(3)
        ],
        "all_candidate_interaction_records": [
            {
                "candidate_id": candidate_id,
                "risky": False,
                "maximum_risk_duration_s": 0.0,
                "minimum_predicted_separation_m": 1.0,
            }
            for candidate_id in range(3)
        ],
    }
    return {
        "plan": {
            "selection_source": "gat_stage1",
            "references": [points[selected]],
            "available": [True],
            "candidate_records": [record],
            "selection_plan_hash": "before",
        }
    }


def _controller() -> DirectionContinuityTieBreak:
    return DirectionContinuityTieBreak(
        DirectionContinuityTieBreakConfig(
            near_tie_probability_margin=0.1,
            warning_safety_margin_m=0.35,
        ),
        num_agents=1,
    )


def _env() -> SimpleNamespace:
    return SimpleNamespace(
        dynamics=[SimpleNamespace(p=[0.0, 0.0, 0.0])],
        goals=[[10.0, 0.0, 0.0]],
    )


def _prime(controller: DirectionContinuityTieBreak) -> None:
    env = _env()
    controller.apply_plan(
        _selection(0, [0.3, 0.6, 0.9], 0.01),
        env=env,
        agent_ids=[0],
        event_types={0: "INITIAL_SELECTION"},
        active_safety_margins_m={0: None},
        initial=True,
        step=0,
    )
    controller.apply_plan(
        _selection(1, [0.3, 0.6, 0.9], 0.01),
        env=env,
        agent_ids=[0],
        event_types={0: "NORMAL_REPROPOSAL"},
        active_safety_margins_m={0: 1.0},
        initial=False,
        step=10,
    )


def test_near_tie_reversal_uses_safe_ranked_continuation() -> None:
    controller = _controller()
    _prime(controller)
    selection = _selection(0, [-0.2, 0.1, 0.9], 0.01)
    # Candidate 2 is the GAT rank-2 continuation.
    selection["plan"]["candidate_records"][0]["class_probabilities"] = [
        0.01,
        0.55,
        0.08,
        0.36,
    ]
    result = controller.apply_plan(
        selection,
        env=_env(),
        agent_ids=[0],
        event_types={0: "NORMAL_REPROPOSAL"},
        active_safety_margins_m={0: 1.0},
        initial=False,
        step=20,
    )
    record = result["plan"]["candidate_records"][0]
    assert record["selected_candidate_id"] == 2
    assert record["dctb_replaced"] is True
    assert result["plan"]["references"][0] == record["candidate_world_points"][2]
    assert record["temporary_reference"] == record["candidate_world_points"][2]


def test_high_confidence_and_safety_critical_states_preserve_gat_top1() -> None:
    for margin, safety_margin, event, expected_reason in (
        (0.2, 1.0, "NORMAL_REPROPOSAL", "GAT_NOT_NEAR_TIE"),
        (0.01, 0.2, "NORMAL_REPROPOSAL", "EXISTING_SAFETY_CRITICAL_BYPASS"),
        (0.01, 1.0, "EMERGENCY_REPROPOSAL", "EXISTING_SAFETY_CRITICAL_BYPASS"),
    ):
        controller = _controller()
        _prime(controller)
        selection = _selection(0, [-0.2, 0.1, 0.9], margin)
        result = controller.apply_plan(
            selection,
            env=_env(),
            agent_ids=[0],
            event_types={0: event},
            active_safety_margins_m={0: safety_margin},
            initial=False,
            step=20,
        )
        record = result["plan"]["candidate_records"][0]
        assert record["selected_candidate_id"] == 0
        assert record["dctb_replaced"] is False
        assert record["dctb_reason"] == expected_reason


def test_terminal_or_null_commitment_resets_direction_history() -> None:
    controller = _controller()
    _prime(controller)
    selection = _selection(0, [-0.2, 0.1, 0.9], 0.01)
    selection["plan"]["available"] = [False]
    selection["plan"]["candidate_records"][0]["selected_candidate_id"] = None
    controller.apply_plan(
        selection,
        env=_env(),
        agent_ids=[0],
        event_types={0: "NORMAL_REPROPOSAL"},
        active_safety_margins_m={0: 1.0},
        initial=False,
        step=20,
    )
    assert controller.trace_rows()[-1]["reason"] == "SEMANTIC_HISTORY_RESET"
