from types import SimpleNamespace

import numpy as np

from planning.transient_aware_candidate_veto import TACVConfig, select_tacv_candidate


def preview(candidate_id: int, accelerations: list[list[float]], clearance: float):
    trajectory = SimpleNamespace(accelerations=np.asarray(accelerations, dtype=float))
    execution = SimpleNamespace(trajectory=trajectory, min_clearance=float(clearance))
    return SimpleNamespace(candidate_id=candidate_id, score=float(-candidate_id), preview=execution)


def interactions(risky_ids=()):
    return [
        {
            "candidate_id": candidate_id,
            "interaction_edge_count": int(candidate_id in risky_ids),
            "minimum_predicted_separation_m": 0.5 if candidate_id in risky_ids else None,
            "maximum_risk_duration_s": 0.1 if candidate_id in risky_ids else 0.0,
            "risky": candidate_id in risky_ids,
        }
        for candidate_id in range(3)
    ]


def config(threshold=1.0, reduction=0.2):
    return TACVConfig(
        dt_s=0.1,
        activation_threshold=threshold,
        minimum_relative_reduction=reduction,
        maximum_gat_candidate_rank=3,
        apply_to_initial_selection=False,
    )


def records(clearances=(1.0, 1.0, 1.0)):
    return [
        preview(0, [[3.0, 0.0, 0.0]] * 4, clearances[0]),
        preview(1, [[0.2, 0.0, 0.0]] * 4, clearances[1]),
        preview(2, [[0.1, 0.0, 0.0]] * 4, clearances[2]),
    ]


def test_initial_selection_is_unchanged():
    decision = select_tacv_candidate(
        original_selected_candidate_id=0,
        class_logits=[-2.0, 3.0, 2.0, 1.0],
        preview_records=records(),
        interaction_records=interactions(),
        current_applied_acceleration=np.zeros(3),
        event_type="INITIAL_SELECTION",
        config=config(),
    )
    assert not decision["activated"]
    assert not decision["replaced"]
    assert decision["effective_selected_candidate_id"] == 0


def test_first_gat_ranked_admissible_material_alternative_is_selected():
    decision = select_tacv_candidate(
        original_selected_candidate_id=0,
        class_logits=[-2.0, 3.0, 2.0, 1.0],
        preview_records=records(),
        interaction_records=interactions(),
        current_applied_acceleration=np.zeros(3),
        event_type="NORMAL_REPROPOSAL",
        config=config(),
    )
    assert decision["activated"]
    assert decision["replaced"]
    assert decision["effective_selected_candidate_id"] == 1
    assert decision["replacement_gat_candidate_rank"] == 2


def test_clearance_inferior_alternative_is_rejected():
    decision = select_tacv_candidate(
        original_selected_candidate_id=0,
        class_logits=[-2.0, 3.0, 2.0, 1.0],
        preview_records=records((1.0, 0.9, 0.8)),
        interaction_records=interactions(),
        current_applied_acceleration=np.zeros(3),
        event_type="NORMAL_REPROPOSAL",
        config=config(),
    )
    assert decision["activated"]
    assert not decision["replaced"]
    assert decision["effective_selected_candidate_id"] == 0


def test_safe_original_cannot_be_replaced_by_risky_candidate():
    decision = select_tacv_candidate(
        original_selected_candidate_id=0,
        class_logits=[-2.0, 3.0, 2.0, 1.0],
        preview_records=records(),
        interaction_records=interactions((1, 2)),
        current_applied_acceleration=np.zeros(3),
        event_type="NORMAL_REPROPOSAL",
        config=config(),
    )
    assert decision["activated"]
    assert not decision["replaced"]


def test_below_activation_threshold_is_exact_noop():
    decision = select_tacv_candidate(
        original_selected_candidate_id=0,
        class_logits=[-2.0, 3.0, 2.0, 1.0],
        preview_records=records(),
        interaction_records=interactions(),
        current_applied_acceleration=np.zeros(3),
        event_type="NORMAL_REPROPOSAL",
        config=config(threshold=1.0e12),
    )
    assert not decision["activated"]
    assert not decision["replaced"]
    assert decision["effective_selected_candidate_id"] == 0
