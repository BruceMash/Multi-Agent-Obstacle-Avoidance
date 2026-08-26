from __future__ import annotations

import itertools

import numpy as np

from planning.gat.interaction_aware_supervision import (
    NEUTRAL,
    RISKY,
    SAFE,
    CandidateRiskDescriptor,
    minimum_l2_risk_consistency_projection,
    project_v1_soft_target,
)


def _descriptors(statuses: list[str], candidate_ids: list[int] | None = None):
    ids = candidate_ids if candidate_ids is not None else list(range(len(statuses)))
    return tuple(
        CandidateRiskDescriptor(index, candidate_id, status, 1.0, 0.0, 1)
        for index, (candidate_id, status) in enumerate(zip(ids, statuses))
    )


def test_already_consistent_target_is_bit_exact() -> None:
    target = (0.2, 0.4, 0.3, 0.1)
    result = project_v1_soft_target(target, _descriptors([SAFE, SAFE, RISKY]))
    assert result.target == target
    assert not result.changed


def test_minimum_l2_pooling_preserves_null_neutral_and_mass() -> None:
    target = (0.2, 0.08, 0.48, 0.24)
    result = project_v1_soft_target(target, _descriptors([SAFE, RISKY, NEUTRAL]))
    assert result.target[0] == target[0]
    assert result.target[3] == target[3]
    assert np.allclose(result.target, (0.2, 0.28, 0.28, 0.24), atol=1e-15)
    assert np.isclose(sum(result.target), 1.0)


def test_no_both_safe_and_risky_is_entire_target_bit_exact() -> None:
    target = (0.1, 0.2, 0.3, 0.4)
    assert project_v1_soft_target(target, _descriptors([SAFE, SAFE, NEUTRAL])).target == target
    assert project_v1_soft_target(target, _descriptors([RISKY, RISKY, NEUTRAL])).target == target


def test_zero_proposal_mass_is_entire_target_bit_exact() -> None:
    target = (1.0, 0.0, 0.0)
    assert project_v1_soft_target(target, _descriptors([SAFE, RISKY])).target == target


def test_projection_is_exactly_order_invariant_after_mapping_candidate_ids() -> None:
    ids = np.asarray([17, 4, 91, 2, 33])
    q = np.asarray([0.04, 0.32, 0.28, 0.21, 0.15])
    statuses = np.asarray([SAFE, RISKY, RISKY, SAFE, NEUTRAL], dtype=object)
    baseline, _, _, _ = minimum_l2_risk_consistency_projection(q, statuses, ids)
    baseline_by_id = dict(zip(ids.tolist(), baseline.tolist()))
    rng = np.random.default_rng(20260818)
    for _ in range(100):
        order = rng.permutation(len(ids))
        projected, _, _, _ = minimum_l2_risk_consistency_projection(
            q[order], statuses[order], ids[order]
        )
        assert {int(ids[index]): projected[position] for position, index in enumerate(order)} == baseline_by_id


def test_projection_matches_brute_force_grid_for_three_classes() -> None:
    q = np.asarray([0.08, 0.57, 0.35])
    statuses = [SAFE, RISKY, NEUTRAL]
    projected, _, _, _ = minimum_l2_risk_consistency_projection(q, statuses, [0, 1, 2])
    candidates = []
    for safe_value in np.linspace(0.0, 0.65, 1301):
        risky_value = 0.65 - safe_value
        if safe_value >= risky_value:
            vector = np.asarray([safe_value, risky_value, 0.35])
            candidates.append(float(np.sum(np.square(vector - q))))
    assert float(np.sum(np.square(projected - q))) <= min(candidates) + 1e-6


def test_all_small_status_patterns_satisfy_constraint_and_mass() -> None:
    rng = np.random.default_rng(7)
    for size in range(2, 7):
        for statuses in itertools.product((SAFE, RISKY, NEUTRAL), repeat=size):
            if SAFE not in statuses or RISKY not in statuses:
                continue
            q = rng.dirichlet(np.ones(size))
            projected, _, _, _ = minimum_l2_risk_consistency_projection(q, statuses, range(size))
            safe = projected[np.asarray(statuses) == SAFE]
            risky = projected[np.asarray(statuses) == RISKY]
            assert np.isclose(projected.sum(), 1.0)
            assert safe.min() + 1e-12 >= risky.max()
