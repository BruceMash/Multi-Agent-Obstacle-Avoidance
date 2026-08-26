from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "Multi-agent_Algo_lib/scripts/audit_gat_v1_joint_compatibility_recoverability.py"
SPEC = importlib.util.spec_from_file_location("joint_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_pair_compatibility_uses_strict_frozen_rule() -> None:
    left = np.zeros((4, 3), dtype=float)
    safe = np.full((4, 3), [0.6, 0.0, 0.0], dtype=float)
    risky = np.full((4, 3), [0.59, 0.0, 0.0], dtype=float)
    exact = audit.pair_compatibility(left, safe, d_safe=0.6, dt=0.1)
    below = audit.pair_compatibility(left, risky, d_safe=0.6, dt=0.1)
    assert exact["compatible"] is True
    assert exact["T_risk"] == 0.0
    assert below["compatible"] is False
    assert below["T_risk"] == 0.4


def test_minimum_deviation_priority_is_lexicographic() -> None:
    baseline = (0, 0, 0)
    ranks = [{0: 1, 1: 2, 2: 3} for _ in range(3)]
    logits = [[3.0, 2.0, 1.0] for _ in range(3)]
    one_change = (1, 0, 0)
    two_change = (1, 1, 0)
    assert audit.combination_key(one_change, baseline, ranks, logits) < audit.combination_key(
        two_change, baseline, ranks, logits
    )


def test_h4_signal_gate() -> None:
    gate = {
        "strong_minimum_recall": 0.7,
        "strong_maximum_success_false_veto_rate": 0.15,
        "moderate_minimum_recall": 0.5,
        "moderate_maximum_success_false_veto_rate": 0.25,
        "weak_requires_true_positive_count": 1,
    }
    assert audit.classify_h4_signal(
        {"recall": 0.75, "successful_false_veto_rate": 0.1, "TP": 3}, gate
    ) == "STRONG"
    assert audit.classify_h4_signal(
        {"recall": 0.5, "successful_false_veto_rate": 0.2, "TP": 2}, gate
    ) == "MODERATE"
    assert audit.classify_h4_signal(
        {"recall": 0.2, "successful_false_veto_rate": 0.4, "TP": 1}, gate
    ) == "WEAK"
    assert audit.classify_h4_signal(
        {"recall": 0.0, "successful_false_veto_rate": 0.0, "TP": 0}, gate
    ) == "NO"


def test_theory_gate_requires_all_conditions() -> None:
    gate = {
        "allowed_h4_signals": ["STRONG", "MODERATE"],
        "minimum_local_alternative_failures": 6,
        "maximum_success_false_veto_rate": 0.25,
        "allowed_promises": ["HIGH", "MODERATE"],
    }
    assert audit.theory_extension_gate(
        signal="MODERATE",
        local_failure_count=6,
        success_false_veto_rate=0.25,
        promise="MODERATE",
        gate=gate,
    )
    assert not audit.theory_extension_gate(
        signal="WEAK",
        local_failure_count=15,
        success_false_veto_rate=0.0,
        promise="LOW",
        gate=gate,
    )


def test_preview_runtime_is_the_only_excluded_reproduction_field() -> None:
    left = [{"candidate_id": 0, "score": 1.0, "preview_runtime_ms": 4.0}]
    right = [{"candidate_id": 0, "score": 1.0, "preview_runtime_ms": 9.0}]
    assert audit.strip_preview_runtime(left) == audit.strip_preview_runtime(right)
    assert left != right


def test_float32_reproduction_requires_exact_ranking() -> None:
    result = audit.float_vector_reproduction(
        [1.0, 0.5, -0.25],
        [1.0 + 1.2e-7, 0.5, -0.25],
        atol=1e-6,
        rtol=1e-6,
    )
    assert result["bitwise_exact"] is False
    assert result["numerically_reproduced"] is True
    assert result["complete_ranking_exact"] is True


def test_expected_rows_follow_actual_top_k_counts() -> None:
    rows = [
        {"scenario": "x", "seed": 1, "agent_id": 0, "candidate_count_reproduced": 10},
        {"scenario": "x", "seed": 1, "agent_id": 1, "candidate_count_reproduced": 5},
        {"scenario": "x", "seed": 1, "agent_id": 2, "candidate_count_reproduced": 3},
    ]
    counts = audit.expected_reproduction_row_counts(rows)
    assert counts["h4_candidate_trajectory_rows"] == 11 + 6 + 4
    assert counts["pairwise_compatibility_rows"] == 11 * 6 + 11 * 4 + 6 * 4
