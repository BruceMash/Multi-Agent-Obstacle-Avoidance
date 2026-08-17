from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from planning.gat.stage1_training import (
    empirical_random_metrics,
    load_stage1_examples,
    masked_soft_target_cross_entropy,
    ragged_soft_target_cross_entropy,
    resolve_split,
    spearman_coefficient,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "training" / "gat_stage1.json"


def _config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_masked_soft_target_loss_excludes_padding_before_softmax():
    logits_a = torch.tensor([[1.0, 2.0, -50.0], [3.0, 0.0, 4.0]])
    logits_b = torch.tensor([[1.0, 2.0, 1.0e9], [3.0, 0.0, -1.0e9]])
    targets = torch.tensor([[0.25, 0.75, 0.0], [0.6, 0.4, 0.0]])
    mask = torch.tensor([[True, True, False], [True, True, False]])
    loss_a = masked_soft_target_cross_entropy(logits_a, targets, mask)
    loss_b = masked_soft_target_cross_entropy(logits_b, targets, mask)
    torch.testing.assert_close(loss_a, loss_b)


def test_masked_soft_target_loss_rejects_target_mass_on_invalid_class():
    logits = torch.zeros((1, 3))
    targets = torch.tensor([[0.4, 0.5, 0.1]])
    mask = torch.tensor([[True, True, False]])
    with pytest.raises(ValueError, match="zero target mass"):
        masked_soft_target_cross_entropy(logits, targets, mask)


def test_ragged_loss_uses_each_graph_real_class_count_only():
    logits = torch.tensor([1.0, 2.0, 0.5, -0.5, 3.0])
    ptr = torch.tensor([0, 2, 5])
    targets = [torch.tensor([0.25, 0.75]), torch.tensor([0.2, 0.3, 0.5])]
    actual = ragged_soft_target_cross_entropy(logits, ptr, targets)
    expected = torch.stack(
        [
            -(targets[0] * torch.log_softmax(logits[:2], dim=0)).sum(),
            -(targets[1] * torch.log_softmax(logits[2:], dim=0)).sum(),
        ]
    ).mean()
    torch.testing.assert_close(actual, expected)


def test_ragged_loss_rejects_nonexistent_class_target():
    with pytest.raises(ValueError, match="real class count"):
        ragged_soft_target_cross_entropy(
            torch.zeros(3), torch.tensor([0, 2, 3]), [[0.5, 0.5, 0.0], [1.0]]
        )


def test_empirical_random_metrics_use_each_graph_actual_class_count():
    metrics = empirical_random_metrics([1, 2, 4])
    assert metrics["top1_accuracy"] == pytest.approx((1.0 + 0.5 + 0.25) / 3.0)
    assert metrics["top3_accuracy"] == pytest.approx((1.0 + 1.0 + 0.75) / 3.0)
    expected_mrr = (1.0 + (1.0 + 0.5) / 2.0 + (1.0 + 0.5 + 1 / 3 + 0.25) / 4.0) / 3.0
    assert metrics["mrr"] == pytest.approx(expected_mrr)


def test_spearman_coefficient_uses_average_tie_ranks():
    assert spearman_coefficient([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]) == pytest.approx(1.0)
    assert spearman_coefficient([1.0, 1.0, 3.0], [10.0, 10.0, 30.0]) == pytest.approx(1.0)
    assert np.isnan(spearman_coefficient([1.0, 1.0], [2.0, 3.0]))


def test_requested_seed_split_is_disjoint_and_exhaustive():
    split_config = _config()["dataset_split"]
    assert [resolve_split(seed, split_config) for seed in range(10)] == [
        "train",
        "train",
        "train",
        "train",
        "train",
        "train",
        "train",
        "validation",
        "test",
        "test",
    ]


def test_stage1_loader_preserves_variable_k_and_group_split():
    config = _config()
    examples, audit, split_rows = load_stage1_examples(
        ROOT / config["dataset_dir"],
        split_config=config["dataset_split"],
        supervision_config=config["supervision"],
        interaction_config=config["interaction_diagnostic"],
    )
    assert len(examples) == 492
    assert audit["split_graph_counts"] == {
        "test": 102,
        "train": 342,
        "validation": 48,
    }
    assert audit["group_leakage_count"] == 0
    assert audit["variable_K"] is True
    assert audit["padding_used_in_training"] is False
    assert {item.class_count for item in examples} == set(range(1, 12))
    group_splits = {}
    for row in split_rows:
        previous = group_splits.setdefault(row["state_group_id"], row["split"])
        assert previous == row["split"]


def test_interaction_diagnostic_cannot_select_checkpoint():
    config = _config()
    diagnostic = config["interaction_diagnostic"]
    assert diagnostic["used_for_checkpoint_selection"] is False
    assert diagnostic["used_for_early_stopping"] is False
    assert diagnostic["new_online_feature_added"] is False


def test_metric_definitions_are_frozen_before_training():
    definitions = _config()["offline_metric_definitions"]
    assert definitions["definitions_frozen_before_training"] is True
    assert definitions["reference_class"] == "argmax_soft_target"
    assert definitions["top3"] == "reference_class_in_predicted_top_min_3_C"
    assert definitions["spearman_excludes_null"] is True
    assert definitions["spearman_minimum_proposal_count"] == 2
