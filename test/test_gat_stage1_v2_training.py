from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from planning.gat.stage1_v2_training import (
    Stage1V2Example,
    V2TrainingRunResult,
    classify_gain,
    examples_for_target,
    load_initialized_model,
    load_v2_examples,
    metrics_for_scores,
    model_state_hash,
    smoke_gate_result_v2,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "training" / "gat_stage1_v2.json"
V1_CONFIG_PATH = REPO_ROOT / "configs" / "training" / "gat_stage1.json"


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _examples() -> tuple[list[Stage1V2Example], dict]:
    config = _config()
    dataset = REPO_ROOT / config["dataset_path"]
    if not dataset.exists():
        pytest.skip("frozen V2 artifact is unavailable")
    interaction = json.loads(V1_CONFIG_PATH.read_text("utf-8"))[
        "interaction_diagnostic"
    ]
    return load_v2_examples(
        dataset,
        split_config=config["dataset_split"],
        interaction_config=interaction,
        expected_h_preview=4,
        expected_temperature=0.25,
    )


def test_v2_loader_uses_effective_split_and_preserves_source_split() -> None:
    examples, audit = _examples()
    assert audit["effective_split_graph_counts"] == {
        "train": 342,
        "validation": 48,
        "test": 102,
    }
    assert audit["source_split_graph_counts"] == {
        "train": 288,
        "validation": 102,
        "test": 102,
    }
    assert audit["effective_split_used_for_training"] is True
    assert audit["source_split_used_for_training"] is False
    assert all(
        item.split
        == ("train" if item.seed <= 6 else "validation" if item.seed == 7 else "test")
        for item in examples
    )


def test_v2_loader_keeps_real_variable_class_counts_and_targets() -> None:
    examples, audit = _examples()
    assert audit["variable_K"] is True
    assert audit["minimum_class_count"] == 1
    assert audit["maximum_class_count"] == 11
    for item in examples:
        assert item.class_count == item.proposal_count + 1
        assert len(item.soft_target) == item.class_count
        assert np.isclose(sum(item.soft_target), 1.0, rtol=0.0, atol=1.0e-10)
        assert len(item.v1_scalar_h6_soft_target) == item.class_count
        assert len(item.v1_historical_h6_soft_target) == item.class_count
        assert len(item.v2_long_horizon_tier) == item.class_count


def test_cross_target_view_does_not_modify_graph_or_original_target() -> None:
    examples, _ = _examples()
    original = examples[0]
    graph_identity = id(original.graph)
    historical = examples_for_target([original], "v1_historical_h6")[0]
    assert id(historical.graph) == graph_identity
    assert original.soft_target == original.v2_long_horizon_soft_target
    assert historical.soft_target == original.v1_historical_h6_soft_target
    assert historical.target_quality == original.v1_historical_h6_utility
    assert original.soft_target == original.v2_long_horizon_soft_target


def test_v1_checkpoint_initialization_is_strict_and_exact() -> None:
    config = _config()
    checkpoint = REPO_ROOT / config["initial_checkpoint"]
    if not checkpoint.exists():
        pytest.skip("frozen V1 checkpoint is unavailable")
    model, payload, source_hash = load_initialized_model(
        checkpoint, config, torch.device("cpu")
    )
    assert model_state_hash(model.state_dict()) == source_hash
    assert model_state_hash(payload["model_state_dict"]) == source_hash
    assert len(model.state_dict()) == 68


def test_same_score_vectors_can_be_evaluated_against_all_frozen_targets() -> None:
    examples, _ = _examples()
    subset = examples[-12:]
    scores = [np.arange(item.class_count, dtype=float) for item in subset]
    for target_name in [
        "v1_scalar_h6",
        "v1_historical_h6",
        "v2_long_horizon",
    ]:
        target_examples = examples_for_target(subset, target_name)
        metrics = metrics_for_scores(target_examples, scores)
        assert metrics["graph_count"] == len(subset)
        assert np.isfinite(metrics["loss"])
        assert 0.0 <= metrics["top1_accuracy"] <= 1.0


def test_smoke_gate_enforces_all_frozen_checks(tmp_path: Path) -> None:
    good_metrics = {
        "top1_accuracy": 0.5,
        "empirical_random_top1_accuracy": 0.2,
        "null_prediction_rate": 0.4,
        "maximum_fixed_proposal_rank_prediction_rate": 0.4,
    }
    result = V2TrainingRunResult(
        optimization_seed=0,
        best_epoch=1,
        epochs_completed=1,
        early_stopped=False,
        best_validation_loss=1.0,
        best_validation_metrics=good_metrics,
        best_checkpoint=tmp_path / "best.pt",
        last_checkpoint=tmp_path / "last.pt",
        history=(
            {
                "train_loss": 1.1,
                "validation_loss": 1.0,
            },
        ),
        initialization_model_hash="hash",
        fresh_optimizer_state_at_start=True,
        gradients_finite=True,
        maximum_gradient_norm_pre_clip=1.0,
        runtime_s=0.1,
    )
    smoke = smoke_gate_result_v2(result, _config()["smoke_gate"])
    assert smoke["status"] == "PASS"
    failed = replace(result, gradients_finite=False)
    assert smoke_gate_result_v2(failed, _config()["smoke_gate"])["status"] == "FAIL"


def test_decision_gain_thresholds_are_frozen() -> None:
    assert classify_gain(0.05, yes_threshold=0.05) == "YES"
    assert classify_gain(0.01, yes_threshold=0.05) == "WEAK"
    assert classify_gain(0.0, yes_threshold=0.05) == "NO"
    assert classify_gain(-0.01, yes_threshold=0.05) == "NO"
