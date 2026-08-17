from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("torch_geometric")

from planning.gat.stage1_training import load_model_checkpoint, load_stage1_examples
from planning.gat.stage2_coordination_training import (
    PAIR_INDICES,
    ablate_joint_groups,
    build_joint_state_groups,
    clone_with_edge_ablation,
    compute_interaction_metrics,
    gradient_scale_audit,
    iter_joint_batches,
    joint_batch_losses,
    null_escape_audit,
    select_lambda_candidate,
    split_joint_groups,
    train_one_stage2_seed,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def frozen_data():
    stage1 = json.loads(
        (ROOT / "configs" / "training" / "gat_stage1.json").read_text(
            encoding="utf-8"
        )
    )
    stage2 = json.loads(
        (ROOT / "configs" / "training" / "gat_stage2_coordination.json").read_text(
            encoding="utf-8"
        )
    )
    examples, _, _ = load_stage1_examples(
        ROOT / stage1["dataset_dir"],
        split_config=stage1["dataset_split"],
        supervision_config=stage1["supervision"],
        interaction_config=stage1["interaction_diagnostic"],
    )
    groups = build_joint_state_groups(examples, required_horizon=4)
    split = split_joint_groups(groups)
    model = load_model_checkpoint(
        ROOT / stage2["stage1_checkpoint"], stage1, torch.device("cpu")
    )
    return stage1, stage2, groups, split, model


def _positive_group(groups):
    return next(
        group
        for group in groups
        if any(np.any(pair.overlap > 0.0) for pair in group.pair_matrices)
    )


def test_stage2_config_freezes_null_pair_and_batch_semantics(frozen_data):
    _, stage2, _, _, _ = frozen_data
    assert stage2["overlap"]["class_domain"] == "proposal_proposal_only"
    assert stage2["overlap"]["probability_source"] == "full_null_plus_K_softmax"
    assert stage2["overlap"]["proposal_probability_renormalization"] is False
    assert stage2["overlap"]["uav_pair_order"] == "unordered_i_lt_j"
    assert stage2["batching"]["regular_joint_states_per_batch"] == 10
    assert stage2["batching"]["regular_ego_graphs_per_batch"] == 30
    assert stage2["initialization"]["optimizer"] == "fresh_AdamW_per_optimization_seed"


def test_joint_groups_are_complete_and_split_without_leakage(frozen_data):
    _, _, groups, split, _ = frozen_data
    assert len(groups) == 164
    assert {name: len(value) for name, value in split.items()} == {
        "test": 34,
        "train": 114,
        "validation": 16,
    }
    assert all([item.ego_agent_id for item in group.examples] == [0, 1, 2] for group in groups)
    assert len({group.state_group_id for group in groups}) == len(groups)


def test_pair_matrix_matches_exact_h4_formula(frozen_data):
    _, _, groups, _, _ = frozen_data
    group = _positive_group(groups)
    pair_index = next(
        index for index, pair in enumerate(group.pair_matrices) if np.any(pair.overlap > 0)
    )
    left_index, right_index = PAIR_INDICES[pair_index]
    pair = group.pair_matrices[pair_index]
    cell = np.argwhere(pair.overlap > 0)[0]
    left_trajectory = np.asarray(
        group.examples[left_index].graph.candidate_preview_positions[int(cell[0])]
    )
    right_trajectory = np.asarray(
        group.examples[right_index].graph.candidate_preview_positions[int(cell[1])]
    )
    distances = np.linalg.norm(left_trajectory - right_trajectory, axis=1)
    expected = np.mean(np.maximum(pair.d_safe - distances, 0.0) / pair.d_safe)
    assert pair.horizon_steps == 4
    assert pair.overlap[int(cell[0]), int(cell[1])] == pytest.approx(expected)
    assert pair.minimum_distance[int(cell[0]), int(cell[1])] == pytest.approx(
        distances.min()
    )


def test_batch_budget_keeps_three_uav_groups_atomic(frozen_data):
    _, _, _, split, _ = frozen_data
    batches = list(
        iter_joint_batches(split["train"], graph_budget=32, shuffle_seed=123)
    )
    assert all(len(batch) <= 10 for batch in batches)
    assert all(3 * len(batch) <= 30 for batch in batches)
    assert sum(len(batch) for batch in batches) == len(split["train"])
    assert len({group.state_group_id for batch in batches for group in batch}) == 114


def test_overlap_loss_uses_three_unordered_pairs_and_raw_proposal_mass(frozen_data):
    _, _, groups, _, model = frozen_data
    group = _positive_group(groups)
    selection_loss, overlap_loss, output, examples = joint_batch_losses(
        model, [group], device=torch.device("cpu")
    )
    assert torch.isfinite(selection_loss)
    manual_terms = []
    normalized_terms = []
    for pair_index, (left_index, right_index) in enumerate(PAIR_INDICES):
        left = output.probabilities_for_graph(left_index)[1:]
        right = output.probabilities_for_graph(right_index)[1:]
        matrix = torch.tensor(group.pair_matrices[pair_index].overlap, dtype=left.dtype)
        raw = left @ matrix @ right if matrix.numel() else left.new_zeros(())
        manual_terms.append(raw)
        if left.numel() and right.numel():
            normalized_terms.append((left / left.sum()) @ matrix @ (right / right.sum()))
        else:
            normalized_terms.append(raw)
    manual = torch.stack(manual_terms).mean()
    normalized = torch.stack(normalized_terms).mean()
    assert len(examples) == 3
    assert len(manual_terms) == 3
    assert float(overlap_loss.detach()) == pytest.approx(
        float(manual.detach()), abs=1e-8
    )
    assert float(overlap_loss.detach()) <= float(normalized.detach()) + 1e-12


def test_interaction_metrics_report_null_without_null_trajectory(frozen_data):
    _, _, groups, _, model = frozen_data
    group = _positive_group(groups)
    with torch.no_grad():
        _, _, output, examples = joint_batch_losses(model, [group], device=torch.device("cpu"))
    scores = {
        item.sample_id: output.logits_for_graph(index).detach().numpy()
        for index, item in enumerate(examples)
    }
    metrics, pairs = compute_interaction_metrics([group], scores)
    assert metrics["proposal_probability_renormalized_for_loss"] is False
    assert metrics["null_trajectory_used"] is False
    assert metrics["unordered_uav_pair_count"] == 3
    assert len(pairs) == 3
    assert 0.0 <= metrics["mean_null_probability"] <= 1.0


def test_null_escape_requires_rate_or_probability_and_mass_attribution():
    config = {
        "null_top1_increase_threshold": 0.05,
        "mean_null_probability_increase_threshold": 0.05,
        "mass_transfer_primary_fraction_threshold": 0.5,
    }
    before_metrics = {"null_top1_selection_rate": 0.2, "mean_null_probability": 0.2}
    after_metrics = {"null_top1_selection_rate": 0.3, "mean_null_probability": 0.3}
    before_pairs = [
        {
            "state_group_id": "g",
            "left_agent_id": 0,
            "right_agent_id": 1,
            "expected_overlap": 0.08,
            "proposal_mass_product": 0.8,
            "conditional_proposal_overlap_diagnostic": 0.1,
        }
    ]
    after_pairs = [
        {
            "state_group_id": "g",
            "left_agent_id": 0,
            "right_agent_id": 1,
            "expected_overlap": 0.04,
            "proposal_mass_product": 0.4,
            "conditional_proposal_overlap_diagnostic": 0.1,
        }
    ]
    audit = null_escape_audit(
        before_metrics, after_metrics, before_pairs, after_pairs, config
    )
    assert audit["NULL_ESCAPE_DETECTED"] == "YES"
    assert audit["mass_transfer_is_primary"] is True


def test_edge_ablation_is_defensive_and_preserves_pair_matrices(frozen_data):
    _, _, groups, _, _ = frozen_data
    group = _positive_group(groups)
    original = group.examples[0].graph["align", "spatiotemporal", "proposal"]
    original_attr = original.edge_attr.clone()
    original_index = original.edge_index.clone()
    zero = clone_with_edge_ablation(group.examples[0], "zero_st_edge")
    removed_groups = ablate_joint_groups([group], "remove_align_message")
    assert torch.equal(original.edge_attr, original_attr)
    assert torch.equal(original.edge_index, original_index)
    assert torch.count_nonzero(
        zero.graph["align", "spatiotemporal", "proposal"].edge_attr
    ) == 0
    assert (
        removed_groups[0]
        .examples[0]
        .graph["align", "spatiotemporal", "proposal"]
        .edge_index.shape[1]
        == 0
    )
    assert removed_groups[0].pair_matrices is group.pair_matrices


def test_gradient_scale_audit_uses_independent_nonzero_gradients(frozen_data):
    _, _, groups, _, model = frozen_data
    positive_groups = [
        group
        for group in groups
        if any(np.any(pair.overlap > 0.0) for pair in group.pair_matrices)
    ][:2]
    rows = gradient_scale_audit(
        model,
        positive_groups,
        [0.1],
        graph_budget=32,
        device=torch.device("cpu"),
    )
    row = rows[0]
    assert row["separate_backward_passes"] is True
    assert row["zero_grad_between_measurements"] is True
    assert row["g_select"] > 0.0
    assert row["g_overlap_weighted"] > 0.0
    assert row["OVERLAP_GRADIENT_RATIO"] > 0.0
    assert row["overlap_spatiotemporal_edge_encoder_gradient_norm"] > 0.0
    assert row["overlap_align_message_gradient_norm"] > 0.0
    assert all(parameter.grad is None for parameter in model.parameters())


def test_no_completion_or_closed_loop_capability_enabled(frozen_data):
    _, stage2, _, _, _ = frozen_data
    exclusions = stage2["strict_exclusions"]
    assert exclusions["completion_loss"] is False
    assert exclusions["closed_loop_evaluation"] is False
    assert exclusions["new_dataset"] is False
    assert exclusions["gat_architecture_modification"] is False


def test_lambda_decision_uses_validation_total_loss_field():
    rows = [
        {
            "lambda_overlap": value,
            "validation_total_loss": 1.0,
            "top1_accuracy": 0.7,
            "expected_overlap": expected,
            "risky_selection_rate": 0.0,
        }
        for value, expected in ((0.05, 0.0095), (0.1, 0.009), (0.2, 0.008))
    ]
    config = {
        "numeric_improvement_tolerance": 1e-8,
        "maximum_validation_top1_regression": 0.05,
        "yes_minimum_expected_overlap_absolute_reduction": 1e-5,
        "yes_minimum_expected_overlap_relative_reduction": 0.05,
        "weak_minimum_consistent_candidate_count": 2,
    }
    selected, signal, _ = select_lambda_candidate(
        rows,
        {"top1_accuracy": 0.7, "expected_overlap": 0.01},
        config,
    )
    assert signal == "YES"
    assert selected["lambda_overlap"] == 0.2


def test_lambda_tie_within_frozen_tolerance_prefers_smaller_value():
    rows = [
        {
            "lambda_overlap": value,
            "validation_total_loss": 1.0,
            "top1_accuracy": 0.7,
            "expected_overlap": 0.01 + delta,
            "risky_selection_rate": 0.0,
        }
        for value, delta in ((0.05, 5e-12), (0.1, 1e-12), (0.2, 3e-12))
    ]
    config = {
        "numeric_improvement_tolerance": 1e-8,
        "maximum_validation_top1_regression": 0.05,
        "yes_minimum_expected_overlap_absolute_reduction": 1e-5,
        "yes_minimum_expected_overlap_relative_reduction": 0.05,
        "weak_minimum_consistent_candidate_count": 2,
    }
    selected, signal, _ = select_lambda_candidate(
        rows,
        {"top1_accuracy": 0.7, "expected_overlap": 0.01},
        config,
    )
    assert signal == "NO"
    assert selected["lambda_overlap"] == 0.05


def test_stage1_initialization_remains_epoch_zero_checkpoint(frozen_data, tmp_path):
    stage1, stage2, _, split, _ = frozen_data
    local = json.loads(json.dumps(stage2))
    local["training"]["learning_rate"] = 0.0
    local["training"]["max_epochs"] = 1
    local["training"]["early_stopping_patience"] = 1
    result = train_one_stage2_seed(
        local,
        stage1,
        split["train"][:1],
        split["validation"][:1],
        stage1_checkpoint=ROOT / stage2["stage1_checkpoint"],
        lambda_overlap=0.1,
        optimization_seed=0,
        checkpoint_dir=tmp_path,
    )
    payload = torch.load(result.best_checkpoint, map_location="cpu", weights_only=False)
    assert result.best_epoch == 0
    assert payload["epoch"] == 0
    assert result.best_validation_total_loss == pytest.approx(
        result.initial_validation_total_loss
    )
