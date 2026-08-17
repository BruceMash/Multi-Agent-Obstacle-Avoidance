"""Stage-II coordination supervision for the frozen Stage-I GAT.

The module consumes only the existing heterogeneous graphs and their stored
H=4 proposal preview trajectories.  It never generates candidates, executes
the environment, or changes the graph/model schema.
"""

from __future__ import annotations

import math
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from planning.gat.candidate_selector import (
    EdgeEnhancedGATConfig,
    PolicyPreviewEdgeEnhancedGATSelector,
    batch_candidate_graphs,
)
from planning.gat.stage1_training import (
    Stage1Example,
    compute_offline_metrics,
    load_model_checkpoint,
    ragged_soft_target_cross_entropy,
    resolve_device,
    seed_everything,
)


SCHEMA_VERSION = "gat_stage2_coordination_v1"
PAIR_INDICES = ((0, 1), (0, 2), (1, 2))


@dataclass(frozen=True)
class PairPreviewMatrix:
    left_agent_id: int
    right_agent_id: int
    overlap: np.ndarray
    minimum_distance: np.ndarray
    risk_duration: np.ndarray
    d_safe: float
    dt: float
    horizon_steps: int


@dataclass(frozen=True)
class JointStateGroup:
    state_group_id: str
    split: str
    scenario: str
    seed: int
    timestep: int
    examples: tuple[Stage1Example, Stage1Example, Stage1Example]
    pair_matrices: tuple[PairPreviewMatrix, PairPreviewMatrix, PairPreviewMatrix]


@dataclass(frozen=True)
class Stage2EpochRecord:
    optimization_seed: int
    lambda_overlap: float
    epoch: int
    train_selection_loss: float
    train_overlap_loss: float
    train_total_loss: float
    validation_selection_loss: float
    validation_overlap_loss: float
    validation_total_loss: float
    validation_top1: float
    validation_mrr: float
    validation_expected_overlap: float
    validation_risky_selection_rate: float | None
    validation_null_probability: float
    validation_null_top1_rate: float
    validation_fixed_rank_max_rate: float
    epoch_runtime_s: float
    improved: bool


@dataclass(frozen=True)
class Stage2TrainingRunResult:
    optimization_seed: int
    lambda_overlap: float
    best_epoch: int
    epochs_completed: int
    early_stopped: bool
    initial_validation_total_loss: float
    best_validation_total_loss: float
    best_validation_metrics: Mapping[str, Any]
    best_checkpoint: Path
    last_checkpoint: Path
    history: tuple[Stage2EpochRecord, ...]
    runtime_s: float


def _readonly_array(value: Any, *, dtype: Any = float) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _preview_trajectories(example: Stage1Example) -> tuple[np.ndarray, ...]:
    trajectories = getattr(example.graph, "candidate_preview_positions", None)
    if trajectories is None or len(trajectories) != example.proposal_count:
        raise ValueError(f"missing proposal preview trajectories: {example.sample_id}")
    horizon = int(example.graph.graph_metadata["H"])
    result: list[np.ndarray] = []
    for index, trajectory in enumerate(trajectories):
        values = np.asarray(trajectory, dtype=float)
        if values.shape != (horizon, 3) or not np.isfinite(values).all():
            raise ValueError(
                f"invalid H={horizon} trajectory for {example.sample_id}, proposal {index}"
            )
        result.append(values)
    return tuple(result)


def build_pair_preview_matrix(
    left: Stage1Example,
    right: Stage1Example,
    *,
    required_horizon: int,
) -> PairPreviewMatrix:
    """Build the paper overlap matrix from existing H=4 proposal trajectories."""

    left_meta = left.graph.graph_metadata
    right_meta = right.graph.graph_metadata
    horizon = int(left_meta["H"])
    if horizon != int(required_horizon) or int(right_meta["H"]) != horizon:
        raise ValueError("overlap horizon must equal the frozen graph preview horizon")
    d_safe = float(left_meta["d_safe"])
    dt = float(left_meta["dt"])
    if not math.isclose(float(right_meta["d_safe"]), d_safe):
        raise ValueError("paired graphs must use the same d_safe")
    if not math.isclose(float(right_meta["dt"]), dt):
        raise ValueError("paired graphs must use the same dt")
    left_trajectories = _preview_trajectories(left)
    right_trajectories = _preview_trajectories(right)
    shape = (left.proposal_count, right.proposal_count)
    overlap = np.zeros(shape, dtype=np.float64)
    minimum_distance = np.full(shape, np.inf, dtype=np.float64)
    risk_duration = np.zeros(shape, dtype=np.float64)
    for left_index, left_trajectory in enumerate(left_trajectories):
        for right_index, right_trajectory in enumerate(right_trajectories):
            distances = np.linalg.norm(left_trajectory - right_trajectory, axis=1)
            overlap[left_index, right_index] = float(
                np.mean(np.maximum(d_safe - distances, 0.0) / d_safe)
            )
            minimum_distance[left_index, right_index] = float(np.min(distances))
            risk_duration[left_index, right_index] = float(
                np.count_nonzero(distances < d_safe) * dt
            )
    return PairPreviewMatrix(
        left_agent_id=int(left.ego_agent_id),
        right_agent_id=int(right.ego_agent_id),
        overlap=_readonly_array(overlap),
        minimum_distance=_readonly_array(minimum_distance),
        risk_duration=_readonly_array(risk_duration),
        d_safe=d_safe,
        dt=dt,
        horizon_steps=horizon,
    )


def build_joint_state_groups(
    examples: Sequence[Stage1Example],
    *,
    required_horizon: int,
) -> tuple[JointStateGroup, ...]:
    grouped: dict[str, list[Stage1Example]] = defaultdict(list)
    for example in examples:
        grouped[example.state_group_id].append(example)
    result: list[JointStateGroup] = []
    for state_group_id in sorted(grouped):
        members = sorted(grouped[state_group_id], key=lambda item: item.ego_agent_id)
        if len(members) != 3 or len({item.ego_agent_id for item in members}) != 3:
            raise ValueError(f"joint state must contain three unique ego graphs: {state_group_id}")
        if [item.ego_agent_id for item in members] != [0, 1, 2]:
            raise ValueError(f"joint state must contain ego ids 0,1,2: {state_group_id}")
        invariant_fields = (
            {item.split for item in members},
            {item.scenario for item in members},
            {item.seed for item in members},
            {item.timestep for item in members},
        )
        if any(len(values) != 1 for values in invariant_fields):
            raise ValueError(f"inconsistent joint-state metadata: {state_group_id}")
        pair_matrices = tuple(
            build_pair_preview_matrix(
                members[left_index],
                members[right_index],
                required_horizon=required_horizon,
            )
            for left_index, right_index in PAIR_INDICES
        )
        result.append(
            JointStateGroup(
                state_group_id=state_group_id,
                split=members[0].split,
                scenario=members[0].scenario,
                seed=int(members[0].seed),
                timestep=int(members[0].timestep),
                examples=(members[0], members[1], members[2]),
                pair_matrices=pair_matrices,  # type: ignore[arg-type]
            )
        )
    return tuple(result)


def split_joint_groups(
    groups: Sequence[JointStateGroup],
) -> dict[str, tuple[JointStateGroup, ...]]:
    by_split: dict[str, list[JointStateGroup]] = defaultdict(list)
    for group in groups:
        by_split[group.split].append(group)
    expected = {"train", "validation", "test"}
    if set(by_split) != expected:
        raise ValueError(f"missing joint split: {expected - set(by_split)}")
    return {name: tuple(by_split[name]) for name in sorted(by_split)}


def flatten_joint_groups(groups: Sequence[JointStateGroup]) -> list[Stage1Example]:
    return [example for group in groups for example in group.examples]


def iter_joint_batches(
    groups: Sequence[JointStateGroup],
    *,
    graph_budget: int,
    shuffle_seed: int | None,
) -> Iterable[list[JointStateGroup]]:
    if int(graph_budget) < 3:
        raise ValueError("graph budget must accommodate one complete joint state")
    groups_per_batch = int(graph_budget) // 3
    indices = np.arange(len(groups))
    if shuffle_seed is not None:
        np.random.default_rng(int(shuffle_seed)).shuffle(indices)
    for start in range(0, len(indices), groups_per_batch):
        yield [groups[int(index)] for index in indices[start : start + groups_per_batch]]


def joint_batch_losses(
    model: PolicyPreviewEdgeEnhancedGATSelector,
    groups: Sequence[JointStateGroup],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, Any, list[Stage1Example]]:
    examples = flatten_joint_groups(groups)
    batch = batch_candidate_graphs([item.graph for item in examples]).to(device)
    output = model(batch)
    selection_loss = ragged_soft_target_cross_entropy(
        output.candidate_logits,
        output.candidate_ptr,
        [item.soft_target for item in examples],
    )
    group_losses: list[torch.Tensor] = []
    graph_offset = 0
    for group in groups:
        probabilities = [
            output.probabilities_for_graph(graph_offset + local_index)
            for local_index in range(3)
        ]
        pair_losses: list[torch.Tensor] = []
        for pair_index, (left_index, right_index) in enumerate(PAIR_INDICES):
            left_probability = probabilities[left_index][1:]
            right_probability = probabilities[right_index][1:]
            matrix = torch.tensor(
                np.asarray(group.pair_matrices[pair_index].overlap),
                dtype=output.candidate_probabilities.dtype,
                device=device,
            )
            if tuple(matrix.shape) != (
                left_probability.numel(),
                right_probability.numel(),
            ):
                raise ValueError("overlap matrix and real proposal classes disagree")
            if matrix.numel() == 0:
                pair_losses.append(output.candidate_logits.new_zeros(()))
            else:
                pair_losses.append(left_probability @ matrix @ right_probability)
        # Exactly the three unordered pairs (0,1), (0,2), (1,2).
        group_losses.append(torch.stack(pair_losses).mean())
        graph_offset += 3
    overlap_loss = torch.stack(group_losses).mean()
    return selection_loss, overlap_loss, output, examples


def _softmax_numpy(logits: Sequence[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - np.max(values)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials)


def infer_joint_groups(
    model: PolicyPreviewEdgeEnhancedGATSelector,
    groups: Sequence[JointStateGroup],
    *,
    graph_budget: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    result: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for batch_groups in iter_joint_batches(
            groups, graph_budget=graph_budget, shuffle_seed=None
        ):
            examples = flatten_joint_groups(batch_groups)
            output = model(
                batch_candidate_graphs([item.graph for item in examples]).to(device)
            )
            for graph_index, example in enumerate(examples):
                result[example.sample_id] = (
                    output.logits_for_graph(graph_index).detach().cpu().numpy()
                )
    if len(result) != 3 * len(groups):
        raise RuntimeError("joint inference did not cover every ego graph exactly once")
    return result


def compute_interaction_metrics(
    groups: Sequence[JointStateGroup],
    scores_by_sample: Mapping[str, Sequence[float] | np.ndarray],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    all_examples = flatten_joint_groups(groups)
    all_scores = [np.asarray(scores_by_sample[item.sample_id]) for item in all_examples]
    probabilities = {
        item.sample_id: _softmax_numpy(scores_by_sample[item.sample_id])
        for item in all_examples
    }
    null_probabilities = [float(probabilities[item.sample_id][0]) for item in all_examples]
    null_top1 = [
        int(np.argmax(probabilities[item.sample_id])) == 0 for item in all_examples
    ]
    rich_examples = [
        item for item in all_examples if item.interaction_group == "interaction-rich"
    ]
    rich_scores = [np.asarray(scores_by_sample[item.sample_id]) for item in rich_examples]
    risk_examples = [item for item in all_examples if item.descriptor_risk_positive]
    risk_scores = [np.asarray(scores_by_sample[item.sample_id]) for item in risk_examples]
    rich_metrics = (
        compute_offline_metrics(rich_examples, rich_scores)
        if rich_examples
        else {"top1_accuracy": None, "mrr": None}
    )
    risk_metrics = (
        compute_offline_metrics(risk_examples, risk_scores)
        if risk_examples
        else {"top1_accuracy": None, "mrr": None}
    )

    pair_records: list[dict[str, Any]] = []
    expected_values: list[float] = []
    conditional_values: list[float] = []
    selected_overlap: list[float] = []
    selected_minimum_distance: list[float] = []
    selected_risk: list[bool] = []
    null_involved = 0
    for group in groups:
        local_probabilities = [probabilities[item.sample_id] for item in group.examples]
        for pair_index, (left_index, right_index) in enumerate(PAIR_INDICES):
            pair = group.pair_matrices[pair_index]
            left = local_probabilities[left_index]
            right = local_probabilities[right_index]
            left_proposal = left[1:]
            right_proposal = right[1:]
            left_mass = float(np.sum(left_proposal))
            right_mass = float(np.sum(right_proposal))
            mass_product = left_mass * right_mass
            expected = (
                float(left_proposal @ pair.overlap @ right_proposal)
                if pair.overlap.size
                else 0.0
            )
            conditional = (
                float(expected / mass_product) if mass_product > 1.0e-15 else None
            )
            expected_values.append(expected)
            if conditional is not None:
                conditional_values.append(conditional)
            left_class = int(np.argmax(left))
            right_class = int(np.argmax(right))
            selected_valid = left_class > 0 and right_class > 0
            selected_overlap_value = None
            selected_distance_value = None
            selected_risk_value = None
            if selected_valid:
                matrix_left = left_class - 1
                matrix_right = right_class - 1
                selected_overlap_value = float(
                    pair.overlap[matrix_left, matrix_right]
                )
                selected_distance_value = float(
                    pair.minimum_distance[matrix_left, matrix_right]
                )
                selected_risk_value = bool(
                    selected_distance_value < pair.d_safe
                    or pair.risk_duration[matrix_left, matrix_right] > 0.0
                )
                selected_overlap.append(selected_overlap_value)
                selected_minimum_distance.append(selected_distance_value)
                selected_risk.append(selected_risk_value)
            else:
                null_involved += 1
            pair_records.append(
                {
                    "state_group_id": group.state_group_id,
                    "scenario": group.scenario,
                    "seed": group.seed,
                    "timestep": group.timestep,
                    "left_agent_id": pair.left_agent_id,
                    "right_agent_id": pair.right_agent_id,
                    "left_proposal_mass": left_mass,
                    "right_proposal_mass": right_mass,
                    "proposal_mass_product": mass_product,
                    "expected_overlap": expected,
                    "conditional_proposal_overlap_diagnostic": conditional,
                    "left_selected_class": left_class,
                    "right_selected_class": right_class,
                    "selected_pair_valid": selected_valid,
                    "selected_pair_overlap": selected_overlap_value,
                    "selected_minimum_distance": selected_distance_value,
                    "selected_risky": selected_risk_value,
                }
            )
    pair_count = len(pair_records)
    valid_pair_count = len(selected_overlap)
    summary = {
        "joint_state_count": len(groups),
        "ego_graph_count": len(all_examples),
        "unordered_uav_pair_count": pair_count,
        "expected_overlap": float(np.mean(expected_values)),
        "conditional_proposal_overlap_diagnostic": (
            float(np.mean(conditional_values)) if conditional_values else None
        ),
        "selected_pair_valid_count": valid_pair_count,
        "selected_pair_null_involved_count": null_involved,
        "selected_pair_null_involved_rate": (
            float(null_involved / pair_count) if pair_count else None
        ),
        "selected_pair_overlap": (
            float(np.mean(selected_overlap)) if selected_overlap else None
        ),
        "risky_selection_count": int(sum(selected_risk)),
        "risky_selection_rate": (
            float(np.mean(selected_risk)) if selected_risk else None
        ),
        "selected_d_min_mean": (
            float(np.mean(selected_minimum_distance))
            if selected_minimum_distance
            else None
        ),
        "selected_d_min_median": (
            float(np.median(selected_minimum_distance))
            if selected_minimum_distance
            else None
        ),
        "selected_d_min_p05": (
            float(np.percentile(selected_minimum_distance, 5))
            if selected_minimum_distance
            else None
        ),
        "mean_null_probability": float(np.mean(null_probabilities)),
        "null_top1_selection_rate": float(np.mean(null_top1)),
        "interaction_rich_mean_null_probability": (
            float(np.mean([probabilities[item.sample_id][0] for item in rich_examples]))
            if rich_examples
            else None
        ),
        "interaction_rich_null_top1_rate": (
            float(
                np.mean(
                    [
                        int(np.argmax(probabilities[item.sample_id])) == 0
                        for item in rich_examples
                    ]
                )
            )
            if rich_examples
            else None
        ),
        "interaction_rich_graph_count": len(rich_examples),
        "interaction_rich_top1": rich_metrics["top1_accuracy"],
        "interaction_rich_mrr": rich_metrics["mrr"],
        "secondary_risk_graph_count": len(risk_examples),
        "secondary_risk_top1": risk_metrics["top1_accuracy"],
        "secondary_risk_mrr": risk_metrics["mrr"],
        "proposal_probability_renormalized_for_loss": False,
        "null_trajectory_used": False,
    }
    return summary, pair_records


@torch.no_grad()
def evaluate_joint_model(
    model: PolicyPreviewEdgeEnhancedGATSelector,
    groups: Sequence[JointStateGroup],
    *,
    lambda_overlap: float,
    graph_budget: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, np.ndarray], list[dict[str, Any]]]:
    model.eval()
    selection_values: list[float] = []
    selection_weights: list[int] = []
    overlap_values: list[float] = []
    overlap_weights: list[int] = []
    scores_by_sample: dict[str, np.ndarray] = {}
    for batch_groups in iter_joint_batches(
        groups, graph_budget=graph_budget, shuffle_seed=None
    ):
        selection_loss, overlap_loss, output, examples = joint_batch_losses(
            model, batch_groups, device=device
        )
        selection_values.append(float(selection_loss.item()))
        selection_weights.append(len(examples))
        overlap_values.append(float(overlap_loss.item()))
        overlap_weights.append(len(batch_groups))
        for graph_index, example in enumerate(examples):
            scores_by_sample[example.sample_id] = (
                output.logits_for_graph(graph_index).detach().cpu().numpy()
            )
    selection_mean = float(np.average(selection_values, weights=selection_weights))
    overlap_mean = float(np.average(overlap_values, weights=overlap_weights))
    examples = flatten_joint_groups(groups)
    scores = [scores_by_sample[item.sample_id] for item in examples]
    offline = compute_offline_metrics(examples, scores, loss=selection_mean)
    interaction, pair_records = compute_interaction_metrics(groups, scores_by_sample)
    metrics = {
        **offline,
        **interaction,
        "selection_loss": selection_mean,
        "overlap_loss": overlap_mean,
        "weighted_overlap_loss": float(lambda_overlap * overlap_mean),
        "total_loss": float(selection_mean + lambda_overlap * overlap_mean),
        "lambda_overlap": float(lambda_overlap),
    }
    return metrics, scores_by_sample, pair_records


def null_escape_audit(
    stage1_metrics: Mapping[str, Any],
    stage2_metrics: Mapping[str, Any],
    stage1_pairs: Sequence[Mapping[str, Any]],
    stage2_pairs: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    key = lambda row: (
        row["state_group_id"],
        int(row["left_agent_id"]),
        int(row["right_agent_id"]),
    )
    left_index = {key(row): row for row in stage1_pairs}
    right_index = {key(row): row for row in stage2_pairs}
    if set(left_index) != set(right_index):
        raise ValueError("Stage-I/II pair records must describe identical joint states")
    baseline_expected: list[float] = []
    stage2_expected: list[float] = []
    mass_counterfactual: list[float] = []
    for pair_key in sorted(left_index):
        before = left_index[pair_key]
        after = right_index[pair_key]
        conditional_before = before["conditional_proposal_overlap_diagnostic"]
        if conditional_before is None:
            continue
        baseline_expected.append(float(before["expected_overlap"]))
        stage2_expected.append(float(after["expected_overlap"]))
        mass_counterfactual.append(
            float(after["proposal_mass_product"]) * float(conditional_before)
        )
    before_mean = float(np.mean(baseline_expected)) if baseline_expected else 0.0
    after_mean = float(np.mean(stage2_expected)) if stage2_expected else 0.0
    counterfactual_mean = (
        float(np.mean(mass_counterfactual)) if mass_counterfactual else 0.0
    )
    total_reduction = before_mean - after_mean
    mass_transfer_reduction = before_mean - counterfactual_mean
    conditional_redistribution_reduction = counterfactual_mean - after_mean
    mass_fraction = (
        mass_transfer_reduction / total_reduction if total_reduction > 1.0e-15 else None
    )
    null_top1_delta = float(stage2_metrics["null_top1_selection_rate"]) - float(
        stage1_metrics["null_top1_selection_rate"]
    )
    null_probability_delta = float(stage2_metrics["mean_null_probability"]) - float(
        stage1_metrics["mean_null_probability"]
    )
    rate_trigger = null_top1_delta > float(config["null_top1_increase_threshold"])
    probability_trigger = null_probability_delta > float(
        config["mean_null_probability_increase_threshold"]
    )
    primarily_mass = bool(
        mass_fraction is not None
        and mass_fraction > float(config["mass_transfer_primary_fraction_threshold"])
        and mass_transfer_reduction > 0.0
    )
    detected = bool(
        (rate_trigger or probability_trigger)
        and total_reduction > 0.0
        and primarily_mass
    )
    return {
        "NULL_ESCAPE_DETECTED": "YES" if detected else "NO",
        "stage1_null_top1_rate": float(stage1_metrics["null_top1_selection_rate"]),
        "stage2_null_top1_rate": float(stage2_metrics["null_top1_selection_rate"]),
        "null_top1_rate_change": null_top1_delta,
        "stage1_mean_null_probability": float(stage1_metrics["mean_null_probability"]),
        "stage2_mean_null_probability": float(stage2_metrics["mean_null_probability"]),
        "mean_null_probability_change": null_probability_delta,
        "overlap_reduction_on_decomposable_pairs": total_reduction,
        "mass_transfer_reduction": mass_transfer_reduction,
        "conditional_redistribution_reduction": conditional_redistribution_reduction,
        "mass_transfer_fraction_of_reduction": mass_fraction,
        "mass_transfer_is_primary": primarily_mass,
        "decomposable_pair_count": len(baseline_expected),
        "proposal_only_probability_renormalization_used_for_loss": False,
    }


def clone_with_edge_ablation(example: Stage1Example, variant: str) -> Stage1Example:
    if variant == "full":
        return example
    graph = example.graph.clone()
    store = graph["align", "spatiotemporal", "proposal"]
    if variant == "zero_st_edge":
        store.edge_attr = torch.zeros_like(store.edge_attr)
        store.edge_attr_normalized = torch.zeros_like(store.edge_attr_normalized)
    elif variant == "remove_align_message":
        store.edge_index = store.edge_index[:, :0]
        store.edge_attr = store.edge_attr[:0]
        store.edge_attr_normalized = store.edge_attr_normalized[:0]
        for optional in (
            "minimum_step_index",
            "candidate_minimum_position",
            "neighbor_minimum_position",
        ):
            if hasattr(store, optional):
                values = getattr(store, optional)
                setattr(store, optional, values[:0])
    else:
        raise ValueError(f"unknown edge ablation variant: {variant}")
    return replace(example, graph=graph)


def ablate_joint_groups(
    groups: Sequence[JointStateGroup], variant: str
) -> tuple[JointStateGroup, ...]:
    if variant == "full":
        return tuple(groups)
    result: list[JointStateGroup] = []
    for group in groups:
        examples = tuple(clone_with_edge_ablation(item, variant) for item in group.examples)
        result.append(replace(group, examples=examples))  # pair matrices stay frozen
    return tuple(result)


def edge_ablation_audit(
    model: PolicyPreviewEdgeEnhancedGATSelector,
    groups: Sequence[JointStateGroup],
    *,
    graph_budget: int,
    device: torch.device,
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    variant_outputs: dict[str, tuple[dict[str, Any], dict[str, np.ndarray]]] = {}
    rows: list[dict[str, Any]] = []
    for variant in config["variants"]:
        variant_groups = ablate_joint_groups(groups, variant)
        metrics, scores, _ = evaluate_joint_model(
            model,
            variant_groups,
            lambda_overlap=0.0,
            graph_budget=graph_budget,
            device=device,
        )
        variant_outputs[str(variant)] = (metrics, scores)
        rows.append(
            {
                "variant": variant,
                "scope": "overall",
                "graph_count": metrics["graph_count"],
                "top1": metrics["top1_accuracy"],
                "top3": metrics["top3_accuracy"],
                "mrr": metrics["mrr"],
                "spearman": metrics["spearman_proposal_only_mean"],
                "spearman_n": metrics["spearman_valid_graph_count"],
            }
        )
        rows.append(
            {
                "variant": variant,
                "scope": "interaction_rich",
                "graph_count": metrics["interaction_rich_graph_count"],
                "top1": metrics["interaction_rich_top1"],
                "top3": None,
                "mrr": metrics["interaction_rich_mrr"],
                "spearman": None,
                "spearman_n": None,
            }
        )
        rows.append(
            {
                "variant": variant,
                "scope": "secondary_high_risk",
                "graph_count": metrics["secondary_risk_graph_count"],
                "top1": metrics["secondary_risk_top1"],
                "top3": None,
                "mrr": metrics["secondary_risk_mrr"],
                "spearman": None,
                "spearman_n": None,
            }
        )
    full_scores = variant_outputs["full"][1]
    all_examples = flatten_joint_groups(groups)
    for variant in ("zero_st_edge", "remove_align_message"):
        alternative = variant_outputs[variant][1]
        max_delta = 0.0
        mean_delta: list[float] = []
        top1_changes = 0
        ranking_changes = 0
        for example in all_examples:
            before = np.asarray(full_scores[example.sample_id])
            after = np.asarray(alternative[example.sample_id])
            delta = np.abs(before - after)
            max_delta = max(max_delta, float(np.max(delta)))
            mean_delta.append(float(np.mean(delta)))
            top1_changes += int(np.argmax(before) != np.argmax(after))
            ranking_changes += int(
                not np.array_equal(
                    np.argsort(-before, kind="stable"),
                    np.argsort(-after, kind="stable"),
                )
            )
        for row in rows:
            if row["variant"] == variant and row["scope"] == "overall":
                row.update(
                    {
                        "max_abs_logit_delta_vs_full": max_delta,
                        "mean_abs_logit_delta_vs_full": float(np.mean(mean_delta)),
                        "top1_changed_count_vs_full": top1_changes,
                        "full_ranking_changed_count_vs_full": ranking_changes,
                    }
                )
    decisive_scopes = [
        scope
        for scope in ("overall", "interaction_rich", "secondary_high_risk")
        if next(
            int(row["graph_count"])
            for row in rows
            if row["variant"] == "full" and row["scope"] == scope
        )
        >= int(config["minimum_decisive_subset_graph_count"])
    ]
    if not decisive_scopes:
        classification = "NOT_ESTABLISHED"
    else:
        tolerance = float(config["numeric_logit_change_tolerance"])
        numeric_dependency = all(
            next(
                float(row.get("max_abs_logit_delta_vs_full", 0.0))
                for row in rows
                if row["variant"] == variant and row["scope"] == "overall"
            )
            > tolerance
            for variant in ("zero_st_edge", "remove_align_message")
        )
        if not numeric_dependency:
            classification = "NONE"
        else:
            threshold = float(config["clear_minimum_top1_degradation"])
            full_by_scope = {
                row["scope"]: float(row["top1"])
                for row in rows
                if row["variant"] == "full"
            }
            clear_variants = []
            for variant in ("zero_st_edge", "remove_align_message"):
                alt_by_scope = {
                    row["scope"]: float(row["top1"])
                    for row in rows
                    if row["variant"] == variant
                }
                clear_variants.append(
                    any(
                        full_by_scope[scope] - alt_by_scope[scope] >= threshold
                        for scope in decisive_scopes
                    )
                )
            classification = "CLEAR" if all(clear_variants) else "WEAK"
    return rows, classification


def _gradient_norms(model: nn.Module) -> dict[str, Any]:
    total_squared = 0.0
    edge_squared = 0.0
    message_squared = 0.0
    finite = True
    edge_nonzero = 0
    message_nonzero = 0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        finite = finite and bool(torch.isfinite(gradient).all().item())
        norm = float(torch.linalg.vector_norm(gradient).item())
        total_squared += norm * norm
        if "edge_encoders.encoders.spatiotemporal" in name:
            edge_squared += norm * norm
            edge_nonzero += int(norm > 0.0)
        if "relations.spatiotemporal" in name:
            message_squared += norm * norm
            message_nonzero += int(norm > 0.0)
    return {
        "gradient_norm": math.sqrt(total_squared),
        "gradient_finite": finite,
        "spatiotemporal_edge_encoder_gradient_norm": math.sqrt(edge_squared),
        "spatiotemporal_edge_encoder_nonzero_tensor_count": edge_nonzero,
        "align_message_gradient_norm": math.sqrt(message_squared),
        "align_message_nonzero_tensor_count": message_nonzero,
    }


def _full_split_objective_gradient(
    model: PolicyPreviewEdgeEnhancedGATSelector,
    groups: Sequence[JointStateGroup],
    *,
    objective: str,
    lambda_overlap: float,
    graph_budget: int,
    device: torch.device,
) -> tuple[float, dict[str, Any]]:
    model.train()
    model.zero_grad(set_to_none=True)
    total_graphs = 3 * len(groups)
    total_groups = len(groups)
    objective_value = 0.0
    for batch_groups in iter_joint_batches(
        groups, graph_budget=graph_budget, shuffle_seed=None
    ):
        selection_loss, overlap_loss, _, examples = joint_batch_losses(
            model, batch_groups, device=device
        )
        if objective == "selection":
            weight = len(examples) / total_graphs
            value = selection_loss
        elif objective == "weighted_overlap":
            weight = len(batch_groups) / total_groups
            value = float(lambda_overlap) * overlap_loss
        else:
            raise ValueError(f"unknown gradient objective: {objective}")
        scaled = value * weight
        scaled.backward()
        objective_value += float(value.detach().item()) * weight
    norms = _gradient_norms(model)
    model.zero_grad(set_to_none=True)
    return objective_value, norms


def gradient_scale_audit(
    model: PolicyPreviewEdgeEnhancedGATSelector,
    groups: Sequence[JointStateGroup],
    lambdas: Sequence[float],
    *,
    graph_budget: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Measure independent full-train gradients with a zero_grad boundary."""

    rows: list[dict[str, Any]] = []
    for lambda_overlap in lambdas:
        selection_value, selection_norms = _full_split_objective_gradient(
            model,
            groups,
            objective="selection",
            lambda_overlap=float(lambda_overlap),
            graph_budget=graph_budget,
            device=device,
        )
        weighted_overlap, overlap_norms = _full_split_objective_gradient(
            model,
            groups,
            objective="weighted_overlap",
            lambda_overlap=float(lambda_overlap),
            graph_budget=graph_budget,
            device=device,
        )
        unweighted_overlap = (
            weighted_overlap / float(lambda_overlap) if lambda_overlap else 0.0
        )
        g_select = float(selection_norms["gradient_norm"])
        g_overlap = float(overlap_norms["gradient_norm"])
        rows.append(
            {
                "lambda_overlap": float(lambda_overlap),
                "selection_loss": selection_value,
                "overlap_loss": unweighted_overlap,
                "weighted_overlap_loss": weighted_overlap,
                "OVERLAP_WEIGHTED_LOSS_RATIO": (
                    weighted_overlap / selection_value
                    if abs(selection_value) > 1.0e-15
                    else None
                ),
                "g_select": g_select,
                "g_overlap_weighted": g_overlap,
                "OVERLAP_GRADIENT_RATIO": (
                    g_overlap / g_select if g_select > 1.0e-15 else None
                ),
                "selection_gradient_finite": selection_norms["gradient_finite"],
                "overlap_gradient_finite": overlap_norms["gradient_finite"],
                "overlap_spatiotemporal_edge_encoder_gradient_norm": overlap_norms[
                    "spatiotemporal_edge_encoder_gradient_norm"
                ],
                "overlap_spatiotemporal_edge_encoder_nonzero_tensor_count": overlap_norms[
                    "spatiotemporal_edge_encoder_nonzero_tensor_count"
                ],
                "overlap_align_message_gradient_norm": overlap_norms[
                    "align_message_gradient_norm"
                ],
                "overlap_align_message_nonzero_tensor_count": overlap_norms[
                    "align_message_nonzero_tensor_count"
                ],
                "separate_backward_passes": True,
                "zero_grad_between_measurements": True,
            }
        )
    return rows


def _checkpoint_payload(
    model: PolicyPreviewEdgeEnhancedGATSelector,
    optimizer: torch.optim.Optimizer,
    *,
    optimization_seed: int,
    epoch: int,
    lambda_overlap: float,
    validation_metrics: Mapping[str, Any],
    stage1_config: Mapping[str, Any],
    stage2_config: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "optimization_seed": int(optimization_seed),
        "epoch": int(epoch),
        "lambda_overlap": float(lambda_overlap),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation_metrics": dict(validation_metrics),
        "model_config": dict(stage1_config["model"]),
        "supervision": dict(stage1_config["supervision"]),
        "offline_metric_definitions": dict(stage2_config["offline_metric_definitions"]),
        "initialization": dict(stage2_config["initialization"]),
        "overlap": dict(stage2_config["overlap"]),
    }


def train_one_stage2_seed(
    stage2_config: Mapping[str, Any],
    stage1_config: Mapping[str, Any],
    train_groups: Sequence[JointStateGroup],
    validation_groups: Sequence[JointStateGroup],
    *,
    stage1_checkpoint: Path,
    lambda_overlap: float,
    optimization_seed: int,
    checkpoint_dir: Path,
) -> Stage2TrainingRunResult:
    seed_everything(optimization_seed)
    training = stage2_config["training"]
    device = resolve_device(training["device"])
    model = load_model_checkpoint(stage1_checkpoint, stage1_config, device)
    # The Stage-I model weights are frozen as initialization; AdamW state is fresh.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    graph_budget = int(stage2_config["batching"]["stage1_graph_budget"])
    max_epochs = int(training["max_epochs"])
    patience = int(training["early_stopping_patience"])
    gradient_clip = float(training["gradient_clip_norm"])
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    label = f"seed_{optimization_seed:03d}_lambda_{lambda_overlap:g}"
    best_path = checkpoint_dir / f"{label}_best_validation.pt"
    last_path = checkpoint_dir / f"{label}_last.pt"
    initial_validation, _, _ = evaluate_joint_model(
        model,
        validation_groups,
        lambda_overlap=lambda_overlap,
        graph_budget=graph_budget,
        device=device,
    )
    # Fine-tuning starts from the authoritative Stage-I checkpoint.  Epoch 0
    # must remain eligible; otherwise a degraded fine-tuning epoch could be
    # incorrectly reported as the best Stage-II checkpoint.
    best_total = float(initial_validation["total_loss"])
    best_epoch = 0
    best_metrics: dict[str, Any] = dict(initial_validation)
    epochs_without_improvement = 0
    history: list[Stage2EpochRecord] = []
    started = time.perf_counter()
    last_validation = initial_validation
    torch.save(
        _checkpoint_payload(
            model,
            optimizer,
            optimization_seed=optimization_seed,
            epoch=0,
            lambda_overlap=lambda_overlap,
            validation_metrics=initial_validation,
            stage1_config=stage1_config,
            stage2_config=stage2_config,
        ),
        best_path,
    )
    for epoch in range(1, max_epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        selection_values: list[float] = []
        selection_weights: list[int] = []
        overlap_values: list[float] = []
        overlap_weights: list[int] = []
        for batch_groups in iter_joint_batches(
            train_groups,
            graph_budget=graph_budget,
            shuffle_seed=optimization_seed * 100_000 + epoch,
        ):
            optimizer.zero_grad(set_to_none=True)
            selection_loss, overlap_loss, _, examples = joint_batch_losses(
                model, batch_groups, device=device
            )
            total_loss = selection_loss + float(lambda_overlap) * overlap_loss
            if not all(
                bool(torch.isfinite(value).item())
                for value in (selection_loss, overlap_loss, total_loss)
            ):
                raise FloatingPointError(
                    f"non-finite Stage-II loss at seed={optimization_seed}, epoch={epoch}"
                )
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            selection_values.append(float(selection_loss.detach().item()))
            selection_weights.append(len(examples))
            overlap_values.append(float(overlap_loss.detach().item()))
            overlap_weights.append(len(batch_groups))
        train_selection = float(np.average(selection_values, weights=selection_weights))
        train_overlap = float(np.average(overlap_values, weights=overlap_weights))
        train_total = train_selection + float(lambda_overlap) * train_overlap
        validation, _, _ = evaluate_joint_model(
            model,
            validation_groups,
            lambda_overlap=lambda_overlap,
            graph_budget=graph_budget,
            device=device,
        )
        last_validation = validation
        validation_total = float(validation["total_loss"])
        improved = validation_total < best_total - 1.0e-12
        if improved:
            best_total = validation_total
            best_epoch = epoch
            best_metrics = dict(validation)
            epochs_without_improvement = 0
            torch.save(
                _checkpoint_payload(
                    model,
                    optimizer,
                    optimization_seed=optimization_seed,
                    epoch=epoch,
                    lambda_overlap=lambda_overlap,
                    validation_metrics=validation,
                    stage1_config=stage1_config,
                    stage2_config=stage2_config,
                ),
                best_path,
            )
        else:
            epochs_without_improvement += 1
        history.append(
            Stage2EpochRecord(
                optimization_seed=int(optimization_seed),
                lambda_overlap=float(lambda_overlap),
                epoch=epoch,
                train_selection_loss=train_selection,
                train_overlap_loss=train_overlap,
                train_total_loss=train_total,
                validation_selection_loss=float(validation["selection_loss"]),
                validation_overlap_loss=float(validation["overlap_loss"]),
                validation_total_loss=validation_total,
                validation_top1=float(validation["top1_accuracy"]),
                validation_mrr=float(validation["mrr"]),
                validation_expected_overlap=float(validation["expected_overlap"]),
                validation_risky_selection_rate=validation["risky_selection_rate"],
                validation_null_probability=float(validation["mean_null_probability"]),
                validation_null_top1_rate=float(validation["null_top1_selection_rate"]),
                validation_fixed_rank_max_rate=float(
                    validation["maximum_fixed_proposal_rank_prediction_rate"]
                ),
                epoch_runtime_s=float(time.perf_counter() - epoch_started),
                improved=bool(improved),
            )
        )
        if epochs_without_improvement >= patience:
            break
    torch.save(
        _checkpoint_payload(
            model,
            optimizer,
            optimization_seed=optimization_seed,
            epoch=len(history),
            lambda_overlap=lambda_overlap,
            validation_metrics=last_validation,
            stage1_config=stage1_config,
            stage2_config=stage2_config,
        ),
        last_path,
    )
    return Stage2TrainingRunResult(
        optimization_seed=int(optimization_seed),
        lambda_overlap=float(lambda_overlap),
        best_epoch=best_epoch,
        epochs_completed=len(history),
        early_stopped=len(history) < max_epochs,
        initial_validation_total_loss=float(initial_validation["total_loss"]),
        best_validation_total_loss=best_total,
        best_validation_metrics=best_metrics,
        best_checkpoint=best_path,
        last_checkpoint=last_path,
        history=tuple(history),
        runtime_s=float(time.perf_counter() - started),
    )


def load_stage2_checkpoint(
    checkpoint_path: Path,
    stage1_config: Mapping[str, Any],
    device: torch.device,
) -> PolicyPreviewEdgeEnhancedGATSelector:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = PolicyPreviewEdgeEnhancedGATSelector(
        EdgeEnhancedGATConfig.from_mapping(stage1_config["model"])
    ).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model


def smoke_gate_result(
    result: Stage2TrainingRunResult,
    gradient_row: Mapping[str, Any],
    smoke_config: Mapping[str, Any],
) -> dict[str, Any]:
    metrics = result.best_validation_metrics
    minimum_gradient = float(smoke_config["minimum_nonzero_gradient_norm"])
    finite_loss = all(
        math.isfinite(float(value))
        for value in (
            metrics["total_loss"],
            metrics["selection_loss"],
            metrics["overlap_loss"],
        )
    )
    validation_limit = result.initial_validation_total_loss * (
        1.0 + float(smoke_config["maximum_validation_total_loss_increase_fraction"])
    )
    checks = {
        "total_selection_overlap_loss_finite": finite_loss,
        "overlap_gradient_nonzero_finite": bool(
            gradient_row["overlap_gradient_finite"]
            and float(gradient_row["g_overlap_weighted"]) > minimum_gradient
        ),
        "spatiotemporal_edge_encoder_gradient_nonzero": float(
            gradient_row["overlap_spatiotemporal_edge_encoder_gradient_norm"]
        )
        > minimum_gradient,
        "align_message_gradient_nonzero": float(
            gradient_row["overlap_align_message_gradient_norm"]
        )
        > minimum_gradient,
        "validation_loss_no_collapse": result.best_validation_total_loss
        <= validation_limit,
        "no_null_collapse": float(metrics["null_top1_selection_rate"])
        < float(smoke_config["maximum_null_prediction_rate"]),
        "no_fixed_rank_collapse": float(
            metrics["maximum_fixed_proposal_rank_prediction_rate"]
        )
        < float(smoke_config["maximum_fixed_proposal_rank_prediction_rate"]),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "initial_validation_total_loss": result.initial_validation_total_loss,
        "best_validation_total_loss": result.best_validation_total_loss,
        "best_epoch": result.best_epoch,
    }


def select_lambda_candidate(
    rows: list[dict[str, Any]],
    baseline: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, str, dict[str, Any]]:
    """Apply the frozen validation-only lambda decision rule."""

    baseline_top1 = float(baseline["top1_accuracy"])
    baseline_expected = float(baseline["expected_overlap"])
    tolerance = float(config["numeric_improvement_tolerance"])
    for row in rows:
        row["validation_top1_change_vs_stage1"] = (
            float(row["top1_accuracy"]) - baseline_top1
        )
        row["expected_overlap_absolute_reduction"] = baseline_expected - float(
            row["expected_overlap"]
        )
        row["expected_overlap_relative_reduction"] = (
            row["expected_overlap_absolute_reduction"] / baseline_expected
            if baseline_expected > 1.0e-15
            else None
        )
        row["eligible"] = bool(
            math.isfinite(float(row["validation_total_loss"]))
            and row["validation_top1_change_vs_stage1"]
            >= -float(config["maximum_validation_top1_regression"])
        )
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        return None, "NO", {"reason": "no_eligible_lambda"}

    def risk_value(row: Mapping[str, Any]) -> float:
        value = row["risky_selection_rate"]
        return math.inf if value is None else float(value)

    minimum_risk = min(risk_value(row) for row in eligible)
    risk_tied = [
        row
        for row in eligible
        if (
            risk_value(row) == minimum_risk
            or abs(risk_value(row) - minimum_risk) <= tolerance
        )
    ]
    minimum_expected = min(float(row["expected_overlap"]) for row in risk_tied)
    expected_tied = [
        row
        for row in risk_tied
        if float(row["expected_overlap"]) <= minimum_expected + tolerance
    ]
    maximum_top1 = max(float(row["top1_accuracy"]) for row in expected_tied)
    top1_tied = [
        row
        for row in expected_tied
        if float(row["top1_accuracy"]) >= maximum_top1 - tolerance
    ]
    selected = min(top1_tied, key=lambda row: float(row["lambda_overlap"]))
    improving_count = sum(
        float(row["expected_overlap_absolute_reduction"]) > tolerance
        for row in eligible
    )
    absolute_reduction = float(selected["expected_overlap_absolute_reduction"])
    relative_reduction = selected["expected_overlap_relative_reduction"]
    if (
        absolute_reduction
        >= float(config["yes_minimum_expected_overlap_absolute_reduction"])
        and relative_reduction is not None
        and float(relative_reduction)
        >= float(config["yes_minimum_expected_overlap_relative_reduction"])
    ):
        signal = "YES"
    elif (
        improving_count >= int(config["weak_minimum_consistent_candidate_count"])
        and absolute_reduction > tolerance
    ):
        signal = "WEAK"
    else:
        signal = "NO"
    return selected, signal, {
        "eligible_lambda_count": len(eligible),
        "improving_lambda_count": improving_count,
        "selected_lambda": float(selected["lambda_overlap"]),
        "selected_absolute_reduction": absolute_reduction,
        "selected_relative_reduction": relative_reduction,
    }


def training_result_record(result: Stage2TrainingRunResult) -> dict[str, Any]:
    record = asdict(result)
    record["best_checkpoint"] = str(result.best_checkpoint)
    record["last_checkpoint"] = str(result.last_checkpoint)
    record["history"] = [asdict(item) for item in result.history]
    return record
