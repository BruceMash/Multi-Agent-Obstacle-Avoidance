"""Stage-I supervised training utilities for the frozen candidate GAT.

This module is deliberately limited to offline candidate-ranking supervision.
It does not generate data, execute environments, or alter graph/model schemas.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
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


SCHEMA_VERSION = "gat_stage1_training_v1"


@dataclass(frozen=True)
class Stage1Example:
    sample_id: str
    state_group_id: str
    scenario: str
    seed: int
    timestep: int
    ego_agent_id: int
    split: str
    graph_path: Path
    label_path: Path
    class_count: int
    proposal_count: int
    soft_target: tuple[float, ...]
    target_quality: tuple[float, ...]
    fp_shep_quality: tuple[float, ...]
    proposal_scores: tuple[float, ...]
    interaction_group: str
    descriptor_risk_positive: bool
    graph: Any

    @property
    def reference_class(self) -> int:
        return int(np.argmax(np.asarray(self.soft_target, dtype=float)))


@dataclass(frozen=True)
class EpochRecord:
    optimization_seed: int
    epoch: int
    train_loss: float
    validation_loss: float
    validation_top1: float
    validation_top3: float
    validation_mrr: float
    validation_random_top1: float
    validation_null_prediction_rate: float
    validation_fixed_rank_max_rate: float
    learning_rate: float
    epoch_runtime_s: float
    improved: bool


@dataclass(frozen=True)
class TrainingRunResult:
    optimization_seed: int
    best_epoch: int
    epochs_completed: int
    early_stopped: bool
    best_validation_loss: float
    first_train_loss: float
    best_train_loss: float
    first_validation_loss: float
    best_validation_metrics: Mapping[str, Any]
    best_checkpoint: Path
    last_checkpoint: Path
    history: tuple[EpochRecord, ...]
    runtime_s: float


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def stable_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_split(seed: int, split_config: Mapping[str, Any]) -> str:
    memberships = {
        "train": set(int(value) for value in split_config["train_seeds"]),
        "validation": set(
            int(value) for value in split_config["validation_seeds"]
        ),
        "test": set(int(value) for value in split_config["test_seeds"]),
    }
    hits = [name for name, values in memberships.items() if int(seed) in values]
    if len(hits) != 1:
        raise ValueError(f"seed {seed} must belong to exactly one split, got {hits}")
    return hits[0]


def scenario_interaction_group(
    scenario: str, interaction_config: Mapping[str, Any]
) -> str:
    hits = [
        group
        for group, scenarios in interaction_config["scenario_groups"].items()
        if scenario in scenarios
    ]
    if len(hits) != 1:
        raise ValueError(
            f"scenario {scenario!r} must belong to exactly one interaction group"
        )
    return hits[0]


def graph_descriptor_risk_positive(graph: Any) -> bool:
    store = graph["align", "spatiotemporal", "proposal"]
    edge_attr = store.edge_attr
    if int(edge_attr.shape[0]) == 0:
        return False
    metadata = graph.graph_metadata
    d_safe = float(metadata["d_safe"])
    d_min = edge_attr[:, 1]
    risk_duration = edge_attr[:, 2]
    return bool(torch.any((d_min < d_safe) | (risk_duration > 0.0)).item())


def _soft_target_key(temperature: float) -> str:
    text = f"{float(temperature):g}"
    return f"tau_{text}"


def load_stage1_examples(
    dataset_dir: Path,
    *,
    split_config: Mapping[str, Any],
    supervision_config: Mapping[str, Any],
    interaction_config: Mapping[str, Any],
    map_location: str | torch.device = "cpu",
) -> tuple[list[Stage1Example], dict[str, Any], list[dict[str, Any]]]:
    """Load the existing graph/label artifacts without regenerating data."""

    dataset_dir = Path(dataset_dir).resolve()
    graph_rows = read_csv(dataset_dir / "graph_records.csv")
    h_label = int(supervision_config["H_label"])
    target_name = str(supervision_config["target_name"])
    tau_key = _soft_target_key(supervision_config["soft_target_temperature"])
    examples: list[Stage1Example] = []
    split_rows: list[dict[str, Any]] = []
    group_to_split: dict[str, str] = {}
    class_histogram: Counter[int] = Counter()
    scenario_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()

    for row in graph_rows:
        sample_id = row["sample_id"]
        seed = int(row["seed"])
        split = resolve_split(seed, split_config)
        state_group_id = row["state_group_id"]
        previous = group_to_split.setdefault(state_group_id, split)
        if previous != split:
            raise ValueError(f"group leakage detected for {state_group_id}")
        graph_path = dataset_dir / row["graph_path"]
        label_path = dataset_dir / "labels" / f"{sample_id}__H{h_label}.json"
        if not graph_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"missing graph/label for {sample_id}")
        label = json.loads(label_path.read_text(encoding="utf-8"))
        if int(label["H_label"]) != h_label:
            raise ValueError(f"unexpected H_label for {sample_id}")
        if int(label["H_preview_formal"]) != int(supervision_config["H_preview"]):
            raise ValueError(f"unexpected H_preview for {sample_id}")
        soft_target = tuple(
            float(value) for value in label["soft_targets"][tau_key][target_name]
        )
        target_quality = tuple(float(value) for value in label[target_name])
        fp_shep_quality = tuple(float(value) for value in label["formal_J_preview_3"])
        class_count = int(row["class_count"])
        proposal_count = int(row["K_actual"])
        if class_count != proposal_count + 1:
            raise ValueError(f"invalid null+K class count for {sample_id}")
        if not (
            len(soft_target)
            == len(target_quality)
            == len(fp_shep_quality)
            == len(label["class_mapping"])
            == class_count
        ):
            raise ValueError(f"class/target length mismatch for {sample_id}")
        target_array = np.asarray(soft_target, dtype=float)
        if not np.isfinite(target_array).all() or np.any(target_array < 0.0):
            raise ValueError(f"invalid soft target for {sample_id}")
        if not math.isclose(float(target_array.sum()), 1.0, abs_tol=1.0e-9):
            raise ValueError(f"soft target does not sum to one for {sample_id}")
        graph = torch.load(graph_path, map_location=map_location, weights_only=False)
        if int(graph["null"].x.shape[0]) != 1:
            raise ValueError(f"graph must contain one null node: {sample_id}")
        if int(graph["proposal"].x.shape[0]) != proposal_count:
            raise ValueError(f"graph proposal count mismatch: {sample_id}")
        proposal_scores = tuple(
            float(value) for value in graph["proposal"].proposal_score.tolist()
        )
        if len(proposal_scores) != proposal_count:
            raise ValueError(f"proposal score count mismatch: {sample_id}")
        interaction_group = scenario_interaction_group(
            row["scenario"], interaction_config
        )
        descriptor_risk = graph_descriptor_risk_positive(graph)
        examples.append(
            Stage1Example(
                sample_id=sample_id,
                state_group_id=state_group_id,
                scenario=row["scenario"],
                seed=seed,
                timestep=int(row["timestep"]),
                ego_agent_id=int(row["ego_agent_id"]),
                split=split,
                graph_path=graph_path,
                label_path=label_path,
                class_count=class_count,
                proposal_count=proposal_count,
                soft_target=soft_target,
                target_quality=target_quality,
                fp_shep_quality=fp_shep_quality,
                proposal_scores=proposal_scores,
                interaction_group=interaction_group,
                descriptor_risk_positive=descriptor_risk,
                graph=graph,
            )
        )
        split_rows.append(
            {
                "sample_id": sample_id,
                "state_group_id": state_group_id,
                "scenario": row["scenario"],
                "seed": seed,
                "timestep": int(row["timestep"]),
                "ego_agent_id": int(row["ego_agent_id"]),
                "split": split,
                "class_count": class_count,
                "proposal_count": proposal_count,
            }
        )
        class_histogram[class_count] += 1
        scenario_counts[row["scenario"]] += 1
        split_counts[split] += 1

    expected_splits = {"train", "validation", "test"}
    if set(split_counts) != expected_splits:
        raise ValueError(f"missing dataset split: {expected_splits - set(split_counts)}")
    dataset_audit = {
        "schema_version": SCHEMA_VERSION,
        "dataset_dir": str(dataset_dir),
        "graph_count": len(examples),
        "state_group_count": len(group_to_split),
        "group_key": split_config["group_key"],
        "group_leakage_count": 0,
        "split_graph_counts": dict(sorted(split_counts.items())),
        "scenario_graph_counts": dict(sorted(scenario_counts.items())),
        "class_count_histogram": {
            str(key): value for key, value in sorted(class_histogram.items())
        },
        "minimum_class_count": min(class_histogram),
        "maximum_class_count": max(class_histogram),
        "variable_K": len(class_histogram) > 1,
        "batching": supervision_config["batching"],
        "padding_used_in_training": False,
        "invalid_class_probability_mass_allowed": False,
        "H_label": h_label,
        "H_preview": int(supervision_config["H_preview"]),
        "target_name": target_name,
        "soft_target_temperature": float(
            supervision_config["soft_target_temperature"]
        ),
        "SELECT_LABEL_INTERACTION_AWARE": "PARTIAL",
        "select_label_interaction_rationale": (
            "real per-candidate rollout and failure tiers include observed inter-agent "
            "effects, but labels do not enumerate joint candidate combinations and are "
            "not a complete multi-agent coordination target"
        ),
        "claim_scope": "GAT learns candidate ranking",
    }
    return examples, dataset_audit, split_rows


def masked_soft_target_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Soft CE for padded inputs, masking invalid classes before softmax.

    The actual Stage-I loader uses ragged concatenation and therefore creates no
    padding.  This explicit implementation defines and tests the required
    behavior if padded tensors are supplied by another caller.
    """

    if logits.ndim != 2 or targets.shape != logits.shape or valid_mask.shape != logits.shape:
        raise ValueError("logits, targets and valid_mask must share shape [B, C]")
    mask = valid_mask.to(dtype=torch.bool, device=logits.device)
    targets = targets.to(dtype=logits.dtype, device=logits.device)
    if not torch.all(mask.any(dim=1)):
        raise ValueError("every graph must contain at least one valid class")
    if torch.any(targets < 0.0) or not torch.isfinite(targets).all():
        raise ValueError("targets must be finite and non-negative")
    if torch.any(targets.masked_select(~mask) != 0.0):
        raise ValueError("invalid/padded classes must have zero target mass")
    valid_sums = (targets * mask).sum(dim=1)
    if not torch.allclose(valid_sums, torch.ones_like(valid_sums), atol=1.0e-6):
        raise ValueError("target mass over valid classes must sum to one")
    masked_logits = logits.masked_fill(~mask, -torch.inf)
    log_probabilities = torch.log_softmax(masked_logits, dim=1)
    safe_log_probabilities = torch.where(mask, log_probabilities, torch.zeros_like(log_probabilities))
    return -(targets * safe_log_probabilities).sum(dim=1).mean()


def ragged_soft_target_cross_entropy(
    logits: torch.Tensor,
    candidate_ptr: torch.Tensor,
    targets: Sequence[torch.Tensor | Sequence[float]],
) -> torch.Tensor:
    """Per-graph soft CE over each graph's real null + K proposal classes."""

    if logits.ndim != 1 or candidate_ptr.ndim != 1:
        raise ValueError("ragged logits/ptr must be one-dimensional")
    if len(targets) != int(candidate_ptr.numel() - 1):
        raise ValueError("one target vector is required per graph")
    losses: list[torch.Tensor] = []
    for graph_index, raw_target in enumerate(targets):
        start = int(candidate_ptr[graph_index].item())
        stop = int(candidate_ptr[graph_index + 1].item())
        local_logits = logits[start:stop]
        target = torch.as_tensor(
            raw_target, dtype=logits.dtype, device=logits.device
        )
        if local_logits.numel() <= 0 or target.shape != local_logits.shape:
            raise ValueError("ragged target must match the graph's real class count")
        if torch.any(target < 0.0) or not torch.isfinite(target).all():
            raise ValueError("target values must be finite and non-negative")
        if not torch.allclose(target.sum(), target.new_tensor(1.0), atol=1.0e-6):
            raise ValueError("per-graph target mass must sum to one")
        losses.append(-(target * torch.log_softmax(local_logits, dim=0)).sum())
    return torch.stack(losses).mean()


def empirical_random_metrics(class_counts: Sequence[int]) -> dict[str, float]:
    if not class_counts or any(int(count) <= 0 for count in class_counts):
        raise ValueError("positive per-graph class counts are required")
    top1 = [1.0 / int(count) for count in class_counts]
    top3 = [min(3, int(count)) / int(count) for count in class_counts]
    mrr = [
        sum(1.0 / rank for rank in range(1, int(count) + 1)) / int(count)
        for count in class_counts
    ]
    return {
        "top1_accuracy": float(np.mean(top1)),
        "top3_accuracy": float(np.mean(top3)),
        "mrr": float(np.mean(mrr)),
    }


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks, including deterministic tie handling."""

    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        average = 0.5 * ((start + 1) + stop)
        ranks[order[start:stop]] = average
        start = stop
    return ranks


def spearman_coefficient(left: Sequence[float], right: Sequence[float]) -> float:
    """Spearman correlation without a SciPy/OpenMP runtime dependency."""

    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if left_array.shape != right_array.shape or left_array.ndim != 1:
        raise ValueError("Spearman inputs must be equal-length vectors")
    if len(left_array) < 2:
        return math.nan
    left_rank = _average_ranks(left_array)
    right_rank = _average_ranks(right_array)
    left_centered = left_rank - left_rank.mean()
    right_centered = right_rank - right_rank.mean()
    denominator = float(
        np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    )
    if denominator <= 0.0:
        return math.nan
    return float(np.dot(left_centered, right_centered) / denominator)


def compute_offline_metrics(
    examples: Sequence[Stage1Example],
    score_vectors: Sequence[Sequence[float] | np.ndarray],
    *,
    selectable_masks: Sequence[Sequence[bool] | np.ndarray] | None = None,
    loss: float | None = None,
) -> dict[str, Any]:
    """Compute the metric definitions frozen in the Stage-I config."""

    if len(examples) != len(score_vectors):
        raise ValueError("examples and score vectors must have equal length")
    if selectable_masks is None:
        selectable_masks = [np.ones(item.class_count, dtype=bool) for item in examples]
    if len(selectable_masks) != len(examples):
        raise ValueError("one selectable mask is required per graph")
    correct_top1 = 0
    correct_top3 = 0
    reciprocal_ranks: list[float] = []
    references: list[int] = []
    predictions: list[int] = []
    proposal_condition_correct = 0
    proposal_condition_count = 0
    proposal_spearman: list[float] = []
    fixed_rank_counts: Counter[int] = Counter()

    for example, raw_scores, raw_mask in zip(examples, score_vectors, selectable_masks):
        scores = np.asarray(raw_scores, dtype=float)
        mask = np.asarray(raw_mask, dtype=bool)
        if scores.shape != (example.class_count,) or mask.shape != scores.shape:
            raise ValueError(f"invalid score/mask shape for {example.sample_id}")
        if not np.isfinite(scores[mask]).all() or not mask.any():
            raise ValueError(f"invalid selectable scores for {example.sample_id}")
        reference = example.reference_class
        selectable_indices = np.flatnonzero(mask)
        order = selectable_indices[
            np.argsort(-scores[selectable_indices], kind="stable")
        ]
        prediction = int(order[0])
        references.append(reference)
        predictions.append(prediction)
        correct_top1 += int(prediction == reference)
        top_m = order[: min(3, example.class_count)]
        correct_top3 += int(reference in set(int(value) for value in top_m))
        if mask[reference]:
            rank_positions = np.flatnonzero(order == reference)
            reciprocal_ranks.append(
                1.0 / float(int(rank_positions[0]) + 1)
                if rank_positions.size
                else 0.0
            )
        else:
            reciprocal_ranks.append(0.0)
        if reference != 0:
            proposal_condition_count += 1
            proposal_condition_correct += int(prediction == reference)
        if prediction != 0:
            fixed_rank_counts[prediction - 1] += 1
        if example.proposal_count >= 2:
            coefficient = spearman_coefficient(
                scores[1:], np.asarray(example.target_quality[1:], dtype=float)
            )
            if math.isfinite(coefficient):
                proposal_spearman.append(coefficient)

    graph_count = len(examples)
    target_null = [value == 0 for value in references]
    predicted_null = [value == 0 for value in predictions]
    null_tp = sum(t and p for t, p in zip(target_null, predicted_null))
    null_fp = sum((not t) and p for t, p in zip(target_null, predicted_null))
    null_fn = sum(t and (not p) for t, p in zip(target_null, predicted_null))
    maximum_fixed_rank_rate = (
        max(fixed_rank_counts.values(), default=0) / graph_count if graph_count else 0.0
    )
    result = {
        "graph_count": graph_count,
        "loss": None if loss is None else float(loss),
        "top1_accuracy": _safe_ratio(correct_top1, graph_count),
        "top3_accuracy": _safe_ratio(correct_top3, graph_count),
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else None,
        "spearman_proposal_only_mean": (
            float(np.mean(proposal_spearman)) if proposal_spearman else None
        ),
        "spearman_valid_graph_count": len(proposal_spearman),
        "spearman_excludes_null": True,
        "spearman_minimum_proposal_count": 2,
        "null_precision": _safe_ratio(null_tp, null_tp + null_fp),
        "null_recall": _safe_ratio(null_tp, null_tp + null_fn),
        "null_reference_rate": float(np.mean(target_null)) if target_null else None,
        "null_prediction_rate": float(np.mean(predicted_null)) if predicted_null else None,
        "proposal_selection_accuracy": _safe_ratio(
            proposal_condition_correct, proposal_condition_count
        ),
        "proposal_reference_graph_count": proposal_condition_count,
        "maximum_fixed_proposal_rank_prediction_rate": float(
            maximum_fixed_rank_rate
        ),
        "fixed_proposal_rank_prediction_counts": {
            str(key): value for key, value in sorted(fixed_rank_counts.items())
        },
        "reference_classes": references,
        "predicted_classes": predictions,
    }
    result.update(
        {
            f"empirical_random_{name}": value
            for name, value in empirical_random_metrics(
                [item.class_count for item in examples]
            ).items()
        }
    )
    return result


def proposal_score_vectors(
    examples: Sequence[Stage1Example],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    scores: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for example in examples:
        if example.class_count == 1:
            scores.append(np.asarray([0.0], dtype=float))
            masks.append(np.asarray([True], dtype=bool))
        else:
            scores.append(
                np.asarray([-np.inf, *example.proposal_scores], dtype=float)
            )
            masks.append(
                np.asarray([False, *([True] * example.proposal_count)], dtype=bool)
            )
    return scores, masks


def fp_shep_score_vectors(examples: Sequence[Stage1Example]) -> list[np.ndarray]:
    return [np.asarray(item.fp_shep_quality, dtype=float) for item in examples]


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(requested: str) -> torch.device:
    normalized = str(requested).lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _iter_batches(
    examples: Sequence[Stage1Example],
    batch_size: int,
    *,
    shuffle_seed: int | None,
) -> Iterable[list[Stage1Example]]:
    indices = np.arange(len(examples))
    if shuffle_seed is not None:
        np.random.default_rng(int(shuffle_seed)).shuffle(indices)
    for start in range(0, len(indices), int(batch_size)):
        yield [examples[int(index)] for index in indices[start : start + batch_size]]


def _batch_loss(
    model: PolicyPreviewEdgeEnhancedGATSelector,
    batch_examples: Sequence[Stage1Example],
    device: torch.device,
) -> tuple[torch.Tensor, Any]:
    batch = batch_candidate_graphs([item.graph for item in batch_examples]).to(device)
    output = model(batch)
    loss = ragged_soft_target_cross_entropy(
        output.candidate_logits,
        output.candidate_ptr,
        [item.soft_target for item in batch_examples],
    )
    return loss, output


@torch.no_grad()
def evaluate_model(
    model: PolicyPreviewEdgeEnhancedGATSelector,
    examples: Sequence[Stage1Example],
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, Any], list[np.ndarray]]:
    model.eval()
    losses: list[float] = []
    weights: list[int] = []
    score_vectors: list[np.ndarray] = []
    for batch_examples in _iter_batches(examples, batch_size, shuffle_seed=None):
        loss, output = _batch_loss(model, batch_examples, device)
        losses.append(float(loss.item()))
        weights.append(len(batch_examples))
        for graph_index in range(output.graph_count):
            score_vectors.append(
                output.logits_for_graph(graph_index).detach().cpu().numpy()
            )
    mean_loss = float(np.average(losses, weights=weights))
    return (
        compute_offline_metrics(examples, score_vectors, loss=mean_loss),
        score_vectors,
    )


def _checkpoint_payload(
    model: PolicyPreviewEdgeEnhancedGATSelector,
    optimizer: torch.optim.Optimizer,
    *,
    optimization_seed: int,
    epoch: int,
    validation_metrics: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "optimization_seed": int(optimization_seed),
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation_metrics": dict(validation_metrics),
        "model_config": dict(config["model"]),
        "supervision": dict(config["supervision"]),
        "offline_metric_definitions": dict(config["offline_metric_definitions"]),
    }


def train_one_seed(
    config: Mapping[str, Any],
    train_examples: Sequence[Stage1Example],
    validation_examples: Sequence[Stage1Example],
    *,
    optimization_seed: int,
    checkpoint_dir: Path,
) -> TrainingRunResult:
    seed_everything(optimization_seed)
    training = config["training"]
    device = resolve_device(training["device"])
    model = PolicyPreviewEdgeEnhancedGATSelector(
        EdgeEnhancedGATConfig.from_mapping(config["model"])
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    batch_size = int(training["batch_size"])
    max_epochs = int(training["max_epochs"])
    patience = int(training["early_stopping_patience"])
    gradient_clip = float(training["gradient_clip_norm"])
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / f"seed_{optimization_seed:03d}_best_validation.pt"
    last_path = checkpoint_dir / f"seed_{optimization_seed:03d}_last.pt"
    history: list[EpochRecord] = []
    best_loss = math.inf
    best_epoch = 0
    best_metrics: dict[str, Any] = {}
    epochs_without_improvement = 0
    started = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        train_losses: list[float] = []
        train_weights: list[int] = []
        for batch_index, batch_examples in enumerate(
            _iter_batches(
                train_examples,
                batch_size,
                shuffle_seed=optimization_seed * 100_000 + epoch,
            )
        ):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = _batch_loss(model, batch_examples, device)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite loss at seed={optimization_seed}, epoch={epoch}, "
                    f"batch={batch_index}"
                )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            train_losses.append(float(loss.item()))
            train_weights.append(len(batch_examples))
        train_loss = float(np.average(train_losses, weights=train_weights))
        validation_metrics, _ = evaluate_model(
            model,
            validation_examples,
            batch_size=batch_size,
            device=device,
        )
        validation_loss = float(validation_metrics["loss"])
        improved = validation_loss < best_loss - 1.0e-12
        if improved:
            best_loss = validation_loss
            best_epoch = epoch
            best_metrics = dict(validation_metrics)
            epochs_without_improvement = 0
            torch.save(
                _checkpoint_payload(
                    model,
                    optimizer,
                    optimization_seed=optimization_seed,
                    epoch=epoch,
                    validation_metrics=validation_metrics,
                    config=config,
                ),
                best_path,
            )
        else:
            epochs_without_improvement += 1
        history.append(
            EpochRecord(
                optimization_seed=int(optimization_seed),
                epoch=epoch,
                train_loss=train_loss,
                validation_loss=validation_loss,
                validation_top1=float(validation_metrics["top1_accuracy"]),
                validation_top3=float(validation_metrics["top3_accuracy"]),
                validation_mrr=float(validation_metrics["mrr"]),
                validation_random_top1=float(
                    validation_metrics["empirical_random_top1_accuracy"]
                ),
                validation_null_prediction_rate=float(
                    validation_metrics["null_prediction_rate"]
                ),
                validation_fixed_rank_max_rate=float(
                    validation_metrics[
                        "maximum_fixed_proposal_rank_prediction_rate"
                    ]
                ),
                learning_rate=float(optimizer.param_groups[0]["lr"]),
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
            validation_metrics=validation_metrics,
            config=config,
        ),
        last_path,
    )
    return TrainingRunResult(
        optimization_seed=int(optimization_seed),
        best_epoch=best_epoch,
        epochs_completed=len(history),
        early_stopped=len(history) < max_epochs,
        best_validation_loss=best_loss,
        first_train_loss=float(history[0].train_loss),
        best_train_loss=float(min(item.train_loss for item in history)),
        first_validation_loss=float(history[0].validation_loss),
        best_validation_metrics=best_metrics,
        best_checkpoint=best_path,
        last_checkpoint=last_path,
        history=tuple(history),
        runtime_s=float(time.perf_counter() - started),
    )


def smoke_gate_result(
    result: TrainingRunResult, smoke_config: Mapping[str, Any]
) -> dict[str, Any]:
    train_improvement = (
        result.first_train_loss - result.best_train_loss
    ) / max(abs(result.first_train_loss), 1.0e-12)
    validation_improvement = (
        result.first_validation_loss - result.best_validation_loss
    ) / max(abs(result.first_validation_loss), 1.0e-12)
    metrics = result.best_validation_metrics
    top1_gain = float(metrics["top1_accuracy"]) - float(
        metrics["empirical_random_top1_accuracy"]
    )
    finite_values = [
        result.first_train_loss,
        result.best_train_loss,
        result.first_validation_loss,
        result.best_validation_loss,
        float(metrics["top1_accuracy"]),
        float(metrics["mrr"]),
    ]
    checks = {
        "finite_metrics": all(math.isfinite(value) for value in finite_values),
        "train_loss_decreased": train_improvement
        >= float(smoke_config["minimum_train_loss_improvement_fraction"]),
        "validation_loss_decreased": validation_improvement
        >= float(smoke_config["minimum_validation_loss_improvement_fraction"]),
        "top1_above_empirical_random": top1_gain
        >= float(smoke_config["minimum_top1_gain_over_empirical_random"]),
        "no_null_collapse": float(metrics["null_prediction_rate"])
        < float(smoke_config["maximum_null_prediction_rate"]),
        "no_fixed_rank_collapse": float(
            metrics["maximum_fixed_proposal_rank_prediction_rate"]
        )
        < float(smoke_config["maximum_fixed_proposal_rank_prediction_rate"]),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "train_loss_improvement_fraction": float(train_improvement),
        "validation_loss_improvement_fraction": float(validation_improvement),
        "validation_top1_gain_over_empirical_random": float(top1_gain),
        "best_epoch": result.best_epoch,
    }


def load_model_checkpoint(
    checkpoint_path: Path,
    config: Mapping[str, Any],
    device: torch.device,
) -> PolicyPreviewEdgeEnhancedGATSelector:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = PolicyPreviewEdgeEnhancedGATSelector(
        EdgeEnhancedGATConfig.from_mapping(config["model"])
    ).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model


def completion_signal_audit(
    dataset_dir: Path,
    completion_artifact_dir: Path,
    *,
    h_label: int,
) -> dict[str, Any]:
    """Join only already-executed branches; never synthesize completion labels."""

    dataset_dir = Path(dataset_dir)
    completion_artifact_dir = Path(completion_artifact_dir)
    all_horizon_class_rows = [
        row
        for row in read_csv(dataset_dir / "candidate_class_records.csv")
        if int(row["H_label"]) == int(h_label)
    ]
    class_rows = [row for row in all_horizon_class_rows if int(row["timestep"]) == 0]
    index: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in class_rows:
        index[(row["scenario"], row["seed"], row["ego_agent_id"])].append(row)
    observations: dict[
        tuple[tuple[str, str, str], int], list[tuple[str, bool]]
    ] = defaultdict(list)
    matched_execution_observations = 0
    for row in read_csv(completion_artifact_dir / "per_agent.csv"):
        key = (row["scenario"], row["seed"], row["agent_id"])
        if key not in index:
            continue
        method = row["method"]
        if method == "terminal_goal_baseline":
            observations[(key, 0)].append(
                (method, bool(row["terminal_completed_step"]))
            )
            matched_execution_observations += 1
            continue
        if method not in {"proposal_top1_one_shot", "fp_shep_top1_one_shot"}:
            continue
        selected = np.asarray(json.loads(row["selected_world_reference"]), dtype=float)
        matches = []
        for candidate_row in index[key]:
            candidate = np.asarray(
                json.loads(candidate_row["candidate_xyz"]), dtype=float
            )
            if float(np.linalg.norm(selected - candidate)) < 1.0e-6:
                matches.append(candidate_row)
        if len(matches) != 1:
            continue
        class_index = int(matches[0]["class_index"])
        completion = (
            row["reference_reached"] == "True"
            and row["terminal_completed_after_reference"] == "True"
        )
        observations[(key, class_index)].append((method, completion))
        matched_execution_observations += 1

    reliable: dict[tuple[tuple[str, str, str], int], bool] = {}
    conflicts: dict[str, Any] = {}
    for branch, values in observations.items():
        labels = {value for _, value in values}
        if len(labels) == 1:
            reliable[branch] = next(iter(labels))
        else:
            conflicts[f"{branch[0]}::class_{branch[1]}"] = [
                {"method": method, "completion": value}
                for method, value in values
            ]
    per_graph: dict[tuple[str, str, str], list[bool]] = defaultdict(list)
    scenario_labels: dict[str, list[bool]] = defaultdict(list)
    for (key, _), label in reliable.items():
        per_graph[key].append(label)
        scenario_labels[key[0]].append(label)
    graph_distribution = Counter()
    for values in per_graph.values():
        graph_distribution[
            "all_positive"
            if all(values)
            else "all_negative"
            if not any(values)
            else "mixed_label"
        ] += 1
    unique_dataset_classes = len(all_horizon_class_rows)
    positive = sum(reliable.values())
    negative = len(reliable) - positive
    return {
        "schema_version": SCHEMA_VERSION,
        "audit_scope": "existing_artifacts_only_no_new_rollout",
        "COMPLETION_LABEL_AVAILABLE": "PARTIAL",
        "CANDIDATE_LEVEL_COMPLETION_ATTRIBUTION": "NOT_RELIABLE",
        "matched_execution_observation_count": matched_execution_observations,
        "unique_observed_branch_count": len(observations),
        "reliable_branch_count": len(reliable),
        "conflicting_branch_count": len(conflicts),
        "positive_count": positive,
        "negative_count": negative,
        "positive_rate": _safe_ratio(positive, len(reliable)),
        "dataset_class_count_at_H_label": unique_dataset_classes,
        "class_coverage_rate": _safe_ratio(len(reliable), unique_dataset_classes),
        "graph_with_any_completion_label_count": len(per_graph),
        "graph_label_distribution": dict(sorted(graph_distribution.items())),
        "scenario_positive_rates": {
            scenario: {
                "label_count": len(values),
                "positive_count": sum(values),
                "positive_rate": _safe_ratio(sum(values), len(values)),
            }
            for scenario, values in sorted(scenario_labels.items())
        },
        "conflicts": conflicts,
        "team_failure_copied_to_all_candidates": False,
        "completion_loss_constructed": False,
        "attribution_rationale": (
            "coverage is sparse and selected-branch-only; identical graph/candidate "
            "branches have conflicting outcomes under different joint selector contexts"
        ),
    }


def training_result_record(result: TrainingRunResult) -> dict[str, Any]:
    record = asdict(result)
    record["best_checkpoint"] = str(result.best_checkpoint)
    record["last_checkpoint"] = str(result.last_checkpoint)
    record.pop("history", None)
    return record
