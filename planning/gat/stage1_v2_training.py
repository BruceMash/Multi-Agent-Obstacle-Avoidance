"""Stage-I GAT V2 training on deployment-aligned graph/target artifacts.

This module consumes the frozen ``supervision_v2_dataset.pt`` artifact.  It
does not generate labels, execute an environment, or change the GAT schema.
Every optimization seed starts from the same Stage-I V1 model state and a new
AdamW optimizer.
"""

from __future__ import annotations

import copy
import hashlib
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from planning.gat.candidate_selector import (
    EdgeEnhancedGATConfig,
    PolicyPreviewEdgeEnhancedGATSelector,
)
from planning.gat.stage1_training import (
    _batch_loss,
    _iter_batches,
    compute_offline_metrics,
    evaluate_model,
    graph_descriptor_risk_positive,
    scenario_interaction_group,
    seed_everything,
)
from planning.gat.typed_encoders import EDGE_INPUT_DIMS, NODE_INPUT_DIMS
from planning.gat_supervision_v2 import pairwise_ordering_agreement


SCHEMA_VERSION = "gat_stage1_v2_training"
DATASET_SCHEMA = "gat_supervision_v2"
TARGET_FIELDS = {
    "v1_scalar_h6": ("v1_scalar_h6_soft_target", "v1_scalar_h6_utility"),
    "v1_historical_h6": (
        "v1_historical_h6_soft_target",
        "v1_historical_h6_utility",
    ),
    "v2_long_horizon": (
        "v2_long_horizon_soft_target",
        "v2_long_horizon_utility",
    ),
}


@dataclass(frozen=True)
class Stage1V2Example:
    sample_id: str
    state_group_id: str
    scenario: str
    seed: int
    timestep: int
    ego_agent_id: int
    split: str
    source_split: str
    class_count: int
    proposal_count: int
    soft_target: tuple[float, ...]
    target_quality: tuple[float, ...]
    proposal_scores: tuple[float, ...]
    interaction_group: str
    descriptor_risk_positive: bool
    graph: Any
    class_mapping: tuple[Mapping[str, Any], ...]
    v1_scalar_h6_soft_target: tuple[float, ...]
    v1_scalar_h6_utility: tuple[float, ...]
    v1_historical_h6_soft_target: tuple[float, ...]
    v1_historical_h6_utility: tuple[float, ...]
    v2_long_horizon_soft_target: tuple[float, ...]
    v2_long_horizon_utility: tuple[float, ...]
    v2_long_horizon_tier: tuple[int, ...]
    background_selector: str
    background_plan_hash: str
    non_ego_plan_hash: str

    @property
    def reference_class(self) -> int:
        return int(np.argmax(np.asarray(self.soft_target, dtype=float)))


@dataclass(frozen=True)
class V2TrainingRunResult:
    optimization_seed: int
    best_epoch: int
    epochs_completed: int
    early_stopped: bool
    best_validation_loss: float
    best_validation_metrics: Mapping[str, Any]
    best_checkpoint: Path
    last_checkpoint: Path
    history: tuple[Mapping[str, Any], ...]
    initialization_model_hash: str
    fresh_optimizer_state_at_start: bool
    gradients_finite: bool
    maximum_gradient_norm_pre_clip: float
    runtime_s: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_state_hash(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _expected_split(seed: int, split_config: Mapping[str, Any]) -> str:
    memberships = {
        "train": set(int(value) for value in split_config["train_seeds"]),
        "validation": set(
            int(value) for value in split_config["validation_seeds"]
        ),
        "test": set(int(value) for value in split_config["test_seeds"]),
    }
    matches = [name for name, values in memberships.items() if int(seed) in values]
    if len(matches) != 1:
        raise ValueError(f"seed {seed} belongs to {len(matches)} effective splits")
    return matches[0]


def _tuple_floats(value: Any, *, name: str) -> tuple[float, ...]:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite one-dimensional vector")
    return tuple(float(item) for item in array)


def _validate_soft_target(value: Sequence[float], *, name: str) -> None:
    array = np.asarray(value, dtype=float)
    if np.any(array < 0.0) or not np.isclose(
        float(array.sum()), 1.0, rtol=0.0, atol=1.0e-10
    ):
        raise ValueError(f"{name} must be a non-negative unit-mass distribution")


def _validate_graph_schema(graph: Any, *, sample_id: str, h_preview: int) -> None:
    expected_nodes = set(NODE_INPUT_DIMS)
    expected_edges = {
        ("agent", "smooth", "proposal"),
        ("align", "spatiotemporal", "proposal"),
    }
    if set(graph.node_types) != expected_nodes:
        raise ValueError(f"node schema mismatch for {sample_id}")
    if set(graph.edge_types) != expected_edges:
        raise ValueError(f"edge schema mismatch for {sample_id}")
    for node_type, dimension in NODE_INPUT_DIMS.items():
        if int(graph[node_type].x.shape[1]) != int(dimension):
            raise ValueError(f"node dimension mismatch for {sample_id}:{node_type}")
    for relation, dimension in EDGE_INPUT_DIMS.items():
        edge_type = (
            ("agent", "smooth", "proposal")
            if relation == "smooth"
            else ("align", "spatiotemporal", "proposal")
        )
        if int(graph[edge_type].edge_attr.shape[1]) != int(dimension):
            raise ValueError(f"edge dimension mismatch for {sample_id}:{relation}")
    metadata = graph.graph_metadata
    if int(metadata["H"]) != int(h_preview):
        raise ValueError(f"H_preview mismatch for {sample_id}")
    if metadata["feature_schema_version"] != "heterogeneous_candidate_graph_v1":
        raise ValueError(f"feature schema version mismatch for {sample_id}")


def load_v2_examples(
    dataset_path: Path,
    *,
    split_config: Mapping[str, Any],
    interaction_config: Mapping[str, Any],
    expected_h_preview: int = 4,
    expected_temperature: float = 0.25,
    map_location: str | torch.device = "cpu",
) -> tuple[list[Stage1V2Example], dict[str, Any]]:
    """Load only frozen V2 tensors and their effective split metadata."""

    dataset_path = Path(dataset_path).resolve()
    payload = torch.load(dataset_path, map_location=map_location, weights_only=False)
    if payload.get("schema_version") != DATASET_SCHEMA:
        raise ValueError("unexpected V2 dataset schema")
    if int(payload.get("H_preview", -1)) != int(expected_h_preview):
        raise ValueError("V2 dataset H_preview is not frozen at 4")
    if not np.isclose(
        float(payload.get("soft_target_temperature", math.nan)),
        float(expected_temperature),
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError("V2 dataset tau is not frozen at 0.25")

    examples: list[Stage1V2Example] = []
    split_counts: dict[str, int] = {"train": 0, "validation": 0, "test": 0}
    source_split_counts: dict[str, int] = {
        "train": 0,
        "validation": 0,
        "test": 0,
    }
    class_histogram: dict[int, int] = {}
    state_group_splits: dict[str, str] = {}
    for raw in payload["samples"]:
        seed = int(raw["seed"])
        split = str(raw["effective_split"])
        if split != _expected_split(seed, split_config):
            raise ValueError(f"effective split mismatch for seed {seed}")
        state_group_id = str(raw["state_group_id"])
        previous = state_group_splits.setdefault(state_group_id, split)
        if previous != split:
            raise ValueError(f"state-group split leakage: {state_group_id}")
        ego_id = int(raw["ego_id"])
        sample_id = f"{state_group_id}__ego{ego_id}"
        graph = raw["graph"]
        _validate_graph_schema(
            graph, sample_id=sample_id, h_preview=int(expected_h_preview)
        )
        proposal_count = int(graph["proposal"].num_nodes)
        class_count = proposal_count + 1
        if len(raw["class_mapping"]) != class_count:
            raise ValueError(f"class mapping mismatch for {sample_id}")
        if int(graph["null"].num_nodes) != 1:
            raise ValueError(f"null count mismatch for {sample_id}")
        target_values: dict[str, tuple[float, ...]] = {}
        for target_name, (soft_field, utility_field) in TARGET_FIELDS.items():
            soft = _tuple_floats(raw[soft_field], name=f"{sample_id}:{soft_field}")
            utility = _tuple_floats(
                raw[utility_field], name=f"{sample_id}:{utility_field}"
            )
            if len(soft) != class_count or len(utility) != class_count:
                raise ValueError(f"target length mismatch for {sample_id}:{target_name}")
            _validate_soft_target(soft, name=f"{sample_id}:{soft_field}")
            target_values[soft_field] = soft
            target_values[utility_field] = utility
        tiers = tuple(int(value) for value in raw["v2_long_horizon_tier"])
        if len(tiers) != class_count or any(value < 0 or value > 5 for value in tiers):
            raise ValueError(f"V2 tier mismatch for {sample_id}")
        v2_soft = target_values["v2_long_horizon_soft_target"]
        if int(raw["v2_hard_target"]) != int(np.argmax(v2_soft)):
            raise ValueError(f"V2 hard target mismatch for {sample_id}")
        class_mapping = tuple(copy.deepcopy(raw["class_mapping"]))
        for class_index, mapping in enumerate(class_mapping):
            if int(mapping["class_index"]) != class_index:
                raise ValueError(f"class order mismatch for {sample_id}")
        proposal_scores = tuple(
            float(value) for value in graph["proposal"].proposal_score.tolist()
        )
        if len(proposal_scores) != proposal_count:
            raise ValueError(f"proposal score mismatch for {sample_id}")
        scenario = str(raw["scenario"])
        source_split = str(raw["source_split"])
        example = Stage1V2Example(
            sample_id=sample_id,
            state_group_id=state_group_id,
            scenario=scenario,
            seed=seed,
            timestep=int(raw["sample_t"]),
            ego_agent_id=ego_id,
            split=split,
            source_split=source_split,
            class_count=class_count,
            proposal_count=proposal_count,
            soft_target=target_values["v2_long_horizon_soft_target"],
            target_quality=target_values["v2_long_horizon_utility"],
            proposal_scores=proposal_scores,
            interaction_group=scenario_interaction_group(
                scenario, interaction_config
            ),
            descriptor_risk_positive=graph_descriptor_risk_positive(graph),
            graph=graph,
            class_mapping=class_mapping,
            v1_scalar_h6_soft_target=target_values["v1_scalar_h6_soft_target"],
            v1_scalar_h6_utility=target_values["v1_scalar_h6_utility"],
            v1_historical_h6_soft_target=target_values[
                "v1_historical_h6_soft_target"
            ],
            v1_historical_h6_utility=target_values["v1_historical_h6_utility"],
            v2_long_horizon_soft_target=target_values[
                "v2_long_horizon_soft_target"
            ],
            v2_long_horizon_utility=target_values["v2_long_horizon_utility"],
            v2_long_horizon_tier=tiers,
            background_selector=str(raw["background_selector"]),
            background_plan_hash=str(raw["background_plan_hash"]),
            non_ego_plan_hash=str(raw["non_ego_plan_hash"]),
        )
        examples.append(example)
        split_counts[split] += 1
        source_split_counts[source_split] += 1
        class_histogram[class_count] = class_histogram.get(class_count, 0) + 1

    if split_counts != {"train": 342, "validation": 48, "test": 102}:
        raise ValueError(f"unexpected effective split counts: {split_counts}")
    if source_split_counts != {"train": 288, "validation": 102, "test": 102}:
        raise ValueError(f"unexpected source split provenance: {source_split_counts}")
    audit = {
        "schema_version": SCHEMA_VERSION,
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "graph_count": len(examples),
        "state_group_count": len(state_group_splits),
        "effective_split_graph_counts": split_counts,
        "source_split_graph_counts": source_split_counts,
        "effective_split_used_for_training": True,
        "source_split_used_for_training": False,
        "group_leakage_count": 0,
        "class_count_histogram": {
            str(key): value for key, value in sorted(class_histogram.items())
        },
        "minimum_class_count": min(class_histogram),
        "maximum_class_count": max(class_histogram),
        "variable_K": len(class_histogram) > 1,
        "H_preview": int(expected_h_preview),
        "soft_target_temperature": float(expected_temperature),
        "labels_regenerated": False,
    }
    return examples, audit


def examples_for_target(
    examples: Sequence[Stage1V2Example], target_name: str
) -> list[Stage1V2Example]:
    if target_name not in TARGET_FIELDS:
        raise ValueError(f"unknown target: {target_name}")
    soft_field, utility_field = TARGET_FIELDS[target_name]
    return [
        replace(
            item,
            soft_target=getattr(item, soft_field),
            target_quality=getattr(item, utility_field),
        )
        for item in examples
    ]


def load_initialized_model(
    checkpoint_path: Path,
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[PolicyPreviewEdgeEnhancedGATSelector, dict[str, Any], str]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    configured_model = {
        key: value for key, value in config["model"].items() if key != "residual"
    }
    if dict(payload["model_config"]) != configured_model:
        raise ValueError("V1 checkpoint model config differs from frozen V2 config")
    if config["model"].get("residual") is not True:
        raise ValueError("residual connection must remain enabled")
    model = PolicyPreviewEdgeEnhancedGATSelector(
        EdgeEnhancedGATConfig.from_mapping(config["model"])
    ).to(device)
    incompatible = model.load_state_dict(payload["model_state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("strict V1 checkpoint initialization failed")
    source_hash = model_state_hash(payload["model_state_dict"])
    loaded_hash = model_state_hash(model.state_dict())
    if source_hash != loaded_hash:
        raise RuntimeError("loaded V1 model state differs from checkpoint")
    return model, payload, source_hash


def _checkpoint_payload_v2(
    *,
    model: PolicyPreviewEdgeEnhancedGATSelector,
    optimizer: torch.optim.Optimizer,
    optimization_seed: int,
    epoch: int,
    validation_metrics: Mapping[str, Any],
    config: Mapping[str, Any],
    initialization_checkpoint: Path,
    initialization_model_hash: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "optimization_seed": int(optimization_seed),
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation_metrics": dict(validation_metrics),
        "model_config": {
            key: value for key, value in config["model"].items() if key != "residual"
        },
        "supervision": dict(config["supervision"]),
        "offline_metric_definitions": dict(config["offline_metric_definitions"]),
        "initialization_checkpoint": str(Path(initialization_checkpoint).resolve()),
        "initialization_model_hash": initialization_model_hash,
        "optimizer_initialized_fresh": True,
        "optimizer_state_inherited": False,
        "checkpoint_selection": "validation_loss_only",
    }


def train_one_seed_v2(
    config: Mapping[str, Any],
    train_examples: Sequence[Stage1V2Example],
    validation_examples: Sequence[Stage1V2Example],
    *,
    optimization_seed: int,
    initialization_checkpoint: Path,
    checkpoint_dir: Path,
    device: torch.device,
) -> V2TrainingRunResult:
    seed_everything(int(optimization_seed))
    model, _, initialization_hash = load_initialized_model(
        initialization_checkpoint, config, device
    )
    training = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    fresh_optimizer = len(optimizer.state) == 0
    if not fresh_optimizer or bool(training["inherit_optimizer_state"]):
        raise RuntimeError("V2 optimizer must start fresh without inherited state")
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / f"seed_{int(optimization_seed):03d}_best_validation.pt"
    last_path = checkpoint_dir / f"seed_{int(optimization_seed):03d}_last.pt"
    best_loss = math.inf
    best_epoch = 0
    best_metrics: dict[str, Any] = {}
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    gradients_finite = True
    maximum_gradient_norm = 0.0
    started = time.perf_counter()

    for epoch in range(1, int(training["max_epochs"]) + 1):
        epoch_started = time.perf_counter()
        model.train()
        train_losses: list[float] = []
        train_weights: list[int] = []
        epoch_max_gradient_norm = 0.0
        for batch_index, batch_examples in enumerate(
            _iter_batches(
                train_examples,
                int(training["batch_size"]),
                shuffle_seed=int(optimization_seed) * 100_000 + epoch,
            )
        ):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = _batch_loss(model, batch_examples, device)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite V2 loss seed={optimization_seed} "
                    f"epoch={epoch} batch={batch_index}"
                )
            loss.backward()
            batch_gradients_finite = all(
                bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
                if parameter.grad is not None
            )
            gradients_finite &= batch_gradients_finite
            if not batch_gradients_finite:
                raise FloatingPointError(
                    f"non-finite V2 gradient seed={optimization_seed} "
                    f"epoch={epoch} batch={batch_index}"
                )
            gradient_norm = nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip_norm"])
            )
            gradient_norm_value = float(gradient_norm.item())
            if not math.isfinite(gradient_norm_value):
                raise FloatingPointError("non-finite gradient norm before clipping")
            epoch_max_gradient_norm = max(
                epoch_max_gradient_norm, gradient_norm_value
            )
            maximum_gradient_norm = max(maximum_gradient_norm, gradient_norm_value)
            optimizer.step()
            train_losses.append(float(loss.item()))
            train_weights.append(len(batch_examples))

        train_loss = float(np.average(train_losses, weights=train_weights))
        validation_metrics, _ = evaluate_model(
            model,
            validation_examples,
            batch_size=int(training["batch_size"]),
            device=device,
        )
        validation_loss = float(validation_metrics["loss"])
        if not math.isfinite(train_loss) or not math.isfinite(validation_loss):
            raise FloatingPointError("non-finite epoch-level V2 loss")
        improved = validation_loss < best_loss - 1.0e-12
        if improved:
            best_loss = validation_loss
            best_epoch = epoch
            best_metrics = dict(validation_metrics)
            epochs_without_improvement = 0
            torch.save(
                _checkpoint_payload_v2(
                    model=model,
                    optimizer=optimizer,
                    optimization_seed=optimization_seed,
                    epoch=epoch,
                    validation_metrics=validation_metrics,
                    config=config,
                    initialization_checkpoint=initialization_checkpoint,
                    initialization_model_hash=initialization_hash,
                ),
                best_path,
            )
        else:
            epochs_without_improvement += 1
        history.append(
            {
                "optimization_seed": int(optimization_seed),
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_top1": float(validation_metrics["top1_accuracy"]),
                "validation_top3": float(validation_metrics["top3_accuracy"]),
                "validation_mrr": float(validation_metrics["mrr"]),
                "validation_spearman": validation_metrics[
                    "spearman_proposal_only_mean"
                ],
                "validation_random_top1": float(
                    validation_metrics["empirical_random_top1_accuracy"]
                ),
                "validation_null_prediction_rate": float(
                    validation_metrics["null_prediction_rate"]
                ),
                "validation_fixed_rank_max_rate": float(
                    validation_metrics[
                        "maximum_fixed_proposal_rank_prediction_rate"
                    ]
                ),
                "gradients_finite": gradients_finite,
                "maximum_gradient_norm_pre_clip": epoch_max_gradient_norm,
                "gradient_clip_norm": float(training["gradient_clip_norm"]),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "epoch_runtime_s": float(time.perf_counter() - epoch_started),
                "improved": improved,
            }
        )
        if epochs_without_improvement >= int(training["early_stopping_patience"]):
            break

    torch.save(
        _checkpoint_payload_v2(
            model=model,
            optimizer=optimizer,
            optimization_seed=optimization_seed,
            epoch=len(history),
            validation_metrics=validation_metrics,
            config=config,
            initialization_checkpoint=initialization_checkpoint,
            initialization_model_hash=initialization_hash,
        ),
        last_path,
    )
    return V2TrainingRunResult(
        optimization_seed=int(optimization_seed),
        best_epoch=best_epoch,
        epochs_completed=len(history),
        early_stopped=len(history) < int(training["max_epochs"]),
        best_validation_loss=best_loss,
        best_validation_metrics=best_metrics,
        best_checkpoint=best_path,
        last_checkpoint=last_path,
        history=tuple(history),
        initialization_model_hash=initialization_hash,
        fresh_optimizer_state_at_start=fresh_optimizer,
        gradients_finite=gradients_finite,
        maximum_gradient_norm_pre_clip=maximum_gradient_norm,
        runtime_s=float(time.perf_counter() - started),
    )


def smoke_gate_result_v2(
    result: V2TrainingRunResult, smoke_config: Mapping[str, Any]
) -> dict[str, Any]:
    metrics = result.best_validation_metrics
    losses = [
        float(row["train_loss"])
        for row in result.history
    ] + [float(row["validation_loss"]) for row in result.history]
    top1_gain = float(metrics["top1_accuracy"]) - float(
        metrics["empirical_random_top1_accuracy"]
    )
    checks = {
        "finite_train_and_validation_loss": bool(losses)
        and all(math.isfinite(value) for value in losses),
        "finite_gradients": bool(result.gradients_finite),
        "top1_not_below_empirical_random": top1_gain
        >= float(smoke_config["minimum_top1_gain_over_empirical_random"]),
        "null_prediction_rate_below_95_percent": float(
            metrics["null_prediction_rate"]
        )
        < float(smoke_config["maximum_null_prediction_rate"]),
        "fixed_rank_collapse_absent": float(
            metrics["maximum_fixed_proposal_rank_prediction_rate"]
        )
        < float(smoke_config["maximum_fixed_proposal_rank_prediction_rate"]),
        "fresh_optimizer": bool(result.fresh_optimizer_state_at_start),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "validation_top1_gain_over_empirical_random": top1_gain,
        "best_epoch": result.best_epoch,
        "best_validation_loss": result.best_validation_loss,
        "automatic_adjustment_performed": False,
    }


def load_v2_checkpoint_model(
    checkpoint_path: Path,
    config: Mapping[str, Any],
    device: torch.device,
) -> PolicyPreviewEdgeEnhancedGATSelector:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected V2 checkpoint schema")
    model = PolicyPreviewEdgeEnhancedGATSelector(
        EdgeEnhancedGATConfig.from_mapping(config["model"])
    ).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model


def model_score_vectors(
    model: PolicyPreviewEdgeEnhancedGATSelector,
    examples: Sequence[Stage1V2Example],
    *,
    batch_size: int,
    device: torch.device,
) -> list[np.ndarray]:
    _, scores = evaluate_model(
        model, examples, batch_size=batch_size, device=device
    )
    return scores


def _cross_entropy_from_scores(
    examples: Sequence[Stage1V2Example], score_vectors: Sequence[np.ndarray]
) -> float:
    losses = []
    for example, scores in zip(examples, score_vectors, strict=True):
        logits = np.asarray(scores, dtype=float)
        target = np.asarray(example.soft_target, dtype=float)
        shifted = logits - float(np.max(logits))
        log_probabilities = shifted - math.log(float(np.sum(np.exp(shifted))))
        losses.append(float(-np.sum(target * log_probabilities)))
    return float(np.mean(losses))


def metrics_for_scores(
    examples: Sequence[Stage1V2Example], score_vectors: Sequence[np.ndarray]
) -> dict[str, Any]:
    return compute_offline_metrics(
        examples,
        score_vectors,
        loss=_cross_entropy_from_scores(examples, score_vectors),
    )


def target_agreement_analysis(
    examples: Sequence[Stage1V2Example],
) -> list[dict[str, Any]]:
    comparisons = (
        ("v1_scalar_h6", "v1_historical_h6"),
        ("v1_historical_h6", "v2_long_horizon"),
        ("v1_scalar_h6", "v2_long_horizon"),
    )
    rows = []
    for left_name, right_name in comparisons:
        left_examples = examples_for_target(examples, left_name)
        right_examples = examples_for_target(examples, right_name)
        top1 = []
        top3 = []
        pairwise = []
        for left, right in zip(left_examples, right_examples, strict=True):
            left_values = np.asarray(left.target_quality, dtype=float)
            right_values = np.asarray(right.target_quality, dtype=float)
            left_order = np.argsort(-left_values, kind="stable")
            right_order = np.argsort(-right_values, kind="stable")
            top1.append(int(left_order[0]) == int(right_order[0]))
            m = min(3, left.class_count)
            top3.append(
                len(set(left_order[:m].tolist()) & set(right_order[:m].tolist())) / m
            )
            pairwise.append(pairwise_ordering_agreement(left_values, right_values))
        rows.append(
            {
                "analysis": "intrinsic_target_agreement",
                "left": left_name,
                "right": right_name,
                "graph_count": len(examples),
                "top1_agreement": float(np.mean(top1)),
                "top3_overlap": float(np.mean(top3)),
                "pairwise_agreement": float(np.mean(pairwise)),
            }
        )
    return rows


def prediction_disagreement(
    first_scores: Sequence[np.ndarray], second_scores: Sequence[np.ndarray]
) -> dict[str, Any]:
    first = [int(np.argmax(value)) for value in first_scores]
    second = [int(np.argmax(value)) for value in second_scores]
    if len(first) != len(second):
        raise ValueError("prediction vectors must have equal graph count")
    disagreement = [left != right for left, right in zip(first, second, strict=True)]
    return {
        "analysis": "prediction_disagreement",
        "graph_count": len(first),
        "disagreement_count": int(sum(disagreement)),
        "disagreement_rate": float(np.mean(disagreement)),
        "agreement_rate": float(1.0 - np.mean(disagreement)),
    }


def long_horizon_selection_diagnostics(
    examples: Sequence[Stage1V2Example],
    score_vectors: Sequence[np.ndarray],
    outcome_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    lookup = {
        (str(row["sample_id"]), int(row["class_index"])): row
        for row in outcome_rows
    }
    maximum_tier_hits = []
    predicted_proposal_count = 0
    predicted_reference_reached = 0
    predicted_team_success = 0
    predicted_ego_terminal = 0
    post_indices: list[int] = []
    for graph_index, (example, scores) in enumerate(
        zip(examples, score_vectors, strict=True)
    ):
        prediction = int(np.argmax(np.asarray(scores, dtype=float)))
        maximum_tier_hits.append(
            int(example.v2_long_horizon_tier[prediction])
            == max(example.v2_long_horizon_tier)
        )
        selected = lookup.get((example.sample_id, prediction))
        if selected is None:
            raise ValueError(f"missing branch outcome {example.sample_id}:{prediction}")
        predicted_team_success += int(bool(selected["team_success"]))
        predicted_ego_terminal += int(bool(selected["ego_terminal_reached"]))
        if prediction != 0:
            predicted_proposal_count += 1
            predicted_reference_reached += int(
                bool(selected["ego_reference_reached"])
            )
        has_post_reference_signal = any(
            bool(lookup[(example.sample_id, class_index)]["ego_reference_reached"])
            and not bool(
                lookup[(example.sample_id, class_index)]["ego_terminal_reached"]
            )
            for class_index in range(1, example.class_count)
        )
        if has_post_reference_signal:
            post_indices.append(graph_index)
    post_examples = [examples[index] for index in post_indices]
    post_scores = [score_vectors[index] for index in post_indices]
    post_metrics = (
        metrics_for_scores(post_examples, post_scores) if post_examples else {}
    )
    graph_count = len(examples)
    return {
        "graph_count": graph_count,
        "maximum_tier_prediction_rate": float(np.mean(maximum_tier_hits)),
        "predicted_proposal_count": predicted_proposal_count,
        "selected_proposal_reference_reach_rate": (
            None
            if predicted_proposal_count == 0
            else float(predicted_reference_reached / predicted_proposal_count)
        ),
        "selected_team_success_rate": float(predicted_team_success / graph_count),
        "selected_ego_terminal_reach_rate": float(
            predicted_ego_terminal / graph_count
        ),
        "post_reference_signal_graph_count": len(post_indices),
        "post_reference_subset_top1": post_metrics.get("top1_accuracy"),
        "post_reference_subset_top3": post_metrics.get("top3_accuracy"),
        "post_reference_subset_mrr": post_metrics.get("mrr"),
        "post_reference_subset_spearman": post_metrics.get(
            "spearman_proposal_only_mean"
        ),
    }


def classify_gain(value: float, *, yes_threshold: float) -> str:
    if float(value) >= float(yes_threshold):
        return "YES"
    if float(value) > 0.0:
        return "WEAK"
    return "NO"

