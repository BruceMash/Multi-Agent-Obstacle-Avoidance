#!/usr/bin/env python3
"""Train GAT-RS with frozen secondary smoothness-pair supervision."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
for search_path in (REPO_ROOT, ALGO_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from planning.gat.candidate_selector import (  # noqa: E402
    EdgeEnhancedGATConfig,
    PolicyPreviewEdgeEnhancedGATSelector,
    batch_candidate_graphs,
)
from planning.gat.stage1_training import (  # noqa: E402
    Stage1Example,
    compute_offline_metrics,
    ragged_soft_target_cross_entropy,
    resolve_device,
    seed_everything,
    sha256_file,
)
from scripts.train_gat_recurrent_r import (  # noqa: E402
    build_examples,
    read_json,
    read_jsonl,
    write_csv,
    write_json,
)


SCHEMA_VERSION = "gat_recurrent_rs_training_v1"
DEFAULT_CONFIG = REPO_ROOT / "configs/training/gat_recurrent_rs.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def iter_batches(
    examples: Sequence[Stage1Example],
    batch_size: int,
    *,
    shuffle_seed: int | None,
) -> Iterable[list[Stage1Example]]:
    indices = np.arange(len(examples))
    if shuffle_seed is not None:
        np.random.default_rng(int(shuffle_seed)).shuffle(indices)
    for start in range(0, len(indices), int(batch_size)):
        yield [examples[int(index)] for index in indices[start : start + int(batch_size)]]


def build_pair_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, tuple[tuple[int, int], ...]]:
    result: dict[str, tuple[tuple[int, int], ...]] = {}
    for row in rows:
        pairs = tuple((int(pair[0]), int(pair[1])) for pair in row["smoothness_pairs"])
        if any(preferred == 0 or disfavored == 0 for preferred, disfavored in pairs):
            raise RuntimeError(f"null class found in smoothness pair: {row['state_id']}")
        if any(preferred == disfavored for preferred, disfavored in pairs):
            raise RuntimeError(f"self-pair found: {row['state_id']}")
        class_count = int(row["class_count"])
        if any(not (0 <= preferred < class_count and 0 <= disfavored < class_count) for preferred, disfavored in pairs):
            raise RuntimeError(f"out-of-range smoothness pair: {row['state_id']}")
        result[str(row["state_id"])] = pairs
    return result


def pair_loss_for_output(
    output: Any,
    batch_examples: Sequence[Stage1Example],
    pair_map: Mapping[str, tuple[tuple[int, int], ...]],
    *,
    margin: float,
) -> tuple[torch.Tensor, int]:
    state_losses: list[torch.Tensor] = []
    pair_count = 0
    for graph_index, example in enumerate(batch_examples):
        local = output.logits_for_graph(graph_index)
        pairs = pair_map[example.sample_id]
        if pairs:
            terms = [
                F.softplus(local[disfavored] - local[preferred] + float(margin))
                for preferred, disfavored in pairs
            ]
            state_losses.append(torch.stack(terms).mean())
            pair_count += len(terms)
        else:
            state_losses.append(local.sum() * 0.0)
    return torch.stack(state_losses).mean(), pair_count


def batch_loss(
    model: PolicyPreviewEdgeEnhancedGATSelector,
    batch_examples: Sequence[Stage1Example],
    pair_map: Mapping[str, tuple[tuple[int, int], ...]],
    device: torch.device,
    *,
    pair_weight: float,
    pair_margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any, int]:
    batch = batch_candidate_graphs([item.graph for item in batch_examples]).to(device)
    output = model(batch)
    primary = ragged_soft_target_cross_entropy(
        output.candidate_logits,
        output.candidate_ptr,
        [item.soft_target for item in batch_examples],
    )
    secondary, pair_count = pair_loss_for_output(
        output,
        batch_examples,
        pair_map,
        margin=pair_margin,
    )
    total = primary + float(pair_weight) * secondary
    return total, primary, secondary, output, pair_count


@torch.no_grad()
def evaluate(
    model: PolicyPreviewEdgeEnhancedGATSelector,
    examples: Sequence[Stage1Example],
    pair_map: Mapping[str, tuple[tuple[int, int], ...]],
    *,
    batch_size: int,
    device: torch.device,
    pair_weight: float,
    pair_margin: float,
) -> tuple[dict[str, Any], list[np.ndarray]]:
    model.eval()
    total_losses: list[float] = []
    primary_losses: list[float] = []
    secondary_losses: list[float] = []
    weights: list[int] = []
    score_vectors: list[np.ndarray] = []
    pair_correct = pair_total = 0
    pair_margins: list[float] = []
    for batch_examples in iter_batches(examples, batch_size, shuffle_seed=None):
        total, primary, secondary, output, _ = batch_loss(
            model,
            batch_examples,
            pair_map,
            device,
            pair_weight=pair_weight,
            pair_margin=pair_margin,
        )
        total_losses.append(float(total.item()))
        primary_losses.append(float(primary.item()))
        secondary_losses.append(float(secondary.item()))
        weights.append(len(batch_examples))
        for graph_index, example in enumerate(batch_examples):
            logits = output.logits_for_graph(graph_index).detach().cpu().numpy()
            score_vectors.append(logits)
            for preferred, disfavored in pair_map[example.sample_id]:
                difference = float(logits[preferred] - logits[disfavored])
                pair_margins.append(difference)
                pair_correct += int(difference > 0.0)
                pair_total += 1
    total_loss = float(np.average(total_losses, weights=weights))
    metrics = compute_offline_metrics(examples, score_vectors, loss=total_loss)
    metrics.update(
        {
            "total_loss": total_loss,
            "primary_soft_target_loss": float(np.average(primary_losses, weights=weights)),
            "secondary_smoothness_pair_loss": float(np.average(secondary_losses, weights=weights)),
            "smoothness_pair_count": pair_total,
            "smoothness_pair_accuracy": pair_correct / pair_total if pair_total else None,
            "smoothness_pair_margin_mean": float(np.mean(pair_margins)) if pair_margins else None,
            "smoothness_pair_margin_median": float(np.median(pair_margins)) if pair_margins else None,
        }
    )
    return metrics, score_vectors


def checkpoint_payload(
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


def train_seed(
    config: Mapping[str, Any],
    train_examples: Sequence[Stage1Example],
    validation_examples: Sequence[Stage1Example],
    pair_map: Mapping[str, tuple[tuple[int, int], ...]],
    *,
    optimization_seed: int,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    seed_everything(optimization_seed)
    training = config["training"]
    supervision = config["supervision"]
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
    pair_weight = float(supervision["smoothness_pairwise_weight"])
    pair_margin = float(supervision["smoothness_pairwise_margin"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / f"seed_{optimization_seed:03d}_best_validation.pt"
    last_path = checkpoint_dir / f"seed_{optimization_seed:03d}_last.pt"
    history: list[dict[str, Any]] = []
    best_loss = math.inf
    best_epoch = 0
    best_metrics: dict[str, Any] = {}
    epochs_without_improvement = 0
    started = time.perf_counter()
    validation_metrics: dict[str, Any] = {}

    for epoch in range(1, max_epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        totals: list[float] = []
        primaries: list[float] = []
        secondaries: list[float] = []
        weights: list[int] = []
        for batch_index, batch_examples in enumerate(
            iter_batches(
                train_examples,
                batch_size,
                shuffle_seed=optimization_seed * 100_000 + epoch,
            )
        ):
            optimizer.zero_grad(set_to_none=True)
            total, primary, secondary, _, _ = batch_loss(
                model,
                batch_examples,
                pair_map,
                device,
                pair_weight=pair_weight,
                pair_margin=pair_margin,
            )
            if not torch.isfinite(total):
                raise FloatingPointError(
                    f"non-finite loss seed={optimization_seed} epoch={epoch} batch={batch_index}"
                )
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            totals.append(float(total.item()))
            primaries.append(float(primary.item()))
            secondaries.append(float(secondary.item()))
            weights.append(len(batch_examples))
        train_total = float(np.average(totals, weights=weights))
        validation_metrics, _ = evaluate(
            model,
            validation_examples,
            pair_map,
            batch_size=batch_size,
            device=device,
            pair_weight=pair_weight,
            pair_margin=pair_margin,
        )
        validation_loss = float(validation_metrics["total_loss"])
        improved = validation_loss < best_loss - 1.0e-12
        if improved:
            best_loss = validation_loss
            best_epoch = epoch
            best_metrics = dict(validation_metrics)
            epochs_without_improvement = 0
            torch.save(
                checkpoint_payload(
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
            {
                "schema_version": SCHEMA_VERSION,
                "variant": "GAT-RS",
                "optimization_seed": int(optimization_seed),
                "epoch": epoch,
                "train_total_loss": train_total,
                "train_primary_loss": float(np.average(primaries, weights=weights)),
                "train_secondary_loss": float(np.average(secondaries, weights=weights)),
                "validation_total_loss": validation_loss,
                "validation_primary_loss": validation_metrics["primary_soft_target_loss"],
                "validation_secondary_loss": validation_metrics["secondary_smoothness_pair_loss"],
                "validation_top1": validation_metrics["top1_accuracy"],
                "validation_top3": validation_metrics["top3_accuracy"],
                "validation_mrr": validation_metrics["mrr"],
                "validation_pair_accuracy": validation_metrics["smoothness_pair_accuracy"],
                "validation_pair_margin_mean": validation_metrics["smoothness_pair_margin_mean"],
                "validation_null_prediction_rate": validation_metrics["null_prediction_rate"],
                "validation_fixed_rank_max_rate": validation_metrics["maximum_fixed_proposal_rank_prediction_rate"],
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "epoch_runtime_s": float(time.perf_counter() - epoch_started),
                "improved": improved,
            }
        )
        if epochs_without_improvement >= patience:
            break

    torch.save(
        checkpoint_payload(
            model,
            optimizer,
            optimization_seed=optimization_seed,
            epoch=len(history),
            validation_metrics=validation_metrics,
            config=config,
        ),
        last_path,
    )
    return {
        "optimization_seed": int(optimization_seed),
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "early_stopped": len(history) < max_epochs,
        "best_validation_loss": best_loss,
        "best_validation_metrics": best_metrics,
        "best_checkpoint": best_path,
        "last_checkpoint": last_path,
        "history": history,
        "runtime_s": float(time.perf_counter() - started),
    }


def preflight(
    config_path: Path,
    config: Mapping[str, Any],
    artifact_root: Path,
    rows: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    dataset = config["dataset"]
    reconciliation_path = artifact_root / str(dataset["reconciliation"])
    label_freeze_path = artifact_root / str(dataset["label_freeze"])
    examples_path = artifact_root / str(dataset["examples"])
    pairs_path = artifact_root / str(dataset["eligible_smoothness_pairs"])
    diagnostic_path = artifact_root / str(dataset["gat_r_dev_diagnostic"])
    reconciliation = read_json(reconciliation_path)
    diagnostic = read_json(diagnostic_path)
    if reconciliation["status"] != "PASS" or diagnostic["status"] != "PASS":
        raise RuntimeError("dataset or GAT-R diagnostic did not pass integrity checks")
    if int(reconciliation["eligible_smoothness_pair_count"]) != int(dataset["expected_pair_count"]):
        raise RuntimeError("reconciled smoothness pair count mismatch")
    if len(pair_rows) != int(dataset["expected_pair_count"]):
        raise RuntimeError("smoothness pair CSV count mismatch")
    row_index = {str(row["state_id"]): row for row in rows}
    pair_counter = Counter()
    partition_pair_counts = Counter()
    for pair in pair_rows:
        state_id = str(pair["state_id"])
        if state_id not in row_index:
            raise RuntimeError(f"smoothness pair state missing: {state_id}")
        if pair["safety_equivalent"] != "True" or pair["progress_equivalent"] != "True":
            raise RuntimeError(f"ineligible pair retained: {state_id}")
        if pair["null_involved"] != "False":
            raise RuntimeError(f"null-involving smoothness pair retained: {state_id}")
        preferred = int(pair["preferred_class_index"])
        disfavored = int(pair["disfavored_class_index"])
        pair_counter[(state_id, preferred, disfavored)] += 1
        partition_pair_counts[str(row_index[state_id]["partition"])] += 1
    json_counter = Counter(
        (str(row["state_id"]), int(pair[0]), int(pair[1]))
        for row in rows
        for pair in row["smoothness_pairs"]
    )
    if pair_counter != json_counter or any(value != 1 for value in pair_counter.values()):
        raise RuntimeError("JSONL and eligible smoothness pair CSV do not match exactly")
    state_counts = Counter(str(row["partition"]) for row in rows)
    scene_counts = {
        partition: len({str(row["scenario_id"]) for row in rows if row["partition"] == partition})
        for partition in ("train", "validation")
    }
    if scene_counts != {
        "train": int(dataset["train_scenes"]),
        "validation": int(dataset["validation_scenes"]),
    }:
        raise RuntimeError("scene split mismatch")
    supervision = config["supervision"]
    if not supervision["smoothness_pairwise_loss_enabled"]:
        raise RuntimeError("GAT-RS must enable smoothness pairwise supervision")
    if float(supervision["smoothness_pairwise_weight"]) != 0.1:
        raise RuntimeError("unexpected predeclared smoothness weight")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "FROZEN_BEFORE_TRAINING",
        "variant": "GAT-RS",
        "config": str(config_path.relative_to(REPO_ROOT).as_posix()),
        "config_sha256": sha256_file(config_path),
        "training_script": str(Path(__file__).resolve().relative_to(REPO_ROOT).as_posix()),
        "training_script_sha256": sha256_file(Path(__file__).resolve()),
        "examples": str(examples_path.relative_to(REPO_ROOT).as_posix()),
        "examples_sha256": sha256_file(examples_path),
        "eligible_smoothness_pairs": str(pairs_path.relative_to(REPO_ROOT).as_posix()),
        "eligible_smoothness_pairs_sha256": sha256_file(pairs_path),
        "reconciliation_sha256": sha256_file(reconciliation_path),
        "label_freeze_sha256": sha256_file(label_freeze_path),
        "gat_r_dev_diagnostic_sha256_order_evidence_only": sha256_file(diagnostic_path),
        "state_counts": dict(sorted(state_counts.items())),
        "scene_counts": scene_counts,
        "pair_counts": dict(sorted(partition_pair_counts.items())),
        "total_pair_count": len(pair_rows),
        "pair_csv_matches_jsonl": True,
        "all_pairs_safety_equivalent": True,
        "all_pairs_progress_equivalent": True,
        "null_pair_count": 0,
        "smoothness_pairwise_weight": 0.1,
        "smoothness_weight_search": False,
        "architecture_equal_to_gat_r": True,
        "initialized_from_gat_r": False,
        "runtime_graph_schema_changed": False,
        "runtime_goal_distance_scale_m": 100.0,
        "runtime_future_information_added": False,
        "dev_used_for_training_or_checkpoint_selection": False,
        "optimization_seeds": list(config["training"]["optimization_seeds"]),
        "checkpoint_selection": config["training"]["checkpoint_selection"],
    }


def verify_frozen(freeze: Mapping[str, Any], config_path: Path) -> None:
    checks = {
        "config": sha256_file(config_path) == freeze["config_sha256"],
        "script": sha256_file(Path(__file__).resolve()) == freeze["training_script_sha256"],
        "examples": sha256_file(REPO_ROOT / freeze["examples"]) == freeze["examples_sha256"],
        "pairs": sha256_file(REPO_ROOT / freeze["eligible_smoothness_pairs"]) == freeze["eligible_smoothness_pairs_sha256"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"GAT-RS pretraining freeze mismatch: {checks}")


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve() if args.config.is_absolute() else (REPO_ROOT / args.config).resolve()
    config = read_json(config_path)
    if config["schema_version"] != SCHEMA_VERSION:
        raise RuntimeError("unexpected GAT-RS config schema")
    artifact_root = (REPO_ROOT / str(config["artifact_root"])).resolve()
    output_dir = artifact_root / "06_gat_rs_training"
    output_dir.mkdir(parents=True, exist_ok=True)
    examples_path = artifact_root / str(config["dataset"]["examples"])
    pairs_path = artifact_root / str(config["dataset"]["eligible_smoothness_pairs"])
    rows = read_jsonl(examples_path)
    pair_rows = read_csv(pairs_path)
    freeze = preflight(config_path, config, artifact_root, rows, pair_rows)
    freeze_path = output_dir / "PRETRAINING_FREEZE.json"
    if args.preflight_only:
        if freeze_path.exists() and read_json(freeze_path) != freeze:
            raise RuntimeError("existing GAT-RS pretraining freeze differs")
        if not freeze_path.exists():
            write_json(freeze_path, freeze)
        print(json.dumps(freeze, indent=2))
        return
    if not freeze_path.exists():
        raise RuntimeError("run --preflight-only before GAT-RS training")
    frozen = read_json(freeze_path)
    verify_frozen(frozen, config_path)
    checkpoints = output_dir / "checkpoints"
    if checkpoints.exists():
        raise RuntimeError("GAT-RS checkpoint directory exists; refusing to overwrite")

    examples = build_examples(artifact_root, rows)
    pair_map = build_pair_map(rows)
    train_examples = [example for example in examples if example.split == "train"]
    validation_examples = [example for example in examples if example.split == "validation"]
    results = []
    for optimization_seed in config["training"]["optimization_seeds"]:
        print(f"[GAT-RS] training optimization seed {optimization_seed}", flush=True)
        results.append(
            train_seed(
                config,
                train_examples,
                validation_examples,
                pair_map,
                optimization_seed=int(optimization_seed),
                checkpoint_dir=checkpoints,
            )
        )
    selected = min(results, key=lambda item: (item["best_validation_loss"], item["optimization_seed"]))
    selected_path = checkpoints / "best_validation.pt"
    shutil.copy2(selected["best_checkpoint"], selected_path)
    device = resolve_device(config["training"]["device"])
    payload = torch.load(selected_path, map_location=device, weights_only=False)
    model = PolicyPreviewEdgeEnhancedGATSelector(
        EdgeEnhancedGATConfig.from_mapping(config["model"])
    ).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    split_metrics = []
    for split_name, split_examples in (("train", train_examples), ("validation", validation_examples)):
        metrics, _ = evaluate(
            model,
            split_examples,
            pair_map,
            batch_size=int(config["training"]["batch_size"]),
            device=device,
            pair_weight=float(config["supervision"]["smoothness_pairwise_weight"]),
            pair_margin=float(config["supervision"]["smoothness_pairwise_margin"]),
        )
        split_metrics.append({"schema_version": SCHEMA_VERSION, "variant": "GAT-RS", "split": split_name, **metrics})

    curves = [record for result in results for record in result["history"]]
    selection_rows = [
        {
            "schema_version": SCHEMA_VERSION,
            "variant": "GAT-RS",
            "optimization_seed": result["optimization_seed"],
            "best_epoch": result["best_epoch"],
            "epochs_completed": result["epochs_completed"],
            "early_stopped": result["early_stopped"],
            "best_validation_loss": result["best_validation_loss"],
            "best_validation_primary_loss": result["best_validation_metrics"]["primary_soft_target_loss"],
            "best_validation_secondary_loss": result["best_validation_metrics"]["secondary_smoothness_pair_loss"],
            "best_validation_pair_accuracy": result["best_validation_metrics"]["smoothness_pair_accuracy"],
            "best_validation_top1": result["best_validation_metrics"]["top1_accuracy"],
            "best_checkpoint": str(result["best_checkpoint"].relative_to(REPO_ROOT).as_posix()),
            "best_checkpoint_sha256": sha256_file(result["best_checkpoint"]),
            "runtime_s": result["runtime_s"],
            "selected": result is selected,
            "selection_criterion": "minimum internal validation total loss",
            "dev_used": False,
        }
        for result in results
    ]
    write_csv(output_dir / "training_curves_gat_rs.csv", curves)
    write_csv(output_dir / "checkpoint_selection_history.csv", selection_rows)
    write_csv(output_dir / "offline_training_metrics.csv", split_metrics)
    manifest = {
        **frozen,
        "status": "TRAINING_COMPLETE",
        "training_result_count": len(results),
        "selected_optimization_seed": selected["optimization_seed"],
        "selected_best_epoch": selected["best_epoch"],
        "selected_validation_loss": selected["best_validation_loss"],
        "selected_checkpoint": str(selected_path.relative_to(REPO_ROOT).as_posix()),
        "selected_checkpoint_sha256": sha256_file(selected_path),
        "selected_metrics": split_metrics,
        "smoothness_supervision_added": True,
        "smoothness_supervision_role": "SECONDARY",
        "gat_runtime_input_changed": False,
        "gat_runtime_role_changed": False,
        "core_runtime_theory_changed": False,
        "dev_used_for_training_or_checkpoint_selection": False,
    }
    write_json(output_dir / "GAT_RS_TRAINING_MANIFEST.json", manifest)
    print(
        json.dumps(
            {
                "selected_seed": selected["optimization_seed"],
                "best_epoch": selected["best_epoch"],
                "validation_loss": selected["best_validation_loss"],
                "checkpoint": str(selected_path),
                "checkpoint_sha256": sha256_file(selected_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
