#!/usr/bin/env python3
"""Fine-tune focal-safe FP-anchored GAT-R and GAT-RS selectors.

Both variants start from the same frozen GAT-V1 weights and use the same
long-range recurrent graphs and primary targets.  GAT-RS differs only by a
small secondary smoothness pair loss.  Runtime architecture and inputs are
unchanged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
)


SCHEMA = "gat_recurrent_fp_anchored_training_v2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["status"])
        writer.writeheader()
        writer.writerows(rows)


def batches(
    examples: Sequence[Stage1Example], batch_size: int, *, shuffle_seed: int | None
) -> Iterable[list[Stage1Example]]:
    indices = np.arange(len(examples))
    if shuffle_seed is not None:
        np.random.default_rng(int(shuffle_seed)).shuffle(indices)
    for start in range(0, len(indices), int(batch_size)):
        yield [examples[int(index)] for index in indices[start : start + int(batch_size)]]


def build_examples(root: Path, rows: Sequence[Mapping[str, Any]]) -> list[Stage1Example]:
    result: list[Stage1Example] = []
    for row in rows:
        graph_path = root / str(row["graph_file"])
        graph = torch.load(graph_path, map_location="cpu", weights_only=False)
        scenario_id = str(row["scenario_id"])
        result.append(
            Stage1Example(
                sample_id=str(row["state_id"]),
                state_group_id=scenario_id,
                scenario=str(row["stage"]),
                seed=int.from_bytes(scenario_id.encode("utf-8"), "little") % (2**31 - 1),
                timestep=int(row["event_step"]),
                ego_agent_id=int(row["agent_id"]),
                split=str(row["partition"]),
                graph_path=graph_path,
                label_path=root / "13_objective_revision/dataset_v2/recurrent_training_examples_fp_anchored.jsonl",
                class_count=int(row["class_count"]),
                proposal_count=int(row["proposal_count"]),
                soft_target=tuple(float(value) for value in row["soft_target_gat_r"]),
                target_quality=tuple(float(value) for value in row["target_quality_gat_r"]),
                fp_shep_quality=tuple(float(value) for value in row["fp_shep_quality"]),
                proposal_scores=tuple(float(value) for value in row["proposal_scores"]),
                interaction_group=(
                    "interaction-rich" if bool(row.get("peer_rich_pre_rollout")) else "low-interaction"
                ),
                descriptor_risk_positive=False,
                graph=graph,
            )
        )
    return result


def load_initialized_model(
    checkpoint: Path, model_config: Mapping[str, Any], device: torch.device
) -> PolicyPreviewEdgeEnhancedGATSelector:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if dict(payload["model_config"]) != dict(model_config):
        raise RuntimeError("GAT-V1 initialization architecture differs from frozen recurrent architecture")
    model = PolicyPreviewEdgeEnhancedGATSelector(
        EdgeEnhancedGATConfig.from_mapping(model_config)
    ).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model


def pair_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, tuple[tuple[int, int], ...]]:
    return {
        str(row["state_id"]): tuple(
            (int(pair[0]), int(pair[1])) for pair in row["smoothness_pairs"]
        )
        for row in rows
    }


def pair_loss(
    output: Any,
    examples: Sequence[Stage1Example],
    pairs_by_state: Mapping[str, tuple[tuple[int, int], ...]],
) -> tuple[torch.Tensor, int]:
    losses: list[torch.Tensor] = []
    count = 0
    for graph_index, example in enumerate(examples):
        logits = output.logits_for_graph(graph_index)
        pairs = pairs_by_state[example.sample_id]
        if pairs:
            losses.append(
                torch.stack(
                    [F.softplus(logits[disfavored] - logits[preferred]) for preferred, disfavored in pairs]
                ).mean()
            )
            count += len(pairs)
        else:
            losses.append(logits.sum() * 0.0)
    return torch.stack(losses).mean(), count


def loss_for_batch(
    model: PolicyPreviewEdgeEnhancedGATSelector,
    examples: Sequence[Stage1Example],
    pairs_by_state: Mapping[str, tuple[tuple[int, int], ...]],
    device: torch.device,
    pair_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
    graph_batch = batch_candidate_graphs([item.graph for item in examples]).to(device)
    output = model(graph_batch)
    primary = ragged_soft_target_cross_entropy(
        output.candidate_logits,
        output.candidate_ptr,
        [item.soft_target for item in examples],
    )
    secondary, _ = pair_loss(output, examples, pairs_by_state)
    total = primary + float(pair_weight) * secondary
    return total, primary, secondary, output


@torch.no_grad()
def evaluate(
    model: PolicyPreviewEdgeEnhancedGATSelector,
    examples: Sequence[Stage1Example],
    row_by_state: Mapping[str, Mapping[str, Any]],
    pairs_by_state: Mapping[str, tuple[tuple[int, int], ...]],
    *,
    batch_size: int,
    device: torch.device,
    pair_weight: float,
) -> tuple[dict[str, Any], list[np.ndarray]]:
    model.eval()
    totals: list[float] = []
    primaries: list[float] = []
    secondaries: list[float] = []
    weights: list[int] = []
    scores: list[np.ndarray] = []
    pair_correct = pair_total = 0
    retain_correct = retain_total = correction_correct = correction_total = 0
    for batch_examples in batches(examples, batch_size, shuffle_seed=None):
        total, primary, secondary, output = loss_for_batch(
            model, batch_examples, pairs_by_state, device, pair_weight
        )
        totals.append(float(total.item()))
        primaries.append(float(primary.item()))
        secondaries.append(float(secondary.item()))
        weights.append(len(batch_examples))
        for graph_index, example in enumerate(batch_examples):
            logits = output.logits_for_graph(graph_index).detach().cpu().numpy()
            scores.append(logits)
            predicted = int(np.argmax(logits))
            target = int(np.argmax(np.asarray(example.soft_target)))
            reason = str(row_by_state[example.sample_id]["fp_anchor_target_reason"])
            if reason == "retain_fp_shep":
                retain_total += 1
                retain_correct += int(predicted == target)
            else:
                correction_total += 1
                correction_correct += int(predicted == target)
            for preferred, disfavored in pairs_by_state[example.sample_id]:
                pair_correct += int(float(logits[preferred]) > float(logits[disfavored]))
                pair_total += 1
    mean_total = float(np.average(totals, weights=weights))
    metrics = compute_offline_metrics(examples, scores, loss=mean_total)
    metrics.update(
        {
            "total_loss": mean_total,
            "primary_loss": float(np.average(primaries, weights=weights)),
            "secondary_pair_loss": float(np.average(secondaries, weights=weights)),
            "retain_fp_state_count": retain_total,
            "retain_fp_accuracy": retain_correct / retain_total if retain_total else None,
            "correction_state_count": correction_total,
            "correction_accuracy": correction_correct / correction_total if correction_total else None,
            "smoothness_pair_count": pair_total,
            "smoothness_pair_accuracy": pair_correct / pair_total if pair_total else None,
        }
    )
    return metrics, scores


def checkpoint_payload(
    model: PolicyPreviewEdgeEnhancedGATSelector,
    optimizer: torch.optim.Optimizer,
    *,
    variant: str,
    seed: int,
    epoch: int,
    metrics: Mapping[str, Any],
    config: Mapping[str, Any],
    initialization_checkpoint: Path,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "variant": variant,
        "optimization_seed": int(seed),
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation_metrics": dict(metrics),
        "model_config": dict(config["model"]),
        "supervision": dict(config["supervision"]),
        "offline_metric_definitions": dict(config["offline_metric_definitions"]),
        "initialization_checkpoint": str(initialization_checkpoint.resolve()),
        "initialization_checkpoint_sha256": sha256_file(initialization_checkpoint),
        "runtime_graph_schema_changed": False,
        "runtime_input_semantics_changed": False,
    }


def better(metrics: Mapping[str, Any], best: Mapping[str, Any] | None) -> bool:
    if best is None:
        return True
    current_key = (
        float(metrics["top1_accuracy"]),
        -float(metrics["total_loss"]),
        float(metrics["mrr"]),
    )
    best_key = (
        float(best["top1_accuracy"]),
        -float(best["total_loss"]),
        float(best["mrr"]),
    )
    return current_key > best_key


def train_seed(
    config: Mapping[str, Any],
    variant: str,
    train: Sequence[Stage1Example],
    validation: Sequence[Stage1Example],
    row_by_state: Mapping[str, Mapping[str, Any]],
    pairs_by_state: Mapping[str, tuple[tuple[int, int], ...]],
    *,
    seed: int,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    seed_everything(seed)
    training = config["training"]
    pair_weight = float(config["supervision"]["smoothness_pairwise_weight"])
    device = resolve_device(training["device"])
    initialization = REPO_ROOT / str(config["initialization_checkpoint"])
    model = load_initialized_model(initialization, config["model"], device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / f"seed_{seed:03d}_best_validation.pt"
    last_path = checkpoint_dir / f"seed_{seed:03d}_last.pt"
    history: list[dict[str, Any]] = []
    best_metrics: dict[str, Any] | None = None
    best_epoch = 0
    stale = 0
    started = time.perf_counter()
    for epoch in range(1, int(training["max_epochs"]) + 1):
        epoch_started = time.perf_counter()
        model.train()
        train_total: list[float] = []
        train_primary: list[float] = []
        train_secondary: list[float] = []
        batch_weights: list[int] = []
        for batch_examples in batches(
            train,
            int(training["batch_size"]),
            shuffle_seed=seed * 100_000 + epoch,
        ):
            optimizer.zero_grad(set_to_none=True)
            total, primary, secondary, _ = loss_for_batch(
                model, batch_examples, pairs_by_state, device, pair_weight
            )
            if not torch.isfinite(total):
                raise FloatingPointError(f"non-finite loss seed={seed} epoch={epoch}")
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]))
            optimizer.step()
            train_total.append(float(total.item()))
            train_primary.append(float(primary.item()))
            train_secondary.append(float(secondary.item()))
            batch_weights.append(len(batch_examples))
        validation_metrics, _ = evaluate(
            model,
            validation,
            row_by_state,
            pairs_by_state,
            batch_size=int(training["batch_size"]),
            device=device,
            pair_weight=pair_weight,
        )
        improved = better(validation_metrics, best_metrics)
        if improved:
            best_metrics = dict(validation_metrics)
            best_epoch = epoch
            stale = 0
            torch.save(
                checkpoint_payload(
                    model,
                    optimizer,
                    variant=variant,
                    seed=seed,
                    epoch=epoch,
                    metrics=validation_metrics,
                    config=config,
                    initialization_checkpoint=initialization,
                ),
                best_path,
            )
        else:
            stale += 1
        history.append(
            {
                "schema_version": SCHEMA,
                "variant": variant,
                "optimization_seed": seed,
                "epoch": epoch,
                "train_total_loss": float(np.average(train_total, weights=batch_weights)),
                "train_primary_loss": float(np.average(train_primary, weights=batch_weights)),
                "train_secondary_loss": float(np.average(train_secondary, weights=batch_weights)),
                "validation_total_loss": validation_metrics["total_loss"],
                "validation_primary_loss": validation_metrics["primary_loss"],
                "validation_secondary_loss": validation_metrics["secondary_pair_loss"],
                "validation_top1": validation_metrics["top1_accuracy"],
                "validation_top3": validation_metrics["top3_accuracy"],
                "validation_mrr": validation_metrics["mrr"],
                "validation_retain_fp_accuracy": validation_metrics["retain_fp_accuracy"],
                "validation_correction_accuracy": validation_metrics["correction_accuracy"],
                "validation_smoothness_pair_accuracy": validation_metrics["smoothness_pair_accuracy"],
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "epoch_runtime_s": time.perf_counter() - epoch_started,
                "improved": improved,
            }
        )
        if stale >= int(training["early_stopping_patience"]):
            break
    if best_metrics is None:
        raise RuntimeError("training produced no valid checkpoint")
    final_metrics, _ = evaluate(
        model,
        validation,
        row_by_state,
        pairs_by_state,
        batch_size=int(training["batch_size"]),
        device=device,
        pair_weight=pair_weight,
    )
    torch.save(
        checkpoint_payload(
            model,
            optimizer,
            variant=variant,
            seed=seed,
            epoch=len(history),
            metrics=final_metrics,
            config=config,
            initialization_checkpoint=initialization,
        ),
        last_path,
    )
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "best_metrics": best_metrics,
        "best_checkpoint": best_path,
        "last_checkpoint": last_path,
        "history": history,
        "runtime_s": time.perf_counter() - started,
    }


def output_dir(config: Mapping[str, Any]) -> Path:
    return REPO_ROOT / str(config["artifact_root"]) / str(config["output_subdir"])


def preflight(config_path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    root = REPO_ROOT / str(config["artifact_root"])
    examples_path = root / str(config["dataset"]["examples"])
    summary_path = root / str(config["dataset"]["summary"])
    pairs_path = root / str(config["dataset"]["pairs"])
    contract_path = root / str(config["dataset"]["contract"])
    initialization = REPO_ROOT / str(config["initialization_checkpoint"])
    rows = read_jsonl(examples_path)
    summary = read_json(summary_path)
    if summary["status"] != "PASS" or len(rows) != int(summary["state_count"]):
        raise RuntimeError("FP-anchored dataset did not reconcile")
    variant = str(config["variant"])
    pair_weight = float(config["supervision"]["smoothness_pairwise_weight"])
    if (variant == "GAT-R") != (pair_weight == 0.0):
        raise RuntimeError("GAT-R must have zero pair weight and GAT-RS must have a positive pair weight")
    return {
        "schema_version": SCHEMA,
        "status": "FROZEN_BEFORE_TRAINING",
        "variant": variant,
        "config": str(config_path.relative_to(REPO_ROOT).as_posix()),
        "config_sha256": sha256_file(config_path),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "examples_sha256": sha256_file(examples_path),
        "pairs_sha256": sha256_file(pairs_path),
        "dataset_summary_sha256": sha256_file(summary_path),
        "target_contract_sha256": sha256_file(contract_path),
        "initialization_checkpoint": str(initialization.relative_to(REPO_ROOT).as_posix()),
        "initialization_checkpoint_sha256": sha256_file(initialization),
        "state_counts": dict(Counter(str(row["partition"]) for row in rows)),
        "scene_counts": {
            split: len({str(row["scenario_id"]) for row in rows if row["partition"] == split})
            for split in ("train", "validation")
        },
        "smoothness_pairwise_weight": pair_weight,
        "checkpoint_selection": "maximum internal-validation exact-target Top-1, then minimum total loss, then MRR",
        "runtime_architecture_changed": False,
        "runtime_graph_changed": False,
        "runtime_input_semantics_changed": False,
        "runtime_role_changed": False,
        "dev_used": False,
        "holdout_used": False,
        "formal_v1_rows_used": 0,
        "formal_v2_generated": False,
    }


def verify_preflight(config_path: Path, config: Mapping[str, Any], freeze: Mapping[str, Any]) -> None:
    root = REPO_ROOT / str(config["artifact_root"])
    checks = {
        "config": sha256_file(config_path) == freeze["config_sha256"],
        "script": sha256_file(Path(__file__).resolve()) == freeze["script_sha256"],
        "examples": sha256_file(root / config["dataset"]["examples"]) == freeze["examples_sha256"],
        "pairs": sha256_file(root / config["dataset"]["pairs"]) == freeze["pairs_sha256"],
        "initialization": sha256_file(REPO_ROOT / config["initialization_checkpoint"]) == freeze["initialization_checkpoint_sha256"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"pretraining freeze mismatch: {checks}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "train"))
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve() if args.config.is_absolute() else (REPO_ROOT / args.config).resolve()
    config = read_json(config_path)
    out = output_dir(config)
    freeze_path = out / "PRETRAINING_FREEZE.json"
    if args.phase == "prepare":
        if out.exists() and any(out.iterdir()):
            raise RuntimeError(f"training output already exists: {out}")
        freeze = preflight(config_path, config)
        write_json(freeze_path, freeze)
        print(json.dumps(freeze, indent=2, ensure_ascii=False))
        return
    if not freeze_path.exists():
        raise RuntimeError("run prepare before train")
    freeze = read_json(freeze_path)
    verify_preflight(config_path, config, freeze)
    checkpoints = out / "checkpoints"
    if checkpoints.exists():
        raise RuntimeError("checkpoint directory already exists; refusing overwrite")

    root = REPO_ROOT / str(config["artifact_root"])
    rows = read_jsonl(root / str(config["dataset"]["examples"]))
    row_by_state = {str(row["state_id"]): row for row in rows}
    examples = build_examples(root, rows)
    train = [item for item in examples if item.split == "train"]
    validation = [item for item in examples if item.split == "validation"]
    pairs_by_state = pair_map(rows)
    device = resolve_device(config["training"]["device"])
    initialization = load_initialized_model(
        REPO_ROOT / str(config["initialization_checkpoint"]), config["model"], device
    )
    initial_metrics = []
    for split, members in (("train", train), ("validation", validation)):
        metrics, _ = evaluate(
            initialization,
            members,
            row_by_state,
            pairs_by_state,
            batch_size=int(config["training"]["batch_size"]),
            device=device,
            pair_weight=float(config["supervision"]["smoothness_pairwise_weight"]),
        )
        initial_metrics.append({"split": split, **metrics})

    results = []
    for seed in config["training"]["optimization_seeds"]:
        print(f"[{config['variant']}] training seed {seed}", flush=True)
        results.append(
            train_seed(
                config,
                str(config["variant"]),
                train,
                validation,
                row_by_state,
                pairs_by_state,
                seed=int(seed),
                checkpoint_dir=checkpoints,
            )
        )
    selected = max(
        results,
        key=lambda result: (
            float(result["best_metrics"]["top1_accuracy"]),
            -float(result["best_metrics"]["total_loss"]),
            float(result["best_metrics"]["mrr"]),
            -int(result["seed"]),
        ),
    )
    selected_path = checkpoints / "best_validation.pt"
    shutil.copy2(selected["best_checkpoint"], selected_path)
    payload = torch.load(selected_path, map_location=device, weights_only=False)
    selected_model = PolicyPreviewEdgeEnhancedGATSelector(
        EdgeEnhancedGATConfig.from_mapping(config["model"])
    ).to(device)
    selected_model.load_state_dict(payload["model_state_dict"], strict=True)
    split_metrics = []
    for split, members in (("train", train), ("validation", validation)):
        metrics, _ = evaluate(
            selected_model,
            members,
            row_by_state,
            pairs_by_state,
            batch_size=int(config["training"]["batch_size"]),
            device=device,
            pair_weight=float(config["supervision"]["smoothness_pairwise_weight"]),
        )
        split_metrics.append({"schema_version": SCHEMA, "variant": config["variant"], "split": split, **metrics})
    curves = [row for result in results for row in result["history"]]
    selection = [
        {
            "schema_version": SCHEMA,
            "variant": config["variant"],
            "seed": result["seed"],
            "best_epoch": result["best_epoch"],
            "epochs_completed": result["epochs_completed"],
            "best_validation_top1": result["best_metrics"]["top1_accuracy"],
            "best_validation_loss": result["best_metrics"]["total_loss"],
            "best_validation_mrr": result["best_metrics"]["mrr"],
            "best_validation_retain_fp_accuracy": result["best_metrics"]["retain_fp_accuracy"],
            "best_validation_correction_accuracy": result["best_metrics"]["correction_accuracy"],
            "best_validation_pair_accuracy": result["best_metrics"]["smoothness_pair_accuracy"],
            "checkpoint": str(result["best_checkpoint"].relative_to(REPO_ROOT).as_posix()),
            "checkpoint_sha256": sha256_file(result["best_checkpoint"]),
            "runtime_s": result["runtime_s"],
            "selected": result is selected,
        }
        for result in results
    ]
    write_csv(out / f"training_curves_{str(config['variant']).lower().replace('-', '_')}.csv", curves)
    write_csv(out / "checkpoint_selection_history.csv", selection)
    write_csv(out / "offline_training_metrics.csv", split_metrics)
    write_json(out / "INITIALIZATION_OFFLINE_METRICS.json", {"metrics": initial_metrics})
    manifest = {
        **freeze,
        "status": "TRAINING_COMPLETE",
        "selected_seed": selected["seed"],
        "selected_epoch": selected["best_epoch"],
        "selected_checkpoint": str(selected_path.relative_to(REPO_ROOT).as_posix()),
        "selected_checkpoint_sha256": sha256_file(selected_path),
        "initialization_metrics": initial_metrics,
        "selected_metrics": split_metrics,
        "training_result_count": len(results),
        "smoothness_supervision_added": config["variant"] == "GAT-RS",
        "smoothness_supervision_role": "SECONDARY" if config["variant"] == "GAT-RS" else "NONE",
        "gat_runtime_input_changed": False,
        "gat_runtime_role_changed": False,
        "core_runtime_theory_changed": False,
    }
    write_json(out / f"{str(config['variant']).replace('-', '_')}_TRAINING_MANIFEST.json", manifest)
    print(
        json.dumps(
            {
                "variant": config["variant"],
                "selected_seed": selected["seed"],
                "selected_epoch": selected["best_epoch"],
                "checkpoint": str(selected_path),
                "checkpoint_sha256": sha256_file(selected_path),
                "validation_metrics": next(row for row in split_metrics if row["split"] == "validation"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
