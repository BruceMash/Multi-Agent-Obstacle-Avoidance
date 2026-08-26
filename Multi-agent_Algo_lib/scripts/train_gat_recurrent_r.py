#!/usr/bin/env python3
"""Train GAT-R on the frozen long-range recurrent hierarchical targets."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from planning.gat.stage1_training import (  # noqa: E402
    Stage1Example,
    evaluate_model,
    graph_descriptor_risk_positive,
    load_model_checkpoint,
    sha256_file,
    train_one_seed,
    training_result_record,
)


SCHEMA_VERSION = "gat_recurrent_r_training_v1"
DEFAULT_CONFIG = REPO_ROOT / "configs/training/gat_recurrent_r.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    if not fields:
        fields = ["schema_version", "status"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(json_ready(value), ensure_ascii=False)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_examples(artifact_root: Path, rows: Sequence[Mapping[str, Any]]) -> list[Stage1Example]:
    result: list[Stage1Example] = []
    for row in rows:
        graph_path = artifact_root / str(row["graph_file"])
        graph = torch.load(graph_path, map_location="cpu", weights_only=False)
        scenario_id = str(row["scenario_id"])
        numeric_seed = int.from_bytes(scenario_id.encode("utf-8"), "little") % (2**31 - 1)
        result.append(
            Stage1Example(
                sample_id=str(row["state_id"]),
                state_group_id=scenario_id,
                scenario=str(row["stage"]),
                seed=numeric_seed,
                timestep=int(row["event_step"]),
                ego_agent_id=int(row["agent_id"]),
                split=str(row["partition"]),
                graph_path=graph_path,
                label_path=artifact_root / "02_recurrent_dataset/recurrent_training_examples.jsonl",
                class_count=int(row["class_count"]),
                proposal_count=int(row["proposal_count"]),
                soft_target=tuple(float(value) for value in row["soft_target_gat_r"]),
                target_quality=tuple(float(value) for value in row["target_quality_gat_r"]),
                fp_shep_quality=tuple(float(value) for value in row["fp_shep_quality"]),
                proposal_scores=tuple(float(value) for value in row["proposal_scores"]),
                interaction_group=(
                    "interaction-rich"
                    if int(row["smoothness_pair_count"]) > 0
                    else "low-interaction"
                ),
                descriptor_risk_positive=graph_descriptor_risk_positive(graph),
                graph=graph,
            )
        )
    return result


def preflight_payload(
    config_path: Path,
    config: Mapping[str, Any],
    artifact_root: Path,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    reconciliation_path = artifact_root / str(config["dataset"]["reconciliation"])
    label_freeze_path = artifact_root / str(config["dataset"]["label_freeze"])
    examples_path = artifact_root / str(config["dataset"]["examples"])
    reconciliation = read_json(reconciliation_path)
    if reconciliation["status"] != "PASS":
        raise RuntimeError("recurrent dataset reconciliation did not pass")
    counts = {
        "train_states": sum(row["partition"] == "train" for row in rows),
        "validation_states": sum(row["partition"] == "validation" for row in rows),
        "train_scenes": len({row["scenario_id"] for row in rows if row["partition"] == "train"}),
        "validation_scenes": len({row["scenario_id"] for row in rows if row["partition"] == "validation"}),
    }
    expected = {
        "train_states": int(reconciliation["internal_train_state_count"]),
        "validation_states": int(reconciliation["internal_validation_state_count"]),
        "train_scenes": int(config["dataset"]["train_scenes"]),
        "validation_scenes": int(config["dataset"]["validation_scenes"]),
    }
    if counts != expected:
        raise RuntimeError(f"preflight split mismatch: {counts} != {expected}")
    if config["supervision"]["smoothness_pairwise_loss_enabled"]:
        raise RuntimeError("GAT-R must not enable smoothness supervision")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "FROZEN_BEFORE_TRAINING",
        "variant": "GAT-R",
        "config": str(config_path.relative_to(REPO_ROOT).as_posix()),
        "config_sha256": sha256_file(config_path),
        "training_script": str(Path(__file__).resolve().relative_to(REPO_ROOT).as_posix()),
        "training_script_sha256": sha256_file(Path(__file__).resolve()),
        "examples": str(examples_path.relative_to(REPO_ROOT).as_posix()),
        "examples_sha256": sha256_file(examples_path),
        "semantic_dataset_sha256": reconciliation["training_examples_semantic_sha256"],
        "reconciliation_sha256": sha256_file(reconciliation_path),
        "label_freeze_sha256": sha256_file(label_freeze_path),
        "counts": counts,
        "architecture_equal_to_gat_v1": True,
        "runtime_graph_schema_changed": False,
        "runtime_goal_distance_scale_m": 100.0,
        "runtime_future_information_added": False,
        "dev_holdout_formal_used": False,
        "checkpoint_selection": config["training"]["checkpoint_selection"],
        "optimization_seeds": config["training"]["optimization_seeds"],
        "hyperparameter_sweep": False,
    }


def verify_frozen(preflight: Mapping[str, Any], config_path: Path) -> None:
    checks = {
        "config": sha256_file(config_path) == preflight["config_sha256"],
        "script": sha256_file(Path(__file__).resolve()) == preflight["training_script_sha256"],
        "examples": sha256_file(REPO_ROOT / preflight["examples"]) == preflight["examples_sha256"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"pretraining freeze mismatch: {checks}")


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve() if args.config.is_absolute() else (REPO_ROOT / args.config).resolve()
    config = read_json(config_path)
    artifact_root = (REPO_ROOT / config["artifact_root"]).resolve()
    output_dir = artifact_root / "05_gat_r_training"
    output_dir.mkdir(parents=True, exist_ok=True)
    examples_path = artifact_root / config["dataset"]["examples"]
    rows = read_jsonl(examples_path)
    freeze = preflight_payload(config_path, config, artifact_root, rows)
    freeze_path = output_dir / "PRETRAINING_FREEZE.json"
    if args.preflight_only:
        if freeze_path.exists():
            existing = read_json(freeze_path)
            if existing != freeze:
                raise RuntimeError("existing pretraining freeze differs from current preflight")
        else:
            write_json(freeze_path, freeze)
        print(json.dumps(freeze, indent=2))
        return
    if not freeze_path.exists():
        raise RuntimeError("run --preflight-only before training")
    frozen = read_json(freeze_path)
    verify_frozen(frozen, config_path)
    if (output_dir / "checkpoints").exists():
        raise RuntimeError("checkpoint directory already exists; refusing to overwrite")

    examples = build_examples(artifact_root, rows)
    train = [item for item in examples if item.split == "train"]
    validation = [item for item in examples if item.split == "validation"]
    checkpoints = output_dir / "checkpoints"
    results = []
    for optimization_seed in config["training"]["optimization_seeds"]:
        print(f"[GAT-R] training optimization seed {optimization_seed}", flush=True)
        results.append(
            train_one_seed(
                config,
                train,
                validation,
                optimization_seed=int(optimization_seed),
                checkpoint_dir=checkpoints,
            )
        )
    selected = min(results, key=lambda item: (item.best_validation_loss, item.optimization_seed))
    selected_path = checkpoints / "best_validation.pt"
    shutil.copy2(selected.best_checkpoint, selected_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_model = load_model_checkpoint(selected_path, config, device)
    split_metrics = []
    for split_name, split_examples in (("train", train), ("validation", validation)):
        metrics, _ = evaluate_model(
            selected_model,
            split_examples,
            batch_size=int(config["training"]["batch_size"]),
            device=device,
        )
        split_metrics.append({"schema_version": SCHEMA_VERSION, "variant": "GAT-R", "split": split_name, **metrics})

    curves = []
    for result in results:
        for record in result.history:
            curves.append({"schema_version": SCHEMA_VERSION, "variant": "GAT-R", **record.__dict__})
    selection_rows = []
    for result in results:
        selection_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "variant": "GAT-R",
                **training_result_record(result),
                "selected": result.optimization_seed == selected.optimization_seed,
                "selection_criterion": "minimum internal validation soft-target loss",
                "dev_used": False,
            }
        )
    write_csv(output_dir / "training_curves_gat_r.csv", curves)
    write_csv(output_dir / "checkpoint_selection_history.csv", selection_rows)
    write_csv(output_dir / "offline_training_metrics.csv", split_metrics)
    manifest = {
        **frozen,
        "status": "TRAINING_COMPLETE",
        "training_result_count": len(results),
        "training_results": [training_result_record(result) for result in results],
        "selected_optimization_seed": selected.optimization_seed,
        "selected_best_epoch": selected.best_epoch,
        "selected_validation_loss": selected.best_validation_loss,
        "selected_checkpoint": str(selected_path.relative_to(REPO_ROOT).as_posix()),
        "selected_checkpoint_sha256": sha256_file(selected_path),
        "selected_metrics": split_metrics,
        "smoothness_supervision_added": False,
        "gat_runtime_input_changed": False,
        "gat_runtime_role_changed": False,
        "core_runtime_theory_changed": False,
        "dev_used_for_training_or_checkpoint_selection": False,
    }
    write_json(output_dir / "GAT_R_TRAINING_MANIFEST.json", manifest)
    print(json.dumps({
        "selected_seed": selected.optimization_seed,
        "best_epoch": selected.best_epoch,
        "validation_loss": selected.best_validation_loss,
        "checkpoint": str(selected_path),
        "checkpoint_sha256": sha256_file(selected_path),
    }, indent=2))


if __name__ == "__main__":
    main()
