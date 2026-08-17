"""Train Stage-I GAT V2 on frozen deployment-aligned graph/target artifacts.

The runner deliberately separates model selection from test evaluation.  It
first executes optimization seed 0, applies the frozen smoke gate, and only
then permits seeds 1 and 2.  The final checkpoint is selected by validation
loss before any test metric is computed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from planning.gat.stage1_training import (  # noqa: E402
    empirical_random_metrics,
    evaluate_model,
    resolve_device,
)
from planning.gat.stage1_v2_training import (  # noqa: E402
    SCHEMA_VERSION,
    Stage1V2Example,
    V2TrainingRunResult,
    classify_gain,
    examples_for_target,
    load_initialized_model,
    load_v2_checkpoint_model,
    load_v2_examples,
    long_horizon_selection_diagnostics,
    metrics_for_scores,
    model_score_vectors,
    model_state_hash,
    prediction_disagreement,
    sha256_file,
    smoke_gate_result_v2,
    target_agreement_analysis,
    train_one_seed_v2,
)


CRITICAL_FILES = {
    "proposal": REPO_ROOT / "Guidance" / "reference_point_proposal_demo.py",
    "fp_shep": REPO_ROOT / "planning" / "policy_preview.py",
    "graph_builder": REPO_ROOT / "planning" / "heterogeneous_candidate_graph.py",
    "gat_selector": REPO_ROOT / "planning" / "gat" / "candidate_selector.py",
    "gat_layer": REPO_ROOT / "planning" / "gat" / "edge_enhanced_gat.py",
    "typed_encoders": REPO_ROOT / "planning" / "gat" / "typed_encoders.py",
    "sac_actor": REPO_ROOT / "baseline" / "sac" / "net.py",
    "dmp": REPO_ROOT / "Controller" / "dmp_rl.py",
    "environment": REPO_ROOT / "Environment" / "multi_agent_dmp_env.py",
}

METRIC_FIELDS = [
    "schema_version",
    "model",
    "target",
    "split",
    "optimization_seed",
    "comparison_role",
    "graph_count",
    "loss",
    "top1_accuracy",
    "top3_accuracy",
    "mrr",
    "spearman_proposal_only_mean",
    "spearman_valid_graph_count",
    "null_precision",
    "null_recall",
    "null_reference_rate",
    "null_prediction_rate",
    "proposal_selection_accuracy",
    "proposal_reference_graph_count",
    "maximum_fixed_proposal_rank_prediction_rate",
    "empirical_random_top1_accuracy",
    "empirical_random_top3_accuracy",
    "empirical_random_mrr",
]


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_ready(value), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_csv(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(fieldnames), extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: _json_ready(row.get(key)) for key in fieldnames}
            )


def _metric_row(
    *,
    model_name: str,
    target_name: str,
    split: str,
    seed: int | str,
    role: str,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model": model_name,
        "target": target_name,
        "split": split,
        "optimization_seed": seed,
        "comparison_role": role,
        **{field: metrics.get(field) for field in METRIC_FIELDS[6:]},
    }


def _result_record(result: V2TrainingRunResult) -> dict[str, Any]:
    return {
        "optimization_seed": result.optimization_seed,
        "best_epoch": result.best_epoch,
        "epochs_completed": result.epochs_completed,
        "early_stopped": result.early_stopped,
        "best_validation_loss": result.best_validation_loss,
        "best_validation_metrics": dict(result.best_validation_metrics),
        "best_checkpoint": str(result.best_checkpoint),
        "last_checkpoint": str(result.last_checkpoint),
        "initialization_model_hash": result.initialization_model_hash,
        "fresh_optimizer_state_at_start": result.fresh_optimizer_state_at_start,
        "gradients_finite": result.gradients_finite,
        "maximum_gradient_norm_pre_clip": result.maximum_gradient_norm_pre_clip,
        "runtime_s": result.runtime_s,
    }


def _frozen_protocol_checks(config: Mapping[str, Any]) -> dict[str, bool]:
    model = config["model"]
    training = config["training"]
    supervision = config["supervision"]
    split = config["dataset_split"]
    exclusions = config["strict_exclusions"]
    return {
        "architecture_frozen": (
            model["hidden_dim"] == 64
            and model["edge_dim"] == 32
            and model["num_heads"] == 4
            and model["num_layers"] == 2
            and model["activation"] == "gelu"
            and model["layer_norm"] is True
            and model["residual"] is True
        ),
        "training_hyperparameters_frozen": (
            training["optimizer"] == "AdamW"
            and training["learning_rate"] == 1.0e-3
            and training["weight_decay"] == 1.0e-4
            and training["batch_size"] == 32
            and training["max_epochs"] == 100
            and training["early_stopping_patience"] == 15
            and training["gradient_clip_norm"] == 5.0
            and training["optimization_seeds"] == [0, 1, 2]
        ),
        "effective_split_frozen": (
            split["train_seeds"] == [0, 1, 2, 3, 4, 5, 6]
            and split["validation_seeds"] == [7]
            and split["test_seeds"] == [8, 9]
            and split["field"] == "effective_split"
        ),
        "supervision_frozen": (
            supervision["H_preview"] == 4
            and supervision["input_graph"] == "historical_vector_gate_H4"
            and supervision["soft_target_temperature"] == 0.25
            and supervision["target_name"] == "v2_long_horizon_soft_target"
            and supervision["labels_regenerated"] is False
        ),
        "initialization_and_optimizer_frozen": (
            training["initialization"] == "stage1_v1_best_model_state_dict"
            and training["optimizer_state_source"] == "fresh"
            and training["inherit_optimizer_state"] is False
        ),
        "selection_validation_loss_only": (
            training["checkpoint_selection"] == "validation_loss_only"
        ),
        "strict_exclusions_active": all(value is False for value in exclusions.values()),
        "combined_interpretation_frozen": (
            config["comparison_interpretation"]["name"]
            == "deployment_aligned_input_plus_target_adaptation"
            and config["comparison_interpretation"]["target_only_claim_allowed"]
            is False
        ),
    }


def _hash_inputs(config: Mapping[str, Any]) -> dict[str, str]:
    paths = {
        "dataset": _resolve(config["dataset_path"]),
        "branch_outcomes": _resolve(config["branch_outcomes_path"]),
        "initial_checkpoint": _resolve(config["initial_checkpoint"]),
    }
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    if hashes != dict(config["frozen_sha256"]):
        raise ValueError(f"frozen input hash mismatch: {hashes}")
    return hashes


def _load_v2_validity(config: Mapping[str, Any]) -> dict[str, Any]:
    directory = _resolve(config["v2_artifact_dir"])
    validity = json.loads((directory / "validity_gate.json").read_text("utf-8"))
    conclusion = json.loads((directory / "conclusion.json").read_text("utf-8"))
    if not validity.get("passed") or conclusion.get("V2_SUPERVISION_VALID") != "YES":
        raise RuntimeError("frozen V2 supervision validity gate is not satisfied")
    return {
        "validity_gate_passed": True,
        "V2_SUPERVISION_VALID": conclusion["V2_SUPERVISION_VALID"],
        "TRAINING_DEPLOYMENT_GATE_ALIGNMENT": conclusion[
            "TRAINING_DEPLOYMENT_GATE_ALIGNMENT"
        ],
        "SOURCE_STATE_REPRODUCTION": conclusion["SOURCE_STATE_REPRODUCTION"],
        "CANDIDATE_IDENTITY_MATCH": conclusion["CANDIDATE_IDENTITY_MATCH"],
        "GRAPH_SCHEMA_MATCH": conclusion["GRAPH_SCHEMA_MATCH"],
        "BRANCH_ROLLOUT_DETERMINISM": conclusion[
            "BRANCH_ROLLOUT_DETERMINISM"
        ],
    }


def _initialization_audit(
    checkpoint: Path, config: Mapping[str, Any], device: torch.device
) -> dict[str, Any]:
    model, payload, state_hash = load_initialized_model(checkpoint, config, device)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_schema": payload.get("schema_version"),
        "checkpoint_optimization_seed": payload.get("optimization_seed"),
        "checkpoint_epoch": payload.get("epoch"),
        "strict_model_state_load": True,
        "model_state_hash": state_hash,
        "model_tensor_count": len(model.state_dict()),
        "model_parameter_count": parameter_count,
        "optimizer_state_loaded": False,
        "fresh_AdamW_required": True,
    }


def _random_metrics(examples: Sequence[Stage1V2Example]) -> dict[str, Any]:
    values = empirical_random_metrics([item.class_count for item in examples])
    return {
        "graph_count": len(examples),
        "top1_accuracy": values["top1_accuracy"],
        "top3_accuracy": values["top3_accuracy"],
        "mrr": values["mrr"],
        "empirical_random_top1_accuracy": values["top1_accuracy"],
        "empirical_random_top3_accuracy": values["top3_accuracy"],
        "empirical_random_mrr": values["mrr"],
        "spearman_valid_graph_count": 0,
    }


def _native_v1_provenance(config: Mapping[str, Any]) -> dict[str, Any]:
    path = _resolve(config["v1_training_artifact_dir"]) / "test_metrics.csv"
    rows = list(csv.DictReader(path.open("r", encoding="utf-8-sig", newline="")))
    checkpoint = torch.load(
        _resolve(config["initial_checkpoint"]), map_location="cpu", weights_only=False
    )
    seed = str(int(checkpoint["optimization_seed"]))
    matches = [
        row
        for row in rows
        if row["optimization_seed"] == seed
        and row["split"] == "test"
        and row["method"] == "gat"
        and row["subset_type"] == "overall"
    ]
    if len(matches) != 1:
        raise ValueError("cannot uniquely identify V1 native checkpoint test metrics")
    return {
        "role": "provenance_only_not_paired_comparison",
        "source": str(path),
        "checkpoint_seed": int(seed),
        "native_v1_test_metrics": matches[0],
        "mixed_with_v2_paired_metrics": False,
    }


def _history_rows(results: Sequence[V2TrainingRunResult]) -> list[Mapping[str, Any]]:
    return [row for result in results for row in result.history]


HISTORY_FIELDS = [
    "optimization_seed",
    "epoch",
    "train_loss",
    "validation_loss",
    "validation_top1",
    "validation_top3",
    "validation_mrr",
    "validation_spearman",
    "validation_random_top1",
    "validation_null_prediction_rate",
    "validation_fixed_rank_max_rate",
    "gradients_finite",
    "maximum_gradient_norm_pre_clip",
    "gradient_clip_norm",
    "learning_rate",
    "epoch_runtime_s",
    "improved",
]


def _empty_required_outputs(run_dir: Path) -> None:
    write_csv(run_dir / "seed_metrics.csv", [], METRIC_FIELDS)
    write_csv(run_dir / "v1_vs_v2_offline_metrics.csv", [], METRIC_FIELDS)
    write_csv(
        run_dir / "target_alignment_analysis.csv",
        [],
        ["analysis", "left", "right", "metric", "value", "graph_count", "notes"],
    )


def _analysis_rows(
    *,
    test_examples: Sequence[Stage1V2Example],
    v1_scores: Sequence[np.ndarray],
    v2_scores: Sequence[np.ndarray],
    v1_long: Mapping[str, Any],
    v2_long: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in target_agreement_analysis(test_examples):
        for metric in ["top1_agreement", "top3_overlap", "pairwise_agreement"]:
            rows.append(
                {
                    "analysis": "intrinsic_target_agreement",
                    "left": record["left"],
                    "right": record["right"],
                    "metric": metric,
                    "value": record[metric],
                    "graph_count": record["graph_count"],
                    "notes": "targets read directly from frozen V2 dataset",
                }
            )
    disagreement = prediction_disagreement(v1_scores, v2_scores)
    for metric in ["disagreement_count", "disagreement_rate", "agreement_rate"]:
        rows.append(
            {
                "analysis": "v1_v2_prediction_disagreement",
                "left": "v1_checkpoint_on_v2_graph",
                "right": "v2_trained_on_v2_graph",
                "metric": metric,
                "value": disagreement[metric],
                "graph_count": disagreement["graph_count"],
                "notes": "same frozen V2 test graphs",
            }
        )
    keys = [
        "maximum_tier_prediction_rate",
        "selected_proposal_reference_reach_rate",
        "selected_team_success_rate",
        "selected_ego_terminal_reach_rate",
        "post_reference_signal_graph_count",
        "post_reference_subset_top1",
        "post_reference_subset_top3",
        "post_reference_subset_mrr",
        "post_reference_subset_spearman",
    ]
    for model_name, diagnostics in [
        ("v1_checkpoint_on_v2_graph", v1_long),
        ("v2_trained_on_v2_graph", v2_long),
    ]:
        for key in keys:
            rows.append(
                {
                    "analysis": "long_horizon_selection_diagnostic",
                    "left": model_name,
                    "right": "v2_long_horizon_branch_truth",
                    "metric": key,
                    "value": diagnostics.get(key),
                    "graph_count": diagnostics["graph_count"],
                    "notes": "diagnostic only; never used for checkpoint selection",
                }
            )
    return rows


def _build_integrity(
    *,
    critical_before: Mapping[str, str],
    input_hashes: Mapping[str, str],
    protocol_checks: Mapping[str, bool],
    dataset_audit: Mapping[str, Any],
    initialization_audit: Mapping[str, Any],
    seed_count: int,
    smoke_status: str,
    test_evaluated: bool,
) -> dict[str, Any]:
    critical_after = {name: sha256_file(path) for name, path in CRITICAL_FILES.items()}
    return {
        "schema_version": SCHEMA_VERSION,
        "frozen_input_hashes": dict(input_hashes),
        "critical_file_hashes_before": dict(critical_before),
        "critical_file_hashes_after": critical_after,
        "critical_files_unchanged": dict(critical_before) == critical_after,
        "protocol_checks": dict(protocol_checks),
        "all_protocol_checks_passed": all(protocol_checks.values()),
        "dataset_audit": dict(dataset_audit),
        "initialization_audit": dict(initialization_audit),
        "smoke_gate": smoke_status,
        "optimization_seed_count": seed_count,
        "checkpoint_selected_by_validation_loss_only": True,
        "test_split_evaluated_after_checkpoint_selection": test_evaluated,
        "V1_and_V2_use_identical_V2_test_graphs": test_evaluated,
        "V1_native_metrics_provenance_only": True,
        "target_labels_regenerated": False,
        "optimizer_state_inherited": False,
        "closed_loop_run": False,
        "stress_test_run": False,
        "overlap_loss_enabled": False,
        "completion_loss_enabled": False,
        "architecture_modified": False,
        "SAC_DMP_modified": False,
    }


def _report(
    *,
    run_dir: Path,
    dataset_audit: Mapping[str, Any],
    smoke: Mapping[str, Any],
    results: Sequence[V2TrainingRunResult],
    paired_rows: Sequence[Mapping[str, Any]],
    conclusion: Mapping[str, Any],
    integrity: Mapping[str, Any],
) -> None:
    lookup = {
        (row["model"], row["target"]): row
        for row in paired_rows
        if row["split"] == "test"
    }
    lines = [
        "# Stage-I GAT V2 Offline Retraining Report",
        "",
        "## 实验口径",
        "",
        "- 本实验属于 deployment-aligned input + target adaptation，不是纯 label-only ablation。",
        "- V1 与 V2 均在同一组 V2 historical-vector-gate H4 test graphs 上评估。",
        "- V2 由 V1 best model_state_dict 初始化，optimizer 为全新 AdamW。",
        "- checkpoint 仅依据 validation loss 选择；test 与 long-horizon 诊断不参与选模。",
        "- 本轮未运行 closed-loop、24-layout stress、overlap/completion loss 或 SAC-DMP 修改。",
        "",
        "## 数据与 smoke gate",
        "",
        f"- Graphs: {dataset_audit['graph_count']}；effective split: "
        f"`{json.dumps(dataset_audit['effective_split_graph_counts'])}`。",
        f"- Variable class count: {dataset_audit['minimum_class_count']}–"
        f"{dataset_audit['maximum_class_count']}。",
        f"- `SMOKE_GATE = {smoke['status']}`；checks: "
        f"`{json.dumps(smoke['checks'], ensure_ascii=False)}`。",
        "",
        "## Optimization seeds",
        "",
        "| seed | best epoch | epochs | validation loss | runtime (s) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result.optimization_seed} | {result.best_epoch} | "
            f"{result.epochs_completed} | {result.best_validation_loss:.6f} | "
            f"{result.runtime_s:.2f} |"
        )
    if paired_rows:
        lines.extend(
            [
                "",
                "## 同一 V2 test graph 配对结果",
                "",
                "| model | target | Top-1 | Top-3 | MRR | Spearman | Null rate |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for model_name in ["v1_checkpoint_on_v2_graph", "v2_trained_on_v2_graph"]:
            for target_name in [
                "v1_scalar_h6",
                "v1_historical_h6",
                "v2_long_horizon",
            ]:
                row = lookup[(model_name, target_name)]
                lines.append(
                    f"| {model_name} | {target_name} | "
                    f"{float(row['top1_accuracy']):.4f} | "
                    f"{float(row['top3_accuracy']):.4f} | "
                    f"{float(row['mrr']):.4f} | "
                    f"{float(row['spearman_proposal_only_mean']):.4f} | "
                    f"{float(row['null_prediction_rate']):.4f} |"
                )
    lines.extend(["", "## 结论", ""])
    for key in [
        "V2_TRAINING_VALID",
        "V2_OFFLINE_IMPROVEMENT_OVER_V1",
        "V2_TARGET_ALIGNMENT_GAIN",
        "NULL_BEHAVIOR_CHANGE",
        "LONG_HORIZON_SIGNAL_LEARNED",
        "RECOMMENDED_NEXT_STEP",
        "FINAL_CHECKPOINT",
    ]:
        lines.append(f"- `{key} = {conclusion[key]}`")
    lines.extend(
        [
            "",
            "## 完整性",
            "",
            f"- Critical files unchanged: {integrity['critical_files_unchanged']}。",
            f"- All protocol checks passed: {integrity['all_protocol_checks_passed']}。",
            "- V1 native test metrics 仅作为 provenance，未与本轮 paired metrics 混用。",
            "- 最终解释不得将差异完全归因于 label replacement。",
        ]
    )
    (run_dir / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", "utf-8")


def run(config_path: Path) -> Path:
    raw_config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    output_root = _resolve(raw_config["output_root"])
    run_dir = output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = run_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)

    protocol_checks = _frozen_protocol_checks(raw_config)
    if not all(protocol_checks.values()):
        raise ValueError(f"frozen protocol mismatch: {protocol_checks}")
    input_hashes = _hash_inputs(raw_config)
    v2_validity = _load_v2_validity(raw_config)
    critical_before = {name: sha256_file(path) for name, path in CRITICAL_FILES.items()}
    device = resolve_device(raw_config["training"]["device"])
    initialization_checkpoint = _resolve(raw_config["initial_checkpoint"])
    initialization = _initialization_audit(
        initialization_checkpoint, raw_config, device
    )

    config = dict(raw_config)
    config.update(
        {
            "resolved_dataset_path": str(_resolve(raw_config["dataset_path"])),
            "resolved_branch_outcomes_path": str(
                _resolve(raw_config["branch_outcomes_path"])
            ),
            "resolved_initial_checkpoint": str(initialization_checkpoint),
            "resolved_output_dir": str(run_dir),
            "created_at": datetime.now().astimezone().isoformat(),
            "python_executable": sys.executable,
            "torch_version": torch.__version__,
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "protocol_checks": protocol_checks,
            "v2_validity": v2_validity,
        }
    )
    write_json(run_dir / "config.json", config)

    stage1_config = json.loads(
        (REPO_ROOT / "configs" / "training" / "gat_stage1.json").read_text("utf-8")
    )
    examples, dataset_audit = load_v2_examples(
        _resolve(config["dataset_path"]),
        split_config=config["dataset_split"],
        interaction_config=stage1_config["interaction_diagnostic"],
        expected_h_preview=int(config["supervision"]["H_preview"]),
        expected_temperature=float(config["supervision"]["soft_target_temperature"]),
    )
    dataset_audit["V2_validity"] = v2_validity
    write_json(run_dir / "dataset_audit.json", dataset_audit)
    by_split = {
        split: [item for item in examples if item.split == split]
        for split in ["train", "validation", "test"]
    }

    smoke_seed = int(config["training"]["smoke_seed"])
    results = [
        train_one_seed_v2(
            config,
            by_split["train"],
            by_split["validation"],
            optimization_seed=smoke_seed,
            initialization_checkpoint=initialization_checkpoint,
            checkpoint_dir=checkpoint_dir,
            device=device,
        )
    ]
    smoke = smoke_gate_result_v2(results[0], config["smoke_gate"])
    write_json(run_dir / "smoke_gate.json", smoke)
    write_csv(run_dir / "training_history.csv", _history_rows(results), HISTORY_FIELDS)

    if smoke["status"] != "PASS":
        shutil.copy2(results[0].best_checkpoint, checkpoint_dir / "best_validation.pt")
        shutil.copy2(results[0].last_checkpoint, checkpoint_dir / "last.pt")
        _empty_required_outputs(run_dir)
        integrity = _build_integrity(
            critical_before=critical_before,
            input_hashes=input_hashes,
            protocol_checks=protocol_checks,
            dataset_audit=dataset_audit,
            initialization_audit=initialization,
            seed_count=1,
            smoke_status="FAIL",
            test_evaluated=False,
        )
        conclusion = {
            "schema_version": SCHEMA_VERSION,
            "V2_TRAINING_VALID": "NO",
            "V2_OFFLINE_IMPROVEMENT_OVER_V1": "NO",
            "V2_TARGET_ALIGNMENT_GAIN": "NO",
            "NULL_BEHAVIOR_CHANGE": "NO",
            "LONG_HORIZON_SIGNAL_LEARNED": "NO",
            "RECOMMENDED_NEXT_STEP": "STOP",
            "FINAL_CHECKPOINT": str(checkpoint_dir / "best_validation.pt"),
            "SMOKE_GATE": "FAIL",
            "optimization_seed_results": [_result_record(results[0])],
            "test_split_evaluated": False,
            "automatic_adjustment_performed": False,
            "interpretation": "deployment_aligned_input_plus_target_adaptation",
        }
        write_json(run_dir / "integrity_manifest.json", integrity)
        write_json(run_dir / "conclusion.json", conclusion)
        _report(
            run_dir=run_dir,
            dataset_audit=dataset_audit,
            smoke=smoke,
            results=results,
            paired_rows=[],
            conclusion=conclusion,
            integrity=integrity,
        )
        return run_dir

    # The test split remains untouched until all seeds finish and the checkpoint
    # has been selected solely by validation loss.
    for seed in config["training"]["optimization_seeds"]:
        seed = int(seed)
        if seed == smoke_seed:
            continue
        results.append(
            train_one_seed_v2(
                config,
                by_split["train"],
                by_split["validation"],
                optimization_seed=seed,
                initialization_checkpoint=initialization_checkpoint,
                checkpoint_dir=checkpoint_dir,
                device=device,
            )
        )
        write_csv(
            run_dir / "training_history.csv", _history_rows(results), HISTORY_FIELDS
        )

    best_result = min(results, key=lambda item: item.best_validation_loss)
    shutil.copy2(best_result.best_checkpoint, checkpoint_dir / "best_validation.pt")
    shutil.copy2(best_result.last_checkpoint, checkpoint_dir / "last.pt")
    selection_record = {
        "criterion": "validation_loss_only",
        "selected_seed": best_result.optimization_seed,
        "selected_best_epoch": best_result.best_epoch,
        "selected_validation_loss": best_result.best_validation_loss,
        "candidate_validation_losses": {
            str(item.optimization_seed): item.best_validation_loss for item in results
        },
        "test_metrics_available_at_selection_time": False,
        "long_horizon_diagnostics_available_at_selection_time": False,
        "closed_loop_outcomes_available_at_selection_time": False,
    }
    write_json(run_dir / "checkpoint_selection.json", selection_record)

    # Test evaluation begins only after the validation-only selection above.
    seed_metric_rows: list[dict[str, Any]] = []
    for result in results:
        model = load_v2_checkpoint_model(result.best_checkpoint, config, device)
        for split in ["train", "validation", "test"]:
            metrics, _ = evaluate_model(
                model,
                by_split[split],
                batch_size=int(config["training"]["batch_size"]),
                device=device,
            )
            seed_metric_rows.append(
                _metric_row(
                    model_name="v2_trained_on_v2_graph",
                    target_name="v2_long_horizon",
                    split=split,
                    seed=result.optimization_seed,
                    role="per_optimization_seed_diagnostic",
                    metrics=metrics,
                )
            )
    write_csv(run_dir / "seed_metrics.csv", seed_metric_rows, METRIC_FIELDS)

    v1_model, _, _ = load_initialized_model(initialization_checkpoint, config, device)
    v1_model.eval()
    v2_model = load_v2_checkpoint_model(
        checkpoint_dir / "best_validation.pt", config, device
    )
    test_examples = by_split["test"]
    v1_scores = model_score_vectors(
        v1_model,
        test_examples,
        batch_size=int(config["training"]["batch_size"]),
        device=device,
    )
    v2_scores = model_score_vectors(
        v2_model,
        test_examples,
        batch_size=int(config["training"]["batch_size"]),
        device=device,
    )
    paired_metric_rows: list[dict[str, Any]] = []
    for model_name, scores, seed in [
        ("v1_checkpoint_on_v2_graph", v1_scores, "v1_checkpoint"),
        ("v2_trained_on_v2_graph", v2_scores, best_result.optimization_seed),
    ]:
        for target_name in config["cross_target_evaluation"]:
            target_examples = examples_for_target(test_examples, target_name)
            paired_metric_rows.append(
                _metric_row(
                    model_name=model_name,
                    target_name=target_name,
                    split="test",
                    seed=seed,
                    role="primary_same_v2_graph_paired_comparison",
                    metrics=metrics_for_scores(target_examples, scores),
                )
            )
    paired_metric_rows.append(
        _metric_row(
            model_name="empirical_random_class_count_conditioned",
            target_name="v2_long_horizon",
            split="test",
            seed="analytic",
            role="reference_baseline",
            metrics=_random_metrics(test_examples),
        )
    )
    write_csv(
        run_dir / "v1_vs_v2_offline_metrics.csv", paired_metric_rows, METRIC_FIELDS
    )

    native_provenance = _native_v1_provenance(config)
    write_json(run_dir / "v1_native_metrics_provenance.json", native_provenance)
    outcomes = pd.read_parquet(_resolve(config["branch_outcomes_path"]))
    test_sample_ids = {item.sample_id for item in test_examples}
    outcome_rows = outcomes.loc[
        outcomes["sample_id"].isin(test_sample_ids)
    ].to_dict("records")
    v1_long = long_horizon_selection_diagnostics(
        test_examples, v1_scores, outcome_rows
    )
    v2_long = long_horizon_selection_diagnostics(
        test_examples, v2_scores, outcome_rows
    )
    analysis_rows = _analysis_rows(
        test_examples=test_examples,
        v1_scores=v1_scores,
        v2_scores=v2_scores,
        v1_long=v1_long,
        v2_long=v2_long,
    )
    write_csv(
        run_dir / "target_alignment_analysis.csv",
        analysis_rows,
        ["analysis", "left", "right", "metric", "value", "graph_count", "notes"],
    )

    paired_lookup = {
        (row["model"], row["target"]): row for row in paired_metric_rows
    }
    v1_primary = paired_lookup[("v1_checkpoint_on_v2_graph", "v2_long_horizon")]
    v2_primary = paired_lookup[("v2_trained_on_v2_graph", "v2_long_horizon")]
    top1_gain = float(v2_primary["top1_accuracy"]) - float(
        v1_primary["top1_accuracy"]
    )
    spearman_gain = float(v2_primary["spearman_proposal_only_mean"]) - float(
        v1_primary["spearman_proposal_only_mean"]
    )
    maximum_tier_gain = float(v2_long["maximum_tier_prediction_rate"]) - float(
        v1_long["maximum_tier_prediction_rate"]
    )
    null_rate_change = float(v2_primary["null_prediction_rate"]) - float(
        v1_primary["null_prediction_rate"]
    )
    offline_label = classify_gain(
        top1_gain,
        yes_threshold=float(config["decision_thresholds"]["offline_top1_gain_yes"]),
    )
    alignment_label = classify_gain(
        spearman_gain,
        yes_threshold=float(
            config["decision_thresholds"]["target_spearman_gain_yes"]
        ),
    )
    maximum_tier_nondecrease = maximum_tier_gain >= float(
        config["decision_thresholds"]["maximum_tier_rate_non_degradation"]
    )
    if (
        offline_label == "YES"
        and alignment_label == "YES"
        and maximum_tier_nondecrease
    ):
        long_signal = "YES"
    elif (top1_gain > 0.0 or spearman_gain > 0.0) and maximum_tier_nondecrease:
        long_signal = "WEAK"
    else:
        long_signal = "NO"
    training_valid = (
        smoke["status"] == "PASS"
        and len(results) == 3
        and all(item.gradients_finite for item in results)
        and all(math.isfinite(item.best_validation_loss) for item in results)
    )
    recommended = (
        "CLOSED_LOOP_V2_EVALUATION"
        if training_valid
        and long_signal
        in set(config["decision_thresholds"]["closed_loop_recommendation_requires_long_horizon_signal"])
        else "STOP"
    )
    conclusion = {
        "schema_version": SCHEMA_VERSION,
        "V2_TRAINING_VALID": "YES" if training_valid else "NO",
        "V2_OFFLINE_IMPROVEMENT_OVER_V1": offline_label,
        "V2_TARGET_ALIGNMENT_GAIN": alignment_label,
        "NULL_BEHAVIOR_CHANGE": (
            "YES"
            if abs(null_rate_change)
            >= float(
                config["decision_thresholds"]["null_prediction_rate_change_yes"]
            )
            else "NO"
        ),
        "LONG_HORIZON_SIGNAL_LEARNED": long_signal,
        "RECOMMENDED_NEXT_STEP": recommended,
        "FINAL_CHECKPOINT": str(checkpoint_dir / "best_validation.pt"),
        "SMOKE_GATE": smoke["status"],
        "best_checkpoint_source_seed": best_result.optimization_seed,
        "best_checkpoint_validation_loss": best_result.best_validation_loss,
        "checkpoint_selection": selection_record,
        "optimization_seed_results": [_result_record(item) for item in results],
        "paired_comparison": {
            "scope": "same frozen V2 test graphs",
            "interpretation": "deployment_aligned_input_plus_target_adaptation",
            "target_only_claim_allowed": False,
            "test_graph_count": len(test_examples),
            "v1_v2_top1_gain": top1_gain,
            "v1_v2_spearman_gain": spearman_gain,
            "v1_v2_null_prediction_rate_change": null_rate_change,
            "v1_v2_maximum_tier_prediction_rate_gain": maximum_tier_gain,
            "v1_metrics": v1_primary,
            "v2_metrics": v2_primary,
        },
        "long_horizon_diagnostics": {
            "v1_checkpoint_on_v2_graph": v1_long,
            "v2_trained_on_v2_graph": v2_long,
        },
        "v1_native_metrics_role": "provenance_only_not_paired_comparison",
        "automatic_adjustment_performed": False,
        "closed_loop_started": False,
        "stress_test_started": False,
        "interpretation_limit": (
            "Observed V1/V2 differences combine deployment-aligned H4 graph input "
            "with long-horizon target adaptation and cannot be attributed solely "
            "to label replacement."
        ),
    }
    integrity = _build_integrity(
        critical_before=critical_before,
        input_hashes=input_hashes,
        protocol_checks=protocol_checks,
        dataset_audit=dataset_audit,
        initialization_audit=initialization,
        seed_count=len(results),
        smoke_status=smoke["status"],
        test_evaluated=True,
    )
    final_payload = torch.load(
        checkpoint_dir / "best_validation.pt", map_location="cpu", weights_only=False
    )
    integrity["final_checkpoint_model_state_hash"] = model_state_hash(
        final_payload["model_state_dict"]
    )
    integrity["final_checkpoint_sha256"] = sha256_file(
        checkpoint_dir / "best_validation.pt"
    )
    integrity["all_acceptance_checks_passed"] = bool(
        training_valid
        and integrity["critical_files_unchanged"]
        and integrity["all_protocol_checks_passed"]
    )
    write_json(run_dir / "integrity_manifest.json", integrity)
    write_json(run_dir / "conclusion.json", conclusion)
    _report(
        run_dir=run_dir,
        dataset_audit=dataset_audit,
        smoke=smoke,
        results=results,
        paired_rows=paired_metric_rows,
        conclusion=conclusion,
        integrity=integrity,
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "training" / "gat_stage1_v2.json",
    )
    args = parser.parse_args()
    print(run(args.config))


if __name__ == "__main__":
    main()
