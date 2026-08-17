"""Run Stage-II coordination-supervision validation without closed-loop rollout."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from planning.gat.stage1_training import (  # noqa: E402
    load_model_checkpoint,
    load_stage1_examples,
    resolve_device,
    sha256_file,
)
from planning.gat.stage2_coordination_training import (  # noqa: E402
    SCHEMA_VERSION,
    JointStateGroup,
    Stage2TrainingRunResult,
    build_joint_state_groups,
    edge_ablation_audit,
    evaluate_joint_model,
    flatten_joint_groups,
    gradient_scale_audit,
    load_stage2_checkpoint,
    null_escape_audit,
    select_lambda_candidate,
    smoke_gate_result,
    split_joint_groups,
    train_one_stage2_seed,
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
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
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
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_ready(row.get(key)) for key in fieldnames})


def _hashes() -> dict[str, str]:
    return {name: sha256_file(path) for name, path in CRITICAL_FILES.items()}


def _assert_frozen_configuration(
    stage2: Mapping[str, Any], stage1: Mapping[str, Any]
) -> None:
    frozen = stage2["frozen_stage1"]
    supervision = stage1["supervision"]
    expected = {
        "H_preview": supervision["H_preview"],
        "H_label": supervision["H_label"],
        "target_name": supervision["target_name"],
        "target_feature_count": supervision["target_feature_count"],
        "soft_target_temperature": supervision["soft_target_temperature"],
    }
    for key, value in expected.items():
        if frozen[key] != value:
            raise ValueError(f"Stage-II changed frozen supervision field {key}")
    for key in (
        "optimizer",
        "learning_rate",
        "weight_decay",
        "batch_size",
        "max_epochs",
        "early_stopping_patience",
        "gradient_clip_norm",
        "optimization_seeds",
        "smoke_seed",
        "device",
        "num_workers",
    ):
        if stage2["training"][key] != stage1["training"][key]:
            raise ValueError(f"Stage-II changed frozen training field {key}")
    if stage2["overlap"]["probability_source"] != "full_null_plus_K_softmax":
        raise ValueError("overlap probabilities must come from full null+K softmax")
    if stage2["overlap"]["proposal_probability_renormalization"]:
        raise ValueError("proposal-only probability renormalization is forbidden")
    if stage2["overlap"]["uav_pair_order"] != "unordered_i_lt_j":
        raise ValueError("Stage-II must use unordered UAV pairs only")
    if stage2["batching"]["regular_ego_graphs_per_batch"] != 30:
        raise ValueError("regular joint batch must contain 30 ego graphs")


def _metric_row(
    *,
    model: str,
    optimization_seed: int | str,
    split: str,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    keys = (
        "graph_count",
        "selection_loss",
        "overlap_loss",
        "weighted_overlap_loss",
        "total_loss",
        "top1_accuracy",
        "top3_accuracy",
        "mrr",
        "spearman_proposal_only_mean",
        "spearman_valid_graph_count",
        "null_precision",
        "null_recall",
        "proposal_selection_accuracy",
        "mean_null_probability",
        "null_top1_selection_rate",
        "interaction_rich_null_top1_rate",
        "expected_overlap",
        "selected_pair_overlap",
        "risky_selection_rate",
        "selected_d_min_mean",
        "selected_d_min_median",
        "selected_d_min_p05",
        "interaction_rich_top1",
        "interaction_rich_mrr",
        "secondary_risk_top1",
        "secondary_risk_mrr",
    )
    return {
        "model": model,
        "optimization_seed": optimization_seed,
        "split": split,
        **{key: metrics.get(key) for key in keys},
    }


def _paired_rows(
    groups: Sequence[JointStateGroup],
    stage1_scores: Mapping[str, np.ndarray],
    stage2_scores: Mapping[str, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    counts = {
        "STAGE2_ONLY_CORRECT": 0,
        "STAGE1_ONLY_CORRECT": 0,
        "BOTH_CORRECT": 0,
        "BOTH_WRONG": 0,
    }
    for example in flatten_joint_groups(groups):
        reference = example.reference_class
        stage1_prediction = int(np.argmax(stage1_scores[example.sample_id]))
        stage2_prediction = int(np.argmax(stage2_scores[example.sample_id]))
        stage1_correct = stage1_prediction == reference
        stage2_correct = stage2_prediction == reference
        if stage1_correct and stage2_correct:
            category = "BOTH_CORRECT"
        elif stage1_correct:
            category = "STAGE1_ONLY_CORRECT"
        elif stage2_correct:
            category = "STAGE2_ONLY_CORRECT"
        else:
            category = "BOTH_WRONG"
        counts[category] += 1
        rows.append(
            {
                "sample_id": example.sample_id,
                "state_group_id": example.state_group_id,
                "scenario": example.scenario,
                "seed": example.seed,
                "ego_agent_id": example.ego_agent_id,
                "reference_class": reference,
                "stage1_prediction": stage1_prediction,
                "stage2_prediction": stage2_prediction,
                "stage1_correct": stage1_correct,
                "stage2_correct": stage2_correct,
                "paired_category": category,
            }
        )
    return rows, counts


def _edge_sensitivity(rows: Sequence[Mapping[str, Any]]) -> float:
    full = {
        row["scope"]: float(row["top1"])
        for row in rows
        if row["variant"] == "full" and row["scope"] in {"overall", "interaction_rich"}
    }
    declines: list[float] = []
    for variant in ("zero_st_edge", "remove_align_message"):
        for row in rows:
            if row["variant"] == variant and row["scope"] in full:
                declines.append(max(0.0, full[row["scope"]] - float(row["top1"])))
    return float(np.mean(declines)) if declines else 0.0


def _coordination_decision(
    comparisons: Sequence[Mapping[str, Any]],
    *,
    edge_gain: str,
    null_escape: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    family_checks: dict[str, dict[str, Any]] = {}
    expected_hits = [
        float(row["expected_overlap_relative_reduction"])
        >= float(config["expected_overlap_relative_reduction"])
        for row in comparisons
    ]
    risk_hits = [
        row["risky_selection_rate_reduction"] is not None
        and float(row["risky_selection_rate_reduction"])
        >= float(config["risky_selection_rate_absolute_reduction"])
        for row in comparisons
    ]
    dmin_hits = [
        row["selected_d_min_p05_increase"] is not None
        and float(row["selected_d_min_p05_increase"])
        >= float(config["selected_d_min_p05_increase_m"])
        for row in comparisons
    ]
    rich_hits = [
        max(
            float(row["interaction_rich_top1_change"]),
            float(row["interaction_rich_mrr_change"]),
        )
        >= float(config["interaction_rich_top1_or_mrr_increase"])
        for row in comparisons
    ]
    minimum_seeds = int(config["yes_minimum_consistent_seed_count"])
    for name, hits in (
        ("expected_overlap", expected_hits),
        ("risky_selection", risk_hits),
        ("selected_d_min_p05", dmin_hits),
        ("interaction_rich", rich_hits),
    ):
        family_checks[name] = {
            "hit_count": int(sum(hits)),
            "seed_count": len(hits),
            "achieved": int(sum(hits)) >= minimum_seeds,
        }
    family_checks["edge_usage_gain"] = {
        "hit_count": 1 if edge_gain in {"YES", "WEAK"} else 0,
        "seed_count": 1,
        "achieved": edge_gain == "YES",
    }
    achieved_count = sum(bool(item["achieved"]) for item in family_checks.values())
    mean_top1_change = float(np.mean([row["top1_change"] for row in comparisons]))
    top1_acceptable = mean_top1_change >= -float(
        config["overall_top1_maximum_regression"]
    )
    if (
        achieved_count >= int(config["yes_minimum_metric_families"])
        and top1_acceptable
        and null_escape == "NO"
    ):
        signal = "YES"
    elif achieved_count >= int(config["weak_minimum_metric_families"]) and top1_acceptable:
        signal = "WEAK"
    else:
        signal = "NO"
    return {
        "SPATIOTEMPORAL_COORDINATION_SIGNAL": signal,
        "achieved_metric_family_count": achieved_count,
        "family_checks": family_checks,
        "mean_test_top1_change": mean_top1_change,
        "overall_top1_regression_acceptable": top1_acceptable,
    }


def _comparison_row(
    seed: int,
    stage1: Mapping[str, Any],
    stage2: Mapping[str, Any],
) -> dict[str, Any]:
    expected_before = float(stage1["expected_overlap"])
    expected_after = float(stage2["expected_overlap"])
    before_risk = stage1["risky_selection_rate"]
    after_risk = stage2["risky_selection_rate"]
    before_dmin = stage1["selected_d_min_p05"]
    after_dmin = stage2["selected_d_min_p05"]
    return {
        "optimization_seed": seed,
        "top1_change": float(stage2["top1_accuracy"]) - float(stage1["top1_accuracy"]),
        "mrr_change": float(stage2["mrr"]) - float(stage1["mrr"]),
        "expected_overlap_change": expected_after - expected_before,
        "expected_overlap_relative_reduction": (
            (expected_before - expected_after) / expected_before
            if expected_before > 1.0e-15
            else 0.0
        ),
        "selected_pair_overlap_change": (
            None
            if stage1["selected_pair_overlap"] is None
            or stage2["selected_pair_overlap"] is None
            else float(stage2["selected_pair_overlap"])
            - float(stage1["selected_pair_overlap"])
        ),
        "risky_selection_rate_reduction": (
            None
            if before_risk is None or after_risk is None
            else float(before_risk) - float(after_risk)
        ),
        "selected_d_min_mean_change": (
            None
            if stage1["selected_d_min_mean"] is None
            or stage2["selected_d_min_mean"] is None
            else float(stage2["selected_d_min_mean"])
            - float(stage1["selected_d_min_mean"])
        ),
        "selected_d_min_p05_increase": (
            None
            if before_dmin is None or after_dmin is None
            else float(after_dmin) - float(before_dmin)
        ),
        "interaction_rich_top1_change": float(stage2["interaction_rich_top1"])
        - float(stage1["interaction_rich_top1"]),
        "interaction_rich_mrr_change": float(stage2["interaction_rich_mrr"])
        - float(stage1["interaction_rich_mrr"]),
        "null_top1_rate_change": float(stage2["null_top1_selection_rate"])
        - float(stage1["null_top1_selection_rate"]),
        "mean_null_probability_change": float(stage2["mean_null_probability"])
        - float(stage1["mean_null_probability"]),
    }


def _write_empty_remaining_artifacts(run_dir: Path) -> None:
    definitions = {
        "training_history.csv": ["optimization_seed", "lambda_overlap", "epoch"],
        "seed_metrics.csv": ["model", "optimization_seed", "split"],
        "test_metrics.csv": ["model", "optimization_seed", "split"],
        "interaction_metrics.csv": ["model", "optimization_seed", "split"],
        "stage1_vs_stage2.csv": ["optimization_seed"],
        "paired_analysis.csv": ["sample_id", "paired_category"],
        "stage2_edge_ablation.csv": ["variant", "scope"],
    }
    for name, fields in definitions.items():
        if not (run_dir / name).exists():
            write_csv(run_dir / name, [], fields)


def _final_report(conclusion: Mapping[str, Any], run_dir: Path) -> str:
    selected = conclusion.get("selected_lambda")
    best = conclusion.get("BEST_CHECKPOINT")
    lines = [
        "# Spatiotemporal Edge-Enhanced GAT Stage-II Report",
        "",
        "## Frozen protocol",
        "",
        "- Stage-I architecture, dataset, split, optimizer and supervision are unchanged.",
        "- Overlap uses existing H=4 proposal trajectories only.",
        "- Null has no synthetic trajectory; proposal probabilities come from full null+K softmax without renormalization.",
        "- Each three-UAV state uses unordered pairs (0,1), (0,2), (1,2).",
        "- No completion loss, new feature, new scenario, rollout, or closed-loop evaluation was used.",
        "",
        "## Decisions",
        "",
    ]
    for key in (
        "STAGE1_SPATIOTEMPORAL_EDGE_USAGE",
        "OVERLAP_SUPERVISION_SIGNAL",
        "STAGE2_TRAINING_SIGNAL",
        "STAGE2_GENERALIZATION_SIGNAL",
        "INTERACTION_RISK_REDUCTION",
        "INTERACTION_RICH_GAIN",
        "STAGE2_SPATIOTEMPORAL_EDGE_USAGE",
        "EDGE_USAGE_GAIN_OVER_STAGE1",
        "SPATIOTEMPORAL_COORDINATION_SIGNAL",
        "STAGE2_ADDS_OVER_STAGE1",
        "NULL_ESCAPE_DETECTED",
        "OVERLAP_SCALE_ADEQUACY",
        "RECOMMENDED_GAT_CHECKPOINT",
        "PROCEED_TO_CLOSED_LOOP",
    ):
        lines.append(f"- `{key} = {conclusion.get(key, 'NOT_RUN')}`")
    lines.extend(
        [
            "",
            f"- Selected lambda: `{selected}`",
            f"- Best checkpoint: `{best}`",
            f"- Next step: `{conclusion.get('NEXT_STEP')}`",
            "",
            "## Stop rule",
            "",
            "Stage-II stopped after offline validation. No closed-loop or additional tuning was started.",
        ]
    )
    lambda_rows = conclusion.get("lambda_validation", [])
    if lambda_rows:
        lines.extend(
            [
                "",
                "## Lambda validation",
                "",
                "| lambda | best epoch | val Top-1 | expected overlap | relative reduction | eligible |",
                "|---:|---:|---:|---:|---:|:---:|",
            ]
        )
        for row in lambda_rows:
            lines.append(
                "| {lambda_overlap:.2f} | {best_epoch} | {top1_accuracy:.4f} | "
                "{expected_overlap:.8f} | {expected_overlap_relative_reduction:.3e} | "
                "{eligible} |".format(**row)
            )
    gradient_rows = conclusion.get("gradient_scale_audit", [])
    if gradient_rows:
        lines.extend(
            [
                "",
                "## Loss and gradient scale",
                "",
                "| lambda | weighted loss ratio | gradient ratio |",
                "|---:|---:|---:|",
            ]
        )
        for row in gradient_rows:
            lines.append(
                f"| {float(row['lambda_overlap']):.2f} | "
                f"{float(row['OVERLAP_WEIGHTED_LOSS_RATIO']):.6e} | "
                f"{float(row['OVERLAP_GRADIENT_RATIO']):.6e} |"
            )
    if conclusion.get("stop_reason"):
        lines.extend(["", "## Gate result", "", str(conclusion["stop_reason"])])
    if conclusion.get("INTERPRETATION_SCOPE"):
        lines.extend(
            [
                "",
                "## Interpretation scope",
                "",
                str(conclusion["INTERPRETATION_SCOPE"]),
            ]
        )
    return "\n".join(lines) + "\n"


def run(config_path: Path) -> Path:
    config_path = _resolve(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    stage1_config_path = _resolve(config["stage1_config"])
    stage1_config = json.loads(stage1_config_path.read_text(encoding="utf-8"))
    _assert_frozen_configuration(config, stage1_config)
    stage1_checkpoint = _resolve(config["stage1_checkpoint"])
    if not stage1_checkpoint.is_file():
        raise FileNotFoundError(stage1_checkpoint)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = _resolve(config["output_root"]) / timestamp
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    hashes_before = _hashes()
    resolved_config = {
        **config,
        "resolved_stage1_config": str(stage1_config_path),
        "resolved_stage1_checkpoint": str(stage1_checkpoint),
        "resolved_dataset_dir": str(_resolve(stage1_config["dataset_dir"])),
        "resolved_output_dir": str(run_dir),
        "created_at": datetime.now().isoformat(),
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    write_json(run_dir / "config.json", resolved_config)

    examples, dataset_audit, split_rows = load_stage1_examples(
        _resolve(stage1_config["dataset_dir"]),
        split_config=stage1_config["dataset_split"],
        supervision_config=stage1_config["supervision"],
        interaction_config=stage1_config["interaction_diagnostic"],
    )
    groups = build_joint_state_groups(
        examples, required_horizon=int(config["overlap"]["horizon_steps"])
    )
    by_split = split_joint_groups(groups)
    if any(len(group.examples) != 3 for group in groups):
        raise RuntimeError("joint-state completeness audit failed")
    device = resolve_device(config["training"]["device"])
    graph_budget = int(config["batching"]["stage1_graph_budget"])
    stage1_model = load_model_checkpoint(stage1_checkpoint, stage1_config, device)

    stage1_metrics: dict[str, dict[str, Any]] = {}
    stage1_scores: dict[str, dict[str, np.ndarray]] = {}
    stage1_pairs: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "validation", "test"):
        metrics, scores, pair_records = evaluate_joint_model(
            stage1_model,
            by_split[split],
            lambda_overlap=0.0,
            graph_budget=graph_budget,
            device=device,
        )
        stage1_metrics[split] = metrics
        stage1_scores[split] = scores
        stage1_pairs[split] = pair_records

    stage1_edge_rows, stage1_edge_usage = edge_ablation_audit(
        stage1_model,
        by_split["test"],
        graph_budget=graph_budget,
        device=device,
        config=config["edge_ablation"],
    )
    edge_fields = [
        "variant",
        "scope",
        "graph_count",
        "top1",
        "top3",
        "mrr",
        "spearman",
        "spearman_n",
        "max_abs_logit_delta_vs_full",
        "mean_abs_logit_delta_vs_full",
        "top1_changed_count_vs_full",
        "full_ranking_changed_count_vs_full",
    ]
    write_csv(run_dir / "stage1_edge_ablation.csv", stage1_edge_rows, edge_fields)

    lambdas = [float(value) for value in config["lambda_validation"]["candidates"]]
    gradient_rows = gradient_scale_audit(
        stage1_model,
        by_split["train"],
        lambdas,
        graph_budget=graph_budget,
        device=device,
    )
    overlap_audit = {
        "schema_version": SCHEMA_VERSION,
        "graph_count": len(examples),
        "joint_state_count": len(groups),
        "joint_states_by_split": {name: len(value) for name, value in by_split.items()},
        "unordered_pairs_per_joint_state": 3,
        "pair_order": [[0, 1], [0, 2], [1, 2]],
        "trajectory_horizon": int(config["overlap"]["horizon_steps"]),
        "trajectory_source": config["overlap"]["trajectory_source"],
        "null_trajectory_available": False,
        "class_domain": "proposal_proposal_only",
        "probability_source": "full_null_plus_K_softmax",
        "proposal_probability_renormalization": False,
        "gradient_scale_audit": gradient_rows,
        "stage1_validation_interaction_metrics": {
            key: stage1_metrics["validation"].get(key)
            for key in (
                "expected_overlap",
                "selected_pair_overlap",
                "risky_selection_rate",
                "selected_pair_valid_count",
                "selected_pair_null_involved_count",
                "mean_null_probability",
                "null_top1_selection_rate",
            )
        },
    }
    write_json(run_dir / "overlap_loss_audit.json", overlap_audit)

    lambda_results: dict[float, Stage2TrainingRunResult] = {}
    lambda_rows: list[dict[str, Any]] = []
    all_history: list[dict[str, Any]] = []
    for lambda_overlap in lambdas:
        result = train_one_stage2_seed(
            config,
            stage1_config,
            by_split["train"],
            by_split["validation"],
            stage1_checkpoint=stage1_checkpoint,
            lambda_overlap=lambda_overlap,
            optimization_seed=int(config["lambda_validation"]["selection_seed"]),
            checkpoint_dir=checkpoint_dir / "lambda_validation",
        )
        lambda_results[lambda_overlap] = result
        all_history.extend(asdict(item) for item in result.history)
        best_model = load_stage2_checkpoint(result.best_checkpoint, stage1_config, device)
        validation_metrics, _, validation_pairs = evaluate_joint_model(
            best_model,
            by_split["validation"],
            lambda_overlap=lambda_overlap,
            graph_budget=graph_budget,
            device=device,
        )
        null_audit = null_escape_audit(
            stage1_metrics["validation"],
            validation_metrics,
            stage1_pairs["validation"],
            validation_pairs,
            config["null_escape_audit"],
        )
        gradient = next(
            row for row in gradient_rows if row["lambda_overlap"] == lambda_overlap
        )
        lambda_rows.append(
            {
                "lambda_overlap": lambda_overlap,
                "best_epoch": result.best_epoch,
                "epochs_completed": result.epochs_completed,
                "validation_total_loss": validation_metrics["total_loss"],
                "validation_selection_loss": validation_metrics["selection_loss"],
                "validation_overlap_loss": validation_metrics["overlap_loss"],
                "top1_accuracy": validation_metrics["top1_accuracy"],
                "mrr": validation_metrics["mrr"],
                "expected_overlap": validation_metrics["expected_overlap"],
                "selected_pair_overlap": validation_metrics["selected_pair_overlap"],
                "risky_selection_rate": validation_metrics["risky_selection_rate"],
                "selected_d_min_p05": validation_metrics["selected_d_min_p05"],
                "mean_null_probability": validation_metrics["mean_null_probability"],
                "null_top1_selection_rate": validation_metrics[
                    "null_top1_selection_rate"
                ],
                "NULL_ESCAPE_DETECTED": null_audit["NULL_ESCAPE_DETECTED"],
                "mass_transfer_fraction_of_reduction": null_audit[
                    "mass_transfer_fraction_of_reduction"
                ],
                "OVERLAP_WEIGHTED_LOSS_RATIO": gradient[
                    "OVERLAP_WEIGHTED_LOSS_RATIO"
                ],
                "g_select": gradient["g_select"],
                "g_overlap_weighted": gradient["g_overlap_weighted"],
                "OVERLAP_GRADIENT_RATIO": gradient["OVERLAP_GRADIENT_RATIO"],
                "best_checkpoint": str(result.best_checkpoint),
            }
        )
    selected_row, overlap_signal, lambda_detail = select_lambda_candidate(
        lambda_rows,
        stage1_metrics["validation"],
        config["lambda_validation"],
    )
    lambda_fields = list(dict.fromkeys(key for row in lambda_rows for key in row))
    write_csv(run_dir / "lambda_validation.csv", lambda_rows, lambda_fields)
    history_fields = list(dict.fromkeys(key for row in all_history for key in row))
    write_csv(run_dir / "training_history.csv", all_history, history_fields)

    integrity = {
        "schema_version": SCHEMA_VERSION,
        "critical_file_hashes_before": hashes_before,
        "dataset_audit": dataset_audit,
        "split_group_leakage_count": 0,
        "joint_state_count": len(groups),
        "joint_state_size": 3,
        "regular_batch_ego_graph_count": 30,
        "stage1_checkpoint_sha256": sha256_file(stage1_checkpoint),
        "test_used_for_lambda_selection": False,
        "closed_loop_evaluation_run": False,
        "completion_loss_enabled": False,
        "proposal_probability_renormalized": False,
        "null_trajectory_synthesized": False,
        "unordered_uav_pairs_only": True,
        "stage2_optimization_seeds_1_and_2_started_before_gate": False,
    }

    if selected_row is None or overlap_signal == "NO":
        diagnostic_checkpoint = None
        diagnostic_lambda = None
        if selected_row is not None:
            diagnostic_lambda = float(selected_row["lambda_overlap"])
            diagnostic_result = lambda_results[diagnostic_lambda]
            shutil.copy2(
                diagnostic_result.best_checkpoint,
                checkpoint_dir / "best_validation.pt",
            )
            shutil.copy2(diagnostic_result.last_checkpoint, checkpoint_dir / "last.pt")
            diagnostic_checkpoint = str(checkpoint_dir / "best_validation.pt")
        maximum_gradient_ratio = max(
            float(row["OVERLAP_GRADIENT_RATIO"]) for row in gradient_rows
        )
        scale_adequacy = (
            "ADEQUATE"
            if maximum_gradient_ratio
            >= float(
                config["gradient_scale_audit"][
                    "minimum_adequate_gradient_ratio"
                ]
            )
            else "WEAK"
        )
        _write_empty_remaining_artifacts(run_dir)
        integrity["critical_file_hashes_after"] = _hashes()
        integrity["critical_files_unchanged"] = (
            integrity["critical_file_hashes_before"]
            == integrity["critical_file_hashes_after"]
        )
        write_json(run_dir / "integrity_manifest.json", integrity)
        conclusion = {
            "schema_version": SCHEMA_VERSION,
            "STAGE1_SPATIOTEMPORAL_EDGE_USAGE": stage1_edge_usage,
            "OVERLAP_SUPERVISION_SIGNAL": "NO",
            "STAGE2_TRAINING_SIGNAL": "NO",
            "STAGE2_GENERALIZATION_SIGNAL": "NOT_ESTABLISHED",
            "INTERACTION_RISK_REDUCTION": "NO",
            "INTERACTION_RICH_GAIN": "NO",
            "STAGE2_SPATIOTEMPORAL_EDGE_USAGE": "NOT_ESTABLISHED",
            "EDGE_USAGE_GAIN_OVER_STAGE1": "NO",
            "SPATIOTEMPORAL_COORDINATION_SIGNAL": "NOT_ESTABLISHED",
            "STAGE2_ADDS_OVER_STAGE1": "NO",
            "NULL_ESCAPE_DETECTED": (
                "NO"
                if selected_row is None
                else selected_row["NULL_ESCAPE_DETECTED"]
            ),
            "OVERLAP_SCALE_ADEQUACY": scale_adequacy,
            "RECOMMENDED_GAT_CHECKPOINT": "STAGE1",
            "PROCEED_TO_CLOSED_LOOP": "NO",
            "BEST_CHECKPOINT": str(stage1_checkpoint),
            "selected_lambda": None,
            "diagnostic_lambda": diagnostic_lambda,
            "STAGE2_DIAGNOSTIC_CHECKPOINT": diagnostic_checkpoint,
            "lambda_decision": lambda_detail,
            "lambda_validation": lambda_rows,
            "gradient_scale_audit": gradient_rows,
            "stop_reason": (
                "No eligible lambda satisfied the frozen gate."
                if selected_row is None
                else "All trained updates failed to improve the epoch-0 Stage-I "
                "initialization; validation expected-overlap changes remained at "
                "floating-point scale."
            ),
            "INTERPRETATION_SCOPE": (
                "Coordination improvement was not established under the tested "
                "overlap-regularization strength. This result does not establish "
                "that overlap supervision itself is ineffective."
                if scale_adequacy == "WEAK"
                else "Coordination improvement was not established under the tested "
                "configuration."
            ),
            "NEXT_STEP": "Retain Stage-I; tested lambda set produced no validation overlap signal",
            "automatic_next_stage_started": False,
        }
        write_json(run_dir / "conclusion.json", conclusion)
        (run_dir / "FINAL_REPORT.md").write_text(
            _final_report(conclusion, run_dir), encoding="utf-8"
        )
        return run_dir

    selected_lambda = float(selected_row["lambda_overlap"])
    selected_gradient = next(
        row for row in gradient_rows if row["lambda_overlap"] == selected_lambda
    )
    seed_results: dict[int, Stage2TrainingRunResult] = {
        0: lambda_results[selected_lambda]
    }
    smoke = smoke_gate_result(
        seed_results[0], selected_gradient, config["smoke_gate"]
    )
    if smoke["status"] != "PASS":
        diagnostic_result = seed_results[0]
        shutil.copy2(
            diagnostic_result.best_checkpoint,
            checkpoint_dir / "best_validation.pt",
        )
        shutil.copy2(diagnostic_result.last_checkpoint, checkpoint_dir / "last.pt")
        _write_empty_remaining_artifacts(run_dir)
        scale_adequacy = (
            "ADEQUATE"
            if float(selected_gradient["OVERLAP_GRADIENT_RATIO"])
            >= float(
                config["gradient_scale_audit"][
                    "minimum_adequate_gradient_ratio"
                ]
            )
            else "WEAK"
        )
        integrity["critical_file_hashes_after"] = _hashes()
        integrity["critical_files_unchanged"] = (
            integrity["critical_file_hashes_before"]
            == integrity["critical_file_hashes_after"]
        )
        write_json(run_dir / "integrity_manifest.json", integrity)
        conclusion = {
            "schema_version": SCHEMA_VERSION,
            "STAGE1_SPATIOTEMPORAL_EDGE_USAGE": stage1_edge_usage,
            "OVERLAP_SUPERVISION_SIGNAL": overlap_signal,
            "STAGE2_TRAINING_SIGNAL": "NO",
            "STAGE2_GENERALIZATION_SIGNAL": "NOT_ESTABLISHED",
            "INTERACTION_RISK_REDUCTION": "NOT_ESTABLISHED",
            "INTERACTION_RICH_GAIN": "NOT_ESTABLISHED",
            "STAGE2_SPATIOTEMPORAL_EDGE_USAGE": "NOT_ESTABLISHED",
            "EDGE_USAGE_GAIN_OVER_STAGE1": "NO",
            "SPATIOTEMPORAL_COORDINATION_SIGNAL": "NOT_ESTABLISHED",
            "STAGE2_ADDS_OVER_STAGE1": "NO",
            "NULL_ESCAPE_DETECTED": selected_row["NULL_ESCAPE_DETECTED"],
            "OVERLAP_SCALE_ADEQUACY": scale_adequacy,
            "RECOMMENDED_GAT_CHECKPOINT": "STAGE1",
            "PROCEED_TO_CLOSED_LOOP": "NO",
            "BEST_CHECKPOINT": str(stage1_checkpoint),
            "selected_lambda": selected_lambda,
            "STAGE2_DIAGNOSTIC_CHECKPOINT": str(
                checkpoint_dir / "best_validation.pt"
            ),
            "lambda_validation": lambda_rows,
            "gradient_scale_audit": gradient_rows,
            "stop_reason": "Selected lambda failed the frozen Stage-II smoke gate.",
            "INTERPRETATION_SCOPE": (
                "Coordination improvement was not established under the tested "
                "overlap-regularization strength. This result does not establish "
                "that overlap supervision itself is ineffective."
                if scale_adequacy == "WEAK"
                else "Coordination improvement was not established because the "
                "frozen smoke gate failed."
            ),
            "SMOKE_GATE": smoke,
            "NEXT_STEP": "Stop: Stage-II smoke gate failed",
            "automatic_next_stage_started": False,
        }
        write_json(run_dir / "conclusion.json", conclusion)
        (run_dir / "FINAL_REPORT.md").write_text(
            _final_report(conclusion, run_dir), encoding="utf-8"
        )
        return run_dir

    for seed in config["training"]["optimization_seeds"]:
        seed = int(seed)
        if seed == 0:
            continue
        result = train_one_stage2_seed(
            config,
            stage1_config,
            by_split["train"],
            by_split["validation"],
            stage1_checkpoint=stage1_checkpoint,
            lambda_overlap=selected_lambda,
            optimization_seed=seed,
            checkpoint_dir=checkpoint_dir,
        )
        seed_results[seed] = result
        all_history.extend(asdict(item) for item in result.history)
    history_fields = list(dict.fromkeys(key for row in all_history for key in row))
    write_csv(run_dir / "training_history.csv", all_history, history_fields)

    seed_metrics_rows: list[dict[str, Any]] = []
    interaction_rows: list[dict[str, Any]] = []
    stage2_metrics_by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    stage2_scores_by_seed: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    stage2_pairs_by_seed: dict[int, dict[str, list[dict[str, Any]]]] = {}
    null_audits_by_seed: dict[int, dict[str, Any]] = {}
    comparison_rows: list[dict[str, Any]] = []
    for seed, result in sorted(seed_results.items()):
        model = load_stage2_checkpoint(result.best_checkpoint, stage1_config, device)
        stage2_metrics_by_seed[seed] = {}
        stage2_scores_by_seed[seed] = {}
        stage2_pairs_by_seed[seed] = {}
        for split in ("train", "validation", "test"):
            metrics, scores, pairs = evaluate_joint_model(
                model,
                by_split[split],
                lambda_overlap=selected_lambda,
                graph_budget=graph_budget,
                device=device,
            )
            stage2_metrics_by_seed[seed][split] = metrics
            stage2_scores_by_seed[seed][split] = scores
            stage2_pairs_by_seed[seed][split] = pairs
            row = _metric_row(
                model="stage2", optimization_seed=seed, split=split, metrics=metrics
            )
            seed_metrics_rows.append(row)
            interaction_rows.append(row)
        null_audits_by_seed[seed] = null_escape_audit(
            stage1_metrics["test"],
            stage2_metrics_by_seed[seed]["test"],
            stage1_pairs["test"],
            stage2_pairs_by_seed[seed]["test"],
            config["null_escape_audit"],
        )
        comparison_rows.append(
            _comparison_row(
                seed, stage1_metrics["test"], stage2_metrics_by_seed[seed]["test"]
            )
        )
    stage1_reference_row = _metric_row(
        model="stage1", optimization_seed="reference", split="test", metrics=stage1_metrics["test"]
    )
    interaction_rows.insert(0, stage1_reference_row)
    metric_fields = list(dict.fromkeys(key for row in seed_metrics_rows for key in row))
    write_csv(run_dir / "seed_metrics.csv", seed_metrics_rows, metric_fields)
    write_csv(
        run_dir / "test_metrics.csv",
        [row for row in seed_metrics_rows if row["split"] == "test"],
        metric_fields,
    )
    interaction_fields = list(dict.fromkeys(key for row in interaction_rows for key in row))
    write_csv(run_dir / "interaction_metrics.csv", interaction_rows, interaction_fields)
    comparison_fields = list(dict.fromkeys(key for row in comparison_rows for key in row))
    write_csv(run_dir / "stage1_vs_stage2.csv", comparison_rows, comparison_fields)

    best_seed = min(
        seed_results,
        key=lambda seed: seed_results[seed].best_validation_total_loss,
    )
    best_result = seed_results[best_seed]
    best_model = load_stage2_checkpoint(best_result.best_checkpoint, stage1_config, device)
    shutil.copy2(best_result.best_checkpoint, checkpoint_dir / "best_validation.pt")
    shutil.copy2(best_result.last_checkpoint, checkpoint_dir / "last.pt")
    best_test_metrics = stage2_metrics_by_seed[best_seed]["test"]
    best_test_scores = stage2_scores_by_seed[best_seed]["test"]
    paired_rows, paired_counts = _paired_rows(
        by_split["test"], stage1_scores["test"], best_test_scores
    )
    paired_fields = list(dict.fromkeys(key for row in paired_rows for key in row))
    write_csv(run_dir / "paired_analysis.csv", paired_rows, paired_fields)

    stage2_edge_rows, stage2_edge_usage = edge_ablation_audit(
        best_model,
        by_split["test"],
        graph_budget=graph_budget,
        device=device,
        config=config["edge_ablation"],
    )
    write_csv(run_dir / "stage2_edge_ablation.csv", stage2_edge_rows, edge_fields)
    stage1_sensitivity = _edge_sensitivity(stage1_edge_rows)
    stage2_sensitivity = _edge_sensitivity(stage2_edge_rows)
    class_rank = {"NONE": 0, "WEAK": 1, "CLEAR": 2, "NOT_ESTABLISHED": -1}
    if class_rank[stage2_edge_usage] > class_rank[stage1_edge_usage]:
        edge_gain = "YES" if stage2_edge_usage == "CLEAR" else "WEAK"
    elif stage2_sensitivity > stage1_sensitivity + 0.01:
        edge_gain = "WEAK"
    else:
        edge_gain = "NO"

    best_null_audit = null_audits_by_seed[best_seed]
    coordination = _coordination_decision(
        comparison_rows,
        edge_gain=edge_gain,
        null_escape=best_null_audit["NULL_ESCAPE_DETECTED"],
        config=config["coordination_decision"],
    )
    coordination_signal = coordination["SPATIOTEMPORAL_COORDINATION_SIGNAL"]
    risk_families = coordination["family_checks"]
    risk_achieved = sum(
        bool(risk_families[name]["achieved"])
        for name in ("expected_overlap", "risky_selection", "selected_d_min_p05")
    )
    interaction_risk_reduction = (
        "YES" if risk_achieved >= 2 else "WEAK" if risk_achieved == 1 else "NO"
    )
    rich_mean = float(
        np.mean([row["interaction_rich_top1_change"] for row in comparison_rows])
    )
    rich_gain = (
        "YES"
        if risk_families["interaction_rich"]["achieved"]
        else "WEAK"
        if rich_mean > 0.0
        else "NO"
    )
    test_random = float(stage1_metrics["test"]["empirical_random_top1_accuracy"])
    stage2_test_top1_mean = float(
        np.mean([stage2_metrics_by_seed[s]["test"]["top1_accuracy"] for s in seed_results])
    )
    generalization = "YES" if stage2_test_top1_mean - test_random >= 0.05 else "WEAK"
    training_signal = "YES" if smoke["status"] == "PASS" else "NO"
    adds_over = (
        "YES"
        if coordination_signal == "YES"
        else "WEAK"
        if coordination_signal == "WEAK"
        else "NO"
    )
    gradient_ratio = float(selected_gradient["OVERLAP_GRADIENT_RATIO"])
    scale_adequacy = (
        "ADEQUATE"
        if coordination_signal == "YES"
        or gradient_ratio
        >= float(config["gradient_scale_audit"]["minimum_adequate_gradient_ratio"])
        else "WEAK"
    )
    recommended = "STAGE2" if adds_over == "YES" else "STAGE1"
    proceed = "YES" if adds_over == "YES" else "NO"

    integrity["critical_file_hashes_after"] = _hashes()
    integrity["critical_files_unchanged"] = (
        integrity["critical_file_hashes_before"] == integrity["critical_file_hashes_after"]
    )
    integrity.update(
        {
            "selected_lambda": selected_lambda,
            "smoke_gate": smoke,
            "optimization_seed_count": len(seed_results),
            "test_used_for_lambda_selection": False,
            "closed_loop_evaluation_run": False,
            "completion_loss_enabled": False,
            "proposal_probability_renormalized": False,
            "unordered_uav_pairs_only": True,
        }
    )
    write_json(run_dir / "integrity_manifest.json", integrity)
    conclusion = {
        "schema_version": SCHEMA_VERSION,
        "STAGE1_SPATIOTEMPORAL_EDGE_USAGE": stage1_edge_usage,
        "OVERLAP_SUPERVISION_SIGNAL": overlap_signal,
        "STAGE2_TRAINING_SIGNAL": training_signal,
        "STAGE2_GENERALIZATION_SIGNAL": generalization,
        "INTERACTION_RISK_REDUCTION": interaction_risk_reduction,
        "INTERACTION_RICH_GAIN": rich_gain,
        "STAGE2_SPATIOTEMPORAL_EDGE_USAGE": stage2_edge_usage,
        "EDGE_USAGE_GAIN_OVER_STAGE1": edge_gain,
        "SPATIOTEMPORAL_COORDINATION_SIGNAL": coordination_signal,
        "STAGE2_ADDS_OVER_STAGE1": adds_over,
        "NULL_ESCAPE_DETECTED": best_null_audit["NULL_ESCAPE_DETECTED"],
        "OVERLAP_SCALE_ADEQUACY": scale_adequacy,
        "RECOMMENDED_GAT_CHECKPOINT": recommended,
        "PROCEED_TO_CLOSED_LOOP": proceed,
        "BEST_CHECKPOINT": str(
            checkpoint_dir / "best_validation.pt" if recommended == "STAGE2" else stage1_checkpoint
        ),
        "selected_lambda": selected_lambda,
        "selected_gradient_scale": selected_gradient,
        "lambda_decision": lambda_detail,
        "SMOKE_GATE": smoke,
        "best_stage2_seed": best_seed,
        "best_stage2_validation_total_loss": best_result.best_validation_total_loss,
        "stage1_edge_sensitivity": stage1_sensitivity,
        "stage2_edge_sensitivity": stage2_sensitivity,
        "coordination_decision": coordination,
        "null_escape_audit": best_null_audit,
        "paired_stage1_stage2": paired_counts,
        "stage2_test_top1_mean": stage2_test_top1_mean,
        "automatic_next_stage_started": False,
        "NEXT_STEP": (
            "Review Stage-II offline evidence before separately authorized closed-loop"
            if proceed == "YES"
            else "Retain Stage-I; do not proceed without a new coordination-supervision design"
        ),
    }
    write_json(run_dir / "conclusion.json", conclusion)
    (run_dir / "FINAL_REPORT.md").write_text(
        _final_report(conclusion, run_dir), encoding="utf-8"
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "training" / "gat_stage2_coordination.json",
    )
    args = parser.parse_args()
    output = run(args.config)
    print(output)


if __name__ == "__main__":
    main()
