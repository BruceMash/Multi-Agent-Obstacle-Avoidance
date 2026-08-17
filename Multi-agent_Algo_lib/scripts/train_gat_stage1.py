"""Run the frozen-architecture GAT Stage-I offline training protocol."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from planning.gat.stage1_training import (  # noqa: E402
    SCHEMA_VERSION,
    Stage1Example,
    completion_signal_audit,
    compute_offline_metrics,
    empirical_random_metrics,
    evaluate_model,
    fp_shep_score_vectors,
    load_model_checkpoint,
    load_stage1_examples,
    proposal_score_vectors,
    read_csv,
    resolve_device,
    sha256_file,
    smoke_gate_result,
    stable_hash,
    train_one_seed,
    training_result_record,
)


CRITICAL_FILES = {
    "proposal": REPO_ROOT / "Guidance" / "reference_point_proposal_demo.py",
    "fp_shep": REPO_ROOT / "planning" / "policy_preview.py",
    "graph_builder": REPO_ROOT / "planning" / "heterogeneous_candidate_graph.py",
    "gat_selector": REPO_ROOT / "planning" / "gat" / "candidate_selector.py",
    "gat_layer": REPO_ROOT / "planning" / "gat" / "edge_enhanced_gat.py",
    "sac_actor": REPO_ROOT / "baseline" / "sac" / "net.py",
    "dmp": REPO_ROOT / "Controller" / "dmp_rl.py",
    "environment": REPO_ROOT / "Environment" / "multi_agent_dmp_env.py",
}


def _resolve(path_value: str | Path) -> Path:
    path = Path(path_value)
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
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_ready(value), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_ready(row.get(key)) for key in fieldnames})


def _metric_row(
    *,
    method: str,
    split: str,
    metrics: Mapping[str, Any],
    optimization_seed: int | str,
    subset_type: str = "overall",
    subset_name: str = "overall",
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "optimization_seed": optimization_seed,
        "split": split,
        "subset_type": subset_type,
        "subset_name": subset_name,
        "method": method,
        "graph_count": metrics.get("graph_count"),
        "loss": metrics.get("loss"),
        "top1_accuracy": metrics.get("top1_accuracy"),
        "top3_accuracy": metrics.get("top3_accuracy"),
        "mrr": metrics.get("mrr"),
        "spearman_proposal_only_mean": metrics.get("spearman_proposal_only_mean"),
        "spearman_valid_graph_count": metrics.get("spearman_valid_graph_count"),
        "null_precision": metrics.get("null_precision"),
        "null_recall": metrics.get("null_recall"),
        "null_reference_rate": metrics.get("null_reference_rate"),
        "null_prediction_rate": metrics.get("null_prediction_rate"),
        "proposal_selection_accuracy": metrics.get("proposal_selection_accuracy"),
        "proposal_reference_graph_count": metrics.get("proposal_reference_graph_count"),
        "maximum_fixed_proposal_rank_prediction_rate": metrics.get(
            "maximum_fixed_proposal_rank_prediction_rate"
        ),
        "empirical_random_top1_accuracy": metrics.get(
            "empirical_random_top1_accuracy"
        ),
        "empirical_random_top3_accuracy": metrics.get(
            "empirical_random_top3_accuracy"
        ),
        "empirical_random_mrr": metrics.get("empirical_random_mrr"),
    }


METRIC_FIELDS = [
    "schema_version",
    "optimization_seed",
    "split",
    "subset_type",
    "subset_name",
    "method",
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


def _random_row(examples: Sequence[Stage1Example], split: str) -> dict[str, Any]:
    values = empirical_random_metrics([item.class_count for item in examples])
    metrics = {
        "graph_count": len(examples),
        "top1_accuracy": values["top1_accuracy"],
        "top3_accuracy": values["top3_accuracy"],
        "mrr": values["mrr"],
        "empirical_random_top1_accuracy": values["top1_accuracy"],
        "empirical_random_top3_accuracy": values["top3_accuracy"],
        "empirical_random_mrr": values["mrr"],
        "spearman_valid_graph_count": 0,
    }
    return _metric_row(
        method="random_class_count_conditioned_expectation",
        split=split,
        metrics=metrics,
        optimization_seed="analytic",
    )


def _selector_metrics(
    examples: Sequence[Stage1Example],
    gat_scores: Sequence[np.ndarray],
) -> dict[str, dict[str, Any]]:
    proposal_scores, proposal_masks = proposal_score_vectors(examples)
    fp_scores = fp_shep_score_vectors(examples)
    return {
        "proposal_coarse_top1": compute_offline_metrics(
            examples, proposal_scores, selectable_masks=proposal_masks
        ),
        "fp_shep_h4_three_feature": compute_offline_metrics(examples, fp_scores),
        "gat": compute_offline_metrics(examples, gat_scores),
    }


def _paired_rows(
    examples: Sequence[Stage1Example],
    gat_scores: Sequence[np.ndarray],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    fp_scores = fp_shep_score_vectors(examples)
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for example, fp, gat in zip(examples, fp_scores, gat_scores):
        reference = example.reference_class
        fp_prediction = int(np.argmax(fp))
        gat_prediction = int(np.argmax(gat))
        fp_correct = fp_prediction == reference
        gat_correct = gat_prediction == reference
        category = (
            "BOTH_CORRECT"
            if fp_correct and gat_correct
            else "FP_SHEP_ONLY_CORRECT"
            if fp_correct
            else "GAT_ONLY_CORRECT"
            if gat_correct
            else "BOTH_WRONG"
        )
        counts[category] += 1
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "sample_id": example.sample_id,
                "state_group_id": example.state_group_id,
                "scenario": example.scenario,
                "seed": example.seed,
                "interaction_group": example.interaction_group,
                "descriptor_risk_positive": example.descriptor_risk_positive,
                "class_count": example.class_count,
                "reference_class": reference,
                "fp_shep_prediction": fp_prediction,
                "gat_prediction": gat_prediction,
                "fp_shep_correct": fp_correct,
                "gat_correct": gat_correct,
                "paired_category": category,
            }
        )
    return rows, counts


def _subset_rows(
    examples: Sequence[Stage1Example],
    gat_scores: Sequence[np.ndarray],
    *,
    split: str,
    best_seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    subsets: list[tuple[str, str, list[int]]] = []
    for scenario in sorted({item.scenario for item in examples}):
        subsets.append(
            (
                "scenario",
                scenario,
                [index for index, item in enumerate(examples) if item.scenario == scenario],
            )
        )
    for group in ["low-interaction", "obstacle-pressure", "interaction-rich"]:
        subsets.append(
            (
                "interaction_group",
                group,
                [
                    index
                    for index, item in enumerate(examples)
                    if item.interaction_group == group
                ],
            )
        )
    for flag, name in [(False, "risk_absent"), (True, "risk_positive")]:
        subsets.append(
            (
                "existing_descriptor",
                name,
                [
                    index
                    for index, item in enumerate(examples)
                    if item.descriptor_risk_positive is flag
                ],
            )
        )
    for subset_type, subset_name, indices in subsets:
        if not indices:
            continue
        local_examples = [examples[index] for index in indices]
        local_gat = [gat_scores[index] for index in indices]
        metrics_by_method = _selector_metrics(local_examples, local_gat)
        random_values = empirical_random_metrics(
            [item.class_count for item in local_examples]
        )
        random_metrics = {
            "graph_count": len(local_examples),
            "top1_accuracy": random_values["top1_accuracy"],
            "top3_accuracy": random_values["top3_accuracy"],
            "mrr": random_values["mrr"],
            "empirical_random_top1_accuracy": random_values["top1_accuracy"],
            "empirical_random_top3_accuracy": random_values["top3_accuracy"],
            "empirical_random_mrr": random_values["mrr"],
            "spearman_valid_graph_count": 0,
        }
        rows.append(
            _metric_row(
                method="random_class_count_conditioned_expectation",
                split=split,
                metrics=random_metrics,
                optimization_seed="analytic",
                subset_type=subset_type,
                subset_name=subset_name,
            )
        )
        for method, metrics in metrics_by_method.items():
            rows.append(
                _metric_row(
                    method=method,
                    split=split,
                    metrics=metrics,
                    optimization_seed=best_seed,
                    subset_type=subset_type,
                    subset_name=subset_name,
                )
            )
    return rows


def _quality_label_interaction_audit(
    examples: Sequence[Stage1Example], dataset_dir: Path,
) -> dict[str, Any]:
    collision_rows = [
        row
        for row in read_csv(Path(dataset_dir) / "candidate_class_records.csv")
        if int(row["H_label"]) == 6
    ]
    inter_agent_collision_count = sum(
        row["inter_agent_collision"] == "True" for row in collision_rows
    )
    return {
        "SELECT_LABEL_INTERACTION_AWARE": "PARTIAL",
        "H_label": 6,
        "target": "provisional_primary_target_three_feature",
        "real_rollout_inter_agent_collision_branch_count": inter_agent_collision_count,
        "spatiotemporal_edges_present_in_graph": any(
            int(item.graph["align", "spatiotemporal", "proposal"].edge_index.shape[1])
            > 0
            for item in examples
        ),
        "joint_candidate_combination_supervision_present": False,
        "rationale": (
            "labels can reflect observed peer clearance/collision during an ego branch, "
            "but they do not supervise joint candidate combinations or isolate peer intent"
        ),
        "allowed_stage1_claim": "GAT learns candidate ranking",
        "spatiotemporal_coordination_claim": "NOT_ESTABLISHED_FROM_LABEL_ALONE",
    }


def _decision_labels(
    config: Mapping[str, Any],
    run_results: Sequence[Any],
    best_test: Mapping[str, Any],
    selectors: Mapping[str, Mapping[str, Any]],
    interaction_rows: Sequence[Mapping[str, Any]],
    completion: Mapping[str, Any],
) -> dict[str, Any]:
    thresholds = config["decision_thresholds"]
    all_converged = all(
        result.best_validation_loss < result.first_validation_loss
        and math.isfinite(result.best_validation_loss)
        for result in run_results
    )
    training_signal = "YES" if all_converged and len(run_results) == 3 else "WEAK"
    random_top1 = float(best_test["empirical_random_top1_accuracy"])
    gat_top1 = float(best_test["top1_accuracy"])
    random_gain = gat_top1 - random_top1
    if random_gain >= float(thresholds["generalization_yes_top1_gain_over_random"]):
        generalization = "YES"
    elif random_gain > float(thresholds["generalization_weak_top1_gain_over_random"]):
        generalization = "WEAK"
    else:
        generalization = "NO"
    proposal_top1 = float(selectors["proposal_coarse_top1"]["top1_accuracy"])
    fp_top1 = float(selectors["fp_shep_h4_three_feature"]["top1_accuracy"])
    beats_proposal = "YES" if gat_top1 > proposal_top1 else "NO"
    fp_gain = gat_top1 - fp_top1
    if fp_gain >= float(thresholds["adds_over_fp_shep_yes_top1_gain"]):
        adds_fp = "YES"
    elif fp_gain > 0.0:
        adds_fp = "WEAK"
    else:
        adds_fp = "NO"
    lookup = {
        (row["subset_name"], row["method"]): row
        for row in interaction_rows
        if row["subset_type"] == "interaction_group"
    }
    interaction_gat = lookup.get(("interaction-rich", "gat"))
    interaction_proposal = lookup.get(("interaction-rich", "proposal_coarse_top1"))
    interaction_fp = lookup.get(("interaction-rich", "fp_shep_h4_three_feature"))
    if not interaction_gat or int(interaction_gat["graph_count"]) < int(
        thresholds["interaction_minimum_graph_count"]
    ):
        interaction_gain = "NOT_ESTABLISHED"
        interaction_delta = None
    else:
        interaction_delta = float(interaction_gat["top1_accuracy"]) - max(
            float(interaction_proposal["top1_accuracy"]),
            float(interaction_fp["top1_accuracy"]),
        )
        if interaction_delta >= float(thresholds["interaction_gain_yes_top1_gain"]):
            interaction_gain = "YES"
        elif interaction_delta > 0.0:
            interaction_gain = "WEAK"
        else:
            interaction_gain = "NO"
    proceed = (
        training_signal == "YES"
        and generalization == "YES"
        and beats_proposal == "YES"
        and adds_fp in {"YES", "WEAK"}
    )
    return {
        "GAT_TRAINING_SIGNAL": training_signal,
        "GAT_GENERALIZATION_SIGNAL": generalization,
        "GAT_BEATS_COARSE_PROPOSAL": beats_proposal,
        "GAT_ADDS_OVER_FP_SHEP": adds_fp,
        "INTERACTION_AWARE_GAIN": interaction_gain,
        "SELECT_LABEL_INTERACTION_AWARE": "PARTIAL",
        "GAT_SPATIOTEMPORAL_COORDINATION_LEARNED": "NOT_ESTABLISHED",
        "COMPLETION_LABEL_AVAILABLE": completion["COMPLETION_LABEL_AVAILABLE"],
        "CANDIDATE_LEVEL_COMPLETION_ATTRIBUTION": completion[
            "CANDIDATE_LEVEL_COMPLETION_ATTRIBUTION"
        ],
        "COMPLETION_AUXILIARY_WORTH_TESTING": "NO",
        "PROCEED_TO_GAT_CLOSED_LOOP": "YES" if proceed else "NO",
        "test_top1_gain_over_empirical_random": random_gain,
        "test_top1_gain_over_proposal": gat_top1 - proposal_top1,
        "test_top1_gain_over_fp_shep": fp_gain,
        "interaction_rich_top1_gain_over_best_baseline": interaction_delta,
    }


def _report(
    *,
    run_dir: Path,
    config: Mapping[str, Any],
    dataset_audit: Mapping[str, Any],
    completion: Mapping[str, Any],
    smoke: Mapping[str, Any],
    run_results: Sequence[Any],
    selector_rows: Sequence[Mapping[str, Any]],
    paired_counts: Mapping[str, int],
    conclusion: Mapping[str, Any],
    integrity: Mapping[str, Any],
) -> None:
    selector_map = {row["method"]: row for row in selector_rows}
    lines = [
        "# Spatiotemporal Edge-Enhanced GAT Stage-I Training Report",
        "",
        "## Protocol",
        "",
        "- Frozen architecture: hidden 64, edge 32, 4 heads, 2 layers.",
        "- Supervision: H_label=6, three-feature provisional primary target, tau=0.25.",
        "- Loss: per-graph soft-target cross entropy over real null + K classes only.",
        "- Split: train seeds 0-6, validation seed 7, test seeds 8-9 by state_group_id.",
        "- Interaction subsets are diagnostic only and never select checkpoints.",
        "- Completion audit is read-only; no completion/overlap auxiliary loss is used.",
        "",
        "## Dataset",
        "",
        f"- Graphs: {dataset_audit['graph_count']}; state groups: {dataset_audit['state_group_count']}.",
        f"- Split graph counts: `{json.dumps(dataset_audit['split_graph_counts'], ensure_ascii=False)}`.",
        f"- Group leakage: {dataset_audit['group_leakage_count']}.",
        f"- SELECT_LABEL_INTERACTION_AWARE: {dataset_audit['SELECT_LABEL_INTERACTION_AWARE']}.",
        "",
        "## Smoke gate",
        "",
        f"- Status: **{smoke['status']}**.",
        f"- Checks: `{json.dumps(smoke['checks'], ensure_ascii=False)}`.",
        "",
        "## Training runs",
        "",
        "| seed | best epoch | epochs | best validation loss | runtime s |",
        "|---:|---:|---:|---:|---:|",
    ]
    for result in run_results:
        lines.append(
            f"| {result.optimization_seed} | {result.best_epoch} | "
            f"{result.epochs_completed} | {result.best_validation_loss:.6f} | "
            f"{result.runtime_s:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Test selector comparison",
            "",
            "| method | Top-1 | Top-3 | MRR | proposal-only Spearman | n |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name in [
        "random_class_count_conditioned_expectation",
        "proposal_coarse_top1",
        "fp_shep_h4_three_feature",
        "gat",
    ]:
        row = selector_map[name]
        spear = row.get("spearman_proposal_only_mean")
        lines.append(
            f"| {name} | {float(row['top1_accuracy']):.4f} | "
            f"{float(row['top3_accuracy']):.4f} | {float(row['mrr']):.4f} | "
            f"{'' if spear in (None, '') else f'{float(spear):.4f}'} | "
            f"{row.get('spearman_valid_graph_count', 0)} |"
        )
    lines.extend(
        [
            "",
            "## FP-SHEP vs GAT paired outcome",
            "",
            f"- `{json.dumps(dict(paired_counts), ensure_ascii=False)}`",
            "",
            "## Completion signal audit",
            "",
            f"- Availability: {completion['COMPLETION_LABEL_AVAILABLE']}.",
            f"- Candidate attribution: {completion['CANDIDATE_LEVEL_COMPLETION_ATTRIBUTION']}.",
            f"- Reliable branches: {completion['reliable_branch_count']}; conflicts: {completion['conflicting_branch_count']}.",
            f"- Class coverage: {float(completion['class_coverage_rate']):.4%}.",
            "- Completion labels were not used in training.",
            "",
            "## Conclusions",
            "",
        ]
    )
    for key in [
        "GAT_TRAINING_SIGNAL",
        "GAT_GENERALIZATION_SIGNAL",
        "GAT_BEATS_COARSE_PROPOSAL",
        "GAT_ADDS_OVER_FP_SHEP",
        "INTERACTION_AWARE_GAIN",
        "COMPLETION_LABEL_AVAILABLE",
        "CANDIDATE_LEVEL_COMPLETION_ATTRIBUTION",
        "COMPLETION_AUXILIARY_WORTH_TESTING",
        "PROCEED_TO_GAT_CLOSED_LOOP",
        "BEST_CHECKPOINT",
        "NEXT_STEP",
    ]:
        lines.append(f"- `{key} = {conclusion[key]}`")
    lines.extend(
        [
            "",
            "## Integrity",
            "",
            f"- Critical files unchanged: {integrity['critical_files_unchanged']}.",
            "- GAT architecture/schema, Proposal, FP-SHEP, SAC-DMP and environment were not modified.",
            "- No closed-loop evaluation or auxiliary training was executed.",
        ]
    )
    (run_dir / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_smoke_failure_outputs(
    *,
    run_dir: Path,
    checkpoint_dir: Path,
    result: Any,
    smoke: Mapping[str, Any],
    completion: Mapping[str, Any],
    critical_before: Mapping[str, str],
) -> None:
    """Finalize a failed smoke run without reading or evaluating the test split."""

    shutil.copy2(result.best_checkpoint, checkpoint_dir / "best_validation.pt")
    shutil.copy2(result.last_checkpoint, checkpoint_dir / "last.pt")
    write_csv(run_dir / "test_metrics.csv", [], METRIC_FIELDS)
    write_csv(run_dir / "selector_comparison.csv", [], METRIC_FIELDS)
    write_csv(run_dir / "scenario_metrics.csv", [], METRIC_FIELDS)
    write_csv(
        run_dir / "paired_selector_analysis.csv",
        [],
        [
            "schema_version",
            "sample_id",
            "state_group_id",
            "scenario",
            "seed",
            "interaction_group",
            "descriptor_risk_positive",
            "class_count",
            "reference_class",
            "fp_shep_prediction",
            "gat_prediction",
            "fp_shep_correct",
            "gat_correct",
            "paired_category",
        ],
    )
    critical_after = {name: sha256_file(path) for name, path in CRITICAL_FILES.items()}
    integrity = {
        "critical_file_hashes_before": dict(critical_before),
        "critical_file_hashes_after": critical_after,
        "critical_files_unchanged": dict(critical_before) == critical_after,
        "test_split_evaluated": False,
        "optimization_seed_count": 1,
        "completion_loss_enabled": False,
        "overlap_loss_enabled": False,
        "closed_loop_evaluation_run": False,
    }
    conclusion = {
        "schema_version": SCHEMA_VERSION,
        "GAT_TRAINING_SIGNAL": "NO",
        "GAT_GENERALIZATION_SIGNAL": "NO",
        "GAT_BEATS_COARSE_PROPOSAL": "NO",
        "GAT_ADDS_OVER_FP_SHEP": "NOT_ESTABLISHED",
        "INTERACTION_AWARE_GAIN": "NOT_ESTABLISHED",
        "SELECT_LABEL_INTERACTION_AWARE": "PARTIAL",
        "GAT_SPATIOTEMPORAL_COORDINATION_LEARNED": "NOT_ESTABLISHED",
        "COMPLETION_LABEL_AVAILABLE": completion["COMPLETION_LABEL_AVAILABLE"],
        "CANDIDATE_LEVEL_COMPLETION_ATTRIBUTION": completion[
            "CANDIDATE_LEVEL_COMPLETION_ATTRIBUTION"
        ],
        "COMPLETION_AUXILIARY_WORTH_TESTING": "NO",
        "PROCEED_TO_GAT_CLOSED_LOOP": "NO",
        "SMOKE_GATE": "FAIL",
        "smoke_gate_details": dict(smoke),
        "optimization_seed_results": [training_result_record(result)],
        "BEST_CHECKPOINT": str(checkpoint_dir / "best_validation.pt"),
        "best_checkpoint_source_seed": result.optimization_seed,
        "NEXT_STEP": "Stop after failed smoke gate; do not tune automatically",
        "automatic_next_stage_started": False,
        "integrity": integrity,
    }
    write_json(run_dir / "conclusion.json", conclusion)
    lines = [
        "# Spatiotemporal Edge-Enhanced GAT Stage-I Smoke Report",
        "",
        "## Outcome",
        "",
        "- `SMOKE_GATE = FAIL`",
        f"- Checks: `{json.dumps(smoke['checks'], ensure_ascii=False)}`",
        "- Optimization seeds 1 and 2 were not started.",
        "- The test split was not evaluated.",
        "- No automatic hyperparameter adjustment was performed.",
        "- No completion/overlap loss or closed-loop evaluation was started.",
        "",
        "## Required conclusions",
        "",
    ]
    for key in [
        "GAT_TRAINING_SIGNAL",
        "GAT_GENERALIZATION_SIGNAL",
        "GAT_BEATS_COARSE_PROPOSAL",
        "GAT_ADDS_OVER_FP_SHEP",
        "INTERACTION_AWARE_GAIN",
        "COMPLETION_LABEL_AVAILABLE",
        "CANDIDATE_LEVEL_COMPLETION_ATTRIBUTION",
        "COMPLETION_AUXILIARY_WORTH_TESTING",
        "PROCEED_TO_GAT_CLOSED_LOOP",
        "BEST_CHECKPOINT",
        "NEXT_STEP",
    ]:
        lines.append(f"- `{key} = {conclusion[key]}`")
    (run_dir / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config_path: Path) -> Path:
    raw_config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    dataset_dir = _resolve(raw_config["dataset_dir"])
    completion_dir = _resolve(raw_config["completion_artifact_dir"])
    output_root = _resolve(raw_config["output_root"])
    run_dir = output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    critical_before = {name: sha256_file(path) for name, path in CRITICAL_FILES.items()}
    config = dict(raw_config)
    config.update(
        {
            "resolved_dataset_dir": str(dataset_dir),
            "resolved_completion_artifact_dir": str(completion_dir),
            "resolved_output_dir": str(run_dir),
            "created_at": datetime.now().astimezone().isoformat(),
            "python_executable": sys.executable,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "critical_file_hashes_before": critical_before,
        }
    )
    write_json(run_dir / "config.json", config)
    examples, dataset_audit, split_rows = load_stage1_examples(
        dataset_dir,
        split_config=config["dataset_split"],
        supervision_config=config["supervision"],
        interaction_config=config["interaction_diagnostic"],
    )
    label_interaction = _quality_label_interaction_audit(examples, dataset_dir)
    dataset_audit["label_interaction_audit"] = label_interaction
    dataset_audit["dataset_manifest_hash"] = stable_hash(split_rows)
    write_json(run_dir / "dataset_audit.json", dataset_audit)
    write_json(
        run_dir / "split_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "group_key": config["dataset_split"]["group_key"],
            "group_leakage_count": 0,
            "split_seeds": {
                "train": config["dataset_split"]["train_seeds"],
                "validation": config["dataset_split"]["validation_seeds"],
                "test": config["dataset_split"]["test_seeds"],
            },
            "records": split_rows,
        },
    )
    completion = completion_signal_audit(
        dataset_dir,
        completion_dir,
        h_label=int(config["supervision"]["H_label"]),
    )
    write_json(run_dir / "completion_signal_audit.json", completion)
    by_split = {
        split: [item for item in examples if item.split == split]
        for split in ["train", "validation", "test"]
    }
    smoke_seed = int(config["training"]["smoke_seed"])
    run_results = [
        train_one_seed(
            config,
            by_split["train"],
            by_split["validation"],
            optimization_seed=smoke_seed,
            checkpoint_dir=checkpoint_dir,
        )
    ]
    smoke = smoke_gate_result(run_results[0], config["smoke_gate"])
    if smoke["status"] == "PASS":
        for seed in config["training"]["optimization_seeds"]:
            seed = int(seed)
            if seed == smoke_seed:
                continue
            run_results.append(
                train_one_seed(
                    config,
                    by_split["train"],
                    by_split["validation"],
                    optimization_seed=seed,
                    checkpoint_dir=checkpoint_dir,
                )
            )
    best_result = min(run_results, key=lambda item: item.best_validation_loss)
    shutil.copy2(best_result.best_checkpoint, checkpoint_dir / "best_validation.pt")
    shutil.copy2(best_result.last_checkpoint, checkpoint_dir / "last.pt")
    history_rows = [
        {**record.__dict__}
        for result in run_results
        for record in result.history
    ]
    write_csv(
        run_dir / "training_history.csv",
        history_rows,
        [
            "optimization_seed",
            "epoch",
            "train_loss",
            "validation_loss",
            "validation_top1",
            "validation_top3",
            "validation_mrr",
            "validation_random_top1",
            "validation_null_prediction_rate",
            "validation_fixed_rank_max_rate",
            "learning_rate",
            "epoch_runtime_s",
            "improved",
        ],
    )
    if smoke["status"] != "PASS":
        _write_smoke_failure_outputs(
            run_dir=run_dir,
            checkpoint_dir=checkpoint_dir,
            result=run_results[0],
            smoke=smoke,
            completion=completion,
            critical_before=critical_before,
        )
        return run_dir
    device = resolve_device(config["training"]["device"])
    test_rows: list[dict[str, Any]] = []
    per_seed_test: dict[int, dict[str, Any]] = {}
    for result in run_results:
        model = load_model_checkpoint(result.best_checkpoint, config, device)
        for split in ["train", "validation", "test"]:
            metrics, _ = evaluate_model(
                model,
                by_split[split],
                batch_size=int(config["training"]["batch_size"]),
                device=device,
            )
            test_rows.append(
                _metric_row(
                    method="gat",
                    split=split,
                    metrics=metrics,
                    optimization_seed=result.optimization_seed,
                )
            )
            if split == "test":
                per_seed_test[result.optimization_seed] = metrics
    write_csv(run_dir / "test_metrics.csv", test_rows, METRIC_FIELDS)
    best_model = load_model_checkpoint(best_result.best_checkpoint, config, device)
    best_test_metrics, best_gat_scores = evaluate_model(
        best_model,
        by_split["test"],
        batch_size=int(config["training"]["batch_size"]),
        device=device,
    )
    selector_metrics = _selector_metrics(by_split["test"], best_gat_scores)
    selector_rows = [_random_row(by_split["test"], "test")]
    selector_rows.extend(
        _metric_row(
            method=method,
            split="test",
            metrics=metrics,
            optimization_seed=best_result.optimization_seed,
        )
        for method, metrics in selector_metrics.items()
    )
    write_csv(run_dir / "selector_comparison.csv", selector_rows, METRIC_FIELDS)
    paired_rows, paired_counts = _paired_rows(by_split["test"], best_gat_scores)
    write_csv(
        run_dir / "paired_selector_analysis.csv",
        paired_rows,
        [
            "schema_version",
            "sample_id",
            "state_group_id",
            "scenario",
            "seed",
            "interaction_group",
            "descriptor_risk_positive",
            "class_count",
            "reference_class",
            "fp_shep_prediction",
            "gat_prediction",
            "fp_shep_correct",
            "gat_correct",
            "paired_category",
        ],
    )
    scenario_rows = _subset_rows(
        by_split["test"],
        best_gat_scores,
        split="test",
        best_seed=best_result.optimization_seed,
    )
    write_csv(run_dir / "scenario_metrics.csv", scenario_rows, METRIC_FIELDS)
    if smoke["status"] == "PASS" and len(run_results) == 3:
        decisions = _decision_labels(
            config,
            run_results,
            best_test_metrics,
            selector_metrics,
            scenario_rows,
            completion,
        )
    else:
        decisions = {
            "GAT_TRAINING_SIGNAL": "NO",
            "GAT_GENERALIZATION_SIGNAL": "NO",
            "GAT_BEATS_COARSE_PROPOSAL": "NO",
            "GAT_ADDS_OVER_FP_SHEP": "NOT_ESTABLISHED",
            "INTERACTION_AWARE_GAIN": "NOT_ESTABLISHED",
            "SELECT_LABEL_INTERACTION_AWARE": "PARTIAL",
            "GAT_SPATIOTEMPORAL_COORDINATION_LEARNED": "NOT_ESTABLISHED",
            "COMPLETION_LABEL_AVAILABLE": completion["COMPLETION_LABEL_AVAILABLE"],
            "CANDIDATE_LEVEL_COMPLETION_ATTRIBUTION": completion[
                "CANDIDATE_LEVEL_COMPLETION_ATTRIBUTION"
            ],
            "COMPLETION_AUXILIARY_WORTH_TESTING": "NO",
            "PROCEED_TO_GAT_CLOSED_LOOP": "NO",
        }
    critical_after = {name: sha256_file(path) for name, path in CRITICAL_FILES.items()}
    integrity = {
        "critical_file_hashes_before": critical_before,
        "critical_file_hashes_after": critical_after,
        "critical_files_unchanged": critical_before == critical_after,
        "group_leakage_count": dataset_audit["group_leakage_count"],
        "ragged_batching_without_padding": True,
        "metric_definitions_frozen_before_training": True,
        "interaction_used_for_checkpoint_selection": False,
        "interaction_used_for_early_stopping": False,
        "completion_loss_enabled": False,
        "overlap_loss_enabled": False,
        "closed_loop_evaluation_run": False,
        "optimization_seed_count": len(run_results),
    }
    conclusion = {
        "schema_version": SCHEMA_VERSION,
        **decisions,
        "SMOKE_GATE": smoke["status"],
        "smoke_gate_details": smoke,
        "optimization_seed_results": [
            training_result_record(result) for result in run_results
        ],
        "paired_fp_shep_vs_gat": dict(sorted(paired_counts.items())),
        "BEST_CHECKPOINT": str(checkpoint_dir / "best_validation.pt"),
        "best_checkpoint_source_seed": best_result.optimization_seed,
        "best_checkpoint_validation_loss": best_result.best_validation_loss,
        "NEXT_STEP": (
            "Review Stage-I offline evidence before any separately authorized closed-loop test"
            if decisions["PROCEED_TO_GAT_CLOSED_LOOP"] == "YES"
            else "Do not enter closed-loop; inspect Stage-I failure/generalization evidence"
        ),
        "automatic_next_stage_started": False,
        "integrity": integrity,
    }
    write_json(run_dir / "conclusion.json", conclusion)
    _report(
        run_dir=run_dir,
        config=config,
        dataset_audit=dataset_audit,
        completion=completion,
        smoke=smoke,
        run_results=run_results,
        selector_rows=selector_rows,
        paired_counts=paired_counts,
        conclusion=conclusion,
        integrity=integrity,
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "training" / "gat_stage1.json",
    )
    args = parser.parse_args()
    run_dir = run(args.config)
    print(run_dir)


if __name__ == "__main__":
    main()
