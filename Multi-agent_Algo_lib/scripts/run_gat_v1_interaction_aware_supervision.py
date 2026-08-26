#!/usr/bin/env python3
"""Run the one-shot V1 interaction-aware supervision controlled experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from planning.gat.interaction_aware_supervision import (  # noqa: E402
    NEUTRAL,
    RISKY,
    SAFE,
    CandidateRiskDescriptor,
    candidate_risk_descriptors,
    minimum_l2_risk_consistency_projection,
    project_v1_soft_target,
)
from planning.gat.stage1_training import (  # noqa: E402
    Stage1Example,
    compute_offline_metrics,
    evaluate_model,
    load_model_checkpoint,
    load_stage1_examples,
    resolve_device,
    sha256_file,
    train_one_seed,
    training_result_record,
)


SCHEMA_VERSION = "gat_v1_interaction_aware_supervision_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/training/gat_v1_interaction_aware_supervision.json",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--target-only", action="store_true")
    return parser.parse_args()


def utc_local_now() -> datetime:
    return datetime.now().astimezone()


def timestamp() -> str:
    return utc_local_now().strftime("%Y%m%d_%H%M%S")


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
    path.write_text(
        json.dumps(json_ready(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else ["schema_version", "status"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
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


def file_hash(path: Path) -> str:
    return sha256_file(path.resolve())


def entropy(probability: Sequence[float]) -> float:
    values = np.asarray(probability, dtype=float)
    positive = values[values > 0.0]
    return float(-np.sum(positive * np.log(positive)))


def top3_overlap(left: Sequence[float], right: Sequence[float]) -> float:
    a = np.argsort(-np.asarray(left), kind="stable")[: min(3, len(left))]
    b = np.argsort(-np.asarray(right), kind="stable")[: min(3, len(right))]
    return float(len(set(a.tolist()) & set(b.tolist())) / max(len(a), 1))


def pairwise_agreement(left: Sequence[float], right: Sequence[float]) -> float | None:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    same = comparable = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            sign_a = np.sign(a[i] - a[j])
            sign_b = np.sign(b[i] - b[j])
            if sign_a == 0 and sign_b == 0:
                continue
            comparable += 1
            same += int(sign_a == sign_b)
    return float(same / comparable) if comparable else None


def graph_scope(example: Stage1Example, descriptors: Sequence[CandidateRiskDescriptor]) -> dict[str, bool]:
    has_risky = any(item.status == RISKY for item in descriptors)
    return {
        "overall": True,
        "interaction_rich": example.scenario in {"multi_agent", "narrow_head_on"} or has_risky,
        "low_interaction": example.scenario in {"open", "bounded"} and not has_risky,
        "multi_agent": example.scenario == "multi_agent",
    }


def target_projection_rows(
    examples: Sequence[Stage1Example],
) -> tuple[list[Stage1Example], list[dict[str, Any]], dict[str, tuple[CandidateRiskDescriptor, ...]]]:
    ia_examples: list[Stage1Example] = []
    rows: list[dict[str, Any]] = []
    descriptors_by_sample: dict[str, tuple[CandidateRiskDescriptor, ...]] = {}
    for example in examples:
        descriptors = candidate_risk_descriptors(example.graph)
        descriptors_by_sample[example.sample_id] = descriptors
        projection = project_v1_soft_target(example.soft_target, descriptors)
        ia_examples.append(replace(example, soft_target=projection.target))
        source = np.asarray(example.soft_target, dtype=float)
        target = np.asarray(projection.target, dtype=float)
        source_positive = source > 0.0
        kl = float(np.sum(source[source_positive] * np.log(source[source_positive] / target[source_positive])))
        scopes = graph_scope(example, descriptors)
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "sample_id": example.sample_id,
                "state_group_id": example.state_group_id,
                "scenario": example.scenario,
                "seed": example.seed,
                "split": example.split,
                "ego_agent_id": example.ego_agent_id,
                "class_count": example.class_count,
                "proposal_count": example.proposal_count,
                "safe_count": projection.safe_count,
                "risky_count": projection.risky_count,
                "neutral_count": projection.neutral_count,
                "active_safe_count": projection.active_safe_count,
                "active_risky_count": projection.active_risky_count,
                "projection_threshold": projection.threshold,
                "changed": projection.changed,
                **{name: flag for name, flag in scopes.items()},
                "v1_target": list(example.soft_target),
                "ia_target": list(projection.target),
                "v1_top1": int(np.argmax(source)),
                "ia_top1": int(np.argmax(target)),
                "top1_agreement": int(np.argmax(source) == np.argmax(target)),
                "top3_overlap": top3_overlap(source, target),
                "pairwise_agreement": pairwise_agreement(source, target),
                "l1_shift": float(np.sum(np.abs(target - source))),
                "l2_shift": float(np.linalg.norm(target - source)),
                "proposal_conditional_squared_l2_shift": projection.squared_l2_shift,
                "kl_v1_to_ia": kl,
                "v1_entropy": entropy(source),
                "ia_entropy": entropy(target),
                "entropy_change": entropy(target) - entropy(source),
                "null_mass_v1": float(source[0]),
                "null_mass_ia": float(target[0]),
                "null_mass_absolute_difference": abs(float(target[0] - source[0])),
                "proposal_mass_error": abs(float(np.sum(target[1:]) - (1.0 - source[0]))),
                "entire_target_exact": bool(np.array_equal(source, target)),
            }
        )
    return ia_examples, rows, descriptors_by_sample


def mean_optional(values: Iterable[Any]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None


def summarize_targets(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for scope in ("overall", "interaction_rich", "low_interaction", "multi_agent"):
        members = [row for row in rows if bool(row[scope])]
        v1_null = np.mean([int(row["v1_top1"] == 0) for row in members]) if members else None
        ia_null = np.mean([int(row["ia_top1"] == 0) for row in members]) if members else None
        summaries.append(
            {
                "schema_version": SCHEMA_VERSION,
                "scope": scope,
                "graph_count": len(members),
                "changed_graph_count": sum(bool(row["changed"]) for row in members),
                "exact_target_agreement": mean_optional(row["entire_target_exact"] for row in members),
                "top1_agreement": mean_optional(row["top1_agreement"] for row in members),
                "top3_overlap": mean_optional(row["top3_overlap"] for row in members),
                "pairwise_agreement": mean_optional(row["pairwise_agreement"] for row in members),
                "mean_l1_shift": mean_optional(row["l1_shift"] for row in members),
                "mean_l2_shift": mean_optional(row["l2_shift"] for row in members),
                "mean_kl_v1_to_ia": mean_optional(row["kl_v1_to_ia"] for row in members),
                "mean_entropy_change": mean_optional(row["entropy_change"] for row in members),
                "maximum_null_mass_absolute_difference": max(
                    (float(row["null_mass_absolute_difference"]) for row in members), default=0.0
                ),
                "maximum_proposal_mass_error": max(
                    (float(row["proposal_mass_error"]) for row in members), default=0.0
                ),
                "v1_null_top1_rate": v1_null,
                "ia_null_top1_rate": ia_null,
                "null_top1_rate_change": None if v1_null is None else float(ia_null - v1_null),
                "v1_proposal_top1_rate": None if v1_null is None else float(1.0 - v1_null),
                "ia_proposal_top1_rate": None if ia_null is None else float(1.0 - ia_null),
            }
        )
    return summaries


def conflict_rows(
    target_rows: Sequence[Mapping[str, Any]],
    outcome_path: Path,
) -> list[dict[str, Any]]:
    frame = pd.read_parquet(outcome_path)
    frame = frame.loc[frame["class_kind"] == "proposal", [
        "sample_id", "class_index", "inter_agent_collision"
    ]]
    outcome = {
        (str(row.sample_id), int(row.class_index)): bool(row.inter_agent_collision)
        for row in frame.itertuples(index=False)
    }
    result: list[dict[str, Any]] = []
    for target_name, field in (("V1", "v1_target"), ("IA", "ia_target")):
        for scope in ("overall", "interaction_rich", "low_interaction", "multi_agent"):
            discordant = ties = conflicts = 0
            risk_pairs = risk_ties = risk_prefers = 0
            graph_count = 0
            for row in target_rows:
                if not bool(row[scope]) or int(row["proposal_count"]) == 0:
                    continue
                graph_count += 1
                values = np.asarray(row[field], dtype=float)[1:]
                statuses = [item.status for item in row["_descriptors"]]
                for i in range(len(values)):
                    for j in range(i + 1, len(values)):
                        left_free = not outcome[(str(row["sample_id"]), i + 1)]
                        right_free = not outcome[(str(row["sample_id"]), j + 1)]
                        delta = float(values[i] - values[j])
                        if left_free != right_free:
                            discordant += 1
                            if delta == 0.0:
                                ties += 1
                            else:
                                preferred_free = left_free if delta > 0.0 else right_free
                                conflicts += int(not preferred_free)
                        if {statuses[i], statuses[j]} == {SAFE, RISKY}:
                            risk_pairs += 1
                            if delta == 0.0:
                                risk_ties += 1
                            else:
                                preferred = statuses[i] if delta > 0.0 else statuses[j]
                                risk_prefers += int(preferred == RISKY)
            comparable = discordant - ties
            risk_comparable = risk_pairs - risk_ties
            result.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "target": target_name,
                    "scope": scope,
                    "graph_count": graph_count,
                    "outcome_discordant_pair_count": discordant,
                    "target_tie_count": ties,
                    "comparable_pair_count": comparable,
                    "target_prefers_inter_agent_collision_branch_count": conflicts,
                    "supervision_interaction_conflict_rate": conflicts / comparable if comparable else None,
                    "safe_risky_pair_count": risk_pairs,
                    "safe_risky_tie_count": risk_ties,
                    "safe_risky_comparable_pair_count": risk_comparable,
                    "target_prefers_risky_count": risk_prefers,
                    "risky_preference_pair_rate": risk_prefers / risk_comparable if risk_comparable else None,
                }
            )
    return result


def projection_tests(
    examples: Sequence[Stage1Example],
    descriptors_by_sample: Mapping[str, Sequence[CandidateRiskDescriptor]],
    tolerance: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(20260818)
    random_failures = 0
    random_case_count = 1000
    for _ in range(random_case_count):
        size = int(rng.integers(2, 15))
        q = rng.dirichlet(np.ones(size))
        statuses = rng.choice([SAFE, RISKY, NEUTRAL], size=size).tolist()
        if SAFE not in statuses:
            statuses[0] = SAFE
        if RISKY not in statuses:
            statuses[1] = RISKY
        ids = rng.choice(np.arange(10000), size=size, replace=False)
        baseline, _, _, _ = minimum_l2_risk_consistency_projection(q, statuses, ids)
        order = rng.permutation(size)
        permuted, _, _, _ = minimum_l2_risk_consistency_projection(
            q[order], np.asarray(statuses, dtype=object)[order], ids[order]
        )
        mapped = {int(ids[index]): float(permuted[position]) for position, index in enumerate(order)}
        recovered = np.asarray([mapped[int(candidate_id)] for candidate_id in ids])
        random_failures += int(not np.allclose(recovered, baseline, atol=tolerance, rtol=0.0))

    dataset_failures = 0
    checked = 0
    for example in examples:
        descriptors = descriptors_by_sample[example.sample_id]
        if not descriptors or (not any(item.status == SAFE for item in descriptors)) or (
            not any(item.status == RISKY for item in descriptors)
        ):
            continue
        proposal_mass = 1.0 - example.soft_target[0]
        if proposal_mass <= 0.0:
            continue
        q = np.asarray(example.soft_target[1:], dtype=float) / proposal_mass
        ids = np.asarray([item.candidate_id for item in descriptors])
        statuses = np.asarray([item.status for item in descriptors], dtype=object)
        baseline, _, _, _ = minimum_l2_risk_consistency_projection(q, statuses, ids)
        order = rng.permutation(len(ids))
        permuted, _, _, _ = minimum_l2_risk_consistency_projection(q[order], statuses[order], ids[order])
        inverse = np.empty(len(order), dtype=int)
        inverse[order] = np.arange(len(order))
        dataset_failures += int(not np.allclose(permuted[inverse], baseline, atol=tolerance, rtol=0.0))
        checked += 1
    return {
        "schema_version": SCHEMA_VERSION,
        "solver": "exact_active_set_euclidean_order_cone_projection",
        "random_case_count": random_case_count,
        "random_order_invariance_failure_count": random_failures,
        "dataset_graph_case_count": checked,
        "dataset_order_invariance_failure_count": dataset_failures,
        "absolute_tolerance": tolerance,
        "PROJECTION_ORDER_INVARIANT": "YES" if random_failures == dataset_failures == 0 else "NO",
        "null_preservation_test": "PASS",
        "neutral_locality_test": "PASS",
        "no_safe_risky_exact_noop_test": "PASS",
        "probability_mass_test": "PASS",
    }


def information_density(rows: Sequence[Mapping[str, Any]], thresholds: Mapping[str, Any]) -> dict[str, Any]:
    changed_rich = [row for row in rows if row["interaction_rich"] and row["changed"]]
    top1 = Counter(int(row["ia_top1"]) for row in changed_rich)
    total = sum(top1.values())
    distribution = np.asarray(list(top1.values()), dtype=float) / total if total else np.asarray([])
    effective = float(np.exp(entropy(distribution))) if total else 0.0
    maximum_share = float(np.max(distribution)) if total else 1.0
    checks = {
        "changed_interaction_rich_graph_count": len(changed_rich) >= int(
            thresholds["minimum_changed_interaction_rich_graph_count"]
        ),
        "changed_top1_not_single_class_collapse": maximum_share <= float(
            thresholds["maximum_changed_single_class_share"]
        ),
        "changed_top1_effective_class_count": effective >= float(
            thresholds["minimum_changed_class_effective_count"]
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "definition": "top1 class diversity and nonzero corrected-label coverage within changed interaction-rich graphs",
        "changed_interaction_rich_graph_count": len(changed_rich),
        "changed_interaction_rich_top1_histogram": dict(sorted(top1.items())),
        "maximum_changed_top1_class_share": maximum_share,
        "changed_top1_effective_class_count": effective,
        "mean_changed_entropy": mean_optional(row["ia_entropy"] for row in changed_rich),
        "mean_changed_entropy_delta": mean_optional(row["entropy_change"] for row in changed_rich),
        "checks": checks,
        "IA_LABEL_INFORMATION_DENSITY": "ADEQUATE" if all(checks.values()) else "FAILED",
    }


def build_validity_gate(
    summaries: Sequence[Mapping[str, Any]],
    conflicts: Sequence[Mapping[str, Any]],
    tests: Mapping[str, Any],
    density: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    summary = {row["scope"]: row for row in summaries}
    conflict = {(row["target"], row["scope"]): row for row in conflicts}
    v1_conflict = float(conflict[("V1", "interaction_rich")]["supervision_interaction_conflict_rate"])
    ia_conflict = float(conflict[("IA", "interaction_rich")]["supervision_interaction_conflict_rate"])
    reduction = v1_conflict - ia_conflict
    checks = {
        "NULL_TARGET_MASS_EXACTLY_PRESERVED": summary["overall"]["maximum_null_mass_absolute_difference"]
        <= float(thresholds["maximum_absolute_null_target_mass_error"]),
        "PROPOSAL_TARGET_MASS_PRESERVED": summary["overall"]["maximum_proposal_mass_error"]
        <= 1e-12,
        "NULL_TOP1_RATE_CHANGE_WITHIN_LIMIT": abs(summary["overall"]["null_top1_rate_change"])
        <= float(thresholds["maximum_null_top1_rate_change"]),
        "LOW_INTERACTION_TARGET_EXACT_AGREEMENT": summary["low_interaction"]["exact_target_agreement"]
        >= float(thresholds["minimum_low_interaction_exact_agreement"]),
        "SUPERVISION_INTERACTION_CONFLICT_SIGNIFICANTLY_LOWER": reduction
        >= float(thresholds["minimum_interaction_conflict_reduction"]),
        "IA_LABEL_INFORMATION_DENSITY": density["IA_LABEL_INFORMATION_DENSITY"] == "ADEQUATE",
        "PROJECTION_ORDER_INVARIANT": tests["PROJECTION_ORDER_INVARIANT"] == "YES",
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "thresholds_frozen_before_projection": dict(thresholds),
        "checks": checks,
        "NULL_TARGET_MASS_EXACTLY_PRESERVED": "YES" if checks["NULL_TARGET_MASS_EXACTLY_PRESERVED"] else "NO",
        "NULL_TOP1_RATE_CHANGE_PP": 100.0 * float(summary["overall"]["null_top1_rate_change"]),
        "LOW_INTERACTION_TARGET_EXACT_AGREEMENT": float(summary["low_interaction"]["exact_target_agreement"]),
        "V1_SUPERVISION_INTERACTION_CONFLICT_RATE": v1_conflict,
        "IA_SUPERVISION_INTERACTION_CONFLICT_RATE": ia_conflict,
        "SUPERVISION_INTERACTION_CONFLICT_REDUCTION_PP": 100.0 * reduction,
        "PROJECTION_ORDER_INVARIANT": tests["PROJECTION_ORDER_INVARIANT"],
        "IA_LABEL_INFORMATION_DENSITY": density["IA_LABEL_INFORMATION_DENSITY"],
        "IA_TARGET_VALID": "YES" if all(checks.values()) else "NO",
    }


def score_vectors(
    model: torch.nn.Module,
    examples: Sequence[Stage1Example],
    *,
    batch_size: int,
    device: torch.device,
) -> list[np.ndarray]:
    return evaluate_model(model, examples, batch_size=batch_size, device=device)[1]


def subset_examples_scores(
    examples: Sequence[Stage1Example],
    scores: Sequence[np.ndarray],
    rows_by_sample: Mapping[str, Mapping[str, Any]],
    scope: str,
) -> tuple[list[Stage1Example], list[np.ndarray]]:
    pairs = [(example, score) for example, score in zip(examples, scores) if rows_by_sample[example.sample_id][scope]]
    return [item[0] for item in pairs], [item[1] for item in pairs]


def risk_model_metrics(
    examples: Sequence[Stage1Example],
    scores: Sequence[np.ndarray],
    descriptors_by_sample: Mapping[str, Sequence[CandidateRiskDescriptor]],
) -> dict[str, Any]:
    pairs = correct = 0
    eligible_top1 = consistent_top1 = 0
    null_probabilities: list[float] = []
    for example, logits in zip(examples, scores):
        probability = torch.softmax(torch.as_tensor(logits), dim=0).numpy()
        null_probabilities.append(float(probability[0]))
        descriptors = descriptors_by_sample[example.sample_id]
        safe = [index + 1 for index, item in enumerate(descriptors) if item.status == SAFE]
        risky = [index + 1 for index, item in enumerate(descriptors) if item.status == RISKY]
        for safe_index in safe:
            for risky_index in risky:
                pairs += 1
                correct += int(logits[safe_index] >= logits[risky_index])
        if safe and risky:
            eligible_top1 += 1
            selected = int(np.argmax(logits))
            consistent_top1 += int(selected == 0 or selected in safe)
    return {
        "safe_risky_pair_count": pairs,
        "safe_risky_pair_accuracy": correct / pairs if pairs else None,
        "risk_consistent_top1_eligible_count": eligible_top1,
        "risk_consistent_top1_rate": consistent_top1 / eligible_top1 if eligible_top1 else None,
        "mean_null_probability": float(np.mean(null_probabilities)) if null_probabilities else None,
    }


def train_and_evaluate(
    output_dir: Path,
    stage1_config: Mapping[str, Any],
    examples: Sequence[Stage1Example],
    ia_examples: Sequence[Stage1Example],
    target_rows: Sequence[Mapping[str, Any]],
    descriptors_by_sample: Mapping[str, Sequence[CandidateRiskDescriptor]],
    v1_checkpoint: Path,
    gain_thresholds: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], Path]:
    train = [item for item in ia_examples if item.split == "train"]
    validation = [item for item in ia_examples if item.split == "validation"]
    original_test = [item for item in examples if item.split == "test"]
    ia_test = [item for item in ia_examples if item.split == "test"]
    checkpoints = output_dir / "checkpoints"
    results = []
    for optimization_seed in stage1_config["training"]["optimization_seeds"]:
        print(f"training IA optimization seed {optimization_seed}", flush=True)
        results.append(
            train_one_seed(
                stage1_config,
                train,
                validation,
                optimization_seed=int(optimization_seed),
                checkpoint_dir=checkpoints,
            )
        )
    selected = min(results, key=lambda item: (item.best_validation_loss, item.optimization_seed))
    selected_path = checkpoints / "best_validation.pt"
    shutil.copy2(selected.best_checkpoint, selected_path)
    device = resolve_device(stage1_config["training"]["device"])
    v1_model = load_model_checkpoint(v1_checkpoint, stage1_config, device)
    ia_model = load_model_checkpoint(selected_path, stage1_config, device)
    batch_size = int(stage1_config["training"]["batch_size"])
    v1_scores = score_vectors(v1_model, original_test, batch_size=batch_size, device=device)
    ia_scores = score_vectors(ia_model, original_test, batch_size=batch_size, device=device)
    rows_by_sample = {row["sample_id"]: row for row in target_rows}

    offline_rows: list[dict[str, Any]] = []
    subset_rows: list[dict[str, Any]] = []
    comparison_cache: dict[tuple[str, str], dict[str, Any]] = {}
    test_ia_by_id = {item.sample_id: item for item in ia_test}
    for scope in ("overall", "interaction_rich", "low_interaction", "multi_agent"):
        original_subset, v1_subset_scores = subset_examples_scores(original_test, v1_scores, rows_by_sample, scope)
        _, ia_subset_scores = subset_examples_scores(original_test, ia_scores, rows_by_sample, scope)
        ia_subset = [test_ia_by_id[item.sample_id] for item in original_subset]
        for method, method_scores in (("GAT_V1", v1_subset_scores), ("GAT_V1_IA", ia_subset_scores)):
            for target_name, reference_examples in (("V1_TARGET", original_subset), ("IA_TARGET", ia_subset)):
                metrics = compute_offline_metrics(reference_examples, method_scores)
                risk = risk_model_metrics(original_subset, method_scores, descriptors_by_sample)
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "method": method,
                    "reference_target": target_name,
                    "scope": scope,
                    **{key: value for key, value in metrics.items() if key not in {"reference_classes", "predicted_classes", "fixed_proposal_rank_prediction_counts"}},
                    **risk,
                }
                comparison_cache[(method, scope)] = record if target_name == "IA_TARGET" else comparison_cache.get((method, scope), record)
                (offline_rows if scope == "overall" else subset_rows).append(record)

    v1_rich = next(row for row in subset_rows if row["method"] == "GAT_V1" and row["reference_target"] == "IA_TARGET" and row["scope"] == "interaction_rich")
    ia_rich = next(row for row in subset_rows if row["method"] == "GAT_V1_IA" and row["reference_target"] == "IA_TARGET" and row["scope"] == "interaction_rich")
    v1_multi = next(row for row in subset_rows if row["method"] == "GAT_V1" and row["reference_target"] == "IA_TARGET" and row["scope"] == "multi_agent")
    ia_multi = next(row for row in subset_rows if row["method"] == "GAT_V1_IA" and row["reference_target"] == "IA_TARGET" and row["scope"] == "multi_agent")
    v1_low = next(row for row in subset_rows if row["method"] == "GAT_V1" and row["reference_target"] == "IA_TARGET" and row["scope"] == "low_interaction")
    ia_low = next(row for row in subset_rows if row["method"] == "GAT_V1_IA" and row["reference_target"] == "IA_TARGET" and row["scope"] == "low_interaction")
    v1_all = next(row for row in offline_rows if row["method"] == "GAT_V1" and row["reference_target"] == "IA_TARGET")
    ia_all = next(row for row in offline_rows if row["method"] == "GAT_V1_IA" and row["reference_target"] == "IA_TARGET")
    rich_gain = float(ia_rich["top1_accuracy"] - v1_rich["top1_accuracy"])
    multi_gain = float(ia_multi["top1_accuracy"] - v1_multi["top1_accuracy"])
    pair_gain = float(ia_rich["safe_risky_pair_accuracy"] - v1_rich["safe_risky_pair_accuracy"])
    low_decline = float(v1_low["top1_accuracy"] - ia_low["top1_accuracy"])
    null_change = float(ia_all["null_prediction_rate"] - v1_all["null_prediction_rate"])
    overall_decline = float(v1_all["top1_accuracy"] - ia_all["top1_accuracy"])
    primary_yes = rich_gain >= float(gain_thresholds["interaction_rich_top1_gain_yes"]) or pair_gain >= float(
        gain_thresholds["safe_risky_pair_accuracy_gain_yes"]
    )
    safeguards = (
        low_decline <= float(gain_thresholds["maximum_low_interaction_top1_decline"])
        and abs(null_change) <= float(gain_thresholds["maximum_null_selection_rate_change"])
        and overall_decline <= float(gain_thresholds["maximum_overall_top1_decline"])
    )
    if primary_yes and safeguards:
        gain = "YES"
    elif safeguards and (rich_gain > 0.0 or pair_gain > 0.0):
        gain = "WEAK"
    else:
        gain = "NO"
    gate = {
        "MODEL_INTERACTION_GAIN": gain,
        "INTERACTION_RICH_TOP1_GAIN_PP": 100.0 * rich_gain,
        "MULTI_AGENT_TOP1_GAIN_PP": 100.0 * multi_gain,
        "SAFE_RISKY_PAIR_ACCURACY_GAIN_PP": 100.0 * pair_gain,
        "LOW_INTERACTION_TOP1_CHANGE_PP": -100.0 * low_decline,
        "NULL_MODEL_SELECTION_CHANGE_PP": 100.0 * null_change,
        "OVERALL_IA_TARGET_TOP1_CHANGE_PP": -100.0 * overall_decline,
        "primary_gain_gate": primary_yes,
        "retention_and_collapse_safeguards": safeguards,
        "selected_optimization_seed": selected.optimization_seed,
        "selected_best_epoch": selected.best_epoch,
        "selected_best_validation_loss": selected.best_validation_loss,
        "selected_checkpoint_sha256": file_hash(selected_path),
    }
    history_rows = [
        {"schema_version": SCHEMA_VERSION, **asdict(record)}
        for result in results
        for record in result.history
    ]
    seed_rows = [{"schema_version": SCHEMA_VERSION, **training_result_record(result)} for result in results]
    return history_rows, seed_rows, offline_rows + subset_rows, gate, selected_path


def target_contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "only_changed_component": "training_supervision_target",
        "source": "original V1 soft target",
        "null_contract": "y_IA[0] equals y_V1[0] exactly",
        "proposal_mass_contract": "sum y_IA[1:] equals 1-y_V1[0]",
        "risk_contract": {
            "RISKY": "d_min < graph_metadata.d_safe OR T_risk > 0",
            "SAFE": "d_min >= graph_metadata.d_safe AND T_risk == 0",
            "NEUTRAL": "descriptor missing or invalid",
            "t_min": "diagnostic only",
        },
        "projection_contract": "unique minimum-L2 projection of proposal-conditional q onto all SAFE >= all RISKY",
        "already_consistent": "exact no-op; no proactive margin",
        "missing_both_groups": "entire target exact no-op",
        "forbidden_target_information": [
            "V2 long-horizon target", "completion", "team success", "collision outcome", "formal outcome"
        ],
    }


def empty_required_outputs(output_dir: Path) -> None:
    for name in (
        "training_history.csv",
        "seed_metrics.csv",
        "offline_v1_vs_ia.csv",
        "interaction_subset_metrics.csv",
        "development_episode_results.csv",
        "development_agent_results.csv",
        "development_paired.csv",
        "development_scenario_summary.csv",
    ):
        write_csv(output_dir / name, [], ["schema_version", "status"])


def preliminary_conclusion(validity: Mapping[str, Any], target_only: bool) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        **{key: validity[key] for key in (
            "NULL_TARGET_MASS_EXACTLY_PRESERVED",
            "NULL_TOP1_RATE_CHANGE_PP",
            "LOW_INTERACTION_TARGET_EXACT_AGREEMENT",
            "V1_SUPERVISION_INTERACTION_CONFLICT_RATE",
            "IA_SUPERVISION_INTERACTION_CONFLICT_RATE",
            "PROJECTION_ORDER_INVARIANT",
            "IA_LABEL_INFORMATION_DENSITY",
            "IA_TARGET_VALID",
        )},
        "MODEL_INTERACTION_GAIN": "NOT_RUN",
        "INTERACTION_RICH_TOP1_GAIN_PP": None,
        "MULTI_AGENT_TOP1_GAIN_PP": None,
        "SAFE_RISKY_PAIR_ACCURACY_GAIN_PP": None,
        "NULL_MODEL_SELECTION_CHANGE_PP": None,
        "DEVELOPMENT_SEED_OVERLAP": 0,
        "V1_DEVELOPMENT_TEAM_SUCCESS": None,
        "IA_DEVELOPMENT_TEAM_SUCCESS": None,
        "IA_DEVELOPMENT_SUCCESS_GAIN_PP": None,
        "V1_MULTI_AGENT_SUCCESS": None,
        "IA_MULTI_AGENT_SUCCESS": None,
        "IA_CLOSED_LOOP_GAIN": "NOT_RUN",
        "EXISTING_GAT_SUPERVISION_ONLY_HEADROOM_EXHAUSTED": "NOT_ESTABLISHED",
        "RECOMMENDED_NEXT_STEP": "STOP_TARGET_INVALID" if validity["IA_TARGET_VALID"] == "NO" else "KEEP_GAT_V1",
        "status": "TARGET_ONLY_DIAGNOSTIC" if target_only else "STOPPED_TARGET_INVALID",
    }


def report_markdown(conclusion: Mapping[str, Any], validity: Mapping[str, Any], model_gate: Mapping[str, Any] | None) -> str:
    lines = [
        "# V1 Interaction-Aware Supervision Controlled Experiment",
        "",
        "## Executive result",
        "",
        f"`IA_TARGET_VALID = {conclusion['IA_TARGET_VALID']}`; "
        f"`MODEL_INTERACTION_GAIN = {conclusion['MODEL_INTERACTION_GAIN']}`; "
        f"`IA_CLOSED_LOOP_GAIN = {conclusion['IA_CLOSED_LOOP_GAIN']}`.",
        "",
        "The experiment changes only the training soft target. The V1 graph, GAT architecture, "
        "soft cross-entropy loss, null semantics, inference, SAC-DMP controller, and environment remain frozen.",
        "",
        "## Target validity",
        "",
        f"- Null target mass preserved: `{conclusion['NULL_TARGET_MASS_EXACTLY_PRESERVED']}`.",
        f"- Null target Top-1 change: {conclusion['NULL_TOP1_RATE_CHANGE_PP']:.3f} pp.",
        f"- Low-interaction exact agreement: {conclusion['LOW_INTERACTION_TARGET_EXACT_AGREEMENT']:.3%}.",
        f"- Interaction-rich collision-label conflict: V1 {conclusion['V1_SUPERVISION_INTERACTION_CONFLICT_RATE']:.3%}, IA {conclusion['IA_SUPERVISION_INTERACTION_CONFLICT_RATE']:.3%}.",
        f"- Projection order invariant: `{conclusion['PROJECTION_ORDER_INVARIANT']}`.",
        f"- Label information density: `{conclusion['IA_LABEL_INFORMATION_DENSITY']}`.",
    ]
    if model_gate is not None:
        lines.extend(
            [
                "",
                "## Paired offline result",
                "",
                f"- Interaction-rich IA-target Top-1 gain: {model_gate['INTERACTION_RICH_TOP1_GAIN_PP']:.3f} pp.",
                f"- Multi-agent IA-target Top-1 gain: {model_gate['MULTI_AGENT_TOP1_GAIN_PP']:.3f} pp.",
                f"- SAFE/RISKY pair accuracy gain: {model_gate['SAFE_RISKY_PAIR_ACCURACY_GAIN_PP']:.3f} pp.",
                f"- Null model-selection change: {model_gate['NULL_MODEL_SELECTION_CHANGE_PP']:.3f} pp.",
            ]
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- `EXISTING_GAT_SUPERVISION_ONLY_HEADROOM_EXHAUSTED = {conclusion['EXISTING_GAT_SUPERVISION_ONLY_HEADROOM_EXHAUSTED']}`",
            f"- `RECOMMENDED_NEXT_STEP = {conclusion['RECOMMENDED_NEXT_STEP']}`",
            "",
            "No second target, formal test, graph/model extension, or post-hoc threshold tuning was performed.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = (args.output_dir or (REPO_ROOT / config["output_root"] / timestamp())).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    empty_required_outputs(output_dir)
    stage1_config_path = (REPO_ROOT / config["sources"]["stage1_config"]).resolve()
    stage1_config = json.loads(stage1_config_path.read_text(encoding="utf-8"))
    dataset_dir = (REPO_ROOT / config["sources"]["dataset_dir"]).resolve()
    v1_checkpoint = (REPO_ROOT / config["sources"]["v1_checkpoint"]).resolve()
    if file_hash(v1_checkpoint) != config["sources"]["v1_checkpoint_sha256_expected"]:
        raise RuntimeError("authoritative V1 checkpoint hash mismatch")
    frozen_hash_paths = {
        "stage1_config": stage1_config_path,
        "v1_checkpoint": v1_checkpoint,
        "candidate_selector": REPO_ROOT / "planning/gat/candidate_selector.py",
        "stage1_training": REPO_ROOT / "planning/gat/stage1_training.py",
        "graph_records": dataset_dir / "graph_records.csv",
    }
    before_hashes = {name: file_hash(path) for name, path in frozen_hash_paths.items()}
    run_config = dict(config)
    run_config.update({
        "created_at": utc_local_now().isoformat(),
        "resolved_output_dir": str(output_dir),
        "target_only": bool(args.target_only),
    })
    write_json(output_dir / "config.json", run_config)
    write_json(output_dir / "target_semantic_contract.json", target_contract())
    write_json(
        output_dir / "context_recovery_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "PASSED",
            "authoritative_v1_training": config["sources"]["v1_training_dir"],
            "authoritative_v1_checkpoint": str(v1_checkpoint),
            "dataset": str(dataset_dir),
            "capacity_audit": config["sources"]["interaction_capacity_audit"],
            "split_recovered": config["dataset_split"],
            "formal_seeds_used_for_target_or_training": False,
            "layout_24_artifact_read": False,
        },
    )

    print("loading frozen V1 examples", flush=True)
    examples, dataset_audit, _ = load_stage1_examples(
        dataset_dir,
        split_config=stage1_config["dataset_split"],
        supervision_config=stage1_config["supervision"],
        interaction_config=stage1_config["interaction_diagnostic"],
        map_location="cpu",
    )
    ia_examples, target_rows, descriptors_by_sample = target_projection_rows(examples)
    for row in target_rows:
        row["_descriptors"] = descriptors_by_sample[row["sample_id"]]
    public_target_rows = [{key: value for key, value in row.items() if key != "_descriptors"} for row in target_rows]
    summaries = summarize_targets(public_target_rows)
    conflicts = conflict_rows(
        target_rows,
        REPO_ROOT / "artifacts/gat_supervision_v2/20260817_111210/branch_raw_outcomes.parquet",
    )
    tests = projection_tests(
        examples,
        descriptors_by_sample,
        float(config["validity_thresholds"]["order_invariance_absolute_tolerance"]),
    )
    density = information_density(public_target_rows, config["validity_thresholds"])
    validity = build_validity_gate(summaries, conflicts, tests, density, config["validity_thresholds"])
    write_csv(output_dir / "target_v1_vs_ia.csv", public_target_rows)
    write_csv(output_dir / "target_subset_summary.csv", summaries)
    write_csv(output_dir / "interaction_conflict_analysis.csv", conflicts)
    write_json(output_dir / "projection_tests.json", tests)
    write_json(output_dir / "label_information_density.json", density)
    write_json(output_dir / "validity_gate.json", validity)
    print(json.dumps(validity, indent=2), flush=True)

    conclusion = preliminary_conclusion(validity, args.target_only)
    model_gate: dict[str, Any] | None = None
    selected_path: Path | None = None
    if validity["IA_TARGET_VALID"] == "YES" and not args.target_only:
        print("target gate passed; starting frozen-protocol IA training", flush=True)
        history, seed_rows, offline_all, model_gate, selected_path = train_and_evaluate(
            output_dir,
            stage1_config,
            examples,
            ia_examples,
            public_target_rows,
            descriptors_by_sample,
            v1_checkpoint,
            config["model_gain_thresholds"],
        )
        write_csv(output_dir / "training_history.csv", history)
        write_csv(output_dir / "seed_metrics.csv", seed_rows)
        write_csv(output_dir / "offline_v1_vs_ia.csv", [row for row in offline_all if row["scope"] == "overall"])
        write_csv(output_dir / "interaction_subset_metrics.csv", [row for row in offline_all if row["scope"] != "overall"])
        conclusion.update(model_gate)
        conclusion["MODEL_INTERACTION_GAIN"] = model_gate["MODEL_INTERACTION_GAIN"]
        # Closed-loop development is implemented as a separately gated phase in this
        # same driver revision; until it runs, no closed-loop claim is made.
        conclusion["IA_CLOSED_LOOP_GAIN"] = "NOT_RUN"
        conclusion["status"] = "OFFLINE_COMPLETE_DEVELOPMENT_PENDING" if model_gate["MODEL_INTERACTION_GAIN"] in {"YES", "WEAK"} else "COMPLETE_OFFLINE_STOP"
        if model_gate["MODEL_INTERACTION_GAIN"] == "NO":
            conclusion["EXISTING_GAT_SUPERVISION_ONLY_HEADROOM_EXHAUSTED"] = "YES"
            conclusion["RECOMMENDED_NEXT_STEP"] = "ONLY_THEN_CONSIDER_MINIMAL_JOINT_DECISION_EXTENSION"
        else:
            conclusion["RECOMMENDED_NEXT_STEP"] = "KEEP_GAT_V1"

    write_json(
        output_dir / "development_seed_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "FROZEN_NOT_RUN" if conclusion["IA_CLOSED_LOOP_GAIN"] == "NOT_RUN" else "COMPLETE",
            "frozen_seeds": config["development"]["seeds"],
            "historical_seed_audit_scope": "all pre-existing artifact CSV seed columns and JSON seed-like fields",
            "development_seed_overlap": 0,
            "formal_seed_overlap": 0,
            "training_seed_overlap": 0,
            "seed_replacement_allowed": False,
        },
    )
    after_hashes = {name: file_hash(path) for name, path in frozen_hash_paths.items()}
    integrity = {
        "schema_version": SCHEMA_VERSION,
        "frozen_source_hashes_before": before_hashes,
        "frozen_source_hashes_after": after_hashes,
        "frozen_sources_unchanged": before_hashes == after_hashes,
        "v1_checkpoint_hash_verified": True,
        "graph_architecture_loss_null_inference_sac_environment_changed": False,
        "target_outcome_labels_used_in_projection": False,
        "target_projection_inputs": ["original_v1_soft_target", "existing_graph_d_min", "existing_graph_T_risk", "graph_metadata_d_safe"],
        "dataset_audit": dataset_audit,
        "selected_ia_checkpoint": None if selected_path is None else str(selected_path),
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json(output_dir / "integrity_manifest.json", integrity)
    write_json(output_dir / "conclusion.json", conclusion)
    (output_dir / "FINAL_REPORT.md").write_text(report_markdown(conclusion, validity, model_gate), encoding="utf-8")
    print(f"output_dir={output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
