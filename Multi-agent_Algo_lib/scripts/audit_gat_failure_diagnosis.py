"""Read-only Stage-I GAT training sufficiency and OOD failure diagnosis.

The script never trains a model and never starts a new long-horizon episode.
It evaluates frozen checkpoints, rebuilds frozen stress graphs, and runs only
the explicitly authorised six-step supervision branches on isolated copies.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


# The environment has an incompatible optional pyarrow build.  This audit does
# not use pyarrow; suppressing the optional import keeps pandas transitive
# imports from loading that binary.
sys.modules.setdefault("pyarrow", None)

REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = Path(__file__).resolve().parent
for search_path in (REPO_ROOT, ALGO_ROOT, SCRIPTS_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from Guidance.reference_point_proposal_demo import ProposalConfig  # noqa: E402
from planning.candidate_execution_benchmark import (  # noqa: E402
    environment_state_fingerprint,
    real_candidate_rollout,
)
from planning.candidate_execution_interface import (  # noqa: E402
    graph_ready_candidate_execution,
)
from planning.candidate_supervision import (  # noqa: E402
    CandidateQualitySpec,
    average_ranks,
    branch_is_failure,
    compute_candidate_quality,
    target_bundle,
)
from planning.gat.candidate_selector import batch_candidate_graphs  # noqa: E402
from planning.gat.stage1_training import (  # noqa: E402
    Stage1Example,
    compute_offline_metrics,
    evaluate_model,
    fp_shep_score_vectors,
    load_model_checkpoint,
    load_stage1_examples,
    proposal_score_vectors,
    resolve_device,
    spearman_coefficient,
)
from planning.heterogeneous_candidate_graph import (  # noqa: E402
    HeterogeneousCandidateGraphConfig,
    build_heterogeneous_candidate_graph_from_env,
)
from planning.historical_forcing_gate import (  # noqa: E402
    HISTORICAL_GATE_NAME,
    scoped_historical_preview_and_multi_agent_transition,
)
from planning.policy_preview import (  # noqa: E402
    build_preview_inputs_from_env,
    preview_candidate,
)
from planning.pre_gat_220step_revalidation import (  # noqa: E402
    generate_immutable_candidate_bundle,
)
from planning.pre_gat_closed_loop import (  # noqa: E402
    FPSHEPOnlineScoreSpec,
    score_fp_shep_candidates,
)
from scripts import evaluate_gat_24_layout_stress as stress  # noqa: E402
from scripts import evaluate_gat_closed_loop as closed_loop  # noqa: E402
from scripts.evaluate_single_policy_aligned_multi_agent import (  # noqa: E402
    build_single_distribution_multi_config,
)


SCHEMA_VERSION = "gat_failure_diagnosis_v1"
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs/evaluation/gat_failure_diagnosis.json"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return None
        return "positive_infinity" if value > 0 else "negative_infinity"
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(_jsonable(value), ensure_ascii=False)
                    if isinstance(value, (dict, list, tuple, np.ndarray))
                    else _jsonable(value)
                    for key, value in row.items()
                }
            )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _optional_float(value: Any) -> float | None:
    if value is None or str(value).strip() in {"", "None", "null"}:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _parse_json_cell(value: Any, default: Any = None) -> Any:
    if value is None or str(value).strip() == "":
        return default
    return json.loads(str(value))


def _softmax(logits: Sequence[float]) -> np.ndarray:
    values = np.asarray(logits, dtype=float)
    shifted = values - np.max(values)
    weights = np.exp(shifted)
    return weights / np.sum(weights)


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {
            "count": int(array.size),
            "finite_count": 0,
            "mean": None,
            "median": None,
            "p05": None,
            "p90": None,
            "p95": None,
            "positive_infinity_rate": float(np.mean(np.isposinf(array)))
            if array.size
            else None,
        }
    return {
        "count": int(array.size),
        "finite_count": int(finite.size),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite, ddof=1)) if finite.size >= 2 else 0.0,
        "median": float(np.median(finite)),
        "p05": float(np.quantile(finite, 0.05)),
        "p90": float(np.quantile(finite, 0.90)),
        "p95": float(np.quantile(finite, 0.95)),
        "positive_infinity_rate": float(np.mean(np.isposinf(array))),
    }


def _rank_percentile(values: Sequence[float], index: int) -> float:
    ranks = average_ranks(values, descending=True)
    if len(ranks) <= 1:
        return 1.0
    return float(1.0 - (ranks[int(index)] - 1.0) / (len(ranks) - 1.0))


def _graph_level_top1_agreement(
    class_rows: Sequence[Mapping[str, Any]],
) -> tuple[float, int, int]:
    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in class_rows:
        groups[(str(row["layout_id"]), int(row["agent_id"]))].append(row)
    pairs: list[tuple[int, int]] = []
    for rows in groups.values():
        formal = [int(row["class_index"]) for row in rows if bool(row["formal_h6_top1"])]
        companion = [
            int(row["class_index"]) for row in rows if bool(row["companion_h6_top1"])
        ]
        if len(formal) != 1 or len(companion) != 1:
            raise ValueError("each graph must have exactly one formal and companion Top-1")
        pairs.append((formal[0], companion[0]))
    if not pairs:
        raise ValueError("no graph-level Top-1 rows were provided")
    changed = sum(left != right for left, right in pairs)
    return float(1.0 - changed / len(pairs)), changed, len(pairs)


def _safe_spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if (
        not np.isfinite(left_array).all()
        or not np.isfinite(right_array).all()
        or np.std(left_array) <= 0.0
        or np.std(right_array) <= 0.0
    ):
        return None
    value = spearman_coefficient(left_array, right_array)
    return float(value) if math.isfinite(value) else None


def _quality_row(value: Any) -> dict[str, float]:
    return {
        "task_progress": float(value.task_progress),
        "min_clearance": float(value.min_clearance),
        "max_execution_deviation": float(value.max_execution_deviation),
        "terminal_speed": float(value.terminal_speed),
    }


def _checkpoint_probabilities(score_vectors: Sequence[np.ndarray]) -> list[np.ndarray]:
    return [_softmax(scores) for scores in score_vectors]


def _augment_null_statistics(
    metrics: Mapping[str, Any], score_vectors: Sequence[np.ndarray]
) -> dict[str, Any]:
    result = dict(metrics)
    probabilities = _checkpoint_probabilities(score_vectors)
    null_probabilities = [float(values[0]) for values in probabilities]
    distribution = _distribution(null_probabilities)
    result.update(
        {
            "mean_null_probability": distribution["mean"],
            "median_null_probability": distribution["median"],
            "null_probability_p90": distribution["p90"],
            "null_probability_p95": distribution["p95"],
        }
    )
    return result


def _load_training_examples(
    config: Mapping[str, Any], stage1_config: Mapping[str, Any]
) -> tuple[list[Stage1Example], dict[str, list[Stage1Example]], dict[str, Any]]:
    examples, audit, _ = load_stage1_examples(
        REPO_ROOT / str(config["stage1_dataset_dir"]),
        split_config=stage1_config["dataset_split"],
        supervision_config=stage1_config["supervision"],
        interaction_config=stage1_config["interaction_diagnostic"],
        map_location="cpu",
    )
    by_split = {
        split: [item for item in examples if item.split == split]
        for split in ("train", "validation", "test")
    }
    return examples, by_split, audit


def _checkpoint_audit(
    *,
    config: Mapping[str, Any],
    stage1_config: Mapping[str, Any],
    by_split: Mapping[str, Sequence[Stage1Example]],
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[tuple[int, str, str], dict[str, Any]], dict[str, list[np.ndarray]]]:
    checkpoint_dir = (
        REPO_ROOT / str(config["stage1_training_dir"]) / "checkpoints"
    )
    rows: list[dict[str, Any]] = []
    index: dict[tuple[int, str, str], dict[str, Any]] = {}
    formal_scores: dict[str, list[np.ndarray]] = {}
    for seed in config["optimization_seeds"]:
        for checkpoint_kind in ("best_validation", "last"):
            checkpoint_path = checkpoint_dir / f"seed_{int(seed):03d}_{checkpoint_kind}.pt"
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model = load_model_checkpoint(checkpoint_path, stage1_config, device)
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            for split, examples in by_split.items():
                metrics, score_vectors = evaluate_model(
                    model,
                    examples,
                    batch_size=int(stage1_config["training"]["batch_size"]),
                    device=device,
                )
                metrics = _augment_null_statistics(metrics, score_vectors)
                row = {
                    "schema_version": SCHEMA_VERSION,
                    "optimization_seed": int(seed),
                    "checkpoint_kind": checkpoint_kind,
                    "checkpoint_epoch": int(payload["epoch"]),
                    "checkpoint_sha256": _sha256_file(checkpoint_path),
                    "split": split,
                    **{
                        key: metrics.get(key)
                        for key in (
                            "graph_count",
                            "loss",
                            "top1_accuracy",
                            "top3_accuracy",
                            "mrr",
                            "spearman_proposal_only_mean",
                            "spearman_valid_graph_count",
                            "null_prediction_rate",
                            "mean_null_probability",
                            "median_null_probability",
                            "null_probability_p90",
                            "null_probability_p95",
                        )
                    },
                }
                rows.append(row)
                index[(int(seed), checkpoint_kind, split)] = row
                if (
                    int(seed) == int(config["formal_checkpoint_seed"])
                    and checkpoint_kind == "best_validation"
                ):
                    formal_scores[split] = score_vectors
    return rows, index, formal_scores


def _training_sufficiency(
    *,
    config: Mapping[str, Any],
    checkpoint_index: Mapping[tuple[int, str, str], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    history = _read_csv(
        REPO_ROOT / str(config["stage1_training_dir"]) / "training_history.csv"
    )
    thresholds = config["decision_thresholds"]
    rows: list[dict[str, Any]] = []
    curve_seeds: list[dict[str, Any]] = []
    checkpoint_support: list[int] = []
    for seed in config["optimization_seeds"]:
        seed_rows = [row for row in history if int(row["optimization_seed"]) == int(seed)]
        seed_rows.sort(key=lambda row: int(row["epoch"]))
        best = min(seed_rows, key=lambda row: float(row["validation_loss"]))
        last = seed_rows[-1]
        best_epoch = int(best["epoch"])
        last_epoch = int(last["epoch"])
        best_train = checkpoint_index[(int(seed), "best_validation", "train")]
        best_validation = checkpoint_index[(int(seed), "best_validation", "validation")]
        best_test = checkpoint_index[(int(seed), "best_validation", "test")]
        last_validation = checkpoint_index[(int(seed), "last", "validation")]
        last_test = checkpoint_index[(int(seed), "last", "test")]
        train_drop = 1.0 - float(last["train_loss"]) / float(best["train_loss"])
        validation_change = (
            float(last["validation_loss"]) / float(best["validation_loss"]) - 1.0
        )
        patience_rows = seed_rows[best_epoch - 1 :]
        validation_top1_range = max(float(row["validation_top1"]) for row in patience_rows) - min(
            float(row["validation_top1"]) for row in patience_rows
        )
        validation_loss_range = max(
            float(row["validation_loss"]) for row in patience_rows
        ) - min(float(row["validation_loss"]) for row in patience_rows)
        last_window = seed_rows[-int(thresholds["validation_plateau_last_window"]) :]
        last_window_relative_range = (
            max(float(row["validation_loss"]) for row in last_window)
            - min(float(row["validation_loss"]) for row in last_window)
        ) / float(best["validation_loss"])
        overfit_signal = (
            train_drop >= float(thresholds["post_best_train_loss_clear_drop_fraction"])
            and validation_change
            >= float(thresholds["post_best_validation_loss_clear_worsening_fraction"])
        )
        noisy_signal = (
            validation_top1_range >= float(thresholds["ranking_clear_difference_pp"])
            and validation_loss_range / float(best["validation_loss"])
            >= float(thresholds["validation_plateau_relative_range"])
        )
        plateau_signal = last_window_relative_range <= float(
            thresholds["validation_plateau_relative_range"]
        )
        split_support: list[bool] = []
        for best_row, last_row in (
            (best_validation, last_validation),
            (best_test, last_test),
        ):
            top1_delta = float(last_row["top1_accuracy"]) - float(
                best_row["top1_accuracy"]
            )
            mrr_delta = float(last_row["mrr"]) - float(best_row["mrr"])
            nonregression = top1_delta >= 0.0 and mrr_delta >= 0.0
            clear_rank_gain = (
                top1_delta >= float(thresholds["ranking_clear_difference_pp"])
                or mrr_delta
                >= float(thresholds["checkpoint_mrr_clear_difference"])
            )
            loss_ok = float(last_row["loss"]) <= float(best_row["loss"]) * (
                1.0
                + float(thresholds["checkpoint_loss_max_relative_deterioration"])
            )
            split_support.append(nonregression and clear_rank_gain and loss_ok)
        seed_supports_checkpoint_issue = all(split_support)
        if seed_supports_checkpoint_issue:
            checkpoint_support.append(int(seed))
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "optimization_seed": int(seed),
                "best_epoch": best_epoch,
                "last_epoch": last_epoch,
                "train_loss_at_best": float(best["train_loss"]),
                "validation_loss_at_best": float(best["validation_loss"]),
                "train_top1_at_best": best_train["top1_accuracy"],
                "validation_top1_at_best": best_validation["top1_accuracy"],
                "test_top1_at_best": best_test["top1_accuracy"],
                "train_top3_at_best": best_train["top3_accuracy"],
                "validation_top3_at_best": best_validation["top3_accuracy"],
                "test_top3_at_best": best_test["top3_accuracy"],
                "train_mrr_at_best": best_train["mrr"],
                "validation_mrr_at_best": best_validation["mrr"],
                "test_mrr_at_best": best_test["mrr"],
                "post_best_train_loss_drop_fraction": train_drop,
                "post_best_validation_loss_change_fraction": validation_change,
                "overfit_signal": overfit_signal,
                "noisy_validation_signal": noisy_signal,
                "validation_plateau_signal": plateau_signal,
                "last_checkpoint_selection_support": seed_supports_checkpoint_issue,
            }
        )
        curve_seeds.append(
            {
                "optimization_seed": int(seed),
                "best_epoch": best_epoch,
                "last_epoch": last_epoch,
                "early_stopping_epochs_after_best": last_epoch - best_epoch,
                "train_loss_curve_available": True,
                "train_top1_curve_available": False,
                "train_top1_curve_unavailable_reason": (
                    "training_history.csv stores train_loss but not per-epoch train Top-1"
                ),
                "validation_loss_curve_available": True,
                "validation_top1_curve_available": True,
                "post_best_train_loss_drop_fraction": train_drop,
                "post_best_validation_loss_change_fraction": validation_change,
                "validation_top1_range_from_best_to_last": validation_top1_range,
                "validation_loss_relative_range_from_best_to_last": (
                    validation_loss_range / float(best["validation_loss"])
                ),
                "last_window_validation_loss_relative_range": last_window_relative_range,
                "overfit_signal": overfit_signal,
                "noisy_validation_signal": noisy_signal,
                "validation_plateau_signal": plateau_signal,
            }
        )
    overfit_count = sum(bool(row["overfit_signal"]) for row in curve_seeds)
    noisy_count = sum(bool(row["noisy_validation_signal"]) for row in curve_seeds)
    plateau_count = sum(bool(row["validation_plateau_signal"]) for row in curve_seeds)
    if overfit_count >= 2 and noisy_count >= 2:
        curve_state = "MIXED"
    elif overfit_count >= 2:
        curve_state = "OVERFIT"
    elif noisy_count >= 2:
        curve_state = "NOISY_VALIDATION"
    elif plateau_count >= 2:
        curve_state = "CONVERGED"
    else:
        curve_state = "NOT_ESTABLISHED"
    formal_seed = int(config["formal_checkpoint_seed"])
    checkpoint_issue = (
        formal_seed in checkpoint_support
        and len(checkpoint_support)
        >= int(thresholds["checkpoint_supporting_seed_count"])
    )
    curve = {
        "schema_version": SCHEMA_VERSION,
        "TRAINING_CURVE_STATE": curve_state,
        "per_seed": curve_seeds,
        "overfit_seed_count": overfit_count,
        "noisy_validation_seed_count": noisy_count,
        "plateau_seed_count": plateau_count,
        "early_stopping_premature_signal": checkpoint_issue,
        "CHECKPOINT_SELECTION_ISSUE": "YES" if checkpoint_issue else "NO",
        "checkpoint_issue_supporting_seeds": checkpoint_support,
        "thresholds_frozen_before_results": copy.deepcopy(thresholds),
    }
    gap_rows = {
        split: checkpoint_index[(formal_seed, "best_validation", split)]
        for split in ("train", "validation", "test")
    }
    train_val_test_gap = {
        "schema_version": SCHEMA_VERSION,
        "formal_checkpoint_seed": formal_seed,
        "metrics": {
            split: {
                key: gap_rows[split][key]
                for key in ("loss", "top1_accuracy", "top3_accuracy", "mrr")
            }
            for split in gap_rows
        },
        "top1_gaps": {
            "train_minus_validation": float(gap_rows["train"]["top1_accuracy"])
            - float(gap_rows["validation"]["top1_accuracy"]),
            "train_minus_test": float(gap_rows["train"]["top1_accuracy"])
            - float(gap_rows["test"]["top1_accuracy"]),
            "validation_minus_test": float(
                gap_rows["validation"]["top1_accuracy"]
            )
            - float(gap_rows["test"]["top1_accuracy"]),
        },
        "mrr_gaps": {
            "train_minus_validation": float(gap_rows["train"]["mrr"])
            - float(gap_rows["validation"]["mrr"]),
            "train_minus_test": float(gap_rows["train"]["mrr"])
            - float(gap_rows["test"]["mrr"]),
            "validation_minus_test": float(gap_rows["validation"]["mrr"])
            - float(gap_rows["test"]["mrr"]),
        },
        "diagnosis_rule": (
            "moderate train fit with small-to-moderate split gaps indicates a mixed "
            "label/feature/optimization ceiling; high train and large test gap indicates coverage"
        ),
    }
    train_top1 = float(gap_rows["train"]["top1_accuracy"])
    test_top1 = float(gap_rows["test"]["top1_accuracy"])
    if train_top1 >= 0.90 and train_top1 - test_top1 >= 0.10:
        split_diagnosis = "HIGH_TRAIN_FIT_WITH_GENERALIZATION_GAP"
    elif train_top1 < 0.80:
        split_diagnosis = "MODERATE_TRAIN_FIT_LABEL_FEATURE_OR_OPTIMIZATION_CEILING"
    else:
        split_diagnosis = "MIXED_FIT_AND_GENERALIZATION_SIGNAL"
    train_val_test_gap["diagnosis"] = split_diagnosis
    return rows, curve, train_val_test_gap


def _score_spec(stress_config: Mapping[str, Any]) -> FPSHEPOnlineScoreSpec:
    return FPSHEPOnlineScoreSpec.from_mapping(
        {
            "name": stress_config["fp_shep"]["score_name"],
            "definition_status": "frozen_closed_loop_baseline",
            "H_preview": int(stress_config["H_preview"]),
            "formula": stress_config["fp_shep"]["formula"],
            "weights": {
                "progress": 1.0,
                "clearance": 1.0,
                "deviation": 1.0,
                "terminal_speed": 0.0,
            },
            "normalization": copy.deepcopy(
                stress_config["fp_shep"]["normalization"]
            ),
            "normalization_recalibrated": False,
            "real_outcome_used_to_tune_score": False,
        }
    )


def _real_quality(
    rollouts: Sequence[Any], quality_spec: CandidateQualitySpec
) -> tuple[Any, dict[str, Any]]:
    failure = np.asarray([branch_is_failure(item) for item in rollouts], dtype=bool)
    quality = compute_candidate_quality(
        [_quality_row(item) for item in rollouts],
        failure_mask=failure,
        spec=quality_spec,
    )
    return quality, target_bundle(quality, (0.25,))


def _stress_inputs(
    *,
    config: Mapping[str, Any],
    stage1_config: Mapping[str, Any],
    formal_model: torch.nn.Module,
    device: torch.device,
) -> tuple[
    list[Stage1Example],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    stress_config = _load_json(REPO_ROOT / str(config["stress_config"]))
    manifest_path = REPO_ROOT / str(config["stress_manifest"])
    if _sha256_file(manifest_path) != str(config["stress_manifest_sha256"]):
        raise RuntimeError("frozen stress manifest hash mismatch")
    manifest = _load_json(manifest_path)
    environment_builder = stress.build_manifest_environment_builder(
        manifest, stress_config
    )
    execution_settings = stress._build_execution_settings(stress_config)
    multi_config = build_single_distribution_multi_config(num_agents=3, max_steps=220)
    policy, loaded_checkpoint = stress._load_policy(execution_settings, multi_config)
    if _sha256_file(loaded_checkpoint) != str(config["sac_checkpoint_sha256"]):
        raise RuntimeError("SAC checkpoint hash mismatch during stress reconstruction")
    proposal_config = ProposalConfig(**dict(stress_config["proposal_config"]))
    graph_config = HeterogeneousCandidateGraphConfig(
        horizon_steps=int(config["H_preview"]),
        d_align=float(stress_config["graph"]["d_align"]),
        d_align_source=str(stress_config["graph"]["d_align_source"]),
    )
    score_spec = _score_spec(stress_config)
    dataset_config = _load_json(
        REPO_ROOT / str(config["stage1_dataset_dir"]) / "config.json"
    )
    quality_spec = CandidateQualitySpec.from_mapping(dataset_config["quality"])
    stress_agent_rows = _read_csv(
        REPO_ROOT / str(config["stress_dir"]) / "per_agent_results.csv"
    )
    existing_index = {
        (row["layout_id"], row["method"], int(row["agent_id"])): row
        for row in stress_agent_rows
    }
    expected_hash = {
        layout_id: existing_index[(layout_id, "gat_stage1", 0)][
            "candidate_bundle_hash"
        ]
        for layout_id in {row["layout_id"] for row in stress_agent_rows}
    }
    examples: list[Stage1Example] = []
    class_rows: list[dict[str, Any]] = []
    graph_rows: list[dict[str, Any]] = []
    reproduction_rows: list[dict[str, Any]] = []
    source_fingerprints: list[tuple[str, str, str]] = []
    for layout_number, layout in enumerate(manifest["layouts"], start=1):
        layout_id = str(layout["layout_id"])
        family = str(layout["family"])
        seed = int(layout["evaluation_seed"])
        env, _ = environment_builder(
            config=multi_config,
            scenario=str(stress_config["scenario"]),
            seed=seed,
            peer_radius=float(stress_config["peer_radius_m"]),
        )
        try:
            initial_fingerprint = environment_state_fingerprint(env)
            immutable = generate_immutable_candidate_bundle(
                env,
                scenario=str(stress_config["scenario"]),
                seed=seed,
                proposal_config=proposal_config,
                consumer_top_k=int(config["top_k"]),
            )
            if immutable.candidate_set_hash != expected_hash[layout_id]:
                raise RuntimeError(f"candidate bundle mismatch for {layout_id}")
            proposals_by_agent = [
                closed_loop.reconstruct_proposals(candidates)[0]
                for candidates in immutable.per_agent
            ]
            previews_by_agent: list[tuple[Any, ...]] = []
            null_previews: list[Any] = []
            graphs: list[Any] = []
            preview_trace: list[str] = []

            def observe_preview(kwargs: dict[str, Any], transition: Any) -> None:
                del kwargs
                preview_trace.append(
                    str(transition.controller_info.get("forcing_gate_semantics"))
                )

            with scoped_historical_preview_and_multi_agent_transition(
                preview_observer=observe_preview
            ):
                for agent_id, proposals in enumerate(proposals_by_agent):
                    records = score_fp_shep_candidates(
                        env=env,
                        agent_index=agent_id,
                        proposals=proposals,
                        policy=policy,
                        spec=score_spec,
                    )
                    previews_by_agent.append(tuple(item.preview for item in records))
                    initial_state, local_context = build_preview_inputs_from_env(
                        env, agent_id
                    )
                    null_preview = preview_candidate(
                        initial_state=initial_state,
                        local_context=local_context,
                        candidate_goal=np.asarray(env.goals[agent_id], dtype=float),
                        policy=policy,
                        horizon=int(config["H_preview"]),
                        dmp_config=env.dmps[agent_id].config,
                        dynamics=env.dynamics[agent_id],
                    )
                    null_previews.append(null_preview)
                    executions = tuple(
                        graph_ready_candidate_execution(index, item.preview)
                        for index, item in enumerate(records)
                    )
                    graph = build_heterogeneous_candidate_graph_from_env(
                        env=env,
                        agent_index=agent_id,
                        proposals=proposals,
                        executions=executions,
                        proposal_config=proposal_config,
                        config=graph_config,
                    )
                    graphs.append(graph)
            if not preview_trace or any(
                value != HISTORICAL_GATE_NAME for value in preview_trace
            ):
                raise RuntimeError("stress graph preview did not use historical gate")

            scalar_rollouts_by_agent: list[tuple[Any, ...]] = []
            historical_rollouts_by_agent: list[tuple[Any, ...]] = []
            for agent_id, proposals in enumerate(proposals_by_agent):
                class_goals = (
                    np.asarray(env.goals[agent_id], dtype=float).copy(),
                    *(
                        np.asarray(proposal.point, dtype=float).copy()
                        for proposal in proposals
                    ),
                )
                scalar_rollouts_by_agent.append(
                    tuple(
                        real_candidate_rollout(
                            initial_env=env,
                            agent_index=agent_id,
                            candidate_goal=goal,
                            policy=policy,
                            horizon=int(config["H_label"]),
                        )
                        for goal in class_goals
                    )
                )
            with scoped_historical_preview_and_multi_agent_transition():
                for agent_id, proposals in enumerate(proposals_by_agent):
                    class_goals = (
                        np.asarray(env.goals[agent_id], dtype=float).copy(),
                        *(
                            np.asarray(proposal.point, dtype=float).copy()
                            for proposal in proposals
                        ),
                    )
                    historical_rollouts_by_agent.append(
                        tuple(
                            real_candidate_rollout(
                                initial_env=env,
                                agent_index=agent_id,
                                candidate_goal=goal,
                                policy=policy,
                                horizon=int(config["H_label"]),
                            )
                            for goal in class_goals
                        )
                    )

            for agent_id, (proposals, previews, null_preview, graph) in enumerate(
                zip(
                    proposals_by_agent,
                    previews_by_agent,
                    null_previews,
                    graphs,
                    strict=True,
                )
            ):
                scalar_quality, scalar_targets = _real_quality(
                    scalar_rollouts_by_agent[agent_id], quality_spec
                )
                historical_quality, historical_targets = _real_quality(
                    historical_rollouts_by_agent[agent_id], quality_spec
                )
                formal_preview_quality = compute_candidate_quality(
                    [_quality_row(null_preview)]
                    + [_quality_row(item) for item in previews],
                    failure_mask=np.zeros(len(proposals) + 1, dtype=bool),
                    spec=quality_spec,
                )
                formal_scores = formal_preview_quality.target_three_feature
                proposal_scores = tuple(float(item.score) for item in proposals)
                soft_target = tuple(
                    float(value)
                    for value in scalar_targets["soft_targets"]["tau_0.25"][
                        "provisional_primary_target"
                    ]
                )
                target_quality = tuple(
                    float(value) for value in scalar_quality.target_three_feature
                )
                example = Stage1Example(
                    sample_id=f"{layout_id}__ego{agent_id}",
                    state_group_id=layout_id,
                    scenario=str(stress_config["scenario"]),
                    seed=seed,
                    timestep=0,
                    ego_agent_id=agent_id,
                    split="stress",
                    graph_path=Path("<reconstructed-read-only>"),
                    label_path=Path("<offline-h6-read-only>"),
                    class_count=len(proposals) + 1,
                    proposal_count=len(proposals),
                    soft_target=soft_target,
                    target_quality=target_quality,
                    fp_shep_quality=tuple(float(value) for value in formal_scores),
                    proposal_scores=proposal_scores,
                    interaction_group="stress",
                    descriptor_risk_positive=True,
                    graph=graph,
                )
                examples.append(example)
                graph_rows.append(
                    {
                        "layout_id": layout_id,
                        "family": family,
                        "agent_id": agent_id,
                        "graph": graph,
                    }
                )
                scalar_top = int(np.argmax(scalar_quality.target_three_feature))
                historical_top = int(
                    np.argmax(historical_quality.target_three_feature)
                )
                for class_index, (scalar_rollout, historical_rollout) in enumerate(
                    zip(
                        scalar_rollouts_by_agent[agent_id],
                        historical_rollouts_by_agent[agent_id],
                        strict=True,
                    )
                ):
                    class_rows.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "layout_id": layout_id,
                            "family": family,
                            "seed": seed,
                            "agent_id": agent_id,
                            "class_index": class_index,
                            "class_kind": "null" if class_index == 0 else "proposal",
                            "candidate_id": None if class_index == 0 else class_index - 1,
                            "proposal_score": None
                            if class_index == 0
                            else proposal_scores[class_index - 1],
                            "formal_h6_gate": config["formal_h6_target_gate"],
                            "formal_h6_J_target_3": float(
                                scalar_quality.target_three_feature[class_index]
                            ),
                            "formal_h6_soft_target_tau_0p25": soft_target[class_index],
                            "formal_h6_rank": float(
                                average_ranks(
                                    scalar_quality.target_three_feature,
                                    descending=True,
                                )[class_index]
                            ),
                            "formal_h6_top1": class_index == scalar_top,
                            "formal_h6_failure": branch_is_failure(scalar_rollout),
                            "formal_h6_collision": bool(scalar_rollout.collision),
                            "formal_h6_termination": (
                                "collision"
                                if scalar_rollout.collision
                                else "terminated"
                                if scalar_rollout.terminated
                                else "truncated"
                                if scalar_rollout.truncated
                                else "horizon"
                            ),
                            "companion_h6_gate": config["companion_h6_target_gate"],
                            "companion_h6_J_target_3": float(
                                historical_quality.target_three_feature[class_index]
                            ),
                            "companion_h6_rank": float(
                                average_ranks(
                                    historical_quality.target_three_feature,
                                    descending=True,
                                )[class_index]
                            ),
                            "companion_h6_top1": class_index == historical_top,
                            "companion_h6_failure": branch_is_failure(
                                historical_rollout
                            ),
                            "formal_preview_h4_J_FP": float(formal_scores[class_index]),
                            "formal_preview_h4_rank": float(
                                average_ranks(formal_scores, descending=True)[class_index]
                            ),
                            "formal_preview_h4_gate": HISTORICAL_GATE_NAME,
                        }
                    )
                existing = existing_index[(layout_id, "gat_stage1", agent_id)]
                historical_records = _parse_json_cell(
                    existing.get("fp_shep_candidate_records"), []
                )
                preview_errors: list[float] = []
                historical_null_current_positive_infinity_count = 0
                for candidate_id, preview in enumerate(previews):
                    if candidate_id >= len(historical_records):
                        preview_errors.append(float("inf"))
                        continue
                    old = historical_records[candidate_id]
                    for attribute, old_name in (
                        ("task_progress", "preview_task_progress"),
                        ("min_clearance", "preview_min_clearance"),
                        ("max_execution_deviation", "preview_max_execution_deviation"),
                        ("terminal_speed", "preview_terminal_speed"),
                    ):
                        left = float(getattr(preview, attribute))
                        old_value = old.get(old_name)
                        if old_value is None and math.isinf(left) and left > 0.0:
                            historical_null_current_positive_infinity_count += 1
                            preview_errors.append(0.0)
                            continue
                        right = float(old_value)
                        if math.isinf(left) and math.isinf(right):
                            preview_errors.append(0.0)
                        else:
                            preview_errors.append(abs(left - right))
                reproduction_rows.append(
                    {
                        "layout_id": layout_id,
                        "family": family,
                        "agent_id": agent_id,
                        "candidate_bundle_hash_match": True,
                        "candidate_count_match": len(proposals)
                        == int(existing["K_t"]),
                        "preview_feature_max_absolute_error": max(
                            preview_errors, default=0.0
                        ),
                        "historical_null_current_positive_infinity_count": (
                            historical_null_current_positive_infinity_count
                        ),
                        "stress_selected_class_historical": int(
                            existing["selected_class"]
                        ),
                    }
                )
            final_fingerprint = environment_state_fingerprint(env)
            source_fingerprints.append(
                (layout_id, initial_fingerprint, final_fingerprint)
            )
            if initial_fingerprint != final_fingerprint:
                raise RuntimeError(f"H6 supervision mutated source env for {layout_id}")
        finally:
            env.close()
        print(f"[offline H6 {layout_number}/24] {layout_id}", flush=True)

    stress_metrics, gat_scores = evaluate_model(
        formal_model,
        examples,
        batch_size=int(stage1_config["training"]["batch_size"]),
        device=device,
    )
    del stress_metrics
    max_probability_error = 0.0
    selection_match = True
    for example, logits, reproduction in zip(
        examples, gat_scores, reproduction_rows, strict=True
    ):
        probabilities = _softmax(logits)
        existing = existing_index[
            (example.state_group_id, "gat_stage1", example.ego_agent_id)
        ]
        old_probabilities = np.asarray(
            _parse_json_cell(existing["class_probabilities"], []), dtype=float
        )
        if old_probabilities.shape != probabilities.shape:
            max_probability_error = float("inf")
            selection_match = False
        else:
            max_probability_error = max(
                max_probability_error,
                float(np.max(np.abs(probabilities - old_probabilities))),
            )
        selection_match &= int(np.argmax(probabilities)) == int(
            existing["selected_class"]
        )
        reproduction["current_selected_class"] = int(np.argmax(probabilities))
        reproduction["selection_match"] = int(np.argmax(probabilities)) == int(
            existing["selected_class"]
        )
        reproduction["probability_max_absolute_error"] = (
            float(np.max(np.abs(probabilities - old_probabilities)))
            if old_probabilities.shape == probabilities.shape
            else float("inf")
        )
    semantic_top1_agreement, semantic_top1_changes, semantic_graph_count = (
        _graph_level_top1_agreement(class_rows)
    )
    semantic_audit = {
        "formal_target_gate": config["formal_h6_target_gate"],
        "companion_target_gate": config["companion_h6_target_gate"],
        "companion_used_for_formal_ranking": False,
        "graph_gate": HISTORICAL_GATE_NAME,
        "source_environment_side_effect_free": all(
            before == after for _, before, after in source_fingerprints
        ),
        "candidate_bundle_reproduction_match": all(
            row["candidate_bundle_hash_match"] and row["candidate_count_match"]
            for row in reproduction_rows
        ),
        "preview_feature_max_absolute_error": max(
            float(row["preview_feature_max_absolute_error"])
            for row in reproduction_rows
        ),
        "historical_null_current_positive_infinity_count": sum(
            int(row["historical_null_current_positive_infinity_count"])
            for row in reproduction_rows
        ),
        "gat_probability_max_absolute_error": max_probability_error,
        "gat_selection_reproduction_match": selection_match,
        "formal_vs_companion_top1_agreement": semantic_top1_agreement,
        "formal_vs_companion_top1_change_count": semantic_top1_changes,
        "formal_vs_companion_graph_count": semantic_graph_count,
    }
    return examples, class_rows, graph_rows, reproduction_rows, semantic_audit


def _companion_h6_examples(
    examples: Sequence[Stage1Example],
    class_rows: Sequence[Mapping[str, Any]],
    *,
    temperature: float,
) -> list[Stage1Example]:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in class_rows:
        grouped[(str(row["layout_id"]), int(row["agent_id"]))].append(row)
    result: list[Stage1Example] = []
    for example in examples:
        rows = sorted(
            grouped[(example.state_group_id, example.ego_agent_id)],
            key=lambda row: int(row["class_index"]),
        )
        quality = np.asarray(
            [float(row["companion_h6_J_target_3"]) for row in rows], dtype=float
        )
        shifted = (quality - float(np.max(quality))) / float(temperature)
        weights = np.exp(shifted)
        soft = weights / np.sum(weights)
        result.append(
            replace(
                example,
                soft_target=tuple(float(value) for value in soft),
                target_quality=tuple(float(value) for value in quality),
            )
        )
    return result


def _selector_metrics_rows(
    *,
    train_test_examples: Sequence[Stage1Example],
    test_gat_scores: Sequence[np.ndarray],
    stress_examples: Sequence[Stage1Example],
    stress_companion_examples: Sequence[Stage1Example],
    stress_gat_scores: Sequence[np.ndarray],
    stress_gat_loss: float,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    metric_index: dict[str, dict[str, Any]] = {}

    def add(
        dataset: str,
        method: str,
        examples: Sequence[Stage1Example],
        scores: Sequence[np.ndarray],
        masks: Sequence[np.ndarray] | None = None,
        loss: float | None = None,
        scope: str = "null_plus_K",
    ) -> None:
        metrics = compute_offline_metrics(examples, scores, selectable_masks=masks, loss=loss)
        row = {
            "schema_version": SCHEMA_VERSION,
            "dataset": dataset,
            "method": method,
            "selection_scope": scope,
            **{
                key: metrics.get(key)
                for key in (
                    "graph_count",
                    "loss",
                    "top1_accuracy",
                    "top3_accuracy",
                    "mrr",
                    "spearman_proposal_only_mean",
                    "spearman_valid_graph_count",
                    "null_prediction_rate",
                )
            },
        }
        rows.append(row)
        metric_index[f"{dataset}:{method}"] = row

    proposal_test, proposal_test_masks = proposal_score_vectors(train_test_examples)
    proposal_stress, proposal_stress_masks = proposal_score_vectors(stress_examples)
    add(
        "stage1_test",
        "proposal",
        train_test_examples,
        proposal_test,
        proposal_test_masks,
        scope="proposal_only_no_null_score",
    )
    add(
        "stress_h6_scalar_target",
        "proposal",
        stress_examples,
        proposal_stress,
        proposal_stress_masks,
        scope="proposal_only_no_null_score",
    )
    add(
        "stage1_test",
        "fp_shep",
        train_test_examples,
        fp_shep_score_vectors(train_test_examples),
        scope="null_plus_K_stage1_offline_definition",
    )
    add(
        "stress_h6_scalar_target",
        "fp_shep",
        stress_examples,
        fp_shep_score_vectors(stress_examples),
        scope="null_plus_K_for_direct_test_comparison",
    )
    fp_stress_masks = [
        np.asarray([False] + [True] * item.proposal_count, dtype=bool)
        if item.proposal_count
        else np.asarray([True], dtype=bool)
        for item in stress_examples
    ]
    add(
        "stress_h6_scalar_target",
        "fp_shep_deployed_proposal_only",
        stress_examples,
        fp_shep_score_vectors(stress_examples),
        fp_stress_masks,
        scope="proposal_only_deployment_companion",
    )
    add(
        "stage1_test",
        "gat_stage1",
        train_test_examples,
        test_gat_scores,
        scope="null_plus_K",
    )
    companion_proposal, companion_proposal_masks = proposal_score_vectors(
        stress_companion_examples
    )
    add(
        "stress_h6_historical_companion",
        "proposal",
        stress_companion_examples,
        companion_proposal,
        companion_proposal_masks,
        scope="proposal_only_no_null_score_diagnostic",
    )
    add(
        "stress_h6_historical_companion",
        "fp_shep",
        stress_companion_examples,
        fp_shep_score_vectors(stress_companion_examples),
        scope="null_plus_K_diagnostic_only",
    )
    add(
        "stress_h6_historical_companion",
        "gat_stage1",
        stress_companion_examples,
        stress_gat_scores,
        scope="null_plus_K_diagnostic_only",
    )
    add(
        "stress_h6_scalar_target",
        "gat_stage1",
        stress_examples,
        stress_gat_scores,
        loss=stress_gat_loss,
        scope="null_plus_K",
    )
    for method in ("proposal", "fp_shep", "gat_stage1"):
        test_row = metric_index[f"stage1_test:{method}"]
        stress_row = metric_index[f"stress_h6_scalar_target:{method}"]
        stress_row["top1_degradation_from_stage1_test"] = float(
            test_row["top1_accuracy"]
        ) - float(stress_row["top1_accuracy"])
        stress_row["mrr_degradation_from_stage1_test"] = float(test_row["mrr"]) - float(
            stress_row["mrr"]
        )
    return rows, metric_index


def _feature_values(graph_rows: Sequence[Mapping[str, Any]]) -> dict[str, list[float]]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in graph_rows:
        graph = row["graph"]
        proposal = graph["proposal"].x_raw.detach().cpu().numpy()
        if proposal.size:
            for name, index in (
                ("candidate_distance", 3),
                ("sector_safety", 8),
                ("preview_progress", 9),
                ("preview_clearance", 10),
                ("preview_deviation", 11),
                ("preview_terminal_speed", 12),
            ):
                values[name].extend(float(value) for value in proposal[:, index])
        edge = graph["align", "spatiotemporal", "proposal"].edge_attr.detach().cpu().numpy()
        edge_count = int(edge.shape[0])
        values["st_edge_count"].append(float(edge_count))
        values["st_edge_zero_flag"].append(float(edge_count == 0))
        if edge.size:
            values["t_min"].extend(float(value) for value in edge[:, 0])
            values["d_min"].extend(float(value) for value in edge[:, 1])
            values["T_risk"].extend(float(value) for value in edge[:, 2])
        null = graph["null"].x_raw.detach().cpu().numpy()[0]
        values["proposal_count"].append(float(graph["proposal"].num_nodes))
        values["null_goal_distance"].append(float(null[3]))
        values["null_goal_sector_safety"].append(float(null[4]))
    return dict(values)


def _feature_shift(
    *,
    train_examples: Sequence[Stage1Example],
    stress_graph_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_rows = [{"graph": item.graph} for item in train_examples]
    train_values = _feature_values(train_rows)
    stress_values = _feature_values(stress_graph_rows)
    rows: list[dict[str, Any]] = []
    feature_to_group = {
        feature: group
        for group, features in config["feature_groups"].items()
        for feature in features
    }
    topology_features = set(config["topology_diagnostics"]["features"])
    feature_to_group.update(
        {feature: "graph_topology_diagnostic" for feature in topology_features}
    )
    group_rates: dict[str, list[float]] = defaultdict(list)
    for feature in sorted(feature_to_group):
        train_array = np.asarray(train_values.get(feature, []), dtype=float)
        stress_array = np.asarray(stress_values.get(feature, []), dtype=float)
        train_finite = train_array[np.isfinite(train_array)]
        stress_finite = stress_array[np.isfinite(stress_array)]
        if not train_finite.size or not stress_finite.size:
            outside_rate = None
            train_p05 = train_p95 = None
        else:
            train_p05 = float(np.quantile(train_finite, 0.05))
            train_p95 = float(np.quantile(train_finite, 0.95))
            outside_rate = float(
                np.mean((stress_finite < train_p05) | (stress_finite > train_p95))
            )
            if feature not in topology_features:
                group_rates[feature_to_group[feature]].append(outside_rate)
        for dataset, array in (("train", train_array), ("stress", stress_array)):
            distribution = _distribution(array)
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "feature_group": feature_to_group[feature],
                    "feature": feature,
                    "dataset": dataset,
                    **distribution,
                    "train_p05_reference": train_p05,
                    "train_p95_reference": train_p95,
                    "stress_outside_train_p05_p95_rate": outside_rate
                    if dataset == "stress"
                    else None,
                }
            )
    group_summary = {
        group: float(np.mean(rates)) if rates else None
        for group, rates in group_rates.items()
    }
    thresholds = config["decision_thresholds"]
    strong_groups = [
        group
        for group, value in group_summary.items()
        if value is not None
        and value >= float(thresholds["feature_strong_ood_outside_fraction"])
    ]
    weak_groups = [
        group
        for group, value in group_summary.items()
        if value is not None
        and value >= float(thresholds["feature_weak_ood_outside_fraction"])
    ]
    return rows, {
        "group_mean_outside_rates": group_summary,
        "strong_shift_groups": strong_groups,
        "weak_shift_groups": weak_groups,
    }


def _null_distribution_rows(
    *,
    by_split: Mapping[str, Sequence[Stage1Example]],
    formal_scores: Mapping[str, Sequence[np.ndarray]],
    stress_examples: Sequence[Stage1Example],
    stress_scores: Sequence[np.ndarray],
    stress_agent_index: Mapping[tuple[str, str, int], Mapping[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    del by_split  # split sizes are represented by the frozen score vectors

    def add(dataset: str, scores: Sequence[np.ndarray], subset: str = "overall") -> None:
        probabilities = [_softmax(value) for value in scores]
        null_values = [float(value[0]) for value in probabilities]
        distribution = _distribution(null_values)
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "dataset": dataset,
                "subset": subset,
                **distribution,
                "null_top1_rate": float(
                    np.mean([int(np.argmax(value)) == 0 for value in probabilities])
                )
                if probabilities
                else None,
            }
        )

    for split in ("train", "validation", "test"):
        add(f"stage1_{split}", formal_scores[split])
    add("stress", stress_scores)
    for success_value, name in ((True, "team_success"), (False, "team_failure")):
        selected_scores = []
        for example, score in zip(stress_examples, stress_scores, strict=True):
            outcome = stress_agent_index[
                (example.state_group_id, "gat_stage1", example.ego_agent_id)
            ]
            if _bool(outcome["team_success"]) == success_value:
                selected_scores.append(score)
        add("stress", selected_scores, name)
    return rows


def _stress_outcome_audits(
    *,
    config: Mapping[str, Any],
    stress_examples: Sequence[Stage1Example],
    stress_scores: Sequence[np.ndarray],
    class_rows: Sequence[Mapping[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    stress_dir = REPO_ROOT / str(config["stress_dir"])
    agents = _read_csv(stress_dir / "per_agent_results.csv")
    episodes = _read_csv(stress_dir / "episode_results.csv")
    agent_index = {
        (row["layout_id"], row["method"], int(row["agent_id"])): row
        for row in agents
    }
    episode_index = {
        (row["layout_id"], row["method"]): row for row in episodes
    }
    target_index = {
        (row["layout_id"], int(row["agent_id"]), int(row["class_index"])): row
        for row in class_rows
    }
    score_index = {
        (example.state_group_id, example.ego_agent_id): np.asarray(scores, dtype=float)
        for example, scores in zip(stress_examples, stress_scores, strict=True)
    }
    alignment_rows: list[dict[str, Any]] = []
    null_rows: list[dict[str, Any]] = []
    override_rows: list[dict[str, Any]] = []
    post_rows: list[dict[str, Any]] = []
    for example in stress_examples:
        layout_id = example.state_group_id
        agent_id = example.ego_agent_id
        family = next(
            row["family"]
            for row in class_rows
            if row["layout_id"] == layout_id and int(row["agent_id"]) == agent_id
        )
        gat_agent = agent_index[(layout_id, "gat_stage1", agent_id)]
        fp_agent = agent_index[(layout_id, "fp_shep", agent_id)]
        gat_episode = episode_index[(layout_id, "gat_stage1")]
        fp_episode = episode_index[(layout_id, "fp_shep")]
        gat_class = int(gat_agent["selected_class"])
        fp_candidate = int(fp_agent["selected_candidate_id"])
        fp_class = fp_candidate + 1
        logits = score_index[(layout_id, agent_id)]
        probabilities = _softmax(logits)
        target = target_index[(layout_id, agent_id, gat_class)]
        target_values = np.asarray(example.target_quality, dtype=float)
        selected_percentile = _rank_percentile(target_values, gat_class)
        alignment_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "layout_id": layout_id,
                "family": family,
                "agent_id": agent_id,
                "method": "gat_stage1",
                "selected_class": gat_class,
                "selected_null": gat_class == 0,
                "formal_h6_J_target_3": float(target["formal_h6_J_target_3"]),
                "formal_h6_target_rank": float(target["formal_h6_rank"]),
                "formal_h6_target_rank_percentile": selected_percentile,
                "formal_h6_target_top1": _bool(target["formal_h6_top1"]),
                "companion_h6_J_target_3": float(
                    target["companion_h6_J_target_3"]
                ),
                "fp_shep_h4_score": float(target["formal_preview_h4_J_FP"]),
                "gat_probability": float(probabilities[gat_class]),
                "gat_rank": int(np.flatnonzero(np.argsort(-logits) == gat_class)[0])
                + 1,
                "reference_reached": _bool(gat_agent["reference_reached"]),
                "terminal_completed_after_reference": _bool(
                    gat_agent["terminal_completed_after_reference"]
                ),
                "team_success": _bool(gat_episode["team_success"]),
                "collision": _bool(gat_episode["collision"]),
                "timeout": _bool(gat_episode["timeout"]),
            }
        )
        fp_target = target_index[(layout_id, agent_id, fp_class)]
        alignment_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "layout_id": layout_id,
                "family": family,
                "agent_id": agent_id,
                "method": "fp_shep",
                "selected_class": fp_class,
                "selected_null": False,
                "formal_h6_J_target_3": float(fp_target["formal_h6_J_target_3"]),
                "formal_h6_target_rank": float(fp_target["formal_h6_rank"]),
                "formal_h6_target_rank_percentile": _rank_percentile(
                    target_values, fp_class
                ),
                "formal_h6_target_top1": _bool(fp_target["formal_h6_top1"]),
                "companion_h6_J_target_3": float(
                    fp_target["companion_h6_J_target_3"]
                ),
                "fp_shep_h4_score": float(fp_target["formal_preview_h4_J_FP"]),
                "gat_probability": None,
                "gat_rank": None,
                "reference_reached": _bool(fp_agent["reference_reached"]),
                "terminal_completed_after_reference": _bool(
                    fp_agent["terminal_completed_after_reference"]
                ),
                "team_success": _bool(fp_episode["team_success"]),
                "collision": _bool(fp_episode["collision"]),
                "timeout": _bool(fp_episode["timeout"]),
            }
        )
        fp_scores = np.asarray(example.fp_shep_quality, dtype=float)
        fp_rank_gat = (
            None
            if gat_class == 0
            else int(np.flatnonzero(np.argsort(-fp_scores[1:]) == (gat_class - 1))[0])
            + 1
        )
        fp_top_proposal_score = float(np.max(fp_scores[1:]))
        score_gap = fp_top_proposal_score - float(fp_scores[gat_class])
        sorted_probabilities = np.sort(probabilities)[::-1]
        override_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "layout_id": layout_id,
                "family": family,
                "agent_id": agent_id,
                "selector_disagreement": gat_class != fp_class,
                "gat_selected_class": gat_class,
                "gat_selected_null": gat_class == 0,
                "fp_selected_class": fp_class,
                "fp_rank_of_gat_selection_1based": fp_rank_gat,
                "fp_score_gap_top1_minus_gat_selection": score_gap,
                "gat_confidence": float(probabilities[gat_class]),
                "gat_top1_top2_margin": float(
                    sorted_probabilities[0] - sorted_probabilities[1]
                ),
                "gat_team_success": _bool(gat_episode["team_success"]),
                "fp_team_success": _bool(fp_episode["team_success"]),
                "gat_collision": _bool(gat_episode["collision"]),
                "gat_timeout": _bool(gat_episode["timeout"]),
                "harmful_layout_context": _bool(fp_episode["team_success"])
                and not _bool(gat_episode["team_success"]),
            }
        )
        if gat_class == 0:
            if _bool(gat_episode["team_success"]) and _bool(fp_episode["team_success"]):
                category = "BOTH_SUCCESS"
            elif _bool(gat_episode["team_success"]) and not _bool(
                fp_episode["team_success"]
            ):
                category = "GAT_NULL_SUCCESS_FP_FAILURE"
            elif not _bool(gat_episode["team_success"]) and _bool(
                fp_episode["team_success"]
            ):
                category = "GAT_NULL_FAILURE_FP_SUCCESS"
            else:
                category = "BOTH_FAIL"
            null_target = target_index[(layout_id, agent_id, 0)]
            null_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "layout_id": layout_id,
                    "family": family,
                    "agent_id": agent_id,
                    "gat_null_probability": float(probabilities[0]),
                    "gat_null_team_success": _bool(gat_episode["team_success"]),
                    "gat_null_collision": _bool(gat_episode["collision"]),
                    "gat_null_timeout": _bool(gat_episode["timeout"]),
                    "fp_team_success": _bool(fp_episode["team_success"]),
                    "fp_collision": _bool(fp_episode["collision"]),
                    "fp_timeout": _bool(fp_episode["timeout"]),
                    "paired_actual_category": category,
                    "fp_selected_class": fp_class,
                    "formal_h6_null_target": float(
                        null_target["formal_h6_J_target_3"]
                    ),
                    "formal_h6_fp_selected_target": float(
                        fp_target["formal_h6_J_target_3"]
                    ),
                    "formal_h6_fp_minus_null_target": float(
                        fp_target["formal_h6_J_target_3"]
                    )
                    - float(null_target["formal_h6_J_target_3"]),
                    "fp_h4_top_proposal_minus_null_score": fp_top_proposal_score
                    - float(fp_scores[0]),
                    "counterfactual_execution_generated": False,
                }
            )

        if _bool(gat_agent["reference_reached"]) and not _bool(
            gat_agent["terminal_completed_after_reference"]
        ):
            collision_after = _bool(gat_agent["collision_after_reference"])
            obstacle = _bool(gat_episode["obstacle_collision"])
            inter_agent = _bool(gat_episode["inter_agent_collision"])
            timeout_after = _bool(gat_agent["timeout_after_reference"])
            if collision_after and obstacle and not inter_agent:
                category = "OBSTACLE_COLLISION_AFTER_REFERENCE"
            elif collision_after and inter_agent and not obstacle:
                category = "INTER_AGENT_COLLISION_AFTER_REFERENCE"
            elif collision_after:
                category = "COLLISION_TYPE_AMBIGUOUS_AFTER_REFERENCE"
            elif timeout_after:
                category = "TIMEOUT_AFTER_REFERENCE"
            else:
                category = "OTHER_UNRESOLVED_FROM_EXISTING_LOGS"
            post_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "layout_id": layout_id,
                    "family": family,
                    "agent_id": agent_id,
                    "selected_class": gat_class,
                    "primary_category": category,
                    "collision_after_reference": collision_after,
                    "team_obstacle_collision": obstacle,
                    "team_inter_agent_collision": inter_agent,
                    "timeout_after_reference": timeout_after,
                    "stagnation_after_reference_available": False,
                    "terminal_direction_geometry_label_available": False,
                    "raw_existing_logs_only": True,
                }
            )

    gat_alignment = [row for row in alignment_rows if row["method"] == "gat_stage1"]
    proposal_selected = [row for row in gat_alignment if not row["selected_null"]]
    reach_left = [float(row["formal_h6_target_rank_percentile"]) for row in proposal_selected]
    reach_right = [float(bool(row["reference_reached"])) for row in proposal_selected]
    reached = [row for row in proposal_selected if row["reference_reached"]]
    completion_left = [
        float(row["formal_h6_target_rank_percentile"]) for row in reached
    ]
    completion_right = [
        float(bool(row["terminal_completed_after_reference"])) for row in reached
    ]
    layout_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in gat_alignment:
        layout_groups[row["layout_id"]].append(row)
    team_left = [
        float(np.mean([row["formal_h6_target_rank_percentile"] for row in rows]))
        for rows in layout_groups.values()
    ]
    team_right = [float(bool(rows[0]["team_success"])) for rows in layout_groups.values()]
    correlations = {
        "target_rank_percentile_vs_reference_reach": {
            "n": len(reach_left),
            "spearman": _safe_spearman(reach_left, reach_right),
        },
        "target_rank_percentile_vs_reached_to_terminal": {
            "n": len(completion_left),
            "spearman": _safe_spearman(completion_left, completion_right),
        },
        "layout_mean_target_rank_percentile_vs_team_success": {
            "n": len(team_left),
            "spearman": _safe_spearman(team_left, team_right),
        },
    }
    available = [
        float(row["spearman"])
        for row in correlations.values()
        if row["spearman"] is not None
        and int(row["n"])
        >= int(config["decision_thresholds"]["alignment_minimum_sample_count"])
    ]
    if len(available) < 2:
        alignment_class = "NOT_ESTABLISHED"
    else:
        median = float(np.median(available))
        if median >= float(
            config["decision_thresholds"]["alignment_strong_absolute_correlation"]
        ) and all(value > 0.0 for value in available):
            alignment_class = "STRONG"
        elif median >= float(
            config["decision_thresholds"]["alignment_moderate_absolute_correlation"]
        ) and sum(value > 0.0 for value in available) >= 2:
            alignment_class = "MODERATE"
        elif median > 0.0:
            alignment_class = "WEAK"
        else:
            alignment_class = "NO"
    summary = {
        "H6_TARGET_LONG_HORIZON_ALIGNMENT": alignment_class,
        "correlations": correlations,
        "gat_selected_h6_top1_but_team_failed_agent_decisions": sum(
            bool(row["formal_h6_target_top1"]) and not bool(row["team_success"])
            for row in gat_alignment
        ),
        "gat_selected_h6_non_top1_but_team_succeeded_agent_decisions": sum(
            not bool(row["formal_h6_target_top1"]) and bool(row["team_success"])
            for row in gat_alignment
        ),
    }
    return alignment_rows, null_rows, override_rows, post_rows, summary


def _null_and_override_conclusions(
    *,
    config: Mapping[str, Any],
    null_rows: Sequence[Mapping[str, Any]],
    null_distribution_rows: Sequence[Mapping[str, Any]],
    override_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    thresholds = config["decision_thresholds"]
    distribution_index = {
        (row["dataset"], row["subset"]): row for row in null_distribution_rows
    }
    test = distribution_index[("stage1_test", "overall")]
    stress_row = distribution_index[("stress", "overall")]
    mean_shift = float(stress_row["mean"]) - float(test["mean"])
    median_shift = float(stress_row["median"]) - float(test["median"])
    top1_shift = float(stress_row["null_top1_rate"]) - float(
        test["null_top1_rate"]
    )
    if (
        max(mean_shift, median_shift)
        >= float(thresholds["null_strong_probability_shift"])
        and top1_shift >= float(thresholds["null_strong_top1_shift"])
    ):
        null_shift = "YES"
    elif max(mean_shift, median_shift, top1_shift) >= float(
        thresholds["null_weak_shift"]
    ):
        null_shift = "WEAK"
    else:
        null_shift = "NO"

    by_layout: dict[str, set[str]] = defaultdict(set)
    for row in null_rows:
        by_layout[str(row["layout_id"])].add(str(row["paired_actual_category"]))
    harmful_layouts = sum(
        "GAT_NULL_FAILURE_FP_SUCCESS" in categories
        for categories in by_layout.values()
    )
    beneficial_layouts = sum(
        "GAT_NULL_SUCCESS_FP_FAILURE" in categories
        for categories in by_layout.values()
    )
    null_layout_count = len(by_layout)
    if null_layout_count < int(thresholds["null_harm_minimum_layout_count"]):
        null_harm = "NOT_ESTABLISHED"
    elif (
        harmful_layouts - beneficial_layouts
        >= int(thresholds["null_harm_yes_excess_count"])
        and (harmful_layouts - beneficial_layouts) / null_layout_count >= 0.10
    ):
        null_harm = "YES"
    elif harmful_layouts > beneficial_layouts:
        null_harm = "WEAK"
    else:
        null_harm = "NO"

    harmful_rows = [row for row in override_rows if row["harmful_layout_context"]]
    harmful_by_layout: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in harmful_rows:
        harmful_by_layout[str(row["layout_id"])].append(row)
    pattern_counts: Counter[str] = Counter()
    for rows in harmful_by_layout.values():
        patterns = {
            "NULL_SELECTION": any(bool(row["gat_selected_null"]) for row in rows),
            "LOW_FP_RANK_SELECTION": any(
                row["fp_rank_of_gat_selection_1based"] is not None
                and int(row["fp_rank_of_gat_selection_1based"])
                >= int(thresholds["low_fp_rank_threshold_1based"])
                for row in rows
            ),
            "LARGE_FP_SCORE_GAP": any(
                float(row["fp_score_gap_top1_minus_gat_selection"])
                >= float(thresholds["large_fp_score_gap"])
                for row in rows
            ),
            "HIGH_CONFIDENCE_OVERRIDE": any(
                float(row["gat_confidence"])
                >= float(thresholds["high_gat_confidence"])
                or float(row["gat_top1_top2_margin"])
                >= float(thresholds["high_gat_margin"])
                for row in rows
            ),
            "COLLISION": any(bool(row["gat_collision"]) for row in rows),
            "TIMEOUT": any(bool(row["gat_timeout"]) for row in rows),
        }
        pattern_counts.update(key for key, value in patterns.items() if value)
    harmful_count = len(harmful_by_layout)
    pattern_rates = {
        key: value / harmful_count if harmful_count else None
        for key, value in pattern_counts.items()
    }
    if harmful_count < int(thresholds["harmful_override_minimum_layout_count"]):
        harmful_pattern = "NOT_ESTABLISHED"
    elif (
        harmful_count
        >= int(thresholds["harmful_override_clear_minimum_layout_count"])
        and max(pattern_rates.values(), default=0.0)
        >= float(thresholds["harmful_override_clear_pattern_fraction"])
    ):
        harmful_pattern = "CLEAR"
    elif max(pattern_rates.values(), default=0.0) >= float(
        thresholds["harmful_override_partial_pattern_fraction"]
    ):
        harmful_pattern = "PARTIAL"
    else:
        harmful_pattern = "NO"
    return {
        "NULL_OOD_SHIFT": null_shift,
        "null_mean_probability_shift_stress_minus_test": mean_shift,
        "null_median_probability_shift_stress_minus_test": median_shift,
        "null_top1_rate_shift_stress_minus_test": top1_shift,
        "NULL_SELECTION_HARM_SIGNAL": null_harm,
        "null_selected_agent_decision_count": len(null_rows),
        "null_selected_unique_layout_count": null_layout_count,
        "null_harmful_layout_count": harmful_layouts,
        "null_beneficial_layout_count": beneficial_layouts,
        "HARMFUL_OVERRIDE_PATTERN": harmful_pattern,
        "harmful_override_layout_count": harmful_count,
        "harmful_override_pattern_counts": dict(pattern_counts),
        "harmful_override_pattern_rates": pattern_rates,
    }


def _family_diagnosis(
    *,
    config: Mapping[str, Any],
    stress_graph_rows: Sequence[Mapping[str, Any]],
    train_feature_rows: Sequence[Mapping[str, Any]],
    override_rows: Sequence[Mapping[str, Any]],
    post_rows: Sequence[Mapping[str, Any]],
    null_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    stress_dir = REPO_ROOT / str(config["stress_dir"])
    family_summary = _read_csv(stress_dir / "family_summary.csv")
    train_ranges = {
        row["feature"]: (row["p05"], row["p95"])
        for row in train_feature_rows
        if row["dataset"] == "train"
    }
    train_topology = {
        row["feature"]: row
        for row in train_feature_rows
        if row["dataset"] == "train"
        and row["feature"] in set(config["topology_diagnostics"]["features"])
    }
    rows: list[dict[str, Any]] = []
    for family in config["families"]:
        family_graphs = [row for row in stress_graph_rows if row["family"] == family]
        family_values = _feature_values(family_graphs)
        outside_by_feature: dict[str, float] = {}
        for feature, values in family_values.items():
            if feature not in train_ranges:
                continue
            low, high = train_ranges[feature]
            if low is None or high is None:
                continue
            array = np.asarray(values, dtype=float)
            array = array[np.isfinite(array)]
            if array.size:
                outside_by_feature[feature] = float(
                    np.mean((array < float(low)) | (array > float(high)))
                )
        group_rates: dict[str, float] = {}
        for group, features in config["feature_groups"].items():
            available = [outside_by_feature[item] for item in features if item in outside_by_feature]
            if available:
                group_rates[group] = float(np.mean(available))
        fp = next(
            row
            for row in family_summary
            if row["family"] == family and row["method"] == "fp_shep"
        )
        gat = next(
            row
            for row in family_summary
            if row["family"] == family and row["method"] == "gat_stage1"
        )
        family_overrides = [row for row in override_rows if row["family"] == family]
        family_null = [row for row in null_rows if row["family"] == family]
        family_post = [row for row in post_rows if row["family"] == family]
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "family": family,
                "fp_shep_success_count": int(fp["team_success_count"]),
                "gat_success_count": int(gat["team_success_count"]),
                "gat_minus_fp_success_rate": float(gat["team_success_rate"])
                - float(fp["team_success_rate"]),
                "gat_obstacle_collision_count": int(gat["obstacle_collision_count"]),
                "gat_inter_agent_collision_count": int(
                    gat["inter_agent_collision_count"]
                ),
                "gat_timeout_count": int(gat["timeout_count"]),
                "selector_disagreement_rate": float(
                    np.mean([row["selector_disagreement"] for row in family_overrides])
                ),
                "null_selection_rate": len(family_null) / 12.0,
                "post_reference_failure_agent_count": len(family_post),
                "candidate_distance_median": _distribution(
                    family_values.get("candidate_distance", [])
                )["median"],
                "sector_safety_median": _distribution(
                    family_values.get("sector_safety", [])
                )["median"],
                "preview_progress_median": _distribution(
                    family_values.get("preview_progress", [])
                )["median"],
                "preview_clearance_median": _distribution(
                    family_values.get("preview_clearance", [])
                )["median"],
                "preview_deviation_median": _distribution(
                    family_values.get("preview_deviation", [])
                )["median"],
                "t_min_median": _distribution(family_values.get("t_min", []))[
                    "median"
                ],
                "d_min_median": _distribution(family_values.get("d_min", []))[
                    "median"
                ],
                "T_risk_median": _distribution(
                    family_values.get("T_risk", [])
                )["median"],
                "st_edge_count_median": _distribution(
                    family_values.get("st_edge_count", [])
                )["median"],
                "st_edge_zero_graph_rate": _distribution(
                    family_values.get("st_edge_zero_flag", [])
                )["mean"],
                "train_st_edge_zero_graph_rate_reference": train_topology[
                    "st_edge_zero_flag"
                ]["mean"],
                "topology_diagnostic_used_for_coverage_classification": False,
                "feature_group_outside_train_rates": group_rates,
                "strong_ood_group_count": sum(
                    value
                    >= float(
                        config["decision_thresholds"][
                            "feature_strong_ood_outside_fraction"
                        ]
                    )
                    for value in group_rates.values()
                ),
                "family_sample_count_note": "4 layouts; diagnostic only, no significance claim",
            }
        )
    return rows


def _final_conclusion(
    *,
    config: Mapping[str, Any],
    curve: Mapping[str, Any],
    metric_index: Mapping[str, Mapping[str, Any]],
    feature_summary: Mapping[str, Any],
    alignment_summary: Mapping[str, Any],
    null_summary: Mapping[str, Any],
    semantic_audit: Mapping[str, Any],
) -> dict[str, Any]:
    thresholds = config["decision_thresholds"]
    test_gat = metric_index["stage1_test:gat_stage1"]
    test_fp = metric_index["stage1_test:fp_shep"]
    stress_gat = metric_index["stress_h6_scalar_target:gat_stage1"]
    stress_fp = metric_index["stress_h6_scalar_target:fp_shep"]
    test_advantage = float(test_gat["top1_accuracy"]) - float(
        test_fp["top1_accuracy"]
    )
    stress_advantage = float(stress_gat["top1_accuracy"]) - float(
        stress_fp["top1_accuracy"]
    )
    companion_gat = metric_index["stress_h6_historical_companion:gat_stage1"]
    companion_fp = metric_index["stress_h6_historical_companion:fp_shep"]
    companion_advantage = float(companion_gat["top1_accuracy"]) - float(
        companion_fp["top1_accuracy"]
    )
    stress_summary_rows = _read_csv(
        REPO_ROOT / str(config["stress_dir"]) / "method_summary.csv"
    )
    stress_overall = {
        row["method"]: row
        for row in stress_summary_rows
        if row["scenario"] == "overall"
    }
    stress_closed_loop_advantage = float(
        stress_overall["gat_stage1"]["team_success_rate"]
    ) - float(stress_overall["fp_shep"]["team_success_rate"])
    main_summary_rows = _read_csv(
        REPO_ROOT / str(config["main_closed_loop_dir"]) / "method_summary.csv"
    )
    main_overall = {
        row["method"]: row
        for row in main_summary_rows
        if row["scenario"] == "overall"
    }
    main_closed_loop_advantage = float(
        main_overall["gat_stage1"]["team_success_rate"]
    ) - float(main_overall["fp_shep"]["team_success_rate"])
    ood_gap = (
        test_advantage >= float(thresholds["ranking_clear_difference_pp"])
        and stress_advantage <= 0.0
    )
    short_long_signal = (
        stress_advantage >= float(thresholds["ranking_clear_difference_pp"])
        and stress_closed_loop_advantage <= 0.0
    )
    horizon_only_confounded = short_long_signal and companion_advantage < float(
        thresholds["ranking_clear_difference_pp"]
    )
    strong_coverage = (
        len(feature_summary["strong_shift_groups"])
        >= int(thresholds["feature_strong_ood_minimum_groups"])
    )
    weak_coverage = (
        len(feature_summary["weak_shift_groups"])
        >= int(thresholds["feature_weak_ood_minimum_groups"])
    )
    if ood_gap and strong_coverage:
        data_coverage = "YES"
    elif ood_gap or weak_coverage:
        data_coverage = "WEAK"
    else:
        data_coverage = "NO"
    checkpoint_issue = str(curve["CHECKPOINT_SELECTION_ISSUE"])
    if checkpoint_issue == "YES" or curve["TRAINING_CURVE_STATE"] == "UNDERTRAINED":
        optimization = "YES"
    elif curve["TRAINING_CURVE_STATE"] in {"OVERFIT", "CONVERGED", "MIXED"}:
        optimization = "NO"
    else:
        optimization = "NOT_ESTABLISHED"
    alignment = str(alignment_summary["H6_TARGET_LONG_HORIZON_ALIGNMENT"])
    if short_long_signal and alignment in {"WEAK", "NO"}:
        supervision = "WEAK" if horizon_only_confounded else "YES"
    elif alignment in {"WEAK", "NO"}:
        supervision = "WEAK"
    elif alignment == "NOT_ESTABLISHED":
        supervision = "NOT_ESTABLISHED"
    else:
        supervision = "NO"
    if data_coverage == "YES" and null_summary["NULL_OOD_SHIFT"] == "YES":
        primary = "MIXED"
        next_direction = "MORE_TRAINING_GEOMETRIES"
    elif data_coverage in {"YES", "WEAK"}:
        primary = "DATA_COVERAGE"
        next_direction = "MORE_TRAINING_GEOMETRIES"
    elif supervision in {"YES", "WEAK"} and short_long_signal:
        primary = "SUPERVISION_MISMATCH"
        next_direction = "SUPERVISION_TARGET_REDESIGN"
    elif (
        null_summary["NULL_OOD_SHIFT"] == "YES"
        and null_summary["NULL_SELECTION_HARM_SIGNAL"] in {"YES", "WEAK"}
    ):
        primary = "NULL_CALIBRATION"
        next_direction = "NULL_CALIBRATION"
    elif optimization == "YES":
        primary = "OPTIMIZATION"
        next_direction = "MORE_EPOCHS"
    else:
        primary = "NOT_ESTABLISHED"
        next_direction = "OTHER"
    return {
        "schema_version": SCHEMA_VERSION,
        "OPTIMIZATION_INSUFFICIENCY": optimization,
        "CHECKPOINT_SELECTION_ISSUE": checkpoint_issue,
        "TRAINING_CURVE_STATE": curve["TRAINING_CURVE_STATE"],
        "TRAINING_DATA_COVERAGE_GAP": data_coverage,
        "OOD_RANKING_GENERALIZATION_GAP": "YES" if ood_gap else "NO",
        "SHORT_LONG_HORIZON_MISMATCH_SIGNAL": "YES"
        if short_long_signal
        else "NO",
        "SUPERVISION_HORIZON_MISMATCH": supervision,
        "HORIZON_ONLY_ATTRIBUTION": (
            "CONFOUNDED_BY_TRAINING_DEPLOYMENT_GATE_SEMANTICS"
            if horizon_only_confounded
            else "SUPPORTED_WITHOUT_CLEAR_GATE_CONFOUND"
            if short_long_signal
            else "NOT_ESTABLISHED"
        ),
        "SUPERVISION_MISMATCH_SCOPE": (
            "TRAINING_DEPLOYMENT_GATE_SEMANTICS_PLUS_WEAK_SHORT_LONG_ALIGNMENT"
            if horizon_only_confounded
            else "SHORT_LABEL_HORIZON_VS_LONG_HORIZON_EXECUTION"
            if short_long_signal
            else "NOT_ESTABLISHED"
        ),
        "H6_TARGET_LONG_HORIZON_ALIGNMENT": alignment,
        "NULL_OOD_SHIFT": null_summary["NULL_OOD_SHIFT"],
        "NULL_SELECTION_HARM_SIGNAL": null_summary[
            "NULL_SELECTION_HARM_SIGNAL"
        ],
        "HARMFUL_OVERRIDE_PATTERN": null_summary["HARMFUL_OVERRIDE_PATTERN"],
        "PRIMARY_GAT_LIMITATION": primary,
        "RECOMMENDED_NEXT_DIRECTION": next_direction,
        "STRESS_SET_STATUS": config["stress_set_status"],
        "stage1_test_gat_minus_fp_top1": test_advantage,
        "stress_offline_gat_minus_fp_top1": stress_advantage,
        "stress_h6_historical_companion_gat_minus_fp_top1": companion_advantage,
        "main_closed_loop_gat_minus_fp_success": main_closed_loop_advantage,
        "stress_closed_loop_gat_minus_fp_success": stress_closed_loop_advantage,
        "training_deployment_gate_semantic_shift_audited": True,
        "formal_h6_target_gate": config["formal_h6_target_gate"],
        "companion_h6_target_gate": config["companion_h6_target_gate"],
        "formal_vs_companion_h6_top1_agreement": semantic_audit[
            "formal_vs_companion_top1_agreement"
        ],
        "formal_vs_companion_h6_top1_change_count": semantic_audit[
            "formal_vs_companion_top1_change_count"
        ],
        "formal_vs_companion_h6_top1_change_rate": 1.0
        - float(semantic_audit["formal_vs_companion_top1_agreement"]),
        "automatic_fix_or_training_started": False,
    }


def _report(
    *,
    conclusion: Mapping[str, Any],
    curve: Mapping[str, Any],
    metric_index: Mapping[str, Mapping[str, Any]],
    feature_summary: Mapping[str, Any],
    alignment: Mapping[str, Any],
    null_summary: Mapping[str, Any],
    semantic: Mapping[str, Any],
) -> str:
    lines = [
        "# Stage-I GAT Training Sufficiency & OOD Failure Audit",
        "",
        "This is a read-only diagnostic. No model, dataset, method, threshold, "
        "geometry, or long-horizon execution was modified.",
        "",
        "## Primary conclusion",
        "",
        f"- PRIMARY_GAT_LIMITATION = `{conclusion['PRIMARY_GAT_LIMITATION']}`",
        f"- RECOMMENDED_NEXT_DIRECTION = `{conclusion['RECOMMENDED_NEXT_DIRECTION']}`",
        f"- STRESS_SET_STATUS = `{conclusion['STRESS_SET_STATUS']}`",
        "",
        "## Training sufficiency",
        "",
        f"- TRAINING_CURVE_STATE = `{curve['TRAINING_CURVE_STATE']}`",
        f"- OPTIMIZATION_INSUFFICIENCY = `{conclusion['OPTIMIZATION_INSUFFICIENCY']}`",
        f"- CHECKPOINT_SELECTION_ISSUE = `{conclusion['CHECKPOINT_SELECTION_ISSUE']}`",
        "",
        "## Offline ranking",
        "",
        f"- Stage-I test GAT Top-1 = {metric_index['stage1_test:gat_stage1']['top1_accuracy']:.4f}",
        f"- Stage-I test FP-SHEP Top-1 = {metric_index['stage1_test:fp_shep']['top1_accuracy']:.4f}",
        f"- Stress H6 GAT Top-1 = {metric_index['stress_h6_scalar_target:gat_stage1']['top1_accuracy']:.4f}",
        f"- Stress H6 FP-SHEP Top-1 = {metric_index['stress_h6_scalar_target:fp_shep']['top1_accuracy']:.4f}",
        f"- Historical-gate companion GAT Top-1 = {metric_index['stress_h6_historical_companion:gat_stage1']['top1_accuracy']:.4f}",
        f"- Historical-gate companion FP-SHEP Top-1 = {metric_index['stress_h6_historical_companion:fp_shep']['top1_accuracy']:.4f}",
        f"- OOD_RANKING_GENERALIZATION_GAP = `{conclusion['OOD_RANKING_GENERALIZATION_GAP']}`",
        "",
        "## Horizon and long-horizon alignment",
        "",
        f"- H6_TARGET_LONG_HORIZON_ALIGNMENT = `{alignment['H6_TARGET_LONG_HORIZON_ALIGNMENT']}`",
        f"- SUPERVISION_HORIZON_MISMATCH = `{conclusion['SUPERVISION_HORIZON_MISMATCH']}`",
        f"- HORIZON_ONLY_ATTRIBUTION = `{conclusion['HORIZON_ONLY_ATTRIBUTION']}`",
        f"- SUPERVISION_MISMATCH_SCOPE = `{conclusion['SUPERVISION_MISMATCH_SCOPE']}`",
        f"- main closed-loop GAT - FP-SHEP success = {conclusion['main_closed_loop_gat_minus_fp_success']:+.4f}",
        f"- stress closed-loop GAT - FP-SHEP success = {conclusion['stress_closed_loop_gat_minus_fp_success']:+.4f}",
        f"- scalar-training vs historical-deployment H6 Top-1 agreement = {semantic['formal_vs_companion_top1_agreement']:.4f}",
        f"- changed H6 Top-1 graphs = {semantic['formal_vs_companion_top1_change_count']}/{semantic['formal_vs_companion_graph_count']}",
        f"- post-reference failures = {conclusion['post_reference_failure_count']}: {conclusion['post_reference_failure_categories']}",
        "",
        "## Null and selector override",
        "",
        f"- NULL_OOD_SHIFT = `{null_summary['NULL_OOD_SHIFT']}`",
        f"- NULL_SELECTION_HARM_SIGNAL = `{null_summary['NULL_SELECTION_HARM_SIGNAL']}`",
        f"- HARMFUL_OVERRIDE_PATTERN = `{null_summary['HARMFUL_OVERRIDE_PATTERN']}`",
        "",
        "## Feature coverage",
        "",
        f"- strong shift groups: `{feature_summary['strong_shift_groups']}`",
        f"- weak shift groups: `{feature_summary['weak_shift_groups']}`",
        f"- TRAINING_DATA_COVERAGE_GAP = `{conclusion['TRAINING_DATA_COVERAGE_GAP']}`",
        "",
        "## Stop rule",
        "",
        "The audit stopped after offline diagnosis. No retraining, data generation, "
        "threshold tuning, FP-SHEP change, or new closed-loop experiment was started.",
    ]
    return "\n".join(lines) + "\n"


def run_audit(config: Mapping[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    config_path = DEFAULT_CONFIG_PATH
    evaluator_path = Path(__file__).resolve()
    config_hash_before = _sha256_file(config_path)
    evaluator_hash_before = _sha256_file(evaluator_path)
    if not config["strict_read_only"]["threshold_adjustment_after_results_allowed"]:
        thresholds_frozen = True
    else:
        raise RuntimeError("diagnostic thresholds must be frozen before results")
    resolved_config = copy.deepcopy(dict(config))
    resolved_config.update(
        {
            "resolved_output_dir": str(output_dir.resolve()),
            "created_at": datetime.now().isoformat(),
            "config_sha256_before_results": config_hash_before,
            "evaluator_sha256_before_results": evaluator_hash_before,
            "thresholds_frozen_before_results": thresholds_frozen,
            "python_executable": sys.executable,
            "torch_version": torch.__version__,
        }
    )
    _write_json(output_dir / "config.json", resolved_config)

    stage1_training_dir = REPO_ROOT / str(config["stage1_training_dir"])
    stage1_config = _load_json(stage1_training_dir / "config.json")
    checkpoint_path = REPO_ROOT / str(config["stage1_checkpoint"])
    if _sha256_file(checkpoint_path) != str(config["stage1_checkpoint_sha256"]):
        raise RuntimeError("formal Stage-I checkpoint hash mismatch")
    if _sha256_file(REPO_ROOT / str(config["sac_checkpoint"])) != str(
        config["sac_checkpoint_sha256"]
    ):
        raise RuntimeError("SAC checkpoint hash mismatch")
    protected_paths = {
        "stage1_trainer": REPO_ROOT / "planning/gat/stage1_training.py",
        "gat_selector": REPO_ROOT / "planning/gat/candidate_selector.py",
        "gat_layer": REPO_ROOT / "planning/gat/edge_enhanced_gat.py",
        "graph_builder": REPO_ROOT / "planning/heterogeneous_candidate_graph.py",
        "fp_shep": REPO_ROOT / "planning/policy_preview.py",
        "sac_actor": REPO_ROOT / "baseline/sac/net.py",
        "dmp": REPO_ROOT / "Controller/dmp_rl.py",
        "environment": REPO_ROOT / "Environment/multi_agent_dmp_env.py",
        "formal_gat_checkpoint": checkpoint_path,
        "sac_checkpoint": REPO_ROOT / str(config["sac_checkpoint"]),
        "stage1_training_history": stage1_training_dir / "training_history.csv",
        "stress_episode_results": REPO_ROOT
        / str(config["stress_dir"])
        / "episode_results.csv",
    }
    hashes_before = {name: _sha256_file(path) for name, path in protected_paths.items()}

    _, by_split, dataset_audit = _load_training_examples(config, stage1_config)
    device = resolve_device(stage1_config["training"]["device"])
    checkpoint_rows, checkpoint_index, formal_scores = _checkpoint_audit(
        config=config,
        stage1_config=stage1_config,
        by_split=by_split,
        device=device,
    )
    training_rows, curve, gap = _training_sufficiency(
        config=config, checkpoint_index=checkpoint_index
    )
    _write_csv(output_dir / "training_sufficiency.csv", training_rows)
    _write_csv(output_dir / "checkpoint_comparison.csv", checkpoint_rows)
    _write_json(output_dir / "learning_curve_diagnosis.json", curve)
    _write_json(output_dir / "train_val_test_gap.json", gap)

    formal_model = load_model_checkpoint(checkpoint_path, stage1_config, device)
    formal_model.eval()
    for parameter in formal_model.parameters():
        parameter.requires_grad_(False)
    stress_examples, class_rows, graph_rows, reproduction_rows, semantic = _stress_inputs(
        config=config,
        stage1_config=stage1_config,
        formal_model=formal_model,
        device=device,
    )
    stress_metrics, stress_scores = evaluate_model(
        formal_model,
        stress_examples,
        batch_size=int(stage1_config["training"]["batch_size"]),
        device=device,
    )
    stress_companion_examples = _companion_h6_examples(
        stress_examples,
        class_rows,
        temperature=float(config["soft_target_temperature"]),
    )
    selector_rows, metric_index = _selector_metrics_rows(
        train_test_examples=by_split["test"],
        test_gat_scores=formal_scores["test"],
        stress_examples=stress_examples,
        stress_companion_examples=stress_companion_examples,
        stress_gat_scores=stress_scores,
        stress_gat_loss=float(stress_metrics["loss"]),
    )
    _write_csv(output_dir / "stress_offline_ranking.csv", selector_rows)
    _write_csv(output_dir / "stress_h6_class_targets.csv", class_rows)
    _write_csv(output_dir / "stress_reproduction_audit.csv", reproduction_rows)

    alignment_rows, null_rows, override_rows, post_rows, alignment_summary = (
        _stress_outcome_audits(
            config=config,
            stress_examples=stress_examples,
            stress_scores=stress_scores,
            class_rows=class_rows,
        )
    )
    _write_csv(output_dir / "stress_target_alignment.csv", alignment_rows)
    _write_csv(output_dir / "null_audit.csv", null_rows)
    _write_csv(output_dir / "selector_override_analysis.csv", override_rows)
    _write_csv(output_dir / "post_reference_failure.csv", post_rows)

    stress_agent_rows = _read_csv(
        REPO_ROOT / str(config["stress_dir"]) / "per_agent_results.csv"
    )
    stress_agent_index = {
        (row["layout_id"], row["method"], int(row["agent_id"])): row
        for row in stress_agent_rows
    }
    null_distribution_rows = _null_distribution_rows(
        by_split=by_split,
        formal_scores=formal_scores,
        stress_examples=stress_examples,
        stress_scores=stress_scores,
        stress_agent_index=stress_agent_index,
    )
    _write_csv(output_dir / "null_distribution_shift.csv", null_distribution_rows)
    null_summary = _null_and_override_conclusions(
        config=config,
        null_rows=null_rows,
        null_distribution_rows=null_distribution_rows,
        override_rows=override_rows,
    )

    feature_rows, feature_summary = _feature_shift(
        train_examples=by_split["train"],
        stress_graph_rows=graph_rows,
        config=config,
    )
    _write_csv(output_dir / "feature_distribution_shift.csv", feature_rows)
    family_rows = _family_diagnosis(
        config=config,
        stress_graph_rows=graph_rows,
        train_feature_rows=feature_rows,
        override_rows=override_rows,
        post_rows=post_rows,
        null_rows=null_rows,
    )
    _write_csv(output_dir / "family_diagnosis.csv", family_rows)

    conclusion = _final_conclusion(
        config=config,
        curve=curve,
        metric_index=metric_index,
        feature_summary=feature_summary,
        alignment_summary=alignment_summary,
        null_summary=null_summary,
        semantic_audit=semantic,
    )
    conclusion["null_diagnosis"] = null_summary
    conclusion["alignment_diagnosis"] = alignment_summary
    conclusion["feature_shift"] = feature_summary
    conclusion["gate_semantic_audit"] = semantic
    conclusion["post_reference_failure_count"] = len(post_rows)
    conclusion["post_reference_failure_categories"] = dict(
        Counter(str(row["primary_category"]) for row in post_rows)
    )
    _write_json(output_dir / "conclusion.json", conclusion)
    (output_dir / "FINAL_REPORT.md").write_text(
        _report(
            conclusion=conclusion,
            curve=curve,
            metric_index=metric_index,
            feature_summary=feature_summary,
            alignment=alignment_summary,
            null_summary=null_summary,
            semantic=semantic,
        ),
        encoding="utf-8",
    )

    hashes_after = {name: _sha256_file(path) for name, path in protected_paths.items()}
    integrity = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASSED",
        "strict_read_only": True,
        "training_performed": False,
        "optimizer_created": False,
        "new_long_horizon_execution_performed": False,
        "short_h6_supervision_rollout_only": True,
        "counterfactual_long_horizon_execution_performed": False,
        "thresholds_frozen_before_results": thresholds_frozen,
        "config_unchanged_after_results": _sha256_file(config_path)
        == config_hash_before,
        "evaluator_unchanged_after_results": _sha256_file(evaluator_path)
        == evaluator_hash_before,
        "protected_hashes_before": hashes_before,
        "protected_hashes_after": hashes_after,
        "protected_inputs_unchanged": hashes_before == hashes_after,
        "dataset_graph_count": dataset_audit["graph_count"],
        "stress_layout_count": len({item.state_group_id for item in stress_examples}),
        "stress_ego_decision_count": len(stress_examples),
        "stress_h6_class_count": len(class_rows),
        "candidate_bundle_reproduction_match": semantic[
            "candidate_bundle_reproduction_match"
        ],
        "gat_selection_reproduction_match": semantic[
            "gat_selection_reproduction_match"
        ],
        "stress_source_environment_side_effect_free": semantic[
            "source_environment_side_effect_free"
        ],
        "runtime_seconds": time.perf_counter() - started,
        "STRESS_SET_STATUS": config["stress_set_status"],
    }
    if not all(
        (
            integrity["config_unchanged_after_results"],
            integrity["evaluator_unchanged_after_results"],
            integrity["protected_inputs_unchanged"],
            integrity["candidate_bundle_reproduction_match"],
            integrity["gat_selection_reproduction_match"],
            integrity["stress_source_environment_side_effect_free"],
        )
    ):
        integrity["status"] = "FAILED"
    _write_json(output_dir / "integrity_manifest.json", integrity)
    if integrity["status"] != "PASSED":
        raise RuntimeError("read-only diagnosis integrity gate failed")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = _load_json(config_path)
    if config_path != DEFAULT_CONFIG_PATH.resolve():
        raise ValueError("formal audit requires the frozen default config")
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = REPO_ROOT / str(config["output_root"]) / timestamp
    else:
        output_dir = args.output_dir.resolve()
    result = run_audit(config, output_dir)
    print(result, flush=True)


if __name__ == "__main__":
    main()
