"""Audit candidate-supervision labels and create traceable visual artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
for search_path in (REPO_ROOT, ALGO_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

# Reuse the project's established numerical-runtime initialization before
# importing Matplotlib in standalone audit runs.
import experiment_config as _experiment_config  # noqa: E402,F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from planning.candidate_supervision import pearson, spearman  # noqa: E402


plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Microsoft JhengHei", "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False
REAL_LINESTYLE = "-"
PREVIEW_LINESTYLE = "--"
CASE_COLORS = (
    "#00A6D6", "#F28E2B", "#59A14F", "#E15759", "#AF7AA1",
    "#76B7B2", "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
)


def _coerce(value: str) -> Any:
    if value == "":
        return None
    if value == "null":
        return "null"
    if value == "True":
        return True
    if value == "False":
        return False
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return [
            {key: _coerce(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(_jsonable(row.get(key)), ensure_ascii=False)
                if isinstance(row.get(key), (list, tuple, dict, np.ndarray))
                else _jsonable(row.get(key))
                for key in fields
            })


def _finite(values: Iterable[Any]) -> np.ndarray:
    output: list[float] = []
    for value in values:
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            output.append(number)
    return np.asarray(output, dtype=float)


def _mean(rows: Sequence[dict[str, Any]], key: str) -> float | None:
    values = _finite(row.get(key) for row in rows)
    return float(np.mean(values)) if values.size else None


def _median(rows: Sequence[dict[str, Any]], key: str) -> float | None:
    values = _finite(row.get(key) for row in rows)
    return float(np.median(values)) if values.size else None


def _normalize(values: Sequence[Any]) -> np.ndarray:
    array = np.asarray(
        [float(value) if value is not None and np.isfinite(float(value)) else np.nan for value in values],
        dtype=float,
    )
    finite = np.isfinite(array)
    result = np.full(array.shape, np.nan, dtype=float)
    if not np.any(finite):
        return result
    minimum = float(np.min(array[finite]))
    span = float(np.max(array[finite]) - minimum)
    result[finite] = 0.0 if span <= 1.0e-12 else (array[finite] - minimum) / span
    return result


def _scope_rows(rows: Sequence[dict[str, Any]], scope: str) -> list[dict[str, Any]]:
    if scope == "train_validation":
        return [row for row in rows if row["split"] in {"train", "validation"}]
    if scope == "test_heldout":
        return [row for row in rows if row["split"] == "test"]
    return [row for row in rows if row["split"] == scope]


def _group(
    rows: Sequence[dict[str, Any]],
    key: Callable[[dict[str, Any]], Any],
) -> dict[Any, list[dict[str, Any]]]:
    result: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        result[key(row)].append(row)
    return dict(result)


def _comparison_table(sample_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for scope in ("train", "validation", "train_validation", "test_heldout"):
        scoped = _scope_rows(sample_rows, scope)
        scenarios = ["overall"] + sorted({str(row["scenario"]) for row in scoped})
        for scenario in scenarios:
            selected = scoped if scenario == "overall" else [
                row for row in scoped if row["scenario"] == scenario
            ]
            for h_label in sorted({int(row["H_label"]) for row in selected}):
                h_rows = [row for row in selected if int(row["H_label"]) == h_label]
                for variant in (3, 4):
                    output.append({
                        "scope": scope,
                        "test_used_for_selection": False,
                        "scenario": scenario,
                        "H_label": h_label,
                        "target_variant": (
                            "provisional_primary_target" if variant == 3 else "companion_target"
                        ),
                        "sample_count": len(h_rows),
                        "proposal_spearman": _mean(h_rows, f"quality{variant}_proposal_spearman_proposals_only"),
                        "fp_shep_spearman": _mean(h_rows, f"quality{variant}_fp_shep_spearman_all_classes"),
                        "proposal_top1_hit": _mean(h_rows, f"quality{variant}_proposal_top1_hit_full_target"),
                        "fp_shep_top1_hit": _mean(h_rows, f"quality{variant}_fp_shep_top1_hit_full_target"),
                        "proposal_top3_hit": _mean(h_rows, f"quality{variant}_proposal_top3_oracle_recall_full_target"),
                        "fp_shep_top3_hit": _mean(h_rows, f"quality{variant}_fp_shep_top3_oracle_recall_full_target"),
                        "proposal_pairwise_accuracy": _mean(h_rows, f"quality{variant}_proposal_pairwise_accuracy_proposals_only"),
                        "fp_shep_pairwise_accuracy": _mean(h_rows, f"quality{variant}_fp_shep_pairwise_accuracy_all_classes"),
                    })
    return output


def _horizon_table(sample_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = _scope_rows(sample_rows, "train_validation")
    output: list[dict[str, Any]] = []
    for h_label in sorted({int(row["H_label"]) for row in selected}):
        rows = [row for row in selected if int(row["H_label"]) == h_label]
        output.append({
            "scope": "train_validation_only",
            "test_used_for_selection": False,
            "H_label": h_label,
            "sample_count": len(rows),
            "candidate_discriminability_quality_std_3": _mean(rows, "quality3_candidate_quality_std"),
            "candidate_discriminability_quality_std_4": _mean(rows, "quality4_candidate_quality_std"),
            "proposal_spearman_3": _mean(rows, "quality3_proposal_spearman_proposals_only"),
            "fp_shep_spearman_3": _mean(rows, "quality3_fp_shep_spearman_all_classes"),
            "diagnostic_equal_horizon_spearman_3": _mean(rows, "diagnostic_quality3_spearman_all_classes"),
            "proposal_top1_3": _mean(rows, "quality3_proposal_top1_hit_full_target"),
            "fp_shep_top1_3": _mean(rows, "quality3_fp_shep_top1_hit_full_target"),
            "fp_shep_pairwise_3": _mean(rows, "quality3_fp_shep_pairwise_accuracy_all_classes"),
            "collision_rate": _mean(rows, "collision_branch_ratio"),
            "truncated_rate": _mean(rows, "truncated_branch_ratio"),
            "null_top1_rate_3": _mean(rows, "null_top1_3"),
            "null_top1_rate_4": _mean(rows, "null_top1_4"),
            "mean_runtime_per_candidate_ms": _mean(rows, "mean_real_runtime_per_candidate_ms"),
            "median_runtime_per_candidate_ms": _median(rows, "median_real_runtime_per_candidate_ms"),
            "mean_preview_real_position_error": _mean(rows, "mean_preview_real_position_error"),
        })
    return output


def _feature_correlation_table(class_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    feature_pairs = (
        ("task_progress", "formal_preview_task_progress", "real_task_progress"),
        ("min_clearance", "formal_preview_min_clearance", "real_min_clearance"),
        ("max_execution_deviation", "formal_preview_max_execution_deviation", "real_max_execution_deviation"),
        ("terminal_speed", "formal_preview_terminal_speed", "real_terminal_speed"),
    )
    for scope in ("train_validation", "test_heldout"):
        scoped = _scope_rows(class_rows, scope)
        scenarios = ["overall"] + sorted({str(row["scenario"]) for row in scoped})
        for scenario in scenarios:
            scenario_rows = scoped if scenario == "overall" else [
                row for row in scoped if row["scenario"] == scenario
            ]
            for h_label in sorted({int(row["H_label"]) for row in scenario_rows}):
                rows = [row for row in scenario_rows if int(row["H_label"]) == h_label]
                for feature, preview_key, real_key in feature_pairs:
                    pairs = [
                        (float(row[preview_key]), float(row[real_key]))
                        for row in rows
                        if row.get(preview_key) is not None and row.get(real_key) is not None
                        and np.isfinite(float(row[preview_key])) and np.isfinite(float(row[real_key]))
                    ]
                    left = [item[0] for item in pairs]
                    right = [item[1] for item in pairs]
                    clearance_semantics_match = feature != "min_clearance"
                    output.append({
                        "scope": scope,
                        "test_used_for_selection": False,
                        "scenario": scenario,
                        "H_label": h_label,
                        "feature": feature,
                        "sample_count": len(pairs),
                        "pearson": pearson(left, right),
                        "spearman": spearman(left, right),
                        "mean_absolute_error": (
                            float(np.mean(np.abs(np.asarray(left) - np.asarray(right))))
                            if pairs and clearance_semantics_match else None
                        ),
                        "absolute_error_interpretation": (
                            "strict_same_metric_definition"
                            if clearance_semantics_match
                            else "not_reported_preview_and_real_clearance_sources_differ"
                        ),
                        "preview_source": "formal_H_preview_4",
                        "real_source": (
                            "real_environment_sensor_packet"
                            if feature == "min_clearance" else "real_execution_state"
                        ),
                    })
    return output


def _distribution_tables(
    sample_rows: list[dict[str, Any]],
    class_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_samples = _scope_rows(sample_rows, "train_validation")
    selected_classes = _scope_rows(class_rows, "train_validation")
    top_counts: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    scenarios = ["overall"] + sorted({str(row["scenario"]) for row in selected_samples})
    for scenario in scenarios:
        srows = selected_samples if scenario == "overall" else [
            row for row in selected_samples if row["scenario"] == scenario
        ]
        crows = selected_classes if scenario == "overall" else [
            row for row in selected_classes if row["scenario"] == scenario
        ]
        for h_label in sorted({int(row["H_label"]) for row in srows}):
            h_srows = [row for row in srows if int(row["H_label"]) == h_label]
            h_crows = [row for row in crows if int(row["H_label"]) == h_label]
            labels = [
                "null" if int(row["hard_target_3_class_index"]) == 0
                else f"proposal_rank_{int(row['hard_target_3_class_index']) - 1}"
                for row in h_srows
            ]
            for label in sorted(set(labels)):
                count = labels.count(label)
                top_counts.append({
                    "scope": "train_validation_only", "scenario": scenario,
                    "H_label": h_label, "target_class": label, "count": count,
                    "ratio": float(count / len(labels)) if labels else 0.0,
                })
            total = len(h_crows)
            for status, predicate in (
                ("invalid", lambda row: not bool(np.all(row.get("real_feature_valid_mask", [])))),
                ("collision", lambda row: bool(row.get("collision"))),
                ("success", lambda row: bool(row.get("success"))),
                ("truncated", lambda row: bool(row.get("truncated"))),
                ("completed_horizon", lambda row: bool(row.get("completed_label_horizon"))),
            ):
                count = sum(predicate(row) for row in h_crows)
                status_rows.append({
                    "scope": "train_validation_only", "scenario": scenario,
                    "H_label": h_label, "status": status, "count": count,
                    "ratio": float(count / total) if total else 0.0,
                })
    return top_counts, status_rows


def _divergence_table(class_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = _scope_rows(class_rows, "train_validation")
    buckets: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in selected:
        steps = row.get("preview_real_error_steps") or []
        errors = row.get("preview_real_position_errors") or []
        for step, error in zip(steps, errors):
            buckets[("overall", int(row["H_label"]), int(step))].append(float(error))
            buckets[(str(row["scenario"]), int(row["H_label"]), int(step))].append(float(error))
    return [
        {
            "scope": "train_validation_only",
            "scenario": scenario,
            "H_label": h_label,
            "preview_step": step,
            "sample_count": len(values),
            "mean_position_error": float(np.mean(values)),
            "median_position_error": float(np.median(values)),
            "p90_position_error": float(np.quantile(values, 0.9)),
        }
        for (scenario, h_label, step), values in sorted(buckets.items())
    ]


def _tau_sensitivity_table(
    sample_rows: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    temperatures = [float(value) for value in config["soft_target_temperatures"]]
    for scope in ("train", "validation", "train_validation", "test_heldout"):
        scoped = _scope_rows(sample_rows, scope)
        for h_label in sorted({int(row["H_label"]) for row in scoped}):
            rows = [row for row in scoped if int(row["H_label"]) == h_label]
            for temperature in temperatures:
                safe_key = f"tau_{temperature:g}".replace(".", "p")
                for variant in ("primary", "companion"):
                    prefix = f"soft_{variant}_{safe_key}_"
                    output.append({
                        "scope": scope,
                        "test_used_for_selection": False,
                        "H_label": h_label,
                        "temperature": temperature,
                        "target_variant": (
                            "provisional_primary_target" if variant == "primary"
                            else "companion_target"
                        ),
                        "sample_count": len(rows),
                        "mean_entropy": _mean(rows, prefix + "entropy"),
                        "mean_normalized_entropy": _mean(rows, prefix + "normalized_entropy"),
                        "mean_effective_candidate_count": _mean(rows, prefix + "effective_candidate_count"),
                        "mean_maximum_probability": _mean(rows, prefix + "maximum_probability"),
                    })
    return output


def _dataset_composition_table(graph_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for scope in ("train", "validation", "train_validation", "test_heldout"):
        scoped = _scope_rows(graph_rows, scope)
        scenarios = ["overall"] + sorted({str(row["scenario"]) for row in scoped})
        for scenario in scenarios:
            rows = scoped if scenario == "overall" else [
                row for row in scoped if row["scenario"] == scenario
            ]
            k_values = np.asarray([int(row["K_actual"]) for row in rows], dtype=int)
            output.append({
                "scope": scope,
                "test_used_for_selection": False,
                "scenario": scenario,
                "ego_graph_count": len(rows),
                "state_group_count": len({str(row["state_group_id"]) for row in rows}),
                "mean_K_actual": float(np.mean(k_values)) if k_values.size else None,
                "median_K_actual": float(np.median(k_values)) if k_values.size else None,
                "minimum_K_actual": int(np.min(k_values)) if k_values.size else None,
                "maximum_K_actual": int(np.max(k_values)) if k_values.size else None,
                "K_zero_ratio": float(np.mean(k_values == 0)) if k_values.size else None,
                "K_less_than_requested_ratio": float(np.mean(k_values < 10)) if k_values.size else None,
                "K_equal_requested_ratio": float(np.mean(k_values == 10)) if k_values.size else None,
                "padding_or_duplicate_candidates_added": False,
            })
    return output


def _plot_comparison(path: Path, rows: list[dict[str, Any]]) -> None:
    selected = [
        row for row in rows
        if row["scope"] == "train_validation" and row["scenario"] == "overall"
        and row["target_variant"] == "provisional_primary_target"
    ]
    if not selected:
        return
    metrics = (
        ("Spearman", "proposal_spearman", "fp_shep_spearman"),
        ("Top-1 hit", "proposal_top1_hit", "fp_shep_top1_hit"),
        ("Top-3 hit", "proposal_top3_hit", "fp_shep_top3_hit"),
        ("Pairwise", "proposal_pairwise_accuracy", "fp_shep_pairwise_accuracy"),
    )
    fig, axes = plt.subplots(1, len(selected), figsize=(5.0 * len(selected), 4.4), squeeze=False)
    for axis, row in zip(axes[0], sorted(selected, key=lambda item: int(item["H_label"]))):
        x = np.arange(len(metrics))
        width = 0.36
        proposal = [row[left] if row[left] is not None else np.nan for _, left, _ in metrics]
        preview = [row[right] if row[right] is not None else np.nan for _, _, right in metrics]
        axis.bar(x - width / 2, proposal, width, label="Proposal.score", color="#9C755F")
        axis.bar(x + width / 2, preview, width, label="FP-SHEP H=4", color="#00A6D6")
        axis.set_xticks(x, [item[0] for item in metrics], rotation=20)
        axis.set_ylim(-1.0, 1.05)
        axis.set_title(f"H_label={row['H_label']}")
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].set_ylabel("metric")
    axes[0, 0].legend()
    fig.suptitle("Proposal vs operational FP-SHEP relative to real target\n(train+validation only)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_comparison_by_scenario(directory: Path, rows: list[dict[str, Any]]) -> None:
    for h_label in sorted({
        int(row["H_label"]) for row in rows
        if row["scope"] == "train_validation"
    }):
        selected = [
            row for row in rows
            if row["scope"] == "train_validation"
            and row["scenario"] != "overall"
            and int(row["H_label"]) == h_label
            and row["target_variant"] == "provisional_primary_target"
        ]
        if not selected:
            continue
        selected = sorted(selected, key=lambda row: str(row["scenario"]))
        scenarios = [str(row["scenario"]) for row in selected]
        metrics = (
            ("Spearman", "proposal_spearman", "fp_shep_spearman"),
            ("Top-1 hit", "proposal_top1_hit", "fp_shep_top1_hit"),
            ("Top-3 hit", "proposal_top3_hit", "fp_shep_top3_hit"),
            ("Pairwise", "proposal_pairwise_accuracy", "fp_shep_pairwise_accuracy"),
        )
        figure, axes = plt.subplots(2, 2, figsize=(14.0, 8.0))
        x = np.arange(len(scenarios))
        width = 0.36
        for axis, (title, proposal_key, preview_key) in zip(axes.flat, metrics):
            axis.bar(
                x - width / 2,
                [row[proposal_key] for row in selected],
                width, label="Proposal.score", color="#9C755F",
            )
            axis.bar(
                x + width / 2,
                [row[preview_key] for row in selected],
                width, label="FP-SHEP H=4", color="#00A6D6",
            )
            axis.set_xticks(x, scenarios, rotation=25, ha="right")
            axis.set_title(title)
            axis.set_ylim(-1.0 if title == "Spearman" else 0.0, 1.05)
            axis.grid(axis="y", alpha=0.25)
        axes[0, 0].legend()
        figure.suptitle(
            f"Proposal vs FP-SHEP by scenario, H_label={h_label}\n"
            "train+validation only"
        )
        figure.tight_layout()
        figure.savefig(directory / f"proposal_vs_fp_by_scenario_H{h_label}.png", dpi=180)
        plt.close(figure)


def _plot_horizon(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    rows = sorted(rows, key=lambda row: int(row["H_label"]))
    h = [int(row["H_label"]) for row in rows]
    panels = (
        ("Discriminability", "candidate_discriminability_quality_std_3"),
        ("Proposal Spearman", "proposal_spearman_3"),
        ("FP-SHEP Spearman", "fp_shep_spearman_3"),
        ("Collision rate", "collision_rate"),
        ("Truncated rate", "truncated_rate"),
        ("Null top-1 rate", "null_top1_rate_3"),
        ("Mean runtime/candidate", "mean_runtime_per_candidate_ms"),
        ("Median runtime/candidate", "median_runtime_per_candidate_ms"),
    )
    fig, axes = plt.subplots(2, 4, figsize=(16.0, 7.4))
    for axis, (title, key) in zip(axes.flat, panels):
        values = [float(row[key]) if row[key] is not None else np.nan for row in rows]
        axis.plot(h, values, marker="o", color="#00A6D6")
        axis.set_title(title)
        axis.set_xlabel("H_label")
        axis.grid(alpha=0.25)
    fig.suptitle("H_label sensitivity (train+validation only; no test selection)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_label_distributions(
    directory: Path,
    sample_rows: list[dict[str, Any]],
    top_counts: list[dict[str, Any]],
    status_rows: list[dict[str, Any]],
    default_temperature: float,
) -> None:
    selected = _scope_rows(sample_rows, "train_validation")
    if not selected:
        return
    overall_top = [row for row in top_counts if row["scenario"] == "overall"]
    labels = sorted({str(row["target_class"]) for row in overall_top})
    horizons = sorted({int(row["H_label"]) for row in overall_top})
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0))
    width = 0.8 / max(1, len(horizons))
    x = np.arange(len(labels))
    for index, h_label in enumerate(horizons):
        by_label = {
            row["target_class"]: row["ratio"]
            for row in overall_top if int(row["H_label"]) == h_label
        }
        axes[0, 0].bar(
            x + (index - (len(horizons) - 1) / 2) * width,
            [by_label.get(label, 0.0) for label in labels],
            width,
            label=f"H={h_label}",
        )
    axes[0, 0].set_xticks(x, labels, rotation=45, ha="right")
    axes[0, 0].set_title("Target top-1 class distribution")
    axes[0, 0].legend()
    for h_label in horizons:
        h_rows = [row for row in selected if int(row["H_label"]) == h_label]
        safe_key = f"tau_{float(default_temperature):g}".replace(".", "p")
        entropy = _finite(
            row.get(f"soft_primary_{safe_key}_entropy") for row in h_rows
        )
        gap = _finite(row.get("quality3_top1_top2_target_gap") for row in h_rows)
        std = _finite(row.get("quality3_candidate_quality_std") for row in h_rows)
        if entropy.size:
            axes[0, 1].hist(entropy, bins=20, alpha=0.45, label=f"H={h_label}")
        if gap.size:
            axes[1, 0].hist(gap, bins=20, alpha=0.45, label=f"H={h_label}")
        if std.size:
            axes[1, 1].hist(std, bins=20, alpha=0.45, label=f"H={h_label}")
    axes[0, 1].set_title("Soft-target entropy")
    axes[1, 0].set_title("Top1-Top2 J_target gap")
    axes[1, 1].set_title("Candidate quality std")
    for axis in axes.flat[1:]:
        axis.legend()
        axis.grid(alpha=0.2)
    fig.suptitle("Supervision label distributions (train+validation only)")
    fig.tight_layout()
    fig.savefig(directory / "label_distribution_overall.png", dpi=180)
    plt.close(fig)

    statuses = sorted({str(row["status"]) for row in status_rows})
    scenes = sorted({str(row["scenario"]) for row in status_rows if row["scenario"] != "overall"})
    for h_label in horizons:
        fig, axis = plt.subplots(figsize=(max(9.0, len(scenes) * 1.5), 4.8))
        width = 0.8 / max(1, len(statuses))
        x = np.arange(len(scenes))
        for status_index, status in enumerate(statuses):
            values = []
            for scene in scenes:
                match = next((
                    row for row in status_rows
                    if row["scenario"] == scene and int(row["H_label"]) == h_label
                    and row["status"] == status
                ), None)
                values.append(0.0 if match is None else float(match["ratio"]))
            axis.bar(
                x + (status_index - (len(statuses) - 1) / 2) * width,
                values,
                width,
                label=status,
            )
        axis.set_title(f"Branch status ratios by scenario, H_label={h_label}")
        axis.set_xticks(x, scenes, rotation=25)
        axis.set_ylim(0.0, 1.05)
        axis.legend(ncol=3)
        fig.tight_layout()
        fig.savefig(directory / f"branch_status_by_scenario_H{h_label}.png", dpi=180)
        plt.close(fig)

    scenarios = sorted({str(row["scenario"]) for row in selected})
    figure, axes = plt.subplots(2, 2, figsize=(14.0, 8.0))
    x = np.arange(len(scenarios))
    width = 0.8 / max(1, len(horizons))
    metric_panels = (
        ("Null top-1 rate", lambda rows: _mean(rows, "null_top1_3")),
        (
            f"Soft-target entropy (tau={default_temperature:g})",
            lambda rows: _mean(
                rows,
                f"soft_primary_{f'tau_{float(default_temperature):g}'.replace('.', 'p')}_entropy",
            ),
        ),
        ("Top1-Top2 target gap", lambda rows: _mean(rows, "quality3_top1_top2_target_gap")),
        ("Candidate quality std", lambda rows: _mean(rows, "quality3_candidate_quality_std")),
    )
    for axis, (title, reducer) in zip(axes.flat, metric_panels):
        for horizon_index, h_label in enumerate(horizons):
            values = []
            for scenario in scenarios:
                rows = [
                    row for row in selected
                    if row["scenario"] == scenario and int(row["H_label"]) == h_label
                ]
                value = reducer(rows)
                values.append(np.nan if value is None else value)
            axis.bar(
                x + (horizon_index - (len(horizons) - 1) / 2) * width,
                values,
                width,
                label=f"H={h_label}",
            )
        axis.set_xticks(x, scenarios, rotation=25, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend()
    figure.suptitle("Label distribution diagnostics by scenario (train+validation only)")
    figure.tight_layout()
    figure.savefig(directory / "label_distribution_by_scenario.png", dpi=180)
    plt.close(figure)


def _plot_tau_sensitivity(path: Path, rows: list[dict[str, Any]]) -> None:
    selected = [
        row for row in rows
        if row["scope"] == "train_validation"
        and row["target_variant"] == "provisional_primary_target"
    ]
    if not selected:
        return
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    for h_label in sorted({int(row["H_label"]) for row in selected}):
        h_rows = sorted(
            [row for row in selected if int(row["H_label"]) == h_label],
            key=lambda row: float(row["temperature"]),
        )
        temperatures = [row["temperature"] for row in h_rows]
        axes[0].plot(
            temperatures, [row["mean_normalized_entropy"] for row in h_rows],
            marker="o", label=f"H={h_label}",
        )
        axes[1].plot(
            temperatures, [row["mean_effective_candidate_count"] for row in h_rows],
            marker="o", label=f"H={h_label}",
        )
    axes[0].set_title("Normalized soft-target entropy")
    axes[1].set_title("Effective candidate count")
    for axis in axes:
        axis.set_xlabel("tau_label")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Tau sensitivity (train+validation only)")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_divergence(directory: Path, rows: list[dict[str, Any]]) -> None:
    overall = [row for row in rows if row["scenario"] == "overall"]
    if not overall:
        return
    fig, axis = plt.subplots(figsize=(8.2, 5.0))
    for h_label in sorted({int(row["H_label"]) for row in overall}):
        h_rows = sorted(
            [row for row in overall if int(row["H_label"]) == h_label],
            key=lambda row: int(row["preview_step"]),
        )
        axis.plot(
            [row["preview_step"] for row in h_rows],
            [row["mean_position_error"] for row in h_rows],
            marker="o", label=f"H_label={h_label}",
        )
    axis.set_xlabel("preview/real step")
    axis.set_ylabel("mean position error (m)")
    axis.set_title("Equal-horizon diagnostic preview-real divergence")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(directory / "divergence_by_step_and_horizon.png", dpi=180)
    plt.close(fig)

    final_rows = []
    for (scenario, h_label), group in _group(
        [row for row in rows if row["scenario"] != "overall"],
        lambda row: (row["scenario"], int(row["H_label"])),
    ).items():
        last = max(group, key=lambda row: int(row["preview_step"]))
        final_rows.append({**last, "scenario": scenario, "H_label": h_label})
    scenarios = sorted({str(row["scenario"]) for row in final_rows})
    horizons = sorted({int(row["H_label"]) for row in final_rows})
    matrix = np.full((len(scenarios), len(horizons)), np.nan)
    for row in final_rows:
        matrix[scenarios.index(str(row["scenario"])), horizons.index(int(row["H_label"]))] = float(row["mean_position_error"])
    fig, axis = plt.subplots(figsize=(7.0, max(4.5, len(scenarios) * 0.6)))
    image = axis.imshow(matrix, aspect="auto", cmap="magma")
    axis.set_xticks(range(len(horizons)), [f"H={item}" for item in horizons])
    axis.set_yticks(range(len(scenarios)), scenarios)
    axis.set_title("Terminal preview-real error by scenario")
    fig.colorbar(image, ax=axis, label="mean error (m)")
    fig.tight_layout()
    fig.savefig(directory / "divergence_by_scenario.png", dpi=180)
    plt.close(fig)


def _candidate_rows_for_sample(
    rows: list[dict[str, Any]], sample_id: str, h_label: int
) -> list[dict[str, Any]]:
    return sorted(
        [
            row for row in rows
            if row["sample_id"] == sample_id and int(row["H_label"]) == int(h_label)
        ],
        key=lambda row: int(row["class_index"]),
    )


def _case_selectors(
    sample_rows: list[dict[str, Any]],
    class_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, dict[str, Any] | None]:
    rows = _scope_rows(sample_rows, "train_validation")
    by_key = {
        (str(row["sample_id"]), int(row["H_label"])): _candidate_rows_for_sample(
            class_rows, str(row["sample_id"]), int(row["H_label"])
        )
        for row in rows
    }
    def prop(row: dict[str, Any]) -> int | None:
        value = row.get("quality3_proposal_top1_class_index")
        return None if value is None else int(value)
    def fp(row: dict[str, Any]) -> int:
        return int(row["quality3_fp_shep_top1_class_index"])
    def oracle(row: dict[str, Any]) -> int:
        return int(row["quality3_oracle_top1_class_index"])
    def first(predicate: Callable[[dict[str, Any]], bool], *, score=None):
        candidates = [row for row in rows if predicate(row)]
        if not candidates:
            return None
        return max(candidates, key=score) if score is not None else candidates[0]
    proposal_fail = first(
        lambda row: prop(row) != oracle(row) and fp(row) == oracle(row),
        score=lambda row: float(row.get("quality3_top1_top2_target_gap") or 0.0),
    )
    proposal_agree = first(lambda row: prop(row) == oracle(row) and prop(row) is not None)
    null_top1 = first(lambda row: oracle(row) == 0)
    gap_threshold = float(config.get("proposal_better_than_null_gap_threshold", 0.25))
    proposal_over_null = first(
        lambda row: oracle(row) != 0 and (
            max(float(item["J_target_3"]) for item in by_key[(str(row["sample_id"]), int(row["H_label"]))] if int(item["class_index"]) != 0)
            - float(by_key[(str(row["sample_id"]), int(row["H_label"]))][0]["J_target_3"])
        ) >= gap_threshold
    )
    near_threshold = float(config.get("near_optimal_gap_threshold", 0.1))
    near_optimal = first(
        lambda row: float(row.get("quality3_top1_top2_target_gap") or float("inf")) <= near_threshold,
        score=lambda row: -float(row.get("quality3_top1_top2_target_gap") or 0.0),
    )
    def collision_downgrade_strength(row: dict[str, Any]) -> float:
        items = by_key[(str(row["sample_id"]), int(row["H_label"]))]
        failures = [item for item in items if bool(item.get("failure_tier"))]
        safe = [item for item in items if not bool(item.get("failure_tier"))]
        if not failures or not safe:
            return float("-inf")
        nominal_advantage = (
            max(float(item["real_quality_3_nominal"]) for item in failures)
            - max(float(item["real_quality_3_nominal"]) for item in safe)
        )
        failure_is_below_all_safe = (
            max(float(item["J_target_3"]) for item in failures)
            < min(float(item["J_target_3"]) for item in safe)
        )
        return nominal_advantage if failure_is_below_all_safe else float("-inf")
    collision_downgraded = first(
        lambda row: collision_downgrade_strength(row) > 0.0,
        score=collision_downgrade_strength,
    )
    multi_agent = first(
        lambda row: str(row["scenario"]) in {"multi_agent", "narrow_head_on"}
        and (prop(row) != oracle(row) or fp(row) != oracle(row))
        and (
            int(row.get("graph_spatiotemporal_edge_count") or 0) > 0
            or any(bool(item.get("inter_agent_collision")) for item in by_key[(str(row["sample_id"]), int(row["H_label"]))])
        ),
        score=lambda row: float(row.get("quality3_top1_top2_target_gap") or 0.0),
    )
    divergence = first(
        lambda row: True,
        score=lambda row: float(row.get("mean_preview_real_position_error") or 0.0),
    )
    return {
        "A_proposal_fail_execution_aware_success": proposal_fail,
        "B_proposal_target_agreement": proposal_agree,
        "C_null_top1": null_top1,
        "D_proposal_better_than_null": proposal_over_null,
        "E_near_optimal_soft_label": near_optimal,
        "F_collision_downgraded": collision_downgraded,
        "G_multi_agent_interaction": multi_agent,
        "H_preview_real_divergence": divergence,
    }


def _case_artifact(
    output_dir: Path,
    category: str,
    sample: dict[str, Any],
    class_rows: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    sample_id = str(sample["sample_id"])
    h_label = int(sample["H_label"])
    rows = _candidate_rows_for_sample(class_rows, sample_id, h_label)
    manifest = next(
        row for row in manifest_rows
        if row["sample_id"] == sample_id and int(row["H_label"]) == h_label
    )
    data_path = output_dir / str(manifest["rollout_path"])
    data = np.load(data_path, allow_pickle=False)
    case_dir = output_dir / "representative_cases" / category
    case_dir.mkdir(parents=True, exist_ok=True)
    _write_json(case_dir / "case.json", {
        "category": category,
        "sample": sample,
        "manifest": manifest,
        "candidates": rows,
        "visualization_semantics": {
            "real_rollout": "solid_line",
            "FP_SHEP_preview": "dashed_line",
            "same_candidate_same_color": True,
            "diagnostic_preview_is_offline_only": True,
        },
    })
    _write_csv(case_dir / "candidate_table.csv", rows)
    proposal_top = sample.get("quality3_proposal_top1_class_index")
    fp_top = int(sample["quality3_fp_shep_top1_class_index"])
    oracle_top = int(sample["quality3_oracle_top1_class_index"])
    highlighted = {0, fp_top, oracle_top}
    if proposal_top is not None:
        highlighted.add(int(proposal_top))
    figure = plt.figure(figsize=(10.0, 7.2))
    axis = figure.add_subplot(111, projection="3d")
    goals = np.asarray(data["candidate_goals"], dtype=float)
    visible = np.asarray(data["visible_surface_points"], dtype=float)
    if visible.size:
        axis.scatter(*visible.T, s=5, c="#888888", alpha=0.25, label="visible LiDAR surfaces")
    axis.scatter(*np.asarray(data["ego_initial_position"], dtype=float), s=80, c="black", marker="o", label="ego initial")
    axis.scatter(*np.asarray(data["task_goal"], dtype=float), s=130, c="#D62728", marker="*", label="task goal / null ref")
    proposal_goals = goals[1:]
    if proposal_goals.size:
        axis.scatter(*proposal_goals.T, s=28, c="#BBBBBB", marker="^", label="proposal candidates")
    for class_index in sorted(highlighted):
        if class_index >= goals.shape[0]:
            continue
        color = CASE_COLORS[class_index % len(CASE_COLORS)]
        real_mask = np.asarray(data["real_position_mask"][class_index], dtype=bool)
        diagnostic_mask = np.asarray(data["diagnostic_preview_position_mask"][class_index], dtype=bool)
        real = np.asarray(data["real_positions"][class_index], dtype=float)[real_mask]
        preview = np.asarray(data["diagnostic_preview_positions"][class_index], dtype=float)[diagnostic_mask]
        label = "null" if class_index == 0 else f"candidate {class_index - 1}"
        axis.plot(*real.T, color=color, linestyle=REAL_LINESTYLE, linewidth=2.3, label=f"{label} real")
        axis.plot(*preview.T, color=color, linestyle=PREVIEW_LINESTYLE, linewidth=1.8, label=f"{label} FP-SHEP diagnostic")
        axis.scatter(*goals[class_index], color=color, marker="x", s=70)
    axis.set_title(
        f"{category}\n{sample_id}, H_label={h_label} | real solid, preview dashed"
    )
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_zlabel("z (m)")
    axis.legend(fontsize=8, loc="best")
    figure.tight_layout()
    trajectory_path = case_dir / "trajectory_comparison.png"
    figure.savefig(trajectory_path, dpi=190)
    plt.close(figure)

    labels = ["null"] + [f"candidate {index}" for index in range(len(rows) - 1)]
    proposal = _normalize([row.get("proposal_score") for row in rows])
    preview = _normalize([row["formal_J_preview_3"] for row in rows])
    target = _normalize([row["J_target_3"] for row in rows])
    figure, axis = plt.subplots(figsize=(max(9.0, len(rows) * 0.9), 5.0))
    x = np.arange(len(rows))
    width = 0.25
    proposal_plot = np.nan_to_num(proposal, nan=0.0)
    bars = axis.bar(x - width, proposal_plot, width, label="Proposal.score", color="#9C755F")
    if len(bars):
        bars[0].set_hatch("//")
        bars[0].set_alpha(0.25)
        axis.text(x[0] - width, 0.03, "N/A", ha="center", va="bottom", fontsize=8)
    axis.bar(x, preview, width, label="FP-SHEP quality H=4", color="#00A6D6")
    axis.bar(x + width, target, width, label="real J_target", color="#59A14F")
    axis.set_xticks(x, labels, rotation=30, ha="right")
    axis.set_ylim(0.0, 1.08)
    axis.set_title("Per-state normalized candidate ranking")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    ranking_path = case_dir / "ranking_comparison.png"
    figure.savefig(ranking_path, dpi=190)
    plt.close(figure)
    return {
        "category": category,
        "sample_id": sample_id,
        "H_label": h_label,
        "scenario": sample["scenario"],
        "trajectory_figure": trajectory_path.relative_to(output_dir).as_posix(),
        "ranking_figure": ranking_path.relative_to(output_dir).as_posix(),
        "underlying_json": (case_dir / "case.json").relative_to(output_dir).as_posix(),
        "underlying_csv": (case_dir / "candidate_table.csv").relative_to(output_dir).as_posix(),
    }


def audit_candidate_supervision_artifacts(output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    sample_rows = _read_csv(output_dir / "sample_records.csv")
    class_rows = _read_csv(output_dir / "candidate_class_records.csv")
    manifest_rows = _read_csv(output_dir / "manifest.csv")
    graph_rows = _read_csv(output_dir / "graph_records.csv")
    if not sample_rows or not class_rows:
        raise ValueError("candidate-supervision artifact tables are empty")
    comparison = _comparison_table(sample_rows)
    horizons = _horizon_table(sample_rows)
    correlations = _feature_correlation_table(class_rows)
    top_counts, status_rows = _distribution_tables(sample_rows, class_rows)
    divergence = _divergence_table(class_rows)
    tau_sensitivity = _tau_sensitivity_table(sample_rows, config)
    composition = _dataset_composition_table(graph_rows)
    _write_csv(output_dir / "summary" / "proposal_fp_target_comparison.csv", comparison)
    _write_csv(output_dir / "horizon_sensitivity" / "h_label_sensitivity.csv", horizons)
    _write_csv(output_dir / "summary" / "feature_outcome_correlations.csv", correlations)
    _write_csv(output_dir / "label_distribution" / "target_top1_distribution.csv", top_counts)
    _write_csv(output_dir / "label_distribution" / "branch_status_ratios.csv", status_rows)
    _write_csv(output_dir / "trajectory_comparison" / "preview_real_divergence.csv", divergence)
    _write_csv(output_dir / "label_distribution" / "tau_sensitivity.csv", tau_sensitivity)
    _write_csv(output_dir / "summary" / "dataset_composition.csv", composition)
    _plot_comparison(output_dir / "summary" / "proposal_vs_fp_vs_target.png", comparison)
    _plot_comparison_by_scenario(output_dir / "summary", comparison)
    _plot_horizon(output_dir / "horizon_sensitivity" / "h_label_sensitivity.png", horizons)
    _plot_label_distributions(
        output_dir / "label_distribution",
        sample_rows,
        top_counts,
        status_rows,
        float(config.get("default_temperature_for_distribution_plots_only", 0.25)),
    )
    _plot_tau_sensitivity(
        output_dir / "label_distribution" / "tau_sensitivity.png",
        tau_sensitivity,
    )
    _plot_divergence(output_dir / "trajectory_comparison", divergence)
    selected_cases = _case_selectors(sample_rows, class_rows, config)
    case_artifacts: list[dict[str, Any]] = []
    missing_cases: list[str] = []
    for category, sample in selected_cases.items():
        if sample is None:
            missing_cases.append(category)
            continue
        case_artifacts.append(
            _case_artifact(output_dir, category, sample, class_rows, manifest_rows)
        )
    _write_json(output_dir / "representative_cases" / "case_index.json", {
        "selection_scope": "train_validation_only",
        "test_used_for_case_selection": False,
        "cases": case_artifacts,
        "missing_case_categories": missing_cases,
    })
    selection_rows = _scope_rows(sample_rows, "train_validation")
    test_rows = _scope_rows(sample_rows, "test_heldout")
    summary = {
        "selection_scope": "train_validation_only",
        "test_used_for_H_label_quality_or_tau_selection": False,
        "test_rows_generated_as_heldout": len(test_rows),
        "train_validation_rows": len(selection_rows),
        "formal_Graph_Builder_H_preview": 4,
        "diagnostic_preview_at_H_label_offline_only": True,
        "provisional_primary_target": "3-feature quality",
        "companion_target": "4-feature quality",
        "final_supervision_target_frozen": False,
        "representative_case_count": len(case_artifacts),
        "missing_representative_case_categories": missing_cases,
        "H_label_statistics": horizons,
        "dataset_composition_train_validation": [
            row for row in composition
            if row["scope"] == "train_validation" and row["scenario"] == "overall"
        ],
        "comparison_train_validation": [
            row for row in comparison
            if row["scope"] == "train_validation" and row["scenario"] == "overall"
        ],
        "heldout_test_statistics_not_used_for_selection": [
            row for row in comparison
            if row["scope"] == "test_heldout" and row["scenario"] == "overall"
        ],
        "GAT_training_started": False,
    }
    _write_json(output_dir / "summary" / "audit_summary.json", summary)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = audit_candidate_supervision_artifacts(args.artifact_dir)
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
