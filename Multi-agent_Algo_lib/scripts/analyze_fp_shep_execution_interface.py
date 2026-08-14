"""Audit H=4 FP-SHEP raw execution features for graph-stage normalization.

The input validation dataset is read-only.  This script never rewrites raw
candidate records and does not fit normalization parameters from the dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from planning.candidate_execution_interface import (  # noqa: E402
    EXECUTION_FEATURE_ORDER,
    ExecutionNormalizationSpec,
)


DEFAULT_INPUT = (
    REPO_ROOT
    / "artifacts"
    / "fp_shep_validation"
    / "20260812_203321"
    / "candidate_level_results.csv"
)
FEATURE_COLUMNS = {
    "task_progress": "preview_task_progress",
    "min_clearance": "preview_min_clearance",
    "max_execution_deviation": "preview_execution_deviation",
    "terminal_speed": "preview_terminal_speed",
}
PERCENTILES = (1, 5, 95, 99)


def _read_horizon_rows(path: Path, horizon: int) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"candidate dataset not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"H", "scene_type", *FEATURE_COLUMNS.values()}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"candidate dataset is missing columns: {sorted(missing)}")
        rows = [row for row in reader if int(row["H"]) == int(horizon)]
    if not rows:
        raise ValueError(f"candidate dataset contains no H={horizon} rows")
    return rows


def _skewness(values: np.ndarray) -> float | None:
    if values.size < 3:
        return None
    standard_deviation = float(np.std(values))
    if standard_deviation <= 1.0e-12:
        return 0.0
    centered = (values - float(np.mean(values))) / standard_deviation
    return float(np.mean(centered**3))


def _feature_statistics(
    values: Iterable[float],
    *,
    scene_type: str,
    feature: str,
    horizon: int,
) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=float)
    finite = array[np.isfinite(array)]
    result: dict[str, Any] = {
        "H": int(horizon),
        "scene_type": scene_type,
        "feature": feature,
        "count": int(array.size),
        "finite_count": int(finite.size),
        "positive_inf_count": int(np.isposinf(array).sum()),
        "negative_inf_count": int(np.isneginf(array).sum()),
        "nan_count": int(np.isnan(array).sum()),
    }
    if finite.size == 0:
        for name in (
            "min", "max", "mean", "std", "median", "P1", "P5", "P95",
            "P99", "skewness",
        ):
            result[name] = None
        result["near_zero_variance"] = None
        result["absolute_p99_to_median_ratio"] = None
        return result

    percentiles = np.percentile(finite, PERCENTILES)
    median = float(np.median(finite))
    result.update(
        min=float(np.min(finite)),
        max=float(np.max(finite)),
        mean=float(np.mean(finite)),
        std=float(np.std(finite)),
        median=median,
        P1=float(percentiles[0]),
        P5=float(percentiles[1]),
        P95=float(percentiles[2]),
        P99=float(percentiles[3]),
        skewness=_skewness(finite),
        near_zero_variance=bool(float(np.std(finite)) <= 1.0e-8),
        absolute_p99_to_median_ratio=(
            None
            if abs(median) <= 1.0e-12
            else float(abs(percentiles[3]) / abs(median))
        ),
    )
    return result


def analyze_dataset(path: Path, horizon: int = 4) -> dict[str, Any]:
    rows = _read_horizon_rows(path, horizon)
    scene_types = sorted({row["scene_type"] for row in rows})
    statistics: list[dict[str, Any]] = []
    for scene_type in ("overall", *scene_types):
        group = rows if scene_type == "overall" else [
            row for row in rows if row["scene_type"] == scene_type
        ]
        for feature in EXECUTION_FEATURE_ORDER:
            column = FEATURE_COLUMNS[feature]
            statistics.append(
                _feature_statistics(
                    (float(row[column]) for row in group),
                    scene_type=scene_type,
                    feature=feature,
                    horizon=horizon,
                )
            )

    overall = {
        row["feature"]: row
        for row in statistics
        if row["scene_type"] == "overall"
    }
    clearance = overall["min_clearance"]
    progress = overall["task_progress"]
    deviation = overall["max_execution_deviation"]
    audit_findings = {
        "contains_nan": bool(any(row["nan_count"] for row in statistics)),
        "overall_clearance_inf_count": int(
            clearance["positive_inf_count"] + clearance["negative_inf_count"]
        ),
        "open_space_clearance_inf_expected": True,
        "task_progress_contains_negative_values": bool(progress["min"] < 0.0),
        "execution_deviation_is_nonnegative": bool(deviation["min"] >= 0.0),
        "execution_deviation_skewness": deviation["skewness"],
        "near_zero_variance_features": [
            {"scene_type": row["scene_type"], "feature": row["feature"]}
            for row in statistics
            if row["near_zero_variance"] is True
        ],
        "clearance_semantics": (
            "positive infinity means no initially visible frozen LiDAR surface; "
            "it is not exact global clearance"
        ),
    }
    spec = ExecutionNormalizationSpec()
    normalization = {
        "version": spec.version,
        "raw_features_are_preserved": True,
        "task_progress_scale": spec.task_progress_scale,
        "min_clearance_scale": spec.min_clearance_scale,
        "max_execution_deviation_scale": spec.max_execution_deviation_scale,
        "terminal_speed_scale": spec.terminal_speed_scale,
        "open_space_clearance_encoding": {
            "normalized_value": 1.0,
            "clearance_finite_mask": False,
            "open_space_flag": True,
        },
    }
    return {
        "source_dataset": str(path.resolve()),
        "H": int(horizon),
        "candidate_count": len(rows),
        "scene_types": scene_types,
        "feature_statistics": statistics,
        "audit_findings": audit_findings,
        "normalization_spec": normalization,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_artifacts(report: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=False)
    statistics = report["feature_statistics"]
    with (output_dir / "feature_statistics_h4.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(statistics[0]))
        writer.writeheader()
        writer.writerows(statistics)
    with (output_dir / "feature_statistics_h4.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(_json_safe(report), handle, ensure_ascii=False, indent=2)
    return output_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> Path:
    args = _parse_args()
    input_path = args.input.expanduser().resolve()
    report = analyze_dataset(input_path, horizon=int(args.horizon))
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = REPO_ROOT / "artifacts" / "fp_shep_execution_interface" / timestamp
    else:
        output_dir = args.output_dir.expanduser().resolve()
    write_artifacts(report, output_dir)
    print(f"Artifacts written to: {output_dir}")
    return output_dir


if __name__ == "__main__":
    main()
