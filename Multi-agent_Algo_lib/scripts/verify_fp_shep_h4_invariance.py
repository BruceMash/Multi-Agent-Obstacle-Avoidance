"""Verify that interface hardening did not change validated H=4 outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


FLOAT_FIELDS = (
    "proposal_score",
    "preview_task_progress",
    "preview_min_clearance",
    "preview_execution_deviation",
    "preview_terminal_speed",
    "J_preview",
)
RANK_FIELDS = (
    "candidate_rank",
    "geometry_rank",
    "preview_rank",
)


def _load_horizon(path: Path, horizon: int) -> dict[tuple[Any, ...], dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            row for row in csv.DictReader(handle) if int(row["H"]) == int(horizon)
        ]
    if not rows:
        raise ValueError(f"{path} contains no H={horizon} candidate rows")
    keyed: dict[tuple[Any, ...], dict[str, str]] = {}
    for row in rows:
        key = (
            row["scenario_type"],
            int(row["seed"]),
            int(row["agent_id"]),
            int(row["candidate_id"]),
            tuple(float(value) for value in json.loads(row["candidate_xyz"])),
        )
        if key in keyed:
            raise ValueError(f"duplicate candidate key in {path}: {key}")
        keyed[key] = row
    return keyed


def compare_validation_outputs(
    reference_path: Path,
    repeat_path: Path,
    *,
    horizon: int = 4,
    absolute_tolerance: float = 1.0e-12,
) -> dict[str, Any]:
    reference = _load_horizon(reference_path, horizon)
    repeat = _load_horizon(repeat_path, horizon)
    keys_match = reference.keys() == repeat.keys()
    float_results: dict[str, Any] = {}
    rank_results: dict[str, Any] = {}
    if keys_match:
        for field in FLOAT_FIELDS:
            finite_differences: list[float] = []
            infinity_mismatch_count = 0
            for key in reference:
                left = float(reference[key][field])
                right = float(repeat[key][field])
                if math.isinf(left) or math.isinf(right):
                    infinity_mismatch_count += int(left != right)
                elif math.isnan(left) or math.isnan(right):
                    infinity_mismatch_count += int(not (math.isnan(left) and math.isnan(right)))
                else:
                    finite_differences.append(abs(left - right))
            maximum = max(finite_differences, default=0.0)
            float_results[field] = {
                "maximum_absolute_difference": maximum,
                "nonfinite_mismatch_count": infinity_mismatch_count,
                "within_tolerance": bool(
                    maximum <= float(absolute_tolerance)
                    and infinity_mismatch_count == 0
                ),
            }
        for field in RANK_FIELDS:
            mismatches = sum(
                int(reference[key][field]) != int(repeat[key][field])
                for key in reference
            )
            rank_results[field] = {
                "mismatch_count": mismatches,
                "identical": mismatches == 0,
            }

    k_reference = {
        (key[0], key[1], key[2]): int(row["K_t"])
        for key, row in reference.items()
    }
    k_repeat = {
        (key[0], key[1], key[2]): int(row["K_t"])
        for key, row in repeat.items()
    }
    passed = bool(
        keys_match
        and k_reference == k_repeat
        and all(row["within_tolerance"] for row in float_results.values())
        and all(row["identical"] for row in rank_results.values())
    )
    return {
        "passed": passed,
        "H": int(horizon),
        "absolute_tolerance": float(absolute_tolerance),
        "reference_dataset": str(reference_path.resolve()),
        "repeat_dataset": str(repeat_path.resolve()),
        "reference_candidate_count": len(reference),
        "repeat_candidate_count": len(repeat),
        "candidate_keys_identical": keys_match,
        "K_t_maps_identical": k_reference == k_repeat,
        "float_fields": float_results,
        "rank_fields": rank_results,
        "runtime_fields_excluded": True,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--repeat", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--atol", type=float, default=1.0e-12)
    return parser.parse_args()


def main() -> Path:
    args = _parse_args()
    report = compare_validation_outputs(
        args.reference.expanduser().resolve(),
        args.repeat.expanduser().resolve(),
        horizon=int(args.horizon),
        absolute_tolerance=float(args.atol),
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)
    return output


if __name__ == "__main__":
    main()
