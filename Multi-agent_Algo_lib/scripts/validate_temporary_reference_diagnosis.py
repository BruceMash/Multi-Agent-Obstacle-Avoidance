"""Validate Temporary-Reference Interface Diagnosis artifacts and test logs."""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Sequence

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, ALGO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from planning.temporary_reference_diagnosis import (  # noqa: E402
    PROTOCOL_EXISTING_BOUNDARY_FREE,
    PROTOCOL_FIXED_PERIOD,
    PROTOCOL_ONE_SHOT,
)
from scripts.evaluate_temporary_reference_interface import write_json  # noqa: E402


FIGURES = (
    "D1_protocol_success_comparison",
    "D2_obstacle_collision_comparison",
    "D3_trajectory_smoothness_comparison",
    "D4_representative_one_shot_trajectory",
    "D5_fixed_period_switching_failure",
    "D6_dmp_acceleration_jump",
    "D7_active_terminal_goal_distance",
    "D8_reference_lifecycle_timeline",
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_junit(paths: Sequence[Path]) -> dict[str, Any]:
    totals = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    files: list[str] = []
    for path in paths:
        root = ET.parse(path).getroot()
        suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
        for suite in suites:
            for key in totals:
                totals[key] += int(suite.attrib.get(key, 0))
        files.append(str(path.resolve()))
    totals["passed"] = totals["tests"] - totals["failures"] - totals["errors"] - totals["skipped"]
    totals["status"] = "PASSED" if totals["failures"] == 0 and totals["errors"] == 0 else "FAILED"
    totals["junit_files"] = files
    return totals


def record_test_results(run_dir: Path, junit_paths: Sequence[Path]) -> dict[str, Any]:
    result = parse_junit(junit_paths)
    write_json(run_dir / "tests" / "regression_summary.json", result)
    return result


def validate(run_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    required = (
        "config.json",
        "manifest.json",
        "protocol_summary/protocol_summary.json",
        "protocol_summary/stage_c_gate.json",
        "per_episode/episodes.csv",
        "per_step/per_step.csv",
        "reference_events/reference_events.json",
        "dmp_switch_diagnostics/switch_diagnostics.json",
        "observation_diagnostics/observation_diagnostics.json",
        "failure_attribution/failure_attribution.json",
        "representative_cases/case_index.json",
        "FINAL_REPORT.md",
    )
    for relative in required:
        path = run_dir / relative
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing_or_empty:{relative}")

    config = _load(run_dir / "config.json")
    manifest = _load(run_dir / "manifest.json")
    episodes = [
        _load(path)
        for path in sorted((run_dir / "per_episode").glob("*.json"))
        if path.name != "episodes.json"
    ]
    if config["protocol_metadata"][PROTOCOL_ONE_SHOT]["reached_tolerance_m"] != 0.25:
        errors.append("one_shot_tolerance_not_0.25")
    if config["protocol_metadata"][PROTOCOL_FIXED_PERIOD]["reached_tolerance_m"] != 0.30:
        errors.append("fixed_period_tolerance_not_0.30")
    if config["existing_waypoint"]["boundary_filter_enabled"] is not False:
        errors.append("boundary_free_protocol_filter_is_enabled")
    if set(config["development_seeds"]) & set(
        config["formal_seeds_excluded_during_interface_design"]
    ):
        errors.append("development_formal_seed_overlap")

    by_pair: dict[tuple[str, int], set[str]] = {}
    for row in episodes:
        by_pair.setdefault((str(row["scenario"]), int(row["seed"])), set()).add(
            str(row["initial_condition_hash"])
        )
        if row["protocol"] == PROTOCOL_EXISTING_BOUNDARY_FREE:
            if row.get("workspace_boundary_filter_enabled") is not False:
                errors.append(f"boundary_filter_enabled:{row['pair_id']}")
            if int(row.get("boundary_rejection_count", 0)) != 0:
                errors.append(f"boundary_rejection_nonzero:{row['pair_id']}")
    for pair, hashes in by_pair.items():
        if len(hashes) != 1:
            errors.append(f"paired_initial_hash_mismatch:{pair}")

    integrity = manifest.get("integrity", {})
    if integrity.get("policy_parameters_unchanged") is not True:
        errors.append("policy_parameters_changed")
    if integrity.get("critical_files_unchanged") is not True:
        errors.append("critical_files_changed")
    if integrity.get("GAT_training_started") is not False:
        errors.append("GAT_training_started")
    if integrity.get("SAC_training_performed") is not False:
        errors.append("SAC_training_performed")

    for name in FIGURES:
        for suffix, folder in ((".pdf", "figures"), (".png", "figures"), (".csv", "figure_data"), (".json", "figure_data")):
            path = run_dir / folder / f"{name}{suffix}"
            if not path.is_file() or path.stat().st_size == 0:
                errors.append(f"missing_figure_artifact:{folder}/{name}{suffix}")
        png = run_dir / "figures" / f"{name}.png"
        if png.is_file():
            with Image.open(png) as image:
                dpi = image.info.get("dpi", (0.0, 0.0))
                if min(float(dpi[0]), float(dpi[1])) < 590.0:
                    errors.append(f"png_dpi_below_600:{name}:{dpi}")
        if re.search(r"[^\x00-\x7F]", name):
            errors.append(f"non_ascii_figure_filename:{name}")

    report = (run_dir / "FINAL_REPORT.md").read_text(encoding="utf-8")
    for section in "ABCDEFGHIJKLMNOPQRSTUV":
        if f"## {section}." not in report and f"## {section}–" not in report:
            errors.append(f"missing_report_section:{section}")
    test_path = run_dir / "tests" / "regression_summary.json"
    if test_path.is_file():
        tests = _load(test_path)
        if tests.get("status") != "PASSED":
            errors.append("regression_tests_failed")
    else:
        warnings.append("regression_summary_not_recorded")

    result = {
        "status": "PASSED" if not errors else "FAILED",
        "errors": errors,
        "warnings": warnings,
        "episode_count": len(episodes),
        "figure_count": len(list((run_dir / "figures").glob("*.pdf"))),
        "protocol_count": len({str(row["protocol"]) for row in episodes}),
    }
    write_json(run_dir / "tests" / "artifact_validation.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--junit", type=Path, action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if args.junit:
        record_test_results(run_dir, [path.resolve() for path in args.junit])
    result = validate(run_dir)
    print(json.dumps(result, indent=2), flush=True)
    if result["status"] != "PASSED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

