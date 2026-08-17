"""Validate and optionally promote Pre-GAT paper-ready artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
for search_path in (REPO_ROOT, ALGO_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from pre_gat_closed_loop_constants import (  # noqa: E402
    METHOD_DISPLAY_NAMES,
    METHOD_FP_SHEP,
    METHOD_FROZEN,
    METHOD_ORDER,
    METHOD_PROPOSAL,
)


CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def _coerce(value: str) -> Any:
    if value == "":
        return None
    if value == "True":
        return True
    if value == "False":
        return False
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return [
            {key: _coerce(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]


def _finite_csv(path: Path) -> bool:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        for row in csv.reader(stream):
            for value in row:
                if value.strip().lower() in {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity"}:
                    return False
    return True


def _language_check(root: Path) -> list[str]:
    failures: list[str] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if CJK_PATTERN.search(relative):
            failures.append(f"filename contains CJK characters: {relative}")
        if not path.is_file() or path.suffix.lower() not in {".csv", ".json", ".tex", ".md"}:
            continue
        text = path.read_text(encoding="utf-8-sig")
        if CJK_PATTERN.search(text):
            failures.append(f"publication-facing text contains CJK characters: {relative}")
    return failures


def _png_dpi(path: Path) -> float:
    with Image.open(path) as image:
        dpi = image.info.get("dpi", (0.0, 0.0))
        if isinstance(dpi, (int, float)):
            return float(dpi)
        return float(min(dpi)) if dpi else 0.0


def _close(left: Any, right: Any, atol: float = 1.0e-9) -> bool:
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    return bool(np.isclose(float(left), float(right), rtol=1.0e-9, atol=atol))


def validate_run(run_dir: Path, *, promote: bool = False) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    settings = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    paper = run_dir / "paper_ready"
    figures = paper / "figures"
    data_dir = paper / "figure_data"
    tables = paper / "tables"
    failures: list[str] = []

    pdfs = sorted(figures.glob("*.pdf"))
    pngs = sorted(figures.glob("*.png"))
    pdf_stems = {path.stem for path in pdfs}
    png_stems = {path.stem for path in pngs}
    if pdf_stems != png_stems:
        failures.append("PDF/PNG figure stem sets differ")
    minimum_dpi = float(settings["paper_ready"]["minimum_png_dpi"])
    final_dpi = float(settings["paper_ready"]["png_dpi"])
    dpi_records = []
    for stem in sorted(pdf_stems | png_stems):
        pdf = figures / f"{stem}.pdf"
        png = figures / f"{stem}.png"
        csv_path = data_dir / f"{stem}.csv"
        json_path = data_dir / f"{stem}.json"
        for path in (pdf, png, csv_path, json_path):
            if not path.is_file() or path.stat().st_size <= 0:
                failures.append(f"missing or empty figure artifact: {path.relative_to(run_dir)}")
        if png.is_file():
            dpi = _png_dpi(png)
            dpi_records.append({"figure": stem, "dpi": dpi})
            if dpi + 1.0 < minimum_dpi:
                failures.append(f"PNG DPI below minimum for {stem}: {dpi}")
            if dpi + 1.0 < final_dpi:
                failures.append(f"paper-final PNG DPI below configured target for {stem}: {dpi}")
        if csv_path.is_file() and not _finite_csv(csv_path):
            failures.append(f"figure data contains unexplained NaN/Inf: {csv_path.name}")

    required_tables = (
        "table_overall_metrics.csv", "table_scenario_metrics.csv",
        "table_overall_metrics.tex", "table_scenario_metrics.tex",
    )
    for name in required_tables:
        path = tables / name
        if not path.is_file() or path.stat().st_size <= 0:
            failures.append(f"missing table: {name}")
    manifest = paper / "PAPER_READY_MANIFEST.md"
    if not manifest.is_file() or manifest.stat().st_size <= 0:
        failures.append("PAPER_READY_MANIFEST.md is missing or empty")
    else:
        manifest_text = manifest.read_text(encoding="utf-8")
        for stem in sorted(pdf_stems):
            if stem not in manifest_text:
                failures.append(f"manifest does not list figure: {stem}")
        for name in ("table_overall_metrics", "table_scenario_metrics"):
            if name not in manifest_text:
                failures.append(f"manifest does not list table: {name}")

    language_failures = _language_check(paper)
    failures.extend(language_failures)

    episode_rows = _read_csv(run_dir / "per_episode" / "all_episode_records.csv")
    formal = [row for row in episode_rows if row["phase"] == "formal"]
    configured_seeds = sorted(int(item) for item in settings["seeds"]["formal"])
    actual_seeds = sorted({int(row["seed"]) for row in formal})
    if actual_seeds != configured_seeds:
        failures.append("formal seed coverage differs from config")
    expected_count = len(settings["scenario_types"]) * len(configured_seeds) * len(METHOD_ORDER)
    if len(formal) != expected_count:
        failures.append(f"formal episode count {len(formal)} != expected {expected_count}")
    by_pair: dict[str, list[dict[str, Any]]] = {}
    for row in formal:
        by_pair.setdefault(str(row["pair_id"]), []).append(row)
    for pair_id, rows in by_pair.items():
        if {row["method"] for row in rows} != set(METHOD_ORDER):
            failures.append(f"paired methods incomplete: {pair_id}")
        hashes = {row["initial_condition_hash"] for row in rows}
        if len(hashes) != 1:
            failures.append(f"paired initial condition mismatch: {pair_id}")
        m_values = {int(row["M_upper"]) for row in rows}
        if len(m_values) != 1:
            failures.append(f"paired M_upper mismatch: {pair_id}")
    if set(actual_seeds) & set(int(item) for item in settings["seeds"]["development"]):
        failures.append("development seeds leak into formal results")

    event_rows = _read_csv(
        run_dir / "per_replanning_event" / "all_replanning_events.csv"
    )
    formal_events = [row for row in event_rows if row["phase"] == "formal"]
    if any(row["method"] == METHOD_FROZEN for row in formal_events):
        failures.append("baseline unexpectedly contains candidate replanning events")
    for row in formal_events:
        timestep = int(row["timestep"])
        m_upper = int(row["M_upper"])
        if timestep % m_upper != 0 or not bool(row["replanning_rule_satisfied"]):
            failures.append("event outside strict fixed-period replanning schedule")
            break
        if not bool(row["terminal_goal_unchanged"]):
            failures.append("temporary reference changed terminal task goal")
            break
        if not bool(row["phase_preserved_on_switch"]):
            failures.append("temporary reference switch reset DMP phase")
            break
        if bool(row["GAT_used_for_selection"]) or bool(row["supervision_target_used_online"]):
            failures.append("forbidden learned/oracle selector entered online execution")
            break
        if int(row["candidate_count"]) != int(row["K_t"]):
            failures.append("candidate_count and K_t differ")
            break
        if int(row["K_t"]) > int(row["K_requested"]):
            failures.append("actual candidate count exceeds explicit consumer K")
            break
        if int(row["K_t"]) == 0:
            if row["selection_kind"] != "no_candidate_fallback":
                failures.append("K_t=0 event does not use no_candidate_fallback")
                break
            if row.get("selected_candidate_id") is not None:
                failures.append("K_t=0 fallback was counted as candidate selection")
                break
        elif bool(row["no_candidate_fallback"]):
            failures.append("nonempty candidate event incorrectly counted as fallback")
            break
        if row["method"] == METHOD_FP_SHEP:
            if bool(row["terminal_speed_used_for_online_ranking"]):
                failures.append("terminal speed entered the three-feature online ranking")
                break
            candidate_records = row.get("fp_shep_candidate_records") or []
            if len(candidate_records) != int(row["K_t"]):
                failures.append("FP-SHEP candidate record count differs from K_t")
                break
            required = {
                "preview_task_progress", "preview_min_clearance",
                "preview_max_execution_deviation", "preview_terminal_speed",
            }
            if any(not required.issubset(record) for record in candidate_records):
                failures.append("FP-SHEP event does not retain all four preview features")
                break

    initial_events: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in formal_events:
        if int(row["timestep"]) != 0:
            continue
        key = (str(row["pair_id"]), int(row["agent_id"]), str(row["method"]))
        initial_events[key] = row
    for pair_id in by_pair:
        for agent_id in range(int(settings["num_agents"])):
            proposal = initial_events.get((pair_id, agent_id, METHOD_PROPOSAL))
            fp_shep = initial_events.get((pair_id, agent_id, METHOD_FP_SHEP))
            if proposal is None or fp_shep is None:
                failures.append(f"missing initial candidate event: {pair_id}, agent={agent_id}")
                continue
            if (
                proposal["candidate_world_positions"] != fp_shep["candidate_world_positions"]
                or proposal["proposal_scores"] != fp_shep["proposal_scores"]
            ):
                failures.append(
                    f"initial Proposal/FP-SHEP candidate set mismatch: {pair_id}, agent={agent_id}"
                )

    selector = settings["fp_shep_online_selector"]
    if int(selector["H_preview"]) != 4:
        failures.append("online FP-SHEP horizon is not H=4")
    if float(selector["weights"]["terminal_speed"]) != 0.0:
        failures.append("terminal speed weight is nonzero in online selector")
    if selector["used_for_online_ranking"].get("terminal_speed") is not False:
        failures.append("terminal speed ranking metadata is not explicitly false")
    if not bool(selector.get("all_four_preview_features_recorded", False)):
        failures.append("selector config does not require all four preview features")

    expected_names = {METHOD_DISPLAY_NAMES[item] for item in METHOD_ORDER}
    overall_table = _read_csv(tables / "table_overall_metrics.csv") if (tables / "table_overall_metrics.csv").is_file() else []
    if {row.get("Method") for row in overall_table} != expected_names:
        failures.append("overall table method naming is inconsistent")
    for method in METHOD_ORDER:
        raw = [row for row in formal if row["method"] == method]
        table = next((row for row in overall_table if row.get("Method") == METHOD_DISPLAY_NAMES[method]), None)
        if table is None:
            continue
        success_count = sum(bool(row["success"]) for row in raw)
        collision_count = sum(bool(row["collision"]) for row in raw)
        if int(table["Success Count"]) != success_count or int(table["Episode Count"]) != len(raw):
            failures.append(f"success numerator/denominator mismatch for {method}")
        if int(table["Collision Count"]) != collision_count:
            failures.append(f"collision numerator mismatch for {method}")
        if not _close(table["Success Rate (%)"], 100.0 * success_count / len(raw)):
            failures.append(f"success rate cannot be recomputed for {method}")
        successful = [row for row in raw if row["success"]]
        expected_path = (
            float(np.mean([row["path_length_success_team_mean_m"] for row in successful]))
            if successful else None
        )
        expected_time = (
            float(np.mean([row["completion_time_success_s"] for row in successful]))
            if successful else None
        )
        if not _close(table["Path Length Mean (m)"], expected_path):
            failures.append(f"successful-only path aggregation mismatch for {method}")
        if not _close(table["Completion Time Mean (s)"], expected_time):
            failures.append(f"successful-only completion-time aggregation mismatch for {method}")
        if int(table["Path Length N"]) != len(successful):
            failures.append(f"successful-only path sample count mismatch for {method}")

    generation = json.loads((run_dir / "summary" / "generation_summary.json").read_text(encoding="utf-8"))
    if not generation.get("checkpoint_unchanged", False):
        failures.append("checkpoint hash changed")
    if not generation.get("policy_parameters_unchanged", False):
        failures.append("frozen policy parameters changed")
    if not generation.get("critical_execution_semantics_unchanged", False):
        failures.append("critical execution source hash changed")
    if generation.get("GAT_training_started", True):
        failures.append("GAT training was started")
    if generation.get("GAT_forward_used_for_selection", True):
        failures.append("GAT forward was used for selection")
    if generation.get("supervision_target_used_online", True):
        failures.append("supervision target was used online")

    forbidden_final_names = {"smoke", "debug", "cache", "upper_period_sensitivity"}
    if any(any(token in path.name.lower() for token in forbidden_final_names) for path in paper.rglob("*")):
        failures.append("smoke/debug/sensitivity artifact leaked into paper_ready")

    passed = not failures
    validation = {
        "run_dir": str(run_dir),
        "paper_ready_dir": str(paper),
        "validation_passed": passed,
        "figure_count": len(pdf_stems),
        "table_count": 2,
        "dpi_records": dpi_records,
        "publication_language_check_passed": not language_failures,
        "formal_seed_count": len(actual_seeds),
        "formal_episode_count": len(formal),
        "paired_count": len(by_pair),
        "formal_replanning_event_count": len(formal_events),
        "online_semantics_check_passed": not any(
            marker in failure
            for failure in failures
            for marker in (
                "replanning", "terminal task goal", "DMP phase", "selector",
                "candidate", "fallback", "terminal speed", "preview features",
            )
        ),
        "strict_fixed_period_rule_checked": "timestep % M_upper == 0",
        "initial_candidate_fairness_pairs_checked": len(by_pair) * int(settings["num_agents"]),
        "selector_score_specification_checked": True,
        "terminal_speed_used_for_online_ranking": False,
        "failures": failures,
        "promoted": False,
        "paper_final_candidate_dir": None,
    }
    final_value = Path(settings["paper_final_candidate_dir"])
    final_dir = final_value if final_value.is_absolute() else REPO_ROOT / final_value
    if promote and passed:
        if final_dir.exists():
            existing_config = final_dir / "experiment_config.json"
            if (
                not existing_config.is_file()
                or json.loads(existing_config.read_text(encoding="utf-8")) != settings
            ):
                failures.append(
                    f"paper-final candidate exists with a different experiment config: {final_dir}"
                )
                validation["validation_passed"] = False
            else:
                shutil.copytree(paper, final_dir, dirs_exist_ok=True)
                validation["promoted"] = True
                validation["promotion_mode"] = "existing_matching_candidate_refreshed"
                validation["paper_final_candidate_dir"] = str(final_dir.resolve())
                (final_dir / "validation_summary.json").write_text(
                    json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8"
                )
        else:
            shutil.copytree(paper, final_dir)
            shutil.copy2(run_dir / "config.json", final_dir / "experiment_config.json")
            validation["promoted"] = True
            validation["promotion_mode"] = "new_candidate_created"
            validation["paper_final_candidate_dir"] = str(final_dir.resolve())
            (final_dir / "validation_summary.json").write_text(
                json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8"
            )
    (run_dir / "summary" / "paper_ready_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return validation


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--promote", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = validate_run(args.run_dir, promote=args.promote)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["validation_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
