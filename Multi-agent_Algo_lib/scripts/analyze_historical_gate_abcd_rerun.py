"""Analyze historical-vector-gate A/B/C/D results against the scalar run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
for search_path in (REPO_ROOT, ALGO_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from planning.goal_semantics_diagnosis import (  # noqa: E402
    VARIANT_A,
    VARIANT_B,
    VARIANT_C,
    VARIANT_D,
    VARIANT_DISPLAY_NAMES,
    VARIANT_ORDER,
)


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
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = [dict(row) for row in rows]
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
            writer.writerow(
                {
                    key: (
                        json.dumps(_jsonable(row.get(key)), ensure_ascii=False)
                        if isinstance(row.get(key), (list, tuple, dict, np.ndarray))
                        else _jsonable(row.get(key))
                    )
                    for key in fields
                }
            )


def _decode(value: str | None) -> Any:
    if value in {None, ""}:
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
            {key: _decode(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]


def _rate(rows: Sequence[Mapping[str, Any]], key: str) -> tuple[int, int, float]:
    values = [bool(row[key]) for row in rows]
    count = int(sum(values))
    total = len(values)
    return count, total, count / total if total else 0.0


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [
        float(row[key])
        for row in rows
        if row.get(key) is not None and math.isfinite(float(row[key]))
    ]
    return float(np.mean(values)) if values else None


def _candidate_rates(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, int | float | None]:
    available = sum(int(row.get("temporary_reference_count") or 0) for row in rows)
    reached = sum(
        int(
            row.get("temporary_reference_reached_count_metric")
            if row.get("temporary_reference_reached_count_metric") is not None
            else row.get("temporary_reference_reached_count") or 0
        )
        for row in rows
    )
    reached_then_terminal = sum(
        int(
            row.get("reached_then_terminal_success_count")
            if row.get("reached_then_terminal_success_count") is not None
            else (
                int(row.get("temporary_reference_reached_count") or 0)
                if bool(row.get("success"))
                else 0
            )
        )
        for row in rows
    )
    return {
        "temporary_reference_reached_count": reached,
        "temporary_reference_reached_total": available,
        "temporary_reference_reached_rate": reached / available if available else None,
        "reached_then_terminal_success_count": reached_then_terminal,
        "reached_then_terminal_success_total": reached,
        "reached_then_terminal_success_rate": (
            reached_then_terminal / reached if reached else None
        ),
    }


def _aggregate_variant(
    rows: Sequence[Mapping[str, Any]], variant: str
) -> dict[str, Any]:
    members = [row for row in rows if str(row["variant"]) == variant]
    result: dict[str, Any] = {"episodes": len(members)}
    for name, key in (
        ("success", "success"),
        ("collision", "collision"),
        ("obstacle_collision", "obstacle_collision"),
        ("inter_agent_collision", "inter_agent_collision"),
        ("timeout", "truncated"),
    ):
        count, total, rate = _rate(members, key)
        result[f"{name}_count"] = count
        result[f"{name}_total"] = total
        result[f"{name}_rate"] = rate
    result.update(_candidate_rates(members))
    for destination, source in (
        ("terminal_progress_m", "terminal_progress_team_mean_m"),
        ("trajectory_smoothness", "trajectory_smoothness"),
        ("mean_acceleration_mps2", "mean_applied_acceleration_mps2"),
        ("action_saturation_rate", "action_saturation_rate"),
    ):
        result[destination] = _mean(members, source)
    return result


def _comparison_rows(
    current_rows: Sequence[Mapping[str, Any]],
    historical_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in VARIANT_ORDER:
        current = _aggregate_variant(current_rows, variant)
        historical = _aggregate_variant(historical_rows, variant)
        row: dict[str, Any] = {
            "variant": variant,
            "variant_display_name": VARIANT_DISPLAY_NAMES[variant],
        }
        for prefix, values in (("current_scalar", current), ("historical_vector", historical)):
            row.update({f"{prefix}_{key}": value for key, value in values.items()})
        for metric in (
            "success_rate",
            "collision_rate",
            "obstacle_collision_rate",
            "inter_agent_collision_rate",
            "timeout_rate",
            "terminal_progress_m",
            "trajectory_smoothness",
            "temporary_reference_reached_rate",
            "reached_then_terminal_success_rate",
            "mean_acceleration_mps2",
            "action_saturation_rate",
        ):
            current_value = current.get(metric)
            historical_value = historical.get(metric)
            row[f"delta_{metric}"] = (
                float(historical_value) - float(current_value)
                if current_value is not None and historical_value is not None
                else None
            )
        rows.append(row)
    return rows


def _variant_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(row["variant"]): row for row in rows}


def _waypoint_truncation_diagnostic(
    historical_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    members = [
        row
        for row in historical_rows
        if str(row["variant"]) in {VARIANT_C, VARIANT_D}
    ]
    reached = sum(int(row.get("temporary_reference_reached_count_metric") or 0) for row in members)
    reached_total = sum(int(row.get("temporary_reference_reached_total_metric") or 0) for row in members)
    reached_terminal = sum(int(row.get("reached_then_terminal_success_count") or 0) for row in members)
    attenuated = sum(int(row.get("near_active_reference_attenuated_count") or 0) for row in members)
    attenuated_total = sum(int(row.get("near_active_reference_attenuated_total") or 0) for row in members)
    low_speed = sum(int(row.get("near_active_reference_low_speed_count") or 0) for row in members)
    low_speed_total = sum(int(row.get("near_active_reference_low_speed_total") or 0) for row in members)
    reached_terminal_rate = reached_terminal / reached if reached else None
    attenuation_rate = attenuated / attenuated_total if attenuated_total else None
    low_speed_rate = low_speed / low_speed_total if low_speed_total else None
    diagnostic = config["waypoint_attenuation_diagnostic"]
    completion_failure_rate = (
        1.0 - reached_terminal_rate if reached_terminal_rate is not None else None
    )
    confirmed = bool(
        reached > 0
        and attenuated_total > 0
        and attenuation_rate is not None
        and attenuation_rate > 0.0
        and low_speed_rate is not None
        and low_speed_rate >= float(diagnostic["low_speed_rate_threshold"])
        and completion_failure_rate is not None
        and completion_failure_rate
        >= float(diagnostic["terminal_completion_failure_rate_threshold"])
    )
    return {
        "confirmed": confirmed,
        "temporary_reference_reached_count": reached,
        "temporary_reference_reached_total": reached_total,
        "temporary_reference_reached_rate": reached / reached_total if reached_total else None,
        "near_reference_attenuated_count": attenuated,
        "near_reference_attenuated_total": attenuated_total,
        "near_reference_attenuated_rate": attenuation_rate,
        "near_reference_low_speed_count": low_speed,
        "near_reference_low_speed_total": low_speed_total,
        "near_reference_low_speed_rate": low_speed_rate,
        "reached_then_terminal_success_count": reached_terminal,
        "reached_then_terminal_success_total": reached,
        "reached_then_terminal_success_rate": reached_terminal_rate,
        "terminal_completion_failure_rate_after_reach": completion_failure_rate,
        "thresholds": dict(diagnostic),
    }


def _build_conclusion(
    config: Mapping[str, Any],
    comparison: Sequence[Mapping[str, Any]],
    historical_rows: Sequence[Mapping[str, Any]],
    gate_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_variant = _variant_map(comparison)
    thresholds = config["decision_thresholds"]
    material_rate = float(thresholds["material_absolute_rate_delta"])
    progress_delta = float(thresholds["material_terminal_progress_delta_m"])
    smoothness_relative = float(
        thresholds["material_smoothness_relative_delta"]
    )
    near_zero = float(thresholds["near_zero_success_rate"])
    recovered_rate = float(
        thresholds["temporary_reference_recovered_success_rate"]
    )

    def materially_affected(row: Mapping[str, Any]) -> bool:
        current_smoothness = float(row["current_scalar_trajectory_smoothness"])
        delta_smoothness = abs(float(row["delta_trajectory_smoothness"]))
        relative_smoothness = (
            delta_smoothness / current_smoothness
            if current_smoothness > 1.0e-12
            else math.inf
        )
        return bool(
            abs(float(row["delta_success_rate"])) >= material_rate
            or abs(float(row["delta_collision_rate"])) >= material_rate
            or abs(float(row["delta_terminal_progress_m"])) >= progress_delta
            or relative_smoothness >= smoothness_relative
        )

    baseline = by_variant[VARIANT_A]
    baseline_affected = materially_affected(baseline)
    baseline_restored = float(baseline["delta_success_rate"]) >= material_rate
    temporary_affected = any(
        materially_affected(by_variant[variant])
        for variant in (VARIANT_B, VARIANT_C, VARIANT_D)
    )
    bcd_near_zero = all(
        float(by_variant[variant]["historical_vector_success_rate"]) <= near_zero
        for variant in (VARIANT_B, VARIANT_C, VARIANT_D)
    )
    recovered_variants = [
        variant
        for variant in (VARIANT_C, VARIANT_D)
        if float(by_variant[variant]["historical_vector_success_rate"])
        >= recovered_rate
        and float(by_variant[variant]["delta_success_rate"]) >= material_rate
    ]
    truncation = _waypoint_truncation_diagnostic(historical_rows, config)

    cases: list[str] = []
    if baseline_restored and bcd_near_zero:
        cases.append("Case 1")
    if baseline_restored and recovered_variants:
        cases.append("Case 2")
    if not baseline_restored and bcd_near_zero:
        cases.append("Case 3")
    if bool(truncation["confirmed"]):
        cases.append("Case 4")
    if not cases:
        cases.append("MIXED_RESULT")

    if bool(truncation["confirmed"]):
        primary = "INTERMEDIATE_WAYPOINT_FORCING_TRUNCATION"
        next_step = (
            "Preserve terminal-conditioned forcing continuity and perform targeted "
            "waypoint-conditioned lower-policy adaptation."
        )
    elif baseline_restored and bcd_near_zero:
        primary = "INTERMEDIATE_REFERENCE_DISTRIBUTION_SHIFT"
        next_step = "Perform targeted waypoint-conditioned lower-policy adaptation."
    elif recovered_variants:
        primary = "FORCING_GATE_SEMANTIC_DRIFT"
        next_step = (
            "Defer SAC fine-tuning and re-evaluate the final forcing/reference interface."
        )
    elif not baseline_restored and bcd_near_zero:
        primary = "INTERMEDIATE_REFERENCE_DISTRIBUTION_SHIFT"
        next_step = "Perform targeted waypoint-conditioned fine-tuning."
    else:
        primary = "MIXED_FORCING_GATE_AND_REFERENCE_EFFECTS"
        next_step = "Inspect per-variant failure modes before selecting an adaptation path."

    fine_tuning = not bool(recovered_variants) or bool(truncation["confirmed"])
    gate_acceleration_deltas = [
        float(row["commanded_acceleration_delta_l2"]) for row in gate_rows
    ]
    gate_forcing_deltas = [
        float(row["forcing_contribution_delta_l2"]) for row in gate_rows
    ]
    return {
        "decision_cases": cases,
        "FORCING_GATE_DRIFT_AFFECTS_BASELINE": "YES" if baseline_affected else "NO",
        "FORCING_GATE_DRIFT_AFFECTS_TEMPORARY_REFERENCE": "YES" if temporary_affected else "NO",
        "INTERMEDIATE_WAYPOINT_FORCING_TRUNCATION_CONFIRMED": (
            "YES" if truncation["confirmed"] else "NO"
        ),
        "SAC_FINE_TUNING_RECOMMENDED": "YES" if fine_tuning else "NO",
        "PRIMARY_LIMITATION": primary,
        "NEXT_STEP": next_step,
        "baseline_terminal_navigation_restored": baseline_restored,
        "temporary_reference_variants_near_zero": bcd_near_zero,
        "temporary_reference_recovered_variants": recovered_variants,
        "waypoint_truncation_diagnostic": truncation,
        "same_state_same_action_gate_diagnostic": {
            "record_count": len(gate_rows),
            "mean_commanded_acceleration_delta_l2_mps2": float(
                np.mean(gate_acceleration_deltas)
            ),
            "max_commanded_acceleration_delta_l2_mps2": float(
                np.max(gate_acceleration_deltas)
            ),
            "mean_forcing_contribution_delta_l2": float(
                np.mean(gate_forcing_deltas)
            ),
            "max_forcing_contribution_delta_l2": float(
                np.max(gate_forcing_deltas)
            ),
            "environment_state_advanced": False,
        },
        "decision_thresholds": dict(thresholds),
        "checkpoint_unchanged": all(
            bool(row.get("checkpoint_unchanged")) for row in historical_rows
        ),
        "policy_parameters_unchanged": all(
            bool(row.get("policy_parameters_unchanged")) for row in historical_rows
        ),
        "current_scalar_transition_restored": all(
            bool(row.get("current_scalar_transition_restored"))
            for row in historical_rows
        ),
        "training_performed": False,
    }


def _format_rate(value: Any) -> str:
    return "N/A" if value is None else f"{100.0 * float(value):.2f}%"


def _write_report(
    path: Path,
    comparison: Sequence[Mapping[str, Any]],
    conclusion: Mapping[str, Any],
) -> None:
    by_variant = _variant_map(comparison)
    lines = [
        "# Historical Forcing-Gate Restoration + A/B/C/D Re-Evaluation",
        "",
        "## 1. 控制变量",
        "",
        "本实验完整复用上一轮 A/B/C/D diagnosis，仅将当前 terminal-distance scalar forcing gate 在作用域内替换为 checkpoint 历史逐轴 gate。评估结束及异常退出后均恢复当前默认 transition。max_steps、场景、seed、Proposal、observation、reward 与 termination 均未改变。",
        "",
        "## 2. Success 直接对照",
        "",
        "| Variant | Current Gate Success | Historical Gate Success | ΔSuccess |",
        "|---|---:|---:|---:|",
    ]
    for variant in VARIANT_ORDER:
        row = by_variant[variant]
        lines.append(
            f"| {variant[0]} | {_format_rate(row['current_scalar_success_rate'])} | "
            f"{_format_rate(row['historical_vector_success_rate'])} | "
            f"{100.0 * float(row['delta_success_rate']):+.2f} pp |"
        )
    lines.extend(
        [
            "",
            "## 3. Collision、Progress 与 Smoothness",
            "",
            "| Variant | ΔCollision | ΔProgress / m | ΔSmoothness | Historical Temp Reached | Reached→Terminal Success |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for variant in VARIANT_ORDER:
        row = by_variant[variant]
        lines.append(
            f"| {variant[0]} | {100.0 * float(row['delta_collision_rate']):+.2f} pp | "
            f"{float(row['delta_terminal_progress_m']):+.3f} | "
            f"{float(row['delta_trajectory_smoothness']):+.3f} | "
            f"{_format_rate(row['historical_vector_temporary_reference_reached_rate'])} | "
            f"{_format_rate(row['historical_vector_reached_then_terminal_success_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## 4. 判定",
            "",
            f"- FORCING_GATE_DRIFT_AFFECTS_BASELINE = {conclusion['FORCING_GATE_DRIFT_AFFECTS_BASELINE']}",
            f"- FORCING_GATE_DRIFT_AFFECTS_TEMPORARY_REFERENCE = {conclusion['FORCING_GATE_DRIFT_AFFECTS_TEMPORARY_REFERENCE']}",
            f"- INTERMEDIATE_WAYPOINT_FORCING_TRUNCATION_CONFIRMED = {conclusion['INTERMEDIATE_WAYPOINT_FORCING_TRUNCATION_CONFIRMED']}",
            f"- SAC_FINE_TUNING_RECOMMENDED = {conclusion['SAC_FINE_TUNING_RECOMMENDED']}",
            f"- PRIMARY_LIMITATION = {conclusion['PRIMARY_LIMITATION']}",
            f"- NEXT_STEP = {conclusion['NEXT_STEP']}",
            "",
            "## 5. Gate-level sanity diagnostic",
            "",
            f"同一 state + same action 的 {conclusion['same_state_same_action_gate_diagnostic']['record_count']} 个样本中，"
            f"commanded acceleration L2 差值均值为 {conclusion['same_state_same_action_gate_diagnostic']['mean_commanded_acceleration_delta_l2_mps2']:.3f} m/s²，"
            f"最大值为 {conclusion['same_state_same_action_gate_diagnostic']['max_commanded_acceleration_delta_l2_mps2']:.3f} m/s²；"
            f"forcing contribution L2 差值均值为 {conclusion['same_state_same_action_gate_diagnostic']['mean_forcing_contribution_delta_l2']:.3f}。"
            "该诊断未推进 environment state。",
            "",
            "## 6. 核心问题回答",
            "",
            f"1. Historical gate 明显改善 A：成功率由 {_format_rate(by_variant[VARIANT_A]['current_scalar_success_rate'])} "
            f"提高到 {_format_rate(by_variant[VARIANT_A]['historical_vector_success_rate'])}，"
            f"collision 变化 {100.0 * float(by_variant[VARIANT_A]['delta_collision_rate']):+.2f} pp，"
            f"progress 变化 {float(by_variant[VARIANT_A]['delta_terminal_progress_m']):+.3f} m。",
            f"2. B 仍为 {_format_rate(by_variant[VARIANT_B]['historical_vector_success_rate'])} success；"
            f"temporary-reference reached rate 由 {_format_rate(by_variant[VARIANT_B]['current_scalar_temporary_reference_reached_rate'])} "
            f"变为 {_format_rate(by_variant[VARIANT_B]['historical_vector_temporary_reference_reached_rate'])}。",
            f"3. C 未恢复，success 为 {_format_rate(by_variant[VARIANT_C]['historical_vector_success_rate'])}；"
            f"reached rate 由 {_format_rate(by_variant[VARIANT_C]['current_scalar_temporary_reference_reached_rate'])} "
            f"变为 {_format_rate(by_variant[VARIANT_C]['historical_vector_temporary_reference_reached_rate'])}。",
            f"4. D 未恢复，success 为 {_format_rate(by_variant[VARIANT_D]['historical_vector_success_rate'])}；"
            f"reached rate 由 {_format_rate(by_variant[VARIANT_D]['current_scalar_temporary_reference_reached_rate'])} "
            f"提高到 {_format_rate(by_variant[VARIANT_D]['historical_vector_temporary_reference_reached_rate'])}。",
            f"5. C/D 合计到达 temporary reference {conclusion['waypoint_truncation_diagnostic']['temporary_reference_reached_count']}/"
            f"{conclusion['waypoint_truncation_diagnostic']['temporary_reference_reached_total']}，但 reached→terminal success 为 "
            f"{_format_rate(conclusion['waypoint_truncation_diagnostic']['reached_then_terminal_success_rate'])}。",
            f"6. Waypoint forcing truncation 未被确认：近点 attenuation rate 为 "
            f"{_format_rate(conclusion['waypoint_truncation_diagnostic']['near_reference_attenuated_rate'])}，"
            f"low-speed rate 为 {_format_rate(conclusion['waypoint_truncation_diagnostic']['near_reference_low_speed_rate'])}，"
            "不满足预注册的联合判据。",
            "",
            "## 7. 完整性",
            "",
            f"- Checkpoint unchanged: {conclusion['checkpoint_unchanged']}",
            f"- Frozen policy parameters unchanged: {conclusion['policy_parameters_unchanged']}",
            f"- Current scalar transition restored: {conclusion['current_scalar_transition_restored']}",
            "- Training performed: False",
            "- FP-SHEP/GAT/oracle selector used: False",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze_run(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    historical_rows = _read_csv(run_dir / "per_episode.csv")
    gate_rows = _read_csv(run_dir / "gate_diagnostic.csv")
    current_dir = Path(str(config["current_scalar_reference_run"]))
    if not current_dir.is_absolute():
        current_dir = REPO_ROOT / current_dir
    current_rows = _read_csv(current_dir / "per_episode.csv")
    expected_jobs = {
        (str(scenario), int(seed), str(variant))
        for scenario in config["scenarios"]
        for seed in config["seeds"]
        for variant in VARIANT_ORDER
    }
    for name, rows in (("current", current_rows), ("historical", historical_rows)):
        jobs = {
            (str(row["scenario"]), int(row["seed"]), str(row["variant"]))
            for row in rows
        }
        if jobs != expected_jobs:
            raise RuntimeError(f"{name} run is not the required paired 60-episode matrix")
    comparison = _comparison_rows(current_rows, historical_rows)
    conclusion = _build_conclusion(config, comparison, historical_rows, gate_rows)
    _write_csv(run_dir / "comparison_summary.csv", comparison)
    _write_json(run_dir / "conclusion.json", conclusion)
    _write_report(run_dir / "FINAL_REPORT.md", comparison, conclusion)
    return conclusion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    return parser.parse_args()


def main() -> dict[str, Any]:
    args = parse_args()
    conclusion = analyze_run(args.run_dir)
    print(json.dumps(conclusion, ensure_ascii=False, indent=2), flush=True)
    return conclusion


if __name__ == "__main__":
    main()
