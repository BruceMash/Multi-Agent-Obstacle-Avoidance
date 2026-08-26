"""Read-only lagged signed reference-to-turn response audit.

The prior binary reversal-window statistic was near saturation.  This audit
instead measures the signed change in executed yaw/pitch rate at fixed lags
after each accepted local-reference turn, separately for Original and Strong.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr


LAGS_S = tuple(round(0.1 * index, 1) for index in range(1, 11))
NUMERIC_EPS = 1.0e-6
BOOTSTRAP_REPLICATES = 500
BOOTSTRAP_SEED = 20260825


def wrap(value: float | np.ndarray) -> float | np.ndarray:
    return (value + np.pi) % (2.0 * np.pi) - np.pi


def signed_horizontal(task: np.ndarray, ref: np.ndarray) -> float:
    task_xy = task[:2]
    ref_xy = ref[:2]
    if np.linalg.norm(task_xy) <= 1.0e-12 or np.linalg.norm(ref_xy) <= 1.0e-12:
        return float("nan")
    cross = task_xy[0] * ref_xy[1] - task_xy[1] * ref_xy[0]
    return float(math.atan2(float(cross), float(np.dot(task_xy, ref_xy))))


def elevation(vector: np.ndarray) -> float:
    horizontal = float(np.linalg.norm(vector[:2]))
    if horizontal <= 1.0e-12 and abs(float(vector[2])) <= 1.0e-12:
        return float("nan")
    return float(math.atan2(float(vector[2]), horizontal))


def execution_rates(velocities: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    speed = np.linalg.norm(velocities, axis=1)
    valid = speed > NUMERIC_EPS
    yaw = np.full(len(speed), np.nan)
    pitch = np.full(len(speed), np.nan)
    yaw[valid] = np.arctan2(velocities[valid, 1], velocities[valid, 0])
    horizontal = np.linalg.norm(velocities[:, :2], axis=1)
    pitch[valid] = np.arctan2(velocities[valid, 2], horizontal[valid])
    yaw_rate = np.full(len(speed), np.nan)
    pitch_rate = np.full(len(speed), np.nan)
    consecutive = valid[1:] & valid[:-1]
    indexes = np.flatnonzero(consecutive) + 1
    yaw_rate[indexes] = wrap(yaw[indexes] - yaw[indexes - 1]) / dt
    pitch_rate[indexes] = (pitch[indexes] - pitch[indexes - 1]) / dt
    return yaw_rate, pitch_rate


def load_method_events(method: str, root: Path) -> list[dict[str, Any]]:
    record_paths = sorted(root.glob("GATRS_DEV_*.json"))
    if len(record_paths) != 100:
        raise RuntimeError(f"{method}: expected 100 records, found {len(record_paths)}")
    rows: list[dict[str, Any]] = []
    for record_path in record_paths:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        scenario_id = str(record["entry_identity"]["scenario_id"])
        stage = str(record["entry_identity"]["stage"])
        trajectory_name = str(record["trajectory_file"])
        trajectory = np.load(root / trajectory_name)
        positions = np.asarray(trajectory["positions"], dtype=float)
        velocities = np.asarray(trajectory["velocities"], dtype=float)
        dt = float(trajectory["dt"])
        if not np.isclose(dt, 0.1, rtol=0.0, atol=0.0):
            raise RuntimeError(f"{scenario_id}: unexpected dt {dt}")
        if "terminal_goals" in trajectory.files:
            terminal_goals = np.asarray(trajectory["terminal_goals"], dtype=float)
        else:
            terminal_goals = np.repeat(
                np.asarray(record["episode"].get("terminal_goals", []), dtype=float)[None, :, :],
                len(positions),
                axis=0,
            ) if record["episode"].get("terminal_goals") else None
            if terminal_goals is None:
                # Original CRT archive stores command trajectories but not terminal goals;
                # the initial event's old active goal is the terminal goal for
                # that same agent.
                goals = [
                    next(
                        event["old_active_goal"]
                        for event in record["events"]
                        if int(event["agent_id"]) == agent_id
                        and str(event["event"]) == "INITIAL_SELECTION"
                    )
                    for agent_id in range(3)
                ]
                terminal_goals = np.repeat(np.asarray(goals, dtype=float)[None, :, :], len(positions), axis=0)
        signals = [execution_rates(velocities[:, agent, :], dt) for agent in range(3)]
        by_agent: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for event in record["events"]:
            if bool(event.get("goal_changed")):
                by_agent[int(event["agent_id"])].append(event)
        for agent_id in range(3):
            history: list[dict[str, float]] = []
            yaw_rate, pitch_rate = signals[agent_id]
            for event in sorted(by_agent[agent_id], key=lambda item: (int(item["step"]), str(item["event"]))):
                if str(event.get("new_active_goal_type")) != "reference":
                    history.clear()
                    continue
                step = min(max(int(event["step"]), 0), len(positions) - 1)
                ego = positions[step, agent_id]
                reference = np.asarray(event["new_active_goal"], dtype=float)
                terminal = terminal_goals[step, agent_id]
                ref_vector = reference - ego
                task_vector = terminal - ego
                alpha = signed_horizontal(task_vector, ref_vector)
                beta = elevation(ref_vector) - elevation(task_vector)
                if history:
                    delta_alpha = float(wrap(alpha - history[-1]["alpha"]))
                    delta_beta = float(beta - history[-1]["beta"])
                    pre_index = step - 1
                    if pre_index >= 0:
                        for lag_s in LAGS_S:
                            target = step + int(round(lag_s / dt))
                            if target >= len(positions):
                                continue
                            for axis, reference_turn, rate_signal in (
                                ("yaw", delta_alpha, yaw_rate),
                                ("pitch", delta_beta, pitch_rate),
                            ):
                                before = rate_signal[pre_index]
                                after = rate_signal[target]
                                if not (np.isfinite(before) and np.isfinite(after)):
                                    continue
                                response = float(after - before)
                                sign_eligible = bool(
                                    abs(reference_turn) > NUMERIC_EPS
                                    and abs(response) > NUMERIC_EPS
                                )
                                rows.append(
                                    {
                                        "method": method,
                                        "scenario_id": scenario_id,
                                        "stage": stage,
                                        "agent_id": agent_id,
                                        "event_step": step,
                                        "event_type": str(event["event"]),
                                        "axis": axis,
                                        "lag_s": lag_s,
                                        "reference_turn_rad": reference_turn,
                                        "executed_rate_before_radps": float(before),
                                        "executed_rate_after_radps": float(after),
                                        "executed_response_radps": response,
                                        "sign_eligible": sign_eligible,
                                        "sign_agreement": bool(
                                            sign_eligible and reference_turn * response > 0.0
                                        ),
                                    }
                                )
                history.append({"alpha": alpha, "beta": beta})
    return rows


def bootstrap_rho(rows: list[dict[str, Any]]) -> dict[str, Any]:
    x = np.asarray([float(row["reference_turn_rad"]) for row in rows], dtype=float)
    y = np.asarray([float(row["executed_response_radps"]) for row in rows], dtype=float)
    rho = float(spearmanr(x, y).statistic)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row["scenario_id"])].append(index)
    indexes = {key: np.asarray(value, dtype=int) for key, value in groups.items()}
    names = sorted(indexes)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        chosen = rng.choice(names, size=len(names), replace=True)
        sample = np.concatenate([indexes[str(name)] for name in chosen])
        value = float(spearmanr(x[sample], y[sample]).statistic)
        if np.isfinite(value):
            values.append(value)
    ci = np.quantile(values, [0.025, 0.975])
    return {
        "rho": rho,
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "bootstrap_replicates": len(values),
        "episode_clusters": len(names),
    }


def summarize(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for method in ("Original", "Strong"):
        for axis in ("yaw", "pitch"):
            for lag_s in LAGS_S:
                members = [
                    row for row in rows
                    if row["method"] == method and row["axis"] == axis and row["lag_s"] == lag_s
                ]
                stats = bootstrap_rho(members)
                eligible = [row for row in members if row["sign_eligible"]]
                sign_agreement = (
                    float(np.mean([row["sign_agreement"] for row in eligible]))
                    if eligible else None
                )
                stage_rho: dict[str, float | None] = {}
                for stage in ("stage_1", "stage_2", "stage_3", "stage_4"):
                    subset = [row for row in members if row["stage"] == stage]
                    stage_rho[stage] = (
                        float(spearmanr(
                            [row["reference_turn_rad"] for row in subset],
                            [row["executed_response_radps"] for row in subset],
                        ).statistic)
                        if len(subset) >= 3 else None
                    )
                table.append(
                    {
                        "method": method,
                        "axis": axis,
                        "lag_s": lag_s,
                        "event_count": len(members),
                        "rho": stats["rho"],
                        "rho_ci95_low": stats["ci_low"],
                        "rho_ci95_high": stats["ci_high"],
                        "sign_agreement_count": sum(bool(row["sign_agreement"]) for row in eligible),
                        "sign_eligible_count": len(eligible),
                        "sign_agreement_rate": sign_agreement,
                        "stage_1_rho": stage_rho["stage_1"],
                        "stage_2_rho": stage_rho["stage_2"],
                        "stage_3_rho": stage_rho["stage_3"],
                        "stage_4_rho": stage_rho["stage_4"],
                    }
                )

    summary: dict[str, Any] = {
        "schema_version": "lagged_reference_execution_coupling_v1",
        "lags_s": list(LAGS_S),
        "bootstrap": {
            "type": "episode_cluster_resampling_with_pooled_spearman_recomputation",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
        },
        "methods": {},
    }
    rank = {"NONE": 0, "WEAK": 1, "MODERATE": 2, "STRONG": 3}
    for method in ("Original", "Strong"):
        method_summary: dict[str, Any] = {}
        classifications: list[str] = []
        for axis in ("yaw", "pitch"):
            members = [row for row in table if row["method"] == method and row["axis"] == axis]
            peak = max(members, key=lambda row: float(row["rho"]))
            positive_stages = sum(
                1 for key in ("stage_1_rho", "stage_2_rho", "stage_3_rho", "stage_4_rho")
                if peak[key] is not None and float(peak[key]) > 0.0
            )
            if (
                peak["rho"] >= 0.30
                and peak["rho_ci95_low"] >= 0.20
                and (peak["sign_agreement_rate"] or 0.0) >= 0.60
                and positive_stages >= 3
            ):
                label = "STRONG"
            elif (
                peak["rho"] >= 0.15
                and peak["rho_ci95_low"] >= 0.10
                and (peak["sign_agreement_rate"] or 0.0) >= 0.55
                and positive_stages >= 3
            ):
                label = "MODERATE"
            elif peak["rho"] > 0.0 or (peak["sign_agreement_rate"] or 0.0) > 0.50:
                label = "WEAK"
            else:
                label = "NONE"
            classifications.append(label)
            method_summary[axis] = {
                "peak_positive_correlation": peak["rho"],
                "lag_at_peak_s": peak["lag_s"],
                "ci95": [peak["rho_ci95_low"], peak["rho_ci95_high"]],
                "response_sign_agreement_at_peak": peak["sign_agreement_rate"],
                "positive_stage_count_at_peak": positive_stages,
                "classification": label,
            }
        overall = max(classifications, key=lambda label: rank[label])
        method_summary["overall_classification"] = overall
        summary["methods"][method] = method_summary

    original_peak = max(
        summary["methods"]["Original"][axis]["lag_at_peak_s"]
        for axis in ("yaw", "pitch")
    )
    strong_peak = max(
        summary["methods"]["Strong"][axis]["lag_at_peak_s"]
        for axis in ("yaw", "pitch")
    )
    summary["ORIGINAL_REFERENCE_TO_EXECUTION_COUPLING"] = summary["methods"]["Original"]["overall_classification"]
    summary["STRONG_REFERENCE_TO_EXECUTION_COUPLING"] = summary["methods"]["Strong"]["overall_classification"]
    summary["STRONG_RESPONSE_DELAY_RELATIVE_TO_ORIGINAL"] = (
        "YES" if strong_peak - original_peak >= 0.2 else "NO"
    )
    original_authorizes = summary["ORIGINAL_REFERENCE_TO_EXECUTION_COUPLING"] in {"STRONG", "MODERATE"}
    meaningful_positive = any(
        summary["methods"]["Original"][axis]["response_sign_agreement_at_peak"] >= 0.55
        and summary["methods"]["Original"][axis]["peak_positive_correlation"] >= 0.15
        for axis in ("yaw", "pitch")
    )
    selected = "DCTB" if original_authorizes and meaningful_positive else "TURN_DIRECTION_HYSTERESIS"
    summary["SELECTED_BRANCH"] = selected
    summary["branch_rule"] = (
        "DCTB only when Original coupling is MODERATE/STRONG and signed reference turn "
        "meaningfully predicts the subsequent executed response; otherwise lower-level hysteresis."
    )
    return table, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--strong-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = load_method_events("Original", args.original_root)
    rows.extend(load_method_events("Strong", args.strong_root))
    table, summary = summarize(rows)
    write_csv(args.output_root / "LAGGED_REFERENCE_EXECUTION_COUPLING.csv", table)
    (args.output_root / "LAGGED_COUPLING_SUMMARY.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    branch = {
        "schema_version": "residual_zigzag_branch_decision_v1",
        "ORIGINAL_REFERENCE_TO_EXECUTION_COUPLING": summary["ORIGINAL_REFERENCE_TO_EXECUTION_COUPLING"],
        "STRONG_REFERENCE_TO_EXECUTION_COUPLING": summary["STRONG_REFERENCE_TO_EXECUTION_COUPLING"],
        "STRONG_RESPONSE_DELAY_RELATIVE_TO_ORIGINAL": summary["STRONG_RESPONSE_DELAY_RELATIVE_TO_ORIGINAL"],
        "SELECTED_BRANCH": summary["SELECTED_BRANCH"],
        "exactly_one_branch": True,
        "performance_scenarios_observed_before_decision": 0,
    }
    (args.output_root / "RESIDUAL_ZIGZAG_BRANCH_DECISION.json").write_text(
        json.dumps(branch, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(branch, indent=2))


if __name__ == "__main__":
    main()
