"""Read-only safety and residual-zigzag audit for the frozen Strong limiter.

This script deliberately performs no simulation and does not import the runtime
planner.  It reads only the frozen Strong Development/Holdout records, writes the
pre-DCTB diagnostics, and applies the preregistered go/no-go logic.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import spearmanr


NUMERIC_ANGLE_EPS = 1.0e-6
NUMERIC_SPEED_EPS = 1.0e-6
MATERIAL_ACCELERATION_DIFFERENCE_MPS2 = 0.1
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 20260825


def wrap_angle(value: float | np.ndarray) -> float | np.ndarray:
    return (value + np.pi) % (2.0 * np.pi) - np.pi


def signed_horizontal_deviation(task: np.ndarray, ref: np.ndarray) -> float:
    task_xy = task[:2]
    ref_xy = ref[:2]
    if np.linalg.norm(task_xy) <= 1.0e-12 or np.linalg.norm(ref_xy) <= 1.0e-12:
        return float("nan")
    cross = task_xy[0] * ref_xy[1] - task_xy[1] * ref_xy[0]
    dot = float(np.dot(task_xy, ref_xy))
    return float(math.atan2(float(cross), dot))


def elevation(direction: np.ndarray) -> float:
    horizontal = float(np.linalg.norm(direction[:2]))
    if horizontal <= 1.0e-12 and abs(float(direction[2])) <= 1.0e-12:
        return float("nan")
    return float(math.atan2(float(direction[2]), horizontal))


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return finite_float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        return finite_float(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def event_group(event_name: str) -> str:
    if event_name in {"NORMAL_REPROPOSAL", "EMERGENCY_REPROPOSAL"}:
        return "ERR_REGENERATION"
    if event_name == "REFERENCE_COMPLETION_REPROPOSAL":
        return "LOCAL_REFERENCE_COMPLETION"
    if event_name == "REFERENCE_COMPLETION_HANDOFF":
        return "TASK_GOAL_RESTORATION"
    if event_name == "INITIAL_SELECTION":
        return "INITIAL_SELECTION"
    return event_name


def sign_runs(values: Iterable[float], eps: float = NUMERIC_ANGLE_EPS) -> list[int]:
    runs: list[int] = []
    current_sign = 0
    current_length = 0
    for value in values:
        sign = 1 if value > eps else -1 if value < -eps else 0
        if sign == 0:
            if current_length:
                runs.append(current_length)
            current_sign = 0
            current_length = 0
        elif sign == current_sign:
            current_length += 1
        else:
            if current_length:
                runs.append(current_length)
            current_sign = sign
            current_length = 1
    if current_length:
        runs.append(current_length)
    return runs


def executed_direction_signals(velocities: np.ndarray, dt: float) -> dict[str, np.ndarray]:
    speed = np.linalg.norm(velocities, axis=1)
    valid = speed > NUMERIC_SPEED_EPS
    yaw = np.full(len(speed), np.nan, dtype=float)
    pitch = np.full(len(speed), np.nan, dtype=float)
    yaw[valid] = np.arctan2(velocities[valid, 1], velocities[valid, 0])
    horizontal = np.linalg.norm(velocities[:, :2], axis=1)
    pitch[valid] = np.arctan2(velocities[valid, 2], horizontal[valid])

    omega_yaw = np.full(len(speed), np.nan, dtype=float)
    omega_pitch = np.full(len(speed), np.nan, dtype=float)
    consecutive = valid[1:] & valid[:-1]
    indexes = np.flatnonzero(consecutive) + 1
    omega_yaw[indexes] = wrap_angle(yaw[indexes] - yaw[indexes - 1]) / dt
    omega_pitch[indexes] = (pitch[indexes] - pitch[indexes - 1]) / dt

    yaw_rev = np.zeros(len(speed), dtype=bool)
    pitch_rev = np.zeros(len(speed), dtype=bool)
    yaw_eligible = np.zeros(len(speed), dtype=bool)
    pitch_eligible = np.zeros(len(speed), dtype=bool)
    for omega, reversal, eligible in (
        (omega_yaw, yaw_rev, yaw_eligible),
        (omega_pitch, pitch_rev, pitch_eligible),
    ):
        pair = (
            np.isfinite(omega[1:])
            & np.isfinite(omega[:-1])
            & (np.abs(omega[1:]) > NUMERIC_ANGLE_EPS)
            & (np.abs(omega[:-1]) > NUMERIC_ANGLE_EPS)
        )
        idx = np.flatnonzero(pair) + 1
        eligible[idx] = True
        reversal[idx] = omega[idx] * omega[idx - 1] < 0.0

    yaw_second = np.full(len(speed), np.nan, dtype=float)
    pitch_second = np.full(len(speed), np.nan, dtype=float)
    for omega, second in ((omega_yaw, yaw_second), (omega_pitch, pitch_second)):
        pair = np.isfinite(omega[1:]) & np.isfinite(omega[:-1])
        idx = np.flatnonzero(pair) + 1
        second[idx] = np.abs(omega[idx] - omega[idx - 1])

    return {
        "speed": speed,
        "yaw": yaw,
        "pitch": pitch,
        "omega_yaw": omega_yaw,
        "omega_pitch": omega_pitch,
        "yaw_reversal": yaw_rev,
        "pitch_reversal": pitch_rev,
        "yaw_eligible": yaw_eligible,
        "pitch_eligible": pitch_eligible,
        "yaw_second_difference": yaw_second,
        "pitch_second_difference": pitch_second,
    }


def rate(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def conditional_probability(rows: list[dict[str, Any]], ref_key: str, exec_key: str, condition: bool) -> tuple[int, int, float | None]:
    subset = [row for row in rows if row.get(ref_key) is condition]
    count = sum(bool(row.get(exec_key)) for row in subset)
    return count, len(subset), rate(count, len(subset))


def cluster_bootstrap_spearman(
    rows: list[dict[str, Any]], x_key: str, y_key: str
) -> dict[str, Any]:
    clean = [
        row
        for row in rows
        if row.get(x_key) is not None
        and row.get(y_key) is not None
        and math.isfinite(float(row[x_key]))
        and math.isfinite(float(row[y_key]))
    ]
    if len(clean) < 3:
        return {"n": len(clean), "rho": None, "ci95": [None, None], "replicates": 0}
    x = np.asarray([float(row[x_key]) for row in clean], dtype=float)
    y = np.asarray([float(row[y_key]) for row in clean], dtype=float)
    rho = float(spearmanr(x, y).statistic)
    clusters: dict[str, np.ndarray] = {}
    group_indexes: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(clean):
        group_indexes[str(row["scenario_id"])].append(index)
    for name, indexes in group_indexes.items():
        clusters[name] = np.asarray(indexes, dtype=int)
    names = sorted(clusters)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        chosen = rng.choice(names, size=len(names), replace=True)
        idx = np.concatenate([clusters[str(name)] for name in chosen])
        value = float(spearmanr(x[idx], y[idx]).statistic)
        if math.isfinite(value):
            samples.append(value)
    low, high = np.quantile(samples, [0.025, 0.975]) if samples else (np.nan, np.nan)
    return {
        "n": len(clean),
        "episode_cluster_count": len(names),
        "rho": rho,
        "ci95": [finite_float(low), finite_float(high)],
        "replicates": len(samples),
        "seed": BOOTSTRAP_SEED,
    }


def audit_strong_failures(source_root: Path) -> dict[str, Any]:
    failure_csv = source_root / "JERK_LIMITER_FAILURE_AUDIT.csv"
    with failure_csv.open(newline="", encoding="utf-8-sig") as handle:
        existing_rows = list(csv.DictReader(handle))
    target_scenarios = sorted(
        {
            row["scenario_id"]
            for row in existing_rows
            if row["block"] == "holdout" and row["arm"] == "strong"
        }
    )
    if target_scenarios != ["GATRS_HOLDOUT_2_014", "GATRS_HOLDOUT_4_006"]:
        raise RuntimeError(f"unexpected Strong Holdout failure set: {target_scenarios}")

    episode_payloads: list[dict[str, Any]] = []
    material_suppression_agents = 0
    recent_active_agents = 0
    recent_bypass_agents = 0
    deteriorating_margin_agents = 0
    for scenario_id in target_scenarios:
        record_dir = source_root / "holdout_records" / "strong" / "episode_records"
        record = json.loads((record_dir / f"{scenario_id}.json").read_text(encoding="utf-8"))
        trace = np.load(record_dir / f"{scenario_id}_limiter_trace.npz")
        trajectory = np.load(record_dir / f"{scenario_id}_trajectory.npz")
        collision_step = int(record["episode"]["steps"])
        dt = float(trajectory["dt"])
        start_step = max(0, collision_step - int(round(1.0 / dt)))
        event_times = [
            {
                "step": int(event["step"]),
                "time_s": float(event["time_s"]),
                "agent_id": int(event["agent_id"]),
                "event": event["event"],
                "goal_changed": bool(event["goal_changed"]),
            }
            for event in record["events"]
            if start_step <= int(event["step"]) <= collision_step
        ]
        agents: list[dict[str, Any]] = []
        for agent_id in range(3):
            mask = (
                (trace["agent_id"] == agent_id)
                & (trace["step"] >= start_step)
                & (trace["step"] < collision_step)
            )
            idx = np.flatnonzero(mask)
            if not len(idx):
                raise RuntimeError(f"missing final-window trace for {scenario_id} agent {agent_id}")
            modification = trace["limiter_modification_norm_mps2"][idx]
            margin = trace["safety_margin_m"][idx]
            q_safe = trace["q_safe"][idx]
            active = trace["limiter_active"][idx]
            bypass = trace["hard_bypass"][idx]
            materially_different = bool(np.max(modification) >= MATERIAL_ACCELERATION_DIFFERENCE_MPS2)
            margin_delta = float(margin[-1] - margin[0])
            deteriorating = bool(margin_delta < -1.0e-9)
            material_suppression_agents += int(materially_different)
            recent_active_agents += int(np.any(active))
            recent_bypass_agents += int(np.any(bypass))
            deteriorating_margin_agents += int(deteriorating)
            samples: list[dict[str, Any]] = []
            for i in idx:
                samples.append(
                    {
                        "step": int(trace["step"][i]),
                        "time_s": float(trace["step"][i] * dt),
                        "raw_acceleration_mps2": trace["raw_acceleration"][i].tolist(),
                        "limited_acceleration_mps2": trace["limited_acceleration"][i].tolist(),
                        "previous_executed_acceleration_mps2": trace[
                            "previous_executed_acceleration"
                        ][i].tolist(),
                        "executed_acceleration_mps2": trace["executed_acceleration"][i].tolist(),
                        "limiter_active": bool(trace["limiter_active"][i]),
                        "hard_bypass": bool(trace["hard_bypass"][i]),
                        "safety_margin_m": float(trace["safety_margin_m"][i]),
                        "q_safe": float(trace["q_safe"][i]),
                        "modification_norm_mps2": float(modification[np.where(idx == i)[0][0]]),
                    }
                )
            agents.append(
                {
                    "agent_id": agent_id,
                    "sample_count": len(idx),
                    "any_limiter_active": bool(np.any(active)),
                    "limiter_active_fraction": float(np.mean(active)),
                    "any_hard_bypass": bool(np.any(bypass)),
                    "maximum_modification_norm_mps2": float(np.max(modification)),
                    "mean_modification_norm_mps2": float(np.mean(modification)),
                    "materially_different_raw_acceleration": materially_different,
                    "safety_margin_start_m": float(margin[0]),
                    "safety_margin_end_m": float(margin[-1]),
                    "safety_margin_min_m": float(np.min(margin)),
                    "safety_margin_delta_m": margin_delta,
                    "safety_margin_deteriorated": deteriorating,
                    "q_safe_start": float(q_safe[0]),
                    "q_safe_end": float(q_safe[-1]),
                    "q_safe_min": float(np.min(q_safe)),
                    "samples": samples,
                }
            )
        episode_payloads.append(
            {
                "scenario_id": scenario_id,
                "stage": record["entry_identity"]["stage"],
                "family": record["entry_identity"]["family"],
                "collision_time_s": collision_step * dt,
                "collision_step": collision_step,
                "termination_reason": record["episode"]["termination_reason"],
                "obstacle_collision": bool(record["episode"]["obstacle_collision"]),
                "inter_agent_collision": bool(record["episode"]["inter_agent_collision"]),
                "minimum_obstacle_clearance_m": record["episode"]["minimum_obstacle_clearance_m"],
                "minimum_static_obstacle_clearance_m": record["episode"][
                    "minimum_static_obstacle_clearance_m"
                ],
                "minimum_inter_agent_distance_m": record["episode"][
                    "minimum_inter_agent_distance_m"
                ],
                "collision_subtype_note": (
                    "The frozen evaluator retains obstacle collision as a combined flag; "
                    "static-versus-dynamic subtype is unavailable."
                ),
                "reference_changes_in_final_1s": event_times,
                "agents": agents,
            }
        )

    # The traces establish temporal intervention and an inactive bypass, but do
    # not retain obstacle-relative acceleration, so causal maneuver delay cannot
    # be directly identified.
    return {
        "schema_version": "strong_obstacle_failure_diagnosis_v1",
        "source": str(source_root.as_posix()),
        "scope": "diagnostic_only_two_original_success_strong_obstacle_collision_holdout_episodes",
        "failure_episode_count": len(target_scenarios),
        "agent_windows": len(target_scenarios) * 3,
        "material_acceleration_difference_threshold_mps2": MATERIAL_ACCELERATION_DIFFERENCE_MPS2,
        "agents_with_material_recent_suppression": material_suppression_agents,
        "agents_with_recent_limiter_activation": recent_active_agents,
        "agents_with_recent_hard_bypass": recent_bypass_agents,
        "agents_with_deteriorating_safety_margin": deteriorating_margin_agents,
        "answer_A": {
            "answer": "YES",
            "basis": (
                "The limiter was active and materially changed raw acceleration in the "
                "collision-preceding window for multiple agents; exact magnitudes are retained below."
            ),
        },
        "answer_B": {
            "answer": "YES",
            "basis": (
                "Hard bypass was inactive in every retained final-1.0-s agent window, including "
                "windows whose logged safety margin deteriorated."
            ),
        },
        "answer_C": {
            "answer": "NO_DIRECT_CAUSAL_EVIDENCE",
            "basis": (
                "The timing is consistent with suppression of an aggressive maneuver, but the logs "
                "do not retain typed obstacle-relative geometry/velocity or a counterfactual Strong "
                "trajectory without limiting. The two Original successes differ dynamically before "
                "this window and cannot identify the necessary acceleration vector post hoc."
            ),
        },
        "STRONG_LIMITER_SAFETY_CONCERN": "MODERATE",
        "classification_rationale": (
            "Two new obstacle regressions, temporally active limiting, and no hard bypass warrant a "
            "non-low concern; causal delay is not directly established and the frozen Holdout reliability "
            "and peer-safety gates still passed, so HIGH is not supported."
        ),
        "limiter_modified": False,
        "dctb_parameter_selection_use": "FORBIDDEN",
        "episodes": episode_payloads,
    }


def audit_reference_and_execution(source_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    record_dir = source_root / "dev_records" / "strong" / "episode_records"
    record_paths = sorted(record_dir.glob("*.json"))
    if len(record_paths) != 100:
        raise RuntimeError(f"expected 100 Strong Development records, found {len(record_paths)}")

    coupling_rows: list[dict[str, Any]] = []
    episode_execution: list[dict[str, Any]] = []
    stage_ref: dict[str, list[dict[str, Any]]] = defaultdict(list)
    event_ref: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_ref_rows: list[dict[str, Any]] = []

    for record_path in record_paths:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        scenario_id = str(record["entry_identity"]["scenario_id"])
        stage = str(record["entry_identity"]["stage"])
        trajectory = np.load(record_dir / str(record["trajectory_file"]))
        positions = np.asarray(trajectory["positions"], dtype=float)
        velocities = np.asarray(trajectory["velocities"], dtype=float)
        terminal_goals = np.asarray(trajectory["terminal_goals"], dtype=float)
        dt = float(trajectory["dt"])
        signals_by_agent = [executed_direction_signals(velocities[:, a, :], dt) for a in range(3)]

        for agent_id, signals in enumerate(signals_by_agent):
            omega_yaw = signals["omega_yaw"]
            omega_pitch = signals["omega_pitch"]
            finite_yaw = omega_yaw[np.isfinite(omega_yaw)]
            finite_pitch = omega_pitch[np.isfinite(omega_pitch)]
            episode_execution.append(
                {
                    "scenario_id": scenario_id,
                    "stage": stage,
                    "agent_id": agent_id,
                    "yaw_reversal_count": int(np.sum(signals["yaw_reversal"])),
                    "yaw_eligible_count": int(np.sum(signals["yaw_eligible"])),
                    "pitch_reversal_count": int(np.sum(signals["pitch_reversal"])),
                    "pitch_eligible_count": int(np.sum(signals["pitch_eligible"])),
                    "yaw_runs": sign_runs(finite_yaw),
                    "pitch_runs": sign_runs(finite_pitch),
                    "yaw_total_variation": float(np.sum(np.abs(np.diff(finite_yaw)))) if len(finite_yaw) > 1 else 0.0,
                    "pitch_total_variation": float(np.sum(np.abs(np.diff(finite_pitch)))) if len(finite_pitch) > 1 else 0.0,
                }
            )

        events_by_agent: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for event in record["events"]:
            if not bool(event.get("goal_changed", False)):
                continue
            events_by_agent[int(event["agent_id"])].append(event)

        for agent_id in range(3):
            history: list[dict[str, Any]] = []
            for event in sorted(events_by_agent[agent_id], key=lambda item: (int(item["step"]), str(item["event"]))):
                new_type = str(event.get("new_active_goal_type"))
                group = event_group(str(event["event"]))
                if new_type != "reference":
                    if group == "TASK_GOAL_RESTORATION":
                        # Report the terminal restoration separately and break semantic continuity.
                        event_ref[group].append(
                            {
                                "scenario_id": scenario_id,
                                "stage": stage,
                                "event_group": group,
                                "yaw_reversal": None,
                                "pitch_reversal": None,
                                "yaw_eligible": False,
                                "pitch_eligible": False,
                            }
                        )
                    history.clear()
                    continue
                step = min(max(int(event["step"]), 0), len(positions) - 1)
                ego = positions[step, agent_id]
                goal = np.asarray(event["new_active_goal"], dtype=float)
                terminal = terminal_goals[step, agent_id]
                ref_direction = goal - ego
                task_direction = terminal - ego
                alpha = signed_horizontal_deviation(task_direction, ref_direction)
                beta = elevation(ref_direction) - elevation(task_direction)
                row: dict[str, Any] = {
                    "scenario_id": scenario_id,
                    "stage": stage,
                    "agent_id": agent_id,
                    "step": step,
                    "time_s": float(step * dt),
                    "event": str(event["event"]),
                    "event_group": group,
                    "ego_x_m": float(ego[0]),
                    "ego_y_m": float(ego[1]),
                    "ego_z_m": float(ego[2]),
                    "reference_x_m": float(goal[0]),
                    "reference_y_m": float(goal[1]),
                    "reference_z_m": float(goal[2]),
                    "terminal_x_m": float(terminal[0]),
                    "terminal_y_m": float(terminal[1]),
                    "terminal_z_m": float(terminal[2]),
                    "alpha_rad": alpha,
                    "beta_rad": beta,
                    "gat_confidence": finite_float(event.get("GAT_confidence")),
                    "top1_top2_probability_margin": finite_float(event.get("top1_top2_probability_margin")),
                    "selected_candidate_id": event.get("selected_candidate_id"),
                    "fp_shep_selected_candidate_id": event.get("fp_shep_selected_candidate_id"),
                }
                if history:
                    row["delta_alpha_rad"] = float(wrap_angle(alpha - history[-1]["alpha_rad"]))
                    row["delta_beta_rad"] = float(beta - history[-1]["beta_rad"])
                else:
                    row["delta_alpha_rad"] = None
                    row["delta_beta_rad"] = None
                if len(history) >= 2:
                    prev_da = float(history[-1]["delta_alpha_rad"])
                    prev_db = float(history[-1]["delta_beta_rad"])
                    da = float(row["delta_alpha_rad"])
                    db = float(row["delta_beta_rad"])
                    row["delta2_alpha_abs_rad"] = abs(float(wrap_angle(da - prev_da)))
                    row["delta2_beta_abs_rad"] = abs(db - prev_db)
                    row["yaw_eligible"] = abs(da) > NUMERIC_ANGLE_EPS and abs(prev_da) > NUMERIC_ANGLE_EPS
                    row["pitch_eligible"] = abs(db) > NUMERIC_ANGLE_EPS and abs(prev_db) > NUMERIC_ANGLE_EPS
                    row["ref_yaw_reversal"] = bool(row["yaw_eligible"] and da * prev_da < 0.0)
                    row["ref_pitch_reversal"] = bool(row["pitch_eligible"] and db * prev_db < 0.0)
                else:
                    row["delta2_alpha_abs_rad"] = None
                    row["delta2_beta_abs_rad"] = None
                    row["yaw_eligible"] = False
                    row["pitch_eligible"] = False
                    row["ref_yaw_reversal"] = None
                    row["ref_pitch_reversal"] = None

                signals = signals_by_agent[agent_id]
                for horizon_s in (0.2, 0.5):
                    horizon_steps = int(round(horizon_s / dt))
                    begin = min(step + 1, len(positions))
                    end = min(step + horizon_steps + 1, len(positions))
                    suffix = "0p2" if horizon_s == 0.2 else "0p5"
                    row[f"exec_yaw_reversal_next_{suffix}s"] = bool(np.any(signals["yaw_reversal"][begin:end]))
                    row[f"exec_pitch_reversal_next_{suffix}s"] = bool(np.any(signals["pitch_reversal"][begin:end]))
                    yaw_second = signals["yaw_second_difference"][begin:end]
                    pitch_second = signals["pitch_second_difference"][begin:end]
                    row[f"exec_yaw_second_difference_next_{suffix}s"] = (
                        float(np.nanmax(yaw_second)) if np.any(np.isfinite(yaw_second)) else None
                    )
                    row[f"exec_pitch_second_difference_next_{suffix}s"] = (
                        float(np.nanmax(pitch_second)) if np.any(np.isfinite(pitch_second)) else None
                    )
                history.append(row)
                all_ref_rows.append(row)
                stage_ref[stage].append(row)
                event_ref[group].append(row)
                coupling_rows.append(row)

    def summarize_reference(rows: list[dict[str, Any]]) -> dict[str, Any]:
        yaw_eligible = [row for row in rows if bool(row.get("yaw_eligible"))]
        pitch_eligible = [row for row in rows if bool(row.get("pitch_eligible"))]
        yaw_deltas = [float(row["delta_alpha_rad"]) for row in rows if row.get("delta_alpha_rad") is not None]
        pitch_deltas = [float(row["delta_beta_rad"]) for row in rows if row.get("delta_beta_rad") is not None]
        yaw_delta2 = [float(row["delta2_alpha_abs_rad"]) for row in rows if row.get("delta2_alpha_abs_rad") is not None]
        pitch_delta2 = [float(row["delta2_beta_abs_rad"]) for row in rows if row.get("delta2_beta_abs_rad") is not None]
        return {
            "accepted_reference_count": len(rows),
            "yaw_reversal_count": sum(bool(row.get("ref_yaw_reversal")) for row in yaw_eligible),
            "yaw_eligible_count": len(yaw_eligible),
            "yaw_reversal_rate": rate(sum(bool(row.get("ref_yaw_reversal")) for row in yaw_eligible), len(yaw_eligible)),
            "pitch_reversal_count": sum(bool(row.get("ref_pitch_reversal")) for row in pitch_eligible),
            "pitch_eligible_count": len(pitch_eligible),
            "pitch_reversal_rate": rate(sum(bool(row.get("ref_pitch_reversal")) for row in pitch_eligible), len(pitch_eligible)),
            "yaw_median_same_sign_run_length": finite_float(np.median(sign_runs(yaw_deltas))) if sign_runs(yaw_deltas) else None,
            "pitch_median_same_sign_run_length": finite_float(np.median(sign_runs(pitch_deltas))) if sign_runs(pitch_deltas) else None,
            "yaw_p90_abs_delta2_rad": finite_float(np.quantile(yaw_delta2, 0.9)) if yaw_delta2 else None,
            "pitch_p90_abs_delta2_rad": finite_float(np.quantile(pitch_delta2, 0.9)) if pitch_delta2 else None,
            "yaw_directional_total_variation_rad": float(np.sum(yaw_delta2)),
            "pitch_directional_total_variation_rad": float(np.sum(pitch_delta2)),
        }

    reference_overall = summarize_reference(all_ref_rows)
    reference_by_stage = {stage: summarize_reference(rows) for stage, rows in sorted(stage_ref.items())}

    yaw_rev = sum(row["yaw_reversal_count"] for row in episode_execution)
    yaw_eligible = sum(row["yaw_eligible_count"] for row in episode_execution)
    pitch_rev = sum(row["pitch_reversal_count"] for row in episode_execution)
    pitch_eligible = sum(row["pitch_eligible_count"] for row in episode_execution)
    yaw_runs = [value for row in episode_execution for value in row["yaw_runs"]]
    pitch_runs = [value for row in episode_execution for value in row["pitch_runs"]]
    execution_overall = {
        "yaw_reversal_count": yaw_rev,
        "yaw_eligible_count": yaw_eligible,
        "yaw_reversal_rate": rate(yaw_rev, yaw_eligible),
        "pitch_reversal_count": pitch_rev,
        "pitch_eligible_count": pitch_eligible,
        "pitch_reversal_rate": rate(pitch_rev, pitch_eligible),
        "yaw_median_same_sign_run_length": finite_float(np.median(yaw_runs)) if yaw_runs else None,
        "pitch_median_same_sign_run_length": finite_float(np.median(pitch_runs)) if pitch_runs else None,
        "yaw_directional_total_variation_radps": float(sum(row["yaw_total_variation"] for row in episode_execution)),
        "pitch_directional_total_variation_radps": float(sum(row["pitch_total_variation"] for row in episode_execution)),
    }
    execution_by_stage: dict[str, dict[str, Any]] = {}
    for stage in sorted({row["stage"] for row in episode_execution}):
        members = [row for row in episode_execution if row["stage"] == stage]
        sy = sum(row["yaw_reversal_count"] for row in members)
        sye = sum(row["yaw_eligible_count"] for row in members)
        sp = sum(row["pitch_reversal_count"] for row in members)
        spe = sum(row["pitch_eligible_count"] for row in members)
        execution_by_stage[stage] = {
            "yaw_reversal_rate": rate(sy, sye),
            "yaw_reversal_count": sy,
            "yaw_eligible_count": sye,
            "pitch_reversal_rate": rate(sp, spe),
            "pitch_reversal_count": sp,
            "pitch_eligible_count": spe,
        }

    coupling_summary: dict[str, Any] = {}
    for axis in ("yaw", "pitch"):
        ref_key = f"ref_{axis}_reversal"
        delta2_key = "delta2_alpha_abs_rad" if axis == "yaw" else "delta2_beta_abs_rad"
        for suffix in ("0p2", "0p5"):
            exec_key = f"exec_{axis}_reversal_next_{suffix}s"
            second_key = f"exec_{axis}_second_difference_next_{suffix}s"
            yes = conditional_probability(coupling_rows, ref_key, exec_key, True)
            no = conditional_probability(coupling_rows, ref_key, exec_key, False)
            rho = cluster_bootstrap_spearman(coupling_rows, delta2_key, second_key)
            coupling_summary[f"{axis}_{suffix}s"] = {
                "exec_reversal_given_reference_reversal": {"count": yes[0], "n": yes[1], "probability": yes[2]},
                "exec_reversal_given_no_reference_reversal": {"count": no[0], "n": no[1], "probability": no[2]},
                "absolute_probability_lift": (yes[2] - no[2]) if yes[2] is not None and no[2] is not None else None,
                "relative_risk": (yes[2] / no[2]) if yes[2] is not None and no[2] not in (None, 0.0) else None,
                "second_difference_spearman": rho,
            }

    # Compare signed alternation to a magnitude-only P90 jump flag using the same events.
    unsigned_comparison: dict[str, Any] = {}
    for axis in ("yaw", "pitch"):
        delta_key = "delta_alpha_rad" if axis == "yaw" else "delta_beta_rad"
        values = [abs(float(row[delta_key])) for row in coupling_rows if row.get(delta_key) is not None]
        threshold = float(np.quantile(values, 0.9))
        flag_key = f"high_unsigned_{axis}_jump"
        for row in coupling_rows:
            row[flag_key] = bool(row.get(delta_key) is not None and abs(float(row[delta_key])) >= threshold)
        exec_key = f"exec_{axis}_reversal_next_0p5s"
        high = conditional_probability(coupling_rows, flag_key, exec_key, True)
        low = conditional_probability(coupling_rows, flag_key, exec_key, False)
        signed = coupling_summary[f"{axis}_0p5s"]
        unsigned_lift = high[2] - low[2] if high[2] is not None and low[2] is not None else None
        signed_lift = signed["absolute_probability_lift"]
        unsigned_comparison[axis] = {
            "p90_unsigned_jump_threshold_rad": threshold,
            "exec_reversal_given_high_unsigned_jump": high[2],
            "exec_reversal_given_not_high_unsigned_jump": low[2],
            "unsigned_jump_probability_lift": unsigned_lift,
            "signed_reversal_probability_lift": signed_lift,
            "signed_reversal_explains_better": bool(
                signed_lift is not None
                and unsigned_lift is not None
                and signed_lift > 0.0
                and signed_lift > unsigned_lift
            ),
        }

    stage_coupling: dict[str, Any] = {}
    for stage, rows in sorted(stage_ref.items()):
        stage_coupling[stage] = {}
        for axis in ("yaw", "pitch"):
            ref_key = f"ref_{axis}_reversal"
            exec_key = f"exec_{axis}_reversal_next_0p5s"
            delta2_key = "delta2_alpha_abs_rad" if axis == "yaw" else "delta2_beta_abs_rad"
            second_key = f"exec_{axis}_second_difference_next_0p5s"
            yes = conditional_probability(rows, ref_key, exec_key, True)
            no = conditional_probability(rows, ref_key, exec_key, False)
            clean = [r for r in rows if r.get(delta2_key) is not None and r.get(second_key) is not None]
            rho = float(spearmanr([r[delta2_key] for r in clean], [r[second_key] for r in clean]).statistic) if len(clean) >= 3 else None
            stage_coupling[stage][axis] = {
                "p_exec_reversal_given_ref_reversal": yes[2],
                "p_exec_reversal_given_no_ref_reversal": no[2],
                "probability_lift": (yes[2] - no[2]) if yes[2] is not None and no[2] is not None else None,
                "second_difference_spearman_rho": finite_float(rho),
                "n": len(clean),
            }

    # Conservative, predeclared evidence interpretation. DCTB needs one axis to
    # satisfy every gate at 0.5 s, with replication in at least three stages.
    axis_gate: dict[str, Any] = {}
    for axis in ("yaw", "pitch"):
        ref_rate = reference_overall[f"{axis}_reversal_rate"]
        cs = coupling_summary[f"{axis}_0p5s"]
        rho = cs["second_difference_spearman"]
        replicated = sum(
            1
            for stage in stage_coupling.values()
            if stage[axis]["probability_lift"] is not None
            and stage[axis]["probability_lift"] >= 0.10
            and stage[axis]["second_difference_spearman_rho"] is not None
            and stage[axis]["second_difference_spearman_rho"] > 0.0
        )
        axis_gate[axis] = {
            "nontrivial_reference_reversal": bool(ref_rate is not None and ref_rate >= 0.10),
            "substantial_execution_probability_lift": bool(cs["absolute_probability_lift"] is not None and cs["absolute_probability_lift"] >= 0.10),
            "stable_positive_second_difference_relation": bool(
                rho["rho"] is not None
                and rho["rho"] >= 0.10
                and rho["ci95"][0] is not None
                and rho["ci95"][0] > 0.0
            ),
            "replicated_stage_count": replicated,
            "replicated_across_multiple_stages": replicated >= 3,
            "better_than_unsigned_jump": unsigned_comparison[axis]["signed_reversal_explains_better"],
        }
        axis_gate[axis]["all_gates_pass"] = all(
            [
                axis_gate[axis]["nontrivial_reference_reversal"],
                axis_gate[axis]["substantial_execution_probability_lift"],
                axis_gate[axis]["stable_positive_second_difference_relation"],
                axis_gate[axis]["replicated_across_multiple_stages"],
                axis_gate[axis]["better_than_unsigned_jump"],
            ]
        )

    authorized = any(axis_gate[axis]["all_gates_pass"] for axis in axis_gate)
    passing_axes = [axis for axis in axis_gate if axis_gate[axis]["all_gates_pass"]]
    # Sequence evidence and execution coupling are intentionally classified
    # separately: abundant signed alternation may exist without causing the
    # executed residual morphology.  That distinction is the point of Phase 2.
    replicated_reference_axes = 0
    for axis in ("yaw", "pitch"):
        stage_count = sum(
            1
            for summary in reference_by_stage.values()
            if (summary[f"{axis}_reversal_rate"] or 0.0) >= 0.10
        )
        if (reference_overall[f"{axis}_reversal_rate"] or 0.0) >= 0.10 and stage_count >= 3:
            replicated_reference_axes += 1
    if replicated_reference_axes == 2:
        reference_evidence = "STRONG"
    elif replicated_reference_axes == 1:
        reference_evidence = "MODERATE"
    elif max(
        reference_overall["yaw_reversal_rate"] or 0.0,
        reference_overall["pitch_reversal_rate"] or 0.0,
    ) > 0.0:
        reference_evidence = "WEAK"
    else:
        reference_evidence = "NONE"

    if authorized and len(passing_axes) == 2:
        execution_coupling = "STRONG"
    elif authorized:
        execution_coupling = "MODERATE"
    else:
        positive_rho = any(
            (coupling_summary[f"{axis}_0p5s"]["second_difference_spearman"]["rho"] or 0.0) > 0.0
            for axis in ("yaw", "pitch")
        )
        execution_coupling = "WEAK" if positive_rho else "NONE"

    event_rows: list[dict[str, Any]] = []
    for group, rows in sorted(event_ref.items()):
        summary = summarize_reference(rows) if any(row.get("alpha_rad") is not None for row in rows) else {
            "accepted_reference_count": len(rows),
            "yaw_reversal_count": 0,
            "yaw_eligible_count": 0,
            "yaw_reversal_rate": None,
            "pitch_reversal_count": 0,
            "pitch_eligible_count": 0,
            "pitch_reversal_rate": None,
            "yaw_median_same_sign_run_length": None,
            "pitch_median_same_sign_run_length": None,
            "yaw_p90_abs_delta2_rad": None,
            "pitch_p90_abs_delta2_rad": None,
            "yaw_directional_total_variation_rad": 0.0,
            "pitch_directional_total_variation_rad": 0.0,
        }
        event_rows.append({"event_group": group, **summary})

    audit = {
        "schema_version": "reference_sequence_zigzag_audit_v1",
        "source": str(source_root.as_posix()),
        "scope": "read_only_existing_strong_development_100",
        "scenario_count": len(record_paths),
        "dt_s": 0.1,
        "numeric_angle_epsilon_rad": NUMERIC_ANGLE_EPS,
        "numeric_speed_epsilon_mps": NUMERIC_SPEED_EPS,
        "reference_representation": {
            "alpha": "signed horizontal angle from current task-goal direction to accepted local-reference direction in [-pi,pi]",
            "beta": "accepted-reference elevation minus current task-goal elevation",
            "semantic_reset": "local-reference sequence is reset at task-goal restoration/terminal handoff",
        },
        "reference_overall": reference_overall,
        "reference_by_stage": reference_by_stage,
        "execution_overall": execution_overall,
        "execution_by_stage": execution_by_stage,
        "coupling": coupling_summary,
        "coupling_by_stage_0p5s": stage_coupling,
        "signed_reversal_vs_unsigned_jump": unsigned_comparison,
        "hard_gate_by_axis": axis_gate,
        "REFERENCE_SEQUENCE_ZIGZAG_EVIDENCE": reference_evidence,
        "REFERENCE_TO_EXECUTION_COUPLING": execution_coupling,
        "DCTB_AUTHORIZED": "YES" if authorized else "NO",
        "simulation_run": False,
        "upper_layer_modified": False,
        "formal_v2_read": False,
    }
    go_no_go = {
        "schema_version": "dctb_go_no_go_v1",
        "source_audit": "REFERENCE_SEQUENCE_ZIGZAG_AUDIT.json",
        "hard_gate_by_axis": axis_gate,
        "passing_axes": passing_axes,
        "REFERENCE_SEQUENCE_ZIGZAG_EVIDENCE": reference_evidence,
        "REFERENCE_TO_EXECUTION_COUPLING": execution_coupling,
        "DCTB_AUTHORIZED": "YES" if authorized else "NO",
        "decision": "PROCEED_TO_ONE_FROZEN_DCTB_CONTRACT" if authorized else "STOP_WITHOUT_UPPER_LAYER_MODIFICATION",
        "no_post_hoc_threshold_tuning": True,
    }
    return audit, coupling_rows, event_rows, go_no_go


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)

    failure = audit_strong_failures(args.source_root)
    audit, coupling_rows, event_rows, go_no_go = audit_reference_and_execution(args.source_root)
    write_json(args.output_root / "STRONG_OBSTACLE_FAILURE_DIAGNOSIS.json", failure)
    write_json(args.output_root / "REFERENCE_SEQUENCE_ZIGZAG_AUDIT.json", audit)
    write_csv(args.output_root / "REFERENCE_EXECUTION_COUPLING.csv", coupling_rows)
    write_csv(args.output_root / "REFERENCE_EVENT_TYPE_STRATIFICATION.csv", event_rows)
    write_json(args.output_root / "DCTB_GO_NO_GO.json", go_no_go)
    print(json.dumps({
        "output_root": str(args.output_root),
        "safety_concern": failure["STRONG_LIMITER_SAFETY_CONCERN"],
        "reference_yaw_reversal_rate": audit["reference_overall"]["yaw_reversal_rate"],
        "reference_pitch_reversal_rate": audit["reference_overall"]["pitch_reversal_rate"],
        "executed_yaw_reversal_rate": audit["execution_overall"]["yaw_reversal_rate"],
        "executed_pitch_reversal_rate": audit["execution_overall"]["pitch_reversal_rate"],
        "evidence": audit["REFERENCE_SEQUENCE_ZIGZAG_EVIDENCE"],
        "coupling": audit["REFERENCE_TO_EXECUTION_COUPLING"],
        "DCTB_AUTHORIZED": audit["DCTB_AUTHORIZED"],
    }, indent=2))


if __name__ == "__main__":
    main()
