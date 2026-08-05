from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class InitialCase:
    name: str
    dims: int
    start: np.ndarray
    goal: np.ndarray
    velocity: np.ndarray


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    seen = set(fieldnames)
    for row in rows[1:]:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def git_metadata() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    status = run("status", "--short")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status),
        "status_short": status,
    }


def build_cases(config: dict[str, Any], *, review: bool = False) -> list[InitialCase]:
    distances = config["review_distances"] if review else config["distances"]
    speed = float(config["review_initial_speed"] if review else config["initial_speed"])
    cases: list[InitialCase] = []
    for dims in (2, 3):
        for distance in distances:
            start = np.zeros(dims, dtype=float)
            direction = np.ones(dims, dtype=float)
            direction /= np.linalg.norm(direction)
            goal = direction * float(distance)
            perpendicular = np.zeros(dims, dtype=float)
            perpendicular[0] = -direction[1]
            perpendicular[1] = direction[0]
            velocities = {
                "stationary": np.zeros(dims, dtype=float),
                "toward": speed * direction,
                "perpendicular": speed * perpendicular,
                "away": -speed * direction,
            }
            for velocity_name, velocity in velocities.items():
                cases.append(
                    InitialCase(
                        name=f"d{dims}_{float(distance):g}_{velocity_name}",
                        dims=dims,
                        start=start.copy(),
                        goal=goal.copy(),
                        velocity=velocity.copy(),
                    )
                )
    return cases


def simulate_case(
    case: InitialCase,
    candidate: dict[str, float],
    sim: dict[str, Any],
    *,
    keep_trace: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dt = float(sim["dt"])
    max_steps = int(sim["max_steps"])
    goal_tolerance = float(sim["goal_tolerance"])
    settling_speed = float(sim["settling_speed"])
    acceleration_limit = float(sim["acceleration_limit"])
    velocity_limit = float(sim["velocity_limit"])
    k_alpha = float(candidate["k_alpha"])
    k_beta = float(candidate["k_beta"])
    tau = float(candidate["tau"])

    position = case.start.copy()
    velocity = case.velocity.copy()
    initial_delta = case.goal - case.start
    initial_distance = float(np.linalg.norm(initial_delta))
    path_direction = initial_delta / max(initial_distance, 1.0e-12)
    previous_acceleration: np.ndarray | None = None
    reached_step: int | None = None
    settled_step: int | None = None
    path_length = 0.0
    overshoot = 0.0
    acceleration_clipped_steps = 0
    velocity_clipped_steps = 0
    mid_ratio_values: list[float] = []
    acceleration_norms: list[float] = []
    jerk_norms: list[float] = []
    trace: list[dict[str, Any]] = []

    for step in range(1, max_steps + 1):
        goal_delta = case.goal - position
        distance = float(np.linalg.norm(goal_delta))
        spring = k_alpha * k_beta * goal_delta
        damping = -k_alpha * tau * velocity
        nominal_drive = spring + damping
        commanded_acceleration = nominal_drive / tau**2
        applied_acceleration = np.clip(
            commanded_acceleration,
            -acceleration_limit,
            acceleration_limit,
        )
        acceleration_clipped = bool(
            np.any(~np.isclose(commanded_acceleration, applied_acceleration, atol=1.0e-8))
        )
        acceleration_clipped_steps += int(acceleration_clipped)

        candidate_velocity = velocity + applied_acceleration * dt
        next_velocity = np.clip(candidate_velocity, -velocity_limit, velocity_limit)
        velocity_clipped = bool(np.any(~np.isclose(candidate_velocity, next_velocity, atol=1.0e-8)))
        velocity_clipped_steps += int(velocity_clipped)
        next_position = position + velocity * dt + 0.5 * applied_acceleration * dt**2
        path_length += float(np.linalg.norm(next_position - position))

        acceleration_norm = float(np.linalg.norm(applied_acceleration))
        acceleration_norms.append(acceleration_norm)
        jerk_norm = 0.0
        if previous_acceleration is not None:
            jerk_norm = float(np.linalg.norm(applied_acceleration - previous_acceleration) / dt)
            jerk_norms.append(jerk_norm)

        spring_norm = float(np.linalg.norm(spring))
        damping_norm = float(np.linalg.norm(damping))
        damping_spring_ratio = damping_norm / (spring_norm + 1.0e-8)
        if 0.3 * initial_distance <= distance <= 0.7 * initial_distance:
            mid_ratio_values.append(damping_spring_ratio)

        projected_progress = float(np.dot(next_position - case.start, path_direction))
        overshoot = max(overshoot, projected_progress - initial_distance)
        next_distance = float(np.linalg.norm(case.goal - next_position))
        next_speed = float(np.linalg.norm(next_velocity))
        if reached_step is None and next_distance <= goal_tolerance:
            reached_step = step
        if settled_step is None and next_distance <= goal_tolerance and next_speed <= settling_speed:
            settled_step = step

        if keep_trace:
            row = {
                "case": case.name,
                "dims": case.dims,
                "step": step,
                "time": step * dt,
                "distance": next_distance,
                "speed": next_speed,
                "spring_norm": spring_norm,
                "damping_norm": damping_norm,
                "damping_spring_ratio": damping_spring_ratio,
                "nominal_drive_norm": float(np.linalg.norm(nominal_drive)),
                "commanded_acceleration_norm": float(np.linalg.norm(commanded_acceleration)),
                "applied_acceleration_norm": acceleration_norm,
                "jerk_norm": jerk_norm,
                "acceleration_clipped": acceleration_clipped,
                "velocity_clipped": velocity_clipped,
            }
            for axis in range(case.dims):
                row[f"position_{axis}"] = float(next_position[axis])
                row[f"velocity_{axis}"] = float(next_velocity[axis])
                row[f"acceleration_{axis}"] = float(applied_acceleration[axis])
            trace.append(row)

        previous_acceleration = applied_acceleration.copy()
        position = next_position
        velocity = next_velocity

    final_distance = float(np.linalg.norm(case.goal - position))
    summary = {
        "case": case.name,
        "dims": case.dims,
        "initial_distance": initial_distance,
        "initial_speed": float(np.linalg.norm(case.velocity)),
        "success": reached_step is not None,
        "settled": settled_step is not None,
        "flight_time": float(reached_step * dt) if reached_step is not None else math.nan,
        "settling_time": float(settled_step * dt) if settled_step is not None else math.nan,
        "terminal_error": final_distance,
        "path_length": path_length,
        "overshoot": overshoot,
        "acceleration_clip_rate": acceleration_clipped_steps / max_steps,
        "velocity_clip_rate": velocity_clipped_steps / max_steps,
        "acceleration_rms": float(np.sqrt(np.mean(np.square(acceleration_norms)))),
        "acceleration_peak": float(np.max(acceleration_norms)),
        "jerk_rms": float(np.sqrt(np.mean(np.square(jerk_norms)))) if jerk_norms else 0.0,
        "jerk_peak": float(np.max(jerk_norms)) if jerk_norms else 0.0,
        "mid_damping_spring_ratio_mean": float(np.mean(mid_ratio_values)) if mid_ratio_values else math.nan,
        "mid_damping_dominance_rate": float(np.mean(np.asarray(mid_ratio_values) > 1.0)) if mid_ratio_values else math.nan,
    }
    return summary, trace


def aggregate_candidate(
    stage: str,
    candidate_index: int,
    candidate: dict[str, float],
    rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    success_rate = float(np.mean([row["success"] for row in rows]))
    settled_rate = float(np.mean([row["settled"] for row in rows]))
    result = {
        "stage": stage,
        "candidate_id": f"{stage}_{candidate_index:02d}",
        **candidate,
        "case_count": len(rows),
        "success_rate": success_rate,
        "settled_rate": settled_rate,
        "terminal_error_max": float(max(row["terminal_error"] for row in rows)),
        "terminal_error_mean": float(np.mean([row["terminal_error"] for row in rows])),
        "acceleration_clip_rate_mean": float(np.mean([row["acceleration_clip_rate"] for row in rows])),
        "acceleration_clip_rate_max": float(max(row["acceleration_clip_rate"] for row in rows)),
        "velocity_clip_rate_mean": float(np.mean([row["velocity_clip_rate"] for row in rows])),
        "jerk_rms_mean": float(np.mean([row["jerk_rms"] for row in rows])),
        "jerk_peak_max": float(max(row["jerk_peak"] for row in rows)),
        "overshoot_max": float(max(row["overshoot"] for row in rows)),
        "mid_damping_dominance_rate_mean": float(np.nanmean([row["mid_damping_dominance_rate"] for row in rows])),
        "flight_time_mean": float(np.nanmean([row["flight_time"] for row in rows])),
    }
    result["hard_pass"] = bool(
        success_rate == 1.0
        and settled_rate == 1.0
        and result["terminal_error_max"] <= float(config["terminal_error_limit"])
        and result["overshoot_max"] <= float(config["overshoot_limit"])
        and result["flight_time_mean"] <= float(config["flight_time_mean_limit"])
        and result["acceleration_clip_rate_max"] <= float(config["acceleration_clip_rate_max_limit"])
        and np.isfinite(result["jerk_rms_mean"])
    )
    return result


def candidate_order(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        not bool(row["hard_pass"]),
        -float(row["success_rate"]),
        float(row["acceleration_clip_rate_mean"]),
        float(row["velocity_clip_rate_mean"]),
        float(row["jerk_rms_mean"]),
        float(row["overshoot_max"]),
        float(row["mid_damping_dominance_rate_mean"]),
        float(row["flight_time_mean"]),
        float(row["k_alpha"]) * float(row["k_beta"]),
    )


def evaluate_candidates(
    stage: str,
    candidates: list[dict[str, float]],
    cases: list[InitialCase],
    config: dict[str, Any],
    *,
    keep_traces: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    aggregates: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(candidates):
        candidate_case_rows = []
        candidate_id = f"{stage}_{candidate_index:02d}"
        for case in cases:
            summary, trace = simulate_case(
                case,
                candidate,
                config["simulation"],
                keep_trace=keep_traces,
            )
            summary.update({"stage": stage, "candidate_id": candidate_id, **candidate})
            candidate_case_rows.append(summary)
            case_rows.append(summary)
            for row in trace:
                row.update({"stage": stage, "candidate_id": candidate_id, **candidate})
                traces.append(row)
        aggregates.append(
            aggregate_candidate(stage, candidate_index, candidate, candidate_case_rows, config)
        )
    return aggregates, case_rows, traces


def run_phase_scan(
    selected: dict[str, Any],
    traces: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    dt = float(config["simulation"]["dt"])
    tau = float(selected["tau"])
    integrator = str(config["phase_scan"]["integrator"])
    phase_radius = float(config["phase_scan"]["phase_radius"])
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in traces:
        if row["candidate_id"] != selected["candidate_id"]:
            continue
        grouped.setdefault(str(row["case"]), []).append(row)

    results: list[dict[str, Any]] = []
    for alpha_s in config["phase_scan"]["alpha_s"]:
        for phase_end in config["phase_scan"]["phase_end_threshold"]:
            errors = []
            speeds = []
            times = []
            missing = 0
            for case_rows in grouped.values():
                phase = 1.0
                crossing = None
                for row in case_rows:
                    decay = float(alpha_s) * dt / tau
                    phase = phase * (math.exp(-decay) if integrator == "exponential" else (1.0 - decay))
                    phase = max(0.0, min(1.0, phase))
                    if phase <= float(phase_end):
                        crossing = row
                        break
                if crossing is None:
                    missing += 1
                    continue
                errors.append(float(crossing["distance"]))
                speeds.append(float(crossing["speed"]))
                times.append(float(crossing["time"]))
            result = {
                "alpha_s": float(alpha_s),
                "phase_end_threshold": float(phase_end),
                "integrator": integrator,
                "case_count": len(grouped),
                "missing_crossing_count": missing,
                "phase_crossing_time_mean": float(np.mean(times)) if times else math.nan,
                "goal_error_at_crossing_mean": float(np.mean(errors)) if errors else math.nan,
                "goal_error_at_crossing_max": float(np.max(errors)) if errors else math.nan,
                "speed_at_crossing_mean": float(np.mean(speeds)) if speeds else math.nan,
                "phase_radius_violation_rate": float(np.mean(np.asarray(errors) > phase_radius)) if errors else 1.0,
            }
            results.append(result)
    return sorted(
        results,
        key=lambda row: (
            row["missing_crossing_count"] > 0,
            row["phase_radius_violation_rate"],
            row["phase_crossing_time_mean"],
            row["goal_error_at_crossing_max"],
        ),
    )


def plot_review(output_dir: Path, traces: list[dict[str, Any]], selected_id: str) -> None:
    selected = [row for row in traces if row["candidate_id"] == selected_id]
    if not selected:
        return
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in selected:
        grouped.setdefault(str(row["case"]), []).append(row)
    representative_names = [name for name in grouped if name.endswith("stationary")]

    figures = [
        ("distance_time.png", "distance", "Distance to goal"),
        ("spring_damping.png", None, "Spring and damping norms"),
        ("damping_spring_ratio.png", "damping_spring_ratio", "Damping/spring ratio"),
        ("acceleration_jerk.png", None, "Acceleration and jerk"),
    ]
    for filename, key, title in figures:
        fig, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
        for name in representative_names:
            rows = grouped[name]
            times = [row["time"] for row in rows]
            if filename == "spring_damping.png":
                axis.plot(times, [row["spring_norm"] for row in rows], label=f"{name}/spring")
                axis.plot(times, [row["damping_norm"] for row in rows], linestyle="--", label=f"{name}/damping")
            elif filename == "acceleration_jerk.png":
                axis.plot(times, [row["applied_acceleration_norm"] for row in rows], label=f"{name}/acc")
                axis.plot(times, [row["jerk_norm"] for row in rows], linestyle="--", label=f"{name}/jerk")
            else:
                axis.plot(times, [row[key] for row in rows], label=name)
        axis.set_title(title)
        axis.set_xlabel("time (s)")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=7, ncol=2)
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    for name in representative_names:
        rows = grouped[name]
        x = [row["position_0"] for row in rows]
        y = [row.get("position_1", 0.0) for row in rows]
        axis.plot(x, y, label=name)
    axis.set_title("Pure-DMP trajectories")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.axis("equal")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=7)
    fig.savefig(output_dir / "trajectories.png", dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run staged Pure-DMP and Classic phase calibration.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "calibration" / "pure_dmp_scan.json",
    )
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "artifacts" / "calibration")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / f"pure_dmp_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    base = config["baseline"]
    scan = config["scan"]
    stage1_candidates = [
        {"k_alpha": float(k_alpha), "k_beta": float(k_beta), "tau": float(base["tau"])}
        for k_alpha in scan["k_alpha"]
        for k_beta in scan["k_beta"]
    ]
    cases = build_cases(config)
    stage1, stage1_cases, _ = evaluate_candidates(
        "gain_scan", stage1_candidates, cases, config, keep_traces=False
    )
    top_gain = sorted(stage1, key=candidate_order)[: int(scan["gain_candidates_for_tau"])]
    stage2_candidates = [
        {
            "k_alpha": float(row["k_alpha"]),
            "k_beta": float(row["k_beta"]),
            "tau": float(tau),
        }
        for row in top_gain
        for tau in scan["tau"]
    ]
    stage2, stage2_cases, _ = evaluate_candidates(
        "tau_scan", stage2_candidates, cases, config, keep_traces=False
    )
    top_local = sorted(stage2, key=candidate_order)[: int(scan["local_review_candidates"])]
    local_candidates = [
        {
            "k_alpha": float(base["k_alpha"]),
            "k_beta": float(base["k_beta"]),
            "tau": float(base["tau"]),
        }
    ]
    for row in top_local:
        candidate = {
            "k_alpha": row["k_alpha"],
            "k_beta": row["k_beta"],
            "tau": row["tau"],
        }
        if candidate not in local_candidates:
            local_candidates.append(candidate)
        if len(local_candidates) >= int(scan["local_review_candidates"]):
            break
    review_cases = build_cases(config, review=True)
    local, local_cases, local_traces = evaluate_candidates(
        "local_review", local_candidates, review_cases, config, keep_traces=True
    )
    selected = sorted(local, key=candidate_order)[0]
    phase_rows = run_phase_scan(selected, local_traces, config)
    selected_phase = phase_rows[0]

    write_csv(output_dir / "candidate_summary.csv", stage1 + stage2 + local)
    write_csv(output_dir / "case_summary.csv", stage1_cases + stage2_cases + local_cases)
    write_csv(output_dir / "review_traces.csv", local_traces)
    write_csv(output_dir / "phase_summary.csv", phase_rows)
    plot_review(output_dir, local_traces, str(selected["candidate_id"]))
    report = {
        "status": "completed",
        "launch_command": subprocess.list2cmdline([sys.executable, *sys.argv]),
        "config_path": str(args.config.resolve()),
        "git": git_metadata(),
        "selection_rule": [
            "hard constraints",
            "success rate",
            "acceleration clipping",
            "velocity clipping",
            "jerk",
            "overshoot",
            "mid-flight damping dominance",
            "flight time",
            "minimum gain",
        ],
        "stage_counts": {
            "gain_scan": len(stage1),
            "tau_scan": len(stage2),
            "local_review": len(local),
        },
        "recommended_dmp": selected,
        "recommended_classic_phase": selected_phase,
        "artifacts": {
            "candidate_summary": "candidate_summary.csv",
            "case_summary": "case_summary.csv",
            "review_traces": "review_traces.csv",
            "phase_summary": "phase_summary.csv",
            "plots": [
                "trajectories.png",
                "distance_time.png",
                "spring_damping.png",
                "damping_spring_ratio.png",
                "acceleration_jerk.png",
            ],
        },
    }
    write_json(output_dir / "report.json", report)
    print(json.dumps({"output_dir": str(output_dir), **report["stage_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
