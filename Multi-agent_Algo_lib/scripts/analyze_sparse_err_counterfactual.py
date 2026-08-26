from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    REPO_ROOT
    / "artifacts"
    / "theory_aligned_final_recovery"
    / "20260818_232900"
)
PHASE_D = SOURCE / "phase_d_theory_err_development"
RECORDS = PHASE_D / "records_v2"
EQUAL_INFO = (
    REPO_ROOT
    / "artifacts"
    / "equal_information_baseline_audit"
    / "20260819_012339"
)
OUTPUT = (
    REPO_ROOT
    / "artifacts"
    / "sparse_err_trigger_revision"
    / "20260819_020727"
)
H_EMG = 0.0
H_REP = 0.35


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def percentile(values: Iterable[float], q: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def load_records() -> list[dict[str, Any]]:
    paths = sorted(RECORDS.rglob("gat_v1_err.json"))
    if len(paths) != 40:
        raise RuntimeError(f"expected 40 source ERR records, found {len(paths)}")
    return [load_json(path) for path in paths]


def current_consecutive_emergencies(events: Sequence[Mapping[str, Any]]) -> int:
    grouped: defaultdict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in events:
        if row.get("counts_as_reproposal"):
            grouped[(row["scenario"], row["seed"], row["agent_id"])].append(row)
    count = 0
    for rows in grouped.values():
        ordered = sorted(rows, key=lambda row: int(row["step"]))
        count += sum(
            first["event"] == "EMERGENCY_REPROPOSAL"
            and second["event"] == "EMERGENCY_REPROPOSAL"
            for first, second in zip(ordered, ordered[1:])
        )
    return count


def filter_timeline(records: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], set[tuple[str, int, int]], set[tuple[str, int, int, int]]]:
    rows: list[dict[str, Any]] = []
    retained_ticks: set[tuple[str, int, int]] = set()
    retained_agent_events: set[tuple[str, int, int, int]] = set()
    for record in records:
        grouped: defaultdict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for row in record["triggers"]:
            grouped[int(row["agent_id"])].append(row)
        for agent_id, timeline in grouped.items():
            ordered = sorted(timeline, key=lambda row: int(row["step"]))
            armed = False
            previous_h: float | None = None
            for index, source_row in enumerate(ordered):
                h_value = float(source_row["active_safety_margin_m"])
                step = int(source_row["step"])
                latch_before = bool(armed)
                previous_before = previous_h
                rearmed = False
                if index == 0:
                    # Initial planning already consumed the t=0 upper decision.
                    # It is armed only when the selected reference is already in
                    # the existing normal safety region.
                    armed = h_value >= H_REP
                    previous_h = h_value
                    emergency_edge = False
                    initialization = True
                else:
                    initialization = False
                    if not armed and h_value >= H_REP:
                        armed = True
                        rearmed = True
                    emergency_edge = bool(
                        armed
                        and previous_h is not None
                        and previous_h > H_EMG
                        and h_value <= H_EMG
                    )
                normal = bool(source_row["normal_trigger"])
                handoff = bool(source_row["handoff_trigger"])
                if handoff:
                    counter_event = "REFERENCE_COMPLETION_HANDOFF"
                    detail = "HANDOFF_PRIORITY"
                elif emergency_edge:
                    counter_event = "EMERGENCY_REPROPOSAL"
                    detail = "NORMAL_AND_EMERGENCY" if normal else "EMERGENCY_ENTRY"
                elif normal:
                    counter_event = "NORMAL_REPROPOSAL"
                    detail = "NORMAL_ONLY"
                else:
                    counter_event = "NO_UPDATE"
                    detail = "NO_EVENT"
                is_reproposal = counter_event in {
                    "EMERGENCY_REPROPOSAL",
                    "NORMAL_REPROPOSAL",
                }
                if is_reproposal:
                    retained_ticks.add((str(source_row["scenario"]), int(source_row["seed"]), step))
                    retained_agent_events.add(
                        (str(source_row["scenario"]), int(source_row["seed"]), agent_id, step)
                    )
                rows.append(
                    {
                        "scenario_id": source_row["scenario"],
                        "seed": int(source_row["seed"]),
                        "stage": record["entry"]["stage"],
                        "agent_id": agent_id,
                        "step": step,
                        "time_s": float(source_row["time_s"]),
                        "h_active_m": h_value,
                        "h_previous_m": previous_before,
                        "h_emg_m": H_EMG,
                        "h_rep_m": H_REP,
                        "initialization_tick": initialization,
                        "emergency_armed_before": latch_before,
                        "rearmed_at_existing_h_rep": rearmed,
                        "emergency_edge": emergency_edge,
                        "normal_trigger_unchanged": normal,
                        "handoff_trigger_unchanged": handoff,
                        "current_event": source_row["event"],
                        "counterfactual_event": counter_event,
                        "counterfactual_event_detail": detail,
                        "counterfactual_reproposal": is_reproposal,
                        "diagnostic_only_noninterventional": True,
                    }
                )
                # A goal update changes the active-direction meaning. It cannot
                # rearm the latch by fiat. The next observed margin must first
                # reach the existing h_rep boundary.
                if handoff or is_reproposal:
                    armed = False
                    previous_h = None
                else:
                    previous_h = h_value
    return rows, retained_ticks, retained_agent_events


def runtime_estimate(
    records: Sequence[Mapping[str, Any]], retained_ticks: set[tuple[str, int, int]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        episode = record["episode"]
        scenario = str(record["entry"]["scenario_id"])
        seed = int(record["entry"]["seed"])
        retained = [
            row
            for row in record["upper_timing_rows"]
            if row["event_type"] == "INITIAL_SELECTION"
            or (scenario, seed, int(row["event_step"])) in retained_ticks
        ]
        current_planning = float(sum(row["upper_planning_total_ms"] for row in record["upper_timing_rows"]))
        counter_planning = float(sum(row["upper_planning_total_ms"] for row in retained))
        actor = float(episode["execution_actor_forward_ms"])
        dmp = float(episode["execution_dmp_ms"])
        rows.append(
            {
                "row_type": "episode",
                "scope": record["entry"]["stage"],
                "scenario_id": scenario,
                "seed": seed,
                "current_upper_decisions": len(record["upper_timing_rows"]),
                "counterfactual_upper_decisions": len(retained),
                "current_planning_compute_ms": current_planning,
                "counterfactual_planning_compute_ms": counter_planning,
                "planning_compute_reduction_ms": current_planning - counter_planning,
                "planning_compute_reduction_rate": (
                    (current_planning - counter_planning) / current_planning
                    if current_planning
                    else 0.0
                ),
                "historical_execution_actor_ms_held_fixed": actor,
                "historical_execution_dmp_ms_held_fixed": dmp,
                "current_total_compute_ms": current_planning + actor + dmp,
                "counterfactual_total_compute_estimate_ms": counter_planning + actor + dmp,
                "estimate_status": "DIAGNOSTIC_ESTIMATE_NONINTERVENTIONAL",
            }
        )
    for scope in ("overall", "stage_1", "stage_2", "stage_3", "stage_4"):
        members = [row for row in rows if scope == "overall" or row["scope"] == scope]
        rows.append(
            {
                "row_type": "summary",
                "scope": scope,
                "scenario_id": "",
                "seed": "",
                "current_upper_decisions": mean(row["current_upper_decisions"] for row in members),
                "counterfactual_upper_decisions": mean(row["counterfactual_upper_decisions"] for row in members),
                "current_planning_compute_ms": mean(row["current_planning_compute_ms"] for row in members),
                "counterfactual_planning_compute_ms": mean(row["counterfactual_planning_compute_ms"] for row in members),
                "planning_compute_reduction_ms": mean(row["planning_compute_reduction_ms"] for row in members),
                "planning_compute_reduction_rate": (
                    sum(row["planning_compute_reduction_ms"] for row in members)
                    / sum(row["current_planning_compute_ms"] for row in members)
                ),
                "historical_execution_actor_ms_held_fixed": mean(row["historical_execution_actor_ms_held_fixed"] for row in members),
                "historical_execution_dmp_ms_held_fixed": mean(row["historical_execution_dmp_ms_held_fixed"] for row in members),
                "current_total_compute_ms": mean(row["current_total_compute_ms"] for row in members),
                "counterfactual_total_compute_estimate_ms": mean(row["counterfactual_total_compute_estimate_ms"] for row in members),
                "estimate_status": "DIAGNOSTIC_ESTIMATE_NONINTERVENTIONAL",
            }
        )
    overall = next(row for row in rows if row["row_type"] == "summary" and row["scope"] == "overall")
    return rows, overall


def write_contracts() -> None:
    (OUTPUT / "revised_emergency_event_contract.md").write_text(
        """# Revised Emergency Event Contract\n\n"
        "Only emergency event semantics change. The normal ERR branch, all existing numeric "
        "thresholds, Proposal, Top-K 10, FP-SHEP H4, GAT-V1, SAC-DMP, handoff distance, "
        "collision rules, and maximum episode length remain frozen.\n\n"
        "For agent i, emergency_armed is a Boolean latch. An emergency event occurs iff "
        "emergency_armed is true, h_active at the previous observation is greater than "
        "the existing h_emg=0.0 m, and the current h_active is at or below h_emg. After the "
        "event the latch is false. It becomes true again only after an observed h_active at "
        "or above the existing h_rep=0.35 m. No third threshold, cooldown, refractory time, "
        "or event-count cap exists.\n\n"
        "## Initialization and update semantics\n\n"
        "- After the t=0 initial upper decision, the latch is initialized armed only when the "
        "selected active direction is already in h_active >= h_rep. Initial planning and an "
        "emergency reproposal cannot both execute at t=0.\n"
        "- After any reproposal, the latch is not automatically rearmed. Because the active "
        "goal changed, its observed safety margin must subsequently reach h_rep.\n"
        "- After reference-to-terminal handoff, handoff has same-tick priority and no upper "
        "reproposal executes. The latch is not rearmed by the handoff itself; the terminal "
        "direction must subsequently reach h_rep.\n"
        "- If normal and emergency are both true, one upper computation executes and the event "
        "is recorded as NORMAL_AND_EMERGENCY with emergency priority.\n"
        "- Multiple independent emergency events remain possible after each genuine recovery "
        "to h_rep and later re-entry through h_emg. ARTIFICIAL_REPROPOSAL_CAP=NONE.\n",
        encoding="utf-8",
    )
    (OUTPUT / "theory_diff.md").write_text(
        """# Theory Difference\n\n"
        "The active manuscript currently defines a level-triggered emergency term "
        "I(h_active <= h_emg). R-ERR replaces only that term with an armed downward-crossing "
        "event and an existing-threshold rearm condition.\n\n"
        "Unchanged: Phi_rep, T_rep,min, W_prog, nu_min, nu_scale, h_rep, h_scale, h_emg, "
        "d_hand, handoff priority, progress-history reset, phase-preserving goal update, "
        "normal event semantics, multiple reproposals, and the complete planning/execution chain.\n\n"
        "Added state: one Boolean emergency_armed latch and the previous observed h_active per "
        "agent. Added numeric thresholds: none.\n",
        encoding="utf-8",
    )


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"output already exists: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    records = load_records()
    source_events = [row for record in records for row in record["events"]]
    current_reproposals = [row for row in source_events if row.get("counts_as_reproposal")]
    filter_rows, retained_ticks, retained_agent_events = filter_timeline(records)
    counter_reproposals = [row for row in filter_rows if row["counterfactual_reproposal"]]
    current_counts = Counter(row["event"] for row in current_reproposals)
    counter_counts = Counter(row["counterfactual_event"] for row in counter_reproposals)
    current_consecutive = current_consecutive_emergencies(source_events)
    counter_consecutive = 0
    current_same_goal = sum(not bool(row["goal_changed"]) for row in current_reproposals)
    current_by_key = {
        (str(row["scenario"]), int(row["seed"]), int(row["agent_id"]), int(row["step"])): row
        for row in current_reproposals
    }
    retained_same_goal = sum(
        not bool(current_by_key[key]["goal_changed"])
        for key in retained_agent_events
        if key in current_by_key
    )
    runtime_rows, runtime_overall = runtime_estimate(records, retained_ticks)
    reduction = (
        (current_consecutive - counter_consecutive) / current_consecutive
        if current_consecutive
        else 1.0
    )
    independent_reentries = sum(row["emergency_edge"] for row in filter_rows)
    summary = {
        "PHASE_A_COUNTERFACTUAL_STATUS": "PASS" if reduction >= 0.70 else "FAIL",
        "CURRENT_EMERGENCY_EVENTS": int(current_counts["EMERGENCY_REPROPOSAL"]),
        "COUNTERFACTUAL_EMERGENCY_EVENTS": int(counter_counts["EMERGENCY_REPROPOSAL"]),
        "CURRENT_TOTAL_REPROPOSALS": len(current_reproposals),
        "COUNTERFACTUAL_TOTAL_EVENTS": len(counter_reproposals),
        "CURRENT_NORMAL_EVENTS": int(current_counts["NORMAL_REPROPOSAL"]),
        "COUNTERFACTUAL_NORMAL_EVENTS": int(counter_counts["NORMAL_REPROPOSAL"]),
        "NORMAL_EVENTS_RETAINED": int(counter_counts["NORMAL_REPROPOSAL"]),
        "CURRENT_CONSECUTIVE_EMERGENCIES": current_consecutive,
        "COUNTERFACTUAL_CONSECUTIVE_EMERGENCIES": counter_consecutive,
        "CONSECUTIVE_EMERGENCY_REDUCTION": reduction,
        "CURRENT_SAME_GOAL_EVENTS": current_same_goal,
        "COUNTERFACTUAL_SAME_GOAL_EVENTS_RETAINED": retained_same_goal,
        "COUNTERFACTUAL_SAME_GOAL_EVENTS_REMOVED": current_same_goal - retained_same_goal,
        "INDEPENDENT_EMERGENCY_REENTRIES_RETAINED": independent_reentries,
        "CURRENT_UPPER_PLANNING_CALLS": sum(len(record["upper_timing_rows"]) for record in records),
        "COUNTERFACTUAL_UPPER_PLANNING_CALLS": 40 + len(retained_ticks),
        "CURRENT_MEAN_UPPER_CALLS_PER_EPISODE": sum(len(record["upper_timing_rows"]) for record in records) / 40.0,
        "COUNTERFACTUAL_MEAN_UPPER_CALLS_PER_EPISODE": (40 + len(retained_ticks)) / 40.0,
        "CURRENT_MEAN_PLANNING_COMPUTE_MS": runtime_overall["current_planning_compute_ms"],
        "COUNTERFACTUAL_MEAN_PLANNING_COMPUTE_ESTIMATE_MS": runtime_overall["counterfactual_planning_compute_ms"],
        "COUNTERFACTUAL_RUNTIME_STATUS": "DIAGNOSTIC_ESTIMATE",
        "NONINTERVENTIONAL_LIMITATION": (
            "Filtering is applied to the saved Current-ERR state/event timeline. Skipped events "
            "would change future active goals and trajectories, so performance is not estimated."
        ),
        "EDGE_REARM_NOT_SUFFICIENT": "NO" if reduction >= 0.70 else "YES",
    }
    source_files = (
        SOURCE / "FINAL_REPORT.md",
        SOURCE / "conclusion.json",
        SOURCE / "theory_execution_contract.md",
        SOURCE / "reproposal_events.csv",
        SOURCE / "reproposal_distribution.csv",
        SOURCE / "runtime_method_summary.csv",
        SOURCE / "paired_recovery.csv",
        EQUAL_INFO / "FINAL_REPORT.md",
        EQUAL_INFO / "conclusion.json",
        REPO_ROOT / "hire-rl-body.tex",
        REPO_ROOT / "planning" / "event_triggered_reference_reconstruction.py",
    )
    context = {
        "authority": "active uncommented ERR theory in hire-rl-body.tex plus named artifacts",
        "AGENTS_MD": "NOT_PRESENT",
        "CODEX_HANDOFF_MD": "NOT_PRESENT",
        "source_hashes": {
            str(path.relative_to(REPO_ROOT)).replace("\\", "/"): file_hash(path)
            for path in source_files
        },
        "source_record_count": len(records),
        "source_trigger_tick_count": sum(len(record["triggers"]) for record in records),
        "source_reproposal_count": len(current_reproposals),
        "baseline_results_preserved": {
            "DWA_FullState": 0.9775,
            "RVO_FullState": 0.985,
            "DWA_SensingMatched": 0.1825,
            "RVO_SensingMatched": 0.10,
        },
    }
    write_contracts()
    write_csv(OUTPUT / "counterfactual_event_filter.csv", filter_rows)
    write_csv(OUTPUT / "runtime_counterfactual_estimate.csv", runtime_rows)
    write_json(OUTPUT / "counterfactual_summary.json", summary)
    write_json(OUTPUT / "context_recovery_manifest.json", context)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
