#!/usr/bin/env python3
"""Diagnose GAT-R target drift and predeclare conservative FP-anchored corrections.

Only frozen training/internal-validation records are read. Development, Holdout,
Formal V1 episode rows, and Formal V2 are not inputs.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "artifacts/gat_recurrent_smoothness_rescue/20260821_180244"
INPUT_EXAMPLES = ROOT / "02_recurrent_dataset/recurrent_training_examples.jsonl"
INPUT_BRANCHES = ROOT / "03_counterfactual_rollouts/candidate_counterfactual_rollouts.csv"
OUTPUT = ROOT / "13_objective_revision"
SCHEMA = "gat_recurrent_fp_anchor_diagnostic_v2"
SERIOUS_NEAR_COLLISION_BUFFER_M = 0.15


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def flag(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def numeric(row: Mapping[str, str], key: str, default: float) -> float:
    value = str(row.get(key, "")).strip()
    return float(value) if value else float(default)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def branch_view(row: Mapping[str, str]) -> dict[str, Any]:
    focal_collision = flag(row["focal_collision"])
    h4_risky = flag(row["h4_risky"])
    peer_margin = numeric(row, "minimum_peer_margin_m", -1e9)
    obstacle_margin = numeric(row, "minimum_obstacle_clearance_m", -1e9)
    focal_hard_safe = bool(
        not focal_collision
        and not h4_risky
        and peer_margin >= SERIOUS_NEAR_COLLISION_BUFFER_M
        and obstacle_margin >= SERIOUS_NEAR_COLLISION_BUFFER_M
    )
    return {
        "class_index": int(row["class_index"]),
        "null_branch": flag(row["null_branch"]),
        "recorded_hard_safe": flag(row["hard_safe"]),
        # A candidate branch changes only the focal UAV's temporary reference.
        # Team collisions that do not involve that UAV are context outcomes, not
        # evidence that the focal candidate itself is unsafe.
        "hard_safe": focal_hard_safe,
        "collision": flag(row["collision"]),
        "focal_collision": focal_collision,
        "peer_collision": flag(row["peer_collision"]),
        "h4_risky": h4_risky,
        "peer_margin": peer_margin,
        "obstacle_margin": obstacle_margin,
        "peer_margin_capped": numeric(row, "peer_margin_capped_m", -1e9),
        "obstacle_margin_capped": numeric(row, "obstacle_clearance_capped_m", -1e9),
        "progress": numeric(row, "terminal_progress_m", -1e9),
        "reference_reached": flag(row["reference_reached"]),
        "deviation": numeric(row, "execution_deviation_m", 1e9),
        "jerk": numeric(row, "trajectory_smoothness_focal_m2_s6", 1e9),
        "emergency": flag(row["subsequent_emergency_trigger"]),
    }


def rank_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """Safety first, then viability, progress, deviation; jerk is excluded."""
    return (
        int(bool(row["hard_safe"])),
        int(not bool(row["h4_risky"])),
        float(row["peer_margin_capped"]),
        float(row["obstacle_margin_capped"]),
        int(not bool(row["emergency"])),
        float(row["progress"]),
        int(bool(row["reference_reached"])),
        -float(row["deviation"]),
        -int(row["class_index"]),
    )


def conservative_target(
    fp: Mapping[str, Any],
    proposals: list[Mapping[str, Any]],
) -> tuple[int, str, list[int]]:
    """Return a strict FP-anchored safety/progress correction target.

    No smoothness value participates. The rule switches away from FP only for a
    hard-safety repair, an H4-risk repair, or strict non-inferior safety plus a
    material progress/clearance improvement.
    """
    non_null = [row for row in proposals if not row["null_branch"]]
    hard_safe = [row for row in non_null if row["hard_safe"]]
    if not fp["hard_safe"] and hard_safe:
        best = max(hard_safe, key=rank_key)
        return int(best["class_index"]), "repair_hard_unsafe_fp", [int(row["class_index"]) for row in hard_safe]

    nonrisk_safe = [row for row in hard_safe if not row["h4_risky"]]
    if fp["h4_risky"] and nonrisk_safe:
        best = max(nonrisk_safe, key=rank_key)
        return int(best["class_index"]), "repair_h4_risky_fp", [int(row["class_index"]) for row in nonrisk_safe]

    # Strict tolerance bands are deliberately narrower than the original 0.15 m
    # safety-equivalence band, because FP is the proven closed-loop reference.
    peer_tol = 0.05
    obstacle_tol = 0.05
    progress_tol = 0.05
    peer_gain = 0.15
    obstacle_gain = 0.15
    progress_gain = 0.20
    dominated = []
    for row in nonrisk_safe:
        if row["peer_margin_capped"] < fp["peer_margin_capped"] - peer_tol:
            continue
        if row["obstacle_margin_capped"] < fp["obstacle_margin_capped"] - obstacle_tol:
            continue
        if row["progress"] < fp["progress"] - progress_tol:
            continue
        material = (
            row["peer_margin_capped"] >= fp["peer_margin_capped"] + peer_gain
            or row["obstacle_margin_capped"] >= fp["obstacle_margin_capped"] + obstacle_gain
            or row["progress"] >= fp["progress"] + progress_gain
        )
        if material:
            dominated.append(row)
    if dominated:
        best = max(dominated, key=rank_key)
        return int(best["class_index"]), "strict_pareto_correction", [int(row["class_index"]) for row in dominated]
    return int(fp["class_index"]), "retain_fp", [int(fp["class_index"])]


def correction_available(
    fp: Mapping[str, Any],
    proposals: list[Mapping[str, Any]],
    *,
    degradation_tolerance: float,
    safety_gain: float,
    progress_gain: float,
) -> tuple[bool, bool, bool]:
    """Report whether a strict noninferior safety/progress correction exists."""
    any_correction = False
    any_safety_gain = False
    any_progress_gain = False
    for row in proposals:
        if row["null_branch"] or not row["hard_safe"] or row["h4_risky"]:
            continue
        if row["peer_margin_capped"] < fp["peer_margin_capped"] - degradation_tolerance:
            continue
        if row["obstacle_margin_capped"] < fp["obstacle_margin_capped"] - degradation_tolerance:
            continue
        if row["progress"] < fp["progress"] - degradation_tolerance:
            continue
        safety_better = bool(
            row["peer_margin_capped"] >= fp["peer_margin_capped"] + safety_gain
            or row["obstacle_margin_capped"] >= fp["obstacle_margin_capped"] + safety_gain
        )
        progress_better = bool(row["progress"] >= fp["progress"] + progress_gain)
        if safety_better or progress_better:
            any_correction = True
            any_safety_gain |= safety_better
            any_progress_gain |= progress_better
    return any_correction, any_safety_gain, any_progress_gain


def main() -> None:
    examples = read_jsonl(INPUT_EXAMPLES)
    by_state: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    horizon_rows = 0
    for row in read_csv(INPUT_BRANCHES):
        if row["horizon"] != "horizon_50":
            continue
        horizon_rows += 1
        by_state[row["state_id"]][int(row["class_index"])] = branch_view(row)
    if horizon_rows != 11852:
        raise RuntimeError(f"expected 11852 primary branches, observed {horizon_rows}")

    diagnostics: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    partition_reasons: dict[str, Counter[str]] = defaultdict(Counter)
    current_target_mismatch = 0
    fp_unsafe_recoverable = 0
    fp_h4_recoverable = 0
    current_target_contains_fp = 0
    current_target_all_hard_safe = 0
    current_target_any_hard_unsafe = 0
    background_only_collision_rows = 0
    focal_collision_rows = 0
    peer_collision_rows = 0
    h4_risky_rows = 0
    recorded_unsafe_corrected_safe_rows = 0
    corrected_unsafe_rows = 0
    state_uniform_team_collision = 0
    state_mixed_team_collision = 0
    state_any_focal_collision = 0
    state_mixed_focal_collision = 0
    sensitivity: dict[str, Counter[str]] = {}
    for tolerance in (0.0, 0.025, 0.05):
        for safety_gain in (0.15, 0.30):
            for progress_gain in (0.20, 0.50, 1.00):
                key = f"tol={tolerance:.3f}|safety_gain={safety_gain:.2f}|progress_gain={progress_gain:.2f}"
                sensitivity[key] = Counter()

    for state_rows in by_state.values():
        non_null = [row for row in state_rows.values() if not row["null_branch"]]
        team_values = {bool(row["collision"]) for row in non_null}
        focal_values = {bool(row["focal_collision"]) for row in non_null}
        state_uniform_team_collision += int(team_values == {True})
        state_mixed_team_collision += int(len(team_values) > 1)
        state_any_focal_collision += int(any(row["focal_collision"] for row in non_null))
        state_mixed_focal_collision += int(len(focal_values) > 1)
        for row in state_rows.values():
            background_only_collision_rows += int(row["collision"] and not row["focal_collision"])
            focal_collision_rows += int(row["focal_collision"])
            peer_collision_rows += int(row["peer_collision"])
            h4_risky_rows += int(row["h4_risky"])
            recorded_unsafe_corrected_safe_rows += int(
                not row["recorded_hard_safe"] and row["hard_safe"]
            )
            corrected_unsafe_rows += int(not row["hard_safe"])

    for example in examples:
        state_id = str(example["state_id"])
        branches = by_state[state_id]
        fp_class = int(example["behavior_fp_selected_class"])
        if fp_class not in branches:
            raise RuntimeError(f"FP class missing for {state_id}: {fp_class}")
        fp = branches[fp_class]
        proposals = list(branches.values())
        for key, counts in sensitivity.items():
            components = {
                item.split("=")[0]: float(item.split("=")[1])
                for item in key.split("|")
            }
            available, safety_better, progress_better = correction_available(
                fp,
                proposals,
                degradation_tolerance=components["tol"],
                safety_gain=components["safety_gain"],
                progress_gain=components["progress_gain"],
            )
            counts["any"] += int(available)
            counts["safety"] += int(safety_better)
            counts["progress"] += int(progress_better)
        current_refs = [int(value) for value in example["reference_classes"]]
        target_class, reason, eligible = conservative_target(fp, proposals)
        reasons[reason] += 1
        partition_reasons[str(example["partition"])][reason] += 1
        current_target_mismatch += int(fp_class not in current_refs)
        current_target_contains_fp += int(fp_class in current_refs)
        current_target_all_hard_safe += int(all(branches[value]["hard_safe"] for value in current_refs))
        current_target_any_hard_unsafe += int(any(not branches[value]["hard_safe"] for value in current_refs))
        safe_alt = [row for row in proposals if not row["null_branch"] and row["hard_safe"]]
        nonrisk_alt = [row for row in safe_alt if not row["h4_risky"]]
        fp_unsafe_recoverable += int(not fp["hard_safe"] and bool(safe_alt))
        fp_h4_recoverable += int(fp["h4_risky"] and bool(nonrisk_alt))
        selected = branches[target_class]
        diagnostics.append(
            {
                "schema_version": SCHEMA,
                "state_id": state_id,
                "scenario_id": example["scenario_id"],
                "partition": example["partition"],
                "stage": example["stage"],
                "family": example["family"],
                "progress_bin": example["progress_bin"],
                "replan_bin": example["replan_bin"],
                "fp_class": fp_class,
                "current_gat_r_reference_classes": json.dumps(current_refs),
                "current_target_contains_fp": fp_class in current_refs,
                "conservative_target_class": target_class,
                "conservative_switch": target_class != fp_class,
                "conservative_reason": reason,
                "eligible_correction_classes": json.dumps(eligible),
                "fp_hard_safe": fp["hard_safe"],
                "target_hard_safe": selected["hard_safe"],
                "fp_recorded_hard_safe": fp["recorded_hard_safe"],
                "target_recorded_hard_safe": selected["recorded_hard_safe"],
                "fp_h4_risky": fp["h4_risky"],
                "target_h4_risky": selected["h4_risky"],
                "fp_peer_margin_m": fp["peer_margin"],
                "target_peer_margin_m": selected["peer_margin"],
                "fp_obstacle_margin_m": fp["obstacle_margin"],
                "target_obstacle_margin_m": selected["obstacle_margin"],
                "fp_progress_m": fp["progress"],
                "target_progress_m": selected["progress"],
                "fp_jerk_m2_s6": fp["jerk"],
                "target_jerk_m2_s6": selected["jerk"],
            }
        )

    total_states = len(examples)
    summary = {
        "schema_version": SCHEMA,
        "status": "PASS",
        "data_scope": "training plus internal validation only",
        "development_performance_read": False,
        "holdout_read": False,
        "formal_v1_episode_rows_read": 0,
        "formal_v2_generated": False,
        "state_count": total_states,
        "horizon_50_branch_count": horizon_rows,
        "current_gat_r_target_contains_fp_count": current_target_contains_fp,
        "current_gat_r_target_contains_fp_rate": current_target_contains_fp / total_states,
        "current_gat_r_target_excludes_fp_count": current_target_mismatch,
        "current_gat_r_target_excludes_fp_rate": current_target_mismatch / total_states,
        "current_target_all_hard_safe_count": current_target_all_hard_safe,
        "current_target_any_hard_unsafe_count": current_target_any_hard_unsafe,
        "fp_hard_unsafe_with_safe_alternative_count": fp_unsafe_recoverable,
        "fp_h4_risky_with_nonrisk_safe_alternative_count": fp_h4_recoverable,
        "primary_branch_collision_count": sum(row["collision"] for state in by_state.values() for row in state.values()),
        "background_only_collision_count": background_only_collision_rows,
        "focal_collision_count": focal_collision_rows,
        "explicit_peer_collision_count": peer_collision_rows,
        "h4_risky_branch_count": h4_risky_rows,
        "recorded_unsafe_but_focal_attribution_safe_branch_count": recorded_unsafe_corrected_safe_rows,
        "focal_attribution_hard_unsafe_branch_count": corrected_unsafe_rows,
        "state_all_non_null_candidates_team_collide_count": state_uniform_team_collision,
        "state_mixed_team_collision_across_candidates_count": state_mixed_team_collision,
        "state_any_focal_collision_count": state_any_focal_collision,
        "state_mixed_focal_collision_across_candidates_count": state_mixed_focal_collision,
        "conservative_target_reason_counts": dict(reasons),
        "conservative_target_switch_count": total_states - reasons["retain_fp"],
        "conservative_target_switch_rate": (total_states - reasons["retain_fp"]) / total_states,
        "partition_reason_counts": {key: dict(value) for key, value in partition_reasons.items()},
        "strict_correction_threshold_sensitivity": {
            key: {
                **dict(counts),
                "rate": counts["any"] / total_states,
            }
            for key, counts in sensitivity.items()
        },
        "proposed_next_target": {
            "hard_safety_attribution": "focal collision/margins plus existing candidate H4 risk; background-only team collision excluded",
            "default": "one-hot FP-SHEP selected class",
            "corrections": [
                "FP branch hard-unsafe and a hard-safe candidate exists",
                "FP branch H4-risky and a hard-safe nonrisk candidate exists",
                "strict safety-noninferior candidate with material peer/obstacle/progress gain",
            ],
            "smoothness_used": False,
            "runtime_change": False,
            "selection_before_training_performance": True,
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "fp_anchor_state_diagnostic.csv", diagnostics)
    write_json(OUTPUT / "FP_ANCHORED_TARGET_DIAGNOSTIC.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
