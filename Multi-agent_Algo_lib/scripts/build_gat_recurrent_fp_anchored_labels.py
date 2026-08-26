#!/usr/bin/env python3
"""Build focal-attribution, FP-anchored recurrent GAT supervision.

This is a Train/internal-validation-only label revision.  It reads the already
executed counterfactual branches, never opens Dev/Holdout/Formal, and never
changes the runtime graph.  The FP-SHEP action is the default target; a target
may move away from it only for a focal safety repair or a strict safety-
noninferior material 5 s progress/safety improvement.  Smoothness is emitted
only as an eligible-pair preference after safety and progress equivalence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "artifacts/gat_recurrent_smoothness_rescue/20260821_180244"
SOURCE_EXAMPLES = ROOT / "02_recurrent_dataset/recurrent_training_examples.jsonl"
SOURCE_BRANCHES = ROOT / "03_counterfactual_rollouts/candidate_counterfactual_rollouts.csv"
OUTPUT = ROOT / "13_objective_revision/dataset_v2"
CONTRACT = OUTPUT / "FP_ANCHORED_TARGET_CONTRACT_FREEZE.json"
EXAMPLES = OUTPUT / "recurrent_training_examples_fp_anchored.jsonl"
PAIRS = OUTPUT / "eligible_smoothness_pairs_fp_anchored.csv"
AUDIT = OUTPUT / "fp_anchored_target_audit.csv"
SUMMARY = OUTPUT / "FP_ANCHORED_DATASET_SUMMARY.json"

SCHEMA = "gat_recurrent_fp_anchored_labels_v2"
PRIMARY_HORIZON = "horizon_50"
SERIOUS_BUFFER_M = 0.15
PEER_CAP_M = 0.60
OBSTACLE_CAP_M = 1.15
SAFETY_GAIN_M = 0.15
PROGRESS_GAIN_M = 0.50
DEGRADATION_TOLERANCE_M = 0.0
PAIR_SAFETY_BAND_M = 0.15
PAIR_PROGRESS_BAND_M = 0.10
PAIR_DEVIATION_BAND_M = 0.10
PAIR_JERK_MIN_ABSOLUTE = 10.0
PAIR_JERK_MIN_RELATIVE = 0.05


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["status"])
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def flag(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def number(row: Mapping[str, str], key: str, default: float) -> float:
    value = str(row.get(key, "")).strip()
    return float(value) if value else float(default)


def branch(row: Mapping[str, str]) -> dict[str, Any]:
    peer = number(row, "minimum_peer_margin_m", -1.0e9)
    obstacle = number(row, "minimum_obstacle_clearance_m", -1.0e9)
    focal_collision = flag(row["focal_collision"])
    h4_risky = flag(row["h4_risky"])
    hard_safe = bool(
        not focal_collision
        and not h4_risky
        and peer >= SERIOUS_BUFFER_M
        and obstacle >= SERIOUS_BUFFER_M
    )
    return {
        "class_index": int(row["class_index"]),
        "null_branch": flag(row["null_branch"]),
        "recorded_hard_safe": flag(row["hard_safe"]),
        "hard_safe": hard_safe,
        "team_collision": flag(row["collision"]),
        "focal_collision": focal_collision,
        "peer_collision": flag(row["peer_collision"]),
        "h4_risky": h4_risky,
        "peer_margin": peer,
        "peer_cap": min(peer, PEER_CAP_M),
        "obstacle_margin": obstacle,
        "obstacle_cap": min(obstacle, OBSTACLE_CAP_M),
        "progress": number(row, "terminal_progress_m", -1.0e9),
        "progress_rate": number(row, "progress_per_second_mps", -1.0e9),
        "reference_reached": flag(row["reference_reached"]),
        "emergency": flag(row["subsequent_emergency_trigger"]),
        "deviation": number(row, "execution_deviation_m", 1.0e9),
        "jerk": number(row, "trajectory_smoothness_focal_m2_s6", math.inf),
    }


def rank_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        int(bool(row["hard_safe"])),
        float(row["peer_cap"]),
        float(row["obstacle_cap"]),
        int(not bool(row["emergency"])),
        float(row["progress"]),
        float(row["progress_rate"]),
        int(bool(row["reference_reached"])),
        -float(row["deviation"]),
        -int(row["class_index"]),
    )


def target_for_state(
    fp: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> tuple[Mapping[str, Any], str, list[int]]:
    proposals = [row for row in rows if not row["null_branch"]]
    safe = [row for row in proposals if row["hard_safe"] and not row["h4_risky"]]
    if not fp["hard_safe"] and safe:
        selected = max(safe, key=rank_key)
        return selected, "focal_hard_safety_repair", [int(row["class_index"]) for row in safe]

    corrections = []
    for row in safe:
        if row["peer_cap"] < fp["peer_cap"] - DEGRADATION_TOLERANCE_M:
            continue
        if row["obstacle_cap"] < fp["obstacle_cap"] - DEGRADATION_TOLERANCE_M:
            continue
        if row["progress"] < fp["progress"] - DEGRADATION_TOLERANCE_M:
            continue
        material = bool(
            row["peer_cap"] >= fp["peer_cap"] + SAFETY_GAIN_M
            or row["obstacle_cap"] >= fp["obstacle_cap"] + SAFETY_GAIN_M
            or row["progress"] >= fp["progress"] + PROGRESS_GAIN_M
        )
        if material:
            corrections.append(row)
    if corrections:
        selected = max(corrections, key=rank_key)
        return selected, "strict_safety_noninferior_material_gain", [
            int(row["class_index"]) for row in corrections
        ]
    return fp, "retain_fp_shep", [int(fp["class_index"])]


def smoothness_pairs(rows: Sequence[Mapping[str, Any]]) -> list[tuple[int, int]]:
    proposals = [
        row for row in rows
        if not row["null_branch"] and row["hard_safe"] and math.isfinite(float(row["jerk"]))
    ]
    result: list[tuple[int, int]] = []
    for index, left in enumerate(proposals):
        for right in proposals[index + 1 :]:
            safety_equal = bool(
                abs(float(left["peer_cap"]) - float(right["peer_cap"])) <= PAIR_SAFETY_BAND_M
                and abs(float(left["obstacle_cap"]) - float(right["obstacle_cap"])) <= PAIR_SAFETY_BAND_M
                and bool(left["emergency"]) == bool(right["emergency"])
            )
            progress_equal = bool(
                abs(float(left["progress"]) - float(right["progress"])) <= PAIR_PROGRESS_BAND_M
                and bool(left["reference_reached"]) == bool(right["reference_reached"])
                and abs(float(left["deviation"]) - float(right["deviation"])) <= PAIR_DEVIATION_BAND_M
            )
            if not (safety_equal and progress_equal):
                continue
            left_jerk = float(left["jerk"])
            right_jerk = float(right["jerk"])
            difference = abs(left_jerk - right_jerk)
            threshold = max(
                PAIR_JERK_MIN_ABSOLUTE,
                PAIR_JERK_MIN_RELATIVE * max(abs(left_jerk), abs(right_jerk), 1.0e-12),
            )
            if difference < threshold:
                continue
            preferred, disfavored = (left, right) if left_jerk < right_jerk else (right, left)
            result.append((int(preferred["class_index"]), int(disfavored["class_index"])))
    return result


def prepare() -> None:
    if CONTRACT.exists() or EXAMPLES.exists() or PAIRS.exists():
        raise RuntimeError("FP-anchored target dataset was already prepared")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    contract = {
        "schema_version": SCHEMA,
        "status": "FROZEN_BEFORE_LABEL_MATERIALIZATION_AND_RETRAINING",
        "source_examples": str(SOURCE_EXAMPLES.relative_to(REPO_ROOT).as_posix()),
        "source_examples_sha256": sha256_file(SOURCE_EXAMPLES),
        "source_branches": str(SOURCE_BRANCHES.relative_to(REPO_ROOT).as_posix()),
        "source_branches_sha256": sha256_file(SOURCE_BRANCHES),
        "builder_script": str(Path(__file__).resolve().relative_to(REPO_ROOT).as_posix()),
        "builder_script_sha256": sha256_file(Path(__file__).resolve()),
        "primary_horizon": PRIMARY_HORIZON,
        "hard_safety_attribution": "focal collision, focal margins, existing per-candidate H4 risk; background-only team collision excluded",
        "default_target": "FP-SHEP selected class",
        "thresholds": {
            "serious_buffer_m": SERIOUS_BUFFER_M,
            "peer_cap_m": PEER_CAP_M,
            "obstacle_cap_m": OBSTACLE_CAP_M,
            "safety_gain_m": SAFETY_GAIN_M,
            "progress_gain_m": PROGRESS_GAIN_M,
            "degradation_tolerance_m": DEGRADATION_TOLERANCE_M,
            "pair_safety_band_m": PAIR_SAFETY_BAND_M,
            "pair_progress_band_m": PAIR_PROGRESS_BAND_M,
            "pair_deviation_band_m": PAIR_DEVIATION_BAND_M,
            "pair_jerk_min_absolute_m2_s6": PAIR_JERK_MIN_ABSOLUTE,
            "pair_jerk_min_relative": PAIR_JERK_MIN_RELATIVE,
        },
        "target_order": ["focal hard safety", "peer and obstacle safety", "5 s progress and viability", "execution deviation"],
        "smoothness_role": "secondary pairwise preference only after safety/progress equivalence",
        "development_read": False,
        "holdout_read": False,
        "formal_v1_rows_read": 0,
        "formal_v2_generated": False,
        "runtime_graph_changed": False,
        "runtime_input_semantics_changed": False,
    }
    write_json(CONTRACT, contract)
    print(json.dumps(contract, indent=2, ensure_ascii=False))


def verify_contract() -> dict[str, Any]:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    checks = {
        "examples": sha256_file(SOURCE_EXAMPLES) == contract["source_examples_sha256"],
        "branches": sha256_file(SOURCE_BRANCHES) == contract["source_branches_sha256"],
        "script": sha256_file(Path(__file__).resolve()) == contract["builder_script_sha256"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"FP-anchored target contract mismatch: {checks}")
    return contract


def build() -> None:
    contract = verify_contract()
    if EXAMPLES.exists() or PAIRS.exists() or AUDIT.exists() or SUMMARY.exists():
        raise RuntimeError("FP-anchored target outputs already exist; refusing overwrite")
    source_examples = read_jsonl(SOURCE_EXAMPLES)
    by_state: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in read_csv(SOURCE_BRANCHES):
        if row["horizon"] == PRIMARY_HORIZON:
            by_state[str(row["state_id"])][int(row["class_index"])] = branch(row)

    output_examples: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    partition_reasons: dict[str, Counter[str]] = defaultdict(Counter)
    corrected_safe = corrected_unsafe = background_only = 0
    for example in source_examples:
        state_id = str(example["state_id"])
        rows = list(by_state[state_id].values())
        fp_class = int(example["behavior_fp_selected_class"])
        if fp_class not in by_state[state_id]:
            raise RuntimeError(f"missing FP branch: {state_id}/{fp_class}")
        fp = by_state[state_id][fp_class]
        target, reason, eligible = target_for_state(fp, rows)
        target_class = int(target["class_index"])
        pairs = smoothness_pairs(rows)
        reasons[reason] += 1
        partition_reasons[str(example["partition"])][reason] += 1
        corrected_safe += sum(int(row["hard_safe"]) for row in rows)
        corrected_unsafe += sum(int(not row["hard_safe"]) for row in rows)
        background_only += sum(int(row["team_collision"] and not row["focal_collision"]) for row in rows)
        class_count = int(example["class_count"])
        soft = [0.0] * class_count
        soft[target_class] = 1.0
        quality = [
            -1.0 if class_index in by_state[state_id] and not by_state[state_id][class_index]["hard_safe"]
            else 1.0 if class_index == target_class
            else 0.0
            for class_index in range(class_count)
        ]
        output_examples.append(
            {
                **example,
                "schema_version": SCHEMA,
                "soft_target_gat_r": soft,
                "target_quality_gat_r": quality,
                "reference_classes": [target_class],
                "smoothness_pairs": [list(pair) for pair in pairs],
                "smoothness_pair_count": len(pairs),
                "selection_trace": [reason],
                "label_horizon": PRIMARY_HORIZON,
                "fp_anchor_target_reason": reason,
                "fp_anchor_eligible_correction_classes": eligible,
                "background_only_team_collision_excluded_from_hard_safety": True,
            }
        )
        audit_rows.append(
            {
                "schema_version": SCHEMA,
                "state_id": state_id,
                "scenario_id": example["scenario_id"],
                "partition": example["partition"],
                "stage": example["stage"],
                "family": example["family"],
                "fp_class": fp_class,
                "target_class": target_class,
                "switched_from_fp": target_class != fp_class,
                "reason": reason,
                "eligible_classes": json.dumps(eligible),
                "fp_hard_safe": fp["hard_safe"],
                "target_hard_safe": target["hard_safe"],
                "fp_peer_margin_m": fp["peer_margin"],
                "target_peer_margin_m": target["peer_margin"],
                "fp_obstacle_margin_m": fp["obstacle_margin"],
                "target_obstacle_margin_m": target["obstacle_margin"],
                "fp_progress_m": fp["progress"],
                "target_progress_m": target["progress"],
                "progress_gain_m": target["progress"] - fp["progress"],
                "smoothness_pair_count": len(pairs),
            }
        )
        row_index = {int(row["class_index"]): row for row in rows}
        for preferred, disfavored in pairs:
            pair_rows.append(
                {
                    "schema_version": SCHEMA,
                    "state_id": state_id,
                    "scenario_id": example["scenario_id"],
                    "partition": example["partition"],
                    "preferred_class_index": preferred,
                    "disfavored_class_index": disfavored,
                    "preferred_jerk_m2_s6": row_index[preferred]["jerk"],
                    "disfavored_jerk_m2_s6": row_index[disfavored]["jerk"],
                    "safety_equivalent": True,
                    "progress_equivalent": True,
                    "null_involved": False,
                }
            )

    write_jsonl(EXAMPLES, output_examples)
    write_csv(PAIRS, pair_rows)
    write_csv(AUDIT, audit_rows)
    partitions = Counter(str(row["partition"]) for row in output_examples)
    summary = {
        "schema_version": SCHEMA,
        "status": "PASS",
        "contract_sha256": sha256_file(CONTRACT),
        "state_count": len(output_examples),
        "partition_state_counts": dict(partitions),
        "scene_count": len({str(row["scenario_id"]) for row in output_examples}),
        "target_reason_counts": dict(reasons),
        "partition_target_reason_counts": {key: dict(value) for key, value in partition_reasons.items()},
        "switch_from_fp_count": sum(bool(row["switched_from_fp"]) for row in audit_rows),
        "switch_from_fp_rate": sum(bool(row["switched_from_fp"]) for row in audit_rows) / len(audit_rows),
        "corrected_hard_safe_branch_count": corrected_safe,
        "corrected_hard_unsafe_branch_count": corrected_unsafe,
        "background_only_team_collision_branch_count_excluded": background_only,
        "eligible_smoothness_pair_count": len(pair_rows),
        "examples_sha256": sha256_file(EXAMPLES),
        "pairs_sha256": sha256_file(PAIRS),
        "audit_sha256": sha256_file(AUDIT),
        "development_read": False,
        "holdout_read": False,
        "formal_v1_rows_read": 0,
        "formal_v2_generated": False,
        "runtime_graph_changed": False,
        "runtime_input_semantics_changed": False,
        "target_contract": contract,
    }
    write_json(SUMMARY, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "build"))
    args = parser.parse_args()
    prepare() if args.phase == "prepare" else build()


if __name__ == "__main__":
    main()
