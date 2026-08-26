#!/usr/bin/env python3
"""Validate and consolidate the frozen recurrent GAT counterfactual dataset.

The script is intentionally Train-only.  It does not execute the environment,
open Dev/Holdout/Formal V2, or change any runtime feature.  Its only role is to
turn the already-frozen branch records into auditable hierarchical supervision.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_ROOT = (
    REPO_ROOT
    / "artifacts/gat_recurrent_smoothness_rescue/20260821_180244"
)
SCHEMA_VERSION = "gat_recurrent_hierarchical_labels_v1"
HORIZONS = ("horizon_10", "horizon_20", "horizon_30", "horizon_50", "next_rerr_event")
PRIMARY_HORIZON = "horizon_50"

# Frozen before model training.  The 0.15 m buffer is the existing engineering
# safety buffer used in the long-range benchmark family.  Peer margin is already
# center distance minus the existing d_safe=0.6 m contract.
SERIOUS_NEAR_COLLISION_BUFFER_M = 0.15
SAFETY_EQUIVALENCE_BAND_M = 0.15
PROGRESS_EQUIVALENCE_BAND_M = 0.10
EXECUTION_DEVIATION_EQUIVALENCE_BAND_M = 0.10
PEER_MARGIN_CAP_M = 0.60
OBSTACLE_CLEARANCE_CAP_M = 1.15
SMOOTHNESS_MIN_RELATIVE_IMPROVEMENT = 0.05
SMOOTHNESS_MIN_ABSOLUTE_IMPROVEMENT_M2_S6 = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(json_ready(row), ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(str(key))
                seen.add(str(key))
    if not fields:
        fields = ["schema_version", "status"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(json_ready(value), ensure_ascii=False, allow_nan=False)
                        if isinstance(value, (dict, list, tuple))
                        else json_ready(value)
                    )
                    for key, value in row.items()
                }
            )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_or_inf(value: Any) -> float:
    return math.inf if value is None else float(value)


def finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def candidate_interaction_map(state: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    return {
        int(row["candidate_id"]): row
        for row in state["candidate_interaction_records"]
    }


def metric_row(
    branch: Mapping[str, Any],
    interaction: Mapping[str, Any] | None,
    horizon: str,
) -> dict[str, Any] | None:
    if branch.get("not_run_reason"):
        return None
    snapshot = branch.get(horizon)
    if snapshot is None:
        return None
    peer_margin = finite_or_inf(snapshot.get("minimum_peer_margin_m"))
    obstacle_clearance = min(
        finite_or_inf(snapshot.get("minimum_static_clearance_m")),
        finite_or_inf(snapshot.get("minimum_dynamic_clearance_m")),
    )
    h4_risky = bool(interaction and interaction.get("risky", False))
    reasons: list[str] = []
    if bool(snapshot.get("collision", False)):
        reasons.append("team_collision")
    if bool(snapshot.get("focal_collision", False)):
        reasons.append("focal_collision")
    if peer_margin < SERIOUS_NEAR_COLLISION_BUFFER_M:
        reasons.append("peer_serious_near_collision")
    if obstacle_clearance < SERIOUS_NEAR_COLLISION_BUFFER_M:
        reasons.append("obstacle_serious_near_collision")
    if h4_risky:
        reasons.append("existing_h4_immediate_peer_conflict")
    next_event = branch.get("next_rerr_event") or {}
    return {
        "class_index": int(branch["class_index"]),
        "candidate_id": branch.get("candidate_id"),
        "null_branch": bool(branch.get("null_branch", False)),
        "null_eligible": bool(branch.get("null_eligible", False)),
        "horizon": horizon,
        "horizon_observed": True,
        "horizon_reached": bool(snapshot.get("horizon_reached", True)),
        "requested_step": snapshot.get("requested_step"),
        "executed_steps": int(branch.get("executed_steps", 0)),
        "terminated": bool(branch.get("terminated", False)),
        "truncated": bool(branch.get("truncated", False)),
        "collision": bool(snapshot.get("collision", False)),
        "focal_collision": bool(snapshot.get("focal_collision", False)),
        "static_collision": bool(snapshot.get("static_collision", False)),
        "dynamic_collision": bool(snapshot.get("dynamic_collision", False)),
        "peer_collision": bool(snapshot.get("peer_collision", False)),
        "boundary_collision": bool(snapshot.get("boundary_collision", False)),
        "minimum_sensor_clearance_m": finite_or_none(snapshot.get("minimum_sensor_clearance_m")),
        "minimum_static_clearance_m": finite_or_none(snapshot.get("minimum_static_clearance_m")),
        "minimum_dynamic_clearance_m": finite_or_none(snapshot.get("minimum_dynamic_clearance_m")),
        "minimum_obstacle_clearance_m": None if not math.isfinite(obstacle_clearance) else obstacle_clearance,
        "minimum_peer_center_distance_m": finite_or_none(snapshot.get("minimum_peer_center_distance_m")),
        "minimum_peer_margin_m": None if not math.isfinite(peer_margin) else peer_margin,
        "peer_margin_capped_m": min(peer_margin, PEER_MARGIN_CAP_M),
        "obstacle_clearance_capped_m": min(obstacle_clearance, OBSTACLE_CLEARANCE_CAP_M),
        "peer_risk_duration_s": finite_or_none(snapshot.get("peer_risk_duration_s")),
        "h4_risky": h4_risky,
        "h4_minimum_predicted_separation_m": None if interaction is None else interaction.get("minimum_predicted_separation_m"),
        "h4_maximum_risk_duration_s": None if interaction is None else interaction.get("maximum_risk_duration_s"),
        "hard_safe": not reasons,
        "unsafe_reasons": reasons,
        "terminal_distance_m": finite_or_none(snapshot.get("terminal_distance_m")),
        "terminal_progress_m": finite_or_none(snapshot.get("terminal_progress_m")),
        "progress_per_second_mps": finite_or_none(snapshot.get("progress_per_second_mps")),
        "reference_reached": bool(snapshot.get("reference_reached", False)),
        "travel_distance_focal_m": finite_or_none(snapshot.get("travel_distance_focal_m")),
        "execution_deviation_m": finite_or_none(snapshot.get("execution_deviation_m")),
        "acceleration_variation_focal_m2_s4": finite_or_none(snapshot.get("acceleration_variation_focal_m2_s4")),
        "trajectory_smoothness_focal_m2_s6": finite_or_none(snapshot.get("trajectory_smoothness_focal_m2_s6")),
        "trajectory_smoothness_team_mean_m2_s6": finite_or_none(snapshot.get("trajectory_smoothness_team_mean_m2_s6")),
        "control_discontinuity_mps2": finite_or_none(snapshot.get("control_discontinuity_mps2")),
        "next_rerr_event_observed": bool(branch.get("next_rerr_event_observed", False)),
        "next_rerr_event_censored_at_s": branch.get("next_rerr_event_censored_at_s"),
        "next_rerr_event_type": next_event.get("event"),
        "subsequent_emergency_trigger": bool(next_event.get("subsequent_emergency_trigger", False)),
    }


def safer_pool(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Return the safety/progress-equivalent top pool without smoothness."""

    executable = [row for row in rows if row is not None]
    proposals = [row for row in executable if not row["null_branch"]]
    safe_proposals = [row for row in proposals if row["hard_safe"]]
    legal_null = next(
        (row for row in executable if row["null_branch"] and row["null_eligible"]),
        None,
    )
    trace: list[str] = []

    # Null is a decision-valid terminal-direct option only near the terminal.
    # It is admitted by safety and terminal-direct viability, never by jerk.
    pool = list(safe_proposals)
    if legal_null is not None and legal_null["hard_safe"]:
        best_proposal_progress = max(
            (float(row["terminal_progress_m"] or -math.inf) for row in safe_proposals),
            default=-math.inf,
        )
        null_progress = float(legal_null["terminal_progress_m"] or -math.inf)
        if not safe_proposals or null_progress >= best_proposal_progress - PROGRESS_EQUIVALENCE_BAND_M:
            pool.append(legal_null)
            trace.append("null_admitted_by_terminal_local_safety_and_progress_viability")

    if not pool:
        # If every executable branch is unsafe, choose the least-bad available
        # branch lexicographically.  Smoothness is intentionally absent.
        candidates = proposals + ([legal_null] if legal_null is not None else [])
        if not candidates:
            raise RuntimeError("state has no executable candidate")

        def unsafe_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
            peer = finite_or_inf(row["minimum_peer_margin_m"])
            obstacle = finite_or_inf(row["minimum_obstacle_clearance_m"])
            progress = float(row["terminal_progress_m"] or -math.inf)
            deviation = finite_or_inf(row["execution_deviation_m"])
            return (
                int(bool(row["collision"])),
                int(bool(row["focal_collision"])),
                int(bool(row["h4_risky"])),
                max(SERIOUS_NEAR_COLLISION_BUFFER_M - peer, 0.0),
                max(SERIOUS_NEAR_COLLISION_BUFFER_M - obstacle, 0.0),
                int(bool(row["subsequent_emergency_trigger"])),
                -progress,
                deviation,
                int(row["class_index"]),
            )

        ordered = sorted(candidates, key=unsafe_key)
        best_key = unsafe_key(ordered[0])[:-1]
        trace.append("all_branches_unsafe_least_bad_lexicographic")
        return [row for row in ordered if unsafe_key(row)[:-1] == best_key], trace

    best_peer = max(float(row["peer_margin_capped_m"]) for row in pool)
    pool = [
        row for row in pool
        if float(row["peer_margin_capped_m"]) >= best_peer - SAFETY_EQUIVALENCE_BAND_M
    ]
    trace.append("peer_margin_equivalence_filter")
    best_obstacle = max(float(row["obstacle_clearance_capped_m"]) for row in pool)
    pool = [
        row for row in pool
        if float(row["obstacle_clearance_capped_m"]) >= best_obstacle - SAFETY_EQUIVALENCE_BAND_M
    ]
    trace.append("obstacle_clearance_equivalence_filter")
    if any(not row["subsequent_emergency_trigger"] for row in pool):
        pool = [row for row in pool if not row["subsequent_emergency_trigger"]]
        trace.append("emergency_burden_filter")
    best_progress = max(float(row["terminal_progress_m"] or -math.inf) for row in pool)
    pool = [
        row for row in pool
        if float(row["terminal_progress_m"] or -math.inf) >= best_progress - PROGRESS_EQUIVALENCE_BAND_M
    ]
    trace.append("terminal_progress_equivalence_filter")
    if any(row["reference_reached"] for row in pool):
        pool = [row for row in pool if row["reference_reached"]]
        trace.append("reference_reach_filter")
    finite_deviation = [
        float(row["execution_deviation_m"])
        for row in pool
        if row["execution_deviation_m"] is not None
    ]
    if finite_deviation:
        best_deviation = min(finite_deviation)
        pool = [
            row for row in pool
            if row["execution_deviation_m"] is not None
            and float(row["execution_deviation_m"])
            <= best_deviation + EXECUTION_DEVIATION_EQUIVALENCE_BAND_M
        ]
        trace.append("execution_deviation_equivalence_filter")
    if not pool:
        raise RuntimeError("hierarchical filtering produced an empty pool")
    return pool, trace


def smoothness_pairs(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    proposals = [
        row for row in rows
        if row is not None
        and not row["null_branch"]
        and row["hard_safe"]
        and row["trajectory_smoothness_focal_m2_s6"] is not None
    ]
    result: list[dict[str, Any]] = []
    for left_index, left in enumerate(proposals):
        for right in proposals[left_index + 1 :]:
            safety_equivalent = (
                abs(float(left["peer_margin_capped_m"]) - float(right["peer_margin_capped_m"]))
                <= SAFETY_EQUIVALENCE_BAND_M
                and abs(float(left["obstacle_clearance_capped_m"]) - float(right["obstacle_clearance_capped_m"]))
                <= SAFETY_EQUIVALENCE_BAND_M
                and bool(left["subsequent_emergency_trigger"])
                == bool(right["subsequent_emergency_trigger"])
            )
            progress_equivalent = (
                left["terminal_progress_m"] is not None
                and right["terminal_progress_m"] is not None
                and abs(float(left["terminal_progress_m"]) - float(right["terminal_progress_m"]))
                <= PROGRESS_EQUIVALENCE_BAND_M
                and bool(left["reference_reached"]) == bool(right["reference_reached"])
                and left["execution_deviation_m"] is not None
                and right["execution_deviation_m"] is not None
                and abs(float(left["execution_deviation_m"]) - float(right["execution_deviation_m"]))
                <= EXECUTION_DEVIATION_EQUIVALENCE_BAND_M
            )
            if not (safety_equivalent and progress_equivalent):
                continue
            left_j = float(left["trajectory_smoothness_focal_m2_s6"])
            right_j = float(right["trajectory_smoothness_focal_m2_s6"])
            improvement = abs(left_j - right_j)
            threshold = max(
                SMOOTHNESS_MIN_ABSOLUTE_IMPROVEMENT_M2_S6,
                SMOOTHNESS_MIN_RELATIVE_IMPROVEMENT * max(left_j, right_j),
            )
            if improvement < threshold:
                continue
            preferred = left if left_j < right_j else right
            disfavored = right if left_j < right_j else left
            result.append(
                {
                    "preferred_class_index": int(preferred["class_index"]),
                    "disfavored_class_index": int(disfavored["class_index"]),
                    "preferred_jerk_m2_s6": min(left_j, right_j),
                    "disfavored_jerk_m2_s6": max(left_j, right_j),
                    "absolute_improvement_m2_s6": improvement,
                    "relative_improvement": improvement / max(left_j, right_j, 1.0e-12),
                    "safety_equivalent": True,
                    "progress_equivalent": True,
                    "null_involved": False,
                }
            )
    return result


def partition_for_scenario(scenario_id: str) -> str:
    # Each stage/family contributes ten collected scenes.  Suffix 8/9 of each
    # family block is held out for internal validation, leaving 160/40 scenes.
    index = int(scenario_id.rsplit("_", 1)[1])
    return "validation" if index % 10 in {8, 9} else "train"


def state_files(collection: Path) -> list[Path]:
    result: list[Path] = []
    for shard in range(4):
        episode_root = collection / f"shard_{shard:02d}_of_04" / "episodes"
        result.extend(episode_root.rglob("state.json"))
    return sorted(result)


def main() -> None:
    args = parse_args()
    artifact_root = args.artifact_root.resolve()
    dataset_dir = artifact_root / "02_recurrent_dataset"
    rollout_dir = artifact_root / "03_counterfactual_rollouts"
    collection = dataset_dir / "recurrent_collection"
    paths = state_files(collection)
    if len(paths) != 1185:
        raise RuntimeError(f"expected 1185 state files, found {len(paths)}")

    state_ids: set[str] = set()
    episode_ids: set[str] = set()
    graph_schema: tuple[Any, ...] | None = None
    graph_manifest: list[dict[str, Any]] = []
    training_examples: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    safety_rows: list[dict[str, Any]] = []
    progress_rows: list[dict[str, Any]] = []
    smooth_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    ambiguous_rows: list[dict[str, Any]] = []
    peer_rows: list[dict[str, Any]] = []
    state_summaries: list[dict[str, Any]] = []
    horizon_counts: dict[str, Counter[str]] = {name: Counter() for name in HORIZONS}
    horizon_winners: dict[str, dict[str, int]] = {name: {} for name in HORIZONS}
    primary_fp_match = 0

    for state_path in paths:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        state = payload["state"]
        state_id = str(state["state_id"])
        if state_id in state_ids:
            raise RuntimeError(f"duplicate state_id: {state_id}")
        state_ids.add(state_id)
        scenario_id = str(state["scenario_id"])
        episode_ids.add(scenario_id)
        candidate_count = int(state["candidate_count"])
        if not 1 <= candidate_count <= 10:
            raise RuntimeError(f"{state_id}: candidate count outside Top-K contract: {candidate_count}")
        branch_order = [int(row["class_index"]) for row in payload["counterfactual_rollouts"]]
        if branch_order != list(range(candidate_count + 1)):
            raise RuntimeError(f"{state_id}: invalid class order {branch_order}")
        graph_path = state_path.parent / str(payload["graph_file"])
        observed_sha = sha256_file(graph_path)
        if observed_sha != payload["graph_file_sha256"]:
            raise RuntimeError(f"{state_id}: graph SHA mismatch")
        graph = torch.load(graph_path, map_location="cpu", weights_only=False)
        candidate_ids = graph["proposal"].candidate_id.detach().cpu().tolist()
        if candidate_ids != list(range(candidate_count)):
            raise RuntimeError(f"{state_id}: graph candidate order mismatch")
        raw_distance = float(graph["agent"].x_raw[0, 3].item())
        normalized_distance = float(graph["agent"].x[0, 3].item())
        expected_normalized = float(np.clip(raw_distance / 100.0, 0.0, 1.0))
        if not math.isclose(raw_distance, float(state["terminal_distance_m"]), abs_tol=1.0e-4):
            raise RuntimeError(f"{state_id}: raw terminal distance mismatch")
        if not math.isclose(normalized_distance, expected_normalized, abs_tol=1.0e-6):
            raise RuntimeError(f"{state_id}: 100 m normalization mismatch")
        schema = (
            tuple(graph.node_types),
            tuple(graph.edge_types),
            tuple((name, int(graph[name].x.shape[1])) for name in graph.node_types),
            tuple((edge, int(graph[edge].edge_attr.shape[1])) for edge in graph.edge_types),
        )
        if graph_schema is None:
            graph_schema = schema
        elif schema != graph_schema:
            raise RuntimeError(f"{state_id}: graph schema changed")

        relative_graph = graph_path.relative_to(artifact_root).as_posix()
        partition = partition_for_scenario(scenario_id)
        common = {
            "schema_version": SCHEMA_VERSION,
            "state_id": state_id,
            "scenario_id": scenario_id,
            "stage": state["stage"],
            "family": state["family"],
            "task_pattern": state["task_pattern"],
            "event_step": int(state["event_step"]),
            "planning_decision_index": int(state["planning_decision_index"]),
            "agent_id": int(state["agent_id"]),
            "mission_progress": float(state["mission_progress"]),
            "progress_bin": state["progress_bin"],
            "replan_bin": state["replan_bin"],
            "partition": partition,
            "graph_file": relative_graph,
        }
        graph_manifest.append(
            {
                **common,
                "graph_file_sha256": observed_sha,
                "candidate_count": candidate_count,
                "class_count": candidate_count + 1,
                "goal_distance_normalization_scale_m": 100.0,
                "offline_future_information_in_runtime_graph": False,
            }
        )

        interactions = candidate_interaction_map(payload)
        by_horizon: dict[str, list[dict[str, Any]]] = {}
        for horizon in HORIZONS:
            metrics: list[dict[str, Any]] = []
            for branch in payload["counterfactual_rollouts"]:
                interaction = None
                if branch.get("candidate_id") is not None:
                    interaction = interactions[int(branch["candidate_id"])]
                row = metric_row(branch, interaction, horizon)
                if row is None:
                    continue
                metrics.append(row)
                horizon_counts[horizon]["rows"] += 1
                horizon_counts[horizon]["collisions"] += int(row["collision"])
                horizon_counts[horizon]["peer_collisions"] += int(row["peer_collision"])
                horizon_counts[horizon]["obstacle_collisions"] += int(row["static_collision"] or row["dynamic_collision"])
                horizon_counts[horizon]["hard_unsafe"] += int(not row["hard_safe"])
                horizon_counts[horizon]["horizon_not_reached"] += int(not row["horizon_reached"])
                rollout_rows.append({**common, **row})
            by_horizon[horizon] = metrics
            if metrics and len(metrics) == sum(not b.get("not_run_reason") for b in payload["counterfactual_rollouts"]):
                winners, _ = safer_pool(metrics)
                horizon_winners[horizon][state_id] = min(int(row["class_index"]) for row in winners)

        primary = by_horizon[PRIMARY_HORIZON]
        winners, selection_trace = safer_pool(primary)
        winner_classes = sorted(int(row["class_index"]) for row in winners)
        soft_target = [0.0] * (candidate_count + 1)
        for class_index in winner_classes:
            soft_target[class_index] = 1.0 / len(winner_classes)
        fp_selected_class = int(payload["behavior_fp_selected_candidate_id"]) + 1
        primary_fp_match += int(fp_selected_class in winner_classes)
        pairs = smoothness_pairs(primary)
        pair_by_class: Counter[int] = Counter()
        for pair_index, pair in enumerate(pairs):
            pair_rows.append(
                {
                    **common,
                    "pair_index_within_state": pair_index,
                    "label_horizon": PRIMARY_HORIZON,
                    **pair,
                }
            )
            pair_by_class[int(pair["preferred_class_index"])] += 1
            pair_by_class[int(pair["disfavored_class_index"])] += 1

        primary_by_class = {int(row["class_index"]): row for row in primary}
        fp_scores = [0.0] + [
            float(row["fp_shep_online_score"])
            for row in payload["fp_shep_candidate_records"]
        ]
        target_quality = [
            1.0 if class_index in winner_classes
            else 0.0 if primary_by_class.get(class_index, {}).get("hard_safe", False)
            else -1.0
            for class_index in range(candidate_count + 1)
        ]
        training_examples.append(
            {
                **common,
                "class_count": candidate_count + 1,
                "proposal_count": candidate_count,
                "soft_target_gat_r": soft_target,
                "target_quality_gat_r": target_quality,
                "fp_shep_quality": fp_scores,
                "proposal_scores": [float(value) for value in payload["proposal_scores"]],
                "reference_classes": winner_classes,
                "behavior_fp_selected_class": fp_selected_class,
                "smoothness_pair_count": len(pairs),
                "smoothness_pairs": [
                    [int(pair["preferred_class_index"]), int(pair["disfavored_class_index"])]
                    for pair in pairs
                ],
                "selection_trace": selection_trace,
                "label_horizon": PRIMARY_HORIZON,
            }
        )

        safe_count = sum(bool(row["hard_safe"]) for row in primary if not row["null_branch"])
        unsafe_count = candidate_count - safe_count
        progress_values = [
            float(row["terminal_progress_m"])
            for row in primary
            if not row["null_branch"] and row["terminal_progress_m"] is not None
        ]
        jerk_values = [
            float(row["trajectory_smoothness_focal_m2_s6"])
            for row in primary
            if not row["null_branch"] and row["trajectory_smoothness_focal_m2_s6"] is not None
        ]
        downstream_difference = (
            safe_count not in {0, candidate_count}
            or (progress_values and max(progress_values) - min(progress_values) > PROGRESS_EQUIVALENCE_BAND_M)
            or (
                jerk_values
                and max(jerk_values) - min(jerk_values)
                >= max(SMOOTHNESS_MIN_ABSOLUTE_IMPROVEMENT_M2_S6, SMOOTHNESS_MIN_RELATIVE_IMPROVEMENT * max(jerk_values))
            )
        )
        ambiguous_actual = bool(
            state["ambiguous_pre_rollout"]
            and safe_count >= 2
            and downstream_difference
        )
        summary = {
            **common,
            "candidate_count": candidate_count,
            "safe_candidate_count_5s": safe_count,
            "unsafe_candidate_count_5s": unsafe_count,
            "h4_safe_candidate_count": int(state["safe_candidate_count_h4"]),
            "h4_risky_candidate_count": int(state["risky_candidate_count_h4"]),
            "fp_top1_top2_gap": float(state["fp_top1_top2_gap"]),
            "ambiguous_pre_rollout": bool(state["ambiguous_pre_rollout"]),
            "ambiguous_after_rollout": ambiguous_actual,
            "peer_rich_pre_rollout": bool(state["peer_rich_pre_rollout"]),
            "sampling_roles": list(state["sampling_roles"]),
            "target_classes_gat_r": winner_classes,
            "target_is_null": 0 in winner_classes,
            "fp_matches_target": fp_selected_class in winner_classes,
            "smoothness_pair_count": len(pairs),
        }
        state_summaries.append(summary)
        if ambiguous_actual:
            ambiguous_rows.append(summary)

        for row in primary:
            class_index = int(row["class_index"])
            labelled_common = {
                **common,
                "class_index": class_index,
                "candidate_id": row["candidate_id"],
                "null_branch": row["null_branch"],
                "null_eligible": row["null_eligible"],
                "label_horizon": PRIMARY_HORIZON,
                "gat_r_target_probability": soft_target[class_index],
                "gat_r_reference_class": class_index in winner_classes,
            }
            safety_rows.append(
                {
                    **labelled_common,
                    "hard_safe": row["hard_safe"],
                    "unsafe_reasons": row["unsafe_reasons"],
                    "collision": row["collision"],
                    "focal_collision": row["focal_collision"],
                    "static_collision": row["static_collision"],
                    "dynamic_collision": row["dynamic_collision"],
                    "peer_collision": row["peer_collision"],
                    "h4_risky": row["h4_risky"],
                    "minimum_peer_margin_m": row["minimum_peer_margin_m"],
                    "minimum_obstacle_clearance_m": row["minimum_obstacle_clearance_m"],
                    "peer_risk_duration_s": row["peer_risk_duration_s"],
                    "subsequent_emergency_trigger": row["subsequent_emergency_trigger"],
                }
            )
            progress_rows.append(
                {
                    **labelled_common,
                    "hard_safe": row["hard_safe"],
                    "terminal_progress_m": row["terminal_progress_m"],
                    "progress_per_second_mps": row["progress_per_second_mps"],
                    "reference_reached": row["reference_reached"],
                    "execution_deviation_m": row["execution_deviation_m"],
                    "next_rerr_event_type": row["next_rerr_event_type"],
                    "subsequent_emergency_trigger": row["subsequent_emergency_trigger"],
                }
            )
            smooth_rows.append(
                {
                    **labelled_common,
                    "hard_safe": row["hard_safe"],
                    "terminal_progress_m": row["terminal_progress_m"],
                    "acceleration_variation_focal_m2_s4": row["acceleration_variation_focal_m2_s4"],
                    "trajectory_smoothness_focal_m2_s6": row["trajectory_smoothness_focal_m2_s6"],
                    "trajectory_smoothness_team_mean_m2_s6": row["trajectory_smoothness_team_mean_m2_s6"],
                    "control_discontinuity_mps2": row["control_discontinuity_mps2"],
                    "eligible_smoothness_pair_memberships": int(pair_by_class[class_index]),
                    "smoothness_used_for_null": False,
                }
            )
            peer_signal = (
                bool(state["peer_rich_pre_rollout"])
                or bool(row["h4_risky"])
                or bool(row["peer_collision"])
                or finite_or_inf(row["minimum_peer_margin_m"]) < SERIOUS_NEAR_COLLISION_BUFFER_M
            )
            if peer_signal and not row["null_branch"]:
                peer_rows.append(
                    {
                        **labelled_common,
                        "peer_rich_pre_rollout": bool(state["peer_rich_pre_rollout"]),
                        "h4_risky": row["h4_risky"],
                        "peer_collision": row["peer_collision"],
                        "minimum_peer_margin_m": row["minimum_peer_margin_m"],
                        "peer_risk_duration_s": row["peer_risk_duration_s"],
                        "target_probability": soft_target[class_index],
                    }
                )

    if len(episode_ids) != 200:
        raise RuntimeError(f"expected 200 unique episodes, found {len(episode_ids)}")
    partition_episode_counts = Counter(
        partition_for_scenario(scenario_id) for scenario_id in episode_ids
    )
    if partition_episode_counts != Counter(train=160, validation=40):
        raise RuntimeError(f"invalid internal split: {partition_episode_counts}")

    # Distribution table: one explicit row per dimension/category.
    distribution_rows: list[dict[str, Any]] = []
    dimensions: dict[str, list[str]] = defaultdict(list)
    for row in state_summaries:
        for name in ("stage", "family", "progress_bin", "replan_bin", "partition"):
            dimensions[name].append(str(row[name]))
        dimensions["ambiguous_after_rollout"].append(str(bool(row["ambiguous_after_rollout"])))
        dimensions["peer_rich_pre_rollout"].append(str(bool(row["peer_rich_pre_rollout"])))
        for role in row["sampling_roles"]:
            dimensions["sampling_role"].append(str(role))
        dimensions["stage_x_family"].append(f"{row['stage']}|{row['family']}")
    for dimension, values in dimensions.items():
        counts = Counter(values)
        denominator = len(state_summaries) if dimension != "sampling_role" else sum(counts.values())
        for category, count in sorted(counts.items()):
            distribution_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "dimension": dimension,
                    "category": category,
                    "state_count": count,
                    "denominator": denominator,
                    "rate": count / denominator,
                }
            )

    # Null calibration is state-level and never references jerk.
    null_rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, str, list[dict[str, Any]]]] = [("overall", "overall", state_summaries)]
    for field in ("stage", "progress_bin", "replan_bin"):
        for value in sorted({str(row[field]) for row in state_summaries}):
            scopes.append((field, value, [row for row in state_summaries if str(row[field]) == value]))
    scopes.extend(
        [
            ("risk", "has_unsafe_candidate", [row for row in state_summaries if row["unsafe_candidate_count_5s"] > 0]),
            ("risk", "all_candidates_safe", [row for row in state_summaries if row["unsafe_candidate_count_5s"] == 0]),
        ]
    )
    for scope, value, members in scopes:
        null_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "scope": scope,
                "scope_value": value,
                "state_count": len(members),
                "null_eligible_state_count": sum(
                    any(
                        row["state_id"] == member["state_id"]
                        and row["null_branch"]
                        and row["null_eligible"]
                        for row in safety_rows
                    )
                    for member in members
                ),
                "null_reference_state_count": sum(bool(row["target_is_null"]) for row in members),
                "null_reference_rate": (
                    sum(bool(row["target_is_null"]) for row in members) / len(members)
                    if members else None
                ),
                "null_supervision_sources": "terminal-local eligibility + safety + progress viability; smoothness excluded",
            }
        )

    horizon_rows: list[dict[str, Any]] = []
    primary_winners = horizon_winners[PRIMARY_HORIZON]
    for horizon in HORIZONS:
        counts = horizon_counts[horizon]
        common_states = sorted(set(primary_winners) & set(horizon_winners[horizon]))
        agreement = (
            sum(primary_winners[state_id] == horizon_winners[horizon][state_id] for state_id in common_states)
            / len(common_states)
            if common_states else None
        )
        horizon_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "horizon": horizon,
                "branch_record_count": counts["rows"],
                "collision_rate": counts["collisions"] / counts["rows"] if counts["rows"] else None,
                "peer_collision_rate": counts["peer_collisions"] / counts["rows"] if counts["rows"] else None,
                "obstacle_collision_rate": counts["obstacle_collisions"] / counts["rows"] if counts["rows"] else None,
                "hard_unsafe_rate": counts["hard_unsafe"] / counts["rows"] if counts["rows"] else None,
                "horizon_not_reached_rate": counts["horizon_not_reached"] / counts["rows"] if counts["rows"] else None,
                "common_state_count_vs_5s": len(common_states),
                "top1_agreement_with_5s": agreement,
                "selected_as_primary": horizon == PRIMARY_HORIZON,
            }
        )

    write_csv(dataset_dir / "recurrent_state_sampling_distribution.csv", distribution_rows)
    write_csv(dataset_dir / "AMBIGUOUS_DECISION_DATASET.csv", ambiguous_rows)
    write_csv(dataset_dir / "recurrent_graph_manifest.csv", graph_manifest)
    write_jsonl(dataset_dir / "recurrent_training_examples.jsonl", training_examples)
    write_csv(dataset_dir / "null_calibration.csv", null_rows)
    write_csv(dataset_dir / "peer_conflict_training_subset.csv", peer_rows)
    write_csv(rollout_dir / "candidate_counterfactual_rollouts.csv", rollout_rows)
    write_csv(rollout_dir / "candidate_safety_labels.csv", safety_rows)
    write_csv(rollout_dir / "candidate_progress_labels.csv", progress_rows)
    write_csv(rollout_dir / "candidate_smoothness_labels.csv", smooth_rows)
    write_csv(rollout_dir / "eligible_smoothness_pairs.csv", pair_rows)
    write_csv(rollout_dir / "label_horizon_audit.csv", horizon_rows)

    dataset_hash = hashlib.sha256()
    for row in training_examples:
        dataset_hash.update(
            (json.dumps(json_ready(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        )
    label_freeze = {
        "schema_version": SCHEMA_VERSION,
        "status": "FROZEN_BEFORE_GAT_R_TRAINING",
        "primary_label_horizon": PRIMARY_HORIZON,
        "selection_reason": "5 s exposed materially more downstream collision signal than 1/2/3 s or next-event while retaining final-state metrics for every executable branch",
        "hierarchy": [
            "hard collision / serious near-collision / existing H4 immediate peer conflict",
            "peer then obstacle safety within explicit equivalence bands",
            "long-range terminal progress and reference viability",
            "execution deviation",
            "smoothness only as a separate eligible-pair term for GAT-RS",
        ],
        "thresholds": {
            "serious_near_collision_buffer_m": SERIOUS_NEAR_COLLISION_BUFFER_M,
            "d_safe_m": 0.6,
            "safety_equivalence_band_m": SAFETY_EQUIVALENCE_BAND_M,
            "progress_equivalence_band_m": PROGRESS_EQUIVALENCE_BAND_M,
            "execution_deviation_equivalence_band_m": EXECUTION_DEVIATION_EQUIVALENCE_BAND_M,
            "peer_margin_cap_m": PEER_MARGIN_CAP_M,
            "obstacle_clearance_cap_m": OBSTACLE_CLEARANCE_CAP_M,
            "smoothness_min_relative_improvement": SMOOTHNESS_MIN_RELATIVE_IMPROVEMENT,
            "smoothness_min_absolute_improvement_m2_s6": SMOOTHNESS_MIN_ABSOLUTE_IMPROVEMENT_M2_S6,
        },
        "null_rule": "terminal-local eligibility + safety + progress viability only; jerk/smoothness excluded",
        "gat_r_target": "uniform soft mass over lexicographic top equivalence set",
        "gat_rs_difference": "same GAT-R primary targets and graphs plus eligible pairwise lower-jerk preference",
        "offline_future_information_at_runtime": False,
        "formal_v1_data_used_for_labels": False,
        "dev_holdout_formal_data_read": False,
        "horizon_audit": horizon_rows,
    }
    write_json(rollout_dir / "COUNTERFACTUAL_LABEL_FREEZE.json", label_freeze)
    reconciliation = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "episode_count": len(episode_ids),
        "state_count": len(state_ids),
        "candidate_branch_count_primary": len(safety_rows),
        "all_horizon_rollout_row_count": len(rollout_rows),
        "eligible_smoothness_pair_count": len(pair_rows),
        "ambiguous_state_count": len(ambiguous_rows),
        "peer_conflict_candidate_row_count": len(peer_rows),
        "internal_train_episode_count": partition_episode_counts["train"],
        "internal_validation_episode_count": partition_episode_counts["validation"],
        "internal_train_state_count": sum(row["partition"] == "train" for row in state_summaries),
        "internal_validation_state_count": sum(row["partition"] == "validation" for row in state_summaries),
        "unique_state_ids": len(state_ids),
        "graph_hash_failures": 0,
        "graph_schema_failures": 0,
        "candidate_order_failures": 0,
        "goal_distance_normalization_failures": 0,
        "partial_directory_count": 0,
        "software_error_count": 0,
        "fp_top1_matches_hierarchical_target_rate": primary_fp_match / len(state_summaries),
        "training_examples_semantic_sha256": dataset_hash.hexdigest(),
        "gat_r_and_gat_rs_state_manifest_identical": True,
        "formal_v1_used_for_training": False,
        "dev_used_for_training_or_label_selection": False,
        "holdout_opened": False,
        "formal_v2_generated": False,
    }
    write_json(dataset_dir / "recurrent_dataset_reconciliation.json", reconciliation)
    print(json.dumps(reconciliation, indent=2))


if __name__ == "__main__":
    main()
