"""Read-only local-MDP and reference-lifecycle audit for long-range setup."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment_config import SACExperimentConfig  # noqa: E402
from Guidance.reference_point_proposal_demo import ProposalConfig  # noqa: E402
from planning.semi_structured_long_range_benchmark import (  # noqa: E402
    MISSION_DISTANCE_RANGE_M,
    SENSOR_RANGE_M,
)


SCHEMA = "long_range_local_mdp_distribution_audit_v1"
DEFAULT_ARTIFACT = REPO_ROOT / "artifacts/semi_structured_long_range_main_benchmark/20260820_193228"
HISTORICAL_CLASSES = REPO_ROOT / "artifacts/candidate_supervision/20260814_121319/candidate_class_records.csv"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, fieldnames: Iterable[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def quantiles(name: str, values: np.ndarray, *, source: str, status: str) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    q = np.quantile(values, [0.05, 0.50, 0.95, 0.99]) if values.size else [math.nan] * 4
    return {
        "distribution": name,
        "source": source,
        "status": status,
        "n": int(values.size),
        "p05_m": float(q[0]),
        "p50_m": float(q[1]),
        "p95_m": float(q[2]),
        "p99_m": float(q[3]),
        "max_m": float(np.max(values)) if values.size else math.nan,
        "goal_distance_clip_m": 2.0 * 4.5,
        "normalized_p50": float(np.clip(q[1] / 9.0, 0.0, 1.0)),
        "normalized_p95": float(np.clip(q[2] / 9.0, 0.0, 1.0)),
    }


def sample_sac_training_goal_distances(count: int = 200_000) -> np.ndarray:
    config = SACExperimentConfig()
    rng = np.random.default_rng(20260820)
    start_lo, start_hi = map(np.asarray, config.start_position_bounds)
    goal_lo, goal_hi = map(np.asarray, config.goal_position_bounds)
    accepted: list[np.ndarray] = []
    remaining = count
    while remaining > 0:
        batch = max(remaining * 2, 4096)
        starts = rng.uniform(start_lo, start_hi, size=(batch, 3))
        goals = rng.uniform(goal_lo, goal_hi, size=(batch, 3))
        distances = np.linalg.norm(goals - starts, axis=1)
        distances = distances[distances >= config.min_start_goal_distance]
        take = distances[:remaining]
        accepted.append(take)
        remaining -= len(take)
    return np.concatenate(accepted)


def historical_candidate_distances() -> tuple[np.ndarray, np.ndarray]:
    all_proposals: list[float] = []
    rank_zero: list[float] = []
    with HISTORICAL_CLASSES.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["class_kind"] != "proposal":
                continue
            distance = float(row["proposal_distance"])
            all_proposals.append(distance)
            if int(row["proposal_rank_zero_based"]) == 0:
                rank_zero.append(distance)
    return np.asarray(all_proposals), np.asarray(rank_zero)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT)
    args = parser.parse_args()
    root = args.artifact_root.resolve()

    training = sample_sac_training_goal_distances()
    candidates, rank_zero = historical_candidate_distances()
    rows = [
        quantiles(
            "historical_sac_training_terminal_goal",
            training,
            source="SACExperimentConfig start/goal distribution; deterministic 200k reconstruction",
            status="HISTORICAL_TRAINING_SUPPORT",
        ),
        quantiles(
            "historical_gat_dataset_all_candidate_references",
            candidates,
            source=str(HISTORICAL_CLASSES.relative_to(REPO_ROOT)).replace("\\", "/"),
            status="HISTORICAL_EXECUTED_GRAPH_SUPPORT",
        ),
        quantiles(
            "historical_gat_dataset_proposal_rank0_references",
            rank_zero,
            source=str(HISTORICAL_CLASSES.relative_to(REPO_ROOT)).replace("\\", "/"),
            status="HISTORICAL_COARSE_SELECTION_SUPPORT",
        ),
    ]
    # These rows are contracts, not empirical distributions. The empirical
    # pilot/development rows are appended by their respective frozen runners.
    proposal = ProposalConfig()
    rows.extend(
        [
            {
                "distribution": "long_range_initial_local_reference_contract",
                "source": "ProposalConfig",
                "status": "CONTRACT_RANGE_NOT_EMPIRICAL",
                "n": 0,
                "p05_m": math.nan,
                "p50_m": math.nan,
                "p95_m": math.nan,
                "p99_m": math.nan,
                "max_m": proposal.s_max,
                "goal_distance_clip_m": 9.0,
                "normalized_p50": math.nan,
                "normalized_p95": math.nan,
            },
            {
                "distribution": "legacy_far_terminal_after_local_handoff",
                "source": "65-85 m mission minus <=1.05 m first local segment",
                "status": "COUNTERFACTUAL_CONTRACT_BOUND",
                "n": 0,
                "p05_m": MISSION_DISTANCE_RANGE_M[0] - proposal.s_max,
                "p50_m": math.nan,
                "p95_m": math.nan,
                "p99_m": math.nan,
                "max_m": MISSION_DISTANCE_RANGE_M[1],
                "goal_distance_clip_m": 9.0,
                "normalized_p50": math.nan,
                "normalized_p95": 1.0,
            },
            {
                "distribution": "adapted_direct_terminal_handoff_scope",
                "source": "reference lifecycle contract",
                "status": "CONTRACT_RANGE_NOT_EMPIRICAL",
                "n": 0,
                "p05_m": 0.0,
                "p50_m": math.nan,
                "p95_m": math.nan,
                "p99_m": math.nan,
                "max_m": SENSOR_RANGE_M,
                "goal_distance_clip_m": 9.0,
                "normalized_p50": math.nan,
                "normalized_p95": 0.5,
            },
        ]
    )
    distribution_fields = (
        "distribution",
        "source",
        "status",
        "n",
        "p05_m",
        "p50_m",
        "p95_m",
        "p99_m",
        "max_m",
        "goal_distance_clip_m",
        "normalized_p50",
        "normalized_p95",
    )
    write_csv(root / "07_training/sac_active_reference_distribution.csv", distribution_fields, rows)

    graph_rows = (
        {
            "feature_block": "proposal_local_position_and_distance",
            "historical_support": "0.05-1.05 m; scale 1.05 m",
            "long_range_initial_support": "same ProposalConfig range",
            "shift": "LOW",
            "initial_action": "preserve",
        },
        {
            "feature_block": "terminal_task_goal_distance",
            "historical_support": "approximately 6-9 m; scale 9 m",
            "long_range_initial_support": "65-85 m; normalized value saturates at 1",
            "shift": "STRONG_RAW_PARTIAL_NORMALIZED",
            "initial_action": "pilot then adapt normalization/retrain GAT on independent training data",
        },
        {
            "feature_block": "ego_velocity",
            "historical_support": "component cap 4 m/s",
            "long_range_initial_support": "common norm cap 3.2 m/s within historical component support",
            "shift": "LOW_TO_MODERATE",
            "initial_action": "preserve actor/GAT semantics",
        },
        {
            "feature_block": "sector_safety_56_direction",
            "historical_support": "56 directions, 4.5 m local range",
            "long_range_initial_support": "same",
            "shift": "LOW",
            "initial_action": "preserve",
        },
        {
            "feature_block": "align_peer_relative_position",
            "historical_support": "exact all-peer event state; graph scale 1.2 m",
            "long_range_initial_support": "anonymous local peers <=4.5 m decoded from native block",
            "shift": "STRONG_INFORMATION_CONTRACT",
            "initial_action": "pilot only; GAT retraining/re-normalization likely",
        },
        {
            "feature_block": "align_peer_relative_velocity",
            "historical_support": "exact relative velocity; graph scale 8 m/s",
            "long_range_initial_support": "native ally block clips relative components at 4 m/s before graph",
            "shift": "MODERATE",
            "initial_action": "respect legal clipping; never bypass",
        },
        {
            "feature_block": "spatiotemporal_peer_candidate_edges",
            "historical_support": "H4, dt 0.1, d_safe 0.6, d_align 1.2/1.0 provenance",
            "long_range_initial_support": "same H4 definition and d_safe; parameters development-tunable",
            "shift": "LOW_DEFINITION_MODERATE_CONTEXT",
            "initial_action": "preserve edge semantics",
        },
        {
            "feature_block": "episode_composition",
            "historical_support": "<=220 steps and few upper events",
            "long_range_initial_support": "<=1500 steps and roughly 62-243 local references by geometry bounds",
            "shift": "STRONG_SEQUENCE_LENGTH",
            "initial_action": "use R-ERR lifecycle; do not enlarge local MDP uniformly",
        },
    )
    write_csv(
        root / "07_training/gat_feature_distribution_audit.csv",
        ("feature_block", "historical_support", "long_range_initial_support", "shift", "initial_action"),
        graph_rows,
    )

    minimum_references = math.ceil(MISSION_DISTANCE_RANGE_M[0] / proposal.s_max)
    maximum_nominal_references = math.ceil(MISSION_DISTANCE_RANGE_M[1] / proposal.s_min)
    lifecycle = {
        "schema_version": SCHEMA,
        "status": "PASS_FOR_CONTRACT_PILOT",
        "historical_behavior": "completed local reference -> direct terminal handoff",
        "historical_behavior_long_range_consequence": {
            "minimum_far_terminal_distance_after_first_max_reference_m": MISSION_DISTANCE_RANGE_M[0] - proposal.s_max,
            "actor_goal_distance_scalar": "saturates at 1.0 because historical clip is 9 m",
            "local_reference_semantics_preserved": False,
        },
        "adapted_behavior": "completed nonterminal reference -> existing R-ERR upper reconstruction unless terminal is within 4.5 m",
        "when_to_reconstruct_owner": "R-ERR",
        "new_planner_or_trigger_module": False,
        "same_upper_chain": "Proposal -> coarse Top-K -> FP-SHEP -> GAT",
        "phase_position_velocity_continuity_required": True,
        "geometry_reference_count_bounds": {
            "minimum_at_65m_using_1.05m_steps": minimum_references,
            "maximum_nominal_at_85m_using_0.35m_steps": maximum_nominal_references,
            "interpretation": "composition-pressure bounds, not predicted event counts",
        },
        "pilot_must_measure": [
            "empirical active-reference distance quantiles",
            "null/far-terminal fallback duration",
            "reference completion reconstruction count",
            "same-goal rate",
            "per-event and per-episode compute",
            "phase/position/velocity continuity",
        ],
        "CORE_THEORY_CHANGED": "NO",
    }
    write_json(root / "07_training/reference_lifecycle_audit.json", lifecycle)

    decision = {
        "schema_version": SCHEMA,
        "SAC_LOCAL_MDP_DISTRIBUTION_MATCH": "PARTIAL_STRONG_LOCAL_GEOMETRY",
        "GAT_FEATURE_DISTRIBUTION_MATCH": "PARTIAL_WITH_STRONG_PEER_AND_TASK_DISTANCE_SHIFT",
        "LONG_RANGE_REFERENCE_LIFECYCLE_AUDIT": "PASS",
        "HISTORICAL_SAC_CHECKPOINT_ALLOWED_FOR_CONTRACT_PILOT": "YES",
        "HISTORICAL_GAT_CHECKPOINT_ALLOWED_FOR_CONTRACT_PILOT": "YES_DIAGNOSTIC_ONLY",
        "HISTORICAL_CHECKPOINT_SELECTED_AS_FINAL": "NO",
        "RETRAIN_BEFORE_PILOT": "NO",
        "RETRAINING_DECISION_AFTER_PILOT": "DEVELOPMENT_ONLY",
        "reason": "Pilot checks compatibility, not success; local sensing/candidate/action semantics remain valid while graph/task shifts must be quantified before adaptation.",
        "new_performance_episode_count": 0,
    }
    write_json(root / "07_training/local_mdp_distribution_decision.json", decision)
    print(json.dumps(decision, ensure_ascii=False))


if __name__ == "__main__":
    main()
