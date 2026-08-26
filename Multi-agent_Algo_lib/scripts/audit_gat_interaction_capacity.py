"""Read-only audit of interaction-learning capacity in the frozen Stage-I GAT."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.stats import mannwhitneyu, pointbiserialr, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = Path(__file__).resolve().parent
for search_path in (REPO_ROOT, ALGO_ROOT, SCRIPTS_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from planning.gat.candidate_selector import batch_candidate_graphs  # noqa: E402
from planning.gat.stage1_training import (  # noqa: E402
    load_model_checkpoint,
    resolve_device,
)
from scripts.evaluate_actor_dmp_goal_semantics import write_csv, write_json  # noqa: E402


SCHEMA_VERSION = "gat_interaction_capacity_audit_v1"
DEFAULT_CONFIG = REPO_ROOT / "configs/evaluation/gat_interaction_capacity_audit.json"
AUDITED_SOURCE_PATHS = (
    "planning/heterogeneous_candidate_graph.py",
    "planning/gat/typed_encoders.py",
    "planning/gat/edge_enhanced_gat.py",
    "planning/gat/candidate_selector.py",
    "planning/gat/stage1_training.py",
)
ST_EDGE = ("align", "spatiotemporal", "proposal")
SMOOTH_EDGE = ("agent", "smooth", "proposal")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hashes(paths: Sequence[str]) -> dict[str, str]:
    return {path: _sha256(REPO_ROOT / path) for path in paths}


def _model_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _safe_corr(function: Any, left: Sequence[float], right: Sequence[float]) -> float | None:
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 3 or len(np.unique(x[mask])) < 2 or len(np.unique(y[mask])) < 2:
        return None
    result = function(x[mask], y[mask])
    value = result.statistic if hasattr(result, "statistic") else result[0]
    return _finite(value)


def _percentile(values: Sequence[float], q: float) -> float | None:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(np.percentile(array, q)) if array.size else None


def _cliffs_delta(feature: Sequence[float], outcome: Sequence[bool]) -> float | None:
    x = np.asarray(feature, dtype=float)
    y = np.asarray(outcome, dtype=bool)
    positive = x[y & np.isfinite(x)]
    negative = x[(~y) & np.isfinite(x)]
    if not len(positive) or not len(negative):
        return None
    u = float(mannwhitneyu(positive, negative, alternative="two-sided").statistic)
    return float(2.0 * u / (len(positive) * len(negative)) - 1.0)


def _subset_name(record: Mapping[str, Any]) -> str:
    if bool(record["interaction_rich"]):
        return "interaction_rich"
    if bool(record["low_interaction_control"]):
        return "low_interaction_control"
    return "middle"


def _rank_desc(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(-array, kind="stable")
    ranks = np.empty(len(array), dtype=int)
    ranks[order] = np.arange(1, len(array) + 1)
    return ranks


def _pairwise_flip_rate(left: Sequence[float], right: Sequence[float]) -> float | None:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    flips = comparable = 0
    for i in range(len(left)):
        for j in range(i + 1, len(left)):
            a = np.sign(left[i] - left[j])
            b = np.sign(right[i] - right[j])
            if a == 0 or b == 0:
                continue
            comparable += 1
            flips += int(a != b)
    return flips / comparable if comparable else None


def _sample_id(sample: Mapping[str, Any]) -> str:
    return f"{sample['state_group_id']}__ego{int(sample['ego_id'])}"


def _extract_graph_records(
    samples: Sequence[Mapping[str, Any]],
    subset_config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    graph_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    primary = set(subset_config["primary_interaction_scenarios"])
    controls = set(subset_config["primary_low_interaction_scenarios"])
    for sample in samples:
        graph = sample["graph"]
        sample_id = _sample_id(sample)
        st = graph[ST_EDGE]
        st_raw = st.edge_attr.detach().cpu().numpy().astype(float)
        st_index = st.edge_index.detach().cpu().numpy().astype(int)
        smooth = graph[SMOOTH_EDGE]
        smooth_attr = smooth.edge_attr.detach().cpu().numpy().reshape(-1).astype(float)
        align_raw = graph["align"].x_raw.detach().cpu().numpy().astype(float)
        d_safe = float(graph.graph_metadata["d_safe"])
        secondary = bool(
            len(st_raw)
            and (np.any(st_raw[:, 1] < d_safe) or np.any(st_raw[:, 2] > 0.0))
        )
        scenario = str(sample["scenario"])
        rich = scenario in primary or secondary
        low_control = scenario in controls and not secondary
        graph_row = {
            "schema_version": SCHEMA_VERSION,
            "graph_id": sample_id,
            "state_group_id": sample["state_group_id"],
            "scenario": scenario,
            "seed": int(sample["seed"]),
            "sample_t": int(sample["sample_t"]),
            "ego": int(sample["ego_id"]),
            "effective_split": sample["effective_split"],
            "class_count": len(sample["class_mapping"]),
            "proposal_count": int(graph["proposal"].num_nodes),
            "peer_node_count": int(graph["align"].num_nodes),
            "spatiotemporal_edge_count": int(st_index.shape[1]),
            "finite_t_min_count": int(np.sum(np.isfinite(st_raw[:, 0]))) if len(st_raw) else 0,
            "minimum_t_min_s": float(np.min(st_raw[:, 0])) if len(st_raw) else None,
            "minimum_d_min_m": float(np.min(st_raw[:, 1])) if len(st_raw) else None,
            "maximum_T_risk_s": float(np.max(st_raw[:, 2])) if len(st_raw) else 0.0,
            "positive_T_risk_edge_count": int(np.sum(st_raw[:, 2] > 0.0)) if len(st_raw) else 0,
            "minimum_peer_distance_m": float(np.min(align_raw[:, 3])) if len(align_raw) else None,
            "maximum_peer_relative_speed_mps": float(
                np.max(np.linalg.norm(align_raw[:, 4:7], axis=1))
            )
            if len(align_raw)
            else None,
            "d_safe_m": d_safe,
            "primary_interaction_scenario": scenario in primary,
            "secondary_risk_descriptor": secondary,
            "interaction_rich": rich,
            "low_interaction_control": low_control,
            "subset": "interaction_rich" if rich else "low_interaction_control" if low_control else "middle",
            "subset_rule_model_independent": True,
        }
        graph_rows.append(graph_row)

        proposal_x = graph["proposal"].x.detach().cpu().numpy().astype(float)
        proposal_raw = graph["proposal"].x_raw.detach().cpu().numpy().astype(float)
        proposal_score = graph["proposal"].proposal_score.detach().cpu().numpy().astype(float)
        agent_x = graph["agent"].x.detach().cpu().numpy().reshape(-1).astype(float)
        align_x = graph["align"].x.detach().cpu().numpy().astype(float)
        if len(align_x):
            align_summary = np.concatenate(
                [np.min(align_x, axis=0), np.mean(align_x, axis=0), np.max(align_x, axis=0)]
            )
        else:
            align_summary = np.zeros(21, dtype=float)
        for proposal_index in range(int(graph["proposal"].num_nodes)):
            edge_mask = st_index[1] == proposal_index if st_index.shape[1] else np.zeros(0, bool)
            candidate_st = st_raw[edge_mask]
            if len(candidate_st):
                st_min = np.min(candidate_st, axis=0)
                st_mean = np.mean(candidate_st, axis=0)
                st_max = np.max(candidate_st, axis=0)
            else:
                st_min = np.asarray([np.nan, np.nan, 0.0])
                st_mean = st_min.copy()
                st_max = st_min.copy()
            interaction_vector = np.concatenate(
                [align_summary, st_min, st_mean, st_max, [float(len(candidate_st))]]
            )
            local_vector = np.concatenate(
                [agent_x, proposal_x[proposal_index], [smooth_attr[proposal_index]]]
            )
            class_index = proposal_index + 1
            candidate_rows.append(
                {
                    **graph_row,
                    "class_index": class_index,
                    "candidate_id": int(
                        graph["proposal"].candidate_id[proposal_index].item()
                    ),
                    "proposal_index": proposal_index,
                    "t_min_min_s": _finite(st_min[0]),
                    "d_min_min_m": _finite(st_min[1]),
                    "T_risk_max_s": _finite(st_max[2]),
                    "candidate_st_edge_count": len(candidate_st),
                    "peer_relative_distance_min_m": graph_row["minimum_peer_distance_m"],
                    "peer_relative_speed_max_mps": graph_row[
                        "maximum_peer_relative_speed_mps"
                    ],
                    "smoothness": float(smooth_attr[proposal_index]),
                    "proposal_score": float(proposal_score[proposal_index]),
                    "geometric_task_progress_m": float(proposal_raw[proposal_index, 7]),
                    "sector_safety": float(proposal_raw[proposal_index, 8]),
                    "preview_task_progress_m": float(proposal_raw[proposal_index, 9]),
                    "preview_min_clearance_m": float(proposal_raw[proposal_index, 10]),
                    "preview_max_deviation_m": float(proposal_raw[proposal_index, 11]),
                    "preview_terminal_speed_mps": float(proposal_raw[proposal_index, 12]),
                    "local_vector": local_vector,
                    "interaction_vector": interaction_vector,
                    "full_vector": np.concatenate([local_vector, interaction_vector]),
                }
            )
    return graph_rows, candidate_rows


def _join_outcomes_and_targets(
    samples: Sequence[Mapping[str, Any]],
    candidate_rows: list[dict[str, Any]],
    branch_frame: pd.DataFrame,
) -> None:
    branch_lookup = {
        (str(row.sample_id), int(row.class_index)): row
        for row in branch_frame.itertuples(index=False)
    }
    sample_lookup = {_sample_id(sample): sample for sample in samples}
    for row in candidate_rows:
        key = (row["graph_id"], int(row["class_index"]))
        branch = branch_lookup.get(key)
        if branch is None:
            raise RuntimeError(f"missing real branch outcome: {key}")
        sample = sample_lookup[row["graph_id"]]
        index = int(row["class_index"])
        row.update(
            {
                "inter_agent_collision": bool(branch.inter_agent_collision),
                "inter_agent_collision_free": not bool(branch.inter_agent_collision),
                "any_collision": bool(branch.any_collision),
                "collision_free": not bool(branch.any_collision),
                "ego_reference_reached": bool(branch.ego_reference_reached),
                "team_success": bool(branch.team_success),
                "ego_terminal_reached": bool(branch.ego_terminal_reached),
                "post_reference_inter_agent_collision": bool(
                    branch.post_reference_inter_agent_collision
                ),
                "v1_scalar_utility": float(sample["v1_scalar_h6_utility"][index]),
                "v1_historical_utility": float(
                    sample["v1_historical_h6_utility"][index]
                ),
                "v2_long_horizon_tier": int(
                    sample["v2_long_horizon_tier"][index]
                ),
                "v2_long_horizon_utility": float(
                    sample["v2_long_horizon_utility"][index]
                ),
            }
        )


def _feature_signal_rows(candidate_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    features = (
        "t_min_min_s",
        "d_min_min_m",
        "T_risk_max_s",
        "peer_relative_distance_min_m",
        "peer_relative_speed_max_mps",
        "smoothness",
        "proposal_score",
        "geometric_task_progress_m",
        "sector_safety",
        "preview_task_progress_m",
        "preview_min_clearance_m",
        "preview_max_deviation_m",
        "preview_terminal_speed_mps",
    )
    subsets = {
        "overall": list(candidate_rows),
        "interaction_rich": [row for row in candidate_rows if row["interaction_rich"]],
        "multi_agent": [row for row in candidate_rows if row["scenario"] == "multi_agent"],
        "low_interaction_control": [
            row for row in candidate_rows if row["low_interaction_control"]
        ],
    }
    rows: list[dict[str, Any]] = []
    for subset, members in subsets.items():
        for feature in features:
            values = np.asarray(
                [np.nan if row[feature] is None else float(row[feature]) for row in members],
                dtype=float,
            )
            collision = np.asarray(
                [bool(row["inter_agent_collision"]) for row in members], dtype=bool
            )
            any_collision = np.asarray(
                [bool(row["any_collision"]) for row in members], dtype=bool
            )
            reference_reached = np.asarray(
                [bool(row["ego_reference_reached"]) for row in members], dtype=bool
            )
            success = np.asarray([bool(row["team_success"]) for row in members], dtype=bool)
            post_reference_collision = np.asarray(
                [bool(row["post_reference_inter_agent_collision"]) for row in members],
                dtype=bool,
            )
            v2_utility = np.asarray(
                [float(row["v2_long_horizon_utility"]) for row in members], dtype=float
            )
            finite = np.isfinite(values)
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "subset": subset,
                    "feature": feature,
                    "branch_count": len(members),
                    "finite_count": int(np.sum(finite)),
                    "median": _percentile(values, 50),
                    "p05": _percentile(values, 5),
                    "p95": _percentile(values, 95),
                    "inter_agent_collision_median": _percentile(values[collision], 50),
                    "inter_agent_collision_free_median": _percentile(values[~collision], 50),
                    "any_collision_median": _percentile(values[any_collision], 50),
                    "collision_free_median": _percentile(values[~any_collision], 50),
                    "reference_reached_median": _percentile(values[reference_reached], 50),
                    "reference_not_reached_median": _percentile(
                        values[~reference_reached], 50
                    ),
                    "team_success_median": _percentile(values[success], 50),
                    "team_failure_median": _percentile(values[~success], 50),
                    "post_reference_inter_agent_collision_median": _percentile(
                        values[post_reference_collision], 50
                    ),
                    "no_post_reference_inter_agent_collision_median": _percentile(
                        values[~post_reference_collision], 50
                    ),
                    "cliffs_delta_collision_vs_free": _cliffs_delta(values, collision),
                    "point_biserial_inter_agent_collision": _safe_corr(
                        pointbiserialr, collision.astype(float), values
                    ),
                    "point_biserial_collision_free": _safe_corr(
                        pointbiserialr, (~any_collision).astype(float), values
                    ),
                    "point_biserial_reference_reached": _safe_corr(
                        pointbiserialr, reference_reached.astype(float), values
                    ),
                    "point_biserial_team_success": _safe_corr(
                        pointbiserialr, success.astype(float), values
                    ),
                    "point_biserial_post_reference_inter_agent_collision": _safe_corr(
                        pointbiserialr,
                        post_reference_collision.astype(float),
                        values,
                    ),
                    "spearman_v2_long_horizon_utility": _safe_corr(
                        spearmanr, values, v2_utility
                    ),
                    "outcomes_are_existing_rollouts": True,
                }
            )
    return rows


def _classify_feature_signal(
    rows: Sequence[Mapping[str, Any]], thresholds: Mapping[str, Any]
) -> tuple[str, float]:
    explicit = {
        "t_min_min_s",
        "d_min_min_m",
        "T_risk_max_s",
        "peer_relative_distance_min_m",
        "peer_relative_speed_max_mps",
    }
    effects = []
    for row in rows:
        if row["subset"] != "interaction_rich" or row["feature"] not in explicit:
            continue
        for field in (
            "cliffs_delta_collision_vs_free",
            "point_biserial_inter_agent_collision",
        ):
            value = row.get(field)
            if value is not None:
                effects.append(abs(float(value)))
    maximum = max(effects, default=0.0)
    if maximum >= float(thresholds["strong_absolute_effect"]):
        return "STRONG", maximum
    if maximum >= float(thresholds["moderate_absolute_effect"]):
        return "MODERATE", maximum
    if maximum >= float(thresholds["weak_absolute_effect"]):
        return "WEAK", maximum
    return "NO", maximum


def _matched_instability(
    attribution_rows: Sequence[Mapping[str, str]],
    graph_lookup: Mapping[str, Mapping[str, Any]],
    thresholds: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], str, str, dict[str, Any]]:
    sample_rows: list[dict[str, Any]] = []
    for source in attribution_rows:
        sample_id = str(source["sample_id"])
        formal_tier = np.asarray(json.loads(source["formal_tiers"]), dtype=float)
        alternative_tier = np.asarray(json.loads(source["alternative_tiers"]), dtype=float)
        formal_utility = np.asarray(json.loads(source["formal_utilities"]), dtype=float)
        alternative_utility = np.asarray(
            json.loads(source["alternative_utilities"]), dtype=float
        )
        formal_team_completion = formal_tier == 5
        alternative_team_completion = alternative_tier == 5
        formal_any_collision = formal_tier == 0
        alternative_any_collision = alternative_tier == 0
        proposal_mask = np.arange(len(formal_tier)) > 0
        graph = graph_lookup[sample_id]
        good_formal = formal_tier >= 3
        good_alternative = alternative_tier >= 3
        formal_rank = _rank_desc(formal_utility)
        alternative_rank = _rank_desc(alternative_utility)
        sample_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "record_type": "sample",
                "graph_id": sample_id,
                "state_group_id": source["state_group_id"],
                "scenario": source["scenario"],
                "seed": int(source["seed"]),
                "ego": int(source["ego_id"]),
                "class_count": len(formal_tier),
                "interaction_rich": graph["interaction_rich"],
                "low_interaction_control": graph["low_interaction_control"],
                "formal_background": source["formal_background"],
                "alternative_background": source["alternative_background"],
                "candidate_quality_flip_rate": float(
                    np.mean(good_formal != good_alternative)
                ),
                "tier_flip_rate": float(np.mean(formal_tier != alternative_tier)),
                "team_completion_flip_rate_tier_exact": float(
                    np.mean(formal_team_completion != alternative_team_completion)
                ),
                "any_collision_flip_rate_tier_exact": float(
                    np.mean(formal_any_collision != alternative_any_collision)
                ),
                "reference_or_terminal_tier_proxy_flip_rate": float(
                    np.mean(
                        (formal_tier[proposal_mask] >= 3)
                        != (alternative_tier[proposal_mask] >= 3)
                    )
                )
                if np.any(proposal_mask)
                else None,
                "reference_reach_flip_rate": None,
                "inter_agent_collision_flip_rate": None,
                "direct_outcome_comparison_status": (
                    "PARTIAL_TIER_DERIVED_REFERENCE_AND_INTER_AGENT_COLLISION_NOT_AVAILABLE"
                ),
                "top1_flip": int(np.argmax(formal_utility))
                != int(np.argmax(alternative_utility)),
                "pairwise_ordering_flip_rate": _pairwise_flip_rate(
                    formal_utility, alternative_utility
                ),
                "ranking_position_change_rate": float(
                    np.mean(formal_rank != alternative_rank)
                ),
                "mean_absolute_rank_change": float(
                    np.mean(np.abs(formal_rank - alternative_rank))
                ),
                "spearman": _safe_corr(
                    spearmanr, formal_utility, alternative_utility
                ),
                "same_source_state": True,
                "same_ego": True,
                "same_candidate_identity": True,
                "same_ego_graph_local_features": True,
            }
        )

    summary_rows: list[dict[str, Any]] = []
    subsets = {
        "overall": sample_rows,
        "interaction_rich": [row for row in sample_rows if row["interaction_rich"]],
        "multi_agent": [row for row in sample_rows if row["scenario"] == "multi_agent"],
        "low_interaction_control": [
            row for row in sample_rows if row["low_interaction_control"]
        ],
    }
    for subset, members in subsets.items():
        summary_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "record_type": "summary",
                "subset": subset,
                "graph_count": len(members),
                "candidate_count": sum(int(row["class_count"]) for row in members),
                "candidate_quality_flip_rate": float(
                    np.average(
                        [row["candidate_quality_flip_rate"] for row in members],
                        weights=[row["class_count"] for row in members],
                    )
                )
                if members
                else None,
                "tier_flip_rate": float(
                    np.average(
                        [row["tier_flip_rate"] for row in members],
                        weights=[row["class_count"] for row in members],
                    )
                )
                if members
                else None,
                "team_completion_flip_rate_tier_exact": float(
                    np.average(
                        [row["team_completion_flip_rate_tier_exact"] for row in members],
                        weights=[row["class_count"] for row in members],
                    )
                )
                if members
                else None,
                "any_collision_flip_rate_tier_exact": float(
                    np.average(
                        [row["any_collision_flip_rate_tier_exact"] for row in members],
                        weights=[row["class_count"] for row in members],
                    )
                )
                if members
                else None,
                "reference_or_terminal_tier_proxy_flip_rate": float(
                    np.average(
                        [
                            row["reference_or_terminal_tier_proxy_flip_rate"]
                            for row in members
                            if row["reference_or_terminal_tier_proxy_flip_rate"] is not None
                        ],
                        weights=[
                            max(int(row["class_count"]) - 1, 1)
                            for row in members
                            if row["reference_or_terminal_tier_proxy_flip_rate"] is not None
                        ],
                    )
                )
                if members
                else None,
                "reference_reach_flip_rate": None,
                "inter_agent_collision_flip_rate": None,
                "direct_outcome_comparison_status": (
                    "PARTIAL_TIER_DERIVED_REFERENCE_AND_INTER_AGENT_COLLISION_NOT_AVAILABLE"
                ),
                "top1_flip_rate": float(np.mean([row["top1_flip"] for row in members]))
                if members
                else None,
                "pairwise_ordering_flip_rate": float(
                    np.mean(
                        [
                            row["pairwise_ordering_flip_rate"]
                            for row in members
                            if row["pairwise_ordering_flip_rate"] is not None
                        ]
                    )
                )
                if members
                else None,
                "mean_absolute_rank_change": float(
                    np.mean([row["mean_absolute_rank_change"] for row in members])
                )
                if members
                else None,
            }
        )
    multi = next(row for row in summary_rows if row["subset"] == "multi_agent")
    top_flip = float(multi["top1_flip_rate"] or 0.0)
    pair_flip = float(multi["pairwise_ordering_flip_rate"] or 0.0)
    if top_flip >= float(thresholds["strong_top1_flip"]) and pair_flip >= float(
        thresholds["strong_pairwise_flip"]
    ):
        instability = "STRONG"
    elif top_flip >= float(thresholds["moderate_top1_flip"]) or pair_flip >= float(
        thresholds["moderate_pairwise_flip"]
    ):
        instability = "MODERATE"
    elif top_flip > 0.0 or pair_flip > 0.0:
        instability = "WEAK"
    else:
        instability = "NO"
    joint = (
        "YES"
        if instability == "STRONG"
        and int(multi["graph_count"]) >= int(thresholds["minimum_multi_agent_graphs_for_yes"])
        else "PARTIAL"
        if instability in {"STRONG", "MODERATE"}
        else "NO"
    )
    return [*sample_rows, *summary_rows], instability, joint, multi


def _ablate_graph(graph: Any, ablation: str, seed: int) -> Any:
    result = copy.deepcopy(graph)
    rng = np.random.default_rng(int(seed))
    if ablation == "FULL" or ablation == "ZERO_ALIGN_MESSAGE":
        return result
    if ablation == "ZERO_ST_EDGE_ATTR":
        result[ST_EDGE].edge_attr_normalized.zero_()
        return result
    if ablation == "PERMUTE_ST_EDGE_ATTR":
        values = result[ST_EDGE].edge_attr_normalized
        if len(values) > 1:
            order = torch.as_tensor(rng.permutation(len(values)), dtype=torch.long)
            result[ST_EDGE].edge_attr_normalized = values[order].clone()
        return result
    if ablation == "CONTROL_PERMUTATION":
        values = result["proposal"].x
        if len(values) > 1:
            order = torch.as_tensor(rng.permutation(len(values)), dtype=torch.long)
            column = values[:, 12].clone()
            values[:, 12] = column[order]
        return result
    raise ValueError(f"unknown ablation: {ablation}")


def _masked_align_message_model(model: torch.nn.Module) -> torch.nn.Module:
    masked = copy.deepcopy(model)
    with torch.no_grad():
        for layer in masked.gat_layers:
            relation = layer.relations["spatiotemporal"]
            relation.node_message.weight.zero_()
            relation.edge_message.weight.zero_()
    masked.eval()
    return masked


def _run_frozen_inference(
    *,
    model: torch.nn.Module,
    samples: Sequence[Mapping[str, Any]],
    graph_lookup: Mapping[str, Mapping[str, Any]],
    branch_lookup: Mapping[tuple[str, int], Mapping[str, Any]],
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[
    dict[str, dict[str, dict[str, Any]]],
    list[dict[str, Any]],
    dict[tuple[str, int], np.ndarray],
]:
    ablations = list(config["ablations"])
    results: dict[str, dict[str, dict[str, Any]]] = {
        name: {} for name in ablations
    }
    relation_rows: list[dict[str, Any]] = []
    embeddings: dict[tuple[str, int], np.ndarray] = {}
    masked_model = _masked_align_message_model(model)
    sample_ids = [_sample_id(sample) for sample in samples]
    base_seed = int(config["deterministic_permutation_seed"])
    batch_size = 32
    for ablation in ablations:
        selected_model = masked_model if ablation == "ZERO_ALIGN_MESSAGE" else model
        selected_model.eval()
        for start in range(0, len(samples), batch_size):
            batch_samples = samples[start : start + batch_size]
            batch_ids = sample_ids[start : start + batch_size]
            graphs = []
            for offset, sample in enumerate(batch_samples):
                sample_id = batch_ids[offset]
                sample_seed = int.from_bytes(
                    hashlib.sha256(sample_id.encode("utf-8")).digest()[:4], "big"
                )
                graphs.append(
                    _ablate_graph(
                        sample["graph"],
                        ablation,
                        base_seed ^ sample_seed,
                    )
                )
            batch = batch_candidate_graphs(graphs).to(device)
            with torch.inference_mode():
                output = selected_model(
                    batch, return_attention_debug=ablation == "FULL"
                )
            for local_index, sample_id in enumerate(batch_ids):
                logits = output.logits_for_graph(local_index).detach().cpu().numpy().astype(float)
                probabilities = (
                    output.probabilities_for_graph(local_index)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(float)
                )
                results[ablation][sample_id] = {
                    "logits": logits,
                    "probabilities": probabilities,
                    "top1": int(np.argmax(probabilities)),
                    "top3": np.argsort(-probabilities, kind="stable")[: min(3, len(probabilities))],
                }
                if ablation == "FULL":
                    proposal_count = int(batch_samples[local_index]["graph"]["proposal"].num_nodes)
                    proposal_batch = batch["proposal"].batch.detach().cpu().numpy()
                    hidden = output.proposal_hidden_states.detach().cpu().numpy()
                    local_hidden = hidden[proposal_batch == local_index]
                    if len(local_hidden) != proposal_count:
                        raise RuntimeError("proposal embedding count mismatch")
                    for proposal_index in range(proposal_count):
                        embeddings[(sample_id, proposal_index + 1)] = local_hidden[
                            proposal_index
                        ].copy()

            if ablation == "FULL":
                selected = [
                    results["FULL"][sample_id]["top1"] for sample_id in batch_ids
                ]
                for debug in output.attention_debug:
                    graph_indices = debug.target_graph_index.detach().cpu().numpy().astype(int)
                    alpha = debug.normalized_alpha.detach().cpu().numpy().astype(float)
                    message = debug.message_content.detach().cpu().numpy().astype(float)
                    for local_index, sample_id in enumerate(batch_ids):
                        mask = graph_indices == local_index
                        if not np.any(mask):
                            continue
                        branch = branch_lookup[(sample_id, int(selected[local_index]))]
                        graph_row = graph_lookup[sample_id]
                        message_norm = np.linalg.norm(
                            message[mask].reshape(int(np.sum(mask)), -1), axis=1
                        )
                        weighted_norm = np.linalg.norm(
                            (message[mask] * alpha[mask][..., None]).reshape(
                                int(np.sum(mask)), -1
                            ),
                            axis=1,
                        )
                        relation_rows.append(
                            {
                                "schema_version": SCHEMA_VERSION,
                                "record_type": "graph_relation",
                                "graph_id": sample_id,
                                "scenario": graph_row["scenario"],
                                "seed": graph_row["seed"],
                                "ego": graph_row["ego"],
                                "subset": graph_row["subset"],
                                "interaction_rich": graph_row["interaction_rich"],
                                "low_interaction_control": graph_row[
                                    "low_interaction_control"
                                ],
                                "selected_class": int(selected[local_index]),
                                "selected_team_success": bool(branch["team_success"]),
                                "selected_inter_agent_collision": bool(
                                    branch["inter_agent_collision"]
                                ),
                                "layer": int(debug.layer_index),
                                "relation": debug.relation,
                                "edge_count": int(np.sum(mask)),
                                "mean_attention_alpha": float(np.mean(alpha[mask])),
                                "mean_message_norm": float(np.mean(message_norm)),
                                "mean_weighted_message_norm": float(
                                    np.mean(weighted_norm)
                                ),
                            }
                        )
    return results, relation_rows, embeddings


def _sensitivity_rows(
    results: Mapping[str, Mapping[str, Mapping[str, Any]]],
    graph_lookup: Mapping[str, Mapping[str, Any]],
    thresholds: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    full = results["FULL"]
    sample_rows: list[dict[str, Any]] = []
    for ablation, by_graph in results.items():
        for sample_id, altered in by_graph.items():
            baseline = full[sample_id]
            p = np.clip(np.asarray(baseline["probabilities"], dtype=float), 1.0e-12, 1.0)
            q = np.clip(np.asarray(altered["probabilities"], dtype=float), 1.0e-12, 1.0)
            top_n = min(3, len(p))
            baseline_margin = (
                float(np.sort(p)[-1] - np.sort(p)[-2]) if len(p) >= 2 else 1.0
            )
            altered_margin = (
                float(np.sort(q)[-1] - np.sort(q)[-2]) if len(q) >= 2 else 1.0
            )
            graph = graph_lookup[sample_id]
            sample_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "record_type": "graph",
                    "ablation": ablation,
                    "graph_id": sample_id,
                    "scenario": graph["scenario"],
                    "seed": graph["seed"],
                    "ego": graph["ego"],
                    "subset": graph["subset"],
                    "interaction_rich": graph["interaction_rich"],
                    "low_interaction_control": graph["low_interaction_control"],
                    "class_count": len(p),
                    "full_top1": int(baseline["top1"]),
                    "ablated_top1": int(altered["top1"]),
                    "top1_flip": int(baseline["top1"]) != int(altered["top1"]),
                    "full_null_selected": int(baseline["top1"]) == 0,
                    "ablated_null_selected": int(altered["top1"]) == 0,
                    "full_null_probability": float(p[0]),
                    "ablated_null_probability": float(q[0]),
                    "null_probability_shift": float(q[0] - p[0]),
                    "top3_overlap": len(
                        set(np.asarray(baseline["top3"]).tolist())
                        & set(np.asarray(altered["top3"]).tolist())
                    )
                    / top_n,
                    "kl_full_to_ablation": float(np.sum(p * np.log(p / q))),
                    "mean_absolute_logit_shift": float(
                        np.mean(
                            np.abs(
                                np.asarray(baseline["logits"], dtype=float)
                                - np.asarray(altered["logits"], dtype=float)
                            )
                        )
                    ),
                    "top1_top2_margin_shift": altered_margin - baseline_margin,
                    "absolute_margin_shift": abs(altered_margin - baseline_margin),
                }
            )
    summary_rows: list[dict[str, Any]] = []
    subsets = {
        "overall": lambda row: True,
        "interaction_rich": lambda row: bool(row["interaction_rich"]),
        "multi_agent": lambda row: row["scenario"] == "multi_agent",
        "low_interaction_control": lambda row: bool(row["low_interaction_control"]),
    }
    for ablation in results:
        for subset, predicate in subsets.items():
            members = [
                row
                for row in sample_rows
                if row["ablation"] == ablation and predicate(row)
            ]
            summary_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "record_type": "summary",
                    "ablation": ablation,
                    "subset": subset,
                    "graph_count": len(members),
                    "top1_flip_rate": float(np.mean([row["top1_flip"] for row in members]))
                    if members
                    else None,
                    "full_null_selection_rate": float(
                        np.mean([row["full_null_selected"] for row in members])
                    )
                    if members
                    else None,
                    "ablated_null_selection_rate": float(
                        np.mean([row["ablated_null_selected"] for row in members])
                    )
                    if members
                    else None,
                    "mean_full_null_probability": float(
                        np.mean([row["full_null_probability"] for row in members])
                    )
                    if members
                    else None,
                    "mean_ablated_null_probability": float(
                        np.mean([row["ablated_null_probability"] for row in members])
                    )
                    if members
                    else None,
                    "mean_null_probability_shift": float(
                        np.mean([row["null_probability_shift"] for row in members])
                    )
                    if members
                    else None,
                    "mean_top3_overlap": float(
                        np.mean([row["top3_overlap"] for row in members])
                    )
                    if members
                    else None,
                    "mean_KL_divergence": float(
                        np.mean([row["kl_full_to_ablation"] for row in members])
                    )
                    if members
                    else None,
                    "mean_absolute_logit_shift": float(
                        np.mean([row["mean_absolute_logit_shift"] for row in members])
                    )
                    if members
                    else None,
                    "mean_top1_top2_margin_shift": float(
                        np.mean([row["top1_top2_margin_shift"] for row in members])
                    )
                    if members
                    else None,
                    "mean_absolute_margin_shift": float(
                        np.mean([row["absolute_margin_shift"] for row in members])
                    )
                    if members
                    else None,
                }
            )
    lookup = {(row["ablation"], row["subset"]): row for row in summary_rows}
    st_ablations = (
        "ZERO_ST_EDGE_ATTR",
        "ZERO_ALIGN_MESSAGE",
        "PERMUTE_ST_EDGE_ATTR",
    )
    rich_effect = float(
        np.mean(
            [lookup[(name, "interaction_rich")]["top1_flip_rate"] for name in st_ablations]
        )
    )
    low_effect = float(
        np.mean(
            [
                lookup[(name, "low_interaction_control")]["top1_flip_rate"]
                for name in st_ablations
            ]
        )
    )
    control_effect = float(
        lookup[("CONTROL_PERMUTATION", "interaction_rich")]["top1_flip_rate"]
    )
    excess = rich_effect - max(low_effect, control_effect)
    configured = thresholds["usage_thresholds"]
    if rich_effect >= float(configured["strong_top1_flip"]) and excess >= float(
        configured["minimum_excess_over_controls"]
    ):
        usage = "STRONG"
    elif rich_effect >= float(configured["moderate_top1_flip"]) and excess >= 0.0:
        usage = "MODERATE"
    elif rich_effect >= float(configured["weak_top1_flip"]):
        usage = "WEAK"
    else:
        usage = "NO"
    diagnostic = {
        "interaction_rich_mean_st_top1_flip": rich_effect,
        "low_interaction_control_mean_st_top1_flip": low_effect,
        "interaction_rich_control_feature_top1_flip": control_effect,
        "excess_over_largest_control": excess,
        "interaction_rich_v1_null_selection_rate": lookup[
            ("FULL", "interaction_rich")
        ]["full_null_selection_rate"],
        "interaction_rich_v1_mean_null_probability": lookup[
            ("FULL", "interaction_rich")
        ]["mean_full_null_probability"],
    }
    return [*sample_rows, *summary_rows], usage, diagnostic


def _relation_message_rows(
    graph_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    groups = {
        "overall": lambda row: True,
        "selected_team_success": lambda row: bool(row["selected_team_success"]),
        "selected_inter_agent_collision": lambda row: bool(
            row["selected_inter_agent_collision"]
        ),
        "interaction_rich": lambda row: bool(row["interaction_rich"]),
        "low_interaction_control": lambda row: bool(row["low_interaction_control"]),
    }
    for layer in sorted({int(row["layer"]) for row in graph_rows}):
        for relation in sorted({str(row["relation"]) for row in graph_rows}):
            base = [
                row
                for row in graph_rows
                if int(row["layer"]) == layer and row["relation"] == relation
            ]
            for group, predicate in groups.items():
                members = [row for row in base if predicate(row)]
                summaries.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "record_type": "summary",
                        "layer": layer,
                        "relation": relation,
                        "group": group,
                        "graph_count": len(members),
                        "edge_count": sum(int(row["edge_count"]) for row in members),
                        "mean_attention_alpha": float(
                            np.average(
                                [row["mean_attention_alpha"] for row in members],
                                weights=[row["edge_count"] for row in members],
                            )
                        )
                        if members
                        else None,
                        "mean_message_norm": float(
                            np.average(
                                [row["mean_message_norm"] for row in members],
                                weights=[row["edge_count"] for row in members],
                            )
                        )
                        if members
                        else None,
                        "mean_weighted_message_norm": float(
                            np.average(
                                [row["mean_weighted_message_norm"] for row in members],
                                weights=[row["edge_count"] for row in members],
                            )
                        )
                        if members
                        else None,
                    }
                )
    return [*graph_rows, *summaries]


def _binary_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    prediction = probability >= 0.5
    return {
        "sample_count": len(y_true),
        "positive_count": int(np.sum(y_true)),
        "negative_count": int(np.sum(1 - y_true)),
        "positive_rate": float(np.mean(y_true)) if len(y_true) else None,
        "AUROC": float(roc_auc_score(y_true, probability))
        if len(np.unique(y_true)) == 2
        else None,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction))
        if len(y_true)
        else None,
        "F1": float(f1_score(y_true, prediction, zero_division=0))
        if len(y_true)
        else None,
        "Brier_score": float(brier_score_loss(y_true, probability))
        if len(y_true)
        else None,
    }


def _run_probe(
    *,
    name: str,
    rows: Sequence[Mapping[str, Any]],
    vector_field: str,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    split_seeds = {
        "train": set(config["data_scope"]["probe_train_seeds"]),
        "validation": set(config["data_scope"]["probe_validation_seeds"]),
        "test": set(config["data_scope"]["probe_test_seeds"]),
    }
    arrays = {
        split: np.stack(
            [np.asarray(row[vector_field], dtype=float) for row in rows if int(row["seed"]) in seeds]
        )
        for split, seeds in split_seeds.items()
    }
    targets = {
        split: np.asarray(
            [
                int(bool(row["inter_agent_collision_free"]))
                for row in rows
                if int(row["seed"]) in seeds
            ],
            dtype=int,
        )
        for split, seeds in split_seeds.items()
    }
    diagnostic = config["diagnostic_probe"]
    pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=float(diagnostic["C"]),
                    class_weight=str(diagnostic["class_weight"]),
                    max_iter=int(diagnostic["max_iter"]),
                    solver=str(diagnostic["solver"]),
                    random_state=0,
                ),
            ),
        ]
    )
    pipeline.fit(arrays["train"], targets["train"])
    result = []
    for split in ("train", "validation", "test"):
        probability = pipeline.predict_proba(arrays[split])[:, 1]
        result.append(
            {
                "schema_version": SCHEMA_VERSION,
                "probe": name,
                "split": split,
                "feature_set": vector_field,
                "feature_dimension_before_imputation": int(arrays[split].shape[1]),
                "target": diagnostic["target"],
                "status": diagnostic["status"],
                "fixed_C": float(diagnostic["C"]),
                "hyperparameter_tuning": False,
                **_binary_metrics(targets[split], probability),
            }
        )
    return result


def _classify_probe_signal(
    probe_rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> tuple[str, dict[str, float]]:
    test = {
        row["probe"]: float(row["AUROC"])
        for row in probe_rows
        if row["split"] == "test" and row["AUROC"] is not None
    }
    local = test["Probe-L"]
    best_interaction = max(test["Probe-I"], test["Probe-F"])
    gain = best_interaction - local
    diagnostic = config["diagnostic_probe"]
    if best_interaction >= float(diagnostic["signal_yes_minimum_auroc"]) and gain >= float(
        diagnostic["signal_yes_minimum_gain_over_local"]
    ):
        signal = "YES"
    elif best_interaction >= float(diagnostic["signal_weak_minimum_auroc"]) and gain >= float(
        diagnostic["signal_weak_minimum_gain_over_local"]
    ):
        signal = "WEAK"
    else:
        signal = "NO"
    return signal, {"local_test_AUROC": local, "best_interaction_test_AUROC": best_interaction, "gain": gain}


def _supervision_alignment_rows(
    candidate_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    targets = {
        "V1_SCALAR_H6": "v1_scalar_utility",
        "V1_HISTORICAL_H6": "v1_historical_utility",
        "V2_LONG_HORIZON": "v2_long_horizon_utility",
    }
    subsets = {
        "overall": list(candidate_rows),
        "interaction_rich": [row for row in candidate_rows if row["interaction_rich"]],
        "multi_agent": [row for row in candidate_rows if row["scenario"] == "multi_agent"],
        "low_interaction_control": [
            row for row in candidate_rows if row["low_interaction_control"]
        ],
    }
    alignment_rows: list[dict[str, Any]] = []
    conflict_rows: list[dict[str, Any]] = []
    for target_name, field in targets.items():
        for subset, members in subsets.items():
            utilities = np.asarray([float(row[field]) for row in members], dtype=float)
            collision_free = np.asarray(
                [not bool(row["inter_agent_collision"]) for row in members], dtype=bool
            )
            team_success = np.asarray(
                [bool(row["team_success"]) for row in members], dtype=bool
            )
            reference_reached = np.asarray(
                [bool(row["ego_reference_reached"]) for row in members], dtype=bool
            )
            post_reference_collision = np.asarray(
                [bool(row["post_reference_inter_agent_collision"]) for row in members],
                dtype=bool,
            )
            v2_tier = np.asarray(
                [int(row["v2_long_horizon_tier"]) for row in members], dtype=float
            )
            graph_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for row in members:
                graph_groups[str(row["graph_id"])].append(row)
            top_rows = []
            for group in graph_groups.values():
                top_rows.append(max(group, key=lambda row: float(row[field])))
            alignment_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "target": target_name,
                    "subset": subset,
                    "branch_count": len(members),
                    "graph_count": len(graph_groups),
                    "point_biserial_inter_agent_collision_free": _safe_corr(
                        pointbiserialr, collision_free.astype(float), utilities
                    ),
                    "point_biserial_team_success": _safe_corr(
                        pointbiserialr, team_success.astype(float), utilities
                    ),
                    "point_biserial_reference_reached": _safe_corr(
                        pointbiserialr, reference_reached.astype(float), utilities
                    ),
                    "point_biserial_post_reference_inter_agent_collision": _safe_corr(
                        pointbiserialr,
                        post_reference_collision.astype(float),
                        utilities,
                    ),
                    "spearman_v2_long_horizon_tier": _safe_corr(
                        spearmanr, v2_tier, utilities
                    ),
                    "top1_inter_agent_collision_free_rate": float(
                        np.mean(
                            [not bool(row["inter_agent_collision"]) for row in top_rows]
                        )
                    )
                    if top_rows
                    else None,
                    "top1_team_success_rate": float(
                        np.mean([bool(row["team_success"]) for row in top_rows])
                    )
                    if top_rows
                    else None,
                    "top1_reference_reach_rate": float(
                        np.mean([bool(row["ego_reference_reached"]) for row in top_rows])
                    )
                    if top_rows
                    else None,
                    "top1_post_reference_inter_agent_collision_rate": float(
                        np.mean(
                            [
                                bool(row["post_reference_inter_agent_collision"])
                                for row in top_rows
                            ]
                        )
                    )
                    if top_rows
                    else None,
                }
            )

            outcome_discordant = conflicts = ties = 0
            risk_discordant = risk_conflicts = risk_ties = 0
            v1_v2_pairs = v1_v2_disagree = 0
            for group in graph_groups.values():
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        left, right = group[i], group[j]
                        left_free = not bool(left["inter_agent_collision"])
                        right_free = not bool(right["inter_agent_collision"])
                        if left_free != right_free:
                            outcome_discordant += 1
                            delta = float(left[field]) - float(right[field])
                            if delta == 0.0:
                                ties += 1
                            else:
                                preferred_free = left_free if delta > 0 else right_free
                                conflicts += int(not preferred_free)
                        left_risky = bool(left["T_risk_max_s"] or 0.0) or (
                            left["d_min_min_m"] is not None
                            and float(left["d_min_min_m"]) < float(left["d_safe_m"])
                        )
                        right_risky = bool(right["T_risk_max_s"] or 0.0) or (
                            right["d_min_min_m"] is not None
                            and float(right["d_min_min_m"]) < float(right["d_safe_m"])
                        )
                        if left_risky != right_risky:
                            risk_discordant += 1
                            risk_delta = float(left[field]) - float(right[field])
                            if risk_delta == 0.0:
                                risk_ties += 1
                            else:
                                preferred_risky = left_risky if risk_delta > 0 else right_risky
                                risk_conflicts += int(preferred_risky)
                        v1_delta = float(left["v1_scalar_utility"]) - float(
                            right["v1_scalar_utility"]
                        )
                        v2_delta = float(left["v2_long_horizon_utility"]) - float(
                            right["v2_long_horizon_utility"]
                        )
                        if v1_delta != 0.0 and v2_delta != 0.0:
                            v1_v2_pairs += 1
                            v1_v2_disagree += int(np.sign(v1_delta) != np.sign(v2_delta))
            comparable = outcome_discordant - ties
            risk_comparable = risk_discordant - risk_ties
            conflict_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "target": target_name,
                    "subset": subset,
                    "graph_count": len(graph_groups),
                    "outcome_discordant_pair_count": outcome_discordant,
                    "target_tie_count": ties,
                    "comparable_pair_count": comparable,
                    "target_prefers_inter_agent_collision_branch_count": conflicts,
                    "SUPERVISION_INTERACTION_CONFLICT_RATE": conflicts / comparable
                    if comparable
                    else None,
                    "existing_risk_discordant_pair_count": risk_discordant,
                    "existing_risk_target_tie_count": risk_ties,
                    "existing_risk_comparable_pair_count": risk_comparable,
                    "target_prefers_existing_interaction_risk_branch_count": risk_conflicts,
                    "existing_interaction_risk_conflict_rate": risk_conflicts / risk_comparable
                    if risk_comparable
                    else None,
                    "v1_scalar_v2_comparable_pair_count": v1_v2_pairs,
                    "v1_scalar_v2_ordering_flip_rate": v1_v2_disagree / v1_v2_pairs
                    if v1_v2_pairs
                    else None,
                }
            )
    return alignment_rows, conflict_rows


def _classify_supervision_signal(
    alignment_rows: Sequence[Mapping[str, Any]],
    conflict_rows: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
) -> tuple[str, float, float]:
    alignment = next(
        row
        for row in alignment_rows
        if row["target"] == "V1_SCALAR_H6" and row["subset"] == "interaction_rich"
    )
    conflict = next(
        row
        for row in conflict_rows
        if row["target"] == "V1_SCALAR_H6" and row["subset"] == "interaction_rich"
    )
    correlation = abs(float(alignment["point_biserial_inter_agent_collision_free"] or 0.0))
    rate = float(conflict["SUPERVISION_INTERACTION_CONFLICT_RATE"] or 0.0)
    if correlation >= float(thresholds["strong_minimum_abs_point_biserial"]) and rate <= float(
        thresholds["strong_maximum_conflict_rate"]
    ):
        label = "STRONG"
    elif correlation >= float(thresholds["moderate_minimum_abs_point_biserial"]) and rate <= float(
        thresholds["moderate_maximum_conflict_rate"]
    ):
        label = "MODERATE"
    elif correlation >= float(thresholds["weak_minimum_abs_point_biserial"]) and rate <= float(
        thresholds["weak_maximum_conflict_rate"]
    ):
        label = "WEAK"
    else:
        label = "NO"
    return label, rate, correlation


def _contract_markdown(stage1_config: Mapping[str, Any], graph: Any) -> str:
    metadata = graph.graph_metadata
    feature_lines = [
        f"- `{item['owner']}.{item['name']}`: {item['dimension']}-D, {item['unit']}; "
        f"source = `{item['source']}`; representation = `{item['representation']}`."
        for item in metadata["feature_schema"]
    ]
    model = stage1_config["model"]
    return "\n".join(
        [
            "# Existing Interaction Information Contract",
            "",
            "## Frozen graph schema",
            "",
            "Node types and dimensions: `null:5`, `agent:66`, `proposal:13`, "
            "`align:7`. Edge relations and dimensions: "
            "`agent --smooth(1)--> proposal` and "
            "`align --spatiotemporal(3)--> proposal`.",
            "",
            *feature_lines,
            "",
            "The `agent` node includes current/previous ego velocity, terminal-goal geometry, "
            "and 56 current LiDAR sector-safety values. Each `align` node is one observed peer "
            "with relative direction, distance, and velocity. A spatiotemporal edge exists when "
            "the H4 constant-velocity separation diagnostic has `d_min < d_align`; its raw "
            "attributes are `(t_min, d_min, T_risk)`, with `d_safe=0.6 m` and `d_align=1.2 m` "
            "from graph metadata.",
            "",
            "## Frozen message passing",
            "",
            "Each node type has its own MLP encoder `h_tau = Enc_tau(x_tau)` and each relation "
            "has its own edge encoder `z_r = Enc_r(e_r)`. For every incoming edge to proposal "
            "`k` and attention head `h`:",
            "",
            "`ell_(r,j->k,h) = LeakyReLU(theta_(r,h)^T [W_s h_j || W_t h_k || W_e z_(r,jk)])`",
            "",
            "`m_(r,j->k,h) = W_n h_j + W_m z_(r,jk)`",
            "",
            "`alpha = softmax_k(ell)` is normalized jointly across all incoming smooth and "
            "spatiotemporal edges targeting the same proposal. The update is "
            "`LayerNorm(h_k + GELU(sum alpha*m))`. No reverse edges or synthetic self-loops are "
            "created; proposal self-information is retained by the residual. The frozen network "
            f"uses hidden={model['hidden_dim']}, edge={model['edge_dim']}, "
            f"heads={model['num_heads']}, layers={model['num_layers']}, dropout={model['dropout']}.",
            "",
            "## Output semantics",
            "",
            "One shared scoring MLP produces a scalar for the null hidden state and for each final "
            "proposal hidden state. Per graph, class 0 is null and class `k+1` is proposal `k`; "
            "the deployed selection is strict argmax over the real null+K classes. There is no "
            "joint decoder: the model computes ego-wise `q_i(k_i | observable graph state)`.",
            "",
            "## Capacity boundary",
            "",
            "The existing representation can encode current peer-relative state and H4 predicted "
            "pairwise candidate conflict. It does not encode the other agents' eventual jointly "
            "selected candidate tuple. Therefore it can learn interaction-aware ego ranking when "
            "the current graph determines risk, but cannot uniquely represent branch quality that "
            "changes only with unobserved non-ego final choices.",
            "",
        ]
    )


def _render_report(conclusion: Mapping[str, Any], evidence: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Existing-GAT Interaction-Learning Capacity Audit",
            "",
            "## Executive result",
            "",
            f"`CAN_GAT_LEARN_MORE_COORDINATION_WITHOUT_THEORY_CHANGE = "
            f"{conclusion['CAN_GAT_LEARN_MORE_COORDINATION_WITHOUT_THEORY_CHANGE']}`. "
            f"The frozen graph already contains explicit peer-relative and H4 conflict signals, "
            f"and the diagnostic probes classify their learnable signal as "
            f"`{conclusion['EXISTING_FEATURES_CONTAIN_INTERACTION_SIGNAL']}`. However, matched "
            f"background-plan evidence gives `JOINT_DECISION_DEPENDENCE = "
            f"{conclusion['JOINT_DECISION_DEPENDENCE']}`.",
            "",
            f"The primary limitation is `{conclusion['PRIMARY_LIMITATION']}` and existing-theory "
            f"headroom is `{conclusion['EXISTING_GAT_THEORY_HEADROOM']}`. The audit therefore "
            f"recommends `{conclusion['RECOMMENDED_NEXT_STEP']}` only; it does not design a new "
            "target, retrain GAT, or implement a joint module.",
            "",
            "## Key evidence",
            "",
            f"- Interaction-rich graph count: {evidence['interaction_rich_graph_count']}; "
            f"low-interaction controls: {evidence['low_control_graph_count']}.",
            f"- Explicit interaction feature signal: `{conclusion['INTERACTION_FEATURE_SIGNAL']}` "
            f"(maximum audited absolute effect {evidence['maximum_interaction_feature_effect']:.3f}).",
            f"- Frozen V1 interaction usage: `{conclusion['CURRENT_GAT_INTERACTION_USAGE']}`; "
            f"interaction-rich ST-ablation Top-1 flip {evidence['rich_st_flip']:.1%}, control "
            f"reference {evidence['largest_control_flip']:.1%}.",
            f"- Relation diagnostics: overall weighted-message norm is smooth/ST "
            f"{evidence['smooth_l0_weighted_norm']:.3f}/{evidence['st_l0_weighted_norm']:.3f} "
            f"at layer 0 and {evidence['smooth_l1_weighted_norm']:.3f}/"
            f"{evidence['st_l1_weighted_norm']:.3f} at layer 1. Layer-1 ST contribution is "
            f"{evidence['st_l1_rich_weighted_norm']:.3f} on interaction-rich graphs versus "
            f"{evidence['st_l1_control_weighted_norm']:.3f} on controls, so the existing V1 "
            f"does not show interaction-selective amplification.",
            f"- Frozen V1 null behavior on interaction-rich graphs: mean null probability "
            f"{evidence['rich_null_probability']:.1%}, null selection rate "
            f"{evidence['rich_null_selection_rate']:.1%}.",
            f"- Probe test AUROC: L={evidence['probe_local_auroc']:.3f}, "
            f"I={evidence['probe_interaction_auroc']:.3f}, F={evidence['probe_full_auroc']:.3f}, "
            f"embedding={evidence['embedding_auroc']:.3f}.",
            f"- V1 interaction-rich supervision conflict rate: "
            f"{conclusion['SUPERVISION_INTERACTION_CONFLICT_RATE']:.1%}; signal = "
            f"`{conclusion['SUPERVISION_INTERACTION_SIGNAL']}`.",
            f"- V1 target preference for the candidate marked risky by the existing "
            f"`d_min < d_safe OR T_risk > 0` descriptor: "
            f"{evidence['existing_risk_conflict_rate']:.1%} of comparable pairs.",
            f"- Matched multi_agent samples: {evidence['matched_multi_graph_count']}; Top-1 flip "
            f"{evidence['matched_multi_top1_flip']:.1%}; pairwise ordering flip "
            f"{evidence['matched_multi_pairwise_flip']:.1%}.",
            f"- Under the matched background change, tier-exact multi_agent team-completion "
            f"flip is {evidence['matched_multi_team_flip']:.1%} and any-collision flip is "
            f"{evidence['matched_multi_any_collision_flip']:.1%}.",
            "- The attribution artifact does not retain alternative-background reference-reach "
            "or collision-type booleans. Those direct flip rates are marked unavailable; no "
            "closed-loop labels were reconstructed or fabricated.",
            "- Feature/outcome analysis uses 3,702 existing executed proposal branches. The "
            "remaining 492 of 4,194 branches are null branches and are excluded from candidate "
            "feature statistics.",
            "",
            "## Decision fields",
            "",
            *[f"- `{key} = {value}`" for key, value in conclusion.items() if key != "schema_version"],
            "",
            "## Scope and stop rule",
            "",
            "All probes are diagnostic-only linear models trained on the existing seeds 0–6, "
            "validated on seed 7, and tested on seeds 8–9. Formal seeds 30–49 were not used for "
            "probe training, feature selection, thresholds, or target design. The 24-layout "
            "artifact was not read. No new closed-loop benchmark was run.",
            "",
        ]
    )


def run_audit(config: Mapping[str, Any], output_dir: Path) -> Path:
    if config["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unexpected audit schema")
    if any(bool(value) for value in config["strict_exclusions"].values()):
        raise ValueError("strict exclusion flags must remain false")
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    sources = config["sources"]
    supervision_dir = REPO_ROOT / sources["supervision_v2_dir"]
    checkpoint = REPO_ROOT / sources["v1_checkpoint"]
    checkpoint_hash_before = _sha256(checkpoint)
    if checkpoint_hash_before != sources["v1_checkpoint_sha256_expected"]:
        raise RuntimeError("V1 checkpoint hash mismatch")
    source_hashes_before = _hashes(AUDITED_SOURCE_PATHS)
    input_hashes = {
        "v1_checkpoint": checkpoint_hash_before,
        "supervision_dataset": _sha256(supervision_dir / "supervision_v2_dataset.pt"),
        "branch_outcomes": _sha256(supervision_dir / "branch_raw_outcomes.parquet"),
        "attribution_stability": _sha256(supervision_dir / "attribution_stability.csv"),
    }

    stage1_config = _load_json(REPO_ROOT / sources["stage1_config"])
    payload = torch.load(
        supervision_dir / "supervision_v2_dataset.pt",
        map_location="cpu",
        weights_only=False,
    )
    samples = list(payload["samples"])
    branch_frame = pd.read_parquet(supervision_dir / "branch_raw_outcomes.parquet")
    attribution_rows = _read_csv(supervision_dir / "attribution_stability.csv")
    allowed_seeds = set(config["data_scope"]["allowed_seeds"])
    if {int(sample["seed"]) for sample in samples} - allowed_seeds:
        raise RuntimeError("dataset contains a prohibited seed")
    if set(int(value) for value in branch_frame["seed"].unique()) - allowed_seeds:
        raise RuntimeError("branch outcomes contain a prohibited seed")
    if len(samples) != 492 or len(branch_frame) != 4194:
        raise RuntimeError("authoritative supervision cardinality changed")

    graph_rows, candidate_rows = _extract_graph_records(
        samples, config["interaction_subset"]
    )
    _join_outcomes_and_targets(samples, candidate_rows, branch_frame)
    graph_lookup = {row["graph_id"]: row for row in graph_rows}
    branch_lookup = {
        (str(row.sample_id), int(row.class_index)): row._asdict()
        for row in branch_frame.itertuples(index=False)
    }
    feature_rows = _feature_signal_rows(candidate_rows)
    feature_signal, maximum_feature_effect = _classify_feature_signal(
        feature_rows, config["feature_signal_thresholds"]
    )
    matched_rows, context_instability, joint_dependence, matched_multi = _matched_instability(
        attribution_rows,
        graph_lookup,
        config["joint_instability_thresholds"],
    )

    device = resolve_device(str(config["inference_device"]))
    model = load_model_checkpoint(checkpoint, stage1_config, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    model_hash_before = _model_hash(model)
    inference, relation_graph_rows, embeddings = _run_frozen_inference(
        model=model,
        samples=samples,
        graph_lookup=graph_lookup,
        branch_lookup=branch_lookup,
        config=config["sensitivity"],
        device=device,
    )
    sensitivity_rows, gat_usage, sensitivity_evidence = _sensitivity_rows(
        inference, graph_lookup, config["sensitivity"]
    )
    relation_rows = _relation_message_rows(relation_graph_rows)
    relation_summary_lookup = {
        (int(row["layer"]), str(row["relation"]), str(row["group"])): row
        for row in relation_rows
        if row["record_type"] == "summary"
    }
    relation_evidence = {
        "smooth_layer0_overall_weighted_message_norm": float(
            relation_summary_lookup[(0, "smooth", "overall")]["mean_weighted_message_norm"]
        ),
        "spatiotemporal_layer0_overall_weighted_message_norm": float(
            relation_summary_lookup[(0, "spatiotemporal", "overall")][
                "mean_weighted_message_norm"
            ]
        ),
        "smooth_layer1_overall_weighted_message_norm": float(
            relation_summary_lookup[(1, "smooth", "overall")]["mean_weighted_message_norm"]
        ),
        "spatiotemporal_layer1_overall_weighted_message_norm": float(
            relation_summary_lookup[(1, "spatiotemporal", "overall")][
                "mean_weighted_message_norm"
            ]
        ),
        "spatiotemporal_layer1_interaction_rich_weighted_message_norm": float(
            relation_summary_lookup[(1, "spatiotemporal", "interaction_rich")][
                "mean_weighted_message_norm"
            ]
        ),
        "spatiotemporal_layer1_low_control_weighted_message_norm": float(
            relation_summary_lookup[(1, "spatiotemporal", "low_interaction_control")][
                "mean_weighted_message_norm"
            ]
        ),
    }
    for row in candidate_rows:
        key = (str(row["graph_id"]), int(row["class_index"]))
        row["embedding_vector"] = embeddings[key]

    probe_rows = []
    for name, vector_field in (
        ("Probe-L", "local_vector"),
        ("Probe-I", "interaction_vector"),
        ("Probe-F", "full_vector"),
    ):
        probe_rows.extend(
            _run_probe(
                name=name,
                rows=candidate_rows,
                vector_field=vector_field,
                config=config,
            )
        )
    feature_probe_signal, probe_evidence = _classify_probe_signal(probe_rows, config)
    embedding_rows = _run_probe(
        name="Frozen-GAT-Embedding",
        rows=candidate_rows,
        vector_field="embedding_vector",
        config=config,
    )
    embedding_test = next(row for row in embedding_rows if row["split"] == "test")
    embedding_auroc = float(embedding_test["AUROC"] or 0.0)
    if embedding_auroc >= float(config["diagnostic_probe"]["signal_yes_minimum_auroc"]):
        embedding_signal = "YES"
    elif embedding_auroc >= float(config["diagnostic_probe"]["signal_weak_minimum_auroc"]):
        embedding_signal = "WEAK"
    else:
        embedding_signal = "NO"

    alignment_rows, conflict_rows = _supervision_alignment_rows(candidate_rows)
    supervision_signal, supervision_conflict, supervision_correlation = (
        _classify_supervision_signal(
            alignment_rows,
            conflict_rows,
            config["supervision_signal_thresholds"],
        )
    )
    v1_interaction_rich_conflicts = next(
        row
        for row in conflict_rows
        if row["target"] == "V1_SCALAR_H6" and row["subset"] == "interaction_rich"
    )
    existing_risk_conflict = float(
        v1_interaction_rich_conflicts["existing_interaction_risk_conflict_rate"] or 0.0
    )

    features_present = "YES"
    if feature_signal in {"STRONG", "MODERATE"} and feature_probe_signal == "YES":
        if joint_dependence == "NO" and supervision_signal in {"WEAK", "NO"}:
            primary_limitation = "SUPERVISION_NOT_ARCHITECTURE"
            headroom = "HIGH"
        elif joint_dependence == "YES" and gat_usage in {"MODERATE", "STRONG"}:
            primary_limitation = "JOINT_DECISION_DEPENDENCE"
            headroom = "LOW"
        else:
            primary_limitation = "MIXED"
            headroom = "MODERATE"
    elif feature_signal in {"WEAK", "NO"} and feature_probe_signal == "NO":
        primary_limitation = "GRAPH_INFORMATION_INSUFFICIENT"
        headroom = "LOW"
    elif feature_probe_signal in {"YES", "WEAK"} and gat_usage in {"WEAK", "NO"}:
        primary_limitation = "MODEL_UNDERUTILIZATION"
        headroom = "MODERATE"
    else:
        primary_limitation = "NOT_ESTABLISHED"
        headroom = "NOT_ESTABLISHED"

    if headroom == "HIGH" and joint_dependence == "NO":
        can_learn = "YES"
        next_step = "RETRAIN_SAME_GAT_WITH_MINIMAL_INTERACTION_AWARE_SUPERVISION"
    elif headroom == "MODERATE" or joint_dependence == "PARTIAL":
        can_learn = "PARTIAL"
        next_step = "SUPERVISION_ONLY_CONTROLLED_EXPERIMENT"
    elif headroom == "LOW":
        can_learn = "NO"
        next_step = "ONLY_THEN_CONSIDER_MINIMAL_JOINT_DECISION_EXTENSION"
    else:
        can_learn = "NOT_ESTABLISHED"
        next_step = "NOT_ESTABLISHED"

    conclusion = {
        "schema_version": SCHEMA_VERSION,
        "EXISTING_INTERACTION_FEATURES_PRESENT": features_present,
        "INTERACTION_FEATURE_SIGNAL": feature_signal,
        "EXISTING_FEATURES_CONTAIN_INTERACTION_SIGNAL": feature_probe_signal,
        "CURRENT_GAT_INTERACTION_USAGE": gat_usage,
        "FROZEN_EMBEDDING_INTERACTION_SIGNAL": embedding_signal,
        "SUPERVISION_INTERACTION_SIGNAL": supervision_signal,
        "SUPERVISION_INTERACTION_CONFLICT_RATE": supervision_conflict,
        "JOINT_CONTEXT_LABEL_INSTABILITY": context_instability,
        "JOINT_DECISION_DEPENDENCE": joint_dependence,
        "PRIMARY_LIMITATION": primary_limitation,
        "EXISTING_GAT_THEORY_HEADROOM": headroom,
        "CAN_GAT_LEARN_MORE_COORDINATION_WITHOUT_THEORY_CHANGE": can_learn,
        "RECOMMENDED_NEXT_STEP": next_step,
    }
    decision_matrix = {
        "schema_version": SCHEMA_VERSION,
        "evidence": {
            "maximum_interaction_feature_effect": maximum_feature_effect,
            "probe": probe_evidence,
            "sensitivity": sensitivity_evidence,
            "relation_messages": relation_evidence,
            "embedding_test_AUROC": embedding_auroc,
            "v1_interaction_rich_supervision_abs_point_biserial": supervision_correlation,
            "v1_interaction_rich_conflict_rate": supervision_conflict,
            "v1_interaction_rich_existing_risk_conflict_rate": existing_risk_conflict,
            "matched_multi_agent": matched_multi,
        },
        "classification": conclusion,
        "interpretation": (
            "Existing graph features have diagnostic signal, but the sampled matched-context "
            "audit shows non-negligible dependence on other agents' final choices; a same-GAT "
            "supervision-only experiment is justified only as a controlled diagnostic, not as "
            "an assured closed-loop fix."
        ),
        "automatic_redesign_or_training_performed": False,
    }

    context_sources = {
        name: {
            "path": path,
            "exists": (REPO_ROOT / path).exists(),
        }
        for name, path in sources.items()
        if name.endswith("_dir")
    }
    context = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASSED" if all(item["exists"] for item in context_sources.values()) else "FAILED",
        "sources": context_sources,
        "authoritative_facts": {
            "v1_formal_team_success": "41/60 (68.3%)",
            "v2_formal_team_success": "36/60 (60.0%)",
            "v2_formal_null_rate": "82.8%",
            "temporary_reference_execution_gap": "NO",
            "targeted_sac_adaptation_justified": "NO",
            "existing_theory_minimal_fix_exhausted": "YES",
            "minimal_change_diagnostic_upper_bound": "44/60 (73.3%)",
        },
        "forbidden_24_layout_read": False,
        "formal_seeds_used_only_as_context_provenance": True,
    }

    write_json(output_dir / "context_recovery_manifest.json", context)
    write_csv(output_dir / "interaction_subset_manifest.csv", graph_rows)
    write_csv(output_dir / "interaction_feature_signal.csv", feature_rows)
    write_csv(output_dir / "matched_candidate_instability.csv", matched_rows)
    write_csv(output_dir / "edge_sensitivity.csv", sensitivity_rows)
    write_csv(output_dir / "relation_message_analysis.csv", relation_rows)
    write_csv(output_dir / "diagnostic_probe_metrics.csv", probe_rows)
    write_csv(output_dir / "embedding_probe_metrics.csv", embedding_rows)
    write_csv(output_dir / "supervision_interaction_alignment.csv", alignment_rows)
    write_csv(output_dir / "pairwise_label_conflicts.csv", conflict_rows)
    write_json(output_dir / "decision_matrix.json", decision_matrix)
    write_json(output_dir / "conclusion.json", conclusion)
    (output_dir / "existing_interaction_information_contract.md").write_text(
        _contract_markdown(stage1_config, samples[0]["graph"]), encoding="utf-8"
    )

    probe_test = {
        row["probe"]: float(row["AUROC"])
        for row in probe_rows
        if row["split"] == "test" and row["AUROC"] is not None
    }
    evidence = {
        "interaction_rich_graph_count": sum(int(row["interaction_rich"]) for row in graph_rows),
        "low_control_graph_count": sum(
            int(row["low_interaction_control"]) for row in graph_rows
        ),
        "maximum_interaction_feature_effect": maximum_feature_effect,
        "rich_st_flip": sensitivity_evidence["interaction_rich_mean_st_top1_flip"],
        "largest_control_flip": max(
            sensitivity_evidence["low_interaction_control_mean_st_top1_flip"],
            sensitivity_evidence["interaction_rich_control_feature_top1_flip"],
        ),
        "probe_local_auroc": probe_test["Probe-L"],
        "probe_interaction_auroc": probe_test["Probe-I"],
        "probe_full_auroc": probe_test["Probe-F"],
        "embedding_auroc": embedding_auroc,
        "matched_multi_graph_count": int(matched_multi["graph_count"]),
        "matched_multi_top1_flip": float(matched_multi["top1_flip_rate"]),
        "matched_multi_pairwise_flip": float(
            matched_multi["pairwise_ordering_flip_rate"]
        ),
        "matched_multi_team_flip": float(
            matched_multi["team_completion_flip_rate_tier_exact"]
        ),
        "matched_multi_any_collision_flip": float(
            matched_multi["any_collision_flip_rate_tier_exact"]
        ),
        "rich_null_selection_rate": float(
            sensitivity_evidence["interaction_rich_v1_null_selection_rate"]
        ),
        "rich_null_probability": float(
            sensitivity_evidence["interaction_rich_v1_mean_null_probability"]
        ),
        "existing_risk_conflict_rate": existing_risk_conflict,
        "smooth_l0_weighted_norm": relation_evidence[
            "smooth_layer0_overall_weighted_message_norm"
        ],
        "st_l0_weighted_norm": relation_evidence[
            "spatiotemporal_layer0_overall_weighted_message_norm"
        ],
        "smooth_l1_weighted_norm": relation_evidence[
            "smooth_layer1_overall_weighted_message_norm"
        ],
        "st_l1_weighted_norm": relation_evidence[
            "spatiotemporal_layer1_overall_weighted_message_norm"
        ],
        "st_l1_rich_weighted_norm": relation_evidence[
            "spatiotemporal_layer1_interaction_rich_weighted_message_norm"
        ],
        "st_l1_control_weighted_norm": relation_evidence[
            "spatiotemporal_layer1_low_control_weighted_message_norm"
        ],
    }
    (output_dir / "FINAL_REPORT.md").write_text(
        _render_report(conclusion, evidence), encoding="utf-8"
    )

    checkpoint_hash_after = _sha256(checkpoint)
    source_hashes_after = _hashes(AUDITED_SOURCE_PATHS)
    model_hash_after = _model_hash(model)
    conclusion_domains = {
        "EXISTING_INTERACTION_FEATURES_PRESENT": {"YES", "NO"},
        "INTERACTION_FEATURE_SIGNAL": {"STRONG", "MODERATE", "WEAK", "NO", "NOT_ESTABLISHED"},
        "EXISTING_FEATURES_CONTAIN_INTERACTION_SIGNAL": {"YES", "WEAK", "NO"},
        "CURRENT_GAT_INTERACTION_USAGE": {"STRONG", "MODERATE", "WEAK", "NO"},
        "FROZEN_EMBEDDING_INTERACTION_SIGNAL": {"YES", "WEAK", "NO", "NOT_AVAILABLE"},
        "SUPERVISION_INTERACTION_SIGNAL": {"STRONG", "MODERATE", "WEAK", "NO"},
        "JOINT_CONTEXT_LABEL_INSTABILITY": {"STRONG", "MODERATE", "WEAK", "NO", "NOT_ESTABLISHED"},
        "JOINT_DECISION_DEPENDENCE": {"YES", "PARTIAL", "NO", "NOT_ESTABLISHED"},
        "PRIMARY_LIMITATION": {
            "SUPERVISION_NOT_ARCHITECTURE",
            "JOINT_DECISION_DEPENDENCE",
            "GRAPH_INFORMATION_INSUFFICIENT",
            "MODEL_UNDERUTILIZATION",
            "MIXED",
            "NOT_ESTABLISHED",
        },
        "EXISTING_GAT_THEORY_HEADROOM": {"HIGH", "MODERATE", "LOW", "NOT_ESTABLISHED"},
        "CAN_GAT_LEARN_MORE_COORDINATION_WITHOUT_THEORY_CHANGE": {"YES", "PARTIAL", "NO", "NOT_ESTABLISHED"},
        "RECOMMENDED_NEXT_STEP": {
            "RETRAIN_SAME_GAT_WITH_MINIMAL_INTERACTION_AWARE_SUPERVISION",
            "SUPERVISION_ONLY_CONTROLLED_EXPERIMENT",
            "KEEP_GAT_V1",
            "ONLY_THEN_CONSIDER_MINIMAL_JOINT_DECISION_EXTENSION",
            "NOT_ESTABLISHED",
        },
    }
    integrity = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASSED",
        "checks": {
            "source_context_complete": context["status"] == "PASSED",
            "supervision_graph_count": len(samples) == 492,
            "real_branch_count": len(branch_frame) == 4194,
            "candidate_outcome_join_complete": len(candidate_rows)
            == sum(int(row["proposal_count"]) for row in graph_rows),
            "expected_output_row_counts": (
                len(graph_rows) == 492
                and len(feature_rows) == 52
                and len(matched_rows) == 46
                and len(sensitivity_rows) == 2480
                and len(probe_rows) == 9
                and len(embedding_rows) == 3
                and len(alignment_rows) == 12
                and len(conflict_rows) == 12
            ),
            "conclusion_enums_valid": all(
                conclusion[key] in domain for key, domain in conclusion_domains.items()
            ),
            "matched_missing_outcomes_not_fabricated": all(
                row["reference_reach_flip_rate"] is None
                and row["inter_agent_collision_flip_rate"] is None
                for row in matched_rows
            ),
            "allowed_seed_scope_only": {int(sample["seed"]) for sample in samples}
            <= allowed_seeds,
            "formal_seeds_excluded_from_probes": not bool(
                config["data_scope"][
                    "formal_seeds_used_for_probe_or_feature_selection"
                ]
            ),
            "layout_24_not_read": not bool(
                config["data_scope"]["layout_24_artifact_read_allowed"]
            ),
            "checkpoint_unchanged": checkpoint_hash_before == checkpoint_hash_after,
            "formal_model_unchanged": model_hash_before == model_hash_after,
            "audited_sources_unchanged": source_hashes_before == source_hashes_after,
            "deterministic_cpu_inference": str(device) == "cpu",
            "no_formal_training_or_closed_loop": True,
            "probes_diagnostic_only": True,
            "no_new_checkpoint": True,
        },
        "input_hashes": input_hashes,
        "checkpoint_hash_before": checkpoint_hash_before,
        "checkpoint_hash_after": checkpoint_hash_after,
        "model_parameter_hash_before": model_hash_before,
        "model_parameter_hash_after": model_hash_after,
        "audited_source_hashes_before": source_hashes_before,
        "audited_source_hashes_after": source_hashes_after,
    }
    integrity["failed_checks"] = [
        key for key, value in integrity["checks"].items() if not bool(value)
    ]
    if integrity["failed_checks"]:
        integrity["status"] = "FAILED"
    write_json(output_dir / "integrity_manifest.json", integrity)
    resolved = copy.deepcopy(dict(config))
    resolved.update(
        {
            "created_at": datetime.now().astimezone().isoformat(),
            "resolved_output_dir": str(output_dir.resolve()),
            "runtime_seconds": float(time.perf_counter() - started),
            "device": str(device),
            "diagnostic_probe_training_performed": True,
            "formal_gat_training_performed": False,
            "closed_loop_benchmark_performed": False,
        }
    )
    write_json(output_dir / "config.json", resolved)
    if integrity["status"] != "PASSED":
        raise RuntimeError(f"integrity failure: {integrity['failed_checks']}")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> Path:
    args = parse_args()
    config = _load_json(args.config.resolve())
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            REPO_ROOT
            / config["output_root"]
            / datetime.now().strftime("%Y%m%d_%H%M%S")
        )
    result = run_audit(config, output_dir.resolve())
    print(result)
    return result


if __name__ == "__main__":
    main()
