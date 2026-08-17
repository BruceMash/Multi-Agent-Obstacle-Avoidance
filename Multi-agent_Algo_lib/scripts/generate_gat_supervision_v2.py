"""Generate deployment-aligned Stage-I GAT supervision V2.

The program has a strict two-stage gate.  Stage A reproduces every original
source state/candidate and rebuilds historical-gate H4 graphs.  Stage B runs
the historical H6 diagnostic and absolute-episode-horizon V2 branches only
after Stage A passes.  It never trains a model or runs a method benchmark.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = Path(__file__).resolve().parent
for search_path in (REPO_ROOT, ALGO_ROOT, SCRIPTS_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from Environment.frozen_sac_dmp_execution import freeze_policy  # noqa: E402
from Guidance.reference_point_proposal_demo import ProposalConfig  # noqa: E402
from experiment_config import EXPERIMENT_CONFIG as SINGLE_AGENT_CONFIG  # noqa: E402
from planning.candidate_execution_benchmark import (  # noqa: E402
    environment_state_fingerprint,
    real_candidate_rollout,
)
from planning.candidate_execution_interface import (  # noqa: E402
    ExecutionNormalizationSpec,
    graph_ready_candidate_execution,
)
from planning.candidate_supervision import (  # noqa: E402
    branch_is_failure,
    compute_candidate_quality,
    target_bundle,
)
from planning.candidate_supervision_dataset import (  # noqa: E402
    _build_formal_previews,
    _quality_row,
)
from planning.gat_supervision_v2 import (  # noqa: E402
    BACKGROUND_FP_SHEP,
    BACKGROUND_NULL,
    HISTORICAL_GATE_NAME,
    SCHEMA_VERSION,
    BranchRollout,
    TieredTarget,
    build_background_plan,
    build_tiered_target,
    classify_attribution_stability,
    information_density,
    pairwise_ordering_agreement,
    run_long_horizon_branch,
    stable_hash,
)
from planning.heterogeneous_candidate_graph import (  # noqa: E402
    HeterogeneousCandidateGraphConfig,
    build_heterogeneous_candidate_graph_from_env,
)
from planning.historical_forcing_gate import (  # noqa: E402
    scoped_historical_preview_and_multi_agent_transition,
)
from planning.policy_preview import adapt_candidate_proposals  # noqa: E402
from planning.pre_gat_closed_loop import FPSHEPOnlineScoreSpec  # noqa: E402
from runner_sac import build_env as build_single_env  # noqa: E402
from runner_sac import build_model as build_single_model  # noqa: E402
from runner_sac import load_checkpoint  # noqa: E402
from scripts.generate_candidate_supervision_dataset import (  # noqa: E402
    _advance_environment_one_step,
    _build_supervision_environment,
    _generate_proposals,
    _policy_parameter_sha256,
)
from scripts.evaluate_single_policy_aligned_multi_agent import (  # noqa: E402
    build_single_distribution_multi_config,
)


DEFAULT_CONFIG_PATH = REPO_ROOT / "configs/evaluation/gat_supervision_v2.json"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            encoded = {}
            for key in fields:
                value = _jsonable(row.get(key))
                encoded[key] = (
                    json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                    if isinstance(value, (list, tuple, dict))
                    else value
                )
            writer.writerow(encoded)


def _parquet_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Keep numeric infinities numeric; serialize only nested provenance maps."""

    result: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, np.ndarray):
            result[str(key)] = value.tolist()
        elif isinstance(value, np.generic):
            result[str(key)] = value.item()
        elif isinstance(value, Mapping):
            result[str(key)] = json.dumps(
                _jsonable(value), ensure_ascii=False, sort_keys=True
            )
        elif isinstance(value, tuple):
            result[str(key)] = list(value)
        else:
            result[str(key)] = value
    return result


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _split(seed: int, mapping: Mapping[str, Sequence[int]]) -> str:
    matches = [name for name, seeds in mapping.items() if name != "role" and seed in seeds]
    if len(matches) != 1:
        raise ValueError(f"seed {seed} belongs to {len(matches)} splits")
    return matches[0]


def _assert_config(config: Mapping[str, Any]) -> None:
    if config["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unexpected supervision V2 schema")
    if int(config["H_preview"]) != 4 or int(config["H_v1_diagnostic"]) != 6:
        raise ValueError("V2 requires H4 graph input and H6 companion diagnostic")
    if int(config["max_absolute_episode_steps"]) != 220:
        raise ValueError("formal V2 branch must keep absolute episode horizon 220")
    if config["branch_budget"] != "max_0_220_minus_source_env_steps":
        raise ValueError("fixed-future-220 is forbidden for formal V2 truth")
    if bool(config["fixed_future_220_sensitivity_enabled"]):
        raise ValueError("fixed-future-220 sensitivity is outside this goal")
    if config["forcing_gate"]["name"] != HISTORICAL_GATE_NAME:
        raise ValueError("all post-source V2 transitions must use historical gate")
    if float(config["soft_target_temperature"]) != 0.25:
        raise ValueError("tau must remain frozen at 0.25")
    if list(config["effective_split"]["train"]) != list(range(7)):
        raise ValueError("effective train split must be seeds 0..6")
    if list(config["effective_split"]["validation"]) != [7]:
        raise ValueError("effective validation split must be seed 7")
    if list(config["effective_split"]["test"]) != [8, 9]:
        raise ValueError("effective test split must be seeds 8..9")
    if any(bool(value) for value in config["strict_exclusions"].values()):
        raise ValueError("strict exclusions must all remain false")
    if bool(config["stress_isolation"]["artifact_may_be_read_by_generator"]):
        raise ValueError("stress artifacts cannot enter V2 generation")


def _protected_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    source = REPO_ROOT / str(config["source_dataset_dir"])
    return {
        "checkpoint": REPO_ROOT / str(config["checkpoint"]),
        "proposal": REPO_ROOT / "Guidance/reference_point_proposal_demo.py",
        "fp_shep": REPO_ROOT / "planning/policy_preview.py",
        "graph_builder": REPO_ROOT / "planning/heterogeneous_candidate_graph.py",
        "gat_architecture": REPO_ROOT / "planning/gat/edge_enhanced_gat.py",
        "stage1_loss": REPO_ROOT / "planning/gat/stage1_training.py",
        "dmp": REPO_ROOT / "Controller/dmp_rl.py",
        "environment": REPO_ROOT / "Environment/multi_agent_dmp_env.py",
        "source_manifest": source / "manifest.csv",
        "source_graph_records": source / "graph_records.csv",
        "source_class_mapping": source / "class_mapping.csv",
        "source_state_groups": source / "state_groups.csv",
        "source_config": source / "config.json",
    }


def _proposal_metadata(proposal: Any) -> dict[str, Any]:
    names = (
        "azimuth_index",
        "elevation_index",
        "direction",
        "point",
        "distance",
        "raw_obstacle_distance",
        "obstacle_distance",
        "effective_safe_radius",
        "braking_distance",
        "safety_margin",
        "normalized_margin",
        "distance_progress",
        "normalized_progress",
        "alignment",
        "smoothness",
        "usable_length",
        "score",
    )
    return {
        name: _jsonable(getattr(proposal, name))
        for name in names
        if hasattr(proposal, name)
    }


def _graph_schema(graph: Any) -> dict[str, Any]:
    node_dims = {
        node_type: int(graph[node_type].x.shape[1]) for node_type in graph.node_types
    }
    edge_dims = {}
    for edge_type in graph.edge_types:
        store = graph[edge_type]
        edge_attr = getattr(store, "edge_attr", None)
        edge_dims["|".join(edge_type)] = (
            0 if edge_attr is None else int(edge_attr.shape[1])
        )
    return {
        "node_types": list(graph.node_types),
        "edge_types": [list(item) for item in graph.edge_types],
        "node_feature_dims": node_dims,
        "edge_feature_dims": edge_dims,
    }


def _graph_shift(old: Any, new: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    values: list[np.ndarray] = []
    for node_type in old.node_types:
        first = old[node_type].x.detach().cpu().numpy().astype(float)
        second = new[node_type].x.detach().cpu().numpy().astype(float)
        if first.shape != second.shape:
            result[f"node_{node_type}_shape_match"] = False
            continue
        delta = np.abs(second - first)
        result[f"node_{node_type}_shape_match"] = True
        result[f"node_{node_type}_mean_abs_shift"] = (
            float(np.mean(delta)) if delta.size else 0.0
        )
        result[f"node_{node_type}_max_abs_shift"] = float(np.max(delta)) if delta.size else 0.0
        if delta.size:
            values.append(delta.reshape(-1))
    for edge_type in old.edge_types:
        name = "|".join(edge_type)
        old_attr = getattr(old[edge_type], "edge_attr", None)
        new_attr = getattr(new[edge_type], "edge_attr", None)
        if old_attr is None and new_attr is None:
            continue
        if old_attr is None or new_attr is None or old_attr.shape != new_attr.shape:
            result[f"edge_{name}_shape_match"] = False
            continue
        delta = np.abs(
            new_attr.detach().cpu().numpy().astype(float)
            - old_attr.detach().cpu().numpy().astype(float)
        )
        result[f"edge_{name}_shape_match"] = True
        result[f"edge_{name}_mean_abs_shift"] = float(np.mean(delta)) if delta.size else 0.0
        result[f"edge_{name}_max_abs_shift"] = float(np.max(delta)) if delta.size else 0.0
        if delta.size:
            values.append(delta.reshape(-1))
    concatenated = np.concatenate(values) if values else np.zeros(1)
    result["overall_mean_abs_shift"] = float(np.mean(concatenated))
    result["overall_max_abs_shift"] = float(np.max(concatenated))
    return result


class GateAudit:
    def __init__(self, *, atol: float):
        self.category = "unclassified"
        self.atol = float(atol)
        self.counts: Counter[str] = Counter()
        self.mismatches: Counter[str] = Counter()
        self.maximum_formula_error = 0.0

    def set_category(self, category: str) -> None:
        self.category = str(category)

    def observe(self, kwargs: Mapping[str, Any], transition: Any) -> None:
        self.counts[self.category] += 1
        info = transition.controller_info
        semantics = str(info.get("forcing_gate_semantics"))
        expected = np.tanh(
            np.abs(np.asarray(info["goal_eff"], dtype=float) - np.asarray(kwargs["position"], dtype=float))
        )
        actual = np.asarray(info["forcing_gate"], dtype=float)
        error = float(np.max(np.abs(expected - actual)))
        self.maximum_formula_error = max(self.maximum_formula_error, error)
        if semantics != HISTORICAL_GATE_NAME or error > self.atol:
            self.mismatches[self.category] += 1

    def summary(self) -> dict[str, Any]:
        return {
            "counts": dict(self.counts),
            "mismatches": dict(self.mismatches),
            "maximum_formula_error": self.maximum_formula_error,
            "all_observed_calls_historical": sum(self.mismatches.values()) == 0,
        }


def _build_historical_graph(
    *,
    env: Any,
    agent_id: int,
    proposals: Sequence[Any],
    policy: Any,
    proposal_config: ProposalConfig,
    graph_config: HeterogeneousCandidateGraphConfig,
) -> tuple[Any, tuple[Any, ...]]:
    class_goals = (np.asarray(env.goals[agent_id], dtype=float).copy(),) + tuple(
        np.asarray(item.point, dtype=float).copy() for item in proposals
    )
    previews = _build_formal_previews(
        env=env,
        agent_index=agent_id,
        class_goals=class_goals,
        policy=policy,
        horizon=4,
    )
    executions = tuple(
        graph_ready_candidate_execution(candidate_id, previews[candidate_id + 1])
        for candidate_id in range(len(proposals))
    )
    graph = build_heterogeneous_candidate_graph_from_env(
        env=env,
        agent_index=agent_id,
        proposals=proposals,
        executions=executions,
        proposal_config=proposal_config,
        config=graph_config,
        execution_normalization=ExecutionNormalizationSpec(),
    )
    return graph, previews


def _source_indices(source_dir: Path) -> dict[str, Any]:
    state_rows = _read_csv(source_dir / "state_groups.csv")
    graph_rows = _read_csv(source_dir / "graph_records.csv")
    manifest_rows = [
        row for row in _read_csv(source_dir / "manifest.csv") if int(row["H_label"]) == 6
    ]
    class_rows = [
        row
        for row in _read_csv(source_dir / "candidate_class_records.csv")
        if int(row["H_label"]) == 6
    ]
    return {
        "state_rows": {row["state_group_id"]: row for row in state_rows},
        "graph_rows": {row["sample_id"]: row for row in graph_rows},
        "manifest_rows": {row["sample_id"]: row for row in manifest_rows},
        "class_rows": {
            (row["sample_id"], int(row["class_index"])): row for row in class_rows
        },
        "group_ids": {row["state_group_id"] for row in state_rows},
    }


def _iterate_source_states(
    *,
    config: Mapping[str, Any],
    policy: Any,
    multi_config: Any,
    source_index: Mapping[str, Any],
    callback: Callable[[Any, str, int, int], None],
) -> None:
    requested = set(source_index["group_ids"])
    visited: set[str] = set()
    for scenario in config["scenario_types"]:
        for seed in config["seeds"]:
            env, _ = _build_supervision_environment(
                config=multi_config,
                scene_type=str(scenario),
                seed=int(seed),
                peer_radius=float(config["peer_radius"]),
            )
            try:
                for timestep in range(max(config["state_sample_timesteps"]) + 1):
                    group_id = (
                        f"{scenario}__seed{int(seed):03d}__episode000__t{int(timestep):04d}"
                    )
                    if group_id in requested:
                        callback(env, group_id, int(seed), int(timestep))
                        visited.add(group_id)
                    if timestep >= max(config["state_sample_timesteps"]):
                        break
                    terminated, truncated, _ = _advance_environment_one_step(env, policy)
                    if terminated or truncated:
                        break
            finally:
                env.close()
    missing = sorted(requested - visited)
    if missing:
        raise RuntimeError(f"failed to reproduce {len(missing)} source states: {missing[:3]}")


def _candidate_check(
    *,
    source_dir: Path,
    source_index: Mapping[str, Any],
    sample_id: str,
    proposals: Sequence[Any],
    all_proposal_count: int,
    tolerance: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], bool]:
    label = _load_json(source_dir / "labels" / f"{sample_id}__H6.json")
    old_mapping = label["class_mapping"]
    old_graph_row = source_index["graph_rows"][sample_id]
    rows: list[dict[str, Any]] = []
    all_match = (
        int(old_graph_row["proposal_count_before_consumer"]) == int(all_proposal_count)
        and int(old_graph_row["K_actual"]) == len(proposals)
        and len(old_mapping) == len(proposals) + 1
    )
    rows.append(
        {
            "sample_id": sample_id,
            "class_index": 0,
            "class_kind": "null",
            "candidate_id_match": old_mapping[0]["candidate_id"] is None,
            "class_mapping_match": old_mapping[0]["class_index"] == 0,
            "position_max_abs_error": 0.0,
            "position_match": True,
            "sector_match": True,
            "score_match": True,
        }
    )
    for candidate_id, proposal in enumerate(proposals):
        class_index = candidate_id + 1
        old = old_mapping[class_index]
        old_row = source_index["class_rows"][(sample_id, class_index)]
        old_point = np.asarray(old["candidate_xyz"], dtype=float)
        new_point = np.asarray(proposal.point, dtype=float)
        error = float(np.max(np.abs(old_point - new_point)))
        position_match = bool(
            np.allclose(
                old_point,
                new_point,
                atol=float(tolerance["candidate_position_atol"]),
                rtol=float(tolerance["candidate_position_rtol"]),
            )
        )
        sector_match = (
            int(float(old_row["proposal_azimuth_index"])) == int(proposal.azimuth_index)
            and int(float(old_row["proposal_elevation_index"]))
            == int(proposal.elevation_index)
        )
        score_match = bool(
            np.isclose(
                float(old_row["proposal_score"]),
                float(proposal.score),
                atol=float(tolerance["float_atol"]),
                rtol=float(tolerance["float_rtol"]),
            )
        )
        identity_match = (
            old["class_index"] == class_index
            and old["candidate_id"] == candidate_id
            and old["proposal_node_index"] == candidate_id
        )
        match = position_match and sector_match and score_match and identity_match
        all_match &= match
        rows.append(
            {
                "sample_id": sample_id,
                "class_index": class_index,
                "class_kind": "proposal",
                "candidate_id": candidate_id,
                "candidate_id_match": identity_match,
                "class_mapping_match": identity_match,
                "candidate_position": new_point.tolist(),
                "position_max_abs_error": error,
                "position_match": position_match,
                "sector_match": sector_match,
                "score_match": score_match,
                "azimuth_index": int(proposal.azimuth_index),
                "elevation_index": int(proposal.elevation_index),
                "proposal_score": float(proposal.score),
                "proposal_metadata": _proposal_metadata(proposal),
            }
        )
    return rows, bool(all_match)


def _select_hash_subset(
    group_ids: Iterable[str], *, count: int, salt: str
) -> set[str]:
    ordered = sorted(
        set(group_ids),
        key=lambda value: hashlib.sha256(f"{salt}:{value}".encode()).hexdigest(),
    )
    return set(ordered[: int(count)])


def _scenario_subset(
    state_rows: Mapping[str, Mapping[str, Any]], *, per_scenario: int, salt: str
) -> set[str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for group_id, row in state_rows.items():
        grouped[str(row["scenario"])].append(group_id)
    selected: set[str] = set()
    for scenario, members in grouped.items():
        selected |= _select_hash_subset(
            members, count=int(per_scenario), salt=f"{salt}:{scenario}"
        )
    return selected


def _rankdata(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="mergesort")
    result = np.empty(len(array), dtype=float)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        result[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return result


def _spearman(first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) < 2:
        return 1.0
    x = _rankdata(first)
    y = _rankdata(second)
    if np.std(x) <= 1.0e-15 and np.std(y) <= 1.0e-15:
        return 1.0
    if np.std(x) <= 1.0e-15 or np.std(y) <= 1.0e-15:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _target_stats(values: Sequence[float], probabilities: Sequence[float]) -> dict[str, Any]:
    values_array = np.asarray(values, dtype=float)
    probabilities_array = np.asarray(probabilities, dtype=float)
    order = np.argsort(-values_array, kind="mergesort")
    entropy = float(
        -np.sum(
            probabilities_array
            * np.log(np.clip(probabilities_array, 1.0e-15, 1.0))
        )
    )
    return {
        "top1": int(order[0]),
        "top3": order[: min(3, len(order))].astype(int).tolist(),
        "entropy": entropy,
        "effective_class_count": float(np.exp(entropy)),
        "null_top1": int(order[0]) == 0,
        "proposal_top1": int(order[0]) != 0,
        "top1_top2_margin": (
            float(values_array[order[0]] - values_array[order[1]])
            if len(order) > 1
            else None
        ),
    }


def _agreement_row(
    *,
    sample_id: str,
    scenario: str,
    seed: int,
    effective_split: str,
    scalar_values: Sequence[float],
    scalar_probabilities: Sequence[float],
    historical_values: Sequence[float],
    historical_probabilities: Sequence[float],
    v2_target: TieredTarget,
) -> dict[str, Any]:
    scalar = _target_stats(scalar_values, scalar_probabilities)
    historical = _target_stats(historical_values, historical_probabilities)
    v2 = _target_stats(v2_target.utilities, v2_target.probabilities)
    return {
        "sample_id": sample_id,
        "scenario": scenario,
        "seed": int(seed),
        "effective_split": effective_split,
        "class_count": len(v2_target.utilities),
        "v1_scalar_top1": scalar["top1"],
        "v1_historical_top1": historical["top1"],
        "v2_top1": v2["top1"],
        "v1_scalar_historical_top1_agreement": scalar["top1"] == historical["top1"],
        "v1_historical_v2_top1_agreement": historical["top1"] == v2["top1"],
        "v1_scalar_historical_top3_overlap": len(set(scalar["top3"]) & set(historical["top3"])) / len(scalar["top3"]),
        "v1_historical_v2_top3_overlap": len(set(historical["top3"]) & set(v2["top3"])) / len(historical["top3"]),
        "v1_scalar_historical_pairwise_agreement": pairwise_ordering_agreement(scalar_values, historical_values),
        "v1_historical_v2_pairwise_agreement": pairwise_ordering_agreement(historical_values, v2_target.utilities),
        "v1_scalar_entropy": scalar["entropy"],
        "v1_historical_entropy": historical["entropy"],
        "v2_entropy": v2["entropy"],
        "v1_scalar_effective_class_count": scalar["effective_class_count"],
        "v1_historical_effective_class_count": historical["effective_class_count"],
        "v2_effective_class_count": v2["effective_class_count"],
        "v1_scalar_null_top1": scalar["null_top1"],
        "v1_historical_null_top1": historical["null_top1"],
        "v2_null_top1": v2["null_top1"],
        "v1_scalar_top1_top2_margin": scalar["top1_top2_margin"],
        "v1_historical_top1_top2_margin": historical["top1_top2_margin"],
        "v2_top1_top2_margin": v2["top1_top2_margin"],
    }


def _compare_branch(first: BranchRollout, second: BranchRollout, tolerance: Mapping[str, Any]) -> dict[str, Any]:
    discrete_fields = (
        "termination_type",
        "ego_reference_reached",
        "reference_reach_step",
        "ego_terminal_reached",
        "collision_step",
        "final_absolute_env_steps",
    )
    discrete_match = all(first.outcome[key] == second.outcome[key] for key in discrete_fields)
    arrays = ("actions", "positions", "velocities", "phases", "dmp_goals")
    errors = {}
    shapes_match = True
    for name in arrays:
        left = np.asarray(getattr(first, name), dtype=float)
        right = np.asarray(getattr(second, name), dtype=float)
        if left.shape != right.shape:
            shapes_match = False
            errors[name] = float("inf")
        else:
            errors[name] = float(np.max(np.abs(left - right))) if left.size else 0.0
    numeric_match = all(
        value <= float(tolerance["trajectory_atol"]) for value in errors.values()
    )
    return {
        "trajectory_hash_match": first.trajectory_hash == second.trajectory_hash,
        "discrete_match": discrete_match,
        "shapes_match": shapes_match,
        "numeric_match": numeric_match,
        "maximum_absolute_error": max(errors.values()),
        "per_field_max_absolute_error": errors,
        "match": bool(discrete_match and shapes_match and numeric_match),
    }


def _report(conclusion: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage-I GAT Supervision Target Redesign V2",
            "",
            "This dataset-generation audit used the original Stage-I state distribution,",
            "historical-vector-gate H4 graph inputs, and absolute-episode-horizon branches.",
            "No GAT/SAC training or formal closed-loop/stress benchmark was run.",
            "",
            "## Validity gate",
            "",
            f"- V2_SUPERVISION_VALID = `{conclusion['V2_SUPERVISION_VALID']}`",
            f"- RECOMMENDED_NEXT_STEP = `{conclusion['RECOMMENDED_NEXT_STEP']}`",
            f"- TRAINING_DEPLOYMENT_GATE_ALIGNMENT = `{conclusion['TRAINING_DEPLOYMENT_GATE_ALIGNMENT']}`",
            f"- CANDIDATE_IDENTITY_MATCH = `{conclusion['CANDIDATE_IDENTITY_MATCH']}`",
            f"- GRAPH_SCHEMA_MATCH = `{conclusion['GRAPH_SCHEMA_MATCH']}`",
            f"- BRANCH_ROLLOUT_DETERMINISM = `{conclusion['BRANCH_ROLLOUT_DETERMINISM']}`",
            f"- V2_LABEL_INFORMATION_DENSITY = `{conclusion['V2_LABEL_INFORMATION_DENSITY']}`",
            f"- CANDIDATE_ATTRIBUTION_STABILITY = `{conclusion['CANDIDATE_ATTRIBUTION_STABILITY']}`",
            "",
            "## Target comparison",
            "",
            f"- V1 scalar vs historical Top-1 agreement = {conclusion['V1_SCALAR_HISTORICAL_TOP1_AGREEMENT']:.4f}",
            f"- V1 historical vs V2 Top-1 agreement = {conclusion['V1_HISTORICAL_V2_TOP1_AGREEMENT']:.4f}",
            f"- POST_REFERENCE_SIGNAL_PRESENT = `{conclusion['POST_REFERENCE_SIGNAL_PRESENT']}`",
            f"- STRESS_SET_USED_FOR_TARGET_TUNING = `{conclusion['STRESS_SET_USED_FOR_TARGET_TUNING']}`",
            "",
            "Generation stopped after dataset construction and validity analysis.",
        ]
    )


def run_generation(config: Mapping[str, Any], output_dir: Path) -> Path:
    _assert_config(config)
    source_dir = REPO_ROOT / str(config["source_dataset_dir"])
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    checkpoint = REPO_ROOT / str(config["checkpoint"])
    if _sha256_file(checkpoint) != config["checkpoint_sha256"]:
        raise RuntimeError("frozen SAC checkpoint hash mismatch")
    output_dir.mkdir(parents=True, exist_ok=False)
    config_path = DEFAULT_CONFIG_PATH.resolve()
    evaluator_path = Path(__file__).resolve()
    config_hash_before = _sha256_file(config_path)
    evaluator_hash_before = _sha256_file(evaluator_path)
    protected_paths = _protected_paths(config)
    protected_before = {name: _sha256_file(path) for name, path in protected_paths.items()}
    _write_json(output_dir / "config.json", config)
    semantic_contract = {
        "schema_version": SCHEMA_VERSION,
        "source_state": "deterministic_V1_scalar_replay_provenance_only",
        "source_state_post_recovery_gate": HISTORICAL_GATE_NAME,
        "absolute_episode_horizon": 220,
        "remaining_budget": "max(0,220-source_env_steps)",
        "fixed_future_220_diagnostic_run": False,
        "formal_background_selector": BACKGROUND_FP_SHEP,
        "attribution_background_selector": BACKGROUND_NULL,
        "one_shot_handoff": True,
        "high_level_replanning": False,
        "terminal_goal_storage": "env.goals",
        "temporary_reference_storage": "dmp.goal",
        "phase_reset_on_handoff": False,
        "GAT_in_label_generation": False,
        "stress_artifact_read": False,
    }
    _write_json(output_dir / "semantic_contract.json", semantic_contract)

    source_index = _source_indices(source_dir)
    source_config = _load_json(source_dir / "config.json")
    proposal_config = ProposalConfig(**config["proposal_config"])
    graph_config = HeterogeneousCandidateGraphConfig(**config["graph_config"])
    score_spec = FPSHEPOnlineScoreSpec.from_mapping(config["fp_shep_online_selector"])
    multi_config = build_single_distribution_multi_config(
        num_agents=int(config["num_agents"]),
        max_steps=int(config["max_absolute_episode_steps"]),
    )
    reference_env = build_single_env(
        config=SINGLE_AGENT_CONFIG, action_guidance_enabled=False
    )
    policy = build_single_model(reference_env, config=SINGLE_AGENT_CONFIG, verbose=0)
    load_checkpoint(policy, checkpoint)
    freeze_policy(policy)
    parameter_hash_before = _policy_parameter_sha256(policy)
    started = time.perf_counter()

    candidate_rows: list[dict[str, Any]] = []
    graph_audit_rows: list[dict[str, Any]] = []
    graph_dataset: dict[str, Any] = {}
    stage_a_failures: list[str] = []
    gate_audit = GateAudit(atol=float(config["reproduction_tolerance"]["float_atol"]))

    def stage_a(env: Any, group_id: str, seed: int, timestep: int) -> None:
        expected_state = source_index["state_rows"][group_id]
        actual_fingerprint = environment_state_fingerprint(env)
        state_match = actual_fingerprint == expected_state["environment_fingerprint"]
        if not state_match:
            stage_a_failures.append(f"source_state:{group_id}")
        proposals_by_agent: list[tuple[Any, ...]] = []
        for agent_id in config["agent_indices"]:
            all_proposals = _generate_proposals(env, int(agent_id), proposal_config)
            selected = tuple(
                adapt_candidate_proposals(
                    all_proposals, consumer_top_k=int(config["K_requested"])
                )
            )
            proposals_by_agent.append(selected)
            sample_id = f"{group_id}__ego{int(agent_id)}"
            rows, candidates_match = _candidate_check(
                source_dir=source_dir,
                source_index=source_index,
                sample_id=sample_id,
                proposals=selected,
                all_proposal_count=len(all_proposals),
                tolerance=config["reproduction_tolerance"],
            )
            for row in rows:
                row.update(
                    {
                        "state_group_id": group_id,
                        "scenario": expected_state["scenario"],
                        "seed": seed,
                        "sample_t": timestep,
                        "ego_id": int(agent_id),
                        "source_state_match": state_match,
                    }
                )
            candidate_rows.extend(rows)
            if not candidates_match:
                stage_a_failures.append(f"candidate:{sample_id}")
            source_before = environment_state_fingerprint(env)
            gate_audit.set_category("graph_h4_preview")
            with scoped_historical_preview_and_multi_agent_transition(
                preview_observer=gate_audit.observe,
                execution_observer=gate_audit.observe,
            ):
                graph, previews = _build_historical_graph(
                    env=env,
                    agent_id=int(agent_id),
                    proposals=selected,
                    policy=policy,
                    proposal_config=proposal_config,
                    graph_config=graph_config,
                )
            source_after = environment_state_fingerprint(env)
            old_graph_path = source_dir / source_index["graph_rows"][sample_id]["graph_path"]
            old_graph = torch.load(old_graph_path, map_location="cpu", weights_only=False)
            old_schema = _graph_schema(old_graph)
            new_schema = _graph_schema(graph)
            schema_match = old_schema == new_schema
            proposal_count_match = int(graph["proposal"].num_nodes) == len(selected)
            graph_state_unchanged = source_before == source_after
            if not (schema_match and proposal_count_match and graph_state_unchanged):
                stage_a_failures.append(f"graph:{sample_id}")
            shift = _graph_shift(old_graph, graph)
            graph_audit_rows.append(
                {
                    "sample_id": sample_id,
                    "state_group_id": group_id,
                    "scenario": expected_state["scenario"],
                    "seed": seed,
                    "sample_t": timestep,
                    "ego_id": int(agent_id),
                    "source_state_fingerprint_match": state_match,
                    "old_vs_v2_graph_schema_match": schema_match,
                    "proposal_count_match": proposal_count_match,
                    "source_environment_unchanged": graph_state_unchanged,
                    "old_schema": old_schema,
                    "v2_schema": new_schema,
                    "H4_feature_shift": shift,
                    "historical_h4_preview_count": len(previews),
                }
            )
            graph_dataset[sample_id] = {
                "graph": graph,
                "state_group_id": group_id,
                "scenario": expected_state["scenario"],
                "seed": seed,
                "sample_t": timestep,
                "ego_id": int(agent_id),
                "source_split": expected_state["split"],
                "effective_split": _split(seed, config["effective_split"]),
            }
        print(
            f"[Stage A {len(graph_dataset) // 3}/164] {group_id}: "
            f"graphs={len(graph_dataset)} failures={len(stage_a_failures)}",
            flush=True,
        )

    try:
        _iterate_source_states(
            config=config,
            policy=policy,
            multi_config=multi_config,
            source_index=source_index,
            callback=stage_a,
        )
        _write_csv(output_dir / "candidate_reproduction.csv", candidate_rows)
        _write_csv(output_dir / "historical_h4_graph_audit.csv", graph_audit_rows)
        gate_stage_a = gate_audit.summary()
        candidate_identity_match = not any(
            item.startswith("candidate:") for item in stage_a_failures
        )
        source_state_match = not any(
            item.startswith("source_state:") for item in stage_a_failures
        )
        graph_schema_match = not any(
            item.startswith("graph:") for item in stage_a_failures
        )
        stage_a_gate_match = (
            gate_stage_a["all_observed_calls_historical"]
            and gate_stage_a["counts"].get("graph_h4_preview", 0) > 0
        )
        if not (
            source_state_match
            and candidate_identity_match
            and graph_schema_match
            and stage_a_gate_match
        ):
            _write_json(
                output_dir / "validity_gate.json",
                {
                    "stage": "SOURCE_AND_GRAPH_REPRODUCTION",
                    "passed": False,
                    "SOURCE_STATE_REPRODUCTION": source_state_match,
                    "CANDIDATE_IDENTITY_MATCH": candidate_identity_match,
                    "GRAPH_SCHEMA_MATCH": graph_schema_match,
                    "GRAPH_PREVIEW_GATE_ALIGNMENT": stage_a_gate_match,
                    "failures": stage_a_failures,
                    "gate_audit": gate_stage_a,
                },
            )
            raise RuntimeError("Stage A source/candidate/graph gate failed")
        _write_json(
            output_dir / "generation_progress.json",
            {
                "stage": "LONG_HORIZON_BRANCHES",
                "stage_a_state_groups_completed": len(source_index["group_ids"]),
                "stage_a_graphs_completed": len(graph_dataset),
                "stage_a_passed": True,
                "stage_b_state_groups_completed": 0,
                "stage_b_graphs_completed": 0,
                "formal_branch_count_completed": 0,
            },
        )

        determinism_groups = _select_hash_subset(
            source_index["group_ids"],
            count=int(config["determinism_audit"]["state_group_count"]),
            salt="gat_supervision_v2_determinism",
        )
        attribution_groups = _scenario_subset(
            source_index["state_rows"],
            per_scenario=int(
                config["attribution_stability_audit"]["state_groups_per_scenario"]
            ),
            salt="gat_supervision_v2_attribution",
        )
        raw_outcomes: list[dict[str, Any]] = []
        branch_manifest_rows: list[dict[str, Any]] = []
        v1_scalar_rows: list[dict[str, Any]] = []
        v1_historical_rows: list[dict[str, Any]] = []
        v2_target_rows: list[dict[str, Any]] = []
        agreement_rows: list[dict[str, Any]] = []
        v2_targets: list[TieredTarget] = []
        targets_by_sample: dict[str, TieredTarget] = {}
        attribution_rows: list[dict[str, Any]] = []
        determinism_rows: list[dict[str, Any]] = []
        branch_counter = 0

        old_quality = source_config["quality"]
        quality_spec = None
        # Reuse the exact V1 quality constructor through its public config path.
        from planning.candidate_supervision import CandidateQualitySpec

        quality_spec = CandidateQualitySpec.from_mapping(old_quality)
        tier_config = config["tier_definition"]

        def stage_b(env: Any, group_id: str, seed: int, timestep: int) -> None:
            nonlocal branch_counter
            expected_state = source_index["state_rows"][group_id]
            if environment_state_fingerprint(env) != expected_state["environment_fingerprint"]:
                raise RuntimeError(f"Stage B source-state reproduction mismatch: {group_id}")
            proposals_by_agent: list[tuple[Any, ...]] = []
            for agent_id in config["agent_indices"]:
                proposals_by_agent.append(
                    tuple(
                        adapt_candidate_proposals(
                            _generate_proposals(env, int(agent_id), proposal_config),
                            consumer_top_k=int(config["K_requested"]),
                        )
                    )
                )
            with scoped_historical_preview_and_multi_agent_transition(
                preview_observer=gate_audit.observe,
                execution_observer=gate_audit.observe,
            ):
                gate_audit.set_category("non_ego_background_selection")
                formal_background = build_background_plan(
                    env=env,
                    proposals_by_agent=proposals_by_agent,
                    policy=policy,
                    score_spec=score_spec,
                    selector=BACKGROUND_FP_SHEP,
                )
                null_background = None
                if group_id in attribution_groups:
                    null_background = build_background_plan(
                        env=env,
                        proposals_by_agent=proposals_by_agent,
                        policy=policy,
                        score_spec=score_spec,
                        selector=BACKGROUND_NULL,
                    )
                for agent_id in config["agent_indices"]:
                    agent_id = int(agent_id)
                    sample_id = f"{group_id}__ego{agent_id}"
                    proposals = proposals_by_agent[agent_id]
                    class_goals: tuple[np.ndarray, ...] = (
                        np.asarray(env.goals[agent_id], dtype=float).copy(),
                    ) + tuple(np.asarray(item.point, dtype=float).copy() for item in proposals)
                    label = _load_json(source_dir / "labels" / f"{sample_id}__H6.json")
                    scalar_values = np.asarray(label["provisional_primary_target"], dtype=float)
                    scalar_probabilities = np.asarray(
                        label["soft_targets"]["tau_0.25"]["provisional_primary_target"],
                        dtype=float,
                    )
                    if len(scalar_values) != len(class_goals):
                        raise RuntimeError(f"V1 scalar class count mismatch: {sample_id}")

                    gate_audit.set_category("v1_historical_h6")
                    h6_rollouts = tuple(
                        real_candidate_rollout(
                            initial_env=env,
                            agent_index=agent_id,
                            candidate_goal=goal,
                            policy=policy,
                            horizon=6,
                        )
                        for goal in class_goals
                    )
                    h6_failure = np.asarray(
                        [branch_is_failure(item) for item in h6_rollouts], dtype=bool
                    )
                    h6_quality = compute_candidate_quality(
                        [_quality_row(item) for item in h6_rollouts],
                        failure_mask=h6_failure,
                        spec=quality_spec,
                    )
                    h6_targets = target_bundle(h6_quality, [0.25])
                    historical_values = np.asarray(
                        h6_targets["provisional_primary_target"], dtype=float
                    )
                    historical_probabilities = np.asarray(
                        h6_targets["soft_targets"]["tau_0.25"][
                            "provisional_primary_target"
                        ],
                        dtype=float,
                    )

                    formal_rollouts: list[BranchRollout] = []
                    formal_outcomes: list[dict[str, Any]] = []
                    non_ego_plan_hash = stable_hash(
                        {
                            "ego_id": agent_id,
                            "references": [
                                formal_background.references[index]
                                for index in range(int(env.num_agents))
                                if index != agent_id
                            ],
                            "available": [
                                bool(formal_background.available[index])
                                for index in range(int(env.num_agents))
                                if index != agent_id
                            ],
                            "candidate_ids": [
                                formal_background.selected_candidate_ids[index]
                                for index in range(int(env.num_agents))
                                if index != agent_id
                            ],
                        }
                    )
                    gate_audit.set_category("v2_long_horizon")
                    for class_index, goal in enumerate(class_goals):
                        rollout = run_long_horizon_branch(
                            initial_env=env,
                            ego_agent_id=agent_id,
                            ego_candidate_goal=None if class_index == 0 else goal,
                            background_plan=formal_background,
                            policy=policy,
                            max_absolute_episode_steps=int(
                                config["max_absolute_episode_steps"]
                            ),
                            reference_reached_tolerance_m=float(
                                config["reference_reached_tolerance_m"]
                            ),
                            class_index=class_index,
                            candidate_id=None if class_index == 0 else class_index - 1,
                        )
                        outcome = dict(rollout.outcome)
                        outcome.update(
                            {
                                "state_group_id": group_id,
                                "sample_id": sample_id,
                                "scenario_id": expected_state["scenario"],
                                "seed": seed,
                                "sample_t": timestep,
                                "source_split": expected_state["split"],
                                "effective_split": _split(seed, config["effective_split"]),
                                "candidate_geometry": (
                                    np.asarray(goal, dtype=float).tolist()
                                ),
                                "candidate_metadata": (
                                    None
                                    if class_index == 0
                                    else _proposal_metadata(proposals[class_index - 1])
                                ),
                                "non_ego_plan_hash": non_ego_plan_hash,
                                "rollout_semantics": HISTORICAL_GATE_NAME,
                                "runtime_ms": rollout.runtime_ms,
                            }
                        )
                        raw_outcomes.append(outcome)
                        formal_outcomes.append(outcome)
                        formal_rollouts.append(rollout)
                        branch_manifest_rows.append(
                            {
                                "state_group_id": group_id,
                                "sample_id": sample_id,
                                "ego_id": agent_id,
                                "class_index": class_index,
                                "candidate_id": outcome["candidate_id"],
                                "background_selector": BACKGROUND_FP_SHEP,
                                "background_plan_hash": formal_background.plan_hash,
                                "non_ego_plan_hash": non_ego_plan_hash,
                                "source_state_fingerprint": outcome[
                                    "source_state_fingerprint"
                                ],
                                "trajectory_hash": rollout.trajectory_hash,
                                "source_env_steps": timestep,
                                "remaining_budget": outcome[
                                    "remaining_budget_at_source"
                                ],
                                "effective_transition_count": outcome[
                                    "effective_transition_count"
                                ],
                                "final_absolute_env_steps": outcome[
                                    "final_absolute_env_steps"
                                ],
                                "termination_type": outcome["termination_type"],
                                "forcing_gate": HISTORICAL_GATE_NAME,
                            }
                        )
                        branch_counter += 1

                    v2_target = build_tiered_target(
                        formal_outcomes,
                        progress_epsilon=float(
                            tier_config["positive_terminal_progress_epsilon_m"]
                        ),
                        within_tier_scale=float(tier_config["within_tier_scale"]),
                        round_decimals=int(tier_config["float_comparison_round_decimals"]),
                        temperature=float(config["soft_target_temperature"]),
                    )
                    v2_targets.append(v2_target)
                    targets_by_sample[sample_id] = v2_target
                    graph_dataset[sample_id].update(
                        {
                            "class_mapping": label["class_mapping"],
                            "v1_scalar_h6_utility": scalar_values,
                            "v1_scalar_h6_soft_target": scalar_probabilities,
                            "v1_historical_h6_utility": historical_values,
                            "v1_historical_h6_soft_target": historical_probabilities,
                            "v2_long_horizon_tier": v2_target.tiers,
                            "v2_long_horizon_utility": v2_target.utilities,
                            "v2_long_horizon_soft_target": v2_target.probabilities,
                            "v2_hard_target": v2_target.hard_target,
                            "background_selector": BACKGROUND_FP_SHEP,
                            "background_plan_hash": formal_background.plan_hash,
                            "non_ego_plan_hash": non_ego_plan_hash,
                        }
                    )
                    for class_index in range(len(class_goals)):
                        common = {
                            "state_group_id": group_id,
                            "sample_id": sample_id,
                            "scenario": expected_state["scenario"],
                            "seed": seed,
                            "sample_t": timestep,
                            "ego_id": agent_id,
                            "class_index": class_index,
                            "candidate_id": None if class_index == 0 else class_index - 1,
                            "source_split": expected_state["split"],
                            "effective_split": _split(seed, config["effective_split"]),
                        }
                        v1_scalar_rows.append(
                            {
                                **common,
                                "utility": float(scalar_values[class_index]),
                                "soft_target_tau_0p25": float(
                                    scalar_probabilities[class_index]
                                ),
                                "gate": "scalar_terminal_distance_gate_V1_baseline_only",
                            }
                        )
                        v1_historical_rows.append(
                            {
                                **common,
                                "utility": float(historical_values[class_index]),
                                "soft_target_tau_0p25": float(
                                    historical_probabilities[class_index]
                                ),
                                "gate": HISTORICAL_GATE_NAME,
                            }
                        )
                        v2_target_rows.append(
                            {
                                **common,
                                "outcome_tier": int(v2_target.tiers[class_index]),
                                "tie_break_key": v2_target.tie_break_keys[class_index],
                                "within_tier_utility": float(
                                    v2_target.utilities[class_index]
                                    - v2_target.tiers[class_index]
                                ),
                                "final_utility": float(v2_target.utilities[class_index]),
                                "rank_zero_based": int(
                                    v2_target.ranks_zero_based[class_index]
                                ),
                                "soft_target_tau_0p25": float(
                                    v2_target.probabilities[class_index]
                                ),
                                "hard_target": class_index == v2_target.hard_target,
                                "top1_tied": v2_target.top1_tied,
                            }
                        )
                    agreement_rows.append(
                        _agreement_row(
                            sample_id=sample_id,
                            scenario=str(expected_state["scenario"]),
                            seed=seed,
                            effective_split=_split(seed, config["effective_split"]),
                            scalar_values=scalar_values,
                            scalar_probabilities=scalar_probabilities,
                            historical_values=historical_values,
                            historical_probabilities=historical_probabilities,
                            v2_target=v2_target,
                        )
                    )

                    if group_id in determinism_groups:
                        class_count = len(class_goals)
                        requested_count = min(
                            int(config["determinism_audit"]["classes_per_ego"]),
                            class_count,
                        )
                        class_ids = sorted(
                            set(
                                np.linspace(
                                    0, class_count - 1, requested_count, dtype=int
                                ).tolist()
                            )
                        )
                        gate_audit.set_category("determinism_repeat")
                        for class_index in class_ids:
                            repeated = run_long_horizon_branch(
                                initial_env=env,
                                ego_agent_id=agent_id,
                                ego_candidate_goal=(
                                    None if class_index == 0 else class_goals[class_index]
                                ),
                                background_plan=formal_background,
                                policy=policy,
                                max_absolute_episode_steps=int(
                                    config["max_absolute_episode_steps"]
                                ),
                                reference_reached_tolerance_m=float(
                                    config["reference_reached_tolerance_m"]
                                ),
                                class_index=class_index,
                                candidate_id=(
                                    None if class_index == 0 else class_index - 1
                                ),
                            )
                            comparison = _compare_branch(
                                formal_rollouts[class_index],
                                repeated,
                                config["determinism_audit"],
                            )
                            determinism_rows.append(
                                {
                                    "state_group_id": group_id,
                                    "sample_id": sample_id,
                                    "ego_id": agent_id,
                                    "class_index": class_index,
                                    **comparison,
                                }
                            )
                            if not comparison["match"]:
                                raise RuntimeError(
                                    f"branch determinism failed: {sample_id}:{class_index}"
                                )

                    if group_id in attribution_groups and null_background is not None:
                        gate_audit.set_category("attribution_null_background")
                        alternative_outcomes = []
                        for class_index, goal in enumerate(class_goals):
                            rollout = run_long_horizon_branch(
                                initial_env=env,
                                ego_agent_id=agent_id,
                                ego_candidate_goal=None if class_index == 0 else goal,
                                background_plan=null_background,
                                policy=policy,
                                max_absolute_episode_steps=int(
                                    config["max_absolute_episode_steps"]
                                ),
                                reference_reached_tolerance_m=float(
                                    config["reference_reached_tolerance_m"]
                                ),
                                class_index=class_index,
                                candidate_id=None if class_index == 0 else class_index - 1,
                            )
                            alternative_outcomes.append(rollout.outcome)
                        alternative = build_tiered_target(
                            alternative_outcomes,
                            progress_epsilon=float(
                                tier_config["positive_terminal_progress_epsilon_m"]
                            ),
                            within_tier_scale=float(tier_config["within_tier_scale"]),
                            round_decimals=int(
                                tier_config["float_comparison_round_decimals"]
                            ),
                            temperature=float(config["soft_target_temperature"]),
                        )
                        attribution_rows.append(
                            {
                                "state_group_id": group_id,
                                "sample_id": sample_id,
                                "scenario": expected_state["scenario"],
                                "seed": seed,
                                "ego_id": agent_id,
                                "class_count": len(class_goals),
                                "formal_background": BACKGROUND_FP_SHEP,
                                "alternative_background": BACKGROUND_NULL,
                                "formal_top1": v2_target.hard_target,
                                "alternative_top1": alternative.hard_target,
                                "top1_agreement": v2_target.hard_target
                                == alternative.hard_target,
                                "spearman": _spearman(
                                    v2_target.utilities, alternative.utilities
                                ),
                                "pairwise_agreement": pairwise_ordering_agreement(
                                    v2_target.utilities, alternative.utilities
                                ),
                                "tier_agreement": float(
                                    np.mean(v2_target.tiers == alternative.tiers)
                                ),
                                "formal_tiers": v2_target.tiers.tolist(),
                                "alternative_tiers": alternative.tiers.tolist(),
                                "formal_utilities": v2_target.utilities.tolist(),
                                "alternative_utilities": alternative.utilities.tolist(),
                            }
                        )
            print(
                f"[V2 {len(targets_by_sample)}/492] {group_id}: branches={branch_counter}",
                flush=True,
            )
            _write_json(
                output_dir / "generation_progress.json",
                {
                    "stage": "LONG_HORIZON_BRANCHES",
                    "stage_a_state_groups_completed": len(source_index["group_ids"]),
                    "stage_a_graphs_completed": len(graph_dataset),
                    "stage_a_passed": True,
                    "stage_b_state_groups_completed": len(targets_by_sample) // 3,
                    "stage_b_graphs_completed": len(targets_by_sample),
                    "formal_branch_count_completed": branch_counter,
                    "latest_state_group_id": group_id,
                },
            )

        _iterate_source_states(
            config=config,
            policy=policy,
            multi_config=multi_config,
            source_index=source_index,
            callback=stage_b,
        )

        gate_summary = gate_audit.summary()
        required_gate_categories = {
            "graph_h4_preview",
            "non_ego_background_selection",
            "v1_historical_h6",
            "v2_long_horizon",
            "determinism_repeat",
            "attribution_null_background",
        }
        gate_alignment = bool(
            gate_summary["all_observed_calls_historical"]
            and all(gate_summary["counts"].get(name, 0) > 0 for name in required_gate_categories)
        )
        branch_determinism = bool(
            determinism_rows
            and len({row["state_group_id"] for row in determinism_rows})
            >= int(config["determinism_audit"]["state_group_count"])
            and all(bool(row["match"]) for row in determinism_rows)
        )
        density_overall = information_density(
            v2_targets, config["information_density_gate"]
        )
        density_by_split = {}
        for split_name in ("train", "validation", "test"):
            subset = [
                targets_by_sample[row["sample_id"]]
                for row in agreement_rows
                if row["effective_split"] == split_name
            ]
            density_by_split[split_name] = information_density(
                subset, config["information_density_gate"]
            )
        density_by_scenario = {}
        for scenario in config["scenario_types"]:
            subset = [
                targets_by_sample[row["sample_id"]]
                for row in agreement_rows
                if row["scenario"] == scenario
            ]
            density_by_scenario[scenario] = information_density(
                subset, config["information_density_gate"]
            )
        density_payload = {
            "schema_version": SCHEMA_VERSION,
            "thresholds_frozen_before_results": config["information_density_gate"],
            "overall": density_overall,
            "by_effective_split": density_by_split,
            "by_scenario": density_by_scenario,
        }

        attribution_metrics = {
            "top1_agreement": float(
                np.mean([row["top1_agreement"] for row in attribution_rows])
            ),
            "spearman": float(np.mean([row["spearman"] for row in attribution_rows])),
            "pairwise_agreement": float(
                np.mean([row["pairwise_agreement"] for row in attribution_rows])
            ),
            "tier_agreement": float(
                np.mean([row["tier_agreement"] for row in attribution_rows])
            ),
        }
        attribution_classification = classify_attribution_stability(
            attribution_metrics, config["attribution_stability_audit"]
        )
        for row in attribution_rows:
            row["overall_classification"] = attribution_classification

        _write_csv(output_dir / "branch_rollout_manifest.csv", branch_manifest_rows)
        table = pa.Table.from_pylist([_parquet_row(row) for row in raw_outcomes])
        pq.write_table(table, output_dir / "branch_raw_outcomes.parquet", compression="zstd")
        _write_csv(output_dir / "target_v1_scalar_h6.csv", v1_scalar_rows)
        _write_csv(output_dir / "target_v1_historical_h6.csv", v1_historical_rows)
        _write_csv(output_dir / "target_v2_long_horizon.csv", v2_target_rows)
        _write_csv(output_dir / "target_agreement.csv", agreement_rows)
        _write_json(output_dir / "label_information_density.json", density_payload)
        _write_csv(output_dir / "attribution_stability.csv", attribution_rows)
        _write_csv(output_dir / "branch_determinism.csv", determinism_rows)

        dataset_payload = {
            "schema_version": SCHEMA_VERSION,
            "effective_split": config["effective_split"],
            "source_split": config["source_split"],
            "H_preview": 4,
            "max_absolute_episode_steps": 220,
            "soft_target_temperature": 0.25,
            "samples": [graph_dataset[key] for key in sorted(graph_dataset)],
        }
        torch.save(dataset_payload, output_dir / "supervision_v2_dataset.pt")
        post_reference_rows = [
            row
            for row in raw_outcomes
            if bool(row["ego_reference_applicable"])
            and bool(row["ego_reference_reached"])
            and not bool(row["ego_terminal_reached"])
        ]
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "state_group_count": len(source_index["group_ids"]),
            "ego_graph_count": len(graph_dataset),
            "branch_count": len(raw_outcomes),
            "source_split_graph_counts": dict(
                Counter(row["source_split"] for row in graph_dataset.values())
            ),
            "effective_split_graph_counts": dict(
                Counter(row["effective_split"] for row in graph_dataset.values())
            ),
            "candidate_count_distribution": dict(
                sorted(
                    Counter(
                        len(item["class_mapping"]) - 1
                        for item in graph_dataset.values()
                    ).items()
                )
            ),
            "forcing_gate": HISTORICAL_GATE_NAME,
            "absolute_episode_horizon": 220,
            "fixed_future_220_sensitivity_run": False,
            "formal_background_selector": BACKGROUND_FP_SHEP,
            "attribution_background_selector": BACKGROUND_NULL,
            "tier_definition": config["tier_definition"],
            "tau": 0.25,
            "hash_provenance": protected_before,
        }
        _write_json(output_dir / "supervision_v2_metadata.json", metadata)

        scalar_historical_agreement = float(
            np.mean(
                [row["v1_scalar_historical_top1_agreement"] for row in agreement_rows]
            )
        )
        historical_v2_agreement = float(
            np.mean([row["v1_historical_v2_top1_agreement"] for row in agreement_rows])
        )
        post_reference_signal = bool(post_reference_rows)
        density_valid = density_overall["classification"] in {"STRONG", "ADEQUATE"}
        attribution_valid = attribution_classification != "WEAK"
        no_stress_tuning = True
        valid = all(
            (
                gate_alignment,
                source_state_match,
                candidate_identity_match,
                graph_schema_match,
                branch_determinism,
                density_valid,
                attribution_valid,
                no_stress_tuning,
            )
        )
        if valid:
            next_step = "TRAIN_STAGE1_GAT_WITH_V2"
        elif not (
            gate_alignment
            and source_state_match
            and candidate_identity_match
            and graph_schema_match
            and branch_determinism
        ):
            next_step = "NOT_ESTABLISHED"
        elif not density_valid or not attribution_valid:
            next_step = "REDESIGN_V2_TARGET"
        else:
            next_step = "NOT_ESTABLISHED"
        conclusion = {
            "schema_version": SCHEMA_VERSION,
            "TRAINING_DEPLOYMENT_GATE_ALIGNMENT": "YES" if gate_alignment else "NO",
            "SOURCE_STATE_REPRODUCTION": "YES" if source_state_match else "NO",
            "CANDIDATE_IDENTITY_MATCH": "YES" if candidate_identity_match else "NO",
            "GRAPH_SCHEMA_MATCH": "YES" if graph_schema_match else "NO",
            "BRANCH_ROLLOUT_DETERMINISM": "YES" if branch_determinism else "NO",
            "V1_SCALAR_HISTORICAL_TOP1_AGREEMENT": scalar_historical_agreement,
            "V1_HISTORICAL_V2_TOP1_AGREEMENT": historical_v2_agreement,
            "V2_LABEL_INFORMATION_DENSITY": density_overall["classification"],
            "CANDIDATE_ATTRIBUTION_STABILITY": attribution_classification,
            "POST_REFERENCE_SIGNAL_PRESENT": "YES" if post_reference_signal else "NO",
            "post_reference_incomplete_branch_count": len(post_reference_rows),
            "post_reference_obstacle_collision_count": sum(
                bool(row["post_reference_obstacle_collision"])
                for row in post_reference_rows
            ),
            "post_reference_inter_agent_collision_count": sum(
                bool(row["post_reference_inter_agent_collision"])
                for row in post_reference_rows
            ),
            "post_reference_timeout_count": sum(
                bool(row["post_reference_timeout"]) for row in post_reference_rows
            ),
            "STRESS_SET_USED_FOR_TARGET_TUNING": "NO",
            "NO_STRESS_SET_TUNING": "YES",
            "V2_SUPERVISION_VALID": "YES" if valid else "NO",
            "RECOMMENDED_NEXT_STEP": next_step,
            "gate_audit": gate_summary,
            "label_information_density": density_overall,
            "attribution_metrics": attribution_metrics,
            "formal_GAT_training_started": False,
            "formal_closed_loop_started": False,
            "stress_rerun_started": False,
            "automatic_target_adjustment_performed": False,
        }
        validity = {
            "schema_version": SCHEMA_VERSION,
            "passed": valid,
            "requirements": {
                "TRAINING_DEPLOYMENT_GATE_ALIGNMENT": gate_alignment,
                "SOURCE_STATE_REPRODUCTION": source_state_match,
                "CANDIDATE_IDENTITY_MATCH": candidate_identity_match,
                "GRAPH_SCHEMA_MATCH": graph_schema_match,
                "BRANCH_ROLLOUT_DETERMINISM": branch_determinism,
                "V2_LABEL_INFORMATION_DENSITY_ALLOWED": density_valid,
                "CANDIDATE_ATTRIBUTION_STABILITY_NOT_WEAK": attribution_valid,
                "NO_STRESS_SET_TUNING": no_stress_tuning,
            },
            "thresholds_frozen_before_results": True,
        }
        _write_json(output_dir / "validity_gate.json", validity)
        _write_json(output_dir / "conclusion.json", conclusion)
        (output_dir / "FINAL_REPORT.md").write_text(
            _report(conclusion), encoding="utf-8"
        )

        parameter_hash_after = _policy_parameter_sha256(policy)
        protected_after = {
            name: _sha256_file(path) for name, path in protected_paths.items()
        }
        integrity = {
            "schema_version": SCHEMA_VERSION,
            "status": "PASSED",
            "config_hash_before": config_hash_before,
            "config_hash_after": _sha256_file(config_path),
            "evaluator_hash_before": evaluator_hash_before,
            "evaluator_hash_after": _sha256_file(evaluator_path),
            "protected_hashes_before": protected_before,
            "protected_hashes_after": protected_after,
            "protected_files_unchanged": protected_before == protected_after,
            "policy_parameter_hash_before": parameter_hash_before,
            "policy_parameter_hash_after": parameter_hash_after,
            "policy_parameters_unchanged": parameter_hash_before == parameter_hash_after,
            "source_state_count": len(source_index["group_ids"]),
            "graph_count": len(graph_dataset),
            "branch_count": len(raw_outcomes),
            "stress_artifact_read_count": 0,
            "GAT_training_performed": False,
            "formal_closed_loop_performed": False,
            "stress_rerun_performed": False,
            "elapsed_seconds": time.perf_counter() - started,
        }
        if not all(
            (
                integrity["config_hash_before"] == integrity["config_hash_after"],
                integrity["evaluator_hash_before"] == integrity["evaluator_hash_after"],
                integrity["protected_files_unchanged"],
                integrity["policy_parameters_unchanged"],
            )
        ):
            integrity["status"] = "FAILED"
        _write_json(output_dir / "integrity_manifest.json", integrity)
        _write_json(
            output_dir / "generation_progress.json",
            {
                "stage": "COMPLETE",
                "stage_a_state_groups_completed": len(source_index["group_ids"]),
                "stage_a_graphs_completed": len(graph_dataset),
                "stage_a_passed": True,
                "stage_b_state_groups_completed": len(source_index["group_ids"]),
                "stage_b_graphs_completed": len(graph_dataset),
                "formal_branch_count_completed": len(raw_outcomes),
                "validity_gate_passed": valid,
            },
        )
        if integrity["status"] != "PASSED":
            raise RuntimeError("V2 read-only integrity gate failed")
        return output_dir
    finally:
        reference_env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    config_path = args.config.resolve()
    if config_path != DEFAULT_CONFIG_PATH.resolve():
        raise ValueError("formal V2 generation requires the frozen default config")
    config = _load_json(config_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else REPO_ROOT / str(config["output_root"]) / timestamp
    )
    result = run_generation(config, output_dir)
    print(result, flush=True)


if __name__ == "__main__":
    main()
