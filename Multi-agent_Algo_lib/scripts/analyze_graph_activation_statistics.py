"""Audit proposal-align graph activation without changing formal thresholds."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch  # Load Torch DLLs before runner_sac -> pandas/pyarrow on Windows.
from torch_geometric.data import Batch  # Keep native import order aligned with validation.


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, ALGO_ROOT, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Guidance.reference_point_proposal_demo import ProposalConfig  # noqa: E402
from planning.candidate_execution_interface import (  # noqa: E402
    constant_velocity_conflict_diagnostic,
)
from scripts.evaluate_single_policy_multi_agent import (  # noqa: E402
    SinglePolicyMultiAgentEnv,
    build_policy_observations,
)
from scripts.evaluate_single_policy_aligned_multi_agent import (  # noqa: E402
    STAGE_SPECS,
    build_single_distribution_multi_config,
    build_stage_scenario,
)
from scripts.validate_heterogeneous_candidate_graph import _load_policy  # noqa: E402
from scripts.validate_edge_enhanced_gat_forward import (  # noqa: E402
    _build_current_graphs,
    _graph_config,
)


DEFAULT_CONFIG = REPO_ROOT / "configs" / "evaluation" / "edge_enhanced_gat_forward.json"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _percentile(values: list[float], percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if values else None


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "p25": _percentile(values, 25),
        "median": _percentile(values, 50),
        "mean": float(np.mean(values)) if values else None,
        "p75": _percentile(values, 75),
        "p95": _percentile(values, 95),
        "max": max(values) if values else None,
    }


def _all_pair_diagnostics(graph: Any, d_safe: float) -> list[dict[str, float | int]]:
    dt = float(graph.graph_metadata["dt"])
    rows: list[dict[str, float | int]] = []
    align_raw = graph["align"].x_raw.detach().cpu().numpy()
    ego_position = graph["agent"].world_position[0].detach().cpu().numpy()
    ego_velocity = graph["agent"].current_velocity[0].detach().cpu().numpy()
    for neighbor_node in range(graph["align"].num_nodes):
        relative_direction = align_raw[neighbor_node, :3]
        relative_distance = float(align_raw[neighbor_node, 3])
        relative_velocity = align_raw[neighbor_node, 4:7]
        neighbor_position = ego_position + relative_direction * relative_distance
        neighbor_velocity = ego_velocity + relative_velocity
        for proposal_node, trajectory in enumerate(graph.candidate_preview_positions):
            diagnostic = constant_velocity_conflict_diagnostic(
                candidate_preview_positions=np.asarray(trajectory, dtype=float),
                neighbor_current_position=neighbor_position,
                neighbor_current_velocity=neighbor_velocity,
                dt=dt,
                risk_separation_threshold=d_safe,
            )
            rows.append(
                {
                    "neighbor_node": neighbor_node,
                    "proposal_node": proposal_node,
                    "d_min": diagnostic.minimum_separation,
                    "t_min": diagnostic.time_to_minimum_separation,
                    "T_risk": diagnostic.risk_duration,
                }
            )
    return rows


def _aggregate_threshold(
    graph_records: list[dict[str, Any]], threshold: float
) -> dict[str, Any]:
    graph_count = len(graph_records)
    graph_has_edge = 0
    total_edges = 0
    total_possible = 0
    total_proposals = 0
    proposals_with_edge = 0
    degrees: list[int] = []
    d_min_values: list[float] = []
    t_min_values: list[float] = []
    risk_values: list[float] = []
    for record in graph_records:
        k = int(record["candidate_count"])
        n = int(record["neighbor_count"])
        pairs = record["pair_diagnostics"]
        active = [row for row in pairs if float(row["d_min"]) < threshold]
        if active:
            graph_has_edge += 1
        total_edges += len(active)
        total_possible += k * n
        total_proposals += k
        degree = np.zeros(k, dtype=int)
        for row in active:
            degree[int(row["proposal_node"])] += 1
            d_min_values.append(float(row["d_min"]))
            t_min_values.append(float(row["t_min"]))
            risk_values.append(float(row["T_risk"]))
        proposals_with_edge += int(np.count_nonzero(degree))
        degrees.extend(degree.tolist())
    return {
        "d_align": float(threshold),
        "graphs": graph_count,
        "graph_activation_rate": graph_has_edge / graph_count if graph_count else 0.0,
        "edge_count": total_edges,
        "possible_edge_count": total_possible,
        "edge_density": total_edges / total_possible if total_possible else 0.0,
        "proposal_activation_rate": proposals_with_edge / total_proposals if total_proposals else 0.0,
        "mean_align_degree": float(np.mean(degrees)) if degrees else 0.0,
        "align_degree_distribution": _distribution([float(value) for value in degrees]),
        "d_min_distribution": _distribution(d_min_values),
        "t_min_distribution": _distribution(t_min_values),
        "T_risk_distribution": _distribution(risk_values),
    }


def run(config_path: Path, output_root: Path | None = None) -> Path:
    settings = json.loads(config_path.read_text(encoding="utf-8"))
    audit = settings["activation_statistics"]
    root = (output_root or REPO_ROOT / settings["output_dir"]) / (
        "activation_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    root.mkdir(parents=True, exist_ok=False)
    graph_config = _graph_config(settings)
    proposal_config = ProposalConfig(top_k=10)
    environment_config = build_single_distribution_multi_config(
        num_agents=3,
        max_steps=max(
            30,
            int(audit["states_per_seed"]) * int(audit["state_stride"]) + 5,
        ),
    )
    stages_by_name = {str(item["name"]): item for item in STAGE_SPECS}
    checkpoint = (REPO_ROOT / settings["checkpoint"]).resolve()
    policy = reference = None
    graph_records: list[dict[str, Any]] = []
    try:
        policy, reference = _load_policy(checkpoint)
        for scenario in audit["scenarios"]:
            for seed in audit["seeds"]:
                env = SinglePolicyMultiAgentEnv(
                    **environment_config.build_core_env_kwargs(),
                    observation_mode="peer_spheres",
                    peer_radius=0.3,
                    include_boundaries_in_sensor=False,
                    terminate_on_boundary_collision=False,
                )
                try:
                    scenario_name = str(scenario)
                    if scenario_name not in stages_by_name:
                        raise ValueError(f"unknown activation scenario: {scenario_name}")
                    captured = 0
                    episode_index = 0
                    rollout_seed = int(seed)
                    aligned_options = build_stage_scenario(
                        environment_config,
                        stages_by_name[scenario_name],
                        seed=rollout_seed,
                    )
                    env.reset(
                        seed=rollout_seed,
                        options=copy.deepcopy(aligned_options),
                    )
                    step_index = 0
                    while captured < int(audit["states_per_seed"]):
                        if step_index % int(audit["state_stride"]) == 0:
                            graphs = _build_current_graphs(
                                env=env,
                                policy=policy,
                                proposal_config=proposal_config,
                                graph_config=graph_config,
                            )
                            for agent_index, graph in enumerate(graphs):
                                pairs = _all_pair_diagnostics(
                                    graph, graph_config.d_safe
                                )
                                graph_records.append(
                                    {
                                        "scenario": scenario,
                                        "seed": int(seed),
                                        "episode_index": episode_index,
                                        "rollout_seed": rollout_seed,
                                        "state_index": captured,
                                        "environment_step": step_index,
                                        "agent_id": agent_index,
                                        "candidate_count": graph["proposal"].num_nodes,
                                        "neighbor_count": graph["align"].num_nodes,
                                        "agent_proposal_edge_count": graph[
                                            "agent", "smooth", "proposal"
                                        ].edge_index.shape[1],
                                        "formal_spatiotemporal_edge_count": graph[
                                            "align", "spatiotemporal", "proposal"
                                        ].edge_index.shape[1],
                                        "pair_diagnostics": pairs,
                                    }
                                )
                            captured += 1
                        observations = build_policy_observations(env)
                        actions, _ = policy.predict(observations, deterministic=True)
                        _, _, terminated, truncated, _ = env.step(
                            np.asarray(actions, dtype=np.float32)
                        )
                        step_index += 1
                        if (terminated or truncated) and captured < int(
                            audit["states_per_seed"]
                        ):
                            # Continue with a new seed-dependent scenario rather
                            # than duplicating the original reset state.
                            episode_index += 1
                            rollout_seed = (
                                int(seed)
                                + episode_index * 1009
                                + sum(ord(char) for char in scenario_name) * 100003
                            )
                            aligned_options = build_stage_scenario(
                                environment_config,
                                stages_by_name[scenario_name],
                                seed=rollout_seed,
                            )
                            env.reset(
                                seed=rollout_seed,
                                options=copy.deepcopy(aligned_options),
                            )
                            step_index = 0
                finally:
                    env.close()
    finally:
        if reference is not None:
            reference.close()

    sensitivity = [
        _aggregate_threshold(graph_records, float(value))
        for value in audit["d_align_values"]
    ]
    formal_threshold = float(audit["formal_d_align"])
    formal = next(
        row for row in sensitivity if math.isclose(row["d_align"], formal_threshold)
    )
    per_scenario = {
        str(scenario): _aggregate_threshold(
            [
                record
                for record in graph_records
                if str(record["scenario"]) == str(scenario)
            ],
            formal_threshold,
        )
        for scenario in audit["scenarios"]
    }
    graph_rows = [
        {key: value for key, value in record.items() if key != "pair_diagnostics"}
        for record in graph_records
    ]
    pair_rows = []
    for graph_index, record in enumerate(graph_records):
        for pair in record["pair_diagnostics"]:
            pair_rows.append(
                {
                    "graph_index": graph_index,
                    "scenario": record["scenario"],
                    "seed": record["seed"],
                    "episode_index": record["episode_index"],
                    "rollout_seed": record["rollout_seed"],
                    "state_index": record["state_index"],
                    "agent_id": record["agent_id"],
                    **pair,
                }
            )
    summary = {
        "graph_count": len(graph_records),
        "scenario_count": len(audit["scenarios"]),
        "seeds": audit["seeds"],
        "states_per_seed": int(audit["states_per_seed"]),
        "ego_graphs_per_state": 3,
        "candidate_count_distribution": _distribution(
            [float(row["candidate_count"]) for row in graph_records]
        ),
        "neighbor_count_distribution": _distribution(
            [float(row["neighbor_count"]) for row in graph_records]
        ),
        "agent_proposal_edge_count_distribution": _distribution(
            [float(row["agent_proposal_edge_count"]) for row in graph_records]
        ),
        "formal_d_align": formal_threshold,
        "formal_activation": formal,
        "per_scenario_formal_activation": per_scenario,
        "d_align_sensitivity": sensitivity,
        "automatic_threshold_selection": False,
        "formal_graph_builder_default_modified": False,
    }
    _write_json(root / "config.json", settings)
    _write_json(root / "summary.json", summary)
    _write_csv(root / "graph_records.csv", graph_rows)
    _write_csv(root / "pair_diagnostics.csv", pair_rows)
    _write_csv(
        root / "d_align_sensitivity.csv",
        [
            {
                key: value
                for key, value in row.items()
                if not key.endswith("_distribution")
            }
            for row in sensitivity
        ],
    )
    print(root)
    return root


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(args.config.resolve(), args.output_root)
