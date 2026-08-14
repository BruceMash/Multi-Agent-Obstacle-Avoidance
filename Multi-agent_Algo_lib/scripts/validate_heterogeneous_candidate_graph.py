"""Build and validate three-UAV FP-SHEP heterogeneous candidate graphs."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, ALGO_ROOT, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Environment.frozen_sac_dmp_execution import freeze_policy  # noqa: E402
from Guidance.reference_point_proposal_demo import (  # noqa: E402
    ProposalConfig,
    propose_reference_points,
)
from experiment_config import EXPERIMENT_CONFIG  # noqa: E402
from planning.candidate_execution_interface import (  # noqa: E402
    GraphReadyCandidateExecution,
    constant_velocity_conflict_diagnostic,
    graph_ready_candidate_execution,
)
from planning.heterogeneous_candidate_graph import (  # noqa: E402
    EgoGraphState,
    HeterogeneousCandidateGraphConfig,
    NeighborGraphState,
    build_heterogeneous_candidate_graph,
    build_heterogeneous_candidate_graph_from_env,
    graph_debug_summary,
)
from planning.policy_preview import (  # noqa: E402
    adapt_candidate_proposals,
    build_preview_inputs_from_env,
    preview_candidates,
)
from runner_sac import build_env as build_single_env  # noqa: E402
from runner_sac import build_model as build_single_model  # noqa: E402
from runner_sac import load_checkpoint  # noqa: E402
from scripts.evaluate_single_policy_multi_agent import (  # noqa: E402
    SinglePolicyMultiAgentEnv,
    build_aligned_config,
    build_policy_observations,
)
from scripts.visualize_heterogeneous_candidate_graph import (  # noqa: E402
    plot_heterogeneous_candidate_graph,
)


DEFAULT_CONFIG = REPO_ROOT / "configs" / "evaluation" / "fp_shep_heterogeneous_graph.json"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if hasattr(value, "detach"):
        return _jsonable(value.detach().cpu().numpy())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return "Inf" if value > 0.0 else ("-Inf" if value < 0.0 else "NaN")
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(
            {
                key: json.dumps(_jsonable(value), ensure_ascii=False)
                if isinstance(value, (dict, list, tuple, np.ndarray))
                else value
                for key, value in row.items()
            }
            for row in rows
        )


def _build_environment(num_agents: int) -> SinglePolicyMultiAgentEnv:
    config = build_aligned_config(num_agents=num_agents, max_steps=30)
    return SinglePolicyMultiAgentEnv(
        **config.build_core_env_kwargs(),
        observation_mode="peer_spheres",
        peer_radius=0.3,
        include_boundaries_in_sensor=False,
        terminate_on_boundary_collision=False,
    )


def _load_policy(checkpoint: Path):
    reference = build_single_env(config=EXPERIMENT_CONFIG, action_guidance_enabled=False)
    model = build_single_model(reference, config=EXPERIMENT_CONFIG, verbose=0)
    load_checkpoint(model, checkpoint)
    freeze_policy(model)
    return model, reference


def _scenario_options() -> dict[str, Any]:
    return {
        "starts": np.asarray([[0.5, 0.6, 0.6], [0.5, 2.2, 1.2], [0.5, 3.8, 1.8]], dtype=float),
        "goals": np.asarray([[8.2, 3.8, 1.8], [8.2, 2.2, 1.2], [8.2, 0.6, 0.6]], dtype=float),
        "static_obstacles": [],
        "dynamic_obstacles": [],
    }


def _synthetic_execution(
    candidate_id: int,
    candidate_world_position: np.ndarray,
    preview_positions: np.ndarray,
) -> GraphReadyCandidateExecution:
    positions = np.asarray(preview_positions, dtype=float)
    horizon = int(positions.shape[0])
    return GraphReadyCandidateExecution(
        candidate_id=candidate_id,
        candidate_world_position=np.asarray(candidate_world_position, dtype=float),
        preview_positions=positions,
        preview_velocities=np.zeros_like(positions),
        task_progress_raw=0.0,
        min_clearance_raw=float("inf"),
        max_execution_deviation_raw=0.0,
        terminal_speed_raw=0.0,
        requested_horizon_steps=horizon,
        effective_horizon_steps=horizon,
        effective_horizon_ratio=1.0,
        preview_completed=True,
        termination_reason="completed_horizon",
        feature_valid_mask=np.asarray([True, False, True, True], dtype=bool),
        feature_full_horizon_mask=np.asarray([True, False, True, True], dtype=bool),
        obstacle_clearance_source="frozen_lidar_surface_samples",
        obstacle_clearance_is_approximate=True,
        boundary_clearance_source="not_separately_available_from_untyped_lidar",
        boundary_clearance_is_approximate=True,
        clearance_finite_mask=False,
        open_space_flag=True,
    )


def _synthetic_interaction_evidence(
    *,
    root: Path,
    config: HeterogeneousCandidateGraphConfig,
) -> list[dict[str, Any]]:
    """Exercise dynamic edge semantics when the real smoke state has no edge."""

    cases = [
        {
            "name": "current_far_future_cross",
            "candidate_world_position": np.asarray([4.0, 0.0, 0.0]),
            "preview_positions": np.asarray(
                [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]]
            ),
            "neighbor_position": np.asarray([3.0, 3.0, 0.0]),
            "neighbor_velocity": np.asarray([0.0, -10.0, 0.0]),
            "expected_edge": True,
        },
        {
            "name": "current_near_future_diverge",
            "candidate_world_position": np.asarray([4.0, 0.0, 0.0]),
            "preview_positions": np.asarray(
                [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]]
            ),
            "neighbor_position": np.asarray([0.1, 0.0, 0.0]),
            "neighbor_velocity": np.asarray([-10.0, 0.0, 0.0]),
            "expected_edge": False,
        },
        {
            "name": "d_safe_below_d_min_below_d_align",
            "candidate_world_position": np.asarray([1.0, 0.0, 0.0]),
            "preview_positions": np.asarray([[1.0, 0.0, 0.0]] * 4),
            "neighbor_position": np.asarray([1.0, 0.8, 0.0]),
            "neighbor_velocity": np.asarray([0.0, 0.0, 0.0]),
            "expected_edge": True,
        },
    ]
    evidence: list[dict[str, Any]] = []
    ego = EgoGraphState(
        agent_id=0,
        position=np.zeros(3),
        velocity=np.zeros(3),
        previous_velocity=np.zeros(3),
        task_goal=np.asarray([6.0, 0.0, 0.0]),
        sector_safety=np.ones(56),
        goal_sector_id=0,
    )
    for case_index, case in enumerate(cases):
        candidate = np.asarray(case["candidate_world_position"], dtype=float)
        proposal = SimpleNamespace(
            point=candidate,
            direction=candidate / max(float(np.linalg.norm(candidate)), 1.0e-8),
            distance_progress=1.0,
            normalized_margin=1.0,
            score=1.0,
            azimuth_index=0,
            elevation_index=0,
        )
        execution = _synthetic_execution(
            candidate_id=case_index,
            candidate_world_position=candidate,
            preview_positions=np.asarray(case["preview_positions"], dtype=float),
        )
        neighbor = NeighborGraphState(
            agent_id=9,
            position=case["neighbor_position"],
            velocity=case["neighbor_velocity"],
        )
        diagnostic = constant_velocity_conflict_diagnostic(
            candidate_preview_positions=execution.preview_positions,
            neighbor_current_position=neighbor.position,
            neighbor_current_velocity=neighbor.velocity,
            dt=config.dt,
            risk_separation_threshold=config.d_safe,
        )
        graph = build_heterogeneous_candidate_graph(
            ego=ego,
            proposals=[proposal],
            executions=[execution],
            neighbors=[neighbor],
            config=config,
        )
        edge_count = int(
            graph["align", "spatiotemporal", "proposal"].edge_index.shape[1]
        )
        expected_edge = bool(case["expected_edge"])
        if edge_count != int(expected_edge):
            raise AssertionError(
                f"synthetic case {case['name']} edge_count={edge_count}, expected={expected_edge}"
            )
        plot_heterogeneous_candidate_graph(
            graph,
            ego_position=ego.position,
            task_goal=ego.task_goal,
            neighbor_positions=np.asarray([neighbor.position]),
            neighbor_velocities=np.asarray([neighbor.velocity]),
            output_path=root / f"synthetic_{case['name']}.png",
            title=str(case["name"]),
        )
        evidence.append(
            {
                "case": case["name"],
                "current_distance": float(np.linalg.norm(neighbor.position - ego.position)),
                "minimum_separation": diagnostic.minimum_separation,
                "time_to_minimum_separation": diagnostic.time_to_minimum_separation,
                "risk_duration_strict_distance_lt_d_safe": diagnostic.risk_duration,
                "d_safe": config.d_safe,
                "d_align": config.d_align,
                "edge_count": edge_count,
                "expected_edge": expected_edge,
                "passed": True,
            }
        )
    return evidence


def run_validation(config_path: Path, output_root: Path | None = None) -> Path:
    settings = json.loads(config_path.read_text(encoding="utf-8"))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = (output_root or (REPO_ROOT / settings["output_dir"])) / stamp
    root.mkdir(parents=True, exist_ok=False)
    checkpoint = (REPO_ROOT / settings["checkpoint"]).resolve()
    env = _build_environment(int(settings["num_agents"]))
    policy, reference = _load_policy(checkpoint)
    proposal_config = ProposalConfig(top_k=int(settings["consumer_top_k"]))
    graph_config = HeterogeneousCandidateGraphConfig(
        horizon_steps=int(settings["H"]),
        dt=float(settings["dt"]),
        d_safe=float(settings["d_safe"]),
        d_align=float(settings["d_align"]),
        d_safe_source=str(settings["d_safe_source"]),
        d_align_source=str(settings["d_align_source"]),
    )
    graph_rows: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    last_graph = None
    try:
        env.reset(seed=int(settings["seed"]), options=copy.deepcopy(_scenario_options()))
        evaluation_states = int(settings["evaluation_states"])
        if evaluation_states <= 0:
            raise ValueError("evaluation_states must be positive")
        for state_id in range(evaluation_states):
            for agent_index in range(int(env.num_agents)):
                packet = env.latest_sensor_packets[agent_index]
                proposals = propose_reference_points(
                    env.dynamics[agent_index].p,
                    env.goals[agent_index],
                    env.dynamics[agent_index].v,
                    packet,
                    env.sensors[agent_index],
                    proposal_config,
                    env.env_config.goal_tolerance,
                )
                selected = adapt_candidate_proposals(
                    proposals, consumer_top_k=proposal_config.top_k
                )
                initial, context = build_preview_inputs_from_env(env, agent_index)
                previews = preview_candidates(
                    initial_state=initial,
                    local_context=context,
                    candidates=selected,
                    policy=policy,
                    horizon=graph_config.horizon_steps,
                    dmp_config=env.dmps[agent_index].config,
                    dynamics=env.dynamics[agent_index],
                )
                executions = [
                    graph_ready_candidate_execution(candidate_id, preview)
                    for candidate_id, preview in enumerate(previews)
                ]
                graph = build_heterogeneous_candidate_graph_from_env(
                    env=env,
                    agent_index=agent_index,
                    proposals=selected,
                    executions=executions,
                    proposal_config=proposal_config,
                    config=graph_config,
                )
                last_graph = graph
                conflict = graph["align", "spatiotemporal", "proposal"]
                report = {
                    "state_id": state_id,
                    "agent_id": agent_index,
                    "candidate_count": graph["proposal"].num_nodes,
                    "neighbor_count": graph["align"].num_nodes,
                    "smooth_edge_count": graph["agent", "smooth", "proposal"].edge_index.shape[1],
                    "spatiotemporal_edge_count": conflict.edge_index.shape[1],
                    "candidate_ids": graph.proposal_node_to_candidate_id,
                    "neighbor_ids": graph.neighbor_node_to_agent_id,
                    "current_velocity": env.dynamics[agent_index].v.copy(),
                    "previous_velocity": env.previous_velocities[agent_index].copy(),
                    "node_shapes": {
                        node_type: list(graph[node_type].x.shape)
                        for node_type in graph.node_types
                    },
                    "edge_shapes": {
                        str(edge_type): list(graph[edge_type].edge_attr.shape)
                        for edge_type in graph.edge_types
                    },
                }
                reports.append(report)
                graph_rows.append(report)
                (root / f"state_{state_id}_agent_{agent_index}_debug_summary.txt").write_text(
                    graph_debug_summary(graph), encoding="utf-8"
                )
                neighbor_states = env.observable_neighbor_states(agent_index)
                plot_heterogeneous_candidate_graph(
                    graph,
                    ego_position=env.dynamics[agent_index].p,
                    task_goal=env.goals[agent_index],
                    neighbor_positions=np.asarray([item.position for item in neighbor_states], dtype=float),
                    neighbor_velocities=np.asarray([item.velocity for item in neighbor_states], dtype=float),
                    output_path=root / f"state_{state_id}_agent_{agent_index}_graph.png",
                    title=f"Three-UAV graph: state {state_id}, ego agent {agent_index}",
                )
            if state_id + 1 < evaluation_states:
                policy_observations = build_policy_observations(env)
                actions, _ = policy.predict(policy_observations, deterministic=True)
                _, _, terminated, truncated, _ = env.step(
                    np.asarray(actions, dtype=np.float32)
                )
                if terminated or truncated:
                    raise RuntimeError(
                        "three-UAV smoke environment terminated before all evaluation states"
                    )
        if last_graph is None:
            raise RuntimeError("three-UAV smoke test produced no graph")
        metadata = last_graph.graph_metadata
        interaction_evidence = _synthetic_interaction_evidence(
            root=root,
            config=graph_config,
        )
        _write_json(root / "interaction_case_evidence.json", interaction_evidence)
        _write_json(root / "config.json", settings)
        _write_json(root / "schema.json", metadata)
        _write_json(root / "summary.json", {
            "passed": True,
            "scenario": "deterministic_three_uav_crossing",
            "seed": int(settings["seed"]),
            "evaluation_state_count": evaluation_states,
            "graphs": reports,
            "synthetic_interaction_evidence": interaction_evidence,
            "unknown_global_environment_information_plotted": False,
            "gat_implemented": False,
        })
        _write_csv(root / "graph_summary.csv", graph_rows)
        _write_csv(root / "feature_schema.csv", metadata["feature_schema"])
    finally:
        env.close()
        reference.close()
    print(root)
    return root


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_validation(args.config.resolve(), args.output_root)
