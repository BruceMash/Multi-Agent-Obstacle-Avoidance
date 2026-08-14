"""Validate the untrained Edge-Enhanced GAT forward path on real graphs."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, ALGO_ROOT, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Guidance.reference_point_proposal_demo import ProposalConfig, propose_reference_points  # noqa: E402
from planning.candidate_execution_interface import graph_ready_candidate_execution  # noqa: E402
from planning.gat import (  # noqa: E402
    EdgeEnhancedGATConfig,
    PolicyPreviewEdgeEnhancedGATSelector,
    batch_candidate_graphs,
)
from planning.heterogeneous_candidate_graph import (  # noqa: E402
    HeterogeneousCandidateGraphConfig,
    build_heterogeneous_candidate_graph_from_env,
)
from planning.policy_preview import (  # noqa: E402
    adapt_candidate_proposals,
    build_preview_inputs_from_env,
    preview_candidates,
)
from scripts.evaluate_single_policy_multi_agent import build_policy_observations  # noqa: E402
from scripts.evaluate_single_policy_aligned_multi_agent import (  # noqa: E402
    STAGE_SPECS,
    build_single_distribution_multi_config,
    build_stage_scenario,
)
from scripts.validate_heterogeneous_candidate_graph import (  # noqa: E402
    _build_environment,
    _load_policy,
    _scenario_options,
)


DEFAULT_CONFIG = REPO_ROOT / "configs" / "evaluation" / "edge_enhanced_gat_forward.json"


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return "Inf" if value > 0 else ("-Inf" if value < 0 else "NaN")
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
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(_jsonable(value), ensure_ascii=False)
                    if isinstance(value, (list, tuple, dict, np.ndarray, torch.Tensor))
                    else value
                    for key, value in row.items()
                }
            )


def _graph_config(settings: dict[str, Any]) -> HeterogeneousCandidateGraphConfig:
    graph_settings = json.loads(
        (REPO_ROOT / settings["graph_config"]).read_text(encoding="utf-8")
    )
    return HeterogeneousCandidateGraphConfig(
        horizon_steps=int(graph_settings["H"]),
        dt=float(graph_settings["dt"]),
        d_safe=float(graph_settings["d_safe"]),
        d_align=float(graph_settings["d_align"]),
        d_safe_source=str(graph_settings["d_safe_source"]),
        d_align_source=str(graph_settings["d_align_source"]),
    )


def _build_current_graphs(
    *,
    env: Any,
    policy: Any,
    proposal_config: ProposalConfig,
    graph_config: HeterogeneousCandidateGraphConfig,
) -> list[Any]:
    graphs: list[Any] = []
    for agent_index in range(int(env.num_agents)):
        proposals = propose_reference_points(
            env.dynamics[agent_index].p,
            env.goals[agent_index],
            env.dynamics[agent_index].v,
            env.latest_sensor_packets[agent_index],
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
        graphs.append(
            build_heterogeneous_candidate_graph_from_env(
                env=env,
                agent_index=agent_index,
                proposals=selected,
                executions=executions,
                proposal_config=proposal_config,
                config=graph_config,
            )
        )
    return graphs


def _latency_ms(model: Any, graph: Any, warmup: int, measured: int) -> dict[str, float]:
    with torch.no_grad():
        for _ in range(warmup):
            model(graph)
        values: list[float] = []
        for _ in range(measured):
            start = time.perf_counter_ns()
            model(graph)
            values.append((time.perf_counter_ns() - start) / 1_000_000.0)
    array = np.asarray(values, dtype=float)
    return {
        "device": str(next(model.parameters()).device),
        "torch_num_threads": int(torch.get_num_threads()),
        "iterations": measured,
        "mean_ms": float(np.mean(array)),
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.percentile(array, 95)),
        "min_ms": float(np.min(array)),
        "max_ms": float(np.max(array)),
    }


def run(config_path: Path, output_root: Path | None = None) -> Path:
    settings = json.loads(config_path.read_text(encoding="utf-8"))
    root = (output_root or REPO_ROOT / settings["output_dir"]) / datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    root.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(int(settings["random_seed"]))
    np.random.seed(int(settings["random_seed"]))
    graph_config = _graph_config(settings)
    proposal_config = ProposalConfig(top_k=10)
    env = _build_environment(3)
    checkpoint = (REPO_ROOT / settings["checkpoint"]).resolve()
    policy, reference = _load_policy(checkpoint)
    try:
        env.reset(
            seed=int(settings["random_seed"]),
            options=copy.deepcopy(_scenario_options()),
        )
        # Use a non-reset real state so previous_velocity/current_velocity are distinct.
        for _ in range(2):
            observations = build_policy_observations(env)
            actions, _ = policy.predict(observations, deterministic=True)
            _, _, terminated, truncated, _ = env.step(np.asarray(actions, dtype=np.float32))
            if terminated or truncated:
                raise RuntimeError("validation scenario terminated before graph capture")
        graphs = _build_current_graphs(
            env=env,
            policy=policy,
            proposal_config=proposal_config,
            graph_config=graph_config,
        )
        active_config = build_single_distribution_multi_config(
            num_agents=3, max_steps=30
        )
        active_stage = next(
            item
            for item in STAGE_SPECS
            if item["name"] == "D_permuted_peer_spheres"
        )
        active_env = type(env)(
            **active_config.build_core_env_kwargs(),
            observation_mode="peer_spheres",
            peer_radius=0.3,
            include_boundaries_in_sensor=False,
            terminate_on_boundary_collision=False,
        )
        try:
            active_env.reset(
                seed=0,
                options=build_stage_scenario(active_config, active_stage, seed=0),
            )
            active_graphs = _build_current_graphs(
                env=active_env,
                policy=policy,
                proposal_config=proposal_config,
                graph_config=graph_config,
            )
        finally:
            active_env.close()
    finally:
        env.close()
        reference.close()

    model = PolicyPreviewEdgeEnhancedGATSelector(
        EdgeEnhancedGATConfig.from_mapping(settings["network"])
    ).eval()
    batch = batch_candidate_graphs(graphs)
    with torch.no_grad():
        output = model(batch, return_attention_debug=True)
        active_batch = batch_candidate_graphs(active_graphs)
        active_output = model(active_batch, return_attention_debug=True)
    model.train()
    model.zero_grad(set_to_none=True)
    gradient_output = model(batch)
    gradient_loss = gradient_output.candidate_logits.square().mean()
    gradient_loss.backward()
    gat_parameter_count = sum(1 for _ in model.parameters())
    gat_gradient_count = sum(
        parameter.grad is not None for parameter in model.parameters()
    )
    sac_actor_gradient_count = sum(
        parameter.grad is not None for parameter in policy.actor.parameters()
    )
    gradient_audit = {
        "dummy_loss": float(gradient_loss.detach()),
        "gat_parameter_tensors": gat_parameter_count,
        "gat_parameter_tensors_with_gradient_on_real_zero_align_graphs": gat_gradient_count,
        "sac_actor_parameter_tensors_with_gradient": sac_actor_gradient_count,
        "graph_node_inputs_require_gradient": any(
            bool(batch[node_type].x.requires_grad) for node_type in batch.node_types
        ),
        "graph_edge_inputs_require_gradient": any(
            bool(batch[edge_type].edge_attr.requires_grad) for edge_type in batch.edge_types
        ),
        "upstream_gradient_isolated": sac_actor_gradient_count == 0,
        "full_all_component_gradient_coverage_source": "synthetic unit test with active smooth and spatiotemporal relations",
    }
    model.zero_grad(set_to_none=True)
    model.eval()
    mapping_rows = [vars(item) for item in output.class_mapping]
    attention_rows: list[dict[str, Any]] = []
    for debug_source, debug_items in (
        ("natural_zero_align_capture", output.attention_debug),
        ("natural_active_align_capture", active_output.attention_debug),
    ):
        for debug in debug_items:
            for edge in range(debug.source_node_index.numel()):
                attention_rows.append(
                    {
                        "capture": debug_source,
                        "layer": debug.layer_index,
                        "relation": debug.relation,
                        "edge": edge,
                        "source_node": int(debug.source_node_index[edge]),
                        "target_proposal": int(debug.target_proposal_index[edge]),
                        "graph_index": int(debug.target_graph_index[edge]),
                        "raw_attention_logits": debug.raw_attention_logits[edge],
                        "normalized_alpha": debug.normalized_alpha[edge],
                        "message_l2_per_head": torch.linalg.vector_norm(
                            debug.message_content[edge], dim=-1
                        ),
                    }
                )

    runtime = settings["runtime"]
    latency = {
        "single_graph_K10_N2_H4": _latency_ms(
            model,
            graphs[0],
            int(runtime["warmup_iterations"]),
            int(runtime["measured_iterations"]),
        ),
        "batch_three_ego_graphs": _latency_ms(
            model,
            batch,
            int(runtime["warmup_iterations"]),
            int(runtime["measured_iterations"]),
        ),
    }
    graph_rows = []
    for graph_index, graph in enumerate(graphs):
        graph_rows.append(
            {
                "graph_index": graph_index,
                "candidate_count": graph["proposal"].num_nodes,
                "neighbor_count": graph["align"].num_nodes,
                "smooth_edge_count": graph["agent", "smooth", "proposal"].edge_index.shape[1],
                "spatiotemporal_edge_count": graph[
                    "align", "spatiotemporal", "proposal"
                ].edge_index.shape[1],
                "candidate_logits": output.logits_for_graph(graph_index),
                "candidate_probabilities": output.probabilities_for_graph(graph_index),
                "probability_sum": float(
                    output.probabilities_for_graph(graph_index).sum()
                ),
                "selected_class_index": int(output.selected_class_index[graph_index]),
                "selected_candidate_id": output.selected_candidate_id[graph_index],
                "selected_original_index": output.selected_original_index[graph_index],
                "selected_world_goal": output.selected_world_goal[graph_index],
            }
        )
    summary = {
        "passed": bool(
            torch.isfinite(output.candidate_logits).all()
            and torch.isfinite(output.candidate_probabilities).all()
            and all(
                math.isclose(row["probability_sum"], 1.0, abs_tol=1.0e-6)
                for row in graph_rows
            )
        ),
        "network_is_untrained": True,
        "selection_has_no_quality_interpretation": True,
        "parameter_count": model.parameter_counts(),
        "node_shapes": {key: list(batch[key].x.shape) for key in batch.node_types},
        "edge_shapes": {
            str(key): list(batch[key].edge_attr.shape) for key in batch.edge_types
        },
        "graph_outputs": graph_rows,
        "runtime": latency,
        "gradient_audit": gradient_audit,
        "attention_debug_rows": len(attention_rows),
        "natural_zero_align_capture": {
            "graphs": len(graphs),
            "spatiotemporal_edge_count": int(
                sum(
                    graph["align", "spatiotemporal", "proposal"].edge_index.shape[1]
                    for graph in graphs
                )
            ),
        },
        "natural_active_align_capture": {
            "scenario": "D_permuted_peer_spheres",
            "seed": 0,
            "graphs": len(active_graphs),
            "spatiotemporal_edge_count": int(
                sum(
                    graph["align", "spatiotemporal", "proposal"].edge_index.shape[1]
                    for graph in active_graphs
                )
            ),
            "all_logits_finite": bool(torch.isfinite(active_output.candidate_logits).all()),
            "all_probabilities_finite": bool(
                torch.isfinite(active_output.candidate_probabilities).all()
            ),
        },
        "standard_layer_audit": settings["standard_layer_audit"],
        "training_logic_implemented": False,
        "goal_switching_implemented": False,
    }
    _write_json(root / "config.json", settings)
    _write_json(root / "summary.json", summary)
    _write_json(root / "parameter_count.json", model.parameter_counts())
    _write_json(root / "runtime.json", latency)
    _write_json(root / "gradient_audit.json", gradient_audit)
    _write_csv(root / "graph_outputs.csv", graph_rows)
    _write_csv(root / "candidate_mapping.csv", mapping_rows)
    _write_csv(root / "attention_debug.csv", attention_rows)
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
