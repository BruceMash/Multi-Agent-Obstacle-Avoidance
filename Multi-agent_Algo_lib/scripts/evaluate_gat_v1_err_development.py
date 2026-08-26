"""Controlled development evaluation for event-triggered reference reconstruction.

The only new mechanism is a per-agent execution-interface supervisor.  Every
initial selection and every triggered reconstruction invokes the existing
Proposal -> TopK=10 -> FP-SHEP H4 -> frozen Stage-I GAT V1 pipeline.  Actor,
SAC-DMP, reward, environment, graph, Proposal, and FP-SHEP implementations are
not modified.
"""

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
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = Path(__file__).resolve().parent
for search_path in (REPO_ROOT, ALGO_ROOT, SCRIPTS_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from Guidance.reference_point_proposal_demo import (  # noqa: E402
    ProposalConfig,
    propose_reference_points,
)
from planning.candidate_execution_interface import (  # noqa: E402
    graph_ready_candidate_execution,
)
from planning.continuous_reference_transition import (  # noqa: E402
    CRTConfig,
    ContinuousReferenceTransition,
)
from planning.event_triggered_reference_reconstruction import (  # noqa: E402
    ACTIVE_GOAL_REFERENCE,
    ACTIVE_GOAL_TERMINAL,
    ACTIVE_SAFETY_MARGIN_SOURCE,
    EMERGENCY_SEMANTICS_EDGE_REARM,
    EMERGENCY_SEMANTICS_LEVEL,
    ERRConfig,
    EVENT_EMERGENCY_REPROPOSAL,
    EVENT_NO_UPDATE,
    EVENT_NORMAL_REPROPOSAL,
    EVENT_REFERENCE_COMPLETION_REPROPOSAL,
    EVENT_REFERENCE_HANDOFF,
    EventTriggeredReferenceSupervisor,
    apply_far_terminal_null_policy,
    apply_interaction_feasibility_policy,
    active_direction_safety_margin,
    set_active_goal_preserve_dmp_phase,
)
from planning.gat.candidate_selector import batch_candidate_graphs  # noqa: E402
from planning.gat.stage1_training import (  # noqa: E402
    load_model_checkpoint,
    resolve_device,
)
from planning.goal_semantics_diagnosis import (  # noqa: E402
    action_saturation_mask,
    task_aware_checkpoint_observations,
    temporary_checkpoint_observations,
)
from planning.heterogeneous_candidate_graph import (  # noqa: E402
    GRAPH_SCHEMA_VERSION,
    HeterogeneousCandidateGraphConfig,
    build_heterogeneous_candidate_graph_from_env,
)
from planning.historical_forcing_gate import (  # noqa: E402
    HISTORICAL_GATE_NAME,
    scoped_historical_preview_and_multi_agent_transition,
)
from planning.multi_agent_obstacle_scenario_audit import (  # noqa: E402
    minimum_static_surface_clearances,
)
from planning.pre_gat_220step_revalidation import (  # noqa: E402
    ImmutableCandidate,
    ImmutableCandidateBundle,
    generate_immutable_candidate_bundle,
    stable_hash,
)
from planning.pre_gat_closed_loop import (  # noqa: E402
    FPSHEPOnlineScoreSpec,
    score_fp_shep_candidates,
    score_fp_shep_candidates_batched,
    termination_reason,
    trajectory_metrics,
)
from planning.policy_preview import adapt_candidate_proposals  # noqa: E402
from planning.online_runtime_instrumentation import synchronize_cuda  # noqa: E402
from scripts.evaluate_gat_closed_loop import (  # noqa: E402
    CORE_METHOD_PATHS,
    _file_hashes,
    _make_method_plan,
    _model_hash,
    _policy_parameter_sha256,
    _rank_1based,
    _selected_edge_diagnostics,
    _sha256_file,
    reconstruct_proposals,
)
from scripts.evaluate_pre_gat_closed_loop import (  # noqa: E402
    _scenario_hash,
    _scene_snapshot,
    build_closed_loop_environment,
)
from scripts.evaluate_single_policy_aligned_multi_agent import (  # noqa: E402
    build_single_distribution_multi_config,
)
from scripts.evaluate_temporary_reference_interface import _load_policy  # noqa: E402


SCHEMA_VERSION = "gat_v1_err_development_v1"
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs/evaluation/gat_v1_err_development.json"
METHOD_ONE_SHOT = "gat_v1_one_shot"
METHOD_ERR = "gat_v1_err"
METHOD_RERR_GAT = "gat_v1_rerr"
METHOD_RERR_FP_SHEP = "fp_shep_rerr"
METHOD_DISPLAY = {
    METHOD_ONE_SHOT: "GAT-V1 one-shot",
    METHOD_ERR: "GAT-V1 + ERR",
    METHOD_RERR_GAT: "GAT-V1 + R-ERR",
    METHOD_RERR_FP_SHEP: "FP-SHEP Top-1 + R-ERR",
}
ERR_METHODS = {METHOD_ERR, METHOD_RERR_GAT, METHOD_RERR_FP_SHEP}
EDGE_ERR_METHODS = {METHOD_RERR_GAT, METHOD_RERR_FP_SHEP}
FP_SELECTOR_METHODS = {METHOD_RERR_FP_SHEP}
INITIAL_SELECTION = "INITIAL_SELECTION"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    encoded = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not encoded:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in encoded:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in encoded:
            writer.writerow(
                {
                    key: (
                        json.dumps(_jsonable(row.get(key)), ensure_ascii=False)
                        if isinstance(row.get(key), (list, tuple, dict, np.ndarray))
                        else _jsonable(row.get(key))
                    )
                    for key in fields
                }
            )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _assert_config(config: Mapping[str, Any]) -> None:
    if config["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unexpected ERR schema")
    if list(config["methods"]) != [METHOD_ONE_SHOT, METHOD_ERR]:
        raise ValueError("controlled comparison must contain one-shot and ERR only")
    if list(config["development_scenarios"]) != [
        "open",
        "sparse_static",
        "multi_agent",
    ]:
        raise ValueError("development scenario set changed")
    if list(config["development_seeds"]) != [50, 51, 52, 53, 54]:
        raise ValueError("new development seeds must remain 50..54")
    frozen = {
        "num_agents": 3,
        "top_k": 10,
        "H_preview": 4,
        "max_steps": 220,
        "dt": 0.1,
        "handoff_threshold_m": 0.25,
        "forcing_gate": HISTORICAL_GATE_NAME,
        "actor_observation_dim": 122,
    }
    for key, expected in frozen.items():
        if config[key] != expected:
            raise ValueError(f"frozen setting changed: {key}")
    if any(bool(value) for value in config["strict_exclusions"].values()):
        raise ValueError("strict exclusion flags must all remain false")
    err = ERRConfig.from_mapping({"dt": config["dt"], **config["err"]})
    if not np.isclose(err.handoff_distance_m, config["handoff_threshold_m"]):
        raise ValueError("ERR and established handoff distances must match")


def _execution_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    settings = _load_json((REPO_ROOT / config["base_execution_config"]).resolve())
    settings.update(
        {
            "checkpoint": config["sac_checkpoint"],
            "checkpoint_sha256_expected": config[
                "sac_checkpoint_sha256_expected"
            ],
            "deterministic_policy": True,
            "num_agents": int(config["num_agents"]),
            "max_steps": int(config["max_steps"]),
            "dt": float(config["dt"]),
            "peer_radius": float(config["peer_radius"]),
            "scenarios": list(config["development_scenarios"]),
            "seeds": list(config["development_seeds"]),
            "temporary_reference": {
                **settings["temporary_reference"],
                "K_requested": int(config["top_k"]),
                "reached_tolerance_m": float(config["handoff_threshold_m"]),
                "replanning_enabled": True,
            },
            "proposal_config": copy.deepcopy(config["proposal_config"]),
        }
    )
    return settings


def _score_spec(config: Mapping[str, Any]) -> FPSHEPOnlineScoreSpec:
    return FPSHEPOnlineScoreSpec.from_mapping(
        {
            "name": config["fp_shep"]["score_name"],
            "definition_status": "frozen_ERR_development",
            "H_preview": int(config["H_preview"]),
            "formula": config["fp_shep"]["formula"],
            "weights": {
                "progress": 1.0,
                "clearance": 1.0,
                "deviation": 1.0,
                "terminal_speed": 0.0,
            },
            "normalization": copy.deepcopy(config["fp_shep"]["normalization"]),
            "normalization_recalibrated": False,
            "real_outcome_used_to_tune_score": False,
        }
    )


def _selected_edge_records(graph: Any, candidate_id: int | None) -> list[dict[str, Any]]:
    """Return read-only peer/candidate diagnostics for a selected proposal.

    These fields are retained for post-hoc causal audits only.  They are not
    consumed by selection, triggering, or execution.
    """

    if candidate_id is None:
        return []
    store = graph["align", "spatiotemporal", "proposal"]
    edge_index = store.edge_index.detach().cpu().numpy()
    edge_attr = store.edge_attr.detach().cpu().numpy()
    neighbor_ids = tuple(int(value) for value in graph.neighbor_node_to_agent_id)
    records: list[dict[str, Any]] = []
    for edge_id in range(int(edge_index.shape[1])):
        source_node = int(edge_index[0, edge_id])
        target_node = int(edge_index[1, edge_id])
        if target_node != int(candidate_id):
            continue
        records.append(
            {
                "neighbor_agent_id": neighbor_ids[source_node],
                "time_to_minimum_separation_s": float(edge_attr[edge_id, 0]),
                "minimum_separation_m": float(edge_attr[edge_id, 1]),
                "risk_duration_s": float(edge_attr[edge_id, 2]),
                "d_safe_m": float(graph.graph_metadata["d_safe"]),
                "d_align_m": float(graph.graph_metadata["d_align"]),
            }
        )
    return records


def _candidate_interaction_records(graph: Any, proposal_count: int) -> list[dict[str, Any]]:
    d_safe = float(graph.graph_metadata["d_safe"])
    rows: list[dict[str, Any]] = []
    for candidate_id in range(int(proposal_count)):
        edges = _selected_edge_records(graph, candidate_id)
        minimum_d = (
            min(float(edge["minimum_separation_m"]) for edge in edges)
            if edges
            else None
        )
        maximum_risk_duration = (
            max(float(edge["risk_duration_s"]) for edge in edges) if edges else 0.0
        )
        risky = bool(
            minimum_d is not None
            and (minimum_d < d_safe or maximum_risk_duration > 0.0)
        )
        rows.append(
            {
                "candidate_id": candidate_id,
                "interaction_edge_count": len(edges),
                "minimum_predicted_separation_m": minimum_d,
                "maximum_risk_duration_s": maximum_risk_duration,
                "d_safe_m": d_safe,
                "risky": risky,
            }
        )
    return rows


def build_online_gat_plan(
    *,
    env: Any,
    config: Mapping[str, Any],
    policy: Any,
    gat_model: torch.nn.Module,
    gat_device: torch.device,
    scenario: str,
    seed: int,
    runtime_recorder: Any | None = None,
    selector: str = "gat",
    compute_agent_ids: Sequence[int] | None = None,
    fp_shep_vectorized: bool = False,
) -> dict[str, Any]:
    """Invoke the frozen upper pipeline with GAT or FP-SHEP final ranking."""

    if selector not in {"gat", "fp_shep"}:
        raise ValueError("selector must be gat or fp_shep")

    all_agent_ids = tuple(range(int(env.num_agents)))
    active_agent_ids = (
        all_agent_ids
        if compute_agent_ids is None
        else tuple(sorted({int(value) for value in compute_agent_ids}))
    )
    if not active_agent_ids or any(value not in all_agent_ids for value in active_agent_ids):
        raise ValueError("compute_agent_ids must be a non-empty subset of agent ids")

    upper_started = time.perf_counter_ns()
    terminal_goals = np.asarray(env.goals, dtype=float)
    lifecycle_config = ERRConfig.from_mapping({"dt": config["dt"], **config["err"]})
    proposal_config = ProposalConfig(**dict(config["proposal_config"]))
    proposal_started = time.perf_counter_ns()
    coarse_ranking_ms = 0.0
    if runtime_recorder is None:
        immutable = generate_immutable_candidate_bundle(
            env,
            scenario=scenario,
            seed=int(seed),
            proposal_config=proposal_config,
            consumer_top_k=int(config["top_k"]),
        )
    else:
        per_agent: list[tuple[ImmutableCandidate, ...]] = []
        counts: list[int] = []
        for agent_index in range(int(env.num_agents)):
            packet = env.latest_sensor_packets[agent_index]
            if packet is None:
                raise RuntimeError("environment must be reset before candidate generation")
            timing_sink: dict[str, float] = {}
            all_proposals = propose_reference_points(
                env.dynamics[agent_index].p,
                env.goals[agent_index],
                env.dynamics[agent_index].v,
                packet,
                env.sensors[agent_index],
                proposal_config,
                float(env.env_config.goal_tolerance),
                timing_sink=timing_sink,
            )
            coarse_ranking_ms += float(timing_sink.get("coarse_ranking_ms", 0.0))
            truncation_started = time.perf_counter_ns()
            selected = adapt_candidate_proposals(
                all_proposals, consumer_top_k=int(config["top_k"])
            )
            coarse_ranking_ms += (
                time.perf_counter_ns() - truncation_started
            ) / 1.0e6
            per_agent.append(
                tuple(
                    ImmutableCandidate.from_proposal(proposal, index)
                    for index, proposal in enumerate(selected)
                )
            )
            counts.append(len(all_proposals))
        immutable = ImmutableCandidateBundle(
            scenario=str(scenario),
            seed=int(seed),
            per_agent=tuple(per_agent),
            count_before_consumer=tuple(counts),
        )
    proposal_total_ms = (time.perf_counter_ns() - proposal_started) / 1.0e6
    proposal_generation_ms = max(0.0, proposal_total_ms - coarse_ranking_ms)
    candidate_hash_before = immutable.candidate_set_hash
    candidate_pack_started = time.perf_counter_ns()
    proposals_by_agent: list[tuple[Any, ...]] = []
    reconstruction_equivalent = True
    for candidates in immutable.per_agent:
        proposals, _, equivalent = reconstruct_proposals(candidates)
        proposals_by_agent.append(proposals)
        reconstruction_equivalent &= bool(equivalent)
    if not reconstruction_equivalent:
        raise RuntimeError("Proposal reconstruction changed candidate semantics")

    candidate_pack_ms = (time.perf_counter_ns() - candidate_pack_started) / 1.0e6

    previews_by_agent: list[tuple[Any, ...]] = [tuple() for _ in all_agent_ids]
    graphs_by_agent: dict[int, Any] = {}
    preview_gate_names: list[str | None] = []
    fp_shep_total_ms = 0.0
    graph_build_ms = 0.0
    graph_feature_build_ms = 0.0
    graph_edge_build_ms = 0.0
    graph_tensor_transfer_ms = 0.0
    fp_shep_state_prepare_ms = 0.0
    fp_shep_actor_forward_ms = 0.0
    fp_shep_dmp_rollout_ms = 0.0
    fp_shep_geometry_metric_ms = 0.0
    agent_candidate_compute_ms = {
        agent_id: 0.0 for agent_id in all_agent_ids
    }

    def preview_observer(_: dict[str, Any], transition: Any) -> None:
        preview_gate_names.append(
            transition.controller_info.get("forcing_gate_semantics")
        )

    preview_actor_start = (
        len(runtime_recorder.actor_rows) if runtime_recorder is not None else 0
    )
    actor_context = (
        policy.timing_mode("fp_shep_preview_actor")
        if runtime_recorder is not None and hasattr(policy, "timing_mode")
        else nullcontext()
    )
    with actor_context, scoped_historical_preview_and_multi_agent_transition(
        preview_observer=preview_observer
    ):
        for agent_id, proposals in enumerate(proposals_by_agent):
            if agent_id not in active_agent_ids:
                continue
            agent_compute_started = time.perf_counter_ns()
            preview_started = time.perf_counter_ns()
            preview_timing: dict[str, float] = {}
            if fp_shep_vectorized:
                previews = score_fp_shep_candidates_batched(
                    env=env,
                    agent_index=agent_id,
                    proposals=proposals,
                    policy=policy,
                    spec=_score_spec(config),
                    timing_sink=preview_timing,
                    observation_extension=config.get("sac_observation_extension"),
                )
            else:
                previews = score_fp_shep_candidates(
                    env=env,
                    agent_index=agent_id,
                    proposals=proposals,
                    policy=policy,
                    spec=_score_spec(config),
                    observation_extension=config.get("sac_observation_extension"),
                )
                for record in previews:
                    performance = record.preview.performance
                    fp_shep_state_prepare_ms += float(performance.observation_ms)
                    fp_shep_actor_forward_ms += float(performance.policy_ms)
                    fp_shep_dmp_rollout_ms += float(performance.transition_ms)
                    accounted = (
                        float(performance.observation_ms)
                        + float(performance.policy_ms)
                        + float(performance.transition_ms)
                    )
                    fp_shep_geometry_metric_ms += max(
                        0.0, float(performance.total_ms) - accounted
                    )
            fp_shep_total_ms += (time.perf_counter_ns() - preview_started) / 1.0e6
            previews_by_agent[agent_id] = previews
            fp_shep_state_prepare_ms += float(
                preview_timing.get("fp_shep_state_prepare_ms", 0.0)
            )
            fp_shep_actor_forward_ms += float(
                preview_timing.get("fp_shep_actor_forward_ms", 0.0)
            )
            fp_shep_dmp_rollout_ms += float(
                preview_timing.get("fp_shep_dmp_rollout_ms", 0.0)
            )
            fp_shep_geometry_metric_ms += float(
                preview_timing.get("fp_shep_geometry_metric_ms", 0.0)
            )
            if selector == "gat":
                graph_started = time.perf_counter_ns()
                graph_timing: dict[str, float] = {}
                executions = tuple(
                    graph_ready_candidate_execution(record.candidate_id, record.preview)
                    for record in previews
                )
                graph = build_heterogeneous_candidate_graph_from_env(
                    env=env,
                    agent_index=agent_id,
                    proposals=proposals,
                    executions=executions,
                    proposal_config=proposal_config,
                    config=HeterogeneousCandidateGraphConfig(
                        horizon_steps=int(config["H_preview"]),
                        d_safe=float(config["graph"].get("d_safe", 0.6)),
                        d_align=float(config["graph"]["d_align"]),
                        d_align_source=str(config["graph"]["d_align_source"]),
                        task_goal_distance_scale=float(
                            config["graph"].get("task_goal_distance_scale_m", 9.0)
                        ),
                    ),
                    timing_sink=graph_timing,
                )
                graphs_by_agent[agent_id] = graph
                graph_build_ms += (time.perf_counter_ns() - graph_started) / 1.0e6
                graph_feature_build_ms += float(
                    graph_timing.get("graph_feature_build_ms", 0.0)
                )
                graph_edge_build_ms += float(
                    graph_timing.get("graph_edge_build_ms", 0.0)
                )
            agent_candidate_compute_ms[agent_id] = (
                time.perf_counter_ns() - agent_compute_started
            ) / 1.0e6

    if immutable.candidate_set_hash != candidate_hash_before:
        raise RuntimeError("FP-SHEP preview mutated candidates")
    if preview_gate_names and not all(
        name == HISTORICAL_GATE_NAME for name in preview_gate_names
    ):
        raise RuntimeError("FP-SHEP preview gate differs from historical gate")
    if selector == "gat" and not all(
        graph.graph_metadata["feature_schema_version"] == GRAPH_SCHEMA_VERSION
        and int(graph.graph_metadata["H"]) == int(config["H_preview"])
        for graph in graphs_by_agent.values()
    ):
        raise RuntimeError("GAT graph schema mismatch")

    diagnostics: list[dict[str, Any]] | None
    if selector == "gat":
        graphs = [graphs_by_agent[agent_id] for agent_id in active_agent_ids]
        batch_started = time.perf_counter_ns()
        batched = batch_candidate_graphs(graphs).to(gat_device)
        if runtime_recorder is not None:
            synchronize_cuda()
        graph_tensor_transfer_ms = (time.perf_counter_ns() - batch_started) / 1.0e6
        graph_build_ms += graph_tensor_transfer_ms
        if runtime_recorder is not None:
            synchronize_cuda()
        gat_started = time.perf_counter_ns()
        with torch.inference_mode():
            output = gat_model(batched)
        if runtime_recorder is not None:
            synchronize_cuda()
        gat_forward_ms = (time.perf_counter_ns() - gat_started) / 1.0e6
        if not torch.isfinite(output.candidate_logits).all():
            raise RuntimeError("GAT produced non-finite logits")

        selected_ids: list[int | None] = [None for _ in all_agent_ids]
        diagnostics = [
            {
                "selected_class": None,
                "selected_null": None,
                "selected_candidate_id": None,
                "optimization_scope": "untriggered_output_not_computed",
            }
            for _ in all_agent_ids
        ]
        selection_decode_started = time.perf_counter_ns()
        for graph_index, agent_id in enumerate(active_agent_ids):
            graph = graphs_by_agent[agent_id]
            proposals = proposals_by_agent[agent_id]
            previews = previews_by_agent[agent_id]
            probabilities = output.probabilities_for_graph(graph_index).detach().cpu().numpy()
            logits = output.logits_for_graph(graph_index).detach().cpu().numpy()
            raw_class_index = int(output.selected_class_index[graph_index])
            raw_selected_id = output.selected_candidate_id[graph_index]
            expected_raw = None if raw_class_index == 0 else raw_class_index - 1
            if (
                expected_raw != raw_selected_id
                or not 0 <= raw_class_index < len(proposals) + 1
            ):
                raise RuntimeError("GAT null/proposal class mapping is invalid")
            terminal_goal_distance_m = float(
                np.linalg.norm(
                    terminal_goals[agent_id]
                    - np.asarray(env.dynamics[agent_id].p, dtype=float)
                )
            )
            lifecycle = apply_far_terminal_null_policy(
                raw_class_index=raw_class_index,
                class_logits=logits,
                proposal_count=len(proposals),
                terminal_goal_distance_m=terminal_goal_distance_m,
                terminal_local_scope_m=lifecycle_config.terminal_local_scope_m,
                policy=lifecycle_config.far_terminal_null_policy,
            )
            candidate_interactions = _candidate_interaction_records(
                graph, len(proposals)
            )
            interaction = apply_interaction_feasibility_policy(
                selected_candidate_id=lifecycle.effective_selected_candidate_id,
                class_logits=logits,
                candidate_risky=[row["risky"] for row in candidate_interactions],
                policy=str(
                    config["graph"].get(
                        "interaction_feasibility_policy", "allow_risky"
                    )
                ),
            )
            selected_id = interaction.effective_selected_candidate_id
            class_index = 0 if selected_id is None else int(selected_id) + 1
            selected_ids[agent_id] = selected_id
            ordered = np.sort(probabilities)[::-1]
            diagnostics[agent_id] = {
                    "selected_class": class_index,
                    "selected_null": selected_id is None,
                    "selected_candidate_id": selected_id,
                    "raw_selected_class": lifecycle.raw_class_index,
                    "raw_selected_null": lifecycle.raw_selected_candidate_id is None,
                    "raw_selected_candidate_id": lifecycle.raw_selected_candidate_id,
                    "terminal_goal_distance_m_at_selection": (
                        lifecycle.terminal_goal_distance_m
                    ),
                    "terminal_null_eligible": lifecycle.terminal_null_eligible,
                    "far_terminal_null_policy": lifecycle.policy,
                    "far_terminal_null_mask_applied": (
                        lifecycle.far_terminal_null_mask_applied
                    ),
                    "far_terminal_null_mask_unavailable_no_candidate": (
                        lifecycle.far_terminal_null_mask_unavailable_no_candidate
                    ),
                    "selected_class_before_interaction_mask": (
                        lifecycle.effective_class_index
                    ),
                    "selected_candidate_id_before_interaction_mask": (
                        interaction.selected_candidate_id_before_mask
                    ),
                    "interaction_feasibility_policy": interaction.policy,
                    "interaction_feasibility_mask_applied": (
                        interaction.interaction_mask_applied
                    ),
                    "selected_candidate_was_risky_before_interaction_mask": (
                        interaction.selected_candidate_was_risky
                    ),
                    "interaction_safe_candidate_count": (
                        interaction.safe_candidate_count
                    ),
                    "interaction_risky_candidate_count": (
                        interaction.risky_candidate_count
                    ),
                    "all_candidate_interaction_records": candidate_interactions,
                    "fp_shep_selected_candidate_id": (
                        int(np.argmax([record.score for record in previews]))
                        if previews
                        else None
                    ),
                    "selected_proposal_rank_by_FP_SHEP": _rank_1based(
                        [record.score for record in previews], selected_id
                    ),
                    "GAT_confidence": float(probabilities[class_index]),
                    "top1_top2_probability_margin": (
                        float(ordered[0] - ordered[1]) if len(ordered) >= 2 else None
                    ),
                    "class_count": len(probabilities),
                    "class_mapping_valid": True,
                    "logits_finite": bool(np.all(np.isfinite(logits))),
                    "class_probabilities": probabilities.tolist(),
                    "class_logits": logits.tolist(),
                    "selected_edge_records": _selected_edge_records(
                        graph, selected_id
                    ),
                    "graph_neighbor_agent_ids": [
                        int(value) for value in graph.neighbor_node_to_agent_id
                    ],
                    **_selected_edge_diagnostics(graph, selected_id),
                }
        selection_decode_ms = (
            time.perf_counter_ns() - selection_decode_started
        ) / 1.0e6
    else:
        gat_forward_ms = 0.0
        selection_decode_started = time.perf_counter_ns()
        selected_ids = [None for _ in all_agent_ids]
        for agent_id in active_agent_ids:
            previews = previews_by_agent[agent_id]
            selected_ids[agent_id] = (
                int(np.argmax([record.score for record in previews]))
                if previews
                else None
            )
        diagnostics = [
            {
                "selected_class": None,
                "selected_null": selected_id is None,
                "selected_candidate_id": selected_id,
                "fp_shep_selected_candidate_id": selected_id,
                "selected_proposal_rank_by_FP_SHEP": (
                    1 if selected_id is not None else None
                ),
                "GAT_confidence": None,
                "gat_diagnostic_not_executed": True,
            }
            for selected_id in selected_ids
        ]
        selection_decode_ms = (
            time.perf_counter_ns() - selection_decode_started
        ) / 1.0e6
    plan = _make_method_plan(
        source="gat_stage1" if selector == "gat" else "fp_shep_h4",
        terminal_goals=terminal_goals,
        proposals_by_agent=proposals_by_agent,
        previews_by_agent=previews_by_agent,
        selected_ids=selected_ids,
        gat_diagnostics=diagnostics,
    )
    upper_planning_total_ms = (time.perf_counter_ns() - upper_started) / 1.0e6
    preview_actor_ms = (
        float(
            sum(
                row["runtime_ms"]
                for row in runtime_recorder.actor_rows[preview_actor_start:]
                if row["actor_mode"] == "fp_shep_preview_actor"
            )
        )
        if runtime_recorder is not None
        else None
    )
    runtime_components = {
        "proposal_generation_ms": float(proposal_generation_ms),
        "coarse_ranking_ms": float(coarse_ranking_ms),
        "candidate_pack_ms": float(candidate_pack_ms),
        "fp_shep_state_prepare_ms": float(fp_shep_state_prepare_ms),
        "fp_shep_actor_forward_ms": float(fp_shep_actor_forward_ms),
        "fp_shep_dmp_rollout_ms": float(fp_shep_dmp_rollout_ms),
        "fp_shep_geometry_metric_ms": float(fp_shep_geometry_metric_ms),
        "fp_shep_total_ms": float(fp_shep_total_ms),
        "fp_shep_preview_actor_ms": preview_actor_ms,
        "graph_build_ms": float(graph_build_ms),
        "graph_feature_build_ms": float(graph_feature_build_ms),
        "graph_edge_build_ms": float(graph_edge_build_ms),
        "graph_tensor_transfer_ms": float(graph_tensor_transfer_ms),
        "gat_forward_ms": float(gat_forward_ms),
        "selection_decode_ms": float(selection_decode_ms),
        "upper_planning_total_ms": float(upper_planning_total_ms),
        **{
            f"agent_{agent_id}_candidate_compute_ms": float(value)
            for agent_id, value in agent_candidate_compute_ms.items()
        },
    }
    if runtime_recorder is not None:
        runtime_recorder.record_upper_event(runtime_components)
    return {
        "plan": plan,
        "candidate_bundle_hash": immutable.candidate_set_hash,
        "preview_historical_gate_verified": bool(preview_gate_names)
        and all(name == HISTORICAL_GATE_NAME for name in preview_gate_names),
        "preview_transition_call_count": len(preview_gate_names),
        "proposal_reconstruction_equivalent": True,
        "graph_schema_match": True if selector == "gat" else None,
        "graph_schema_applicable": selector == "gat",
        "selector": selector,
        "candidate_count_per_agent": [len(items) for items in proposals_by_agent],
        "computed_agent_ids": list(active_agent_ids),
        "fp_shep_vectorized": bool(fp_shep_vectorized),
        "_audit_graphs_by_agent": dict(graphs_by_agent),
        "runtime_components": runtime_components,
    }


def build_online_gat_plan_optimized(
    *,
    env: Any,
    config: Mapping[str, Any],
    policy: Any,
    gat_model: torch.nn.Module,
    gat_device: torch.device,
    scenario: str,
    seed: int,
    runtime_recorder: Any | None = None,
    selector: str = "gat",
) -> dict[str, Any]:
    """Exact-update optimization used only after dependency audit approval.

    Initial selection still computes all agents.  At later event ticks the
    runtime context identifies the agents whose active references may be
    updated; only those agents execute FP-SHEP and GAT.  Proposal generation is
    deliberately retained team-wide so the immutable candidate bundle remains
    available for exact audit and logging.
    """

    context = runtime_recorder.context if runtime_recorder is not None else {}
    triggered = context.get("triggered_agent_ids")
    compute_agent_ids = None if not triggered else [int(value) for value in triggered]
    return build_online_gat_plan(
        env=env,
        config=config,
        policy=policy,
        gat_model=gat_model,
        gat_device=gat_device,
        scenario=scenario,
        seed=seed,
        runtime_recorder=runtime_recorder,
        selector=selector,
        compute_agent_ids=compute_agent_ids,
        fp_shep_vectorized=True,
    )


def _selection_event_row(
    *,
    method: str,
    scenario: str,
    seed: int,
    step: int,
    agent_id: int,
    event: str,
    record: Mapping[str, Any],
    old_goal: Sequence[float],
    new_goal: Sequence[float],
    new_goal_type: str,
    goal_changed: bool,
    decision: Any | None,
    selection: Mapping[str, Any] | None,
    phase_before: float,
    phase_after: float,
    speed_before: float,
    speed_after: float,
) -> dict[str, Any]:
    base = {
        "schema_version": SCHEMA_VERSION,
        "method": method,
        "scenario": scenario,
        "seed": int(seed),
        "step": int(step),
        "time_s": float(step) * 0.1,
        "agent_id": int(agent_id),
        "event": event,
        "counts_as_reproposal": event
        in {
            EVENT_NORMAL_REPROPOSAL,
            EVENT_EMERGENCY_REPROPOSAL,
            EVENT_REFERENCE_COMPLETION_REPROPOSAL,
        },
        "goal_changed": bool(goal_changed),
        "old_active_goal": list(map(float, old_goal)),
        "new_active_goal": list(map(float, new_goal)),
        "new_active_goal_type": new_goal_type,
        "phase_before": float(phase_before),
        "phase_after": float(phase_after),
        "speed_before_mps": float(speed_before),
        "speed_after_mps": float(speed_after),
        "selected_candidate_id": record.get("selected_candidate_id"),
        "selected_null": bool(record.get("selected_null", False)),
        "raw_selected_class": record.get("raw_selected_class"),
        "raw_selected_candidate_id": record.get("raw_selected_candidate_id"),
        "raw_selected_null": record.get("raw_selected_null"),
        "terminal_goal_distance_m_at_selection": record.get(
            "terminal_goal_distance_m_at_selection"
        ),
        "terminal_null_eligible": record.get("terminal_null_eligible"),
        "far_terminal_null_policy": record.get("far_terminal_null_policy"),
        "far_terminal_null_mask_applied": bool(
            record.get("far_terminal_null_mask_applied", False)
        ),
        "far_terminal_null_mask_unavailable_no_candidate": bool(
            record.get("far_terminal_null_mask_unavailable_no_candidate", False)
        ),
        "selected_class_before_interaction_mask": record.get(
            "selected_class_before_interaction_mask"
        ),
        "selected_candidate_id_before_interaction_mask": record.get(
            "selected_candidate_id_before_interaction_mask"
        ),
        "interaction_feasibility_policy": record.get(
            "interaction_feasibility_policy"
        ),
        "interaction_feasibility_mask_applied": bool(
            record.get("interaction_feasibility_mask_applied", False)
        ),
        "selected_candidate_was_risky_before_interaction_mask": record.get(
            "selected_candidate_was_risky_before_interaction_mask"
        ),
        "interaction_safe_candidate_count": record.get(
            "interaction_safe_candidate_count"
        ),
        "interaction_risky_candidate_count": record.get(
            "interaction_risky_candidate_count"
        ),
        "all_candidate_interaction_records": record.get(
            "all_candidate_interaction_records"
        ),
        "K_t": int(record.get("K_t", 0)),
        "candidate_world_points": record.get("candidate_world_points", []),
        "candidate_scores": record.get("proposal_scores", []),
        "fp_shep_scores": [
            item.get("score")
            for item in record.get("fp_shep_candidate_records", [])
        ],
        "fp_shep_selected_candidate_id": record.get(
            "fp_shep_selected_candidate_id",
            (
                record.get("selected_candidate_id")
                if record.get("selection_source") == "fp_shep_h4"
                else None
            ),
        ),
        "gat_selected_candidate_id": (
            record.get("selected_candidate_id")
            if record.get("selection_source") == "gat_stage1"
            else None
        ),
        "GAT_confidence": record.get("GAT_confidence"),
        "top1_top2_probability_margin": record.get(
            "top1_top2_probability_margin"
        ),
        "class_probabilities": record.get("class_probabilities"),
        "class_logits": record.get("class_logits"),
        "selected_minimum_t_min": record.get("selected_minimum_t_min"),
        "selected_minimum_d_min": record.get("selected_minimum_d_min"),
        "selected_maximum_T_risk": record.get("selected_maximum_T_risk"),
        "selected_align_edge_count": record.get("selected_align_edge_count"),
        "selected_edge_records": record.get("selected_edge_records", []),
        "graph_neighbor_agent_ids": record.get("graph_neighbor_agent_ids", []),
        "dctb_activated": bool(record.get("dctb_activated", False)),
        "dctb_replaced": bool(record.get("dctb_replaced", False)),
        "dctb_original_candidate_id": record.get("dctb_original_candidate_id"),
        "dctb_replacement_candidate_id": record.get(
            "dctb_replacement_candidate_id"
        ),
        "dctb_replacement_gat_rank": record.get("dctb_replacement_gat_rank"),
        "dctb_reason": record.get("dctb_reason"),
        "candidate_bundle_hash": (
            selection.get("candidate_bundle_hash") if selection is not None else None
        ),
        "selection_plan_hash": (
            selection["plan"]["selection_plan_hash"]
            if selection is not None
            else None
        ),
        "upper_planning_runtime_ms": (
            selection.get("runtime_components", {}).get("upper_planning_total_ms")
            if selection is not None
            else None
        ),
        "runtime_components": (
            selection.get("runtime_components") if selection is not None else None
        ),
    }
    if decision is not None:
        base.update(
            {
                "trigger_reasons": list(decision.trigger_reasons),
                "active_goal_distance_m": decision.active_goal_distance_m,
                "active_age_s": decision.active_age_s,
                "progress_rate_mps": decision.progress_rate_mps,
                "progress_valid": decision.progress_valid,
                "active_safety_margin_m": decision.active_safety_margin_m,
                "phi_tau": decision.phi_tau,
                "phi_p": decision.phi_p,
                "phi_h": decision.phi_h,
                "S_rep": decision.S_rep,
                "dwell_satisfied": decision.dwell_satisfied,
            }
        )
    if record.get("selection_source") == "gat_stage1":
        base["selection_agreement"] = (
            base["gat_selected_candidate_id"]
            == base["fp_shep_selected_candidate_id"]
        )
    else:
        base["selection_agreement"] = None
    return base


def _trajectory_rows(
    *,
    method: str,
    scenario: str,
    seed: int,
    step: int,
    env: Any,
    supervisor: EventTriggeredReferenceSupervisor,
    policy_actions: np.ndarray | None = None,
    applied_accelerations: np.ndarray | None = None,
    commanded_accelerations: np.ndarray | None = None,
    step_info: Mapping[str, Any] | None = None,
    executed_references: np.ndarray | None = None,
    executed_reference_velocities: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    positions = np.asarray(env._positions(), dtype=float)
    static_clearances = minimum_static_surface_clearances(
        positions, env.static_obstacles
    )
    collision_info = dict(step_info or env._check_collision())
    obstacle_mask = np.asarray(
        collision_info.get("obstacle_collision_mask", np.zeros(env.num_agents)),
        dtype=bool,
    )
    peer_mask = np.asarray(
        collision_info.get("inter_agent_collision_mask", np.zeros(env.num_agents)),
        dtype=bool,
    )
    executed = (
        np.stack([state.active_goal for state in supervisor.states])
        if executed_references is None
        else np.asarray(executed_references, dtype=float)
    )
    executed_velocities = (
        np.zeros((int(env.num_agents), 3), dtype=float)
        if executed_reference_velocities is None
        else np.asarray(executed_reference_velocities, dtype=float)
    )
    if executed.shape != (int(env.num_agents), 3):
        raise ValueError("executed_references must have shape (num_agents, 3)")
    if executed_velocities.shape != (int(env.num_agents), 3):
        raise ValueError(
            "executed_reference_velocities must have shape (num_agents, 3)"
        )
    for agent_id, state in enumerate(supervisor.states):
        position = np.asarray(env.dynamics[agent_id].p, dtype=float)
        velocity = np.asarray(env.dynamics[agent_id].v, dtype=float)
        terminal_goal = np.asarray(env.goals[agent_id], dtype=float)
        packet = env.latest_sensor_packets[agent_id]
        action = (
            np.asarray(policy_actions[agent_id], dtype=float)
            if policy_actions is not None
            else np.full(3, np.nan, dtype=float)
        )
        applied = (
            np.asarray(applied_accelerations[agent_id], dtype=float)
            if applied_accelerations is not None
            else np.full(3, np.nan, dtype=float)
        )
        commanded = (
            np.asarray(commanded_accelerations[agent_id], dtype=float)
            if commanded_accelerations is not None
            else np.full(3, np.nan, dtype=float)
        )
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "method": method,
                "scenario": scenario,
                "seed": int(seed),
                "step": int(step),
                "time_s": float(step) * 0.1,
                "agent_id": int(agent_id),
                "x_m": float(position[0]),
                "y_m": float(position[1]),
                "z_m": float(position[2]),
                "vx_mps": float(velocity[0]),
                "vy_mps": float(velocity[1]),
                "vz_mps": float(velocity[2]),
                "active_goal_x_m": float(state.active_goal[0]),
                "active_goal_y_m": float(state.active_goal[1]),
                "active_goal_z_m": float(state.active_goal[2]),
                "commanded_reference_x_m": float(state.active_goal[0]),
                "commanded_reference_y_m": float(state.active_goal[1]),
                "commanded_reference_z_m": float(state.active_goal[2]),
                "executed_reference_x_m": float(executed[agent_id, 0]),
                "executed_reference_y_m": float(executed[agent_id, 1]),
                "executed_reference_z_m": float(executed[agent_id, 2]),
                "executed_reference_vx_mps": float(executed_velocities[agent_id, 0]),
                "executed_reference_vy_mps": float(executed_velocities[agent_id, 1]),
                "executed_reference_vz_mps": float(executed_velocities[agent_id, 2]),
                "command_execution_error_m": float(
                    np.linalg.norm(state.active_goal - executed[agent_id])
                ),
                "reference_filter_speed_mps": float(
                    np.linalg.norm(executed_velocities[agent_id])
                ),
                "active_goal_type": state.active_goal_type,
                "goal_version": int(state.goal_version),
                "terminal_goal_x_m": float(terminal_goal[0]),
                "terminal_goal_y_m": float(terminal_goal[1]),
                "terminal_goal_z_m": float(terminal_goal[2]),
                "active_goal_distance_m": float(
                    np.linalg.norm(state.active_goal - position)
                ),
                "terminal_goal_distance_m": float(
                    np.linalg.norm(terminal_goal - position)
                ),
                "sensor_min_clearance_m": (
                    float(packet.min_clearance) if packet is not None else None
                ),
                "static_obstacle_clearance_m": float(static_clearances[agent_id]),
                "obstacle_collision": bool(obstacle_mask[agent_id]),
                "inter_agent_collision": bool(peer_mask[agent_id]),
                "policy_action_0": float(action[0]),
                "policy_action_1": float(action[1]),
                "policy_action_2": float(action[2]),
                "commanded_ax_mps2": float(commanded[0]),
                "commanded_ay_mps2": float(commanded[1]),
                "commanded_az_mps2": float(commanded[2]),
                "applied_ax_mps2": float(applied[0]),
                "applied_ay_mps2": float(applied[1]),
                "applied_az_mps2": float(applied[2]),
            }
        )
    return rows


def run_episode(
    *,
    config: Mapping[str, Any],
    settings: Mapping[str, Any],
    multi_config: Any,
    policy: Any,
    gat_model: torch.nn.Module,
    gat_device: torch.device,
    method: str,
    scenario: str,
    seed: int,
    environment_builder: Callable[..., tuple[Any, dict[str, Any]]] | None = None,
    runtime_recorder: Any | None = None,
    upper_plan_builder: Callable[..., dict[str, Any]] = build_online_gat_plan,
    crt: Mapping[str, Any] | None = None,
    sac_observation_extension: Mapping[str, Any] | None = None,
    execution_acceleration_limiter: Any | None = None,
    candidate_commitment_tiebreak: Any | None = None,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    builder = environment_builder or build_closed_loop_environment
    env, scene_metadata = builder(
        config=multi_config,
        scenario=scenario,
        seed=int(seed),
        peer_radius=float(config["peer_radius"]),
    )
    if execution_acceleration_limiter is not None:
        if int(execution_acceleration_limiter.num_agents) != int(env.num_agents):
            raise ValueError("execution limiter agent count does not match environment")
        if not np.isclose(
            float(execution_acceleration_limiter.config.dt),
            float(config["dt"]),
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError("execution limiter dt does not match frozen runtime")
        execution_acceleration_limiter.reset()
        env.execution_acceleration_limiter = execution_acceleration_limiter
    if candidate_commitment_tiebreak is not None:
        if int(candidate_commitment_tiebreak.num_agents) != int(env.num_agents):
            raise ValueError(
                "candidate commitment tie-break agent count does not match environment"
            )
        candidate_commitment_tiebreak.reset()
    started = time.perf_counter_ns()
    actor_row_start = len(runtime_recorder.actor_rows) if runtime_recorder is not None else 0
    dmp_row_start = len(runtime_recorder.dmp_rows) if runtime_recorder is not None else 0
    upper_row_start = len(runtime_recorder.upper_rows) if runtime_recorder is not None else 0
    proposal_config = ProposalConfig(**dict(config["proposal_config"]))
    try:
        snapshot = _scene_snapshot(env)
        initial_hash = _scenario_hash(snapshot)
        starts = np.asarray(env.starts, dtype=float).copy()
        terminal_goals = np.asarray(env.goals, dtype=float).copy()
        initial_context = (
            runtime_recorder.scoped_context(
                planning_decision_index=0, event_step=0, event_type=INITIAL_SELECTION
            )
            if runtime_recorder is not None
            else nullcontext()
        )
        with initial_context:
            initial_selection = upper_plan_builder(
                env=env,
                config=config,
                policy=policy,
                gat_model=gat_model,
                gat_device=gat_device,
                scenario=scenario,
                seed=int(seed),
                runtime_recorder=runtime_recorder,
                selector=("fp_shep" if method in FP_SELECTOR_METHODS else "gat"),
            )
        if candidate_commitment_tiebreak is not None:
            initial_selection = candidate_commitment_tiebreak.apply_plan(
                initial_selection,
                env=env,
                agent_ids=tuple(range(int(env.num_agents))),
                event_types={
                    agent_id: INITIAL_SELECTION
                    for agent_id in range(int(env.num_agents))
                },
                active_safety_margins_m={
                    agent_id: None for agent_id in range(int(env.num_agents))
                },
                initial=True,
                step=0,
            )
        plan = initial_selection["plan"]
        initial_selection_semantic_hash = stable_hash(
            {
                "references": plan["references"],
                "available": plan["available"],
                "selected_candidate_ids": [
                    record.get("selected_candidate_id")
                    for record in plan["candidate_records"]
                ],
                "selected_classes": [
                    record.get("selected_class")
                    for record in plan["candidate_records"]
                ],
            }
        )
        active_goals = np.asarray(plan["references"], dtype=float)
        available = np.asarray(plan["available"], dtype=bool)
        goal_types = [
            ACTIVE_GOAL_REFERENCE if value else ACTIVE_GOAL_TERMINAL
            for value in available
        ]
        err_config = ERRConfig.from_mapping({"dt": config["dt"], **config["err"]})
        supervisor = EventTriggeredReferenceSupervisor(
            err_config,
            terminal_goals=terminal_goals,
            active_goals=active_goals,
            active_goal_types=goal_types,
            initial_positions=starts,
            emergency_event_semantics=(
                EMERGENCY_SEMANTICS_EDGE_REARM
                if method in EDGE_ERR_METHODS
                else EMERGENCY_SEMANTICS_LEVEL
            ),
        )
        crt_config: CRTConfig | None = None
        crt_filters: list[ContinuousReferenceTransition] | None = None
        crt_pending_settlements: list[dict[str, Any] | None] = [
            None for _ in range(int(env.num_agents))
        ]
        crt_trace_rows: list[dict[str, Any]] = []
        crt_update_runtime_ns = 0
        crt_safety_runtime_ns = 0
        crt_bandwidth_override_count = 0
        crt_instant_replacement_count = 0
        if crt is not None:
            crt_config = CRTConfig(
                dt=float(config["dt"]),
                settling_time_s=float(crt["settling_time_s"]),
                safety_bandwidth_multiplier=float(
                    crt.get("safety_bandwidth_multiplier", 2.0)
                ),
                settled_absolute_tolerance_m=float(
                    crt.get("settled_absolute_tolerance_m", 1.0e-6)
                ),
            )
            hard_margin = float(
                crt.get(
                    "hard_safety_margin_m",
                    config["err"]["h_emg_m"],
                )
            )
            if not np.isclose(
                hard_margin,
                float(config["err"]["h_emg_m"]),
                rtol=0.0,
                atol=0.0,
            ):
                raise ValueError(
                    "CRT hard safety margin must equal the frozen R-ERR h_emg_m"
                )
            crt_filters = [
                ContinuousReferenceTransition(crt_config, state.active_goal)
                for state in supervisor.states
            ]

        def crt_executed_arrays() -> tuple[np.ndarray, np.ndarray]:
            if crt_filters is None:
                return (
                    np.stack([state.active_goal for state in supervisor.states]),
                    np.zeros((int(env.num_agents), 3), dtype=float),
                )
            return (
                np.stack([item.executed for item in crt_filters]),
                np.stack([item.velocity for item in crt_filters]),
            )

        def crt_accept_command(
            agent_id: int,
            command: Sequence[float],
            event_row: dict[str, Any],
        ) -> None:
            if crt_filters is None or crt_config is None:
                return
            transition = crt_filters[int(agent_id)]
            previous_command = transition.command.copy()
            executed_before = transition.executed.copy()
            velocity_before = transition.velocity.copy()
            changed = transition.set_command(command)
            initial_error = float(np.linalg.norm(transition.command - executed_before))
            event_row.update(
                {
                    "crt_enabled": True,
                    "crt_previous_command": previous_command.tolist(),
                    "crt_command": transition.command.tolist(),
                    "crt_executed_at_command_update": executed_before.tolist(),
                    "crt_velocity_at_command_update": velocity_before.tolist(),
                    "crt_command_changed": changed,
                    "crt_initial_tracking_error_m": initial_error,
                    "crt_settle_threshold_m": transition.settled_threshold_m(
                        initial_error
                    ),
                    "crt_settle_step": None,
                    "crt_settle_time_s": None,
                    "crt_settle_censored_by_new_command": False,
                }
            )
            previous_pending = crt_pending_settlements[int(agent_id)]
            if previous_pending is not None:
                previous_pending["crt_settle_censored_by_new_command"] = True
                previous_pending["crt_settle_censor_step"] = int(event_row["step"])
            crt_pending_settlements[int(agent_id)] = event_row if changed else None

        phase_switch_deltas: list[float] = []
        for agent_id, state in enumerate(supervisor.states):
            phase_before = float(env.dmps[agent_id].phase)
            executed_reference = (
                state.active_goal
                if crt_filters is None
                else crt_filters[agent_id].executed
            )
            set_active_goal_preserve_dmp_phase(
                env.dmps[agent_id], executed_reference
            )
            phase_switch_deltas.append(float(env.dmps[agent_id].phase) - phase_before)
            if method in EDGE_ERR_METHODS:
                initial_safety = active_direction_safety_margin(
                    env, agent_id, state.active_goal, proposal_config
                )
                supervisor.synchronize_emergency_latch(
                    agent_id,
                    active_safety_margin_m=initial_safety.value_m,
                )

        event_rows: list[dict[str, Any]] = []
        trigger_rows: list[dict[str, Any]] = []
        initial_executed, initial_executed_velocity = crt_executed_arrays()
        path_rows = _trajectory_rows(
            method=method,
            scenario=scenario,
            seed=seed,
            step=0,
            env=env,
            supervisor=supervisor,
            executed_references=initial_executed,
            executed_reference_velocities=initial_executed_velocity,
        )
        reference_segments: list[list[dict[str, Any]]] = [
            [] for _ in range(int(env.num_agents))
        ]
        open_segments: list[dict[str, Any] | None] = [None] * int(env.num_agents)
        for agent_id, (state, record) in enumerate(
            zip(supervisor.states, plan["candidate_records"], strict=True)
        ):
            if state.active_goal_type == ACTIVE_GOAL_REFERENCE:
                segment = {
                    "start_step": 0,
                    "goal": state.active_goal.tolist(),
                    "source_event": INITIAL_SELECTION,
                    "reached": False,
                    "end_step": None,
                    "end_reason": None,
                }
                reference_segments[agent_id].append(segment)
                open_segments[agent_id] = segment
            speed = float(np.linalg.norm(env.dynamics[agent_id].v))
            initial_event_row = _selection_event_row(
                    method=method,
                    scenario=scenario,
                    seed=seed,
                    step=0,
                    agent_id=agent_id,
                    event=INITIAL_SELECTION,
                    record=record,
                    old_goal=terminal_goals[agent_id],
                    new_goal=state.active_goal,
                    new_goal_type=state.active_goal_type,
                    goal_changed=not np.array_equal(
                        terminal_goals[agent_id], state.active_goal
                    ),
                    decision=None,
                    selection=initial_selection,
                    phase_before=float(env.dmps[agent_id].phase),
                    phase_after=float(env.dmps[agent_id].phase),
                    speed_before=speed,
                    speed_after=speed,
                )
            if crt_filters is not None:
                initial_event_row.update(
                    {
                        "crt_enabled": True,
                        "crt_previous_command": terminal_goals[agent_id].tolist(),
                        "crt_command": state.active_goal.tolist(),
                        "crt_executed_at_command_update": state.active_goal.tolist(),
                        "crt_velocity_at_command_update": [0.0, 0.0, 0.0],
                        "crt_command_changed": bool(
                            not np.array_equal(
                                terminal_goals[agent_id], state.active_goal
                            )
                        ),
                        "crt_initial_tracking_error_m": 0.0,
                        "crt_settle_threshold_m": float(
                            crt_config.settled_absolute_tolerance_m
                        ),
                        "crt_settle_step": 0,
                        "crt_settle_time_s": 0.0,
                        "crt_settle_censored_by_new_command": False,
                    }
                )
            event_rows.append(initial_event_row)

        positions = [env._positions().copy()]
        velocities = [env._velocities().copy()]
        accelerations: list[np.ndarray] = []
        previous_applied_accelerations = np.zeros((int(env.num_agents), 3), dtype=float)
        min_clearances = [
            float(min(packet.min_clearance for packet in env.latest_sensor_packets))
        ]
        min_inter_agent_distances = [
            float(env._check_collision()["min_inter_agent_distance"])
        ]
        static_clearances = minimum_static_surface_clearances(
            env._positions(), env.static_obstacles
        )
        collision = obstacle_collision = inter_agent_collision = False
        obstacle_collision_mask = np.zeros(int(env.num_agents), dtype=bool)
        inter_agent_collision_mask = np.zeros(int(env.num_agents), dtype=bool)
        completed_mask = np.zeros(int(env.num_agents), dtype=bool)
        completion_steps: list[int | None] = [None] * int(env.num_agents)
        applied_norms: list[float] = []
        commanded_norms: list[float] = []
        saturation_values: list[float] = []
        terminated = truncated = False
        info: dict[str, Any] = {}
        execution_gate_names: list[str | None] = []
        upper_pipeline_invocations = 1

        def execution_observer(_: dict[str, Any], transition: Any) -> None:
            execution_gate_names.append(
                transition.controller_info.get("forcing_gate_semantics")
            )

        with scoped_historical_preview_and_multi_agent_transition(
            execution_observer=execution_observer
        ):
            while not (terminated or truncated):
                step = int(env.steps)
                decisions: dict[int, Any] = {}
                execution_safety_margins: dict[int, float] = {}
                handoff_ids: list[int] = []
                reproposal_ids: list[int] = []
                for agent_id, state in enumerate(supervisor.states):
                    if completed_mask[agent_id]:
                        continue
                    if method in ERR_METHODS:
                        safety = active_direction_safety_margin(
                            env, agent_id, state.active_goal, proposal_config
                        )
                        decision = supervisor.evaluate(
                            agent_id,
                            current_step=step,
                            position=env.dynamics[agent_id].p,
                            active_safety_margin_m=safety.value_m,
                        )
                        decisions[agent_id] = decision
                        execution_safety_margins[agent_id] = float(safety.value_m)
                        trigger_rows.append(
                            {
                                "schema_version": SCHEMA_VERSION,
                                "method": method,
                                "scenario": scenario,
                                "seed": int(seed),
                                "step": step,
                                "time_s": step * float(config["dt"]),
                                "agent_id": agent_id,
                                "active_goal_type": state.active_goal_type,
                                "event": decision.event,
                                "trigger_reasons": list(decision.trigger_reasons),
                                "active_goal_distance_m": decision.active_goal_distance_m,
                                "active_age_s": decision.active_age_s,
                                "progress_rate_mps": decision.progress_rate_mps,
                                "progress_valid": decision.progress_valid,
                                "active_safety_margin_m": decision.active_safety_margin_m,
                                "active_safety_margin_normalized": safety.normalized_value,
                                "active_safety_margin_source": safety.source,
                                "active_safety_margin_unit": safety.unit,
                                "active_direction_sector": safety.sector_index_flat,
                                "raw_obstacle_distance_m": safety.raw_obstacle_distance_m,
                                "guarded_obstacle_distance_m": safety.guarded_obstacle_distance_m,
                                "effective_safe_radius_m": safety.effective_safe_radius_m,
                                "braking_distance_m": safety.braking_distance_m,
                                "phi_tau": decision.phi_tau,
                                "phi_p": decision.phi_p,
                                "phi_h": decision.phi_h,
                                "S_rep": decision.S_rep,
                                "dwell_satisfied": decision.dwell_satisfied,
                                "normal_trigger": decision.normal_trigger,
                                "emergency_trigger": decision.emergency_trigger,
                                "handoff_trigger": decision.handoff_trigger,
                                "reference_completion_trigger": (
                                    decision.reference_completion_trigger
                                ),
                                "terminal_goal_distance_m": (
                                    decision.terminal_goal_distance_m
                                ),
                                "reference_completion_action": (
                                    decision.reference_completion_action
                                ),
                                "emergency_event_semantics": (
                                    decision.emergency_event_semantics
                                ),
                                "emergency_armed_before": (
                                    decision.emergency_armed_before
                                ),
                                "emergency_armed_after": (
                                    decision.emergency_armed_after
                                ),
                                "emergency_rearmed": decision.emergency_rearmed,
                                "previous_active_safety_margin_m": (
                                    decision.previous_active_safety_margin_m
                                ),
                            }
                        )
                        if decision.event == EVENT_REFERENCE_HANDOFF:
                            handoff_ids.append(agent_id)
                        elif decision.event in {
                            EVENT_NORMAL_REPROPOSAL,
                            EVENT_EMERGENCY_REPROPOSAL,
                            EVENT_REFERENCE_COMPLETION_REPROPOSAL,
                        }:
                            reproposal_ids.append(agent_id)
                    elif (
                        state.active_goal_type == ACTIVE_GOAL_REFERENCE
                        and np.linalg.norm(
                            env.dynamics[agent_id].p - state.active_goal
                        )
                        <= float(config["handoff_threshold_m"])
                    ):
                        handoff_ids.append(agent_id)

                # Handoff has strict priority and cannot replan on this tick.
                for agent_id in handoff_ids:
                    if candidate_commitment_tiebreak is not None:
                        candidate_commitment_tiebreak.reset_agent(
                            agent_id,
                            step=step,
                            reason=EVENT_REFERENCE_HANDOFF,
                        )
                    state = supervisor.states[agent_id]
                    old_goal = state.active_goal.copy()
                    phase_before = float(env.dmps[agent_id].phase)
                    speed_before = float(np.linalg.norm(env.dynamics[agent_id].v))
                    if open_segments[agent_id] is not None:
                        open_segments[agent_id]["reached"] = True
                        open_segments[agent_id]["end_step"] = step
                        open_segments[agent_id]["end_reason"] = EVENT_REFERENCE_HANDOFF
                        open_segments[agent_id] = None
                    changed = supervisor.handoff_to_terminal(
                        agent_id,
                        current_step=step,
                        position=env.dynamics[agent_id].p,
                    )
                    if crt_filters is None:
                        set_active_goal_preserve_dmp_phase(
                            env.dmps[agent_id],
                            supervisor.states[agent_id].active_goal,
                        )
                    handoff_safety_value = None
                    if method in EDGE_ERR_METHODS:
                        handoff_safety = active_direction_safety_margin(
                            env,
                            agent_id,
                            supervisor.states[agent_id].active_goal,
                            proposal_config,
                        )
                        supervisor.synchronize_emergency_latch(
                            agent_id,
                            active_safety_margin_m=handoff_safety.value_m,
                        )
                        handoff_safety_value = handoff_safety.value_m
                    phase_after = float(env.dmps[agent_id].phase)
                    phase_switch_deltas.append(phase_after - phase_before)
                    speed_after = float(np.linalg.norm(env.dynamics[agent_id].v))
                    handoff_row = _selection_event_row(
                            method=method,
                            scenario=scenario,
                            seed=seed,
                            step=step,
                            agent_id=agent_id,
                            event=EVENT_REFERENCE_HANDOFF,
                            record={},
                            old_goal=old_goal,
                            new_goal=supervisor.states[agent_id].active_goal,
                            new_goal_type=ACTIVE_GOAL_TERMINAL,
                            goal_changed=changed,
                            decision=decisions.get(agent_id),
                            selection=None,
                            phase_before=phase_before,
                            phase_after=phase_after,
                            speed_before=speed_before,
                            speed_after=speed_after,
                        )
                    handoff_row["post_update_active_safety_margin_m"] = (
                        handoff_safety_value
                    )
                    if handoff_safety_value is not None:
                        execution_safety_margins[agent_id] = float(
                            handoff_safety_value
                        )
                    crt_accept_command(
                        agent_id,
                        supervisor.states[agent_id].active_goal,
                        handoff_row,
                    )
                    event_rows.append(handoff_row)

                if method in ERR_METHODS and reproposal_ids:
                    positions_before = env._positions().copy()
                    velocities_before = env._velocities().copy()
                    phases_before = np.asarray(
                        [dmp.phase for dmp in env.dmps], dtype=float
                    )
                    trigger_types = sorted(
                        {decisions[agent_id].event for agent_id in reproposal_ids}
                    )
                    replan_context = (
                        runtime_recorder.scoped_context(
                            planning_decision_index=upper_pipeline_invocations,
                            event_step=step,
                            event_type="|".join(trigger_types),
                            triggered_agent_ids=list(reproposal_ids),
                        )
                        if runtime_recorder is not None
                        else nullcontext()
                    )
                    with replan_context:
                        selection = upper_plan_builder(
                            env=env,
                            config=config,
                            policy=policy,
                            gat_model=gat_model,
                            gat_device=gat_device,
                            scenario=scenario,
                            seed=int(seed),
                            runtime_recorder=runtime_recorder,
                            selector=(
                                "fp_shep"
                                if method in FP_SELECTOR_METHODS
                                else "gat"
                            ),
                        )
                    if candidate_commitment_tiebreak is not None:
                        selection = candidate_commitment_tiebreak.apply_plan(
                            selection,
                            env=env,
                            agent_ids=tuple(reproposal_ids),
                            event_types={
                                agent_id: decisions[agent_id].event
                                for agent_id in reproposal_ids
                            },
                            active_safety_margins_m={
                                agent_id: decisions[agent_id].active_safety_margin_m
                                for agent_id in reproposal_ids
                            },
                            initial=False,
                            step=step,
                        )
                    upper_pipeline_invocations += 1
                    if not np.array_equal(env._positions(), positions_before):
                        raise RuntimeError("upper pipeline changed real positions")
                    if not np.array_equal(env._velocities(), velocities_before):
                        raise RuntimeError("upper pipeline changed real velocities")
                    if not np.array_equal(
                        np.asarray([dmp.phase for dmp in env.dmps]), phases_before
                    ):
                        raise RuntimeError("upper pipeline changed real DMP phase")
                    for agent_id in reproposal_ids:
                        state = supervisor.states[agent_id]
                        old_goal = state.active_goal.copy()
                        record = selection["plan"]["candidate_records"][agent_id]
                        is_reference = bool(selection["plan"]["available"][agent_id])
                        new_goal = np.asarray(
                            selection["plan"]["references"][agent_id], dtype=float
                        )
                        new_type = (
                            ACTIVE_GOAL_REFERENCE
                            if is_reference
                            else ACTIVE_GOAL_TERMINAL
                        )
                        if open_segments[agent_id] is not None:
                            open_segments[agent_id]["end_step"] = step
                            if (
                                decisions[agent_id].event
                                == EVENT_REFERENCE_COMPLETION_REPROPOSAL
                            ):
                                open_segments[agent_id]["reached"] = True
                                open_segments[agent_id]["end_reason"] = (
                                    EVENT_REFERENCE_COMPLETION_REPROPOSAL
                                )
                            else:
                                open_segments[agent_id]["end_reason"] = (
                                    "REPLACED_BEFORE_REACH"
                                )
                            open_segments[agent_id] = None
                        phase_before = float(env.dmps[agent_id].phase)
                        speed_before = float(np.linalg.norm(env.dynamics[agent_id].v))
                        changed = supervisor.update_goal(
                            agent_id,
                            new_goal=new_goal,
                            new_goal_type=new_type,
                            current_step=step,
                            position=env.dynamics[agent_id].p,
                            event=decisions[agent_id].event,
                        )
                        if crt_filters is None:
                            set_active_goal_preserve_dmp_phase(
                                env.dmps[agent_id], new_goal
                            )
                        post_update_safety_value = None
                        if method in EDGE_ERR_METHODS:
                            new_goal_safety = active_direction_safety_margin(
                                env, agent_id, new_goal, proposal_config
                            )
                            supervisor.synchronize_emergency_latch(
                                agent_id,
                                active_safety_margin_m=new_goal_safety.value_m,
                            )
                            post_update_safety_value = new_goal_safety.value_m
                        phase_after = float(env.dmps[agent_id].phase)
                        speed_after = float(np.linalg.norm(env.dynamics[agent_id].v))
                        phase_switch_deltas.append(phase_after - phase_before)
                        if new_type == ACTIVE_GOAL_REFERENCE:
                            segment = {
                                "start_step": step,
                                "goal": new_goal.tolist(),
                                "source_event": decisions[agent_id].event,
                                "reached": False,
                                "end_step": None,
                                "end_reason": None,
                            }
                            reference_segments[agent_id].append(segment)
                            open_segments[agent_id] = segment
                        replan_row = _selection_event_row(
                                method=method,
                                scenario=scenario,
                                seed=seed,
                                step=step,
                                agent_id=agent_id,
                                event=decisions[agent_id].event,
                                record=record,
                                old_goal=old_goal,
                                new_goal=new_goal,
                                new_goal_type=new_type,
                                goal_changed=changed,
                                decision=decisions[agent_id],
                                selection=selection,
                                phase_before=phase_before,
                                phase_after=phase_after,
                                speed_before=speed_before,
                                speed_after=speed_after,
                            )
                        replan_row["post_update_active_safety_margin_m"] = (
                            post_update_safety_value
                        )
                        if post_update_safety_value is not None:
                            execution_safety_margins[agent_id] = float(
                                post_update_safety_value
                            )
                        crt_accept_command(agent_id, new_goal, replan_row)
                        event_rows.append(replan_row)
                    if not np.array_equal(env._positions(), positions_before):
                        raise RuntimeError("goal updates changed positions")
                    if not np.array_equal(env._velocities(), velocities_before):
                        raise RuntimeError("goal updates changed velocities")

                if crt_filters is not None and crt_config is not None:
                    hard_margin_m = float(config["err"]["h_emg_m"])
                    for agent_id, (state, transition_filter) in enumerate(
                        zip(supervisor.states, crt_filters, strict=True)
                    ):
                        if not np.array_equal(
                            transition_filter.command, state.active_goal
                        ):
                            raise RuntimeError(
                                "CRT command diverged from the frozen R-ERR active goal"
                            )
                        update_started = time.perf_counter_ns()
                        error_before = float(
                            np.linalg.norm(
                                transition_filter.command
                                - transition_filter.executed
                            )
                        )
                        executed_margin_m: float | None = None
                        post_step_margin_m: float | None = None
                        command_margin_m: float | None = None
                        bandwidth_override = False
                        instant_replacement = False
                        omega_multiplier = 1.0
                        if (
                            not completed_mask[agent_id]
                            and error_before
                            > float(crt_config.settled_absolute_tolerance_m)
                        ):
                            safety_started = time.perf_counter_ns()
                            executed_margin_m = active_direction_safety_margin(
                                env,
                                agent_id,
                                transition_filter.executed,
                                proposal_config,
                            ).value_m
                            crt_safety_runtime_ns += (
                                time.perf_counter_ns() - safety_started
                            )
                            if executed_margin_m <= hard_margin_m:
                                omega_multiplier = float(
                                    crt_config.safety_bandwidth_multiplier
                                )
                                bandwidth_override = True
                                crt_bandwidth_override_count += 1
                        if completed_mask[agent_id]:
                            transition_step = transition_filter.snapshot()
                        else:
                            transition_step = transition_filter.exact_step(
                                omega_multiplier=omega_multiplier
                            )
                        urgent_existing_event = bool(
                            agent_id in decisions
                            and decisions[agent_id].event
                            == EVENT_EMERGENCY_REPROPOSAL
                        )
                        if bandwidth_override and urgent_existing_event:
                            safety_started = time.perf_counter_ns()
                            post_step_margin_m = active_direction_safety_margin(
                                env,
                                agent_id,
                                transition_filter.executed,
                                proposal_config,
                            ).value_m
                            command_margin_m = active_direction_safety_margin(
                                env,
                                agent_id,
                                transition_filter.command,
                                proposal_config,
                            ).value_m
                            crt_safety_runtime_ns += (
                                time.perf_counter_ns() - safety_started
                            )
                            if (
                                post_step_margin_m <= hard_margin_m
                                and command_margin_m > hard_margin_m
                            ):
                                transition_step = (
                                    transition_filter.direct_replace_with_command()
                                )
                                instant_replacement = True
                                crt_instant_replacement_count += 1
                        set_active_goal_preserve_dmp_phase(
                            env.dmps[agent_id],
                            transition_filter.executed,
                        )
                        update_elapsed_ns = time.perf_counter_ns() - update_started
                        crt_update_runtime_ns += update_elapsed_ns
                        pending = crt_pending_settlements[agent_id]
                        if pending is not None:
                            threshold = float(pending["crt_settle_threshold_m"])
                            if transition_step.command_error_m <= threshold:
                                settle_step = int(step) + 1
                                pending["crt_settle_step"] = settle_step
                                pending["crt_settle_time_s"] = float(
                                    (settle_step - int(pending["step"]))
                                    * float(config["dt"])
                                )
                                crt_pending_settlements[agent_id] = None
                        crt_trace_rows.append(
                            {
                                "schema_version": "continuous_reference_transition_trace_v1",
                                "method": method,
                                "scenario": scenario,
                                "seed": int(seed),
                                "agent_id": int(agent_id),
                                "control_step": int(step),
                                "state_step": int(step) + 1,
                                "state_time_s": float(int(step) + 1)
                                * float(config["dt"]),
                                "g_cmd": transition_filter.command.tolist(),
                                "g_exec": transition_filter.executed.tolist(),
                                "gdot_exec": transition_filter.velocity.tolist(),
                                "command_error_m": transition_step.command_error_m,
                                "filter_speed_mps": transition_step.filter_speed_mps,
                                "omega_rad_s": transition_step.omega_rad_s,
                                "omega_multiplier": omega_multiplier,
                                "executed_margin_m_before": executed_margin_m,
                                "post_step_margin_m": post_step_margin_m,
                                "command_margin_m": command_margin_m,
                                "bandwidth_override": bandwidth_override,
                                "instant_replacement": instant_replacement,
                                "update_runtime_ms": float(update_elapsed_ns / 1.0e6),
                            }
                        )
                    active = np.stack(
                        [item.executed for item in crt_filters]
                    )
                else:
                    active = np.stack(
                        [state.active_goal for state in supervisor.states]
                    )
                if execution_acceleration_limiter is not None:
                    for agent_id, state in enumerate(supervisor.states):
                        if completed_mask[agent_id]:
                            continue
                        if agent_id not in execution_safety_margins:
                            raise RuntimeError(
                                "limiter requires the existing per-step active-direction safety margin"
                            )
                        if hasattr(execution_acceleration_limiter, "set_execution_velocity"):
                            execution_acceleration_limiter.set_execution_velocity(
                                agent_id,
                                np.asarray(env._velocities()[agent_id], dtype=float),
                            )
                        execution_acceleration_limiter.set_context(
                            agent_id,
                            step=step,
                            safety_margin_m=execution_safety_margins[agent_id],
                            time_since_reference_change_s=max(
                                0,
                                step - int(state.last_reference_update_step),
                            )
                            * float(config["dt"]),
                        )
                if sac_observation_extension is None:
                    observations = temporary_checkpoint_observations(env, active)
                else:
                    switch_ages_s = np.asarray(
                        [
                            max(0, step - int(state.last_reference_update_step))
                            * float(config["dt"])
                            for state in supervisor.states
                        ],
                        dtype=float,
                    )
                    observations = task_aware_checkpoint_observations(
                        env,
                        active,
                        switch_ages_s=switch_ages_s,
                        previous_applied_accelerations=previous_applied_accelerations,
                        task_distance_scale_m=float(
                            sac_observation_extension.get("task_distance_scale_m", 100.0)
                        ),
                        switch_age_scale_s=float(
                            sac_observation_extension.get("switch_age_scale_s", 5.0)
                        ),
                        acceleration_scale_mps2=float(
                            sac_observation_extension.get("acceleration_scale_mps2", 4.0)
                        ),
                    )
                execution_context = (
                    runtime_recorder.scoped_context(event_step=step)
                    if runtime_recorder is not None
                    else nullcontext()
                )
                actor_context = (
                    policy.timing_mode("execution_actor")
                    if runtime_recorder is not None and hasattr(policy, "timing_mode")
                    else nullcontext()
                )
                with execution_context, actor_context:
                    actions, _ = policy.predict(observations, deterministic=True)
                actions = np.asarray(actions, dtype=np.float32)
                if actions.shape != tuple(env.action_shape):
                    raise RuntimeError("frozen SAC returned an unexpected action shape")
                saturation_values.extend(
                    action_saturation_mask(
                        actions,
                        env.action_space.low,
                        env.action_space.high,
                        relative_tolerance=float(
                            settings["action_saturation"]["relative_tolerance"]
                        ),
                    )
                    .astype(float)
                    .reshape(-1)
                    .tolist()
                )
                dmp_context = (
                    runtime_recorder.dmp_mode("execution_dmp")
                    if runtime_recorder is not None
                    else nullcontext()
                )
                dmp_step_context = (
                    runtime_recorder.scoped_context(event_step=step)
                    if runtime_recorder is not None
                    else nullcontext()
                )
                with dmp_step_context, dmp_context:
                    _, _, terminated, truncated, info = env.step(actions)
                positions.append(env._positions().copy())
                velocities.append(env._velocities().copy())
                applied = np.asarray(info["applied_accelerations"], dtype=float)
                commanded = np.asarray(info["commanded_accelerations"], dtype=float)
                previous_applied_accelerations = applied.copy()
                accelerations.append(applied.copy())
                applied_norms.extend(np.linalg.norm(applied, axis=1).tolist())
                commanded_norms.extend(np.linalg.norm(commanded, axis=1).tolist())
                min_clearances.append(float(np.min(info["min_clearances"])))
                min_inter_agent_distances.append(
                    float(info["min_inter_agent_distance"])
                )
                static_clearances = np.minimum(
                    static_clearances,
                    minimum_static_surface_clearances(
                        env._positions(), env.static_obstacles
                    ),
                )
                collision |= bool(info["collision"])
                obstacle_collision |= bool(np.any(info["obstacle_collision_mask"]))
                inter_agent_collision |= bool(
                    np.any(info["inter_agent_collision_mask"])
                )
                obstacle_collision_mask |= np.asarray(
                    info["obstacle_collision_mask"], dtype=bool
                )
                inter_agent_collision_mask |= np.asarray(
                    info["inter_agent_collision_mask"], dtype=bool
                )
                success_mask = np.asarray(info["success_mask"], dtype=bool)
                for agent_id in np.flatnonzero(success_mask):
                    completed_mask[int(agent_id)] = True
                    if completion_steps[int(agent_id)] is None:
                        completion_steps[int(agent_id)] = int(env.steps)
                path_rows.extend(
                    _trajectory_rows(
                        method=method,
                        scenario=scenario,
                        seed=seed,
                        step=int(env.steps),
                        env=env,
                        supervisor=supervisor,
                        policy_actions=actions,
                        applied_accelerations=applied,
                        commanded_accelerations=commanded,
                        step_info=info,
                        executed_references=active,
                        executed_reference_velocities=(
                            np.stack([item.velocity for item in crt_filters])
                            if crt_filters is not None
                            else None
                        ),
                    )
                )

        for agent_id, segment in enumerate(open_segments):
            if segment is not None:
                segment["end_step"] = int(env.steps)
                segment["end_reason"] = "EPISODE_END_UNREACHED"
        for pending in crt_pending_settlements:
            if pending is not None:
                pending["crt_settle_censored_by_episode_end"] = True
        if phase_switch_deltas and not np.allclose(
            phase_switch_deltas, 0.0, rtol=0.0, atol=0.0
        ):
            raise RuntimeError("active-goal update changed DMP phase")
        if not np.array_equal(env.goals, terminal_goals):
            raise RuntimeError("terminal task goals changed")
        if execution_gate_names and not all(
            name == HISTORICAL_GATE_NAME for name in execution_gate_names
        ):
            raise RuntimeError("real execution did not use historical vector gate")

        position_array = np.stack(positions)
        velocity_array = np.stack(velocities)
        acceleration_array = np.stack(accelerations)
        metrics = trajectory_metrics(
            position_array,
            velocity_array,
            acceleration_array,
            dt=float(config["dt"]),
        )
        final_positions = position_array[-1]
        initial_terminal_distances = np.linalg.norm(terminal_goals - starts, axis=1)
        final_terminal_distances = np.linalg.norm(
            terminal_goals - final_positions, axis=1
        )
        terminal_progress = initial_terminal_distances - final_terminal_distances
        team_success = bool(info.get("success", False))
        reason = termination_reason(
            success=team_success,
            collision=collision,
            terminated=bool(terminated),
            truncated=bool(truncated),
        )
        replans = sum(
            state.number_of_reproposal_events for state in supervisor.states
        )
        emergencies = sum(
            state.number_of_emergency_events for state in supervisor.states
        )
        handoffs = sum(state.number_of_reference_handoffs for state in supervisor.states)
        reference_selected = sum(len(items) for items in reference_segments)
        reference_reached = sum(
            int(segment["reached"])
            for items in reference_segments
            for segment in items
        )
        agent_rows: list[dict[str, Any]] = []
        path_lengths = np.asarray(metrics["path_lengths"], dtype=float)
        speed_norms = np.linalg.norm(velocity_array, axis=2)
        acceleration_norms = np.linalg.norm(acceleration_array, axis=2)
        for agent_id, state in enumerate(supervisor.states):
            selections = len(reference_segments[agent_id])
            reached = sum(
                int(segment["reached"]) for segment in reference_segments[agent_id]
            )
            agent_success = bool(
                completion_steps[agent_id] is not None
                or final_terminal_distances[agent_id]
                <= float(env.env_config.goal_tolerance)
            )
            agent_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "method": method,
                    "scenario": scenario,
                    "seed": int(seed),
                    "agent_id": agent_id,
                    "success": agent_success,
                    "terminal_completion_step": completion_steps[agent_id],
                    "collision": bool(
                        obstacle_collision_mask[agent_id]
                        or inter_agent_collision_mask[agent_id]
                    ),
                    "obstacle_collision": bool(obstacle_collision_mask[agent_id]),
                    "inter_agent_collision": bool(
                        inter_agent_collision_mask[agent_id]
                    ),
                    "timeout": bool(truncated and not agent_success),
                    "terminal_progress_m": float(terminal_progress[agent_id]),
                    "final_terminal_distance_m": float(
                        final_terminal_distances[agent_id]
                    ),
                    "path_length_m": float(path_lengths[agent_id]),
                    "mean_speed_mps": float(np.mean(speed_norms[:, agent_id])),
                    "peak_speed_mps": float(np.max(speed_norms[:, agent_id])),
                    "mean_acceleration_mps2": float(
                        np.mean(acceleration_norms[:, agent_id])
                    ),
                    "peak_acceleration_mps2": float(
                        np.max(acceleration_norms[:, agent_id])
                    ),
                    "minimum_static_obstacle_clearance_m": float(
                        static_clearances[agent_id]
                    ),
                    "reference_selection_count": selections,
                    "reference_reached_count": reached,
                    "reference_executability_rate": (
                        float(reached / selections) if selections else None
                    ),
                    "replanning_count": state.number_of_reproposal_events,
                    "emergency_replanning_count": state.number_of_emergency_events,
                    "handoff_count": state.number_of_reference_handoffs,
                    "trigger_check_count": state.number_of_trigger_checks,
                    "final_active_goal_type": state.active_goal_type,
                    "reference_segments": reference_segments[agent_id],
                    "team_success": team_success,
                }
            )
        episode_actor_rows = (
            runtime_recorder.actor_rows[actor_row_start:]
            if runtime_recorder is not None
            else []
        )
        episode_dmp_rows = (
            runtime_recorder.dmp_rows[dmp_row_start:]
            if runtime_recorder is not None
            else []
        )
        episode_upper_rows = (
            runtime_recorder.upper_rows[upper_row_start:]
            if runtime_recorder is not None
            else []
        )
        execution_actor_rows = [
            row for row in episode_actor_rows if row["actor_mode"] == "execution_actor"
        ]
        preview_actor_rows = [
            row
            for row in episode_actor_rows
            if row["actor_mode"] == "fp_shep_preview_actor"
        ]
        cumulative_upper_ms = float(
            sum(row["upper_planning_total_ms"] for row in episode_upper_rows)
        )
        execution_actor_ms = float(sum(row["runtime_ms"] for row in execution_actor_rows))
        execution_dmp_ms = float(sum(row["runtime_ms"] for row in episode_dmp_rows))
        crt_total_ms = float(crt_update_runtime_ns / 1.0e6)
        crt_safety_ms = float(crt_safety_runtime_ns / 1.0e6)
        crt_filter_only_ms = float(max(0.0, crt_total_ms - crt_safety_ms))
        crt_errors = np.asarray(
            [row["command_error_m"] for row in crt_trace_rows], dtype=float
        )
        crt_filter_speeds = np.asarray(
            [row["filter_speed_mps"] for row in crt_trace_rows], dtype=float
        )
        crt_settle_times = np.asarray(
            [
                float(row["crt_settle_time_s"])
                for row in event_rows
                if row.get("crt_settle_time_s") is not None
                and row.get("event") != INITIAL_SELECTION
            ],
            dtype=float,
        )
        episode = {
            "schema_version": SCHEMA_VERSION,
            "method": method,
            "method_display_name": METHOD_DISPLAY[method],
            "scenario": scenario,
            "seed": int(seed),
            "num_agents": int(env.num_agents),
            "dt": float(config["dt"]),
            "pair_id": f"{scenario}__seed{int(seed):03d}",
            "initial_condition_hash": initial_hash,
            "initial_selection_plan_hash": plan["selection_plan_hash"],
            "initial_selection_semantic_hash": initial_selection_semantic_hash,
            "initial_candidate_bundle_hash": initial_selection[
                "candidate_bundle_hash"
            ],
            "team_success": team_success,
            "agent_completion_rate": float(
                np.mean([row["success"] for row in agent_rows])
            ),
            "collision": collision,
            "obstacle_collision": obstacle_collision,
            "inter_agent_collision": inter_agent_collision,
            "timeout": bool(truncated),
            "termination_reason": reason,
            "steps": int(env.steps),
            "completion_time_s": (
                float(env.steps) * float(config["dt"]) if team_success else None
            ),
            "team_path_length_m": float(metrics["path_length_team_sum"]),
            "team_path_length_mean_agent_m": float(
                metrics["path_length_team_mean"]
            ),
            "trajectory_smoothness": float(
                metrics["trajectory_smoothness_team_mean"]
            ),
            "terminal_progress_team_mean_m": float(np.mean(terminal_progress)),
            "minimum_obstacle_clearance_m": float(np.min(min_clearances)),
            "minimum_static_obstacle_clearance_m": float(
                np.min(static_clearances)
            ),
            "minimum_inter_agent_distance_m": float(
                np.min(min_inter_agent_distances)
            ),
            "mean_applied_acceleration_mps2": float(np.mean(applied_norms)),
            "max_applied_acceleration_mps2": float(np.max(applied_norms)),
            "mean_commanded_acceleration_mps2": float(np.mean(commanded_norms)),
            "max_commanded_acceleration_mps2": float(np.max(commanded_norms)),
            "action_saturation_rate": float(np.mean(saturation_values)),
            "reference_selection_count": reference_selected,
            "reference_reached_count": reference_reached,
            "reference_executability_rate": (
                float(reference_reached / reference_selected)
                if reference_selected
                else None
            ),
            "replanning_count": replans,
            "emergency_replanning_count": emergencies,
            "handoff_count": handoffs,
            "trigger_check_count": sum(
                state.number_of_trigger_checks for state in supervisor.states
            ),
            "upper_pipeline_invocation_count": upper_pipeline_invocations,
            "planning_decision_count": len(episode_upper_rows),
            "proposal_generation_ms": float(
                sum(row["proposal_generation_ms"] for row in episode_upper_rows)
            ),
            "coarse_ranking_ms": float(
                sum(row["coarse_ranking_ms"] for row in episode_upper_rows)
            ),
            "fp_shep_total_ms": float(
                sum(row["fp_shep_total_ms"] for row in episode_upper_rows)
            ),
            "fp_shep_preview_actor_ms": float(
                sum(row["runtime_ms"] for row in preview_actor_rows)
            ),
            "graph_build_ms": float(
                sum(row["graph_build_ms"] for row in episode_upper_rows)
            ),
            "gat_forward_ms": float(
                sum(row["gat_forward_ms"] for row in episode_upper_rows)
            ),
            "upper_planning_total_ms": cumulative_upper_ms,
            "execution_actor_forward_ms": execution_actor_ms,
            "execution_actor_call_count": len(execution_actor_rows),
            "execution_dmp_ms": execution_dmp_ms,
            "execution_dmp_call_count": len(episode_dmp_rows),
            "total_online_algorithm_compute_ms": (
                cumulative_upper_ms
                + execution_actor_ms
                + execution_dmp_ms
                + crt_total_ms
            ),
            "crt_enabled": crt_filters is not None,
            "crt_type": (
                "SECOND_ORDER_CRITICALLY_DAMPED"
                if crt_filters is not None
                else None
            ),
            "crt_settling_time_s": (
                float(crt_config.settling_time_s)
                if crt_config is not None
                else None
            ),
            "crt_omega_rad_s": (
                float(crt_config.omega_rad_s)
                if crt_config is not None
                else None
            ),
            "crt_mean_command_execution_error_m": (
                float(np.mean(crt_errors)) if crt_errors.size else None
            ),
            "crt_p95_command_execution_error_m": (
                float(np.percentile(crt_errors, 95)) if crt_errors.size else None
            ),
            "crt_mean_filter_speed_mps": (
                float(np.mean(crt_filter_speeds))
                if crt_filter_speeds.size
                else None
            ),
            "crt_p95_filter_speed_mps": (
                float(np.percentile(crt_filter_speeds, 95))
                if crt_filter_speeds.size
                else None
            ),
            "crt_mean_settle_time_s": (
                float(np.mean(crt_settle_times))
                if crt_settle_times.size
                else None
            ),
            "crt_p95_settle_time_s": (
                float(np.percentile(crt_settle_times, 95))
                if crt_settle_times.size
                else None
            ),
            "crt_settled_event_count": int(crt_settle_times.size),
            "crt_bandwidth_override_count": int(
                crt_bandwidth_override_count
            ),
            "crt_instant_replacement_count": int(
                crt_instant_replacement_count
            ),
            "crt_total_runtime_ms": crt_total_ms,
            "crt_safety_evaluation_runtime_ms": crt_safety_ms,
            "crt_filter_only_runtime_ms": crt_filter_only_ms,
            "crt_runtime_per_control_step_ms": (
                float(crt_total_ms / int(env.steps)) if int(env.steps) else None
            ),
            "crt_filter_only_runtime_per_agent_step_ms": (
                float(crt_filter_only_ms / len(crt_trace_rows))
                if crt_trace_rows
                else None
            ),
            "crt_required_state_bytes": (
                int(env.num_agents) * 6 * 8 if crt_filters is not None else 0
            ),
            "crt_implementation_state_bytes": (
                int(env.num_agents) * 9 * 8 if crt_filters is not None else 0
            ),
            "phase_reset_on_switch": False,
            "maximum_phase_switch_delta": float(
                np.max(np.abs(phase_switch_deltas))
            ),
            "terminal_task_goals_unchanged": True,
            "historical_gate": HISTORICAL_GATE_NAME,
            "execution_historical_gate_verified": bool(execution_gate_names)
            and all(name == HISTORICAL_GATE_NAME for name in execution_gate_names),
            "preview_historical_gate_verified": initial_selection[
                "preview_historical_gate_verified"
            ],
            "proposal_reconstruction_equivalent": True,
            "graph_schema_match": initial_selection["graph_schema_match"],
            "selector": (
                "fp_shep" if method in FP_SELECTOR_METHODS else "gat"
            ),
            "emergency_event_semantics": supervisor.emergency_event_semantics,
            "GAT_checkpoint_used": method not in FP_SELECTOR_METHODS,
            "FP_SHEP_selector_used": True,
            "retraining_performed": False,
            "episode_runtime_ms": float(
                (time.perf_counter_ns() - started) / 1.0e6
            ),
            **scene_metadata,
        }
        for row in event_rows:
            row["episode_team_success"] = team_success
            row["episode_termination_reason"] = reason
            row["steps_after_event"] = int(env.steps) - int(row["step"])
        scene_record = {
            "method": method,
            "scenario": scenario,
            "seed": int(seed),
            "initial_condition_hash": initial_hash,
            "snapshot": snapshot,
        }
        return episode, agent_rows, event_rows, trigger_rows, {
            "path_rows": path_rows,
            "crt_trace_rows": crt_trace_rows,
            "jerk_limiter_trace_rows": (
                execution_acceleration_limiter.trace_rows()
                if execution_acceleration_limiter is not None
                else []
            ),
            "direction_continuity_trace_rows": (
                candidate_commitment_tiebreak.trace_rows()
                if candidate_commitment_tiebreak is not None
                else []
            ),
            "scene_record": scene_record,
            "actor_timing_rows": [dict(row) for row in episode_actor_rows],
            "dmp_timing_rows": [dict(row) for row in episode_dmp_rows],
            "upper_timing_rows": [dict(row) for row in episode_upper_rows],
        }
    finally:
        env.close()


def _finite(values: Iterable[Any]) -> np.ndarray:
    result = np.asarray(
        [float(value) for value in values if value is not None], dtype=float
    )
    return result[np.isfinite(result)]


def aggregate_summary(
    episodes: Sequence[Mapping[str, Any]],
    agents: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in (METHOD_ONE_SHOT, METHOD_ERR):
        for scope in [*sorted({row["scenario"] for row in episodes}), "overall"]:
            members = [
                row
                for row in episodes
                if row["method"] == method
                and (scope == "overall" or row["scenario"] == scope)
            ]
            agent_members = [
                row
                for row in agents
                if row["method"] == method
                and (scope == "overall" or row["scenario"] == scope)
            ]
            replan_values = _finite(row["replanning_count"] for row in members)
            selected = sum(int(row["reference_selection_count"]) for row in members)
            reached = sum(int(row["reference_reached_count"]) for row in members)
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "method": method,
                    "scenario": scope,
                    "episode_count": len(members),
                    "team_success_rate": float(
                        np.mean([row["team_success"] for row in members])
                    ),
                    "agent_completion_rate": float(
                        np.mean([row["success"] for row in agent_members])
                    ),
                    "collision_rate": float(
                        np.mean([row["collision"] for row in members])
                    ),
                    "obstacle_collision_rate": float(
                        np.mean([row["obstacle_collision"] for row in members])
                    ),
                    "inter_agent_collision_rate": float(
                        np.mean([row["inter_agent_collision"] for row in members])
                    ),
                    "timeout_rate": float(
                        np.mean([row["timeout"] for row in members])
                    ),
                    "mean_terminal_progress_m": float(
                        np.mean(
                            [row["terminal_progress_team_mean_m"] for row in members]
                        )
                    ),
                    "mean_team_path_length_m": float(
                        np.mean([row["team_path_length_m"] for row in members])
                    ),
                    "reference_selection_count": selected,
                    "reference_reached_count": reached,
                    "reference_executability_rate": (
                        float(reached / selected) if selected else None
                    ),
                    "mean_replans_per_episode": float(np.mean(replan_values)),
                    "p50_replans_per_episode": float(
                        np.percentile(replan_values, 50)
                    ),
                    "p90_replans_per_episode": float(
                        np.percentile(replan_values, 90)
                    ),
                    "max_replans_per_episode": int(np.max(replan_values)),
                    "mean_handoffs_per_episode": float(
                        np.mean([row["handoff_count"] for row in members])
                    ),
                }
            )
    return rows


def build_paired(episodes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_pair: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in episodes:
        by_pair[str(row["pair_id"])][str(row["method"])] = row
    rows: list[dict[str, Any]] = []
    for pair_id, methods in sorted(by_pair.items()):
        if set(methods) != {METHOD_ONE_SHOT, METHOD_ERR}:
            raise RuntimeError(f"incomplete method pair: {pair_id}")
        one = methods[METHOD_ONE_SHOT]
        err = methods[METHOD_ERR]
        if one["initial_condition_hash"] != err["initial_condition_hash"]:
            raise RuntimeError("paired initial condition mismatch")
        if one["initial_candidate_bundle_hash"] != err["initial_candidate_bundle_hash"]:
            raise RuntimeError("paired initial candidate bundle mismatch")
        if one["initial_selection_semantic_hash"] != err["initial_selection_semantic_hash"]:
            raise RuntimeError("paired initial GAT decision mismatch")
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "pair_id": pair_id,
                "scenario": one["scenario"],
                "seed": one["seed"],
                "initial_condition_hash": one["initial_condition_hash"],
                "initial_candidate_bundle_hash": one[
                    "initial_candidate_bundle_hash"
                ],
                "initial_selection_semantic_hash": one[
                    "initial_selection_semantic_hash"
                ],
                "raw_plan_hash_equal": one["initial_selection_plan_hash"]
                == err["initial_selection_plan_hash"],
                "one_shot_team_success": one["team_success"],
                "err_team_success": err["team_success"],
                "team_success_delta": int(err["team_success"])
                - int(one["team_success"]),
                "one_shot_collision": one["collision"],
                "err_collision": err["collision"],
                "collision_delta": int(err["collision"]) - int(one["collision"]),
                "one_shot_timeout": one["timeout"],
                "err_timeout": err["timeout"],
                "timeout_delta": int(err["timeout"]) - int(one["timeout"]),
                "one_shot_agent_completion_rate": one["agent_completion_rate"],
                "err_agent_completion_rate": err["agent_completion_rate"],
                "agent_completion_delta": err["agent_completion_rate"]
                - one["agent_completion_rate"],
                "one_shot_reference_executability": one[
                    "reference_executability_rate"
                ],
                "err_reference_executability": err[
                    "reference_executability_rate"
                ],
                "terminal_progress_delta_m": err["terminal_progress_team_mean_m"]
                - one["terminal_progress_team_mean_m"],
                "path_length_delta_m": err["team_path_length_m"]
                - one["team_path_length_m"],
                "err_replanning_count": err["replanning_count"],
                "recovered_one_shot_failure": bool(
                    not one["team_success"] and err["team_success"]
                ),
                "regressed_one_shot_success": bool(
                    one["team_success"] and not err["team_success"]
                ),
            }
        )
    return rows


def _sanity_audit(
    trigger_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    checks = len(trigger_rows)
    replans = sum(
        row["event"]
        in {
            EVENT_NORMAL_REPROPOSAL,
            EVENT_EMERGENCY_REPROPOSAL,
            EVENT_REFERENCE_COMPLETION_REPROPOSAL,
        }
        for row in event_rows
    )
    event_rate = float(replans / checks) if checks else 0.0
    clearly_always = event_rate >= float(
        config["sanity_gate"]["clearly_always_trigger_event_rate"]
    )
    clearly_never = replans == 0 and event_rate < float(
        config["sanity_gate"]["clearly_never_trigger_event_rate"]
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "sanity_scenarios": list(config["development_scenarios"]),
        "sanity_seed": int(config["sanity_seed"]),
        "trigger_check_count": checks,
        "reproposal_event_count": int(replans),
        "reproposal_event_rate": event_rate,
        "emergency_event_count": int(
            sum(row["event"] == EVENT_EMERGENCY_REPROPOSAL for row in event_rows)
        ),
        "handoff_event_count": int(
            sum(row["event"] == EVENT_REFERENCE_HANDOFF for row in event_rows)
        ),
        "reference_completion_reproposal_count": int(
            sum(
                row["event"] == EVENT_REFERENCE_COMPLETION_REPROPOSAL
                for row in event_rows
            )
        ),
        "progress_valid_rate": float(
            np.mean([row["progress_valid"] for row in trigger_rows])
        )
        if trigger_rows
        else 0.0,
        "phi_tau_nonnegative_rate": float(
            np.mean([row["phi_tau"] >= 0.0 for row in trigger_rows])
        )
        if trigger_rows
        else 0.0,
        "phi_p_nonnegative_rate_when_valid": float(
            np.mean(
                [
                    row["phi_p"] >= 0.0
                    for row in trigger_rows
                    if row["phi_p"] is not None
                ]
            )
        )
        if any(row["phi_p"] is not None for row in trigger_rows)
        else None,
        "phi_h_nonnegative_rate": float(
            np.mean([row["phi_h"] >= 0.0 for row in trigger_rows])
        )
        if trigger_rows
        else 0.0,
        "clearly_always_trigger": clearly_always,
        "clearly_never_trigger_and_unusable": clearly_never,
        "TRIGGER_SCALE_VALID": "YES"
        if not clearly_always and not clearly_never
        else "NO",
        "sanity_correction_count": 0,
        "parameter_search_performed": False,
    }


def _chattering(event_rows: Sequence[Mapping[str, Any]], dwell_s: float) -> dict[str, Any]:
    by_agent: dict[tuple[str, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in event_rows:
        if row["event"] in {
            EVENT_NORMAL_REPROPOSAL,
            EVENT_EMERGENCY_REPROPOSAL,
            EVENT_REFERENCE_COMPLETION_REPROPOSAL,
        }:
            by_agent[(row["scenario"], int(row["seed"]), int(row["agent_id"]))].append(
                row
            )
    intervals: list[float] = []
    rapid_bursts: list[dict[str, Any]] = []
    violating: list[dict[str, Any]] = []
    for key, rows in by_agent.items():
        ordered = sorted(rows, key=lambda row: int(row["step"]))
        for first, second in zip(ordered[:-1], ordered[1:]):
            interval = float(second["time_s"]) - float(first["time_s"])
            intervals.append(interval)
            if interval < float(dwell_s):
                record = {
                    "scenario": key[0],
                    "seed": key[1],
                    "agent_id": key[2],
                    "interval_s": interval,
                    "second_event": second["event"],
                    "emergency_dwell_bypass": second["event"]
                    == EVENT_EMERGENCY_REPROPOSAL,
                }
                rapid_bursts.append(record)
                if second["event"] != EVENT_EMERGENCY_REPROPOSAL:
                    violating.append(record)
    return {
        # Emergency events are allowed to bypass dwell semantically, but a
        # repeated sub-dwell burst is still reported as observed chattering.
        "CHATTERING_PRESENT": "YES" if rapid_bursts else "NO",
        "rapid_sub_dwell_interval_count": len(rapid_bursts),
        "emergency_bypass_burst_count": sum(
            int(row["emergency_dwell_bypass"]) for row in rapid_bursts
        ),
        "non_emergency_dwell_violation_count": len(violating),
        "replanning_interval_mean_s": float(np.mean(intervals)) if intervals else None,
        "replanning_interval_p50_s": float(np.percentile(intervals, 50))
        if intervals
        else None,
        "replanning_interval_p90_s": float(np.percentile(intervals, 90))
        if intervals
        else None,
        "replanning_interval_max_s": float(np.max(intervals)) if intervals else None,
        "non_emergency_violations": violating,
        "rapid_bursts": rapid_bursts,
    }


def build_conclusion(
    summaries: Sequence[Mapping[str, Any]],
    paired: Sequence[Mapping[str, Any]],
    sanity: Mapping[str, Any],
    chatter: Mapping[str, Any],
) -> dict[str, Any]:
    one = next(
        row
        for row in summaries
        if row["method"] == METHOD_ONE_SHOT and row["scenario"] == "overall"
    )
    err = next(
        row
        for row in summaries
        if row["method"] == METHOD_ERR and row["scenario"] == "overall"
    )
    multi_one = next(
        row
        for row in summaries
        if row["method"] == METHOD_ONE_SHOT and row["scenario"] == "multi_agent"
    )
    multi_err = next(
        row
        for row in summaries
        if row["method"] == METHOD_ERR and row["scenario"] == "multi_agent"
    )
    success_gain = float(err["team_success_rate"] - one["team_success_rate"])
    collision_change = float(err["collision_rate"] - one["collision_rate"])
    timeout_change = float(err["timeout_rate"] - one["timeout_rate"])
    exec_gain = float(
        (err["reference_executability_rate"] or 0.0)
        - (one["reference_executability_rate"] or 0.0)
    )
    recovered = sum(int(row["recovered_one_shot_failure"]) for row in paired)
    regressed = sum(int(row["regressed_one_shot_success"]) for row in paired)
    if success_gain >= 0.1 or (
        collision_change <= -0.1 and timeout_change <= 0.05
    ):
        gain = "YES"
    elif recovered > regressed and success_gain > 0.0:
        gain = "WEAK"
    else:
        gain = "NO"
    reference_executability_gain = (
        "YES" if exec_gain >= 0.1 else "WEAK" if exec_gain > 0.0 else "NO"
    )
    recovery_signal = (
        "YES"
        if recovered > regressed and success_gain >= 0.1
        else "WEAK"
        if recovered > regressed
        else "NO"
    )
    low_level_limitation = (
        "YES"
        if err["reference_executability_rate"] is not None
        and err["reference_executability_rate"] < 0.9
        else "WEAK"
        if chatter["CHATTERING_PRESENT"] == "YES"
        else "NO"
    )
    multi_success_gain = float(
        multi_err["team_success_rate"] - multi_one["team_success_rate"]
    )
    multi_collision_change = float(
        multi_err["collision_rate"] - multi_one["collision_rate"]
    )
    multi_agent_recovery = (
        "YES"
        if multi_success_gain >= 0.1
        or (multi_collision_change <= -0.1 and multi_success_gain >= 0.0)
        else "WEAK"
        if multi_success_gain > 0.0 and multi_collision_change <= 0.0
        else "NO"
    )
    pipeline_valid = sanity["TRIGGER_SCALE_VALID"] == "YES"
    next_step = (
        "FAILURE_AUDIT"
        if not pipeline_valid
        else "FORMAL_ERR_EVALUATION"
        if gain == "YES"
        else "STOP_AND_KEEP_ONE_SHOT"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "ACTIVE_SAFETY_MARGIN_SOURCE": ACTIVE_SAFETY_MARGIN_SOURCE,
        "TRIGGER_SCALE_VALID": sanity["TRIGGER_SCALE_VALID"],
        "ERR_PIPELINE_VALID": "YES" if pipeline_valid else "NO",
        "ONE_SHOT_TEAM_SUCCESS": one["team_success_rate"],
        "ERR_TEAM_SUCCESS": err["team_success_rate"],
        "SUCCESS_GAIN_PP": 100.0 * success_gain,
        "COLLISION_CHANGE_PP": 100.0 * collision_change,
        "TIMEOUT_CHANGE_PP": 100.0 * timeout_change,
        "MEAN_REPLANS_PER_EPISODE": err["mean_replans_per_episode"],
        "P50_REPLANS_PER_EPISODE": err["p50_replans_per_episode"],
        "P90_REPLANS_PER_EPISODE": err["p90_replans_per_episode"],
        "MAX_REPLANS_PER_EPISODE": err["max_replans_per_episode"],
        "CHATTERING_PRESENT": chatter["CHATTERING_PRESENT"],
        "REFERENCE_EXECUTABILITY_GAIN": reference_executability_gain,
        "REFERENCE_EXECUTABILITY_CHANGE_PP": 100.0 * exec_gain,
        "REPLANNING_RECOVERY_SIGNAL": recovery_signal,
        "RECOVERED_ONE_SHOT_FAILURE_COUNT": recovered,
        "REGRESSED_ONE_SHOT_SUCCESS_COUNT": regressed,
        "LOW_LEVEL_EXECUTION_LIMITATION_REMAINS": low_level_limitation,
        "MULTI_AGENT_RECOVERY": multi_agent_recovery,
        "ERR_CLOSED_LOOP_GAIN": gain,
        "RECOMMENDED_NEXT_STEP": next_step,
        "development_only_not_formal_claim": True,
    }


def _render_report(
    summaries: Sequence[Mapping[str, Any]],
    conclusion: Mapping[str, Any],
    sanity: Mapping[str, Any],
    chatter: Mapping[str, Any],
) -> str:
    lines = [
        "# GAT-V1 Event-Triggered Reference Reconstruction 开发闭环报告",
        "",
        "本次仅比较冻结的 GAT-V1 one-shot 与同一模型加 ERR；未训练、未调参，也未修改 Proposal、FP-SHEP、GAT、SAC-DMP、reward 或 environment core。",
        "",
        "## 总体结果",
        "",
        "| 方法 | 团队成功率 | 碰撞率 | 超时率 | 单机完成率 | 航点可执行率 | 平均重规划 | P90 重规划 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in (METHOD_ONE_SHOT, METHOD_ERR):
        row = next(
            item
            for item in summaries
            if item["method"] == method and item["scenario"] == "overall"
        )
        executable = row["reference_executability_rate"]
        lines.append(
            f"| {METHOD_DISPLAY[method]} | {row['team_success_rate']:.1%} | "
            f"{row['collision_rate']:.1%} | {row['timeout_rate']:.1%} | "
            f"{row['agent_completion_rate']:.1%} | "
            f"{executable:.1%} | {row['mean_replans_per_episode']:.2f} | "
            f"{row['p90_replans_per_episode']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## ERR 触发审计",
            "",
            f"- 安全裕度来源：`{ACTIVE_SAFETY_MARGIN_SOURCE}`（m，按 Proposal `h_max` 归一化）。",
            f"- sanity 触发检查 {sanity['trigger_check_count']} 次，重规划 {sanity['reproposal_event_count']} 次，事件率 {sanity['reproposal_event_rate']:.2%}。",
            f"- `TRIGGER_SCALE_VALID = {sanity['TRIGGER_SCALE_VALID']}`；`CHATTERING_PRESENT = {chatter['CHATTERING_PRESENT']}`。",
            "- 每次更新均保持位置、速度、DMP phase、LiDAR 历史和环境时间连续；handoff 优先于同 tick 重规划。",
            "",
            "## 结论",
            "",
            f"- one-shot 团队成功率：{conclusion['ONE_SHOT_TEAM_SUCCESS']:.1%}",
            f"- ERR 团队成功率：{conclusion['ERR_TEAM_SUCCESS']:.1%}",
            f"- 成功率变化：{conclusion['SUCCESS_GAIN_PP']:+.1f} pp；碰撞变化：{conclusion['COLLISION_CHANGE_PP']:+.1f} pp；超时变化：{conclusion['TIMEOUT_CHANGE_PP']:+.1f} pp。",
            f"- `ERR_CLOSED_LOOP_GAIN = {conclusion['ERR_CLOSED_LOOP_GAIN']}`",
            f"- `RECOMMENDED_NEXT_STEP = {conclusion['RECOMMENDED_NEXT_STEP']}`",
            "",
            "这是 5 seeds/scenario 的受控 development 结果，不是正式论文结论。规划候选、初始/重构航点与真实执行轨迹保存在 `path_visualization_data.json` 和 `trajectory_points.csv`。",
        ]
    )
    return "\n".join(lines) + "\n"


def _semantic_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "state_per_agent": [
            "active_goal",
            "terminal_goal_immutable",
            "active_goal_type",
            "last_reference_update_step",
            "progress_distance_history",
        ],
        "forbidden_state_absent": ["pending_reference", "candidate_cache"],
        "progress_definition": "(d(t-W_p)-d(t))/(W_p*dt) m/s; invalid until W_p+1 samples",
        "phi_tau": "(tau_active-T_rep)/T_rep",
        "phi_p": "(p_min-p)/p_scale when progress is valid",
        "phi_h": "(h_rep-h_active)/h_scale",
        "S_rep": "max(valid phi_tau, phi_p, phi_h)",
        "normal_trigger": "S_rep>=0 and tau_active>=T_dwell",
        "emergency_trigger": "h_active<=h_emg; bypass dwell",
        "event_priority": [
            EVENT_REFERENCE_HANDOFF,
            EVENT_EMERGENCY_REPROPOSAL,
            EVENT_NORMAL_REPROPOSAL,
            EVENT_NO_UPDATE,
        ],
        "handoff_same_tick_replanning_forbidden": True,
        "active_safety_margin": {
            "source": ACTIVE_SAFETY_MARGIN_SOURCE,
            "unit": "m",
            "range": "unbounded signed physical margin; <=0 is Proposal infeasible boundary",
            "normalization": f"clip(h_active/{float(ProposalConfig(**dict(config['proposal_config'])).h_max):g} m,0,1)",
            "ground_truth_future_collision_used": False,
        },
        "goal_update_continuity": {
            "position_preserved": True,
            "velocity_preserved": True,
            "dmp_phase_preserved": True,
            "SAC_recurrent_state": "not_applicable_stateless_policy",
            "lidar_history_preserved": True,
            "environment_time_preserved": True,
        },
        "online_reconstruction_pipeline": "Proposal -> TopK=10 -> FP-SHEP H4 -> frozen GAT-V1",
        "null_or_K_zero_policy": "immutable terminal goal fallback",
        "err_parameters": config["err"],
        "parameter_provenance": config["err_parameter_provenance"],
    }


def run_experiment(config: Mapping[str, Any], output_dir: Path) -> Path:
    _assert_config(config)
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    resolved = copy.deepcopy(dict(config))
    resolved.update(
        {
            "resolved_output_dir": str(output_dir),
            "created_at": datetime.now().isoformat(),
            "python_executable": sys.executable,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
    )
    write_json(output_dir / "config.json", resolved)
    write_json(output_dir / "trigger_semantic_contract.json", _semantic_contract(config))

    gat_checkpoint = (REPO_ROOT / config["gat_checkpoint"]).resolve()
    sac_checkpoint = (REPO_ROOT / config["sac_checkpoint"]).resolve()
    gat_hash_before = _sha256_file(gat_checkpoint)
    sac_hash_before = _sha256_file(sac_checkpoint)
    if gat_hash_before != config["gat_checkpoint_sha256_expected"]:
        raise RuntimeError("formal GAT-V1 checkpoint hash mismatch")
    if sac_hash_before != config["sac_checkpoint_sha256_expected"]:
        raise RuntimeError("frozen SAC checkpoint hash mismatch")
    core_before = _file_hashes(CORE_METHOD_PATHS)
    for path, expected in config["expected_core_hashes"].items():
        if core_before[path] != expected:
            raise RuntimeError(f"core hash changed before ERR evaluation: {path}")

    settings = _execution_settings(config)
    multi_config = build_single_distribution_multi_config(
        num_agents=int(config["num_agents"]), max_steps=int(config["max_steps"])
    )
    policy, loaded_sac = _load_policy(settings, multi_config)
    if loaded_sac.resolve() != sac_checkpoint:
        raise RuntimeError("SAC loader resolved a different checkpoint")
    policy_hash_before = _policy_parameter_sha256(policy)
    stage1_config = _load_json((REPO_ROOT / config["stage1_config"]).resolve())
    gat_device = resolve_device(stage1_config["training"]["device"])
    gat_model = load_model_checkpoint(gat_checkpoint, stage1_config, gat_device)
    for parameter in gat_model.parameters():
        parameter.requires_grad_(False)
    gat_model.eval()
    gat_model_hash_before = _model_hash(gat_model)

    episodes: list[dict[str, Any]] = []
    agents: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    triggers: list[dict[str, Any]] = []
    paths: list[dict[str, Any]] = []
    scenes: list[dict[str, Any]] = []

    # One pre-outcome sanity pass: one ERR episode per required scenario.
    for scenario in config["development_scenarios"]:
        result = run_episode(
            config=config,
            settings=settings,
            multi_config=multi_config,
            policy=policy,
            gat_model=gat_model,
            gat_device=gat_device,
            method=METHOD_ERR,
            scenario=str(scenario),
            seed=int(config["sanity_seed"]),
        )
        episode, agent_rows, event_rows, trigger_rows, extra = result
        episodes.append(episode)
        agents.extend(agent_rows)
        events.extend(event_rows)
        triggers.extend(trigger_rows)
        paths.extend(extra["path_rows"])
        scenes.append(extra["scene_record"])
        print(
            json.dumps(
                {
                    "phase": "sanity",
                    "method": METHOD_ERR,
                    "scenario": scenario,
                    "seed": config["sanity_seed"],
                    "success": episode["team_success"],
                    "replans": episode["replanning_count"],
                }
            ),
            flush=True,
        )
    sanity = _sanity_audit(triggers, events, config)
    write_json(output_dir / "sanity_audit.json", sanity)
    if sanity["TRIGGER_SCALE_VALID"] != "YES":
        write_csv(output_dir / "trigger_distribution.csv", triggers)
        write_csv(output_dir / "replanning_events.csv", events)
        write_csv(output_dir / "trajectory_points.csv", paths)
        write_json(
            output_dir / "conclusion.json",
            {
                "schema_version": SCHEMA_VERSION,
                "ACTIVE_SAFETY_MARGIN_SOURCE": ACTIVE_SAFETY_MARGIN_SOURCE,
                "TRIGGER_SCALE_VALID": "NO",
                "ERR_PIPELINE_VALID": "NO",
                "RECOMMENDED_NEXT_STEP": "FAILURE_AUDIT",
                "stop_reason": "SANITY_TRIGGER_SCALE_INVALID",
            },
        )
        (output_dir / "FINAL_REPORT.md").write_text(
            "# ERR development audit\n\nSanity trigger scale invalid; controlled comparison was not started.\n",
            encoding="utf-8",
        )
        return output_dir

    # Complete all one-shot pairs and the remaining ERR seeds.  Sanity ERR rows
    # are reused unchanged; no experiment is rerun and no parameter is adjusted.
    for method in (METHOD_ONE_SHOT, METHOD_ERR):
        seeds = list(config["development_seeds"])
        if method == METHOD_ERR:
            seeds = [seed for seed in seeds if seed != int(config["sanity_seed"])]
        for scenario in config["development_scenarios"]:
            for seed in seeds:
                result = run_episode(
                    config=config,
                    settings=settings,
                    multi_config=multi_config,
                    policy=policy,
                    gat_model=gat_model,
                    gat_device=gat_device,
                    method=method,
                    scenario=str(scenario),
                    seed=int(seed),
                )
                episode, agent_rows, event_rows, trigger_rows, extra = result
                episodes.append(episode)
                agents.extend(agent_rows)
                events.extend(event_rows)
                triggers.extend(trigger_rows)
                paths.extend(extra["path_rows"])
                scenes.append(extra["scene_record"])
                print(
                    json.dumps(
                        {
                            "phase": "development",
                            "method": method,
                            "scenario": scenario,
                            "seed": seed,
                            "success": episode["team_success"],
                            "replans": episode["replanning_count"],
                        }
                    ),
                    flush=True,
                )

    # Persist the completed raw development run before aggregation so a report
    # or bookkeeping defect can never discard expensive trajectories.
    write_csv(output_dir / "_raw_episode_results.csv", episodes)
    write_csv(output_dir / "_raw_agent_results.csv", agents)
    write_csv(output_dir / "_raw_trigger_distribution.csv", triggers)
    write_csv(output_dir / "_raw_replanning_events.csv", events)
    write_csv(output_dir / "_raw_trajectory_points.csv", paths)
    write_json(output_dir / "_raw_scene_snapshots.json", scenes)

    episodes.sort(key=lambda row: (row["scenario"], int(row["seed"]), row["method"]))
    agents.sort(
        key=lambda row: (
            row["scenario"],
            int(row["seed"]),
            row["method"],
            int(row["agent_id"]),
        )
    )
    summaries = aggregate_summary(episodes, agents)
    paired = build_paired(episodes)
    failure_recovery = [
        {
            **row,
            "one_shot_failed": not row["one_shot_team_success"],
            "recovery_class": (
                "RECOVERED"
                if row["recovered_one_shot_failure"]
                else "REGRESSED"
                if row["regressed_one_shot_success"]
                else "BOTH_SUCCESS"
                if row["one_shot_team_success"] and row["err_team_success"]
                else "BOTH_FAILED"
            ),
        }
        for row in paired
    ]
    chatter = _chattering(events, float(config["err"]["T_dwell_s"]))
    conclusion = build_conclusion(summaries, paired, sanity, chatter)

    core_after = _file_hashes(CORE_METHOD_PATHS)
    integrity = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASSED",
        "core_hashes_before": core_before,
        "core_hashes_after": core_after,
        "core_hashes_unchanged": core_before == core_after,
        "gat_checkpoint_path": str(gat_checkpoint),
        "gat_checkpoint_sha256_before": gat_hash_before,
        "gat_checkpoint_sha256_after": _sha256_file(gat_checkpoint),
        "sac_checkpoint_path": str(sac_checkpoint),
        "sac_checkpoint_sha256_before": sac_hash_before,
        "sac_checkpoint_sha256_after": _sha256_file(sac_checkpoint),
        "gat_model_parameter_hash_before": gat_model_hash_before,
        "gat_model_parameter_hash_after": _model_hash(gat_model),
        "sac_policy_parameter_hash_before": policy_hash_before,
        "sac_policy_parameter_hash_after": _policy_parameter_sha256(policy),
        "all_gat_parameters_frozen": not any(
            parameter.requires_grad for parameter in gat_model.parameters()
        ),
        "episode_count": len(episodes),
        "paired_episode_count": len(paired),
        "formal_claim_made": False,
        "training_performed": False,
        "parameter_search_performed": False,
        "sanity_correction_count": 0,
    }
    integrity["status"] = (
        "PASSED"
        if integrity["core_hashes_unchanged"]
        and integrity["gat_checkpoint_sha256_before"]
        == integrity["gat_checkpoint_sha256_after"]
        and integrity["sac_checkpoint_sha256_before"]
        == integrity["sac_checkpoint_sha256_after"]
        and integrity["gat_model_parameter_hash_before"]
        == integrity["gat_model_parameter_hash_after"]
        and integrity["sac_policy_parameter_hash_before"]
        == integrity["sac_policy_parameter_hash_after"]
        else "FAILED"
    )
    if integrity["status"] != "PASSED":
        raise RuntimeError("final frozen-component integrity failed")

    # Choose an actual paired episode for the default spatial path view.
    representative = next(
        (row for row in failure_recovery if row["recovery_class"] == "RECOVERED"),
        max(failure_recovery, key=lambda row: row["terminal_progress_delta_m"]),
    )
    visualization_data = {
        "schema_version": SCHEMA_VERSION,
        "default_pair": {
            "scenario": representative["scenario"],
            "seed": representative["seed"],
            "reason": representative["recovery_class"],
        },
        "available_pairs": [
            {"scenario": row["scenario"], "seed": row["seed"]} for row in paired
        ],
        "scenes": scenes,
        "trajectory_points": paths,
        "planning_events": events,
    }

    write_json(output_dir / "integrity_manifest.json", integrity)
    write_csv(output_dir / "episode_results.csv", episodes)
    write_csv(output_dir / "agent_results.csv", agents)
    write_csv(output_dir / "trigger_distribution.csv", triggers)
    write_csv(output_dir / "replanning_events.csv", events)
    write_csv(output_dir / "one_shot_vs_err_paired.csv", paired)
    write_csv(output_dir / "failure_recovery.csv", failure_recovery)
    write_csv(output_dir / "scenario_summary.csv", summaries)
    write_csv(output_dir / "trajectory_points.csv", paths)
    write_json(output_dir / "chattering_audit.json", chatter)
    write_json(output_dir / "conclusion.json", conclusion)
    write_json(output_dir / "path_visualization_data.json", visualization_data)
    (output_dir / "FINAL_REPORT.md").write_text(
        _render_report(summaries, conclusion, sanity, chatter), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "episode_count": len(episodes),
                "runtime_seconds": time.perf_counter() - started,
                "ERR_PIPELINE_VALID": conclusion["ERR_PIPELINE_VALID"],
                "ERR_CLOSED_LOOP_GAIN": conclusion["ERR_CLOSED_LOOP_GAIN"],
                "RECOMMENDED_NEXT_STEP": conclusion["RECOMMENDED_NEXT_STEP"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> Path:
    args = parse_args()
    config = _load_json(args.config.expanduser().resolve())
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else REPO_ROOT
        / str(config["output_root"])
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    return run_experiment(config, output_dir)


if __name__ == "__main__":
    main()
