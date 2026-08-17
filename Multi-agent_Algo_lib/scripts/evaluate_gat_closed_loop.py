"""Run the frozen Stage-I GAT final closed-loop validation.

The evaluator freezes one t=0 candidate/preview/graph bundle per
``scenario + seed``.  Proposal, FP-SHEP, and Stage-I GAT derive immutable
selection plans from that shared bundle, then enter the established
``run_variant_episode`` execution helper through the same selection-plan
interface.  No selector may reorder candidates to emulate another method.
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
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = Path(__file__).resolve().parent
for search_path in (REPO_ROOT, ALGO_ROOT, SCRIPTS_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from Guidance.reference_point_proposal_demo import Proposal, ProposalConfig  # noqa: E402
from planning.candidate_execution_interface import (  # noqa: E402
    graph_ready_candidate_execution,
)
from planning.gat.candidate_selector import batch_candidate_graphs  # noqa: E402
from planning.gat.stage1_training import (  # noqa: E402
    load_model_checkpoint,
    resolve_device,
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
from planning.pre_gat_220step_revalidation import (  # noqa: E402
    ImmutableCandidate,
    ImmutableCandidateBundle,
    generate_immutable_candidate_bundle,
    stable_hash,
)
from planning.pre_gat_closed_loop import (  # noqa: E402
    FPSHEPOnlineScoreSpec,
    score_fp_shep_candidates,
)
from planning.goal_semantics_diagnosis import VARIANT_A, VARIANT_D  # noqa: E402
from scripts.evaluate_actor_dmp_goal_semantics import (  # noqa: E402
    run_variant_episode,
    write_csv,
    write_json,
)
from scripts.evaluate_pre_gat_closed_loop import (  # noqa: E402
    _policy_parameter_sha256,
    _scenario_hash,
    _scene_snapshot,
    build_closed_loop_environment,
)
from scripts.evaluate_single_policy_aligned_multi_agent import (  # noqa: E402
    build_single_distribution_multi_config,
)
from scripts.evaluate_temporary_reference_interface import _load_policy  # noqa: E402


SCHEMA_VERSION = "gat_closed_loop_v1"
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs/evaluation/gat_closed_loop.json"

METHOD_TERMINAL = "terminal"
METHOD_PROPOSAL = "proposal"
METHOD_FP_SHEP = "fp_shep"
METHOD_GAT = "gat_stage1"
METHOD_ORDER = (METHOD_TERMINAL, METHOD_PROPOSAL, METHOD_FP_SHEP, METHOD_GAT)
METHOD_NAMES = {
    METHOD_TERMINAL: "Terminal",
    METHOD_PROPOSAL: "Proposal",
    METHOD_FP_SHEP: "FP-SHEP",
    METHOD_GAT: "Stage-I GAT",
}

CORE_METHOD_PATHS = (
    "Guidance/reference_point_proposal_demo.py",
    "planning/policy_preview.py",
    "planning/heterogeneous_candidate_graph.py",
    "planning/gat/candidate_selector.py",
    "planning/gat/edge_enhanced_gat.py",
    "baseline/sac/net.py",
    "Controller/dmp_rl.py",
    "Environment/multi_agent_dmp_env.py",
    "Environment/frozen_sac_dmp_execution.py",
)
SMOKE_REUSE_PATHS = CORE_METHOD_PATHS + (
    "Multi-agent_Algo_lib/scripts/evaluate_actor_dmp_goal_semantics.py",
    "Multi-agent_Algo_lib/scripts/evaluate_gat_closed_loop.py",
    "configs/evaluation/gat_closed_loop.json",
)
GRAPH_REQUIRED_PROPOSAL_FIELDS = (
    "azimuth_index",
    "elevation_index",
    "direction",
    "distance_progress",
    "normalized_margin",
)


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_hashes(paths: Sequence[str]) -> dict[str, str]:
    return {relative: _sha256_file(REPO_ROOT / relative) for relative in paths}


def _model_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(key.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _assert_frozen_config(config: Mapping[str, Any]) -> None:
    if str(config["schema_version"]) != SCHEMA_VERSION:
        raise ValueError("unexpected closed-loop schema version")
    if tuple(config["methods"]) != METHOD_ORDER:
        raise ValueError("method order changed")
    if list(config["formal_scenarios"]) != ["open", "sparse_static", "multi_agent"]:
        raise ValueError("formal scenario set changed")
    if list(config["smoke_seeds"]) != [10, 11, 12]:
        raise ValueError("smoke seeds changed")
    if list(config["formal_seeds"]) != list(range(10, 30)):
        raise ValueError("formal seeds must remain 10..29")
    frozen = {
        "top_k": 10,
        "H_preview": 4,
        "max_steps": 220,
        "handoff_threshold_m": 0.25,
    }
    for key, expected in frozen.items():
        if config[key] != expected:
            raise ValueError(f"frozen setting changed: {key}")
    if config["forcing_gate"] != HISTORICAL_GATE_NAME:
        raise ValueError("historical vector gate is required")
    if config["boundary_mode"] != "boundary_free":
        raise ValueError("closed-loop evaluation must remain boundary-free")
    if any(bool(value) for value in config["strict_exclusions"].values()):
        raise ValueError("strict exclusion flags must remain false")
    if config["selection_plan"]["candidate_reordering_allowed"]:
        raise ValueError("candidate reordering is forbidden")
    if config["statistics"]["mcnemar_exact_unit"] != "scenario_seed_team_episode":
        raise ValueError("McNemar unit must be one team episode")


def _build_execution_settings(config: Mapping[str, Any]) -> dict[str, Any]:
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
            "scenarios": list(config["formal_scenarios"]),
            "seeds": list(config["formal_seeds"]),
            "temporary_reference": {
                **settings["temporary_reference"],
                "K_requested": int(config["top_k"]),
                "reached_tolerance_m": float(config["handoff_threshold_m"]),
                "replanning_enabled": False,
            },
            "proposal_config": copy.deepcopy(config["proposal_config"]),
        }
    )
    return settings


def reconstruct_proposals(
    candidates: Sequence[ImmutableCandidate],
) -> tuple[tuple[Proposal, ...], list[dict[str, Any]], bool]:
    """Recover only serialized Proposal fields; never recompute attributes."""

    proposal_fields = tuple(Proposal.__dataclass_fields__)
    metadata_fields = set(proposal_fields) - {"point", "score"}
    recovered: list[Proposal] = []
    audit_rows: list[dict[str, Any]] = []
    all_equivalent = True
    for expected_index, candidate in enumerate(candidates):
        metadata = candidate.metadata
        metadata_keys_exact = set(metadata) == metadata_fields
        kwargs = copy.deepcopy(metadata)
        if "direction" in kwargs:
            kwargs["direction"] = np.asarray(kwargs["direction"], dtype=float)
        proposal = Proposal(point=candidate.point, score=candidate.score, **kwargs)
        recovered.append(proposal)

        field_equivalence = {
            field: _jsonable(getattr(proposal, field)) == _jsonable(metadata[field])
            for field in metadata_fields
            if field in metadata
        }
        graph_metadata_present = all(
            field in metadata for field in GRAPH_REQUIRED_PROPOSAL_FIELDS
        )
        checks = {
            "candidate_index_order": int(candidate.original_index) == expected_index,
            "world_position": np.array_equal(proposal.point, candidate.point),
            "sector_id": (
                int(proposal.azimuth_index) == int(metadata.get("azimuth_index", -1))
                and int(proposal.elevation_index)
                == int(metadata.get("elevation_index", -1))
            ),
            "coarse_score": float(proposal.score) == float(candidate.score),
            "metadata_keys_exact": metadata_keys_exact,
            "graph_required_metadata": graph_metadata_present,
            "all_existing_fields": bool(field_equivalence)
            and all(field_equivalence.values()),
        }
        equivalent = all(checks.values())
        all_equivalent &= equivalent
        audit_rows.append(
            {
                "candidate_index": expected_index,
                "frozen_original_index": int(candidate.original_index),
                "equivalent": equivalent,
                **checks,
                "world_position_frozen": candidate.point.tolist(),
                "world_position_reconstructed": proposal.point.tolist(),
                "sector_id": [
                    int(proposal.azimuth_index),
                    int(proposal.elevation_index),
                ],
                "coarse_score": float(proposal.score),
            }
        )
    count_equivalent = len(recovered) == len(candidates)
    all_equivalent &= count_equivalent
    for row in audit_rows:
        row["candidate_count"] = count_equivalent
        row["equivalent"] = bool(row["equivalent"] and count_equivalent)
    return tuple(recovered), audit_rows, bool(all_equivalent)


def _preview_payload(agent_records: Sequence[Sequence[Any]]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for agent_id, records in enumerate(agent_records):
        for record in records:
            execution = graph_ready_candidate_execution(record.candidate_id, record.preview)
            payload.append(
                {
                    "agent_id": agent_id,
                    "candidate_id": int(record.candidate_id),
                    "candidate_world_position": record.candidate_world_position,
                    "feature_raw": execution.raw_feature_vector,
                    "feature_normalized": record.normalized_features,
                    "preview_positions": execution.preview_positions,
                    "preview_velocities": execution.preview_velocities,
                    "requested_horizon": execution.requested_horizon_steps,
                    "effective_horizon": execution.effective_horizon_steps,
                }
            )
    return payload


def _rank_1based(scores: Sequence[float], candidate_id: int | None) -> int | None:
    if candidate_id is None:
        return None
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable").tolist()
    return int(order.index(int(candidate_id)) + 1)


def _selected_edge_diagnostics(graph: Any, candidate_id: int | None) -> dict[str, Any]:
    if candidate_id is None:
        return {
            "selected_minimum_t_min": None,
            "selected_minimum_d_min": None,
            "selected_maximum_T_risk": None,
            "selected_align_edge_count": 0,
        }
    store = graph["align", "spatiotemporal", "proposal"]
    edge_index = store.edge_index.detach().cpu().numpy()
    raw = store.edge_attr.detach().cpu().numpy()
    mask = edge_index[1] == int(candidate_id) if edge_index.shape[1] else np.zeros(0, bool)
    selected = raw[mask]
    if selected.size == 0:
        return {
            "selected_minimum_t_min": None,
            "selected_minimum_d_min": None,
            "selected_maximum_T_risk": None,
            "selected_align_edge_count": 0,
        }
    return {
        "selected_minimum_t_min": float(np.min(selected[:, 0])),
        "selected_minimum_d_min": float(np.min(selected[:, 1])),
        "selected_maximum_T_risk": float(np.max(selected[:, 2])),
        "selected_align_edge_count": int(selected.shape[0]),
    }


def _candidate_record(
    *,
    agent_id: int,
    proposals: Sequence[Proposal],
    preview_records: Sequence[Any],
    selected_id: int | None,
    selection_source: str,
    terminal_goal: np.ndarray,
    gat_diagnostic: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    selected = proposals[selected_id] if selected_id is not None else None
    no_candidate = len(proposals) == 0 and selection_source != "gat_stage1"
    selected_preview = (
        preview_records[selected_id] if selected_id is not None else None
    )
    return {
        "agent_id": int(agent_id),
        "candidate_available": selected is not None,
        "temporary_reference": (
            np.asarray(selected.point, dtype=float).tolist()
            if selected is not None
            else np.asarray(terminal_goal, dtype=float).tolist()
        ),
        "selected_candidate_id": selected_id,
        "selection_source": selection_source,
        "selected_null": selection_source == "gat_stage1" and selected_id is None,
        "no_candidate_fallback": no_candidate,
        "proposal_count_before_consumer": len(proposals),
        "K_t": len(proposals),
        "candidate_world_points": [item.point.tolist() for item in proposals],
        "proposal_scores": [float(item.score) for item in proposals],
        "candidate_metadata": [
            {
                key: _jsonable(value)
                for key, value in vars(item).items()
                if key not in {"point", "score"}
            }
            for item in proposals
        ],
        "selected_proposal_rank": selected_id,
        "selected_proposal_rank_1based": (
            int(selected_id) + 1 if selected_id is not None else None
        ),
        "selected_proposal_score": (
            float(selected.score) if selected is not None else None
        ),
        "fp_shep_candidate_records": [item.to_record() for item in preview_records],
        "selected_fp_shep_score": (
            float(selected_preview.score) if selected_preview is not None else None
        ),
        "selected_fp_shep_rank_1based": _rank_1based(
            [item.score for item in preview_records], selected_id
        ),
        **dict(gat_diagnostic or {}),
    }


def _make_method_plan(
    *,
    source: str,
    terminal_goals: np.ndarray,
    proposals_by_agent: Sequence[Sequence[Proposal]],
    previews_by_agent: Sequence[Sequence[Any]],
    selected_ids: Sequence[int | None],
    gat_diagnostics: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    references = np.asarray(terminal_goals, dtype=float).copy()
    available = np.zeros(len(proposals_by_agent), dtype=bool)
    for agent_id, (proposals, previews, selected_id) in enumerate(
        zip(proposals_by_agent, previews_by_agent, selected_ids, strict=True)
    ):
        if selected_id is not None:
            if not 0 <= int(selected_id) < len(proposals):
                raise IndexError("selection plan candidate index is outside actual K_t")
            references[agent_id] = proposals[int(selected_id)].point
            available[agent_id] = True
        records.append(
            _candidate_record(
                agent_id=agent_id,
                proposals=proposals,
                preview_records=previews,
                selected_id=selected_id,
                selection_source=source,
                terminal_goal=terminal_goals[agent_id],
                gat_diagnostic=(
                    gat_diagnostics[agent_id] if gat_diagnostics is not None else None
                ),
            )
        )
    payload = {
        "selection_source": source,
        "references": references.tolist(),
        "available": available.tolist(),
        "candidate_records": records,
    }
    payload["selection_plan_hash"] = stable_hash(payload)
    return payload


def build_shared_selection_bundle(
    *,
    config: Mapping[str, Any],
    execution_settings: Mapping[str, Any],
    multi_config: Any,
    policy: Any,
    gat_model: torch.nn.Module,
    gat_device: torch.device,
    scenario: str,
    seed: int,
) -> dict[str, Any]:
    env, scene_metadata = build_closed_loop_environment(
        config=multi_config,
        scenario=scenario,
        seed=int(seed),
        peer_radius=float(config["peer_radius"]),
    )
    try:
        snapshot = _scene_snapshot(env)
        initial_hash = _scenario_hash(snapshot)
        terminal_goals = np.asarray(env.goals, dtype=float).copy()
        immutable = generate_immutable_candidate_bundle(
            env,
            scenario=scenario,
            seed=int(seed),
            proposal_config=ProposalConfig(**dict(config["proposal_config"])),
            consumer_top_k=int(config["top_k"]),
        )
        candidate_hash_before = immutable.candidate_set_hash

        proposals_by_agent: list[tuple[Proposal, ...]] = []
        reconstruction_rows: list[dict[str, Any]] = []
        reconstruction_equivalent = True
        for agent_id, candidates in enumerate(immutable.per_agent):
            proposals, rows, equivalent = reconstruct_proposals(candidates)
            proposals_by_agent.append(proposals)
            reconstruction_equivalent &= equivalent
            reconstruction_rows.extend(
                {
                    "scenario": scenario,
                    "seed": int(seed),
                    "agent_id": agent_id,
                    **row,
                }
                for row in rows
            )
        if not reconstruction_equivalent:
            raise RuntimeError("PROPOSAL_RECONSTRUCTION_EQUIVALENT = NO")

        score_spec = FPSHEPOnlineScoreSpec.from_mapping(
            {
                "name": config["fp_shep"]["score_name"],
                "definition_status": "frozen_closed_loop_baseline",
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
        preview_trace: list[dict[str, Any]] = []

        def observe_preview(kwargs: dict[str, Any], transition: Any) -> None:
            preview_trace.append(
                {
                    "forcing_gate_semantics": transition.controller_info.get(
                        "forcing_gate_semantics"
                    ),
                    "active_goal": _jsonable(kwargs.get("active_goal")),
                }
            )

        previews_by_agent: list[tuple[Any, ...]] = []
        graphs: list[Any] = []
        with scoped_historical_preview_and_multi_agent_transition(
            preview_observer=observe_preview
        ):
            for agent_id, proposals in enumerate(proposals_by_agent):
                previews = score_fp_shep_candidates(
                    env=env,
                    agent_index=agent_id,
                    proposals=proposals,
                    policy=policy,
                    spec=score_spec,
                )
                previews_by_agent.append(previews)
                executions = tuple(
                    graph_ready_candidate_execution(record.candidate_id, record.preview)
                    for record in previews
                )
                graph = build_heterogeneous_candidate_graph_from_env(
                    env=env,
                    agent_index=agent_id,
                    proposals=proposals,
                    executions=executions,
                    proposal_config=ProposalConfig(**dict(config["proposal_config"])),
                    config=HeterogeneousCandidateGraphConfig(
                        horizon_steps=int(config["H_preview"]),
                        d_align=float(config["graph"]["d_align"]),
                        d_align_source=str(config["graph"]["d_align_source"]),
                    ),
                )
                preview_trajectory_match = all(
                    np.array_equal(
                        graph.candidate_preview_positions[index],
                        executions[index].preview_positions,
                    )
                    and np.array_equal(
                        graph.original_proposals[index].point,
                        proposals[index].point,
                    )
                    for index in range(len(proposals))
                )
                reconstruction_equivalent &= preview_trajectory_match
                for row in reconstruction_rows:
                    if row["agent_id"] == agent_id:
                        row["preview_trajectory"] = preview_trajectory_match
                        row["equivalent"] = bool(
                            row["equivalent"] and preview_trajectory_match
                        )
                graphs.append(graph)

        if not reconstruction_equivalent:
            raise RuntimeError("PROPOSAL_RECONSTRUCTION_EQUIVALENT = NO")
        if preview_trace and not all(
            row["forcing_gate_semantics"] == HISTORICAL_GATE_NAME
            for row in preview_trace
        ):
            raise RuntimeError("FP-SHEP preview did not use historical vector gate")

        preview_payload = _preview_payload(previews_by_agent)
        preview_hash = stable_hash(preview_payload)
        candidate_hash_after_preview = immutable.candidate_set_hash
        if candidate_hash_after_preview != candidate_hash_before:
            raise RuntimeError("preview mutated the immutable candidate bundle")

        batched = batch_candidate_graphs(graphs).to(gat_device)
        with torch.inference_mode():
            gat_output = gat_model(batched)
        if not torch.isfinite(gat_output.candidate_logits).all():
            raise RuntimeError("GAT produced non-finite logits")

        gat_selected = list(gat_output.selected_candidate_id)
        gat_diagnostics: list[dict[str, Any]] = []
        for agent_id, (graph, proposals, previews, selected_id) in enumerate(
            zip(
                graphs,
                proposals_by_agent,
                previews_by_agent,
                gat_selected,
                strict=True,
            )
        ):
            probabilities = (
                gat_output.probabilities_for_graph(agent_id).detach().cpu().numpy()
            )
            logits = gat_output.logits_for_graph(agent_id).detach().cpu().numpy()
            class_index = int(gat_output.selected_class_index[agent_id])
            expected_selected = None if class_index == 0 else class_index - 1
            class_mapping_valid = (
                expected_selected == selected_id
                and 0 <= class_index < len(proposals) + 1
            )
            if not class_mapping_valid:
                raise RuntimeError("GAT null/proposal class mapping is invalid")
            sorted_probabilities = np.sort(probabilities)[::-1]
            margin = (
                float(sorted_probabilities[0] - sorted_probabilities[1])
                if len(sorted_probabilities) >= 2
                else None
            )
            diagnostic = {
                "selected_class": class_index,
                "selected_null": selected_id is None,
                "selected_candidate_id": selected_id,
                "selected_proposal_rank_by_coarse_score": (
                    int(selected_id) + 1 if selected_id is not None else None
                ),
                "selected_proposal_rank_by_FP_SHEP": _rank_1based(
                    [record.score for record in previews], selected_id
                ),
                "GAT_confidence": float(probabilities[class_index]),
                "top1_top2_probability_margin": margin,
                "class_count": len(probabilities),
                "class_mapping_valid": class_mapping_valid,
                "logits_finite": bool(np.all(np.isfinite(logits))),
                "class_probabilities": probabilities.tolist(),
                "class_logits": logits.tolist(),
                **_selected_edge_diagnostics(graph, selected_id),
            }
            if selected_id is not None:
                preview = previews[int(selected_id)]
                diagnostic.update(
                    {
                        "selected_preview_progress": preview.preview_task_progress,
                        "selected_preview_clearance": preview.preview_min_clearance,
                        "selected_preview_deviation": (
                            preview.preview_max_execution_deviation
                        ),
                        "selected_preview_terminal_speed": preview.preview_terminal_speed,
                    }
                )
            else:
                diagnostic.update(
                    {
                        "selected_preview_progress": None,
                        "selected_preview_clearance": None,
                        "selected_preview_deviation": None,
                        "selected_preview_terminal_speed": None,
                    }
                )
            gat_diagnostics.append(diagnostic)

        proposal_selected = [0 if proposals else None for proposals in proposals_by_agent]
        fp_selected = [
            int(np.argmax([record.score for record in previews])) if previews else None
            for previews in previews_by_agent
        ]
        plans = {
            METHOD_PROPOSAL: _make_method_plan(
                source="proposal_top1",
                terminal_goals=terminal_goals,
                proposals_by_agent=proposals_by_agent,
                previews_by_agent=previews_by_agent,
                selected_ids=proposal_selected,
            ),
            METHOD_FP_SHEP: _make_method_plan(
                source="fp_shep_h4",
                terminal_goals=terminal_goals,
                proposals_by_agent=proposals_by_agent,
                previews_by_agent=previews_by_agent,
                selected_ids=fp_selected,
            ),
            METHOD_GAT: _make_method_plan(
                source="gat_stage1",
                terminal_goals=terminal_goals,
                proposals_by_agent=proposals_by_agent,
                previews_by_agent=previews_by_agent,
                selected_ids=gat_selected,
                gat_diagnostics=gat_diagnostics,
            ),
        }
        candidate_hashes_by_method = {
            method: immutable.candidate_set_hash for method in plans
        }
        preview_hashes_by_method = {
            METHOD_FP_SHEP: preview_hash,
            METHOD_GAT: preview_hash,
        }
        return {
            "scenario": scenario,
            "seed": int(seed),
            "initial_condition_hash": initial_hash,
            "scenario_snapshot": _jsonable(snapshot),
            "scene_metadata": _jsonable(scene_metadata),
            "candidate_bundle": immutable,
            "candidate_bundle_hash": immutable.candidate_set_hash,
            "candidate_hashes_by_method": candidate_hashes_by_method,
            "preview_bundle_hash": preview_hash,
            "preview_hashes_by_method": preview_hashes_by_method,
            "preview_transition_call_count": len(preview_trace),
            "preview_historical_gate_verified": bool(preview_trace)
            and all(
                row["forcing_gate_semantics"] == HISTORICAL_GATE_NAME
                for row in preview_trace
            ),
            "proposal_reconstruction_equivalent": reconstruction_equivalent,
            "reconstruction_rows": reconstruction_rows,
            "plans": plans,
            "gat_diagnostics": gat_diagnostics,
            "candidate_count_per_agent": [
                len(proposals) for proposals in proposals_by_agent
            ],
            "graph_schema_match": all(
                graph.graph_metadata["feature_schema_version"] == GRAPH_SCHEMA_VERSION
                and int(graph.graph_metadata["H"]) == int(config["H_preview"])
                for graph in graphs
            ),
        }
    finally:
        env.close()


def _reference_selected_type(method: str, record: Mapping[str, Any]) -> str:
    if method == METHOD_TERMINAL:
        return "terminal_baseline"
    if bool(record.get("no_candidate_fallback", False)):
        return "no_candidate_fallback"
    if bool(record.get("selected_null", False)):
        return "null"
    return "proposal"


def _standardize_episode(
    raw: Mapping[str, Any],
    *,
    method: str,
    shared: Mapping[str, Any],
    execution_trace: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    plan = shared["plans"].get(method)
    raw_agents = list(raw["agent_stage_records"])
    agents: list[dict[str, Any]] = []
    for raw_agent in raw_agents:
        agent_id = int(raw_agent["agent_id"])
        plan_record = (
            plan["candidate_records"][agent_id]
            if plan is not None
            else {
                "selected_candidate_id": None,
                "selected_null": False,
                "no_candidate_fallback": False,
            }
        )
        agents.append(
            {
                **copy.deepcopy(dict(raw_agent)),
                "schema_version": SCHEMA_VERSION,
                "method": method,
                "method_display_name": METHOD_NAMES[method],
                "scenario": raw["scenario"],
                "seed": int(raw["seed"]),
                "candidate_bundle_hash": shared["candidate_bundle_hash"],
                "preview_bundle_hash": (
                    shared["preview_bundle_hash"]
                    if method in {METHOD_FP_SHEP, METHOD_GAT}
                    else None
                ),
                "reference_selected_type": _reference_selected_type(
                    method, plan_record
                ),
                "selected_candidate_id": plan_record.get("selected_candidate_id"),
                "selected_null": bool(plan_record.get("selected_null", False)),
                "selection_plan_hash": (
                    plan.get("selection_plan_hash") if plan is not None else None
                ),
            }
        )
    completion_step = raw.get("team_terminal_completion_step") if raw["success"] else None
    minimum_obstacle_clearance = float(raw["minimum_obstacle_clearance_m"])
    episode = {
        "schema_version": SCHEMA_VERSION,
        "method": method,
        "method_display_name": METHOD_NAMES[method],
        "scenario": raw["scenario"],
        "seed": int(raw["seed"]),
        "team_success": bool(raw["success"]),
        "obstacle_collision": bool(raw["obstacle_collision"]),
        "inter_agent_collision": bool(raw["inter_agent_collision"]),
        "collision": bool(raw["collision"]),
        "timeout": bool(raw["truncated"]),
        "termination_reason": raw["termination_reason"],
        "completion_step": completion_step,
        "completion_time_s": (
            float(completion_step) * float(raw.get("dt", 0.1))
            if completion_step is not None
            else None
        ),
        "steps": int(raw["steps"]),
        "team_path_length_m": float(raw["path_length_team_sum_m"]),
        "team_path_length_mean_agent_m": float(raw["path_length_team_mean_m"]),
        "per_agent_path_length_m": raw["path_length_per_agent_m"],
        "trajectory_smoothness": float(raw["trajectory_smoothness"]),
        "mean_speed_mps": float(raw["mean_speed_team_mps"]),
        "peak_speed_mps": float(raw["peak_speed_team_mps"]),
        "mean_acceleration_mps2": float(raw["mean_applied_acceleration_mps2"]),
        "peak_acceleration_mps2": float(raw["max_applied_acceleration_mps2"]),
        "heading_yaw_change_available": False,
        "minimum_obstacle_clearance_m": (
            minimum_obstacle_clearance
            if np.isfinite(minimum_obstacle_clearance)
            else None
        ),
        "minimum_obstacle_clearance_unbounded": bool(
            np.isposinf(minimum_obstacle_clearance)
        ),
        "minimum_inter_agent_distance_m": float(
            raw["minimum_inter_agent_distance_m"]
        ),
        "reference_selected_types": [
            row["reference_selected_type"] for row in agents
        ],
        "reference_selected_type": (
            "mixed"
            if len({row["reference_selected_type"] for row in agents}) > 1
            else agents[0]["reference_selected_type"]
        ),
        "reference_selected_count": sum(
            row["reference_selected_type"] == "proposal" for row in agents
        ),
        "reference_reached_count": int(raw["temporary_reference_reached_count"]),
        "reference_reached": any(bool(row["reference_reached"]) for row in agents),
        "reference_reach_step": raw["reference_reached_steps"],
        "reference_to_terminal_success_count": int(
            raw["reached_then_terminal_completion_count"]
        ),
        "reference_to_terminal_success": bool(
            raw["reached_then_terminal_completion_count"]
            == raw["temporary_reference_reached_count"]
            and raw["temporary_reference_reached_count"] > 0
        ),
        "candidate_bundle_hash": shared["candidate_bundle_hash"],
        "preview_bundle_hash": (
            shared["preview_bundle_hash"]
            if method in {METHOD_FP_SHEP, METHOD_GAT}
            else None
        ),
        "selection_plan_hash": (
            plan.get("selection_plan_hash") if plan is not None else None
        ),
        "initial_condition_hash": raw["initial_condition_hash"],
        "scenario_manifest_hash": shared["initial_condition_hash"],
        "proposal_reconstruction_equivalent": shared[
            "proposal_reconstruction_equivalent"
        ],
        "graph_schema_match": shared["graph_schema_match"],
        "GAT_checkpoint_used": method == METHOD_GAT,
        "FP_SHEP_selector_used": method == METHOD_FP_SHEP,
        "historical_gate": HISTORICAL_GATE_NAME,
        "execution_transition_call_count": len(execution_trace),
        "execution_historical_gate_verified": bool(execution_trace)
        and all(
            row["forcing_gate_semantics"] == HISTORICAL_GATE_NAME
            for row in execution_trace
        ),
        "preview_historical_gate_verified": shared[
            "preview_historical_gate_verified"
        ],
        "handoff_count": int(raw["temporary_reference_reached_count"]),
        "maximum_handoff_count_per_agent": max(
            (int(bool(row["reference_reached"])) for row in agents), default=0
        ),
        "replanning_count": 0,
        "phase_reset_on_switch": bool(raw["phase_reset_on_switch"]),
        "terminal_task_goals_unchanged": bool(raw["terminal_task_goals_unchanged"]),
        "max_steps": 220,
        "episode_runtime_ms": float(raw["episode_runtime_ms"]),
    }
    return episode, agents


def run_method_episode(
    *,
    config: Mapping[str, Any],
    execution_settings: Mapping[str, Any],
    multi_config: Any,
    policy: Any,
    shared: Mapping[str, Any],
    method: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    execution_trace: list[dict[str, Any]] = []

    def observe_execution(kwargs: dict[str, Any], transition: Any) -> None:
        execution_trace.append(
            {
                "forcing_gate_semantics": transition.controller_info.get(
                    "forcing_gate_semantics"
                ),
                "active_goal": _jsonable(kwargs.get("active_goal")),
            }
        )

    variant = VARIANT_A if method == METHOD_TERMINAL else VARIANT_D
    plan = shared["plans"].get(method)
    plan_hash_before = stable_hash(plan) if plan is not None else None
    with scoped_historical_preview_and_multi_agent_transition(
        execution_observer=observe_execution
    ):
        raw, _, _ = run_variant_episode(
            policy=policy,
            multi_config=multi_config,
            settings=execution_settings,
            scenario=str(shared["scenario"]),
            seed=int(shared["seed"]),
            variant=variant,
            selection_plan=plan,
        )
    plan_hash_after = stable_hash(plan) if plan is not None else None
    if plan_hash_before != plan_hash_after:
        raise RuntimeError("execution mutated the immutable selection plan")
    episode, agents = _standardize_episode(
        raw,
        method=method,
        shared=shared,
        execution_trace=execution_trace,
    )
    episode["selection_plan_unchanged"] = plan_hash_before == plan_hash_after
    return episode, agents


def _mean_std_median(values: Iterable[Any]) -> dict[str, float | None]:
    array = np.asarray(
        [float(value) for value in values if value is not None], dtype=float
    )
    array = array[np.isfinite(array)]
    if not array.size:
        return {"mean": None, "std": None, "median": None, "count": 0}
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "count": int(array.size),
    }


def aggregate_method_summary(
    episodes: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scenarios = sorted({str(row["scenario"]) for row in episodes})
    for method in METHOD_ORDER:
        method_rows = [row for row in episodes if row["method"] == method]
        for scope in ["overall", *scenarios]:
            members = (
                method_rows
                if scope == "overall"
                else [row for row in method_rows if row["scenario"] == scope]
            )
            successful = [row for row in members if bool(row["team_success"])]
            record: dict[str, Any] = {
                "method": method,
                "method_display_name": METHOD_NAMES[method],
                "scenario": scope,
                "episode_count": len(members),
            }
            for field in (
                "team_success",
                "obstacle_collision",
                "inter_agent_collision",
                "collision",
                "timeout",
            ):
                count = sum(bool(row[field]) for row in members)
                record[f"{field}_count"] = count
                record[f"{field}_rate"] = count / len(members) if members else None
            for field in (
                "completion_step",
                "completion_time_s",
                "team_path_length_m",
                "trajectory_smoothness",
                "mean_speed_mps",
                "peak_speed_mps",
                "mean_acceleration_mps2",
                "peak_acceleration_mps2",
            ):
                stats = _mean_std_median(row[field] for row in successful)
                for statistic, value in stats.items():
                    record[f"success_{field}_{statistic}"] = value
            for field in (
                "minimum_obstacle_clearance_m",
                "minimum_inter_agent_distance_m",
            ):
                stats = _mean_std_median(row[field] for row in members)
                for statistic, value in stats.items():
                    record[f"{field}_{statistic}"] = value
            rows.append(record)
    return rows


def _failure_category(
    episode: Mapping[str, Any], agent_rows: Sequence[Mapping[str, Any]]
) -> str:
    if bool(episode["team_success"]):
        return "SUCCESS"
    if bool(episode["obstacle_collision"]):
        return "OBSTACLE_COLLISION"
    if bool(episode["inter_agent_collision"]):
        return "INTER_AGENT_COLLISION"
    if bool(episode["timeout"]):
        proposal_rows = [
            row for row in agent_rows if row["reference_selected_type"] == "proposal"
        ]
        if proposal_rows and any(not bool(row["reference_reached"]) for row in proposal_rows):
            return "REFERENCE_UNREACHABLE"
        if proposal_rows and any(
            bool(row["reference_reached"])
            and not bool(row["terminal_completed_after_reference"])
            for row in proposal_rows
        ):
            return "POST_REFERENCE_FAILURE"
        return "TIMEOUT_DIRECT_OR_OTHER"
    return "OTHER_FAILURE"


def build_gat_fp_pairing(
    episodes: Sequence[Mapping[str, Any]],
    agents: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episode_index = {
        (row["scenario"], int(row["seed"]), row["method"]): row for row in episodes
    }
    agent_groups: dict[tuple[str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in agents:
        agent_groups[(row["scenario"], int(row["seed"]), row["method"])].append(row)
    paired: list[dict[str, Any]] = []
    disagreements: list[dict[str, Any]] = []
    keys = sorted({(row["scenario"], int(row["seed"])) for row in episodes})
    for scenario, seed in keys:
        fp = episode_index[(scenario, seed, METHOD_FP_SHEP)]
        gat = episode_index[(scenario, seed, METHOD_GAT)]
        fp_agents = sorted(
            agent_groups[(scenario, seed, METHOD_FP_SHEP)],
            key=lambda row: int(row["agent_id"]),
        )
        gat_agents = sorted(
            agent_groups[(scenario, seed, METHOD_GAT)],
            key=lambda row: int(row["agent_id"]),
        )
        disagreement_count = 0
        for fp_agent, gat_agent in zip(fp_agents, gat_agents, strict=True):
            same = (
                fp_agent["reference_selected_type"]
                == gat_agent["reference_selected_type"]
                and fp_agent["selected_candidate_id"]
                == gat_agent["selected_candidate_id"]
            )
            disagreement_count += int(not same)
            disagreements.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "agent_id": int(fp_agent["agent_id"]),
                    "disagreement": not same,
                    "fp_shep_selected_type": fp_agent["reference_selected_type"],
                    "gat_selected_type": gat_agent["reference_selected_type"],
                    "fp_shep_candidate_id": fp_agent["selected_candidate_id"],
                    "gat_candidate_id": gat_agent["selected_candidate_id"],
                    "candidate_bundle_hash_match": (
                        fp_agent["candidate_bundle_hash"]
                        == gat_agent["candidate_bundle_hash"]
                    ),
                    "preview_bundle_hash_match": (
                        fp_agent["preview_bundle_hash"]
                        == gat_agent["preview_bundle_hash"]
                    ),
                }
            )
        if fp["team_success"] and gat["team_success"]:
            outcome = "BOTH_SUCCESS"
        elif gat["team_success"]:
            outcome = "GAT_ONLY_SUCCESS"
        elif fp["team_success"]:
            outcome = "FP_SHEP_ONLY_SUCCESS"
        else:
            outcome = "BOTH_FAIL"
        both_success = bool(fp["team_success"] and gat["team_success"])
        paired.append(
            {
                "scenario": scenario,
                "seed": seed,
                "pair_outcome": outcome,
                "fp_shep_failure": _failure_category(fp, fp_agents),
                "gat_failure": _failure_category(gat, gat_agents),
                "selector_disagreement_count": disagreement_count,
                "selector_disagreement": disagreement_count > 0,
                "candidate_bundle_hash_match": (
                    fp["candidate_bundle_hash"] == gat["candidate_bundle_hash"]
                ),
                "preview_bundle_hash_match": (
                    fp["preview_bundle_hash"] == gat["preview_bundle_hash"]
                ),
                "gat_minus_fp_success": int(gat["team_success"])
                - int(fp["team_success"]),
                "fp_minus_gat_collision": int(fp["collision"])
                - int(gat["collision"]),
                "fp_minus_gat_timeout": int(fp["timeout"]) - int(gat["timeout"]),
                "both_success": both_success,
                "completion_time_paired_difference_s": (
                    float(gat["completion_time_s"]) - float(fp["completion_time_s"])
                    if both_success
                    else None
                ),
                "path_length_paired_difference_m": (
                    float(gat["team_path_length_m"])
                    - float(fp["team_path_length_m"])
                    if both_success
                    else None
                ),
                "smoothness_paired_difference": (
                    float(gat["trajectory_smoothness"])
                    - float(fp["trajectory_smoothness"])
                    if both_success
                    else None
                ),
                "minimum_inter_agent_distance_difference_m": float(
                    gat["minimum_inter_agent_distance_m"]
                )
                - float(fp["minimum_inter_agent_distance_m"]),
            }
        )
    return paired, disagreements


def _mcnemar_exact(left: Sequence[bool], right: Sequence[bool]) -> dict[str, Any]:
    if len(left) != len(right):
        raise ValueError("paired outcomes must have equal length")
    left_only = sum(a and not b for a, b in zip(left, right, strict=True))
    right_only = sum(b and not a for a, b in zip(left, right, strict=True))
    discordant = left_only + right_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, index)
            for index in range(min(left_only, right_only) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "left_only": left_only,
        "right_only": right_only,
        "discordant_count": discordant,
        "exact_two_sided_p_value": float(p_value),
        "statistical_unit": "scenario_seed_team_episode",
    }


def build_statistical_tests(
    episodes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    index = {
        (row["scenario"], int(row["seed"]), row["method"]): row for row in episodes
    }
    scopes = ["overall", *sorted({str(row["scenario"]) for row in episodes})]
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "statistical_unit": "scenario_seed_team_episode",
        "individual_uav_is_independent_sample": False,
        "comparisons": {},
    }
    for baseline in (METHOD_PROPOSAL, METHOD_FP_SHEP):
        name = f"{METHOD_GAT}_vs_{baseline}"
        result["comparisons"][name] = {}
        for scope in scopes:
            keys = sorted(
                {
                    (row["scenario"], int(row["seed"]))
                    for row in episodes
                    if scope == "overall" or row["scenario"] == scope
                }
            )
            result["comparisons"][name][scope] = {}
            for metric in ("team_success", "collision", "obstacle_collision", "inter_agent_collision", "timeout"):
                baseline_values = [bool(index[(*key, baseline)][metric]) for key in keys]
                gat_values = [bool(index[(*key, METHOD_GAT)][metric]) for key in keys]
                result["comparisons"][name][scope][metric] = _mcnemar_exact(
                    baseline_values, gat_values
                )
    return result


def classify_rate_improvement(value: float, config: Mapping[str, Any]) -> str:
    thresholds = config["effect_classification"]
    if value >= float(thresholds["yes_minimum_improvement"]):
        return "YES"
    if value > float(thresholds["weak_strict_minimum_improvement"]):
        return "WEAK"
    return "NO"


def _combined_gain(improvements: Sequence[float], config: Mapping[str, Any]) -> str:
    material = float(config["effect_classification"]["yes_minimum_improvement"])
    if any(value >= material for value in improvements) and not any(
        value <= -material for value in improvements
    ):
        return "YES"
    if any(value > 0.0 for value in improvements) and not any(
        value <= -material for value in improvements
    ):
        return "WEAK"
    return "NO"


def build_conclusion(
    summaries: Sequence[Mapping[str, Any]],
    paired: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    integrity: Mapping[str, Any],
) -> dict[str, Any]:
    lookup = {(row["method"], row["scenario"]): row for row in summaries}

    def improvements(scope: str, baseline: str) -> dict[str, float]:
        base = lookup[(baseline, scope)]
        gat = lookup[(METHOD_GAT, scope)]
        return {
            "success": float(gat["team_success_rate"])
            - float(base["team_success_rate"]),
            "collision": float(base["collision_rate"])
            - float(gat["collision_rate"]),
            "timeout": float(base["timeout_rate"]) - float(gat["timeout_rate"]),
            "obstacle_collision": float(base["obstacle_collision_rate"])
            - float(gat["obstacle_collision_rate"]),
            "inter_agent_collision": float(base["inter_agent_collision_rate"])
            - float(gat["inter_agent_collision_rate"]),
        }

    vs_proposal = improvements("overall", METHOD_PROPOSAL)
    vs_fp = improvements("overall", METHOD_FP_SHEP)
    interaction = improvements("multi_agent", METHOD_FP_SHEP)
    gat_gain = _combined_gain(
        [vs_fp["success"], vs_fp["collision"], vs_fp["timeout"]], config
    )
    interaction_gain = _combined_gain(
        [interaction["success"], interaction["collision"], interaction["timeout"]],
        config,
    )
    valid = bool(integrity.get("status") == "PASSED")
    best_method = sorted(
        METHOD_ORDER,
        key=lambda method: (
            -float(lookup[(method, "overall")]["team_success_rate"]),
            float(lookup[(method, "overall")]["collision_rate"]),
            float(lookup[(method, "overall")]["timeout_rate"]),
            METHOD_ORDER.index(method),
        ),
    )[0]
    proceed_stress = valid and gat_gain in set(
        config["stress_test_decision"]["recommend_yes_if_closed_loop_gain"]
    )
    pair_counts = Counter(row["pair_outcome"] for row in paired)
    return {
        "schema_version": SCHEMA_VERSION,
        "CLOSED_LOOP_PIPELINE_VALID": "YES" if valid else "NO",
        "CANDIDATE_BUNDLE_FAIRNESS": (
            "YES" if integrity.get("candidate_bundle_fairness") else "NO"
        ),
        "PROPOSAL_RECONSTRUCTION_EQUIVALENT": (
            "YES" if integrity.get("proposal_reconstruction_equivalent") else "NO"
        ),
        "GAT_CHECKPOINT_VALID": (
            "YES" if integrity.get("gat_checkpoint_valid") else "NO"
        ),
        "GAT_VS_PROPOSAL_SUCCESS_GAIN": classify_rate_improvement(
            vs_proposal["success"], config
        ),
        "GAT_VS_FP_SHEP_SUCCESS_GAIN": classify_rate_improvement(
            vs_fp["success"], config
        ),
        "GAT_VS_FP_SHEP_COLLISION_GAIN": classify_rate_improvement(
            vs_fp["collision"], config
        ),
        "INTERACTION_RICH_CLOSED_LOOP_GAIN": interaction_gain,
        "OFFLINE_TO_CLOSED_LOOP_TRANSFER": gat_gain,
        "CLOSED_LOOP_GAT_GAIN": gat_gain,
        "FINAL_GAT_CHECKPOINT": str(config["gat_checkpoint"]),
        "RECOMMENDED_FINAL_METHOD": METHOD_NAMES[best_method],
        "PROCEED_TO_24_LAYOUT_STRESS_TEST": "YES" if proceed_stress else "NO",
        "NEXT_STEP": (
            "Ask for separate authorization before the frozen 24-layout stress test."
            if proceed_stress
            else "Retain the best validated closed-loop baseline; do not auto-tune."
        ),
        "rate_improvements_positive_is_better": {
            "GAT_vs_Proposal": vs_proposal,
            "GAT_vs_FP_SHEP": vs_fp,
            "interaction_rich_GAT_vs_FP_SHEP": interaction,
        },
        "gat_vs_fp_pair_counts": dict(pair_counts),
        "automatic_24_layout_started": False,
        "training_performed": False,
    }


def _scenario_manifest_entry(shared: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "scenario": shared["scenario"],
        "seed": int(shared["seed"]),
        "initial_condition_hash": shared["initial_condition_hash"],
        "candidate_bundle_hash": shared["candidate_bundle_hash"],
        "preview_bundle_hash": shared["preview_bundle_hash"],
        "candidate_count_per_agent": shared["candidate_count_per_agent"],
        "scene_metadata": shared["scene_metadata"],
        "scenario_snapshot": shared["scenario_snapshot"],
    }


def _gat_diagnostic_rows(shared: Mapping[str, Any]) -> list[dict[str, Any]]:
    plan = shared["plans"][METHOD_GAT]
    rows: list[dict[str, Any]] = []
    for agent_id, record in enumerate(plan["candidate_records"]):
        rows.append(
            {
                "scenario": shared["scenario"],
                "seed": int(shared["seed"]),
                "agent_id": agent_id,
                "candidate_bundle_hash": shared["candidate_bundle_hash"],
                "preview_bundle_hash": shared["preview_bundle_hash"],
                "selection_plan_hash": plan["selection_plan_hash"],
                **{
                    key: record.get(key)
                    for key in (
                        "selected_class",
                        "selected_null",
                        "selected_candidate_id",
                        "selected_proposal_rank_by_coarse_score",
                        "selected_proposal_rank_by_FP_SHEP",
                        "GAT_confidence",
                        "top1_top2_probability_margin",
                        "class_count",
                        "class_mapping_valid",
                        "logits_finite",
                        "selected_preview_progress",
                        "selected_preview_clearance",
                        "selected_preview_deviation",
                        "selected_preview_terminal_speed",
                        "selected_minimum_t_min",
                        "selected_minimum_d_min",
                        "selected_maximum_T_risk",
                        "selected_align_edge_count",
                        "class_probabilities",
                        "class_logits",
                    )
                },
            }
        )
    return rows


def evaluate_seed_set(
    *,
    config: Mapping[str, Any],
    execution_settings: Mapping[str, Any],
    multi_config: Any,
    policy: Any,
    gat_model: torch.nn.Module,
    gat_device: torch.device,
    seeds: Sequence[int],
    shared_cache: dict[tuple[str, int], dict[str, Any]],
    phase_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    episodes: list[dict[str, Any]] = []
    agents: list[dict[str, Any]] = []
    reconstruction: list[dict[str, Any]] = []
    gat_diagnostics: list[dict[str, Any]] = []
    jobs = [
        (str(scenario), int(seed), method)
        for scenario in config["formal_scenarios"]
        for seed in seeds
        for method in METHOD_ORDER
    ]
    for index, (scenario, seed, method) in enumerate(jobs, start=1):
        key = (scenario, seed)
        if key not in shared_cache:
            shared_cache[key] = build_shared_selection_bundle(
                config=config,
                execution_settings=execution_settings,
                multi_config=multi_config,
                policy=policy,
                gat_model=gat_model,
                gat_device=gat_device,
                scenario=scenario,
                seed=seed,
            )
            reconstruction.extend(shared_cache[key]["reconstruction_rows"])
            gat_diagnostics.extend(_gat_diagnostic_rows(shared_cache[key]))
        episode, agent_rows = run_method_episode(
            config=config,
            execution_settings=execution_settings,
            multi_config=multi_config,
            policy=policy,
            shared=shared_cache[key],
            method=method,
        )
        episodes.append(episode)
        agents.extend(agent_rows)
        print(
            f"[{phase_name} {index}/{len(jobs)}] {scenario} seed={seed} "
            f"{method}: {episode['termination_reason']}",
            flush=True,
        )
    return episodes, agents, reconstruction, gat_diagnostics


def build_smoke_gate(
    *,
    config: Mapping[str, Any],
    episodes: Sequence[Mapping[str, Any]],
    agents: Sequence[Mapping[str, Any]],
    shared_cache: Mapping[tuple[str, int], Mapping[str, Any]],
    gat_checkpoint_valid: bool,
    code_hash_before: Mapping[str, str],
    code_hash_after: Mapping[str, str],
) -> dict[str, Any]:
    expected = int(config["smoke_gate"]["expected_team_episode_count"])
    initial_groups: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in episodes:
        initial_groups[(row["scenario"], int(row["seed"]))].add(
            str(row["initial_condition_hash"])
        )
    gat_rows = [row for row in agents if row["method"] == METHOD_GAT]
    checks = {
        "episode_count": len(episodes) == expected,
        "all_logits_finite": all(
            bool(record["logits_finite"])
            for shared in shared_cache.values()
            for record in shared["gat_diagnostics"]
        ),
        "all_class_indices_valid": all(
            bool(record["class_mapping_valid"])
            for shared in shared_cache.values()
            for record in shared["gat_diagnostics"]
        ),
        "selected_proposal_in_actual_top_k": all(
            row["selected_candidate_id"] is None
            or 0 <= int(row["selected_candidate_id"]) < int(row["K_t"])
            for row in gat_rows
        ),
        "null_mapping_correct": all(
            not bool(row["selected_null"])
            or row["reference_selected_type"] == "null"
            for row in gat_rows
        ),
        "candidate_bundle_hash_match": all(
            len(set(shared["candidate_hashes_by_method"].values())) == 1
            for shared in shared_cache.values()
        ),
        "preview_bundle_hash_match": all(
            len(set(shared["preview_hashes_by_method"].values())) == 1
            for shared in shared_cache.values()
        ),
        "proposal_reconstruction_equivalent": all(
            bool(shared["proposal_reconstruction_equivalent"])
            for shared in shared_cache.values()
        ),
        "handoff_count_per_agent_max": all(
            int(row["maximum_handoff_count_per_agent"]) <= 1 for row in episodes
        ),
        "no_replanning": all(int(row["replanning_count"]) == 0 for row in episodes),
        "no_nan": all(
            row.get(field) is None or not np.isnan(float(row[field]))
            for row in episodes
            for field in (
                "team_path_length_m",
                "trajectory_smoothness",
                "minimum_obstacle_clearance_m",
                "minimum_inter_agent_distance_m",
            )
        ),
        "finite_required_trajectory_metrics": all(
            np.isfinite(float(row[field]))
            for row in episodes
            for field in (
                "team_path_length_m",
                "trajectory_smoothness",
                "minimum_inter_agent_distance_m",
            )
        ),
        "checkpoint_match": gat_checkpoint_valid,
        "graph_schema_match": all(
            bool(shared["graph_schema_match"]) for shared in shared_cache.values()
        ),
        "paired_initial_state": all(len(values) == 1 for values in initial_groups.values()),
        "historical_gate": all(
            bool(row["execution_historical_gate_verified"]) for row in episodes
        )
        and all(
            bool(shared["preview_historical_gate_verified"])
            for shared in shared_cache.values()
        ),
        "selection_plan_unchanged": all(
            bool(row["selection_plan_unchanged"]) for row in episodes
        ),
        "smoke_code_unchanged": dict(code_hash_before) == dict(code_hash_after),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASSED" if all(checks.values()) else "FAILED",
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "episode_count": len(episodes),
        "code_hash_before": dict(code_hash_before),
        "code_hash_after": dict(code_hash_after),
    }


def build_integrity(
    *,
    config: Mapping[str, Any],
    episodes: Sequence[Mapping[str, Any]],
    shared_cache: Mapping[tuple[str, int], Mapping[str, Any]],
    core_before: Mapping[str, str],
    core_after: Mapping[str, str],
    reuse_before: Mapping[str, str],
    reuse_after: Mapping[str, str],
    gat_checkpoint_hash_before: str,
    gat_checkpoint_hash_after: str,
    sac_checkpoint_hash_before: str,
    sac_checkpoint_hash_after: str,
    gat_model_hash_before: str,
    gat_model_hash_after: str,
    policy_hash_before: str,
    policy_hash_after: str,
) -> dict[str, Any]:
    expected = (
        len(config["formal_scenarios"])
        * len(config["formal_seeds"])
        * len(METHOD_ORDER)
    )
    candidate_fairness = all(
        len(set(shared["candidate_hashes_by_method"].values())) == 1
        and len(set(shared["preview_hashes_by_method"].values())) == 1
        for shared in shared_cache.values()
    )
    proposal_equivalent = all(
        bool(shared["proposal_reconstruction_equivalent"])
        for shared in shared_cache.values()
    )
    checks = {
        "episode_count": len(episodes) == expected,
        "candidate_bundle_fairness": candidate_fairness,
        "proposal_reconstruction_equivalent": proposal_equivalent,
        "core_method_hash_unchanged": dict(core_before) == dict(core_after),
        "smoke_reuse_hash_unchanged": dict(reuse_before) == dict(reuse_after),
        "gat_checkpoint_unchanged": gat_checkpoint_hash_before
        == gat_checkpoint_hash_after,
        "sac_checkpoint_unchanged": sac_checkpoint_hash_before
        == sac_checkpoint_hash_after,
        "gat_model_parameters_unchanged": gat_model_hash_before == gat_model_hash_after,
        "sac_policy_parameters_unchanged": policy_hash_before == policy_hash_after,
        "all_historical_gate": all(
            bool(row["execution_historical_gate_verified"]) for row in episodes
        ),
        "no_replanning": all(int(row["replanning_count"]) == 0 for row in episodes),
        "handoff_at_most_once_per_agent": all(
            int(row["maximum_handoff_count_per_agent"]) <= 1 for row in episodes
        ),
        "terminal_goals_unchanged": all(
            bool(row["terminal_task_goals_unchanged"]) for row in episodes
        ),
        "phase_not_reset": all(not bool(row["phase_reset_on_switch"]) for row in episodes),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASSED" if all(checks.values()) else "FAILED",
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "episode_count": len(episodes),
        "expected_episode_count": expected,
        "candidate_bundle_fairness": candidate_fairness,
        "proposal_reconstruction_equivalent": proposal_equivalent,
        "gat_checkpoint_valid": gat_checkpoint_hash_before
        == config["gat_checkpoint_sha256_expected"],
        "CORE_METHOD_HASH_CHANGED": "NO"
        if dict(core_before) == dict(core_after)
        else "YES",
        "core_hashes_before": dict(core_before),
        "core_hashes_after": dict(core_after),
        "smoke_reuse_hashes_before": dict(reuse_before),
        "smoke_reuse_hashes_after": dict(reuse_after),
        "gat_checkpoint_sha256_before": gat_checkpoint_hash_before,
        "gat_checkpoint_sha256_after": gat_checkpoint_hash_after,
        "sac_checkpoint_sha256_before": sac_checkpoint_hash_before,
        "sac_checkpoint_sha256_after": sac_checkpoint_hash_after,
        "training_performed": False,
        "automatic_24_layout_started": False,
    }


def _render_report(
    summaries: Sequence[Mapping[str, Any]],
    paired: Sequence[Mapping[str, Any]],
    conclusion: Mapping[str, Any],
    smoke_gate: Mapping[str, Any],
    integrity: Mapping[str, Any],
) -> str:
    overall = {row["method"]: row for row in summaries if row["scenario"] == "overall"}
    lines = [
        "# Stage-I GAT Closed-Loop Validation",
        "",
        "## Frozen protocol",
        "",
        "- Top-K=10, H_preview=4, max_steps=220, handoff=0.25 m.",
        "- Historical per-axis vector goal gate; boundary-free one-shot execution.",
        "- Scenarios: open, sparse_static, multi_agent; held-out seeds 10-29.",
        "- Proposal, FP-SHEP and GAT consume one immutable t=0 selection bundle.",
        "- No training, replanning, stress test, candidate reordering or core-method change.",
        "",
        "## Integrity",
        "",
        f"- Smoke gate: `{smoke_gate['status']}`",
        f"- Final integrity: `{integrity['status']}`",
        f"- CORE_METHOD_HASH_CHANGED: `{integrity['CORE_METHOD_HASH_CHANGED']}`",
        f"- PROPOSAL_RECONSTRUCTION_EQUIVALENT: `{conclusion['PROPOSAL_RECONSTRUCTION_EQUIVALENT']}`",
        "",
        "## Overall team results",
        "",
        "| Method | Success | Obstacle collision | Inter-agent collision | Timeout |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHOD_ORDER:
        row = overall[method]
        lines.append(
            f"| {METHOD_NAMES[method]} | {row['team_success_count']}/{row['episode_count']} "
            f"({100*row['team_success_rate']:.1f}%) | "
            f"{row['obstacle_collision_count']}/{row['episode_count']} "
            f"({100*row['obstacle_collision_rate']:.1f}%) | "
            f"{row['inter_agent_collision_count']}/{row['episode_count']} "
            f"({100*row['inter_agent_collision_rate']:.1f}%) | "
            f"{row['timeout_count']}/{row['episode_count']} "
            f"({100*row['timeout_rate']:.1f}%) |"
        )
    pair_counts = Counter(row["pair_outcome"] for row in paired)
    lines.extend(
        [
            "",
            "## GAT vs FP-SHEP paired outcomes",
            "",
            *[f"- {key}: {value}" for key, value in sorted(pair_counts.items())],
            "",
            "## Decisions",
            "",
        ]
    )
    for key in (
        "CLOSED_LOOP_PIPELINE_VALID",
        "CANDIDATE_BUNDLE_FAIRNESS",
        "GAT_CHECKPOINT_VALID",
        "GAT_VS_PROPOSAL_SUCCESS_GAIN",
        "GAT_VS_FP_SHEP_SUCCESS_GAIN",
        "GAT_VS_FP_SHEP_COLLISION_GAIN",
        "INTERACTION_RICH_CLOSED_LOOP_GAIN",
        "OFFLINE_TO_CLOSED_LOOP_TRANSFER",
        "CLOSED_LOOP_GAT_GAIN",
        "RECOMMENDED_FINAL_METHOD",
        "PROCEED_TO_24_LAYOUT_STRESS_TEST",
        "NEXT_STEP",
    ):
        lines.append(f"- `{key} = {conclusion[key]}`")
    lines.extend(
        [
            "",
            "## Stop rule",
            "",
            "The evaluation stopped after the main closed-loop validation. No 24-layout "
            "stress test, training, tuning, overlap loss, completion loss, or method "
            "redesign was started.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_experiment(config: Mapping[str, Any], output_dir: Path) -> Path:
    _assert_frozen_config(config)
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    resolved_config = copy.deepcopy(dict(config))
    resolved_config.update(
        {
            "resolved_output_dir": str(output_dir),
            "created_at": datetime.now().isoformat(),
            "python_executable": sys.executable,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
    )
    write_json(output_dir / "config.json", resolved_config)

    gat_checkpoint = (REPO_ROOT / config["gat_checkpoint"]).resolve()
    sac_checkpoint = (REPO_ROOT / config["sac_checkpoint"]).resolve()
    gat_checkpoint_hash_before = _sha256_file(gat_checkpoint)
    sac_checkpoint_hash_before = _sha256_file(sac_checkpoint)
    if gat_checkpoint_hash_before != config["gat_checkpoint_sha256_expected"]:
        raise RuntimeError("Stage-I GAT checkpoint hash mismatch")
    if sac_checkpoint_hash_before != config["sac_checkpoint_sha256_expected"]:
        raise RuntimeError("SAC checkpoint hash mismatch")

    core_before = _file_hashes(CORE_METHOD_PATHS)
    reuse_before = _file_hashes(SMOKE_REUSE_PATHS)
    execution_settings = _build_execution_settings(config)
    multi_config = build_single_distribution_multi_config(
        num_agents=int(config["num_agents"]), max_steps=int(config["max_steps"])
    )
    policy, loaded_sac = _load_policy(execution_settings, multi_config)
    if loaded_sac.resolve() != sac_checkpoint:
        raise RuntimeError("SAC loader resolved a different checkpoint")
    policy_hash_before = _policy_parameter_sha256(policy)

    stage1_config = _load_json((REPO_ROOT / config["stage1_config"]).resolve())
    gat_device = resolve_device(stage1_config["training"]["device"])
    gat_model = load_model_checkpoint(gat_checkpoint, stage1_config, gat_device)
    gat_model_hash_before = _model_hash(gat_model)
    if any(parameter.requires_grad for parameter in gat_model.parameters()):
        for parameter in gat_model.parameters():
            parameter.requires_grad_(False)
    gat_model.eval()

    shared_cache: dict[tuple[str, int], dict[str, Any]] = {}
    smoke_episodes, smoke_agents, reconstruction_rows, gat_diagnostics = (
        evaluate_seed_set(
            config=config,
            execution_settings=execution_settings,
            multi_config=multi_config,
            policy=policy,
            gat_model=gat_model,
            gat_device=gat_device,
            seeds=config["smoke_seeds"],
            shared_cache=shared_cache,
            phase_name="smoke",
        )
    )
    reuse_after_smoke = _file_hashes(SMOKE_REUSE_PATHS)
    smoke_gate = build_smoke_gate(
        config=config,
        episodes=smoke_episodes,
        agents=smoke_agents,
        shared_cache=shared_cache,
        gat_checkpoint_valid=gat_checkpoint_hash_before
        == config["gat_checkpoint_sha256_expected"],
        code_hash_before=reuse_before,
        code_hash_after=reuse_after_smoke,
    )
    write_csv(output_dir / "smoke_results.csv", smoke_episodes)
    write_json(output_dir / "smoke_gate.json", smoke_gate)
    write_json(
        output_dir / "scenario_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "entries": [
                _scenario_manifest_entry(shared)
                for _, shared in sorted(shared_cache.items())
            ],
        },
    )
    write_csv(output_dir / "proposal_reconstruction_audit.csv", reconstruction_rows)
    write_csv(output_dir / "gat_selection_diagnostics.csv", gat_diagnostics)
    if smoke_gate["status"] != "PASSED":
        write_json(
            output_dir / "conclusion.json",
            {
                "schema_version": SCHEMA_VERSION,
                "CLOSED_LOOP_PIPELINE_VALID": "NO",
                "CANDIDATE_BUNDLE_FAIRNESS": "NO",
                "GAT_CHECKPOINT_VALID": "YES",
                "PROPOSAL_RECONSTRUCTION_EQUIVALENT": (
                    "YES"
                    if smoke_gate["checks"].get(
                        "proposal_reconstruction_equivalent", False
                    )
                    else "NO"
                ),
                "stop_reason": "SMOKE_GATE_FAILED",
                "failed_checks": smoke_gate["failed_checks"],
                "automatic_24_layout_started": False,
            },
        )
        (output_dir / "FINAL_REPORT.md").write_text(
            "# Stage-I GAT Closed-Loop Validation\n\n"
            f"Smoke gate failed: {smoke_gate['failed_checks']}. Formal evaluation was not started.\n",
            encoding="utf-8",
        )
        return output_dir

    # Smoke rows may be reused only when all execution, logging, config, and
    # selector-interface hashes remain exactly unchanged in this same process.
    if reuse_after_smoke != _file_hashes(SMOKE_REUSE_PATHS):
        raise RuntimeError("smoke code changed before formal continuation")
    remaining_seeds = [
        seed for seed in config["formal_seeds"] if seed not in config["smoke_seeds"]
    ]
    formal_episodes, formal_agents, formal_reconstruction, formal_gat = (
        evaluate_seed_set(
            config=config,
            execution_settings=execution_settings,
            multi_config=multi_config,
            policy=policy,
            gat_model=gat_model,
            gat_device=gat_device,
            seeds=remaining_seeds,
            shared_cache=shared_cache,
            phase_name="formal",
        )
    )
    episodes = smoke_episodes + formal_episodes
    agents = smoke_agents + formal_agents
    reconstruction_rows.extend(formal_reconstruction)
    gat_diagnostics.extend(formal_gat)

    core_after = _file_hashes(CORE_METHOD_PATHS)
    reuse_after = _file_hashes(SMOKE_REUSE_PATHS)
    gat_checkpoint_hash_after = _sha256_file(gat_checkpoint)
    sac_checkpoint_hash_after = _sha256_file(sac_checkpoint)
    gat_model_hash_after = _model_hash(gat_model)
    policy_hash_after = _policy_parameter_sha256(policy)
    integrity = build_integrity(
        config=config,
        episodes=episodes,
        shared_cache=shared_cache,
        core_before=core_before,
        core_after=core_after,
        reuse_before=reuse_before,
        reuse_after=reuse_after,
        gat_checkpoint_hash_before=gat_checkpoint_hash_before,
        gat_checkpoint_hash_after=gat_checkpoint_hash_after,
        sac_checkpoint_hash_before=sac_checkpoint_hash_before,
        sac_checkpoint_hash_after=sac_checkpoint_hash_after,
        gat_model_hash_before=gat_model_hash_before,
        gat_model_hash_after=gat_model_hash_after,
        policy_hash_before=policy_hash_before,
        policy_hash_after=policy_hash_after,
    )
    if integrity["status"] != "PASSED":
        raise RuntimeError(f"final integrity failed: {integrity['failed_checks']}")

    summaries = aggregate_method_summary(episodes)
    paired, disagreements = build_gat_fp_pairing(episodes, agents)
    statistical_tests = build_statistical_tests(episodes)
    conclusion = build_conclusion(summaries, paired, config, integrity)

    write_json(output_dir / "integrity_manifest.json", integrity)
    write_json(
        output_dir / "scenario_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "entries": [
                _scenario_manifest_entry(shared)
                for _, shared in sorted(shared_cache.items())
            ],
        },
    )
    write_csv(output_dir / "episode_results.csv", episodes)
    write_csv(output_dir / "per_agent_results.csv", agents)
    write_csv(output_dir / "method_summary.csv", summaries)
    write_csv(
        output_dir / "scenario_summary.csv",
        [row for row in summaries if row["scenario"] != "overall"],
    )
    write_csv(output_dir / "paired_gat_vs_fp_shep.csv", paired)
    write_csv(output_dir / "selector_disagreement.csv", disagreements)
    write_csv(output_dir / "gat_selection_diagnostics.csv", gat_diagnostics)
    write_csv(output_dir / "proposal_reconstruction_audit.csv", reconstruction_rows)
    write_json(output_dir / "statistical_tests.json", statistical_tests)
    write_json(output_dir / "conclusion.json", conclusion)
    (output_dir / "FINAL_REPORT.md").write_text(
        _render_report(summaries, paired, conclusion, smoke_gate, integrity),
        encoding="utf-8",
    )
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "episode_count": len(episodes),
                "runtime_seconds": elapsed,
                "smoke_gate": smoke_gate["status"],
                "integrity": integrity["status"],
                "CLOSED_LOOP_GAT_GAIN": conclusion["CLOSED_LOOP_GAT_GAIN"],
                "automatic_24_layout_started": False,
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
    config_path = args.config.expanduser().resolve()
    config = _load_json(config_path)
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
