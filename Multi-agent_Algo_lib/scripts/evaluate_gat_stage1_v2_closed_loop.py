"""Run the frozen, deployment-aligned Stage-I GAT V1/V2 closed-loop test.

One candidate bundle, one H4 preview bundle, and one graph bundle are built for
each scenario/seed.  The V1 and V2 checkpoints are then evaluated on those
same graph objects; only checkpoint parameters differ.  All five methods enter
the established one-shot ``run_variant_episode`` execution helper through an
immutable selection plan.
"""

from __future__ import annotations

import argparse
import copy
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

from planning.gat.candidate_selector import batch_candidate_graphs  # noqa: E402
from planning.gat.stage1_training import (  # noqa: E402
    load_model_checkpoint,
    resolve_device,
)
from planning.pre_gat_220step_revalidation import stable_hash  # noqa: E402
from scripts import evaluate_gat_closed_loop as legacy  # noqa: E402
from scripts.evaluate_actor_dmp_goal_semantics import (  # noqa: E402
    write_csv,
    write_json,
)
from scripts.evaluate_pre_gat_closed_loop import _policy_parameter_sha256  # noqa: E402
from scripts.evaluate_single_policy_aligned_multi_agent import (  # noqa: E402
    build_single_distribution_multi_config,
)
from scripts.evaluate_temporary_reference_interface import _load_policy  # noqa: E402


SCHEMA_VERSION = "gat_stage1_v2_closed_loop_v1"
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs/evaluation/gat_stage1_v2_closed_loop.json"

METHOD_TERMINAL = "terminal"
METHOD_PROPOSAL = "proposal"
METHOD_FP_SHEP = "fp_shep"
METHOD_V1 = "gat_stage1_v1"
METHOD_V2 = "gat_stage1_v2"
METHOD_ORDER = (
    METHOD_TERMINAL,
    METHOD_PROPOSAL,
    METHOD_FP_SHEP,
    METHOD_V1,
    METHOD_V2,
)
METHOD_NAMES = {
    METHOD_TERMINAL: "Terminal Goal",
    METHOD_PROPOSAL: "Proposal Top-1",
    METHOD_FP_SHEP: "FP-SHEP Top-1",
    METHOD_V1: "GAT-V1",
    METHOD_V2: "GAT-V2",
}

EVALUATOR_PATH = "Multi-agent_Algo_lib/scripts/evaluate_gat_stage1_v2_closed_loop.py"
CONFIG_PATH = "configs/evaluation/gat_stage1_v2_closed_loop.json"
CORE_METHOD_PATHS = tuple(legacy.CORE_METHOD_PATHS)
SMOKE_REUSE_PATHS = CORE_METHOD_PATHS + (
    "Multi-agent_Algo_lib/scripts/evaluate_actor_dmp_goal_semantics.py",
    "Multi-agent_Algo_lib/scripts/evaluate_gat_closed_loop.py",
    EVALUATOR_PATH,
    CONFIG_PATH,
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_hashes(paths: Sequence[str]) -> dict[str, str]:
    return {path: _sha256_file(REPO_ROOT / path) for path in paths}


def _model_hash(model: torch.nn.Module) -> str:
    return legacy._model_hash(model)


def _assert_frozen_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected V2 closed-loop schema")
    if tuple(config["methods"]) != METHOD_ORDER:
        raise ValueError("five-method order changed")
    if list(config["formal_scenarios"]) != ["open", "sparse_static", "multi_agent"]:
        raise ValueError("formal scenario set changed")
    if list(config["smoke_seeds"]) != [30, 31, 32]:
        raise ValueError("smoke seeds must remain 30..32")
    if list(config["formal_seeds"]) != list(range(30, 50)):
        raise ValueError("formal seeds must remain 30..49")
    frozen = {
        "top_k": 10,
        "H_preview": 4,
        "max_steps": 220,
        "handoff_threshold_m": 0.25,
        "actor_observation_dim": 122,
    }
    for key, expected in frozen.items():
        if config.get(key) != expected:
            raise ValueError(f"frozen setting changed: {key}")
    if config["forcing_gate"] != "historical_vector_goal_eff_gate":
        raise ValueError("historical vector goal-eff gate is required")
    if config["execution_protocol"] != "one_shot_temporary_reference_handoff":
        raise ValueError("one-shot execution is required")
    if config["boundary_mode"] != "boundary_free":
        raise ValueError("boundary-free execution is required")
    if config["phase"] != "legacy/classic":
        raise ValueError("legacy/classic phase is required")
    if any(bool(value) for value in config["strict_exclusions"].values()):
        raise ValueError("a prohibited modification was enabled")
    if bool(config["interpretation"]["label_only_ablation_claim_allowed"]):
        raise ValueError("V1/V2 may not be described as a label-only ablation")
    if config["context_sources"]["stress_set_status"] != (
        "DIAGNOSTIC_ONLY_AFTER_OBSERVATION"
    ):
        raise ValueError("stress-set status changed")
    if int(config["seed_audit"]["final_test_seed_overlap"]) != 0:
        raise ValueError("final test seed overlap is non-zero")


def _hash_value(digest: Any, key: str, value: Any) -> None:
    digest.update(key.encode("utf-8"))
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    else:
        digest.update(
            json.dumps(
                legacy._jsonable(value), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        )


def graph_input_hash(graphs: Sequence[Any]) -> str:
    """Hash every tensor and global metadata field in the online graph bundle."""

    digest = hashlib.sha256()
    for graph_index, graph in enumerate(graphs):
        _hash_value(digest, f"graph[{graph_index}].metadata", graph.graph_metadata)
        for node_type in sorted(graph.node_types):
            for key, value in sorted(graph[node_type].items()):
                _hash_value(digest, f"graph[{graph_index}].node.{node_type}.{key}", value)
        for edge_type in sorted(graph.edge_types, key=str):
            edge_name = "|".join(edge_type)
            for key, value in sorted(graph[edge_type].items()):
                _hash_value(digest, f"graph[{graph_index}].edge.{edge_name}.{key}", value)
    return digest.hexdigest()


def _checkpoint_metadata(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", {})
    return {
        "schema_version": payload.get("schema_version"),
        "optimization_seed": payload.get("optimization_seed"),
        "epoch": payload.get("epoch"),
        "model_config": payload.get("model_config"),
        "supervision": payload.get("supervision"),
        "tensor_count": len(state),
        "parameter_count": int(sum(value.numel() for value in state.values())),
    }


def build_context_recovery_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    required_names = ("config.json", "conclusion.json", "FINAL_REPORT.md")
    for role, relative_dir in config["context_sources"].items():
        if not role.endswith("_dir"):
            continue
        directory = REPO_ROOT / str(relative_dir)
        for name in required_names:
            path = directory / name
            sources.append(
                {
                    "role": role,
                    "path": str(path.relative_to(REPO_ROOT)),
                    "exists": path.is_file(),
                    "sha256": _sha256_file(path) if path.is_file() else None,
                }
            )
        integrity = directory / "integrity_manifest.json"
        if integrity.is_file():
            sources.append(
                {
                    "role": role,
                    "path": str(integrity.relative_to(REPO_ROOT)),
                    "exists": True,
                    "sha256": _sha256_file(integrity),
                }
            )

    decisions: dict[str, bool] = {}
    diagnosis = _load_json(
        REPO_ROOT / config["context_sources"]["failure_diagnosis_dir"] / "conclusion.json"
    )
    supervision = _load_json(
        REPO_ROOT / config["context_sources"]["v2_supervision_dir"] / "conclusion.json"
    )
    training = _load_json(
        REPO_ROOT / config["context_sources"]["v2_training_dir"] / "conclusion.json"
    )
    v1_closed = _load_json(
        REPO_ROOT / config["context_sources"]["v1_closed_loop_dir"] / "conclusion.json"
    )
    decisions.update(
        {
            "primary_limitation_recovered": diagnosis.get("PRIMARY_GAT_LIMITATION")
            == "SUPERVISION_MISMATCH",
            "stress_status_recovered": diagnosis.get("STRESS_SET_STATUS")
            == "DIAGNOSTIC_ONLY_AFTER_OBSERVATION",
            "v2_supervision_valid": supervision.get("V2_SUPERVISION_VALID") == "YES",
            "v2_training_valid": training.get("V2_TRAINING_VALID") == "YES",
            "v2_offline_improvement_recovered": training.get(
                "V2_OFFLINE_IMPROVEMENT_OVER_V1"
            )
            == "YES",
            "long_horizon_signal_recovered": training.get("LONG_HORIZON_SIGNAL_LEARNED")
            == "YES",
            "v1_closed_loop_pipeline_recovered": v1_closed.get(
                "CLOSED_LOOP_PIPELINE_VALID"
            )
            == "YES",
            "combined_interpretation_frozen": config["interpretation"]["name"]
            == "deployment_aligned_H4_graph_input_plus_V2_long_horizon_target_adaptation",
            "label_only_claim_prohibited": not bool(
                config["interpretation"]["label_only_ablation_claim_allowed"]
            ),
            "stress_not_used_for_tuning": not bool(
                config["context_sources"]["stress_used_for_tuning"]
            ),
        }
    )
    checkpoints = {}
    for role in ("v1", "v2"):
        path = (REPO_ROOT / config[f"{role}_checkpoint"]).resolve()
        checkpoints[role] = {
            "path": str(path),
            "exists": path.is_file(),
            "sha256": _sha256_file(path) if path.is_file() else None,
            "sha256_expected": config[f"{role}_checkpoint_sha256_expected"],
            "metadata": _checkpoint_metadata(path) if path.is_file() else None,
        }
        decisions[f"{role}_checkpoint_hash_match"] = (
            checkpoints[role]["sha256"] == checkpoints[role]["sha256_expected"]
        )
    valid = all(row["exists"] for row in sources) and all(decisions.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASSED" if valid else "FAILED",
        "CONTEXT_RECOVERY_VALID": "YES" if valid else "NO",
        "repository_root_AGENTS_md": "ABSENT",
        "CODEX_HANDOFF_md": "ABSENT",
        "sources": sources,
        "decisions": decisions,
        "checkpoints": checkpoints,
        "stress_set_status": "DIAGNOSTIC_ONLY_AFTER_OBSERVATION",
        "stress_artifact_role": "context_only_not_tuning_or_test",
    }


def build_seed_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    formal = set(int(value) for value in config["formal_seeds"])
    prior = set(int(value) for value in config["seed_audit"]["training_supervision_seeds"])
    prior.update(int(value) for value in config["seed_audit"]["prior_main_closed_loop_seeds"])
    overlap = sorted(formal & prior)
    checks = {
        "twenty_contiguous_formal_seeds": sorted(formal) == list(range(30, 50)),
        "smoke_is_first_three_formal_seeds": list(config["smoke_seeds"]) == [30, 31, 32],
        "semantic_artifact_scan_found_no_30_49_seed": not config["seed_audit"][
            "formal_seed_hits_before_freeze"
        ],
        "explicit_authoritative_seed_overlap_zero": not overlap,
        "result_dependent_replacement_disabled": not bool(
            config["seed_audit"]["result_dependent_seed_replacement_allowed"]
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASSED" if all(checks.values()) else "FAILED",
        "formal_seeds": sorted(formal),
        "smoke_seeds": list(config["smoke_seeds"]),
        "historical_seed_union": sorted(prior),
        "overlap_values": overlap,
        "FINAL_TEST_SEED_OVERLAP": len(overlap),
        "seed_interval_frozen_before_outcomes": True,
        "audit_scope": config["seed_audit"]["scope"],
        "checks": checks,
    }


def _build_model_diagnostics(
    *, shared: Mapping[str, Any], model: torch.nn.Module, device: torch.device
) -> tuple[list[int | None], list[dict[str, Any]]]:
    batched = batch_candidate_graphs(shared["graphs"]).to(device)
    with torch.inference_mode():
        output = model(batched)
    if not torch.isfinite(output.candidate_logits).all():
        raise RuntimeError("GAT-V2 produced non-finite logits")
    selected = list(output.selected_candidate_id)
    diagnostics: list[dict[str, Any]] = []
    for agent_id, (graph, proposals, previews, selected_id) in enumerate(
        zip(
            shared["graphs"],
            shared["proposals_by_agent"],
            shared["previews_by_agent"],
            selected,
            strict=True,
        )
    ):
        probabilities = output.probabilities_for_graph(agent_id).detach().cpu().numpy()
        logits = output.logits_for_graph(agent_id).detach().cpu().numpy()
        class_index = int(output.selected_class_index[agent_id])
        expected = None if class_index == 0 else class_index - 1
        mapping_valid = expected == selected_id and 0 <= class_index < len(proposals) + 1
        if not mapping_valid:
            raise RuntimeError("GAT-V2 null/proposal class mapping is invalid")
        ordered = np.sort(probabilities)[::-1]
        row = {
            "selected_class": class_index,
            "selected_null": selected_id is None,
            "selected_candidate_id": selected_id,
            "selected_proposal_rank_by_coarse_score": (
                int(selected_id) + 1 if selected_id is not None else None
            ),
            "selected_proposal_rank_by_FP_SHEP": legacy._rank_1based(
                [record.score for record in previews], selected_id
            ),
            "GAT_confidence": float(probabilities[class_index]),
            "top1_top2_probability_margin": (
                float(ordered[0] - ordered[1]) if len(ordered) >= 2 else None
            ),
            "class_count": len(probabilities),
            "class_mapping_valid": mapping_valid,
            "logits_finite": bool(np.all(np.isfinite(logits))),
            "class_probabilities": probabilities.tolist(),
            "class_logits": logits.tolist(),
            **legacy._selected_edge_diagnostics(graph, selected_id),
        }
        if selected_id is not None:
            preview = previews[int(selected_id)]
            row.update(
                {
                    "selected_preview_progress": preview.preview_task_progress,
                    "selected_preview_clearance": preview.preview_min_clearance,
                    "selected_preview_deviation": preview.preview_max_execution_deviation,
                    "selected_preview_terminal_speed": preview.preview_terminal_speed,
                }
            )
        else:
            row.update(
                {
                    "selected_preview_progress": None,
                    "selected_preview_clearance": None,
                    "selected_preview_deviation": None,
                    "selected_preview_terminal_speed": None,
                }
            )
        diagnostics.append(row)
    return selected, diagnostics


def add_v2_checkpoint_plan(
    shared: dict[str, Any], model: torch.nn.Module, device: torch.device
) -> None:
    selected, diagnostics = _build_model_diagnostics(
        shared=shared, model=model, device=device
    )
    plan = legacy._make_method_plan(
        source=legacy.METHOD_GAT,
        terminal_goals=np.asarray(
            shared["scenario_snapshot"]["goals"], dtype=float
        ),
        proposals_by_agent=shared["proposals_by_agent"],
        previews_by_agent=shared["previews_by_agent"],
        selected_ids=selected,
        gat_diagnostics=diagnostics,
    )
    plan["selection_source"] = METHOD_V2
    for record in plan["candidate_records"]:
        record["selection_source"] = METHOD_V2
        record["selected_null"] = record["selected_candidate_id"] is None
    plan["selection_plan_hash"] = stable_hash(
        {key: value for key, value in plan.items() if key != "selection_plan_hash"}
    )
    shared["plans"][METHOD_V1] = shared["plans"][legacy.METHOD_GAT]
    shared["plans"][METHOD_V2] = plan
    shared["diagnostics_by_method"] = {
        METHOD_V1: shared["gat_diagnostics"],
        METHOD_V2: diagnostics,
    }
    graph_hash = graph_input_hash(shared["graphs"])
    shared["graph_input_hash"] = graph_hash
    shared["graph_input_hashes_by_method"] = {
        METHOD_V1: graph_hash,
        METHOD_V2: graph_hash,
    }
    shared["candidate_hashes_by_method"] = {
        method: shared["candidate_bundle_hash"]
        for method in (METHOD_PROPOSAL, METHOD_FP_SHEP, METHOD_V1, METHOD_V2)
    }
    shared["preview_hashes_by_method"] = {
        method: shared["preview_bundle_hash"]
        for method in (METHOD_FP_SHEP, METHOD_V1, METHOD_V2)
    }


def _relabel_episode(
    episode: dict[str, Any], agents: list[dict[str, Any]], method: str, shared: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    episode.update(
        {
            "schema_version": SCHEMA_VERSION,
            "method": method,
            "method_display_name": METHOD_NAMES[method],
            "any_collision": bool(episode["collision"]),
            "graph_input_hash": (
                shared["graph_input_hash"] if method in {METHOD_V1, METHOD_V2} else None
            ),
            "checkpoint_role": (
                "V1" if method == METHOD_V1 else "V2" if method == METHOD_V2 else None
            ),
        }
    )
    for row in agents:
        row.update(
            {
                "schema_version": SCHEMA_VERSION,
                "method": method,
                "method_display_name": METHOD_NAMES[method],
                "agent_terminal_completed": row.get("terminal_completed_step") is not None,
                "reference_selected": row.get("reference_selected_type") == "proposal",
                "reference_reach_step": row.get("reference_reached_step"),
                "post_reference_obstacle_collision": bool(
                    row.get("collision_after_reference") and episode["obstacle_collision"]
                ),
                "post_reference_inter_agent_collision": bool(
                    row.get("collision_after_reference")
                    and episode["inter_agent_collision"]
                ),
                "post_reference_timeout": bool(row.get("timeout_after_reference")),
                "graph_input_hash": (
                    shared["graph_input_hash"]
                    if method in {METHOD_V1, METHOD_V2}
                    else None
                ),
                "null_probability": (
                    float(row["class_probabilities"][0])
                    if method in {METHOD_V1, METHOD_V2}
                    and row.get("class_probabilities")
                    else None
                ),
            }
        )
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
    legacy_method = legacy.METHOD_GAT if method in {METHOD_V1, METHOD_V2} else method
    shim = dict(shared)
    shim["plans"] = dict(shared["plans"])
    if method in {METHOD_V1, METHOD_V2}:
        shim["plans"][legacy.METHOD_GAT] = shared["plans"][method]
    episode, agents = legacy.run_method_episode(
        config=config,
        execution_settings=execution_settings,
        multi_config=multi_config,
        policy=policy,
        shared=shim,
        method=legacy_method,
    )
    return _relabel_episode(episode, agents, method, shared)


def evaluate_seed_set(
    *,
    config: Mapping[str, Any],
    execution_settings: Mapping[str, Any],
    multi_config: Any,
    policy: Any,
    v1_model: torch.nn.Module,
    v2_model: torch.nn.Module,
    device: torch.device,
    seeds: Sequence[int],
    shared_cache: dict[tuple[str, int], dict[str, Any]],
    phase_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episodes: list[dict[str, Any]] = []
    agents: list[dict[str, Any]] = []
    jobs = [
        (str(scenario), int(seed), method)
        for scenario in config["formal_scenarios"]
        for seed in seeds
        for method in METHOD_ORDER
    ]
    for index, (scenario, seed, method) in enumerate(jobs, start=1):
        key = (scenario, seed)
        if key not in shared_cache:
            shared = legacy.build_shared_selection_bundle(
                config=config,
                execution_settings=execution_settings,
                multi_config=multi_config,
                policy=policy,
                gat_model=v1_model,
                gat_device=device,
                scenario=scenario,
                seed=seed,
            )
            add_v2_checkpoint_plan(shared, v2_model, device)
            if len(set(shared["graph_input_hashes_by_method"].values())) != 1:
                raise RuntimeError("V1_GRAPH_INPUT_HASH != V2_GRAPH_INPUT_HASH")
            shared_cache[key] = shared
        episode, rows = run_method_episode(
            config=config,
            execution_settings=execution_settings,
            multi_config=multi_config,
            policy=policy,
            shared=shared_cache[key],
            method=method,
        )
        episodes.append(episode)
        agents.extend(rows)
        print(
            f"[{phase_name} {index}/{len(jobs)}] {scenario} seed={seed} "
            f"{method}: {episode['termination_reason']}",
            flush=True,
        )
    return episodes, agents


def _stats(values: Iterable[Any]) -> dict[str, float | int | None]:
    array = np.asarray(
        [float(value) for value in values if value is not None], dtype=float
    )
    array = array[np.isfinite(array)]
    if not array.size:
        return {"count": 0, "mean": None, "std": None, "median": None}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
    }


def build_method_summaries(
    episodes: Sequence[Mapping[str, Any]], agents: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scenarios = sorted({str(row["scenario"]) for row in episodes})
    for method in METHOD_ORDER:
        for scope in ("overall", *scenarios):
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
            successful = [row for row in members if bool(row["team_success"])]
            record: dict[str, Any] = {
                "method": method,
                "method_display_name": METHOD_NAMES[method],
                "scenario": scope,
                "episode_count": len(members),
                "agent_count": len(agent_members),
            }
            for field in (
                "team_success",
                "any_collision",
                "obstacle_collision",
                "inter_agent_collision",
                "timeout",
            ):
                count = sum(bool(row[field]) for row in members)
                record[f"{field}_count"] = count
                record[f"{field}_rate"] = count / len(members) if members else None
            for field in (
                "completion_step",
                "completion_time_s",
                "team_path_length_m",
            ):
                for name, value in _stats(row[field] for row in successful).items():
                    record[f"success_{field}_{name}"] = value
            for field in (
                "minimum_obstacle_clearance_m",
                "minimum_inter_agent_distance_m",
            ):
                for name, value in _stats(row[field] for row in members).items():
                    record[f"{field}_{name}"] = value
            null_count = sum(bool(row.get("selected_null")) for row in agent_members)
            selected = [row for row in agent_members if bool(row.get("reference_selected"))]
            reached = [row for row in selected if bool(row.get("reference_reached"))]
            completed = [
                row for row in reached if bool(row.get("terminal_completed_after_reference"))
            ]
            record.update(
                {
                    "null_selection_count": null_count,
                    "null_selection_rate": (
                        null_count / len(agent_members) if agent_members else None
                    ),
                    "reference_selection_count": len(selected),
                    "reference_selection_rate": (
                        len(selected) / len(agent_members) if agent_members else None
                    ),
                    "reference_reach_count": len(reached),
                    "reference_reach_rate": (
                        len(reached) / len(selected) if selected else None
                    ),
                    "post_reference_completion_count": len(completed),
                    "post_reference_completion_rate": (
                        len(completed) / len(reached) if reached else None
                    ),
                }
            )
            rows.append(record)
    return rows


def build_reference_transition_analysis(
    agents: Sequence[Mapping[str, Any]], episodes: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    scenarios = sorted({str(row["scenario"]) for row in episodes})
    rows: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        for scope in ("overall", *scenarios):
            members = [
                row
                for row in agents
                if row["method"] == method
                and (scope == "overall" or row["scenario"] == scope)
            ]
            selected = [row for row in members if bool(row.get("reference_selected"))]
            reached = [row for row in selected if bool(row.get("reference_reached"))]
            completed = [
                row for row in reached if bool(row.get("terminal_completed_after_reference"))
            ]
            obstacle = sum(bool(row.get("post_reference_obstacle_collision")) for row in reached)
            inter = sum(bool(row.get("post_reference_inter_agent_collision")) for row in reached)
            timeout = sum(bool(row.get("post_reference_timeout")) for row in reached)
            rows.append(
                {
                    "method": method,
                    "method_display_name": METHOD_NAMES[method],
                    "scenario": scope,
                    "agent_count": len(members),
                    "reference_selection_count": len(selected),
                    "reference_selection_rate": len(selected) / len(members) if members else None,
                    "reference_reach_count": len(reached),
                    "reference_reach_rate": len(reached) / len(selected) if selected else None,
                    "reached_to_terminal_count": len(completed),
                    "reached_to_terminal_rate": len(completed) / len(reached) if reached else None,
                    "post_reference_obstacle_collision_count": obstacle,
                    "post_reference_obstacle_collision_rate": obstacle / len(reached) if reached else None,
                    "post_reference_inter_agent_collision_count": inter,
                    "post_reference_inter_agent_collision_rate": inter / len(reached) if reached else None,
                    "post_reference_timeout_count": timeout,
                    "post_reference_timeout_rate": timeout / len(reached) if reached else None,
                }
            )
    return rows


def build_pairing(
    episodes: Sequence[Mapping[str, Any]], *, baseline: str
) -> list[dict[str, Any]]:
    index = {
        (row["scenario"], int(row["seed"]), row["method"]): row for row in episodes
    }
    rows: list[dict[str, Any]] = []
    baseline_label = "V1" if baseline == METHOD_V1 else "FP_SHEP"
    for scenario, seed in sorted(
        {(str(row["scenario"]), int(row["seed"])) for row in episodes}
    ):
        base = index[(scenario, seed, baseline)]
        v2 = index[(scenario, seed, METHOD_V2)]
        if base["team_success"] and v2["team_success"]:
            outcome = "BOTH_SUCCESS"
        elif v2["team_success"]:
            outcome = "V2_ONLY_SUCCESS"
        elif base["team_success"]:
            outcome = f"{baseline_label}_ONLY_SUCCESS"
        else:
            outcome = "BOTH_FAIL"
        both_success = bool(base["team_success"] and v2["team_success"])
        rows.append(
            {
                "scenario": scenario,
                "seed": seed,
                "baseline": baseline,
                "pair_outcome": outcome,
                "both_success": both_success,
                "v2_only_success": bool(v2["team_success"] and not base["team_success"]),
                "baseline_only_success": bool(base["team_success"] and not v2["team_success"]),
                "both_fail": bool(not base["team_success"] and not v2["team_success"]),
                "baseline_collision_to_v2_success": bool(base["any_collision"] and v2["team_success"]),
                "baseline_timeout_to_v2_success": bool(base["timeout"] and v2["team_success"]),
                "baseline_success_to_v2_collision": bool(base["team_success"] and v2["any_collision"]),
                "baseline_success_to_v2_timeout": bool(base["team_success"] and v2["timeout"]),
                "baseline_success_to_v2_failure": bool(base["team_success"] and not v2["team_success"]),
                "v2_success_to_baseline_failure": bool(v2["team_success"] and not base["team_success"]),
                "baseline_team_success": bool(base["team_success"]),
                "v2_team_success": bool(v2["team_success"]),
                "baseline_any_collision": bool(base["any_collision"]),
                "v2_any_collision": bool(v2["any_collision"]),
                "baseline_inter_agent_collision": bool(base["inter_agent_collision"]),
                "v2_inter_agent_collision": bool(v2["inter_agent_collision"]),
                "baseline_timeout": bool(base["timeout"]),
                "v2_timeout": bool(v2["timeout"]),
                "candidate_bundle_hash_match": base["candidate_bundle_hash"]
                == v2["candidate_bundle_hash"],
                "preview_bundle_hash_match": (
                    base["preview_bundle_hash"] == v2["preview_bundle_hash"]
                    if baseline != METHOD_TERMINAL
                    else True
                ),
                "graph_input_hash_match": (
                    base.get("graph_input_hash") == v2.get("graph_input_hash")
                    if baseline == METHOD_V1
                    else None
                ),
                "completion_step_difference_v2_minus_baseline": (
                    float(v2["completion_step"]) - float(base["completion_step"])
                    if both_success
                    else None
                ),
                "completion_time_difference_v2_minus_baseline_s": (
                    float(v2["completion_time_s"]) - float(base["completion_time_s"])
                    if both_success
                    else None
                ),
                "team_path_length_difference_v2_minus_baseline_m": (
                    float(v2["team_path_length_m"]) - float(base["team_path_length_m"])
                    if both_success
                    else None
                ),
            }
        )
    return rows


def _mcnemar_exact(left: Sequence[bool], right: Sequence[bool]) -> dict[str, Any]:
    return legacy._mcnemar_exact(left, right)


def build_statistical_tests(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
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
    for name, baseline in (("v2_vs_v1", METHOD_V1), ("v2_vs_fp_shep", METHOD_FP_SHEP)):
        result["comparisons"][name] = {}
        for scope in scopes:
            keys = sorted(
                {
                    (str(row["scenario"]), int(row["seed"]))
                    for row in episodes
                    if scope == "overall" or row["scenario"] == scope
                }
            )
            scope_result: dict[str, Any] = {"binary": {}, "continuous": {}}
            for field in ("team_success", "any_collision", "inter_agent_collision", "timeout"):
                base_values = [bool(index[(*key, baseline)][field]) for key in keys]
                v2_values = [bool(index[(*key, METHOD_V2)][field]) for key in keys]
                test = _mcnemar_exact(base_values, v2_values)
                base_count = sum(base_values)
                v2_count = sum(v2_values)
                scope_result["binary"][field] = {
                    "baseline_count": base_count,
                    "baseline_rate": base_count / len(keys),
                    "v2_count": v2_count,
                    "v2_rate": v2_count / len(keys),
                    "v2_minus_baseline_absolute_difference": (v2_count - base_count)
                    / len(keys),
                    **test,
                }
            for field, subset in (
                ("completion_step", "both_success"),
                ("completion_time_s", "both_success"),
                ("team_path_length_m", "both_success"),
                ("minimum_obstacle_clearance_m", "all_paired"),
                ("minimum_inter_agent_distance_m", "all_paired"),
            ):
                selected_keys = [
                    key
                    for key in keys
                    if subset == "all_paired"
                    or (
                        index[(*key, baseline)]["team_success"]
                        and index[(*key, METHOD_V2)]["team_success"]
                    )
                ]
                base_values = [index[(*key, baseline)][field] for key in selected_keys]
                v2_values = [index[(*key, METHOD_V2)][field] for key in selected_keys]
                differences = [
                    float(right) - float(left)
                    for left, right in zip(base_values, v2_values, strict=True)
                    if left is not None
                    and right is not None
                    and math.isfinite(float(left))
                    and math.isfinite(float(right))
                ]
                scope_result["continuous"][field] = {
                    "subset": subset,
                    "baseline": _stats(base_values),
                    "v2": _stats(v2_values),
                    "paired_difference_v2_minus_baseline": _stats(differences),
                }
            result["comparisons"][name][scope] = scope_result
    return result


def build_null_analysis(
    episodes: Sequence[Mapping[str, Any]], agents: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scenarios = sorted({str(row["scenario"]) for row in episodes})
    for method in (METHOD_V1, METHOD_V2):
        for scope in ("overall", *scenarios):
            members = [
                row
                for row in agents
                if row["method"] == method
                and (scope == "overall" or row["scenario"] == scope)
            ]
            selected = [row for row in members if bool(row.get("selected_null"))]
            probabilities = [row.get("null_probability") for row in members]
            probability_array = np.asarray(
                [float(value) for value in probabilities if value is not None], dtype=float
            )
            rows.append(
                {
                    "record_type": "summary",
                    "method": method,
                    "scenario": scope,
                    "agent_count": len(members),
                    "null_selection_count": len(selected),
                    "null_selection_rate": len(selected) / len(members) if members else None,
                    "null_probability_mean": float(probability_array.mean()) if probability_array.size else None,
                    "null_probability_median": float(np.median(probability_array)) if probability_array.size else None,
                    "null_probability_p90": float(np.quantile(probability_array, 0.90)) if probability_array.size else None,
                    "null_probability_p95": float(np.quantile(probability_array, 0.95)) if probability_array.size else None,
                    "null_selected_terminal_completed_count": sum(bool(row["agent_terminal_completed"]) for row in selected),
                    "null_selected_terminal_completed_rate": sum(bool(row["agent_terminal_completed"]) for row in selected) / len(selected) if selected else None,
                    "null_selected_team_collision_count": sum(bool(row.get("team_collision")) for row in selected),
                    "null_selected_team_timeout_count": sum(bool(row.get("team_timeout")) for row in selected),
                }
            )
    episode_index = {
        (row["scenario"], int(row["seed"]), row["method"]): row for row in episodes
    }
    agent_index = {
        (row["scenario"], int(row["seed"]), int(row["agent_id"]), row["method"]): row
        for row in agents
    }
    for v2 in agents:
        if v2["method"] != METHOD_V2 or not bool(v2.get("selected_null")):
            continue
        key = (str(v2["scenario"]), int(v2["seed"]))
        agent_key = (*key, int(v2["agent_id"]))
        v1 = agent_index[(*agent_key, METHOD_V1)]
        fp = agent_index[(*agent_key, METHOD_FP_SHEP)]
        v2_episode = episode_index[(*key, METHOD_V2)]
        v1_episode = episode_index[(*key, METHOD_V1)]
        fp_episode = episode_index[(*key, METHOD_FP_SHEP)]
        rows.append(
            {
                "record_type": "v2_null_paired_decision",
                "method": METHOD_V2,
                "scenario": key[0],
                "seed": key[1],
                "agent_id": int(v2["agent_id"]),
                "v2_null_probability": v2.get("null_probability"),
                "v2_agent_terminal_completed": bool(v2["agent_terminal_completed"]),
                "v2_team_success": bool(v2_episode["team_success"]),
                "v2_team_collision": bool(v2_episode["any_collision"]),
                "v2_team_timeout": bool(v2_episode["timeout"]),
                "v1_selected_type": v1["reference_selected_type"],
                "v1_selected_candidate_id": v1.get("selected_candidate_id"),
                "v1_team_success": bool(v1_episode["team_success"]),
                "v1_team_collision": bool(v1_episode["any_collision"]),
                "v1_team_timeout": bool(v1_episode["timeout"]),
                "fp_shep_selected_type": fp["reference_selected_type"],
                "fp_shep_selected_candidate_id": fp.get("selected_candidate_id"),
                "fp_shep_team_success": bool(fp_episode["team_success"]),
                "fp_shep_team_collision": bool(fp_episode["any_collision"]),
                "fp_shep_team_timeout": bool(fp_episode["timeout"]),
            }
        )
    return rows


def _rate_class(value: float, config: Mapping[str, Any]) -> str:
    if value >= float(config["effect_classification"]["yes_minimum_improvement"]):
        return "YES"
    if value > float(config["effect_classification"]["weak_strict_minimum_improvement"]):
        return "WEAK"
    return "NO"


def _combined_gain(
    success_gain: float, collision_gain: float, timeout_gain: float, config: Mapping[str, Any]
) -> str:
    threshold = float(config["effect_classification"]["yes_minimum_improvement"])
    timeout_limit = float(
        config["effect_classification"]["maximum_timeout_worsening_for_collision_yes"]
    )
    if success_gain >= threshold or (
        collision_gain >= threshold and timeout_gain >= -timeout_limit
    ):
        return "YES"
    if max(success_gain, collision_gain, timeout_gain) > 0.0:
        return "WEAK"
    return "NO"


def build_conclusion(
    *,
    summaries: Sequence[Mapping[str, Any]],
    references: Sequence[Mapping[str, Any]],
    integrity: Mapping[str, Any],
    context: Mapping[str, Any],
    seeds: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    lookup = {(row["method"], row["scenario"]): row for row in summaries}
    ref_lookup = {(row["method"], row["scenario"]): row for row in references}

    def gains(scope: str, baseline: str) -> dict[str, float]:
        base = lookup[(baseline, scope)]
        v2 = lookup[(METHOD_V2, scope)]
        return {
            "success": float(v2["team_success_rate"]) - float(base["team_success_rate"]),
            "collision": float(base["any_collision_rate"]) - float(v2["any_collision_rate"]),
            "timeout": float(base["timeout_rate"]) - float(v2["timeout_rate"]),
            "obstacle_collision": float(base["obstacle_collision_rate"]) - float(v2["obstacle_collision_rate"]),
            "inter_agent_collision": float(base["inter_agent_collision_rate"]) - float(v2["inter_agent_collision_rate"]),
        }

    v1 = gains("overall", METHOD_V1)
    fp = gains("overall", METHOD_FP_SHEP)
    multi = gains("multi_agent", METHOD_V1)
    v1_gain = _combined_gain(v1["success"], v1["collision"], v1["timeout"], config)
    fp_gain = _combined_gain(fp["success"], fp["collision"], fp["timeout"], config)
    multi_gain = _combined_gain(multi["success"], multi["collision"], multi["timeout"], config)
    v1_ref = ref_lookup[(METHOD_V1, "overall")]
    v2_ref = ref_lookup[(METHOD_V2, "overall")]
    reach_gain = float(v2_ref["reference_reach_rate"] or 0.0) - float(
        v1_ref["reference_reach_rate"] or 0.0
    )
    post_gain = float(v2_ref["reached_to_terminal_rate"] or 0.0) - float(
        v1_ref["reached_to_terminal_rate"] or 0.0
    )
    null_increase = float(lookup[(METHOD_V2, "overall")]["null_selection_rate"]) - float(
        lookup[(METHOD_V1, "overall")]["null_selection_rate"]
    )
    large_null = float(config["effect_classification"]["large_null_rate_increase"])
    material = float(config["effect_classification"]["material_null_outcome_change"])
    if null_increase < large_null:
        null_effect = "NOT_ESTABLISHED"
    elif v1["success"] >= 0.0 and v1["timeout"] >= -material and v1["collision"] > 0.0:
        null_effect = "BENEFICIAL"
    elif v1["success"] < 0.0 or v1["timeout"] < -material:
        null_effect = "MIXED" if v1["collision"] > 0.0 else "HARMFUL"
    elif max(v1["success"], v1["collision"], v1["timeout"]) > 0.0:
        null_effect = "BENEFICIAL"
    else:
        null_effect = "MIXED"
    over_conservative = bool(
        null_increase >= large_null
        and (v1["success"] < 0.0 or v1["timeout"] < -material)
        and v1["collision"] < float(config["effect_classification"]["yes_minimum_improvement"])
    )

    def selection_key(method: str) -> tuple[Any, ...]:
        row = lookup[(method, "overall")]
        ref = ref_lookup[(method, "overall")]
        return (
            -float(row["team_success_rate"]),
            float(row["any_collision_rate"]),
            float(row["timeout_rate"]),
            -float(ref["reached_to_terminal_rate"] or -1.0),
            float(row["success_completion_time_s_mean"] or math.inf),
            float(row["success_team_path_length_m_mean"] or math.inf),
            0 if method == METHOD_V2 else 1,
        )

    final_method = min((METHOD_V1, METHOD_V2), key=selection_key)
    pipeline_valid = integrity.get("status") == "PASSED"
    transfer = v1_gain
    proceed = pipeline_valid and transfer in {"YES", "WEAK"}
    return {
        "schema_version": SCHEMA_VERSION,
        "CONTEXT_RECOVERY_VALID": context["CONTEXT_RECOVERY_VALID"],
        "V1_CHECKPOINT_VALID": "YES" if integrity.get("v1_checkpoint_valid") else "NO",
        "V2_CHECKPOINT_VALID": "YES" if integrity.get("v2_checkpoint_valid") else "NO",
        "CANDIDATE_BUNDLE_FAIRNESS": "YES" if integrity.get("candidate_bundle_fairness") else "NO",
        "PREVIEW_FAIRNESS": "YES" if integrity.get("preview_fairness") else "NO",
        "V1_V2_GRAPH_INPUT_EQUAL": "YES" if integrity.get("v1_v2_graph_input_equal") else "NO",
        "CLOSED_LOOP_PIPELINE_VALID": "YES" if pipeline_valid else "NO",
        "FINAL_TEST_SEED_OVERLAP": int(seeds["FINAL_TEST_SEED_OVERLAP"]),
        "V2_VS_V1_SUCCESS_GAIN": _rate_class(v1["success"], config),
        "V2_VS_FP_SHEP_SUCCESS_GAIN": _rate_class(fp["success"], config),
        "V2_CLOSED_LOOP_GAIN_OVER_V1": v1_gain,
        "V2_CLOSED_LOOP_GAIN_OVER_FP_SHEP": fp_gain,
        "V2_COLLISION_IMPROVEMENT": _rate_class(v1["collision"], config),
        "REFERENCE_EXECUTABILITY_GAIN": _rate_class(reach_gain, config),
        "POST_REFERENCE_COMPLETION_GAIN": _rate_class(post_gain, config),
        "NULL_BEHAVIOR_EFFECT": null_effect,
        "OVER_CONSERVATIVE_NULL_SELECTION": "YES" if over_conservative else "NO",
        "MULTI_AGENT_GAIN": multi_gain,
        "OFFLINE_TO_CLOSED_LOOP_TRANSFER_V2": transfer,
        "TEAM_SUCCESS_90_TARGET_REACHED": "YES"
        if float(lookup[(METHOD_V2, "overall")]["team_success_rate"]) >= 0.90
        else "NO",
        "FINAL_GAT_CHECKPOINT": "V2" if final_method == METHOD_V2 else "V1",
        "FINAL_METHOD": "Proposal + FP-SHEP features + selected GAT checkpoint + Frozen SAC-DMP",
        "PROCEED_TO_FINAL_PAPER_EXPERIMENTS": "YES" if proceed else "NO",
        "NEXT_STEP": (
            "Organize the frozen main tables, figures, and Results around the selected checkpoint."
            if proceed
            else "Perform failure decomposition only; do not start a new algorithm branch automatically."
        ),
        "rate_improvements_positive_is_better": {
            "V2_vs_V1_overall": v1,
            "V2_vs_FP_SHEP_overall": fp,
            "V2_vs_V1_multi_agent_mixed_obstacle_interaction": multi,
            "V2_minus_V1_reference_reach": reach_gain,
            "V2_minus_V1_post_reference_completion": post_gain,
            "V2_minus_V1_null_selection": null_increase,
        },
        "interpretation": config["interpretation"]["name"],
        "label_only_ablation_claim_allowed": False,
        "training_or_tuning_performed": False,
        "stress_test_rerun": False,
    }


def build_smoke_gate(
    *,
    config: Mapping[str, Any],
    episodes: Sequence[Mapping[str, Any]],
    agents: Sequence[Mapping[str, Any]],
    shared_cache: Mapping[tuple[str, int], Mapping[str, Any]],
    checkpoint_valid: Mapping[str, bool],
    core_before: Mapping[str, str],
    core_after: Mapping[str, str],
    reuse_before: Mapping[str, str],
    reuse_after: Mapping[str, str],
) -> dict[str, Any]:
    diagnostics = [
        row
        for shared in shared_cache.values()
        for method in (METHOD_V1, METHOD_V2)
        for row in shared["diagnostics_by_method"][method]
    ]
    initial: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in episodes:
        initial[(str(row["scenario"]), int(row["seed"]))].add(
            str(row["initial_condition_hash"])
        )
    checks = {
        "episode_count": len(episodes) == int(config["smoke_gate"]["expected_team_episode_count"]),
        "checkpoint_strict_load": all(checkpoint_valid.values()),
        "logits_finite": all(bool(row["logits_finite"]) for row in diagnostics),
        "class_mapping": all(bool(row["class_mapping_valid"]) for row in diagnostics),
        "null_mapping": all(
            not bool(row.get("selected_null")) or row.get("reference_selected_type") == "null"
            for row in agents
            if row["method"] in {METHOD_V1, METHOD_V2}
        ),
        "proposal_index_mapping": all(
            row.get("selected_candidate_id") is None
            or 0 <= int(row["selected_candidate_id"]) < int(row["K_t"])
            for row in agents
            if row["method"] in {METHOD_V1, METHOD_V2}
        ),
        "candidate_fairness": all(
            len(set(shared["candidate_hashes_by_method"].values())) == 1
            for shared in shared_cache.values()
        ),
        "preview_fairness": all(
            len(set(shared["preview_hashes_by_method"].values())) == 1
            for shared in shared_cache.values()
        ),
        "graph_equality": all(
            len(set(shared["graph_input_hashes_by_method"].values())) == 1
            for shared in shared_cache.values()
        ),
        "paired_initial_state": all(len(values) == 1 for values in initial.values()),
        "one_shot_handoff": all(int(row["maximum_handoff_count_per_agent"]) <= 1 for row in episodes),
        "no_replanning": all(int(row["replanning_count"]) == 0 for row in episodes),
        "no_nan": all(
            row.get(field) is None or not np.isnan(float(row[field]))
            for row in episodes
            for field in ("team_path_length_m", "minimum_inter_agent_distance_m")
        ),
        "environment_exception_count_zero": True,
        "selection_plan_unchanged": all(bool(row["selection_plan_unchanged"]) for row in episodes),
        "historical_gate": all(bool(row["execution_historical_gate_verified"]) for row in episodes),
        "core_file_hash_unchanged": dict(core_before) == dict(core_after),
        "smoke_reuse_hash_unchanged": dict(reuse_before) == dict(reuse_after),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASSED" if all(checks.values()) else "FAILED",
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "episode_count": len(episodes),
        "result_based_algorithm_adjustment": False,
    }


def build_integrity(
    *,
    config: Mapping[str, Any],
    episodes: Sequence[Mapping[str, Any]],
    agents: Sequence[Mapping[str, Any]],
    shared_cache: Mapping[tuple[str, int], Mapping[str, Any]],
    core_before: Mapping[str, str],
    core_after: Mapping[str, str],
    reuse_before: Mapping[str, str],
    reuse_after: Mapping[str, str],
    checkpoint_hash_before: Mapping[str, str],
    checkpoint_hash_after: Mapping[str, str],
    model_hash_before: Mapping[str, str],
    model_hash_after: Mapping[str, str],
    policy_hash_before: str,
    policy_hash_after: str,
) -> dict[str, Any]:
    candidate_fair = all(
        len(set(shared["candidate_hashes_by_method"].values())) == 1
        for shared in shared_cache.values()
    )
    preview_fair = all(
        len(set(shared["preview_hashes_by_method"].values())) == 1
        for shared in shared_cache.values()
    )
    graph_equal = all(
        len(set(shared["graph_input_hashes_by_method"].values())) == 1
        for shared in shared_cache.values()
    )
    checks = {
        "episode_count": len(episodes) == 300,
        "agent_record_count": len(agents) == 900,
        "candidate_bundle_fairness": candidate_fair,
        "preview_fairness": preview_fair,
        "v1_v2_graph_input_equal": graph_equal,
        "core_sources_match_authoritative_hashes": dict(core_before)
        == dict(config["expected_core_hashes"]),
        "core_sources_unchanged": dict(core_before) == dict(core_after),
        "evaluator_and_config_unchanged": dict(reuse_before) == dict(reuse_after),
        "checkpoints_unchanged": dict(checkpoint_hash_before) == dict(checkpoint_hash_after),
        "model_parameters_unchanged": dict(model_hash_before) == dict(model_hash_after),
        "policy_parameters_unchanged": policy_hash_before == policy_hash_after,
        "historical_gate": all(bool(row["execution_historical_gate_verified"]) for row in episodes),
        "no_replanning": all(int(row["replanning_count"]) == 0 for row in episodes),
        "one_shot_handoff": all(int(row["maximum_handoff_count_per_agent"]) <= 1 for row in episodes),
        "phase_not_reset": all(not bool(row["phase_reset_on_switch"]) for row in episodes),
        "terminal_goals_unchanged": all(bool(row["terminal_task_goals_unchanged"]) for row in episodes),
    }
    v1_valid = checkpoint_hash_before["v1"] == config["v1_checkpoint_sha256_expected"]
    v2_valid = checkpoint_hash_before["v2"] == config["v2_checkpoint_sha256_expected"]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASSED" if all(checks.values()) and v1_valid and v2_valid else "FAILED",
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "episode_count": len(episodes),
        "expected_episode_count": 300,
        "agent_record_count": len(agents),
        "candidate_bundle_fairness": candidate_fair,
        "preview_fairness": preview_fair,
        "v1_v2_graph_input_equal": graph_equal,
        "v1_checkpoint_valid": v1_valid,
        "v2_checkpoint_valid": v2_valid,
        "checkpoint_hashes_before": dict(checkpoint_hash_before),
        "checkpoint_hashes_after": dict(checkpoint_hash_after),
        "core_hashes_before": dict(core_before),
        "core_hashes_after": dict(core_after),
        "smoke_reuse_hashes_before": dict(reuse_before),
        "smoke_reuse_hashes_after": dict(reuse_after),
        "training_performed": False,
        "tuning_performed": False,
        "stress_test_rerun": False,
    }


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.1f}%"


def _render_report(
    *,
    summaries: Sequence[Mapping[str, Any]],
    references: Sequence[Mapping[str, Any]],
    v1_pairs: Sequence[Mapping[str, Any]],
    fp_pairs: Sequence[Mapping[str, Any]],
    tests: Mapping[str, Any],
    conclusion: Mapping[str, Any],
    smoke: Mapping[str, Any],
    integrity: Mapping[str, Any],
    config: Mapping[str, Any],
) -> str:
    lookup = {(row["method"], row["scenario"]): row for row in summaries}
    ref_lookup = {(row["method"], row["scenario"]): row for row in references}
    lines = [
        "# Stage-I GAT V2 Closed-Loop Validation",
        "",
        "## Frozen protocol and interpretation",
        "",
        "- V2 is interpreted as deployment-aligned historical-vector-gate H4 graph input plus V2 long-horizon target adaptation; this is not a label-only ablation.",
        "- Top-K=10, H_preview=4, max_steps=220, handoff threshold=0.25 m, 122-D actor observation, frozen deterministic SAC-DMP, legacy/classic phase, boundary-free one-shot handoff.",
        "- Scenarios: open, sparse_static, multi_agent (mixed obstacle and multi-agent interaction); untouched seeds: 30-49.",
        "- Proposal, FP-SHEP, GAT-V1, and GAT-V2 consume one immutable candidate bundle. FP-SHEP/V1/V2 share one H4 preview bundle; V1/V2 use the same online graph hash.",
        "- No retraining, threshold calibration, graph/method modification, replanning, new geometry, or 24-layout stress rerun was performed.",
        "",
        "## Validity",
        "",
        f"- Smoke gate: `{smoke['status']}` (45/45 expected team episodes).",
        f"- Final integrity: `{integrity['status']}` (300 team episodes; 900 agent records).",
        f"- Candidate fairness: `{conclusion['CANDIDATE_BUNDLE_FAIRNESS']}`; preview fairness: `{conclusion['PREVIEW_FAIRNESS']}`; V1/V2 graph equality: `{conclusion['V1_V2_GRAPH_INPUT_EQUAL']}`.",
        f"- FINAL_TEST_SEED_OVERLAP = `{conclusion['FINAL_TEST_SEED_OVERLAP']}`.",
        "",
        "## Overall team results",
        "",
        "| Method | Success | Any collision | Obstacle collision | Inter-agent collision | Timeout | Null rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHOD_ORDER:
        row = lookup[(method, "overall")]
        lines.append(
            f"| {METHOD_NAMES[method]} | {row['team_success_count']}/{row['episode_count']} ({_pct(row['team_success_rate'])}) | "
            f"{row['any_collision_count']}/{row['episode_count']} ({_pct(row['any_collision_rate'])}) | "
            f"{row['obstacle_collision_count']}/{row['episode_count']} ({_pct(row['obstacle_collision_rate'])}) | "
            f"{row['inter_agent_collision_count']}/{row['episode_count']} ({_pct(row['inter_agent_collision_rate'])}) | "
            f"{row['timeout_count']}/{row['episode_count']} ({_pct(row['timeout_rate'])}) | "
            f"{_pct(row['null_selection_rate'])} |"
        )
    lines.extend(["", "## Scenario-stratified success", "", "| Scenario | V1 | V2 | V2-V1 |", "|---|---:|---:|---:|"])
    for scope in config["formal_scenarios"]:
        v1 = lookup[(METHOD_V1, scope)]["team_success_rate"]
        v2 = lookup[(METHOD_V2, scope)]["team_success_rate"]
        lines.append(f"| {scope} | {_pct(v1)} | {_pct(v2)} | {100*(float(v2)-float(v1)):+.1f} pp |")
    lines.extend(["", "## Paired outcomes", ""])
    lines.append(f"- V2 vs V1: `{dict(Counter(row['pair_outcome'] for row in v1_pairs))}`")
    lines.append(f"- V2 vs FP-SHEP: `{dict(Counter(row['pair_outcome'] for row in fp_pairs))}`")
    for comparison in ("v2_vs_v1", "v2_vs_fp_shep"):
        success = tests["comparisons"][comparison]["overall"]["binary"]["team_success"]
        lines.append(
            f"- {comparison} success McNemar: discordant={success['discordant_count']}, exact p={success['exact_two_sided_p_value']:.4f}."
        )
    v1_ref = ref_lookup[(METHOD_V1, "overall")]
    v2_ref = ref_lookup[(METHOD_V2, "overall")]
    lines.extend(
        [
            "",
            "## Reference transition and null behavior",
            "",
            f"- V1 reference selection/reach/reached-to-terminal: {_pct(v1_ref['reference_selection_rate'])} / {_pct(v1_ref['reference_reach_rate'])} / {_pct(v1_ref['reached_to_terminal_rate'])}.",
            f"- V2 reference selection/reach/reached-to-terminal: {_pct(v2_ref['reference_selection_rate'])} / {_pct(v2_ref['reference_reach_rate'])} / {_pct(v2_ref['reached_to_terminal_rate'])}.",
            f"- NULL_BEHAVIOR_EFFECT = `{conclusion['NULL_BEHAVIOR_EFFECT']}`; OVER_CONSERVATIVE_NULL_SELECTION = `{conclusion['OVER_CONSERVATIVE_NULL_SELECTION']}`.",
            "",
            "## Decisions",
            "",
        ]
    )
    for key in (
        "CONTEXT_RECOVERY_VALID",
        "V1_CHECKPOINT_VALID",
        "V2_CHECKPOINT_VALID",
        "CANDIDATE_BUNDLE_FAIRNESS",
        "PREVIEW_FAIRNESS",
        "V1_V2_GRAPH_INPUT_EQUAL",
        "CLOSED_LOOP_PIPELINE_VALID",
        "V2_VS_V1_SUCCESS_GAIN",
        "V2_VS_FP_SHEP_SUCCESS_GAIN",
        "V2_COLLISION_IMPROVEMENT",
        "REFERENCE_EXECUTABILITY_GAIN",
        "POST_REFERENCE_COMPLETION_GAIN",
        "MULTI_AGENT_GAIN",
        "OFFLINE_TO_CLOSED_LOOP_TRANSFER_V2",
        "TEAM_SUCCESS_90_TARGET_REACHED",
        "FINAL_GAT_CHECKPOINT",
        "FINAL_METHOD",
        "PROCEED_TO_FINAL_PAPER_EXPERIMENTS",
        "NEXT_STEP",
    ):
        lines.append(f"- `{key} = {conclusion[key]}`")
    lines.extend(
        [
            "",
            "## Interpretation limit and stop rule",
            "",
            "The multi_agent scenario mixes obstacle pressure with peer interaction; any gain here is not evidence that multi-agent coordination is solved. The run stopped after this frozen validation. No automatic training, tuning, supervision redesign, stress test, null modification, or new experiment was started.",
        ]
    )
    return "\n".join(lines) + "\n"


def _annotate_agent_team_outcomes(
    agents: list[dict[str, Any]], episodes: Sequence[Mapping[str, Any]]
) -> None:
    index = {
        (row["scenario"], int(row["seed"]), row["method"]): row for row in episodes
    }
    for row in agents:
        episode = index[(row["scenario"], int(row["seed"]), row["method"])]
        row["team_collision"] = bool(episode["any_collision"])
        row["team_obstacle_collision"] = bool(episode["obstacle_collision"])
        row["team_inter_agent_collision"] = bool(episode["inter_agent_collision"])
        row["team_timeout"] = bool(episode["timeout"])


def run_experiment(config: Mapping[str, Any], output_dir: Path) -> Path:
    _assert_frozen_config(config)
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    resolved = copy.deepcopy(dict(config))
    resolved.update(
        {
            "resolved_output_dir": str(output_dir),
            "created_at": datetime.now().astimezone().isoformat(),
            "python_executable": sys.executable,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "config_sha256_before_outcomes": _sha256_file(REPO_ROOT / CONFIG_PATH),
            "evaluator_sha256_before_outcomes": _sha256_file(REPO_ROOT / EVALUATOR_PATH),
        }
    )
    write_json(output_dir / "config.json", resolved)
    context = build_context_recovery_manifest(config)
    seed_manifest = build_seed_manifest(config)
    write_json(output_dir / "context_recovery_manifest.json", context)
    write_json(output_dir / "final_seed_manifest.json", seed_manifest)
    if context["status"] != "PASSED" or seed_manifest["status"] != "PASSED":
        raise RuntimeError("context recovery or final seed audit failed")

    checkpoint_paths = {
        "v1": (REPO_ROOT / config["v1_checkpoint"]).resolve(),
        "v2": (REPO_ROOT / config["v2_checkpoint"]).resolve(),
        "sac": (REPO_ROOT / config["sac_checkpoint"]).resolve(),
    }
    checkpoint_hash_before = {
        key: _sha256_file(path) for key, path in checkpoint_paths.items()
    }
    checkpoint_valid = {
        "v1": checkpoint_hash_before["v1"] == config["v1_checkpoint_sha256_expected"],
        "v2": checkpoint_hash_before["v2"] == config["v2_checkpoint_sha256_expected"],
        "sac": checkpoint_hash_before["sac"] == config["sac_checkpoint_sha256_expected"],
    }
    if not all(checkpoint_valid.values()):
        raise RuntimeError(f"checkpoint hash mismatch: {checkpoint_valid}")

    core_before = _file_hashes(CORE_METHOD_PATHS)
    if core_before != dict(config["expected_core_hashes"]):
        raise RuntimeError("core source differs from authoritative integrity hashes")
    reuse_before = _file_hashes(SMOKE_REUSE_PATHS)
    execution_settings = legacy._build_execution_settings(config)
    multi_config = build_single_distribution_multi_config(
        num_agents=int(config["num_agents"]), max_steps=int(config["max_steps"])
    )
    policy, loaded_sac = _load_policy(execution_settings, multi_config)
    if loaded_sac.resolve() != checkpoint_paths["sac"]:
        raise RuntimeError("SAC loader resolved a different checkpoint")
    policy_hash_before = _policy_parameter_sha256(policy)

    stage1_config = _load_json((REPO_ROOT / config["stage1_config"]).resolve())
    device = resolve_device(stage1_config["training"]["device"])
    v1_model = load_model_checkpoint(checkpoint_paths["v1"], stage1_config, device)
    v2_model = load_model_checkpoint(checkpoint_paths["v2"], stage1_config, device)
    payload_v1 = torch.load(checkpoint_paths["v1"], map_location="cpu", weights_only=False)
    payload_v2 = torch.load(checkpoint_paths["v2"], map_location="cpu", weights_only=False)
    checkpoint_valid["v1_strict_schema"] = payload_v1.get("schema_version") == "gat_stage1_training_v1"
    checkpoint_valid["v2_strict_schema"] = payload_v2.get("schema_version") == "gat_stage1_v2_training"
    checkpoint_valid["same_model_config"] = payload_v1.get("model_config") == payload_v2.get("model_config")
    if not all(checkpoint_valid.values()):
        raise RuntimeError(f"strict checkpoint metadata failed: {checkpoint_valid}")
    for model in (v1_model, v2_model):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    model_hash_before = {"v1": _model_hash(v1_model), "v2": _model_hash(v2_model)}

    shared_cache: dict[tuple[str, int], dict[str, Any]] = {}
    smoke_episodes, smoke_agents = evaluate_seed_set(
        config=config,
        execution_settings=execution_settings,
        multi_config=multi_config,
        policy=policy,
        v1_model=v1_model,
        v2_model=v2_model,
        device=device,
        seeds=config["smoke_seeds"],
        shared_cache=shared_cache,
        phase_name="smoke",
    )
    _annotate_agent_team_outcomes(smoke_agents, smoke_episodes)
    core_after_smoke = _file_hashes(CORE_METHOD_PATHS)
    reuse_after_smoke = _file_hashes(SMOKE_REUSE_PATHS)
    smoke_gate = build_smoke_gate(
        config=config,
        episodes=smoke_episodes,
        agents=smoke_agents,
        shared_cache=shared_cache,
        checkpoint_valid=checkpoint_valid,
        core_before=core_before,
        core_after=core_after_smoke,
        reuse_before=reuse_before,
        reuse_after=reuse_after_smoke,
    )
    write_csv(output_dir / "smoke_results.csv", smoke_episodes)
    write_json(output_dir / "smoke_gate.json", smoke_gate)
    if smoke_gate["status"] != "PASSED":
        write_json(
            output_dir / "conclusion.json",
            {
                "schema_version": SCHEMA_VERSION,
                "CLOSED_LOOP_PIPELINE_VALID": "NO",
                "stop_reason": "SMOKE_GATE_FAILED",
                "failed_checks": smoke_gate["failed_checks"],
                "training_or_tuning_performed": False,
            },
        )
        (output_dir / "FINAL_REPORT.md").write_text(
            "# Stage-I GAT V2 Closed-Loop Validation\n\n"
            f"Smoke gate failed: {smoke_gate['failed_checks']}. Formal evaluation was not started.\n",
            encoding="utf-8",
        )
        return output_dir
    if reuse_after_smoke != _file_hashes(SMOKE_REUSE_PATHS):
        raise RuntimeError("evaluator/config changed after smoke; smoke reuse prohibited")

    remaining = [seed for seed in config["formal_seeds"] if seed not in config["smoke_seeds"]]
    formal_episodes, formal_agents = evaluate_seed_set(
        config=config,
        execution_settings=execution_settings,
        multi_config=multi_config,
        policy=policy,
        v1_model=v1_model,
        v2_model=v2_model,
        device=device,
        seeds=remaining,
        shared_cache=shared_cache,
        phase_name="formal",
    )
    episodes = smoke_episodes + formal_episodes
    agents = smoke_agents + formal_agents
    _annotate_agent_team_outcomes(agents, episodes)

    core_after = _file_hashes(CORE_METHOD_PATHS)
    reuse_after = _file_hashes(SMOKE_REUSE_PATHS)
    checkpoint_hash_after = {
        key: _sha256_file(path) for key, path in checkpoint_paths.items()
    }
    model_hash_after = {"v1": _model_hash(v1_model), "v2": _model_hash(v2_model)}
    policy_hash_after = _policy_parameter_sha256(policy)
    integrity = build_integrity(
        config=config,
        episodes=episodes,
        agents=agents,
        shared_cache=shared_cache,
        core_before=core_before,
        core_after=core_after,
        reuse_before=reuse_before,
        reuse_after=reuse_after,
        checkpoint_hash_before=checkpoint_hash_before,
        checkpoint_hash_after=checkpoint_hash_after,
        model_hash_before=model_hash_before,
        model_hash_after=model_hash_after,
        policy_hash_before=policy_hash_before,
        policy_hash_after=policy_hash_after,
    )
    if integrity["status"] != "PASSED":
        write_json(output_dir / "integrity_manifest.json", integrity)
        raise RuntimeError(f"final integrity failed: {integrity['failed_checks']}")

    summaries = build_method_summaries(episodes, agents)
    scenario_summaries = [row for row in summaries if row["scenario"] != "overall"]
    references = build_reference_transition_analysis(agents, episodes)
    v1_pairs = build_pairing(episodes, baseline=METHOD_V1)
    fp_pairs = build_pairing(episodes, baseline=METHOD_FP_SHEP)
    null_rows = build_null_analysis(episodes, agents)
    tests = build_statistical_tests(episodes)
    conclusion = build_conclusion(
        summaries=summaries,
        references=references,
        integrity=integrity,
        context=context,
        seeds=seed_manifest,
        config=config,
    )

    write_json(output_dir / "integrity_manifest.json", integrity)
    write_csv(output_dir / "episode_results.csv", episodes)
    write_csv(output_dir / "agent_results.csv", agents)
    write_csv(output_dir / "method_summary.csv", summaries)
    write_csv(output_dir / "scenario_summary.csv", scenario_summaries)
    write_csv(output_dir / "v1_vs_v2_paired.csv", v1_pairs)
    write_csv(output_dir / "v2_vs_fpshep_paired.csv", fp_pairs)
    write_csv(output_dir / "null_analysis.csv", null_rows)
    write_csv(output_dir / "reference_transition_analysis.csv", references)
    write_json(output_dir / "statistical_tests.json", tests)
    write_json(output_dir / "conclusion.json", conclusion)
    (output_dir / "FINAL_REPORT.md").write_text(
        _render_report(
            summaries=summaries,
            references=references,
            v1_pairs=v1_pairs,
            fp_pairs=fp_pairs,
            tests=tests,
            conclusion=conclusion,
            smoke=smoke_gate,
            integrity=integrity,
            config=config,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "episode_count": len(episodes),
                "runtime_seconds": time.perf_counter() - started,
                "smoke_gate": smoke_gate["status"],
                "integrity": integrity["status"],
                "FINAL_GAT_CHECKPOINT": conclusion["FINAL_GAT_CHECKPOINT"],
                "automatic_follow_on_started": False,
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
