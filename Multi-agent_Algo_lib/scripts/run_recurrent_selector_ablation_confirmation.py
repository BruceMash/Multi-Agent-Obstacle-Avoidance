#!/usr/bin/env python3
"""Run the frozen independent recurrent-selector confirmation block.

The three arms share the same candidate generator, R-ERR supervisor, frozen
SAC-DMP executor, scene manifest, and outcome contract.  Only the final
selector differs: Proposal coarse Top-1, FP-SHEP Top-1, or frozen GAT-R.

The run phase deliberately exposes only record counts and software health.
Performance aggregation is deferred to a separate post-run analysis script.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
for search_path in (REPO_ROOT, ALGO_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

# Import the formal engine first.  On Windows it deliberately initializes
# pandas/pyarrow before torch-backed planning modules.
from scripts import run_long_range_formal_benchmark as base  # noqa: E402
from Guidance.reference_point_proposal_demo import (  # noqa: E402
    ProposalConfig,
    propose_reference_points,
)
from planning.online_runtime_instrumentation import (  # noqa: E402
    OnlineRuntimeRecorder,
    TimedPolicyProxy,
)
from planning.policy_preview import adapt_candidate_proposals  # noqa: E402
from planning.pre_gat_220step_revalidation import (  # noqa: E402
    ImmutableCandidate,
    ImmutableCandidateBundle,
    stable_hash,
)
from scripts.evaluate_gat_closed_loop import reconstruct_proposals  # noqa: E402


ARTIFACT_ROOT = (
    REPO_ROOT
    / "artifacts/recurrent_selector_ablation_confirmation/20260822_143118"
)
SOURCE_STUDY = (
    REPO_ROOT / "artifacts/gat_recurrent_smoothness_rescue/20260821_180244"
)
FREEZE_DIR = ARTIFACT_ROOT / "01_method_freeze"
MANIFEST_DIR = ARTIFACT_ROOT / "02_manifest"
AUDIT_DIR = ARTIFACT_ROOT / "03_proposal_selector_audit"
RECORD_DIR = ARTIFACT_ROOT / "04_runs"
FORMAL_V2_FREEZE = SOURCE_STUDY / "09_final_freeze/FORMAL_V2_RUN_FREEZE.json"
FORMAL_V2_METHODS = SOURCE_STUDY / "09_final_freeze/method_configs"
FORMAL_V2_REGISTRY = SOURCE_STUDY / "10_formal_v2/ALL_USED_SCENE_REGISTRY.csv"
FORMAL_V2_MANIFEST = SOURCE_STUDY / "10_formal_v2/FORMAL_V2_MANIFEST.json"
FINAL_GAT_FREEZE = SOURCE_STUDY / "09_final_freeze/FINAL_GAT_RS_FREEZE.json"
SOURCE_FP_CONFIG = FORMAL_V2_METHODS / "M8_RERR_FP_SHEP_SAC_DMP.json"
SOURCE_GAT_CONFIG = FORMAL_V2_METHODS / "M9_Proposed_RERR_GAT_SAC_DMP.json"
METHOD_FREEZE_AUDIT = FREEZE_DIR / "METHOD_FREEZE_AUDIT.json"
PROPOSAL_AUDIT = AUDIT_DIR / "PROPOSAL_ONLY_SELECTOR_INDEPENDENCE_AUDIT.json"
SCENE_ISOLATION = MANIFEST_DIR / "SELECTOR_ABLATION_SCENE_ISOLATION.json"


METHODS = (
    {
        "method_id": "M1_RERR_Proposal_SAC_DMP",
        "display_name": "R-ERR + Proposal + SAC-DMP",
        "engine": "rerr_proposal",
        "role": "RECURRENT_COARSE_SELECTOR_ABLATION",
        "config": str(SOURCE_FP_CONFIG.relative_to(REPO_ROOT).as_posix()),
    },
    {
        "method_id": "M8_RERR_FP_SHEP_SAC_DMP",
        "display_name": "R-ERR + FP-SHEP + SAC-DMP",
        "engine": "rerr_fp_shep",
        "role": "RECURRENT_EXECUTION_AWARE_SELECTOR",
        "config": str(SOURCE_FP_CONFIG.relative_to(REPO_ROOT).as_posix()),
    },
    {
        "method_id": "M9_Proposed_RERR_GAT_SAC_DMP",
        "display_name": "R-ERR + GAT-R + SAC-DMP",
        "engine": "rerr_gat",
        "role": "RECURRENT_CONTEXTUAL_SELECTOR",
        "config": str(SOURCE_GAT_CONFIG.relative_to(REPO_ROOT).as_posix()),
    },
)
METHOD_BY_ID = {row["method_id"]: row for row in METHODS}
METHOD_ORDER = tuple(METHOD_BY_ID)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def jsonable(value: Any) -> Any:
    return base.json_ready(value)


def proposal_method_config() -> dict[str, Any]:
    config = copy.deepcopy(load_json(SOURCE_FP_CONFIG))
    config.update(
        {
            "method": "proposal_rerr",
            "configuration_id": "SELECTOR_CONFIRMATION_RERR_PROPOSAL_TOP1",
            "final_method_description": (
                "Proposal coarse Top-1 + frozen R-ERR + frozen adapted SAC-DMP"
            ),
        }
    )
    return config


def derived_method_config(method: Mapping[str, str]) -> dict[str, Any]:
    if method["engine"] == "rerr_proposal":
        return proposal_method_config()
    if method["engine"] == "rerr_fp_shep":
        return load_json(SOURCE_FP_CONFIG)
    if method["engine"] == "rerr_gat":
        return load_json(SOURCE_GAT_CONFIG)
    raise ValueError(f"unsupported selector engine: {method['engine']}")


def frozen_method_config(method: Mapping[str, str]) -> dict[str, Any]:
    return load_json(FREEZE_DIR / "method_configs" / f"{method['method_id']}.json")


def historical_scene_registry() -> tuple[list[dict[str, Any]], dict[str, set[Any]]]:
    rows: list[dict[str, Any]] = []
    sets: dict[str, set[Any]] = {
        "seed": set(),
        "geometry_fingerprint": set(),
        "dynamic_track_fingerprint": set(),
        "translation_invariant_fingerprint": set(),
        "start_goal_fingerprint": set(),
    }
    with FORMAL_V2_REGISTRY.open("r", newline="", encoding="utf-8-sig") as handle:
        for source in csv.DictReader(handle):
            row = dict(source)
            if row.get("seed") not in (None, ""):
                row["seed"] = int(row["seed"])
            rows.append(row)
            for key in sets:
                value = row.get(key)
                if value not in (None, ""):
                    sets[key].add(int(value) if key == "seed" else value)
    return rows, sets


def method_gate() -> dict[str, Any]:
    if not METHOD_FREEZE_AUDIT.is_file() or not PROPOSAL_AUDIT.is_file():
        raise FileNotFoundError("method and Proposal-only audits must precede manifest freeze")
    method_audit = load_json(METHOD_FREEZE_AUDIT)
    proposal_audit = load_json(PROPOSAL_AUDIT)
    if method_audit.get("status") != "PASS":
        raise RuntimeError("method freeze audit failed")
    if proposal_audit.get("status") != "PASS":
        raise RuntimeError("Proposal-only independence audit failed")
    if proposal_audit.get("FP_SHEP_AFFECTS_FINAL_SELECTION") != "NO":
        raise RuntimeError("Proposal-only path depends on FP-SHEP")
    if proposal_audit.get("GAT_AFFECTS_FINAL_SELECTION") != "NO":
        raise RuntimeError("Proposal-only path depends on GAT")
    return {
        "confirmatory_design": True,
        "FORMAL_V2_ALREADY_OBSERVED": True,
        "FORMAL_V2_USED_FOR_NEW_ABLATION_TUNING": False,
        "NEW_PARAMETER_TUNING_ALLOWED": False,
        "NEW_NETWORK_TRAINING_ALLOWED": False,
        "METHOD_FREEZE_AUDIT": {
            "path": str(METHOD_FREEZE_AUDIT.relative_to(ARTIFACT_ROOT).as_posix()),
            "sha256": base.sha256_file(METHOD_FREEZE_AUDIT),
        },
        "PROPOSAL_ONLY_SELECTOR_INDEPENDENCE_AUDIT": {
            "path": str(PROPOSAL_AUDIT.relative_to(ARTIFACT_ROOT).as_posix()),
            "sha256": base.sha256_file(PROPOSAL_AUDIT),
        },
        "FORMAL_V2_FREEZE": {
            "path": str(FORMAL_V2_FREEZE.relative_to(REPO_ROOT).as_posix()),
            "sha256": base.sha256_file(FORMAL_V2_FREEZE),
        },
        "FINAL_GAT_FREEZE": {
            "path": str(FINAL_GAT_FREEZE.relative_to(REPO_ROOT).as_posix()),
            "sha256": base.sha256_file(FINAL_GAT_FREEZE),
            "checkpoint_sha256": load_json(FINAL_GAT_FREEZE)[
                "selected_checkpoint_sha256"
            ],
        },
    }


def _proposal_candidate_record(
    *,
    agent_id: int,
    proposals: tuple[Any, ...],
    selected_id: int | None,
    terminal_goal: np.ndarray,
    count_before_consumer: int,
) -> dict[str, Any]:
    selected = proposals[selected_id] if selected_id is not None else None
    return {
        "agent_id": int(agent_id),
        "candidate_available": selected is not None,
        "temporary_reference": (
            np.asarray(selected.point, dtype=float).tolist()
            if selected is not None
            else np.asarray(terminal_goal, dtype=float).tolist()
        ),
        "selected_candidate_id": selected_id,
        "selection_source": "proposal_coarse_top1",
        "selected_null": False,
        "no_candidate_fallback": selected is None,
        "proposal_count_before_consumer": int(count_before_consumer),
        "K_t": len(proposals),
        "candidate_world_points": [item.point.tolist() for item in proposals],
        "proposal_scores": [float(item.score) for item in proposals],
        "candidate_metadata": [
            {
                key: jsonable(value)
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
        "fp_shep_candidate_records": [],
        "selected_fp_shep_score": None,
        "selected_fp_shep_rank_1based": None,
        "fp_shep_selected_candidate_id": None,
        "GAT_confidence": None,
        "gat_diagnostic_not_executed": True,
        "proposal_top1_is_score_argmax": (
            selected_id is None
            or selected_id
            == int(np.argmax([float(item.score) for item in proposals]))
        ),
    }


def build_online_proposal_plan(
    *,
    env: Any,
    config: Mapping[str, Any],
    policy: Any,
    gat_model: Any,
    gat_device: Any,
    scenario: str,
    seed: int,
    runtime_recorder: Any | None = None,
    selector: str = "proposal",
) -> dict[str, Any]:
    """Select Proposal coarse Top-1 without invoking FP-SHEP or GAT."""

    del policy, gat_model, gat_device, selector
    upper_started = time.perf_counter_ns()
    proposal_config = ProposalConfig(**dict(config["proposal_config"]))
    per_agent: list[tuple[ImmutableCandidate, ...]] = []
    counts: list[int] = []
    coarse_ranking_ms = 0.0
    proposal_started = time.perf_counter_ns()
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
        coarse_ranking_ms += (time.perf_counter_ns() - truncation_started) / 1.0e6
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

    pack_started = time.perf_counter_ns()
    proposals_by_agent: list[tuple[Any, ...]] = []
    reconstruction_equivalent = True
    for candidates in immutable.per_agent:
        proposals, _, equivalent = reconstruct_proposals(candidates)
        proposals_by_agent.append(proposals)
        reconstruction_equivalent &= bool(equivalent)
    if not reconstruction_equivalent:
        raise RuntimeError("Proposal reconstruction changed candidate semantics")
    candidate_pack_ms = (time.perf_counter_ns() - pack_started) / 1.0e6

    terminal_goals = np.asarray(env.goals, dtype=float)
    selected_ids = [0 if proposals else None for proposals in proposals_by_agent]
    references = terminal_goals.copy()
    available = np.zeros(int(env.num_agents), dtype=bool)
    records: list[dict[str, Any]] = []
    for agent_id, (proposals, selected_id, count_before) in enumerate(
        zip(proposals_by_agent, selected_ids, counts, strict=True)
    ):
        if selected_id is not None:
            references[agent_id] = proposals[selected_id].point
            available[agent_id] = True
        records.append(
            _proposal_candidate_record(
                agent_id=agent_id,
                proposals=proposals,
                selected_id=selected_id,
                terminal_goal=terminal_goals[agent_id],
                count_before_consumer=count_before,
            )
        )
    plan = {
        "selection_source": "proposal_coarse_top1",
        "references": references.tolist(),
        "available": available.tolist(),
        "candidate_records": records,
    }
    plan["selection_plan_hash"] = stable_hash(plan)
    selection_decode_ms = (time.perf_counter_ns() - pack_started) / 1.0e6
    upper_planning_total_ms = (time.perf_counter_ns() - upper_started) / 1.0e6
    runtime_components = {
        "proposal_generation_ms": float(proposal_generation_ms),
        "coarse_ranking_ms": float(coarse_ranking_ms),
        "candidate_pack_ms": float(candidate_pack_ms),
        "fp_shep_state_prepare_ms": 0.0,
        "fp_shep_actor_forward_ms": 0.0,
        "fp_shep_dmp_rollout_ms": 0.0,
        "fp_shep_geometry_metric_ms": 0.0,
        "fp_shep_total_ms": 0.0,
        "fp_shep_preview_actor_ms": 0.0,
        "graph_build_ms": 0.0,
        "graph_feature_build_ms": 0.0,
        "graph_edge_build_ms": 0.0,
        "graph_tensor_transfer_ms": 0.0,
        "gat_forward_ms": 0.0,
        "selection_decode_ms": float(selection_decode_ms),
        "upper_planning_total_ms": float(upper_planning_total_ms),
    }
    if runtime_recorder is not None:
        runtime_recorder.record_upper_event(runtime_components)
    return {
        "plan": plan,
        "candidate_bundle_hash": immutable.candidate_set_hash,
        "preview_historical_gate_verified": None,
        "preview_transition_call_count": 0,
        "proposal_reconstruction_equivalent": True,
        "graph_schema_match": None,
        "graph_schema_applicable": False,
        "selector": "proposal",
        "candidate_count_per_agent": [len(items) for items in proposals_by_agent],
        "computed_agent_ids": list(range(int(env.num_agents))),
        "fp_shep_vectorized": False,
        "proposal_only_selector_independence": True,
        "runtime_components": runtime_components,
    }


_ORIGINAL_EVALUATE = base.evaluate_formal_episode


def evaluate_selector_episode(
    method: Mapping[str, str],
    runtime_bundle: Mapping[str, Any],
    entry: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray], list[dict[str, Any]]]:
    if method["engine"] != "rerr_proposal":
        return _ORIGINAL_EVALUATE(method, runtime_bundle, entry)

    config = runtime_bundle["config"]
    runtime = runtime_bundle["runtime"]
    recorder = OnlineRuntimeRecorder()
    policy = TimedPolicyProxy(runtime.policy, recorder)
    with recorder.instrument_dmp(), recorder.scoped_context(
        evaluation_block="recurrent_selector_ablation_confirmation",
        method_id=method["method_id"],
        scenario_id=entry["scenario_id"],
        stage=entry["stage"],
        family=entry["family"],
    ):
        episode, agents, events, _, extra = base.run_rerr_episode(
            config=runtime.eval_config,
            settings=runtime.settings,
            multi_config=runtime.multi_config,
            policy=policy,
            gat_model=runtime.gat_model,
            gat_device=runtime.gat_device,
            # Reuse the accepted R-ERR edge/rearm and lifecycle branch exactly.
            method=base.METHOD_RERR_FP_SHEP,
            scenario=entry["scenario_id"],
            seed=int(entry["seed"]),
            environment_builder=runtime.builder,
            runtime_recorder=recorder,
            upper_plan_builder=build_online_proposal_plan,
        )
    trajectory = base._rerr_trajectory(extra, entry)
    episode = dict(episode)
    episode["planning_runtime_ms"] = float(episode["upper_planning_total_ms"])
    episode["normal_replanning_count"] = max(
        0,
        int(episode.get("replanning_count", 0))
        - int(episode.get("emergency_replanning_count", 0)),
    )
    episode["planning_runtime_per_decision_ms"] = (
        float(episode["upper_planning_total_ms"] / episode["planning_decision_count"])
        if episode["planning_decision_count"]
        else None
    )
    episode["proposal_only_selector_independence"] = True
    row, agent_rows = base._standardize(episode, agents, trajectory, entry, method)
    normalized_events = []
    for event in events:
        item = dict(event)
        item["method"] = "proposal_rerr"
        normalized_events.append(item)
    return row, agent_rows, trajectory, normalized_events


def _shared_contract(configs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    keys = (
        "training_config",
        "sac_checkpoint",
        "sac_checkpoint_sha256_expected",
        "base_execution_config",
        "num_agents",
        "top_k",
        "H_preview",
        "max_steps",
        "dt",
        "peer_radius",
        "maximum_speed_norm_mps",
        "handoff_threshold_m",
        "workspace_bounds",
        "sensor",
        "peer_information_contract",
        "proposal_config",
        "err",
    )
    reference = configs[METHOD_ORDER[0]]
    equality = {
        key: all(configs[method_id][key] == reference[key] for method_id in METHOD_ORDER)
        for key in keys
    }
    return {"fields": list(keys), "equality": equality, "all_equal": all(equality.values())}


def preflight_and_audit() -> None:
    for directory in (
        ARTIFACT_ROOT / "00_context",
        FREEZE_DIR,
        MANIFEST_DIR,
        AUDIT_DIR,
        RECORD_DIR,
        ARTIFACT_ROOT / "05_pairing",
        ARTIFACT_ROOT / "06_statistics",
        ARTIFACT_ROOT / "07_failure_analysis",
        ARTIFACT_ROOT / "08_continuous_metrics",
        ARTIFACT_ROOT / "09_runtime",
        ARTIFACT_ROOT / "10_paper_ready",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    if (MANIFEST_DIR / "SELECTOR_ABLATION_MANIFEST.json").exists():
        raise RuntimeError("preflight is forbidden after manifest freeze")

    source_formal_freeze = load_json(FORMAL_V2_FREEZE)
    gat_freeze = load_json(FINAL_GAT_FREEZE)
    fp = load_json(SOURCE_FP_CONFIG)
    gat = load_json(SOURCE_GAT_CONFIG)
    proposal = proposal_method_config()
    configs = {
        METHODS[0]["method_id"]: proposal,
        METHODS[1]["method_id"]: fp,
        METHODS[2]["method_id"]: gat,
    }
    shared = _shared_contract(configs)
    checkpoint_sha = base.sha256_file(REPO_ROOT / gat["gat_checkpoint"])
    method_audit = {
        "schema_version": "recurrent_selector_ablation_method_freeze_v1",
        "status": "PASS" if shared["all_equal"] else "FAIL",
        "created_at": datetime.now().astimezone().isoformat(),
        "confirmatory_not_development": True,
        "METHOD_PARAMETERS_CHANGED": "NO",
        "NETWORK_RETRAINED": "NO",
        "NEW_CHECKPOINT_SELECTION": "NO",
        "shared_contract": shared,
        "selector_only_difference": {
            METHODS[0]["method_id"]: "Proposal coarse-score Top-1",
            METHODS[1]["method_id"]: "FP-SHEP H4 score Top-1",
            METHODS[2]["method_id"]: "frozen GAT-R contextual logits",
        },
        "formal_v2_freeze_sha256": base.sha256_file(FORMAL_V2_FREEZE),
        "source_fp_config_sha256": base.sha256_file(SOURCE_FP_CONFIG),
        "source_gat_config_sha256": base.sha256_file(SOURCE_GAT_CONFIG),
        "gat_checkpoint_path": gat["gat_checkpoint"],
        "gat_checkpoint_sha256": checkpoint_sha,
        "formal_v2_gat_checkpoint_sha256": source_formal_freeze["checkpoint_sha256"]["gat"],
        "final_checkpoint_freeze_sha256": gat_freeze["selected_checkpoint_sha256"],
        "gat_checkpoint_matches_formal_v2": (
            checkpoint_sha
            == source_formal_freeze["checkpoint_sha256"]["gat"]
            == gat_freeze["selected_checkpoint_sha256"]
        ),
        "formal_v2_performance_used_for_tuning": False,
    }
    if not method_audit["gat_checkpoint_matches_formal_v2"]:
        method_audit["status"] = "FAIL"
    base.atomic_json(METHOD_FREEZE_AUDIT, method_audit)
    if method_audit["status"] != "PASS":
        raise RuntimeError(f"method freeze audit failed: {method_audit}")

    integration_manifest_path = (
        REPO_ROOT
        / "artifacts/semi_structured_long_range_main_benchmark/20260820_193228/08_development/development_manifest.json"
    )
    manifest = load_json(integration_manifest_path)
    entry = dict(manifest["entries"][0])
    checks: list[dict[str, Any]] = []
    proposal_events: list[dict[str, Any]] = []
    for method in METHODS:
        runtime = {"config": configs[method["method_id"]], "runtime": base.DevelopmentRuntime(configs[method["method_id"]], manifest)}
        row, agents, trajectory, events = evaluate_selector_episode(method, runtime, entry)
        positions = np.asarray(trajectory["positions"], dtype=float)
        if row["method_id"] != method["method_id"] or len(agents) != 3:
            raise RuntimeError("preflight standardized identity mismatch")
        if positions.ndim != 3 or positions.shape[1:] != (3, 3):
            raise RuntimeError("preflight trajectory shape mismatch")
        checks.append(
            {
                "method_id": method["method_id"],
                "status": "PASS",
                "trajectory_shape": list(positions.shape),
                "event_count": len(events),
                "performance_values_retained": False,
            }
        )
        if method["engine"] == "rerr_proposal":
            proposal_events = events

    selection_events = [event for event in proposal_events if int(event.get("K_t", 0)) > 0]
    score_order = all(
        list(event.get("candidate_scores", []))
        == sorted(event.get("candidate_scores", []), reverse=True)
        for event in selection_events
    )
    selected_top1 = all(event.get("selected_candidate_id") == 0 for event in selection_events)
    no_fp_scores = all(not event.get("fp_shep_scores") for event in selection_events)
    no_gat = all(event.get("gat_selected_candidate_id") is None for event in selection_events)
    zero_fp_compute = all(
        float((event.get("runtime_components") or {}).get("fp_shep_total_ms", 0.0)) == 0.0
        for event in selection_events
    )
    zero_gat_compute = all(
        float((event.get("runtime_components") or {}).get("gat_forward_ms", 0.0)) == 0.0
        for event in selection_events
    )
    proposal_audit = {
        "schema_version": "proposal_only_selector_independence_audit_v1",
        "status": "PASS" if all((score_order, selected_top1, no_fp_scores, no_gat, zero_fp_compute, zero_gat_compute)) else "FAIL",
        "created_at": datetime.now().astimezone().isoformat(),
        "source_definition": "Guidance.reference_point_proposal_demo.propose_reference_points sorts descending by existing coarse score after hard feasibility and positive-progress filtering",
        "top_k_adapter_definition": "planning.policy_preview.adapt_candidate_proposals preserves order and truncates only",
        "selection_definition": "candidate index 0 if and only if at least one proposal exists; otherwise existing terminal fallback",
        "dynamic_preflight_selection_event_count": len(selection_events),
        "candidate_scores_descending": score_order,
        "all_nonempty_selections_are_candidate_zero": selected_top1,
        "fp_shep_score_records_absent": no_fp_scores,
        "gat_selection_absent": no_gat,
        "fp_shep_compute_zero": zero_fp_compute,
        "gat_compute_zero": zero_gat_compute,
        "FP_SHEP_AFFECTS_FINAL_SELECTION": "NO",
        "GAT_AFFECTS_FINAL_SELECTION": "NO",
        "hard_feasibility_reused": True,
        "random_or_index_baseline_used": False,
        "performance_values_retained": False,
    }
    base.atomic_json(PROPOSAL_AUDIT, proposal_audit)
    if proposal_audit["status"] != "PASS":
        raise RuntimeError(f"Proposal-only audit failed: {proposal_audit}")
    preflight = {
        "schema_version": "recurrent_selector_ablation_preflight_v1",
        "status": "PASS",
        "created_at": datetime.now().astimezone().isoformat(),
        "source_split": "preexisting development integration scene",
        "source_manifest": str(integration_manifest_path.relative_to(REPO_ROOT).as_posix()),
        "source_manifest_sha256": base.sha256_file(integration_manifest_path),
        "scenario_id": entry["scenario_id"],
        "method_count": len(checks),
        "checks": checks,
        "formal_data_used": False,
        "performance_values_retained": False,
    }
    base.atomic_json(FREEZE_DIR / "selector_engine_preflight.json", preflight)
    print(json.dumps({"phase": "preflight", "status": "PASS", "method_count": 3}), flush=True)


def configure(*, frozen: bool) -> None:
    base.ARTIFACT_ROOT = ARTIFACT_ROOT
    base.FREEZE_DIR = FREEZE_DIR
    base.MANIFEST_DIR = MANIFEST_DIR
    base.RECORD_DIR = RECORD_DIR
    base.FINAL_METHOD_FREEZE = FREEZE_DIR / "FINAL_SELECTOR_METHOD_FREEZE.json"
    base.FORMAL_RUN_FREEZE = FREEZE_DIR / "SELECTOR_ABLATION_RUN_FREEZE.json"
    base.FORMAL_MANIFEST = MANIFEST_DIR / "SELECTOR_ABLATION_MANIFEST.json"
    base.FORMAL_REGISTRY = MANIFEST_DIR / "ALL_USED_SCENE_REGISTRY.csv"
    base.DEVELOPMENT_SELECTION_FREEZE = METHOD_FREEZE_AUDIT
    base.SHORT_COORDINATION_RECONCILIATION = PROPOSAL_AUDIT
    base.FORMAL_ENGINE_PREFLIGHT = FREEZE_DIR / "selector_engine_preflight.json"
    base.SEED_BASE = 3_800_000_000
    base.SCENARIOS_PER_STAGE = 100
    base.EXPECTED_SCENARIOS = 400
    base.METHODS = METHODS
    base.METHOD_BY_ID = METHOD_BY_ID
    base.METHOD_ORDER = METHOD_ORDER
    base.DEVELOPMENT_EVIDENCE = {}
    base.SOURCE_PATHS = tuple(
        dict.fromkeys(
            (
                *base.SOURCE_PATHS,
                "Multi-agent_Algo_lib/scripts/run_recurrent_selector_ablation_confirmation.py",
                str(FORMAL_V2_FREEZE.relative_to(REPO_ROOT).as_posix()),
                str(FINAL_GAT_FREEZE.relative_to(REPO_ROOT).as_posix()),
            )
        )
    )
    base.development_gate = method_gate
    base.historical_scene_registry = historical_scene_registry
    base.resolved_method_config = frozen_method_config if frozen else derived_method_config
    base.evaluate_formal_episode = evaluate_selector_episode


def prepare() -> None:
    configure(frozen=False)
    base.prepare()
    validation = load_json(MANIFEST_DIR / "formal_manifest_validation.json")
    manifest = load_json(MANIFEST_DIR / "SELECTOR_ABLATION_MANIFEST.json")
    isolation = {
        "schema_version": "selector_ablation_scene_isolation_v1",
        "status": validation["status"],
        "created_at": datetime.now().astimezone().isoformat(),
        "scenario_count": len(manifest["entries"]),
        "history_registry_source": str(FORMAL_V2_REGISTRY.relative_to(REPO_ROOT).as_posix()),
        "history_registry_sha256": base.sha256_file(FORMAL_V2_REGISTRY),
        "historical_scene_row_count": validation["historical_scene_row_count"],
        "historical_overlap": validation["historical_overlap"],
        "internal_duplicates": validation["internal_duplicates"],
        "FORMAL_V2_REUSED": "NO",
        "SCENE_HISTORY_OVERLAP": sum(validation["historical_overlap"].values()),
        "manifest_sha256": manifest["manifest_sha256"],
        "formal_episode_count_when_checked": 0,
    }
    base.atomic_json(SCENE_ISOLATION, isolation)
    if isolation["status"] != "PASS" or isolation["SCENE_HISTORY_OVERLAP"] != 0:
        raise RuntimeError(f"scene isolation failed: {isolation}")
    print(json.dumps({"phase": "isolation", "status": "PASS", "overlap": 0}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("preflight", "prepare", "run", "status", "finalize"))
    parser.add_argument("--method-id", choices=METHOD_ORDER)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.phase == "preflight":
        preflight_and_audit()
        return
    if args.phase == "prepare":
        prepare()
        return
    configure(frozen=True)
    if args.phase == "run":
        if args.method_id is None:
            raise ValueError("--method-id is required for run")
        base.run(args.method_id, args.shard_index, args.shard_count, args.limit)
    elif args.phase == "status":
        base.status()
    else:
        base.finalize()


if __name__ == "__main__":
    main()
