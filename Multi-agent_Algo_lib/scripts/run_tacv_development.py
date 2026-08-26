#!/usr/bin/env python3
"""Run the authorized post-GAT TACV Development and sealed Holdout arms."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as _pandas  # noqa: F401  # Windows pyarrow/torch initialization order


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
for search_path in (REPO_ROOT, ALGO_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from planning.online_runtime_instrumentation import (  # noqa: E402
    OnlineRuntimeRecorder,
    TimedPolicyProxy,
)
from planning.pre_gat_220step_revalidation import stable_hash  # noqa: E402
from planning.transient_aware_candidate_veto import (  # noqa: E402
    TACVConfig,
    select_tacv_candidate,
)
from scripts import evaluate_gat_v1_err_development as evalmod  # noqa: E402
from scripts.evaluate_gat_closed_loop import _rank_1based  # noqa: E402
from scripts.evaluate_gat_v1_err_development import (  # noqa: E402
    METHOD_RERR_GAT,
    run_episode,
)
from scripts.run_continuous_reference_transition import (  # noqa: E402
    atomic_json,
    save_episode,
    sha256_file,
    write_csv,
)
from scripts.run_gat_recurrent_r_development import (  # noqa: E402
    FileBackedBuilder,
    load_json,
    resolve_runtime_config,
)
from scripts.run_long_range_contract_pilot import summarize_record  # noqa: E402
from scripts.run_long_range_development import DevelopmentRuntime  # noqa: E402


ROOT = REPO_ROOT / "artifacts/transient_aware_candidate_veto/20260824_184551"
SOURCE_ROOT = REPO_ROOT / "artifacts/gat_recurrent_smoothness_rescue/20260821_180244"
CRT_ROOT = REPO_ROOT / "artifacts/continuous_reference_transition/20260824_132552"
GATE_PATH = ROOT / "04_gate_decision/TACV_GATE_DECISION.json"
DEV_MANIFEST = CRT_ROOT / "00_context/CRT_DEVELOPMENT_MANIFEST.json"
DEV_SOURCE_RECORDS = CRT_ROOT / "04_development/records/original/episode_records"
DEV_CONFIG = REPO_ROOT / "configs/evaluation/gat_recurrent_r_fp_anchor_dev.json"
HOLDOUT_MANIFEST = ROOT / "11_freeze/TACV_HOLDOUT_MANIFEST.json"
HOLDOUT_SOURCE_RECORDS = (
    SOURCE_ROOT
    / "13_objective_revision/08_holdout/GAT_R_FP_ANCHOR_HOLDOUT400/episode_records"
)
HOLDOUT_CONFIG = REPO_ROOT / "configs/evaluation/gat_recurrent_r_fp_anchor_holdout.json"
FINAL_FREEZE = ROOT / "11_freeze/FINAL_TACV_FREEZE.json"


VARIANTS = {
    "tacv_mild": {
        "activation_source": "Development CLEAN P95",
        "activation_threshold": 920.4453700219636,
        "minimum_relative_reduction": 0.50,
        "maximum_gat_candidate_rank": 3,
    },
    "tacv_medium": {
        "activation_source": "Development CLEAN P90",
        "activation_threshold": 801.701109087464,
        "minimum_relative_reduction": 0.35,
        "maximum_gat_candidate_rank": 3,
    },
    "tacv_strong": {
        "activation_source": "Development CLEAN P90",
        "activation_threshold": 801.701109087464,
        "minimum_relative_reduction": 0.20,
        "maximum_gat_candidate_rank": 3,
    },
}


SOURCE_PATHS = (
    "planning/transient_aware_candidate_veto.py",
    "Multi-agent_Algo_lib/scripts/run_tacv_development.py",
    "planning/policy_preview.py",
    "planning/pre_gat_closed_loop.py",
    "planning/event_triggered_reference_reconstruction.py",
    "Multi-agent_Algo_lib/scripts/evaluate_gat_v1_err_development.py",
    "Environment/frozen_sac_dmp_execution.py",
)


def block_contract(block: str) -> dict[str, Path]:
    if block == "development":
        return {
            "manifest": DEV_MANIFEST,
            "source_records": DEV_SOURCE_RECORDS,
            "config": DEV_CONFIG,
            "output": ROOT / "06_development",
        }
    if block == "holdout":
        return {
            "manifest": HOLDOUT_MANIFEST,
            "source_records": HOLDOUT_SOURCE_RECORDS,
            "config": HOLDOUT_CONFIG,
            "output": ROOT / "08_holdout",
        }
    raise ValueError(f"unknown block {block}")


def output_dir(block: str, variant: str) -> Path:
    return block_contract(block)["output"] / "records" / variant


def config_for(variant: str) -> TACVConfig:
    values = VARIANTS[variant]
    return TACVConfig(
        dt_s=0.1,
        activation_threshold=float(values["activation_threshold"]),
        minimum_relative_reduction=float(values["minimum_relative_reduction"]),
        maximum_gat_candidate_rank=int(values["maximum_gat_candidate_rank"]),
        apply_to_initial_selection=False,
    )


class TACVUpperPlanBuilder:
    """Capture existing previews and apply a post-GAT, pre-commit veto."""

    def __init__(self, config: TACVConfig) -> None:
        self.config = config
        self.event_rows: list[dict[str, Any]] = []
        self.call_rows: list[dict[str, Any]] = []
        self.total_runtime_ms = 0.0
        self._score = evalmod.score_fp_shep_candidates_batched

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        recorder = kwargs.get("runtime_recorder")
        context = dict(recorder.context) if recorder is not None else {}
        event_type = str(context.get("event_type", "UNKNOWN"))
        step = int(context.get("event_step", -1))
        scenario_id = str(kwargs["scenario"])
        preview_by_agent: dict[int, tuple[Any, ...]] = {}
        acceleration_by_agent: dict[int, np.ndarray] = {}

        def capture_score(*, env: Any, agent_index: int, **score_kwargs: Any):
            agent_index = int(agent_index)
            info = env.latest_controller_infos[agent_index] or {}
            current = np.asarray(info.get("applied_acceleration", np.zeros(3)), dtype=float)
            if current.shape != (3,) or not np.all(np.isfinite(current)):
                raise RuntimeError("current applied acceleration is unavailable for TACV")
            records = self._score(env=env, agent_index=agent_index, **score_kwargs)
            preview_by_agent[agent_index] = tuple(records)
            acceleration_by_agent[agent_index] = current.copy()
            return records

        previous = evalmod.score_fp_shep_candidates_batched
        if previous is not self._score:
            raise RuntimeError("FP-SHEP scorer was unexpectedly replaced")
        evalmod.score_fp_shep_candidates_batched = capture_score
        try:
            result = evalmod.build_online_gat_plan_optimized(**kwargs)
        finally:
            evalmod.score_fp_shep_candidates_batched = previous

        started = time.perf_counter_ns()
        plan = result["plan"]
        replaced_ids: list[int] = []
        for agent_id in sorted(preview_by_agent):
            record = plan["candidate_records"][agent_id]
            decision = select_tacv_candidate(
                original_selected_candidate_id=record.get("selected_candidate_id"),
                class_logits=record["class_logits"],
                preview_records=preview_by_agent[agent_id],
                interaction_records=record["all_candidate_interaction_records"],
                current_applied_acceleration=acceleration_by_agent[agent_id],
                event_type=event_type,
                config=self.config,
            )
            diagnostics = list(decision.pop("candidate_diagnostics"))
            original_id = decision["original_selected_candidate_id"]
            effective_id = decision["effective_selected_candidate_id"]
            original_candidate = (
                None if original_id is None else diagnostics[int(original_id)]
            )
            effective_candidate = (
                None if effective_id is None else diagnostics[int(effective_id)]
            )
            any_peer = any(int(row["interaction_edge_count"]) > 0 for row in diagnostics)
            event_row = {
                "scenario_id": scenario_id,
                "seed": int(kwargs["seed"]),
                "step": step,
                "time_s": float(step * self.config.dt_s),
                "event_type": event_type,
                "agent_id": agent_id,
                "any_peer_interaction_edge": any_peer,
                **decision,
                "original_preview_min_clearance_m": (
                    None if original_candidate is None else original_candidate["preview_min_clearance_m"]
                ),
                "replacement_preview_min_clearance_m": (
                    None if not decision["replaced"] else effective_candidate["preview_min_clearance_m"]
                ),
                "original_candidate_risky": (
                    None if original_candidate is None else original_candidate["risky"]
                ),
                "replacement_candidate_risky": (
                    None if not decision["replaced"] else effective_candidate["risky"]
                ),
                "original_minimum_predicted_separation_m": (
                    None if original_candidate is None else original_candidate["minimum_predicted_separation_m"]
                ),
                "replacement_minimum_predicted_separation_m": (
                    None if not decision["replaced"] else effective_candidate["minimum_predicted_separation_m"]
                ),
                "original_maximum_risk_duration_s": (
                    None if original_candidate is None else original_candidate["maximum_risk_duration_s"]
                ),
                "replacement_maximum_risk_duration_s": (
                    None if not decision["replaced"] else effective_candidate["maximum_risk_duration_s"]
                ),
                "original_gat_logit": (
                    None if original_candidate is None else original_candidate["gat_logit"]
                ),
                "replacement_gat_logit": (
                    None if not decision["replaced"] else effective_candidate["gat_logit"]
                ),
            }
            self.event_rows.append(event_row)
            record.update(
                {
                    "tacv_enabled": True,
                    "tacv_activated": bool(decision["activated"]),
                    "tacv_replaced": bool(decision["replaced"]),
                    "tacv_original_selected_candidate_id": original_id,
                    "tacv_effective_selected_candidate_id": effective_id,
                    "tacv_reason": decision["reason"],
                    "tacv_original_J_preview": decision["original_J_preview"],
                    "tacv_replacement_J_preview": decision["replacement_J_preview"],
                    "tacv_predicted_relative_reduction": decision["predicted_relative_reduction"],
                }
            )
            if not decision["replaced"]:
                continue
            replaced_ids.append(agent_id)
            candidate_id = int(effective_id)
            point = np.asarray(record["candidate_world_points"][candidate_id], dtype=float)
            plan["references"][agent_id] = point.tolist()
            plan["available"][agent_id] = True
            record["temporary_reference"] = point.tolist()
            record["selected_candidate_id"] = candidate_id
            record["selected_null"] = False
            record["selected_class"] = candidate_id + 1
            record["selected_proposal_rank"] = candidate_id
            record["selected_proposal_rank_1based"] = candidate_id + 1
            record["selected_proposal_score"] = float(record["proposal_scores"][candidate_id])
            fp_scores = [
                float(row["fp_shep_online_score"])
                for row in record["fp_shep_candidate_records"]
            ]
            record["selected_fp_shep_score"] = fp_scores[candidate_id]
            record["selected_fp_shep_rank_1based"] = _rank_1based(fp_scores, candidate_id)
            probabilities = record["class_probabilities"]
            record["GAT_confidence"] = float(probabilities[candidate_id + 1])
            graph = result["_audit_graphs_by_agent"][agent_id]
            record["selected_edge_records"] = evalmod._selected_edge_records(
                graph, candidate_id
            )
            record.update(evalmod._selected_edge_diagnostics(graph, candidate_id))

        plan["selection_plan_hash"] = stable_hash(
            {
                "selection_source": plan["selection_source"],
                "references": plan["references"],
                "available": plan["available"],
                "candidate_records": plan["candidate_records"],
            }
        )
        runtime_ms = (time.perf_counter_ns() - started) / 1.0e6
        self.total_runtime_ms += runtime_ms
        call_row = {
            "scenario_id": scenario_id,
            "seed": int(kwargs["seed"]),
            "step": step,
            "event_type": event_type,
            "evaluated_agent_count": len(preview_by_agent),
            "replaced_agent_count": len(replaced_ids),
            "tacv_runtime_ms": float(runtime_ms),
        }
        self.call_rows.append(call_row)
        result["runtime_components"]["tacv_veto_ms"] = float(runtime_ms)
        result["runtime_components"]["upper_planning_total_plus_tacv_ms"] = float(
            result["runtime_components"]["upper_planning_total_ms"] + runtime_ms
        )
        return result


def prepare() -> None:
    gate = load_json(GATE_PATH)
    if gate["TACV_AUTHORIZED"] != "YES":
        raise RuntimeError("the two diagnostic gates did not authorize TACV")
    for folder in (
        "05_tacv_implementation",
        "06_development",
        "07_failure_audit",
        "08_holdout",
        "09_runtime",
        "10_figures",
        "11_freeze",
        "12_paper_ready",
    ):
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    contract = {
        "schema_version": "tacv_implementation_contract_v1",
        "status": "FROZEN_BEFORE_DEVELOPMENT",
        "location": "post-GAT and post-existing interaction mask, immediately before active-reference commit",
        "initial_selection_modified": False,
        "graph_nodes_removed": False,
        "gat_logits_changed": False,
        "gat_rerun_after_veto": False,
        "additional_preview_rollout": False,
        "candidate_search_order": "original candidate-logit rank order",
        "maximum_candidate_rank": 3,
        "variants": VARIANTS,
        "safety_contract": str(
            ROOT / "03_replaceability/TACV_SAFETY_ADMISSIBILITY_CONTRACT.json"
        ),
        "formal_v2_authorized": False,
    }
    atomic_json(ROOT / "05_tacv_implementation/TACV_IMPLEMENTATION_CONTRACT.json", contract)
    runtime_config = resolve_runtime_config(load_json(DEV_CONFIG), "gat_r")
    freeze = {
        "schema_version": "tacv_predevelopment_freeze_v1",
        "status": "FROZEN_BEFORE_DEVELOPMENT_PERFORMANCE",
        "gate_decision_sha256": sha256_file(GATE_PATH),
        "implementation_contract_sha256": sha256_file(
            ROOT / "05_tacv_implementation/TACV_IMPLEMENTATION_CONTRACT.json"
        ),
        "development_manifest_sha256": sha256_file(DEV_MANIFEST),
        "development_scenario_count": len(load_json(DEV_MANIFEST)["entries"]),
        "source_sha256": {
            relative: sha256_file(REPO_ROOT / relative) for relative in SOURCE_PATHS
        },
        "checkpoint_sha256": {
            "gat_r": sha256_file(REPO_ROOT / runtime_config["gat_checkpoint"]),
            "sac_dmp": sha256_file(REPO_ROOT / runtime_config["sac_checkpoint"]),
        },
        "variants": VARIANTS,
        "selection_rule": [
            "team reliability",
            "peer-collision behavior",
            "smoothness reduction",
            "switch-transient reduction",
            "completion time",
            "compute",
        ],
        "development_acceptance": {
            "maximum_team_success_loss_pp": 1.0,
            "maximum_peer_collision_increase_pp": 1.0,
            "minimum_smoothness_reduction_percent": 5.0,
            "minimum_switch_transient_reduction_percent": 5.0,
        },
        "formal_v2_run": False,
    }
    atomic_json(ROOT / "11_freeze/PRE_DEVELOPMENT_TACV_FREEZE.json", freeze)


def verify_freeze(block: str, variant: str) -> None:
    freeze = load_json(ROOT / "11_freeze/PRE_DEVELOPMENT_TACV_FREEZE.json")
    if freeze["status"] != "FROZEN_BEFORE_DEVELOPMENT_PERFORMANCE":
        raise RuntimeError("TACV pre-Development freeze is invalid")
    for relative, expected in freeze["source_sha256"].items():
        if sha256_file(REPO_ROOT / relative) != expected:
            raise RuntimeError(f"frozen source changed: {relative}")
    if variant not in VARIANTS:
        raise ValueError(f"unknown TACV variant {variant}")
    if block == "holdout":
        final = load_json(FINAL_FREEZE)
        if final["status"] != "FROZEN_BEFORE_HOLDOUT":
            raise RuntimeError("Holdout final TACV freeze is invalid")
        if variant != final["selected_variant"]:
            raise RuntimeError("Holdout permits only the selected TACV variant")
        if sha256_file(HOLDOUT_MANIFEST) != final["holdout_manifest_sha256"]:
            raise RuntimeError("sealed Holdout manifest changed")


def run(block: str, variant: str, shard_index: int | None, shard_count: int | None, limit: int | None) -> None:
    verify_freeze(block, variant)
    contract = block_contract(block)
    manifest = load_json(contract["manifest"])
    runtime_config = resolve_runtime_config(load_json(contract["config"]), "gat_r")
    runtime = DevelopmentRuntime(runtime_config, {"entries": []})
    runtime.builder = FileBackedBuilder(manifest, runtime_config, SOURCE_ROOT)
    target = output_dir(block, variant)
    completed = {
        path.stem
        for path in (target / "episode_records").glob("*.json")
        if not path.stem.endswith("_SOFTWARE_ERROR")
    }
    indexed = list(enumerate(manifest["entries"]))
    if shard_count is not None:
        if shard_index is None or not 0 <= shard_index < shard_count:
            raise ValueError("invalid shard index/count")
        indexed = [pair for pair in indexed if pair[0] % shard_count == shard_index]
    pending = [entry for _, entry in indexed if str(entry["scenario_id"]) not in completed]
    if limit is not None:
        pending = pending[:limit]
    for entry in pending:
        sid = str(entry["scenario_id"])
        print(f"[TACV:{block}:{variant}] start {sid}", flush=True)
        recorder = OnlineRuntimeRecorder()
        proxy = TimedPolicyProxy(runtime.policy, recorder)
        builder = TACVUpperPlanBuilder(config_for(variant))
        source_scene = load_json(SOURCE_ROOT / str(entry["scenario_file"]))
        try:
            with recorder.instrument_dmp(), recorder.scoped_context(
                evaluation_block=f"tacv_{block}",
                configuration_id=f"TACV_{block.upper()}_{variant.upper()}",
                stage=entry["stage"],
                family=entry["family"],
                scenario_id=sid,
                seed=int(entry["seed"]),
                method=variant,
            ):
                episode, agents, events, triggers, extra = run_episode(
                    config=runtime.eval_config,
                    settings=runtime.settings,
                    multi_config=runtime.multi_config,
                    policy=proxy,
                    gat_model=runtime.gat_model,
                    gat_device=runtime.gat_device,
                    method=METHOD_RERR_GAT,
                    scenario=sid,
                    seed=int(entry["seed"]),
                    environment_builder=runtime.builder,
                    runtime_recorder=recorder,
                    upper_plan_builder=builder,
                    crt=None,
                )
            decisions = [
                row for row in builder.event_rows if row["event_type"] != "INITIAL_SELECTION"
            ]
            candidate_decisions = [
                row for row in decisions if row["original_selected_candidate_id"] is not None
            ]
            replacements = [row for row in candidate_decisions if row["replaced"]]
            episode.update(
                {
                    "tacv_enabled": True,
                    "tacv_variant": variant,
                    "tacv_candidate_decision_count": len(candidate_decisions),
                    "tacv_activation_count": sum(bool(row["activated"]) for row in candidate_decisions),
                    "tacv_replacement_count": len(replacements),
                    "tacv_replacement_rate": float(len(replacements) / max(len(candidate_decisions), 1)),
                    "tacv_gat_effective_top1_retention_rate": float(1.0 - len(replacements) / max(len(candidate_decisions), 1)),
                    "tacv_mean_replacement_gat_rank": (
                        None if not replacements else float(np.mean([row["replacement_gat_candidate_rank"] for row in replacements]))
                    ),
                    "tacv_mean_predicted_relative_reduction": (
                        None if not replacements else float(np.mean([row["predicted_relative_reduction"] for row in replacements]))
                    ),
                    "tacv_peer_interaction_replacement_fraction": (
                        None if not replacements else float(np.mean([bool(row["any_peer_interaction_edge"]) for row in replacements]))
                    ),
                    "tacv_runtime_ms": float(builder.total_runtime_ms),
                    "total_online_algorithm_compute_plus_tacv_ms": float(
                        episode["total_online_algorithm_compute_ms"] + builder.total_runtime_ms
                    ),
                }
            )
            summary = summarize_record(entry, episode, events, triggers, extra["path_rows"])
            summary.update(
                {
                    "configuration_id": f"TACV_{block.upper()}_{variant.upper()}",
                    "variant": variant,
                    "method": "Proposed-TACV",
                    "team_success": bool(episode["team_success"]),
                    "collision": bool(episode["collision"]),
                    "timeout": bool(episode["timeout"]),
                    "performance_used_for_selection": block == "development",
                    **{key: value for key, value in episode.items() if key.startswith("tacv_")},
                    "total_online_algorithm_compute_plus_tacv_ms": episode["total_online_algorithm_compute_plus_tacv_ms"],
                }
            )
            save_episode(target, entry, episode, agents, events, triggers, extra, summary, source_scene)
            atomic_json(target / "tacv_event_records" / f"{sid}.json", {
                "scenario_id": sid,
                "variant": variant,
                "event_rows": builder.event_rows,
                "call_rows": builder.call_rows,
            })
            print(
                f"[TACV:{block}:{variant}] complete {sid} success={int(episode['team_success'])} "
                f"replacement={len(replacements)}",
                flush=True,
            )
        except Exception as error:
            atomic_json(target / "episode_records" / f"{sid}_SOFTWARE_ERROR.json", {
                "scenario_id": sid,
                "seed": int(entry["seed"]),
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
            })
            raise


def finalize(block: str, variant: str) -> None:
    contract = block_contract(block)
    expected = len(load_json(contract["manifest"])["entries"])
    target = output_dir(block, variant)
    rows = [
        load_json(path)["summary"]
        for path in sorted((target / "episode_records").glob("*.json"))
        if not path.stem.endswith("_SOFTWARE_ERROR")
    ]
    errors = list((target / "episode_records").glob("*_SOFTWARE_ERROR.json"))
    write_csv(target / "episode_summary.csv", rows)
    reconciliation = {
        "schema_version": "tacv_variant_reconciliation_v1",
        "block": block,
        "variant": variant,
        "expected_episode_count": expected,
        "completed_episode_count": len(rows),
        "unique_scenario_count": len({row["scenario_id"] for row in rows}),
        "software_error_count": len(errors),
        "status": "PASS" if len(rows) == expected and not errors else "INCOMPLETE",
    }
    atomic_json(target / "reconciliation.json", reconciliation)
    print(json.dumps(reconciliation), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "run", "finalize"))
    parser.add_argument("--block", choices=("development", "holdout"))
    parser.add_argument("--variant", choices=tuple(VARIANTS))
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare()
        return
    if args.block is None or args.variant is None:
        raise ValueError("--block and --variant are required")
    if args.phase == "run":
        run(args.block, args.variant, args.shard_index, args.shard_count, args.limit)
    else:
        finalize(args.block, args.variant)


if __name__ == "__main__":
    main()
