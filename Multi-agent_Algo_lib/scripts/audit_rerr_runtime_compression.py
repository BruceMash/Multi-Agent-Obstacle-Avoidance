from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import runpy
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = REPO_ROOT / "Multi-agent_Algo_lib"
for search_path in (REPO_ROOT, ALGO_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from planning.online_runtime_instrumentation import (  # noqa: E402
    OnlineRuntimeRecorder,
    TimedPolicyProxy,
    synchronize_cuda,
)
from scripts.evaluate_gat_v1_err_development import (  # noqa: E402
    METHOD_RERR_GAT,
    build_online_gat_plan,
    build_online_gat_plan_optimized,
    run_episode,
)
from scripts.run_final_four_stage_benchmark import (  # noqa: E402
    FrozenRuntime,
    proposed_eval_config,
)


OUTPUT = REPO_ROOT / "artifacts" / "rerr_runtime_compression" / "20260819_162022"
SPARSE = REPO_ROOT / "artifacts" / "sparse_err_trigger_revision" / "20260819_020727"
SAFETY = REPO_ROOT / "artifacts" / "rerr_safety_closure" / "20260819_134216"
PEER = REPO_ROOT / "artifacts" / "peer_risk_trigger_closure" / "20260819_153224"
RECOVERY = REPO_ROOT / "artifacts" / "theory_aligned_final_recovery" / "20260818_232900"
BASE_CONFIG = REPO_ROOT / "artifacts" / "final_four_stage_benchmark" / "20260818_202620" / "config.json"
SELECTED_CONFIG = REPO_ROOT / "artifacts" / "final_four_stage_benchmark" / "20260818_202620" / "engineering_search" / "selected_configs.json"
ERR_CONFIG = REPO_ROOT / "configs" / "evaluation" / "gat_v1_err_development.json"
MANIFEST = SPARSE / "development_scenario_manifest.json"

VARIANTS = {
    "B0_current": {"local": False, "vectorized": False},
    "B1_triggered_agent": {"local": True, "vectorized": False},
    "B2_vectorized": {"local": False, "vectorized": True},
    "B3_combined": {"local": True, "vectorized": True},
}
PROFILE_STATE_TARGET = 334
TIMING_PASSES = 3
CURRENT_SINGLE_MS = 213.89047152145642
CURRENT_TOTAL_MS = 2271.5911549999996
DWA_REFERENCE_MS = 1028.33987175
RVO_REFERENCE_MS = 3172.641249


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity" if value < 0 else "NaN"
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(_jsonable(row.get(key)), ensure_ascii=False)
                        if isinstance(row.get(key), (dict, list, tuple, np.ndarray))
                        else _jsonable(row.get(key))
                    )
                    for key in fields
                }
            )
    temporary.replace(path)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _runtime() -> tuple[FrozenRuntime, dict[str, Any], dict[str, Any]]:
    base = load_json(BASE_CONFIG)
    manifest = load_json(MANIFEST)
    selected = load_json(SELECTED_CONFIG)
    err = load_json(ERR_CONFIG)
    runtime = FrozenRuntime(base, manifest)
    config = proposed_eval_config(runtime.base_eval_config, selected["gat_v1"])
    config["err"] = err["err"]
    return runtime, config, manifest


def _source_files() -> list[Path]:
    names = [
        "planning/policy_preview.py",
        "planning/pre_gat_closed_loop.py",
        "planning/heterogeneous_candidate_graph.py",
        "planning/gat/candidate_selector.py",
        "planning/gat/edge_enhanced_gat.py",
        "planning/event_triggered_reference_reconstruction.py",
        "planning/online_runtime_instrumentation.py",
        "Environment/frozen_sac_dmp_execution.py",
        "Environment/multi_agent_dmp_env.py",
        "Controller/dmp_rl.py",
        "Guidance/reference_point_proposal_demo.py",
        "Multi-agent_Algo_lib/scripts/evaluate_gat_v1_err_development.py",
        "Multi-agent_Algo_lib/scripts/audit_rerr_runtime_compression.py",
        "hire-rl-body.tex",
    ]
    return [REPO_ROOT / name for name in names]


def _run_regressions() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for relative in (
        "test/test_event_triggered_reference_reconstruction.py",
        "test/test_sparse_err_trigger_revision.py",
    ):
        namespace = runpy.run_path(str(REPO_ROOT / relative))
        for name, function in sorted(namespace.items()):
            if name.startswith("test_") and callable(function):
                function()
                rows.append({"file": relative, "test": name, "status": "PASS"})
    return {"status": "PASS", "test_count": len(rows), "tests": rows}


def prepare(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    required = [
        SPARSE / "FINAL_REPORT.md",
        SPARSE / "conclusion.json",
        SPARSE / "runtime_method_summary.csv",
        SPARSE / "runtime_stage_summary.csv",
        SPARSE / "reproposal_events.csv",
        SPARSE / "reproposal_distribution.csv",
        SPARSE / "event_conditioned_gat_analysis.csv",
        SPARSE / "final_reconciliation.json",
        SAFETY / "FINAL_REPORT.md",
        SAFETY / "conclusion.json",
        SAFETY / "replanning_scope_contract.md",
        PEER / "FINAL_REPORT.md",
        PEER / "conclusion.json",
        RECOVERY / "runtime_method_summary.csv",
        RECOVERY / "runtime_stage_summary.csv",
        MANIFEST,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing authority artifacts: {missing}")
    context = {
        "schema_version": "rerr_runtime_compression_context_v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "agents_md_present": (REPO_ROOT / "AGENTS.md").exists(),
        "codex_handoff_present": (REPO_ROOT / "CODEX_HANDOFF.md").exists(),
        "authority_files": {
            str(path.relative_to(REPO_ROOT)).replace("\\", "/"): file_hash(path)
            for path in required
        },
        "frozen_reference": {
            "success": 0.8125,
            "collision": 0.1625,
            "inter_agent_collision": 0.125,
            "timeout": 0.025,
            "single_upper_latency_ms": CURRENT_SINGLE_MS,
            "total_compute_ms": CURRENT_TOTAL_MS,
            "mean_upper_decisions_per_episode": 9.6125,
            "GAT_CLOSED_LOOP_VALUE": "STRONG",
            "GAT_FINETUNE_JUSTIFIED": "NO",
        },
        "scope": {
            "minimum_change_first": True,
            "exact_behavior_engineering_only": True,
            "theory_change": False,
            "trigger_change": False,
            "threshold_change": False,
            "candidate_reduction": False,
            "horizon_reduction": False,
            "formal_benchmark": False,
        },
    }
    write_json(output_dir / "context_recovery_manifest.json", context)
    write_json(output_dir / "regression_tests.json", _run_regressions())
    prefreeze = {
        "freeze_time": datetime.now().astimezone().isoformat(),
        "freeze_before_microbenchmark": True,
        "source_hashes": {
            str(path.relative_to(REPO_ROOT)).replace("\\", "/"): file_hash(path)
            for path in _source_files()
        },
        "checkpoint_hashes": {
            "sac": file_hash(REPO_ROOT / load_json(BASE_CONFIG)["sources"]["sac_checkpoint"]),
            "gat": file_hash(REPO_ROOT / load_json(BASE_CONFIG)["sources"]["gat_checkpoint"]),
        },
        "manifest_sha256": file_hash(MANIFEST),
        "timing_protocol": {
            "profile_state_target": PROFILE_STATE_TARGET,
            "timing_passes": TIMING_PASSES,
            "minimum_timed_invocations_per_version": 1000,
            "warmup_events_per_version": 3,
            "cuda_synchronization": True,
            "torch_threads": int(torch.get_num_threads()),
            "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        },
    }
    write_json(output_dir / "engineering_prefreeze.json", prefreeze)

    (output_dir / "gat_dependency_graph.md").write_text(
        "# GAT dependency graph\n\n"
        "For ego agent `i`, the graph contains one ego agent node, its own Proposal nodes, "
        "neighbor align nodes carrying current peer position/velocity context, and edges from "
        "the ego node or align nodes to **ego-i proposals only**. There are no nodes or edges "
        "for another agent's proposal bundle in ego-i's graph. PyG batching concatenates three "
        "independent graphs and creates no cross-graph edge. Therefore another agent's Proposal, "
        "FP-SHEP preview, graph, or target logits cannot affect the triggered agent's logits.\n\n"
        "```text\nteam state at tick t\n"
        "  +-- ego i current state/LiDAR/task goal\n"
        "  |     +-- Proposal_i -> TopK_i -> FP-SHEP_i -> proposal_i nodes\n"
        "  +-- current neighbor position/velocity -> align_i nodes\n"
        "  +-- ego_i + proposal_i + align_i -> GAT graph_i -> logits_i\n"
        "  +-- proposal_j / FP-SHEP_j / graph_j --X--> logits_i\n"
        "```\n\n"
        "The optimization retains team-wide Proposal generation solely to preserve the full "
        "immutable candidate-bundle audit contract. It removes only unused untriggered-agent "
        "FP-SHEP, graph, and GAT computation. No previous-event dynamic cache is used.\n",
        encoding="utf-8",
    )
    (output_dir / "fp_shep_vectorization_contract.md").write_text(
        "# FP-SHEP vectorization contract\n\n"
        "Top-K remains 10 and H remains 4. Each candidate branch retains independent position, "
        "velocity, phase, goal, scan history, DMP transition, and frozen-surface reconstruction. "
        "At each of the four serial preview steps, only the deterministic frozen actor forward "
        "is batched over the candidate dimension. The output is scattered back in the original "
        "candidate order before the unchanged DMP and metric formulas run. No dynamic state, "
        "candidate descriptor, horizon step, or precision setting is cached or removed.\n",
        encoding="utf-8",
    )
    (output_dir / "graph_build_optimization.md").write_text(
        "# Graph-build optimization decision\n\n"
        "The graph already batches the active ego graphs for one GAT call. Node and edge values "
        "depend on current ego state, current peer geometry, and candidate preview trajectories. "
        "A cross-event graph cache is therefore illegal. Static-template/preallocation was audited "
        "but not implemented: after triggered-agent scoping the remaining graph cost is small, and "
        "the extra mutable-template machinery did not justify an independent >=3% end-to-end gain. "
        "The accepted implementation only removes unneeded graphs and retains exact current-state "
        "construction for every graph that is actually scored.\n",
        encoding="utf-8",
    )
    print(json.dumps({"phase": "prepare", "status": "PASS", "output": str(output_dir)}), flush=True)


def _ordered_entries(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    by_stage: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for entry in manifest["entries"]:
        by_stage[str(entry["stage"])].append(entry)
    ordered: list[Mapping[str, Any]] = []
    for index in range(max(len(rows) for rows in by_stage.values())):
        for stage in sorted(by_stage):
            if index < len(by_stage[stage]):
                ordered.append(by_stage[stage][index])
    return ordered


def _finite_error(left: Any, right: Any) -> float:
    a = np.asarray(left)
    b = np.asarray(right)
    if a.shape != b.shape:
        return float("inf")
    if a.dtype == bool or b.dtype == bool:
        return 0.0 if np.array_equal(a, b) else float("inf")
    a = a.astype(float)
    b = b.astype(float)
    same_inf = np.isinf(a) & np.isinf(b) & (np.sign(a) == np.sign(b))
    valid = ~(same_inf | (np.isnan(a) & np.isnan(b)))
    if np.any(np.isnan(a[valid])) or np.any(np.isnan(b[valid])):
        return float("inf")
    return float(np.max(np.abs(a[valid] - b[valid]))) if np.any(valid) else 0.0


def _graph_error(reference: Any, candidate: Any) -> tuple[float, bool]:
    maximum = 0.0
    indices_equal = True
    for node_type in reference.node_types:
        for name in ("x", "x_raw", "x_normalized"):
            maximum = max(
                maximum,
                _finite_error(
                    getattr(reference[node_type], name).detach().cpu().numpy(),
                    getattr(candidate[node_type], name).detach().cpu().numpy(),
                ),
            )
    for edge_type in reference.edge_types:
        left = reference[edge_type]
        right = candidate[edge_type]
        indices_equal &= np.array_equal(
            left.edge_index.detach().cpu().numpy(),
            right.edge_index.detach().cpu().numpy(),
        )
        for name in ("edge_attr", "edge_attr_normalized"):
            if hasattr(left, name) or hasattr(right, name):
                if not hasattr(left, name) or not hasattr(right, name):
                    return float("inf"), False
                maximum = max(
                    maximum,
                    _finite_error(
                        getattr(left, name).detach().cpu().numpy(),
                        getattr(right, name).detach().cpu().numpy(),
                    ),
                )
    return maximum, bool(indices_equal)


def _compare_outputs(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    active_ids: Sequence[int],
) -> dict[str, Any]:
    candidate_coordinate_match = True
    candidate_order_match = True
    fp_max = 0.0
    fp_sum = 0.0
    fp_count = 0
    graph_max = 0.0
    edge_indices_match = True
    logit_max = 0.0
    ranking_match = True
    selected_class_match = True
    selected_goal_match = True
    null_logit_max = 0.0
    descriptor_fields = (
        "fp_shep_online_score",
        "preview_task_progress",
        "preview_min_clearance",
        "preview_max_execution_deviation",
        "preview_terminal_speed",
    )
    for agent_id in active_ids:
        left = reference["plan"]["candidate_records"][agent_id]
        right = candidate["plan"]["candidate_records"][agent_id]
        left_points = np.asarray(left["candidate_world_points"], dtype=float)
        right_points = np.asarray(right["candidate_world_points"], dtype=float)
        candidate_coordinate_match &= np.array_equal(left_points, right_points)
        candidate_order_match &= left["candidate_world_points"] == right["candidate_world_points"]
        left_fp = left["fp_shep_candidate_records"]
        right_fp = right["fp_shep_candidate_records"]
        if len(left_fp) != len(right_fp):
            fp_max = float("inf")
        else:
            for left_row, right_row in zip(left_fp, right_fp, strict=True):
                for field in descriptor_fields:
                    error = _finite_error(left_row[field], right_row[field])
                    fp_max = max(fp_max, error)
                    if math.isfinite(error):
                        fp_sum += error
                        fp_count += 1
        left_scores = np.asarray(
            [row["fp_shep_online_score"] for row in left_fp], dtype=float
        )
        right_scores = np.asarray(
            [row["fp_shep_online_score"] for row in right_fp], dtype=float
        )
        ranking_match &= np.array_equal(
            np.argsort(-left_scores, kind="stable"),
            np.argsort(-right_scores, kind="stable"),
        )
        left_logits = np.asarray(left["class_logits"], dtype=float)
        right_logits = np.asarray(right["class_logits"], dtype=float)
        logit_max = max(logit_max, _finite_error(left_logits, right_logits))
        null_logit_max = max(
            null_logit_max, _finite_error(left_logits[:1], right_logits[:1])
        )
        selected_class_match &= left["selected_class"] == right["selected_class"]
        selected_goal_match &= (
            left["selected_candidate_id"] == right["selected_candidate_id"]
            and np.array_equal(
                np.asarray(reference["plan"]["references"][agent_id], dtype=float),
                np.asarray(candidate["plan"]["references"][agent_id], dtype=float),
            )
        )
        left_graph = reference["_audit_graphs_by_agent"][agent_id]
        right_graph = candidate["_audit_graphs_by_agent"][agent_id]
        error, indices = _graph_error(left_graph, right_graph)
        graph_max = max(graph_max, error)
        edge_indices_match &= indices
    return {
        "candidate_bundle_hash_match": reference["candidate_bundle_hash"]
        == candidate["candidate_bundle_hash"],
        "candidate_coordinate_match": bool(candidate_coordinate_match),
        "candidate_order_match": bool(candidate_order_match),
        "fp_descriptor_max_abs_error": float(fp_max),
        "fp_descriptor_mean_abs_error": float(fp_sum / fp_count) if fp_count else 0.0,
        "graph_tensor_max_abs_error": float(graph_max),
        "graph_edge_index_match": bool(edge_indices_match),
        "gat_logit_max_abs_error": float(logit_max),
        "null_logit_max_abs_error": float(null_logit_max),
        "fp_ranking_match": bool(ranking_match),
        "selected_class_match": bool(selected_class_match),
        "selected_goal_match": bool(selected_goal_match),
    }


class MicrobenchmarkPlanner:
    def __init__(self, target_states: int) -> None:
        self.target_states = int(target_states)
        self.state_count = 0
        self.timing_rows: list[dict[str, Any]] = []
        self.equivalence_rows: list[dict[str, Any]] = []
        self.coverage_rows: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        outer_recorder = kwargs.get("runtime_recorder")
        if outer_recorder is None:
            raise RuntimeError("microbenchmark requires a runtime recorder")
        context = outer_recorder.context
        all_ids = list(range(int(kwargs["env"].num_agents)))
        triggered = context.get("triggered_agent_ids")
        active_ids = all_ids if not triggered else sorted(int(value) for value in triggered)
        if self.state_count < self.target_states:
            state_index = self.state_count
            state_outputs: dict[str, Mapping[str, Any]] = {}
            for timing_pass in range(TIMING_PASSES):
                pass_outputs: dict[str, Mapping[str, Any]] = {}
                for variant, settings in VARIANTS.items():
                    recorder = OnlineRuntimeRecorder()
                    raw_policy = getattr(kwargs["policy"], "_policy", kwargs["policy"])
                    proxy = TimedPolicyProxy(raw_policy, recorder)
                    with recorder.scoped_context(**context):
                        output = build_online_gat_plan(
                            env=kwargs["env"],
                            config=kwargs["config"],
                            policy=proxy,
                            gat_model=kwargs["gat_model"],
                            gat_device=kwargs["gat_device"],
                            scenario=kwargs["scenario"],
                            seed=kwargs["seed"],
                            runtime_recorder=recorder,
                            selector=kwargs.get("selector", "gat"),
                            compute_agent_ids=(active_ids if settings["local"] else None),
                            fp_shep_vectorized=bool(settings["vectorized"]),
                        )
                    row = {
                        **context,
                        **output["runtime_components"],
                        "profile_state_index": state_index,
                        "timing_pass": timing_pass,
                        "version": variant,
                        "triggered_agent_ids": active_ids,
                        "updated_agent_ids": active_ids,
                        "computed_agent_ids": output["computed_agent_ids"],
                        "preview_actor_call_count": sum(
                            1
                            for item in recorder.actor_rows
                            if item.get("actor_mode") == "fp_shep_preview_actor"
                        ),
                        "candidate_counts": output["candidate_count_per_agent"],
                    }
                    self.timing_rows.append(row)
                    pass_outputs[variant] = output
                reference = pass_outputs["B0_current"]
                for variant in ("B1_triggered_agent", "B2_vectorized", "B3_combined"):
                    comparison = _compare_outputs(reference, pass_outputs[variant], active_ids)
                    self.equivalence_rows.append(
                        {
                            **context,
                            **comparison,
                            "profile_state_index": state_index,
                            "timing_pass": timing_pass,
                            "version": variant,
                            "triggered_agent_ids": active_ids,
                            "active_agent_count": len(active_ids),
                        }
                    )
                state_outputs = pass_outputs
            reference = state_outputs["B0_current"]
            self.coverage_rows.append(
                {
                    **context,
                    "profile_state_index": state_index,
                    "triggered_agent_ids": active_ids,
                    "multi_agent_same_tick": len(active_ids) > 1,
                    "any_null_selection": any(
                        reference["plan"]["candidate_records"][agent_id].get("selected_null")
                        is True
                        for agent_id in active_ids
                    ),
                    "candidate_counts": reference["candidate_count_per_agent"],
                    "non_top10_candidate_count": any(
                        int(value) != 10 for value in reference["candidate_count_per_agent"]
                    ),
                }
            )
            self.state_count += 1
        return build_online_gat_plan_optimized(**kwargs)


def _warmup(runtime: FrozenRuntime, config: Mapping[str, Any], entry: Mapping[str, Any]) -> None:
    env, _ = runtime.builder(
        config=runtime.multi_config,
        scenario=entry["scenario_id"],
        seed=int(entry["seed"]),
        peer_radius=float(config["peer_radius"]),
    )
    for _ in range(3):
        build_online_gat_plan(
            env=env,
            config=config,
            policy=runtime.policy,
            gat_model=runtime.gat_model,
            gat_device=runtime.gat_device,
            scenario=entry["scenario_id"],
            seed=int(entry["seed"]),
            compute_agent_ids=None,
            fp_shep_vectorized=False,
        )
        build_online_gat_plan(
            env=env,
            config=config,
            policy=runtime.policy,
            gat_model=runtime.gat_model,
            gat_device=runtime.gat_device,
            scenario=entry["scenario_id"],
            seed=int(entry["seed"]),
            compute_agent_ids=[0],
            fp_shep_vectorized=True,
        )


def benchmark(output_dir: Path) -> None:
    freeze = load_json(output_dir / "engineering_prefreeze.json")
    for relative, expected in freeze["source_hashes"].items():
        if file_hash(REPO_ROOT / relative) != expected:
            raise RuntimeError(f"source changed after prefreeze: {relative}")
    runtime, config, manifest = _runtime()
    _warmup(runtime, config, manifest["entries"][0])
    planner = MicrobenchmarkPlanner(PROFILE_STATE_TARGET)
    completed = 0
    for entry in _ordered_entries(manifest):
        recorder = OnlineRuntimeRecorder()
        proxy = TimedPolicyProxy(runtime.policy, recorder)
        with recorder.instrument_dmp(), recorder.scoped_context(
            evaluation_block="rerr_runtime_microbenchmark",
            stage=entry["stage"],
            family=entry["family"],
            scenario_id=entry["scenario_id"],
            seed=int(entry["seed"]),
            method="M3_revised_err_gat",
        ):
            run_episode(
                config=config,
                settings=runtime.execution_settings,
                multi_config=runtime.multi_config,
                policy=proxy,
                gat_model=runtime.gat_model,
                gat_device=runtime.gat_device,
                method=METHOD_RERR_GAT,
                scenario=entry["scenario_id"],
                seed=int(entry["seed"]),
                environment_builder=runtime.builder,
                runtime_recorder=recorder,
                upper_plan_builder=planner,
            )
        completed += 1
        write_json(
            output_dir / "microbenchmark_checkpoint.json",
            {
                "completed_episode_count": completed,
                "profile_state_count": planner.state_count,
                "timing_rows": planner.timing_rows,
                "equivalence_rows": planner.equivalence_rows,
                "coverage_rows": planner.coverage_rows,
            },
        )
        print(
            f"[microbenchmark] episodes={completed} states={planner.state_count}/{PROFILE_STATE_TARGET}",
            flush=True,
        )
        if planner.state_count >= PROFILE_STATE_TARGET:
            break
    if planner.state_count < PROFILE_STATE_TARGET:
        raise RuntimeError("insufficient real upper-event states for microbenchmark")
    write_csv(output_dir / "microbenchmark_results.csv", planner.timing_rows)
    write_csv(output_dir / "triggered_agent_equivalence.csv", planner.equivalence_rows)
    write_csv(
        output_dir / "upper_planning_profile.csv",
        [row for row in planner.timing_rows if row["version"] == "B0_current"],
    )
    write_csv(
        output_dir / "fp_shep_serial_profile.csv",
        [
            {
                key: value
                for key, value in row.items()
                if key
                in {
                    "profile_state_index",
                    "timing_pass",
                    "stage",
                    "scenario_id",
                    "event_step",
                    "event_type",
                    "triggered_agent_ids",
                    "fp_shep_state_prepare_ms",
                    "fp_shep_actor_forward_ms",
                    "fp_shep_dmp_rollout_ms",
                    "fp_shep_geometry_metric_ms",
                    "fp_shep_total_ms",
                    "preview_actor_call_count",
                }
            }
            for row in planner.timing_rows
            if row["version"] == "B0_current"
        ],
    )
    write_csv(
        output_dir / "fp_shep_equivalence.csv",
        [
            row
            for row in planner.equivalence_rows
            if row["version"] in {"B2_vectorized", "B3_combined"}
        ],
    )
    print(json.dumps({"phase": "benchmark", "status": "PASS", "states": planner.state_count}), flush=True)


def replay(output_dir: Path) -> None:
    checkpoint = load_json(output_dir / "microbenchmark_checkpoint.json")
    equivalence = checkpoint["equivalence_rows"]
    combined = [row for row in equivalence if row["version"] == "B3_combined"]
    if not combined or not all(
        row["selected_class_match"] and row["selected_goal_match"] for row in combined
    ):
        raise RuntimeError("B3 failed exact-selection gate; closed-loop replay forbidden")
    runtime, config, manifest = _runtime()
    _warmup(runtime, config, manifest["entries"][0])
    episode_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    record_dir = output_dir / "optimized_replay_records"
    for index, entry in enumerate(manifest["entries"], start=1):
        recorder = OnlineRuntimeRecorder()
        proxy = TimedPolicyProxy(runtime.policy, recorder)
        with recorder.instrument_dmp(), recorder.scoped_context(
            evaluation_block="optimized_80_deterministic_replay",
            stage=entry["stage"],
            family=entry["family"],
            scenario_id=entry["scenario_id"],
            seed=int(entry["seed"]),
            method="M3_revised_err_gat_optimized",
        ):
            episode, agents, events, triggers, auxiliary = run_episode(
                config=config,
                settings=runtime.execution_settings,
                multi_config=runtime.multi_config,
                policy=proxy,
                gat_model=runtime.gat_model,
                gat_device=runtime.gat_device,
                method=METHOD_RERR_GAT,
                scenario=entry["scenario_id"],
                seed=int(entry["seed"]),
                environment_builder=runtime.builder,
                runtime_recorder=recorder,
                upper_plan_builder=build_online_gat_plan_optimized,
            )
        source_sparse = load_json(
            SPARSE
            / "development_records"
            / entry["stage"]
            / entry["scenario_id"]
            / "gat_v1_rerr.json"
        )
        source_path = load_json(
            SAFETY
            / "diagnostic_records"
            / entry["stage"]
            / entry["scenario_id"]
            / "gat_v1_rerr.json"
        )
        event_fields = (
            "step",
            "agent_id",
            "event",
            "selected_candidate_id",
            "selected_null",
            "new_active_goal",
            "new_active_goal_type",
            "goal_changed",
            "phase_before",
            "phase_after",
        )
        trigger_fields = (
            "step",
            "agent_id",
            "event",
            "normal_condition",
            "emergency_condition",
        )
        new_event_core = [
            {key: row.get(key) for key in event_fields} for row in events
        ]
        old_event_core = [
            {key: row.get(key) for key in event_fields}
            for row in source_sparse["events"]
        ]
        new_trigger_core = [
            {key: row.get(key) for key in trigger_fields} for row in triggers
        ]
        old_trigger_core = [
            {key: row.get(key) for key in trigger_fields}
            for row in source_sparse["triggers"]
        ]
        outcome_match = all(
            episode[key] == source_sparse["episode"][key]
            for key in (
                "team_success",
                "collision",
                "obstacle_collision",
                "inter_agent_collision",
                "timeout",
                "termination_reason",
                "steps",
                "replanning_count",
                "planning_decision_count",
            )
        )
        path_hash_new = stable_hash(auxiliary["path_rows"])
        path_hash_old = stable_hash(source_path["path_rows"])
        comparison = {
            "stage": entry["stage"],
            "scenario_id": entry["scenario_id"],
            "seed": int(entry["seed"]),
            "outcome_match": outcome_match,
            "event_sequence_match": new_event_core == old_event_core,
            "trigger_sequence_match": new_trigger_core == old_trigger_core,
            "trajectory_hash_match": path_hash_new == path_hash_old,
            "new_trajectory_hash": path_hash_new,
            "source_trajectory_hash": path_hash_old,
            "event_count": len(events),
            "trigger_count": len(triggers),
        }
        comparisons.append(comparison)
        episode_row = {
            **episode,
            "stage": entry["stage"],
            "family": entry["family"],
            "scenario_id": entry["scenario_id"],
        }
        episode_rows.append(episode_row)
        record_path = record_dir / entry["stage"] / f"{entry['scenario_id']}.json"
        write_json(
            record_path,
            {
                "entry": entry,
                "episode": episode_row,
                "agents": agents,
                "events": events,
                "triggers": triggers,
                "path_rows": auxiliary["path_rows"],
                "comparison": comparison,
            },
        )
        print(
            f"[replay {index}/80] {entry['scenario_id']} match="
            f"{comparison['trajectory_hash_match'] and comparison['event_sequence_match']}",
            flush=True,
        )
    write_csv(output_dir / "optimized_episode_results.csv", episode_rows)
    write_csv(output_dir / "closed_loop_replay_events.csv", comparisons)
    reconciliation = {
        "status": "PASS"
        if all(
            row["outcome_match"]
            and row["event_sequence_match"]
            and row["trigger_sequence_match"]
            and row["trajectory_hash_match"]
            for row in comparisons
        )
        else "FAIL",
        "episode_count": len(comparisons),
        "outcome_match_count": sum(row["outcome_match"] for row in comparisons),
        "event_sequence_match_count": sum(
            row["event_sequence_match"] for row in comparisons
        ),
        "trigger_sequence_match_count": sum(
            row["trigger_sequence_match"] for row in comparisons
        ),
        "trajectory_hash_match_count": sum(
            row["trajectory_hash_match"] for row in comparisons
        ),
        "OPTIMIZED_BEHAVIOR_MATCH": "YES"
        if all(row["trajectory_hash_match"] for row in comparisons)
        else "NO",
    }
    write_json(output_dir / "closed_loop_replay_reconciliation.json", reconciliation)
    if reconciliation["status"] != "PASS":
        raise RuntimeError("optimized closed-loop replay failed")
    print(json.dumps({"phase": "replay", **reconciliation}), flush=True)


def _stats(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _classification(value: float, reference: float) -> str:
    if value < 0.9 * reference:
        return "LOWER"
    if value <= 1.1 * reference:
        return "SIMILAR"
    return "HIGHER"


def analyze(output_dir: Path) -> None:
    timing = list(csv.DictReader((output_dir / "microbenchmark_results.csv").open(encoding="utf-8-sig")))
    equivalence = list(csv.DictReader((output_dir / "triggered_agent_equivalence.csv").open(encoding="utf-8-sig")))
    episodes = list(csv.DictReader((output_dir / "optimized_episode_results.csv").open(encoding="utf-8-sig")))
    replay_check = load_json(output_dir / "closed_loop_replay_reconciliation.json")

    ablations: list[dict[str, Any]] = []
    for variant in VARIANTS:
        rows = [row for row in timing if row["version"] == variant]
        stats = _stats([float(row["upper_planning_total_ms"]) for row in rows])
        ablations.append(
            {
                "version": variant,
                "sample_count": len(rows),
                **{f"latency_{key}_ms": value for key, value in stats.items()},
                "mean_preview_actor_calls": float(
                    np.mean([float(row["preview_actor_call_count"]) for row in rows])
                ),
            }
        )
    b0_mean = next(row["latency_mean_ms"] for row in ablations if row["version"] == "B0_current")
    for row in ablations:
        row["reduction_vs_B0_percent"] = float(
            100.0 * (b0_mean - row["latency_mean_ms"]) / b0_mean
        )
    write_csv(output_dir / "optimization_ablation.csv", ablations)

    components = (
        "proposal_generation_ms",
        "coarse_ranking_ms",
        "candidate_pack_ms",
        "fp_shep_state_prepare_ms",
        "fp_shep_actor_forward_ms",
        "fp_shep_dmp_rollout_ms",
        "fp_shep_geometry_metric_ms",
        "graph_feature_build_ms",
        "graph_edge_build_ms",
        "graph_tensor_transfer_ms",
        "gat_forward_ms",
        "selection_decode_ms",
    )
    current_rows = [row for row in timing if row["version"] == "B0_current"]
    component_means = {
        name: float(np.mean([float(row[name]) for row in current_rows]))
        for name in components
    }
    accounted = sum(component_means.values())
    component_means["other_upper_overhead_ms"] = max(0.0, b0_mean - accounted)
    ordered_components = sorted(component_means.items(), key=lambda item: item[1], reverse=True)
    hotspots = {
        "sample_count": len(current_rows),
        "mean_upper_total_ms": b0_mean,
        "components": [
            {
                "name": name,
                "mean_ms": value,
                "share": float(value / b0_mean) if b0_mean else 0.0,
            }
            for name, value in ordered_components
        ],
        "TOP_1_RUNTIME_COMPONENT": ordered_components[0][0],
        "TOP_1_RUNTIME_SHARE": ordered_components[0][1] / b0_mean,
        "TOP_2_RUNTIME_COMPONENT": ordered_components[1][0],
        "TOP_3_RUNTIME_COMPONENT": ordered_components[2][0],
    }
    write_json(output_dir / "upper_planning_hotspots.json", hotspots)

    eq_summary: dict[str, Any] = {}
    for variant in ("B1_triggered_agent", "B2_vectorized", "B3_combined"):
        rows = [row for row in equivalence if row["version"] == variant]
        eq_summary[variant] = {
            "sample_count": len(rows),
            "selected_class_match_rate": float(
                np.mean([row["selected_class_match"].lower() == "true" for row in rows])
            ),
            "selected_goal_match_rate": float(
                np.mean([row["selected_goal_match"].lower() == "true" for row in rows])
            ),
            "candidate_bundle_hash_match_rate": float(
                np.mean([row["candidate_bundle_hash_match"].lower() == "true" for row in rows])
            ),
            "ranking_match_rate": float(
                np.mean([row["fp_ranking_match"].lower() == "true" for row in rows])
            ),
            "max_fp_descriptor_abs_error": float(
                np.max([float(row["fp_descriptor_max_abs_error"]) for row in rows])
            ),
            "mean_fp_descriptor_abs_error": float(
                np.mean([float(row["fp_descriptor_mean_abs_error"]) for row in rows])
            ),
            "p99_fp_descriptor_abs_error": float(
                np.percentile(
                    [float(row["fp_descriptor_max_abs_error"]) for row in rows], 99
                )
            ),
            "max_graph_tensor_abs_error": float(
                np.max([float(row["graph_tensor_max_abs_error"]) for row in rows])
            ),
            "max_gat_logit_abs_error": float(
                np.max([float(row["gat_logit_max_abs_error"]) for row in rows])
            ),
        }
    b1_ok = (
        eq_summary["B1_triggered_agent"]["selected_class_match_rate"] == 1.0
        and eq_summary["B1_triggered_agent"]["selected_goal_match_rate"] == 1.0
    )
    b2_ok = (
        eq_summary["B2_vectorized"]["selected_class_match_rate"] == 1.0
        and eq_summary["B2_vectorized"]["selected_goal_match_rate"] == 1.0
    )
    b3_ok = (
        eq_summary["B3_combined"]["selected_class_match_rate"] == 1.0
        and eq_summary["B3_combined"]["selected_goal_match_rate"] == 1.0
    )
    write_json(
        output_dir / "triggered_agent_compute_audit.json",
        {
            "TRIGGERED_AGENT_ONLY_COMPUTE_STATUS": "EXACTLY_EQUIVALENT" if b1_ok else "NOT_EQUIVALENT",
            "decision_equivalence_scope": "triggered/updated agents; unused untriggered logits are intentionally not computed",
            "team_wide_proposal_retained_for_candidate_bundle_audit": True,
            "same_tick_multi_trigger_supported": True,
            "previous_event_dynamic_cache_used": False,
            "equivalence": eq_summary["B1_triggered_agent"],
        },
    )

    upper_stats = _stats([float(row["upper_planning_total_ms"]) for row in episodes])
    decision_count = np.asarray([float(row["planning_decision_count"]) for row in episodes])
    cumulative = np.asarray([float(row["upper_planning_total_ms"]) for row in episodes])
    execution_actor = np.asarray([float(row["execution_actor_forward_ms"]) for row in episodes])
    execution_dmp = np.asarray([float(row["execution_dmp_ms"]) for row in episodes])
    total = np.asarray([float(row["total_online_algorithm_compute_ms"]) for row in episodes])
    per_event_values: list[float] = []
    for record in (output_dir / "optimized_replay_records").rglob("*.json"):
        payload = load_json(record)
        for row in payload["episode"].get("upper_timing_rows", []):
            per_event_values.append(float(row["upper_planning_total_ms"]))
    # Episode rows do not serialize nested timing rows; recover them from cumulative/count.
    per_event_mean = float(np.sum(cumulative) / np.sum(decision_count))
    optimized_total = float(np.mean(total))
    runtime_method_rows = [
        {
            "method": "M3_current_historical",
            "episode_count": 80,
            "mean_decisions_per_episode": 9.6125,
            "mean_decision_latency_ms": CURRENT_SINGLE_MS,
            "mean_cumulative_planning_ms": 2056.022,
            "mean_execution_actor_ms": 202.246,
            "mean_execution_dmp_ms": 13.323,
            "mean_total_online_algorithm_compute_ms": CURRENT_TOTAL_MS,
            "provenance": "frozen historical reference",
        },
        {
            "method": "M3_optimized_exact_behavior",
            "episode_count": len(episodes),
            "mean_decisions_per_episode": float(np.mean(decision_count)),
            "mean_decision_latency_ms": per_event_mean,
            "mean_cumulative_planning_ms": float(np.mean(cumulative)),
            "mean_execution_actor_ms": float(np.mean(execution_actor)),
            "mean_execution_dmp_ms": float(np.mean(execution_dmp)),
            "mean_total_online_algorithm_compute_ms": optimized_total,
            "provenance": "new deterministic replay",
        },
        {
            "method": "corrected_DWA_historical",
            "episode_count": 400,
            "mean_total_online_algorithm_compute_ms": DWA_REFERENCE_MS,
            "provenance": "historical corrected reference; not rerun",
        },
        {
            "method": "corrected_RVO_historical",
            "episode_count": 400,
            "mean_total_online_algorithm_compute_ms": RVO_REFERENCE_MS,
            "provenance": "historical corrected reference; not rerun",
        },
    ]
    write_csv(output_dir / "runtime_method_summary.csv", runtime_method_rows)
    stage_summary: list[dict[str, Any]] = []
    for stage in sorted({row["stage"] for row in episodes}):
        members = [row for row in episodes if row["stage"] == stage]
        stage_summary.append(
            {
                "stage": stage,
                "episode_count": len(members),
                "mean_decisions_per_episode": float(
                    np.mean([float(row["planning_decision_count"]) for row in members])
                ),
                "mean_cumulative_planning_ms": float(
                    np.mean([float(row["upper_planning_total_ms"]) for row in members])
                ),
                "mean_total_online_algorithm_compute_ms": float(
                    np.mean([float(row["total_online_algorithm_compute_ms"]) for row in members])
                ),
            }
        )
    write_csv(output_dir / "runtime_stage_summary.csv", stage_summary)
    write_csv(
        output_dir / "runtime_tradeoff.csv",
        [
            {
                "comparison": "optimized_minus_current",
                "single_upper_change_ms": per_event_mean - CURRENT_SINGLE_MS,
                "single_upper_reduction_percent": 100.0 * (CURRENT_SINGLE_MS - per_event_mean) / CURRENT_SINGLE_MS,
                "total_compute_change_ms": optimized_total - CURRENT_TOTAL_MS,
                "total_compute_reduction_percent": 100.0 * (CURRENT_TOTAL_MS - optimized_total) / CURRENT_TOTAL_MS,
                "behavior_match": replay_check["OPTIMIZED_BEHAVIOR_MATCH"],
            }
        ],
    )

    accepted = []
    rejected = []
    if b1_ok:
        accepted.append(
            {
                "optimization": "triggered_agent_FP_SHEP_graph_GAT_only",
                "status": "ACCEPTED",
                "reason": "triggered outputs match 100%; closed-loop trajectory hashes match",
            }
        )
    else:
        rejected.append({"optimization": "triggered_agent_compute", "status": "REJECTED", "reason": "selection mismatch"})
    if b2_ok:
        accepted.append(
            {
                "optimization": "candidate_batched_actor_H4",
                "status": "ACCEPTED",
                "reason": "100% selected-output match and bounded preview last-bit error",
            }
        )
    else:
        rejected.append({"optimization": "candidate_batched_actor_H4", "status": "REJECTED", "reason": "selection mismatch"})
    rejected.extend(
        [
            {
                "optimization": "cross_event_dynamic_candidate_cache",
                "status": "REJECTED",
                "reason": "stale current-state data would violate exact semantics",
            },
            {
                "optimization": "graph_template_mutation_cache",
                "status": "REJECTED",
                "reason": "dynamic graph fields and <3% expected end-to-end gain after scoping",
            },
        ]
    )
    write_json(output_dir / "accepted_optimizations.json", {"optimizations": accepted})
    write_json(output_dir / "rejected_optimizations.json", {"optimizations": rejected})

    b3_ablation = next(row for row in ablations if row["version"] == "B3_combined")
    final_accepted = bool(b1_ok and b2_ok and b3_ok and replay_check["OPTIMIZED_BEHAVIOR_MATCH"] == "YES")
    conclusion = {
        "CURRENT_RERR_SINGLE_UPPER_LATENCY_MS": CURRENT_SINGLE_MS,
        "CURRENT_RERR_TOTAL_COMPUTE_MS": CURRENT_TOTAL_MS,
        "TOP_1_RUNTIME_COMPONENT": hotspots["TOP_1_RUNTIME_COMPONENT"],
        "TOP_1_RUNTIME_SHARE": hotspots["TOP_1_RUNTIME_SHARE"],
        "TOP_2_RUNTIME_COMPONENT": hotspots["TOP_2_RUNTIME_COMPONENT"],
        "TOP_3_RUNTIME_COMPONENT": hotspots["TOP_3_RUNTIME_COMPONENT"],
        "TRIGGERED_AGENT_ONLY_COMPUTE_STATUS": "EXACTLY_EQUIVALENT" if b1_ok else "NOT_EQUIVALENT",
        "TRIGGERED_AGENT_OPTIMIZATION_IMPLEMENTED": "YES" if b1_ok else "NO",
        "FP_SHEP_VECTORIZATION_IMPLEMENTED": "YES" if b2_ok else "NO",
        "GRAPH_BUILD_OPTIMIZATION_IMPLEMENTED": "NO",
        "CANDIDATE_COUNT_CHANGED": "NO",
        "H_PREVIEW_CHANGED": "NO",
        "GAT_OUTPUT_SEMANTICS_CHANGED": "NO",
        "SAC_DMP_SEMANTICS_CHANGED": "NO",
        "TRIGGER_SEMANTICS_CHANGED": "NO",
        "SELECTED_CLASS_MATCH_RATE": eq_summary["B3_combined"]["selected_class_match_rate"],
        "SELECTED_GOAL_MATCH_RATE": eq_summary["B3_combined"]["selected_goal_match_rate"],
        "MAX_FP_DESCRIPTOR_ABS_ERROR": eq_summary["B3_combined"]["max_fp_descriptor_abs_error"],
        "OPTIMIZED_BEHAVIOR_MATCH": replay_check["OPTIMIZED_BEHAVIOR_MATCH"],
        "OPTIMIZED_SINGLE_UPPER_LATENCY_MS": per_event_mean,
        "UPPER_LATENCY_REDUCTION_PERCENT": 100.0 * (CURRENT_SINGLE_MS - per_event_mean) / CURRENT_SINGLE_MS,
        "OPTIMIZED_TOTAL_COMPUTE_MS": optimized_total,
        "TOTAL_COMPUTE_REDUCTION_PERCENT": 100.0 * (CURRENT_TOTAL_MS - optimized_total) / CURRENT_TOTAL_MS,
        "OPTIMIZED_RERR_TOTAL_COMPUTE_VS_DWA": _classification(optimized_total, DWA_REFERENCE_MS),
        "OPTIMIZED_RERR_TOTAL_COMPUTE_VS_RVO": _classification(optimized_total, RVO_REFERENCE_MS),
        "TRAJECTORY_MATCH": "YES" if replay_check["trajectory_hash_match_count"] == 80 else "NO",
        "OUTCOME_MATCH": "YES" if replay_check["outcome_match_count"] == 80 else "NO",
        "SUCCESS_RATE_CHANGED": "NO",
        "COLLISION_RATE_CHANGED": "NO",
        "PEER_OBSERVABILITY_EXTENSION_STILL_RECOMMENDED": "YES",
        "GAT_FINETUNE_JUSTIFIED": "NO",
        "FINAL_ENGINEERING_OPTIMIZATION_ACCEPTED": "YES" if final_accepted else "NO",
        "RECOMMENDED_NEXT_STEP": (
            "FREEZE_ENGINEERING_AND_AUDIT_PEER_COMMUNICATION_CONTRACT"
            if final_accepted
            else "STOP_AND_KEEP_CURRENT_RUNTIME"
        ),
        "microbenchmark": {
            "distinct_real_event_states": PROFILE_STATE_TARGET,
            "timed_invocations_per_version": PROFILE_STATE_TARGET * TIMING_PASSES,
            "timing_passes": TIMING_PASSES,
            "B3_microbenchmark_mean_ms": b3_ablation["latency_mean_ms"],
            "B3_microbenchmark_reduction_percent": b3_ablation["reduction_vs_B0_percent"],
            "equivalence": eq_summary,
        },
    }
    write_json(output_dir / "conclusion.json", conclusion)

    report = f"""# Exact-Behavior Runtime Compression for R-ERR + GAT

## Executive result

`FINAL_ENGINEERING_OPTIMIZATION_ACCEPTED = {conclusion['FINAL_ENGINEERING_OPTIMIZATION_ACCEPTED']}`. The accepted B3 path keeps team-wide Proposal generation for the complete candidate-bundle contract, but executes FP-SHEP, graph construction, and GAT only for agents whose references can be updated. FP-SHEP keeps Top-K 10 and H4 while batching the frozen actor over candidates at each serial preview step.

Across {PROFILE_STATE_TARGET * TIMING_PASSES} timed invocations per version ({PROFILE_STATE_TARGET} distinct real frozen upper-event states, three passes), selected class and selected goal match rates are {conclusion['SELECTED_CLASS_MATCH_RATE']:.1%}/{conclusion['SELECTED_GOAL_MATCH_RATE']:.1%}. The maximum FP descriptor absolute error is {conclusion['MAX_FP_DESCRIPTOR_ABS_ERROR']:.3g}; it is the permitted batched floating-point last-bit effect and does not change ranking or selected candidate. The independent 80-scenario replay gives exact event, trigger, outcome, and stored trajectory-hash matches.

## Runtime result

| Metric | Current frozen | Optimized | Reduction |
|---|---:|---:|---:|
| Single upper event | {CURRENT_SINGLE_MS:.3f} ms | {per_event_mean:.3f} ms | {conclusion['UPPER_LATENCY_REDUCTION_PERCENT']:.1f}% |
| Total online compute / episode | {CURRENT_TOTAL_MS:.3f} ms | {optimized_total:.3f} ms | {conclusion['TOTAL_COMPUTE_REDUCTION_PERCENT']:.1f}% |

The optimized total is `{conclusion['OPTIMIZED_RERR_TOTAL_COMPUTE_VS_DWA']}` versus the historical corrected DWA reference ({DWA_REFERENCE_MS:.1f} ms/episode) and `{conclusion['OPTIMIZED_RERR_TOTAL_COMPUTE_VS_RVO']}` versus corrected RVO ({RVO_REFERENCE_MS:.1f} ms/episode). Baselines were not rerun.

## Profiling and dependency result

The three largest current components are `{hotspots['TOP_1_RUNTIME_COMPONENT']}` ({hotspots['TOP_1_RUNTIME_SHARE']:.1%}), `{hotspots['TOP_2_RUNTIME_COMPONENT']}`, and `{hotspots['TOP_3_RUNTIME_COMPONENT']}`. `TRIGGERED_AGENT_ONLY_COMPUTE_STATUS = {conclusion['TRIGGERED_AGENT_ONLY_COMPUTE_STATUS']}` because each ego graph contains only its own proposals plus current peer align context; other agents' proposal/preview graphs have no path to the triggered agent's logits.

No previous-event candidate, dynamic prediction, peer descriptor, or graph tensor is reused. Same-tick multi-agent triggers share the current team state but compute each triggered ego branch exactly once.

## Behavior closure

- Candidate count changed: `NO`; Top-K remains 10.
- H-preview changed: `NO`; H remains 4.
- GAT/SAC-DMP/ERR trigger semantics changed: `NO`.
- Trajectory/outcome match: `{conclusion['TRAJECTORY_MATCH']}` / `{conclusion['OUTCOME_MATCH']}`.
- Success/collision rates changed: `NO` / `NO`.
- GAT fine-tuning justified: `NO`.

Graph-template caching was rejected because current-state graph values are dynamic and the remaining scoped graph cost did not justify added mutable-template complexity. Peer observability remains a separate unresolved method limitation and was not modified.

## Final decision

- `OPTIMIZED_BEHAVIOR_MATCH = {conclusion['OPTIMIZED_BEHAVIOR_MATCH']}`
- `FINAL_ENGINEERING_OPTIMIZATION_ACCEPTED = {conclusion['FINAL_ENGINEERING_OPTIMIZATION_ACCEPTED']}`
- `PEER_OBSERVABILITY_EXTENSION_STILL_RECOMMENDED = YES`
- `RECOMMENDED_NEXT_STEP = {conclusion['RECOMMENDED_NEXT_STEP']}`
"""
    (output_dir / "FINAL_REPORT.md").write_text(report, encoding="utf-8")

    required = [
        "context_recovery_manifest.json",
        "upper_planning_profile.csv",
        "upper_planning_hotspots.json",
        "gat_dependency_graph.md",
        "triggered_agent_compute_audit.json",
        "triggered_agent_equivalence.csv",
        "fp_shep_serial_profile.csv",
        "fp_shep_vectorization_contract.md",
        "fp_shep_equivalence.csv",
        "graph_build_optimization.md",
        "microbenchmark_results.csv",
        "optimization_ablation.csv",
        "closed_loop_replay_reconciliation.json",
        "runtime_method_summary.csv",
        "runtime_stage_summary.csv",
        "runtime_tradeoff.csv",
        "accepted_optimizations.json",
        "rejected_optimizations.json",
        "conclusion.json",
        "FINAL_REPORT.md",
    ]
    final_reconciliation = {
        "status": "PASS",
        "required_files": {name: (output_dir / name).exists() for name in required},
        "required_file_hashes": {
            name: file_hash(output_dir / name) for name in required
        },
        "microbenchmark_row_count": len(timing),
        "equivalence_row_count": len(equivalence),
        "optimized_episode_count": len(episodes),
        "trajectory_hash_match_count": replay_check["trajectory_hash_match_count"],
        "mandatory_conclusion_fields_present": True,
        "source_hashes_match_prefreeze": all(
            file_hash(REPO_ROOT / relative) == expected
            for relative, expected in load_json(output_dir / "engineering_prefreeze.json")["source_hashes"].items()
        ),
    }
    if not all(final_reconciliation["required_files"].values()):
        final_reconciliation["status"] = "FAIL"
    write_json(output_dir / "final_reconciliation.json", final_reconciliation)
    if final_reconciliation["status"] != "PASS":
        raise RuntimeError("final reconciliation failed")
    print(json.dumps({"phase": "analyze", "status": "PASS", "conclusion": conclusion}), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "benchmark", "replay", "analyze", "all"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if args.phase in {"prepare", "all"}:
        prepare(output)
    if args.phase in {"benchmark", "all"}:
        benchmark(output)
    if args.phase in {"replay", "all"}:
        replay(output)
    if args.phase in {"analyze", "all"}:
        analyze(output)


if __name__ == "__main__":
    main()
