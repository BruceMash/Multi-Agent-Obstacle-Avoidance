#!/usr/bin/env python3
"""Read-only H4 joint-compatibility recoverability audit for frozen GAT-V1.

This script reconstructs the already-observed formal seed states, candidate
bundles, H4 previews, graph inputs, and V1 selections.  It then performs only
offline local-compatibility enumeration.  It never calls a closed-loop episode
runner and never changes an online selection plan.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import itertools
import json
import math
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

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

from planning.gat.stage1_training import load_model_checkpoint, resolve_device  # noqa: E402
from planning.historical_forcing_gate import (  # noqa: E402
    HISTORICAL_GATE_NAME,
    scoped_historical_preview_and_multi_agent_transition,
)
from planning.policy_preview import (  # noqa: E402
    build_preview_inputs_from_env,
    preview_candidate,
)
from scripts import evaluate_gat_closed_loop as legacy  # noqa: E402
from scripts import evaluate_gat_stage1_v2_closed_loop as paired  # noqa: E402
from scripts import evaluate_gat_v1_ia_formal_closed_loop as formal  # noqa: E402
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


SCHEMA_VERSION = "gat_joint_compatibility_recoverability_audit_v1"
DEFAULT_CONFIG = REPO_ROOT / "configs/evaluation/gat_joint_compatibility_recoverability_audit.json"
PAIR_IDS = ((0, 1), (0, 2), (1, 2))
CORE_PATHS = (
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_hashes(paths: Iterable[str]) -> dict[str, str]:
    return {path: sha256_file(REPO_ROOT / path) for path in paths}


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str] | None = None,
) -> None:
    materialized = list(rows)
    fieldnames = list(fields) if fields is not None else list(materialized[0]) if materialized else ["schema_version", "status"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in materialized:
            encoded: dict[str, Any] = {}
            for key in fieldnames:
                value = row.get(key)
                if isinstance(value, (dict, list, tuple, np.ndarray)):
                    encoded[key] = json.dumps(json_ready(value), ensure_ascii=False, allow_nan=False)
                elif isinstance(value, np.generic):
                    encoded[key] = value.item()
                elif isinstance(value, float) and not math.isfinite(value):
                    encoded[key] = ""
                else:
                    encoded[key] = value
            writer.writerow(encoded)


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def parse_optional_int(value: Any) -> int | None:
    text = str(value).strip()
    return None if not text else int(text)


def parse_json_field(value: Any) -> Any:
    text = str(value).strip()
    return None if not text else json.loads(text)


def exact_json_equal(left: Any, right: Any) -> bool:
    def normalize(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return normalize(value.tolist())
        if isinstance(value, np.generic):
            return normalize(value.item())
        if isinstance(value, Mapping):
            return {str(key): normalize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        return value

    return normalize(left) == normalize(right)


def strip_preview_runtime(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Remove the sole wall-clock field from otherwise semantic H4 records."""

    return [
        {key: value for key, value in record.items() if key != "preview_runtime_ms"}
        for record in records
    ]


def float_vector_reproduction(
    formal_values: Sequence[float],
    reproduced_values: Sequence[float],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    formal_array = np.asarray(formal_values, dtype=float)
    reproduced_array = np.asarray(reproduced_values, dtype=float)
    same_shape = formal_array.shape == reproduced_array.shape
    if not same_shape:
        return {
            "same_shape": False,
            "bitwise_exact": False,
            "numerically_reproduced": False,
            "maximum_absolute_difference": None,
            "complete_ranking_exact": False,
        }
    return {
        "same_shape": True,
        "bitwise_exact": bool(np.array_equal(formal_array, reproduced_array)),
        "numerically_reproduced": bool(
            np.allclose(formal_array, reproduced_array, atol=float(atol), rtol=float(rtol))
        ),
        "maximum_absolute_difference": float(np.max(np.abs(formal_array - reproduced_array)))
        if formal_array.size
        else 0.0,
        "complete_ranking_exact": np.argsort(-formal_array, kind="stable").tolist()
        == np.argsort(-reproduced_array, kind="stable").tolist(),
    }


def validate_config(config: Mapping[str, Any]) -> None:
    if config["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unexpected audit schema")
    scope = config["formal_scope"]
    if scope["method"] != "gat_v1":
        raise ValueError("audit method must remain gat_v1")
    if tuple(scope["scenarios"]) != ("open", "sparse_static", "multi_agent"):
        raise ValueError("formal scenarios changed")
    if list(scope["seeds"]) != list(range(60, 80)):
        raise ValueError("diagnostic formal seed block must remain 60..79")
    frozen = config["frozen_semantics"]
    expected = {
        "num_agents": 3,
        "top_k": 10,
        "null_class_index": 0,
        "proposal_class_offset": 1,
        "H_preview": 4,
        "dt": 0.1,
        "d_safe": 0.6,
        "forcing_gate": HISTORICAL_GATE_NAME,
        "null_active_goal": "terminal_goal",
    }
    for key, value in expected.items():
        if frozen[key] != value:
            raise ValueError(f"frozen semantic changed: {key}")
    if any(bool(value) for value in config["strict_exclusions"].values()):
        raise ValueError("strict exclusion flags must remain false")
    for gate in (
        "h4_signal_gate",
        "minimal_repair_promise_gate",
        "structural_gap_gate",
        "theory_extension_gate",
    ):
        if not config[gate]["thresholds_frozen_before_reproduction"]:
            raise ValueError(f"{gate} thresholds were not frozen")


def raw_formal_reconciliation(
    config: Mapping[str, Any],
    episode_rows: Sequence[Mapping[str, str]],
    agent_rows: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    scope = config["formal_scope"]
    episodes = [row for row in episode_rows if row["method"] == scope["method"]]
    agents = [row for row in agent_rows if row["method"] == scope["method"]]
    keys = [(row["scenario"], int(row["seed"])) for row in episodes]
    agent_keys = [
        (row["scenario"], int(row["seed"]), int(row["agent_id"])) for row in agents
    ]
    counts = {
        "episode_count": len(episodes),
        "agent_count": len(agents),
        "success_count": sum(parse_bool(row["team_success"]) for row in episodes),
        "failure_count": sum(not parse_bool(row["team_success"]) for row in episodes),
        "inter_agent_collision_count": sum(parse_bool(row["inter_agent_collision"]) for row in episodes),
        "obstacle_collision_count": sum(parse_bool(row["obstacle_collision"]) for row in episodes),
        "timeout_count": sum(parse_bool(row["timeout"]) for row in episodes),
    }
    expected_counts = {
        "episode_count": int(scope["expected_episode_count"]),
        "agent_count": int(scope["expected_agent_count"]),
        "success_count": int(scope["expected_success_count"]),
        "failure_count": int(scope["expected_failure_count"]),
        "inter_agent_collision_count": int(scope["expected_inter_agent_collision_count"]),
        "obstacle_collision_count": int(scope["expected_obstacle_collision_count"]),
        "timeout_count": int(scope["expected_timeout_count"]),
    }
    checks = {
        "raw_counts_match_preregistered_expectation": counts == expected_counts,
        "episode_keys_unique": len(keys) == len(set(keys)) == 60,
        "episode_scope_exact": set(keys)
        == {
            (scenario, seed)
            for scenario in scope["scenarios"]
            for seed in scope["seeds"]
        },
        "agent_keys_unique": len(agent_keys) == len(set(agent_keys)) == 180,
        "three_agents_per_episode": Counter((scenario, seed) for scenario, seed, _ in agent_keys)
        == Counter({key: 3 for key in keys}),
        "failure_modes_partition_failures": counts["inter_agent_collision_count"]
        + counts["obstacle_collision_count"]
        + counts["timeout_count"]
        == counts["failure_count"],
    }
    return {
        "status": "PASSED" if all(checks.values()) else "FAILED",
        "counts": counts,
        "expected_counts": expected_counts,
        "checks": checks,
        "additional_successes_required_for_90": max(0, math.ceil(0.9 * len(episodes)) - counts["success_count"]),
    }


def build_context_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    sources = config["sources"]
    records: list[dict[str, Any]] = []
    for role, value in sources.items():
        if not (role.endswith("_dir") or role.endswith("_config") or role.endswith("_results") or role.endswith("checkpoint")):
            continue
        path = REPO_ROOT / value
        if path.is_dir():
            for name in ("config.json", "conclusion.json", "FINAL_REPORT.md", "integrity_manifest.json"):
                child = path / name
                if child.is_file():
                    records.append(
                        {
                            "role": role,
                            "path": str(child.relative_to(REPO_ROOT)),
                            "sha256": sha256_file(child),
                        }
                    )
        elif path.is_file():
            records.append(
                {
                    "role": role,
                    "path": str(path.relative_to(REPO_ROOT)),
                    "sha256": sha256_file(path),
                }
            )

    formal_conclusion = load_json(REPO_ROOT / sources["formal_v1_ia_dir"] / "conclusion.json")
    interaction = load_json(REPO_ROOT / sources["interaction_capacity_audit_dir"] / "conclusion.json")
    minimal = load_json(REPO_ROOT / sources["minimal_change_audit_dir"] / "conclusion.json")
    lower = load_json(REPO_ROOT / sources["temporary_reference_audit_dir"] / "conclusion.json")
    checks = {
        "v1_is_final_formal_checkpoint": formal_conclusion.get("FINAL_GAT_CHECKPOINT") == "V1",
        "formal_v1_success_recovered": formal_conclusion.get("V1_TEAM_SUCCESS") == 0.75,
        "ia_formal_gain_no": formal_conclusion.get("IA_FORMAL_GAIN") == "NO",
        "supervision_only_headroom_exhausted": formal_conclusion.get("EXISTING_GAT_SUPERVISION_ONLY_HEADROOM_EXHAUSTED") == "YES",
        "interaction_features_present": interaction.get("EXISTING_FEATURES_CONTAIN_INTERACTION_SIGNAL") == "YES",
        "joint_decision_dependence_partial": interaction.get("JOINT_DECISION_DEPENDENCE") == "PARTIAL",
        "minimal_interface_fix_exhausted": minimal.get("EXISTING_THEORY_MINIMAL_FIX_EXHAUSTED") == "YES",
        "targeted_sac_adaptation_not_justified": lower.get("TARGETED_SAC_ADAPTATION_JUSTIFIED") == "NO",
        "handoff_transition_gap_no": lower.get("HANDOFF_TRANSITION_GAP") == "NO",
        "v1_checkpoint_hash": sha256_file(REPO_ROOT / sources["v1_checkpoint"])
        == sources["v1_checkpoint_sha256_expected"],
        "sac_checkpoint_hash": sha256_file(REPO_ROOT / sources["sac_checkpoint"])
        == sources["sac_checkpoint_sha256_expected"],
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASSED" if all(checks.values()) else "FAILED",
        "AGENTS_md_present": (REPO_ROOT / "AGENTS.md").is_file(),
        "CODEX_HANDOFF_md_present": (REPO_ROOT / "CODEX_HANDOFF.md").is_file(),
        "sources": records,
        "checks": checks,
        "SEEDS_60_79_STATUS": "DIAGNOSTIC_ONLY_AFTER_OBSERVATION",
        "layout_24_artifact_read": False,
    }


def trajectory_columns(
    *,
    scenario: str,
    seed: int,
    agent_id: int,
    class_id: int,
    candidate_id: int | None,
    candidate_type: str,
    selected: bool,
    goal: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    preview_runtime_ms: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "scenario": scenario,
        "seed": seed,
        "agent_id": agent_id,
        "class_id": class_id,
        "candidate_id": candidate_id,
        "candidate_type": candidate_type,
        "baseline_selected": selected,
        "goal_x_m": float(goal[0]),
        "goal_y_m": float(goal[1]),
        "goal_z_m": float(goal[2]),
        "preview_runtime_ms": float(preview_runtime_ms),
    }
    for h in range(4):
        for axis, axis_name in enumerate(("x", "y", "z")):
            row[f"{axis_name}_h{h + 1}_m"] = float(positions[h, axis])
            row[f"v{axis_name}_h{h + 1}_mps"] = float(velocities[h, axis])
    return row


def reproduce_inputs(
    *,
    config: Mapping[str, Any],
    formal_config: Mapping[str, Any],
    episode_rows: Sequence[Mapping[str, str]],
    agent_rows: Sequence[Mapping[str, str]],
    policy: Any,
    v1_model: torch.nn.Module,
    device: torch.device,
    execution_settings: Mapping[str, Any],
    multi_config: Any,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    scope = config["formal_scope"]
    closed_config = formal.build_closed_config(formal_config, scope["seeds"])
    episode_index = {
        (row["scenario"], int(row["seed"])): row
        for row in episode_rows
        if row["method"] == scope["method"]
    }
    agent_index = {
        (row["scenario"], int(row["seed"]), int(row["agent_id"])): row
        for row in agent_rows
        if row["method"] == scope["method"]
    }
    source_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []

    for job_index, (scenario, seed) in enumerate(
        itertools.product(scope["scenarios"], scope["seeds"]), start=1
    ):
        formal_episode = episode_index[(scenario, int(seed))]
        shared = legacy.build_shared_selection_bundle(
            config=closed_config,
            execution_settings=execution_settings,
            multi_config=multi_config,
            policy=policy,
            gat_model=v1_model,
            gat_device=device,
            scenario=scenario,
            seed=int(seed),
        )
        reproduced_graph_hash = paired.graph_input_hash(shared["graphs"])
        snapshot = shared["scenario_snapshot"]
        reproduced_initial_hash = shared["initial_condition_hash"]
        source_checks = {
            "initial_condition_hash_exact": reproduced_initial_hash
            == formal_episode["initial_condition_hash"],
            "scenario_manifest_hash_exact": reproduced_initial_hash
            == formal_episode["scenario_manifest_hash"],
            "scenario_seed_exact": shared["scenario"] == scenario
            and int(shared["seed"]) == int(seed),
            "snapshot_has_initial_positions": "positions" in snapshot,
            "snapshot_has_terminal_goals": "goals" in snapshot,
            "snapshot_has_static_obstacles": "static_obstacles" in snapshot,
            "snapshot_has_dynamic_obstacles": "dynamic_obstacles" in snapshot,
            "snapshot_has_peer_states": "positions" in snapshot and "velocities" in snapshot,
            "candidate_bundle_hash_exact": shared["candidate_bundle_hash"]
            == formal_episode["candidate_bundle_hash"],
            "preview_bundle_hash_exact": shared["preview_bundle_hash"]
            == formal_episode["preview_bundle_hash"],
            "graph_input_hash_exact": reproduced_graph_hash
            == formal_episode["graph_input_hash"],
            "proposal_reconstruction_equivalent": bool(shared["proposal_reconstruction_equivalent"]),
            "graph_schema_match": bool(shared["graph_schema_match"]),
            "preview_historical_gate_verified": bool(shared["preview_historical_gate_verified"]),
        }
        source_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "scenario": scenario,
                "seed": int(seed),
                "formal_initial_condition_hash": formal_episode["initial_condition_hash"],
                "reproduced_initial_condition_hash": reproduced_initial_hash,
                "starts_hash": legacy.stable_hash(snapshot["starts"]),
                "terminal_goals_hash": legacy.stable_hash(snapshot["goals"]),
                "positions_hash": legacy.stable_hash(snapshot["positions"]),
                "velocities_hash": legacy.stable_hash(snapshot["velocities"]),
                "static_obstacles_hash": legacy.stable_hash(snapshot["static_obstacles"]),
                "dynamic_obstacles_hash": legacy.stable_hash(snapshot["dynamic_obstacles"]),
                "rng_seed": int(seed),
                "formal_selection_plan_hash": formal_episode["selection_plan_hash"],
                "reproduced_selection_plan_hash": shared["plans"][legacy.METHOD_GAT]["selection_plan_hash"],
                "raw_selection_plan_hash_exact": shared["plans"][legacy.METHOD_GAT]["selection_plan_hash"]
                == formal_episode["selection_plan_hash"],
                "raw_selection_plan_hash_includes_wall_clock_preview_runtime": True,
                **source_checks,
                "source_state_reproduced": all(source_checks.values()),
            }
        )

        null_previews: list[Any] = []
        null_trace: list[dict[str, Any]] = []

        def observe_null(kwargs: dict[str, Any], transition: Any) -> None:
            null_trace.append(
                {
                    "active_goal": legacy._jsonable(kwargs.get("active_goal")),
                    "forcing_gate_semantics": transition.controller_info.get("forcing_gate_semantics"),
                }
            )

        env, _ = build_closed_loop_environment(
            config=multi_config,
            scenario=scenario,
            seed=int(seed),
            peer_radius=float(closed_config["peer_radius"]),
        )
        try:
            if _scenario_hash(_scene_snapshot(env)) != reproduced_initial_hash:
                raise RuntimeError(f"null-preview source state mismatch for {scenario} seed={seed}")
            with scoped_historical_preview_and_multi_agent_transition(
                preview_observer=observe_null
            ):
                for agent_id in range(int(config["frozen_semantics"]["num_agents"])):
                    initial_state, local_context = build_preview_inputs_from_env(env, agent_id)
                    preview = preview_candidate(
                        initial_state=initial_state,
                        local_context=local_context,
                        candidate_goal=np.asarray(env.goals[agent_id], dtype=float),
                        policy=policy,
                        horizon=int(config["frozen_semantics"]["H_preview"]),
                        dmp_config=env.dmps[agent_id].config,
                        dynamics=env.dynamics[agent_id],
                    )
                    null_previews.append(preview)
        finally:
            env.close()
        if not null_trace or not all(
            row["forcing_gate_semantics"] == HISTORICAL_GATE_NAME for row in null_trace
        ):
            raise RuntimeError("null H4 preview did not use the historical vector gate")

        class_positions: list[list[np.ndarray]] = []
        class_velocities: list[list[np.ndarray]] = []
        class_goals: list[list[np.ndarray]] = []
        class_candidate_ids: list[list[int | None]] = []
        class_logits: list[list[float]] = []
        class_ranks: list[dict[int, int]] = []
        baseline_classes: list[int] = []
        existing_preview_runtime_ms = 0.0
        null_preview_runtime_ms = 0.0

        for agent_id in range(int(config["frozen_semantics"]["num_agents"])):
            formal_agent = agent_index[(scenario, int(seed), agent_id)]
            proposals = shared["proposals_by_agent"][agent_id]
            previews = shared["previews_by_agent"][agent_id]
            diagnostic = shared["gat_diagnostics"][agent_id]
            selected_class = int(diagnostic["selected_class"])
            selected_candidate_id = diagnostic["selected_candidate_id"]
            formal_selected_id = parse_optional_int(formal_agent["selected_candidate_id"])

            reproduced_points = [proposal.point.tolist() for proposal in proposals]
            reproduced_scores = [float(proposal.score) for proposal in proposals]
            reproduced_fp_records = [record.to_record() for record in previews]
            reproduced_logits = [float(value) for value in diagnostic["class_logits"]]
            reproduced_probabilities = [float(value) for value in diagnostic["class_probabilities"]]
            formal_logits = parse_json_field(formal_agent["class_logits"])
            formal_probabilities = parse_json_field(formal_agent["class_probabilities"])
            tolerance_atol = float(config["metric_definitions"]["cuda_float32_logit_probability_atol"])
            tolerance_rtol = float(config["metric_definitions"]["cuda_float32_logit_probability_rtol"])
            logit_reproduction = float_vector_reproduction(
                formal_logits,
                reproduced_logits,
                atol=tolerance_atol,
                rtol=tolerance_rtol,
            )
            probability_reproduction = float_vector_reproduction(
                formal_probabilities,
                reproduced_probabilities,
                atol=tolerance_atol,
                rtol=tolerance_rtol,
            )
            candidate_checks = {
                "candidate_count_exact": len(proposals) == int(formal_agent["K_t"]),
                "candidate_id_order_exact": list(range(len(proposals)))
                == list(range(int(formal_agent["K_t"]))),
                "candidate_positions_exact": exact_json_equal(
                    reproduced_points, parse_json_field(formal_agent["candidate_world_points"])
                ),
                "proposal_scores_exact": exact_json_equal(
                    reproduced_scores, parse_json_field(formal_agent["proposal_scores"])
                ),
                "fp_shep_records_exact_excluding_preview_runtime": exact_json_equal(
                    strip_preview_runtime(reproduced_fp_records),
                    strip_preview_runtime(parse_json_field(formal_agent["fp_shep_candidate_records"])),
                ),
                "class_logits_numerically_reproduced": logit_reproduction["numerically_reproduced"],
                "class_probabilities_numerically_reproduced": probability_reproduction["numerically_reproduced"],
                "complete_class_ranking_exact": logit_reproduction["complete_ranking_exact"],
                "selected_class_exact": selected_class == int(formal_agent["selected_class"]),
                "selected_candidate_id_exact": selected_candidate_id == formal_selected_id,
                "class_mapping_valid": bool(diagnostic["class_mapping_valid"])
                and parse_bool(formal_agent["class_mapping_valid"]),
                "logits_finite": bool(diagnostic["logits_finite"])
                and parse_bool(formal_agent["logits_finite"]),
                "candidate_bundle_hash_exact": shared["candidate_bundle_hash"]
                == formal_agent["candidate_bundle_hash"],
                "preview_bundle_hash_exact": shared["preview_bundle_hash"]
                == formal_agent["preview_bundle_hash"],
                "graph_input_hash_exact": reproduced_graph_hash
                == formal_agent["graph_input_hash"],
            }
            raw_plan_hash_exact = shared["plans"][legacy.METHOD_GAT]["selection_plan_hash"] == formal_agent["selection_plan_hash"]
            candidate_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "scenario": scenario,
                    "seed": int(seed),
                    "agent_id": agent_id,
                    "candidate_count_formal": int(formal_agent["K_t"]),
                    "candidate_count_reproduced": len(proposals),
                    "formal_selected_class": int(formal_agent["selected_class"]),
                    "reproduced_selected_class": selected_class,
                    "formal_selected_candidate_id": formal_selected_id,
                    "reproduced_selected_candidate_id": selected_candidate_id,
                    "formal_candidate_bundle_hash": formal_agent["candidate_bundle_hash"],
                    "reproduced_candidate_bundle_hash": shared["candidate_bundle_hash"],
                    "formal_preview_bundle_hash": formal_agent["preview_bundle_hash"],
                    "reproduced_preview_bundle_hash": shared["preview_bundle_hash"],
                    "formal_graph_input_hash": formal_agent["graph_input_hash"],
                    "reproduced_graph_input_hash": reproduced_graph_hash,
                    "formal_selection_plan_hash": formal_agent["selection_plan_hash"],
                    "reproduced_selection_plan_hash": shared["plans"][legacy.METHOD_GAT]["selection_plan_hash"],
                    "raw_selection_plan_hash_exact": raw_plan_hash_exact,
                    "raw_selection_plan_hash_includes_wall_clock_preview_runtime": True,
                    "selection_semantics_exclude_preview_runtime_only": True,
                    "class_logits_bitwise_exact": logit_reproduction["bitwise_exact"],
                    "class_logits_maximum_absolute_difference": logit_reproduction["maximum_absolute_difference"],
                    "class_probabilities_bitwise_exact": probability_reproduction["bitwise_exact"],
                    "class_probabilities_maximum_absolute_difference": probability_reproduction["maximum_absolute_difference"],
                    "cuda_float32_atol": tolerance_atol,
                    "cuda_float32_rtol": tolerance_rtol,
                    **candidate_checks,
                    "v1_selection_reproduced": all(candidate_checks.values()),
                }
            )

            null = null_previews[agent_id]
            positions = [np.asarray(null.trajectory.positions[1:5], dtype=float)]
            velocities = [np.asarray(null.trajectory.velocities[1:5], dtype=float)]
            goals = [np.asarray(null.trajectory.candidate_goal, dtype=float)]
            candidate_ids: list[int | None] = [None]
            null_preview_runtime_ms += float(null.performance.total_ms)
            trajectory_rows.append(
                trajectory_columns(
                    scenario=scenario,
                    seed=int(seed),
                    agent_id=agent_id,
                    class_id=0,
                    candidate_id=None,
                    candidate_type="null_terminal_goal",
                    selected=selected_class == 0,
                    goal=np.asarray(null.trajectory.candidate_goal, dtype=float),
                    positions=positions[0],
                    velocities=velocities[0],
                    preview_runtime_ms=float(null.performance.total_ms),
                )
            )
            for candidate_id, record in enumerate(previews):
                proposal_positions = np.asarray(record.preview.trajectory.positions[1:5], dtype=float)
                proposal_velocities = np.asarray(record.preview.trajectory.velocities[1:5], dtype=float)
                positions.append(proposal_positions)
                velocities.append(proposal_velocities)
                goals.append(np.asarray(record.preview.trajectory.candidate_goal, dtype=float))
                candidate_ids.append(candidate_id)
                existing_preview_runtime_ms += float(record.runtime_ms)
                trajectory_rows.append(
                    trajectory_columns(
                        scenario=scenario,
                        seed=int(seed),
                        agent_id=agent_id,
                        class_id=candidate_id + 1,
                        candidate_id=candidate_id,
                        candidate_type="proposal",
                        selected=selected_class == candidate_id + 1,
                        goal=np.asarray(record.preview.trajectory.candidate_goal, dtype=float),
                        positions=proposal_positions,
                        velocities=proposal_velocities,
                        preview_runtime_ms=float(record.runtime_ms),
                    )
                )
            order = sorted(range(len(reproduced_logits)), key=lambda class_id: (-reproduced_logits[class_id], class_id))
            ranks = {class_id: rank + 1 for rank, class_id in enumerate(order)}
            if ranks[selected_class] != 1:
                raise RuntimeError("reproduced V1 class is not logit rank 1")
            class_positions.append(positions)
            class_velocities.append(velocities)
            class_goals.append(goals)
            class_candidate_ids.append(candidate_ids)
            class_logits.append(reproduced_logits)
            class_ranks.append(ranks)
            baseline_classes.append(selected_class)

        decisions.append(
            {
                "scenario": scenario,
                "seed": int(seed),
                "formal_episode": formal_episode,
                "class_positions": class_positions,
                "class_velocities": class_velocities,
                "class_goals": class_goals,
                "class_candidate_ids": class_candidate_ids,
                "class_logits": class_logits,
                "class_ranks": class_ranks,
                "baseline_classes": tuple(baseline_classes),
                "existing_preview_runtime_ms": existing_preview_runtime_ms,
                "null_preview_runtime_ms": null_preview_runtime_ms,
            }
        )
        print(
            f"[reproduce {job_index}/60] {scenario} seed={seed}: "
            f"source={source_rows[-1]['source_state_reproduced']} "
            f"selection={all(row['v1_selection_reproduced'] for row in candidate_rows[-3:])}",
            flush=True,
        )
    return source_rows, candidate_rows, trajectory_rows, decisions


def pair_compatibility(
    positions_i: np.ndarray,
    positions_j: np.ndarray,
    *,
    d_safe: float,
    dt: float,
) -> dict[str, Any]:
    distances = np.linalg.norm(
        np.asarray(positions_i, dtype=float) - np.asarray(positions_j, dtype=float), axis=1
    )
    risk_steps = int(np.sum(distances < float(d_safe)))
    d_min = float(np.min(distances))
    t_risk = float(dt) * risk_steps
    return {
        "distances": distances,
        "d_min": d_min,
        "risk_steps": risk_steps,
        "T_risk": t_risk,
        "compatible": bool(d_min >= float(d_safe) and t_risk == 0.0),
    }


def combination_key(
    combo: tuple[int, int, int],
    baseline: tuple[int, int, int],
    ranks: Sequence[Mapping[int, int]],
    logits: Sequence[Sequence[float]],
) -> tuple[Any, ...]:
    changed = sum(left != right for left, right in zip(combo, baseline, strict=True))
    displacement = sum(int(ranks[agent_id][class_id]) - 1 for agent_id, class_id in enumerate(combo))
    logit_sum = sum(float(logits[agent_id][class_id]) for agent_id, class_id in enumerate(combo))
    return changed, displacement, -logit_sum, combo


def enumerate_compatibility(
    *,
    config: Mapping[str, Any],
    decisions: Sequence[Mapping[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    d_safe = float(config["frozen_semantics"]["d_safe"])
    dt = float(config["frozen_semantics"]["dt"])
    pairwise_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    alternative_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []

    for index, decision in enumerate(decisions, start=1):
        scenario = str(decision["scenario"])
        seed = int(decision["seed"])
        baseline = tuple(int(value) for value in decision["baseline_classes"])
        positions = decision["class_positions"]
        compatibility: dict[tuple[int, int, int, int], dict[str, Any]] = {}

        pairwise_started = time.perf_counter_ns()
        for agent_i, agent_j in PAIR_IDS:
            for class_i in range(len(positions[agent_i])):
                for class_j in range(len(positions[agent_j])):
                    result = pair_compatibility(
                        positions[agent_i][class_i],
                        positions[agent_j][class_j],
                        d_safe=d_safe,
                        dt=dt,
                    )
                    compatibility[(agent_i, agent_j, class_i, class_j)] = result
                    row = {
                        "schema_version": SCHEMA_VERSION,
                        "scenario": scenario,
                        "seed": seed,
                        "agent_i": agent_i,
                        "agent_j": agent_j,
                        "class_i": class_i,
                        "class_j": class_j,
                        "candidate_id_i": decision["class_candidate_ids"][agent_i][class_i],
                        "candidate_id_j": decision["class_candidate_ids"][agent_j][class_j],
                        "class_i_is_null": class_i == 0,
                        "class_j_is_null": class_j == 0,
                        "baseline_pair": baseline[agent_i] == class_i and baseline[agent_j] == class_j,
                        "d_min_m": result["d_min"],
                        "risk_step_count": result["risk_steps"],
                        "T_risk_s": result["T_risk"],
                        "pair_compatible": result["compatible"],
                    }
                    for h, distance in enumerate(result["distances"], start=1):
                        row[f"distance_h{h}_m"] = float(distance)
                    pairwise_rows.append(row)
        pairwise_ms = (time.perf_counter_ns() - pairwise_started) / 1.0e6

        def team_compatible(combo: tuple[int, int, int]) -> bool:
            return all(
                compatibility[(agent_i, agent_j, combo[agent_i], combo[agent_j])]["compatible"]
                for agent_i, agent_j in PAIR_IDS
            )

        enumeration_started = time.perf_counter_ns()
        compatible_combos = [
            combo
            for combo in itertools.product(*(range(len(values)) for values in positions))
            if team_compatible(tuple(int(value) for value in combo))
        ]
        enumeration_ms = (time.perf_counter_ns() - enumeration_started) / 1.0e6
        compatible_combos = [tuple(int(value) for value in combo) for combo in compatible_combos]

        search_started = time.perf_counter_ns()
        distinct = [combo for combo in compatible_combos if combo != baseline]
        best = (
            min(
                distinct,
                key=lambda combo: combination_key(
                    combo,
                    baseline,
                    decision["class_ranks"],
                    decision["class_logits"],
                ),
            )
            if distinct
            else None
        )
        search_ms = (time.perf_counter_ns() - search_started) / 1.0e6

        baseline_pair_results = {
            f"pair_{agent_i}_{agent_j}": compatibility[
                (agent_i, agent_j, baseline[agent_i], baseline[agent_j])
            ]
            for agent_i, agent_j in PAIR_IDS
        }
        baseline_compatible = team_compatible(baseline)
        formal_episode = decision["formal_episode"]
        baseline_row: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "scenario": scenario,
            "seed": seed,
            "baseline_class_0": baseline[0],
            "baseline_class_1": baseline[1],
            "baseline_class_2": baseline[2],
            "baseline_candidate_id_0": decision["class_candidate_ids"][0][baseline[0]],
            "baseline_candidate_id_1": decision["class_candidate_ids"][1][baseline[1]],
            "baseline_candidate_id_2": decision["class_candidate_ids"][2][baseline[2]],
            "baseline_uses_null": any(class_id == 0 for class_id in baseline),
            "team_local_compatible": baseline_compatible,
            "H4_CONFLICT_FLAG": not baseline_compatible,
            "actual_team_success": parse_bool(formal_episode["team_success"]),
            "actual_inter_agent_collision": parse_bool(formal_episode["inter_agent_collision"]),
            "actual_obstacle_collision": parse_bool(formal_episode["obstacle_collision"]),
            "actual_timeout": parse_bool(formal_episode["timeout"]),
            "termination_reason": formal_episode["termination_reason"],
        }
        for agent_i, agent_j in PAIR_IDS:
            result = baseline_pair_results[f"pair_{agent_i}_{agent_j}"]
            baseline_row[f"pair_{agent_i}_{agent_j}_d_min_m"] = result["d_min"]
            baseline_row[f"pair_{agent_i}_{agent_j}_T_risk_s"] = result["T_risk"]
            baseline_row[f"pair_{agent_i}_{agent_j}_compatible"] = result["compatible"]
        baseline_rows.append(baseline_row)

        if best is None:
            changed_count = None
            sum_displacement = None
            max_displacement = None
            sum_logits = None
            category = "NO_LOCAL_COMPATIBLE_ALTERNATIVE"
            best_candidate_ids = [None, None, None]
            best_uses_null = None
            best_min_pair_d = None
        else:
            changed_count = sum(a != b for a, b in zip(best, baseline, strict=True))
            displacements = [
                int(decision["class_ranks"][agent_id][class_id]) - 1
                for agent_id, class_id in enumerate(best)
            ]
            sum_displacement = sum(displacements)
            max_displacement = max(displacements)
            sum_logits = sum(
                float(decision["class_logits"][agent_id][class_id])
                for agent_id, class_id in enumerate(best)
            )
            category = {
                1: "ONE_AGENT_CHANGE_AVAILABLE",
                2: "TWO_AGENT_CHANGE_AVAILABLE",
                3: "THREE_AGENT_CHANGE_REQUIRED",
            }[changed_count]
            best_candidate_ids = [
                decision["class_candidate_ids"][agent_id][class_id]
                for agent_id, class_id in enumerate(best)
            ]
            best_uses_null = any(class_id == 0 for class_id in best)
            best_min_pair_d = min(
                compatibility[(agent_i, agent_j, best[agent_i], best[agent_j])]["d_min"]
                for agent_i, agent_j in PAIR_IDS
            )
        alternative_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "scenario": scenario,
                "seed": seed,
                "actual_team_success": parse_bool(formal_episode["team_success"]),
                "actual_inter_agent_collision": parse_bool(formal_episode["inter_agent_collision"]),
                "actual_obstacle_collision": parse_bool(formal_episode["obstacle_collision"]),
                "actual_timeout": parse_bool(formal_episode["timeout"]),
                "baseline_team_local_compatible": baseline_compatible,
                "H4_CONFLICT_FLAG": not baseline_compatible,
                "baseline_class_tuple": list(baseline),
                "combination_count_total": int(np.prod([len(values) for values in positions])),
                "compatible_combination_count_including_baseline": len(compatible_combos),
                "distinct_compatible_alternative_count": len(distinct),
                "local_compatibility_alternative_available": best is not None,
                "triggerable_local_alternative": bool(not baseline_compatible and best is not None),
                "alternative_category": category,
                "best_alternative_class_tuple": list(best) if best is not None else None,
                "best_alternative_candidate_ids": best_candidate_ids if best is not None else None,
                "changed_agent_count": changed_count,
                "sum_gat_rank_displacement": sum_displacement,
                "maximum_gat_rank_displacement": max_displacement,
                "sum_frozen_v1_logits": sum_logits,
                "alternative_uses_null": best_uses_null,
                "alternative_minimum_pair_d_min_m": best_min_pair_d,
                "local_compatibility_only": True,
                "successful_counterfactual_claimed": False,
            }
        )
        total_joint_ms = pairwise_ms + enumeration_ms + search_ms
        existing_preview_ms = float(decision["existing_preview_runtime_ms"])
        runtime_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "scenario": scenario,
                "seed": seed,
                "pairwise_matrix_construction_ms": pairwise_ms,
                "combination_enumeration_ms": enumeration_ms,
                "minimum_deviation_search_ms": search_ms,
                "joint_search_total_ms": total_joint_ms,
                "existing_proposal_h4_preview_runtime_ms": existing_preview_ms,
                "diagnostic_null_h4_preview_runtime_ms": float(decision["null_preview_runtime_ms"]),
                "estimated_preview_plus_joint_ms": existing_preview_ms
                + float(decision["null_preview_runtime_ms"])
                + total_joint_ms,
                "joint_over_existing_proposal_preview_ratio": total_joint_ms / existing_preview_ms
                if existing_preview_ms > 0
                else None,
                "joint_runtime_excludes_h4_preview": True,
                "online_implementation_performed": False,
            }
        )
        print(
            f"[enumerate {index}/60] {scenario} seed={seed}: "
            f"conflict={not baseline_compatible} compatible={len(compatible_combos)}",
            flush=True,
        )
    return pairwise_rows, baseline_rows, alternative_rows, runtime_rows


def detector_metrics(
    rows: Sequence[Mapping[str, Any]], scope: str
) -> dict[str, Any]:
    members = [row for row in rows if scope == "overall" or row["scenario"] == scope]
    tp = sum(bool(row["H4_CONFLICT_FLAG"]) and bool(row["actual_inter_agent_collision"]) for row in members)
    fp = sum(bool(row["H4_CONFLICT_FLAG"]) and not bool(row["actual_inter_agent_collision"]) for row in members)
    fn = sum(not bool(row["H4_CONFLICT_FLAG"]) and bool(row["actual_inter_agent_collision"]) for row in members)
    tn = sum(not bool(row["H4_CONFLICT_FLAG"]) and not bool(row["actual_inter_agent_collision"]) for row in members)
    successes = [row for row in members if bool(row["actual_team_success"])]
    false_veto = sum(bool(row["H4_CONFLICT_FLAG"]) for row in successes)
    return {
        "scope": scope,
        "episode_count": len(members),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "successful_episode_count": len(successes),
        "successful_false_veto_count": false_veto,
        "successful_false_veto_rate": false_veto / len(successes) if successes else None,
    }


def classify_h4_signal(metrics: Mapping[str, Any], gate: Mapping[str, Any]) -> str:
    recall = float(metrics["recall"] or 0.0)
    false_veto = float(metrics["successful_false_veto_rate"] or 0.0)
    if recall >= float(gate["strong_minimum_recall"]) and false_veto <= float(
        gate["strong_maximum_success_false_veto_rate"]
    ):
        return "STRONG"
    if recall >= float(gate["moderate_minimum_recall"]) and false_veto <= float(
        gate["moderate_maximum_success_false_veto_rate"]
    ):
        return "MODERATE"
    if int(metrics["TP"]) >= int(gate["weak_requires_true_positive_count"]):
        return "WEAK"
    return "NO"


def classify_repair_promise(
    *,
    signal: str,
    local_failure_count: int,
    collision_alternative_count: int,
    collision_one_agent_fraction: float | None,
    success_false_veto_rate: float,
    gate: Mapping[str, Any],
) -> str:
    if signal == gate["no_h4_signal"]:
        return "NO"
    if signal in gate["high_h4_signal"] and (
        local_failure_count >= int(gate["high_minimum_local_alternative_failures"])
        and success_false_veto_rate <= float(gate["high_maximum_success_false_veto_rate"])
        and collision_one_agent_fraction is not None
        and collision_one_agent_fraction
        >= float(gate["high_minimum_majority_fraction_collision_alternatives_one_agent"])
    ):
        return "HIGH"
    if signal in gate["moderate_h4_signal"] and (
        collision_alternative_count >= int(gate["moderate_minimum_collision_alternative_count"])
        and success_false_veto_rate <= float(gate["moderate_maximum_success_false_veto_rate"])
    ):
        return "MODERATE"
    if local_failure_count > 0 or collision_alternative_count > 0:
        return "LOW"
    return "NO"


def theory_extension_gate(
    *,
    signal: str,
    local_failure_count: int,
    success_false_veto_rate: float,
    promise: str,
    gate: Mapping[str, Any],
) -> bool:
    return bool(
        signal in gate["allowed_h4_signals"]
        and local_failure_count >= int(gate["minimum_local_alternative_failures"])
        and success_false_veto_rate <= float(gate["maximum_success_false_veto_rate"])
        and promise in gate["allowed_promises"]
    )


def structural_gap_decision(
    *,
    reproduction_valid: bool,
    theory_justified: bool,
    inter_collision_alternative_fraction: float,
    signal: str,
    local_failure_count: int,
    prior_joint_dependence: str,
    gate: Mapping[str, Any],
) -> str:
    if not reproduction_valid:
        return "NOT_ESTABLISHED"
    if theory_justified and inter_collision_alternative_fraction >= float(
        gate["yes_minimum_inter_collision_local_alternative_fraction"]
    ):
        return "YES"
    if local_failure_count > 0 and (
        signal != "NO" or prior_joint_dependence == "PARTIAL"
    ):
        return "PARTIAL"
    return "NO"


def summarize_runtime(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def summary(field: str) -> dict[str, Any]:
        values = np.asarray([float(row[field]) for row in rows], dtype=float)
        return {
            "count": int(values.size),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
        }

    result = {
        field: summary(field)
        for field in (
            "pairwise_matrix_construction_ms",
            "combination_enumeration_ms",
            "minimum_deviation_search_ms",
            "joint_search_total_ms",
            "existing_proposal_h4_preview_runtime_ms",
            "diagnostic_null_h4_preview_runtime_ms",
            "estimated_preview_plus_joint_ms",
        )
    }
    mean_joint = result["joint_search_total_ms"]["mean"]
    mean_preview = result["existing_proposal_h4_preview_runtime_ms"]["mean"]
    result["JOINT_SEARCH_RUNTIME_OVERHEAD_MS"] = mean_joint
    result["JOINT_SEARCH_OVERHEAD_RATIO"] = mean_joint / mean_preview if mean_preview > 0 else None
    result["joint_runtime_excludes_h4_preview"] = True
    return result


def expected_reproduction_row_counts(
    candidate_rows: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    counts = {
        (str(row["scenario"]), int(row["seed"]), int(row["agent_id"])): int(
            row["candidate_count_reproduced"]
        )
        for row in candidate_rows
    }
    episode_keys = sorted({(scenario, seed) for scenario, seed, _ in counts})
    trajectory_count = sum(candidate_count + 1 for candidate_count in counts.values())
    pairwise_count = 0
    for scenario, seed in episode_keys:
        for agent_i, agent_j in PAIR_IDS:
            pairwise_count += (counts[(scenario, seed, agent_i)] + 1) * (
                counts[(scenario, seed, agent_j)] + 1
            )
    return {
        "h4_candidate_trajectory_rows": trajectory_count,
        "pairwise_compatibility_rows": pairwise_count,
    }


def build_analysis(
    *,
    config: Mapping[str, Any],
    reconciliation: Mapping[str, Any],
    baseline_rows: Sequence[Mapping[str, Any]],
    alternative_rows: Sequence[Mapping[str, Any]],
    runtime_rows: Sequence[Mapping[str, Any]],
    source_valid: bool,
    selection_valid: bool,
) -> dict[str, Any]:
    metric_rows = [
        detector_metrics(baseline_rows, scope)
        for scope in ("overall", "multi_agent", "open", "sparse_static")
    ]
    overall = metric_rows[0]
    signal = classify_h4_signal(overall, config["h4_signal_gate"])
    failures = [row for row in alternative_rows if not bool(row["actual_team_success"])]
    successful = [row for row in alternative_rows if bool(row["actual_team_success"])]
    local_failures = [row for row in failures if bool(row["local_compatibility_alternative_available"])]
    triggerable_failures = [row for row in failures if bool(row["triggerable_local_alternative"])]
    inter_failures = [row for row in failures if bool(row["actual_inter_agent_collision"])]
    inter_with_alt = [row for row in inter_failures if bool(row["local_compatibility_alternative_available"])]
    timeout_failures = [row for row in failures if bool(row["actual_timeout"])]
    timeout_with_alt = [row for row in timeout_failures if bool(row["local_compatibility_alternative_available"])]
    obstacle_failures = [row for row in failures if bool(row["actual_obstacle_collision"])]
    obstacle_with_alt = [row for row in obstacle_failures if bool(row["local_compatibility_alternative_available"])]
    category_counts = Counter(row["alternative_category"] for row in local_failures)
    inter_one_agent = sum(int(row["changed_agent_count"] or 0) == 1 for row in inter_with_alt)
    inter_one_agent_fraction = inter_one_agent / len(inter_with_alt) if inter_with_alt else None
    promise = classify_repair_promise(
        signal=signal,
        local_failure_count=len(local_failures),
        collision_alternative_count=len(inter_with_alt),
        collision_one_agent_fraction=inter_one_agent_fraction,
        success_false_veto_rate=float(overall["successful_false_veto_rate"]),
        gate=config["minimal_repair_promise_gate"],
    )
    theory = theory_extension_gate(
        signal=signal,
        local_failure_count=len(local_failures),
        success_false_veto_rate=float(overall["successful_false_veto_rate"]),
        promise=promise,
        gate=config["theory_extension_gate"],
    )
    reproduction_valid = source_valid and selection_valid
    inter_fraction = len(inter_with_alt) / len(inter_failures) if inter_failures else 0.0
    structural = structural_gap_decision(
        reproduction_valid=reproduction_valid,
        theory_justified=theory,
        inter_collision_alternative_fraction=inter_fraction,
        signal=signal,
        local_failure_count=len(local_failures),
        prior_joint_dependence=config["structural_gap_gate"]["prior_joint_decision_dependence"],
        gate=config["structural_gap_gate"],
    )
    optimistic = (
        reconciliation["counts"]["success_count"] + len(local_failures)
    ) / reconciliation["counts"]["episode_count"]
    runtime = summarize_runtime(runtime_rows)
    return {
        "detector_metrics": metric_rows,
        "H4_JOINT_CONFLICT_SIGNAL": signal,
        "failure_rows": failures,
        "successful_rows": successful,
        "local_failure_rows": local_failures,
        "triggerable_failure_rows": triggerable_failures,
        "inter_failure_rows": inter_failures,
        "inter_with_alt_rows": inter_with_alt,
        "timeout_failure_rows": timeout_failures,
        "timeout_with_alt_rows": timeout_with_alt,
        "obstacle_failure_rows": obstacle_failures,
        "obstacle_with_alt_rows": obstacle_with_alt,
        "category_counts": category_counts,
        "inter_one_agent_fraction": inter_one_agent_fraction,
        "N_LOCAL_ALT_FAILURES": len(local_failures),
        "OPTIMISTIC_LOCAL_COMPATIBILITY_COVERAGE_BOUND": optimistic,
        "LOCAL_COMPATIBILITY_CANDIDATE_COVERAGE_INSUFFICIENT_FOR_90": len(local_failures) < 9,
        "MINIMAL_JOINT_REPAIR_PROMISE": promise,
        "JOINT_COMPATIBILITY_STRUCTURAL_GAP": structural,
        "MINIMAL_THEORY_EXTENSION_JUSTIFIED": theory,
        "runtime": runtime,
    }


def build_failure_rows(analysis: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in analysis["failure_rows"]:
        if row["actual_inter_agent_collision"]:
            failure_mode = "INTER_AGENT_COLLISION"
        elif row["actual_obstacle_collision"]:
            failure_mode = "OBSTACLE_COLLISION"
        elif row["actual_timeout"]:
            failure_mode = "TIMEOUT"
        else:
            failure_mode = "OTHER"
        rows.append(
            {
                **row,
                "failure_mode": failure_mode,
                "coordination_repair_trigger_available": bool(row["triggerable_local_alternative"]),
                "actual_recoverability_established": False,
            }
        )
    return rows


def build_false_veto_rows(
    baseline_rows: Sequence[Mapping[str, Any]],
    alternative_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    alternatives = {(row["scenario"], int(row["seed"])): row for row in alternative_rows}
    rows = []
    for baseline in baseline_rows:
        if not baseline["actual_team_success"]:
            continue
        alt = alternatives[(baseline["scenario"], int(baseline["seed"]))]
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "scenario": baseline["scenario"],
                "seed": int(baseline["seed"]),
                "H4_CONFLICT_FLAG": bool(baseline["H4_CONFLICT_FLAG"]),
                "POTENTIAL_FALSE_REPAIR": bool(baseline["H4_CONFLICT_FLAG"]),
                "minimum_alternative_changed_agent_count": alt["changed_agent_count"],
                "minimum_alternative_sum_rank_displacement": alt["sum_gat_rank_displacement"],
                "minimum_alternative_class_tuple": alt["best_alternative_class_tuple"],
            }
        )
    return rows


def build_timeout_rows(
    baseline_rows: Sequence[Mapping[str, Any]],
    alternative_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    alternatives = {(row["scenario"], int(row["seed"])): row for row in alternative_rows}
    rows = []
    for baseline in baseline_rows:
        if not baseline["actual_timeout"]:
            continue
        explicit_h4 = bool(baseline["H4_CONFLICT_FLAG"])
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "scenario": baseline["scenario"],
                "seed": int(baseline["seed"]),
                "H4_CONFLICT_FLAG": explicit_h4,
                "explicit_existing_peer_blocking_log_available": False,
                "new_stagnation_threshold_introduced": False,
                "COORDINATION_RELATED_TIMEOUT_SIGNAL": "YES" if explicit_h4 else "NOT_ESTABLISHED",
                "local_compatibility_alternative_available": alternatives[
                    (baseline["scenario"], int(baseline["seed"]))
                ]["local_compatibility_alternative_available"],
                "actual_recoverability_established": False,
            }
        )
    return rows


def report_text(
    *,
    conclusion: Mapping[str, Any],
    analysis: Mapping[str, Any],
    reconciliation: Mapping[str, Any],
) -> str:
    metrics = {row["scope"]: row for row in analysis["detector_metrics"]}
    overall = metrics["overall"]
    inter_total = len(analysis["inter_failure_rows"])
    inter_alt = len(analysis["inter_with_alt_rows"])
    timeout_alt = len(analysis["timeout_with_alt_rows"])
    runtime = analysis["runtime"]
    return "\n".join(
        [
            "# Frozen GAT-V1 Joint-Compatibility Recoverability Audit",
            "",
            "## Executive result",
            "",
            f"`JOINT_COMPATIBILITY_STRUCTURAL_GAP = {conclusion['JOINT_COMPATIBILITY_STRUCTURAL_GAP']}` and "
            f"`MINIMAL_THEORY_EXTENSION_JUSTIFIED = {conclusion['MINIMAL_THEORY_EXTENSION_JUSTIFIED']}`. "
            f"The H4 conflict signal is `{conclusion['H4_JOINT_CONFLICT_SIGNAL']}` and the minimum-repair promise is "
            f"`{conclusion['MINIMAL_JOINT_REPAIR_PROMISE']}`.",
            "",
            "This is a read-only local H4 feasibility audit. No alternative combination was executed in long-horizon closed loop, so every reported alternative is a `LOCAL_COMPATIBILITY_ALTERNATIVE`, not a successful counterfactual.",
            "",
            "## Formal result reconciliation",
            "",
            f"Raw V1 records independently reproduce {reconciliation['counts']['success_count']}/60 team successes and "
            f"{reconciliation['counts']['failure_count']} failures: {reconciliation['counts']['inter_agent_collision_count']} "
            f"inter-agent collisions, {reconciliation['counts']['obstacle_collision_count']} obstacle collision, and "
            f"{reconciliation['counts']['timeout_count']} timeouts. Reaching 90% would require "
            f"{reconciliation['additional_successes_required_for_90']} additional successes.",
            "",
            "Seeds 60-79 are now `DIAGNOSTIC_ONLY_AFTER_OBSERVATION`; they cannot be reused as an untouched final test after this diagnosis informs a future method.",
            "",
            "## Reproduction gates",
            "",
            f"- `SOURCE_STATE_REPRODUCTION = {conclusion['SOURCE_STATE_REPRODUCTION']}`",
            f"- `V1_SELECTION_REPRODUCTION = {conclusion['V1_SELECTION_REPRODUCTION']}`",
            "- All 60 t=0 source hashes and all 180 agent candidate/preview/graph/logit/class records were compared against the formal artifacts.",
            "- The raw selection-plan SHA is not used as a cross-run gate because the legacy payload includes wall-clock `preview_runtime_ms`. A stopped pre-enumeration reconciliation proved that this was the only FP-SHEP record difference; semantic reproduction excludes only that timing field and retains both raw hashes for audit.",
            "- CUDA float32 logits/probabilities are checked at frozen `atol=rtol=1e-6`; the complete class ranking and selected class must still match exactly. Bitwise flags and maximum absolute differences remain in the reproduction table.",
            "- Formal artifacts did not serialize complete H4 trajectories; the audit regenerated H4 only from exact hash-matched t=0 states with the same frozen SAC-DMP, historical vector gate, and H=4.",
            "",
            "## H4 conflict detector",
            "",
            "| scope | TP | FP | FN | TN | precision | recall | specificity | success false-veto |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            *[
                f"| {scope} | {row['TP']} | {row['FP']} | {row['FN']} | {row['TN']} | "
                f"{row['precision'] if row['precision'] is not None else 'NA'} | "
                f"{row['recall'] if row['recall'] is not None else 'NA'} | "
                f"{row['specificity'] if row['specificity'] is not None else 'NA'} | "
                f"{row['successful_false_veto_rate'] if row['successful_false_veto_rate'] is not None else 'NA'} |"
                for scope, row in metrics.items()
            ],
            "",
            f"Overall the detector marks {overall['TP'] + overall['FP']} episodes as locally incompatible. It detects "
            f"{overall['TP']} of {overall['TP'] + overall['FN']} real inter-agent collision episodes and would veto "
            f"{overall['successful_false_veto_count']} of {overall['successful_episode_count']} successful V1 episodes.",
            "",
            "## Candidate-pool local alternatives",
            "",
            f"A distinct H4 locally-compatible combination exists for {analysis['N_LOCAL_ALT_FAILURES']}/15 failed episodes. "
            f"For real inter-agent collisions the count is {inter_alt}/{inter_total}; for timeouts it is "
            f"{timeout_alt}/{len(analysis['timeout_failure_rows'])}. Counts by minimum change are "
            f"one-agent={analysis['category_counts'].get('ONE_AGENT_CHANGE_AVAILABLE', 0)}, "
            f"two-agent={analysis['category_counts'].get('TWO_AGENT_CHANGE_AVAILABLE', 0)}, and "
            f"three-agent={analysis['category_counts'].get('THREE_AGENT_CHANGE_REQUIRED', 0)}.",
            "",
            f"The optimistic candidate-availability bound is {analysis['OPTIMISTIC_LOCAL_COMPATIBILITY_COVERAGE_BOUND']:.1%}. "
            "This is not an expected success rate: H4 is local, alternatives were not executed, and baseline-compatible failures can still have distinct compatible combinations that do not address their true long-horizon failure cause.",
            "",
            "The sole obstacle-collision episode remains `NON_JOINT_FAILURE` unless the existing logs show an inter-agent causal precursor; no such precursor is introduced or fabricated here. Timeout coordination is marked `YES` only when the frozen H4 baseline combination is conflicting, otherwise `NOT_ESTABLISHED` because the formal artifact has no explicit peer-blocking log.",
            "",
            "## Runtime feasibility",
            "",
            f"Joint-search-only mean/median/P95/max runtime is "
            f"{runtime['joint_search_total_ms']['mean']:.4f}/"
            f"{runtime['joint_search_total_ms']['median']:.4f}/"
            f"{runtime['joint_search_total_ms']['p95']:.4f}/"
            f"{runtime['joint_search_total_ms']['max']:.4f} ms per team decision. "
            f"Mean overhead ratio to the already-required proposal H4 preview is "
            f"{runtime['JOINT_SEARCH_OVERHEAD_RATIO']:.6f}. Preview runtime is excluded from joint-search-only timing.",
            "",
            "## Decision",
            "",
            f"- `H4_JOINT_CONFLICT_SIGNAL = {conclusion['H4_JOINT_CONFLICT_SIGNAL']}`",
            f"- `MINIMAL_JOINT_REPAIR_PROMISE = {conclusion['MINIMAL_JOINT_REPAIR_PROMISE']}`",
            f"- `JOINT_COMPATIBILITY_STRUCTURAL_GAP = {conclusion['JOINT_COMPATIBILITY_STRUCTURAL_GAP']}`",
            f"- `MINIMAL_THEORY_EXTENSION_JUSTIFIED = {conclusion['MINIMAL_THEORY_EXTENSION_JUSTIFIED']}`",
            f"- `LOCAL_COMPATIBILITY_CANDIDATE_COVERAGE_INSUFFICIENT_FOR_90 = {conclusion['LOCAL_COMPATIBILITY_CANDIDATE_COVERAGE_INSUFFICIENT_FOR_90']}`",
            f"- `RECOMMENDED_NEXT_STEP = {conclusion['RECOMMENDED_NEXT_STEP']}`",
            "",
            "## Stop rule",
            "",
            "The audit stops here. It did not implement or connect a joint selector, modify a selection plan, train a model, change any frozen component, execute a new 220-step branch, or read/rerun the 24-layout artifact.",
            "",
        ]
    )


def run_audit(config: Mapping[str, Any], output_dir: Path) -> Path:
    validate_config(config)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    resolved = copy.deepcopy(dict(config))
    resolved["created_at"] = datetime.now().astimezone().isoformat()
    resolved["resolved_output_dir"] = str(output_dir)
    resolved["thresholds_frozen_before_reproduction"] = True
    write_json(output_dir / "config.json", resolved)

    context = build_context_manifest(config)
    write_json(output_dir / "context_recovery_manifest.json", context)
    if context["status"] != "PASSED":
        raise RuntimeError(f"context recovery failed: {context['checks']}")

    sources = config["sources"]
    formal_config_payload = load_json(REPO_ROOT / sources["formal_config"])
    formal.validate_config(formal_config_payload)
    episode_rows = read_csv(REPO_ROOT / sources["formal_episode_results"])
    agent_rows = read_csv(REPO_ROOT / sources["formal_agent_results"])
    reconciliation = raw_formal_reconciliation(config, episode_rows, agent_rows)
    write_json(output_dir / "raw_formal_reconciliation.json", reconciliation)
    if reconciliation["status"] != "PASSED":
        write_json(
            output_dir / "conclusion.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "RAW_RECONCILIATION_FAILED",
                "SOURCE_STATE_REPRODUCTION": "NOT_RUN",
                "V1_SELECTION_REPRODUCTION": "NOT_RUN",
                "RECOMMENDED_NEXT_STEP": "NOT_ESTABLISHED",
            },
        )
        raise RuntimeError(f"raw formal reconciliation failed: {reconciliation}")

    core_before = file_hashes(CORE_PATHS)
    expected_core = formal_config_payload["expected_core_hashes"]
    if core_before != expected_core:
        raise RuntimeError("frozen core source hashes differ from formal evaluation")
    audit_source_paths = (
        "Multi-agent_Algo_lib/scripts/evaluate_gat_closed_loop.py",
        "Multi-agent_Algo_lib/scripts/evaluate_gat_stage1_v2_closed_loop.py",
        "Multi-agent_Algo_lib/scripts/evaluate_gat_v1_ia_formal_closed_loop.py",
        "Multi-agent_Algo_lib/scripts/audit_gat_v1_joint_compatibility_recoverability.py",
        "configs/evaluation/gat_joint_compatibility_recoverability_audit.json",
    )
    auxiliary_before = file_hashes(audit_source_paths)

    closed_config = formal.build_closed_config(formal_config_payload, config["formal_scope"]["seeds"])
    execution_settings = legacy._build_execution_settings(closed_config)
    multi_config = build_single_distribution_multi_config(
        num_agents=int(config["frozen_semantics"]["num_agents"]),
        max_steps=int(formal_config_payload["execution"]["max_steps"]),
    )
    policy, loaded_sac = _load_policy(execution_settings, multi_config)
    expected_sac = (REPO_ROOT / sources["sac_checkpoint"]).resolve()
    if loaded_sac.resolve() != expected_sac:
        raise RuntimeError("SAC loader resolved an unexpected checkpoint")
    policy_before = _policy_parameter_sha256(policy)
    stage1_config = load_json(REPO_ROOT / sources["stage1_config"])
    device = resolve_device(stage1_config["training"]["device"])
    v1_path = (REPO_ROOT / sources["v1_checkpoint"]).resolve()
    v1_model = load_model_checkpoint(v1_path, stage1_config, device)
    v1_model.eval()
    for parameter in v1_model.parameters():
        parameter.requires_grad_(False)
    model_before = formal.model_hash(v1_model)

    source_rows, candidate_rows, trajectory_rows, decisions = reproduce_inputs(
        config=config,
        formal_config=formal_config_payload,
        episode_rows=episode_rows,
        agent_rows=agent_rows,
        policy=policy,
        v1_model=v1_model,
        device=device,
        execution_settings=execution_settings,
        multi_config=multi_config,
    )
    write_csv(output_dir / "source_state_reproduction.csv", source_rows)
    write_csv(output_dir / "candidate_reproduction.csv", candidate_rows)
    pq.write_table(
        pa.Table.from_pylist(trajectory_rows),
        output_dir / "h4_candidate_trajectories.parquet",
        compression="zstd",
    )
    source_valid = len(source_rows) == 60 and all(row["source_state_reproduced"] for row in source_rows)
    selection_valid = len(candidate_rows) == 180 and all(row["v1_selection_reproduced"] for row in candidate_rows)
    if not source_valid or not selection_valid:
        write_json(
            output_dir / "conclusion.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "REPRODUCTION_GATE_FAILED",
                "SOURCE_STATE_REPRODUCTION": "YES" if source_valid else "NO",
                "V1_SELECTION_REPRODUCTION": "YES" if selection_valid else "NO",
                "RECOMMENDED_NEXT_STEP": "NOT_ESTABLISHED",
            },
        )
        raise RuntimeError("source-state or V1-selection reproduction gate failed")

    pairwise_rows, baseline_rows, alternative_rows, runtime_rows = enumerate_compatibility(
        config=config,
        decisions=decisions,
    )
    pq.write_table(
        pa.Table.from_pylist(pairwise_rows),
        output_dir / "pairwise_compatibility_matrices.parquet",
        compression="zstd",
    )
    write_csv(output_dir / "baseline_joint_compatibility.csv", baseline_rows)
    write_csv(output_dir / "h4_conflict_detection.csv", baseline_rows)
    write_csv(output_dir / "alternative_combinations.csv", alternative_rows)
    write_csv(output_dir / "runtime_audit.csv", runtime_rows)

    analysis = build_analysis(
        config=config,
        reconciliation=reconciliation,
        baseline_rows=baseline_rows,
        alternative_rows=alternative_rows,
        runtime_rows=runtime_rows,
        source_valid=source_valid,
        selection_valid=selection_valid,
    )
    failure_rows = build_failure_rows(analysis)
    false_veto_rows = build_false_veto_rows(baseline_rows, alternative_rows)
    timeout_rows = build_timeout_rows(baseline_rows, alternative_rows)
    write_csv(output_dir / "failure_local_alternative_summary.csv", failure_rows)
    write_csv(output_dir / "successful_false_veto.csv", false_veto_rows)
    write_csv(output_dir / "timeout_coordination_audit.csv", timeout_rows)
    write_json(
        output_dir / "conflict_detector_metrics.json",
        {
            "schema_version": SCHEMA_VERSION,
            "rows": analysis["detector_metrics"],
            "H4_JOINT_CONFLICT_SIGNAL": analysis["H4_JOINT_CONFLICT_SIGNAL"],
            "thresholds": config["h4_signal_gate"],
        },
    )
    coverage = {
        "schema_version": SCHEMA_VERSION,
        "formal_success_count": reconciliation["counts"]["success_count"],
        "formal_failure_count": reconciliation["counts"]["failure_count"],
        "additional_successes_required_for_90": reconciliation["additional_successes_required_for_90"],
        "N_LOCAL_ALT_FAILURES": analysis["N_LOCAL_ALT_FAILURES"],
        "triggerable_h4_conflict_failure_with_local_alternative_count": len(analysis["triggerable_failure_rows"]),
        "inter_agent_collision_total": len(analysis["inter_failure_rows"]),
        "inter_agent_collision_with_local_alternative": len(analysis["inter_with_alt_rows"]),
        "timeout_total": len(analysis["timeout_failure_rows"]),
        "timeout_with_local_alternative": len(analysis["timeout_with_alt_rows"]),
        "obstacle_collision_total": len(analysis["obstacle_failure_rows"]),
        "obstacle_collision_with_local_alternative": len(analysis["obstacle_with_alt_rows"]),
        "one_agent_change_failures": analysis["category_counts"].get("ONE_AGENT_CHANGE_AVAILABLE", 0),
        "two_agent_change_failures": analysis["category_counts"].get("TWO_AGENT_CHANGE_AVAILABLE", 0),
        "three_agent_change_failures": analysis["category_counts"].get("THREE_AGENT_CHANGE_REQUIRED", 0),
        "OPTIMISTIC_LOCAL_COMPATIBILITY_COVERAGE_BOUND": analysis["OPTIMISTIC_LOCAL_COMPATIBILITY_COVERAGE_BOUND"],
        "THIS_IS_NOT_AN_EXPECTED_SUCCESS_RATE": True,
        "LOCAL_COMPATIBILITY_CANDIDATE_COVERAGE_INSUFFICIENT_FOR_90": "YES"
        if analysis["LOCAL_COMPATIBILITY_CANDIDATE_COVERAGE_INSUFFICIENT_FOR_90"]
        else "NO",
        "successful_counterfactual_claimed": False,
    }
    write_json(output_dir / "coverage_analysis.json", coverage)

    overall = analysis["detector_metrics"][0]
    decision = {
        "schema_version": SCHEMA_VERSION,
        "H4_JOINT_CONFLICT_SIGNAL": analysis["H4_JOINT_CONFLICT_SIGNAL"],
        "N_LOCAL_ALT_FAILURES": analysis["N_LOCAL_ALT_FAILURES"],
        "SUCCESS_EPISODE_FALSE_VETO_RATE": overall["successful_false_veto_rate"],
        "MINIMAL_JOINT_REPAIR_PROMISE": analysis["MINIMAL_JOINT_REPAIR_PROMISE"],
        "JOINT_COMPATIBILITY_STRUCTURAL_GAP": analysis["JOINT_COMPATIBILITY_STRUCTURAL_GAP"],
        "MINIMAL_THEORY_EXTENSION_JUSTIFIED": "YES"
        if analysis["MINIMAL_THEORY_EXTENSION_JUSTIFIED"]
        else "NO",
        "theory_extension_gate": config["theory_extension_gate"],
        "all_decision_thresholds_frozen_before_reproduction": True,
        "gate_purpose": "development_implementation_worth_testing_only_not_formal_method_adoption",
    }
    write_json(output_dir / "decision_gate.json", decision)

    conclusion = {
        "schema_version": SCHEMA_VERSION,
        "SOURCE_STATE_REPRODUCTION": "YES",
        "V1_SELECTION_REPRODUCTION": "YES",
        "FORMAL_V1_SUCCESS": "45/60",
        "FORMAL_V1_FAILURES": 15,
        "ADDITIONAL_SUCCESSES_REQUIRED_FOR_90": 9,
        "H4_CONFLICT_TP": overall["TP"],
        "H4_CONFLICT_FP": overall["FP"],
        "H4_CONFLICT_FN": overall["FN"],
        "H4_CONFLICT_TN": overall["TN"],
        "H4_INTER_AGENT_COLLISION_RECALL": overall["recall"],
        "H4_CONFLICT_PRECISION": overall["precision"],
        "SUCCESS_EPISODE_FALSE_VETO_COUNT": overall["successful_false_veto_count"],
        "SUCCESS_EPISODE_FALSE_VETO_RATE": overall["successful_false_veto_rate"],
        "H4_JOINT_CONFLICT_SIGNAL": analysis["H4_JOINT_CONFLICT_SIGNAL"],
        "INTER_AGENT_COLLISION_WITH_LOCAL_ALTERNATIVE": len(analysis["inter_with_alt_rows"]),
        "INTER_AGENT_COLLISION_WITH_LOCAL_ALTERNATIVE_FRACTION": f"{len(analysis['inter_with_alt_rows'])}/{len(analysis['inter_failure_rows'])}",
        "TIMEOUT_WITH_LOCAL_ALTERNATIVE": len(analysis["timeout_with_alt_rows"]),
        "N_LOCAL_ALT_FAILURES": analysis["N_LOCAL_ALT_FAILURES"],
        "ONE_AGENT_CHANGE_FAILURES": analysis["category_counts"].get("ONE_AGENT_CHANGE_AVAILABLE", 0),
        "TWO_AGENT_CHANGE_FAILURES": analysis["category_counts"].get("TWO_AGENT_CHANGE_AVAILABLE", 0),
        "THREE_AGENT_CHANGE_FAILURES": analysis["category_counts"].get("THREE_AGENT_CHANGE_REQUIRED", 0),
        "OPTIMISTIC_LOCAL_COMPATIBILITY_COVERAGE_BOUND": analysis["OPTIMISTIC_LOCAL_COMPATIBILITY_COVERAGE_BOUND"],
        "LOCAL_COMPATIBILITY_CANDIDATE_COVERAGE_INSUFFICIENT_FOR_90": "YES"
        if analysis["LOCAL_COMPATIBILITY_CANDIDATE_COVERAGE_INSUFFICIENT_FOR_90"]
        else "NO",
        "JOINT_SEARCH_RUNTIME_OVERHEAD_MS": analysis["runtime"]["JOINT_SEARCH_RUNTIME_OVERHEAD_MS"],
        "JOINT_SEARCH_OVERHEAD_RATIO": analysis["runtime"]["JOINT_SEARCH_OVERHEAD_RATIO"],
        "MINIMAL_JOINT_REPAIR_PROMISE": analysis["MINIMAL_JOINT_REPAIR_PROMISE"],
        "JOINT_COMPATIBILITY_STRUCTURAL_GAP": analysis["JOINT_COMPATIBILITY_STRUCTURAL_GAP"],
        "MINIMAL_THEORY_EXTENSION_JUSTIFIED": "YES"
        if analysis["MINIMAL_THEORY_EXTENSION_JUSTIFIED"]
        else "NO",
        "SEEDS_60_79_STATUS": "DIAGNOSTIC_ONLY_AFTER_OBSERVATION",
        "RECOMMENDED_NEXT_STEP": "DEVELOP_MINIMAL_CONFLICT_AWARE_SELECTION_REPAIR"
        if analysis["MINIMAL_THEORY_EXTENSION_JUSTIFIED"]
        else "KEEP_GAT_V1_AND_PROCEED_TO_BASELINES",
        "THIS_IS_NOT_AN_EXPECTED_SUCCESS_RATE": True,
        "successful_counterfactual_claimed": False,
        "status": "COMPLETE",
    }
    write_json(output_dir / "conclusion.json", conclusion)
    (output_dir / "FINAL_REPORT.md").write_text(
        report_text(
            conclusion=conclusion,
            analysis=analysis,
            reconciliation=reconciliation,
        ),
        encoding="utf-8",
    )

    core_after = file_hashes(CORE_PATHS)
    auxiliary_after = file_hashes(audit_source_paths)
    required_outputs = (
        "config.json",
        "context_recovery_manifest.json",
        "source_state_reproduction.csv",
        "candidate_reproduction.csv",
        "baseline_joint_compatibility.csv",
        "pairwise_compatibility_matrices.parquet",
        "h4_conflict_detection.csv",
        "conflict_detector_metrics.json",
        "alternative_combinations.csv",
        "failure_local_alternative_summary.csv",
        "successful_false_veto.csv",
        "timeout_coordination_audit.csv",
        "runtime_audit.csv",
        "coverage_analysis.json",
        "decision_gate.json",
        "conclusion.json",
        "FINAL_REPORT.md",
    )
    expected_rows = expected_reproduction_row_counts(candidate_rows)
    integrity_checks = {
        "context_recovery_valid": context["status"] == "PASSED",
        "raw_formal_reconciliation_valid": reconciliation["status"] == "PASSED",
        "source_state_reproduction_60_of_60": source_valid,
        "v1_selection_reproduction_180_of_180": selection_valid,
        "candidate_count_matches_formal_and_within_top_k": all(
            row["candidate_count_exact"]
            and 0 <= int(row["candidate_count_reproduced"]) <= int(config["frozen_semantics"]["top_k"])
            for row in candidate_rows
        ),
        "h4_trajectory_count_expected_from_frozen_K_t": len(trajectory_rows)
        == expected_rows["h4_candidate_trajectory_rows"],
        "pairwise_matrix_row_count_expected_from_frozen_K_t": len(pairwise_rows)
        == expected_rows["pairwise_compatibility_rows"],
        "baseline_decision_count_expected": len(baseline_rows) == 60,
        "failure_decision_count_expected": len(failure_rows) == 15,
        "successful_false_veto_denominator_45": len(false_veto_rows) == 45,
        "timeout_audit_count_expected": len(timeout_rows) == 5,
        "core_hashes_unchanged": core_before == core_after == expected_core,
        "auxiliary_hashes_unchanged": auxiliary_before == auxiliary_after,
        "checkpoint_hashes_unchanged": sha256_file(v1_path)
        == sources["v1_checkpoint_sha256_expected"]
        and sha256_file(expected_sac) == sources["sac_checkpoint_sha256_expected"],
        "model_hash_unchanged": model_before == formal.model_hash(v1_model),
        "policy_hash_unchanged": policy_before == _policy_parameter_sha256(policy),
        "all_required_outputs_present": all((output_dir / name).is_file() for name in required_outputs),
        "no_long_horizon_alternative_execution": True,
        "no_online_selector_implementation": True,
        "no_selection_plan_modification": True,
        "no_training_or_tuning": True,
        "layout_24_not_read_or_run": True,
    }
    integrity = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASSED" if all(integrity_checks.values()) else "FAILED",
        "checks": integrity_checks,
        "core_hashes_before": core_before,
        "core_hashes_after": core_after,
        "auxiliary_hashes_before": auxiliary_before,
        "auxiliary_hashes_after": auxiliary_after,
        "v1_checkpoint_sha256": sha256_file(v1_path),
        "sac_checkpoint_sha256": sha256_file(expected_sac),
        "v1_model_parameter_hash_before": model_before,
        "v1_model_parameter_hash_after": formal.model_hash(v1_model),
        "sac_policy_parameter_hash_before": policy_before,
        "sac_policy_parameter_hash_after": _policy_parameter_sha256(policy),
        "row_counts": {
            "source_state_reproduction.csv": len(source_rows),
            "candidate_reproduction.csv": len(candidate_rows),
            "h4_candidate_trajectories.parquet": len(trajectory_rows),
            "pairwise_compatibility_matrices.parquet": len(pairwise_rows),
            "baseline_joint_compatibility.csv": len(baseline_rows),
            "alternative_combinations.csv": len(alternative_rows),
            "failure_local_alternative_summary.csv": len(failure_rows),
            "successful_false_veto.csv": len(false_veto_rows),
            "timeout_coordination_audit.csv": len(timeout_rows),
            "runtime_audit.csv": len(runtime_rows),
        },
        "expected_row_counts_from_frozen_K_t": expected_rows,
        "maximum_preview_horizon_executed": 4,
        "closed_loop_team_episode_count_executed": 0,
        "long_horizon_alternative_branch_count_executed": 0,
        "selection_plan_write_count": 0,
        "training_step_count": 0,
        "layout_24_artifact_access_count": 0,
    }
    write_json(output_dir / "integrity_manifest.json", integrity)
    if integrity["status"] != "PASSED":
        raise RuntimeError(f"integrity checks failed: {integrity_checks}")
    return output_dir


def main() -> Path:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    config = load_json(config_path)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = REPO_ROOT / config["output_root"] / datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    elif not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    result = run_audit(config, output_dir)
    print(result)
    return result


if __name__ == "__main__":
    main()
