"""Generate controlled candidate-supervision graphs, labels, and rollouts.

No GAT training is performed.  Each graph uses the operational H_preview=4
FP-SHEP features.  H_label previews are serialized only for offline
preview-real divergence diagnostics.
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
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = Path(__file__).resolve().parent
for search_path in (REPO_ROOT, ALGO_ROOT, SCRIPTS_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from Environment.frozen_sac_dmp_execution import freeze_policy, predict_frozen_actions  # noqa: E402
from Guidance.reference_point_proposal_demo import ProposalConfig, propose_reference_points  # noqa: E402
from experiment_config import EXPERIMENT_CONFIG as SINGLE_AGENT_CONFIG  # noqa: E402
from planning.candidate_execution_benchmark import (  # noqa: E402
    _active_goal_observations,
    environment_state_fingerprint,
)
from planning.candidate_supervision import (  # noqa: E402
    SUPERVISION_SCHEMA_VERSION,
    assign_seed_split,
    portable_raw_metric,
    state_group_id,
    validate_sample_timesteps,
    validate_split_seeds,
)
from planning.candidate_supervision_dataset import (  # noqa: E402
    CandidateSupervisionConfig,
    EgoSupervisionEvaluation,
    HorizonSupervisionEvaluation,
    evaluate_ego_supervision,
)
from planning.heterogeneous_candidate_graph import HeterogeneousCandidateGraphConfig  # noqa: E402
from runner_sac import build_env as build_single_env  # noqa: E402
from runner_sac import build_model as build_single_model  # noqa: E402
from runner_sac import load_checkpoint  # noqa: E402
from scripts.evaluate_single_policy_aligned_multi_agent import (  # noqa: E402
    STAGE_SPECS,
    build_single_distribution_multi_config,
    build_stage_scenario,
)
from scripts.evaluate_single_policy_multi_agent import _build_environment  # noqa: E402
from scripts.validate_policy_preview import (  # noqa: E402
    SCENE_ADAPTERS,
    _scene_snapshot,
    build_validation_environment,
)


DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "evaluation" / "candidate_supervision_audit.json"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            encoded: dict[str, Any] = {}
            for key in fieldnames:
                value = _jsonable(row.get(key))
                encoded[key] = (
                    json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (list, tuple, dict))
                    else value
                )
            writer.writerow(encoded)


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _policy_parameter_sha256(policy: Any) -> str:
    hasher = hashlib.sha256()
    state_dict = policy.actor.state_dict()
    for key in sorted(state_dict):
        value = state_dict[key].detach().cpu().contiguous().numpy()
        hasher.update(key.encode("utf-8"))
        hasher.update(str(value.dtype).encode("ascii"))
        hasher.update(str(value.shape).encode("ascii"))
        hasher.update(value.tobytes())
    return hasher.hexdigest()


def _build_supervision_environment(
    *,
    config: Any,
    scene_type: str,
    seed: int,
    peer_radius: float,
) -> tuple[Any, dict[str, Any]]:
    if scene_type in SCENE_ADAPTERS:
        return build_validation_environment(
            config=config,
            scene_type=scene_type,
            seed=seed,
            peer_radius=peer_radius,
        )
    if scene_type != "narrow_head_on":
        raise ValueError(f"unknown candidate-supervision scenario: {scene_type}")
    stage = next(item for item in STAGE_SPECS if item["name"] == "E_head_on_narrow_peer_spheres")
    options = build_stage_scenario(config, stage, seed=int(seed))
    env = _build_environment(
        config,
        observation_mode="peer_spheres",
        peer_radius=float(peer_radius),
        training_distribution=False,
        include_boundaries_in_sensor=False,
        terminate_on_boundary_collision=False,
    )
    env.reset(seed=int(seed), options=copy.deepcopy(options))
    return env, {
        "scene_type": scene_type,
        "source_stage": stage["name"],
        "observation_mode": "peer_spheres",
        "include_boundaries_in_sensor": False,
        "terminate_on_boundary_collision": False,
        "static_obstacle_count": len(options["static_obstacles"]),
        "dynamic_obstacle_count": len(options["dynamic_obstacles"]),
    }


def _generate_proposals(env: Any, agent_index: int, config: ProposalConfig) -> list[Any]:
    packet = env.latest_sensor_packets[agent_index]
    if packet is None:
        raise RuntimeError("environment must be reset before proposal generation")
    return propose_reference_points(
        env.dynamics[agent_index].p,
        env.goals[agent_index],
        env.dynamics[agent_index].v,
        packet,
        env.sensors[agent_index],
        config,
        env.env_config.goal_tolerance,
    )


def _advance_environment_one_step(env: Any, policy: Any) -> tuple[bool, bool, dict[str, Any]]:
    active_goals = np.stack([np.asarray(item.goal, dtype=float) for item in env.dmps])
    observations = _active_goal_observations(env, active_goals)
    actions = predict_frozen_actions(policy, observations, expected_shape=tuple(env.action_shape))
    _, _, terminated, truncated, info = env.step(actions)
    return bool(terminated), bool(truncated), info


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _pad_trajectory(rows: list[np.ndarray], length: int) -> tuple[np.ndarray, np.ndarray]:
    output = np.zeros((len(rows), int(length), 3), dtype=np.float64)
    mask = np.zeros((len(rows), int(length)), dtype=bool)
    for index, row in enumerate(rows):
        array = np.asarray(row, dtype=float)
        count = min(int(length), int(array.shape[0]))
        output[index, :count] = array[:count]
        mask[index, :count] = True
    return output, mask


def _save_rollout_bundle(
    path: Path,
    *,
    evaluation: EgoSupervisionEvaluation,
    horizon_evaluation: HorizonSupervisionEvaluation,
    env: Any,
    agent_index: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    class_count = len(evaluation.formal_previews)
    formal_length = max(item.trajectory.positions.shape[0] for item in evaluation.formal_previews)
    diagnostic_length = max(
        item.trajectory.positions.shape[0] for item in horizon_evaluation.diagnostic_previews
    )
    real_length = int(horizon_evaluation.h_label) + 1
    formal_positions, formal_mask = _pad_trajectory(
        [item.trajectory.positions for item in evaluation.formal_previews], formal_length
    )
    formal_velocities, _ = _pad_trajectory(
        [item.trajectory.velocities for item in evaluation.formal_previews], formal_length
    )
    diagnostic_positions, diagnostic_mask = _pad_trajectory(
        [item.trajectory.positions for item in horizon_evaluation.diagnostic_previews],
        diagnostic_length,
    )
    diagnostic_velocities, _ = _pad_trajectory(
        [item.trajectory.velocities for item in horizon_evaluation.diagnostic_previews],
        diagnostic_length,
    )
    real_positions, real_mask = _pad_trajectory(
        [item.positions for item in horizon_evaluation.real_rollouts], real_length
    )
    real_velocities, _ = _pad_trajectory(
        [item.velocities for item in horizon_evaluation.real_rollouts], real_length
    )
    error_values = np.zeros((class_count, real_length), dtype=np.float64)
    error_mask = np.zeros((class_count, real_length), dtype=bool)
    for index, (diagnostic, real) in enumerate(
        zip(
            horizon_evaluation.diagnostic_previews,
            horizon_evaluation.real_rollouts,
            strict=True,
        )
    ):
        count = min(diagnostic.trajectory.positions.shape[0], real.positions.shape[0])
        error_values[index, :count] = np.linalg.norm(
            diagnostic.trajectory.positions[:count] - real.positions[:count], axis=1
        )
        error_mask[index, :count] = True
    goals = np.stack([item.trajectory.candidate_goal for item in evaluation.formal_previews])
    all_positions = np.stack([np.asarray(item.p, dtype=float) for item in env.dynamics])
    all_velocities = np.stack([np.asarray(item.v, dtype=float) for item in env.dynamics])
    target = horizon_evaluation.target_bundle
    arrays: dict[str, Any] = {
        "schema_version_code": np.asarray([1], dtype=np.int64),
        "class_index": np.arange(class_count, dtype=np.int64),
        "is_null": np.asarray([True] + [False] * (class_count - 1), dtype=bool),
        "candidate_goals": goals,
        "ego_initial_position": all_positions[agent_index],
        "task_goal": np.asarray(env.goals[agent_index], dtype=float),
        "all_agent_initial_positions": all_positions,
        "all_agent_initial_velocities": all_velocities,
        "visible_surface_points": np.asarray(evaluation.visible_surface_points, dtype=float).reshape(-1, 3),
        "formal_preview_positions": formal_positions,
        "formal_preview_position_mask": formal_mask,
        "formal_preview_velocities": formal_velocities,
        "diagnostic_preview_positions": diagnostic_positions,
        "diagnostic_preview_position_mask": diagnostic_mask,
        "diagnostic_preview_velocities": diagnostic_velocities,
        "real_positions": real_positions,
        "real_position_mask": real_mask,
        "real_velocities": real_velocities,
        "preview_real_position_error": error_values,
        "preview_real_position_error_mask": error_mask,
        "formal_J_preview_3": evaluation.formal_preview_quality.target_three_feature,
        "formal_J_preview_4": evaluation.formal_preview_quality.target_four_feature,
        "J_target_3": horizon_evaluation.real_quality.target_three_feature,
        "J_target_4": horizon_evaluation.real_quality.target_four_feature,
        "failure_mask": horizon_evaluation.real_quality.failure_mask,
    }
    for temperature_key, payload in target["soft_targets"].items():
        safe_key = temperature_key.replace(".", "p")
        arrays[f"soft_primary_{safe_key}"] = payload["provisional_primary_target"]
        arrays[f"soft_companion_{safe_key}"] = payload["companion_target"]
    np.savez_compressed(path, **arrays)


def _label_payload(
    evaluation: EgoSupervisionEvaluation,
    horizon_evaluation: HorizonSupervisionEvaluation,
) -> dict[str, Any]:
    target = horizon_evaluation.target_bundle
    return {
        "schema_version": SUPERVISION_SCHEMA_VERSION,
        "H_preview_formal": 4,
        "H_label": horizon_evaluation.h_label,
        "diagnostic_preview_at_H_label": True,
        "diagnostic_preview_role": "offline_equal_horizon_preview_real_divergence_only",
        "diagnostic_preview_excluded_from_graph": True,
        "diagnostic_preview_excluded_from_formal_J_preview": True,
        "class_mapping": [
            {
                "class_index": index,
                "class_kind": "null" if index == 0 else "proposal",
                "proposal_node_index": None if index == 0 else index - 1,
                "candidate_id": None if index == 0 else index - 1,
                "candidate_xyz": preview.trajectory.candidate_goal,
            }
            for index, preview in enumerate(evaluation.formal_previews)
        ],
        "formal_J_preview_3": evaluation.formal_preview_quality.target_three_feature,
        "formal_J_preview_4": evaluation.formal_preview_quality.target_four_feature,
        "provisional_primary_target": target["provisional_primary_target"],
        "companion_target": target["companion_target"],
        "hard_target_three_feature": target["hard_target_three_feature"],
        "hard_target_four_feature": target["hard_target_four_feature"],
        "soft_targets": target["soft_targets"],
        "failure_mask": horizon_evaluation.real_quality.failure_mask,
        "quality_normalized": horizon_evaluation.real_quality.normalized.values,
        "quality_valid_mask": horizon_evaluation.real_quality.normalized.valid_mask,
        "clearance_raw_portable": [
            portable_raw_metric(item.min_clearance)
            for item in horizon_evaluation.real_rollouts
        ],
        "final_supervision_target_frozen": False,
    }


def _critical_hashes(checkpoint: Path) -> dict[str, str]:
    paths = {
        "checkpoint": checkpoint,
        "candidate_generator": REPO_ROOT / "Guidance" / "reference_point_proposal_demo.py",
        "policy_preview": REPO_ROOT / "planning" / "policy_preview.py",
        "real_execution": REPO_ROOT / "planning" / "candidate_execution_benchmark.py",
        "graph_builder": REPO_ROOT / "planning" / "heterogeneous_candidate_graph.py",
        "gat_architecture": REPO_ROOT / "planning" / "gat" / "edge_enhanced_gat.py",
    }
    return {name: _sha256(path) for name, path in paths.items() if path.is_file()}


def run_generation(
    settings: dict[str, Any],
    output_dir: Path,
    *,
    max_ego_samples: int | None = None,
    run_audit: bool = True,
) -> dict[str, Any]:
    if settings.get("deterministic_policy") is not True:
        raise ValueError("candidate supervision requires deterministic_policy=true")
    if int(settings["H_preview"]) != 4:
        raise ValueError("formal Graph Builder and GAT input must keep H_preview=4")
    split_seeds = {
        str(name): [int(value) for value in seeds]
        for name, seeds in settings["split_seeds"].items()
    }
    validate_split_seeds(split_seeds)
    seeds = [int(value) for value in settings["seeds"]]
    if set(seeds) != {seed for values in split_seeds.values() for seed in values}:
        raise ValueError("configured seeds must exactly match the union of split seeds")
    h_labels = tuple(int(value) for value in settings["H_label"])
    timesteps = validate_sample_timesteps(
        settings["state_sample_timesteps"], maximum_label_horizon=max(h_labels)
    )
    if settings.get("selection_scope") != "train_validation_only":
        raise ValueError("H_label/quality/tau selection must use train+validation only")
    if settings.get("test_role") != "held_out_no_selection_before_design_freeze":
        raise ValueError("test split must remain held out before design freeze")
    checkpoint = Path(settings["checkpoint"])
    checkpoint = checkpoint if checkpoint.is_absolute() else REPO_ROOT / checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    supervision_config = CandidateSupervisionConfig.from_mapping({
        "h_preview": settings["H_preview"],
        "h_labels": h_labels,
        "diagnostic_preview_at_H_label": settings["diagnostic_preview_at_H_label"],
        "consumer_top_k": settings["K"],
        "soft_target_temperatures": settings["soft_target_temperatures"],
        "quality": settings["quality"],
    })
    proposal_config = ProposalConfig(**settings.get("proposal_config", {}))
    graph_config = HeterogeneousCandidateGraphConfig(**settings.get("graph_config", {}))
    multi_config = build_single_distribution_multi_config(
        num_agents=int(settings["num_agents"]),
        max_steps=int(settings["max_steps"]),
    )
    if not np.isclose(float(multi_config.time_step), float(settings["dt"])):
        raise ValueError("configured dt differs from checkpoint-aligned environment dt")
    output_dir.mkdir(parents=True, exist_ok=False)
    for name in (
        "graphs", "labels", "rollouts", "summary", "horizon_sensitivity",
        "label_distribution", "representative_cases", "trajectory_comparison",
    ):
        (output_dir / name).mkdir(parents=True, exist_ok=True)

    reference_env = build_single_env(config=SINGLE_AGENT_CONFIG, action_guidance_enabled=False)
    policy = build_single_model(reference_env, config=SINGLE_AGENT_CONFIG, verbose=0)
    load_checkpoint(policy, checkpoint)
    freeze_policy(policy)
    parameter_hash_before = _policy_parameter_sha256(policy)
    critical_hashes_before = _critical_hashes(checkpoint)
    resolved_config = copy.deepcopy(settings)
    resolved_config.update({
        "created_at": datetime.now().astimezone().isoformat(),
        "output_dir": str(output_dir.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": critical_hashes_before["checkpoint"],
        "policy_parameter_sha256_before": parameter_hash_before,
        "state_sample_timesteps_resolved": timesteps,
        "H_label_max": max(h_labels),
        "selection_uses_test_split": False,
        "GAT_training_started": False,
        "critical_file_hashes_before": critical_hashes_before,
        "quality_definition": supervision_config.quality.metadata(),
    })
    _write_json(output_dir / "config.json", resolved_config)

    manifest_rows: list[dict[str, Any]] = []
    graph_rows: list[dict[str, Any]] = []
    class_mapping_rows: list[dict[str, Any]] = []
    class_records: list[dict[str, Any]] = []
    sample_records: list[dict[str, Any]] = []
    state_group_rows: list[dict[str, Any]] = []
    unavailable_state_rows: list[dict[str, Any]] = []
    ego_sample_count = 0
    stop_requested = False
    started = time.perf_counter()
    try:
        for scene_type in [str(value) for value in settings["scenario_types"]]:
            if stop_requested:
                break
            for seed in seeds:
                if stop_requested:
                    break
                split = assign_seed_split(seed, split_seeds)
                env, scene_metadata = _build_supervision_environment(
                    config=multi_config,
                    scene_type=scene_type,
                    seed=seed,
                    peer_radius=float(settings["peer_radius"]),
                )
                reached_timesteps: set[int] = set()
                try:
                    maximum_timestep = max(timesteps)
                    for timestep in range(maximum_timestep + 1):
                        if timestep in timesteps:
                            reached_timesteps.add(timestep)
                            group_id = state_group_id(
                                scenario=scene_type,
                                seed=seed,
                                episode=int(settings.get("episode_index", 0)),
                                timestep=timestep,
                            )
                            state_group_rows.append({
                                "state_group_id": group_id,
                                "scenario": scene_type,
                                "seed": seed,
                                "episode": int(settings.get("episode_index", 0)),
                                "timestep": timestep,
                                "split": split,
                                "environment_fingerprint": environment_state_fingerprint(env),
                                "all_ego_share_split": True,
                                **scene_metadata,
                            })
                            snapshot_path = output_dir / "rollouts" / f"{group_id}__state.json"
                            _write_json(snapshot_path, _scene_snapshot(env))
                            for agent_index in [int(value) for value in settings["agent_indices"]]:
                                if max_ego_samples is not None and ego_sample_count >= int(max_ego_samples):
                                    stop_requested = True
                                    break
                                sample_id = f"{group_id}__ego{agent_index}"
                                proposal_start = time.perf_counter_ns()
                                proposals = _generate_proposals(env, agent_index, proposal_config)
                                proposal_runtime_ms = (time.perf_counter_ns() - proposal_start) / 1.0e6
                                evaluation_start = time.perf_counter_ns()
                                evaluation = evaluate_ego_supervision(
                                    env=env,
                                    agent_index=agent_index,
                                    proposals=proposals,
                                    policy=policy,
                                    proposal_config=proposal_config,
                                    config=supervision_config,
                                    graph_config=graph_config,
                                )
                                evaluation_runtime_ms = (time.perf_counter_ns() - evaluation_start) / 1.0e6
                                graph_path = output_dir / "graphs" / f"{sample_id}.pt"
                                torch.save(evaluation.graph, graph_path)
                                graph_rows.append({
                                    "sample_id": sample_id,
                                    "state_group_id": group_id,
                                    "scenario": scene_type,
                                    "seed": seed,
                                    "split": split,
                                    "timestep": timestep,
                                    "ego_agent_id": agent_index,
                                    "graph_path": _relative(graph_path, output_dir),
                                    "proposal_count_before_consumer": len(proposals),
                                    "K_requested": supervision_config.consumer_top_k,
                                    "K_actual": len(evaluation.selected_proposals),
                                    "class_count": len(evaluation.formal_previews),
                                    "null_class_index": 0,
                                    "H_preview": 4,
                                    "diagnostic_fields_in_graph": False,
                                    "proposal_generation_runtime_ms": proposal_runtime_ms,
                                    "total_ego_evaluation_runtime_ms": evaluation_runtime_ms,
                                    "environment_unchanged": evaluation.initial_state_fingerprint == evaluation.final_state_fingerprint,
                                })
                                for class_index, preview in enumerate(evaluation.formal_previews):
                                    class_mapping_rows.append({
                                        "sample_id": sample_id,
                                        "state_group_id": group_id,
                                        "split": split,
                                        "class_index": class_index,
                                        "class_kind": "null" if class_index == 0 else "proposal",
                                        "proposal_node_index": None if class_index == 0 else class_index - 1,
                                        "candidate_id": None if class_index == 0 else class_index - 1,
                                        "proposal_original_index": None if class_index == 0 else class_index - 1,
                                        "candidate_xyz": preview.trajectory.candidate_goal.tolist(),
                                    })
                                for h_label, horizon_evaluation in evaluation.horizon_evaluations.items():
                                    label_path = output_dir / "labels" / f"{sample_id}__H{h_label}.json"
                                    rollout_path = output_dir / "rollouts" / f"{sample_id}__H{h_label}.npz"
                                    _write_json(label_path, _label_payload(evaluation, horizon_evaluation))
                                    _save_rollout_bundle(
                                        rollout_path,
                                        evaluation=evaluation,
                                        horizon_evaluation=horizon_evaluation,
                                        env=env,
                                        agent_index=agent_index,
                                    )
                                    common = {
                                        "sample_id": sample_id,
                                        "state_group_id": group_id,
                                        "scenario": scene_type,
                                        "seed": seed,
                                        "episode": int(settings.get("episode_index", 0)),
                                        "timestep": timestep,
                                        "split": split,
                                        "ego_agent_id": agent_index,
                                        "H_label": h_label,
                                        "H_preview": 4,
                                    }
                                    sample_record = {**common, **horizon_evaluation.sample_record}
                                    sample_records.append(sample_record)
                                    for record in horizon_evaluation.class_records:
                                        class_records.append({**common, **record})
                                    manifest_rows.append({
                                        **common,
                                        "graph_path": _relative(graph_path, output_dir),
                                        "label_path": _relative(label_path, output_dir),
                                        "rollout_path": _relative(rollout_path, output_dir),
                                        "state_snapshot_path": _relative(snapshot_path, output_dir),
                                        "K_actual": len(evaluation.selected_proposals),
                                        "class_count": len(evaluation.formal_previews),
                                        "selection_eligible": split in {"train", "validation"},
                                        "test_used_for_selection": False,
                                    })
                                ego_sample_count += 1
                                print(
                                    f"[{ego_sample_count}] {sample_id}: proposals={len(proposals)} "
                                    f"K_t={len(evaluation.selected_proposals)} labels={list(h_labels)}",
                                    flush=True,
                                )
                            if stop_requested:
                                break
                        if timestep >= maximum_timestep:
                            break
                        terminated, truncated, info = _advance_environment_one_step(env, policy)
                        if terminated or truncated:
                            for missing in timesteps:
                                if missing > timestep and missing not in reached_timesteps:
                                    unavailable_state_rows.append({
                                        "scenario": scene_type,
                                        "seed": seed,
                                        "split": split,
                                        "requested_timestep": missing,
                                        "last_reached_timestep": int(getattr(env, "steps", timestep + 1)),
                                        "reason": "episode_terminated" if terminated else "episode_truncated",
                                        "state_fabricated": False,
                                        "collision": bool(info.get("collision", False)),
                                        "success": bool(info.get("success", False)),
                                    })
                            break
                finally:
                    env.close()
    finally:
        reference_env.close()

    parameter_hash_after = _policy_parameter_sha256(policy)
    critical_hashes_after = _critical_hashes(checkpoint)
    if parameter_hash_before != parameter_hash_after:
        raise RuntimeError("candidate supervision modified frozen SAC parameters")
    if critical_hashes_before != critical_hashes_after:
        raise RuntimeError("candidate supervision modified a critical implementation file")
    _write_csv(output_dir / "manifest.csv", manifest_rows)
    _write_csv(output_dir / "graph_records.csv", graph_rows)
    _write_csv(output_dir / "class_mapping.csv", class_mapping_rows)
    _write_csv(output_dir / "candidate_class_records.csv", class_records)
    _write_csv(output_dir / "sample_records.csv", sample_records)
    _write_csv(output_dir / "state_groups.csv", state_group_rows)
    _write_csv(output_dir / "unavailable_states.csv", unavailable_state_rows)
    generation_summary = {
        "schema_version": SUPERVISION_SCHEMA_VERSION,
        "elapsed_seconds": time.perf_counter() - started,
        "state_group_count": len(state_group_rows),
        "ego_graph_count": len(graph_rows),
        "label_variant_count": len(manifest_rows),
        "candidate_class_record_count": len(class_records),
        "unavailable_state_count": len(unavailable_state_rows),
        "split_state_group_counts": {
            split: sum(row["split"] == split for row in state_group_rows)
            for split in ("train", "validation", "test")
        },
        "early_termination_states_fabricated": False,
        "policy_parameter_sha256_before": parameter_hash_before,
        "policy_parameter_sha256_after": parameter_hash_after,
        "policy_parameters_unchanged": parameter_hash_before == parameter_hash_after,
        "critical_file_hashes_before": critical_hashes_before,
        "critical_file_hashes_after": critical_hashes_after,
        "critical_files_unchanged": critical_hashes_before == critical_hashes_after,
        "GAT_training_started": False,
        "selection_scope": "train_validation_only",
        "test_used_for_selection": False,
    }
    _write_json(output_dir / "summary" / "generation_summary.json", generation_summary)
    if run_audit and manifest_rows:
        from scripts.audit_candidate_supervision import audit_candidate_supervision_artifacts

        audit_candidate_supervision_artifacts(output_dir)
    return generation_summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--scenes", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--state-timesteps", nargs="+", type=int, default=None)
    parser.add_argument("--agents", nargs="+", type=int, default=None)
    parser.add_argument("--h-labels", nargs="+", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--max-ego-samples", type=int, default=None)
    parser.add_argument("--skip-audit", action="store_true")
    return parser.parse_args()


def main() -> Path:
    args = _parse_args()
    config_path = args.config.expanduser().resolve()
    settings = json.loads(config_path.read_text(encoding="utf-8"))
    if args.scenes is not None:
        settings["scenario_types"] = args.scenes
    if args.seeds is not None:
        settings["seeds"] = args.seeds
        settings["split_seeds"] = {
            name: [seed for seed in values if seed in set(args.seeds)]
            for name, values in settings["split_seeds"].items()
        }
    if args.state_timesteps is not None:
        settings["state_sample_timesteps"] = args.state_timesteps
    if args.agents is not None:
        settings["agent_indices"] = args.agents
    if args.h_labels is not None:
        settings["H_label"] = args.h_labels
    if args.k is not None:
        settings["K"] = args.k
        settings.setdefault("proposal_config", {})["top_k"] = args.k
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is None:
        base = Path(settings["output_dir"])
        base = base if base.is_absolute() else REPO_ROOT / base
        output_dir = base / timestamp
    else:
        output_dir = args.output_dir.expanduser().resolve()
    run_generation(
        settings,
        output_dir,
        max_ego_samples=args.max_ego_samples,
        run_audit=not args.skip_audit,
    )
    print(f"Artifacts written to: {output_dir}")
    return output_dir


if __name__ == "__main__":
    main()

