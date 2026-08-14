"""Controlled candidate-supervision evaluation for one ego-local graph state.

The formal graph is always constructed from the operational H=4 FP-SHEP
preview.  Optional previews at H_label are diagnostic-only and are never fed
to the graph or used as formal J_preview.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from Guidance.reference_point_proposal_demo import ProposalConfig
from planning.candidate_execution_benchmark import (
    RealCandidateRollout,
    environment_state_fingerprint,
    real_candidate_rollout,
)
from planning.candidate_execution_interface import (
    ExecutionNormalizationSpec,
    graph_ready_candidate_execution,
)
from planning.candidate_supervision import (
    FORMAL_PREVIEW_HORIZON,
    CandidateQualityBatch,
    CandidateQualitySpec,
    average_ranks,
    branch_is_failure,
    candidate_set_ranking_metrics,
    compute_candidate_quality,
    target_bundle,
    termination_reason,
)
from planning.heterogeneous_candidate_graph import (
    HeterogeneousCandidateGraphConfig,
    build_heterogeneous_candidate_graph_from_env,
)
from planning.policy_preview import (
    CandidatePreview,
    adapt_candidate_proposals,
    build_preview_inputs_from_env,
    preview_candidate,
)


@dataclass(frozen=True)
class CandidateSupervisionConfig:
    h_preview: int = FORMAL_PREVIEW_HORIZON
    h_labels: tuple[int, ...] = (4, 6, 8)
    diagnostic_preview_at_h_label: bool = True
    consumer_top_k: int = 10
    soft_target_temperatures: tuple[float, ...] = (0.10, 0.25, 0.50)
    quality: CandidateQualitySpec = field(default_factory=CandidateQualitySpec)

    def __post_init__(self) -> None:
        if int(self.h_preview) != FORMAL_PREVIEW_HORIZON:
            raise ValueError("formal Graph Builder preview horizon must remain H_preview=4")
        horizons = tuple(int(value) for value in self.h_labels)
        if not horizons or any(value <= 0 for value in horizons):
            raise ValueError("h_labels must be a non-empty positive sequence")
        if len(set(horizons)) != len(horizons):
            raise ValueError("h_labels must be unique")
        if int(self.consumer_top_k) <= 0:
            raise ValueError("consumer_top_k must be positive")
        temperatures = tuple(float(value) for value in self.soft_target_temperatures)
        if not temperatures or any(not np.isfinite(value) or value <= 0.0 for value in temperatures):
            raise ValueError("soft-target temperatures must be positive and finite")
        object.__setattr__(self, "h_preview", FORMAL_PREVIEW_HORIZON)
        object.__setattr__(self, "h_labels", horizons)
        object.__setattr__(self, "consumer_top_k", int(self.consumer_top_k))
        object.__setattr__(self, "soft_target_temperatures", temperatures)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "CandidateSupervisionConfig":
        return cls(
            h_preview=int(values.get("h_preview", FORMAL_PREVIEW_HORIZON)),
            h_labels=tuple(int(value) for value in values.get("h_labels", (4, 6, 8))),
            diagnostic_preview_at_h_label=bool(
                values.get("diagnostic_preview_at_H_label", True)
            ),
            consumer_top_k=int(values.get("consumer_top_k", 10)),
            soft_target_temperatures=tuple(
                float(value) for value in values.get(
                    "soft_target_temperatures", (0.10, 0.25, 0.50)
                )
            ),
            quality=CandidateQualitySpec.from_mapping(values.get("quality", {})),
        )


@dataclass
class HorizonSupervisionEvaluation:
    h_label: int
    class_records: list[dict[str, Any]]
    sample_record: dict[str, Any]
    real_rollouts: tuple[RealCandidateRollout, ...]
    diagnostic_previews: tuple[CandidatePreview, ...]
    real_quality: CandidateQualityBatch
    target_bundle: dict[str, Any]


@dataclass
class EgoSupervisionEvaluation:
    graph: Any
    selected_proposals: tuple[Any, ...]
    formal_previews: tuple[CandidatePreview, ...]
    formal_preview_quality: CandidateQualityBatch
    horizon_evaluations: dict[int, HorizonSupervisionEvaluation]
    initial_state_fingerprint: str
    final_state_fingerprint: str
    visible_surface_points: np.ndarray


def _quality_row(value: Any) -> dict[str, float]:
    return {
        "task_progress": float(value.task_progress),
        "min_clearance": float(value.min_clearance),
        "max_execution_deviation": float(value.max_execution_deviation),
        "terminal_speed": float(value.terminal_speed),
    }


def _proposal_metadata(proposal: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in vars(proposal).items():
        if name == "point":
            continue
        result[name] = value.copy() if isinstance(value, np.ndarray) else value
    return result


def _position_errors(
    preview: CandidatePreview,
    rollout: RealCandidateRollout,
) -> tuple[np.ndarray, np.ndarray]:
    count = min(
        int(preview.trajectory.positions.shape[0]),
        int(rollout.positions.shape[0]),
    )
    errors = np.linalg.norm(
        preview.trajectory.positions[:count] - rollout.positions[:count], axis=1
    )
    steps = np.arange(count, dtype=np.int64)
    return steps, errors


def _rank_vector(values: np.ndarray) -> np.ndarray:
    return average_ranks(np.asarray(values, dtype=float), descending=True)


def _quality_fields(prefix: str, quality: CandidateQualityBatch, index: int) -> dict[str, Any]:
    normalized = quality.normalized
    return {
        f"{prefix}_task_progress_normalized": float(normalized.values[index, 0]),
        f"{prefix}_min_clearance_normalized": float(normalized.values[index, 1]),
        f"{prefix}_max_execution_deviation_normalized": float(normalized.values[index, 2]),
        f"{prefix}_terminal_speed_normalized": float(normalized.values[index, 3]),
        f"{prefix}_feature_valid_mask": normalized.valid_mask[index].tolist(),
        f"{prefix}_feature_clipped_mask": normalized.clipped_mask[index].tolist(),
        f"{prefix}_clearance_finite_mask": bool(normalized.clearance_finite_mask[index]),
        f"{prefix}_clearance_positive_inf_flag": bool(
            normalized.clearance_positive_inf_flag[index]
        ),
        f"{prefix}_quality_3_nominal": float(quality.nominal_three_feature[index]),
        f"{prefix}_quality_4_nominal": float(quality.nominal_four_feature[index]),
        f"{prefix}_quality_3_target": float(quality.target_three_feature[index]),
        f"{prefix}_quality_4_target": float(quality.target_four_feature[index]),
    }


def _build_formal_previews(
    *,
    env: Any,
    agent_index: int,
    class_goals: Sequence[np.ndarray],
    policy: Any,
    horizon: int,
) -> tuple[CandidatePreview, ...]:
    initial_state, local_context = build_preview_inputs_from_env(env, agent_index)
    previews = tuple(
        preview_candidate(
            initial_state=initial_state,
            local_context=local_context,
            candidate_goal=goal,
            policy=policy,
            horizon=horizon,
            dmp_config=env.dmps[agent_index].config,
            dynamics=env.dynamics[agent_index],
        )
        for goal in class_goals
    )
    return previews


def evaluate_ego_supervision(
    *,
    env: Any,
    agent_index: int,
    proposals: Sequence[Any],
    policy: Any,
    proposal_config: ProposalConfig,
    config: CandidateSupervisionConfig | None = None,
    graph_config: HeterogeneousCandidateGraphConfig | None = None,
    execution_normalization: ExecutionNormalizationSpec | None = None,
) -> EgoSupervisionEvaluation:
    """Build one formal graph and all offline label-horizon branches.

    Class index 0 is the null branch and uses the terminal task goal as its
    active guidance goal.  Class indices 1..K_t map one-to-one to the graph's
    proposal nodes 0..K_t-1.  The source environment is fingerprinted before
    and after every operation.
    """

    config = config or CandidateSupervisionConfig()
    agent_index = int(agent_index)
    if not 0 <= agent_index < int(env.num_agents):
        raise IndexError("agent_index is out of range")
    before = environment_state_fingerprint(env)
    selected = tuple(
        adapt_candidate_proposals(proposals, consumer_top_k=config.consumer_top_k)
    )
    task_goal = np.asarray(env.goals[agent_index], dtype=float).copy()
    class_goals = (task_goal,) + tuple(np.asarray(item.point, dtype=float).copy() for item in selected)
    formal_previews = _build_formal_previews(
        env=env,
        agent_index=agent_index,
        class_goals=class_goals,
        policy=policy,
        horizon=config.h_preview,
    )
    formal_quality = compute_candidate_quality(
        [_quality_row(item) for item in formal_previews],
        failure_mask=np.zeros(len(formal_previews), dtype=bool),
        spec=config.quality,
    )
    graph_executions = tuple(
        graph_ready_candidate_execution(candidate_id, formal_previews[candidate_id + 1])
        for candidate_id in range(len(selected))
    )
    resolved_graph_config = graph_config or HeterogeneousCandidateGraphConfig(
        horizon_steps=FORMAL_PREVIEW_HORIZON
    )
    if int(resolved_graph_config.horizon_steps) != FORMAL_PREVIEW_HORIZON:
        raise ValueError("supervision graph must use the operational H_preview=4")
    graph = build_heterogeneous_candidate_graph_from_env(
        env=env,
        agent_index=agent_index,
        proposals=selected,
        executions=graph_executions,
        proposal_config=proposal_config,
        config=resolved_graph_config,
        execution_normalization=execution_normalization,
    )
    if int(graph.graph_metadata["H"]) != FORMAL_PREVIEW_HORIZON:
        raise RuntimeError("formal graph contains a non-operational preview horizon")
    if int(graph["proposal"].num_nodes) != len(selected):
        raise RuntimeError("graph proposal count does not match selected candidates")

    proposal_scores = np.asarray(
        [float("nan")] + [float(item.score) for item in selected], dtype=float
    )
    formal_three = formal_quality.target_three_feature
    formal_four = formal_quality.target_four_feature
    formal_three_rank = _rank_vector(formal_three)
    formal_four_rank = _rank_vector(formal_four)
    horizon_evaluations: dict[int, HorizonSupervisionEvaluation] = {}

    for h_label in config.h_labels:
        if config.diagnostic_preview_at_h_label:
            diagnostic_previews = (
                formal_previews
                if h_label == config.h_preview
                else _build_formal_previews(
                    env=env,
                    agent_index=agent_index,
                    class_goals=class_goals,
                    policy=policy,
                    horizon=h_label,
                )
            )
        else:
            diagnostic_previews = formal_previews
        real_rollouts = tuple(
            real_candidate_rollout(
                initial_env=env,
                agent_index=agent_index,
                candidate_goal=goal,
                policy=policy,
                horizon=h_label,
                preview=(diagnostic_previews[index] if config.diagnostic_preview_at_h_label else None),
            )
            for index, goal in enumerate(class_goals)
        )
        real_failure = np.asarray([branch_is_failure(item) for item in real_rollouts], dtype=bool)
        real_quality = compute_candidate_quality(
            [_quality_row(item) for item in real_rollouts],
            failure_mask=real_failure,
            spec=config.quality,
        )
        diagnostic_quality = compute_candidate_quality(
            [_quality_row(item) for item in diagnostic_previews],
            failure_mask=np.zeros(len(diagnostic_previews), dtype=bool),
            spec=config.quality,
        )
        targets = target_bundle(real_quality, config.soft_target_temperatures)
        real_three = real_quality.target_three_feature
        real_four = real_quality.target_four_feature
        real_three_rank = _rank_vector(real_three)
        real_four_rank = _rank_vector(real_four)
        metrics_three = candidate_set_ranking_metrics(
            proposal_scores_with_null=proposal_scores,
            formal_preview_quality=formal_three,
            real_target_quality=real_three,
        )
        metrics_four = candidate_set_ranking_metrics(
            proposal_scores_with_null=proposal_scores,
            formal_preview_quality=formal_four,
            real_target_quality=real_four,
        )
        diagnostic_metrics_three = candidate_set_ranking_metrics(
            proposal_scores_with_null=proposal_scores,
            formal_preview_quality=diagnostic_quality.target_three_feature,
            real_target_quality=real_three,
        )
        class_records: list[dict[str, Any]] = []
        for class_index, (goal, formal, diagnostic, real) in enumerate(
            zip(class_goals, formal_previews, diagnostic_previews, real_rollouts, strict=True)
        ):
            proposal_index = class_index - 1
            proposal = selected[proposal_index] if proposal_index >= 0 else None
            steps, position_errors = _position_errors(diagnostic, real)
            record: dict[str, Any] = {
                "class_index": class_index,
                "class_kind": "null" if class_index == 0 else "proposal",
                "class_label": "null" if class_index == 0 else f"candidate_{proposal_index}",
                "candidate_id": None if class_index == 0 else proposal_index,
                "proposal_original_index": None if class_index == 0 else proposal_index,
                "proposal_rank_zero_based": None if class_index == 0 else proposal_index,
                "candidate_xyz": goal.tolist(),
                "proposal_score": None if proposal is None else float(proposal.score),
                "H_preview_formal": config.h_preview,
                "H_label": int(h_label),
                "formal_preview_task_progress": float(formal.task_progress),
                "formal_preview_min_clearance": float(formal.min_clearance),
                "formal_preview_max_execution_deviation": float(formal.max_execution_deviation),
                "formal_preview_terminal_speed": float(formal.terminal_speed),
                "formal_preview_runtime_ms": float(formal.performance.total_ms),
                "formal_J_preview_3": float(formal_three[class_index]),
                "formal_J_preview_4": float(formal_four[class_index]),
                "formal_J_preview_3_rank": float(formal_three_rank[class_index]),
                "formal_J_preview_4_rank": float(formal_four_rank[class_index]),
                "diagnostic_preview_at_H_label": bool(config.diagnostic_preview_at_h_label),
                "diagnostic_preview_excluded_from_graph": True,
                "diagnostic_preview_excluded_from_formal_J_preview": True,
                "diagnostic_preview_task_progress": float(diagnostic.task_progress),
                "diagnostic_preview_min_clearance": float(diagnostic.min_clearance),
                "diagnostic_preview_max_execution_deviation": float(diagnostic.max_execution_deviation),
                "diagnostic_preview_terminal_speed": float(diagnostic.terminal_speed),
                "diagnostic_preview_runtime_ms": float(diagnostic.performance.total_ms),
                "diagnostic_quality_3": float(diagnostic_quality.target_three_feature[class_index]),
                "diagnostic_quality_4": float(diagnostic_quality.target_four_feature[class_index]),
                "real_task_progress": float(real.task_progress),
                "real_min_clearance": float(real.min_clearance),
                "real_max_execution_deviation": float(real.max_execution_deviation),
                "real_terminal_speed": float(real.terminal_speed),
                "J_target_3": float(real_three[class_index]),
                "J_target_4": float(real_four[class_index]),
                "J_target_3_nominal": float(real_quality.nominal_three_feature[class_index]),
                "J_target_4_nominal": float(real_quality.nominal_four_feature[class_index]),
                "J_target_3_rank": float(real_three_rank[class_index]),
                "J_target_4_rank": float(real_four_rank[class_index]),
                "target_role_3": "provisional_primary_target",
                "target_role_4": "companion_target",
                "final_supervision_target_frozen": False,
                "collision": bool(real.collision),
                "obstacle_collision": bool(real.obstacle_collision),
                "inter_agent_collision": bool(real.inter_agent_collision),
                "boundary_collision": bool(real.boundary_collision),
                "success": bool(real.success),
                "agent_success": bool(real.agent_success),
                "terminated": bool(real.terminated),
                "truncated": bool(real.truncated),
                "completed_label_horizon": int(real.effective_steps) == int(h_label),
                "effective_steps": int(real.effective_steps),
                "termination_reason": termination_reason(real, h_label),
                "failure_tier": bool(real_failure[class_index]),
                "minimum_real_inter_agent_distance": float(real.minimum_inter_agent_distance),
                "real_runtime_ms": float(real.runtime_ms),
                "real_clearance_source": str(real.clearance_source),
                "preview_clearance_source": str(formal.metadata["clearance_source"]),
                "preview_clearance_is_approximate": bool(formal.metadata["clearance_is_approximate"]),
                "preview_real_error_steps": steps.tolist(),
                "preview_real_position_errors": position_errors.tolist(),
                "preview_real_mean_position_error": float(np.mean(position_errors[1:])) if len(position_errors) > 1 else 0.0,
                "preview_real_terminal_position_error": float(position_errors[-1]),
                **_quality_fields("formal_preview", formal_quality, class_index),
                **_quality_fields("diagnostic_preview", diagnostic_quality, class_index),
                **_quality_fields("real", real_quality, class_index),
            }
            if proposal is not None:
                record.update({
                    f"proposal_{key}": value.tolist() if isinstance(value, np.ndarray) else value
                    for key, value in _proposal_metadata(proposal).items()
                    if key != "score"
                })
            for temperature_key, soft in targets["soft_targets"].items():
                record[f"soft_primary_{temperature_key}"] = float(
                    soft["provisional_primary_target"][class_index]
                )
                record[f"soft_companion_{temperature_key}"] = float(
                    soft["companion_target"][class_index]
                )
            class_records.append(record)

        sample_record: dict[str, Any] = {
            "H_preview_formal": config.h_preview,
            "H_label": int(h_label),
            "K_requested": config.consumer_top_k,
            "K_actual": len(selected),
            "class_count": len(class_goals),
            "null_class_index": 0,
            "graph_proposal_count": int(graph["proposal"].num_nodes),
            "graph_spatiotemporal_edge_count": int(
                graph["align", "spatiotemporal", "proposal"].edge_index.shape[1]
            ),
            "formal_graph_uses_H_preview_4": True,
            "diagnostic_preview_at_H_label": bool(config.diagnostic_preview_at_h_label),
            "diagnostic_preview_offline_only": True,
            "provisional_primary_target": "quality_3_feature",
            "companion_target": "quality_4_feature",
            "final_supervision_target_frozen": False,
            "hard_target_3_class_index": int(targets["hard_target_three_feature"]),
            "hard_target_4_class_index": int(targets["hard_target_four_feature"]),
            "null_top1_3": float(int(targets["hard_target_three_feature"]) == 0),
            "null_top1_4": float(int(targets["hard_target_four_feature"]) == 0),
            "collision_branch_ratio": float(np.mean(real_failure)),
            "success_branch_ratio": float(np.mean([item.success for item in real_rollouts])),
            "truncated_branch_ratio": float(np.mean([item.truncated for item in real_rollouts])),
            "invalid_branch_ratio": float(np.mean([
                not bool(np.all(real_quality.normalized.valid_mask[index]))
                for index in range(len(real_rollouts))
            ])),
            "mean_real_runtime_per_candidate_ms": float(np.mean([item.runtime_ms for item in real_rollouts])),
            "median_real_runtime_per_candidate_ms": float(np.median([item.runtime_ms for item in real_rollouts])),
            "total_real_runtime_ms": float(np.sum([item.runtime_ms for item in real_rollouts])),
            "total_formal_preview_runtime_ms": float(np.sum([item.performance.total_ms for item in formal_previews])),
            "total_diagnostic_preview_runtime_ms": float(np.sum([item.performance.total_ms for item in diagnostic_previews])),
            "mean_preview_real_position_error": float(np.mean([
                record["preview_real_mean_position_error"] for record in class_records
            ])),
            "median_preview_real_position_error": float(np.median([
                record["preview_real_mean_position_error"] for record in class_records
            ])),
            **{f"quality3_{key}": value for key, value in metrics_three.items()},
            **{f"quality4_{key}": value for key, value in metrics_four.items()},
            "diagnostic_quality3_spearman_all_classes": diagnostic_metrics_three[
                "fp_shep_spearman_all_classes"
            ],
            "diagnostic_quality3_top1_hit_full_target": diagnostic_metrics_three[
                "fp_shep_top1_hit_full_target"
            ],
            "diagnostic_quality3_pairwise_accuracy_all_classes": diagnostic_metrics_three[
                "fp_shep_pairwise_accuracy_all_classes"
            ],
        }
        for temperature_key, soft in targets["soft_targets"].items():
            safe_key = temperature_key.replace(".", "p")
            for variant_name, statistics_key in (
                ("primary", "primary_statistics"),
                ("companion", "companion_statistics"),
            ):
                for statistic_name, value in soft[statistics_key].items():
                    sample_record[
                        f"soft_{variant_name}_{safe_key}_{statistic_name}"
                    ] = float(value)
        horizon_evaluations[int(h_label)] = HorizonSupervisionEvaluation(
            h_label=int(h_label),
            class_records=class_records,
            sample_record=sample_record,
            real_rollouts=real_rollouts,
            diagnostic_previews=tuple(diagnostic_previews),
            real_quality=real_quality,
            target_bundle=targets,
        )

    after = environment_state_fingerprint(env)
    if before != after:
        raise RuntimeError("candidate supervision mutated its source environment")
    _, local_context = build_preview_inputs_from_env(env, agent_index)
    return EgoSupervisionEvaluation(
        graph=graph,
        selected_proposals=selected,
        formal_previews=formal_previews,
        formal_preview_quality=formal_quality,
        horizon_evaluations=horizon_evaluations,
        initial_state_fingerprint=before,
        final_state_fingerprint=after,
        visible_surface_points=local_context.visible_surface_points.copy(),
    )
