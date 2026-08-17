"""Pure contracts and analysis for 220-step Pre-GAT revalidation.

The module contains no training logic.  Candidate bundles are represented by
frozen scalar/tuple values so Proposal and FP-SHEP selectors can consume the
same logical t=0 bundle without sharing mutable NumPy arrays.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from Guidance.reference_point_proposal_demo import ProposalConfig
from planning.pre_gat_closed_loop import generate_candidate_set


SCHEMA_VERSION = "pre_gat_220step_revalidation_v1"
METHOD_TERMINAL = "terminal_goal_baseline"
METHOD_PROPOSAL = "proposal_top1_one_shot"
METHOD_FP_SHEP = "fp_shep_top1_one_shot"
METHOD_ORDER = (METHOD_TERMINAL, METHOD_PROPOSAL, METHOD_FP_SHEP)
METHOD_DISPLAY_NAMES = {
    METHOD_TERMINAL: "Terminal-Goal Baseline",
    METHOD_PROPOSAL: "Proposal Top-1",
    METHOD_FP_SHEP: "FP-SHEP Top-1",
}
NO_CANDIDATE_PAIR = "NO_CANDIDATE_PAIR"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def stable_hash(value: Any) -> str:
    payload = json.dumps(
        _jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ImmutableCandidate:
    """Defensive, hash-stable representation of one existing Proposal."""

    original_index: int
    world_point: tuple[float, float, float]
    proposal_score: float
    metadata_json: str

    @classmethod
    def from_proposal(cls, proposal: Any, original_index: int) -> "ImmutableCandidate":
        point = np.asarray(proposal.point, dtype=float)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError("candidate point must be a finite 3-vector")
        metadata = {
            key: _jsonable(value)
            for key, value in vars(proposal).items()
            if key not in {"point", "score"}
        }
        return cls(
            original_index=int(original_index),
            world_point=tuple(float(value) for value in point),
            proposal_score=float(proposal.score),
            metadata_json=json.dumps(
                metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ),
        )

    @property
    def point(self) -> np.ndarray:
        """Return a defensive array; callers cannot mutate bundle storage."""

        return np.asarray(self.world_point, dtype=float).copy()

    @property
    def score(self) -> float:
        return float(self.proposal_score)

    @property
    def metadata(self) -> dict[str, Any]:
        return json.loads(self.metadata_json)

    def record(self) -> dict[str, Any]:
        return {
            "original_index": int(self.original_index),
            "candidate_world_point": list(self.world_point),
            "proposal_score": float(self.proposal_score),
            "candidate_metadata": self.metadata,
        }


@dataclass(frozen=True)
class ImmutableCandidateBundle:
    scenario: str
    seed: int
    per_agent: tuple[tuple[ImmutableCandidate, ...], ...]
    count_before_consumer: tuple[int, ...]

    @property
    def num_agents(self) -> int:
        return len(self.per_agent)

    @property
    def candidate_set_hash(self) -> str:
        return stable_hash(self.record())

    def record(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "seed": int(self.seed),
            "count_before_consumer": list(self.count_before_consumer),
            "per_agent": [
                [candidate.record() for candidate in candidates]
                for candidates in self.per_agent
            ],
        }


def generate_immutable_candidate_bundle(
    env: Any,
    *,
    scenario: str,
    seed: int,
    proposal_config: ProposalConfig,
    consumer_top_k: int,
) -> ImmutableCandidateBundle:
    """Generate each agent's t=0 candidates exactly once and freeze them."""

    per_agent: list[tuple[ImmutableCandidate, ...]] = []
    counts: list[int] = []
    for agent_index in range(int(env.num_agents)):
        proposals, count_before = generate_candidate_set(
            env,
            agent_index,
            proposal_config,
            consumer_top_k=int(consumer_top_k),
        )
        per_agent.append(
            tuple(
                ImmutableCandidate.from_proposal(proposal, index)
                for index, proposal in enumerate(proposals)
            )
        )
        counts.append(int(count_before))
    return ImmutableCandidateBundle(
        scenario=str(scenario),
        seed=int(seed),
        per_agent=tuple(per_agent),
        count_before_consumer=tuple(counts),
    )


def candidate_bundle_rows(bundle: ImmutableCandidateBundle) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    bundle_hash = bundle.candidate_set_hash
    for agent_id, candidates in enumerate(bundle.per_agent):
        if not candidates:
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "scenario": bundle.scenario,
                    "seed": bundle.seed,
                    "agent_id": agent_id,
                    "candidate_index": None,
                    "K_t": 0,
                    "candidate_set_hash": bundle_hash,
                    "no_candidate_pair": True,
                }
            )
            continue
        for candidate in candidates:
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "scenario": bundle.scenario,
                    "seed": bundle.seed,
                    "agent_id": agent_id,
                    "candidate_index": candidate.original_index,
                    "K_t": len(candidates),
                    "candidate_set_hash": bundle_hash,
                    "no_candidate_pair": False,
                    **candidate.record(),
                }
            )
    return rows


def mean_or_none(values: Iterable[Any]) -> float | None:
    finite = [
        float(value)
        for value in values
        if value is not None and np.isfinite(float(value))
    ]
    return float(np.mean(finite)) if finite else None


def rate(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def classify_failure(agent_rows: Sequence[Mapping[str, Any]]) -> tuple[str, dict[str, bool]]:
    """Return one mutually-exclusive primary category plus overlapping flags."""

    flags = {
        "A_no_valid_candidate": any(bool(row["no_candidate_fallback"]) for row in agent_rows),
        "B_reference_not_reached_before_timeout": any(
            bool(row["reference_available"])
            and not bool(row["reference_reached"])
            and bool(row["team_timeout"])
            for row in agent_rows
        ),
        "C_obstacle_collision_before_reference": any(
            bool(row["obstacle_collision_before_reference"]) for row in agent_rows
        ),
        "D_inter_agent_collision_before_reference": any(
            bool(row["inter_agent_collision_before_reference"]) for row in agent_rows
        ),
        "E_collision_after_handoff": any(
            bool(row["collision_after_reference"]) for row in agent_rows
        ),
        "F_reference_reached_terminal_timeout": any(
            bool(row["reference_reached"])
            and not bool(row["terminal_completed_after_reference"])
            and bool(row["team_timeout"])
            for row in agent_rows
        ),
        "G_partial_reference_team_failure": (
            any(bool(row["reference_reached"]) for row in agent_rows)
            and any(
                bool(row["reference_available"]) and not bool(row["reference_reached"])
                for row in agent_rows
            )
        ),
    }
    priority = (
        "A_no_valid_candidate",
        "C_obstacle_collision_before_reference",
        "D_inter_agent_collision_before_reference",
        "G_partial_reference_team_failure",
        "B_reference_not_reached_before_timeout",
        "E_collision_after_handoff",
        "F_reference_reached_terminal_timeout",
    )
    primary = next((name for name in priority if flags[name]), "H_unknown")
    flags["H_unknown"] = primary == "H_unknown"
    return primary, flags


def aggregate_method_rows(
    episode_rows: Sequence[Mapping[str, Any]],
    agent_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        method_episodes = [row for row in episode_rows if row["method"] == method]
        scopes = ["overall"] + sorted({str(row["scenario"]) for row in method_episodes})
        for scope in scopes:
            episodes = (
                method_episodes
                if scope == "overall"
                else [row for row in method_episodes if row["scenario"] == scope]
            )
            agents = [
                row
                for row in agent_rows
                if row["method"] == method
                and (scope == "overall" or row["scenario"] == scope)
            ]
            eligible = [row for row in agents if bool(row["reference_available"])]
            reached = [row for row in eligible if bool(row["reference_reached"])]
            completed = [
                row for row in reached if bool(row["terminal_completed_after_reference"])
            ]
            before_collision = [
                row for row in eligible if bool(row["collision_before_reference"])
            ]
            before_obstacle_collision = [
                row
                for row in eligible
                if bool(row.get("obstacle_collision_before_reference", False))
            ]
            before_inter_agent_collision = [
                row
                for row in eligible
                if bool(row.get("inter_agent_collision_before_reference", False))
            ]
            after_collision = [
                row for row in reached if bool(row["collision_after_reference"])
            ]
            stage1_timeouts = [row for row in eligible if bool(row["reference_timeout"])]
            stage2_timeouts = [row for row in reached if bool(row["timeout_after_reference"])]
            success_count = sum(bool(row["team_success"]) for row in episodes)
            results.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "method": method,
                    "method_display_name": METHOD_DISPLAY_NAMES[method],
                    "scenario": scope,
                    "episode_count": len(episodes),
                    "team_success_count": success_count,
                    "team_success_rate": rate(success_count, len(episodes)),
                    "team_collision_count": sum(bool(row["collision"]) for row in episodes),
                    "team_collision_rate": rate(
                        sum(bool(row["collision"]) for row in episodes), len(episodes)
                    ),
                    "obstacle_collision_rate": rate(
                        sum(bool(row["obstacle_collision"]) for row in episodes), len(episodes)
                    ),
                    "inter_agent_collision_rate": rate(
                        sum(bool(row["inter_agent_collision"]) for row in episodes), len(episodes)
                    ),
                    "timeout_rate": rate(
                        sum(bool(row["timeout"]) for row in episodes), len(episodes)
                    ),
                    "obstacle_interaction_episode_count": sum(
                        bool(row.get("obstacle_interaction_episode", False))
                        for row in episodes
                    ),
                    "obstacle_interaction_episode_rate": rate(
                        sum(
                            bool(row.get("obstacle_interaction_episode", False))
                            for row in episodes
                        ),
                        len(episodes),
                    ),
                    "inter_agent_interaction_episode_count": sum(
                        bool(row.get("inter_agent_interaction_episode", False))
                        for row in episodes
                    ),
                    "inter_agent_interaction_episode_rate": rate(
                        sum(
                            bool(row.get("inter_agent_interaction_episode", False))
                            for row in episodes
                        ),
                        len(episodes),
                    ),
                    "combined_conflict_episode_count": sum(
                        bool(row.get("combined_conflict_episode", False))
                        for row in episodes
                    ),
                    "combined_conflict_episode_rate": rate(
                        sum(
                            bool(row.get("combined_conflict_episode", False))
                            for row in episodes
                        ),
                        len(episodes),
                    ),
                    "mean_minimum_static_obstacle_clearance_m": mean_or_none(
                        row.get("minimum_static_obstacle_clearance_m")
                        for row in episodes
                    ),
                    "mean_obstacle_pressure_path_deviation_m": mean_or_none(
                        row.get("obstacle_pressure_path_deviation_team_mean_m")
                        for row in episodes
                    ),
                    "mean_maximum_terminal_line_deviation_m": mean_or_none(
                        row.get("maximum_terminal_line_deviation_team_mean_m")
                        for row in episodes
                    ),
                    "reference_available_count": len(eligible),
                    "reference_reached_count": len(reached),
                    "reference_reached_rate": rate(len(reached), len(eligible)),
                    "reference_collision_count": len(before_collision),
                    "reference_collision_rate": rate(len(before_collision), len(eligible)),
                    "obstacle_collision_before_reference_count": len(
                        before_obstacle_collision
                    ),
                    "obstacle_collision_before_reference_rate": rate(
                        len(before_obstacle_collision), len(eligible)
                    ),
                    "inter_agent_collision_before_reference_count": len(
                        before_inter_agent_collision
                    ),
                    "inter_agent_collision_before_reference_rate": rate(
                        len(before_inter_agent_collision), len(eligible)
                    ),
                    "reference_timeout_count": len(stage1_timeouts),
                    "reference_timeout_rate": rate(len(stage1_timeouts), len(eligible)),
                    "mean_reference_reached_step": mean_or_none(
                        row["reference_reached_step"] for row in reached
                    ),
                    "mean_distance_to_reference_at_termination_m": mean_or_none(
                        row["distance_to_reference_at_termination_m"] for row in eligible
                    ),
                    "stage1_minimum_obstacle_clearance_m": mean_or_none(
                        row["stage1_minimum_obstacle_clearance_m"] for row in eligible
                    ),
                    "stage1_minimum_inter_agent_distance_m": mean_or_none(
                        row["stage1_minimum_inter_agent_distance_m"] for row in eligible
                    ),
                    "reached_then_terminal_count": len(completed),
                    "reached_then_terminal_rate": rate(len(completed), len(reached)),
                    "collision_after_reference_count": len(after_collision),
                    "collision_after_reference_rate": rate(len(after_collision), len(reached)),
                    "timeout_after_reference_count": len(stage2_timeouts),
                    "timeout_after_reference_rate": rate(len(stage2_timeouts), len(reached)),
                    "mean_terminal_completion_step": mean_or_none(
                        row["terminal_completed_step"] for row in completed
                    ),
                    "mean_remaining_steps_after_reference": mean_or_none(
                        row["remaining_steps_after_reference"] for row in reached
                    ),
                    "mean_path_length_m": mean_or_none(row["path_length_m"] for row in episodes),
                    "mean_trajectory_smoothness": mean_or_none(
                        row["trajectory_smoothness"] for row in episodes
                    ),
                    "mean_terminal_progress_m": mean_or_none(
                        row["terminal_progress_m"] for row in episodes
                    ),
                    "mean_completion_step_success": mean_or_none(
                        row["steps"] for row in episodes if bool(row["team_success"])
                    ),
                    "no_candidate_fallback_count": sum(
                        bool(row["no_candidate_fallback"]) for row in agents
                    ),
                }
            )
    return results


def build_selector_pairing(
    episode_rows: Sequence[Mapping[str, Any]],
    agent_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped_agents: dict[tuple[str, int, int], dict[str, Mapping[str, Any]]] = {}
    for row in agent_rows:
        if row["method"] not in {METHOD_PROPOSAL, METHOD_FP_SHEP}:
            continue
        key = (str(row["scenario"]), int(row["seed"]), int(row["agent_id"]))
        grouped_agents.setdefault(key, {})[str(row["method"])] = row

    episode_index = {
        (str(row["scenario"]), int(row["seed"]), str(row["method"])): row
        for row in episode_rows
    }
    paired: list[dict[str, Any]] = []
    integrity_errors: list[str] = []
    for key, methods in sorted(grouped_agents.items()):
        if set(methods) != {METHOD_PROPOSAL, METHOD_FP_SHEP}:
            integrity_errors.append(f"missing_selector_pair:{key}")
            continue
        proposal = methods[METHOD_PROPOSAL]
        fp = methods[METHOD_FP_SHEP]
        hashes_match = proposal["candidate_set_hash"] == fp["candidate_set_hash"]
        order_match = proposal["candidate_order_hash"] == fp["candidate_order_hash"]
        if not hashes_match or not order_match:
            integrity_errors.append(f"candidate_bundle_mismatch:{key}")
        no_candidate = bool(proposal["no_candidate_fallback"])
        if no_candidate != bool(fp["no_candidate_fallback"]):
            integrity_errors.append(f"fallback_mismatch:{key}")
        disagreement = bool(
            not no_candidate
            and proposal["selected_candidate_index"] != fp["selected_candidate_index"]
        )
        if no_candidate:
            reach_class = NO_CANDIDATE_PAIR
        elif proposal["reference_reached"] and fp["reference_reached"]:
            reach_class = "BOTH_REACH"
        elif proposal["reference_reached"]:
            reach_class = "PROPOSAL_ONLY_REACH"
        elif fp["reference_reached"]:
            reach_class = "FP_SHEP_ONLY_REACH"
        else:
            reach_class = "BOTH_FAIL"
        p_episode = episode_index[(key[0], key[1], METHOD_PROPOSAL)]
        f_episode = episode_index[(key[0], key[1], METHOD_FP_SHEP)]
        if no_candidate:
            team_class = NO_CANDIDATE_PAIR
        elif p_episode["team_success"] and f_episode["team_success"]:
            team_class = "BOTH_SUCCESS"
        elif p_episode["team_success"]:
            team_class = "PROPOSAL_ONLY_SUCCESS"
        elif f_episode["team_success"]:
            team_class = "FP_SHEP_ONLY_SUCCESS"
        else:
            team_class = "BOTH_FAIL"
        paired.append(
            {
                "schema_version": SCHEMA_VERSION,
                "scenario": key[0],
                "seed": key[1],
                "agent_id": key[2],
                "candidate_set_hash": proposal["candidate_set_hash"],
                "candidate_hash_match": hashes_match,
                "candidate_order_match": order_match,
                "selector_pair_status": NO_CANDIDATE_PAIR if no_candidate else "ELIGIBLE",
                "selector_disagreement": disagreement if not no_candidate else None,
                "proposal_selected_candidate_index": proposal["selected_candidate_index"],
                "fp_shep_selected_candidate_index": fp["selected_candidate_index"],
                "proposal_reference_reached": bool(proposal["reference_reached"]),
                "fp_shep_reference_reached": bool(fp["reference_reached"]),
                "reference_pair_class": reach_class,
                "proposal_team_success": bool(p_episode["team_success"]),
                "fp_shep_team_success": bool(f_episode["team_success"]),
                "team_pair_class": team_class,
            }
        )
    eligible = [row for row in paired if row["selector_pair_status"] == "ELIGIBLE"]
    team_groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in paired:
        team_groups.setdefault((str(row["scenario"]), int(row["seed"])), []).append(row)
    eligible_team_groups = [
        members
        for members in team_groups.values()
        if all(row["selector_pair_status"] == "ELIGIBLE" for row in members)
    ]
    team_classes = [members[0]["team_pair_class"] for members in eligible_team_groups]
    disagreement_team_classes = [
        members[0]["team_pair_class"]
        for members in eligible_team_groups
        if any(bool(row["selector_disagreement"]) for row in members)
    ]
    disagreement_reference_classes = [
        row["reference_pair_class"]
        for row in eligible
        if bool(row["selector_disagreement"])
    ]
    summary = {
        "integrity_status": "PASSED" if not integrity_errors else "FAILED",
        "integrity_errors": integrity_errors,
        "pair_count": len(paired),
        "eligible_selector_pair_count": len(eligible),
        "no_candidate_pair_count": len(paired) - len(eligible),
        "selector_disagreement_count": sum(bool(row["selector_disagreement"]) for row in eligible),
        "selector_disagreement_rate": rate(
            sum(bool(row["selector_disagreement"]) for row in eligible), len(eligible)
        ),
        "reference_pair_class_counts": {
            name: sum(row["reference_pair_class"] == name for row in eligible)
            for name in ("BOTH_REACH", "PROPOSAL_ONLY_REACH", "FP_SHEP_ONLY_REACH", "BOTH_FAIL")
        },
        "disagreement_reference_pair_class_counts": {
            name: sum(value == name for value in disagreement_reference_classes)
            for name in ("BOTH_REACH", "PROPOSAL_ONLY_REACH", "FP_SHEP_ONLY_REACH", "BOTH_FAIL")
        },
        "eligible_team_pair_count": len(eligible_team_groups),
        "no_candidate_team_pair_count": len(team_groups) - len(eligible_team_groups),
        "team_pair_class_counts": {
            name: sum(value == name for value in team_classes)
            for name in ("BOTH_SUCCESS", "PROPOSAL_ONLY_SUCCESS", "FP_SHEP_ONLY_SUCCESS", "BOTH_FAIL")
        },
        "disagreement_episode_count": len(disagreement_team_classes),
        "disagreement_episode_team_class_counts": {
            name: sum(value == name for value in disagreement_team_classes)
            for name in ("BOTH_SUCCESS", "PROPOSAL_ONLY_SUCCESS", "FP_SHEP_ONLY_SUCCESS", "BOTH_FAIL")
        },
    }
    return paired, summary


def build_failure_rows(
    episode_rows: Sequence[Mapping[str, Any]],
    agent_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for episode in episode_rows:
        if bool(episode["team_success"]):
            continue
        selected = [
            row
            for row in agent_rows
            if row["method"] == episode["method"]
            and row["scenario"] == episode["scenario"]
            and int(row["seed"]) == int(episode["seed"])
        ]
        if episode["method"] == METHOD_TERMINAL:
            primary = "TERMINAL_BASELINE_FAILURE"
            flags: dict[str, bool] = {}
        else:
            primary, flags = classify_failure(selected)
        before = any(
            bool(row.get("collision_before_reference"))
            or bool(row.get("reference_timeout"))
            or (
                bool(row.get("reference_available"))
                and not bool(row.get("reference_reached"))
            )
            for row in selected
        )
        after = any(
            bool(row.get("collision_after_reference"))
            or bool(row.get("timeout_after_reference"))
            for row in selected
        )
        results.append(
            {
                "schema_version": SCHEMA_VERSION,
                "method": episode["method"],
                "scenario": episode["scenario"],
                "seed": episode["seed"],
                "primary_failure_category": primary,
                "failure_before_reference": before,
                "failure_after_reference": after,
                **flags,
            }
        )
    return results


def build_conclusion(
    method_rows: Sequence[Mapping[str, Any]],
    selector_summary: Mapping[str, Any],
    failure_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    lookup = {(row["method"], row["scenario"]): row for row in method_rows}
    proposal = lookup[(METHOD_PROPOSAL, "overall")]
    fp = lookup[(METHOD_FP_SHEP, "overall")]
    delta_reached = float(fp["reference_reached_rate"] or 0.0) - float(
        proposal["reference_reached_rate"] or 0.0
    )
    collision_reduction = float(proposal["reference_collision_rate"] or 0.0) - float(
        fp["reference_collision_rate"] or 0.0
    )
    delta_success = float(fp["team_success_rate"] or 0.0) - float(
        proposal["team_success_rate"] or 0.0
    )
    direction_consistent = all(
        float(lookup[(METHOD_FP_SHEP, scene)]["reference_reached_rate"] or 0.0)
        >= float(lookup[(METHOD_PROPOSAL, scene)]["reference_reached_rate"] or 0.0)
        for scene in ("sparse_static", "multi_agent")
    )
    if (delta_reached >= 0.10 or collision_reduction >= 0.10) and direction_consistent:
        signal = "YES"
        signal_reason = "stage1_executability_improvement"
    elif delta_success >= 0.10:
        signal = "YES"
        signal_reason = "team_success_improvement_without_stage1_claim"
    elif abs(delta_reached) < 0.10 and abs(delta_success) < 0.10:
        signal = "WEAK"
        signal_reason = "selector_effect_below_10pp"
    else:
        signal = "NO"
        signal_reason = "fp_shep_not_better_than_proposal"

    selector_failures = [
        row
        for row in failure_rows
        if row["method"] in {METHOD_PROPOSAL, METHOD_FP_SHEP}
    ]
    before_count = sum(bool(row["failure_before_reference"]) for row in selector_failures)
    after_count = sum(bool(row["failure_after_reference"]) for row in selector_failures)
    stage1_primary = before_count > after_count
    interaction_failures = sum(
        bool(row.get("D_inter_agent_collision_before_reference"))
        or bool(row.get("G_partial_reference_team_failure"))
        for row in selector_failures
    )
    open_stable = all(
        float(lookup[(method, "open")]["team_success_rate"] or 0.0) >= 0.8
        for method in (METHOD_PROPOSAL, METHOD_FP_SHEP)
    )
    coordination_gap = bool(open_stable and interaction_failures > 0)
    proceed = signal == "YES"
    if proceed:
        primary = "PROPOSAL_SELECTOR_EXECUTABILITY_LIMITATION"
        next_step = "Proceed to GAT Stage-I training after explicit user approval."
    elif coordination_gap:
        primary = "MULTI_AGENT_SPATIOTEMPORAL_COMPATIBILITY_NOT_YET_ISOLATED"
        next_step = "Audit selector score scaling and optional read-only graph evidence before GAT training."
    else:
        primary = "CANDIDATE_SET_OR_SELECTOR_SIGNAL_INSUFFICIENT"
        next_step = "Inspect candidate-set quality and FP-SHEP online score scaling before GAT training."
    return {
        "FP_SHEP_CLOSED_LOOP_EXECUTABILITY_SIGNAL": signal,
        "FP_SHEP_SIGNAL_REASON": signal_reason,
        "STAGE1_IS_PRIMARY_BOTTLENECK": "YES" if stage1_primary else "NO",
        "MULTI_AGENT_COORDINATION_GAP": "YES" if coordination_gap else "NO",
        "PROCEED_TO_GAT_STAGE_I": "YES" if proceed else "NO",
        "PRIMARY_LIMITATION": primary,
        "NEXT_STEP": next_step,
        "proposal_reference_reached_rate": proposal["reference_reached_rate"],
        "fp_shep_reference_reached_rate": fp["reference_reached_rate"],
        "delta_reference_reached_rate": delta_reached,
        "proposal_collision_before_reference_rate": proposal["reference_collision_rate"],
        "fp_shep_collision_before_reference_rate": fp["reference_collision_rate"],
        "collision_before_reference_reduction": collision_reduction,
        "proposal_team_success_rate": proposal["team_success_rate"],
        "fp_shep_team_success_rate": fp["team_success_rate"],
        "delta_team_success_rate": delta_success,
        "selector_disagreement_rate": selector_summary["selector_disagreement_rate"],
        "failure_before_reference_count": before_count,
        "failure_after_reference_count": after_count,
        "interaction_related_failure_count": interaction_failures,
        "graph_builder_post_hoc_diagnostic": "SKIPPED_BY_CONFIRMED_SCOPE",
        "thresholds_are_statistical_significance": False,
    }
