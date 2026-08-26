"""Minimal post-GAT direction-continuity tie-break.

The normal GAT-R decision remains authoritative.  This helper may replace its
committed candidate only inside a frozen low-confidence gate and only with an
existing rank-2/rank-3 candidate that is FP-feasible, safety-noninferior, and
positive-progress.  It changes neither candidates, previews, logits, nor ERR.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from planning.pre_gat_220step_revalidation import stable_hash


def _wrap(value: float) -> float:
    return float((value + np.pi) % (2.0 * np.pi) - np.pi)


def _horizontal_angle(task: np.ndarray, ref: np.ndarray) -> float:
    task_xy = task[:2]
    ref_xy = ref[:2]
    if np.linalg.norm(task_xy) <= 1.0e-12 or np.linalg.norm(ref_xy) <= 1.0e-12:
        return float("nan")
    cross = task_xy[0] * ref_xy[1] - task_xy[1] * ref_xy[0]
    return float(math.atan2(float(cross), float(np.dot(task_xy, ref_xy))))


def _elevation(vector: np.ndarray) -> float:
    horizontal = float(np.linalg.norm(vector[:2]))
    if horizontal <= 1.0e-12 and abs(float(vector[2])) <= 1.0e-12:
        return float("nan")
    return float(math.atan2(float(vector[2]), horizontal))


@dataclass(frozen=True)
class DirectionContinuityTieBreakConfig:
    near_tie_probability_margin: float
    warning_safety_margin_m: float
    numerical_angle_epsilon_rad: float = 1.0e-6

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.near_tie_probability_margin) <= 1.0:
            raise ValueError("near-tie probability margin must lie in [0,1]")
        if not np.isfinite(float(self.warning_safety_margin_m)):
            raise ValueError("warning safety margin must be finite")
        if float(self.numerical_angle_epsilon_rad) <= 0.0:
            raise ValueError("numerical angle epsilon must be positive")


class DirectionContinuityTieBreak:
    def __init__(self, config: DirectionContinuityTieBreakConfig, *, num_agents: int) -> None:
        self.config = config
        self.num_agents = int(num_agents)
        if self.num_agents <= 0:
            raise ValueError("num_agents must be positive")
        self._history: list[list[tuple[float, float]]] = [[] for _ in range(self.num_agents)]
        self._rows: list[dict[str, Any]] = []

    def reset(self) -> None:
        self._history = [[] for _ in range(self.num_agents)]
        self._rows.clear()

    def reset_agent(self, agent_id: int, *, step: int, reason: str) -> None:
        agent = self._agent(agent_id)
        self._history[agent].clear()
        self._rows.append(
            {
                "step": int(step),
                "agent_id": agent,
                "event_type": str(reason),
                "history_reset": True,
                "activated": False,
                "replaced": False,
                "reason": "SEMANTIC_HISTORY_RESET",
            }
        )

    def apply_plan(
        self,
        selection: dict[str, Any],
        *,
        env: Any,
        agent_ids: Sequence[int],
        event_types: Mapping[int, str],
        active_safety_margins_m: Mapping[int, float | None],
        initial: bool,
        step: int,
    ) -> dict[str, Any]:
        plan = selection["plan"]
        for agent_id in agent_ids:
            agent = self._agent(agent_id)
            record = plan["candidate_records"][agent]
            available = bool(plan["available"][agent])
            selected_id = record.get("selected_candidate_id")
            if not available or selected_id is None:
                self.reset_agent(agent, step=step, reason="TERMINAL_OR_NULL_COMMITMENT")
                continue
            selected_id = int(selected_id)
            candidate_points = np.asarray(record.get("candidate_world_points", []), dtype=float)
            if candidate_points.ndim != 2 or candidate_points.shape[1:] != (3,):
                raise ValueError("DCTB requires retained Top-K candidate world points")
            position = np.asarray(env.dynamics[agent].p, dtype=float)
            terminal = np.asarray(env.goals[agent], dtype=float)
            task = terminal - position
            top_alpha, top_beta = self._angles(position, task, candidate_points[selected_id])
            event_type = str(event_types.get(agent, "INITIAL_SELECTION" if initial else "UNKNOWN"))
            margin = record.get("top1_top2_probability_margin")
            margin = None if margin is None else float(margin)
            history = self._history[agent]
            row: dict[str, Any] = {
                "step": int(step),
                "agent_id": agent,
                "event_type": event_type,
                "initial": bool(initial),
                "history_reset": False,
                "history_length_before": len(history),
                "original_k1_candidate_id": selected_id,
                "selected_candidate_id": selected_id,
                "replacement_candidate_id": None,
                "replacement_gat_rank": None,
                "top1_top2_probability_margin": margin,
                "near_tie_threshold": float(self.config.near_tie_probability_margin),
                "near_tie": bool(margin is not None and margin <= self.config.near_tie_probability_margin),
                "active_safety_margin_m": active_safety_margins_m.get(agent),
                "activated": False,
                "replaced": False,
                "horizontal_reversal_trigger": False,
                "vertical_reversal_trigger": False,
                "reason": "INITIAL_OR_INSUFFICIENT_HISTORY",
            }
            replacement_id: int | None = None
            replacement_rank: int | None = None
            if not initial and len(history) >= 2:
                previous_alpha, previous_beta = history[-1]
                older_alpha, older_beta = history[-2]
                previous_yaw_turn = _wrap(previous_alpha - older_alpha)
                previous_pitch_turn = previous_beta - older_beta
                top_yaw_turn = _wrap(top_alpha - previous_alpha)
                top_pitch_turn = top_beta - previous_beta
                eps = float(self.config.numerical_angle_epsilon_rad)
                yaw_gradual = self._gradual_to_zero(previous_alpha, top_alpha, eps)
                pitch_gradual = self._gradual_to_zero(previous_beta, top_beta, eps)
                yaw_trigger = bool(
                    abs(previous_yaw_turn) > eps
                    and abs(top_yaw_turn) > eps
                    and previous_yaw_turn * top_yaw_turn < 0.0
                    and not yaw_gradual
                )
                pitch_trigger = bool(
                    abs(previous_pitch_turn) > eps
                    and abs(top_pitch_turn) > eps
                    and previous_pitch_turn * top_pitch_turn < 0.0
                    and not pitch_gradual
                )
                current_margin = active_safety_margins_m.get(agent)
                safety_critical = bool(
                    event_type == "EMERGENCY_REPROPOSAL"
                    or (
                        current_margin is not None
                        and float(current_margin) <= float(self.config.warning_safety_margin_m)
                    )
                )
                row.update(
                    {
                        "previous_yaw_turn_rad": previous_yaw_turn,
                        "previous_pitch_turn_rad": previous_pitch_turn,
                        "top1_yaw_turn_rad": top_yaw_turn,
                        "top1_pitch_turn_rad": top_pitch_turn,
                        "horizontal_reversal_trigger": yaw_trigger,
                        "vertical_reversal_trigger": pitch_trigger,
                        "safety_critical_bypass": safety_critical,
                    }
                )
                if not (yaw_trigger or pitch_trigger):
                    row["reason"] = "NO_CLEAR_REVERSAL"
                elif safety_critical:
                    row["reason"] = "EXISTING_SAFETY_CRITICAL_BYPASS"
                elif margin is None or margin > float(self.config.near_tie_probability_margin):
                    row["reason"] = "GAT_NOT_NEAR_TIE"
                else:
                    row["activated"] = True
                    ranked = self._candidate_ranking(record, selected_id)
                    alternatives: list[dict[str, Any]] = []
                    for rank, candidate_id in enumerate(ranked[1:3], start=2):
                        if not self._safety_admissible(record, selected_id, candidate_id):
                            continue
                        alpha, beta = self._angles(position, task, candidate_points[candidate_id])
                        yaw_turn = _wrap(alpha - previous_alpha)
                        pitch_turn = beta - previous_beta
                        yaw_ok = (
                            not yaw_trigger
                            or previous_yaw_turn * yaw_turn >= 0.0
                            or self._gradual_to_zero(previous_alpha, alpha, eps)
                        )
                        pitch_ok = (
                            not pitch_trigger
                            or previous_pitch_turn * pitch_turn >= 0.0
                            or self._gradual_to_zero(previous_beta, beta, eps)
                        )
                        if not (yaw_ok and pitch_ok):
                            continue
                        yaw_cost = abs(_wrap(yaw_turn - previous_yaw_turn))
                        pitch_cost = abs(pitch_turn - previous_pitch_turn)
                        top_yaw_cost = abs(_wrap(top_yaw_turn - previous_yaw_turn))
                        top_pitch_cost = abs(top_pitch_turn - previous_pitch_turn)
                        if yaw_trigger and pitch_trigger:
                            if not (
                                yaw_cost <= top_yaw_cost + eps
                                and pitch_cost <= top_pitch_cost + eps
                                and (yaw_cost < top_yaw_cost - eps or pitch_cost < top_pitch_cost - eps)
                            ):
                                continue
                            key = (max(yaw_cost / max(top_yaw_cost, eps), pitch_cost / max(top_pitch_cost, eps)), rank)
                        elif yaw_trigger:
                            if yaw_cost >= top_yaw_cost - eps:
                                continue
                            key = (yaw_cost, rank)
                        else:
                            if pitch_cost >= top_pitch_cost - eps:
                                continue
                            key = (pitch_cost, rank)
                        alternatives.append(
                            {
                                "candidate_id": candidate_id,
                                "rank": rank,
                                "alpha": alpha,
                                "beta": beta,
                                "yaw_turn": yaw_turn,
                                "pitch_turn": pitch_turn,
                                "yaw_cost": yaw_cost,
                                "pitch_cost": pitch_cost,
                                "key": key,
                            }
                        )
                    if alternatives:
                        chosen = min(alternatives, key=lambda item: item["key"])
                        replacement_id = int(chosen["candidate_id"])
                        replacement_rank = int(chosen["rank"])
                        top_alpha, top_beta = float(chosen["alpha"]), float(chosen["beta"])
                        self._replace(plan, record, agent, selected_id, replacement_id, replacement_rank)
                        row.update(
                            {
                                "selected_candidate_id": replacement_id,
                                "replacement_candidate_id": replacement_id,
                                "replacement_gat_rank": replacement_rank,
                                "replaced": True,
                                "replacement_yaw_turn_rad": chosen["yaw_turn"],
                                "replacement_pitch_turn_rad": chosen["pitch_turn"],
                                "replacement_yaw_second_difference_cost": chosen["yaw_cost"],
                                "replacement_pitch_second_difference_cost": chosen["pitch_cost"],
                                "reason": "DIRECTION_CONTINUITY_REPLACEMENT",
                            }
                        )
                    else:
                        row["reason"] = "NO_ADMISSIBLE_RANK2_OR_RANK3_ALTERNATIVE"
            history.append((top_alpha, top_beta))
            if len(history) > 2:
                del history[:-2]
            record.update(
                {
                    "dctb_activated": bool(row["activated"]),
                    "dctb_replaced": bool(row["replaced"]),
                    "dctb_original_candidate_id": selected_id,
                    "dctb_replacement_candidate_id": replacement_id,
                    "dctb_replacement_gat_rank": replacement_rank,
                    "dctb_reason": row["reason"],
                }
            )
            self._rows.append(row)
        plan["selection_plan_hash"] = stable_hash(
            {key: value for key, value in plan.items() if key != "selection_plan_hash"}
        )
        return selection

    def trace_rows(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self._rows]

    def _angles(self, position: np.ndarray, task: np.ndarray, point: np.ndarray) -> tuple[float, float]:
        ref = point - position
        return _horizontal_angle(task, ref), _elevation(ref) - _elevation(task)

    @staticmethod
    def _gradual_to_zero(previous: float, candidate: float, eps: float) -> bool:
        return bool(
            abs(previous) > eps
            and previous * candidate > 0.0
            and abs(candidate) < abs(previous) - eps
        )

    @staticmethod
    def _candidate_ranking(record: Mapping[str, Any], selected_id: int) -> list[int]:
        probabilities = np.asarray(record.get("class_probabilities", []), dtype=float)
        candidate_count = len(record.get("candidate_world_points", []))
        if probabilities.shape != (candidate_count + 1,):
            raise ValueError("DCTB requires the frozen null-plus-candidate probability vector")
        remaining = sorted(
            (candidate for candidate in range(candidate_count) if candidate != selected_id),
            key=lambda candidate: (-float(probabilities[candidate + 1]), candidate),
        )
        return [selected_id, *remaining]

    @staticmethod
    def _safety_admissible(record: Mapping[str, Any], top: int, alternative: int) -> bool:
        fp_records = {
            int(item["candidate_id"]): item
            for item in record.get("fp_shep_candidate_records", [])
        }
        if top not in fp_records or alternative not in fp_records:
            return False
        top_fp, alt_fp = fp_records[top], fp_records[alternative]
        valid = [bool(value) for value in alt_fp.get("preview_feature_valid_mask", [])]
        if not valid or not all(valid):
            return False
        if float(alt_fp.get("preview_task_progress", -np.inf)) <= 0.0:
            return False
        if float(alt_fp.get("preview_min_clearance", -np.inf)) < float(
            top_fp.get("preview_min_clearance", -np.inf)
        ):
            return False
        interactions = {
            int(item["candidate_id"]): item
            for item in record.get("all_candidate_interaction_records", [])
        }
        top_i = interactions.get(top, {"risky": False})
        alt_i = interactions.get(alternative, {"risky": False})
        if bool(alt_i.get("risky")) and not bool(top_i.get("risky")):
            return False
        if bool(alt_i.get("risky")) and bool(top_i.get("risky")):
            if float(alt_i.get("maximum_risk_duration_s", np.inf)) > float(
                top_i.get("maximum_risk_duration_s", np.inf)
            ):
                return False
            alt_d = alt_i.get("minimum_predicted_separation_m")
            top_d = top_i.get("minimum_predicted_separation_m")
            if alt_d is not None and top_d is not None and float(alt_d) < float(top_d):
                return False
        return True

    @staticmethod
    def _replace(
        plan: dict[str, Any],
        record: dict[str, Any],
        agent: int,
        original: int,
        replacement: int,
        replacement_rank: int,
    ) -> None:
        point = record["candidate_world_points"][replacement]
        plan["references"][agent] = list(point)
        plan["available"][agent] = True
        record["selected_candidate_id"] = replacement
        record["selected_class"] = replacement + 1
        record["selected_null"] = False
        record["selected_proposal_rank"] = replacement
        record["selected_proposal_rank_1based"] = replacement + 1
        record["temporary_reference"] = list(point)
        proposal_scores = record.get("proposal_scores", [])
        record["selected_proposal_score"] = (
            float(proposal_scores[replacement])
            if replacement < len(proposal_scores)
            else None
        )
        fp_records = {
            int(item["candidate_id"]): item
            for item in record.get("fp_shep_candidate_records", [])
        }
        selected_fp = fp_records.get(replacement, {})
        record["selected_fp_shep_score"] = selected_fp.get("fp_shep_online_score")
        ordered_fp = sorted(
            fp_records,
            key=lambda candidate: (
                -float(fp_records[candidate].get("fp_shep_online_score", -np.inf)),
                candidate,
            ),
        )
        record["selected_fp_shep_rank_1based"] = (
            ordered_fp.index(replacement) + 1 if replacement in ordered_fp else None
        )
        interactions = {
            int(item["candidate_id"]): item
            for item in record.get("all_candidate_interaction_records", [])
        }
        selected_interaction = interactions.get(replacement, {})
        record["selected_minimum_d_min"] = selected_interaction.get(
            "minimum_predicted_separation_m"
        )
        record["selected_maximum_T_risk"] = selected_interaction.get(
            "maximum_risk_duration_s"
        )
        record["selected_candidate_risky"] = selected_interaction.get("risky")
        # The retained all-candidate interaction table contains conservative
        # summaries, not the full edge list for every alternative.  Do not
        # leave the original top-1 edge list attached to a replacement.
        record["selected_edge_records"] = []
        record["selected_edge_records_unavailable_after_dctb"] = True
        record["dctb_original_candidate_id"] = original
        record["dctb_replacement_candidate_id"] = replacement
        record["dctb_replacement_gat_rank"] = replacement_rank

    def _agent(self, value: int) -> int:
        agent = int(value)
        if not 0 <= agent < self.num_agents:
            raise IndexError("agent outside DCTB state")
        return agent


__all__ = ["DirectionContinuityTieBreak", "DirectionContinuityTieBreakConfig"]
