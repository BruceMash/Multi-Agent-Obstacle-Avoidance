"""Track-A motion-aligned Proposal eligibility.

This module changes only which already-valid Proposal candidates are eligible
for the existing coarse-ranked Top-K interface.  The full sensor field and the
raw 16 x 16 direction grid remain untouched.  If the motion direction is not
reliable, safety is not comfortable, an escape state is detected, or fewer
than ``top_k`` candidates survive, the original full-sphere proposal list is
returned exactly.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from planning.event_triggered_reference_reconstruction import (
    active_direction_safety_margin,
)


@dataclass(frozen=True)
class MotionAlignedProposalConfig:
    theta_max_rad: float
    top_k: int = 10
    warning_margin_m: float = 0.35
    emergency_margin_m: float = 0.0
    velocity_epsilon_mps: float = 1.0e-6
    enabled: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < float(self.theta_max_rad) <= math.pi:
            raise ValueError("theta_max_rad must be in (0, pi]")
        if int(self.top_k) <= 0:
            raise ValueError("top_k must be positive")
        if float(self.velocity_epsilon_mps) <= 0.0:
            raise ValueError("velocity_epsilon_mps must be positive")
        if float(self.emergency_margin_m) >= float(self.warning_margin_m):
            raise ValueError("emergency margin must be below warning margin")


class MotionAlignedProposalEligibility:
    """Callable wrapper around the frozen Proposal generator.

    ``begin_plan`` binds the current environment for one upper-planning call.
    The evaluator invokes the wrapped Proposal function once per agent, and
    agent identity is recovered from the existing sensor object identity.
    """

    def __init__(
        self,
        config: MotionAlignedProposalConfig,
        original_propose: Callable[..., list[Any]],
    ) -> None:
        self.config = config
        self.original_propose = original_propose
        self.env: Any | None = None
        self.scenario_id = ""
        self.rows: list[dict[str, Any]] = []

    def reset_episode(self, scenario_id: str) -> None:
        self.env = None
        self.scenario_id = str(scenario_id)
        self.rows.clear()

    def begin_plan(self, env: Any) -> None:
        self.env = env

    def _agent_id(self, sensor: Any) -> int:
        if self.env is None:
            raise RuntimeError("begin_plan(env) must precede Proposal generation")
        matches = [index for index, value in enumerate(self.env.sensors) if value is sensor]
        if len(matches) != 1:
            raise RuntimeError("could not recover unique agent from sensor identity")
        return int(matches[0])

    def __call__(
        self,
        position: np.ndarray,
        goal: np.ndarray,
        velocity: np.ndarray,
        sensor_packet: Any,
        sensor: Any,
        proposal_config: Any,
        goal_tolerance: float,
        *,
        timing_sink: dict[str, float] | None = None,
    ) -> list[Any]:
        started = time.perf_counter_ns()
        proposals = self.original_propose(
            position,
            goal,
            velocity,
            sensor_packet,
            sensor,
            proposal_config,
            goal_tolerance,
            timing_sink=timing_sink,
        )
        raw = list(proposals)
        agent_id = self._agent_id(sensor)
        env = self.env
        assert env is not None
        velocity = np.asarray(velocity, dtype=float)
        speed = float(np.linalg.norm(velocity))
        valid_motion = bool(speed > float(self.config.velocity_epsilon_mps))
        angles: list[float] = []
        if valid_motion:
            unit_velocity = velocity / speed
            angles = [
                float(math.acos(float(np.clip(np.dot(unit_velocity, item.direction), -1.0, 1.0))))
                for item in raw
            ]
        cone = [
            item for item, angle in zip(raw, angles)
            if angle <= float(self.config.theta_max_rad)
        ] if valid_motion else []

        active_goal = np.asarray(env.dmps[agent_id].goal, dtype=float)
        safety_margin = None
        fallback_reasons: list[str] = []
        if self.config.enabled:
            safety_margin = float(
                active_direction_safety_margin(
                    env, agent_id, active_goal, proposal_config
                ).value_m
            )
            if safety_margin <= float(self.config.emergency_margin_m):
                fallback_reasons.extend(["E_CRITICAL_ESCAPE_STATE", "A_NOT_COMFORTABLE_SAFE"])
            elif safety_margin <= float(self.config.warning_margin_m):
                fallback_reasons.append("A_NOT_COMFORTABLE_SAFE")
            if not valid_motion:
                fallback_reasons.append("B_LOW_SPEED")
            # The frozen Proposal implementation retains non-positive progress
            # directions only when no positive-progress escape is available.
            if raw and not any(
                float(item.distance_progress) > float(proposal_config.positive_progress_epsilon)
                for item in raw
            ):
                fallback_reasons.append("E_EXISTING_ESCAPE_STATE")
            if len(cone) < int(self.config.top_k):
                fallback_reasons.extend(
                    ["C_INSUFFICIENT_FEASIBLE_CONE", "D_TOPK_INTERFACE_UNAVAILABLE"]
                )

        use_fallback = bool(self.config.enabled and fallback_reasons)
        returned = raw if (not self.config.enabled or use_fallback) else cone
        top_k_success = len(returned) >= int(self.config.top_k)
        row = {
            "scenario_id": self.scenario_id,
            "step": int(env.steps),
            "agent_id": agent_id,
            "enabled": bool(self.config.enabled),
            "raw_direction_count": int(np.prod(np.asarray(sensor.ray_directions).shape[:2])),
            "raw_viable_candidate_count": len(raw),
            "motion_cone_candidate_count": len(cone),
            "returned_candidate_count": len(returned),
            "full_sphere_fallback": use_fallback,
            "fallback_reasons": sorted(set(fallback_reasons)),
            "top_k_construction_success": bool(top_k_success),
            "current_speed_mps": speed,
            "motion_direction_valid": valid_motion,
            "active_safety_margin_m": safety_margin,
            "theta_max_rad": float(self.config.theta_max_rad),
            "theta_max_deg": float(np.degrees(self.config.theta_max_rad)),
            "eligibility_runtime_ms": (time.perf_counter_ns() - started) / 1.0e6,
        }
        self.rows.append(row)
        return returned

    def summary(self) -> dict[str, Any]:
        rows = self.rows
        count = len(rows)
        fallback = sum(bool(row["full_sphere_fallback"]) for row in rows)
        return {
            "proposal_call_count": count,
            "raw_direction_count_per_call": 256,
            "mean_raw_viable_candidate_count": (
                float(np.mean([row["raw_viable_candidate_count"] for row in rows])) if rows else 0.0
            ),
            "mean_motion_cone_candidate_count": (
                float(np.mean([row["motion_cone_candidate_count"] for row in rows])) if rows else 0.0
            ),
            "full_sphere_fallback_count": fallback,
            "full_sphere_fallback_rate": fallback / count if count else 0.0,
            "top_k_construction_success_count": sum(
                bool(row["top_k_construction_success"]) for row in rows
            ),
            "top_k_construction_success_rate": (
                float(np.mean([row["top_k_construction_success"] for row in rows])) if rows else 0.0
            ),
            "eligibility_runtime_ms": float(sum(row["eligibility_runtime_ms"] for row in rows)),
        }


def wrap_plan_builder(
    builder: Callable[..., Mapping[str, Any]],
    eligibility: MotionAlignedProposalEligibility,
) -> Callable[..., Mapping[str, Any]]:
    def wrapped(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
        env = kwargs.get("env")
        if env is None and args:
            env = args[0]
        eligibility.begin_plan(env)
        return builder(*args, **kwargs)

    return wrapped


__all__ = [
    "MotionAlignedProposalConfig",
    "MotionAlignedProposalEligibility",
    "wrap_plan_builder",
]
