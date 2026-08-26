"""Safety-adaptive vector jerk limiting for the final SAC-DMP handoff.

The limiter is deliberately execution-local.  It does not alter the SAC
action, DMP state, reference, phase, observation, or any upper-level planner.
The existing physical acceleration and velocity bounds remain authoritative
and are applied after this module returns its acceleration command.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SafetyAdaptiveJerkLimiterConfig:
    dt: float
    j_smooth_mps3: float
    j_free_mps3: float
    emergency_margin_m: float
    comfortable_margin_m: float
    enabled: bool = True
    early_bypass_enabled: bool = False
    warning_margin_m: float | None = None
    numerical_epsilon: float = 1.0e-12

    def __post_init__(self) -> None:
        finite = {
            "dt": self.dt,
            "j_smooth_mps3": self.j_smooth_mps3,
            "j_free_mps3": self.j_free_mps3,
            "emergency_margin_m": self.emergency_margin_m,
            "comfortable_margin_m": self.comfortable_margin_m,
            "numerical_epsilon": self.numerical_epsilon,
        }
        if not all(np.isfinite(float(value)) for value in finite.values()):
            raise ValueError(f"jerk-limiter settings must be finite: {finite}")
        if float(self.dt) <= 0.0:
            raise ValueError("dt must be positive")
        if float(self.j_smooth_mps3) <= 0.0:
            raise ValueError("j_smooth_mps3 must be positive")
        if float(self.j_free_mps3) < float(self.j_smooth_mps3):
            raise ValueError("j_free_mps3 must not be below j_smooth_mps3")
        if float(self.comfortable_margin_m) <= float(self.emergency_margin_m):
            raise ValueError("comfortable margin must exceed emergency margin")
        warning = (
            float(self.comfortable_margin_m)
            if self.warning_margin_m is None
            else float(self.warning_margin_m)
        )
        if not np.isfinite(warning):
            raise ValueError("warning margin must be finite")
        if not (
            float(self.emergency_margin_m)
            < warning
            <= float(self.comfortable_margin_m)
        ):
            raise ValueError(
                "warning margin must be above emergency and no higher than comfortable margin"
            )
        if float(self.numerical_epsilon) <= 0.0:
            raise ValueError("numerical_epsilon must be positive")


class SafetyAdaptiveVectorJerkLimiter:
    """Stateful per-agent Euclidean acceleration-increment projection."""

    def __init__(
        self,
        config: SafetyAdaptiveJerkLimiterConfig,
        *,
        num_agents: int,
    ) -> None:
        self.config = config
        self.num_agents = int(num_agents)
        if self.num_agents <= 0:
            raise ValueError("num_agents must be positive")
        self.previous_executed = np.zeros((self.num_agents, 3), dtype=float)
        self.control_counts = np.zeros(self.num_agents, dtype=int)
        self._contexts: list[dict[str, Any] | None] = [None] * self.num_agents
        self._pending: list[dict[str, Any] | None] = [None] * self.num_agents
        self.previous_safety_margins = np.full(self.num_agents, np.nan, dtype=float)
        self._rows: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.previous_executed.fill(0.0)
        self.control_counts.fill(0)
        self._contexts = [None] * self.num_agents
        self._pending = [None] * self.num_agents
        self.previous_safety_margins.fill(np.nan)
        self._rows.clear()

    @property
    def warning_margin_m(self) -> float:
        return float(
            self.config.comfortable_margin_m
            if self.config.warning_margin_m is None
            else self.config.warning_margin_m
        )

    def q_safe(self, margin_m: float) -> float:
        margin = float(margin_m)
        if not np.isfinite(margin):
            raise ValueError("safety margin must be finite")
        span = (
            float(self.config.comfortable_margin_m)
            - float(self.config.emergency_margin_m)
        )
        return float(
            np.clip(
                (margin - float(self.config.emergency_margin_m)) / span,
                0.0,
                1.0,
            )
        )

    def set_context(
        self,
        agent_id: int,
        *,
        step: int,
        safety_margin_m: float,
        time_since_reference_change_s: float,
    ) -> None:
        index = self._agent_index(agent_id)
        margin = float(safety_margin_m)
        age = float(time_since_reference_change_s)
        if not np.isfinite(margin) or not np.isfinite(age) or age < 0.0:
            raise ValueError("limiter context must contain finite margin and nonnegative age")
        q_safe = self.q_safe(margin)
        hard_critical = bool(margin <= float(self.config.emergency_margin_m))
        previous_margin = float(self.previous_safety_margins[index])
        previous_available = bool(np.isfinite(previous_margin))
        margin_delta = (
            float(margin - previous_margin) if previous_available else None
        )
        warning_low_margin = bool(margin <= self.warning_margin_m)
        deteriorating_warning = bool(
            previous_available
            and warning_low_margin
            and float(margin_delta) < 0.0
        )
        early_bypass = bool(
            self.config.enabled
            and self.config.early_bypass_enabled
            and (warning_low_margin or deteriorating_warning)
            and not hard_critical
        )
        self._contexts[index] = {
            "step": int(step),
            "safety_margin_m": margin,
            "q_safe": q_safe,
            "hard_critical": hard_critical,
            "previous_safety_margin_m": (
                previous_margin if previous_available else None
            ),
            "safety_margin_delta_m": margin_delta,
            "warning_margin_m": self.warning_margin_m,
            "warning_low_margin": warning_low_margin,
            "deteriorating_warning": deteriorating_warning,
            "early_bypass": early_bypass,
            "time_since_reference_change_s": age,
        }
        self.previous_safety_margins[index] = margin

    def limit(self, agent_id: int, raw_acceleration: np.ndarray) -> np.ndarray:
        index = self._agent_index(agent_id)
        raw = self._vector3(raw_acceleration, "raw_acceleration")
        context = self._contexts[index]
        if context is None:
            raise RuntimeError("set_context must be called before every limiter invocation")
        if self._pending[index] is not None:
            raise RuntimeError("previous limiter invocation was not paired with observe_executed")

        started = time.perf_counter_ns()
        previous = self.previous_executed[index].copy()
        q_safe = float(context["q_safe"])
        hard_bypass = bool(self.config.enabled and context["hard_critical"])
        early_bypass = bool(context["early_bypass"])
        bypass_active = bool(hard_bypass or early_bypass)
        j_max = float(self.config.j_smooth_mps3) + (1.0 - q_safe) * (
            float(self.config.j_free_mps3) - float(self.config.j_smooth_mps3)
        )
        delta = raw - previous
        delta_norm = float(np.linalg.norm(delta))
        delta_max = float(j_max * float(self.config.dt))
        if not self.config.enabled or bypass_active or delta_norm <= delta_max:
            limited = raw.copy()
        else:
            limited = previous + delta * (delta_max / max(delta_norm, self.config.numerical_epsilon))
        if not np.all(np.isfinite(limited)):
            raise FloatingPointError("jerk limiter produced a non-finite acceleration")
        elapsed_ns = int(time.perf_counter_ns() - started)
        modification = float(np.linalg.norm(limited - raw))
        limiter_active = bool(
            self.config.enabled
            and not bypass_active
            and modification > float(self.config.numerical_epsilon)
        )
        self._pending[index] = {
            "agent_id": index,
            "control_index": int(self.control_counts[index]),
            **context,
            "enabled": bool(self.config.enabled),
            "hard_bypass": hard_bypass,
            "early_bypass": early_bypass,
            "bypass_active": bypass_active,
            "j_max_mps3": j_max,
            "delta_a_norm_mps2": delta_norm,
            "delta_a_max_mps2": delta_max,
            "limiter_active": limiter_active,
            "limiter_runtime_ns": elapsed_ns,
            "raw_acceleration": raw.copy(),
            "limited_acceleration": limited.copy(),
            "previous_executed_acceleration": previous,
            "limiter_modification_norm_mps2": modification,
        }
        return limited

    def pending_bypass_active(self, agent_id: int) -> bool:
        """Return the current decision's bypass state for a post-Strong guard."""
        index = self._agent_index(agent_id)
        pending = self._pending[index]
        if pending is None:
            raise RuntimeError("limit must be called before pending_bypass_active")
        return bool(pending["bypass_active"])

    def pending_trace_snapshot(self, agent_id: int) -> dict[str, Any]:
        """Return the current unobserved decision for a composed execution guard."""
        index = self._agent_index(agent_id)
        pending = self._pending[index]
        if pending is None:
            raise RuntimeError("limit must be called before pending_trace_snapshot")
        return {
            key: (value.copy() if isinstance(value, np.ndarray) else value)
            for key, value in pending.items()
        }

    def observe_executed(self, agent_id: int, executed_acceleration: np.ndarray) -> None:
        index = self._agent_index(agent_id)
        executed = self._vector3(executed_acceleration, "executed_acceleration")
        pending = self._pending[index]
        if pending is None:
            raise RuntimeError("observe_executed must follow a limiter invocation")
        previous = np.asarray(pending["previous_executed_acceleration"], dtype=float)
        jerk_norm = float(np.linalg.norm(executed - previous) / float(self.config.dt))
        row = {
            **pending,
            "executed_acceleration": executed.copy(),
            "executed_jerk_norm_mps3": jerk_norm,
            "jerk_sample_valid": bool(int(pending["control_index"]) > 0),
            "physical_acceleration_clip_active": bool(
                not np.allclose(
                    np.asarray(pending["limited_acceleration"], dtype=float),
                    executed,
                    rtol=0.0,
                    atol=1.0e-12,
                )
            ),
            "completely_unchanged": bool(
                np.array_equal(
                    np.asarray(pending["raw_acceleration"], dtype=float),
                    executed,
                )
            ),
        }
        self._rows.append(row)
        self.previous_executed[index] = executed
        self.control_counts[index] += 1
        self._pending[index] = None

    def trace_rows(self) -> list[dict[str, Any]]:
        if any(item is not None for item in self._pending):
            raise RuntimeError("cannot export limiter trace with unobserved acceleration")
        result: list[dict[str, Any]] = []
        for row in self._rows:
            result.append(
                {
                    key: (value.tolist() if isinstance(value, np.ndarray) else value)
                    for key, value in row.items()
                }
            )
        return result

    def _agent_index(self, agent_id: int) -> int:
        index = int(agent_id)
        if not 0 <= index < self.num_agents:
            raise IndexError("agent_id outside limiter state")
        return index

    @staticmethod
    def _vector3(value: np.ndarray, name: str) -> np.ndarray:
        vector = np.asarray(value, dtype=float)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must be a finite 3-vector")
        return vector.copy()


__all__ = [
    "SafetyAdaptiveJerkLimiterConfig",
    "SafetyAdaptiveVectorJerkLimiter",
]
