"""Execution-local horizontal/vertical turn-sign persistence.

This wrapper composes the frozen Strong jerk limiter with one fixed persistence
rule.  It changes no reference, policy, DMP state, or upper-planner output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TurnSignPersistenceConfig:
    n_persist: int
    a_rev_lat_mps2: float
    a_rev_vert_mps2: float
    warning_margin_m: float
    velocity_epsilon_mps: float = 1.0e-9
    command_epsilon_mps2: float = 1.0e-9
    enabled: bool = True

    def __post_init__(self) -> None:
        finite = (
            self.a_rev_lat_mps2,
            self.a_rev_vert_mps2,
            self.warning_margin_m,
            self.velocity_epsilon_mps,
            self.command_epsilon_mps2,
        )
        if not all(np.isfinite(float(value)) for value in finite):
            raise ValueError("turn-persistence thresholds must be finite")
        if int(self.n_persist) != 2:
            raise ValueError("the frozen turn-persistence contract requires N_persist=2")
        if min(float(self.a_rev_lat_mps2), float(self.a_rev_vert_mps2)) <= 0.0:
            raise ValueError("magnitude overrides must be positive")
        if min(float(self.velocity_epsilon_mps), float(self.command_epsilon_mps2)) <= 0.0:
            raise ValueError("numerical epsilons must be positive")


class TurnSignPersistenceExecutionFilter:
    """Compose Early-Bypass Strong with independent lateral/vertical persistence."""

    def __init__(self, strong_limiter: Any, config: TurnSignPersistenceConfig) -> None:
        self.strong_limiter = strong_limiter
        self.config_persistence = config
        self.config = strong_limiter.config
        self.num_agents = int(strong_limiter.num_agents)
        self._velocity = np.full((self.num_agents, 3), np.nan, dtype=float)
        self._accepted = np.zeros((self.num_agents, 2), dtype=np.int8)
        self._pending_sign = np.zeros((self.num_agents, 2), dtype=np.int8)
        self._pending_count = np.zeros((self.num_agents, 2), dtype=np.int64)
        self._pending_rows: list[dict[str, Any] | None] = [None] * self.num_agents
        self._rows: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.strong_limiter.reset()
        self._velocity.fill(np.nan)
        self._accepted.fill(0)
        self._pending_sign.fill(0)
        self._pending_count.fill(0)
        self._pending_rows = [None] * self.num_agents
        self._rows.clear()

    def set_execution_velocity(self, agent_id: int, velocity: np.ndarray) -> None:
        index = self._agent(agent_id)
        value = np.asarray(velocity, dtype=float)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError("executed velocity must be a finite 3-vector")
        self._velocity[index] = value

    def set_context(self, agent_id: int, **kwargs: Any) -> None:
        self.strong_limiter.set_context(agent_id, **kwargs)

    def limit(self, agent_id: int, raw_acceleration: np.ndarray) -> np.ndarray:
        index = self._agent(agent_id)
        raw = self._vector(raw_acceleration)
        if not np.all(np.isfinite(self._velocity[index])):
            raise RuntimeError("set_execution_velocity must precede limit")
        strong = np.asarray(self.strong_limiter.limit(index, raw), dtype=float)
        strong_pending = self.strong_limiter.pending_trace_snapshot(index)
        bypass = bool(strong_pending["bypass_active"])
        output = strong.copy()
        velocity_xy = self._velocity[index, :2]
        speed_xy = float(np.linalg.norm(velocity_xy))
        lateral = {
            "valid_heading": speed_xy > float(self.config_persistence.velocity_epsilon_mps),
            "requested": None,
            "executed": None,
            "activation": False,
            "confirmed_reversal": False,
            "magnitude_override": False,
        }
        vertical = {
            "requested": float(strong[2]),
            "executed": float(strong[2]),
            "activation": False,
            "confirmed_reversal": False,
            "magnitude_override": False,
        }
        if not self.config_persistence.enabled or bypass:
            if bypass:
                output = raw.copy()
        else:
            if lateral["valid_heading"]:
                tangent = velocity_xy / speed_xy
                normal = np.asarray([-tangent[1], tangent[0]], dtype=float)
                a_tan = float(np.dot(strong[:2], tangent))
                a_lat = float(np.dot(strong[:2], normal))
                lateral["requested"] = a_lat
                a_lat_exec, details = self._axis(index, 0, a_lat, self.config_persistence.a_rev_lat_mps2)
                lateral.update(details)
                lateral["executed"] = a_lat_exec
                output[:2] = a_tan * tangent + a_lat_exec * normal
                lateral["tangential_acceleration"] = a_tan
            else:
                self._reset_axis(index, 0)
            a_vert_exec, details = self._axis(
                index, 1, float(strong[2]), self.config_persistence.a_rev_vert_mps2
            )
            vertical.update(details)
            vertical["executed"] = a_vert_exec
            output[2] = a_vert_exec
        if not np.all(np.isfinite(output)):
            raise FloatingPointError("turn persistence produced non-finite acceleration")
        self._pending_rows[index] = {
            **strong_pending,
            "persistence_enabled": bool(self.config_persistence.enabled),
            "execution_velocity": self._velocity[index].copy(),
            "strong_limited_acceleration": strong.copy(),
            "persistence_output_acceleration": output.copy(),
            "horizontal": dict(lateral),
            "vertical": dict(vertical),
            "horizontal_accepted_sign": int(self._accepted[index, 0]),
            "vertical_accepted_sign": int(self._accepted[index, 1]),
            "horizontal_pending_sign": int(self._pending_sign[index, 0]),
            "vertical_pending_sign": int(self._pending_sign[index, 1]),
            "horizontal_pending_count": int(self._pending_count[index, 0]),
            "vertical_pending_count": int(self._pending_count[index, 1]),
            "persistence_bypassed_for_safety": bypass,
            "persistence_modified": bool(
                not np.allclose(output, strong, rtol=0.0, atol=self.config_persistence.command_epsilon_mps2)
            ),
        }
        return output

    def observe_executed(self, agent_id: int, executed_acceleration: np.ndarray) -> None:
        index = self._agent(agent_id)
        executed = self._vector(executed_acceleration)
        pending = self._pending_rows[index]
        if pending is None:
            raise RuntimeError("observe_executed must follow limit")
        self.strong_limiter.observe_executed(index, executed)
        # The wrapped Strong limiter finalizes jerk/runtime diagnostics only in
        # observe_executed.  Merge that completed diagnostic row back into the
        # wrapper trace; this does not participate in the control output.
        # Read only the last finalized internal row.  Calling trace_rows() here
        # would serialize the entire episode history at every control step and
        # turn an O(T) diagnostic into O(T^2); the value is diagnostic-only.
        strong_completed = dict(self.strong_limiter._rows[-1])
        row = {
            **strong_completed,
            **pending,
            "limited_acceleration": np.asarray(pending["strong_limited_acceleration"], dtype=float),
            "executed_acceleration": executed.copy(),
            "physical_acceleration_clip_active": bool(
                not np.allclose(
                    np.asarray(pending["persistence_output_acceleration"], dtype=float),
                    executed,
                    rtol=0.0,
                    atol=1.0e-12,
                )
            ),
            "persistence_unchanged_fraction_flag": bool(
                np.array_equal(
                    np.asarray(pending["strong_limited_acceleration"]),
                    np.asarray(pending["persistence_output_acceleration"]),
                )
            ),
        }
        self._rows.append(row)
        self._pending_rows[index] = None

    def trace_rows(self) -> list[dict[str, Any]]:
        if any(row is not None for row in self._pending_rows):
            raise RuntimeError("cannot export persistence trace with unobserved acceleration")
        return [self._json_row(row) for row in self._rows]

    def _axis(self, agent: int, axis: int, command: float, override: float) -> tuple[float, dict[str, bool]]:
        epsilon = float(self.config_persistence.command_epsilon_mps2)
        details = {"activation": False, "confirmed_reversal": False, "magnitude_override": False}
        if abs(command) <= epsilon:
            self._pending_sign[agent, axis] = 0
            self._pending_count[agent, axis] = 0
            return command, details
        requested_sign = 1 if command > 0.0 else -1
        accepted = int(self._accepted[agent, axis])
        if accepted == 0:
            self._accepted[agent, axis] = requested_sign
            self._reset_pending(agent, axis)
            return command, details
        if requested_sign == accepted:
            self._reset_pending(agent, axis)
            return command, details
        details["activation"] = True
        if abs(command) >= float(override):
            self._accepted[agent, axis] = requested_sign
            self._reset_pending(agent, axis)
            details["confirmed_reversal"] = True
            details["magnitude_override"] = True
            return command, details
        if int(self._pending_sign[agent, axis]) == requested_sign:
            self._pending_count[agent, axis] += 1
        else:
            self._pending_sign[agent, axis] = requested_sign
            self._pending_count[agent, axis] = 1
        if int(self._pending_count[agent, axis]) >= int(self.config_persistence.n_persist):
            self._accepted[agent, axis] = requested_sign
            self._reset_pending(agent, axis)
            details["confirmed_reversal"] = True
            return command, details
        return 0.0, details

    def _reset_axis(self, agent: int, axis: int) -> None:
        self._accepted[agent, axis] = 0
        self._reset_pending(agent, axis)

    def _reset_pending(self, agent: int, axis: int) -> None:
        self._pending_sign[agent, axis] = 0
        self._pending_count[agent, axis] = 0

    def _agent(self, value: int) -> int:
        index = int(value)
        if not 0 <= index < self.num_agents:
            raise IndexError("agent outside turn-persistence state")
        return index

    @staticmethod
    def _vector(value: np.ndarray) -> np.ndarray:
        vector = np.asarray(value, dtype=float)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError("acceleration must be a finite 3-vector")
        return vector.copy()

    @staticmethod
    def _json_row(row: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, np.ndarray):
                result[key] = value.tolist()
            elif isinstance(value, dict):
                result[key] = TurnSignPersistenceExecutionFilter._json_row(value)
            elif isinstance(value, np.generic):
                result[key] = value.item()
            else:
                result[key] = value
        return result


__all__ = ["TurnSignPersistenceConfig", "TurnSignPersistenceExecutionFilter"]
