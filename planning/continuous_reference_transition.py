"""Stateful runtime-only transition from selected to executed references.

The upper planner owns ``g_cmd``.  This module owns only the lower-interface
state ``g_exec`` and ``gdot_exec``.  It neither selects references nor changes
the learned SAC-DMP policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


# For a unit step of a critically damped second-order system with zero initial
# velocity, (1 + lambda) exp(-lambda) = 0.02 at this dimensionless time.
CRITICAL_2_PERCENT_OMEGA_T = 5.83392170191739


def _vector3(value: Sequence[float], name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite")
    return vector.copy()


@dataclass(frozen=True)
class CRTConfig:
    """Configuration for one exact-discrete critically damped filter."""

    dt: float
    settling_time_s: float
    settling_fraction: float = 0.02
    safety_bandwidth_multiplier: float = 2.0
    settled_absolute_tolerance_m: float = 1.0e-6

    def __post_init__(self) -> None:
        if not np.isfinite(self.dt) or float(self.dt) <= 0.0:
            raise ValueError("dt must be positive and finite")
        if not np.isfinite(self.settling_time_s) or float(self.settling_time_s) <= 0.0:
            raise ValueError("settling_time_s must be positive and finite")
        if not np.isclose(float(self.settling_fraction), 0.02, rtol=0.0, atol=0.0):
            raise ValueError("this implementation defines T_ref at exactly 2% error")
        if (
            not np.isfinite(self.safety_bandwidth_multiplier)
            or float(self.safety_bandwidth_multiplier) < 1.0
        ):
            raise ValueError("safety_bandwidth_multiplier must be finite and at least one")
        if (
            not np.isfinite(self.settled_absolute_tolerance_m)
            or float(self.settled_absolute_tolerance_m) <= 0.0
        ):
            raise ValueError("settled_absolute_tolerance_m must be positive and finite")

    @property
    def omega_rad_s(self) -> float:
        return float(CRITICAL_2_PERCENT_OMEGA_T / float(self.settling_time_s))


@dataclass(frozen=True)
class CRTStep:
    executed_reference: np.ndarray
    executed_reference_velocity: np.ndarray
    command_error_m: float
    filter_speed_mps: float
    omega_rad_s: float


class ContinuousReferenceTransition:
    """Exact-discrete state for a three-dimensional critical reference filter."""

    def __init__(self, config: CRTConfig, initial_reference: Sequence[float]) -> None:
        self.config = config
        initial = _vector3(initial_reference, "initial_reference")
        self.command = initial.copy()
        self.executed = initial.copy()
        self.velocity = np.zeros(3, dtype=float)

    def set_command(self, command: Sequence[float]) -> bool:
        """Update g_cmd without modifying g_exec or gdot_exec."""

        new_command = _vector3(command, "command")
        changed = bool(not np.array_equal(new_command, self.command))
        self.command = new_command
        return changed

    def exact_step(self, *, omega_multiplier: float = 1.0) -> CRTStep:
        """Advance one deterministic exact step for a piecewise-constant command."""

        multiplier = float(omega_multiplier)
        if not np.isfinite(multiplier) or multiplier <= 0.0:
            raise ValueError("omega_multiplier must be positive and finite")
        omega = float(self.config.omega_rad_s * multiplier)
        dt = float(self.config.dt)
        error = self.executed - self.command
        velocity = self.velocity
        decay = float(np.exp(-omega * dt))
        next_error = decay * ((1.0 + omega * dt) * error + dt * velocity)
        next_velocity = decay * (
            (1.0 - omega * dt) * velocity - (omega * omega * dt) * error
        )
        next_executed = self.command + next_error
        if not np.all(np.isfinite(next_executed)) or not np.all(np.isfinite(next_velocity)):
            raise FloatingPointError("CRT exact update produced a non-finite state")
        self.executed = next_executed
        self.velocity = next_velocity
        return self.snapshot(omega_rad_s=omega)

    def direct_replace_with_command(self) -> CRTStep:
        """Hard-safety fallback; never used for an ordinary command update."""

        self.executed = self.command.copy()
        self.velocity = np.zeros(3, dtype=float)
        return self.snapshot()

    def snapshot(self, *, omega_rad_s: float | None = None) -> CRTStep:
        return CRTStep(
            executed_reference=self.executed.copy(),
            executed_reference_velocity=self.velocity.copy(),
            command_error_m=float(np.linalg.norm(self.command - self.executed)),
            filter_speed_mps=float(np.linalg.norm(self.velocity)),
            omega_rad_s=float(
                self.config.omega_rad_s if omega_rad_s is None else omega_rad_s
            ),
        )

    def settled_threshold_m(self, command_change_distance_m: float) -> float:
        distance = float(command_change_distance_m)
        if not np.isfinite(distance) or distance < 0.0:
            raise ValueError("command_change_distance_m must be finite and non-negative")
        return float(
            max(
                self.config.settled_absolute_tolerance_m,
                self.config.settling_fraction * distance,
            )
        )


__all__ = [
    "CRITICAL_2_PERCENT_OMEGA_T",
    "CRTConfig",
    "CRTStep",
    "ContinuousReferenceTransition",
]
