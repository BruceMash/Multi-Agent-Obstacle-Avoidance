"""Checkpoint-compatible historical forcing-gate execution primitives.

This module intentionally lives outside the current shared DMP implementation.
It restores only the action-to-dynamics mapping used by the May 2026 SAC
checkpoint and exposes an exception-safe, evaluation-only scoped replacement
for ``MultiAgentDMPEnv.step``.  The current scalar-gate implementation remains
the default before and after every scope.
"""

from __future__ import annotations

from contextlib import contextmanager
import threading
from typing import Any, Callable, Iterator

import numpy as np

from Controller.dmp_rl import DMPConfig, compute_dmp_transition
from Entity.KinematicModel import propagate_point_mass
from Environment.frozen_sac_dmp_execution import SACDMPTransition


HISTORICAL_GATE_NAME = "historical_vector_goal_eff_gate"
CURRENT_GATE_NAME = "current_scalar_terminal_distance_gate"

VARIANT_A = "A_terminal_actor_terminal_dmp"
VARIANT_B = "B_temporary_actor_terminal_dmp"
VARIANT_C = "C_terminal_actor_temporary_dmp"
VARIANT_D = "D_temporary_actor_temporary_dmp"

_TERMINAL_DMP_VARIANTS = frozenset((VARIANT_A, VARIANT_B))
_ACTIVE_DMP_VARIANTS = frozenset((VARIANT_C, VARIANT_D))
_TRANSITION_SCOPE_LOCK = threading.RLock()

TransitionObserver = Callable[[dict[str, Any], SACDMPTransition], None]


def _finite_vector(value: Any, name: str, dimension: int) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (int(dimension),) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite vector of shape ({dimension},)")
    return vector.copy()


def historical_goal_base_for_variant(
    variant: str,
    *,
    terminal_goal: np.ndarray,
    active_goal: np.ndarray,
) -> np.ndarray:
    """Return the DMP base goal dictated by the established A/B/C/D design."""

    if variant in _TERMINAL_DMP_VARIANTS:
        return np.asarray(terminal_goal, dtype=float).copy()
    if variant in _ACTIVE_DMP_VARIANTS:
        return np.asarray(active_goal, dtype=float).copy()
    raise ValueError(f"unknown A/B/C/D variant: {variant}")


def compute_historical_checkpoint_dmp_transition(
    *,
    config: DMPConfig,
    position: np.ndarray,
    velocity: np.ndarray,
    rl_action: np.ndarray,
    active_goal: np.ndarray,
    terminal_goal: np.ndarray,
    phase: float,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Compute the exact historical vector-gated DMP transition.

    ``active_goal`` is the historical ``goal_base``.  For variants A/B it is
    the terminal task goal; for C/D it is the temporary DMP reference until
    the already-established one-shot return condition switches it back.
    ``terminal_goal`` is recorded for diagnostics only and never controls the
    historical gate.
    """

    dims = int(config.dims)
    position = _finite_vector(position, "position", dims)
    velocity = _finite_vector(velocity, "velocity", dims)
    action = _finite_vector(rl_action, "rl_action", 2 * dims)
    active_goal = _finite_vector(active_goal, "active_goal", dims)
    terminal_goal = _finite_vector(terminal_goal, "terminal_goal", dims)
    phase = float(phase)
    if not np.isfinite(phase):
        raise ValueError("phase must be finite")

    raw_forcing = action[:dims].copy()
    forcing = np.clip(
        raw_forcing,
        float(config.forcing_term_min),
        float(config.forcing_term_max),
    )
    goal_offset = np.clip(
        action[dims : 2 * dims],
        -float(config.goal_offset_max),
        float(config.goal_offset_max),
    )
    goal_eff = active_goal + goal_offset
    goal_delta = goal_eff - position
    forcing_gate = np.tanh(np.abs(goal_delta))

    spring_drive = float(config.K_alpha) * float(config.K_beta) * goal_delta
    damping_drive = -float(config.K_alpha) * float(config.tau) * velocity
    forcing_contribution = forcing * forcing_gate
    nominal_drive = spring_drive + damping_drive
    closed_loop_drive = nominal_drive + forcing_contribution
    acceleration = closed_loop_drive / float(config.tau) ** 2

    # Historical checkpoint semantics were classic legacy Euler only.  This
    # deliberately does not dispatch to current configurable FCEP machinery.
    phase_rate = -float(config.alpha_s) * phase / float(config.tau)
    next_phase = max(0.0, phase + phase_rate * float(config.dt))
    terminal_delta = terminal_goal - position
    terminal_distance = float(np.linalg.norm(terminal_delta))

    info = {
        "phase": float(next_phase),
        "previous_phase": phase,
        "phase_rate": float(phase_rate),
        "phase_paused": False,
        "phase_end_reached": bool(next_phase <= float(config.phase_end_threshold)),
        "phase_mode": "historical_classic_legacy_euler",
        "tau": float(config.tau),
        "goal_eff": goal_eff.copy(),
        "goal_delta": goal_delta.copy(),
        "active_goal": active_goal.copy(),
        "terminal_goal": terminal_goal.copy(),
        "terminal_goal_delta": terminal_delta.copy(),
        "terminal_goal_distance": terminal_distance,
        "goal_offset": goal_offset.copy(),
        "raw_forcing": raw_forcing.copy(),
        "forcing": forcing.copy(),
        "forcing_gate": forcing_gate.copy(),
        "forcing_gate_semantics": HISTORICAL_GATE_NAME,
        "spring_drive": spring_drive.copy(),
        "damping_drive": damping_drive.copy(),
        "nominal_drive": nominal_drive.copy(),
        "residual_drive": forcing_contribution.copy(),
        "forcing_contribution": forcing_contribution.copy(),
        "closed_loop_drive": closed_loop_drive.copy(),
    }
    return np.asarray(acceleration, dtype=float), float(next_phase), info


def propagate_historical_sac_dmp_action(
    *,
    position: np.ndarray,
    velocity: np.ndarray,
    phase: float,
    active_goal: np.ndarray,
    terminal_goal: np.ndarray,
    action: np.ndarray,
    dmp_config: DMPConfig,
    dynamics: Any,
    acceleration_limiter: Any | None = None,
    acceleration_limiter_agent_id: int | None = None,
) -> SACDMPTransition:
    """Apply the historical DMP command through the existing point-mass model."""

    action = np.asarray(action, dtype=np.float32)
    acceleration, next_phase, controller_info = (
        compute_historical_checkpoint_dmp_transition(
            config=dmp_config,
            position=position,
            velocity=velocity,
            rl_action=action,
            active_goal=active_goal,
            terminal_goal=terminal_goal,
            phase=phase,
        )
    )
    acceleration_for_dynamics = np.asarray(acceleration, dtype=float)
    if acceleration_limiter is not None:
        if acceleration_limiter_agent_id is None:
            raise ValueError("acceleration_limiter_agent_id is required with a limiter")
        acceleration_for_dynamics = np.asarray(
            acceleration_limiter.limit(
                int(acceleration_limiter_agent_id), acceleration_for_dynamics
            ),
            dtype=float,
        )
    motion = propagate_point_mass(
        position=position,
        velocity=velocity,
        acceleration=acceleration_for_dynamics,
        dt=dynamics.dt,
        acceleration_min=dynamics.accelerate_min,
        acceleration_max=dynamics.accelerate_max,
        velocity_min=dynamics.velocity_min,
        velocity_max=dynamics.velocity_max,
        maximum_speed_norm=getattr(dynamics, "maximum_speed_norm", None),
    )
    if acceleration_limiter is not None:
        acceleration_limiter.observe_executed(
            int(acceleration_limiter_agent_id), motion["applied_acceleration"]
        )
    return SACDMPTransition(
        position=motion["position"].copy(),
        velocity=motion["velocity"].copy(),
        commanded_acceleration=np.asarray(acceleration, dtype=float).copy(),
        applied_acceleration=motion["applied_acceleration"].copy(),
        unclipped_next_velocity=motion["unclipped_next_velocity"].copy(),
        phase=float(next_phase),
        action=action.copy(),
        controller_info=controller_info,
    )


def _angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm < 1.0e-12 or second_norm < 1.0e-12:
        return 0.0
    cosine = float(np.dot(first, second) / (first_norm * second_norm))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def compare_current_and_historical_gate(
    *,
    config: DMPConfig,
    position: np.ndarray,
    velocity: np.ndarray,
    action: np.ndarray,
    active_goal: np.ndarray,
    terminal_goal: np.ndarray,
    phase: float,
) -> dict[str, Any]:
    """Compare both gates on one unchanged state/action without propagation."""

    position_before = np.asarray(position, dtype=float).copy()
    velocity_before = np.asarray(velocity, dtype=float).copy()
    action_before = np.asarray(action, dtype=float).copy()
    current_acceleration, _, current_info = compute_dmp_transition(
        config=config,
        position=position_before,
        velocity=velocity_before,
        rl_action=action_before,
        active_goal=active_goal,
        terminal_goal=terminal_goal,
        phase=phase,
    )
    historical_acceleration, _, historical_info = (
        compute_historical_checkpoint_dmp_transition(
            config=config,
            position=position_before,
            velocity=velocity_before,
            rl_action=action_before,
            active_goal=active_goal,
            terminal_goal=terminal_goal,
            phase=phase,
        )
    )
    current_forcing = np.asarray(current_info["residual_drive"], dtype=float)
    historical_forcing = np.asarray(
        historical_info["forcing_contribution"], dtype=float
    )
    return {
        "current_gate": np.asarray(current_info["forcing_gate"], dtype=float).copy(),
        "historical_gate_xyz": np.asarray(
            historical_info["forcing_gate"], dtype=float
        ).copy(),
        "current_forcing_contribution": current_forcing.copy(),
        "historical_forcing_contribution": historical_forcing.copy(),
        "current_commanded_acceleration": np.asarray(
            current_acceleration, dtype=float
        ).copy(),
        "historical_commanded_acceleration": np.asarray(
            historical_acceleration, dtype=float
        ).copy(),
        "commanded_acceleration_delta_l2": float(
            np.linalg.norm(historical_acceleration - current_acceleration)
        ),
        "commanded_acceleration_direction_delta_deg": _angle_degrees(
            current_acceleration, historical_acceleration
        ),
        "forcing_contribution_delta_l2": float(
            np.linalg.norm(historical_forcing - current_forcing)
        ),
        "state_advanced": False,
    }


@contextmanager
def scoped_historical_multi_agent_transition(
    observer: TransitionObserver | None = None,
) -> Iterator[None]:
    """Temporarily route one synchronous evaluation through historical DMP.

    ``MultiAgentDMPEnv`` imports the shared propagation function into its own
    module namespace.  Replacing that single symbol inside a locked, bounded
    context preserves the complete environment step/reward/termination path
    without copying it or changing its default semantics.  Restoration is
    guaranteed by ``finally``, including when evaluation raises.
    """

    import Environment.multi_agent_dmp_env as environment_module

    with _TRANSITION_SCOPE_LOCK:
        original = environment_module.propagate_sac_dmp_action

        def historical_transition(**kwargs: Any) -> SACDMPTransition:
            transition = propagate_historical_sac_dmp_action(**kwargs)
            if observer is not None:
                observer(dict(kwargs), transition)
            return transition

        environment_module.propagate_sac_dmp_action = historical_transition
        try:
            yield
        finally:
            environment_module.propagate_sac_dmp_action = original


@contextmanager
def scoped_historical_preview_and_multi_agent_transition(
    *,
    preview_observer: TransitionObserver | None = None,
    execution_observer: TransitionObserver | None = None,
) -> Iterator[None]:
    """Route both FP-SHEP preview and real environment execution historically.

    ``policy_preview`` and ``multi_agent_dmp_env`` each bind the shared
    propagation function into their own module namespace at import time.  A
    real/preview identity scope therefore has to replace both bound symbols.
    The existing process-wide re-entrant lock prevents concurrent evaluation
    scopes from observing a partial replacement.  Both symbols are restored
    in ``finally`` even if preview or episode execution raises.

    This context is evaluation-only.  It does not modify the default scalar
    transition, FP-SHEP source code, or the persistent environment semantics.
    """

    import Environment.multi_agent_dmp_env as environment_module
    import planning.policy_preview as preview_module

    with _TRANSITION_SCOPE_LOCK:
        original_execution = environment_module.propagate_sac_dmp_action
        original_preview = preview_module.propagate_sac_dmp_action

        def historical_execution(**kwargs: Any) -> SACDMPTransition:
            transition = propagate_historical_sac_dmp_action(**kwargs)
            if execution_observer is not None:
                execution_observer(dict(kwargs), transition)
            return transition

        def historical_preview(**kwargs: Any) -> SACDMPTransition:
            transition = propagate_historical_sac_dmp_action(**kwargs)
            if preview_observer is not None:
                preview_observer(dict(kwargs), transition)
            return transition

        environment_module.propagate_sac_dmp_action = historical_execution
        preview_module.propagate_sac_dmp_action = historical_preview
        try:
            yield
        finally:
            preview_module.propagate_sac_dmp_action = original_preview
            environment_module.propagate_sac_dmp_action = original_execution
