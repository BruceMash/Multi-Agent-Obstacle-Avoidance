"""Shared, side-effect-free primitives for frozen SAC-DMP execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from Controller.dmp_rl import DMPConfig, compute_dmp_transition
from Entity.KinematicModel import propagate_point_mass


HISTORICAL_CHECKPOINT_OBSERVATION_DIM = 122


def actor_observation_dim(ray_count: int) -> int:
    """Return 3 velocity + 3 direction + 1 distance + two scans + 3 DMP fields."""
    ray_count = int(ray_count)
    if ray_count <= 0:
        raise ValueError("ray_count must be positive")
    return 10 + 2 * ray_count


@dataclass(frozen=True)
class SACDMPTransition:
    position: np.ndarray
    velocity: np.ndarray
    commanded_acceleration: np.ndarray
    applied_acceleration: np.ndarray
    unclipped_next_velocity: np.ndarray
    phase: float
    action: np.ndarray
    controller_info: dict[str, Any]

    @property
    def state(self) -> np.ndarray:
        return np.concatenate([self.position, self.velocity])


def build_actor_observation(
    *,
    velocity: np.ndarray,
    active_goal: np.ndarray,
    position: np.ndarray,
    current_scan: np.ndarray,
    previous_scan: np.ndarray,
    goal_distance_clip: float,
    phase: float,
    k_alpha: float,
    k_beta: float,
) -> np.ndarray:
    """Build the SAC-DMP actor vector for any frozen spatial-ray contract."""
    position = np.asarray(position, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    active_goal = np.asarray(active_goal, dtype=float)
    current_scan = np.asarray(current_scan, dtype=np.float32)
    previous_scan = np.asarray(previous_scan, dtype=np.float32)
    if position.shape != (3,) or velocity.shape != (3,) or active_goal.shape != (3,):
        raise ValueError("position, velocity and active_goal must have shape (3,)")
    if current_scan.shape != previous_scan.shape:
        raise ValueError("current_scan and previous_scan must have identical shapes")
    if current_scan.size <= 0:
        raise ValueError("actor observation requires at least one spatial ray")
    goal_distance_clip = float(goal_distance_clip)
    if not np.isfinite(goal_distance_clip) or goal_distance_clip <= 0.0:
        raise ValueError("goal_distance_clip must be positive and finite")
    delta = active_goal - position
    distance = float(np.linalg.norm(delta))
    direction = np.zeros(3, dtype=float) if distance < 1.0e-8 else delta / distance
    observation = np.concatenate(
        [
            velocity.astype(np.float32),
            direction.astype(np.float32),
            np.asarray([np.clip(distance / goal_distance_clip, 0.0, 1.0)], dtype=np.float32),
            current_scan.reshape(-1),
            previous_scan.reshape(-1),
            np.asarray([phase, k_alpha, k_beta], dtype=np.float32),
        ]
    ).astype(np.float32)
    expected_dim = actor_observation_dim(current_scan.size)
    if observation.shape != (expected_dim,):
        raise ValueError(f"actor observation must have shape ({expected_dim},), got {observation.shape}")
    if not np.all(np.isfinite(observation)):
        raise ValueError("actor observation must be finite")
    return observation


def build_historical_actor_observation(**kwargs: Any) -> np.ndarray:
    """Build the exact 122-D, 56-direction input for the historical checkpoint."""
    observation = build_actor_observation(**kwargs)
    if observation.shape != (HISTORICAL_CHECKPOINT_OBSERVATION_DIM,):
        ray_count = (observation.size - 10) // 2
        raise ValueError(f"historical checkpoint requires 56 rays, got {ray_count}")
    return observation


def predict_policy_action(policy: Any, observation: np.ndarray) -> np.ndarray:
    """Run one deterministic SAC-DMP inference for the policy's frozen input width."""
    observation = np.asarray(observation, dtype=np.float32)
    if observation.ndim != 1 or observation.size < 12:
        raise ValueError("observation must be a one-dimensional SAC-DMP feature vector")
    action, _ = policy.predict(observation, deterministic=True)
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (6,):
        raise ValueError(f"SAC-DMP action must have shape (6,), got {action.shape}")
    if not np.all(np.isfinite(action)):
        raise ValueError("policy action must be finite")
    return action


def predict_frozen_action(policy: Any, observation: np.ndarray) -> np.ndarray:
    """Run one deterministic physical-action inference without policy updates."""
    observation = np.asarray(observation, dtype=np.float32)
    if observation.shape != (HISTORICAL_CHECKPOINT_OBSERVATION_DIM,):
        raise ValueError("observation must have shape (122,)")
    return predict_policy_action(policy, observation)


def predict_frozen_actions(
    policy: Any,
    observations: np.ndarray,
    *,
    expected_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Run deterministic frozen-policy inference for one or more observations."""
    observations = np.asarray(observations, dtype=np.float32)
    if observations.ndim not in {1, 2}:
        raise ValueError("observations must be a feature vector or a batch of feature vectors")
    actions, _ = policy.predict(observations, deterministic=True)
    actions = np.asarray(actions, dtype=np.float32)
    if expected_shape is not None and actions.shape != expected_shape:
        raise ValueError(f"policy action shape {actions.shape} != {expected_shape}")
    if not np.all(np.isfinite(actions)):
        raise ValueError("policy actions must be finite")
    return actions


def freeze_policy(policy: Any) -> Any:
    """Freeze the already-loaded SAC actor in place; never creates a new actor."""
    actor = getattr(policy, "actor", None)
    if actor is None:
        raise ValueError("policy must expose the loaded actor")
    actor.eval()
    actor.requires_grad_(False)
    return policy


def propagate_sac_dmp_action(
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
    """Shared DMP + point-mass transition used by real and preview execution."""
    action = np.asarray(action, dtype=np.float32)
    acceleration, next_phase, controller_info = compute_dmp_transition(
        config=dmp_config,
        position=position,
        velocity=velocity,
        rl_action=action,
        active_goal=active_goal,
        terminal_goal=terminal_goal,
        phase=phase,
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
