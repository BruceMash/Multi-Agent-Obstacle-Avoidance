"""Deterministic sensing-matched receding-horizon optimization baseline.

This module intentionally implements an ``NMPC-style`` baseline rather than
claiming a canonical NMPC package implementation.  It uses fixed-iteration
projected Adam direct shooting, exact point-mass propagation, hard component
acceleration/velocity bounds, and a disclosed soft penalty for current
untyped LiDAR surface returns.  It never reads obstacle geometry, obstacle
state, peer identity, or exact peer state.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import torch

from planning.final_four_stage_benchmark import (
    _trajectory_metrics,
    step_direct_accelerations,
)
from planning.sensing_matched_classical import reconstruct_sensing_matched_perception


METHOD_LABEL = "Sensing-Matched NMPC-style"
SOLVER_LABEL = "fixed_iteration_projected_adam_direct_shooting"
SCHEMA_VERSION = "sensing_matched_nmpc_style_v1"


@dataclass(frozen=True)
class NMPCStyleConfig:
    horizon_steps: int
    iterations: int
    goal_weight: float
    control_weight: float
    smoothness_weight: float
    safety_weight: float
    learning_rate: float = 0.12
    gradient_tolerance: float = 1.0e-5
    safety_distance_m: float = 0.6
    goal_distance_scale_m: float = 9.0
    fallback: str = "previous_admissible_control_else_zero"
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.horizon_steps <= 0 or self.iterations <= 0:
            raise ValueError("horizon and iteration count must be positive")
        for name in (
            "goal_weight",
            "control_weight",
            "smoothness_weight",
            "safety_weight",
            "learning_rate",
            "gradient_tolerance",
            "safety_distance_m",
            "goal_distance_scale_m",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.fallback != "previous_admissible_control_else_zero":
            raise ValueError("unsupported fallback contract")
        if self.device != "cpu":
            raise ValueError("the frozen deterministic baseline uses CPU only")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tensor(value: Any) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float64, device="cpu")


def exact_point_mass_rollout(
    position: torch.Tensor,
    velocity: torch.Tensor,
    controls: torch.Tensor,
    *,
    dt: float,
    acceleration_min: torch.Tensor,
    acceleration_max: torch.Tensor,
    velocity_min: torch.Tensor,
    velocity_max: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match ``PartialDynamic.step`` for a batch of shooting sequences."""

    positions: list[torch.Tensor] = []
    velocities: list[torch.Tensor] = []
    applied: list[torch.Tensor] = []
    p = position
    v = velocity
    for index in range(int(controls.shape[1])):
        acceleration = torch.maximum(
            torch.minimum(controls[:, index], acceleration_max), acceleration_min
        )
        next_velocity = torch.maximum(
            torch.minimum(v + acceleration * float(dt), velocity_max), velocity_min
        )
        next_position = p + v * float(dt) + 0.5 * acceleration * float(dt) ** 2
        positions.append(next_position)
        velocities.append(next_velocity)
        applied.append(acceleration)
        p = next_position
        v = next_velocity
    return (
        torch.stack(positions, dim=1),
        torch.stack(velocities, dim=1),
        torch.stack(applied, dim=1),
    )


class SensingMatchedNMPCStyle:
    """Warm-started deterministic direct-shooting controller."""

    def __init__(self, config: NMPCStyleConfig, num_agents: int = 3) -> None:
        self.config = config
        self.num_agents = int(num_agents)
        self._warm_controls: np.ndarray | None = None
        self._previous_controls = np.zeros((self.num_agents, 3), dtype=float)

    def reset(self) -> None:
        self._warm_controls = None
        self._previous_controls = np.zeros((self.num_agents, 3), dtype=float)

    def _initial_controls(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        goal: np.ndarray,
        a_min: np.ndarray,
        a_max: np.ndarray,
        v_min: np.ndarray,
        v_max: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, bool]:
        horizon = int(self.config.horizon_steps)
        if self._warm_controls is not None:
            shifted = np.concatenate(
                [self._warm_controls[:, 1:, :], self._warm_controls[:, -1:, :]], axis=1
            )
            return np.clip(shifted, a_min[:, None, :], a_max[:, None, :]), True
        duration = max(horizon * float(dt), float(dt))
        desired_velocity = np.clip((goal - position) / duration, v_min, v_max)
        first = np.clip((desired_velocity - velocity) / float(dt), a_min, a_max)
        controls = np.zeros((self.num_agents, horizon, 3), dtype=float)
        controls[:, 0, :] = first
        return controls, False

    @staticmethod
    def _padded_surfaces(perception: Any, num_agents: int) -> tuple[np.ndarray, np.ndarray]:
        maximum = max((len(row) for row in perception.local_surfaces), default=0)
        maximum = max(maximum, 1)
        points = np.zeros((num_agents, maximum, 3), dtype=float)
        mask = np.zeros((num_agents, maximum), dtype=bool)
        for agent_id, row in enumerate(perception.local_surfaces):
            for surface_id, surface in enumerate(row):
                points[agent_id, surface_id] = np.asarray(surface.point, dtype=float)
                mask[agent_id, surface_id] = True
        return points, mask

    def _fallback(self, a_min: np.ndarray, a_max: np.ndarray) -> np.ndarray:
        if (
            self._previous_controls.shape == (self.num_agents, 3)
            and np.all(np.isfinite(self._previous_controls))
        ):
            return np.clip(self._previous_controls, a_min, a_max)
        return np.zeros((self.num_agents, 3), dtype=float)

    def plan(self, env: Any) -> tuple[np.ndarray, dict[str, Any]]:
        started = time.perf_counter_ns()
        perception = reconstruct_sensing_matched_perception(env)
        core_started = time.perf_counter_ns()
        position_np = np.asarray(env._positions(), dtype=float)
        velocity_np = np.asarray(env._velocities(), dtype=float)
        goal_np = np.asarray(env.goals, dtype=float)
        completed = np.asarray(env.success_rewarded_mask, dtype=bool)
        def bound_vector(value: Any) -> np.ndarray:
            return np.broadcast_to(np.asarray(value, dtype=float), (3,)).copy()

        a_min_np = np.stack([bound_vector(item.accelerate_min) for item in env.dynamics])
        a_max_np = np.stack([bound_vector(item.accelerate_max) for item in env.dynamics])
        v_min_np = np.stack([bound_vector(item.velocity_min) for item in env.dynamics])
        v_max_np = np.stack([bound_vector(item.velocity_max) for item in env.dynamics])
        dt = float(env.dynamics[0].dt)
        initial, warm_started = self._initial_controls(
            position_np,
            velocity_np,
            goal_np,
            a_min_np,
            a_max_np,
            v_min_np,
            v_max_np,
            dt,
        )
        surface_np, surface_mask_np = self._padded_surfaces(perception, self.num_agents)

        position = _tensor(position_np)
        velocity = _tensor(velocity_np)
        goal = _tensor(goal_np)
        a_min = _tensor(a_min_np)
        a_max = _tensor(a_max_np)
        v_min = _tensor(v_min_np)
        v_max = _tensor(v_max_np)
        surfaces = _tensor(surface_np)
        surface_mask = torch.as_tensor(surface_mask_np, dtype=torch.bool)
        active_mask = _tensor((~completed).astype(float))
        completed_tensor = torch.as_tensor(completed, dtype=torch.bool)
        controls = _tensor(initial)
        first_cost = None
        final_cost = None
        final_components: dict[str, float] = {}
        final_gradient_norm = float("inf")
        iterations_executed = 0
        solver_status = "fixed_iteration"
        failure_reason = None
        fallback_used = False
        minimum_predicted_clearance = float("inf")
        maximum_safety_violation = 0.0
        try:
            first_moment = torch.zeros_like(controls)
            second_moment = torch.zeros_like(controls)
            for iteration in range(1, int(self.config.iterations) + 1):
                controls = controls.detach().requires_grad_(True)
                predicted_p, _, applied = exact_point_mass_rollout(
                    position,
                    velocity,
                    controls,
                    dt=dt,
                    acceleration_min=a_min,
                    acceleration_max=a_max,
                    velocity_min=v_min,
                    velocity_max=v_max,
                )
                normalized_goal_error = (predicted_p - goal[:, None, :]) / float(
                    self.config.goal_distance_scale_m
                )
                goal_cost_by_agent = (
                    torch.mean(torch.sum(normalized_goal_error**2, dim=2), dim=1)
                    + torch.sum(normalized_goal_error[:, -1, :] ** 2, dim=1)
                )
                a_scale = torch.maximum(torch.abs(a_min), torch.abs(a_max)).clamp_min(1.0e-9)[:, None, :]
                control_cost_by_agent = torch.mean(
                    torch.sum((applied / a_scale) ** 2, dim=2), dim=1
                )
                previous = _tensor(self._previous_controls)[:, None, :]
                deltas = torch.cat([applied[:, :1, :] - previous, torch.diff(applied, dim=1)], dim=1)
                smoothness_cost_by_agent = torch.mean(
                    torch.sum((deltas / a_scale) ** 2, dim=2), dim=1
                )
                distances = torch.linalg.vector_norm(
                    predicted_p[:, :, None, :] - surfaces[:, None, :, :], dim=3
                )
                valid_distances = torch.where(
                    surface_mask[:, None, :], distances, torch.full_like(distances, float("inf"))
                )
                violations = torch.relu(float(self.config.safety_distance_m) - valid_distances)
                safety_cost_by_agent = torch.mean(
                    torch.sum((violations / float(self.config.safety_distance_m)) ** 2, dim=2), dim=1
                )
                per_agent = (
                    float(self.config.goal_weight) * goal_cost_by_agent
                    + float(self.config.control_weight) * control_cost_by_agent
                    + float(self.config.smoothness_weight) * smoothness_cost_by_agent
                    + float(self.config.safety_weight) * safety_cost_by_agent
                )
                denominator = active_mask.sum().clamp_min(1.0)
                loss = torch.sum(per_agent * active_mask) / denominator
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("non-finite objective")
                gradient = torch.autograd.grad(loss, controls)[0]
                if not bool(torch.all(torch.isfinite(gradient))):
                    raise FloatingPointError("non-finite gradient")
                gradient = gradient * active_mask[:, None, None]
                final_gradient_norm = float(torch.linalg.vector_norm(gradient).detach())
                current_cost = float(loss.detach())
                if first_cost is None:
                    first_cost = current_cost
                final_cost = current_cost
                final_components = {
                    "goal": float(torch.sum(goal_cost_by_agent * active_mask).detach() / denominator),
                    "control": float(torch.sum(control_cost_by_agent * active_mask).detach() / denominator),
                    "smoothness": float(torch.sum(smoothness_cost_by_agent * active_mask).detach() / denominator),
                    "safety": float(torch.sum(safety_cost_by_agent * active_mask).detach() / denominator),
                }
                finite_distances = valid_distances[torch.isfinite(valid_distances)]
                minimum_predicted_clearance = (
                    float(torch.min(finite_distances).detach()) if finite_distances.numel() else float("inf")
                )
                maximum_safety_violation = (
                    float(torch.max(violations).detach()) if violations.numel() else 0.0
                )
                iterations_executed = iteration
                if final_gradient_norm <= float(self.config.gradient_tolerance):
                    solver_status = "gradient_tolerance"
                    controls = controls.detach()
                    break
                beta1, beta2 = 0.9, 0.999
                first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
                second_moment = beta2 * second_moment + (1.0 - beta2) * gradient.square()
                corrected_first = first_moment / (1.0 - beta1**iteration)
                corrected_second = second_moment / (1.0 - beta2**iteration)
                with torch.no_grad():
                    controls = controls - float(self.config.learning_rate) * corrected_first / (
                        torch.sqrt(corrected_second) + 1.0e-8
                    )
                    controls = torch.maximum(
                        torch.minimum(controls, a_max[:, None, :]), a_min[:, None, :]
                    )
                    controls[completed_tensor] = 0.0
            plan_np = controls.detach().cpu().numpy()
            if not np.all(np.isfinite(plan_np)):
                raise FloatingPointError("non-finite optimized controls")
            selected = np.clip(plan_np[:, 0, :], a_min_np, a_max_np)
            selected[completed] = 0.0
            self._warm_controls = plan_np.copy()
            self._previous_controls = selected.copy()
        except Exception as exc:  # fixed deterministic safety fallback
            selected = self._fallback(a_min_np, a_max_np)
            selected[completed] = 0.0
            self._warm_controls = None
            fallback_used = True
            solver_status = "fallback"
            failure_reason = f"{type(exc).__name__}: {exc}"

        core_ms = (time.perf_counter_ns() - core_started) / 1.0e6
        total_ms = (time.perf_counter_ns() - started) / 1.0e6
        return selected, {
            "schema_version": SCHEMA_VERSION,
            "planner": METHOD_LABEL,
            "solver": SOLVER_LABEL,
            "runtime_ms": float(total_ms),
            "perception_adapter_runtime_ms": float(perception.adapter_runtime_ms),
            "solver_core_runtime_ms": float(core_ms),
            "visible_hit_count": int(perception.visible_hit_count),
            "observation_equivalence_hashes": perception.observation_hashes,
            "warm_started": bool(warm_started),
            "iterations_executed": int(iterations_executed),
            "solver_status": solver_status,
            "failure_reason": failure_reason,
            "fallback_used": bool(fallback_used),
            "objective_initial": first_cost,
            "objective_final": final_cost,
            "objective_components_unweighted": final_components,
            "gradient_norm_final": float(final_gradient_norm),
            "minimum_predicted_surface_distance_m": float(minimum_predicted_clearance),
            "maximum_soft_safety_violation_m": float(maximum_safety_violation),
        }


def run_sensing_matched_nmpc_episode(
    *,
    environment_builder: Any,
    multi_config: Any,
    scenario: str,
    seed: int,
    peer_radius: float,
    planner_config: NMPCStyleConfig,
    retain_trajectory: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    env, scene_metadata = environment_builder(
        config=multi_config,
        scenario=str(scenario),
        seed=int(seed),
        peer_radius=float(peer_radius),
    )
    controller = SensingMatchedNMPCStyle(planner_config, int(env.num_agents))
    started = time.perf_counter_ns()
    try:
        starts = np.asarray(env.starts, dtype=float).copy()
        goals = np.asarray(env.goals, dtype=float).copy()
        positions = [env._positions().copy()]
        velocities = [env._velocities().copy()]
        accelerations: list[np.ndarray] = []
        runtime_rows: list[dict[str, Any]] = []
        completion_steps: list[int | None] = [None] * int(env.num_agents)
        min_clearance = np.full(int(env.num_agents), float("inf"))
        min_peer = np.full(int(env.num_agents), float("inf"))
        obstacle_collision = inter_agent_collision = boundary_collision = False
        terminated = truncated = False
        last_info: dict[str, Any] = {}
        while not (terminated or truncated):
            acceleration, planner_info = controller.plan(env)
            runtime_rows.append(
                {
                    "stage": scene_metadata.get("stage"),
                    "family": scene_metadata.get("family"),
                    "scenario_id": str(scenario),
                    "seed": int(seed),
                    "method": "sensing_matched_nmpc_style",
                    "decision_index": int(env.steps),
                    **planner_info,
                }
            )
            terminated, truncated, last_info = step_direct_accelerations(env, acceleration)
            positions.append(env._positions().copy())
            velocities.append(env._velocities().copy())
            applied = np.asarray(last_info["applied_accelerations"], dtype=float)
            accelerations.append(applied)
            min_clearance = np.minimum(min_clearance, np.asarray(last_info["min_clearances"], dtype=float))
            pairwise = np.asarray(last_info["pairwise_distances"], dtype=float)
            for agent_id in range(int(env.num_agents)):
                peer_values = np.delete(pairwise[agent_id], agent_id)
                if peer_values.size:
                    min_peer[agent_id] = min(min_peer[agent_id], float(np.min(peer_values)))
                if completion_steps[agent_id] is None and bool(last_info["success_mask"][agent_id]):
                    completion_steps[agent_id] = int(env.steps)
            obstacle_collision |= bool(np.any(last_info["obstacle_collision_mask"]))
            inter_agent_collision |= bool(np.any(last_info["inter_agent_collision_mask"]))
            boundary_collision |= bool(np.any(last_info["boundary_collision_mask"]))
        position_array = np.stack(positions)
        velocity_array = np.stack(velocities)
        acceleration_array = np.stack(accelerations)
        metrics = _trajectory_metrics(position_array, velocity_array, acceleration_array, float(env.dynamics[0].dt))
        success = bool(last_info.get("success", False))
        collision = bool(obstacle_collision or inter_agent_collision or boundary_collision)
        timeout = bool(truncated)
        if success:
            reason = "success"
        elif obstacle_collision:
            reason = "obstacle_collision"
        elif inter_agent_collision:
            reason = "inter_agent_collision"
        elif timeout:
            reason = "timeout"
        else:
            reason = "other"
        path_lengths = np.asarray(metrics["path_lengths"], dtype=float)
        straight = np.linalg.norm(goals - starts, axis=1)
        final_collision_mask = np.asarray(last_info["collision_mask"], dtype=bool)
        agents: list[dict[str, Any]] = []
        for agent_id in range(int(env.num_agents)):
            completed = completion_steps[agent_id] is not None and not final_collision_mask[agent_id]
            agents.append(
                {
                    "stage": scene_metadata.get("stage"),
                    "family": scene_metadata.get("family"),
                    "scenario_id": str(scenario),
                    "seed": int(seed),
                    "method": "sensing_matched_nmpc_style",
                    "agent_id": int(agent_id),
                    "agent_terminal_completed": bool(completed),
                    "agent_collision": bool(final_collision_mask[agent_id]),
                    "agent_path_length_m": float(path_lengths[agent_id]),
                    "agent_path_efficiency": float(straight[agent_id] / max(path_lengths[agent_id], 1.0e-9)) if completed else None,
                    "completion_step": completion_steps[agent_id],
                    "minimum_sensor_clearance_m": float(min_clearance[agent_id]),
                    "minimum_peer_distance_m": float(min_peer[agent_id]),
                }
            )
        decision_times = np.asarray([row["runtime_ms"] for row in runtime_rows], dtype=float)
        episode = {
            "stage": scene_metadata.get("stage"),
            "family": scene_metadata.get("family"),
            "scenario_id": str(scenario),
            "seed": int(seed),
            "method": "sensing_matched_nmpc_style",
            "success": success,
            "collision": collision,
            "any_collision": collision,
            "obstacle_collision": bool(obstacle_collision),
            "inter_agent_collision": bool(inter_agent_collision),
            "boundary_collision": bool(boundary_collision),
            "timeout": timeout,
            "termination_reason": reason,
            "steps": int(env.steps),
            "termination_time_s": float(env.steps) * float(env.dynamics[0].dt),
            "agent_completion_count": int(sum(item is not None for item in completion_steps)),
            "agent_completion_rate": float(sum(item is not None for item in completion_steps) / int(env.num_agents)),
            "per_agent_path_length_m": path_lengths.tolist(),
            "team_path_length_m": float(metrics["path_length_team_sum"]),
            "trajectory_smoothness_team_mean": float(metrics["trajectory_smoothness_team_mean"]),
            "planning_decision_count": len(runtime_rows),
            "planning_runtime_ms": float(np.sum(decision_times)),
            "decision_latency_mean_ms": float(np.mean(decision_times)),
            "decision_latency_p95_ms": float(np.percentile(decision_times, 95)),
            "decision_latency_max_ms": float(np.max(decision_times)),
            "fallback_decision_count": int(sum(bool(row["fallback_used"]) for row in runtime_rows)),
            "episode_wall_runtime_ms": (time.perf_counter_ns() - started) / 1.0e6,
        }
        trajectory = {
            "positions": position_array.tolist() if retain_trajectory else None,
            "velocities": velocity_array.tolist() if retain_trajectory else None,
            "accelerations": acceleration_array.tolist() if retain_trajectory else None,
            **metrics,
        }
        return episode, agents, runtime_rows, trajectory
    finally:
        env.close()


def configuration_from_mapping(value: Mapping[str, Any]) -> NMPCStyleConfig:
    return NMPCStyleConfig(**{key: value[key] for key in NMPCStyleConfig.__dataclass_fields__ if key in value})
