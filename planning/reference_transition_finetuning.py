"""Targeted one-shot reference-transition SAC-DMP fine-tuning helpers.

This module is deliberately isolated from the historical single-agent
environment.  ``env.goal`` remains the terminal task goal for the complete
episode, while ``active_goal`` is the goal exposed to the Actor, DMP and
active-reference progress reward.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from Environment.frozen_sac_dmp_execution import build_historical_actor_observation
from Environment.single_agent_dmp_env import SingleAgentDMPEnv
from Guidance.reference_point_proposal_demo import ProposalConfig, propose_reference_points
from experiment_config import SACExperimentConfig
from planning.historical_forcing_gate import (
    HISTORICAL_GATE_NAME,
    compute_historical_checkpoint_dmp_transition,
)
from planning.policy_preview import adapt_candidate_proposals


TASK_TERMINAL = "terminal_navigation"
TASK_REFERENCE = "one_shot_reference_transition"
TASKS = (TASK_TERMINAL, TASK_REFERENCE)


def _vector3(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector with shape (3,)")
    return result


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


@dataclass
class BlockTaskSampler:
    """Exact episode-level 60/40 sampler using shuffled ten-episode blocks."""

    terminal_episodes_per_block: int = 6
    reference_episodes_per_block: int = 4

    def __post_init__(self) -> None:
        if self.terminal_episodes_per_block <= 0 or self.reference_episodes_per_block <= 0:
            raise ValueError("both task counts must be positive")
        self._pending: list[str] = []

    def sample(self, rng: np.random.Generator) -> str:
        if not self._pending:
            block = np.asarray(
                [TASK_TERMINAL] * int(self.terminal_episodes_per_block)
                + [TASK_REFERENCE] * int(self.reference_episodes_per_block),
                dtype=object,
            )
            rng.shuffle(block)
            self._pending = [str(value) for value in block.tolist()]
        return self._pending.pop()


@dataclass(frozen=True)
class WarmStartAudit:
    checkpoint_sha256_before: str
    actor_checkpoint_sha256: str
    actor_loaded_sha256: str
    critic_checkpoint_sha256: str
    critic_loaded_sha256: str
    critic_target_checkpoint_sha256: str
    critic_target_loaded_sha256: str
    log_alpha_checkpoint: float
    log_alpha_loaded: float
    actor_optimizer_state_entries: int
    critic_optimizer_state_entries: int
    alpha_optimizer_state_entries: int
    replay_size: int
    fine_tuning_num_timesteps: int
    fine_tuning_n_updates: int

    @property
    def weights_match(self) -> bool:
        return (
            self.actor_checkpoint_sha256 == self.actor_loaded_sha256
            and self.critic_checkpoint_sha256 == self.critic_loaded_sha256
            and self.critic_target_checkpoint_sha256 == self.critic_target_loaded_sha256
            and self.log_alpha_checkpoint == self.log_alpha_loaded
        )

    @property
    def fresh_training_state(self) -> bool:
        return (
            self.actor_optimizer_state_entries == 0
            and self.critic_optimizer_state_entries == 0
            and self.alpha_optimizer_state_entries == 0
            and self.replay_size == 0
            and self.fine_tuning_num_timesteps == 0
            and self.fine_tuning_n_updates == 0
        )


def load_checkpoint_weights_only(
    model: Any,
    checkpoint_path: str | Path,
    *,
    learning_rate: float,
) -> WarmStartAudit:
    """Load network/entropy values while retaining a fresh replay and optimizers."""

    checkpoint_path = Path(checkpoint_path)
    checkpoint_hash = sha256_file(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=model.device)

    model.actor.load_state_dict(checkpoint["actor"])
    model.critic.load_state_dict(checkpoint["critic"])
    model.critic_target.load_state_dict(checkpoint["critic_target"])

    # Explicitly discard historical Adam moments because replay distribution changes.
    model.actor.optimizer = torch.optim.Adam(model.actor.parameters(), lr=float(learning_rate))
    model.critic.optimizer = torch.optim.Adam(model.critic.parameters(), lr=float(learning_rate))
    if model.log_ent_coef is None or "log_ent_coef" not in checkpoint:
        raise RuntimeError("checkpoint-compatible automatic entropy state is required")
    model.log_ent_coef = (
        checkpoint["log_ent_coef"].to(model.device).detach().clone().requires_grad_(True)
    )
    model.ent_coef_optimizer = torch.optim.Adam(
        [model.log_ent_coef], lr=float(learning_rate)
    )
    model.num_timesteps = 0
    model._n_updates = 0

    replay_size = int(model.replay_buffer.size())
    audit = WarmStartAudit(
        checkpoint_sha256_before=checkpoint_hash,
        actor_checkpoint_sha256=state_dict_sha256(checkpoint["actor"]),
        actor_loaded_sha256=state_dict_sha256(model.actor.state_dict()),
        critic_checkpoint_sha256=state_dict_sha256(checkpoint["critic"]),
        critic_loaded_sha256=state_dict_sha256(model.critic.state_dict()),
        critic_target_checkpoint_sha256=state_dict_sha256(checkpoint["critic_target"]),
        critic_target_loaded_sha256=state_dict_sha256(model.critic_target.state_dict()),
        log_alpha_checkpoint=float(checkpoint["log_ent_coef"].reshape(-1)[0]),
        log_alpha_loaded=float(model.log_ent_coef.detach().cpu().reshape(-1)[0]),
        actor_optimizer_state_entries=len(model.actor.optimizer.state),
        critic_optimizer_state_entries=len(model.critic.optimizer.state),
        alpha_optimizer_state_entries=len(model.ent_coef_optimizer.state),
        replay_size=replay_size,
        fine_tuning_num_timesteps=int(model.num_timesteps),
        fine_tuning_n_updates=int(model._n_updates),
    )
    if not audit.weights_match or not audit.fresh_training_state:
        raise RuntimeError("weights-only warm start invariant failed")
    return audit


class ReferenceTransitionFineTuningEnv(SingleAgentDMPEnv):
    """Single-agent environment with one optional active-reference handoff.

    Invariant: ``self.goal == self.terminal_goal`` at every externally visible
    point.  The active reference is stored only in ``self.active_goal`` and
    ``self.dmp.goal``.
    """

    def __init__(
        self,
        *args: Any,
        task_sampler: BlockTaskSampler | None = None,
        forced_task: str | None = None,
        proposal_config: ProposalConfig | None = None,
        proposal_top_k: int = 10,
        reference_tolerance: float = 0.25,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if forced_task is not None and forced_task not in TASKS:
            raise ValueError(f"unsupported forced task: {forced_task}")
        if int(proposal_top_k) <= 0:
            raise ValueError("proposal_top_k must be positive")
        if float(reference_tolerance) <= 0.0:
            raise ValueError("reference_tolerance must be positive")
        self.task_sampler = task_sampler or BlockTaskSampler()
        self.forced_task = forced_task
        self.proposal_config = proposal_config or ProposalConfig()
        self.proposal_top_k = int(proposal_top_k)
        self.reference_tolerance = float(reference_tolerance)

        self.terminal_goal = self.goal.copy()
        self.active_goal = self.goal.copy()
        self.requested_task = TASK_TERMINAL
        self.episode_task = TASK_TERMINAL
        self.reference_generation_fallback = False
        self.temporary_reference: np.ndarray | None = None
        self.reference_reached = False
        self.reference_reached_step: int | None = None
        self.reference_handoff_count = 0
        self._progress_baseline_distance = 0.0
        self._last_reward_goal_eff = self.goal.copy()
        self._last_reward_forcing_gate = np.ones(3, dtype=float)

    def _assert_goal_separation_invariant(self) -> None:
        if not np.array_equal(np.asarray(self.goal), np.asarray(self.terminal_goal)):
            raise RuntimeError("env.goal leaked away from terminal_goal")
        if not np.array_equal(np.asarray(self.dmp.goal), np.asarray(self.active_goal)):
            raise RuntimeError("dmp.goal is not the active reference")

    def _retarget_latest_packet(self, goal: np.ndarray) -> None:
        if self.latest_sensor_packet is None:
            raise RuntimeError("sensor packet is unavailable")
        packet = self.latest_sensor_packet
        full = build_historical_actor_observation(
            position=self.dynamics.p,
            velocity=self.dynamics.v,
            active_goal=_vector3(goal, "active_goal"),
            phase=float(self.dmp.phase),
            k_alpha=float(self.dmp.config.K_alpha),
            k_beta=float(self.dmp.config.K_beta),
            current_scan=packet.current_scan,
            previous_scan=packet.previous_scan,
            goal_distance_clip=float(self.sensor.goal_distance_clip),
        )
        packet.observation = full[: self.sensor_observation_dim].copy()
        self.latest_observation = None

    def _set_active_goal_preserve_state(self, goal: np.ndarray) -> None:
        goal = _vector3(goal, "active_goal").copy()
        phase_before = float(self.dmp.phase)
        position_before = self.dynamics.p.copy()
        velocity_before = self.dynamics.v.copy()
        previous_scan_before = self.sensor._previous_scan.copy()
        self.active_goal = goal
        self.dmp.goal = goal.copy()
        self._retarget_latest_packet(goal)
        if float(self.dmp.phase) != phase_before:
            raise RuntimeError("active-goal switch reset DMP phase")
        if not np.array_equal(self.dynamics.p, position_before):
            raise RuntimeError("active-goal switch changed position")
        if not np.array_equal(self.dynamics.v, velocity_before):
            raise RuntimeError("active-goal switch changed velocity")
        if not np.array_equal(self.sensor._previous_scan, previous_scan_before):
            raise RuntimeError("active-goal switch changed LiDAR history")
        self._assert_goal_separation_invariant()

    def _select_temporary_reference(self) -> tuple[np.ndarray | None, int]:
        if self.latest_sensor_packet is None:
            raise RuntimeError("reset must create a sensor packet")
        proposals = propose_reference_points(
            self.dynamics.p,
            self.terminal_goal,
            self.dynamics.v,
            self.latest_sensor_packet,
            self.sensor,
            self.proposal_config,
            float(self.env_config.goal_tolerance),
        )
        selected = adapt_candidate_proposals(
            proposals,
            consumer_top_k=self.proposal_top_k,
        )
        if not selected:
            return None, len(proposals)
        return np.asarray(selected[0].point, dtype=float).copy(), len(proposals)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        observation, info = super().reset(seed=seed, options=options)
        self.terminal_goal = self.goal.copy()
        self.active_goal = self.terminal_goal.copy()
        self.dmp.goal = self.active_goal.copy()
        self.requested_task = (
            self.forced_task
            if self.forced_task is not None
            else self.task_sampler.sample(self.np_random)
        )
        self.episode_task = self.requested_task
        self.reference_generation_fallback = False
        self.temporary_reference = None
        self.reference_reached = False
        self.reference_reached_step = None
        self.reference_handoff_count = 0
        proposal_count = 0

        if self.requested_task == TASK_REFERENCE:
            reference, proposal_count = self._select_temporary_reference()
            if reference is None:
                self.reference_generation_fallback = True
                self.episode_task = TASK_TERMINAL
            else:
                self.temporary_reference = reference.copy()
                self._set_active_goal_preserve_state(reference)

        self._progress_baseline_distance = float(
            np.linalg.norm(self.active_goal - self.dynamics.p)
        )
        observation = self.get_observation()
        info.update(
            self._reference_info(
                handoff_event=False,
                progress=0.0,
                progress_baseline_before=self._progress_baseline_distance,
                progress_baseline_after=self._progress_baseline_distance,
                progress_reference=(
                    TASK_TERMINAL
                    if np.array_equal(self.active_goal, self.terminal_goal)
                    else TASK_REFERENCE
                ),
            )
        )
        info["proposal_count"] = int(proposal_count)
        self._assert_goal_separation_invariant()
        return observation, info

    def _compute_policy_action_direction(self, policy_action: np.ndarray) -> np.ndarray | None:
        """Use the exact DMP-step historical gate for reward reconstruction."""

        _ = policy_action
        controller = self.latest_controller_info
        goal_eff = _vector3(controller["goal_eff"], "controller goal_eff")
        forcing_gate = _vector3(controller["forcing_gate"], "controller forcing_gate")
        # Reward is evaluated after propagation, whereas the controller gate is
        # evaluated at the transition start.  Use the cached controller gate so
        # dynamics and reward share exactly the same action-to-drive mapping.
        goal_offset = _vector3(controller["goal_offset"], "controller goal_offset")
        forcing = _vector3(controller["forcing"], "controller forcing")
        action_component = (
            float(self.dmp.config.K_alpha)
            * float(self.dmp.config.K_beta)
            * goal_offset
            + forcing * forcing_gate
        )
        self._last_reward_goal_eff = goal_eff.copy()
        self._last_reward_forcing_gate = forcing_gate.copy()
        norm = float(np.linalg.norm(action_component))
        if norm < 1.0e-8:
            return None
        return action_component / norm

    def _reference_info(
        self,
        *,
        handoff_event: bool,
        progress: float,
        progress_baseline_before: float,
        progress_baseline_after: float,
        progress_reference: str,
    ) -> dict[str, Any]:
        return {
            "requested_task": self.requested_task,
            "episode_task": self.episode_task,
            "reference_generation_fallback": bool(self.reference_generation_fallback),
            "terminal_goal": self.terminal_goal.copy(),
            "active_goal": self.active_goal.copy(),
            "temporary_reference": (
                None if self.temporary_reference is None else self.temporary_reference.copy()
            ),
            "temporary_reference_reached": bool(self.reference_reached),
            "reference_reached_step": self.reference_reached_step,
            "reference_handoff_event": bool(handoff_event),
            "reference_handoff_count": int(self.reference_handoff_count),
            "active_goal_distance": float(np.linalg.norm(self.active_goal - self.dynamics.p)),
            "terminal_goal_distance": float(
                np.linalg.norm(self.terminal_goal - self.dynamics.p)
            ),
            "progress": float(progress),
            "progress_baseline_before": float(progress_baseline_before),
            "progress_baseline_after": float(progress_baseline_after),
            "progress_reference": progress_reference,
            "forcing_gate_semantics": HISTORICAL_GATE_NAME,
            "reward_goal_eff": self._last_reward_goal_eff.copy(),
            "reward_forcing_gate": self._last_reward_forcing_gate.copy(),
            "GAT_used": False,
            "FP_SHEP_selector_used": False,
            "repeated_waypoint_used": False,
            "hard_boundary_used": False,
        }

    def step(self, action: np.ndarray):
        if self.latest_sensor_packet is None:
            raise RuntimeError("reset must be called before step")
        self._assert_goal_separation_invariant()

        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.action_dim,):
            raise ValueError(f"action must have shape ({self.action_dim},)")
        action = np.clip(action, self.action_space.low, self.action_space.high)
        raw_action = action.copy()
        terminal_distance_before = float(
            np.linalg.norm(self.terminal_goal - self.dynamics.p)
        )
        action, action_guidance_weight = self._apply_action_guidance(
            action, terminal_distance_before
        )

        progress_baseline_before = float(self._progress_baseline_distance)
        progress_goal_before = self.active_goal.copy()
        acceleration, next_phase, controller_info = (
            compute_historical_checkpoint_dmp_transition(
                config=self.dmp.config,
                position=self.dynamics.p,
                velocity=self.dynamics.v,
                rl_action=action,
                active_goal=self.active_goal,
                terminal_goal=self.terminal_goal,
                phase=float(self.dmp.phase),
            )
        )
        self.dmp.phase = float(next_phase)
        applied_acceleration = np.clip(
            acceleration,
            self.dynamics.accelerate_min,
            self.dynamics.accelerate_max,
        )
        self.latest_controller_info = controller_info
        next_state = self.dynamics.step(applied_acceleration)
        for obstacle in self.dynamic_obstacles:
            obstacle.step(self.dynamics.dt)
        self.steps += 1
        self.action_guidance_step += 1

        self.latest_sensor_packet = self.sensor.sense(
            self.dynamics.p,
            self.dynamics.v,
            progress_goal_before,
            self.static_obstacles,
            self.dynamic_obstacles,
        )
        active_distance_after = float(
            np.linalg.norm(progress_goal_before - self.dynamics.p)
        )
        progress = progress_baseline_before - active_distance_after

        obstacle_potential_penalty = self._compute_obstacle_potential_penalty(raw_action)
        boundary_potential_penalty = self._compute_boundary_potential_penalty(raw_action)
        step_reward = float(self.env_config.step_reward_weight) * progress
        step_penalty = float(self.env_config.step_penalty)
        reward = (
            step_reward
            - obstacle_potential_penalty
            - boundary_potential_penalty
            - step_penalty
        )

        terminal_distance_after = float(
            np.linalg.norm(self.terminal_goal - self.dynamics.p)
        )
        terminal_success = terminal_distance_after <= float(self.env_config.goal_tolerance)
        collision = self._check_collision()
        terminated = bool(terminal_success or collision)
        truncated = bool(
            (not terminated) and (self.steps >= int(self.env_config.max_steps))
        )
        if collision:
            reward -= float(self.env_config.collision_penalty)
        elif terminal_success:
            reward += float(self.env_config.success_bonus)
        timeout_penalty = float(self.env_config.timeout_penalty) if truncated else 0.0
        reward -= timeout_penalty

        temporary_reached_now = bool(
            self.episode_task == TASK_REFERENCE
            and not self.reference_reached
            and active_distance_after <= self.reference_tolerance
        )
        handoff_event = bool(
            temporary_reached_now
            and not collision
            and not terminal_success
            and not truncated
        )
        if temporary_reached_now:
            self.reference_reached = True
            self.reference_reached_step = int(self.steps)
        if handoff_event:
            self.reference_handoff_count += 1
            self._set_active_goal_preserve_state(self.terminal_goal)
            # Explicitly establish the new terminal-reference baseline.  No
            # cross-reference distance difference is ever evaluated.
            self._progress_baseline_distance = terminal_distance_after
        else:
            self._progress_baseline_distance = active_distance_after

        observation = self.get_observation()
        info = self._build_info(
            success=terminal_success,
            collision=collision,
            truncated=truncated,
            commanded_acceleration=acceleration,
            applied_acceleration=applied_acceleration,
            next_state=next_state,
            step_reward=step_reward,
            obstacle_potential_penalty=obstacle_potential_penalty,
            boundary_potential_penalty=boundary_potential_penalty,
            step_penalty=step_penalty,
            timeout_penalty=timeout_penalty,
            raw_action=raw_action,
            guided_action=action,
            action_guidance_weight=action_guidance_weight,
        )
        info.update(
            self._reference_info(
                handoff_event=handoff_event,
                progress=progress,
                progress_baseline_before=progress_baseline_before,
                progress_baseline_after=self._progress_baseline_distance,
                progress_reference=(
                    TASK_TERMINAL
                    if np.array_equal(progress_goal_before, self.terminal_goal)
                    else TASK_REFERENCE
                ),
            )
        )
        info.update(
            {
                "success": bool(terminal_success),
                "terminal_success": bool(terminal_success),
                "temporary_reached_this_step": bool(temporary_reached_now),
                "commanded_acceleration": np.asarray(acceleration, dtype=float).copy(),
                "applied_acceleration": np.asarray(applied_acceleration, dtype=float).copy(),
                "raw_action": raw_action.copy(),
                "guided_action": np.asarray(action, dtype=float).copy(),
                "reward_total": float(reward),
                "reward_progress": float(step_reward),
                "progress_reference_before_step": progress_goal_before.copy(),
                "dynamics_goal_eff": np.asarray(controller_info["goal_eff"], dtype=float).copy(),
                "dynamics_forcing_gate": np.asarray(
                    controller_info["forcing_gate"], dtype=float
                ).copy(),
            }
        )
        self._assert_goal_separation_invariant()
        return observation, float(reward), terminated, truncated, info


def final_curriculum_state() -> dict[str, Any]:
    return {
        "enabled": True,
        "level": 4,
        "dense_scene_probability": 0.4,
        "dense_dynamic_probability": 0.5,
        "episode_count": 0,
        "level_episode_count": 0,
        "success_rate": 0.0,
    }


def build_reference_transition_env(
    config: SACExperimentConfig,
    *,
    forced_task: str | None = None,
    proposal_config: ProposalConfig | None = None,
    proposal_top_k: int = 10,
    reference_tolerance: float = 0.25,
    curriculum_state: dict[str, Any] | None = None,
) -> ReferenceTransitionFineTuningEnv:
    """Build the historical 122-D single-agent environment with fixed level 4."""

    state = final_curriculum_state() if curriculum_state is None else curriculum_state
    fixed_box = config.build_fixed_box()
    env = ReferenceTransitionFineTuningEnv(
        dynamics_config=config.build_dynamics_config(),
        sensor_config=config.build_sensor_config(),
        dmp_config=config.build_dmp_config(),
        env_config=replace(config.build_env_config(), action_guidance_enabled=False),
        start_goal_generator=config.build_start_goal_generator(),
        static_obstacles=config.build_static_obstacles(fixed_box),
        static_obstacle_generator=config.build_static_obstacle_generator(
            fixed_box, curriculum_state=state
        ),
        dynamic_obstacles=[],
        dynamic_obstacle_generator=config.build_dynamic_obstacle_generator(
            curriculum_state=state
        ),
        forced_task=forced_task,
        proposal_config=proposal_config,
        proposal_top_k=proposal_top_k,
        reference_tolerance=reference_tolerance,
    )
    env._default_start = np.asarray(config.default_start, dtype=float)
    env._default_goal = np.asarray(config.default_goal, dtype=float)
    env.goal = env._default_goal.copy()
    env.terminal_goal = env.goal.copy()
    env.active_goal = env.goal.copy()
    env.curriculum_state = state
    return env


def smoke_training_config(base: SACExperimentConfig) -> SACExperimentConfig:
    """Return the fixed, non-searchable fine-tuning configuration."""

    return replace(
        base,
        learning_rate=1.0e-4,
        learning_starts=5_000,
        train_freq=1,
        gradient_steps=1,
        batch_size=256,
        buffer_size=1_000_000,
        max_steps=220,
        total_timesteps=100_000,
        training_scene_mixture_enabled=True,
        training_dense_scene_probability=0.4,
        training_dense_dynamic_probability=0.5,
        curriculum_enabled=False,
        action_guidance_enabled=False,
    )
