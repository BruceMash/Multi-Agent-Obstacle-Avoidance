from __future__ import annotations

import copy
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

from Entity.KinematicModel import PartialDynamic
from Entity.sensors import LocalObstacleSensor
from Environment.multi_agent_dmp_env import MultiAgentDMPEnv, MultiAgentEnvConfig


@dataclass
class MultiAgentContinuousEnvConfig(MultiAgentEnvConfig):
    """直接加速度多智能体环境配置。"""


class MultiAgentContinuousEnv(MultiAgentDMPEnv):
    """
    使用三维净线加速度动作的矩阵式多智能体环境。

    该环境复用 MultiAgentDMPEnv 的场景生成、传感器、碰撞检测、奖励与
    终止逻辑，但不创建或调用任何 DMP 控制器。每个智能体的策略动作直接
    表示 ``[a_x, a_y, a_z]``，并在进入质点动力学前按加速度边界裁剪。
    """

    def __init__(
        self,
        dynamics_config,
        sensor_config=None,
        env_config=None,
        start_goal_generator: (
            Callable[[np.random.Generator], tuple[np.ndarray, np.ndarray]] | None
        ) = None,
        static_obstacles=None,
        static_obstacle_generator=None,
        dynamic_obstacles=None,
        dynamic_obstacle_generator=None,
        render_mode=None,
    ):
        if render_mode not in {None, "human"}:
            raise ValueError("render_mode must be None or 'human'")

        self.dynamics_config = copy.deepcopy(dynamics_config)
        self.sensor_config = copy.deepcopy(sensor_config or {})

        if env_config is None:
            self.env_config = MultiAgentContinuousEnvConfig()
        elif isinstance(env_config, MultiAgentEnvConfig):
            self.env_config = env_config
        elif isinstance(env_config, dict):
            self.env_config = MultiAgentContinuousEnvConfig(**env_config)
        else:
            self.env_config = MultiAgentContinuousEnvConfig(**vars(env_config))

        self.num_agents = int(self.env_config.num_agents)
        self.dynamics = [
            PartialDynamic(copy.deepcopy(self.dynamics_config))
            for _ in range(self.num_agents)
        ]
        self.sensors = [
            LocalObstacleSensor(**copy.deepcopy(self.sensor_config))
            for _ in range(self.num_agents)
        ]
        self.state_dim = int(self.dynamics[0].p.shape[0])

        self._initial_static_obstacles = copy.deepcopy(static_obstacles or [])
        self._static_obstacle_generator = static_obstacle_generator
        self._initial_dynamic_obstacles = copy.deepcopy(dynamic_obstacles or [])
        self._dynamic_obstacle_generator = dynamic_obstacle_generator
        self._start_goal_generator = start_goal_generator

        self._default_starts = np.zeros(
            (self.num_agents, self.state_dim),
            dtype=float,
        )
        self._default_goals = np.zeros(
            (self.num_agents, self.state_dim),
            dtype=float,
        )
        for agent_index in range(self.num_agents):
            self._default_starts[agent_index] = np.array(
                [0.0, float(agent_index), 0.0],
                dtype=float,
            )
            self._default_goals[agent_index] = np.array(
                [8.0, float(agent_index), 0.0],
                dtype=float,
            )

        self.static_obstacles = copy.deepcopy(self._initial_static_obstacles)
        self.workspace_boundary_obstacles = (
            self._build_workspace_boundary_obstacles()
        )
        self.dynamic_obstacles = copy.deepcopy(self._initial_dynamic_obstacles)
        self.starts = self._default_starts.copy()
        self.goals = self._default_goals.copy()
        self.steps = 0
        self.render_mode = render_mode

        self.latest_sensor_packets = [None for _ in range(self.num_agents)]
        self.latest_observation = None
        self.latest_collision_info = None
        self.success_rewarded_mask = np.zeros(self.num_agents, dtype=bool)
        self.stagnation_distance_histories = [
            deque(maxlen=int(self.env_config.stagnation_window) + 1)
            for _ in range(self.num_agents)
        ]
        self.stagnation_window_progress = np.zeros(
            self.num_agents,
            dtype=np.float32,
        )
        self.stagnation_counters = np.zeros(
            self.num_agents,
            dtype=np.int32,
        )

        self.action_space = self._build_action_space()
        self.observation_space = self._build_observation_space()

    @property
    def single_agent_action_dim(self) -> int:
        return self.state_dim

    @property
    def extra_observation_dim(self) -> int:
        return 2

    def _build_single_agent_action_bounds(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        low = self._as_state_vector(
            self.dynamics[0].accelerate_min,
            self.state_dim,
        )
        high = self._as_state_vector(
            self.dynamics[0].accelerate_max,
            self.state_dim,
        )
        if np.any(low >= high):
            raise ValueError(
                "dynamics acceleration lower bounds must be smaller than "
                "upper bounds"
            )
        return low, high

    def _build_action_space(self) -> spaces.Box:
        single_low, single_high = self._build_single_agent_action_bounds()
        return spaces.Box(
            low=np.tile(single_low, (self.num_agents, 1)),
            high=np.tile(single_high, (self.num_agents, 1)),
            dtype=np.float32,
        )

    def _build_observation_space(self) -> spaces.Box:
        velocity_low = self._as_state_vector(
            self.dynamics[0].velocity_min,
            self.state_dim,
        )
        velocity_high = self._as_state_vector(
            self.dynamics[0].velocity_max,
            self.state_dim,
        )
        goal_direction_low = np.full(
            self.state_dim,
            -1.0,
            dtype=np.float32,
        )
        goal_direction_high = np.full(
            self.state_dim,
            1.0,
            dtype=np.float32,
        )
        goal_distance_low = np.zeros(1, dtype=np.float32)
        goal_distance_high = np.ones(1, dtype=np.float32)
        scan_frame_count = 2 if self.sensors[0].include_previous_scan else 1
        scan_low = np.zeros(
            scan_frame_count * self.sensors[0].n_rays,
            dtype=np.float32,
        )
        scan_high = np.ones(
            scan_frame_count * self.sensors[0].n_rays,
            dtype=np.float32,
        )
        sensor_low = np.concatenate(
            [
                velocity_low,
                goal_direction_low,
                goal_distance_low,
                scan_low,
            ],
            axis=0,
        )
        sensor_high = np.concatenate(
            [
                velocity_high,
                goal_direction_high,
                goal_distance_high,
                scan_high,
            ],
            axis=0,
        )

        single_pair_low = np.concatenate(
            [
                np.full(self.state_dim, -1.0, dtype=np.float32),
                np.full(self.state_dim, -1.0, dtype=np.float32),
                np.zeros(1, dtype=np.float32),
            ],
            axis=0,
        )
        single_pair_high = np.concatenate(
            [
                np.full(self.state_dim, 1.0, dtype=np.float32),
                np.full(self.state_dim, 1.0, dtype=np.float32),
                np.ones(1, dtype=np.float32),
            ],
            axis=0,
        )
        inter_agent_low = np.tile(
            single_pair_low,
            self.nearest_agent_observation_count,
        )
        inter_agent_high = np.tile(
            single_pair_high,
            self.nearest_agent_observation_count,
        )
        extra_low = np.array([-1.0, 0.0], dtype=np.float32)
        extra_high = np.array([1.0, 1.0], dtype=np.float32)

        single_low = np.concatenate(
            [sensor_low, extra_low, inter_agent_low],
            axis=0,
        )
        single_high = np.concatenate(
            [sensor_high, extra_high, inter_agent_high],
            axis=0,
        )
        return spaces.Box(
            low=np.tile(single_low, (self.num_agents, 1)),
            high=np.tile(single_high, (self.num_agents, 1)),
            dtype=np.float32,
        )

    def _compose_extra_observation(self, agent_index: int) -> np.ndarray:
        progress_scale = max(
            float(self.env_config.stagnation_progress_threshold),
            1e-8,
        )
        normalized_progress = np.clip(
            float(self.stagnation_window_progress[agent_index])
            / progress_scale,
            -1.0,
            1.0,
        )

        growth = float(self.env_config.stagnation_penalty_growth)
        if growth > 0.0:
            ramp_steps = int(
                np.ceil(
                    max(
                        float(self.env_config.stagnation_penalty_max)
                        - float(self.env_config.stagnation_penalty_start),
                        0.0,
                    )
                    / growth
                )
            )
        else:
            ramp_steps = 0
        counter_scale = max(
            int(self.env_config.stagnation_patience) + ramp_steps,
            1,
        )
        normalized_counter = np.clip(
            float(self.stagnation_counters[agent_index])
            / float(counter_scale),
            0.0,
            1.0,
        )
        return np.array(
            [normalized_progress, normalized_counter],
            dtype=np.float32,
        )

    def _compute_policy_action_direction(
        self,
        agent_index: int,
        policy_action: np.ndarray,
    ) -> np.ndarray | None:
        del agent_index
        acceleration = np.asarray(
            policy_action[: self.state_dim],
            dtype=float,
        )
        acceleration_norm = float(np.linalg.norm(acceleration))
        if acceleration_norm < 1e-8:
            return None
        return acceleration / acceleration_norm

    def _build_info(
        self,
        *,
        success_mask,
        new_success_mask,
        successful_episode,
        collision_info,
        truncated,
        distances_to_goals,
        progress,
        commanded_accelerations,
        applied_accelerations,
        acceleration_clip_mask,
        next_states,
        raw_action,
        step_rewards,
        obstacle_potential_penalties,
        boundary_potential_penalties,
        inter_agent_potential_penalties,
        stagnation_penalties,
        acceleration_penalties,
        acceleration_clip_penalties,
        individual_success_bonuses,
        team_success_bonuses,
        team_collision_penalties,
        local_collision_penalties,
        team_timeout_penalties,
    ) -> dict:
        min_clearances = np.array(
            [
                float(packet.min_clearance)
                if packet is not None
                else np.nan
                for packet in self.latest_sensor_packets
            ],
            dtype=np.float32,
        )
        return {
            "success": bool(successful_episode),
            "success_mask": success_mask.astype(bool).copy(),
            "new_success_mask": new_success_mask.astype(bool).copy(),
            "success_rewarded_mask": (
                self.success_rewarded_mask.astype(bool).copy()
            ),
            "per_agent_success_bonus": float(
                self.env_config.individual_success_bonus
            ),
            "collision": bool(collision_info["collision"]),
            "collision_mask": (
                collision_info["collision_mask"].astype(bool).copy()
            ),
            "obstacle_collision_mask": (
                collision_info["obstacle_collision_mask"]
                .astype(bool)
                .copy()
            ),
            "inter_agent_collision_mask": (
                collision_info["inter_agent_collision_mask"]
                .astype(bool)
                .copy()
            ),
            "boundary_collision_mask": (
                collision_info["boundary_collision_mask"]
                .astype(bool)
                .copy()
            ),
            "inter_agent_collision_matrix": (
                collision_info["inter_agent_collision_matrix"]
                .astype(bool)
                .copy()
            ),
            "truncated": bool(truncated),
            "steps": int(self.steps),
            "distance_to_goals": (
                distances_to_goals.astype(np.float32).copy()
            ),
            "progress": progress.astype(np.float32).copy(),
            "pairwise_distances": (
                collision_info["pairwise_distances"]
                .astype(np.float32)
                .copy()
            ),
            "min_inter_agent_distance": float(
                collision_info["min_inter_agent_distance"]
            ),
            "min_boundary_distances": (
                self._compute_min_boundary_distances()
            ),
            "min_clearances": min_clearances,
            "commanded_accelerations": (
                commanded_accelerations.astype(np.float32).copy()
            ),
            "applied_accelerations": (
                applied_accelerations.astype(np.float32).copy()
            ),
            "acceleration_clip_mask": (
                acceleration_clip_mask.astype(bool).copy()
            ),
            "next_states": next_states.astype(np.float32).copy(),
            "raw_action": raw_action.astype(np.float32).copy(),
            "reward_step": step_rewards.astype(np.float32).copy(),
            "reward_progress": step_rewards.astype(np.float32).copy(),
            "reward_obstacle_potential_penalty": (
                obstacle_potential_penalties.astype(np.float32).copy()
            ),
            "reward_boundary_potential_penalty": (
                boundary_potential_penalties.astype(np.float32).copy()
            ),
            "reward_inter_agent_potential_penalty": (
                inter_agent_potential_penalties.astype(np.float32).copy()
            ),
            "reward_stagnation_penalty": (
                stagnation_penalties.astype(np.float32).copy()
            ),
            "reward_acceleration_penalty": (
                acceleration_penalties.astype(np.float32).copy()
            ),
            "reward_acceleration_clip_penalty": (
                acceleration_clip_penalties.astype(np.float32).copy()
            ),
            "reward_individual_success_bonus": (
                individual_success_bonuses.astype(np.float32).copy()
            ),
            "reward_team_success_bonus": (
                team_success_bonuses.astype(np.float32).copy()
            ),
            "reward_team_collision_penalty": (
                team_collision_penalties.astype(np.float32).copy()
            ),
            "reward_local_collision_penalty": (
                local_collision_penalties.astype(np.float32).copy()
            ),
            "reward_team_timeout_penalty": (
                team_timeout_penalties.astype(np.float32).copy()
            ),
            "reward_collision_penalty": (
                team_collision_penalties + local_collision_penalties
            ).astype(np.float32).copy(),
            "reward_timeout_penalty": (
                team_timeout_penalties.astype(np.float32).copy()
            ),
            "stagnation_window_progress": (
                self.stagnation_window_progress.copy()
            ),
            "stagnation_counters": self.stagnation_counters.copy(),
            "stagnation_mask": (stagnation_penalties > 0.0).copy(),
        }

    def reset(self, *, seed=None, options=None):
        try:
            gym.Env.reset(self, seed=seed)
        except TypeError:
            if seed is not None or not hasattr(self, "np_random"):
                self.np_random = np.random.default_rng(seed)
        options = options or {}

        starts, goals = self._resolve_starts_goals(options)
        self.starts = starts.copy()
        self.goals = goals.copy()
        self.steps = 0
        self.success_rewarded_mask = np.zeros(
            self.num_agents,
            dtype=bool,
        )
        self.stagnation_distance_histories = [
            deque(maxlen=int(self.env_config.stagnation_window) + 1)
            for _ in range(self.num_agents)
        ]
        self.stagnation_window_progress = np.zeros(
            self.num_agents,
            dtype=np.float32,
        )
        self.stagnation_counters = np.zeros(
            self.num_agents,
            dtype=np.int32,
        )

        if "static_obstacles" in options:
            self.static_obstacles = copy.deepcopy(
                options["static_obstacles"]
            )
        else:
            self.static_obstacles = self._generate_static_obstacles(
                starts,
                goals,
            )

        if "dynamic_obstacles" in options:
            self.dynamic_obstacles = copy.deepcopy(
                options["dynamic_obstacles"]
            )
        else:
            self.dynamic_obstacles = self._generate_dynamic_obstacles(
                starts,
                goals,
            )

        zero_velocity = np.zeros(self.state_dim, dtype=float)
        for agent_index in range(self.num_agents):
            self.dynamics[agent_index].reset(
                {
                    "position": starts[agent_index].copy(),
                    "velocity": zero_velocity.copy(),
                }
            )
            self.sensors[agent_index].reset()

        for agent_index in range(self.num_agents):
            self.latest_sensor_packets[agent_index] = (
                self.sensors[agent_index].sense(
                    self.dynamics[agent_index].p,
                    self.dynamics[agent_index].v,
                    goals[agent_index],
                    self._sensor_static_obstacles(),
                    self._sensor_dynamic_obstacles(agent_index),
                )
            )

        distances_to_goals = np.array(
            [
                np.linalg.norm(
                    self.goals[index] - self.dynamics[index].p
                )
                for index in range(self.num_agents)
            ],
            dtype=np.float32,
        )
        for agent_index, distance in enumerate(distances_to_goals):
            self.stagnation_distance_histories[agent_index].append(
                float(distance)
            )

        observation = self.get_observation()
        collision_info = self._check_collision()
        self.latest_collision_info = collision_info
        info = {
            "starts": starts.copy(),
            "goals": goals.copy(),
            "num_agents": int(self.num_agents),
            "static_obstacle_count": len(self.static_obstacles),
            "dynamic_obstacle_count": len(self.dynamic_obstacles),
            "distance_to_goals": distances_to_goals,
            "collision": bool(collision_info["collision"]),
            "collision_mask": collision_info["collision_mask"].copy(),
            "obstacle_collision_mask": (
                collision_info["obstacle_collision_mask"].copy()
            ),
            "inter_agent_collision_mask": (
                collision_info["inter_agent_collision_mask"].copy()
            ),
            "boundary_collision_mask": (
                collision_info["boundary_collision_mask"].copy()
            ),
            "pairwise_distances": (
                collision_info["pairwise_distances"].copy()
            ),
            "min_inter_agent_distance": float(
                collision_info["min_inter_agent_distance"]
            ),
            "min_boundary_distances": (
                self._compute_min_boundary_distances()
            ),
            "stagnation_window_progress": (
                self.stagnation_window_progress.copy()
            ),
            "stagnation_counters": self.stagnation_counters.copy(),
        }
        return observation, info

    def step(self, action):
        if any(packet is None for packet in self.latest_sensor_packets):
            raise RuntimeError("reset must be called before step")

        commanded_accelerations = np.asarray(
            action,
            dtype=np.float32,
        )
        if commanded_accelerations.shape != self.action_shape:
            raise ValueError(f"action must have shape {self.action_shape}")
        raw_action = commanded_accelerations.copy()
        bounded_accelerations = np.clip(
            commanded_accelerations,
            self.action_space.low,
            self.action_space.high,
        ).astype(np.float32)
        acceleration_clip_mask = np.not_equal(
            commanded_accelerations,
            bounded_accelerations,
        )

        previous_distances = np.array(
            [
                np.linalg.norm(
                    self.goals[index] - self.dynamics[index].p
                )
                for index in range(self.num_agents)
            ],
            dtype=float,
        )

        applied_accelerations = np.zeros(
            (self.num_agents, self.state_dim),
            dtype=np.float32,
        )
        next_states = np.zeros(
            (self.num_agents, 2 * self.state_dim),
            dtype=np.float32,
        )
        policy_accelerations = bounded_accelerations.copy()

        for agent_index in range(self.num_agents):
            if self.success_rewarded_mask[agent_index]:
                policy_accelerations[agent_index] = 0.0
                next_states[agent_index] = self._freeze_agent(agent_index)
                continue

            applied_acceleration = bounded_accelerations[agent_index]
            applied_accelerations[agent_index] = applied_acceleration
            next_states[agent_index] = self.dynamics[agent_index].step(
                applied_acceleration
            )

        for obstacle in self.dynamic_obstacles:
            obstacle.step(self.dynamics[0].dt)

        self.steps += 1
        for agent_index in range(self.num_agents):
            self.latest_sensor_packets[agent_index] = (
                self.sensors[agent_index].sense(
                    self.dynamics[agent_index].p,
                    self.dynamics[agent_index].v,
                    self.goals[agent_index],
                    self._sensor_static_obstacles(),
                    self._sensor_dynamic_obstacles(agent_index),
                )
            )

        current_distances = np.array(
            [
                np.linalg.norm(
                    self.goals[index] - self.dynamics[index].p
                )
                for index in range(self.num_agents)
            ],
            dtype=float,
        )
        progress = previous_distances - current_distances
        step_rewards = (
            float(self.env_config.step_reward_weight) * progress
        )
        success_mask = (
            current_distances <= float(self.env_config.goal_tolerance)
        )

        collision_info = self._check_collision()
        self.latest_collision_info = collision_info
        obstacle_potential_penalties = (
            self._compute_obstacle_potential_penalties(
                policy_accelerations
            )
        )
        boundary_potential_penalties = (
            self._compute_boundary_potential_penalties(
                policy_accelerations
            )
        )
        inter_agent_potential_penalties = (
            self._compute_inter_agent_potential_penalties(
                collision_info["pairwise_distances"]
            )
        )
        acceleration_penalties = (
            float(self.env_config.acceleration_penalty_weight)
            * np.sum(
                applied_accelerations.astype(float) ** 2,
                axis=1,
            )
        )
        acceleration_clip_penalties = (
            float(self.env_config.acceleration_clip_penalty_weight)
            * np.sum(
                (
                    commanded_accelerations.astype(float)
                    - bounded_accelerations.astype(float)
                )
                ** 2,
                axis=1,
            )
        )
        stagnation_penalties = self._compute_stagnation_penalties(
            current_distances,
            success_mask,
        )

        rewards = (
            step_rewards
            - obstacle_potential_penalties
            - boundary_potential_penalties
            - inter_agent_potential_penalties
            - stagnation_penalties
        ).astype(np.float32)

        new_success_mask = np.logical_and(
            success_mask,
            np.logical_not(self.success_rewarded_mask),
        )
        full_success = bool(np.all(success_mask))
        collision = bool(collision_info["collision"])
        successful_episode = bool(full_success and not collision)
        terminated = bool(successful_episode or collision)
        truncated = bool(
            (not terminated)
            and (self.steps >= int(self.env_config.max_steps))
        )

        individual_success_bonuses = np.zeros(
            self.num_agents,
            dtype=np.float32,
        )
        team_success_bonuses = np.zeros(
            self.num_agents,
            dtype=np.float32,
        )
        team_collision_penalties = np.zeros(
            self.num_agents,
            dtype=np.float32,
        )
        local_collision_penalties = np.zeros(
            self.num_agents,
            dtype=np.float32,
        )
        team_timeout_penalties = np.zeros(
            self.num_agents,
            dtype=np.float32,
        )

        if collision:
            team_collision_penalties[:] = float(
                self.env_config.team_collision_penalty
            )
            local_collision_penalties[
                collision_info["obstacle_collision_mask"]
            ] += float(
                self.env_config.local_obstacle_collision_penalty
            )
            local_collision_penalties[
                collision_info["boundary_collision_mask"]
            ] += float(
                self.env_config.local_boundary_collision_penalty
            )
            local_collision_penalties[
                collision_info["inter_agent_collision_mask"]
            ] += float(
                self.env_config.local_inter_agent_collision_penalty
            )
            rewards -= (
                team_collision_penalties + local_collision_penalties
            )
        else:
            individual_success_bonuses[new_success_mask] = float(
                self.env_config.individual_success_bonus
            )
            rewards += individual_success_bonuses
            self.success_rewarded_mask = np.logical_or(
                self.success_rewarded_mask,
                new_success_mask,
            )
            for agent_index in np.flatnonzero(new_success_mask):
                next_states[agent_index] = self._freeze_agent(
                    int(agent_index)
                )
            if successful_episode:
                team_success_bonuses[:] = float(
                    self.env_config.team_success_bonus
                )
                rewards += team_success_bonuses
            elif truncated:
                team_timeout_penalties[:] = float(
                    self.env_config.team_timeout_penalty
                )
                rewards -= team_timeout_penalties

        observation = self.get_observation()
        info = self._build_info(
            success_mask=success_mask,
            new_success_mask=new_success_mask,
            successful_episode=successful_episode,
            collision_info=collision_info,
            truncated=truncated,
            distances_to_goals=current_distances.astype(np.float32),
            progress=progress.astype(np.float32),
            commanded_accelerations=commanded_accelerations,
            applied_accelerations=applied_accelerations,
            acceleration_clip_mask=acceleration_clip_mask,
            next_states=next_states,
            raw_action=raw_action,
            step_rewards=step_rewards.astype(np.float32),
            obstacle_potential_penalties=obstacle_potential_penalties,
            boundary_potential_penalties=boundary_potential_penalties,
            inter_agent_potential_penalties=(
                inter_agent_potential_penalties
            ),
            stagnation_penalties=stagnation_penalties,
            acceleration_penalties=(
                acceleration_penalties.astype(np.float32)
            ),
            acceleration_clip_penalties=(
                acceleration_clip_penalties.astype(np.float32)
            ),
            individual_success_bonuses=individual_success_bonuses,
            team_success_bonuses=team_success_bonuses,
            team_collision_penalties=team_collision_penalties,
            local_collision_penalties=local_collision_penalties,
            team_timeout_penalties=team_timeout_penalties,
        )
        return (
            observation,
            rewards.astype(np.float32),
            terminated,
            truncated,
            info,
        )


__all__ = [
    "MultiAgentContinuousEnv",
    "MultiAgentContinuousEnvConfig",
]
