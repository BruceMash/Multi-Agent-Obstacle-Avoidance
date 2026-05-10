"""Centralized experiment parameters for SAC training."""

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from Controller.dmp_rl import DMPConfig
from Entity.obstacle_generators import DynamicSpherePositionGenerate, StaticSpherePositionGenerate
from Entity.static_obstacles import AxisAlignedBoxObstacle
from Environment.single_agent_dmp_env import EnvConfig


@dataclass(frozen=True)
class SACExperimentConfig:
    
    # Environment and dynamics
    velocity_clip: tuple[float, float] = (-4.0, 4.0)
    accelerate_clip: tuple[float, float] = (-4.0, 4.0)
    time_step: float = 0.1
    sensing_radius: float = 4.5
    sensor_azimuth_bins: int = 8
    sensor_elevation_bins: int = 7
    randomize_start_goal: bool = True
    default_start: tuple[float, float, float] = (0.0, 0.0, 0.0)
    default_goal: tuple[float, float, float] = (8.0, 0.0, 0.0)
    start_position_bounds: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (0.0, -1.0, -0.4),
        (0.8, 1.0, 0.4),
    )
    goal_position_bounds: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (7.2, -1.0, -0.4),
        (8.0, 1.0, 0.4),
    )
    min_start_goal_distance: float = 6.0
    start_goal_max_attempts: int = 1000

    # Reward and termination
    max_steps: int = 220    # 单个Episode的最大步数
    goal_tolerance: float = 0.3     # 
    obstacle_potential_weight: float = 1.5  # 障碍物势场权重
    step_reward_weight: float = 8.0 # 步进奖励权重
    step_penalty: float = 0.01  # 单步惩罚
    collision_penalty: float = 80.0
    timeout_penalty: float = 50.0
    success_bonus: float = 300.0
    collision_margin: float = 0.0

    # DMP
    dmp_dims: int = 3
    k_alpha: float = 20.0
    k_beta: float = 5.0
    alpha_s: float = 4.0
    tau: float = 1.2
    forcing_term_max: float = 10.0
    forcing_term_min: float = -10.0
    goal_offset_max: float = 1.0

    # Training
    learning_rate: float = 3e-4
    buffer_size: int = 1_000_000
    batch_size: int = 256
    learning_starts: int = 100
    train_freq: int = 1
    gradient_steps: int = 1
    ent_coef: str | float = "auto_0.1"
    hidden_dim: int = 256
    sensor_output_dim: int = 128
    num_sensor_layers: int = 2
    num_observation_layers: int = 2
    total_timesteps: int = 4_000_000
    output_root: str = "artifacts"
    save_freq: int = 50_000
    eval_freq: int = 50_000
    eval_seeds: tuple[int, ...] = (
        202405170,
        202405171,
        202405172,
        202405173,
        202405174,
        202405175,
        202405176,
        202405177,
        202405178,
        202405179,
    )
    eval_deterministic: bool = True
    verbose: int = 1

    # Obstacles
    fixed_box_center: tuple[float, float, float] = (4.8, -0.5, 0.0)
    fixed_box_half_extents: tuple[float, float, float] = (0.45, 0.45, 0.35)
    fixed_box_safety_margin: float = 0.1

    static_obstacle_center: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (1.2, -1.4, -0.3),
        (7.0, 1.4, 0.3),
    )
    static_obstacle_radius: float = 0.45
    static_obstacle_safety_margin: float = 0.1
    static_obstacle_num: int = 1

    dynamic_obstacle_center: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (1.5, -1.6, -0.3),
        (7.0, 1.2, 0.3),
    )
    dynamic_obstacle_radius: float = 0.35
    dynamic_obstacle_velocity: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (-0.3, -1.0, -0.2),
        (0.3, 1.0, 0.2),
    )
    dynamic_obstacle_safety_margin: float = 0.15
    dynamic_obstacle_num: int = 1
    dynamic_obstacle_bounds: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (1.2, -2.0, -1.0),
        (7.3, 1.5, 1.0),
    )
    dynamic_obstacle_min_speed: float = 0.4

    def build_dynamics_config(self) -> dict[str, Any]:
        return {
            "velocity_clip": self.velocity_clip,
            "accelerate_clip": self.accelerate_clip,
            "time_step": self.time_step,
        }

    def build_sensor_config(self) -> dict[str, Any]:
        return {
            "sensing_radius": self.sensing_radius,
            "azimuth_bins": self.sensor_azimuth_bins,
            "elevation_bins": self.sensor_elevation_bins,
        }

    def build_start_goal_generator(self) -> Callable[[np.random.Generator], tuple[np.ndarray, np.ndarray]] | None:
        if not self.randomize_start_goal:
            return None

        start_bounds = np.asarray(self.start_position_bounds, dtype=float)
        goal_bounds = np.asarray(self.goal_position_bounds, dtype=float)
        if start_bounds.shape != (2, 3) or goal_bounds.shape != (2, 3):
            raise ValueError("start_position_bounds and goal_position_bounds must have shape (2, 3)")

        start_lower, start_upper = start_bounds
        goal_lower, goal_upper = goal_bounds
        if np.any(start_lower >= start_upper) or np.any(goal_lower >= goal_upper):
            raise ValueError("each start/goal lower bound must be smaller than upper bound")

        min_distance = float(self.min_start_goal_distance)
        max_attempts = max(1, int(self.start_goal_max_attempts))

        def start_goal_generator(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
            for _ in range(max_attempts):
                start = rng.uniform(start_lower, start_upper)
                goal = rng.uniform(goal_lower, goal_upper)
                if float(np.linalg.norm(goal - start)) >= min_distance:
                    return start.astype(float, copy=True), goal.astype(float, copy=True)
            raise RuntimeError(
                f"failed to sample start/goal pair with minimum distance {min_distance} "
                f"within {max_attempts} attempts"
            )

        return start_goal_generator

    def build_dmp_config(self) -> DMPConfig:
        return DMPConfig(
            dt=self.time_step,
            dims=self.dmp_dims,
            K_alpha=self.k_alpha,
            K_beta=self.k_beta,
            alpha_s=self.alpha_s,
            tau=self.tau,
            forcing_term_max=self.forcing_term_max,
            forcing_term_min=self.forcing_term_min,
            goal_offset_max=self.goal_offset_max,
        )

    def build_env_config(self) -> EnvConfig:
        return EnvConfig(
            max_steps=self.max_steps,
            goal_tolerance=self.goal_tolerance,
            obstacle_potential_weight=self.obstacle_potential_weight,
            step_reward_weight=self.step_reward_weight,
            step_penalty=self.step_penalty,
            collision_penalty=self.collision_penalty,
            timeout_penalty=self.timeout_penalty,
            success_bonus=self.success_bonus,
            collision_margin=self.collision_margin,
        )

    def build_fixed_box(self) -> AxisAlignedBoxObstacle:
        return AxisAlignedBoxObstacle(
            center=list(self.fixed_box_center),
            half_extents=list(self.fixed_box_half_extents),
            safety_margin=self.fixed_box_safety_margin,
        )

    def build_static_obstacle_generator(
        self, fixed_box: AxisAlignedBoxObstacle
    ) -> Callable[[np.ndarray, np.ndarray, int], list[Any]]:
        def static_obstacle_generator(start, goal, seed):
            spheres = StaticSpherePositionGenerate(
                center=[list(point) for point in self.static_obstacle_center],
                radius=self.static_obstacle_radius,
                safety_margin=self.static_obstacle_safety_margin,
                num=self.static_obstacle_num,
                existing_obstacles=[fixed_box],
                seed=seed,
                protected_points=[start, goal],
            )
            spheres.append(fixed_box)
            return spheres

        return static_obstacle_generator

    def build_dynamic_obstacle_generator(self) -> Callable[[np.ndarray, np.ndarray, int, list[Any]], list[Any]]:
        def dynamic_obstacle_generator(start, goal, seed, static_obstacles):
            return DynamicSpherePositionGenerate(
                center=[list(point) for point in self.dynamic_obstacle_center],
                radius=self.dynamic_obstacle_radius,
                velocity=[list(vec) for vec in self.dynamic_obstacle_velocity],
                safety_margin=self.dynamic_obstacle_safety_margin,
                num=self.dynamic_obstacle_num,
                movement_bounds=[list(point) for point in self.dynamic_obstacle_bounds],
                existing_obstacles=static_obstacles,
                seed=seed,
                protected_points=[start, goal],
                min_speed=self.dynamic_obstacle_min_speed,
            )

        return dynamic_obstacle_generator

    def build_static_obstacles(self, fixed_box: AxisAlignedBoxObstacle) -> list[Any]:
        return [fixed_box]

    def build_policy_kwargs(self) -> dict[str, Any]:
        return {
            "hidden_dim": self.hidden_dim,
            "sensor_output_dim": self.sensor_output_dim,
            "num_sensor_layers": self.num_sensor_layers,
            "num_observation_layers": self.num_observation_layers,
        }


EXPERIMENT_CONFIG = SACExperimentConfig()
