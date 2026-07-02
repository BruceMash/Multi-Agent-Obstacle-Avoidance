"""Centralized experiment parameters for SAC training."""

from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


from dataclasses import dataclass
from typing import Any, Callable, TYPE_CHECKING

import numpy as np

from Controller.dmp_rl import DMPConfig
from Environment.multi_agent_dmp_env import MultiAgentEnvConfig
from Environment.single_agent_dmp_env import EnvConfig

if TYPE_CHECKING:
    from Entity.static_obstacles import AxisAlignedBoxObstacle


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
    default_goal: tuple[float, float, float] = (8.0, 8.0, 8.0)
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
    max_steps: int = 50    # 单个Episode的最大步数
    goal_tolerance: float = 0.3     # 
    
    obstacle_potential_weight: float = 1.5  # 障碍物势场权重
    obstacle_influence_distance: float = 1.5    # 障碍物势场的计算范围
    obstacle_potential_penalty_max: float = 20.0    # 能够给予的最大势场惩罚

    boundary_influence_distance: float = 0.6
    boundary_potential_weight: float = 0.3
    boundary_potential_penalty_max: float = 20.0
    boundary_distance_epsilon: float = 1e-3

    step_reward_weight: float = 8.0 # 步进奖励权重
    step_penalty: float = 0.01  # 单步惩罚
    collision_penalty: float = 20.0
    timeout_penalty: float = 20.0
    success_bonus: float = 300.0
    collision_margin: float = 0.0
    action_guidance_enabled: bool = True
    action_guidance_radius: float = 1.0
    action_guidance_initial_weight: float = 0.7
    action_guidance_decay_steps: int = 500_000
    workspace_bounds: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (-0.5, -2.5, -1.2),
        (8.5, 2.0, 1.2),
    )


    # DMP
    dmp_dims: int = 3
    k_alpha: float = 3.0
    k_beta: float = 0.8
    alpha_s: float = 4.0
    tau: float = 2.5
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
    total_timesteps: int = 6_000_000
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
    static_cylinder_center: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (1.2, -1.4, 0.0),
        (7.0, 1.4, 0.0),
    )
    static_cylinder_radius: float = 0.38
    static_cylinder_half_height: float = 0.6
    static_cylinder_safety_margin: float = 0.1
    static_cylinder_num: int = 0

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

    training_scene_mixture_enabled: bool = True
    training_allow_obstacle_overlap: bool = True
    training_dense_scene_probability: float = 0.3
    training_dense_dynamic_probability: float = 0.5
    curriculum_enabled: bool = True
    curriculum_window_episodes: int = 100
    curriculum_check_interval_episodes: int = 20
    curriculum_min_level_episodes: int = 100
    curriculum_rollback_success_threshold: float = 0.45
    curriculum_dense_scene_probabilities: tuple[float, ...] = (0.0, 0.10, 0.20, 0.30, 0.40)
    curriculum_dense_dynamic_probabilities: tuple[float, ...] = (0.0, 0.00, 0.20, 0.50, 0.50)
    curriculum_advance_success_thresholds: tuple[float, ...] = (0.85, 0.80, 0.75, 0.75)
    training_dense_static_obstacle_center: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (1.0, -1.15, -0.45),
        (8.0, 1.15, 0.45),
    )
    training_dense_static_obstacle_radius: float = 0.38
    training_dense_static_obstacle_safety_margin: float = 0.08
    training_dense_static_obstacle_num: int = 4
    training_dense_static_cylinder_center: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (1.0, -1.15, 0.0),
        (8.0, 1.15, 0.0),
    )
    training_dense_static_cylinder_radius: float = 0.36
    training_dense_static_cylinder_half_height: float = 0.62
    training_dense_static_cylinder_safety_margin: float = 0.08
    training_dense_static_cylinder_num: int = 2
    training_dense_dynamic_obstacle_center: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (1.2, -1.35, -0.90),
        (8.0, 1.35, 0.70),
    )
    training_dense_dynamic_obstacle_radius: float = 0.22
    training_dense_dynamic_obstacle_velocity: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (-0.3, -1.0, -0.2),
        (0.3, 1.0, 0.2),
    )
    training_dense_dynamic_obstacle_safety_margin: float = 0.06
    training_dense_dynamic_obstacle_num: int = 2
    training_dense_dynamic_obstacle_bounds: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (0.8, -1.55, -1.00),
        (8.2, 1.55, 0.85),
    )
    training_dense_dynamic_obstacle_min_speed: float = 0.4

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
            obstacle_influence_distance=self.obstacle_influence_distance,
            obstacle_potential_penalty_max=self.obstacle_potential_penalty_max,
            step_reward_weight=self.step_reward_weight,
            step_penalty=self.step_penalty,
            collision_penalty=self.collision_penalty,
            timeout_penalty=self.timeout_penalty,
            success_bonus=self.success_bonus,
            collision_margin=self.collision_margin,
            action_guidance_enabled=self.action_guidance_enabled,
            action_guidance_radius=self.action_guidance_radius,
            action_guidance_initial_weight=self.action_guidance_initial_weight,
            action_guidance_decay_steps=self.action_guidance_decay_steps,
            workspace_bounds=self.workspace_bounds,
            boundary_influence_distance=self.boundary_influence_distance,
            boundary_potential_weight=self.boundary_potential_weight,
            boundary_potential_penalty_max=self.boundary_potential_penalty_max,
            boundary_distance_epsilon=self.boundary_distance_epsilon,
        )

    def build_curriculum_state(self) -> dict[str, float | int | bool]:
        if bool(self.curriculum_enabled):
            dense_probabilities = tuple(float(value) for value in self.curriculum_dense_scene_probabilities)
            dynamic_probabilities = tuple(float(value) for value in self.curriculum_dense_dynamic_probabilities)
            if len(dense_probabilities) != len(dynamic_probabilities):
                raise ValueError("curriculum dense and dynamic probability schedules must have equal length")
            if len(dense_probabilities) < 1:
                raise ValueError("curriculum schedule must contain at least one level")
            return {
                "enabled": True,
                "level": 0,
                "dense_scene_probability": dense_probabilities[0],
                "dense_dynamic_probability": dynamic_probabilities[0],
                "episode_count": 0,
                "level_episode_count": 0,
                "success_rate": 0.0,
            }
        return {
            "enabled": False,
            "level": 0,
            "dense_scene_probability": float(self.training_dense_scene_probability),
            "dense_dynamic_probability": float(self.training_dense_dynamic_probability),
            "episode_count": 0,
            "level_episode_count": 0,
            "success_rate": 0.0,
        }

    def build_fixed_box(self) -> AxisAlignedBoxObstacle:
        from Entity.static_obstacles import AxisAlignedBoxObstacle

        return AxisAlignedBoxObstacle(
            center=list(self.fixed_box_center),
            half_extents=list(self.fixed_box_half_extents),
            safety_margin=self.fixed_box_safety_margin,
        )

    def _sample_probability_event(self, seed: int | None, probability: float) -> bool:
        probability = float(probability)
        if probability <= 0.0:
            return False
        if probability >= 1.0:
            return True
        rng = np.random.default_rng(None if seed is None else int(seed))
        return bool(rng.random() < probability)

    def _curriculum_probability(
        self,
        curriculum_state: dict[str, Any] | None,
        key: str,
        default_value: float,
    ) -> float:
        if curriculum_state is None:
            return float(default_value)
        return float(curriculum_state.get(key, default_value))

    def _cylinder_center_bounds_on_workspace_floor(
        self,
        center_bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
        half_height: float,
    ) -> np.ndarray:
        cylinder_center = np.asarray(center_bounds, dtype=float).copy()
        workspace_bottom_z = float(np.asarray(self.workspace_bounds, dtype=float)[0, 2])
        cylinder_center_z = workspace_bottom_z + float(half_height)
        cylinder_center[0, 2] = cylinder_center_z
        cylinder_center[1, 2] = cylinder_center_z + 1e-6
        return cylinder_center

    def _sample_center_avoiding_protected_points(
        self,
        rng: np.random.Generator,
        lower: np.ndarray,
        upper: np.ndarray,
        protected_points: list[np.ndarray],
        protected_clearance: float,
        max_attempts: int = 1000,
    ) -> np.ndarray:
        for _ in range(max(1, int(max_attempts))):
            center = rng.uniform(lower, upper)
            if not any(float(np.linalg.norm(center - point)) < protected_clearance for point in protected_points):
                return center
        raise RuntimeError("failed to sample obstacle center away from protected start/goal points")

    def _sample_velocity_with_min_speed(
        self,
        rng: np.random.Generator,
        lower: np.ndarray,
        upper: np.ndarray,
        min_speed: float,
        max_attempts: int = 1000,
    ) -> np.ndarray:
        min_speed = float(min_speed)
        for _ in range(max(1, int(max_attempts))):
            velocity = rng.uniform(lower, upper)
            if float(np.linalg.norm(velocity)) >= min_speed:
                return velocity
        raise RuntimeError("failed to sample dynamic obstacle velocity above min_speed")

    def _generate_static_obstacles_allow_overlap(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        seed: int | None,
        static_center: tuple[tuple[float, float, float], tuple[float, float, float]],
        static_radius: float,
        static_safety_margin: float,
        static_num: int,
        cylinder_center: tuple[tuple[float, float, float], tuple[float, float, float]],
        cylinder_radius: float,
        cylinder_half_height: float,
        cylinder_safety_margin: float,
        cylinder_num: int,
        scene_name: str,
    ) -> list[Any]:
        from Entity.static_obstacles import StaticCylinderObstacle, StaticSphereObstacle

        rng = np.random.default_rng(None if seed is None else int(seed))
        protected_points = [np.asarray(start, dtype=float).copy(), np.asarray(goal, dtype=float).copy()]
        obstacles: list[Any] = []

        cylinder_bounds = self._cylinder_center_bounds_on_workspace_floor(cylinder_center, cylinder_half_height)
        cylinder_lower = cylinder_bounds[0]
        cylinder_upper = cylinder_bounds[1]
        cylinder_effective_radius = float(
            np.sqrt(
                (float(cylinder_radius) + float(cylinder_safety_margin)) ** 2
                + (float(cylinder_half_height) + float(cylinder_safety_margin)) ** 2
            )
        )
        for _ in range(max(0, int(cylinder_num))):
            center = self._sample_center_avoiding_protected_points(
                rng=rng,
                lower=cylinder_lower,
                upper=cylinder_upper,
                protected_points=protected_points,
                protected_clearance=cylinder_effective_radius + 0.8,
            )
            center[2] = float(cylinder_lower[2])
            obstacles.append(
                StaticCylinderObstacle(
                    center=center,
                    radius=cylinder_radius,
                    half_height=cylinder_half_height,
                    safety_margin=cylinder_safety_margin,
                )
            )

        sphere_bounds = np.asarray(static_center, dtype=float)
        sphere_lower = sphere_bounds[0]
        sphere_upper = sphere_bounds[1]
        sphere_effective_radius = float(static_radius) + float(static_safety_margin)
        for _ in range(max(0, int(static_num))):
            center = self._sample_center_avoiding_protected_points(
                rng=rng,
                lower=sphere_lower,
                upper=sphere_upper,
                protected_points=protected_points,
                protected_clearance=sphere_effective_radius + 0.8,
            )
            obstacles.append(
                StaticSphereObstacle(
                    center=center,
                    radius=static_radius,
                    safety_margin=static_safety_margin,
                )
            )

        for obstacle in obstacles:
            obstacle.training_scene_name = scene_name
        return obstacles

    def _generate_static_obstacles(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        seed: int | None,
        static_center: tuple[tuple[float, float, float], tuple[float, float, float]],
        static_radius: float,
        static_safety_margin: float,
        static_num: int,
        cylinder_center: tuple[tuple[float, float, float], tuple[float, float, float]],
        cylinder_radius: float,
        cylinder_half_height: float,
        cylinder_safety_margin: float,
        cylinder_num: int,
        scene_name: str,
        curriculum_state: dict[str, Any] | None = None,
    ) -> list[Any]:
        from Entity.obstacle_generators import StaticCylinderPositionGenerate, StaticSpherePositionGenerate

        if (
            bool(self.training_scene_mixture_enabled)
            and bool(self.training_allow_obstacle_overlap)
            and curriculum_state is not None
        ):
            return self._generate_static_obstacles_allow_overlap(
                start=start,
                goal=goal,
                seed=seed,
                static_center=static_center,
                static_radius=static_radius,
                static_safety_margin=static_safety_margin,
                static_num=static_num,
                cylinder_center=cylinder_center,
                cylinder_radius=cylinder_radius,
                cylinder_half_height=cylinder_half_height,
                cylinder_safety_margin=cylinder_safety_margin,
                cylinder_num=cylinder_num,
                scene_name=scene_name,
            )

        obstacles: list[Any] = []
        cylinder_seed = None if seed is None else int(seed) + 10007
        if int(cylinder_num) > 0:
            cylinder_center_bounds = self._cylinder_center_bounds_on_workspace_floor(
                cylinder_center,
                cylinder_half_height,
            )
            cylinders = StaticCylinderPositionGenerate(
                center=cylinder_center_bounds.tolist(),
                radius=cylinder_radius,
                half_height=cylinder_half_height,
                safety_margin=cylinder_safety_margin,
                num=cylinder_num,
                existing_obstacles=[],
                seed=cylinder_seed,
                protected_points=[start, goal],
            )
            cylinder_center_z = float(cylinder_center_bounds[0, 2])
            for cylinder in cylinders:
                cylinder.center[2] = cylinder_center_z
            obstacles.extend(cylinders)

        spheres = StaticSpherePositionGenerate(
            center=[list(point) for point in static_center],
            radius=static_radius,
            safety_margin=static_safety_margin,
            num=static_num,
            existing_obstacles=obstacles,
            seed=seed,
            protected_points=[start, goal],
        )
        obstacles.extend(spheres)
        for obstacle in obstacles:
            obstacle.training_scene_name = scene_name
        return obstacles

    def _static_obstacles_use_dense_training_scene(self, static_obstacles: list[Any]) -> bool:
        return any(getattr(obstacle, "training_scene_name", None) == "dense_training" for obstacle in static_obstacles)

    def _generate_dynamic_obstacles_allow_overlap(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        seed: int | None,
        center: tuple[tuple[float, float, float], tuple[float, float, float]],
        radius: float,
        velocity: tuple[tuple[float, float, float], tuple[float, float, float]],
        safety_margin: float,
        num: int,
        movement_bounds: tuple[tuple[float, float, float], tuple[float, float, float]] | None,
        min_speed: float,
    ) -> list[Any]:
        from Entity.dynamic_obstacles import MovingSphereObstacle

        rng = np.random.default_rng(None if seed is None else int(seed))
        center_bounds = np.asarray(center, dtype=float)
        velocity_bounds = np.asarray(velocity, dtype=float)
        if center_bounds.shape != (2, 3) or velocity_bounds.shape != (2, 3):
            raise ValueError("dynamic obstacle center and velocity bounds must have shape (2, 3)")
        sample_lower = center_bounds[0].copy()
        sample_upper = center_bounds[1].copy()

        obstacle_bounds = None
        effective_radius = float(radius) + float(safety_margin)
        if movement_bounds is not None:
            bounds = np.asarray(movement_bounds, dtype=float)
            if bounds.shape != (2, 3):
                raise ValueError("dynamic obstacle movement_bounds must have shape (2, 3)")
            obstacle_bounds = (bounds[0].copy(), bounds[1].copy())
            sample_lower = np.maximum(sample_lower, obstacle_bounds[0] + effective_radius)
            sample_upper = np.minimum(sample_upper, obstacle_bounds[1] - effective_radius)
            if np.any(sample_lower >= sample_upper):
                raise ValueError("center sampling range must leave room for dynamic obstacle movement_bounds")

        protected_points = [np.asarray(start, dtype=float).copy(), np.asarray(goal, dtype=float).copy()]
        obstacles = []
        for _ in range(max(0, int(num))):
            obstacle_center = self._sample_center_avoiding_protected_points(
                rng=rng,
                lower=sample_lower,
                upper=sample_upper,
                protected_points=protected_points,
                protected_clearance=effective_radius + 0.8,
            )
            obstacle_velocity = self._sample_velocity_with_min_speed(
                rng=rng,
                lower=velocity_bounds[0],
                upper=velocity_bounds[1],
                min_speed=min_speed,
            )
            obstacles.append(
                MovingSphereObstacle(
                    center=obstacle_center,
                    radius=radius,
                    velocity=obstacle_velocity,
                    safety_margin=safety_margin,
                    bounds=obstacle_bounds,
                )
            )
        return obstacles

    def build_static_obstacle_generator(
        self,
        fixed_box: AxisAlignedBoxObstacle,
        curriculum_state: dict[str, Any] | None = None,
    ) -> Callable[[np.ndarray, np.ndarray, int], list[Any]]:
        def static_obstacle_generator(start, goal, seed):
            dense_scene_probability = self._curriculum_probability(
                curriculum_state,
                key="dense_scene_probability",
                default_value=self.training_dense_scene_probability,
            )
            use_dense_scene = bool(self.training_scene_mixture_enabled) and self._sample_probability_event(
                seed=seed,
                probability=dense_scene_probability,
            )
            if use_dense_scene:
                return self._generate_static_obstacles(
                    start=start,
                    goal=goal,
                    seed=seed,
                    static_center=self.training_dense_static_obstacle_center,
                    static_radius=self.training_dense_static_obstacle_radius,
                    static_safety_margin=self.training_dense_static_obstacle_safety_margin,
                    static_num=self.training_dense_static_obstacle_num,
                    cylinder_center=self.training_dense_static_cylinder_center,
                    cylinder_radius=self.training_dense_static_cylinder_radius,
                    cylinder_half_height=self.training_dense_static_cylinder_half_height,
                    cylinder_safety_margin=self.training_dense_static_cylinder_safety_margin,
                    cylinder_num=self.training_dense_static_cylinder_num,
                    scene_name="dense_training",
                    curriculum_state=curriculum_state,
                )
            return self._generate_static_obstacles(
                start=start,
                goal=goal,
                seed=seed,
                static_center=self.static_obstacle_center,
                static_radius=self.static_obstacle_radius,
                static_safety_margin=self.static_obstacle_safety_margin,
                static_num=self.static_obstacle_num,
                cylinder_center=self.static_cylinder_center,
                cylinder_radius=self.static_cylinder_radius,
                cylinder_half_height=self.static_cylinder_half_height,
                cylinder_safety_margin=self.static_cylinder_safety_margin,
                cylinder_num=self.static_cylinder_num,
                scene_name="base_training",
                curriculum_state=curriculum_state,
            )

        return static_obstacle_generator

    def build_dynamic_obstacle_generator(
        self,
        curriculum_state: dict[str, Any] | None = None,
    ) -> Callable[[np.ndarray, np.ndarray, int, list[Any]], list[Any]]:
        from Entity.obstacle_generators import DynamicSpherePositionGenerate

        def dynamic_obstacle_generator(start, goal, seed, static_obstacles):
            dense_dynamic_probability = self._curriculum_probability(
                curriculum_state,
                key="dense_dynamic_probability",
                default_value=self.training_dense_dynamic_probability,
            )
            use_dense_dynamic_scene = (
                bool(self.training_scene_mixture_enabled)
                and self._static_obstacles_use_dense_training_scene(static_obstacles)
                and self._sample_probability_event(
                    seed=seed,
                    probability=dense_dynamic_probability,
                )
            )
            if use_dense_dynamic_scene:
                if (
                    bool(self.training_scene_mixture_enabled)
                    and bool(self.training_allow_obstacle_overlap)
                    and curriculum_state is not None
                ):
                    return self._generate_dynamic_obstacles_allow_overlap(
                        start=start,
                        goal=goal,
                        seed=seed,
                        center=self.training_dense_dynamic_obstacle_center,
                        radius=self.training_dense_dynamic_obstacle_radius,
                        velocity=self.training_dense_dynamic_obstacle_velocity,
                        safety_margin=self.training_dense_dynamic_obstacle_safety_margin,
                        num=self.training_dense_dynamic_obstacle_num,
                        movement_bounds=self.training_dense_dynamic_obstacle_bounds,
                        min_speed=self.training_dense_dynamic_obstacle_min_speed,
                    )
                return DynamicSpherePositionGenerate(
                    center=[list(point) for point in self.training_dense_dynamic_obstacle_center],
                    radius=self.training_dense_dynamic_obstacle_radius,
                    velocity=[list(vec) for vec in self.training_dense_dynamic_obstacle_velocity],
                    safety_margin=self.training_dense_dynamic_obstacle_safety_margin,
                    num=self.training_dense_dynamic_obstacle_num,
                    movement_bounds=[list(point) for point in self.training_dense_dynamic_obstacle_bounds],
                    existing_obstacles=static_obstacles,
                    seed=seed,
                    protected_points=[start, goal],
                    min_speed=self.training_dense_dynamic_obstacle_min_speed,
                )
            if (
                bool(self.training_scene_mixture_enabled)
                and self._static_obstacles_use_dense_training_scene(static_obstacles)
            ):
                return []
            if (
                bool(self.training_scene_mixture_enabled)
                and bool(self.training_allow_obstacle_overlap)
                and curriculum_state is not None
            ):
                return self._generate_dynamic_obstacles_allow_overlap(
                    start=start,
                    goal=goal,
                    seed=seed,
                    center=self.dynamic_obstacle_center,
                    radius=self.dynamic_obstacle_radius,
                    velocity=self.dynamic_obstacle_velocity,
                    safety_margin=self.dynamic_obstacle_safety_margin,
                    num=self.dynamic_obstacle_num,
                    movement_bounds=self.dynamic_obstacle_bounds,
                    min_speed=self.dynamic_obstacle_min_speed,
                )
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
        return []

    def build_policy_kwargs(self) -> dict[str, Any]:
        return {
            "hidden_dim": self.hidden_dim,
            "sensor_output_dim": self.sensor_output_dim,
            "num_sensor_layers": self.num_sensor_layers,
            "num_observation_layers": self.num_observation_layers,
        }


@dataclass(frozen=True)
class MAPPOExperimentConfig:
    """ 面向ray训练的MAPPO实验配置，包含环境、奖励、DMP和训练相关的参数 """
    # MARLlib entry
    environment_name: str = "multi_agent_dmp"
    map_name: str = "default"
    hyperparam_source: str = "test"
    model_core_arch: str = "mlp"
    hidden_dim: int = 256
    sensor_output_dim: int = 128
    num_sensor_layers: int = 2
    num_observation_layers: int = 2
    actor_log_std_min: float = -5.0
    actor_log_std_max: float = 0.0

    # Environment and dynamics
    num_agents: int = 5
    velocity_clip: tuple[float, float] = (-4.0, 4.0)
    accelerate_clip: tuple[float, float] = (-4.0, 4.0)
    time_step: float = 0.1
    sensing_radius: float = 5.0
    sensor_azimuth_bins: int = 24
    sensor_elevation_bins: int = 9
    sensor_elevation_range_deg: tuple[float, float] = (-80.0, 80.0)
    sensor_goal_distance_clip: float | None = None
    max_steps: int = 200
    goal_tolerance: float = 0.3
    workspace_bounds: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (-0.5, -2.5, -1.2),
        (8.5, 2.0, 1.2),
    )
    randomize_start_goal: bool = True
    start_position_bounds: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (0.0, -2.0, -0.8),
        (0.8, 1.5, 0.8),
    )
    goal_position_bounds: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (7.2, -2.0, -0.8),
        (8.0, 1.5, 0.8),
    )
    min_start_distance: float = 0.6
    min_goal_distance: float = 0.0
    min_start_goal_distance: float = 6.0
    start_goal_max_attempts: int = 1000

    # Reward and safety
    obstacle_potential_weight: float = 2.0
    obstacle_influence_distance: float = 1.5
    obstacle_potential_penalty_max: float = 20.0
    boundary_influence_distance: float = 0.6
    boundary_potential_weight: float = 0.3
    boundary_potential_penalty_max: float = 20.0
    boundary_distance_epsilon: float = 1e-3
    step_reward_weight: float = 4.0
    step_penalty: float = 0.01
    collision_penalty: float = 80.0
    timeout_penalty: float = 20
    success_bonus: float = 600.0
    collision_margin: float = 0.0
    action_guidance_enabled: bool = True
    action_guidance_radius: float = 1.0
    action_guidance_initial_weight: float = 0.7
    action_guidance_decay_steps: int = 500_000
    near_goal_bonus_radius_1: float = 1.0
    near_goal_bonus_1: float = 0.2
    near_goal_bonus_radius_2: float = 0.6
    near_goal_bonus_2: float = 0.6
    inter_agent_safe_distance: float = 0.6
    inter_agent_collision_penalty: float = 20.0
    inter_agent_potential_weight: float = 1.0
    inter_agent_influence_distance: float = 1.2
    nearest_agent_observation_count: int | None = None
    acceleration_penalty_weight: float = 0.01
    acceleration_clip_penalty_weight: float = 0.05

    # DMP
    dmp_dims: int = 3
    k_alpha: float = 3.2
    k_beta: float = 0.8
    alpha_s: float = 4.0
    tau: float = 2.5
    forcing_term_max: float = 10.0
    forcing_term_min: float = -10.0
    goal_offset_max: float = 1.0

    # MAPPO hyperparameters
    use_gae: bool = True
    gae_lambda: float = 0.95
    kl_coeff: float = 0.2
    batch_episode: int = 16
    num_sgd_iter: int = 5
    vf_loss_coeff: float = 1.0
    learning_rate: float = 1e-4
    entropy_coeff: float = 0.003
    clip_param: float = 0.3
    vf_clip_param: float = 10.0
    batch_mode: str = "truncate_episodes"
    fixed_batch_timesteps: int | None = 4096

    # Training and Ray
    training_iteration: int = 500000
    stop_timesteps: int = 8_000_000
    stop_reward: float = 999_999.0
    seed: int = 321
    output_root: str = "artifacts/mappo"
    local_mode: bool = False    # ray调度模式，True为单进程调试，False为多进程训练
    share_policy: str = "all"
    evaluation_interval: int | None = None
    framework: str = "torch"
    num_workers: int = 2
    num_gpus: int = 1
    num_cpus_per_worker: int = 2
    num_gpus_per_worker: int = 0
    checkpoint_freq: int = 50
    checkpoint_end: bool = True
    fixed_eval_enabled: bool = True
    fixed_eval_num_scenarios: int = 3
    fixed_eval_deterministic: bool = True
    fixed_eval_output_dirname: str = "fixed_eval"
    max_failures: int = 3
    restore_model_path: str = ""
    restore_params_path: str = ""

    def build_dynamics_config(self) -> dict[str, Any]:
        return {
            "velocity_clip": self.velocity_clip,
            "accelerate_clip": self.accelerate_clip,
            "time_step": self.time_step,
        }

    def build_sensor_config(self) -> dict[str, Any]:
        sensor_config: dict[str, Any] = {
            "sensing_radius": self.sensing_radius,
            "azimuth_bins": self.sensor_azimuth_bins,
            "elevation_bins": self.sensor_elevation_bins,
            "elevation_range_deg": self.sensor_elevation_range_deg,
        }
        if self.sensor_goal_distance_clip is not None:
            sensor_config["goal_distance_clip"] = self.sensor_goal_distance_clip
        return sensor_config

    def build_dmp_config(self) -> dict[str, Any]:
        return {
            "dt": self.time_step,
            "dims": self.dmp_dims,
            "K_alpha": self.k_alpha,
            "K_beta": self.k_beta,
            "alpha_s": self.alpha_s,
            "tau": self.tau,
            "forcing_term_max": self.forcing_term_max,
            "forcing_term_min": self.forcing_term_min,
            "goal_offset_max": self.goal_offset_max,
        }

    def build_env_config(self) -> MultiAgentEnvConfig:
        return MultiAgentEnvConfig(
            num_agents=self.num_agents,
            max_steps=self.max_steps,
            goal_tolerance=self.goal_tolerance,
            obstacle_potential_weight=self.obstacle_potential_weight,
            obstacle_influence_distance=self.obstacle_influence_distance,
            obstacle_potential_penalty_max=self.obstacle_potential_penalty_max,
            step_reward_weight=self.step_reward_weight,
            step_penalty=self.step_penalty,
            collision_penalty=self.collision_penalty,
            timeout_penalty=self.timeout_penalty,
            success_bonus=self.success_bonus,
            collision_margin=self.collision_margin,
            workspace_bounds=self.workspace_bounds,
            randomize_start_goal=self.randomize_start_goal,
            start_position_bounds=self.start_position_bounds,
            goal_position_bounds=self.goal_position_bounds,
            min_start_distance=self.min_start_distance,
            min_goal_distance=self.min_goal_distance,
            min_start_goal_distance=self.min_start_goal_distance,
            start_goal_max_attempts=self.start_goal_max_attempts,
            boundary_influence_distance=self.boundary_influence_distance,
            boundary_potential_weight=self.boundary_potential_weight,
            boundary_potential_penalty_max=self.boundary_potential_penalty_max,
            boundary_distance_epsilon=self.boundary_distance_epsilon,
            action_guidance_enabled=self.action_guidance_enabled,
            action_guidance_radius=self.action_guidance_radius,
            action_guidance_initial_weight=self.action_guidance_initial_weight,
            action_guidance_decay_steps=self.action_guidance_decay_steps,
            near_goal_bonus_radius_1=self.near_goal_bonus_radius_1,
            near_goal_bonus_1=self.near_goal_bonus_1,
            near_goal_bonus_radius_2=self.near_goal_bonus_radius_2,
            near_goal_bonus_2=self.near_goal_bonus_2,
            inter_agent_safe_distance=self.inter_agent_safe_distance,
            inter_agent_collision_penalty=self.inter_agent_collision_penalty,
            inter_agent_potential_weight=self.inter_agent_potential_weight,
            inter_agent_influence_distance=self.inter_agent_influence_distance,
            nearest_agent_observation_count=self.nearest_agent_observation_count,
            acceleration_penalty_weight=self.acceleration_penalty_weight,
            acceleration_clip_penalty_weight=self.acceleration_clip_penalty_weight,
        )

    def build_env_config_dict(self) -> dict[str, Any]:
        return dict(vars(self.build_env_config()))

    def build_core_env_kwargs(self) -> dict[str, Any]:
        return {
            "dynamics_config": self.build_dynamics_config(),
            "sensor_config": self.build_sensor_config(),
            "dmp_config": self.build_dmp_config(),
            "env_config": self.build_env_config_dict(),
        }

    def _build_eval_lane_points(self) -> tuple[np.ndarray, np.ndarray]:
        bounds = np.asarray(self.workspace_bounds, dtype=float)
        if bounds.shape != (2, 3):
            raise ValueError("workspace_bounds must have shape (2, 3)")

        lower, upper = bounds
        x_start = float(np.clip(0.0, lower[0] + 0.4, upper[0] - 0.4))
        x_goal = float(np.clip(8.0, lower[0] + 0.4, upper[0] - 0.4))

        y_margin = min(0.35, max(0.0, 0.2 * float(upper[1] - lower[1])))
        y_lower = float(lower[1] + y_margin)
        y_upper = float(upper[1] - y_margin)
        if y_lower > y_upper:
            y_lower, y_upper = float(lower[1]), float(upper[1])
        y_values = np.linspace(y_lower, y_upper, int(self.num_agents), dtype=float)

        z_margin = min(0.25, max(0.0, 0.2 * float(upper[2] - lower[2])))
        z_abs = max(0.0, min(0.55, 0.5 * float(upper[2] - lower[2]) - z_margin))
        z_values = np.array(
            [(-1.0 if index % 2 == 0 else 1.0) * z_abs for index in range(int(self.num_agents))],
            dtype=float,
        )

        starts = np.stack(
            [np.full(int(self.num_agents), x_start), y_values, z_values],
            axis=1,
        )
        goals = np.stack(
            [np.full(int(self.num_agents), x_goal), y_values, z_values],
            axis=1,
        )
        return starts, goals

    def build_fixed_eval_scenarios(self) -> list[dict[str, Any]]:
        starts, goals = self._build_eval_lane_points()
        scenarios = [
            {
                "name": "parallel_layered",
                "starts": starts.tolist(),
                "goals": goals.tolist(),
            },
            {
                "name": "height_swap",
                "starts": starts.tolist(),
                "goals": np.column_stack([goals[:, 0], goals[:, 1], -goals[:, 2]]).tolist(),
            },
            {
                "name": "lane_swap",
                "starts": starts.tolist(),
                "goals": goals[::-1].tolist(),
            },
        ]
        return scenarios[: max(0, int(self.fixed_eval_num_scenarios))]

    def build_fixed_eval_config(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.fixed_eval_enabled),
            "num_scenarios": int(self.fixed_eval_num_scenarios),
            "deterministic": bool(self.fixed_eval_deterministic),
            "output_dirname": str(self.fixed_eval_output_dirname),
            "checkpoint_freq": int(self.checkpoint_freq),
            "scenarios": self.build_fixed_eval_scenarios(),
        }

    def build_algo_args(self) -> dict[str, Any]:
        return {
            "use_gae": self.use_gae,
            "lambda": self.gae_lambda,
            "kl_coeff": self.kl_coeff,
            "batch_episode": self.batch_episode,
            "num_sgd_iter": self.num_sgd_iter,
            "vf_loss_coeff": self.vf_loss_coeff,
            "lr": self.learning_rate,
            "entropy_coeff": self.entropy_coeff,
            "clip_param": self.clip_param,
            "vf_clip_param": self.vf_clip_param,
            "batch_mode": self.batch_mode,
        }

    def build_model_preference(self) -> dict[str, Any]:
        return {
            "core_arch": self.model_core_arch,
            "hidden_dim": self.hidden_dim,
            "sensor_output_dim": self.sensor_output_dim,
            "num_sensor_layers": self.num_sensor_layers,
            "num_observation_layers": self.num_observation_layers,
            "actor_log_std_min": self.actor_log_std_min,
            "actor_log_std_max": self.actor_log_std_max,
        }

    def build_running_params(self, local_dir: str | None = None) -> dict[str, Any]:
        running_params: dict[str, Any] = {
            "local_mode": self.local_mode,
            "share_policy": self.share_policy,
            "evaluation_interval": self.evaluation_interval,
            "framework": self.framework,
            "num_workers": self.num_workers,
            "num_gpus": self.num_gpus,
            "num_cpus_per_worker": self.num_cpus_per_worker,
            "num_gpus_per_worker": self.num_gpus_per_worker,
            "checkpoint_freq": self.checkpoint_freq,
            "checkpoint_end": self.checkpoint_end,
            "max_failures": self.max_failures,
            "restore_path": {
                "model_path": self.restore_model_path,
                "params_path": self.restore_params_path,
            },
            "stop_iters": self.training_iteration,
            "stop_timesteps": self.stop_timesteps,
            "stop_reward": self.stop_reward,
            "seed": self.seed,
            "local_dir": "" if local_dir is None else local_dir,
            "fixed_eval_config": self.build_fixed_eval_config(),
        }
        if self.fixed_batch_timesteps is not None:
            running_params["fixed_batch_timesteps"] = int(self.fixed_batch_timesteps)
        return running_params

    def build_stop_config(self, training_iteration: int | None = None) -> dict[str, Any]:
        return {
            "training_iteration": self.training_iteration if training_iteration is None else int(training_iteration),
        }


EXPERIMENT_CONFIG = SACExperimentConfig()
MAPPO_EXPERIMENT_CONFIG = MAPPOExperimentConfig()
