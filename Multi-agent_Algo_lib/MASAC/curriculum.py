from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from Entity.dynamic_obstacles import PatternedMovingSphereObstacle
from Entity.static_obstacles import AxisAlignedBoxObstacle, StaticSphereObstacle


@dataclass(frozen=True)
class CurriculumStage:
    phase: int
    level: int
    ground_box_count: int
    aerial_sphere_count: int
    dynamic_sphere_count: int

    @property
    def name(self) -> str:
        return f"phase{self.phase}_level{self.level}"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["name"] = self.name
        return payload


def build_curriculum_stages(
    phase2_box_counts: Sequence[int],
    phase2_sphere_counts: Sequence[int],
    phase3_dynamic_counts: Sequence[int],
) -> tuple[CurriculumStage, ...]:
    box_counts = tuple(int(value) for value in phase2_box_counts)
    sphere_counts = tuple(int(value) for value in phase2_sphere_counts)
    dynamic_counts = tuple(int(value) for value in phase3_dynamic_counts)
    if not box_counts or len(box_counts) != len(sphere_counts):
        raise ValueError("phase2 box and sphere density levels must have equal non-zero length")
    if not dynamic_counts:
        raise ValueError("phase3 dynamic density levels must be non-empty")
    if any(value < 0 for value in box_counts + sphere_counts + dynamic_counts):
        raise ValueError("curriculum obstacle counts must be non-negative")
    if any(right < left for left, right in zip(box_counts, box_counts[1:])):
        raise ValueError("phase2 box counts must be non-decreasing")
    if any(right < left for left, right in zip(sphere_counts, sphere_counts[1:])):
        raise ValueError("phase2 sphere counts must be non-decreasing")
    if any(right < left for left, right in zip(dynamic_counts, dynamic_counts[1:])):
        raise ValueError("phase3 dynamic counts must be non-decreasing")

    stages = [CurriculumStage(1, 0, 0, 0, 0)]
    stages.extend(
        CurriculumStage(2, index + 1, box_count, sphere_count, 0)
        for index, (box_count, sphere_count) in enumerate(zip(box_counts, sphere_counts))
    )
    stages.extend(
        CurriculumStage(
            3,
            index + 1,
            box_counts[-1],
            sphere_counts[-1],
            dynamic_count,
        )
        for index, dynamic_count in enumerate(dynamic_counts)
    )
    return tuple(stages)


class SuccessRateCurriculum:
    """按当前难度整队成功率进行单向晋级的课程控制器。"""

    def __init__(
        self,
        stages: Sequence[CurriculumStage],
        success_threshold: float = 0.8,
        success_window: int = 100,
        enabled: bool = True,
    ):
        self.stages = tuple(stages)
        if not self.stages:
            raise ValueError("curriculum stages must be non-empty")
        self.success_threshold = float(success_threshold)
        self.success_window = int(success_window)
        self.enabled = bool(enabled)
        if not 0.0 <= self.success_threshold <= 1.0:
            raise ValueError("success_threshold must be in [0, 1]")
        if self.success_window <= 0:
            raise ValueError("success_window must be positive")
        self.stage_index = 0
        self.success_history: deque[float] = deque(maxlen=self.success_window)

    @property
    def current_stage(self) -> CurriculumStage:
        return self.stages[self.stage_index]

    @property
    def success_rate(self) -> float:
        if not self.success_history:
            return 0.0
        return float(np.mean(self.success_history))

    @property
    def window_count(self) -> int:
        return len(self.success_history)

    @property
    def is_final_stage(self) -> bool:
        return self.stage_index == len(self.stages) - 1

    def record_episode(self, success: bool) -> dict[str, Any]:
        completed_stage = self.current_stage
        self.success_history.append(float(bool(success)))
        completed_rate = self.success_rate
        completed_count = self.window_count
        advanced = bool(
            self.enabled
            and completed_count >= self.success_window
            and completed_rate >= self.success_threshold
            and not self.is_final_stage
        )
        if advanced:
            self.stage_index += 1
            self.success_history.clear()
        return {
            "completed_stage": completed_stage,
            "completed_success_rate": completed_rate,
            "completed_window_count": completed_count,
            "advanced": advanced,
            "next_stage": self.current_stage,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "success_threshold": self.success_threshold,
            "success_window": self.success_window,
            "stage_index": self.stage_index,
            "current_stage": self.current_stage.to_dict(),
            "stages": [stage.to_dict() for stage in self.stages],
        }


class CurriculumScenarioGenerator:
    """根据课程阶段生成与起终点安全分离的静态和动态障碍物。"""

    MOTION_MODES = ("linear", "curved", "wandering")

    def __init__(self, experiment_config, stage: CurriculumStage):
        self.config = experiment_config
        self.stage = stage
        bounds = np.asarray(experiment_config.workspace_bounds, dtype=float)
        if bounds.shape != (2, 3) or np.any(bounds[0] >= bounds[1]):
            raise ValueError("workspace_bounds must have shape (2, 3)")
        self.lower = bounds[0]
        self.upper = bounds[1]

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.lower.copy(), self.upper.copy()

    def _is_clear(self, center, radius, protected_points, existing) -> bool:
        center = np.asarray(center, dtype=float)
        protected = np.asarray(protected_points, dtype=float).reshape(-1, 3)
        clearance = float(self.config.curriculum_start_goal_clearance)
        if protected.size and np.any(
            np.linalg.norm(protected - center[None, :], axis=1) < radius + clearance
        ):
            return False
        separation = float(self.config.curriculum_obstacle_separation)
        for obstacle in existing:
            other_center = np.asarray(obstacle.center, dtype=float)
            other_radius = float(getattr(obstacle, "effective_radius", 0.0))
            if np.linalg.norm(other_center - center) < radius + other_radius + separation:
                return False
        return True

    def _sample_ground_box(self, rng, protected_points, existing):
        xy_min, xy_max = self.config.curriculum_box_half_extent_range
        height_min, height_max = self.config.curriculum_box_height_range
        margin = float(self.config.curriculum_obstacle_safety_margin)
        attempts = int(self.config.curriculum_placement_attempts)
        for _ in range(attempts):
            half_extents = np.array(
                [rng.uniform(xy_min, xy_max), rng.uniform(xy_min, xy_max), 0.5 * rng.uniform(height_min, height_max)],
                dtype=float,
            )
            center_lower = self.lower + half_extents + margin
            center_upper = self.upper - half_extents - margin
            if np.any(center_lower >= center_upper):
                break
            center = rng.uniform(center_lower, center_upper)
            center[2] = self.lower[2] + half_extents[2]
            bounding_radius = float(np.linalg.norm(half_extents + margin))
            if self._is_clear(center, bounding_radius, protected_points, existing):
                return AxisAlignedBoxObstacle(center, half_extents, margin)
        raise RuntimeError("failed to place a curriculum ground box obstacle")

    def _sample_sphere(self, rng, protected_points, existing, dynamic=False, index=0):
        radius_range = (
            self.config.curriculum_dynamic_sphere_radius_range
            if dynamic
            else self.config.curriculum_aerial_sphere_radius_range
        )
        margin = float(self.config.curriculum_obstacle_safety_margin)
        attempts = int(self.config.curriculum_placement_attempts)
        for _ in range(attempts):
            radius = float(rng.uniform(*radius_range))
            effective_radius = radius + margin
            center_lower = self.lower + effective_radius
            center_upper = self.upper - effective_radius
            center_lower[2] = max(
                center_lower[2],
                float(self.config.curriculum_aerial_min_center_height),
            )
            if np.any(center_lower >= center_upper):
                break
            center = rng.uniform(center_lower, center_upper)
            if not self._is_clear(center, effective_radius, protected_points, existing):
                continue
            if not dynamic:
                return StaticSphereObstacle(center, radius, margin)

            speed = float(rng.uniform(*self.config.curriculum_dynamic_speed_range))
            direction = rng.normal(size=3)
            direction[2] *= 0.45
            norm = float(np.linalg.norm(direction))
            if norm < 1e-8:
                continue
            velocity = speed * direction / norm
            motion_mode = self.MOTION_MODES[index % len(self.MOTION_MODES)]
            obstacle_seed = int(rng.integers(0, np.iinfo(np.uint32).max))
            return PatternedMovingSphereObstacle(
                center=center,
                radius=radius,
                velocity=velocity,
                safety_margin=margin,
                bounds=self.bounds,
                motion_mode=motion_mode,
                turn_rate=float(self.config.curriculum_curved_turn_rate),
                wandering_strength=float(self.config.curriculum_wandering_strength),
                seed=obstacle_seed,
            )
        kind = "dynamic sphere" if dynamic else "aerial sphere"
        raise RuntimeError(f"failed to place a curriculum {kind} obstacle")

    def generate_static(self, *, starts, goals, seed, **_) -> list[Any]:
        rng = np.random.default_rng(int(seed))
        protected_points = np.concatenate([starts, goals], axis=0)
        obstacles: list[Any] = []
        for _ in range(self.stage.ground_box_count):
            obstacles.append(self._sample_ground_box(rng, protected_points, obstacles))
        for _ in range(self.stage.aerial_sphere_count):
            obstacles.append(
                self._sample_sphere(rng, protected_points, obstacles, dynamic=False)
            )
        return obstacles

    def generate_dynamic(self, *, starts, goals, seed, static_obstacles=None, **_) -> list[Any]:
        rng = np.random.default_rng(int(seed))
        protected_points = np.concatenate([starts, goals], axis=0)
        existing = list(static_obstacles or [])
        dynamic_obstacles = []
        for index in range(self.stage.dynamic_sphere_count):
            obstacle = self._sample_sphere(
                rng,
                protected_points,
                existing,
                dynamic=True,
                index=index,
            )
            dynamic_obstacles.append(obstacle)
            existing.append(obstacle)
        return dynamic_obstacles


def build_stage_env_kwargs(experiment_config, stage: CurriculumStage) -> dict[str, Any]:
    generator = CurriculumScenarioGenerator(experiment_config, stage)
    kwargs = experiment_config.build_core_env_kwargs()
    kwargs["static_obstacle_generator"] = generator.generate_static
    kwargs["dynamic_obstacle_generator"] = generator.generate_dynamic
    return kwargs
