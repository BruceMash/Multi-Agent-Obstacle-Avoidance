import unittest
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from Entity.dynamic_obstacles import PatternedMovingSphereObstacle

ALGO_ROOT = Path(__file__).resolve().parents[1] / "Multi-agent_Algo_lib"
if str(ALGO_ROOT) not in sys.path:
    sys.path.insert(0, str(ALGO_ROOT))

from MASAC.curriculum import (
    CurriculumScenarioGenerator,
    SuccessRateCurriculum,
    build_curriculum_stages,
)


def curriculum_test_config():
    return SimpleNamespace(
        workspace_bounds=((0.0, 0.0, 0.0), (9.0, 4.5, 2.4)),
        curriculum_box_half_extent_range=(0.20, 0.38),
        curriculum_box_height_range=(0.45, 1.10),
        curriculum_aerial_sphere_radius_range=(0.18, 0.30),
        curriculum_dynamic_sphere_radius_range=(0.20, 0.30),
        curriculum_dynamic_speed_range=(0.25, 0.60),
        curriculum_aerial_min_center_height=0.65,
        curriculum_obstacle_safety_margin=0.05,
        curriculum_start_goal_clearance=0.55,
        curriculum_obstacle_separation=0.12,
        curriculum_placement_attempts=1000,
        curriculum_curved_turn_rate=0.45,
        curriculum_wandering_strength=0.8,
    )


class TestSuccessRateCurriculum(unittest.TestCase):
    def setUp(self):
        self.stages = build_curriculum_stages((1, 2, 3), (1, 2, 3), (1, 2, 3))

    def test_stage_layout(self):
        self.assertEqual(len(self.stages), 7)
        self.assertEqual(self.stages[0].to_dict()["name"], "phase1_level0")
        self.assertEqual(self.stages[-1].dynamic_sphere_count, 3)

    def test_requires_full_window_before_advancement(self):
        curriculum = SuccessRateCurriculum(self.stages, 0.8, 5)
        for _ in range(4):
            result = curriculum.record_episode(True)
            self.assertFalse(result["advanced"])
        self.assertEqual(curriculum.stage_index, 0)

    def test_advances_one_level_and_resets_window(self):
        curriculum = SuccessRateCurriculum(self.stages, 0.8, 5)
        results = [curriculum.record_episode(value) for value in (True, True, True, True, False)]
        self.assertTrue(results[-1]["advanced"])
        self.assertEqual(curriculum.stage_index, 1)
        self.assertEqual(curriculum.window_count, 0)


class TestCurriculumScenes(unittest.TestCase):
    def setUp(self):
        self.config = curriculum_test_config()
        self.starts = np.array([[0.8, 0.7, 0.7], [0.8, 2.25, 1.2], [0.8, 3.8, 1.7]])
        self.goals = np.array([[8.2, 0.7, 0.7], [8.2, 2.25, 1.2], [8.2, 3.8, 1.7]])
        self.stages = build_curriculum_stages((1, 2, 3), (1, 2, 3), (1, 2, 3))

    def test_obstacle_counts_and_ground_contact(self):
        for stage in self.stages:
            generator = CurriculumScenarioGenerator(self.config, stage)
            static = generator.generate_static(
                starts=self.starts, goals=self.goals, seed=100 + stage.phase * 10 + stage.level
            )
            dynamic = generator.generate_dynamic(
                starts=self.starts,
                goals=self.goals,
                seed=200 + stage.phase * 10 + stage.level,
                static_obstacles=static,
            )
            self.assertEqual(len(static), stage.ground_box_count + stage.aerial_sphere_count)
            self.assertEqual(len(dynamic), stage.dynamic_sphere_count)
            for box in static[: stage.ground_box_count]:
                self.assertAlmostEqual(box.center[2] - box.half_extents[2], 0.0)

    def test_dynamic_modes_stay_inside_bounds(self):
        generator = CurriculumScenarioGenerator(self.config, self.stages[-1])
        static = generator.generate_static(starts=self.starts, goals=self.goals, seed=301)
        dynamic = generator.generate_dynamic(
            starts=self.starts,
            goals=self.goals,
            seed=302,
            static_obstacles=static,
        )
        self.assertEqual(
            [obstacle.motion_mode for obstacle in dynamic],
            ["linear", "curved", "wandering"],
        )
        lower, upper = np.asarray(self.config.workspace_bounds, dtype=float)
        for _ in range(500):
            for obstacle in dynamic:
                obstacle.step(0.1)
                self.assertTrue(np.all(obstacle.center >= lower + obstacle.effective_radius))
                self.assertTrue(np.all(obstacle.center <= upper - obstacle.effective_radius))

    def test_patterned_obstacle_rejects_unknown_mode(self):
        with self.assertRaises(ValueError):
            PatternedMovingSphereObstacle(
                center=np.ones(3),
                radius=0.2,
                velocity=np.ones(3),
                motion_mode="unknown",
            )


if __name__ == "__main__":
    unittest.main()
