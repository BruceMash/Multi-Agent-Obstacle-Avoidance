from __future__ import annotations

import unittest

import numpy as np

from planning.semi_structured_long_range_benchmark import (
    DT,
    LONG_RANGE_MAX_STEPS,
    MISSION_DISTANCE_RANGE_M,
    STAGE_ORDER,
    STAGE_POPULATION,
    STATIC_DYNAMIC_RATIO,
    WORKSPACE_BOUNDS,
    generate_scenario_entry,
    generate_scenario_manifest,
    validate_scenario_manifest,
)


class SemiStructuredLongRangeBenchmarkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = generate_scenario_manifest(counts_per_stage=5, seed_base=910_000, prefix="TEST")

    def test_manifest_contract_passes(self) -> None:
        validation = validate_scenario_manifest(self.manifest)
        self.assertEqual(validation["status"], "PASS", validation["errors"])
        self.assertEqual(validation["exact_geometry_duplicates"], 0)
        self.assertEqual(validation["translation_equivalent_duplicates"], 0)
        self.assertEqual(validation["local_decision_recoverability"], "YES")

    def test_stage_changes_only_obstacle_population(self) -> None:
        for stage in STAGE_ORDER:
            rows = [row for row in self.manifest["entries"] if row["stage"] == stage]
            target = STAGE_POPULATION[stage]
            self.assertTrue(rows)
            for row in rows:
                self.assertEqual(len(row["static_obstacles"]), target["static"])
                self.assertEqual(len(row["dynamic_obstacles"]), target["dynamic"])
                self.assertEqual(target["static"] / target["dynamic"], STATIC_DYNAMIC_RATIO)
                self.assertEqual(tuple(map(tuple, row["workspace_bounds"])), WORKSPACE_BOUNDS)
                self.assertEqual(row["dt"], DT)
                self.assertEqual(row["max_steps"], LONG_RANGE_MAX_STEPS)

    def test_missions_and_routes_are_valid(self) -> None:
        for row in self.manifest["entries"]:
            distances = np.asarray(row["mission"]["straight_line_distances_m"], dtype=float)
            self.assertTrue(np.all(distances >= MISSION_DISTANCE_RANGE_M[0]))
            self.assertTrue(np.all(distances <= MISSION_DISTANCE_RANGE_M[1]))
            self.assertTrue(row["static_route_witness_evaluation_only"]["all_agents_feasible"])
            self.assertGreaterEqual(row["difficulty"]["mission_relevant_static_ratio"], 0.75)
            self.assertGreaterEqual(row["difficulty"]["mission_relevant_dynamic_ratio"], 0.75)

    def test_dynamic_motion_is_constant_translation_and_bounded(self) -> None:
        lower = np.asarray(WORKSPACE_BOUNDS[0], dtype=float)
        upper = np.asarray(WORKSPACE_BOUNDS[1], dtype=float)
        for row in self.manifest["entries"]:
            for spec, raw_track in zip(
                row["dynamic_obstacles"], row["dynamic_obstacle_trajectories"], strict=True
            ):
                track = np.asarray(raw_track, dtype=float)
                velocity = np.asarray(spec["velocity"], dtype=float)
                self.assertEqual(spec["motion_model"], "constant_direction_translation")
                self.assertFalse(spec["future_available_to_planner"])
                self.assertEqual(track.shape, (LONG_RANGE_MAX_STEPS + 1, 3))
                expected_steps = np.repeat((DT * velocity)[None, :], LONG_RANGE_MAX_STEPS, axis=0)
                np.testing.assert_allclose(np.diff(track, axis=0), expected_steps, atol=1.0e-12)
                self.assertTrue(np.all(track >= lower - 1.0e-9))
                self.assertTrue(np.all(track <= upper + 1.0e-9))

    def test_generation_is_deterministic(self) -> None:
        left = generate_scenario_entry(stage="stage_4", scenario_index=17, seed=920_017, prefix="DET")
        right = generate_scenario_entry(stage="stage_4", scenario_index=17, seed=920_017, prefix="DET")
        self.assertEqual(left["geometry_fingerprint"], right["geometry_fingerprint"])
        self.assertEqual(left["dynamic_track_fingerprint"], right["dynamic_track_fingerprint"])
        self.assertEqual(left["environment_fingerprint"], right["environment_fingerprint"])


if __name__ == "__main__":
    unittest.main()
