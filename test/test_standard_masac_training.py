import csv
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALGO_ROOT = PROJECT_ROOT / "Multi-agent_Algo_lib"
SCRIPTS_ROOT = ALGO_ROOT / "scripts"
for path in (PROJECT_ROOT, ALGO_ROOT, SCRIPTS_ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from Environment.multi_agent_dmp_env import MultiAgentDMPEnv
from MASAC.config import MASAC_EXPERIMENT_CONFIG
from MASAC.curriculum import (
    build_curriculum_stages,
    build_stage_env_kwargs,
)
from train_standard_masac_multi_agent import (
    AgentObservationHistory,
    DMP_CONFIG_FIELDS,
    build_dim_info,
    build_env,
    build_network_config,
    build_replay_action_matrix,
    build_standard_env_kwargs,
    parse_args,
    train,
)


class TestStandardMASACTraining(unittest.TestCase):
    @staticmethod
    def small_config(**overrides):
        values = {
            "sensor_azimuth_bins": 2,
            "sensor_elevation_bins": 2,
            "sensor_include_previous_scan": False,
            "max_steps": 4,
        }
        values.update(overrides)
        return replace(MASAC_EXPERIMENT_CONFIG, **values)

    @staticmethod
    def small_args(*extra_args: str):
        return parse_args(
            [
                "--total-steps",
                "6",
                "--start-steps",
                "2",
                "--batch-size",
                "2",
                "--buffer-size",
                "32",
                "--temporal-steps",
                "2",
                "--hidden-dim",
                "16",
                "--sensor-hidden-dim",
                "8",
                "--ally-hidden-dim",
                "8",
                "--sensor-output-dim",
                "8",
                "--ally-output-dim",
                "8",
                "--num-sensor-layers",
                "1",
                "--num-ally-layers",
                "1",
                "--num-observation-layers",
                "1",
                "--max-steps",
                "3",
                "--sensor-azimuth-bins",
                "2",
                "--sensor-elevation-bins",
                "2",
                "--save-interval",
                "3",
                "--eval-interval",
                "3",
                "--eval-episodes",
                "1",
                "--log-interval",
                "1",
                "--progress-interval",
                "3",
                "--success-window",
                "2",
                "--disable-curriculum",
                "--disable-tensorboard",
                "--plain-progress",
                "--device",
                "cpu",
                *extra_args,
            ]
        )

    def test_standard_environment_kwargs_remove_dmp_configuration(self):
        config = self.small_config()
        kwargs = build_standard_env_kwargs(config)
        env = build_env(config)
        try:
            self.assertNotIn("dmp_config", kwargs)
            self.assertFalse(hasattr(env, "dmps"))
            self.assertFalse(hasattr(env, "dmp_config"))
            self.assertEqual(env.action_shape, (3, 3))
            self.assertEqual(env.observation_shape, (3, 27))
        finally:
            env.close()

    def test_standard_and_dmp_environments_share_curriculum_distribution(self):
        config = self.small_config()
        stage = build_curriculum_stages(
            config.curriculum_phase2_box_counts,
            config.curriculum_phase2_sphere_counts,
            config.curriculum_phase3_dynamic_counts,
        )[-1]
        standard_env = build_env(config, stage)
        dmp_env = MultiAgentDMPEnv(
            **build_stage_env_kwargs(config, stage)
        )
        try:
            _, standard_info = standard_env.reset(seed=20260729)
            _, dmp_info = dmp_env.reset(seed=20260729)

            np.testing.assert_allclose(
                standard_info["starts"],
                dmp_info["starts"],
            )
            np.testing.assert_allclose(
                standard_info["goals"],
                dmp_info["goals"],
            )
            self.assertEqual(
                len(standard_env.static_obstacles),
                len(dmp_env.static_obstacles),
            )
            self.assertEqual(
                len(standard_env.dynamic_obstacles),
                len(dmp_env.dynamic_obstacles),
            )
            for standard_obstacle, dmp_obstacle in zip(
                standard_env.static_obstacles,
                dmp_env.static_obstacles,
            ):
                np.testing.assert_allclose(
                    standard_obstacle.center,
                    dmp_obstacle.center,
                )
            for standard_obstacle, dmp_obstacle in zip(
                standard_env.dynamic_obstacles,
                dmp_env.dynamic_obstacles,
            ):
                np.testing.assert_allclose(
                    standard_obstacle.center,
                    dmp_obstacle.center,
                )
                np.testing.assert_allclose(
                    standard_obstacle.velocity,
                    dmp_obstacle.velocity,
                )
        finally:
            standard_env.close()
            dmp_env.close()

    def test_online_history_resets_at_episode_boundary(self):
        history = AgentObservationHistory(
            temporal_steps=3,
            agent_ids=["agent_0", "agent_1"],
        )
        history.reset(
            {
                "agent_0": np.array([1.0, 2.0], dtype=np.float32),
                "agent_1": np.array([3.0, 4.0], dtype=np.float32),
            }
        )
        history.append(
            {
                "agent_0": np.array([5.0, 6.0], dtype=np.float32),
                "agent_1": np.array([7.0, 8.0], dtype=np.float32),
            }
        )
        observations, masks = history.policy_inputs()
        self.assertEqual(observations["agent_0"].shape, (2, 2))
        np.testing.assert_array_equal(
            masks["agent_0"],
            [True, True],
        )

        history.reset(
            {
                "agent_0": np.array([9.0, 10.0], dtype=np.float32),
                "agent_1": np.array([11.0, 12.0], dtype=np.float32),
            }
        )
        observations, masks = history.policy_inputs()
        np.testing.assert_allclose(
            observations["agent_0"],
            [[9.0, 10.0]],
        )
        np.testing.assert_array_equal(masks["agent_0"], [True])

    def test_replay_uses_applied_direct_acceleration(self):
        env = build_env(self.small_config())
        try:
            env.reset(seed=11)
            action = np.array(
                [
                    [10.0, -10.0, 2.0],
                    [0.0, 0.0, 0.0],
                    [-5.0, 5.0, 1.0],
                ],
                dtype=np.float32,
            )
            _, _, _, _, info = env.step(action)
            replay_action = build_replay_action_matrix(info, env)

            np.testing.assert_allclose(
                replay_action,
                [
                    [4.0, -4.0, 2.0],
                    [0.0, 0.0, 0.0],
                    [-4.0, 4.0, 1.0],
                ],
            )
            self.assertEqual(replay_action.shape, (3, 3))
        finally:
            env.close()

    def test_network_configuration_matches_direct_environment(self):
        args = self.small_args()
        config = self.small_config()
        env = build_env(config)
        try:
            agent_ids = [
                f"agent_{index}" for index in range(env.num_agents)
            ]
            dim_info = build_dim_info(env, agent_ids)
            network_config = build_network_config(env, config, args)

            self.assertEqual(
                dim_info,
                {
                    "agent_0": (27, 3),
                    "agent_1": (27, 3),
                    "agent_2": (27, 3),
                },
            )
            self.assertEqual(
                network_config.action_low,
                (-4.0, -4.0, -4.0),
            )
            self.assertEqual(
                network_config.action_high,
                (4.0, 4.0, 4.0),
            )
            self.assertEqual(
                network_config.extra_observation_dim,
                2,
            )
        finally:
            env.close()

    def test_cpu_smoke_simulation_writes_complete_artifacts(self):
        with tempfile.TemporaryDirectory() as output_root:
            args = self.small_args(
                "--output-root",
                output_root,
            )
            outputs = train(args)
            run_dir = Path(outputs["run_dir"])
            metrics_path = Path(outputs["metrics"])
            eval_metrics_path = Path(outputs["eval_metrics"])
            model_dir = Path(outputs["model_dir"])

            self.assertTrue((run_dir / "config.json").is_file())
            self.assertTrue(metrics_path.is_file())
            self.assertTrue(eval_metrics_path.is_file())
            self.assertTrue((model_dir / "MASAC.pth").is_file())
            self.assertTrue(
                (run_dir / "models" / "best" / "MASAC.pth").is_file()
            )
            self.assertTrue(
                (run_dir / "models" / "step_3" / "MASAC.pth").is_file()
            )
            self.assertTrue(
                (run_dir / "models" / "step_6" / "MASAC.pth").is_file()
            )

            with (run_dir / "config.json").open(
                "r",
                encoding="utf-8",
            ) as file:
                run_config = json.load(file)
            self.assertEqual(
                run_config["dim_info"]["agent_0"],
                [27, 3],
            )
            self.assertNotIn(
                "dmp_config",
                run_config["core_env_kwargs"],
            )
            self.assertTrue(
                DMP_CONFIG_FIELDS.isdisjoint(
                    run_config["experiment_config"]
                )
            )
            self.assertTrue(
                DMP_CONFIG_FIELDS.isdisjoint(
                    run_config["network_config"]
                )
            )

            with metrics_path.open(
                "r",
                newline="",
                encoding="utf-8",
            ) as file:
                metric_rows = list(csv.DictReader(file))
            with eval_metrics_path.open(
                "r",
                newline="",
                encoding="utf-8",
            ) as file:
                eval_rows = list(csv.DictReader(file))

            self.assertEqual(len(metric_rows), 2)
            self.assertEqual(len(eval_rows), 2)
            self.assertIn("learn_actor_loss", metric_rows[-1])
            self.assertIn(
                "mean_applied_acceleration_norm",
                metric_rows[-1],
            )
            self.assertIn(
                "acceleration_clip_fraction",
                metric_rows[-1],
            )


if __name__ == "__main__":
    unittest.main()
