from __future__ import annotations

import unittest

import numpy as np

from Environment.multi_agent_dmp_env import MultiAgentDMPEnv


def _set_state(env: MultiAgentDMPEnv, index: int, position, velocity) -> None:
    dynamic = env.dynamics[index]
    dynamic.p = np.asarray(position, dtype=float).copy()
    dynamic.v = np.asarray(velocity, dtype=float).copy()
    dynamic.state = np.concatenate([dynamic.p, dynamic.v])


def _make_env(*, local: bool) -> MultiAgentDMPEnv:
    config = {
        "num_agents": 3,
        "max_steps": 10,
        "goal_tolerance": 0.3,
        "nearest_agent_observation_count": 2,
        "min_start_goal_distance": 0.0,
        "randomize_start_goal": False,
        "workspace_bounds": ((0.0, 0.0, 0.8), (100.0, 100.0, 3.2)),
    }
    if local:
        config.update(
            {
                "inter_agent_influence_distance": 4.5,
                "peer_state_observation_mode": "local_anonymous_ally_block",
                "peer_state_observation_range": 4.5,
            }
        )
    env = MultiAgentDMPEnv(
        dynamics_config={
            "velocity_clip": [-4.0, 4.0],
            "accelerate_clip": [-4.0, 4.0],
            "time_step": 0.1,
        },
        sensor_config={
            "sensing_radius": 4.5,
            "azimuth_bins": 8,
            "elevation_bins": 7,
            "include_previous_scan": True,
        },
        dmp_config={"dt": 0.1, "dims": 3},
        env_config=config,
        static_obstacles=[],
        dynamic_obstacles=[],
    )
    env.reset(
        seed=1,
        options={
            "starts": np.asarray([[10, 10, 2], [12, 10, 2], [20, 20, 2]], dtype=float),
            "goals": np.asarray([[80, 10, 2], [82, 10, 2], [80, 20, 2]], dtype=float),
            "static_obstacles": [],
            "dynamic_obstacles": [],
        },
    )
    return env


class LocalAnonymousPeerObservationTest(unittest.TestCase):
    def test_legacy_mode_preserves_global_exact_event_state(self) -> None:
        env = _make_env(local=False)
        try:
            rows = env.observable_neighbor_states(0)
            self.assertEqual([row.agent_id for row in rows], [1, 2])
            np.testing.assert_allclose(rows[1].position, [20, 20, 2])
        finally:
            env.close()

    def test_local_mode_filters_range_and_zero_pads_flat_block(self) -> None:
        env = _make_env(local=True)
        try:
            rows = env.observable_neighbor_states(0)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].agent_id, 0)
            np.testing.assert_allclose(rows[0].position, [12, 10, 2], atol=1.0e-6)
            flat = env._compose_inter_agent_observation(0).reshape(2, 7)
            self.assertFalse(np.allclose(flat[0], 0.0))
            np.testing.assert_allclose(flat[1], 0.0)
        finally:
            env.close()

    def test_range_external_private_state_cannot_change_local_output(self) -> None:
        env = _make_env(local=True)
        try:
            before_flat = env._compose_inter_agent_observation(0).copy()
            before_rows = env.observable_neighbor_states(0)
            _set_state(env, 2, [95, 95, 3], [-4, 4, 0])
            after_flat = env._compose_inter_agent_observation(0)
            after_rows = env.observable_neighbor_states(0)
            np.testing.assert_array_equal(before_flat, after_flat)
            self.assertEqual(len(before_rows), len(after_rows))
            np.testing.assert_allclose(before_rows[0].position, after_rows[0].position)
            np.testing.assert_allclose(before_rows[0].velocity, after_rows[0].velocity)
        finally:
            env.close()

    def test_simulator_identity_permutation_is_not_observable(self) -> None:
        env = _make_env(local=True)
        try:
            _set_state(env, 1, [12, 11, 2.2], [0.5, -0.3, 0.1])
            _set_state(env, 2, [11, 12, 1.8], [-0.4, 0.2, -0.1])
            before_flat = env._compose_inter_agent_observation(0).copy()
            before_rows = env.observable_neighbor_states(0)
            state_1 = (env.dynamics[1].p.copy(), env.dynamics[1].v.copy())
            state_2 = (env.dynamics[2].p.copy(), env.dynamics[2].v.copy())
            _set_state(env, 1, *state_2)
            _set_state(env, 2, *state_1)
            after_flat = env._compose_inter_agent_observation(0)
            after_rows = env.observable_neighbor_states(0)
            np.testing.assert_array_equal(before_flat, after_flat)
            self.assertEqual([row.agent_id for row in before_rows], [0, 1])
            self.assertEqual([row.agent_id for row in after_rows], [0, 1])
            np.testing.assert_allclose(
                np.stack([row.position for row in before_rows]),
                np.stack([row.position for row in after_rows]),
            )
        finally:
            env.close()

    def test_graph_accessor_cannot_bypass_native_velocity_clipping(self) -> None:
        env = _make_env(local=True)
        try:
            _set_state(env, 0, [10, 10, 2], [-3.0, 0.0, 0.0])
            _set_state(env, 1, [12, 10, 2], [3.0, 0.0, 0.0])
            rows = env.observable_neighbor_states(0)
            self.assertEqual(len(rows), 1)
            # Native relative-velocity scale is 4 m/s, so +6 is clipped to +4.
            np.testing.assert_allclose(rows[0].velocity, [1.0, 0.0, 0.0], atol=1.0e-6)
        finally:
            env.close()

    def test_invalid_local_range_above_native_scale_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            env = MultiAgentDMPEnv(
                dynamics_config={
                    "velocity_clip": [-4.0, 4.0],
                    "accelerate_clip": [-4.0, 4.0],
                    "time_step": 0.1,
                },
                env_config={
                    "num_agents": 3,
                    "inter_agent_influence_distance": 1.2,
                    "peer_state_observation_mode": "local_anonymous_ally_block",
                    "peer_state_observation_range": 4.5,
                },
            )
            env.close()


if __name__ == "__main__":
    unittest.main()
