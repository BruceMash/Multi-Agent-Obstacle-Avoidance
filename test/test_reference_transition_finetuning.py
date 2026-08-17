from __future__ import annotations

import types
import unittest
from pathlib import Path

import numpy as np

from experiment_config import EXPERIMENT_CONFIG
from planning.historical_forcing_gate import HISTORICAL_GATE_NAME
from planning.reference_transition_finetuning import (
    TASK_REFERENCE,
    TASK_TERMINAL,
    BlockTaskSampler,
    build_reference_transition_env,
    load_checkpoint_weights_only,
    sha256_file,
    smoke_training_config,
)
from runner_sac import build_model


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = REPO_ROOT / "artifacts" / "20260520_201912" / "best_eval_model.pt"
EXPECTED_CHECKPOINT_HASH = (
    "0c3595f738b2f2f2b7e88fc90479e6d525c986a216ab2d1b2eecc139833ad3d5"
)


def _options(start=(0.2, 0.0, 0.0), goal=(7.8, 0.0, 0.0)):
    return {
        "start": np.asarray(start, dtype=float),
        "goal": np.asarray(goal, dtype=float),
        "static_obstacles": [],
        "dynamic_obstacles": [],
    }


class ReferenceTransitionFineTuningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = smoke_training_config(EXPERIMENT_CONFIG)

    def build_env(self, task=TASK_TERMINAL):
        return build_reference_transition_env(self.config, forced_task=task)

    def force_reference_at_start(self, env):
        env.requested_task = TASK_REFERENCE
        env.episode_task = TASK_REFERENCE
        env.reference_generation_fallback = False
        env.temporary_reference = env.dynamics.p.copy()
        env.reference_reached = False
        env.reference_reached_step = None
        env.reference_handoff_count = 0
        env._set_active_goal_preserve_state(env.temporary_reference)
        env._progress_baseline_distance = 0.0

    def test_task_sampler_is_exact_sixty_forty_per_block(self):
        sampler = BlockTaskSampler()
        rng = np.random.default_rng(7)
        values = [sampler.sample(rng) for _ in range(20)]
        for first in (values[:10], values[10:]):
            self.assertEqual(first.count(TASK_TERMINAL), 6)
            self.assertEqual(first.count(TASK_REFERENCE), 4)

    def test_terminal_task_keeps_historical_goal_semantics(self):
        env = self.build_env(TASK_TERMINAL)
        try:
            observation, _ = env.reset(seed=1, options=_options())
            self.assertEqual(observation.shape, (122,))
            np.testing.assert_array_equal(env.goal, env.terminal_goal)
            np.testing.assert_array_equal(env.active_goal, env.terminal_goal)
            np.testing.assert_array_equal(env.dmp.goal, env.terminal_goal)
            _, _, _, _, info = env.step(np.zeros(6, dtype=np.float32))
            self.assertEqual(info["forcing_gate_semantics"], HISTORICAL_GATE_NAME)
            self.assertEqual(info["progress_reference"], TASK_TERMINAL)
        finally:
            env.close()

    def test_reference_actor_observation_and_dmp_share_active_goal(self):
        env = self.build_env(TASK_REFERENCE)
        try:
            observation, _ = env.reset(seed=2, options=_options())
            self.assertFalse(env.reference_generation_fallback)
            np.testing.assert_array_equal(env.goal, env.terminal_goal)
            np.testing.assert_array_equal(env.dmp.goal, env.active_goal)
            direction = env.active_goal - env.dynamics.p
            direction /= np.linalg.norm(direction)
            np.testing.assert_allclose(observation[3:6], direction, atol=1e-6)
            self.assertFalse(np.array_equal(env.active_goal, env.terminal_goal))
        finally:
            env.close()

    def test_historical_dynamics_and_reward_use_identical_gate(self):
        env = self.build_env(TASK_REFERENCE)
        try:
            env.reset(seed=3, options=_options())
            action = np.asarray([7.0, -4.0, 2.0, 0.3, -0.2, 0.1], dtype=np.float32)
            position = env.dynamics.p.copy()
            active = env.active_goal.copy()
            _, _, _, _, info = env.step(action)
            expected_goal_eff = active + action[3:]
            expected_gate = np.tanh(np.abs(expected_goal_eff - position))
            np.testing.assert_allclose(info["dynamics_goal_eff"], expected_goal_eff, atol=1e-7)
            np.testing.assert_allclose(info["reward_goal_eff"], expected_goal_eff, atol=1e-7)
            np.testing.assert_allclose(info["dynamics_forcing_gate"], expected_gate, atol=1e-7)
            np.testing.assert_allclose(info["reward_forcing_gate"], expected_gate, atol=1e-7)
        finally:
            env.close()

    def test_handoff_switches_once_without_reset_or_success_bonus(self):
        env = self.build_env(TASK_TERMINAL)
        try:
            env.reset(seed=4, options=_options())
            self.force_reference_at_start(env)
            phase_before = float(env.dmp.phase)
            velocity_before = env.dynamics.v.copy()
            scan_history_before = env.sensor._previous_scan.copy()
            _, reward, terminated, truncated, info = env.step(
                np.zeros(6, dtype=np.float32)
            )
            self.assertTrue(info["reference_handoff_event"])
            self.assertEqual(info["reference_handoff_count"], 1)
            self.assertEqual(info["progress_reference"], TASK_REFERENCE)
            self.assertFalse(info["terminal_success"])
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertLess(reward, 100.0)
            self.assertAlmostEqual(env.dmp.phase, phase_before * 0.84, places=12)
            np.testing.assert_array_equal(env.dynamics.v, velocity_before)
            np.testing.assert_array_equal(env.sensor._previous_scan, scan_history_before)
            np.testing.assert_array_equal(env.active_goal, env.terminal_goal)
            np.testing.assert_array_equal(env.goal, env.terminal_goal)
            _, _, _, _, next_info = env.step(np.zeros(6, dtype=np.float32))
            self.assertFalse(next_info["reference_handoff_event"])
            self.assertEqual(next_info["reference_handoff_count"], 1)
        finally:
            env.close()

    def test_handoff_resets_progress_baseline_without_cross_reference_jump(self):
        env = self.build_env(TASK_TERMINAL)
        try:
            env.reset(seed=5, options=_options())
            self.force_reference_at_start(env)
            _, _, _, _, handoff = env.step(np.zeros(6, dtype=np.float32))
            self.assertAlmostEqual(
                handoff["progress_baseline_after"],
                handoff["terminal_goal_distance"],
                places=12,
            )
            baseline = float(handoff["progress_baseline_after"])
            _, _, _, _, following = env.step(np.zeros(6, dtype=np.float32))
            expected = baseline - float(following["terminal_goal_distance"])
            self.assertAlmostEqual(following["progress"], expected, places=12)
            self.assertAlmostEqual(
                following["reward_progress"],
                self.config.step_reward_weight * expected,
                places=10,
            )
        finally:
            env.close()

    def test_only_terminal_goal_triggers_success_bonus(self):
        env = self.build_env(TASK_TERMINAL)
        try:
            env.reset(
                seed=6,
                options=_options(start=(4.0, 0.0, 0.0), goal=(4.1, 0.0, 0.0)),
            )
            _, reward, terminated, _, info = env.step(np.zeros(6, dtype=np.float32))
            self.assertTrue(info["terminal_success"])
            self.assertTrue(terminated)
            self.assertGreater(reward, 299.0)
        finally:
            env.close()

    def test_forbidden_features_and_hard_boundary_are_absent(self):
        env = self.build_env(TASK_REFERENCE)
        try:
            env.reset(seed=7, options=_options())
            _, _, _, _, info = env.step(np.zeros(6, dtype=np.float32))
            self.assertFalse(info["GAT_used"])
            self.assertFalse(info["FP_SHEP_selector_used"])
            self.assertFalse(info["repeated_waypoint_used"])
            self.assertFalse(info["hard_boundary_used"])
        finally:
            env.close()

    def test_no_candidate_falls_back_without_fake_reference(self):
        env = self.build_env(TASK_REFERENCE)

        def no_reference(_self):
            return None, 0

        env._select_temporary_reference = types.MethodType(no_reference, env)
        try:
            _, info = env.reset(seed=8, options=_options())
            self.assertTrue(info["reference_generation_fallback"])
            self.assertEqual(env.episode_task, TASK_TERMINAL)
            self.assertIsNone(env.temporary_reference)
            np.testing.assert_array_equal(env.active_goal, env.terminal_goal)
        finally:
            env.close()

    def test_weights_only_warm_start_preserves_checkpoint_and_fresh_state(self):
        original_hash = sha256_file(CHECKPOINT)
        self.assertEqual(original_hash, EXPECTED_CHECKPOINT_HASH)
        env = self.build_env(TASK_TERMINAL)
        try:
            model = build_model(env, config=self.config, tensorboard_log=None, verbose=0)
            audit = load_checkpoint_weights_only(
                model,
                CHECKPOINT,
                learning_rate=1.0e-4,
            )
            self.assertTrue(audit.weights_match)
            self.assertTrue(audit.fresh_training_state)
            self.assertEqual(model.replay_buffer.size(), 0)
            self.assertEqual(len(model.actor.optimizer.state), 0)
            self.assertEqual(len(model.critic.optimizer.state), 0)
            self.assertEqual(len(model.ent_coef_optimizer.state), 0)
            self.assertEqual(sha256_file(CHECKPOINT), original_hash)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
