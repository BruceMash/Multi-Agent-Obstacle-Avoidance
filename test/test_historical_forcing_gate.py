from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from Controller.dmp_rl import DMPConfig, compute_dmp_transition
from planning.historical_forcing_gate import (
    VARIANT_A,
    VARIANT_B,
    VARIANT_C,
    VARIANT_D,
    compute_historical_checkpoint_dmp_transition,
    historical_goal_base_for_variant,
    scoped_historical_multi_agent_transition,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = REPO_ROOT / "artifacts" / "20260520_201912" / "best_eval_model.pt"
EXPECTED_CHECKPOINT_SHA256 = (
    "0c3595f738b2f2f2b7e88fc90479e6d525c986a216ab2d1b2eecc139833ad3d5"
)


def historical_config() -> DMPConfig:
    return DMPConfig(
        dt=0.1,
        dims=3,
        K_alpha=3.0,
        K_beta=0.8,
        alpha_s=4.0,
        tau=2.5,
        forcing_term_min=-10.0,
        forcing_term_max=10.0,
        goal_offset_max=1.0,
        phase_mode="classic",
        phase_integrator="legacy_euler",
    )


class HistoricalForcingGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = historical_config()
        self.position = np.asarray([0.2, -0.4, 0.1], dtype=float)
        self.velocity = np.asarray([0.1, -0.2, 0.05], dtype=float)
        self.active_goal = np.asarray([2.0, 0.7, -0.3], dtype=float)
        self.terminal_goal = np.asarray([7.5, 0.2, 0.0], dtype=float)
        self.action = np.asarray([3.0, -4.0, 5.0, 0.3, -0.2, 0.4], dtype=float)

    def transition(self):
        return compute_historical_checkpoint_dmp_transition(
            config=self.config,
            position=self.position,
            velocity=self.velocity,
            rl_action=self.action,
            active_goal=self.active_goal,
            terminal_goal=self.terminal_goal,
            phase=1.0,
        )

    def test_01_historical_gate_is_three_dimensional_vector(self) -> None:
        _, _, info = self.transition()
        self.assertEqual(np.asarray(info["forcing_gate"]).shape, (3,))

    def test_02_historical_gate_matches_goal_eff_per_axis_formula(self) -> None:
        _, _, info = self.transition()
        expected = np.tanh(np.abs(np.asarray(info["goal_eff"]) - self.position))
        np.testing.assert_allclose(info["forcing_gate"], expected, rtol=0.0, atol=1e-12)
        self.assertFalse(np.allclose(expected, np.full(3, expected[0])))

    def test_03_goal_eff_contains_only_current_step_goal_offset(self) -> None:
        _, _, first = self.transition()
        expected = self.active_goal + self.action[3:]
        np.testing.assert_allclose(first["goal_eff"], expected, rtol=0.0, atol=1e-12)
        second_action = self.action.copy()
        second_action[3:] = 0.0
        _, _, second = compute_historical_checkpoint_dmp_transition(
            config=self.config,
            position=self.position,
            velocity=self.velocity,
            rl_action=second_action,
            active_goal=self.active_goal,
            terminal_goal=self.terminal_goal,
            phase=1.0,
        )
        np.testing.assert_allclose(second["goal_eff"], self.active_goal, rtol=0.0, atol=0.0)

    def test_04_variants_a_and_b_use_terminal_dmp_base_goal(self) -> None:
        for variant in (VARIANT_A, VARIANT_B):
            base = historical_goal_base_for_variant(
                variant,
                terminal_goal=self.terminal_goal,
                active_goal=self.active_goal,
            )
            np.testing.assert_array_equal(base, self.terminal_goal)

    def test_05_variants_c_and_d_use_active_temporary_dmp_base_goal(self) -> None:
        for variant in (VARIANT_C, VARIANT_D):
            base = historical_goal_base_for_variant(
                variant,
                terminal_goal=self.terminal_goal,
                active_goal=self.active_goal,
            )
            np.testing.assert_array_equal(base, self.active_goal)

    def test_06_current_scalar_gate_implementation_still_exists(self) -> None:
        _, _, current = compute_dmp_transition(
            config=self.config,
            position=self.position,
            velocity=self.velocity,
            rl_action=self.action,
            active_goal=self.active_goal,
            terminal_goal=self.terminal_goal,
            phase=1.0,
        )
        gate = np.asarray(current["forcing_gate"], dtype=float)
        self.assertEqual(gate.shape, (3,))
        np.testing.assert_allclose(gate, np.full(3, gate[0]), rtol=0.0, atol=0.0)
        _, _, historical = self.transition()
        self.assertFalse(np.allclose(gate, historical["forcing_gate"]))

    def test_07_scoped_wrapper_restores_current_transition(self) -> None:
        import Environment.multi_agent_dmp_env as environment_module

        original = environment_module.propagate_sac_dmp_action
        with scoped_historical_multi_agent_transition():
            self.assertIsNot(environment_module.propagate_sac_dmp_action, original)
        self.assertIs(environment_module.propagate_sac_dmp_action, original)

    def test_08_scoped_wrapper_restores_after_exception(self) -> None:
        import Environment.multi_agent_dmp_env as environment_module

        original = environment_module.propagate_sac_dmp_action
        with self.assertRaisesRegex(RuntimeError, "diagnostic failure"):
            with scoped_historical_multi_agent_transition():
                raise RuntimeError("diagnostic failure")
        self.assertIs(environment_module.propagate_sac_dmp_action, original)

    def test_09_checkpoint_hash_is_unchanged(self) -> None:
        digest = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
        self.assertEqual(digest, EXPECTED_CHECKPOINT_SHA256)

    def test_10_configuration_and_evaluator_forbid_training(self) -> None:
        config_path = (
            REPO_ROOT
            / "configs"
            / "evaluation"
            / "historical_gate_abcd_rerun.json"
        )
        settings = json.loads(config_path.read_text(encoding="utf-8"))
        exclusions = settings["strict_exclusions"]
        self.assertFalse(exclusions["SAC_training"])
        self.assertFalse(exclusions["Actor_update"])
        self.assertFalse(exclusions["Critic_update"])
        self.assertFalse(exclusions["alpha_update"])
        self.assertFalse(exclusions["replay_buffer_training"])
        evaluator = (
            REPO_ROOT
            / "Multi-agent_Algo_lib"
            / "scripts"
            / "evaluate_historical_gate_abcd_rerun.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn(".learn(", evaluator)
        self.assertNotIn("optimizer.step", evaluator)


if __name__ == "__main__":
    unittest.main()
