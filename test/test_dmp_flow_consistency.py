from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALGO_ROOT = PROJECT_ROOT / "Multi-agent_Algo_lib"
for path in (PROJECT_ROOT, ALGO_ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from Controller.dmp_flow import (  # noqa: E402
    compute_dmp_drives,
    compute_dmp_flow_consistency,
    update_dmp_phase,
)
from Controller.dmp_rl import DMPConfig, SecondOrderDMPController  # noqa: E402
from MASAC.MASAC import MASAC  # noqa: E402
from MASAC.config import MASACNetworkConfig  # noqa: E402


class TestFlowConsistencyNumerics(unittest.TestCase):
    def test_zero_residual_is_consistent(self):
        nominal = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        consistency = compute_dmp_flow_consistency(nominal, nominal.copy())
        self.assertAlmostEqual(float(consistency), 1.0, places=6)

    def test_same_direction_is_consistent(self):
        nominal = torch.tensor([[1.0, 0.0, 0.0]])
        closed = torch.tensor([[3.0, 0.0, 0.0]])
        torch.testing.assert_close(
            compute_dmp_flow_consistency(nominal, closed),
            torch.ones(1),
        )

    def test_orthogonal_and_reverse_drives(self):
        nominal = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        closed = torch.tensor([[0.0, 2.0], [-1.0, 0.0]])
        consistency = compute_dmp_flow_consistency(nominal, closed)
        torch.testing.assert_close(consistency, torch.tensor([0.0, -1.0]))

    def test_equilibrium_rules(self):
        nominal = torch.zeros(2, 3)
        closed = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        consistency = compute_dmp_flow_consistency(
            nominal,
            closed,
            zero_threshold=1.0e-4,
        )
        torch.testing.assert_close(consistency, torch.tensor([1.0, 0.0]))

    def test_batch_and_multi_agent_shapes_are_finite(self):
        nominal = torch.randn(4, 3, 3, dtype=torch.float32)
        closed = nominal + 0.2 * torch.randn_like(nominal)
        consistency = compute_dmp_flow_consistency(nominal, closed)
        self.assertEqual(consistency.shape, (4, 3))
        self.assertTrue(torch.isfinite(consistency).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_inputs_are_finite(self):
        nominal = torch.randn(8, 3, device="cuda", dtype=torch.float32)
        closed = nominal + torch.randn_like(nominal)
        consistency = compute_dmp_flow_consistency(nominal, closed)
        self.assertEqual(consistency.device.type, "cuda")
        self.assertTrue(torch.isfinite(consistency).all())


class TestFCEPPhase(unittest.TestCase):
    def test_legacy_euler_preserves_original_baseline_update(self):
        next_phase, rate = update_dmp_phase(
            1.0,
            0.2,
            alpha_s=4.0,
            dt=0.1,
            tau=2.5,
            phase_min=0.0,
            phase_mode="classic",
            phase_integrator="legacy_euler",
        )
        self.assertAlmostEqual(float(next_phase), 0.84, places=7)
        self.assertEqual(float(rate), 1.0)

    def test_zero_residual_matches_exponential_classic_phase(self):
        nominal = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        consistency = compute_dmp_flow_consistency(nominal, nominal)
        classic, _ = update_dmp_phase(
            1.0,
            consistency,
            alpha_s=4.0,
            dt=0.1,
            tau=2.5,
            phase_min=1.0e-4,
            phase_mode="classic",
            phase_integrator="exponential",
        )
        fcep, _ = update_dmp_phase(
            1.0,
            consistency,
            alpha_s=4.0,
            dt=0.1,
            tau=2.5,
            phase_min=1.0e-4,
            phase_mode="fcep",
            phase_integrator="exponential",
        )
        self.assertAlmostEqual(float(classic), float(fcep), places=7)

    def test_nonpositive_consistency_pauses_phase(self):
        for consistency in (0.0, -0.5, -1.0):
            next_phase, rate = update_dmp_phase(
                0.5,
                consistency,
                alpha_s=4.0,
                dt=0.1,
                tau=2.5,
                phase_min=1.0e-4,
                phase_mode="fcep",
                phase_integrator="exponential",
            )
            self.assertEqual(float(next_phase), 0.5)
            self.assertEqual(float(rate), 0.0)

    def test_phase_is_bounded_and_nonincreasing(self):
        phase = torch.ones(4, dtype=torch.float32)
        consistency = torch.tensor([1.0, 0.4, 0.0, -1.0])
        next_phase, _ = update_dmp_phase(
            phase,
            consistency,
            alpha_s=4.0,
            dt=0.1,
            tau=2.5,
            phase_min=1.0e-4,
            phase_mode="fcep",
            phase_integrator="exponential",
        )
        self.assertTrue(torch.all(next_phase <= phase))
        self.assertTrue(torch.all(next_phase >= 1.0e-4))
        self.assertFalse(next_phase.requires_grad)


class TestFlowConstraintGradients(unittest.TestCase):
    @staticmethod
    def _policy() -> MASAC:
        network_config = MASACNetworkConfig(
            sensor_observation_dim=11,
            extra_observation_dim=0,
            ally_feature_dim=0,
            sensor_azimuth_bins=2,
            sensor_elevation_bins=2,
            sensor_include_previous_scan=False,
            sensor_hidden_dim=16,
            sensor_output_dim=16,
            hidden_dim=16,
            action_low=(-10.0, -10.0, -10.0, -1.0, -1.0, -1.0),
            action_high=(10.0, 10.0, 10.0, 1.0, 1.0, 1.0),
            temporal_steps=1,
            critic_encoder="mlp",
            goal_distance_clip=1.0,
            dmp_k_alpha=1.0,
            dmp_k_beta=1.0,
            dmp_tau=1.0,
        )
        return MASAC(
            dim_info={"agent_0": (11, 6)},
            is_continue=True,
            actor_lr=1.0e-4,
            critic_lr=1.0e-4,
            buffer_size=8,
            device="cpu",
            network_config=network_config,
        )

    def test_flow_loss_only_has_direct_action_gradient_on_forcing(self):
        policy = self._policy()
        observation = torch.zeros(1, 11)
        observation[0, 3] = 1.0
        observation[0, 6] = 1.0
        sampled_action = torch.tensor(
            [[-3.0, 1.0, 0.0, 0.2, -0.1, 0.0]],
            requires_grad=True,
        )

        flow_loss, consistency, _ = policy.compute_flow_consistency_loss(
            observation,
            sampled_action,
            minimum_consistency=0.0,
        )
        self.assertLess(float(consistency.item()), 0.0)
        flow_loss.backward()

        self.assertGreater(float(sampled_action.grad[:, :3].abs().sum()), 0.0)
        torch.testing.assert_close(
            sampled_action.grad[:, 3:],
            torch.zeros_like(sampled_action.grad[:, 3:]),
        )

    def test_detached_nominal_drive_has_no_gradient(self):
        nominal = torch.tensor([[1.0, 0.0]], requires_grad=True)
        residual = torch.tensor([[-2.0, 0.3]], requires_grad=True)
        nominal_for_loss = nominal.detach()
        consistency = compute_dmp_flow_consistency(
            nominal_for_loss,
            nominal_for_loss + residual,
        )
        loss = torch.relu(-consistency).square().mean()
        loss.backward()
        self.assertIsNone(nominal.grad)
        self.assertIsNotNone(residual.grad)
        self.assertGreater(float(residual.grad.abs().sum()), 0.0)


class TestControllerCompatibility(unittest.TestCase):
    def test_classic_and_fcep_preserve_action_and_output_shapes(self):
        position = np.zeros(3, dtype=np.float32)
        velocity = np.zeros(3, dtype=np.float32)
        action = np.zeros(6, dtype=np.float32)
        for phase_mode in ("classic", "fcep"):
            controller = SecondOrderDMPController(
                DMPConfig(
                    dt=0.1,
                    phase_mode=phase_mode,
                    phase_integrator="exponential",
                    phase_min=1.0e-4,
                )
            )
            controller.reset(position, np.array([2.0, 0.0, 0.0]))
            acceleration, info = controller.compute_acceleration(
                position,
                velocity,
                action,
            )
            self.assertEqual(acceleration.shape, (3,))
            self.assertEqual(info["forcing"].shape, (3,))
            self.assertEqual(info["goal_offset"].shape, (3,))
            self.assertTrue(np.isfinite(acceleration).all())


if __name__ == "__main__":
    unittest.main()
