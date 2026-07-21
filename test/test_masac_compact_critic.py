import sys
import unittest
from pathlib import Path

import torch


ALGO_LIB_ROOT = Path(__file__).resolve().parents[1] / "Multi-agent_Algo_lib"
if str(ALGO_LIB_ROOT) not in sys.path:
    sys.path.insert(0, str(ALGO_LIB_ROOT))

from net.masac import MASACCritic, ObservationEncoder


class CompactCentralizedCriticTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.agent_ids = ["agent_0", "agent_1", "agent_2"]
        self.sensor_dim = 7 + 2 * (2 * 2)
        self.extra_dim = 3
        self.ally_feature_dim = 7
        self.ally_count = 2
        self.obs_dim = (
            self.sensor_dim
            + self.extra_dim
            + self.ally_count * self.ally_feature_dim
        )
        self.action_dim = 6
        dim_info = {
            agent_id: (self.obs_dim, self.action_dim)
            for agent_id in self.agent_ids
        }
        self.critic = MASACCritic(
            dim_info,
            focal_agent_id="agent_0",
            sensor_observation_dim=self.sensor_dim,
            extra_observation_dim=self.extra_dim,
            ally_feature_dim=self.ally_feature_dim,
            sensor_output_dim=8,
            ally_output_dim=8,
            sensor_hidden_dim=8,
            ally_hidden_dim=8,
            hidden_dim=16,
            num_sensor_layers=1,
            num_ally_layers=1,
            num_observation_layers=1,
            sensor_azimuth_bins=2,
            sensor_elevation_bins=2,
            agent_pooling="mean_max",
            critic_encoder="attention",
        )

    def _inputs(self, batch_size=2, temporal_steps=3):
        observations = {
            agent_id: torch.rand(
                batch_size,
                temporal_steps,
                self.obs_dim,
                requires_grad=True,
            )
            for agent_id in self.agent_ids
        }
        actions = {
            agent_id: torch.rand(
                batch_size,
                self.action_dim,
                requires_grad=True,
            )
            for agent_id in self.agent_ids
        }
        masks = {
            agent_id: torch.ones(batch_size, temporal_steps, dtype=torch.bool)
            for agent_id in self.agent_ids
        }
        return observations, actions, masks

    def test_critic_uses_current_step_without_temporal_rnn(self):
        observations, actions, masks = self._inputs()
        changed_history = {
            agent_id: observation.detach().clone()
            for agent_id, observation in observations.items()
        }
        for observation in changed_history.values():
            observation[:, :-1] = torch.rand_like(observation[:, :-1]) * 100.0

        with torch.no_grad():
            original_q = self.critic.forward_q1(
                observations,
                actions,
                temporal_masks=masks,
            )
            changed_q = self.critic.forward_q1(
                changed_history,
                actions,
                temporal_masks=masks,
            )

        self.assertTrue(torch.equal(original_q, changed_q))
        self.assertIsNone(self.critic.q1.focal_sensor_encoder.rnn_layers)

    def test_neighbor_redundant_blocks_have_no_gradient_path(self):
        observations, actions, masks = self._inputs()
        q = self.critic.forward_q1(observations, actions, temporal_masks=masks)
        q.sum().backward()

        neighbor_gradient = observations["agent_1"].grad
        self.assertIsNotNone(neighbor_gradient)
        current_gradient = neighbor_gradient[:, -1]

        scan_start = 7
        scan_end = self.sensor_dim
        gain_start = self.sensor_dim + 1
        gain_end = self.sensor_dim + self.extra_dim
        ally_start = gain_end

        self.assertTrue(torch.count_nonzero(neighbor_gradient[:, :-1]) == 0)
        self.assertTrue(torch.count_nonzero(current_gradient[:, scan_start:scan_end]) == 0)
        self.assertTrue(torch.count_nonzero(current_gradient[:, gain_start:gain_end]) == 0)
        self.assertTrue(torch.count_nonzero(current_gradient[:, ally_start:]) == 0)

    def test_actor_sensor_encoder_keeps_temporal_rnn_by_default(self):
        encoder = ObservationEncoder(
            input_dim=self.sensor_dim,
            output_dim=8,
            hidden_dim=8,
            num_layers=1,
            sensor_azimuth_bins=2,
            sensor_elevation_bins=2,
        )
        self.assertIsNotNone(encoder.rnn_layers)


if __name__ == "__main__":
    unittest.main()
