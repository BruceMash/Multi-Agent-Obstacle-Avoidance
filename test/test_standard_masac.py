import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


ALGO_LIB_ROOT = Path(__file__).resolve().parents[1] / "Multi-agent_Algo_lib"
if str(ALGO_LIB_ROOT) not in sys.path:
    sys.path.insert(0, str(ALGO_LIB_ROOT))

from MASAC.MASAC import MASAC
from MASAC.config import MASACNetworkConfig
from MASAC.standard_masac import StandardMASAC


class TestStandardMASAC(unittest.TestCase):
    agent_ids = ("agent_0", "agent_1", "agent_2")
    sensor_dim = 7 + 4
    extra_dim = 2
    ally_feature_dim = 7
    ally_count = 2
    obs_dim = sensor_dim + extra_dim + ally_feature_dim * ally_count
    action_dim = 3

    def setUp(self):
        torch.manual_seed(20260728)
        np.random.seed(20260728)

    def build_network_config(
        self,
        *,
        temporal_steps: int = 2,
        action_dim: int = 3,
    ) -> MASACNetworkConfig:
        return MASACNetworkConfig(
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
            sensor_include_previous_scan=False,
            ally_pooling="mean_max",
            agent_pooling="mean_max",
            critic_encoder="attention",
            action_low=tuple(-4.0 for _ in range(action_dim)),
            action_high=tuple(4.0 for _ in range(action_dim)),
            temporal_steps=temporal_steps,
        )

    def build_policy(
        self,
        *,
        buffer_size: int = 64,
        temporal_steps: int = 2,
    ) -> StandardMASAC:
        dim_info = {
            agent_id: (self.obs_dim, self.action_dim)
            for agent_id in self.agent_ids
        }
        return StandardMASAC(
            dim_info=dim_info,
            is_continue=True,
            actor_lr=3e-4,
            critic_lr=3e-4,
            buffer_size=buffer_size,
            device="cpu",
            network_config=self.build_network_config(
                temporal_steps=temporal_steps
            ),
        )

    def random_observation_dict(
        self,
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        return {
            agent_id: rng.random(self.obs_dim).astype(np.float32)
            for agent_id in self.agent_ids
        }

    def add_transitions(
        self,
        policy: StandardMASAC,
        count: int = 8,
    ) -> None:
        rng = np.random.default_rng(17)
        for transition_index in range(count):
            obs = self.random_observation_dict(rng)
            next_obs = self.random_observation_dict(rng)
            actions = {
                agent_id: rng.uniform(
                    -4.0,
                    4.0,
                    size=self.action_dim,
                ).astype(np.float32)
                for agent_id in self.agent_ids
            }
            rewards = {
                agent_id: float(rng.normal())
                for agent_id in self.agent_ids
            }
            done = {
                agent_id: transition_index == count - 1
                for agent_id in self.agent_ids
            }
            policy.add(
                obs,
                actions,
                rewards,
                next_obs,
                done,
                episode_end=done,
            )

    def test_reuses_independent_existing_networks_without_dmp_state(self):
        policy = self.build_policy()

        self.assertEqual(policy.action_dim, 3)
        self.assertFalse(hasattr(policy, "control_dim"))
        self.assertFalse(hasattr(policy, "dmp_k_alpha"))
        self.assertFalse(hasattr(policy, "dmp_k_beta"))
        self.assertFalse(hasattr(policy, "dmp_tau"))
        self.assertFalse(hasattr(policy, "forcing_term_min"))
        self.assertFalse(hasattr(policy, "forcing_term_max"))

        actor_ids = {
            id(policy.agents[agent_id].actor)
            for agent_id in self.agent_ids
        }
        critic_ids = {
            id(policy.agents[agent_id].critic)
            for agent_id in self.agent_ids
        }
        alpha_ids = {
            id(policy.alphas[agent_id])
            for agent_id in self.agent_ids
        }
        self.assertEqual(len(actor_ids), len(self.agent_ids))
        self.assertEqual(len(critic_ids), len(self.agent_ids))
        self.assertEqual(len(alpha_ids), len(self.agent_ids))

    def test_actor_outputs_bounded_three_dimensional_action_and_log_prob(self):
        policy = self.build_policy()
        actor = policy.agents["agent_0"].actor
        observation = torch.rand(5, 2, self.obs_dim)
        temporal_mask = torch.ones(5, 2, dtype=torch.bool)

        action, log_prob = actor(
            observation,
            temporal_mask=temporal_mask,
        )
        deterministic_action, _ = actor(
            observation,
            deterministic=True,
            temporal_mask=temporal_mask,
        )

        self.assertEqual(action.shape, (5, 3))
        self.assertEqual(deterministic_action.shape, (5, 3))
        self.assertEqual(log_prob.shape, (5, 1))
        self.assertTrue(torch.all(action >= -4.0))
        self.assertTrue(torch.all(action <= 4.0))
        self.assertTrue(torch.all(deterministic_action >= -4.0))
        self.assertTrue(torch.all(deterministic_action <= 4.0))
        self.assertTrue(torch.isfinite(log_prob).all())

    def test_online_action_interfaces_use_current_network_design(self):
        policy = self.build_policy()
        rng = np.random.default_rng(23)
        observations = self.random_observation_dict(rng)

        sampled_actions = policy.select_action(observations)
        deterministic_actions_1 = policy.evaluate_action(observations)
        deterministic_actions_2 = policy.evaluate_action(observations)

        self.assertEqual(set(sampled_actions), set(self.agent_ids))
        for agent_id in self.agent_ids:
            self.assertEqual(sampled_actions[agent_id].shape, (3,))
            self.assertTrue(
                np.all(sampled_actions[agent_id] >= -4.0)
            )
            self.assertTrue(
                np.all(sampled_actions[agent_id] <= 4.0)
            )
            np.testing.assert_allclose(
                deterministic_actions_1[agent_id],
                deterministic_actions_2[agent_id],
            )

    def test_actor_and_critic_actions_have_identity_semantics(self):
        policy = self.build_policy()
        actor_action = torch.tensor(
            [[-4.0, 0.5, 3.0]],
            dtype=torch.float32,
        )

        critic_action = policy.actor_action_to_critic_action(
            obs=None,
            actor_action=actor_action,
        )

        self.assertIs(critic_action, actor_action)
        torch.testing.assert_close(critic_action, actor_action)

    def test_centralized_twin_critic_accepts_joint_three_dimensional_actions(self):
        policy = self.build_policy()
        batch_size = 4
        observations = {
            agent_id: torch.rand(batch_size, 2, self.obs_dim)
            for agent_id in self.agent_ids
        }
        actions = {
            agent_id: torch.rand(batch_size, self.action_dim)
            for agent_id in self.agent_ids
        }
        temporal_masks = {
            agent_id: torch.ones(batch_size, 2, dtype=torch.bool)
            for agent_id in self.agent_ids
        }

        q1, q2 = policy.agents["agent_0"].critic(
            observations,
            actions,
            temporal_masks=temporal_masks,
        )

        self.assertEqual(q1.shape, (batch_size, 1))
        self.assertEqual(q2.shape, (batch_size, 1))
        self.assertIsNot(
            policy.agents["agent_0"].critic.q1,
            policy.agents["agent_0"].critic.q2,
        )
        self.assertTrue(torch.isfinite(q1).all())
        self.assertTrue(torch.isfinite(q2).all())

    def test_joint_replay_preserves_direct_acceleration_actions(self):
        policy = self.build_policy()
        self.add_transitions(policy, count=5)

        for agent_id in self.agent_ids:
            buffer = policy.buffers[agent_id]
            self.assertEqual(len(buffer), 5)
            self.assertEqual(buffer.actions.shape[1], 3)
            np.testing.assert_array_equal(
                buffer.transition_ids[:5],
                np.arange(5),
            )
        np.testing.assert_array_equal(
            policy.buffers["agent_0"].transition_ids,
            policy.buffers["agent_1"].transition_ids,
        )

    def test_one_complete_learning_update_changes_actor_and_critic(self):
        policy = self.build_policy()
        self.add_transitions(policy, count=10)
        actor = policy.agents["agent_0"].actor
        critic = policy.agents["agent_0"].critic
        actor_before = [
            parameter.detach().clone()
            for parameter in actor.parameters()
        ]
        critic_before = [
            parameter.detach().clone()
            for parameter in critic.parameters()
        ]

        diagnostics = policy.learn(batch_size=4, gamma=0.99, tau=0.01)

        self.assertEqual(
            set(diagnostics),
            {
                "critic_loss",
                "actor_loss",
                "q_replay",
                "q_policy",
                "q_target",
                "entropy",
                "alpha",
                "alpha_loss",
            },
        )
        self.assertTrue(
            np.isfinite(np.asarray(list(diagnostics.values()))).all()
        )
        self.assertTrue(
            any(
                not torch.equal(before, after)
                for before, after in zip(actor_before, actor.parameters())
            )
        )
        self.assertTrue(
            any(
                not torch.equal(before, after)
                for before, after in zip(critic_before, critic.parameters())
            )
        )

    def test_target_update_uses_soft_update(self):
        policy = self.build_policy()
        agent = policy.agents["agent_0"]
        source_parameter = next(agent.actor.parameters())
        target_parameter = next(agent.actor_target.parameters())
        target_before = target_parameter.detach().clone()
        with torch.no_grad():
            source_parameter.add_(1.0)
        source_after = source_parameter.detach().clone()

        policy.update_target(tau=0.25)

        expected = 0.75 * target_before + 0.25 * source_after
        torch.testing.assert_close(target_parameter, expected)

    def test_actor_save_and_load_preserve_deterministic_actions(self):
        policy = self.build_policy()
        rng = np.random.default_rng(31)
        observations = self.random_observation_dict(rng)
        expected_actions = policy.evaluate_action(observations)
        dim_info = {
            agent_id: (self.obs_dim, self.action_dim)
            for agent_id in self.agent_ids
        }

        with tempfile.TemporaryDirectory() as model_dir:
            policy.save(model_dir)
            loaded = StandardMASAC.load(
                dim_info=dim_info,
                is_continue=True,
                model_dir=model_dir,
                network_config=self.build_network_config(),
                device="cpu",
            )
            actual_actions = loaded.evaluate_action(observations)

        for agent_id in self.agent_ids:
            np.testing.assert_allclose(
                actual_actions[agent_id],
                expected_actions[agent_id],
                rtol=1e-6,
                atol=1e-6,
            )

    def test_existing_dmp_masac_interface_is_unchanged(self):
        dim_info = {
            agent_id: (self.obs_dim, 6)
            for agent_id in self.agent_ids
        }
        policy = MASAC(
            dim_info=dim_info,
            is_continue=True,
            actor_lr=3e-4,
            critic_lr=3e-4,
            buffer_size=8,
            device="cpu",
            network_config=self.build_network_config(action_dim=6),
        )

        self.assertEqual(policy.action_dim, 6)
        self.assertEqual(policy.control_dim, 3)
        self.assertTrue(hasattr(policy, "dmp_k_alpha"))
        self.assertTrue(hasattr(policy, "dmp_k_beta"))
        self.assertTrue(hasattr(policy, "dmp_tau"))


if __name__ == "__main__":
    unittest.main()
