import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALGO_ROOT = PROJECT_ROOT / "Multi-agent_Algo_lib"
SCRIPTS_ROOT = ALGO_ROOT / "scripts"
for path in (PROJECT_ROOT, ALGO_ROOT, SCRIPTS_ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from MASAC.Buffer import Buffer
from train_masac_multi_agent_dmp import (
    AgentObservationHistory,
    build_critic_action_matrix,
)


class TestTemporalReplayBoundaries(unittest.TestCase):
    def test_timeout_bootstraps_without_crossing_episode_boundary(self):
        buffer = Buffer(capacity=8, obs_dim=2, act_dim=1, device=torch.device("cpu"))
        buffer.add([1.0, 1.0], [0.1], 0.0, [2.0, 2.0], False, False)
        # timeout 不是 MDP terminal，因此 done=False；但它仍必须截断历史序列。
        buffer.add([2.0, 2.0], [0.2], 0.0, [3.0, 3.0], False, True)
        buffer.add([10.0, 10.0], [0.3], 0.0, [11.0, 11.0], False, False)

        obs, _, _, _, done, obs_mask, _ = buffer.sample(
            np.asarray([2]),
            sequence_length=3,
        )

        np.testing.assert_array_equal(obs_mask.cpu().numpy(), [[True, False, False]])
        np.testing.assert_allclose(obs.cpu().numpy()[0, 0], [10.0, 10.0])
        self.assertEqual(float(done.item()), 0.0)


class TestOnlineTemporalHistory(unittest.TestCase):
    def test_reset_removes_previous_episode_frames(self):
        history = AgentObservationHistory(temporal_steps=3, agent_ids=["agent_0"])
        history.reset({"agent_0": np.asarray([1.0, 2.0], dtype=np.float32)})
        history.append({"agent_0": np.asarray([3.0, 4.0], dtype=np.float32)})

        observations, masks = history.policy_inputs()
        np.testing.assert_allclose(
            observations["agent_0"],
            [[1.0, 2.0], [3.0, 4.0]],
        )
        np.testing.assert_array_equal(masks["agent_0"], [True, True])

        history.reset({"agent_0": np.asarray([9.0, 8.0], dtype=np.float32)})
        observations, masks = history.policy_inputs()
        np.testing.assert_allclose(observations["agent_0"], [[9.0, 8.0]])
        np.testing.assert_array_equal(masks["agent_0"], [True])


class TestCriticActionSemantics(unittest.TestCase):
    def test_replay_uses_raw_environment_action(self):
        raw_action = np.asarray(
            [[9.0, -8.0, 7.0, 0.3, -0.2, 0.1]],
            dtype=np.float32,
        )
        info = {
            "raw_action": raw_action,
            # 该字段故意与 raw action 不同，用于防止回归到旧的 acceleration 语义。
            "applied_accelerations": np.zeros((1, 3), dtype=np.float32),
            "guided_action": np.zeros((1, 6), dtype=np.float32),
        }
        env = SimpleNamespace(action_shape=(1, 6))

        critic_action = build_critic_action_matrix(info, env)

        np.testing.assert_array_equal(critic_action, raw_action)


if __name__ == "__main__":
    unittest.main()
