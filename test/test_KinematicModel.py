
import unittest
import numpy as np
from Entity.KinematicModel import PartialDynamic  # 请替换为实际的模块名

class TestPartialDynamic(unittest.TestCase):
    def setUp(self):
        """测试前的初始化设置"""
        self.init_config = {
            'velocity_clip': (-10.0, 10.0),
            'accelerate_clip': (-5.0, 5.0),
            'time_step': 0.1
        }
        self.model = PartialDynamic(self.init_config)

    def test_initialization(self):
        """测试初始化是否正确"""
        # 测试默认初始化
        np.testing.assert_array_equal(self.model.p, np.zeros(3))
        np.testing.assert_array_equal(self.model.v, np.zeros(3))
        self.assertEqual(self.model.dt, 0.1)

        # 测试缺少time_step的情况
        with self.assertRaises(ValueError):
            PartialDynamic({'velocity_clip': (-10, 10), 'accelerate_clip': (-5, 5)})

    def test_clip_function(self):
        """测试clip静态方法"""
        self.assertEqual(PartialDynamic.clip(3, 0, 5), 3)
        self.assertEqual(PartialDynamic.clip(-1, 0, 5), 0)
        self.assertEqual(PartialDynamic.clip(6, 0, 5), 5)

    def test_step_basic(self):
        """测试基本步进功能"""
        action = [1.0, 0.0, -1.0]
        initial_state = self.model.state.copy()
        next_state = self.model.step(action)

        # 验证状态更新
        expected_v = initial_state[3:6] + np.array(action) * self.init_config['time_step']
        expected_p = initial_state[0:3] + initial_state[3:6] * self.init_config['time_step'] + 0.5 * np.array(action) * (self.init_config['time_step'] ** 2)
        np.testing.assert_array_almost_equal(next_state[3:6], expected_v)
        np.testing.assert_array_almost_equal(next_state[0:3], expected_p)

    def test_step_clip(self):
        """测试步进时的裁剪功能"""
        # 测试加速度裁剪
        action = [10.0, -10.0, 0.0]  # 超出加速度限制
        self.model.step(action)
        clipped_action = np.clip(np.array(action), self.init_config['accelerate_clip'][0], self.init_config['accelerate_clip'][1])
        np.testing.assert_array_equal(self.model.v, np.clip(clipped_action * self.init_config['time_step'], 
                                                           self.init_config['velocity_clip'][0], 
                                                           self.init_config['velocity_clip'][1]))

        # 测试速度裁剪
        self.model.reset()
        action = [100.0, 0.0, 0.0]  # 会导致速度超出限制
        self.model.step(action)
        self.assertTrue(np.all(self.model.v <= self.init_config['velocity_clip'][1]))
        self.assertTrue(np.all(self.model.v >= self.init_config['velocity_clip'][0]))

    def test_step_invalid_action(self):
        """测试无效action"""
        with self.assertRaises(ValueError):
            self.model.step([1.0, 2.0])  # 形状不正确

    def test_reset_default(self):
        """测试默认重置"""
        self.model.p = np.array([1.0, 2.0, 3.0])
        self.model.v = np.array([0.5, 0.5, 0.5])
        self.model.reset()
        np.testing.assert_array_equal(self.model.p, np.zeros(3))
        np.testing.assert_array_equal(self.model.v, np.zeros(3))

    def test_reset_with_dict(self):
        """测试使用字典重置"""
        init_state = {
            'position': [1.0, 2.0, 3.0],
            'velocity': [0.5, 0.5, 0.5]
        }
        self.model.reset(init_state)
        np.testing.assert_array_equal(self.model.p, np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(self.model.v, np.array([0.5, 0.5, 0.5]))

    def test_reset_with_array(self):
        """测试使用数组重置"""
        init_state = np.array([1.0, 2.0, 3.0, 0.5, 0.5, 0.5])
        self.model.reset(init_state)
        np.testing.assert_array_equal(self.model.p, np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(self.model.v, np.array([0.5, 0.5, 0.5]))

    def test_reset_invalid(self):
        """测试无效重置"""
        # 形状不正确的数组
        with self.assertRaises(ValueError):
            self.model.reset(np.array([1.0, 2.0, 3.0]))

        # 形状不正确的字典
        with self.assertRaises(ValueError):
            self.model.reset({'position': [1.0, 2.0], 'velocity': [0.5, 0.5]})

    def test_compose_state(self):
        """测试状态组合"""
        self.model.p = np.array([1.0, 2.0, 3.0])
        self.model.v = np.array([0.5, 0.5, 0.5])
        state = self.model._compose_state()
        np.testing.assert_array_equal(state, np.concatenate([self.model.p, self.model.v]))

    def test_multiple_steps(self):
        """测试多步进"""
        actions = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]
        for action in actions:
            self.model.step(action)
        
        # 验证最终状态
        self.assertTrue(np.all(self.model.v > 0))
        self.assertTrue(np.all(self.model.p > 0))

if __name__ == '__main__':
    unittest.main()
