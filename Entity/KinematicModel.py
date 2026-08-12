# -*- coding: utf-8 -*-

import numpy as np


def propagate_point_mass(
    *,
    position,
    velocity,
    acceleration,
    dt,
    acceleration_min,
    acceleration_max,
    velocity_min,
    velocity_max,
):
    """按 ``PartialDynamic.step`` 的原始顺序执行无副作用质点传播。"""
    position = np.asarray(position, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    acceleration = np.asarray(acceleration, dtype=float)
    if position.shape != (3,) or velocity.shape != (3,) or acceleration.shape != (3,):
        raise ValueError("position, velocity and acceleration must have shape (3,)")
    if not (
        np.all(np.isfinite(position))
        and np.all(np.isfinite(velocity))
        and np.all(np.isfinite(acceleration))
    ):
        raise ValueError("point-mass transition inputs must be finite")
    dt = float(dt)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be a positive finite scalar")
    applied_acceleration = np.clip(acceleration, acceleration_min, acceleration_max)
    next_velocity_unclipped = velocity + applied_acceleration * dt
    next_velocity = np.clip(next_velocity_unclipped, velocity_min, velocity_max)
    next_position = position + velocity * dt + 0.5 * applied_acceleration * (dt ** 2)
    return {
        "position": next_position,
        "velocity": next_velocity,
        "state": np.concatenate([next_position, next_velocity]),
        "applied_acceleration": applied_acceleration,
        "unclipped_next_velocity": next_velocity_unclipped,
    }

'''
先用质点模型跑通实验，再看是不是可以往
'''


class UAVDynamics:
    '''
    state: [
        p: [x, y, z]
        v: [vx, vy, vz]
        atti: [roll, pitch, yaw]
        omega: [p, q, r]
    ]

    action: [a_x, a_y, a_z]
    '''

    def __init__(self, init_config):     # 初始化无人机类

        """
        初始化无人机类的构造函数
        参数:
            init_config: 包含初始化配置的字典，包括时间步长、加速度限制等
        """
        self.dt = init_config['timestep']    # 设置时间步长
        self.min_accelerate = init_config['clip_accelerate'][0]
        self.max_accelerate = init_config['clip_accelerate'][1]

        self.g_vec = np.array([0.0, 0.0, -9.8], dtype=float)    # 重力矩阵
        
        # 初始化状态量
        self.p = np.zeros(3, dtype=float)   
        self.v = np.zeros(3, dtype=float)
        self.atti = np.zeros(3, dtype=float)
        self.omega = np.zeros(3, dtype=float)

        # 构建状态矩阵
        self.state = np.concatenate([self.p, self.v, self.atti, self.omega])

    def reset(self, init_state=None):
        if init_state is None:
            self.p = np.zeros(3, dtype=float)
            self.v = np.zeros(3, dtype=float)
            self.atti = np.zeros(3, dtype=float)
            self.omega = np.zeros(3, dtype=float)
        else:
            init_state = np.asarray(init_state, dtype=float)
            if init_state.shape != (12,):
                raise ValueError('init_state must have shape (12,)')
            self.p = init_state[0:3].copy()
            self.v = init_state[3:6].copy()
            self.atti = init_state[6:9].copy()
            self.omega = init_state[9:12].copy()

        self.state = np.concatenate([self.p, self.v, self.atti, self.omega])
        return self.state.copy()

    def step(self, action):
        '''
        六自由度模型的单步实现

        '''
        action = np.asarray(action, dtype=float)
        if action.shape != (3,):
            raise ValueError('action must have shape (3,)')

        action = np.clip(action, self.min_accelerate, self.max_accelerate)
        next_state = self.integrate(action)

        self.p = next_state[0:3]
        self.v = next_state[3:6]
        self.atti = next_state[6:9]
        self.omega = next_state[9:12]
        self.state = next_state
        return self.state.copy()

    def integrate(self, action):    # 状态积分
        ax_cmd, ay_cmd, az_cmd = action
        gravity = abs(self.g_vec[2])
        phi, theta, psi = self.atti

        phi_ref = np.clip(-ay_cmd / gravity, -np.pi / 6.0, np.pi / 6.0)
        theta_ref = np.clip(ax_cmd / gravity, -np.pi / 6.0, np.pi / 6.0)
        tau = 0.3

        phi_dot = (phi_ref - phi) / tau
        theta_dot = (theta_ref - theta) / tau
        psi_dot = 0.0
        omega_next = np.array([phi_dot, theta_dot, psi_dot], dtype=float)
        atti_next = self.atti + omega_next * self.dt

        phi_n, theta_n, psi_n = atti_next
        c_phi, s_phi = np.cos(phi_n), np.sin(phi_n)
        c_theta, s_theta = np.cos(theta_n), np.sin(theta_n)
        c_psi, s_psi = np.cos(psi_n), np.sin(psi_n)

        rotation = np.array([
            [c_psi * c_theta, c_psi * s_theta * s_phi - s_psi * c_phi, c_psi * s_theta * c_phi + s_psi * s_phi],
            [s_psi * c_theta, s_psi * s_theta * s_phi + c_psi * c_phi, s_psi * s_theta * c_phi - c_psi * s_phi],
            [-s_theta, c_theta * s_phi, c_theta * c_phi],
        ], dtype=float)

        thrust = max(0.0, gravity + az_cmd)
        thrust_axis = rotation @ np.array([0.0, 0.0, 1.0], dtype=float)
        a_real = thrust * thrust_axis + self.g_vec

        v_next = self.v + a_real * self.dt
        p_next = self.p + self.v * self.dt + 0.5 * a_real * (self.dt ** 2)

        return np.concatenate([p_next, v_next, atti_next, omega_next])



class PartialDynamic:

    '''
    质点模型

    state: [px, py, pz, vx, vy, vz]
    action: [ax, ay, az]

    说明:
    1. 该类只负责基础平动动力学
    2. action 被视为净线加速度
    3. 传感器、避障和碰撞判定由其他模块负责
    '''

    def __init__(self, init_config):
        self.p = np.zeros(3, dtype=float)
        self.v = np.zeros(3, dtype=float)

        self.velocity_min = init_config['velocity_clip'][0]
        self.velocity_max = init_config['velocity_clip'][1]
        self.accelerate_min = init_config['accelerate_clip'][0]
        self.accelerate_max = init_config['accelerate_clip'][1]
        self.dt = init_config.get('time_step', init_config.get('timestep'))

        if self.dt is None:
            raise ValueError("init_config must contain 'time_step' or 'timestep'")

        self.state = self._compose_state()

    def _compose_state(self):
        return np.concatenate([self.p, self.v])

    @staticmethod
    def clip(value, floor, ceiling):    # 限制函数
        return np.clip(value, floor, ceiling)

    def step(self, action): # 步进
        transition = propagate_point_mass(
            position=self.p,
            velocity=self.v,
            acceleration=action,
            dt=self.dt,
            acceleration_min=self.accelerate_min,
            acceleration_max=self.accelerate_max,
            velocity_min=self.velocity_min,
            velocity_max=self.velocity_max,
        )
        self.v = transition["velocity"]
        self.p = transition["position"]
        self.state = self._compose_state()
        return self.state.copy()
    
    def reset(self, init_state=None):   # 重置
        if init_state is None:
            self.p = np.zeros(3, dtype=float)
            self.v = np.zeros(3, dtype=float)
        elif isinstance(init_state, dict):
            self.p = np.asarray(init_state['position'], dtype=float)
            self.v = np.asarray(init_state['velocity'], dtype=float)
        else:
            init_state = np.asarray(init_state, dtype=float)
            if init_state.shape != (6,):
                raise ValueError('init_state must have shape (6,)')
            self.p = init_state[0:3].copy()
            self.v = init_state[3:6].copy()

        if self.p.shape != (3,) or self.v.shape != (3,):
            raise ValueError('position and velocity must have shape (3,)')

        self.state = self._compose_state()
        return self.state.copy()

        

