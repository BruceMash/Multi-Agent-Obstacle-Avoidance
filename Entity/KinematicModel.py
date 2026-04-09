# -*- coding: utf-8 -*-

import numpy as np


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

    def __init__(self, config): #初始化
        self.dt = config['timestep']
        self.min_accelerate = config['clip_accelerate'][0]
        self.max_accelerate = config['clip_accelerate'][1]

        self.g_vec = np.array([0.0, 0.0, -9.8], dtype=float)
        self.p = np.zeros(3, dtype=float)
        self.v = np.zeros(3, dtype=float)
        self.atti = np.zeros(3, dtype=float)
        self.omega = np.zeros(3, dtype=float)
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

    def integrate(self, action):
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
