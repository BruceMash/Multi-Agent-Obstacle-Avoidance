import numpy as np

from Controller.dmp_rl import DMPConfig, HeuristicDMPPolicy
from Entity.dynamic_obstacles import MovingSphereObstacle
from Environment.single_agent_dmp_env import EnvConfig, SingleAgentDMPEnv


def build_demo_env():
    dynamics_config = {
        "velocity_clip": (-2.0, 2.0),
        "accelerate_clip": (-4.0, 4.0),
        "time_step": 0.1,
    }
    sensor_config = {"sensing_radius": 4.5}
    dmp_config = DMPConfig(dt=dynamics_config["time_step"], goal_offset_max=1.0)
    env_config = EnvConfig(max_steps=220, goal_tolerance=0.3)

    dynamic_obstacles = [
        MovingSphereObstacle(
            center=[3.5, -1.5, 0.0],
            radius=0.35,
            velocity=[0.0, 0.8, 0.0],
            safety_margin=0.15,
            bounds=(np.array([3.5, -2.0, -0.5]), np.array([3.5, 1.0, 0.5])),
        )
    ]
    return SingleAgentDMPEnv(
        dynamics_config=dynamics_config,
        sensor_config=sensor_config,
        dmp_config=dmp_config,
        env_config=env_config,
        dynamic_obstacles=dynamic_obstacles,
    )


def run_demo():
    env = build_demo_env()
    policy = HeuristicDMPPolicy(goal_offset_max=1.0, dims=env.dmp.config.dims)
    observation, _ = env.reset(
        options={
            "start": np.array([0.0, 0.0, 0.0]),
            "goal": np.array([8.0, 0.0, 0.0]),
        }
    )

    trajectory = [env.dynamics.p.copy()]
    total_reward = 0.0
    terminated = False
    truncated = False
    info = {}

    while not (terminated or truncated):
        action = policy.act(env.latest_sensor_packet)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        trajectory.append(env.dynamics.p.copy())

    print("demo finished")
    print(f"observation_dim: {observation.shape[0]}")
    print(f"steps: {env.steps}")
    print(f"total_reward: {total_reward:.3f}")
    print(f"success: {info['success']}")
    print(f"collision: {info['collision']}")
    print(f"truncated: {info['truncated']}")
    print(f"distance_to_goal: {info['distance_to_goal']:.3f}")
    print(f"trajectory_points: {len(trajectory)}")
    return np.asarray(trajectory, dtype=float), info


if __name__ == "__main__":
    run_demo()
