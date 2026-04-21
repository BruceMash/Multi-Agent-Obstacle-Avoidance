"""
SAC 训练入口
"""

from pathlib import Path

from Controller.dmp_rl import DMPConfig
from Environment.single_agent_dmp_env import EnvConfig, SingleAgentDMPEnv
from baseline.sac import SAC


class _PolicyModeShim:
    def set_training_mode(self, mode: bool) -> None:
        return None


def build_env() -> SingleAgentDMPEnv:
    dynamics_config = {
        "velocity_clip": (-2.0, 2.0),
        "accelerate_clip": (-4.0, 4.0),
        "time_step": 0.1,
    }
    sensor_config = {"sensing_radius": 4.5}
    dmp_config = DMPConfig(dt=dynamics_config["time_step"], goal_offset_max=1.0)
    env_config = EnvConfig(max_steps=220, goal_tolerance=0.3)

    return SingleAgentDMPEnv(
        dynamics_config=dynamics_config,
        sensor_config=sensor_config,
        dmp_config=dmp_config,
        env_config=env_config,
    )


def train(total_timesteps: int = 2000, model_path: str = "artifacts/sac_model.zip") -> str:
    env = build_env()
    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        batch_size=256,
        learning_starts=100,
        train_freq=1,
        gradient_steps=1,
        verbose=1,
        policy_kwargs={
            "hidden_dim": 256,
            "sensor_output_dim": 128,
            "num_sensor_layers": 2,
            "num_observation_layers": 2,
        },
    )
    if model.policy is None:
        model.policy = _PolicyModeShim()

    model.learn(total_timesteps=total_timesteps)
    save_path = Path(model_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    env.close()
    return str(save_path)


if __name__ == "__main__":
    saved_file = train()
    print(f"训练完成，模型已保存到: {saved_file}")
