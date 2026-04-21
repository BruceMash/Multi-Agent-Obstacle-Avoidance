"""
SAC 训练入口
"""

from pathlib import Path

import torch

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


def save_checkpoint(model: SAC, model_path: str) -> str:
    save_path = Path(model_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "actor": model.actor.state_dict(),
        "critic": model.critic.state_dict(),
        "critic_target": model.critic_target.state_dict(),
        "actor_optimizer": model.actor.optimizer.state_dict(),
        "critic_optimizer": model.critic.optimizer.state_dict(),
    }
    if model.ent_coef_optimizer is not None and model.log_ent_coef is not None:
        checkpoint["log_ent_coef"] = model.log_ent_coef.detach().cpu()
        checkpoint["ent_coef_optimizer"] = model.ent_coef_optimizer.state_dict()
    elif hasattr(model, "ent_coef_tensor"):
        checkpoint["ent_coef_tensor"] = model.ent_coef_tensor.detach().cpu()

    torch.save(checkpoint, str(save_path))
    return str(save_path)


def load_checkpoint(model: SAC, model_path: str) -> SAC:
    checkpoint = torch.load(model_path, map_location=model.device)
    model.actor.load_state_dict(checkpoint["actor"])
    model.critic.load_state_dict(checkpoint["critic"])
    model.critic_target.load_state_dict(checkpoint["critic_target"])
    model.actor.optimizer.load_state_dict(checkpoint["actor_optimizer"])
    model.critic.optimizer.load_state_dict(checkpoint["critic_optimizer"])

    if model.ent_coef_optimizer is not None and "ent_coef_optimizer" in checkpoint and "log_ent_coef" in checkpoint:
        model.log_ent_coef = checkpoint["log_ent_coef"].to(model.device).requires_grad_(True)
        model.ent_coef_optimizer = torch.optim.Adam([model.log_ent_coef], lr=model.lr_schedule(1))
        model.ent_coef_optimizer.load_state_dict(checkpoint["ent_coef_optimizer"])
    elif "ent_coef_tensor" in checkpoint:
        model.ent_coef_tensor = checkpoint["ent_coef_tensor"].to(model.device)

    return model


def train(total_timesteps: int = 2000, model_path: str = "artifacts/sac_model.pt") -> str:
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
    saved_path = save_checkpoint(model, model_path)
    env.close()
    return saved_path


if __name__ == "__main__":
    saved_file = train()
    print(f"训练完成，模型已保存到: {saved_file}")
