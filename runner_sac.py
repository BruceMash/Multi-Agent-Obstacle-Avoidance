"""
SAC 训练入口（用于本项目的单智能体 DMP-RL 环境）。

这份 runner 负责三件事：
1. 构建训练环境和模型
2. 训练过程中按规则保存 checkpoint / best_model
3. 支持从 checkpoint 恢复继续训练
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from Controller.dmp_rl import DMPConfig

# 引入障碍物随机生成
from Entity.obstacle_generators import DynamicSpherePositionGenerate, StaticSpherePositionGenerate
from Entity.static_obstacles import AxisAlignedBoxObstacle

from Environment.single_agent_dmp_env import EnvConfig, SingleAgentDMPEnv
from baseline.common.callbacks import BaseCallback, CallbackList
from baseline.sac import SAC


class _PolicyModeShim:
    """
    兼容占位对象。

    当前自定义 SAC 实现里，`model.policy` 不是核心训练对象，
    但父类训练流程会调用 `set_training_mode()`，这里给一个最小实现避免中断。
    """

    def set_training_mode(self, mode: bool) -> None:
        _ = mode
        return None


def build_env() -> SingleAgentDMPEnv:
    """
    统一构建训练环境。

    这里集中放环境超参数，便于评审时快速确认实验配置。
    """
    dynamics_config = {
        "velocity_clip": (-2.0, 2.0),
        "accelerate_clip": (-4.0, 4.0),
        "time_step": 0.1,
    }   # 动态参数裁剪
    sensor_config = {"sensing_radius": 4.5} # 感知半径

    # dmp与环境测试
    dmp_config = DMPConfig(dt=dynamics_config["time_step"], goal_offset_max=1.0)
    env_config = EnvConfig(max_steps=220, goal_tolerance=0.3)
    # 固定立方体：作为稳定参考场景，每个回合都存在。
    # 随机障碍物生成时必须避让此立方体，避免一开始就重叠。
    fixed_box = AxisAlignedBoxObstacle(
        center=[4.8, -0.5, 0.0], half_extents=[0.45, 0.45, 0.35], safety_margin=0.1
    )

    def static_obstacle_generator(start, goal, seed):
        spheres = StaticSpherePositionGenerate(
            center=[[1.2, -1.4, -0.3], [7.0, 1.4, 0.3]],
            radius=0.45,
            safety_margin=0.1,
            num=3,
            existing_obstacles=[fixed_box],
            seed=seed,
            protected_points=[start, goal],
        )
        spheres.append(fixed_box)
        return spheres

    def dynamic_obstacle_generator(start, goal, seed, static_obstacles):
        return DynamicSpherePositionGenerate(
            center=[[1.5, -1.6, -0.3], [7.0, 1.2, 0.3]],
            radius=0.35,
            velocity=[[-0.3, -1.0, -0.2], [0.3, 1.0, 0.2]],
            safety_margin=0.15,
            num=1,
            movement_bounds=[[1.2, -2.0, -1.0], [7.3, 1.5, 1.0]],
            existing_obstacles=static_obstacles,
            seed=seed,
            protected_points=[start, goal],
            min_speed=0.4,
        )

    # env 构造阶段只放固定障碍物；随机球形障碍物在每次 reset 时按本回合起终点生成。
    static_obstacles = [fixed_box]

    return SingleAgentDMPEnv(
        dynamics_config=dynamics_config,
        sensor_config=sensor_config,
        dmp_config=dmp_config,
        env_config=env_config,
        static_obstacles=static_obstacles,
        static_obstacle_generator=static_obstacle_generator,
        dynamic_obstacles=[],
        dynamic_obstacle_generator=dynamic_obstacle_generator,
    )   # 创建环境


def build_model(
    env: SingleAgentDMPEnv,
    tensorboard_log: str | None = None,
    buffer_size: int = 1_000_000,
    verbose: int = 1,
) -> SAC:
    """
    创建 SAC 模型并注入网络结构参数。
    """
    model = SAC(    
        policy="MlpPolicy", # sb3基础实现的接口占位
        env=env,
        learning_rate=3e-4,
        buffer_size=buffer_size,
        batch_size=256,
        learning_starts=100,
        train_freq=1,
        gradient_steps=1,
        verbose=verbose,
        tensorboard_log=tensorboard_log,
        policy_kwargs={
            # 下面 4 个参数映射到 baseline/sac/net.py 的网络结构
            "hidden_dim": 256,
            "sensor_output_dim": 128,
            "num_sensor_layers": 2,
            "num_observation_layers": 2,
        },
    )

    # 父类训练循环会调用 model.policy.set_training_mode()，这里确保接口存在。
    if model.policy is None:    # 兼容旧版本
        model.policy = _PolicyModeShim()
    return model


def save_checkpoint(model: SAC, model_path: str | Path, extra: dict[str, Any] | None = None) -> str:
    """
    保存可恢复训练的完整状态。

    保存内容包括：
    1. actor / critic / critic_target 参数
    2. actor / critic 优化器状态
    3. 时间步与更新次数
    4. 熵系数相关状态（自动熵或固定熵两种模式）
    5. 额外业务字段（extra）
    """
    save_path = Path(model_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint: dict[str, Any] = {
        "actor": model.actor.state_dict(),
        "critic": model.critic.state_dict(),
        "critic_target": model.critic_target.state_dict(),
        "actor_optimizer": model.actor.optimizer.state_dict(),
        "critic_optimizer": model.critic.optimizer.state_dict(),
        "num_timesteps": int(model.num_timesteps),
        "n_updates": int(getattr(model, "_n_updates", 0)),
    }

    # 自动熵：保存 log_ent_coef + 对应优化器
    if model.ent_coef_optimizer is not None and model.log_ent_coef is not None:
        checkpoint["log_ent_coef"] = model.log_ent_coef.detach().cpu()
        checkpoint["ent_coef_optimizer"] = model.ent_coef_optimizer.state_dict()
    # 固定熵：保存 ent_coef_tensor
    elif hasattr(model, "ent_coef_tensor"):
        checkpoint["ent_coef_tensor"] = model.ent_coef_tensor.detach().cpu()

    if extra:
        checkpoint.update(extra)

    torch.save(checkpoint, str(save_path))
    return str(save_path)


def load_checkpoint(model: SAC, model_path: str | Path) -> SAC:
    """
    从 checkpoint 恢复模型和训练状态。
    """
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

    if "num_timesteps" in checkpoint:
        model.num_timesteps = int(checkpoint["num_timesteps"])
    if "n_updates" in checkpoint:
        model._n_updates = int(checkpoint["n_updates"])
    return model


class CheckpointAndBestCallback(BaseCallback):
    """
    训练过程保存策略：
    1. 周期性保存 checkpoint（固定步数）
    2. 额外维护一个最新 checkpoint（checkpoint.pt）
    3. 发现更高 episode reward 时保存 best_model
    """

    def __init__(self, run_dir: Path, save_freq: int = 500, verbose: int = 1):
        super().__init__(verbose=verbose)
        self.run_dir = run_dir
        self.ckpt_dir = run_dir / "checkpoints"
        self.save_freq = max(1, int(save_freq))
        self.best_model_path = run_dir / "best_model.pt"
        self.latest_checkpoint_path = run_dir / "checkpoint.pt"
        self.best_episode_reward = float("-inf")

    def _init_callback(self) -> None:
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _save_latest_checkpoint(self) -> None:
        save_checkpoint(
            model=self.model,
            model_path=self.latest_checkpoint_path,
            extra={"best_episode_reward": self.best_episode_reward},
        )

    def _save_step_checkpoint(self) -> None:
        step_ckpt = self.ckpt_dir / f"checkpoint_{self.num_timesteps:08d}.pt"
        save_checkpoint(
            model=self.model,
            model_path=step_ckpt,
            extra={"best_episode_reward": self.best_episode_reward},
        )

    def _maybe_save_best(self, infos: list[dict[str, Any]]) -> None:   
        """
        从 env 返回的 info 中读取 episode 统计，维护 best_model。
        """
        for info in infos:
            episode_info = info.get("episode")
            if episode_info is None:
                continue
            episode_reward = float(episode_info["r"])
            if episode_reward > self.best_episode_reward:
                self.best_episode_reward = episode_reward
                save_checkpoint(
                    model=self.model,
                    model_path=self.best_model_path,
                    extra={"best_episode_reward": self.best_episode_reward},
                )
                if self.verbose > 0:
                    print(
                        f"[best_model] timesteps={self.num_timesteps}, "
                        f"episode_reward={self.best_episode_reward:.3f}"
                    )

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            self._save_step_checkpoint()
            self._save_latest_checkpoint()

        infos = self.locals.get("infos", [])
        if isinstance(infos, list):
            self._maybe_save_best(infos)
        return True

    def _on_training_end(self) -> None:
        # 训练结束时再落一次最新 checkpoint，保证最后状态可恢复。
        self._save_latest_checkpoint()


class TensorboardRewardCallback(BaseCallback):
    """
    记录与奖励相关的 TensorBoard 指标。

    记录内容：
    1. step/reward_total: 每步奖励（多环境取均值）
    2. step/reward_step_reward: 步进奖励
    3. step/reward_obstacle_potential_penalty: 势场惩罚
    4. step/reward_step_penalty: 单步惩罚
    5. episode/reward: 每个回合总奖励（由 Monitor 提供）
    """

    def __init__(self, log_dir: Path):
        super().__init__(verbose=0)
        self.log_dir = log_dir
        self.writer: SummaryWriter | None = None

    def _init_callback(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

    def _on_step(self) -> bool:
        if self.writer is None:
            return True

        rewards = self.locals.get("rewards")
        if rewards is not None:
            reward_value = float(rewards.mean()) if hasattr(rewards, "mean") else float(rewards)
            self.writer.add_scalar("step/reward_total", reward_value, self.num_timesteps)

        infos = self.locals.get("infos", [])
        if isinstance(infos, list) and infos:
            step_reward_values: list[float] = []
            obstacle_penalty_values: list[float] = []
            step_penalty_values: list[float] = []
            episode_reward_values: list[float] = []

            for info in infos:
                if "reward_step_reward" in info:
                    step_reward_values.append(float(info["reward_step_reward"]))
                if "reward_obstacle_potential_penalty" in info:
                    obstacle_penalty_values.append(float(info["reward_obstacle_potential_penalty"]))
                if "reward_step_penalty" in info:
                    step_penalty_values.append(float(info["reward_step_penalty"]))
                episode_info = info.get("episode")
                if episode_info is not None and "r" in episode_info:
                    episode_reward_values.append(float(episode_info["r"]))

            if step_reward_values:
                self.writer.add_scalar(
                    "step/reward_step_reward",
                    sum(step_reward_values) / len(step_reward_values),
                    self.num_timesteps,
                )
            if obstacle_penalty_values:
                self.writer.add_scalar(
                    "step/reward_obstacle_potential_penalty",
                    sum(obstacle_penalty_values) / len(obstacle_penalty_values),
                    self.num_timesteps,
                )
            if step_penalty_values:
                self.writer.add_scalar(
                    "step/reward_step_penalty",
                    sum(step_penalty_values) / len(step_penalty_values),
                    self.num_timesteps,
                )
            for episode_reward in episode_reward_values:
                self.writer.add_scalar("episode/reward", episode_reward, self.num_timesteps)

        return True

    def _on_training_end(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()


def train(
    total_timesteps: int = 4000000,
    output_root: str = "artifacts",
    save_freq: int = 50000,
    resume_from: str | None = None,
) -> dict[str, str]:
    """
    执行训练并返回本次产物路径。

    输出目录为时间戳目录：output_root/YYYYMMDD_HHMMSS/
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / timestamp
    tensorboard_dir = run_dir / "tensorboard"

    env = build_env()
    model = build_model(env, tensorboard_log=str(tensorboard_dir))
    if resume_from:
        model = load_checkpoint(model, resume_from)

    checkpoint_callback = CheckpointAndBestCallback(run_dir=run_dir, save_freq=save_freq, verbose=1)
    tensorboard_callback = TensorboardRewardCallback(log_dir=tensorboard_dir)
    callback = CallbackList([checkpoint_callback, tensorboard_callback])

    model.learn(total_timesteps=total_timesteps, callback=callback)

    final_model_path = run_dir / "final_model.pt"
    save_checkpoint(
        model=model,
        model_path=final_model_path,
        extra={"best_episode_reward": checkpoint_callback.best_episode_reward},
    )
    env.close()

    return {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_callback.latest_checkpoint_path),
        "best_model": str(checkpoint_callback.best_model_path),
        "final_model": str(final_model_path),
        "tensorboard_dir": str(tensorboard_dir),
    }


if __name__ == "__main__":
    outputs = train()
    print("训练完成，已保存文件：")
    print(f"run_dir: {outputs['run_dir']}")
    print(f"checkpoint: {outputs['checkpoint']}")
    print(f"best_model: {outputs['best_model']}")
    print(f"final_model: {outputs['final_model']}")
    print(f"tensorboard_dir: {outputs['tensorboard_dir']}")
