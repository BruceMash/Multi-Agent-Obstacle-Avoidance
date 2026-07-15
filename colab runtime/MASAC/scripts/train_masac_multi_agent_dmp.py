# coding: utf-8

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import deque
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


SCRIPT_PATH = Path(__file__).resolve()
ALGO_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]
for path in (PROJECT_ROOT, ALGO_ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from Environment.multi_agent_dmp_env import MultiAgentDMPEnv
from MASAC.MASAC import MASAC
from MASAC.config import MASAC_EXPERIMENT_CONFIG, MASACExperimentConfig, MASACNetworkConfig # 读取配置信息


def matrix_to_agent_dict(values: np.ndarray, agent_ids: list[str]) -> dict[str, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    return {
        agent_id: values[index].astype(np.float32, copy=True)
        for index, agent_id in enumerate(agent_ids)
    }


def vector_to_agent_dict(values: np.ndarray, agent_ids: list[str]) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float32)
    return {
        agent_id: float(values[index])
        for index, agent_id in enumerate(agent_ids)
    }


def agent_dict_to_matrix(values: dict[str, np.ndarray], agent_ids: list[str]) -> np.ndarray:
    return np.stack(
        [np.asarray(values[agent_id], dtype=np.float32) for agent_id in agent_ids],
        axis=0,
    )


def state_bound_tuple(value, state_dim: int) -> tuple[float, ...]:
    values = np.asarray(value, dtype=np.float32)
    if values.ndim == 0:
        values = np.full(state_dim, float(values), dtype=np.float32)
    values = values.reshape(-1)
    if values.shape != (state_dim,):
        raise ValueError(f"bound must have shape ({state_dim},), got {tuple(values.shape)}")
    return tuple(float(item) for item in values)


def build_critic_action_matrix(info: dict, env: MultiAgentDMPEnv) -> np.ndarray:
    applied_accelerations = np.asarray(
        info["applied_accelerations"],
        dtype=np.float32,
    )
    guided_action = np.asarray(info["guided_action"], dtype=np.float32)
    dmp_dims = int(env.dmp_config.dims)
    goal_offset = guided_action[:, dmp_dims : 2 * dmp_dims]
    critic_action = np.concatenate([applied_accelerations, goal_offset], axis=1)
    expected_shape = env.action_shape
    if critic_action.shape != expected_shape:
        raise ValueError(
            f"critic action must have shape {expected_shape}, "
            f"got {critic_action.shape}"
        )
    return critic_action.astype(np.float32, copy=False)


def build_env(config: MASACExperimentConfig) -> MultiAgentDMPEnv:
    return MultiAgentDMPEnv(**config.build_core_env_kwargs())


def build_dim_info(env: MultiAgentDMPEnv, agent_ids: list[str]) -> dict[str, tuple[int, int]]:
    _, obs_dim = env.observation_shape
    _, action_dim = env.action_shape
    return {
        agent_id: (int(obs_dim), int(action_dim))
        for agent_id in agent_ids
    }


def build_network_config(
    env: MultiAgentDMPEnv,
    experiment_config: MASACExperimentConfig,
    args: argparse.Namespace,
) -> MASACNetworkConfig:
    sensor = env.sensors[0]
    state_dim = int(env.state_dim)
    action_low = tuple(float(value) for value in env.action_space.low[0])
    action_high = tuple(float(value) for value in env.action_space.high[0])
    acceleration_low = state_bound_tuple(env.dynamics[0].accelerate_min, state_dim)
    acceleration_high = state_bound_tuple(env.dynamics[0].accelerate_max, state_dim)
    ally_feature_dim = (
        int(env.single_pair_observation_dim)
        if int(env.nearest_agent_observation_count) > 0
        else 0
    )
    return MASACNetworkConfig(
        sensor_observation_dim=int(env.sensor_observation_dim),
        extra_observation_dim=int(env.extra_observation_dim),
        ally_feature_dim=ally_feature_dim,
        sensor_output_dim=int(args.sensor_output_dim),
        ally_output_dim=int(args.ally_output_dim),
        sensor_hidden_dim=int(args.sensor_hidden_dim),
        ally_hidden_dim=int(args.ally_hidden_dim),
        hidden_dim=int(args.hidden_dim),
        num_sensor_layers=int(args.num_sensor_layers),
        num_ally_layers=int(args.num_ally_layers),
        num_observation_layers=int(args.num_observation_layers),
        sensor_azimuth_bins=int(sensor.azimuth_bins),
        sensor_elevation_bins=int(sensor.elevation_bins),
        sensor_elevation_range_deg=tuple(float(v) for v in sensor.elevation_range_deg),
        action_low=action_low,
        action_high=action_high,
        temporal_steps=int(args.temporal_steps),
        actor_log_std_min=float(args.actor_log_std_min),
        actor_log_std_max=float(args.actor_log_std_max),
        ally_pooling=str(experiment_config.ally_pooling),
        agent_pooling=str(experiment_config.agent_pooling),
        goal_distance_clip=float(sensor.goal_distance_clip),
        dmp_k_alpha=float(env.dmp_config.K_alpha),
        dmp_k_beta=float(env.dmp_config.K_beta),
        dmp_tau=float(env.dmp_config.tau),
        forcing_term_min=float(env.dmp_config.forcing_term_min),
        forcing_term_max=float(env.dmp_config.forcing_term_max),
        acceleration_low=acceleration_low,
        acceleration_high=acceleration_high,
    )


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return requested


def make_run_dir(output_root: str, seed: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / f"masac_multi_agent_dmp_seed_{seed}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False, default=str)


def append_metrics(path: Path, row: dict[str, Any]) -> None:
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def mask_count(info: dict[str, Any], key: str) -> int:
    return int(np.sum(np.asarray(info.get(key, []), dtype=bool)))


class TerminalProgress:
    """Render training progress with tqdm, with a plain text fallback."""

    def __init__(self, *, total_steps: int, success_window: int, plain: bool = False):
        self.total_steps = max(1, int(total_steps))
        self.success_window = max(1, int(success_window))
        self.plain = bool(plain) or tqdm is None
        self.last_step = 0
        self.last_status: dict[str, Any] | None = None
        self.pbar = None
        if not self.plain:
            self.pbar = tqdm(
                total=self.total_steps,
                desc="MASAC",
                unit="step",
                dynamic_ncols=True,
                leave=True,
            )

    @staticmethod
    def _format_elapsed(seconds: float) -> str:
        seconds = max(0, int(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _plain_line(self, status: dict[str, Any]) -> str:
        return (
            f"MASAC | step {int(status['global_step']):,}/{self.total_steps:,} "
            f"| ep {int(status['episode_count'])} "
            f"| buf {int(status['buffer_size']):,} "
            f"| SR{self.success_window} {float(status['rolling_success_rate']):.3f} "
            f"| IA {int(status['inter_agent_collision_count'])} "
            f"| OBS {int(status['obstacle_collision_count'])} "
            f"| {self._format_elapsed(float(status['elapsed_sec']))}"
        )

    def update(self, **status: Any) -> None:
        self.last_status = dict(status)
        current_step = min(max(int(status["global_step"]), 0), self.total_steps)
        if self.pbar is None:
            print(self._plain_line(self.last_status), flush=True)
            self.last_step = current_step
            return

        delta = current_step - self.last_step
        if delta > 0:
            self.pbar.update(delta)
            self.last_step = current_step
        self.pbar.set_postfix(
            {
                "ep": int(status["episode_count"]),
                "buf": f"{int(status['buffer_size']):,}",
                f"SR{self.success_window}": f"{float(status['rolling_success_rate']):.3f}",
                "IA": int(status["inter_agent_collision_count"]),
                "OBS": int(status["obstacle_collision_count"]),
                "elapsed": self._format_elapsed(float(status["elapsed_sec"])),
            },
            refresh=True,
        )

    def event(self, message: str) -> None:
        if self.pbar is None:
            print(message, flush=True)
            return
        tqdm.write(message)

    def close(self) -> None:
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None


class ForwardProfiler:
    """Collect inclusive forward-pass timings for selected MASAC network modules."""

    TARGET_CLASS_NAMES = {
        "MASACActor",
        "MASACCritic",
        "MASACObservationEncoder",
        "ObservationEncoder",
        "AllyObservationEncoder",
        "_CentralizedQBranch",
        "MultiheadAttention",
        "GRU",
    }

    def __init__(
        self,
        policy: MASAC,
        *,
        device: torch.device,
        output_path: Path,
        topk: int = 20,
    ) -> None:
        self.device = device
        self.output_path = output_path
        self.topk = max(1, int(topk))
        self.sync_cuda = bool(device.type == "cuda" and torch.cuda.is_available())
        self.handles = []
        self.stats: dict[str, dict[str, Any]] = {}
        self._register(policy)

    def _maybe_sync(self) -> None:
        if self.sync_cuda:
            torch.cuda.synchronize(self.device)

    def _should_track(self, module: torch.nn.Module) -> bool:
        return module.__class__.__name__ in self.TARGET_CLASS_NAMES

    def _register_module(self, label: str, module: torch.nn.Module) -> None:
        if not self._should_track(module):
            return

        class_name = module.__class__.__name__
        key = f"{label}:{class_name}"
        self.stats.setdefault(
            key,
            {
                "module": label,
                "class_name": class_name,
                "calls": 0,
                "total_ms": 0.0,
                "max_ms": 0.0,
            },
        )

        def pre_hook(tracked_module, _inputs):
            self._maybe_sync()
            tracked_module.__masac_profile_start = time.perf_counter()

        def post_hook(tracked_module, _inputs, _output):
            self._maybe_sync()
            start = getattr(tracked_module, "__masac_profile_start", None)
            if start is None:
                return
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            row = self.stats[key]
            row["calls"] += 1
            row["total_ms"] += elapsed_ms
            row["max_ms"] = max(float(row["max_ms"]), elapsed_ms)

        self.handles.append(module.register_forward_pre_hook(pre_hook))
        self.handles.append(module.register_forward_hook(post_hook))

    def _register_network(self, prefix: str, network: torch.nn.Module) -> None:
        for name, module in network.named_modules():
            label = prefix if name == "" else f"{prefix}.{name}"
            self._register_module(label, module)

    def _register(self, policy: MASAC) -> None:
        for agent_id, agent in policy.agents.items():
            self._register_network(f"{agent_id}.actor", agent.actor)
            self._register_network(f"{agent_id}.actor_target", agent.actor_target)
            self._register_network(f"{agent_id}.critic", agent.critic)
            self._register_network(f"{agent_id}.critic_target", agent.critic_target)

    def rows(self, global_step: int) -> list[dict[str, Any]]:
        rows = []
        for key, stat in self.stats.items():
            calls = int(stat["calls"])
            if calls <= 0:
                continue
            total_ms = float(stat["total_ms"])
            rows.append(
                {
                    "global_step": int(global_step),
                    "module": stat["module"],
                    "class_name": stat["class_name"],
                    "calls": calls,
                    "total_ms": total_ms,
                    "mean_ms": total_ms / calls,
                    "max_ms": float(stat["max_ms"]),
                }
            )
        rows.sort(key=lambda row: row["total_ms"], reverse=True)
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
        return rows

    def reset(self) -> None:
        for stat in self.stats.values():
            stat["calls"] = 0
            stat["total_ms"] = 0.0
            stat["max_ms"] = 0.0

    def write_snapshot(self, global_step: int, *, reset: bool = True) -> list[dict[str, Any]]:
        rows = self.rows(global_step)
        for row in rows:
            append_metrics(self.output_path, row)
        if reset:
            self.reset()
        return rows

    def format_summary(self, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "forward_profile: no tracked forward calls yet"
        lines = [f"forward_profile top {min(self.topk, len(rows))} modules:"]
        for row in rows[: self.topk]:
            lines.append(
                (
                    "#{rank} {module} [{class_name}] "
                    "calls={calls} total={total_ms:.2f}ms "
                    "mean={mean_ms:.3f}ms max={max_ms:.3f}ms"
                ).format(**row)
            )
        return "\n".join(lines)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def write_tensorboard_episode(writer, row: dict[str, Any]) -> None:
    if writer is None:
        return
    step = int(row["global_step"])
    writer.add_scalar("episode/mean_reward", row["mean_reward"], step)
    writer.add_scalar("episode/sum_reward", row["sum_reward"], step)
    writer.add_scalar("episode/length", row["episode_step"], step)
    writer.add_scalar("success/full_success", float(row["success"]), step)
    writer.add_scalar("success/agent_success_rate", row["agent_success_rate"], step)
    writer.add_scalar("success/rolling_success_rate", row["rolling_success_rate"], step)
    writer.add_scalar("collision/inter_agent_event", float(row["inter_agent_collision"]), step)
    writer.add_scalar("collision/obstacle_event", float(row["obstacle_collision"]), step)
    writer.add_scalar("collision/boundary_event", float(row["boundary_collision"]), step)
    writer.add_scalar("collision/inter_agent_steps", row["inter_agent_collision_steps"], step)
    writer.add_scalar("collision/obstacle_steps", row["obstacle_collision_steps"], step)
    writer.add_scalar("collision/boundary_steps", row["boundary_collision_steps"], step)
    writer.add_scalar("collision/inter_agent_agent_count", row["inter_agent_collision_agent_count"], step)
    writer.add_scalar("collision/obstacle_agent_count", row["obstacle_collision_agent_count"], step)
    writer.add_scalar("collision/boundary_agent_count", row["boundary_collision_agent_count"], step)
    writer.add_scalar("safety/min_inter_agent_distance", row["min_inter_agent_distance"], step)


def write_tensorboard_step(
    writer,
    *,
    global_step: int,
    buffer_size: int,
    episode_count: int,
    rolling_success_rate: float,
    inter_agent_collision_count: int,
    obstacle_collision_count: int,
    min_inter_agent_distance: float,
) -> None:
    if writer is None:
        return
    writer.add_scalar("train/global_step", global_step, global_step)
    writer.add_scalar("train/buffer_size", buffer_size, global_step)
    writer.add_scalar("train/episode_count", episode_count, global_step)
    writer.add_scalar("success/rolling_success_rate_step", rolling_success_rate, global_step)
    writer.add_scalar("collision/inter_agent_agent_count_step", inter_agent_collision_count, global_step)
    writer.add_scalar("collision/obstacle_agent_count_step", obstacle_collision_count, global_step)
    writer.add_scalar("safety/min_inter_agent_distance_step", min_inter_agent_distance, global_step)


def parse_args() -> argparse.Namespace:
    config = MASAC_EXPERIMENT_CONFIG
    parser = argparse.ArgumentParser(
        description="Train MASAC on the matrix-style MultiAgentDMPEnv."
    )
    parser.add_argument("--seed", type=int, default=int(config.seed))
    # 算法相关
    parser.add_argument("--total-steps", type=int, default=int(config.total_steps))
    parser.add_argument("--start-steps", type=int, default=int(config.start_steps))
    parser.add_argument("--batch-size", type=int, default=int(config.batch_size))
    parser.add_argument("--buffer-size", type=int, default=int(config.buffer_size))
    parser.add_argument("--actor-lr", type=float, default=float(config.actor_lr))
    parser.add_argument("--critic-lr", type=float, default=float(config.critic_lr))
    parser.add_argument("--gamma", type=float, default=float(config.gamma))
    parser.add_argument("--tau", type=float, default=float(config.soft_update_tau))
    parser.add_argument("--learn-interval", type=int, default=int(config.learn_interval))
    parser.add_argument("--updates-per-step", type=int, default=int(config.updates_per_step))
    parser.add_argument("--temporal-steps", type=int, default=int(config.temporal_steps))

    # 网络结构线管
    parser.add_argument("--hidden-dim", type=int, default=int(config.hidden_dim))
    parser.add_argument("--sensor-hidden-dim", type=int, default=int(config.sensor_hidden_dim))
    parser.add_argument("--ally-hidden-dim", type=int, default=int(config.ally_hidden_dim))
    parser.add_argument("--sensor-output-dim", type=int, default=int(config.sensor_output_dim))
    parser.add_argument("--ally-output-dim", type=int, default=int(config.ally_output_dim))
    parser.add_argument("--num-sensor-layers", type=int, default=int(config.num_sensor_layers))
    parser.add_argument("--num-ally-layers", type=int, default=int(config.num_ally_layers))
    parser.add_argument("--num-observation-layers", type=int, default=int(config.num_observation_layers))
    parser.add_argument("--actor-log-std-min", type=float, default=float(config.actor_log_std_min))
    parser.add_argument("--actor-log-std-max", type=float, default=float(config.actor_log_std_max))

    # Runtime and logging options
    parser.add_argument("--device", type=str, default=str(config.device))
    parser.add_argument("--output-root", type=str, default=str(config.output_root))
    parser.add_argument("--save-interval", type=int, default=int(config.save_interval))
    parser.add_argument("--log-interval", type=int, default=int(config.log_interval))
    parser.add_argument("--progress-interval", type=int, default=int(config.progress_interval))
    parser.add_argument("--success-window", type=int, default=int(config.success_window))
    parser.add_argument("--disable-tensorboard", action="store_true", default=bool(config.disable_tensorboard))
    parser.add_argument("--plain-progress", action="store_true", default=bool(config.plain_progress))
    parser.add_argument("--profile-forward", action="store_true", default=False)
    parser.add_argument("--profile-forward-interval", type=int, default=1_000)
    parser.add_argument("--profile-forward-topk", type=int, default=20)
    return parser.parse_args()


def train() -> dict[str, str]:
    args = parse_args() # 读args
    args.progress_interval = max(1, int(args.progress_interval))

    # 训练间隔
    args.log_interval = max(1, int(args.log_interval))
    args.success_window = max(1, int(args.success_window))  # 计算滚动成功率均值
    args.profile_forward_interval = max(1, int(args.profile_forward_interval))
    args.profile_forward_topk = max(1, int(args.profile_forward_topk))
    
    # 固定随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    experiment_config = MASAC_EXPERIMENT_CONFIG
    env = build_env(experiment_config)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(args.seed)

    agent_ids = [f"agent_{index}" for index in range(int(env.num_agents))]
    dim_info = build_dim_info(env, agent_ids)
    network_config = build_network_config(env, experiment_config, args)
    device = resolve_device(args.device)

    policy = MASAC(
        dim_info=dim_info,
        is_continue=True,
        actor_lr=float(args.actor_lr),
        critic_lr=float(args.critic_lr),
        buffer_size=int(args.buffer_size),
        device=device,
        network_config=network_config,
    )

    run_dir = make_run_dir(args.output_root, args.seed)
    model_dir = run_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    forward_profile_path = run_dir / "forward_profile.csv"
    progress = TerminalProgress(
        total_steps=int(args.total_steps),
        success_window=int(args.success_window),
        plain=bool(args.plain_progress),
    )
    writer = None
    if not bool(args.disable_tensorboard) and SummaryWriter is not None:
        writer = SummaryWriter(str(run_dir / "tensorboard"))
        progress.event(f"TensorBoard logging enabled: {run_dir / 'tensorboard'}")
    elif bool(args.disable_tensorboard):
        progress.event("TensorBoard logging disabled by command line argument.")
    else:
        progress.event("TensorBoard logging is unavailable because tensorboard is not installed.")

    forward_profiler = None
    if bool(args.profile_forward):
        forward_profiler = ForwardProfiler(
            policy,
            device=device,
            output_path=forward_profile_path,
            topk=int(args.profile_forward_topk),
        )
        progress.event(
            (
                "Forward profiling enabled: "
                f"{forward_profile_path} "
                f"(interval={int(args.profile_forward_interval)}, "
                f"topk={int(args.profile_forward_topk)}, "
                f"cuda_sync={forward_profiler.sync_cuda})"
            )
        )

    write_json(
        run_dir / "config.json",
        {
            "script_args": vars(args),
            "experiment_config": asdict(experiment_config),
            "core_env_kwargs": experiment_config.build_core_env_kwargs(),
            "dim_info": dim_info,
            "network_config": asdict(network_config),
            "device": str(device),
            "tensorboard_enabled": writer is not None,
        },
    )

    obs_matrix, _ = env.reset(seed=args.seed)
    obs = matrix_to_agent_dict(obs_matrix, agent_ids)
    episode_reward = np.zeros(int(env.num_agents), dtype=np.float64)
    episode_count = 0
    episode_step = 0
    start_time = time.time()
    success_history = deque(maxlen=max(1, int(args.success_window)))
    episode_inter_agent_collision_steps = 0
    episode_obstacle_collision_steps = 0
    episode_boundary_collision_steps = 0
    episode_inter_agent_collision_agent_count = 0
    episode_obstacle_collision_agent_count = 0
    episode_boundary_collision_agent_count = 0
    latest_inter_agent_collision_count = 0
    latest_obstacle_collision_count = 0
    latest_min_inter_agent_distance = float("inf")

    min_learn_size = max(int(args.batch_size), int(args.temporal_steps))
    for global_step in range(1, int(args.total_steps) + 1):
        if global_step <= int(args.start_steps):
            action_matrix = env.action_space.sample().astype(np.float32)
            action = matrix_to_agent_dict(action_matrix, agent_ids)
        else:
            action = policy.select_action(obs)
            action_matrix = agent_dict_to_matrix(action, agent_ids)
            action_matrix = np.clip(
                action_matrix,
                env.action_space.low,
                env.action_space.high,
            ).astype(np.float32)
            action = matrix_to_agent_dict(action_matrix, agent_ids)

        next_obs_matrix, rewards, terminated, truncated, info = env.step(action_matrix)
        latest_inter_agent_collision_count = mask_count(info, "inter_agent_collision_mask")
        latest_obstacle_collision_count = mask_count(info, "obstacle_collision_mask")
        latest_boundary_collision_count = mask_count(info, "boundary_collision_mask")
        latest_min_inter_agent_distance = float(info.get("min_inter_agent_distance", np.inf))
        if latest_inter_agent_collision_count > 0:
            episode_inter_agent_collision_steps += 1
            episode_inter_agent_collision_agent_count += latest_inter_agent_collision_count
        if latest_obstacle_collision_count > 0:
            episode_obstacle_collision_steps += 1
            episode_obstacle_collision_agent_count += latest_obstacle_collision_count
        if latest_boundary_collision_count > 0:
            episode_boundary_collision_steps += 1
            episode_boundary_collision_agent_count += latest_boundary_collision_count

        next_obs = matrix_to_agent_dict(next_obs_matrix, agent_ids)
        reward = vector_to_agent_dict(rewards, agent_ids)
        critic_action_matrix = build_critic_action_matrix(info, env)
        critic_action = matrix_to_agent_dict(critic_action_matrix, agent_ids)
        done_for_buffer = {
            agent_id: bool(terminated)
            for agent_id in agent_ids
        }
        policy.add(obs, critic_action, reward, next_obs, done_for_buffer)

        episode_reward += np.asarray(rewards, dtype=np.float64)
        episode_step += 1
        obs = next_obs

        can_learn = len(policy.buffers[policy.agent_x]) >= min_learn_size
        if (
            can_learn
            and global_step > int(args.start_steps)
            and global_step % int(args.learn_interval) == 0
        ):
            for _ in range(int(args.updates_per_step)):
                policy.learn(
                    batch_size=int(args.batch_size),
                    gamma=float(args.gamma),
                    tau=float(args.tau),
                )

        episode_done = bool(terminated or truncated)
        if episode_done:
            episode_count += 1
            success_mask = np.asarray(
                info.get("success_mask", np.zeros(int(env.num_agents), dtype=bool)),
                dtype=bool,
            )
            success = bool(info.get("success", False))
            success_history.append(float(success))
            rolling_success_rate = float(np.mean(success_history)) if success_history else 0.0
            reached_agent_count = int(np.sum(success_mask))
            agent_success_rate = float(reached_agent_count / float(env.num_agents))
            inter_agent_collision_event = episode_inter_agent_collision_steps > 0
            obstacle_collision_event = episode_obstacle_collision_steps > 0
            boundary_collision_event = episode_boundary_collision_steps > 0
            row = {
                "global_step": global_step,
                "episode": episode_count,
                "episode_step": episode_step,
                "mean_reward": float(np.mean(episode_reward)),
                "sum_reward": float(np.sum(episode_reward)),
                "min_reward": float(np.min(episode_reward)),
                "max_reward": float(np.max(episode_reward)),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "collision": bool(info.get("collision", False)),
                "success": success,
                "reached_agent_count": reached_agent_count,
                "agent_success_rate": agent_success_rate,
                "rolling_success_rate": rolling_success_rate,
                "inter_agent_collision": inter_agent_collision_event,
                "obstacle_collision": obstacle_collision_event,
                "boundary_collision": boundary_collision_event,
                "inter_agent_collision_steps": episode_inter_agent_collision_steps,
                "obstacle_collision_steps": episode_obstacle_collision_steps,
                "boundary_collision_steps": episode_boundary_collision_steps,
                "inter_agent_collision_agent_count": episode_inter_agent_collision_agent_count,
                "obstacle_collision_agent_count": episode_obstacle_collision_agent_count,
                "boundary_collision_agent_count": episode_boundary_collision_agent_count,
                "min_inter_agent_distance": float(info.get("min_inter_agent_distance", np.inf)),
                "elapsed_sec": float(time.time() - start_time),
            }
            append_metrics(metrics_path, row)
            write_tensorboard_episode(writer, row)
            progress.event(
                (
                    "episode={episode} step={global_step} len={episode_step} "
                    "reward={mean_reward:.3f} success={success} "
                    "reached={reached_agent_count}/{num_agents} "
                    "SR{window}={rolling_success_rate:.3f} "
                    "IA_collision={inter_agent_collision} "
                    "OBS_collision={obstacle_collision} "
                    "BD_collision={boundary_collision}"
                ).format(
                    num_agents=int(env.num_agents),
                    window=int(args.success_window),
                    **row,
                )
            )

            obs_matrix, _ = env.reset()
            obs = matrix_to_agent_dict(obs_matrix, agent_ids)
            episode_reward[:] = 0.0
            episode_step = 0
            episode_inter_agent_collision_steps = 0
            episode_obstacle_collision_steps = 0
            episode_boundary_collision_steps = 0
            episode_inter_agent_collision_agent_count = 0
            episode_obstacle_collision_agent_count = 0
            episode_boundary_collision_agent_count = 0

        rolling_success_rate = float(np.mean(success_history)) if success_history else 0.0
        if global_step % int(args.progress_interval) == 0 or global_step == 1:
            progress.update(
                global_step=global_step,
                episode_count=episode_count,
                buffer_size=len(policy.buffers[policy.agent_x]),
                rolling_success_rate=rolling_success_rate,
                inter_agent_collision_count=latest_inter_agent_collision_count,
                obstacle_collision_count=latest_obstacle_collision_count,
                elapsed_sec=float(time.time() - start_time),
            )

        if global_step % int(args.log_interval) == 0:
            progress.event(
                (
                    f"step={global_step} buffer={len(policy.buffers[policy.agent_x])} "
                    f"episodes={episode_count} SR{int(args.success_window)}={rolling_success_rate:.3f} "
                    f"IA={latest_inter_agent_collision_count} OBS={latest_obstacle_collision_count} "
                    f"elapsed={time.time() - start_time:.1f}s"
                )
            )
            write_tensorboard_step(
                writer,
                global_step=global_step,
                buffer_size=len(policy.buffers[policy.agent_x]),
                episode_count=episode_count,
                rolling_success_rate=rolling_success_rate,
                inter_agent_collision_count=latest_inter_agent_collision_count,
                obstacle_collision_count=latest_obstacle_collision_count,
                min_inter_agent_distance=latest_min_inter_agent_distance,
            )

        if (
            forward_profiler is not None
            and global_step % int(args.profile_forward_interval) == 0
        ):
            rows = forward_profiler.write_snapshot(global_step)
            progress.event(forward_profiler.format_summary(rows))

        if global_step % int(args.save_interval) == 0:
            checkpoint_dir = model_dir / f"step_{global_step}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            policy.save(checkpoint_dir)
            progress.event(f"checkpoint saved: {checkpoint_dir}")

    final_dir = model_dir / "final"
    if forward_profiler is not None:
        rows = forward_profiler.write_snapshot(int(args.total_steps), reset=False)
        if rows:
            progress.event(forward_profiler.format_summary(rows))
        forward_profiler.close()
    final_dir.mkdir(parents=True, exist_ok=True)
    policy.save(final_dir)
    progress.event(f"final model saved: {final_dir}")
    env.close()
    if writer is not None:
        writer.close()
    progress.close()
    return {
        "run_dir": str(run_dir),
        "metrics": str(metrics_path),
        "model_dir": str(final_dir),
    }


def main() -> None:
    outputs = train()
    print("MASAC training finished.")
    print(f"run_dir: {outputs['run_dir']}")
    print(f"metrics: {outputs['metrics']}")
    print(f"model_dir: {outputs['model_dir']}")


if __name__ == "__main__":
    main()
