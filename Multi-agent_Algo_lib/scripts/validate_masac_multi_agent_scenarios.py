"""Validate MASAC in bounded custom scenarios or the training distribution.

The script provides both live browser visualization and reproducible rollout
artifacts.  It intentionally uses only Python's standard-library HTTP server
and a dependency-free Canvas frontend so that the generated HTML remains
usable without a CDN.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
import threading
import time
import webbrowser
from dataclasses import asdict, fields
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
for search_path in (REPO_ROOT, ALGO_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from Entity.dynamic_obstacles import MovingSphereObstacle
from Entity.static_obstacles import AxisAlignedBoxObstacle, StaticSphereObstacle
from Environment.multi_agent_dmp_env import MultiAgentDMPEnv
from MASAC.config import MASACExperimentConfig, MASACNetworkConfig
from MASAC.curriculum import CurriculumStage, build_curriculum_stages, build_stage_env_kwargs


CUSTOM_SCENARIO_NAMES = ("boundary_only", "boundary_dynamic", "boundary_mixed")
TRAINING_DISTRIBUTION_SCENARIO = "phase3_level3"
SCENARIO_NAMES = (*CUSTOM_SCENARIO_NAMES, TRAINING_DISTRIBUTION_SCENARIO)
SCENARIO_LABELS = {
    "boundary_only": "有边界无障碍物",
    "boundary_dynamic": "有边界动态障碍物",
    "boundary_mixed": "有边界混合动静态障碍物",
    TRAINING_DISTRIBUTION_SCENARIO: "训练同分布 Phase 3 Level 3",
}
AGENT_COLORS = ("#22d3ee", "#34d399", "#fbbf24", "#fb7185", "#a78bfa", "#60a5fa")
SPATIOTEMPORAL_ENCOUNTER_WINDOW_SECONDS = 0.5


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _dataclass_from_mapping(cls, values: dict[str, Any]):
    allowed = {field.name for field in fields(cls)}
    return cls(**{key: value for key, value in values.items() if key in allowed})


def _resolve_device(device: str):
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _checkpoint_sort_key(path: Path) -> tuple[float, str]:
    return path.stat().st_mtime, str(path)


def _find_latest_checkpoint(output_root: Path) -> Path:
    candidates = list(output_root.rglob("MASAC.pth"))
    if not candidates:
        raise FileNotFoundError(
            f"未在 {output_root} 下找到 MASAC.pth，请通过 --checkpoint 指定模型。"
        )
    preferred = [path for path in candidates if path.parent.name in {"best", "final"}]
    return sorted(preferred or candidates, key=_checkpoint_sort_key)[-1]


def _resolve_checkpoint(path: Path | None, output_root: Path) -> Path:
    if path is None:
        return _find_latest_checkpoint(output_root)
    resolved = path.expanduser().resolve()
    if resolved.is_dir():
        resolved = resolved / "MASAC.pth"
    if not resolved.is_file():
        raise FileNotFoundError(f"checkpoint 不存在：{resolved}")
    return resolved


def _find_run_config(checkpoint: Path, explicit_config: Path | None) -> Path:
    if explicit_config is not None:
        config_path = explicit_config.expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"配置文件不存在：{config_path}")
        return config_path
    for parent in checkpoint.parents:
        candidate = parent / "config.json"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "无法从 checkpoint 目录向上找到 config.json，请使用 --config 指定训练配置。"
    )


def _load_configs(config_path: Path) -> tuple[MASACExperimentConfig, MASACNetworkConfig, dict[str, Any]]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    experiment_values = payload.get("experiment_config", {})
    network_values = payload.get("network_config", {})
    if not experiment_values:
        raise ValueError(f"{config_path} 缺少 experiment_config")
    if not network_values:
        raise ValueError(f"{config_path} 缺少 network_config")
    experiment_config = _dataclass_from_mapping(MASACExperimentConfig, experiment_values)
    network_config = _dataclass_from_mapping(MASACNetworkConfig, network_values)
    return experiment_config, network_config, payload


def _agent_ids(num_agents: int) -> list[str]:
    return [f"agent_{index}" for index in range(num_agents)]


def _matrix_to_agent_dict(values: np.ndarray, agent_ids: list[str]) -> dict[str, np.ndarray]:
    array = np.asarray(values, dtype=np.float32)
    return {
        agent_id: array[index].astype(np.float32, copy=True)
        for index, agent_id in enumerate(agent_ids)
    }


def _agent_dict_to_matrix(values: dict[str, np.ndarray], agent_ids: list[str]) -> np.ndarray:
    return np.stack(
        [np.asarray(values[agent_id], dtype=np.float32) for agent_id in agent_ids],
        axis=0,
    )


def _phase3_level3_stage(config: MASACExperimentConfig) -> CurriculumStage:
    stages = build_curriculum_stages(
        config.curriculum_phase2_box_counts,
        config.curriculum_phase2_sphere_counts,
        config.curriculum_phase3_dynamic_counts,
    )
    matching = [stage for stage in stages if stage.name == TRAINING_DISTRIBUTION_SCENARIO]
    if not matching:
        available = ", ".join(stage.name for stage in stages)
        raise ValueError(
            f"训练配置不包含 {TRAINING_DISTRIBUTION_SCENARIO}；可用阶段：{available}"
        )
    return matching[0]


def _build_environment(
    config: MASACExperimentConfig,
    scenario_name: str | None = None,
) -> MultiAgentDMPEnv:
    if scenario_name == TRAINING_DISTRIBUTION_SCENARIO:
        kwargs = build_stage_env_kwargs(config, _phase3_level3_stage(config))
    else:
        kwargs = config.build_core_env_kwargs()
    return MultiAgentDMPEnv(**copy.deepcopy(kwargs))


def _load_policy(
    checkpoint: Path,
    env: MultiAgentDMPEnv,
    network_config: MASACNetworkConfig,
    device: Any,
) -> tuple[Any, list[str]]:
    from MASAC.MASAC import MASAC

    agent_ids = _agent_ids(int(env.num_agents))
    _, obs_dim = env.observation_shape
    _, action_dim = env.action_shape
    dim_info = {
        agent_id: (int(obs_dim), int(action_dim))
        for agent_id in agent_ids
    }
    policy = MASAC.load(
        dim_info=dim_info,
        is_continue=True,
        model_dir=str(checkpoint.parent),
        network_config=network_config,
        device=device,
    )
    for agent in policy.agents.values():
        agent.actor.eval()
        agent.actor_target.eval()
        agent.critic.eval()
        agent.critic_target.eval()
    return policy, agent_ids


def _warmup_policy(
    policy: Any,
    env: MultiAgentDMPEnv,
    agent_ids: list[str],
    device: Any,
    steps: int,
    seed: int,
) -> None:
    if steps <= 0:
        return
    observation, _ = env.reset(seed=int(seed))
    obs_dict = _matrix_to_agent_dict(observation, agent_ids)
    for _ in range(int(steps)):
        _synchronize_device(device)
        policy.evaluate_action(obs_dict)
        _synchronize_device(device)


def _fixed_starts_goals(config: MASACExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
    bounds = np.asarray(config.workspace_bounds, dtype=float)
    lower, upper = bounds
    count = int(config.num_agents)
    x_start = float(np.clip(lower[0] + 0.65, lower[0] + 0.3, upper[0] - 0.3))
    x_goal = float(np.clip(upper[0] - 0.65, lower[0] + 0.3, upper[0] - 0.3))
    y_margin = min(0.55, 0.18 * float(upper[1] - lower[1]))
    y_values = np.linspace(lower[1] + y_margin, upper[1] - y_margin, count)
    z_span = max(0.0, min(0.45, 0.25 * float(upper[2] - lower[2])))
    z_values = np.linspace(-z_span, z_span, count)
    starts = np.column_stack([np.full(count, x_start), y_values, z_values])
    goals = np.column_stack([np.full(count, x_goal), y_values[::-1], -z_values])
    return starts.astype(float), goals.astype(float)


def _sample_spaced_points(
    rng: np.random.Generator,
    lower: np.ndarray,
    upper: np.ndarray,
    count: int,
    min_distance: float,
    max_attempts: int,
) -> np.ndarray:
    points: list[np.ndarray] = []
    for _ in range(count):
        for _ in range(max_attempts):
            candidate = rng.uniform(lower, upper)
            if all(float(np.linalg.norm(candidate - point)) >= min_distance for point in points):
                points.append(candidate)
                break
        else:
            raise RuntimeError("无法在给定边界内生成满足间距约束的多机点集")
    return np.stack(points, axis=0)


def _seeded_starts_goals(
    config: MASACExperimentConfig,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    count = int(config.num_agents)
    start_bounds = np.asarray(config.start_position_bounds, dtype=float)
    goal_bounds = np.asarray(config.goal_position_bounds, dtype=float)
    start_spacing = max(float(config.min_start_distance), float(config.inter_agent_safe_distance))
    goal_spacing = max(
        float(config.min_goal_distance),
        float(config.inter_agent_safe_distance) + 2.0 * float(config.goal_tolerance),
    )
    for _ in range(int(config.start_goal_max_attempts)):
        starts = _sample_spaced_points(
            rng,
            start_bounds[0],
            start_bounds[1],
            count,
            start_spacing,
            300,
        )
        goals = _sample_spaced_points(
            rng,
            goal_bounds[0],
            goal_bounds[1],
            count,
            goal_spacing,
            300,
        )
        # 对目标索引做确定性打乱，形成平行、交叉等不同多机关系。
        goals = goals[rng.permutation(count)]
        if np.all(np.linalg.norm(goals - starts, axis=1) >= float(config.min_start_goal_distance)):
            return starts.astype(float), goals.astype(float)
    raise RuntimeError(f"seed={seed} 无法生成满足约束的起终点")


def _center_is_clear(
    center: np.ndarray,
    radius: float,
    protected_points: np.ndarray,
    existing: list[Any],
) -> bool:
    if protected_points.size and np.any(
        np.linalg.norm(protected_points - center[None, :], axis=1) < radius + 0.55
    ):
        return False
    for obstacle in existing:
        other_radius = float(getattr(obstacle, "effective_radius", 0.0))
        if float(np.linalg.norm(np.asarray(obstacle.center) - center)) < radius + other_radius + 0.18:
            return False
    return True


def _dynamic_obstacles(
    bounds: np.ndarray,
    mixed: bool,
    rng: np.random.Generator | None = None,
    protected_points: np.ndarray | None = None,
    existing: list[Any] | None = None,
) -> list[MovingSphereObstacle]:
    lower, upper = bounds
    x_span = float(upper[0] - lower[0])
    y_span = float(upper[1] - lower[1])
    z_mid = float(0.5 * (lower[2] + upper[2]))
    movement_bounds = (lower.copy(), upper.copy())
    if mixed:
        specs = [
            (lower[0] + 0.56 * x_span, lower[1] + 0.80 * y_span, z_mid, 0.0, -0.62, 0.12),
            (lower[0] + 0.73 * x_span, lower[1] + 0.22 * y_span, z_mid + 0.25, 0.0, 0.52, -0.10),
        ]
    else:
        specs = [
            (lower[0] + 0.38 * x_span, lower[1] + 0.18 * y_span, z_mid - 0.25, 0.0, 0.62, 0.10),
            (lower[0] + 0.58 * x_span, lower[1] + 0.82 * y_span, z_mid + 0.25, 0.0, -0.68, -0.10),
            (lower[0] + 0.76 * x_span, lower[1] + 0.25 * y_span, z_mid, 0.0, 0.55, 0.14),
        ]
    if rng is None:
        return [
            MovingSphereObstacle(
                center=np.array([x, y, z], dtype=float),
                radius=0.28,
                velocity=np.array([vx, vy, vz], dtype=float),
                safety_margin=0.08,
                bounds=movement_bounds,
            )
            for x, y, z, vx, vy, vz in specs
        ]

    count = 2 if mixed else 3
    protected = np.asarray(protected_points if protected_points is not None else [], dtype=float).reshape(-1, 3)
    occupied = list(existing or [])
    obstacles: list[MovingSphereObstacle] = []
    effective_radius = 0.36
    sample_lower = lower + np.array([0.24 * x_span, 0.10 * y_span, 0.16 * (upper[2] - lower[2])])
    sample_upper = upper - np.array([0.18 * x_span, 0.10 * y_span, 0.16 * (upper[2] - lower[2])])
    for _ in range(count):
        for _ in range(2000):
            center = rng.uniform(sample_lower, sample_upper)
            if not _center_is_clear(center, effective_radius, protected, occupied):
                continue
            velocity = np.array(
                [rng.uniform(-0.08, 0.08), rng.uniform(-0.78, 0.78), rng.uniform(-0.20, 0.20)],
                dtype=float,
            )
            if float(np.linalg.norm(velocity)) < 0.35:
                continue
            obstacle = MovingSphereObstacle(
                center=center,
                radius=0.28,
                velocity=velocity,
                safety_margin=0.08,
                bounds=movement_bounds,
            )
            obstacles.append(obstacle)
            occupied.append(obstacle)
            break
        else:
            raise RuntimeError("无法生成满足间距约束的动态障碍物")
    return obstacles


def _mixed_static_obstacles(
    bounds: np.ndarray,
    rng: np.random.Generator | None = None,
    protected_points: np.ndarray | None = None,
) -> list[Any]:
    lower, upper = bounds
    span = upper - lower
    z_mid = float(0.5 * (lower[2] + upper[2]))
    if rng is None:
        return [
        StaticSphereObstacle(
            center=np.array(
                [lower[0] + 0.36 * span[0], lower[1] + 0.48 * span[1], z_mid + 0.25]
            ),
            radius=0.42,
            safety_margin=0.08,
        ),
        AxisAlignedBoxObstacle(
            center=np.array(
                [lower[0] + 0.66 * span[0], lower[1] + 0.42 * span[1], z_mid - 0.22]
            ),
            half_extents=np.array([0.34, 0.42, 0.46]),
            safety_margin=0.06,
        ),
        ]

    protected = np.asarray(protected_points if protected_points is not None else [], dtype=float).reshape(-1, 3)
    existing: list[Any] = []
    sample_lower = lower + np.array([0.24 * span[0], 0.12 * span[1], 0.18 * span[2]])
    sample_upper = upper - np.array([0.18 * span[0], 0.12 * span[1], 0.18 * span[2]])
    specs = (("sphere", 0.50), ("box", 0.82))
    for obstacle_type, bounding_radius in specs:
        for _ in range(2000):
            center = rng.uniform(sample_lower, sample_upper)
            if not _center_is_clear(center, bounding_radius, protected, existing):
                continue
            if obstacle_type == "sphere":
                obstacle = StaticSphereObstacle(
                    center=center,
                    radius=0.42,
                    safety_margin=0.08,
                )
            else:
                obstacle = AxisAlignedBoxObstacle(
                    center=center,
                    half_extents=np.array([0.34, 0.42, 0.46]),
                    safety_margin=0.06,
                )
            existing.append(obstacle)
            break
        else:
            raise RuntimeError("无法生成满足间距约束的静态障碍物")
    return existing


def build_scenario(
    name: str,
    config: MASACExperimentConfig,
    seed: int | None = None,
) -> dict[str, Any]:
    if name not in SCENARIO_NAMES:
        raise ValueError(f"不支持的场景：{name}")
    if name == TRAINING_DISTRIBUTION_SCENARIO:
        stage = _phase3_level3_stage(config)
        return {
            "name": name,
            "label": SCENARIO_LABELS[name],
            "seed": seed,
            "distribution": "training",
            "curriculum_stage": stage.to_dict(),
        }
    starts, goals = (
        _fixed_starts_goals(config)
        if seed is None
        else _seeded_starts_goals(config, int(seed))
    )
    bounds = np.asarray(config.workspace_bounds, dtype=float)
    protected_points = np.concatenate([starts, goals], axis=0)
    static_obstacles: list[Any] = []
    dynamic_obstacles: list[MovingSphereObstacle] = []
    if name == "boundary_dynamic":
        dynamic_obstacles = _dynamic_obstacles(
            bounds,
            mixed=False,
            rng=None if seed is None else np.random.default_rng(int(seed) + 10_001),
            protected_points=protected_points,
        )
    elif name == "boundary_mixed":
        static_obstacles = _mixed_static_obstacles(
            bounds,
            rng=None if seed is None else np.random.default_rng(int(seed) + 20_003),
            protected_points=protected_points,
        )
        dynamic_obstacles = _dynamic_obstacles(
            bounds,
            mixed=True,
            rng=None if seed is None else np.random.default_rng(int(seed) + 30_007),
            protected_points=protected_points,
            existing=static_obstacles,
        )
    return {
        "name": name,
        "label": SCENARIO_LABELS[name],
        "seed": seed,
        "distribution": "custom_validation",
        "starts": starts,
        "goals": goals,
        "static_obstacles": static_obstacles,
        "dynamic_obstacles": dynamic_obstacles,
    }


def _serialize_obstacle(obstacle: Any, obstacle_id: str, dynamic: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": obstacle_id,
        "type": type(obstacle).__name__,
        "dynamic": bool(dynamic),
        "center": np.asarray(obstacle.center, dtype=float),
        "velocity": np.asarray(getattr(obstacle, "velocity", np.zeros(3)), dtype=float),
        "safety_margin": float(getattr(obstacle, "safety_margin", 0.0)),
    }
    if hasattr(obstacle, "radius"):
        payload["radius"] = float(obstacle.radius)
    if hasattr(obstacle, "effective_radius"):
        payload["effective_radius"] = float(obstacle.effective_radius)
    if hasattr(obstacle, "half_extents"):
        payload["half_extents"] = np.asarray(obstacle.half_extents, dtype=float)
    if hasattr(obstacle, "expanded_half_extents"):
        payload["expanded_half_extents"] = np.asarray(obstacle.expanded_half_extents, dtype=float)
    if getattr(obstacle, "bounds", None) is not None:
        payload["bounds"] = [
            np.asarray(obstacle.bounds[0], dtype=float),
            np.asarray(obstacle.bounds[1], dtype=float),
        ]
    return _jsonable(payload)


def _obstacle_payload(env: MultiAgentDMPEnv) -> list[dict[str, Any]]:
    rows = [
        _serialize_obstacle(obstacle, f"static_{index}", False)
        for index, obstacle in enumerate(env.static_obstacles)
    ]
    rows.extend(
        _serialize_obstacle(obstacle, f"dynamic_{index}", True)
        for index, obstacle in enumerate(env.dynamic_obstacles)
    )
    return rows


def _min_pairwise_distance(positions: np.ndarray) -> float:
    if len(positions) < 2:
        return float("inf")
    distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    values = distances[np.triu_indices(len(positions), k=1)]
    return float(np.min(values))


def _frame_payload(
    *,
    env: MultiAgentDMPEnv,
    scenario_name: str,
    episode_index: int,
    step: int,
    info: dict[str, Any],
    action: np.ndarray | None,
    rewards: np.ndarray | None,
    terminated: bool,
    truncated: bool,
) -> dict[str, Any]:
    positions = np.asarray(env._positions(), dtype=float)
    velocities = np.asarray(env._velocities(), dtype=float)
    distances = np.linalg.norm(env.goals - positions, axis=1)
    zero_actions = np.zeros(env.action_shape, dtype=float)
    action_value = zero_actions if action is None else np.asarray(action, dtype=float)
    reward_value = np.zeros(env.num_agents, dtype=float) if rewards is None else np.asarray(rewards, dtype=float)
    boundary_distances = np.asarray(
        info.get("min_boundary_distances", env._compute_min_boundary_distances()),
        dtype=float,
    )
    success_mask = np.asarray(
        info.get("success_mask", distances <= float(env.env_config.goal_tolerance)),
        dtype=bool,
    )
    return _jsonable(
        {
            "scenario": scenario_name,
            "episode": int(episode_index),
            "step": int(step),
            "sim_time": float(step * env.dynamics[0].dt),
            "wall_time": time.time(),
            "positions": positions,
            "velocities": velocities,
            "starts": env.starts,
            "goals": env.goals,
            "actions": action_value,
            "rewards": reward_value,
            "distance_to_goals": distances,
            "success_mask": success_mask,
            "commanded_accelerations": info.get("commanded_accelerations", np.zeros_like(positions)),
            "applied_accelerations": info.get("applied_accelerations", np.zeros_like(positions)),
            "min_pairwise_distance": _min_pairwise_distance(positions),
            "min_boundary_distances": boundary_distances,
            "collision": bool(info.get("collision", False)),
            "collision_mask": info.get("collision_mask", np.zeros(env.num_agents, dtype=bool)),
            "obstacle_collision_mask": info.get(
                "obstacle_collision_mask", np.zeros(env.num_agents, dtype=bool)
            ),
            "inter_agent_collision_mask": info.get(
                "inter_agent_collision_mask", np.zeros(env.num_agents, dtype=bool)
            ),
            "boundary_collision_mask": info.get(
                "boundary_collision_mask", np.zeros(env.num_agents, dtype=bool)
            ),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "obstacles": _obstacle_payload(env),
        }
    )


class LiveState:
    def __init__(self, workspace_bounds: Any):
        self._lock = threading.Lock()
        self._traces: dict[str, Any] = {}
        self._trace_version = 0
        self._payload: dict[str, Any] = {
            "status": "initializing",
            "workspace_bounds": _jsonable(workspace_bounds),
            "current_frame": None,
            "current_trace": [],
            "completed": [],
            "aggregate": [],
            "trace_version": 0,
            "message": "等待仿真启动",
        }

    def update(self, **values: Any) -> None:
        with self._lock:
            self._payload.update(_jsonable(values))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._payload)

    def add_trace(self, trace: dict[str, Any]) -> None:
        with self._lock:
            self._traces[str(trace["key"])] = _jsonable(trace)
            self._trace_version += 1
            self._payload["trace_version"] = self._trace_version

    def traces_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "trace_version": self._trace_version,
                "traces": copy.deepcopy(self._traces),
            }


def _html_document(static_payload: dict[str, Any] | None = None) -> str:
    payload_json = json.dumps(_jsonable(static_payload or {}), ensure_ascii=False)
    return r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>MASAC 多机场景实时验证</title>
  <style>
    :root{color-scheme:dark;--bg:#09111f;--panel:#111c2e;--line:#263650;--text:#e5edf8;--muted:#91a4bf;--cyan:#22d3ee;--green:#34d399;--red:#fb7185;--amber:#fbbf24}
    *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:"Segoe UI","Microsoft YaHei",sans-serif}header{padding:18px 24px;background:#0c1628;border-bottom:1px solid var(--line)}h1{font-size:21px;margin:0 0 6px}.subtitle{font-size:12px;color:var(--muted)}main{max-width:1600px;margin:auto;padding:16px}.cards{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:10px;margin-bottom:12px}.card,.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px}.card{padding:10px 12px}.card .label{font-size:11px;color:var(--muted)}.card .value{font-size:19px;font-weight:650;margin-top:4px}.layout{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(340px,.65fr);gap:12px}.panel{padding:12px}.toolbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:10px}select,button,input{background:#17253a;color:var(--text);border:1px solid var(--line);border-radius:5px;padding:6px 9px}button{cursor:pointer}button:hover{border-color:#4f6b91;background:#1d304a}canvas{display:block;width:100%;background:radial-gradient(circle at 50% 45%,#10213a 0,#07101e 72%);border:1px solid var(--line);border-radius:6px;cursor:grab;touch-action:none}canvas.dragging{cursor:grabbing}#scene{height:670px}.side{display:grid;gap:12px}.table-wrap{max-height:330px;overflow:auto}table{width:100%;border-collapse:collapse;font-size:11px}th,td{padding:7px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}th{position:sticky;top:0;background:#16243a;color:#c8d6e8}.legend{font-size:12px;line-height:1.7;color:var(--muted)}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}.status-ok{color:var(--green)}.status-bad{color:var(--red)}.view-sep{width:1px;height:25px;background:var(--line)}@media(max-width:1050px){.cards{grid-template-columns:repeat(3,1fr)}.layout{grid-template-columns:1fr}#scene{height:600px}}
  </style>
</head>
<body>
<header><h1>MASAC 多机场景实时验证</h1><div class="subtitle">有边界无障碍物 · 有边界动态障碍物 · 有边界混合动静态障碍物</div></header>
<main>
  <section class="cards" id="cards"></section>
  <section class="layout">
    <div class="panel">
      <div class="toolbar"><select id="scenario"></select><button id="liveBtn">实时</button><button id="playBtn">播放</button><button id="resetBtn">复位</button><input id="slider" type="range" min="0" value="0"><span id="stepText"></span><span class="view-sep"></span><select id="accelMode" title="加速度箭头"><option value="applied" selected>Applied acceleration</option><option value="commanded">Commanded acceleration</option><option value="both">Applied + Commanded</option><option value="none">隐藏加速度</option></select><label class="legend"><input id="accelHistory" type="checkbox"> 历史箭头</label><span class="view-sep"></span><button id="viewDefault">默认视角</button><button id="viewTop">俯视</button><button id="viewSide">侧视</button></div>
      <canvas id="scene"></canvas>
    </div>
    <div class="side">
      <div class="panel"><div class="legend" id="legend"></div></div>
      <div class="panel table-wrap"><table><thead><tr><th>无人机</th><th>目标距离</th><th>速度</th><th>奖励</th><th>状态</th></tr></thead><tbody id="agents"></tbody></table></div>
      <div class="panel table-wrap"><table><thead><tr><th>场景</th><th>状态</th><th>步数</th><th>总奖励</th><th>最小机距</th></tr></thead><tbody id="summary"></tbody></table></div>
    </div>
  </section>
  <section class="panel" style="margin-top:12px"><div class="legend" style="margin-bottom:8px"><b>多种子聚合指标</b>（均值；成功率和碰撞率按 episode 统计）</div><div class="table-wrap"><table><thead><tr><th>场景</th><th>样本数</th><th>成功率</th><th>碰撞率</th><th>路径长度</th><th>飞行时间/s</th><th>平滑代价</th><th>控制代价</th><th>最小安全裕度</th><th>推理时间/ms</th></tr></thead><tbody id="aggregate"></tbody></table></div></section>
</main>
<script>
const STATIC_DATA=__STATIC_PAYLOAD__;
const COLORS=["#22d3ee","#34d399","#fbbf24","#fb7185","#a78bfa","#60a5fa"];
let liveData={status:"offline",current_frame:null,current_trace:[],completed:[],workspace_bounds:null,trace_version:0,aggregate:[]};
let traces=STATIC_DATA.traces||{};let summaries=STATIC_DATA.summaries||[];let aggregateRows=STATIC_DATA.aggregate||[];let selected="live";let index=0;let playing=false;let last=0;
let loadedTraceVersion=Object.keys(traces).length?Number.MAX_SAFE_INTEGER:-1;
const canvas=document.getElementById("scene"),ctx=canvas.getContext("2d"),slider=document.getElementById("slider"),scenario=document.getElementById("scenario"),accelMode=document.getElementById("accelMode"),accelHistory=document.getElementById("accelHistory");
function fmt(v,n=2){return v===null||v===undefined||Number.isNaN(Number(v))?"-":Number(v).toFixed(n)}
function resize(){const r=canvas.getBoundingClientRect(),d=devicePixelRatio||1;canvas.width=Math.round(r.width*d);canvas.height=Math.round(r.height*d);ctx.setTransform(d,0,0,d,0,0);draw()}
function currentTrace(){if(selected==="live")return liveData.current_trace||[];return traces[selected]?.frames||[]}
function currentFrame(){const t=currentTrace();return t.length?t[Math.min(index,t.length-1)]:liveData.current_frame}
function rebuildSelect(){const old=selected;scenario.innerHTML='<option value="live">实时仿真</option>';Object.keys(traces).forEach(k=>{const o=document.createElement("option");o.value=k;o.textContent=traces[k].label||k;scenario.appendChild(o)});selected=[...scenario.options].some(o=>o.value===old)?old:"live";scenario.value=selected}
function bounds(){const sceneBounds=selected!=="live"?traces[selected]?.scene?.workspace_bounds:null;return sceneBounds||liveData.workspace_bounds||STATIC_DATA.workspace_bounds||[[0,0,0],[9,4.5,2.4]]}
const camera={yaw:-0.72,pitch:0.46,zoom:1};let dragging=false,lastPointer=[0,0];
function setCamera(yaw,pitch,zoom=1){camera.yaw=yaw;camera.pitch=pitch;camera.zoom=zoom;draw()}
function sceneGeometry(){const b=bounds(),center=b[0].map((v,i)=>(v+b[1][i])/2),span=b[1].map((v,i)=>v-b[0][i]);return{b,center,span,scale:Math.max(...span)}}
function project3(point){const rect=canvas.getBoundingClientRect(),g=sceneGeometry(),x=(point[0]-g.center[0])/g.scale,y=(point[1]-g.center[1])/g.scale,z=(point[2]-g.center[2])/g.scale,cy=Math.cos(camera.yaw),sy=Math.sin(camera.yaw),cp=Math.cos(camera.pitch),sp=Math.sin(camera.pitch),xr=cy*x-sy*y,yr=sy*x+cy*y,zr=cp*z-sp*yr,depth=sp*z+cp*yr,base=Math.min(rect.width,rect.height)*0.82*camera.zoom,factor=3.1/Math.max(2.1,3.1+depth*.7);return{x:rect.width*.5+xr*base*factor,y:rect.height*.52-zr*base*factor,depth,factor,unit:base*factor/g.scale}}
function line3(a,b,color,width=1,dash=[]){const p=project3(a),q=project3(b);ctx.save();ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);ctx.beginPath();ctx.moveTo(p.x,p.y);ctx.lineTo(q.x,q.y);ctx.stroke();ctx.restore()}
function drawArrow3(a,b,color,options={}){const p=project3(a),q=project3(b),dx=q.x-p.x,dy=q.y-p.y,n=Math.hypot(dx,dy);ctx.save();ctx.strokeStyle=color;ctx.fillStyle=color;ctx.globalAlpha=options.alpha??1;ctx.lineWidth=options.width??1.7;ctx.setLineDash(options.dash||[]);ctx.beginPath();ctx.moveTo(p.x,p.y);ctx.lineTo(q.x,q.y);ctx.stroke();ctx.setLineDash([]);if(n>3){const ux=dx/n,uy=dy/n;ctx.beginPath();ctx.moveTo(q.x,q.y);ctx.lineTo(q.x-8*ux+4*uy,q.y-8*uy-4*ux);ctx.lineTo(q.x-8*ux-4*uy,q.y-8*uy+4*ux);ctx.closePath();ctx.fill()}if(options.label){ctx.font="10px Segoe UI";ctx.fillText(options.label,q.x+5,q.y-5)}ctx.restore()}
function label3(point,text,color="#b8c7dc",dx=0,dy=0){const p=project3(point);ctx.save();ctx.fillStyle=color;ctx.font="11px Segoe UI";ctx.fillText(text,p.x+dx,p.y+dy);ctx.restore()}
function boxCorners(center,half){const c=[];for(const dx of[-1,1])for(const dy of[-1,1])for(const dz of[-1,1])c.push([center[0]+dx*half[0],center[1]+dy*half[1],center[2]+dz*half[2]]);return c}
const BOX_EDGES=[[0,1],[0,2],[0,4],[1,3],[1,5],[2,3],[2,6],[3,7],[4,5],[4,6],[5,7],[6,7]];
function drawWireBox(center,half,color,width=1.2){const c=boxCorners(center,half);BOX_EDGES.forEach(e=>line3(c[e[0]],c[e[1]],color,width))}
function drawWorkspace(){const g=sceneGeometry(),lo=g.b[0],hi=g.b[1],center=g.center,half=g.span.map(v=>v/2);for(let i=1;i<10;i++){const x=lo[0]+g.span[0]*i/10,y=lo[1]+g.span[1]*i/10;line3([x,lo[1],lo[2]],[x,hi[1],lo[2]],"#27405f",.7);line3([lo[0],y,lo[2]],[hi[0],y,lo[2]],"#27405f",.7)}drawWireBox(center,half,"#58708f",1.15);const o=lo,axis=Math.min(...g.span)*.2;drawArrow3(o,[o[0]+axis,o[1],o[2]],"#fb7185");drawArrow3(o,[o[0],o[1]+axis,o[2]],"#34d399");drawArrow3(o,[o[0],o[1],o[2]+axis],"#60a5fa");[["X",[o[0]+axis,o[1],o[2]],"#fb7185"],["Y",[o[0],o[1]+axis,o[2]],"#34d399"],["Z",[o[0],o[1],o[2]+axis],"#60a5fa"]].forEach(a=>{const p=project3(a[1]);ctx.fillStyle=a[2];ctx.font="12px Segoe UI";ctx.fillText(a[0],p.x+4,p.y-4)});label3(o,"O (0, 0, 0)","#e5edf8",-34,16);label3([center[0],lo[1],lo[2]],`X: ${fmt(g.span[0])} m`,"#fb9aaa",0,18);label3([lo[0],center[1],lo[2]],`Y: ${fmt(g.span[1])} m`,"#6ee7b7",-8,18);label3([lo[0],lo[1],center[2]],`Z: ${fmt(g.span[2])} m`,"#93c5fd",8,0)}
function drawSphere3(center,radius,color,label=""){const p=project3(center),r=Math.max(4,radius*p.unit),grad=ctx.createRadialGradient(p.x-r*.32,p.y-r*.35,1,p.x,p.y,r);grad.addColorStop(0,"#ffffffdd");grad.addColorStop(.18,color);grad.addColorStop(1,color+"44");ctx.fillStyle=grad;ctx.strokeStyle=color;ctx.lineWidth=1.3;ctx.beginPath();ctx.arc(p.x,p.y,r,0,Math.PI*2);ctx.fill();ctx.stroke();if(label){ctx.fillStyle="#f8fbff";ctx.font="bold 11px Segoe UI";ctx.textAlign="center";ctx.textBaseline="middle";ctx.fillText(label,p.x,p.y);ctx.textAlign="start";ctx.textBaseline="alphabetic"}}
function drawTarget3(goal,color){const p=project3(goal),r=Math.max(6,.11*p.unit);ctx.save();ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();ctx.arc(p.x,p.y,r,0,Math.PI*2);ctx.stroke();ctx.beginPath();ctx.moveTo(p.x-r-4,p.y);ctx.lineTo(p.x+r+4,p.y);ctx.moveTo(p.x,p.y-r-4);ctx.lineTo(p.x,p.y+r+4);ctx.stroke();ctx.restore()}
function accelerationEndpoint(position,vector){const magnitude=Math.hypot(...vector);if(magnitude<1e-8)return null;const displayLength=Math.min(.95,.16*magnitude);return position.map((v,i)=>v+vector[i]*displayLength/magnitude)}
function drawAccelerationFrame(frame,alpha=1,showLabels=true){const mode=accelMode.value;if(mode==='none'||!frame)return;const modes=mode==='both'?['applied','commanded']:[mode];modes.forEach(kind=>{const vectors=kind==='applied'?frame.applied_accelerations:frame.commanded_accelerations;(frame.positions||[]).forEach((position,i)=>{const vector=vectors?.[i];if(!vector)return;const end=accelerationEndpoint(position,vector);if(!end)return;const magnitude=Math.hypot(...vector),isCommanded=kind==='commanded';drawArrow3(position,end,COLORS[i%COLORS.length],{alpha:isCommanded?alpha*.55:alpha,width:isCommanded?1.25:2.2,dash:isCommanded?[5,4]:[],label:showLabels?(isCommanded?'cmd ':'')+`|a|=${fmt(magnitude,2)}`:''})})})}
function drawScene3(frame,trace){drawWorkspace();if(!frame)return;const upto=Math.min(index,trace.length-1);for(let a=0;a<(frame.positions||[]).length;a++){ctx.strokeStyle=COLORS[a%COLORS.length]+"aa";ctx.lineWidth=2;ctx.beginPath();let started=false;for(let k=0;k<=upto;k++){const pos=trace[k]?.positions?.[a];if(!pos)continue;const p=project3(pos);started?ctx.lineTo(p.x,p.y):ctx.moveTo(p.x,p.y);started=true}ctx.stroke()}if(accelHistory.checked&&upto>0){const stride=Math.max(1,Math.ceil(upto/18));for(let k=0;k<upto;k+=stride)drawAccelerationFrame(trace[k],.22,false)}(frame.goals||[]).forEach((g,i)=>drawTarget3(g,COLORS[i%COLORS.length]));const items=[];(frame.obstacles||[]).forEach(o=>items.push({depth:project3(o.center).depth,draw:()=>{if(o.expanded_half_extents)drawWireBox(o.center,o.expanded_half_extents,o.dynamic?"#a78bfa":"#fb7185",1.6);else drawSphere3(o.center,o.effective_radius||o.radius||.2,o.dynamic?"#a78bfa":"#fb7185");if(o.dynamic&&o.velocity)drawArrow3(o.center,o.center.map((v,i)=>v+o.velocity[i]*.75),"#c4b5fd")}}));(frame.positions||[]).forEach((p,i)=>items.push({depth:project3(p).depth,draw:()=>drawSphere3(p,.10,COLORS[i%COLORS.length],`U${i}`)}));items.sort((a,b)=>b.depth-a.depth).forEach(item=>item.draw());drawAccelerationFrame(frame,1,true)}
function draw(){const rect=canvas.getBoundingClientRect();ctx.clearRect(0,0,rect.width,rect.height);const frame=currentFrame(),trace=currentTrace();drawScene3(frame,trace);ctx.fillStyle="#91a4bf";ctx.font="12px Segoe UI";ctx.fillText("拖动旋转 · 滚轮缩放 · 双击恢复视角",14,22);updatePanels(frame,trace)}
function updatePanels(f,t){const g=sceneGeometry(),size=`${fmt(g.span[0])} × ${fmt(g.span[1])} × ${fmt(g.span[2])} m`,ranges=`X [0, ${fmt(g.span[0])}] m　Y [0, ${fmt(g.span[1])}] m　Z [0, ${fmt(g.span[2])}] m`,vals=[['运行状态',liveData.status],['场景',f?.scenario||'-'],['步数',f?.step??'-'],['总奖励',f?fmt((f.rewards||[]).reduce((a,b)=>a+b,0)): '-'],['最小机距',f?fmt(f.min_pairwise_distance):'-'],['碰撞',f?.collision?'是':'否']];document.getElementById('cards').innerHTML=vals.map(x=>`<div class="card"><div class="label">${x[0]}</div><div class="value">${x[1]}</div></div>`).join('');document.getElementById('stepText').textContent=`${Math.min(index,Math.max(0,t.length-1))}/${Math.max(0,t.length-1)}`;slider.max=Math.max(0,t.length-1);slider.value=Math.min(index,Math.max(0,t.length-1));if(!f)return;document.getElementById('agents').innerHTML=(f.positions||[]).map((_,i)=>`<tr><td style="color:${COLORS[i%COLORS.length]}">U${i}</td><td>${fmt(f.distance_to_goals?.[i])}</td><td>${fmt(Math.hypot(...(f.velocities?.[i]||[0])))}</td><td>${fmt(f.rewards?.[i])}</td><td>${f.success_mask?.[i]?'到达':(f.collision_mask?.[i]?'碰撞':'飞行')}</td></tr>`).join('');document.getElementById('legend').innerHTML=`<b>${f.scenario}</b><br>场景尺寸：${size}<br>显示坐标：${ranges}<br>仿真时间：${fmt(f.sim_time)} s<br>动态障碍物：${(f.obstacles||[]).filter(o=>o.dynamic).length}<br>静态障碍物：${(f.obstacles||[]).filter(o=>!o.dynamic).length}<br>加速度：${accelMode.options[accelMode.selectedIndex].text}${accelHistory.checked?'（含历史）':''}<br><span class="dot" style="background:#a78bfa"></span>动态障碍物　<span class="dot" style="background:#fb7185"></span>静态障碍物<br>${liveData.message||''}`}
function updateSummary(){const rows=(summaries.length?summaries:liveData.completed||[]);document.getElementById('summary').innerHTML=rows.map(r=>`<tr><td>${r.label||r.scenario}</td><td class="${r.status==='success'?'status-ok':'status-bad'}">${r.status}</td><td>${r.steps}</td><td>${fmt(r.total_reward)}</td><td>${fmt(r.min_pairwise_distance)}</td></tr>`).join('')}
function updateAggregate(){const rows=aggregateRows.length?aggregateRows:(liveData.aggregate||[]);document.getElementById('aggregate').innerHTML=rows.map(r=>`<tr><td>${r.label||r.scenario}</td><td>${r.episodes}</td><td>${fmt(100*r.success_rate,1)}%</td><td>${fmt(100*r.collision_rate,1)}%</td><td>${fmt(r.path_length_mean_mean)}</td><td>${fmt(r.flight_time_mean_mean)}</td><td>${fmt(r.smoothness_cost_mean_mean)}</td><td>${fmt(r.control_cost_mean_mean)}</td><td>${fmt(r.min_safety_clearance_mean)}</td><td>${fmt(r.inference_time_mean_ms_mean,3)}</td></tr>`).join('')}
async function loadTracesIfNeeded(){const version=Number(liveData.trace_version||0);if(version===loadedTraceVersion)return;try{const r=await fetch('/api/traces',{cache:'no-store'});if(r.ok){const data=await r.json();Object.assign(traces,data.traces||{});loadedTraceVersion=Number(data.trace_version||version);rebuildSelect()}}catch(e){}}
async function poll(){try{const r=await fetch('/api/state',{cache:'no-store'});if(r.ok){liveData=await r.json();if(selected==='live'){index=Math.max(0,(liveData.current_trace||[]).length-1)}await loadTracesIfNeeded();rebuildSelect();updateSummary();updateAggregate();draw()}}catch(e){}setTimeout(poll,100)}
scenario.onchange=()=>{selected=scenario.value;index=selected==='live'?Math.max(0,currentTrace().length-1):0;playing=false;draw()};slider.oninput=()=>{index=Number(slider.value);playing=false;draw()};document.getElementById('liveBtn').onclick=()=>{selected='live';scenario.value='live';index=Math.max(0,currentTrace().length-1);draw()};document.getElementById('playBtn').onclick=()=>{playing=!playing};document.getElementById('resetBtn').onclick=()=>{index=0;playing=false;draw()};
document.getElementById('viewDefault').onclick=()=>setCamera(-.72,.46,1);document.getElementById('viewTop').onclick=()=>setCamera(0,1.48,.9);document.getElementById('viewSide').onclick=()=>setCamera(-1.57,0,1);
accelMode.onchange=draw;accelHistory.onchange=draw;
canvas.addEventListener('pointerdown',e=>{dragging=true;lastPointer=[e.clientX,e.clientY];canvas.classList.add('dragging');canvas.setPointerCapture(e.pointerId)});canvas.addEventListener('pointermove',e=>{if(!dragging)return;const dx=e.clientX-lastPointer[0],dy=e.clientY-lastPointer[1];lastPointer=[e.clientX,e.clientY];camera.yaw+=dx*.008;camera.pitch=Math.max(-1.48,Math.min(1.48,camera.pitch+dy*.008));draw()});canvas.addEventListener('pointerup',e=>{dragging=false;canvas.classList.remove('dragging');canvas.releasePointerCapture(e.pointerId)});canvas.addEventListener('pointercancel',()=>{dragging=false;canvas.classList.remove('dragging')});canvas.addEventListener('wheel',e=>{e.preventDefault();camera.zoom=Math.max(.45,Math.min(2.8,camera.zoom*Math.exp(-e.deltaY*.001)));draw()},{passive:false});canvas.addEventListener('dblclick',()=>setCamera(-.72,.46,1));
function animate(ts){if(playing&&ts-last>80){const t=currentTrace();index=t.length?((index+1)%t.length):0;last=ts;draw()}requestAnimationFrame(animate)}
rebuildSelect();updateSummary();updateAggregate();resize();window.addEventListener('resize',resize);requestAnimationFrame(animate);if(location.protocol.startsWith('http'))poll();else if(Object.keys(traces).length){selected=Object.keys(traces)[0];scenario.value=selected;draw()}
</script></body></html>'''.replace("__STATIC_PAYLOAD__", payload_json)


def _make_handler(state: LiveState, html_path: Path):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            path = unquote(urlparse(self.path).path)
            if path in {"/", "/index.html"}:
                self._send(200, "text/html; charset=utf-8", html_path.read_bytes())
            elif path == "/api/state":
                body = json.dumps(state.snapshot(), ensure_ascii=False).encode("utf-8")
                self._send(200, "application/json; charset=utf-8", body)
            elif path == "/api/traces":
                body = json.dumps(state.traces_snapshot(), ensure_ascii=False).encode("utf-8")
                self._send(200, "application/json; charset=utf-8", body)
            else:
                self._send(404, "text/plain; charset=utf-8", "not found".encode("utf-8"))

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    return Handler


def _start_server(
    state: LiveState,
    html_path: Path,
    host: str,
    port: int,
) -> tuple[ThreadingHTTPServer, threading.Thread]:
    server = ThreadingHTTPServer((host, port), _make_handler(state, html_path))
    thread = threading.Thread(target=server.serve_forever, name="masac-validation-http", daemon=True)
    thread.start()
    return server, thread


def _serialized_obstacle_clearance(position: np.ndarray, obstacle: dict[str, Any]) -> float:
    center = np.asarray(obstacle["center"], dtype=float)
    if obstacle.get("expanded_half_extents") is not None:
        half = np.asarray(obstacle["expanded_half_extents"], dtype=float)
        q = np.abs(position - center) - half
        return float(np.linalg.norm(np.maximum(q, 0.0)) + min(float(np.max(q)), 0.0))
    radius = float(obstacle.get("effective_radius", obstacle.get("radius", 0.0)))
    return float(np.linalg.norm(position - center) - radius)


def _find_spatiotemporal_path_encounter(
    positions: np.ndarray,
    first_reached_steps: np.ndarray,
    dt: float,
    distance_threshold: float,
) -> dict[str, Any]:
    """Find the closest pre-arrival path encounter within a bounded time offset."""
    num_steps, num_agents, _ = positions.shape
    window_steps = max(
        0,
        int(math.ceil(SPATIOTEMPORAL_ENCOUNTER_WINDOW_SECONDS / max(dt, 1e-9))),
    )
    best: dict[str, Any] | None = None
    for agent_i in range(num_agents):
        last_i = min(num_steps - 1, max(0, int(first_reached_steps[agent_i]) - 1))
        for agent_j in range(agent_i + 1, num_agents):
            last_j = min(num_steps - 1, max(0, int(first_reached_steps[agent_j]) - 1))
            for step_i in range(1, last_i + 1):
                step_j_start = max(1, step_i - window_steps)
                step_j_stop = min(last_j, step_i + window_steps)
                if step_j_start > step_j_stop:
                    continue
                candidate_positions = positions[step_j_start : step_j_stop + 1, agent_j]
                distances = np.linalg.norm(
                    candidate_positions - positions[step_i, agent_i],
                    axis=1,
                )
                local_offset = int(np.argmin(distances))
                distance = float(distances[local_offset])
                if best is not None and distance >= float(best["distance"]):
                    continue
                step_j = step_j_start + local_offset
                best = {
                    "agent_pair": [int(agent_i), int(agent_j)],
                    "distance": distance,
                    "step_i": int(step_i),
                    "step_j": int(step_j),
                    "time_i": float(step_i * dt),
                    "time_j": float(step_j * dt),
                    "time_offset": float(abs(step_i - step_j) * dt),
                }
    if best is None:
        return {
            "detected": False,
            "agent_pair": [],
            "distance": None,
            "step_i": None,
            "step_j": None,
            "time_i": None,
            "time_j": None,
            "time_offset": None,
        }
    best["detected"] = bool(float(best["distance"]) <= float(distance_threshold))
    return best


def _episode_metrics_from_frames(
    scenario: dict[str, Any],
    episode_index: int,
    seed: int,
    frames: list[dict[str, Any]],
    total_reward: float,
    status: str,
    config: MASACExperimentConfig,
    inference_times_ms: list[float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    final = frames[-1]
    positions = np.asarray([frame["positions"] for frame in frames], dtype=float)
    rewards = np.asarray([frame["rewards"] for frame in frames[1:]], dtype=float)
    accelerations = np.asarray(
        [frame["applied_accelerations"] for frame in frames[1:]],
        dtype=float,
    )
    commanded = np.asarray(
        [frame["commanded_accelerations"] for frame in frames[1:]],
        dtype=float,
    )
    dt = float(config.time_step)
    num_agents = positions.shape[1]
    path_lengths = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=2), axis=0)
    direct_lengths = np.linalg.norm(positions[0] - np.asarray(final["goals"], dtype=float), axis=1)
    path_efficiencies = np.divide(
        direct_lengths,
        np.maximum(path_lengths, 1e-9),
    )
    control_costs = (
        np.sum(np.sum(accelerations ** 2, axis=2), axis=0) * dt
        if accelerations.size
        else np.zeros(num_agents)
    )
    if len(accelerations) >= 2:
        jerk = np.diff(accelerations, axis=0) / dt
        smoothness_costs = np.mean(np.sum(jerk ** 2, axis=2), axis=0)
    else:
        smoothness_costs = np.zeros(num_agents)
    if accelerations.size and commanded.shape == accelerations.shape:
        clipped = np.any(~np.isclose(commanded, accelerations, atol=1e-6), axis=2)
        clip_ratios = np.mean(clipped, axis=0)
    else:
        clip_ratios = np.zeros(num_agents)

    first_reached_steps = np.full(num_agents, int(final["step"]), dtype=int)
    reached_once = np.zeros(num_agents, dtype=bool)
    for frame in frames:
        mask = np.asarray(frame["success_mask"], dtype=bool)
        newly_reached = mask & ~reached_once
        first_reached_steps[newly_reached] = int(frame["step"])
        reached_once |= mask
    flight_times = first_reached_steps.astype(float) * dt
    path_efficiencies[~reached_once] = np.nan
    encounter = _find_spatiotemporal_path_encounter(
        positions,
        first_reached_steps,
        dt,
        float(config.inter_agent_influence_distance),
    )
    successful_encounter = bool(status == "success" and encounter["detected"])

    min_pairwise_by_agent = np.full(num_agents, np.inf, dtype=float)
    min_boundary_by_agent = np.full(num_agents, np.inf, dtype=float)
    min_obstacle_by_agent = np.full(num_agents, np.inf, dtype=float)
    min_safety_clearance_by_agent = np.full(num_agents, np.inf, dtype=float)
    for frame in frames:
        frame_positions = np.asarray(frame["positions"], dtype=float)
        boundary = np.asarray(frame["min_boundary_distances"], dtype=float)
        min_boundary_by_agent = np.minimum(min_boundary_by_agent, boundary)
        for agent_index in range(num_agents):
            if num_agents > 1:
                other_distances = np.linalg.norm(
                    np.delete(frame_positions, agent_index, axis=0) - frame_positions[agent_index],
                    axis=1,
                )
                min_pairwise_by_agent[agent_index] = min(
                    min_pairwise_by_agent[agent_index],
                    float(np.min(other_distances)),
                )
            obstacle_clearances = [
                _serialized_obstacle_clearance(frame_positions[agent_index], obstacle)
                for obstacle in frame.get("obstacles", [])
            ]
            if obstacle_clearances:
                min_obstacle_by_agent[agent_index] = min(
                    min_obstacle_by_agent[agent_index],
                    min(obstacle_clearances),
                )
        pair_clearance = min_pairwise_by_agent - float(config.inter_agent_safe_distance)
        min_safety_clearance_by_agent = np.minimum(
            np.minimum(pair_clearance, min_boundary_by_agent),
            min_obstacle_by_agent,
        )

    agent_rows: list[dict[str, Any]] = []
    for agent_index in range(num_agents):
        agent_rows.append(
            {
                "scenario": scenario["name"],
                "seed": int(seed),
                "episode": int(episode_index),
                "agent_id": f"agent_{agent_index}",
                "success": bool(reached_once[agent_index]),
                "path_length": float(path_lengths[agent_index]),
                "path_efficiency": float(path_efficiencies[agent_index]),
                "flight_time": float(flight_times[agent_index]),
                "smoothness_cost": float(smoothness_costs[agent_index]),
                "control_cost": float(control_costs[agent_index]),
                "min_pairwise_distance": float(min_pairwise_by_agent[agent_index]),
                "min_obstacle_clearance": float(min_obstacle_by_agent[agent_index]),
                "min_boundary_distance": float(min_boundary_by_agent[agent_index]),
                "min_safety_clearance": float(min_safety_clearance_by_agent[agent_index]),
                "acceleration_clip_ratio": float(clip_ratios[agent_index]),
                "total_reward": float(np.sum(rewards[:, agent_index])) if rewards.size else 0.0,
            }
        )

    inference = np.asarray(inference_times_ms, dtype=float)
    summary = {
        "scenario": scenario["name"],
        "label": scenario["label"],
        "seed": int(seed),
        "episode": int(episode_index),
        "status": status,
        "steps": int(final["step"]),
        "total_reward": float(total_reward),
        "reached_agent_count": int(sum(bool(value) for value in final["success_mask"])),
        "min_pairwise_distance": float(
            min(frame["min_pairwise_distance"] for frame in frames)
        ),
        "min_boundary_distance": float(
            min(min(frame["min_boundary_distances"]) for frame in frames)
        ),
        "collision": bool(any(frame["collision"] for frame in frames)),
        "obstacle_collision": bool(
            any(any(frame["obstacle_collision_mask"]) for frame in frames)
        ),
        "inter_agent_collision": bool(
            any(any(frame["inter_agent_collision_mask"]) for frame in frames)
        ),
        "boundary_collision": bool(
            any(any(frame["boundary_collision_mask"]) for frame in frames)
        ),
        "final_mean_goal_distance": float(np.mean(final["distance_to_goals"])),
        "path_length_mean": float(np.mean(path_lengths)),
        "path_length_team": float(np.sum(path_lengths)),
        "path_efficiency_mean": (
            float(np.nanmean(path_efficiencies)) if np.any(reached_once) else 0.0
        ),
        "flight_time_mean": float(np.mean(flight_times)),
        "smoothness_cost_mean": float(np.mean(smoothness_costs)),
        "control_cost_mean": float(np.mean(control_costs)),
        "min_safety_clearance": float(np.min(min_safety_clearance_by_agent)),
        "acceleration_clip_ratio": float(np.mean(clip_ratios)),
        "inference_time_mean_ms": float(np.mean(inference)) if inference.size else 0.0,
        "inference_time_median_ms": float(np.median(inference)) if inference.size else 0.0,
        "inference_time_p95_ms": float(np.percentile(inference, 95)) if inference.size else 0.0,
        "spatiotemporal_path_encounter_success": successful_encounter,
        "spatiotemporal_path_encounter_agents": encounter["agent_pair"],
        "spatiotemporal_path_encounter_distance": encounter["distance"],
        "spatiotemporal_path_encounter_step_i": encounter["step_i"],
        "spatiotemporal_path_encounter_step_j": encounter["step_j"],
        "spatiotemporal_path_encounter_time_i": encounter["time_i"],
        "spatiotemporal_path_encounter_time_j": encounter["time_j"],
        "spatiotemporal_path_encounter_time_offset": encounter["time_offset"],
    }
    return summary, agent_rows


def _synchronize_device(device: Any) -> None:
    if str(device).startswith("cuda"):
        import torch

        torch.cuda.synchronize(device)


def _run_episode(
    *,
    scenario: dict[str, Any],
    episode_index: int,
    seed: int,
    config: MASACExperimentConfig,
    policy: Any | None,
    device: Any,
    agent_ids: list[str],
    output_dir: Path,
    live_state: LiveState,
    realtime_delay: float,
    max_steps: int,
    stream_live: bool,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    is_training_distribution = (
        scenario.get("distribution") == "training"
        and scenario["name"] == TRAINING_DISTRIBUTION_SCENARIO
    )
    env = _build_environment(
        config,
        scenario_name=scenario["name"] if is_training_distribution else None,
    )
    scenario_key = f"{scenario['name']}_seed_{seed}"
    episode_dir = output_dir / "scenarios" / scenario["name"] / str(seed)
    episode_dir.mkdir(parents=True, exist_ok=True)
    try:
        if is_training_distribution:
            observation, info = env.reset(seed=seed)
        else:
            observation, info = env.reset(
                seed=seed,
                options={
                    "starts": np.asarray(scenario["starts"], dtype=float),
                    "goals": np.asarray(scenario["goals"], dtype=float),
                    "static_obstacles": copy.deepcopy(scenario["static_obstacles"]),
                    "dynamic_obstacles": copy.deepcopy(scenario["dynamic_obstacles"]),
                },
            )
        scene_payload = {
            "key": scenario_key,
            "scenario": scenario["name"],
            "label": scenario["label"],
            "distribution": scenario.get("distribution", "custom_validation"),
            "curriculum_stage": scenario.get("curriculum_stage"),
            "episode": episode_index,
            "seed": seed,
            "workspace_bounds": config.workspace_bounds,
            "starts": env.starts,
            "goals": env.goals,
            "static_obstacles": [
                _serialize_obstacle(obstacle, f"static_{index}", False)
                for index, obstacle in enumerate(env.static_obstacles)
            ],
            "dynamic_obstacles": [
                _serialize_obstacle(obstacle, f"dynamic_{index}", True)
                for index, obstacle in enumerate(env.dynamic_obstacles)
            ],
        }
        _write_json(episode_dir / "scene.json", scene_payload)

        frames: list[dict[str, Any]] = []
        initial_frame = _frame_payload(
            env=env,
            scenario_name=scenario["name"],
            episode_index=episode_index,
            step=0,
            info=info,
            action=None,
            rewards=None,
            terminated=False,
            truncated=False,
        )
        frames.append(initial_frame)
        if stream_live:
            live_state.update(
                status="running",
                current_frame=initial_frame,
                current_trace=frames,
                message=f"正在运行：{scenario['label']} / seed {seed}",
            )

        total_reward = 0.0
        inference_times_ms: list[float] = []
        terminated = False
        truncated = False
        frames_path = episode_dir / "frames.jsonl"
        with frames_path.open("w", encoding="utf-8") as frame_file:
            frame_file.write(json.dumps(initial_frame, ensure_ascii=False) + "\n")
            for step in range(1, max_steps + 1):
                if policy is None:
                    action_matrix = np.zeros(env.action_shape, dtype=np.float32)
                else:
                    obs_dict = _matrix_to_agent_dict(observation, agent_ids)
                    _synchronize_device(device)
                    inference_start_ns = time.perf_counter_ns()
                    action_dict = policy.evaluate_action(obs_dict)
                    _synchronize_device(device)
                    inference_times_ms.append(
                        float(time.perf_counter_ns() - inference_start_ns) / 1_000_000.0
                    )
                    action_matrix = _agent_dict_to_matrix(action_dict, agent_ids)
                    action_matrix = np.clip(
                        action_matrix,
                        env.action_space.low,
                        env.action_space.high,
                    ).astype(np.float32)
                observation, rewards, terminated, truncated, info = env.step(action_matrix)
                total_reward += float(np.sum(rewards))
                frame = _frame_payload(
                    env=env,
                    scenario_name=scenario["name"],
                    episode_index=episode_index,
                    step=step,
                    info=info,
                    action=action_matrix,
                    rewards=rewards,
                    terminated=terminated,
                    truncated=truncated,
                )
                frames.append(frame)
                frame_file.write(json.dumps(frame, ensure_ascii=False) + "\n")
                if stream_live:
                    frame_file.flush()
                    live_state.update(current_frame=frame, current_trace=frames)
                if realtime_delay > 0.0:
                    time.sleep(realtime_delay)
                if bool(terminated or truncated):
                    break

        if bool(info.get("success", False)):
            status = "success"
        elif bool(info.get("collision", False)):
            status = "collision"
        elif truncated or frames[-1]["step"] >= max_steps:
            status = "timeout"
        else:
            status = "done"
        summary, agent_rows = _episode_metrics_from_frames(
            scenario,
            episode_index,
            seed,
            frames,
            total_reward,
            status,
            config,
            inference_times_ms,
        )
        summary["frames_path"] = str(frames_path.relative_to(output_dir))
        summary["scene_path"] = str((episode_dir / "scene.json").relative_to(output_dir))
        _write_json(episode_dir / "summary.json", summary)
        trace = {
            "key": scenario_key,
            "label": f"{scenario['label']} / E{episode_index}",
            "scene": scene_payload,
            "summary": summary,
            "frames": frames,
        }
        return summary, trace, agent_rows
    finally:
        env.close()


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _numeric_stats(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = np.asarray([row[key] for row in rows], dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return {
            f"{key}_mean": 0.0,
            f"{key}_std": 0.0,
            f"{key}_min": 0.0,
            f"{key}_max": 0.0,
            f"{key}_median": 0.0,
            f"{key}_p95": 0.0,
        }
    return {
        f"{key}_mean": float(np.mean(values)),
        f"{key}_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        f"{key}_min": float(np.min(values)),
        f"{key}_max": float(np.max(values)),
        f"{key}_median": float(np.median(values)),
        f"{key}_p95": float(np.percentile(values, 95)),
    }


def _aggregate_metrics(rows: list[dict[str, Any]], num_agents: int) -> list[dict[str, Any]]:
    metric_keys = (
        "path_length_mean",
        "path_length_team",
        "path_efficiency_mean",
        "flight_time_mean",
        "smoothness_cost_mean",
        "control_cost_mean",
        "min_safety_clearance",
        "inference_time_mean_ms",
        "inference_time_median_ms",
        "inference_time_p95_ms",
        "acceleration_clip_ratio",
    )
    aggregate_rows: list[dict[str, Any]] = []
    for scenario_name in SCENARIO_NAMES:
        scenario_rows = [row for row in rows if row["scenario"] == scenario_name]
        if not scenario_rows:
            continue
        total = len(scenario_rows)
        success_count = sum(row["status"] == "success" for row in scenario_rows)
        collision_count = sum(bool(row["collision"]) for row in scenario_rows)
        success_ci = _wilson_interval(success_count, total)
        collision_ci = _wilson_interval(collision_count, total)
        aggregate: dict[str, Any] = {
            "scenario": scenario_name,
            "label": SCENARIO_LABELS[scenario_name],
            "episodes": total,
            "success_rate": success_count / total,
            "success_ci95_low": success_ci[0],
            "success_ci95_high": success_ci[1],
            "collision_rate": collision_count / total,
            "collision_ci95_low": collision_ci[0],
            "collision_ci95_high": collision_ci[1],
            "obstacle_collision_rate": float(np.mean([row["obstacle_collision"] for row in scenario_rows])),
            "inter_agent_collision_rate": float(np.mean([row["inter_agent_collision"] for row in scenario_rows])),
            "boundary_collision_rate": float(np.mean([row["boundary_collision"] for row in scenario_rows])),
            "timeout_rate": float(np.mean([row["status"] == "timeout" for row in scenario_rows])),
            "agent_success_rate": float(
                np.sum([row["reached_agent_count"] for row in scenario_rows])
                / (total * max(1, int(num_agents)))
            ),
        }
        for key in metric_keys:
            aggregate.update(_numeric_stats(scenario_rows, key))
        aggregate_rows.append(aggregate)
    return aggregate_rows


def _select_representatives(
    rows: list[dict[str, Any]],
    per_scenario: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for scenario_name in SCENARIO_NAMES:
        candidates = [row for row in rows if row["scenario"] == scenario_name]
        if not candidates:
            continue
        encounter_candidates = sorted(
            (
                row
                for row in candidates
                if row.get("spatiotemporal_path_encounter_success", False)
            ),
            key=lambda row: float(row["spatiotemporal_path_encounter_distance"]),
        )
        category_candidates = [
            ("多机时空路径交汇后成功", encounter_candidates),
            ("成功样本", [row for row in candidates if row["status"] == "success"]),
            ("碰撞样本", [row for row in candidates if row["collision"]]),
            (
                "最长路径样本",
                sorted(candidates, key=lambda row: row["path_length_mean"], reverse=True),
            ),
            (
                "最小安全裕度样本",
                sorted(candidates, key=lambda row: row["min_safety_clearance"]),
            ),
            ("首个测试样本", candidates),
        ]
        seen: set[int] = set()
        for category, category_rows in category_candidates:
            row = next(
                (candidate for candidate in category_rows if int(candidate["seed"]) not in seen),
                None,
            )
            if row is None:
                continue
            representative = dict(row)
            representative["representative_category"] = category
            selected.append(representative)
            seed = int(row["seed"])
            seen.add(seed)
            if len(seen) >= max(1, int(per_scenario)):
                break
    return selected


def _load_saved_trace(output_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    frames_path = output_dir / summary["frames_path"]
    scene_path = output_dir / summary["scene_path"]
    frames = [json.loads(line) for line in frames_path.read_text(encoding="utf-8").splitlines() if line]
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    key = f"{summary['scenario']}_seed_{summary['seed']}"
    return {
        "key": key,
        "label": (
            f"{summary['label']} / "
            f"{summary.get('representative_category', '典型样本')} / "
            f"seed={summary['seed']}"
        ),
        "scene": scene,
        "summary": summary,
        "frames": frames,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="验证 MASAC 在定制场景或训练同分布场景中的行为，并生成 HTML 可视化。"
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/masac_validation"))
    parser.add_argument("--model-output-root", type=Path, default=Path("artifacts/masac"))
    parser.add_argument(
        "--scenario",
        choices=("all", *SCENARIO_NAMES),
        default="all",
        help=(
            "all 保持运行原有三类定制场景；phase3_level3 使用训练场景生成器"
            "执行同分布评估。"
        ),
    )
    parser.add_argument(
        "--episodes-per-scenario",
        type=int,
        default=None,
        help="兼容旧参数；设置后覆盖 --num-test-seeds。",
    )
    parser.add_argument("--num-test-seeds", type=int, default=100)
    parser.add_argument("--test-seed-base", type=int, default=20260713)
    parser.add_argument("--seed", type=int, default=None, help="--test-seed-base 的兼容别名。")
    parser.add_argument("--visualized-seeds-per-scenario", type=int, default=6)
    parser.add_argument("--inference-warmup-steps", type=int, default=10)
    parser.add_argument(
        "--save-all-traces",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--realtime-delay", type=float, default=0.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument("--no-server", action="store_true")
    parser.add_argument("--keep-server", action="store_true")
    parser.add_argument("--keep-server-seconds", type=float, default=10.0)
    parser.add_argument(
        "--policy-mode",
        choices=("masac", "zero"),
        default="masac",
        help="zero 仅用于检查场景、记录与页面链路。",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    test_seed_count = int(
        args.episodes_per_scenario
        if args.episodes_per_scenario is not None
        else args.num_test_seeds
    )
    if test_seed_count <= 0:
        raise ValueError("--num-test-seeds 必须为正数")
    if args.realtime_delay < 0.0:
        raise ValueError("--realtime-delay 不能为负数")
    test_seed_base = int(args.seed if args.seed is not None else args.test_seed_base)
    test_seeds = [test_seed_base + offset for offset in range(test_seed_count)]

    checkpoint: Path | None = None
    if args.policy_mode == "masac":
        checkpoint = _resolve_checkpoint(args.checkpoint, args.model_output_root)
        config_path = _find_run_config(checkpoint, args.config)
    else:
        if args.config is None:
            experiment_config = MASACExperimentConfig()
            network_config = MASACNetworkConfig()
            raw_config: dict[str, Any] = {}
            config_path = None
        else:
            config_path = args.config.expanduser().resolve()
    if args.policy_mode == "masac" or args.config is not None:
        experiment_config, network_config, raw_config = _load_configs(config_path)

    device = _resolve_device(args.device) if args.policy_mode == "masac" else "not-used"
    env_probe = _build_environment(experiment_config)
    try:
        if args.policy_mode == "masac":
            policy, agent_ids = _load_policy(checkpoint, env_probe, network_config, device)
            _warmup_policy(
                policy,
                env_probe,
                agent_ids,
                device,
                int(args.inference_warmup_steps),
                test_seed_base,
            )
        else:
            policy = None
            agent_ids = _agent_ids(int(env_probe.num_agents))
    finally:
        env_probe.close()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / f"validation_seed_{test_seed_base}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "index.html"
    html_path.write_text(_html_document(), encoding="utf-8")
    live_state = LiveState(experiment_config.workspace_bounds)

    server = None
    if not args.no_server:
        server, _ = _start_server(live_state, html_path, args.host, args.port)
        url = f"http://{args.host}:{args.port}/"
        print(f"实时页面：{url}")
        if args.open_browser:
            webbrowser.open(url)

    selected_names = (
        list(CUSTOM_SCENARIO_NAMES)
        if args.scenario == "all"
        else [args.scenario]
    )
    max_steps = int(args.max_steps or experiment_config.max_steps)
    stream_live = server is not None and (
        float(args.realtime_delay) > 0.0 or test_seed_count <= 5
    )
    summaries: list[dict[str, Any]] = []
    per_agent_rows: list[dict[str, Any]] = []
    completed_for_live: list[dict[str, Any]] = []
    run_payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_seed_base": test_seed_base,
        "test_seed_count": test_seed_count,
        "test_seeds": test_seeds,
        "checkpoint": checkpoint,
        "config_path": config_path,
        "policy_mode": args.policy_mode,
        "device": str(device),
        "scenario_names": selected_names,
        "scenario_distributions": {
            name: (
                "training"
                if name == TRAINING_DISTRIBUTION_SCENARIO
                else "custom_validation"
            )
            for name in selected_names
        },
        "episodes_per_scenario": test_seed_count,
        "inference_warmup_steps": int(args.inference_warmup_steps),
        "save_all_traces": bool(args.save_all_traces),
        "max_steps": max_steps,
        "spatiotemporal_encounter_time_window_seconds": (
            SPATIOTEMPORAL_ENCOUNTER_WINDOW_SECONDS
        ),
        "spatiotemporal_encounter_distance_threshold": float(
            experiment_config.inter_agent_influence_distance
        ),
        "experiment_config": asdict(experiment_config),
        "network_config": asdict(network_config),
        "training_config": raw_config,
    }
    _write_json(output_dir / "run_config.json", run_payload)
    _write_json(
        output_dir / "test_seeds.json",
        {
            "test_seed_base": test_seed_base,
            "count": test_seed_count,
            "seeds": test_seeds,
            "paired_across_scenarios": True,
        },
    )

    try:
        for scenario_index, scenario_name in enumerate(selected_names):
            for episode_index, episode_seed in enumerate(test_seeds):
                # 同一个 seed 在三类场景中共享起终点，只改变障碍物类别。
                scenario = build_scenario(
                    scenario_name,
                    experiment_config,
                    seed=episode_seed,
                )
                summary, trace, episode_agent_rows = _run_episode(
                    scenario=scenario,
                    episode_index=episode_index,
                    seed=episode_seed,
                    config=experiment_config,
                    policy=policy,
                    device=device,
                    agent_ids=agent_ids,
                    output_dir=output_dir,
                    live_state=live_state,
                    realtime_delay=float(args.realtime_delay),
                    max_steps=max_steps,
                    stream_live=stream_live,
                )
                summaries.append(summary)
                per_agent_rows.extend(episode_agent_rows)
                completed_for_live.append(summary)
                live_state.update(
                    completed=completed_for_live,
                    status="running",
                    message=f"已完成：{trace['label']}，状态={summary['status']}",
                )
                if (episode_index + 1) % 10 == 0 or episode_index == 0:
                    print(
                        f"{scenario['label']} | {episode_index + 1}/{test_seed_count} | "
                        f"seed={episode_seed} | status={summary['status']} | "
                        f"path={summary['path_length_mean']:.3f} | "
                        f"infer={summary['inference_time_mean_ms']:.3f} ms"
                    )

        aggregate_rows = _aggregate_metrics(summaries, int(experiment_config.num_agents))
        representative_rows = _select_representatives(
            summaries,
            int(args.visualized_seeds_per_scenario),
        )
        traces = {
            trace["key"]: trace
            for trace in (
                _load_saved_trace(output_dir, summary)
                for summary in representative_rows
            )
        }
        for trace in traces.values():
            live_state.add_trace(trace)

        if not bool(args.save_all_traces):
            retained = {(row["scenario"], int(row["seed"])) for row in representative_rows}
            for row in summaries:
                if (row["scenario"], int(row["seed"])) in retained:
                    continue
                for key in ("frames_path", "scene_path"):
                    path = output_dir / row[key]
                    if path.is_file():
                        path.unlink()

        overall = {
            "episode_count": len(summaries),
            "scenario_count": len(selected_names),
            "seeds_per_scenario": test_seed_count,
            "success_rate": float(np.mean([row["status"] == "success" for row in summaries])),
            "collision_rate": float(np.mean([row["collision"] for row in summaries])),
            "mean_total_reward": float(np.mean([row["total_reward"] for row in summaries])),
            "aggregate_metrics": aggregate_rows,
            "representative_seeds": {
                scenario_name: [
                    int(row["seed"])
                    for row in representative_rows
                    if row["scenario"] == scenario_name
                ]
                for scenario_name in selected_names
            },
            "representative_details": [
                {
                    "scenario": row["scenario"],
                    "seed": int(row["seed"]),
                    "category": row["representative_category"],
                }
                for row in representative_rows
            ],
        }
        _write_json(output_dir / "summary.json", overall)
        _write_json(output_dir / "episode_metrics.json", summaries)
        _write_json(output_dir / "aggregate_metrics.json", aggregate_rows)
        _write_summary_csv(output_dir / "summary.csv", summaries)
        _write_summary_csv(output_dir / "episode_metrics.csv", summaries)
        _write_summary_csv(output_dir / "per_agent_metrics.csv", per_agent_rows)
        _write_summary_csv(output_dir / "aggregate_metrics.csv", aggregate_rows)
        static_payload = {
            "workspace_bounds": experiment_config.workspace_bounds,
            "summaries": summaries,
            "aggregate": aggregate_rows,
            "traces": traces,
        }
        html_path.write_text(_html_document(static_payload), encoding="utf-8")
        live_state.update(
            status="completed",
            completed=completed_for_live,
            aggregate=aggregate_rows,
            message=f"全部验证完成，结果目录：{output_dir}",
        )
        print(f"验证结果：{output_dir}")
        print(f"离线回放：{html_path}")

        if server is not None:
            if args.keep_server:
                print("HTTP 服务保持运行，按 Ctrl+C 退出。")
                while True:
                    time.sleep(0.5)
            elif args.keep_server_seconds > 0:
                print(f"HTTP 服务将在 {args.keep_server_seconds:g} 秒后关闭。")
                time.sleep(float(args.keep_server_seconds))
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    main()
