"""Matched end-to-end PPO baseline for the frozen long-range benchmark.

The module deliberately contains no Proposal, FP-SHEP, GAT, R-ERR, or
SAC-DMP calls.  One shared policy sees one legal local observation per UAV and
emits a normalized three-axis acceleration command at every 0.1 s control
step.  The three commands are committed together through the existing
``step_direct_accelerations`` physical interface.
"""

from __future__ import annotations

import sys
import types
import importlib.machinery
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import gymnasium as gym
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
ALGO_ROOT = REPO_ROOT / "Multi-agent_Algo_lib"
SCRIPTS_ROOT = ALGO_ROOT / "scripts"
for _path in (REPO_ROOT, ALGO_ROOT, SCRIPTS_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from planning.final_four_stage_benchmark import step_direct_accelerations  # noqa: E402
from planning.semi_structured_long_range_benchmark import (  # noqa: E402
    LONG_RANGE_MAX_STEPS,
    SENSOR_RANGE_M,
    WORKSPACE_BOUNDS,
)
from Environment.multi_agent_dmp_env import (  # noqa: E402
    MultiAgentDMPEnv,
    _AgentAsDynamicObstacle,
)
from Entity.dynamic_obstacles import MovingSphereObstacle  # noqa: E402
from Entity.static_obstacles import (  # noqa: E402
    AxisAlignedBoxObstacle,
    StaticCylinderObstacle,
)


NUM_AGENTS = 3
ACTION_DIM = 3
ACTION_SCALE_MPS2 = 4.0
SPEED_SCALE_MPS = 4.0
PEER_SLOTS = 2
PEER_FEATURES_PER_SLOT = 7
SENSOR_DIRECTIONS = 256
SENSOR_FRAMES = 2
SENSOR_FEATURE_DIM = 3 + 3 + 1 + SENSOR_DIRECTIONS * SENSOR_FRAMES
OBSERVATION_DIM = SENSOR_FEATURE_DIM + PEER_SLOTS * PEER_FEATURES_PER_SLOT


@dataclass(frozen=True)
class PPODirectRewardConfig:
    """Compact navigation reward; simulator labels are training supervision."""

    progress_weight: float = 1.0
    goal_reach_bonus: float = 30.0
    time_step_penalty: float = 0.01
    team_collision_penalty: float = 100.0
    local_collision_penalty: float = 25.0
    control_effort_weight: float = 0.0


def make_multi_config(*, max_steps: int = LONG_RANGE_MAX_STEPS) -> dict[str, Any]:
    """Build the exact Formal-V2 physical and 16x16 sensing configuration."""

    return {
        "dynamics_config": {
            "velocity_clip": [-4.0, 4.0],
            "accelerate_clip": [-4.0, 4.0],
            "time_step": 0.1,
            "maximum_speed_norm": 3.2,
        },
        "sensor_config": {
            "sensing_radius": float(SENSOR_RANGE_M),
            "azimuth_bins": 16,
            "elevation_bins": 16,
            "elevation_range_deg": [-80.0, 80.0],
            "goal_distance_clip": 2.0 * float(SENSOR_RANGE_M),
            "include_previous_scan": True,
            "broad_phase_enabled": True,
        },
        "dmp_config": {
            "dt": 0.1,
            "dims": 3,
            "K_alpha": 3.0,
            "K_beta": 0.8,
            "alpha_s": 4.0,
            "tau": 2.5,
            "forcing_term_min": -10.0,
            "forcing_term_max": 10.0,
            "goal_offset_max": 1.0,
        },
        "env_config": {
            "num_agents": NUM_AGENTS,
            "max_steps": int(max_steps),
            "goal_tolerance": 0.30,
            "workspace_bounds": tuple(tuple(row) for row in WORKSPACE_BOUNDS),
            "randomize_start_goal": False,
            "start_position_bounds": None,
            "goal_position_bounds": None,
            "min_start_goal_distance": 0.0,
            "collision_margin": 0.0,
            "inter_agent_safe_distance": 0.60,
            "inter_agent_influence_distance": float(SENSOR_RANGE_M),
            "nearest_agent_observation_count": PEER_SLOTS,
            "peer_state_observation_mode": "local_anonymous_ally_block",
            "peer_state_observation_range": float(SENSOR_RANGE_M),
            "action_guidance_enabled": False,
        },
    }


def obstacles_from_entry(entry: Mapping[str, Any]) -> tuple[list[Any], list[Any]]:
    static: list[Any] = []
    dynamic: list[Any] = []
    for spec in entry["static_obstacles"]:
        kind = str(spec["type"])
        if kind == "box":
            static.append(
                AxisAlignedBoxObstacle(
                    center=np.asarray(spec["center"], dtype=float),
                    half_extents=np.asarray(spec["half_extents"], dtype=float),
                    safety_margin=float(spec.get("safety_margin", 0.0)),
                )
            )
        elif kind == "cylinder":
            static.append(
                StaticCylinderObstacle(
                    center=np.asarray(spec["center"], dtype=float),
                    radius=float(spec["radius"]),
                    half_height=float(spec["half_height"]),
                    safety_margin=float(spec.get("safety_margin", 0.0)),
                )
            )
        else:
            raise ValueError(f"unsupported static obstacle type: {kind}")
    for spec in entry["dynamic_obstacles"]:
        if str(spec["type"]) != "moving_sphere_constant_translation":
            raise ValueError(f"unsupported dynamic obstacle type: {spec['type']}")
        dynamic.append(
            MovingSphereObstacle(
                center=np.asarray(spec["center"], dtype=float),
                radius=float(spec["radius"]),
                velocity=np.asarray(spec["velocity"], dtype=float),
                safety_margin=float(spec.get("safety_margin", 0.0)),
                bounds=None,
            )
        )
    return static, dynamic


class PPODirectPhysicalEnv(MultiAgentDMPEnv):
    """Frozen environment plus the Formal-V2 peer-sphere LiDAR adapter."""

    def _sensor_dynamic_obstacles(self, agent_index: int) -> list[Any]:
        obstacles = list(super()._sensor_dynamic_obstacles(agent_index))
        for peer_index, dynamic in enumerate(self.dynamics):
            if peer_index == int(agent_index):
                continue
            obstacles.append(
                _AgentAsDynamicObstacle(
                    center=dynamic.p.copy(),
                    velocity=dynamic.v.copy(),
                    radius=0.30,
                    safety_margin=0.0,
                )
            )
        return obstacles


def make_environment(entry: Mapping[str, Any]) -> PPODirectPhysicalEnv:
    """Instantiate one world with the current frozen physics and local sensing."""

    kwargs = make_multi_config(max_steps=int(entry.get("max_steps", LONG_RANGE_MAX_STEPS)))
    static_obstacles, dynamic_obstacles = obstacles_from_entry(entry)
    env = PPODirectPhysicalEnv(
        **kwargs,
        static_obstacles=static_obstacles,
        dynamic_obstacles=dynamic_obstacles,
    )
    reset_environment(env, entry)
    return env


def reset_environment(env: PPODirectPhysicalEnv, entry: Mapping[str, Any]) -> None:
    static_obstacles, dynamic_obstacles = obstacles_from_entry(entry)
    env.reset(
        seed=int(entry["seed"]),
        options={
            "starts": np.asarray(entry["starts"], dtype=float),
            "goals": np.asarray(entry["goals"], dtype=float),
            "static_obstacles": static_obstacles,
            "dynamic_obstacles": dynamic_obstacles,
        },
    )


def build_local_observations(env: PPODirectPhysicalEnv) -> np.ndarray:
    """Return legal per-agent observations with no DMP or upper-planner state.

    The first 519 fields are the native local sensor packet.  Ego velocity is
    deterministically scaled by the native per-axis 4 m/s bound.  The final 14
    fields are the two anonymous, range-limited ally slots already specified
    by the frozen information contract.
    """

    rows: list[np.ndarray] = []
    for agent_id in range(int(env.num_agents)):
        packet = env.latest_sensor_packets[agent_id]
        if packet is None:
            raise RuntimeError("environment must be reset before observation assembly")
        sensor = packet.observation.astype(np.float32, copy=True)
        if sensor.shape != (SENSOR_FEATURE_DIM,):
            raise RuntimeError(
                f"expected {SENSOR_FEATURE_DIM} native sensor fields, got {sensor.shape}"
            )
        sensor[:3] = np.clip(sensor[:3] / SPEED_SCALE_MPS, -1.0, 1.0)
        peers = env._compose_inter_agent_observation(agent_id).astype(np.float32, copy=True)
        if peers.shape != (PEER_SLOTS * PEER_FEATURES_PER_SLOT,):
            raise RuntimeError(f"unexpected anonymous peer block shape: {peers.shape}")
        row = np.concatenate([sensor, peers], axis=0).astype(np.float32, copy=False)
        if row.shape != (OBSERVATION_DIM,) or not np.all(np.isfinite(row)):
            raise RuntimeError("invalid PPO-Direct local observation")
        rows.append(row)
    return np.stack(rows, axis=0).astype(np.float32)


def normalized_actions_to_accelerations(actions: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.shape != (NUM_AGENTS, ACTION_DIM):
        raise ValueError(f"actions must have shape {(NUM_AGENTS, ACTION_DIM)}")
    if not np.all(np.isfinite(actions)):
        raise ValueError("actions must be finite")
    return np.clip(actions, -1.0, 1.0).astype(float) * ACTION_SCALE_MPS2


def compute_rewards(
    info: Mapping[str, Any],
    actions: np.ndarray,
    previously_completed: np.ndarray,
    config: PPODirectRewardConfig,
) -> np.ndarray:
    """Compute local rewards without exposing privileged state to the policy."""

    progress = np.asarray(info["progress"], dtype=np.float32)
    new_success = np.asarray(info["new_success_mask"], dtype=bool)
    collision_mask = np.asarray(info["collision_mask"], dtype=bool)
    active = np.logical_not(np.asarray(previously_completed, dtype=bool))
    rewards = config.progress_weight * progress
    rewards -= float(config.time_step_penalty) * active.astype(np.float32)
    rewards += float(config.goal_reach_bonus) * new_success.astype(np.float32)
    if bool(info["collision"]):
        rewards[active] -= float(config.team_collision_penalty)
        rewards[collision_mask] -= float(config.local_collision_penalty)
    if float(config.control_effort_weight) > 0.0:
        rewards -= float(config.control_effort_weight) * np.sum(
            np.asarray(actions, dtype=np.float32) ** 2,
            axis=1,
        )
    rewards[np.logical_not(active)] = 0.0
    return rewards.astype(np.float32)


class PPODirectWorld:
    """One three-UAV world whose actions are committed atomically."""

    def __init__(
        self,
        entries: Sequence[Mapping[str, Any]],
        *,
        world_id: int = 0,
        world_stride: int = 1,
        reward_config: PPODirectRewardConfig | None = None,
    ) -> None:
        if not entries:
            raise ValueError("entries cannot be empty")
        self.entries = tuple(entries)
        self.world_id = int(world_id)
        self.world_stride = max(1, int(world_stride))
        self.cursor = self.world_id % len(self.entries)
        self.reward_config = reward_config or PPODirectRewardConfig()
        self.current_entry = self.entries[self.cursor]
        self.env = make_environment(self.current_entry)
        self.episode_index = 0

    def _advance_entry(self) -> None:
        self.cursor = (self.cursor + self.world_stride) % len(self.entries)
        self.current_entry = self.entries[self.cursor]

    def reset(self, *, advance: bool = False) -> np.ndarray:
        if advance:
            self._advance_entry()
        reset_environment(self.env, self.current_entry)
        self.episode_index += 1
        return build_local_observations(self.env)

    def step(
        self,
        actions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, bool, dict[str, Any]]:
        # All actions have already been inferred from build_local_observations()
        # at the same world state; conversion is side-effect free.
        normalized = np.clip(np.asarray(actions, dtype=np.float32), -1.0, 1.0)
        previously_completed = self.env.success_rewarded_mask.astype(bool, copy=True)
        accelerations = normalized_actions_to_accelerations(normalized)
        terminated, truncated, info = step_direct_accelerations(
            self.env,
            accelerations,
            refresh_sensors=True,
        )
        observations = build_local_observations(self.env)
        rewards = compute_rewards(info, normalized, previously_completed, self.reward_config)
        done = bool(terminated or truncated)
        info = dict(info)
        info.update(
            {
                "scenario_id": str(self.current_entry["scenario_id"]),
                "stage": str(self.current_entry["stage"]),
                "family": str(self.current_entry["family"]),
                "task_pattern": str(self.current_entry["task_pattern"]),
                "normalized_actions": normalized.copy(),
                "commanded_accelerations": accelerations.astype(np.float32),
                "team_done": done,
                "team_timeout": bool(truncated),
            }
        )
        return observations, rewards, done, info

    def close(self) -> None:
        self.env.close()


def observation_space() -> gym.spaces.Box:
    return gym.spaces.Box(
        low=np.full(OBSERVATION_DIM, -1.0, dtype=np.float32),
        high=np.full(OBSERVATION_DIM, 1.0, dtype=np.float32),
        dtype=np.float32,
    )


def action_space() -> gym.spaces.Box:
    return gym.spaces.Box(
        low=np.full(ACTION_DIM, -1.0, dtype=np.float32),
        high=np.full(ACTION_DIM, 1.0, dtype=np.float32),
        dtype=np.float32,
    )


def make_sb3_vec_env(
    entries: Sequence[Mapping[str, Any]],
    *,
    num_worlds: int,
    reward_config: PPODirectRewardConfig | None = None,
    threaded: bool = True,
) -> Any:
    """Create an SB3 VecEnv with agent rows and atomic three-agent worlds."""

    install_pandas_import_guard()
    from stable_baselines3.common.vec_env import VecEnv

    class _AtomicWorldVecEnv(VecEnv):
        def __init__(self) -> None:
            self.worlds = [
                PPODirectWorld(
                    entries,
                    world_id=index,
                    world_stride=int(num_worlds),
                    reward_config=reward_config,
                )
                for index in range(int(num_worlds))
            ]
            self.pending_actions: np.ndarray | None = None
            self.executor = (
                ThreadPoolExecutor(max_workers=int(num_worlds))
                if threaded and int(num_worlds) > 1
                else None
            )
            super().__init__(
                int(num_worlds) * NUM_AGENTS,
                observation_space(),
                action_space(),
            )

        def reset(self) -> np.ndarray:
            self.reset_infos = [{} for _ in range(self.num_envs)]
            return np.concatenate([world.reset(advance=False) for world in self.worlds], axis=0)

        def step_async(self, actions: np.ndarray) -> None:
            actions = np.asarray(actions, dtype=np.float32)
            expected = (self.num_envs, ACTION_DIM)
            if actions.shape != expected:
                raise ValueError(f"actions must have shape {expected}")
            self.pending_actions = actions.reshape(len(self.worlds), NUM_AGENTS, ACTION_DIM).copy()

        def step_wait(self):
            if self.pending_actions is None:
                raise RuntimeError("step_async must be called before step_wait")
            pairs = list(zip(self.worlds, self.pending_actions, strict=True))
            if self.executor is None:
                results = [world.step(actions) for world, actions in pairs]
            else:
                futures = [self.executor.submit(world.step, actions) for world, actions in pairs]
                results = [future.result() for future in futures]
            self.pending_actions = None
            obs_rows: list[np.ndarray] = []
            reward_rows: list[np.ndarray] = []
            done_rows: list[np.ndarray] = []
            info_rows: list[dict[str, Any]] = []
            for world, (obs, rewards, done, team_info) in zip(self.worlds, results, strict=True):
                terminal_obs = obs.copy()
                per_agent_infos = []
                for agent_id in range(NUM_AGENTS):
                    row = {
                        "agent_id": int(agent_id),
                        "scenario_id": team_info["scenario_id"],
                        "stage": team_info["stage"],
                        "success": bool(team_info["success"]),
                        "collision": bool(team_info["collision"]),
                        "peer_collision": bool(
                            np.asarray(team_info["inter_agent_collision_mask"])[agent_id]
                        ),
                        "obstacle_collision": bool(
                            np.asarray(team_info["obstacle_collision_mask"])[agent_id]
                        ),
                        "agent_completed": bool(
                            np.asarray(team_info["success_mask"])[agent_id]
                        ),
                    }
                    if agent_id == 0 and done:
                        row["team_episode"] = {
                            "scenario_id": team_info["scenario_id"],
                            "stage": team_info["stage"],
                            "success": bool(team_info["success"]),
                            "collision": bool(team_info["collision"]),
                            "peer_collision": bool(
                                np.any(team_info["inter_agent_collision_mask"])
                            ),
                            "timeout": bool(team_info["team_timeout"]),
                            "steps": int(team_info["steps"]),
                            "agent_completion": float(
                                np.mean(np.asarray(team_info["success_mask"], dtype=float))
                            ),
                        }
                    if done:
                        row["terminal_observation"] = terminal_obs[agent_id]
                        row["TimeLimit.truncated"] = bool(team_info["team_timeout"])
                    per_agent_infos.append(row)
                if done:
                    obs = world.reset(advance=True)
                obs_rows.append(obs)
                reward_rows.append(rewards)
                done_rows.append(np.full(NUM_AGENTS, done, dtype=bool))
                info_rows.extend(per_agent_infos)
            return (
                np.concatenate(obs_rows, axis=0).astype(np.float32),
                np.concatenate(reward_rows, axis=0).astype(np.float32),
                np.concatenate(done_rows, axis=0),
                info_rows,
            )

        def close(self) -> None:
            if self.executor is not None:
                self.executor.shutdown(wait=True)
            for world in self.worlds:
                world.close()

        def get_attr(self, attr_name: str, indices=None):
            indices = self._get_indices(indices)
            return [getattr(self.worlds[index // NUM_AGENTS], attr_name) for index in indices]

        def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
            for index in self._get_indices(indices):
                setattr(self.worlds[index // NUM_AGENTS], attr_name, value)

        def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
            return [
                getattr(self.worlds[index // NUM_AGENTS], method_name)(*method_args, **method_kwargs)
                for index in self._get_indices(indices)
            ]

        def env_is_wrapped(self, wrapper_class, indices=None):
            return [False for _ in self._get_indices(indices)]

    return _AtomicWorldVecEnv()


__all__ = [
    "ACTION_DIM",
    "ACTION_SCALE_MPS2",
    "NUM_AGENTS",
    "OBSERVATION_DIM",
    "PPODirectRewardConfig",
    "PPODirectWorld",
    "build_local_observations",
    "make_environment",
    "make_multi_config",
    "make_sb3_vec_env",
    "normalized_actions_to_accelerations",
    "install_pandas_import_guard",
]
def install_pandas_import_guard() -> None:
    """Avoid a broken optional pandas/pyarrow binary in the RL environment.

    Stable-Baselines3 imports pandas only for optional log readers.  PPO-Direct
    writes its own CSV records and never calls those readers, so a tiny
    process-local module is sufficient and does not alter training semantics.
    """

    if "pandas" in sys.modules:
        return
    module = types.ModuleType("pandas")
    module.__spec__ = importlib.machinery.ModuleSpec("pandas", loader=None)
    module.__version__ = "0.0.0-ppo-direct-import-guard"
    module.DataFrame = type("DataFrame", (), {})

    def _unavailable(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("optional pandas log reader is unavailable in this process")

    module.read_csv = _unavailable
    module.concat = _unavailable
    sys.modules["pandas"] = module
