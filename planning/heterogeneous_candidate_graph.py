"""Typed local heterogeneous graph for candidate execution assessment.

This module only constructs graph data.  It contains no neural encoder,
message-passing layer, attention mechanism, candidate selector, or training
logic.  The implementation convention follows the current point-mass code:
the local frame is ego-centred and aligned with the world axes.  No yaw,
heading, or body rotation is invented.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch_geometric.data import HeteroData

from Guidance.reference_point_proposal_demo import (
    ProposalConfig,
    compute_sector_safety_field,
)


GAT_CANONICAL_AZIMUTH_BINS = 8
GAT_CANONICAL_ELEVATION_BINS = 7
GAT_CANONICAL_ELEVATION_RANGE_DEG = (-80.0, 80.0)


def _canonical_gat_sector_projection(
    safety: np.ndarray,
    sensor_directions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project a denser scan onto the frozen GAT's original 8x7 directions."""
    safety = np.asarray(safety, dtype=float).reshape(-1)
    sensor_directions = np.asarray(sensor_directions, dtype=float).reshape(-1, 3)
    if safety.size != sensor_directions.shape[0]:
        raise ValueError("sector safety and direction counts do not match")
    canonical_azimuth = np.linspace(
        -np.pi, np.pi, GAT_CANONICAL_AZIMUTH_BINS, endpoint=False, dtype=float
    )
    canonical_elevation = np.deg2rad(
        np.linspace(
            GAT_CANONICAL_ELEVATION_RANGE_DEG[0],
            GAT_CANONICAL_ELEVATION_RANGE_DEG[1],
            GAT_CANONICAL_ELEVATION_BINS,
            dtype=float,
        )
    )
    canonical = []
    for azimuth in canonical_azimuth:
        for elevation in canonical_elevation:
            cosine = float(np.cos(elevation))
            canonical.append(
                [
                    cosine * float(np.cos(azimuth)),
                    cosine * float(np.sin(azimuth)),
                    float(np.sin(elevation)),
                ]
            )
    canonical_directions = np.asarray(canonical, dtype=float)
    nearest = np.argmax(canonical_directions @ sensor_directions.T, axis=1)
    if len(set(nearest.astype(int).tolist())) != canonical_directions.shape[0]:
        raise RuntimeError("canonical GAT sector projection is not one-to-one")
    return safety[nearest].copy(), canonical_directions
from planning.candidate_execution_interface import (
    ExecutionNormalizationSpec,
    GraphReadyCandidateExecution,
    constant_velocity_conflict_diagnostic,
    normalize_execution_features,
)


GRAPH_SCHEMA_VERSION = "heterogeneous_candidate_graph_v1"
LOCAL_FRAME_CONVENTION = "ego_centered_world_axis_aligned"
ANGLE_REPRESENTATION = "unit_direction_vector_xyz"
DEFAULT_D_ALIGN_SOURCE = (
    "configurable_implementation_parameter_not_paper_constant"
)
EPS = 1.0e-8


@dataclass(frozen=True)
class HeterogeneousCandidateGraphConfig:    # 构造config参数类
    """Fixed graph-construction and finite-normalization parameters."""

    horizon_steps: int = 4
    dt: float = 0.1
    d_safe: float = 0.6
    d_align: float = 1.2
    d_safe_source: str = "MultiAgentEnvConfig.inter_agent_safe_distance"
    d_align_source: str = DEFAULT_D_ALIGN_SOURCE
    candidate_distance_scale: float = 1.05
    task_goal_distance_scale: float = 9.0
    velocity_scale: float = 4.0
    relative_velocity_scale: float = 8.0
    align_distance_scale: float = 1.2
    schema_version: str = GRAPH_SCHEMA_VERSION

    def __post_init__(self) -> None:    # 参数合法化检查
        if int(self.horizon_steps) <= 0:
            raise ValueError("horizon_steps must be positive")
        positive = (
            self.dt,
            self.d_safe,
            self.d_align,
            self.candidate_distance_scale,
            self.task_goal_distance_scale,
            self.velocity_scale,
            self.relative_velocity_scale,
            self.align_distance_scale,
        )
        if any(not np.isfinite(float(value)) or float(value) <= 0.0 for value in positive):
            raise ValueError("graph thresholds, dt, and scales must be positive and finite")
        if not str(self.d_safe_source) or not str(self.d_align_source):
            raise ValueError("threshold provenance must be non-empty")


@dataclass(frozen=True)
class EgoGraphState:
    agent_id: int
    position: np.ndarray
    velocity: np.ndarray
    previous_velocity: np.ndarray
    task_goal: np.ndarray
    sector_safety: np.ndarray
    goal_sector_id: int

    def __post_init__(self) -> None:
        # object: 因为frozen 类，不能修改属性的值，所以只能用object修改
        for name in ("position", "velocity", "previous_velocity", "task_goal"):
            value = _readonly_vector3(getattr(self, name), name)
            object.__setattr__(self, name, value)
        safety = np.asarray(self.sector_safety, dtype=float)
        if safety.ndim != 1 or safety.size == 0 or not np.all(np.isfinite(safety)):
            raise ValueError("sector_safety must be a non-empty finite vector")
        safety = safety.copy()
        safety.setflags(write=False)
        sector_id = int(self.goal_sector_id)
        if not 0 <= sector_id < safety.size:
            raise ValueError("goal_sector_id is outside sector_safety")
        object.__setattr__(self, "agent_id", int(self.agent_id))
        object.__setattr__(self, "sector_safety", safety)
        object.__setattr__(self, "goal_sector_id", sector_id)


@dataclass(frozen=True)
class NeighborGraphState:   # 单机局部图对象
    agent_id: int
    position: np.ndarray
    velocity: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent_id", int(self.agent_id))
        object.__setattr__(
            self, "position", _readonly_vector3(self.position, "neighbor.position")
        )
        object.__setattr__(
            self, "velocity", _readonly_vector3(self.velocity, "neighbor.velocity")
        )

def _readonly_vector3(value: Any, name: str) -> np.ndarray:
    # 创建一个只读的vector3,用于检查解耦的三维状态
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape (3,)")
    result = result.copy()
    result.setflags(write=False)
    return result


def world_to_ego_local(world_position: Any, ego_world_position: Any) -> np.ndarray:
    """Transform into the current point-mass implementation's local frame."""
    # 转换到当前点群实现中的局部坐标系
    world = np.asarray(world_position, dtype=float)
    ego = np.asarray(ego_world_position, dtype=float)
    if world.shape[-1:] != (3,) or ego.shape != (3,):
        raise ValueError("world positions must end in dimension 3 and ego must have shape (3,)")
    if not np.all(np.isfinite(world)) or not np.all(np.isfinite(ego)):
        raise ValueError("world-to-local inputs must be finite")
    return (world - ego).copy()

def _unit(vector: np.ndarray) -> np.ndarray:    # 归一化向量, 避免除零
    norm = float(np.linalg.norm(vector))
    return np.zeros(3, dtype=float) if norm < EPS else vector / norm

def _finite_clip(value: np.ndarray, low: float, high: float) -> np.ndarray: # 控制数值稳定,将极端值或未定义值赋为0
    result = np.asarray(value, dtype=float).copy()
    result[~np.isfinite(result)] = 0.0
    return np.clip(result, low, high)

def _as_neighbor(value: Any) -> NeighborGraphState: # 输入适配器，将临机数据转换为NeighborGraphState对象
    if isinstance(value, NeighborGraphState):
        return value
    if isinstance(value, Mapping):
        return NeighborGraphState(
            agent_id=value["agent_id"],
            position=value["position"],
            velocity=value["velocity"],
        )
    return NeighborGraphState(
        agent_id=getattr(value, "agent_id"),
        position=getattr(value, "position"),
        velocity=getattr(value, "velocity"),
    )


def _tensor(rows: np.ndarray, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:  # 数据类型创建为张量 
    # Some schema inputs are intentionally read-only.  Materialize one writable
    # copy before crossing the NumPy/Torch boundary to avoid undefined write
    # semantics in ``torch.as_tensor``.
    return torch.as_tensor(np.array(rows, copy=True), dtype=dtype)


def _proposal_feature_schema() -> tuple[str, ...]:  # 创建输入特征向量的结构
    return (
        "local_position_x",
        "local_position_y",
        "local_position_z",
        "candidate_distance",
        "sector_direction_x",
        "sector_direction_y",
        "sector_direction_z",
        "geometric_task_progress",
        "sector_safety",
        "preview_task_progress",
        "preview_min_clearance",
        "preview_max_execution_deviation",
        "preview_terminal_speed",
    )


def feature_schema_table(sector_count: int) -> list[dict[str, Any]]:
    """Return the explicit dimension/unit/source contract for every feature."""

    sector_count = int(sector_count)
    rows = [
        {"owner": "null", "name": "goal_direction", "dimension": 3, "unit": "unitless", "source": "normalize(task_goal-ego_position)", "representation": "raw+normalized"},
        {"owner": "null", "name": "goal_distance", "dimension": 1, "unit": "m", "source": "norm(task_goal-ego_position)", "representation": "raw+normalized"},
        {"owner": "null", "name": "goal_sector_safety", "dimension": 1, "unit": "unitless", "source": "proposal sector normalized_margin", "representation": "raw+normalized"},
        {"owner": "agent", "name": "current_velocity", "dimension": 3, "unit": "m/s", "source": "current real state", "representation": "raw+normalized"},
        {"owner": "agent", "name": "goal_distance", "dimension": 1, "unit": "m", "source": "norm(task_goal-ego_position)", "representation": "raw+normalized"},
        {"owner": "agent", "name": "goal_direction", "dimension": 3, "unit": "unitless", "source": "normalize(task_goal-ego_position)", "representation": "raw+normalized"},
        {"owner": "agent", "name": "previous_velocity", "dimension": 3, "unit": "m/s", "source": "previous real execution state", "representation": "raw+normalized"},
        {"owner": "agent", "name": "sector_safety", "dimension": sector_count, "unit": "unitless", "source": "current real LiDAR proposal sector field", "representation": "raw+normalized"},
        {"owner": "proposal", "name": "local_position", "dimension": 3, "unit": "m", "source": "world_to_ego_local(candidate)", "representation": "raw+normalized"},
        {"owner": "proposal", "name": "candidate_distance", "dimension": 1, "unit": "m", "source": "norm(local_position)", "representation": "raw+normalized"},
        {"owner": "proposal", "name": "sector_direction", "dimension": 3, "unit": "unitless", "source": "Proposal.direction", "representation": "raw+normalized"},
        {"owner": "proposal", "name": "geometric_task_progress", "dimension": 1, "unit": "m", "source": "Proposal.distance_progress", "representation": "raw+normalized"},
        {"owner": "proposal", "name": "sector_safety", "dimension": 1, "unit": "unitless", "source": "Proposal.normalized_margin", "representation": "raw+normalized"},
        {"owner": "proposal", "name": "preview_task_progress", "dimension": 1, "unit": "m", "source": "GraphReadyCandidateExecution.task_progress_raw", "representation": "raw+normalized"},
        {"owner": "proposal", "name": "preview_min_clearance", "dimension": 1, "unit": "m", "source": "FP-SHEP approximate frozen-LiDAR clearance", "representation": "raw+normalized"},
        {"owner": "proposal", "name": "preview_max_execution_deviation", "dimension": 1, "unit": "m", "source": "GraphReadyCandidateExecution.max_execution_deviation_raw", "representation": "raw+normalized"},
        {"owner": "proposal", "name": "preview_terminal_speed", "dimension": 1, "unit": "m/s", "source": "GraphReadyCandidateExecution.terminal_speed_raw", "representation": "raw+normalized"},
        {"owner": "align", "name": "relative_direction", "dimension": 3, "unit": "unitless", "source": "normalize(neighbor_position-ego_position)", "representation": "raw+normalized"},
        {"owner": "align", "name": "relative_distance", "dimension": 1, "unit": "m", "source": "norm(neighbor_position-ego_position)", "representation": "raw+normalized"},
        {"owner": "align", "name": "relative_velocity", "dimension": 3, "unit": "m/s", "source": "neighbor_velocity-ego_velocity", "representation": "raw+normalized"},
        {"owner": "agent->proposal", "name": "smoothness", "dimension": 1, "unit": "unitless", "source": "velocity/candidate cosine", "representation": "raw"},
        {"owner": "align->proposal", "name": "time_to_minimum_separation", "dimension": 1, "unit": "s", "source": "constant_velocity_conflict_diagnostic", "representation": "raw+normalized"},
        {"owner": "align->proposal", "name": "minimum_separation", "dimension": 1, "unit": "m", "source": "constant_velocity_conflict_diagnostic", "representation": "raw+normalized"},
        {"owner": "align->proposal", "name": "risk_duration", "dimension": 1, "unit": "s", "source": "constant_velocity_conflict_diagnostic(distance < d_safe)", "representation": "raw+normalized"},
    ]
    return rows


def build_heterogeneous_candidate_graph(
    *,
    ego: EgoGraphState,
    proposals: Sequence[Any],
    executions: Sequence[GraphReadyCandidateExecution],
    neighbors: Sequence[Any],
    config: HeterogeneousCandidateGraphConfig | None = None,
    execution_normalization: ExecutionNormalizationSpec | None = None,
    timing_sink: dict[str, float] | None = None,
) -> HeteroData:
    """Build one ego-local typed graph without mutating or reordering inputs."""

    # 读取配置与输入
    graph_started = time.perf_counter_ns()
    config = config or HeterogeneousCandidateGraphConfig()
    proposals = tuple(proposals)
    executions = tuple(executions)
    neighbors = tuple(_as_neighbor(item) for item in neighbors)

    # 验证输入有效性
    if len(proposals) != len(executions):
        raise ValueError("proposals and executions must have identical lengths")
    if len({item.candidate_id for item in executions}) != len(executions):
        raise ValueError("candidate_id values must be unique")
    if len({item.agent_id for item in neighbors}) != len(neighbors):
        raise ValueError("neighbor agent_id values must be unique")

    # 组织数据
    goal_local = world_to_ego_local(ego.task_goal, ego.position)
    goal_distance = float(np.linalg.norm(goal_local))
    goal_direction = _unit(goal_local)
    goal_sector_safety = float(ego.sector_safety[ego.goal_sector_id])
    sector_count = int(ego.sector_safety.size)

    # 创建数据行

    # 空节点原始\归一化
    null_raw = np.concatenate([goal_direction, [goal_distance, goal_sector_safety]])[None, :]
    null_norm = np.concatenate([
        goal_direction,
        [np.clip(goal_distance / config.task_goal_distance_scale, 0.0, 1.0)],
        [np.clip(goal_sector_safety, 0.0, 1.0)],
    ])[None, :]

    # agent节点原始\归一化
    agent_raw = np.concatenate([
        ego.velocity,
        [goal_distance],
        goal_direction,
        ego.previous_velocity,
        ego.sector_safety,
    ])[None, :]
    agent_norm = np.concatenate([
        _finite_clip(ego.velocity / config.velocity_scale, -1.0, 1.0),
        [np.clip(goal_distance / config.task_goal_distance_scale, 0.0, 1.0)],
        goal_direction,
        _finite_clip(ego.previous_velocity / config.velocity_scale, -1.0, 1.0),
        _finite_clip(ego.sector_safety, 0.0, 1.0),
    ])[None, :]

    proposal_raw_rows: list[np.ndarray] = []
    proposal_norm_rows: list[np.ndarray] = []
    normalized_execution_rows: list[np.ndarray] = []
    local_positions: list[np.ndarray] = []
    world_positions: list[np.ndarray] = []

    # 
    for index, (proposal, execution) in enumerate(zip(proposals, executions)):  # 
        world_position = _readonly_vector3(proposal.point, f"proposal[{index}].point")
        if not np.allclose(
            world_position,
            execution.candidate_world_position,
            rtol=0.0,
            atol=1.0e-7,
        ):
            raise ValueError(f"proposal/execution world-position mismatch at index {index}")
        if int(execution.requested_horizon_steps) != int(config.horizon_steps):
            raise ValueError("execution requested horizon does not match graph config")
        local = world_to_ego_local(world_position, ego.position)
        distance = float(np.linalg.norm(local))
        if not np.isclose(distance, np.linalg.norm(world_position - ego.position)):
            raise RuntimeError("local/world candidate distance invariant failed")
        direction = _readonly_vector3(proposal.direction, f"proposal[{index}].direction")
        normalized_execution = normalize_execution_features(
            execution, execution_normalization
        )
        raw_row = np.concatenate([
            local,
            [distance],
            direction,
            [float(proposal.distance_progress), float(proposal.normalized_margin)],
            execution.raw_feature_vector,
        ])
        norm_row = np.concatenate([
            _finite_clip(local / config.candidate_distance_scale, -1.0, 1.0),
            [np.clip(distance / config.candidate_distance_scale, 0.0, 1.0)],
            _finite_clip(direction, -1.0, 1.0),
            [np.clip(float(proposal.distance_progress) / config.candidate_distance_scale, -1.0, 1.0)],
            [np.clip(float(proposal.normalized_margin), 0.0, 1.0)],
            normalized_execution.values,
        ])
        proposal_raw_rows.append(raw_row)
        proposal_norm_rows.append(norm_row)
        normalized_execution_rows.append(normalized_execution.values)
        local_positions.append(local)
        world_positions.append(world_position)

    proposal_dim = len(_proposal_feature_schema())
    proposal_raw = np.stack(proposal_raw_rows) if proposal_raw_rows else np.empty((0, proposal_dim))
    proposal_norm = np.stack(proposal_norm_rows) if proposal_norm_rows else np.empty((0, proposal_dim))
    local_position_array = np.stack(local_positions) if local_positions else np.empty((0, 3))
    world_position_array = np.stack(world_positions) if world_positions else np.empty((0, 3))

    align_raw_rows: list[np.ndarray] = []
    align_norm_rows: list[np.ndarray] = []
    for neighbor in neighbors:
        relative_position = world_to_ego_local(neighbor.position, ego.position)
        relative_distance = float(np.linalg.norm(relative_position))
        relative_direction = _unit(relative_position)
        relative_velocity = neighbor.velocity - ego.velocity
        align_raw_rows.append(
            np.concatenate([relative_direction, [relative_distance], relative_velocity])
        )
        align_norm_rows.append(
            np.concatenate([
                relative_direction,
                [np.clip(relative_distance / config.align_distance_scale, 0.0, 1.0)],
                _finite_clip(relative_velocity / config.relative_velocity_scale, -1.0, 1.0),
            ])
        )
    align_raw = np.stack(align_raw_rows) if align_raw_rows else np.empty((0, 7))
    align_norm = np.stack(align_norm_rows) if align_norm_rows else np.empty((0, 7))

    data = HeteroData()
    for node_type, raw, normalized in (
        ("null", null_raw, null_norm),
        ("agent", agent_raw, agent_norm),
        ("proposal", proposal_raw, proposal_norm),
        ("align", align_raw, align_norm),
    ):
        data[node_type].x_raw = _tensor(raw)
        data[node_type].x_normalized = _tensor(normalized)
        data[node_type].x = data[node_type].x_normalized

    edge_started = time.perf_counter_ns()
    candidate_count = len(proposals)
    smooth_edge_index = np.vstack([
        np.zeros(candidate_count, dtype=np.int64),
        np.arange(candidate_count, dtype=np.int64),
    ])
    smooth_values: list[list[float]] = []
    speed = float(np.linalg.norm(ego.velocity))
    for local in local_positions:
        distance = float(np.linalg.norm(local))
        denominator = speed * distance
        smoothness = 0.0 if denominator < EPS else float(np.dot(ego.velocity, local) / denominator)
        smooth_values.append([float(np.clip(smoothness, -1.0, 1.0))])
    smooth_edge_attr = np.asarray(smooth_values, dtype=float).reshape(candidate_count, 1)
    smooth_store = data["agent", "smooth", "proposal"]
    smooth_store.edge_index = _tensor(smooth_edge_index, dtype=torch.long)
    smooth_store.edge_attr = _tensor(smooth_edge_attr)

    conflict_sources: list[int] = []
    conflict_targets: list[int] = []
    conflict_raw: list[list[float]] = []
    conflict_norm: list[list[float]] = []
    minimum_step_indices: list[int] = []
    candidate_minimum_positions: list[np.ndarray] = []
    neighbor_minimum_positions: list[np.ndarray] = []
    for neighbor_node, neighbor in enumerate(neighbors):
        for proposal_node, execution in enumerate(executions):
            if execution.preview_positions.shape[0] == 0:
                continue
            diagnostic = constant_velocity_conflict_diagnostic(
                candidate_preview_positions=execution.preview_positions,
                neighbor_current_position=neighbor.position,
                neighbor_current_velocity=neighbor.velocity,
                dt=config.dt,
                risk_separation_threshold=config.d_safe,
            )
            if diagnostic.minimum_separation >= config.d_align:
                continue
            conflict_sources.append(neighbor_node)
            conflict_targets.append(proposal_node)
            conflict_raw.append([
                diagnostic.time_to_minimum_separation,
                diagnostic.minimum_separation,
                diagnostic.risk_duration,
            ])
            conflict_norm.append([
                np.clip(diagnostic.time_to_minimum_separation / (config.horizon_steps * config.dt), 0.0, 1.0),
                np.clip(diagnostic.minimum_separation / config.d_align, 0.0, 1.0),
                np.clip(diagnostic.risk_duration / (config.horizon_steps * config.dt), 0.0, 1.0),
            ])
            minimum_index = int(np.argmin(diagnostic.per_step_distance))
            minimum_step_indices.append(minimum_index)
            candidate_minimum_positions.append(execution.preview_positions[minimum_index])
            step = float(minimum_index + 1)
            neighbor_minimum_positions.append(
                neighbor.position + step * config.dt * neighbor.velocity
            )

    conflict_count = len(conflict_raw)
    conflict_edge_index = np.asarray(
        [conflict_sources, conflict_targets], dtype=np.int64
    ).reshape(2, conflict_count)
    conflict_raw_array = np.asarray(conflict_raw, dtype=float).reshape(conflict_count, 3)
    conflict_norm_array = np.asarray(conflict_norm, dtype=float).reshape(conflict_count, 3)
    conflict_store = data["align", "spatiotemporal", "proposal"]
    conflict_store.edge_index = _tensor(conflict_edge_index, dtype=torch.long)
    conflict_store.edge_attr = _tensor(conflict_raw_array)
    conflict_store.edge_attr_normalized = _tensor(conflict_norm_array)
    conflict_store.minimum_step_index = _tensor(
        np.asarray(minimum_step_indices, dtype=np.int64), dtype=torch.long
    )
    conflict_store.candidate_minimum_position = _tensor(
        np.asarray(candidate_minimum_positions, dtype=float).reshape(conflict_count, 3)
    )
    conflict_store.neighbor_minimum_position = _tensor(
        np.asarray(neighbor_minimum_positions, dtype=float).reshape(conflict_count, 3)
    )
    edge_ns = time.perf_counter_ns() - edge_started

    proposal_store = data["proposal"]
    proposal_store.candidate_id = _tensor(
        np.asarray([item.candidate_id for item in executions], dtype=np.int64),
        dtype=torch.long,
    )
    proposal_store.original_index = torch.arange(candidate_count, dtype=torch.long)
    proposal_store.world_position = _tensor(world_position_array)
    proposal_store.local_position = _tensor(local_position_array)
    proposal_store.proposal_score = _tensor(
        np.asarray([float(item.score) for item in proposals], dtype=float)
    )
    proposal_store.sector_index = _tensor(
        np.asarray(
            [[int(item.azimuth_index), int(item.elevation_index)] for item in proposals],
            dtype=np.int64,
        ).reshape(candidate_count, 2),
        dtype=torch.long,
    )
    proposal_store.execution_feature_raw = _tensor(
        np.asarray([item.raw_feature_vector for item in executions], dtype=float).reshape(candidate_count, 4)
    )
    proposal_store.execution_feature_normalized = _tensor(
        np.asarray(normalized_execution_rows, dtype=float).reshape(candidate_count, 4)
    )
    proposal_store.feature_valid_mask = _tensor(
        np.asarray([item.feature_valid_mask for item in executions], dtype=bool).reshape(candidate_count, 4),
        dtype=torch.bool,
    )
    proposal_store.feature_full_horizon_mask = _tensor(
        np.asarray([item.feature_full_horizon_mask for item in executions], dtype=bool).reshape(candidate_count, 4),
        dtype=torch.bool,
    )
    proposal_store.clearance_finite_mask = _tensor(
        np.asarray([item.clearance_finite_mask for item in executions], dtype=bool),
        dtype=torch.bool,
    )
    proposal_store.open_space_flag = _tensor(
        np.asarray([item.open_space_flag for item in executions], dtype=bool),
        dtype=torch.bool,
    )
    proposal_store.preview_completed = _tensor(
        np.asarray([item.preview_completed for item in executions], dtype=bool),
        dtype=torch.bool,
    )

    data["null"].goal_sector_id = torch.as_tensor([ego.goal_sector_id], dtype=torch.long)
    data["agent"].agent_id = torch.as_tensor([ego.agent_id], dtype=torch.long)
    data["agent"].world_position = _tensor(ego.position[None, :])
    data["agent"].current_velocity = _tensor(ego.velocity[None, :])
    data["agent"].previous_velocity = _tensor(ego.previous_velocity[None, :])
    data["agent"].task_goal = _tensor(ego.task_goal[None, :])
    data["align"].agent_id = _tensor(
        np.asarray([item.agent_id for item in neighbors], dtype=np.int64),
        dtype=torch.long,
    )

    candidate_ids = [int(item.candidate_id) for item in executions]
    data.graph_metadata = {
        "feature_schema_version": config.schema_version,
        "normalization_spec_version": (
            execution_normalization or ExecutionNormalizationSpec()
        ).version,
        "normalization_scales": {
            "candidate_distance": float(config.candidate_distance_scale),
            "task_goal_distance": float(config.task_goal_distance_scale),
            "velocity": float(config.velocity_scale),
            "relative_velocity": float(config.relative_velocity_scale),
            "align_distance": float(config.align_distance_scale),
            **{
                name: float(value)
                for name, value in vars(
                    execution_normalization or ExecutionNormalizationSpec()
                ).items()
                if name.endswith("_scale")
            },
        },
        "local_frame_convention": LOCAL_FRAME_CONVENTION,
        "local_frame_rotation_applied": False,
        "angle_representation": ANGLE_REPRESENTATION,
        "H": int(config.horizon_steps),
        "dt": float(config.dt),
        "d_safe": float(config.d_safe),
        "d_safe_source": config.d_safe_source,
        "d_safe_risk_inequality": "distance < d_safe",
        "d_align": float(config.d_align),
        "d_align_source": config.d_align_source,
        "d_align_edge_inequality": "minimum_separation < d_align",
        "candidate_order_preserved": True,
        "neighbor_order_source": "environment_observation_membership_and_order; ids_may_be_event_local_slots",
        "candidate_count": candidate_count,
        "neighbor_count": len(neighbors),
        "goal_sector_id": int(ego.goal_sector_id),
        "clearance_provenance": [item.obstacle_clearance_source for item in executions],
        "clearance_is_approximate": [item.obstacle_clearance_is_approximate for item in executions],
        "termination_reason": [item.termination_reason for item in executions],
        "feature_schema": feature_schema_table(sector_count),
        "proposal_feature_names": list(_proposal_feature_schema()),
    }
    data.proposal_node_to_candidate_id = tuple(candidate_ids)
    data.proposal_node_to_original_index = tuple(range(candidate_count))
    data.candidate_id_to_proposal_node = {
        candidate_id: index for index, candidate_id in enumerate(candidate_ids)
    }
    data.original_proposals = proposals
    data.neighbor_node_to_agent_id = tuple(item.agent_id for item in neighbors)
    data.candidate_preview_positions = tuple(
        item.preview_positions for item in executions
    )
    data.candidate_preview_velocities = tuple(
        item.preview_velocities for item in executions
    )
    if timing_sink is not None:
        total_ns = time.perf_counter_ns() - graph_started
        timing_sink.update(
            {
                "graph_feature_build_ms": max(0.0, (total_ns - edge_ns) / 1.0e6),
                "graph_edge_build_ms": edge_ns / 1.0e6,
                "graph_total_cpu_ms": total_ns / 1.0e6,
            }
        )
    return data


def build_heterogeneous_candidate_graph_from_env(
    *,
    env: Any,
    agent_index: int,
    proposals: Sequence[Any],
    executions: Sequence[GraphReadyCandidateExecution],
    proposal_config: ProposalConfig,
    config: HeterogeneousCandidateGraphConfig | None = None,
    execution_normalization: ExecutionNormalizationSpec | None = None,
    timing_sink: dict[str, float] | None = None,
) -> HeteroData:
    """Read only the currently observable state and delegate to the pure builder."""

    adapter_started = time.perf_counter_ns()
    agent_index = int(agent_index)
    packet = env.latest_sensor_packets[agent_index]
    if packet is None:
        raise RuntimeError("environment must be reset before graph construction")
    sector_field = compute_sector_safety_field(
        env.dynamics[agent_index].p,
        env.goals[agent_index],
        env.dynamics[agent_index].v,
        packet,
        env.sensors[agent_index],
        proposal_config,
        env.env_config.goal_tolerance,
    )
    goal_vector = np.asarray(env.goals[agent_index], dtype=float) - np.asarray(
        env.dynamics[agent_index].p, dtype=float
    )
    goal_direction = _unit(goal_vector)
    directions = np.asarray(env.sensors[agent_index].ray_directions, dtype=float)
    flat_directions = directions.reshape(-1, 3)
    graph_sector_safety, graph_directions = _canonical_gat_sector_projection(
        sector_field.normalized_margin, flat_directions
    )
    goal_sector_id = int(np.argmax(graph_directions @ goal_direction))
    ego = EgoGraphState(
        agent_id=agent_index,
        position=env.dynamics[agent_index].p,
        velocity=env.dynamics[agent_index].v,
        previous_velocity=env.previous_velocities[agent_index],
        task_goal=env.goals[agent_index],
        sector_safety=graph_sector_safety,
        goal_sector_id=goal_sector_id,
    )
    resolved = config or HeterogeneousCandidateGraphConfig()
    resolved = replace(
        resolved,
        dt=float(env.dynamics[agent_index].dt),
        d_safe=float(env.env_config.inter_agent_safe_distance),
        d_safe_source="MultiAgentEnvConfig.inter_agent_safe_distance",
    )
    adapter_ms = (time.perf_counter_ns() - adapter_started) / 1.0e6
    graph_timing: dict[str, float] = {}
    result = build_heterogeneous_candidate_graph(
        ego=ego,
        proposals=proposals,
        executions=executions,
        neighbors=env.observable_neighbor_states(agent_index),
        config=resolved,
        execution_normalization=execution_normalization,
        timing_sink=graph_timing,
    )
    if timing_sink is not None:
        timing_sink.update(graph_timing)
        timing_sink["graph_feature_build_ms"] = float(
            timing_sink.get("graph_feature_build_ms", 0.0) + adapter_ms
        )
        timing_sink["graph_total_cpu_ms"] = float(
            timing_sink.get("graph_total_cpu_ms", 0.0) + adapter_ms
        )
    return result


def graph_debug_summary(data: HeteroData) -> str:
    """Create a deterministic, switchable human-readable graph summary."""

    metadata = data.graph_metadata
    lines = [
        "=== Heterogeneous Candidate Graph ===",
        f"schema: {metadata['feature_schema_version']}",
        f"frame: {metadata['local_frame_convention']}",
        "ego:",
        f"  agent_id = {int(data['agent'].agent_id[0])}",
        f"  position = {data['agent'].world_position[0].tolist()}",
        f"  velocity = {data['agent'].current_velocity[0].tolist()}",
        f"  previous_velocity = {data['agent'].previous_velocity[0].tolist()}",
        f"  task_goal = {data['agent'].task_goal[0].tolist()}",
        f"  goal_sector_id = {metadata['goal_sector_id']}",
        "nodes:",
        f"  null = {data['null'].num_nodes}",
        f"  agent = {data['agent'].num_nodes}",
        f"  proposals = {data['proposal'].num_nodes}",
        f"  align = {data['align'].num_nodes}",
        "proposal nodes:",
    ]
    for node in range(data["proposal"].num_nodes):
        candidate_id = int(data["proposal"].candidate_id[node])
        original_index = int(data["proposal"].original_index[node])
        world = data["proposal"].world_position[node].tolist()
        local = data["proposal"].local_position[node].tolist()
        raw = data["proposal"].execution_feature_raw[node].tolist()
        lines.append(
            f"  node={node} candidate_id={candidate_id} original_index={original_index} "
            f"world={world} local={local} fp_shep={raw}"
        )
    smooth = data["agent", "smooth", "proposal"]
    conflict = data["align", "spatiotemporal", "proposal"]
    lines.extend([
        "agent -> proposal:",
        f"  edge_count = {smooth.edge_index.shape[1]}",
        f"  smoothness = {smooth.edge_attr.reshape(-1).tolist()}",
        "align -> proposal:",
        f"  edge_count = {conflict.edge_index.shape[1]}",
    ])
    for edge in range(conflict.edge_index.shape[1]):
        source = int(conflict.edge_index[0, edge])
        target = int(conflict.edge_index[1, edge])
        neighbor_id = int(data["align"].agent_id[source])
        candidate_id = int(data["proposal"].candidate_id[target])
        t_min, d_min, risk = conflict.edge_attr[edge].tolist()
        lines.append(
            f"  edge={edge} neighbor_id={neighbor_id} candidate_id={candidate_id} "
            f"t_min={t_min:.6g} d_min={d_min:.6g} T_risk={risk:.6g}"
        )
    return "\n".join(lines)
