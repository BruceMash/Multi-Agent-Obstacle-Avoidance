from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from Environment.multi_agent_dmp_env import MultiAgentDMPEnv
from Guidance.reference_point_proposal_demo import Proposal
from planning.candidate_execution_interface import GraphReadyCandidateExecution
from planning.heterogeneous_candidate_graph import (
    DEFAULT_D_ALIGN_SOURCE,
    EgoGraphState,
    HeterogeneousCandidateGraphConfig,
    NeighborGraphState,
    build_heterogeneous_candidate_graph,
    graph_debug_summary,
    world_to_ego_local,
)


def _proposal(
    candidate_id: int,
    point: tuple[float, float, float],
    *,
    score: float | None = None,
    normalized_margin: float | None = None,
    distance_progress: float | None = None,
) -> Proposal:
    point_array = np.asarray(point, dtype=float)
    distance = float(np.linalg.norm(point_array))
    direction = point_array / distance if distance > 0.0 else np.array([1.0, 0.0, 0.0])
    return Proposal(
        azimuth_index=int(candidate_id) % 8,
        elevation_index=int(candidate_id) % 7,
        direction=direction,
        point=point_array,
        distance=distance,
        raw_obstacle_distance=4.0,
        obstacle_distance=3.9,
        effective_safe_radius=1.15,
        braking_distance=0.1,
        safety_margin=2.65,
        normalized_margin=(
            0.7 if normalized_margin is None else float(normalized_margin)
        ),
        distance_progress=(
            0.25 if distance_progress is None else float(distance_progress)
        ),
        normalized_progress=0.25,
        alignment=0.8,
        smoothness=0.6,
        usable_length=0.8,
        score=float(candidate_id if score is None else score),
    )


def _execution(
    candidate_id: int,
    point: tuple[float, float, float],
    *,
    trajectory: np.ndarray | None = None,
    progress: float = 0.1,
    clearance: float = 1.0,
    deviation: float = 0.2,
    speed: float = 0.8,
) -> GraphReadyCandidateExecution:
    if trajectory is None:
        target = np.asarray(point, dtype=float)
        trajectory = np.stack([target * ratio for ratio in (0.25, 0.5, 0.75, 1.0)])
    trajectory = np.asarray(trajectory, dtype=float)
    horizon = trajectory.shape[0]
    finite_clearance = bool(np.isfinite(clearance))
    return GraphReadyCandidateExecution(
        candidate_id=candidate_id,
        candidate_world_position=np.asarray(point, dtype=float),
        preview_positions=trajectory,
        preview_velocities=np.ones_like(trajectory) * 0.3,
        task_progress_raw=progress,
        min_clearance_raw=clearance,
        max_execution_deviation_raw=deviation,
        terminal_speed_raw=speed,
        requested_horizon_steps=horizon,
        effective_horizon_steps=horizon,
        effective_horizon_ratio=1.0,
        preview_completed=True,
        termination_reason="completed_horizon",
        feature_valid_mask=np.asarray([True, finite_clearance, True, True]),
        feature_full_horizon_mask=np.asarray([True, finite_clearance, True, True]),
        obstacle_clearance_source="frozen_lidar_surface_samples",
        obstacle_clearance_is_approximate=True,
        boundary_clearance_source="not_separately_available_from_untyped_lidar",
        boundary_clearance_is_approximate=True,
        clearance_finite_mask=finite_clearance,
        open_space_flag=not finite_clearance,
    )


def _ego(*, velocity=(1.0, 0.0, 0.0)) -> EgoGraphState:
    safety = np.linspace(0.1, 0.9, 56)
    return EgoGraphState(
        agent_id=0,
        position=np.asarray([0.0, 0.0, 0.0]),
        velocity=np.asarray(velocity, dtype=float),
        previous_velocity=np.asarray([0.7, 0.1, 0.0]),
        task_goal=np.asarray([8.0, 0.0, 0.0]),
        sector_safety=safety,
        goal_sector_id=28,
    )


def _build(
    proposals: list[Proposal],
    executions: list[GraphReadyCandidateExecution],
    *,
    neighbors: list[NeighborGraphState] | None = None,
    ego: EgoGraphState | None = None,
    config: HeterogeneousCandidateGraphConfig | None = None,
):
    return build_heterogeneous_candidate_graph(
        ego=ego or _ego(),
        proposals=proposals,
        executions=executions,
        neighbors=neighbors or [],
        config=config or HeterogeneousCandidateGraphConfig(),
    )


@pytest.mark.parametrize("candidate_count", [10, 6, 2, 1, 0])
def test_variable_candidate_count_is_native_and_unpadded(candidate_count: int):
    proposals = [_proposal(index, (1.0 + index * 0.02, 0.0, 0.0)) for index in range(candidate_count)]
    executions = [_execution(index, tuple(proposal.point)) for index, proposal in enumerate(proposals)]
    graph = _build(proposals, executions)
    assert graph["proposal"].num_nodes == candidate_count
    assert graph["agent", "smooth", "proposal"].edge_index.shape == (2, candidate_count)
    assert tuple(graph.proposal_node_to_candidate_id) == tuple(range(candidate_count))
    assert graph["proposal"].x_normalized.shape == (candidate_count, 13)


@pytest.mark.parametrize("neighbor_count", [0, 1, 3])
def test_variable_neighbor_count_and_zero_neighbor_empty_edges(neighbor_count: int):
    proposal = _proposal(0, (1.0, 0.0, 0.0))
    execution = _execution(0, (1.0, 0.0, 0.0))
    neighbors = [
        NeighborGraphState(index + 1, [5.0 + index, 3.0, 0.0], [0.0, 0.0, 0.0])
        for index in range(neighbor_count)
    ]
    graph = _build([proposal], [execution], neighbors=neighbors)
    assert graph["align"].num_nodes == neighbor_count
    if neighbor_count == 0:
        assert graph["align", "spatiotemporal", "proposal"].edge_index.shape == (2, 0)


def test_world_to_local_is_translation_only_and_preserves_distance():
    ego_position = np.asarray([2.0, -3.0, 1.0])
    world = np.asarray([3.0, -1.0, 4.0])
    local = world_to_ego_local(world, ego_position)
    np.testing.assert_allclose(local, [1.0, 2.0, 3.0])
    assert np.linalg.norm(local) == pytest.approx(np.linalg.norm(world - ego_position))


def test_geometric_and_execution_progress_are_distinct_features():
    proposal = _proposal(0, (1.0, 0.0, 0.0), distance_progress=-0.4)
    execution = _execution(0, (1.0, 0.0, 0.0), progress=0.35)
    graph = _build([proposal], [execution])
    assert graph["proposal"].x_raw[0, 7].item() == pytest.approx(-0.4)
    assert graph["proposal"].x_raw[0, 9].item() == pytest.approx(0.35)


def test_sector_safety_maps_normalized_margin_not_proposal_score():
    proposal = _proposal(0, (1.0, 0.0, 0.0), score=9.5, normalized_margin=0.42)
    graph = _build([proposal], [_execution(0, (1.0, 0.0, 0.0))])
    assert graph["proposal"].x_raw[0, 8].item() == pytest.approx(0.42)
    assert graph["proposal"].x_raw[0, 8].item() != pytest.approx(9.5)


def test_zero_velocity_smoothness_is_finite_zero():
    graph = _build(
        [_proposal(0, (1.0, 0.0, 0.0))],
        [_execution(0, (1.0, 0.0, 0.0))],
        ego=_ego(velocity=(0.0, 0.0, 0.0)),
    )
    edge = graph["agent", "smooth", "proposal"].edge_attr
    assert torch.isfinite(edge).all()
    assert edge.item() == pytest.approx(0.0)


def test_future_crossing_builds_edge_despite_large_current_distance():
    trajectory = np.asarray([[1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]], dtype=float)
    graph = _build(
        [_proposal(0, (4.0, 0.0, 0.0))],
        [_execution(0, (4.0, 0.0, 0.0), trajectory=trajectory)],
        neighbors=[NeighborGraphState(9, [3.0, 3.0, 0.0], [0.0, -10.0, 0.0])],
        config=HeterogeneousCandidateGraphConfig(dt=0.1),
    )
    edge = graph["align", "spatiotemporal", "proposal"]
    assert edge.edge_index.shape[1] == 1
    assert edge.edge_attr[0, 1].item() == pytest.approx(0.0)
    assert edge.edge_attr[0, 0].item() == pytest.approx(0.3)


def test_current_close_but_future_diverging_does_not_build_edge():
    trajectory = np.asarray([[1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]], dtype=float)
    graph = _build(
        [_proposal(0, (4.0, 0.0, 0.0))],
        [_execution(0, (4.0, 0.0, 0.0), trajectory=trajectory)],
        neighbors=[NeighborGraphState(3, [0.1, 0.0, 0.0], [-10.0, 0.0, 0.0])],
        config=HeterogeneousCandidateGraphConfig(dt=0.1),
    )
    assert graph["align", "spatiotemporal", "proposal"].edge_index.shape[1] == 0


def test_d_safe_and_d_align_are_independent_thresholds():
    trajectory = np.asarray([[1.0, 0.0, 0.0]] * 4)
    graph = _build(
        [_proposal(0, (1.0, 0.0, 0.0))],
        [_execution(0, (1.0, 0.0, 0.0), trajectory=trajectory)],
        neighbors=[NeighborGraphState(1, [1.0, 0.8, 0.0], [0.0, 0.0, 0.0])],
    )
    edge = graph["align", "spatiotemporal", "proposal"]
    assert edge.edge_index.shape[1] == 1
    assert edge.edge_attr[0, 1].item() == pytest.approx(0.8)
    assert edge.edge_attr[0, 2].item() == pytest.approx(0.0)
    assert graph.graph_metadata["d_align_source"] == DEFAULT_D_ALIGN_SOURCE


def test_graph_preserves_strict_d_safe_boundary_in_reused_diagnostic():
    trajectory = np.asarray([[1.0, 0.0, 0.0]] * 4)
    graph = _build(
        [_proposal(0, (1.0, 0.0, 0.0))],
        [_execution(0, (1.0, 0.0, 0.0), trajectory=trajectory)],
        neighbors=[NeighborGraphState(1, [1.0, 0.6, 0.0], [0.0, 0.0, 0.0])],
    )
    edge = graph["align", "spatiotemporal", "proposal"]
    assert edge.edge_index.shape[1] == 1
    assert edge.edge_attr[0, 1].item() == pytest.approx(0.6)
    assert edge.edge_attr[0, 2].item() == pytest.approx(0.0)
    assert graph.graph_metadata["d_safe_risk_inequality"] == "distance < d_safe"


def test_candidate_mapping_preserves_input_order_without_score_sorting():
    order = [2, 0, 1]
    proposals = [_proposal(identifier, (1.0, identifier * 0.1, 0.0), score=float(identifier)) for identifier in order]
    executions = [_execution(identifier, tuple(proposal.point)) for identifier, proposal in zip(order, proposals)]
    graph = _build(proposals, executions)
    assert graph.proposal_node_to_candidate_id == (2, 0, 1)
    assert graph.proposal_node_to_original_index == (0, 1, 2)
    assert graph.candidate_id_to_proposal_node == {2: 0, 0: 1, 1: 2}
    np.testing.assert_allclose(graph["proposal"].world_position.numpy(), np.stack([p.point for p in proposals]))


def test_open_space_preserves_raw_inf_and_all_gat_ready_tensors_are_finite():
    proposal = _proposal(0, (1.0, 0.0, 0.0))
    graph = _build([proposal], [_execution(0, tuple(proposal.point), clearance=np.inf)])
    assert torch.isinf(graph["proposal"].x_raw[0, 10])
    assert graph["proposal"].open_space_flag.item() is True
    for node_type in graph.node_types:
        assert torch.isfinite(graph[node_type].x_normalized).all()
        assert torch.isfinite(graph[node_type].x).all()
    for edge_type in graph.edge_types:
        store = graph[edge_type]
        if hasattr(store, "edge_attr_normalized"):
            assert torch.isfinite(store.edge_attr_normalized).all()


def test_determinism_and_no_side_effects():
    proposals = [_proposal(3, (1.0, 0.2, 0.0)), _proposal(7, (1.0, -0.2, 0.0))]
    executions = [_execution(3, tuple(proposals[0].point)), _execution(7, tuple(proposals[1].point))]
    neighbors = [NeighborGraphState(4, [2.0, 1.0, 0.0], [0.0, 0.0, 0.0])]
    before = copy.deepcopy((proposals, executions, neighbors))
    left = _build(proposals, executions, neighbors=neighbors)
    right = _build(proposals, executions, neighbors=neighbors)
    for node_type in left.node_types:
        torch.testing.assert_close(left[node_type].x_raw, right[node_type].x_raw, equal_nan=True)
        torch.testing.assert_close(left[node_type].x_normalized, right[node_type].x_normalized)
    for edge_type in left.edge_types:
        torch.testing.assert_close(left[edge_type].edge_index, right[edge_type].edge_index)
        torch.testing.assert_close(left[edge_type].edge_attr, right[edge_type].edge_attr)
    for old, new in zip(before[0], proposals):
        np.testing.assert_array_equal(old.point, new.point)
    for old, new in zip(before[1], executions):
        np.testing.assert_array_equal(old.preview_positions, new.preview_positions)


def test_schema_has_only_required_types_and_debug_summary_is_traceable():
    graph = _build(
        [_proposal(5, (1.0, 0.0, 0.0))],
        [_execution(5, (1.0, 0.0, 0.0))],
    )
    assert graph.node_types == ["null", "agent", "proposal", "align"]
    assert graph.edge_types == [
        ("agent", "smooth", "proposal"),
        ("align", "spatiotemporal", "proposal"),
    ]
    assert graph.graph_metadata["local_frame_convention"] == "ego_centered_world_axis_aligned"
    assert graph.graph_metadata["local_frame_rotation_applied"] is False
    summary = graph_debug_summary(graph)
    assert "candidate_id=5" in summary
    assert "previous_velocity" in summary
    assert "goal_sector_id" in summary
    assert "agent -> proposal" in summary
    assert "align -> proposal" in summary


def test_horizon_mismatch_is_rejected_instead_of_silently_reinterpreted():
    proposal = _proposal(0, (1.0, 0.0, 0.0))
    execution = _execution(0, tuple(proposal.point), trajectory=np.asarray([[1.0, 0.0, 0.0]] * 2))
    with pytest.raises(ValueError, match="horizon"):
        _build([proposal], [execution])


def _minimal_env(*, nearest_count: int | None = None) -> MultiAgentDMPEnv:
    env = MultiAgentDMPEnv(
        dynamics_config={
            "velocity_clip": [-4.0, 4.0],
            "accelerate_clip": [-4.0, 4.0],
            "time_step": 0.1,
        },
        sensor_config={
            "sensing_radius": 4.5,
            "azimuth_bins": 8,
            "elevation_bins": 7,
            "include_previous_scan": True,
        },
        dmp_config={"dt": 0.1, "dims": 3},
        env_config={
            "num_agents": 3,
            "max_steps": 10,
            "goal_tolerance": 0.3,
            "nearest_agent_observation_count": nearest_count,
            "min_start_goal_distance": 0.0,
        },
        static_obstacles=[],
        dynamic_obstacles=[],
    )
    env.reset(
        seed=1,
        options={
            "starts": np.asarray([[0, 0, 0], [1, 0, 0], [3, 0, 0]], dtype=float),
            "goals": np.asarray([[8, 0, 0], [8, 1.3, 0], [8, 2.6, 0]], dtype=float),
            "static_obstacles": [],
            "dynamic_obstacles": [],
        },
    )
    return env


def test_previous_velocity_cache_is_previous_real_step_and_not_observation_feature():
    env = _minimal_env()
    try:
        observation_before = env.get_observation()
        assert env.previous_velocities.shape == (3, 3)
        np.testing.assert_allclose(env.previous_velocities, 0.0)
        current_velocity = np.asarray([[0.2, 0.1, 0.0], [0.3, 0.0, 0.0], [0.4, -0.1, 0.0]])
        for index in range(3):
            env.dynamics[index].v = current_velocity[index].copy()
            env.dynamics[index].state = np.concatenate([env.dynamics[index].p, env.dynamics[index].v])
        shape_before = observation_before.shape
        env.step(np.zeros(env.action_shape, dtype=np.float32))
        np.testing.assert_allclose(env.previous_velocities, current_velocity)
        assert env.get_observation().shape == shape_before
        assert env.sensor_observation_dim == 119
    finally:
        env.close()


def test_structured_neighbor_accessor_uses_same_nearest_membership_and_physical_state():
    env = _minimal_env(nearest_count=1)
    try:
        neighbors = env.observable_neighbor_states(0)
        assert len(neighbors) == 1
        assert neighbors[0].agent_id == 1
        np.testing.assert_allclose(neighbors[0].position, [1.0, 0.0, 0.0])
        assert neighbors[0].position.flags.writeable is False
        flat = env._compose_inter_agent_observation(0)
        assert flat.shape == (7,)
        assert flat[0] == pytest.approx(1.0 / env.env_config.inter_agent_influence_distance)
    finally:
        env.close()
