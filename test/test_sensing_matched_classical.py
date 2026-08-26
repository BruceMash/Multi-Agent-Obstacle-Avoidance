from __future__ import annotations

import copy
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Entity.sensors import LocalObstacleSensor, SensorPacket  # noqa: E402
from planning.final_four_stage_benchmark import DWAStyleConfig, RVOStyleConfig  # noqa: E402
from planning.sensing_matched_classical import (  # noqa: E402
    _dwa_candidate_scores_scalar_reference,
    _dwa_candidate_scores_vectorized,
    _episode_termination_reason,
    dwa_sensing_matched_accelerations,
    observation_equivalence_hash,
    reconstruct_sensing_matched_perception,
    rvo_sensing_matched_accelerations,
)


def test_typed_collision_reason_precedence_and_boundary_reason() -> None:
    common = {
        "success": False,
        "static_obstacle_collision": False,
        "dynamic_obstacle_collision": False,
        "obstacle_collision": False,
        "inter_agent_collision": False,
        "boundary_collision": False,
        "timeout": False,
        "planner_infeasible_count": 0,
    }
    assert _episode_termination_reason(
        **{**common, "static_obstacle_collision": True, "timeout": True}
    ) == "static_obstacle_collision"
    assert _episode_termination_reason(
        **{**common, "dynamic_obstacle_collision": True}
    ) == "dynamic_obstacle_collision"
    assert _episode_termination_reason(
        **{**common, "boundary_collision": True}
    ) == "boundary_collision"


def _packet(scan: np.ndarray) -> SensorPacket:
    scan = np.asarray(scan, dtype=np.float32)
    return SensorPacket(
        observation=np.zeros(119, dtype=np.float32),
        current_scan=scan.copy(),
        previous_scan=scan.copy(),
        min_clearance=float(np.min(scan) * 4.5),
        collision=False,
    )


def _environment() -> SimpleNamespace:
    sensors = [
        LocalObstacleSensor(
            sensing_radius=4.5,
            azimuth_bins=8,
            elevation_bins=7,
            include_previous_scan=True,
        )
        for _ in range(2)
    ]
    scan0 = np.ones((8, 7), dtype=np.float32)
    scan0[0, 3] = 0.55
    scan0[1, 3] = 0.62
    scan0[7, 3] = 0.58
    scan1 = np.ones((8, 7), dtype=np.float32)
    dynamics = [
        SimpleNamespace(
            p=np.asarray([0.0, 0.0, 0.0], dtype=float),
            v=np.asarray([0.2, 0.0, 0.0], dtype=float),
            dt=0.1,
            accelerate_min=np.full(3, -4.0),
            accelerate_max=np.full(3, 4.0),
            velocity_min=np.full(3, -4.0),
            velocity_max=np.full(3, 4.0),
        ),
        SimpleNamespace(
            p=np.asarray([8.0, 3.0, 0.0], dtype=float),
            v=np.asarray([-0.1, 0.0, 0.0], dtype=float),
            dt=0.1,
            accelerate_min=np.full(3, -4.0),
            accelerate_max=np.full(3, 4.0),
            velocity_min=np.full(3, -4.0),
            velocity_max=np.full(3, 4.0),
        ),
    ]
    return SimpleNamespace(
        num_agents=2,
        dynamics=dynamics,
        goals=np.asarray([[8.0, 0.0, 0.0], [0.0, 3.0, 0.0]], dtype=float),
        sensors=sensors,
        latest_sensor_packets=[_packet(scan0), _packet(scan1)],
        success_rewarded_mask=np.zeros(2, dtype=bool),
        env_config=SimpleNamespace(
            collision_margin=0.0,
            inter_agent_influence_distance=1.2,
            inter_agent_safe_distance=0.6,
        ),
        static_obstacles=[SimpleNamespace(hidden_shape="box")],
        dynamic_obstacles=[
            SimpleNamespace(
                center=np.asarray([6.0, 2.0, 0.0]),
                velocity=np.asarray([1.0, 0.0, 0.0]),
                hidden_motion="wandering",
            )
        ],
    )


def _outputs(env: SimpleNamespace) -> tuple[np.ndarray, np.ndarray]:
    dwa, _ = dwa_sensing_matched_accelerations(env, DWAStyleConfig())
    rvo, _ = rvo_sensing_matched_accelerations(env, RVOStyleConfig())
    return dwa, rvo


def _assert_agent0_equivalent(left: SimpleNamespace, right: SimpleNamespace) -> None:
    assert observation_equivalence_hash(left, 0) == observation_equivalence_hash(right, 0)
    left_dwa, left_rvo = _outputs(left)
    right_dwa, right_rvo = _outputs(right)
    assert np.array_equal(left_dwa[0], right_dwa[0])
    assert np.array_equal(left_rvo[0], right_rvo[0])


def test_adapter_uses_only_visible_lidar_endpoints() -> None:
    env = _environment()
    perception = reconstruct_sensing_matched_perception(env)
    assert len(perception.local_surfaces[0]) == 3
    assert len(perception.local_surfaces[1]) == 0
    assert perception.visible_hit_count == 3
    assert all(obstacle.effective_radius == 0.0 for obstacle in perception.local_surfaces[0])


def test_long_range_mode_includes_only_public_anonymous_peer_block() -> None:
    env = _environment()
    env.env_config.peer_state_observation_mode = "local_anonymous_ally_block"

    def observable_neighbor_states(agent_id: int):
        if int(agent_id) != 0:
            return ()
        return (
            SimpleNamespace(
                agent_id=0,
                position=np.asarray([1.0, 0.0, 0.0], dtype=float),
                velocity=np.asarray([-0.2, 0.0, 0.0], dtype=float),
            ),
        )

    env.observable_neighbor_states = observable_neighbor_states
    perception = reconstruct_sensing_matched_perception(env)
    _, info = dwa_sensing_matched_accelerations(env, DWAStyleConfig())
    assert len(perception.local_anonymous_peers[0]) == 1
    assert len(perception.local_anonymous_peers[1]) == 0
    assert perception.observable_peer_count == 1
    assert info["observable_peer_count"] == 1
    assert info["peer_information_mode"] == "local_anonymous_ally_block"


def test_vectorized_dwa_candidate_scores_match_scalar_oracle() -> None:
    rng = np.random.default_rng(20260820)
    kwargs = {
        "position": rng.normal(size=3),
        "goal": rng.normal(size=3),
        "preferred": rng.normal(size=3),
        "candidates": rng.normal(size=(29, 3)),
        "obstacle_points": rng.normal(size=(37, 3)),
        "peer_positions": rng.normal(size=(2, 3)),
        "peer_velocities": rng.normal(size=(2, 3)),
        "dt": 0.1,
        "steps": 12,
        "obstacle_collision_distance": 0.23,
        "peer_collision_distance": 0.68,
        "sensing_radius": 4.5,
        "peer_influence_distance": 4.5,
        "config": DWAStyleConfig(),
    }
    scalar = _dwa_candidate_scores_scalar_reference(**kwargs)
    batched = _dwa_candidate_scores_vectorized(**kwargs)
    assert np.allclose(batched, scalar, rtol=0.0, atol=1.0e-12)
    assert int(np.argmax(batched)) == int(np.argmax(scalar))


def test_out_of_range_obstacle_change_is_observation_equivalent() -> None:
    left = _environment()
    right = copy.deepcopy(left)
    right.static_obstacles = [SimpleNamespace(hidden_shape="sphere", center=[100.0, 0.0, 0.0])]
    _assert_agent0_equivalent(left, right)


def test_hidden_obstacle_shape_change_is_observation_equivalent() -> None:
    left = _environment()
    right = copy.deepcopy(left)
    right.static_obstacles[0].hidden_shape = "cylinder_with_different_radius"
    _assert_agent0_equivalent(left, right)


def test_hidden_dynamic_velocity_change_is_observation_equivalent() -> None:
    left = _environment()
    right = copy.deepcopy(left)
    right.dynamic_obstacles[0].velocity = np.asarray([-99.0, 77.0, 3.0])
    right.dynamic_obstacles[0].hidden_motion = "different_rng_future"
    _assert_agent0_equivalent(left, right)


def test_hidden_out_of_range_peer_change_is_observation_equivalent() -> None:
    left = _environment()
    right = copy.deepcopy(left)
    right.dynamics[1].p = np.asarray([100.0, -80.0, 50.0])
    right.dynamics[1].v = np.asarray([40.0, -30.0, 20.0])
    _assert_agent0_equivalent(left, right)


def test_same_observation_execution_is_deterministic() -> None:
    env = _environment()
    first_dwa, first_rvo = _outputs(env)
    second_dwa, second_rvo = _outputs(env)
    assert np.array_equal(first_dwa, second_dwa)
    assert np.array_equal(first_rvo, second_rvo)


def test_planner_functions_do_not_read_hidden_world_collections() -> None:
    source = inspect.getsource(dwa_sensing_matched_accelerations)
    source += inspect.getsource(rvo_sensing_matched_accelerations)
    for forbidden in (
        "env.static_obstacles",
        "env.dynamic_obstacles",
        "env._positions",
        "env._velocities",
        "observable_neighbor_states",
    ):
        assert forbidden not in source
