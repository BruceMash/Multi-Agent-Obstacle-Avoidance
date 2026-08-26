from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Entity.dynamic_obstacles import MovingSphereObstacle  # noqa: E402
from Entity.static_obstacles import (  # noqa: E402
    AxisAlignedBoxObstacle,
    StaticCylinderObstacle,
    StaticSphereObstacle,
    WorkspaceBoundaryPlaneObstacle,
)
from planning.final_four_stage_benchmark import (  # noqa: E402
    DWAStyleConfig,
    _dwa_fullstate_candidate_scores_scalar_reference,
    _dwa_fullstate_candidate_scores_vectorized,
    _planner_static_obstacles,
    _signed_distance_batch,
    current_state_dynamic_predictions,
)


def test_boundary_contract_switch() -> None:
    static = object()
    boundary = object()
    env = SimpleNamespace(
        static_obstacles=[static],
        workspace_boundary_obstacles=[boundary],
        terminate_on_boundary_collision=True,
        env_config=SimpleNamespace(),
    )
    assert _planner_static_obstacles(env) == [static, boundary]
    env.terminate_on_boundary_collision = False
    assert _planner_static_obstacles(env) == [static]


def test_vectorized_scores_match_scalar_oracle() -> None:
    rng = np.random.default_rng(20260821)
    lower = np.asarray([0.0, 0.0, 0.8])
    upper = np.asarray([100.0, 100.0, 3.2])
    static_obstacles = [
        WorkspaceBoundaryPlaneObstacle(2, 0.8, lower, upper, True),
        WorkspaceBoundaryPlaneObstacle(2, 3.2, lower, upper, False),
        StaticSphereObstacle([4.0, 3.0, 1.7], 0.8, 0.1),
        AxisAlignedBoxObstacle([6.0, 5.0, 1.8], [0.7, 1.2, 0.4], 0.08),
        StaticCylinderObstacle([2.0, 6.0, 1.8], 0.65, 0.7, 0.06),
    ]
    points = rng.uniform([0.0, 0.0, 0.4], [9.0, 9.0, 3.6], size=(11, 7, 3))
    for obstacle in static_obstacles:
        scalar_distance = np.asarray(
            [obstacle.signed_distance(point) for point in points.reshape(-1, 3)]
        ).reshape(points.shape[:-1])
        np.testing.assert_allclose(
            _signed_distance_batch(points, obstacle),
            scalar_distance,
            rtol=0.0,
            atol=1.0e-12,
        )

    moving = MovingSphereObstacle(
        center=[5.0, 4.0, 1.9],
        radius=0.45,
        velocity=[-0.2, 0.3, 0.0],
        safety_margin=0.05,
    )
    dynamic_predictions = current_state_dynamic_predictions(
        [moving], np.arange(1, 13, dtype=float) * 0.1
    )
    kwargs = {
        "position": np.asarray([1.2, 1.4, 1.5]),
        "goal": np.asarray([8.2, 7.4, 2.0]),
        "preferred": np.asarray([1.7, 1.1, 0.2]),
        "candidates": rng.uniform(-2.8, 2.8, size=(43, 3)),
        "static_obstacles": static_obstacles,
        "dynamic_predictions_by_step": dynamic_predictions,
        "peer_positions": np.asarray([[3.0, 2.2, 1.6], [1.0, 5.0, 2.2]]),
        "peer_velocities": np.asarray([[-0.3, 0.1, 0.0], [0.2, -0.4, 0.1]]),
        "dt": 0.1,
        "collision_margin": 0.0,
        "peer_safe_distance": 0.6,
        "sensing_radius": 4.5,
        "peer_influence_distance": 4.5,
        "config": DWAStyleConfig(),
    }
    scalar = _dwa_fullstate_candidate_scores_scalar_reference(**kwargs)
    vectorized = _dwa_fullstate_candidate_scores_vectorized(**kwargs)
    np.testing.assert_allclose(vectorized, scalar, rtol=0.0, atol=1.0e-10)
    assert int(np.argmax(vectorized)) == int(np.argmax(scalar))


if __name__ == "__main__":
    test_boundary_contract_switch()
    test_vectorized_scores_match_scalar_oracle()
    print("PASSED 2 full-state DWA vectorization tests")
