from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
ALGO_ROOT = REPO_ROOT / "Multi-agent_Algo_lib"
for path in (REPO_ROOT, ALGO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from planning.final_four_stage_benchmark import (  # noqa: E402
    DWAStyleConfig,
    _dwa_fullstate_candidate_scores_scalar_reference,
    _dwa_fullstate_candidate_scores_vectorized,
    _planner_static_obstacles,
    _signed_distance_batch,
    RVOStyleConfig,
    WORKSPACE_BOUNDS,
    current_state_dynamic_predictions,
    generate_scenario_manifest,
    obstacle_from_spec,
    run_classical_episode,
    step_direct_accelerations,
    validate_scenario_manifest,
)
from Entity.static_obstacles import (  # noqa: E402
    AxisAlignedBoxObstacle,
    StaticCylinderObstacle,
    StaticSphereObstacle,
    WorkspaceBoundaryPlaneObstacle,
)
from Entity.dynamic_obstacles import (  # noqa: E402
    MovingSphereObstacle,
    PatternedMovingSphereObstacle,
)
from scripts.evaluate_single_policy_aligned_multi_agent import (  # noqa: E402
    build_single_distribution_multi_config,
)
from scripts.run_final_four_stage_benchmark import (  # noqa: E402
    ManifestEnvironmentBuilder,
)


def _manifest():
    return generate_scenario_manifest(
        counts_per_stage=5,
        seed_base=991000000,
        prefix="T",
        max_steps=20,
        dt=0.1,
    )


def _config():
    base = build_single_distribution_multi_config(num_agents=3, max_steps=20)
    return replace(
        base,
        workspace_bounds=WORKSPACE_BOUNDS,
        randomize_start_goal=False,
        min_start_goal_distance=5.5,
    )


def test_manifest_diversity_difficulty_and_freeze() -> None:
    manifest = _manifest()
    audit = validate_scenario_manifest(manifest)
    assert audit["status"] == "PASSED"
    assert audit["scenario_count"] == 20
    assert audit["scenario_duplication_rate"] == 0.0
    assert audit["GEOMETRY_DIVERSITY_VALID"] == "YES"
    assert audit["DIFFICULTY_MONOTONICITY_VALID"] == "YES"


def test_dynamic_trajectory_is_exactly_reconstructable() -> None:
    entry = next(row for row in _manifest()["entries"] if row["stage"] == "stage_4")
    for spec, expected in zip(
        entry["dynamic_obstacles"], entry["dynamic_obstacle_trajectories"], strict=True
    ):
        obstacle = obstacle_from_spec(spec)
        actual = [obstacle.center.copy()]
        for _ in range(entry["max_steps"]):
            actual.append(obstacle.step(entry["dt"]))
        assert np.array_equal(np.asarray(actual), np.asarray(expected))


def test_manifest_builder_and_direct_step_use_frozen_contract() -> None:
    manifest = _manifest()
    entry = manifest["entries"][0]
    builder = ManifestEnvironmentBuilder(manifest)
    env, metadata = builder(
        config=_config(),
        scenario=entry["scenario_id"],
        seed=entry["seed"],
        peer_radius=0.3,
    )
    try:
        assert np.array_equal(env.starts, np.asarray(entry["starts"]))
        assert np.array_equal(env.goals, np.asarray(entry["goals"]))
        assert metadata["environment_fingerprint"] == entry["environment_fingerprint"]
        terminated, truncated, info = step_direct_accelerations(
            env, np.full((3, 3), 99.0)
        )
        assert not terminated
        assert not truncated
        assert np.max(np.abs(info["applied_accelerations"])) <= 4.0 + 1e-8
        assert env.steps == 1
    finally:
        env.close()


def test_fullstate_planner_observes_boundary_planes_only_when_enforced() -> None:
    boundary = WorkspaceBoundaryPlaneObstacle(
        axis=2,
        bound=0.8,
        lower_bounds=np.asarray([0.0, 0.0, 0.8]),
        upper_bounds=np.asarray([100.0, 100.0, 3.2]),
        is_lower=True,
    )
    static = object()
    env = type("PlannerContractEnv", (), {})()
    env.static_obstacles = [static]
    env.workspace_boundary_obstacles = [boundary]
    env.env_config = type("PlannerContractConfig", (), {})()

    env.terminate_on_boundary_collision = True
    assert _planner_static_obstacles(env) == [static, boundary]

    env.terminate_on_boundary_collision = False
    assert _planner_static_obstacles(env) == [static]


def test_vectorized_fullstate_dwa_matches_scalar_geometry_oracle() -> None:
    rng = np.random.default_rng(20260821)
    lower = np.asarray([0.0, 0.0, 0.8])
    upper = np.asarray([100.0, 100.0, 3.2])
    static_obstacles = [
        WorkspaceBoundaryPlaneObstacle(2, 0.8, lower, upper, True),
        WorkspaceBoundaryPlaneObstacle(2, 3.2, lower, upper, False),
        StaticSphereObstacle(center=[4.0, 3.0, 1.7], radius=0.8, safety_margin=0.1),
        AxisAlignedBoxObstacle(
            center=[6.0, 5.0, 1.8], half_extents=[0.7, 1.2, 0.4], safety_margin=0.08
        ),
        StaticCylinderObstacle(
            center=[2.0, 6.0, 1.8], radius=0.65, half_height=0.7, safety_margin=0.06
        ),
    ]
    points = rng.uniform([0.0, 0.0, 0.4], [9.0, 9.0, 3.6], size=(11, 7, 3))
    for obstacle in static_obstacles:
        expected = np.asarray(
            [obstacle.signed_distance(point) for point in points.reshape(-1, 3)]
        ).reshape(points.shape[:-1])
        np.testing.assert_allclose(
            _signed_distance_batch(points, obstacle), expected, rtol=0.0, atol=1.0e-12
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
    batched = _dwa_fullstate_candidate_scores_vectorized(**kwargs)
    np.testing.assert_allclose(batched, scalar, rtol=0.0, atol=1.0e-10)
    assert int(np.argmax(batched)) == int(np.argmax(scalar))


def test_rvo_style_is_deterministic_on_same_manifest_scene() -> None:
    manifest = _manifest()
    entry = manifest["entries"][0]
    builder = ManifestEnvironmentBuilder(manifest)
    kwargs = dict(
        environment_builder=builder,
        multi_config=_config(),
        scenario=entry["scenario_id"],
        seed=entry["seed"],
        peer_radius=0.3,
        method="rvo_orca_style",
        planner_config=RVOStyleConfig(),
    )
    first, _, _, first_trajectory = run_classical_episode(**kwargs)
    second, _, _, second_trajectory = run_classical_episode(**kwargs)
    assert first["termination_reason"] == second["termination_reason"]
    assert np.array_equal(first_trajectory["positions"], second_trajectory["positions"])


def test_corrected_dynamic_prediction_ignores_private_rng_state() -> None:
    kwargs = dict(
        center=[1.0, 2.0, 3.0],
        radius=0.4,
        velocity=[0.3, -0.2, 0.1],
        safety_margin=0.06,
        bounds=((-2.0, -2.0, -2.0), (8.0, 8.0, 8.0)),
        motion_mode="wandering",
        wandering_strength=0.8,
    )
    first = PatternedMovingSphereObstacle(**kwargs, seed=11)
    second = PatternedMovingSphereObstacle(**kwargs, seed=991)
    first._rng.normal(size=37)
    second._rng.normal(size=3)
    times = [0.1, 0.4, 1.4]
    first_predictions = current_state_dynamic_predictions([first], times)
    second_predictions = current_state_dynamic_predictions([second], times)
    for time_s, left, right in zip(
        times, first_predictions, second_predictions, strict=True
    ):
        expected = np.asarray(kwargs["center"]) + time_s * np.asarray(kwargs["velocity"])
        np.testing.assert_array_equal(left[0].center, expected)
        np.testing.assert_array_equal(right[0].center, expected)
        np.testing.assert_array_equal(left[0].center, right[0].center)


def test_true_wandering_future_does_not_change_corrected_prediction() -> None:
    obstacle = PatternedMovingSphereObstacle(
        center=[0.0, 0.0, 0.0],
        radius=0.3,
        velocity=[0.5, 0.1, 0.0],
        safety_margin=0.05,
        bounds=((-2.0, -2.0, -2.0), (2.0, 2.0, 2.0)),
        motion_mode="wandering",
        wandering_strength=0.8,
        seed=17,
    )
    predicted = current_state_dynamic_predictions([obstacle], [0.8])[0][0].center
    true_clone = PatternedMovingSphereObstacle(
        center=obstacle.center.copy(),
        radius=obstacle.radius,
        velocity=obstacle.velocity.copy(),
        safety_margin=obstacle.safety_margin,
        bounds=obstacle.bounds,
        motion_mode="wandering",
        wandering_strength=obstacle.wandering_strength,
        seed=9001,
    )
    for _ in range(8):
        true_clone.step(0.1)
    assert not np.array_equal(predicted, true_clone.center)
    repeated = current_state_dynamic_predictions([obstacle], [0.8])[0][0].center
    np.testing.assert_array_equal(predicted, repeated)


if __name__ == "__main__":
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            value()
            print(f"PASSED {name}")
