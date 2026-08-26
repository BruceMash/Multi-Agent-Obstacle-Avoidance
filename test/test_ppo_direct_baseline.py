from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from planning.ppo_direct_baseline import (
    OBSERVATION_DIM,
    build_local_observations,
    make_environment,
    normalized_actions_to_accelerations,
)
from planning.final_four_stage_benchmark import step_direct_accelerations


ROOT = Path("artifacts/ppo_direct_full_baseline/20260823_193740")


def _entry():
    manifest = json.loads(
        (ROOT / "03_training_environment/PPO_DIRECT_TRAIN_MANIFEST.json").read_text(
            encoding="utf-8"
        )
    )
    return manifest["entries"][0]


def test_local_observation_has_frozen_legal_shape_and_bounds() -> None:
    env = make_environment(_entry())
    try:
        observation = build_local_observations(env)
        assert observation.shape == (3, OBSERVATION_DIM) == (3, 533)
        assert np.all(np.isfinite(observation))
        assert np.max(observation) <= 1.0
        assert np.min(observation) >= -1.0
        # DMP-only phase/K-alpha/K-beta fields are absent: the native sensor
        # packet is followed immediately by the 14-field anonymous peer block.
        assert env.latest_sensor_packets[0].observation.size == 519
        assert observation.shape[1] - 519 == 14
    finally:
        env.close()


def test_normalized_action_maps_only_to_native_acceleration_bounds() -> None:
    action = np.asarray([[2.0, -2.0, 0.5]] * 3, dtype=np.float32)
    acceleration = normalized_actions_to_accelerations(action)
    np.testing.assert_allclose(acceleration[0], [4.0, -4.0, 2.0])


def test_native_step_enforces_speed_and_acceleration_limits() -> None:
    env = make_environment(_entry())
    try:
        terminated, truncated, info = step_direct_accelerations(
            env,
            np.full((3, 3), 40.0, dtype=float),
            refresh_sensors=True,
        )
        assert not truncated
        assert np.max(np.abs(info["applied_accelerations"])) <= 4.0 + 1e-6
        assert np.max(np.linalg.norm(env._velocities(), axis=1)) <= 3.2 + 1e-6
        assert isinstance(terminated, bool)
    finally:
        env.close()
