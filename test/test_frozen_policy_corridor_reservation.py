"""Contract tests for the frozen-policy corridor feasibility experiment.

The experiment is intentionally stage-gated.  Tests for the implemented Step 1
geometry and execution contracts are executable.  Tests that require the E3
reservation manager or the E4 policy-conditioned rollout are registered as
skipped contracts until the preceding Go/No-Go gate permits those components to
be implemented.  This prevents a test-only surrogate from being mistaken for
experimental functionality.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
ALGO_ROOT = REPO_ROOT / "Multi-agent_Algo_lib"
SCRIPTS_ROOT = ALGO_ROOT / "scripts"
CONFIG_PATH = (
    REPO_ROOT
    / "configs"
    / "evaluation"
    / "frozen_policy_corridor_reservation_minimal.json"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "0c3595f738b2f2f2b7e88fc90479e6d525c986a216ab2d1b2eecc139833ad3d5"
)
STEP1_GATE_REASON = (
    "Step 1 No-Go gate: implementation intentionally not reached"
)

for search_path in (REPO_ROOT, ALGO_ROOT, SCRIPTS_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from scripts import frozen_policy_corridor_reservation as core  # noqa: E402
from scripts import evaluate_frozen_policy_corridor_reservation as evaluator  # noqa: E402


def _settings() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _candidate(
    waypoint_type: str,
    x: float,
    *,
    direction: str = "A_TO_B",
) -> core.CandidateWaypoint:
    return core.CandidateWaypoint(
        waypoint_id=f"test:{waypoint_type}:{x}",
        position=np.asarray([x, 0.0, 0.0], dtype=float),
        waypoint_type=waypoint_type,
        corridor_id="test_corridor",
        direction=direction,
        min_clearance=1.0,
        max_expected_turn=0.0,
        source="unit_test_finite_set",
    )


class _TransitionEnv:
    def __init__(self) -> None:
        self.dynamics = [
            SimpleNamespace(
                p=np.asarray([1.5, 0.0, 0.0], dtype=float),
                v=np.zeros(3, dtype=float),
            )
        ]
        self.dmps = [
            SimpleNamespace(
                phase=0.37,
                goal=np.asarray([2.0, 0.0, 0.0], dtype=float),
            )
        ]
        self.last_policy_action = np.asarray(
            [[1.0, -2.0, 3.0, -0.5, 0.25, 0.75]], dtype=np.float32
        )


class _ExitMetadata:
    def __init__(self, crossed_exit: bool) -> None:
        self.crossed_exit = bool(crossed_exit)

    def has_crossed_exit(self, point, direction, margin=0.0) -> bool:
        del point, direction, margin
        return self.crossed_exit


def _exit_runtime() -> evaluator.AgentRuntime:
    candidates = [
        _candidate("HOLD", 0.0),
        _candidate("ENTRY", 1.0),
        _candidate("EXIT", 2.0),
        _candidate("TERMINAL", 3.0),
    ]
    return evaluator.AgentRuntime(
        route_agent_id=0,
        environment_agent_id=0,
        candidates=candidates,
        candidate_index=2,
        upper_state="OCCUPY",
    )


def test_checkpoint_hash_and_actor_snapshot_freeze_contract() -> None:
    settings = _settings()
    checkpoint = REPO_ROOT / settings["checkpoint"]
    assert checkpoint.is_file()
    assert settings["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256

    digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    assert digest.hexdigest() == EXPECTED_CHECKPOINT_SHA256

    model = SimpleNamespace(actor=torch.nn.Linear(3, 2))
    before = evaluator._actor_snapshot(model)
    model.actor.eval()
    assert evaluator._actor_unchanged(before, model)

    with torch.no_grad():
        next(model.actor.parameters()).add_(1.0)
    assert not evaluator._actor_unchanged(before, model)


def test_policy_action_reaches_environment_step_bitwise_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_action = np.asarray(
        [[1.25, -2.5, 3.75, -0.5, 0.25, 0.75]], dtype=np.float32
    )

    class DummyModel:
        def predict(self, observations, deterministic):
            assert observations.shape == (1, 4)
            assert deterministic is True
            return expected_action, None

    class DummyEnv:
        num_agents = 1
        action_shape = (1, 6)

        def __init__(self) -> None:
            self.starts = np.asarray([[0.0, 0.0, 0.0]], dtype=float)
            self.goals = np.asarray([[1.0, 0.0, 0.0]], dtype=float)
            self.dynamics = [
                SimpleNamespace(p=self.starts[0].copy(), v=np.zeros(3, dtype=float))
            ]
            self.dmps = [SimpleNamespace(phase=0.5, goal=self.goals[0].copy())]
            self.static_obstacles: list[Any] = []
            self.dynamic_obstacles: list[Any] = []
            self.env_config = SimpleNamespace(
                workspace_bounds=np.asarray(
                    [[-10.0, -10.0, -10.0], [10.0, 10.0, 10.0]], dtype=float
                )
            )
            self.action_space = SimpleNamespace(
                low=np.full((1, 6), -10.0, dtype=np.float32),
                high=np.full((1, 6), 10.0, dtype=np.float32),
            )
            self.steps = 0
            self.submitted_action: np.ndarray | None = None

        def _positions(self) -> np.ndarray:
            return np.stack([item.p for item in self.dynamics])

        def _velocities(self) -> np.ndarray:
            return np.stack([item.v for item in self.dynamics])

        def step(self, action: np.ndarray):
            self.submitted_action = np.asarray(action).copy()
            self.steps = 1
            self.dynamics[0].p = self.goals[0].copy()
            info = {
                "success_mask": np.asarray([True]),
                "commanded_accelerations": np.zeros((1, 3), dtype=np.float32),
                "obstacle_collision_mask": np.asarray([False]),
                "inter_agent_collision_mask": np.asarray([False]),
            }
            return np.zeros((1, 4)), np.zeros(1), True, False, info

    class DummyMetadata:
        centerline_point = np.zeros(3, dtype=float)
        centerline_direction = np.asarray([1.0, 0.0, 0.0], dtype=float)
        longitudinal_axis = 0

        @staticmethod
        def longitudinal_coordinate(point) -> float:
            return float(np.asarray(point)[0])

        @staticmethod
        def inside_conflict_region(point) -> bool:
            del point
            return False

    env = DummyEnv()
    runtime = evaluator.AgentRuntime(
        route_agent_id=0,
        environment_agent_id=0,
        candidates=[_candidate("TERMINAL", 1.0)],
    )
    monkeypatch.setattr(
        evaluator,
        "build_active_goal_observations",
        lambda environment, active_goals: np.zeros((1, 4), dtype=np.float32),
    )

    result = evaluator.run_execution_episode(
        model=DummyModel(),
        env=env,
        metadata=DummyMetadata(),
        runtimes=[runtime],
        group=evaluator.E0_GROUP,
        seed=202608050,
        episode_index=0,
        execution_config=_settings()["waypoint_execution"],
        dt=0.1,
    )

    assert env.submitted_action is not None
    np.testing.assert_array_equal(env.submitted_action, expected_action)
    assert result["episodes"][0]["action_contract_unchanged"] is True


@pytest.mark.skip(reason=STEP1_GATE_REASON)
def test_rollout_restores_formal_environment_snapshot() -> None:
    """Enabled after ``rollout_candidate`` exists in the E4 implementation."""


@pytest.mark.skip(reason=STEP1_GATE_REASON)
def test_rollout_preserves_python_numpy_and_torch_rng_states() -> None:
    """Enabled after ``rollout_candidate`` exists in the E4 implementation."""


def test_reservation_schema_and_config_are_capacity_one() -> None:
    settings = _settings()
    reservation = core.CorridorReservation(
        corridor_id="test_corridor",
        owner_agent_id=2,
        status=core.ReservationStatus.AUTHORIZED,
    )
    owner_fields = [
        field.name
        for field in dataclasses.fields(reservation)
        if field.name.startswith("owner_agent")
    ]
    assert owner_fields == ["owner_agent_id"]
    assert isinstance(reservation.to_dict()["owner_agent_id"], int)
    assert settings["reservation"]["capacity"] == 1


@pytest.mark.skip(reason=STEP1_GATE_REASON)
def test_reservation_manager_never_has_more_than_one_owner() -> None:
    """Requires the E3 atomic reservation manager, not only its schema."""


@pytest.mark.skip(reason=STEP1_GATE_REASON)
def test_unauthorized_agent_cannot_enter_corridor() -> None:
    """Requires the E3 authorization guard."""


def test_occupying_agent_cannot_switch_back_to_internal_hold() -> None:
    runtime = _exit_runtime()
    env = _TransitionEnv()

    evaluator._advance_runtime(
        runtime,
        env=env,
        metadata=_ExitMetadata(crossed_exit=False),
        step=10,
        dt=0.1,
        execution_config={"waypoint_tolerance": 0.35},
        reservation_owner=runtime.route_agent_id,
    )

    assert runtime.upper_state == "OCCUPY"
    assert runtime.waypoint_type == "EXIT"
    assert runtime.candidate_index == 2


def test_step1_exit_waypoint_is_kept_until_exit_gate_is_crossed() -> None:
    runtime = _exit_runtime()
    env = _TransitionEnv()

    evaluator._advance_runtime(
        runtime,
        env=env,
        metadata=_ExitMetadata(crossed_exit=False),
        step=10,
        dt=0.1,
        execution_config={"waypoint_tolerance": 0.35},
        reservation_owner=runtime.route_agent_id,
    )
    assert runtime.waypoint_type == "EXIT"

    evaluator._advance_runtime(
        runtime,
        env=env,
        metadata=_ExitMetadata(crossed_exit=True),
        step=11,
        dt=0.1,
        execution_config={"waypoint_tolerance": 0.35},
        reservation_owner=runtime.route_agent_id,
    )
    assert runtime.waypoint_type == "TERMINAL"
    assert runtime.exit_time == pytest.approx(1.1)


@pytest.mark.skip(reason=STEP1_GATE_REASON)
def test_reservation_release_requires_exit_plane_and_release_margin() -> None:
    """Requires the E3 reservation release transition."""


def test_step1_staggered_oracle_uses_fifo_then_agent_id_tie_break() -> None:
    terminal = [_candidate("ENTRY", 1.0), _candidate("EXIT", 2.0)]
    late_low_id = evaluator.AgentRuntime(
        route_agent_id=0,
        environment_agent_id=0,
        candidates=copy.deepcopy(terminal),
        hold_release_step=8,
    )
    early_high_id = evaluator.AgentRuntime(
        route_agent_id=2,
        environment_agent_id=2,
        candidates=copy.deepcopy(terminal),
        hold_release_step=4,
    )
    early_mid_id = evaluator.AgentRuntime(
        route_agent_id=1,
        environment_agent_id=1,
        candidates=copy.deepcopy(terminal),
        hold_release_step=4,
    )

    owner = evaluator._oracle_owner(
        evaluator.E1_GROUP,
        [late_low_id, early_high_id, early_mid_id],
        step=10,
    )
    assert owner == 1


@pytest.mark.skip(reason=STEP1_GATE_REASON)
def test_reservation_aging_prevents_permanent_starvation() -> None:
    """Requires the E3 FIFO/aging queue manager."""


def test_e_corridor_candidates_are_finite_and_route_typed() -> None:
    settings = _settings()
    config = evaluator.build_single_distribution_multi_config(
        num_agents=3,
        max_steps=int(settings["max_steps"]),
    )
    options = evaluator.build_stage_scenario(
        config,
        evaluator._e_stage(),
        seed=int(settings["seed_base"]),
    )
    metadata = core.infer_corridor_metadata(
        options["static_obstacles"],
        config.workspace_bounds,
        settings["corridor_geometry"],
    )

    assert metadata.corridor_width == pytest.approx(0.92)
    assert metadata.usable_length == pytest.approx(6.94)
    expected_directions = ["A_TO_B", "B_TO_A", "BYPASS"]
    allowed_types = {member.value for member in core.WaypointType}
    expected_crossing_types = [
        "PROGRESS",
        "DECELERATION",
        "HOLD",
        "ENTRY",
        "EXIT",
        "TERMINAL",
    ]

    for agent_id, (start, goal) in enumerate(zip(options["starts"], options["goals"])):
        candidates = core.build_route_candidates(
            metadata,
            start,
            goal,
            options["static_obstacles"],
            settings["corridor_geometry"],
            config.workspace_bounds,
        )
        assert candidates
        assert all(candidate.waypoint_type in allowed_types for candidate in candidates)
        assert all(candidate.direction == expected_directions[agent_id] for candidate in candidates)
        assert len({candidate.waypoint_id for candidate in candidates}) == len(candidates)
        assert len({tuple(candidate.position) for candidate in candidates}) == len(candidates)
        if agent_id < 2:
            assert [candidate.waypoint_type for candidate in candidates] == expected_crossing_types
            hold = next(item for item in candidates if item.waypoint_type == "HOLD")
            assert not metadata.inside_conflict_region(hold.position)
        else:
            assert [candidate.waypoint_type for candidate in candidates] == [
                "PROGRESS",
                "TERMINAL",
            ]


def test_candidate_switch_changes_only_active_goal_not_policy_action() -> None:
    env = _TransitionEnv()
    before_action = env.last_policy_action.copy()
    runtime = evaluator.AgentRuntime(
        route_agent_id=0,
        environment_agent_id=0,
        candidates=[_candidate("HOLD", 0.0), _candidate("ENTRY", 1.0)],
        upper_state="HOLD",
    )
    original_phase = env.dmps[0].phase

    evaluator._switch_candidate(
        runtime,
        1,
        env=env,
        step=5,
        dt=0.1,
        reason="unit_test_authorized",
        reservation_owner=0,
    )

    np.testing.assert_array_equal(env.last_policy_action, before_action)
    np.testing.assert_array_equal(env.dmps[0].goal, runtime.candidate.position)
    assert env.dmps[0].phase == original_phase
    assert runtime.waypoint_type == "ENTRY"


def test_first_failure_selection_is_unique_and_deterministic() -> None:
    observed = ["TIMEOUT", "OUT_OF_BOUNDS", "OBSTACLE_COLLISION"]
    selected = evaluator._first_present_failure(observed)
    assert selected == "OBSTACLE_COLLISION"
    assert selected in evaluator.FIRST_FAILURE_TYPES
    assert isinstance(selected, str)


def test_step1_groups_reuse_seed_and_initial_conditions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    seed_count = 2
    generated: dict[int, dict[str, Any]] = {}
    captured: list[dict[str, Any]] = []

    class DummyConfig:
        workspace_bounds = np.asarray(
            [[0.0, -2.5, -1.5], [8.0, 2.5, 1.5]], dtype=float
        )

    class DummyMetadata:
        def to_dict(self) -> dict[str, Any]:
            return {"corridor_id": "stable_test_geometry"}

    class DummyEnv:
        def __init__(self, agent_count: int) -> None:
            self.num_agents = agent_count

        def close(self) -> None:
            return None

    def fake_stage_scenario(config, stage, *, seed):
        del config, stage
        offset = float(seed - int(settings["seed_base"]))
        payload = {
            "starts": np.asarray(
                [[offset, 0.0, 0.0], [8.0 - offset, 0.0, 0.0], [0.0, -1.8, -0.6]],
                dtype=float,
            ),
            "goals": np.asarray(
                [[8.0, 0.0, 0.0], [0.0, 0.0, 0.0], [8.0, -1.8, -0.6]],
                dtype=float,
            ),
            "static_obstacles": [SimpleNamespace(tag=f"walls-{seed}")],
            "dynamic_obstacles": [],
        }
        generated[int(seed)] = copy.deepcopy(payload)
        return payload

    def fake_build_env(config, *, settings, options, observation_mode):
        del config, settings
        captured.append(
            {
                "seed": int(options["seed"]),
                "mode": observation_mode,
                "options": copy.deepcopy(options["scenario_options"]),
            }
        )
        return DummyEnv(len(options["scenario_options"]["starts"]))

    monkeypatch.setattr(
        evaluator,
        "build_single_distribution_multi_config",
        lambda **kwargs: DummyConfig(),
    )
    monkeypatch.setattr(evaluator, "_e_stage", lambda: {"name": evaluator.E_SCENARIO_NAME})
    monkeypatch.setattr(evaluator, "build_stage_scenario", fake_stage_scenario)
    monkeypatch.setattr(
        evaluator.corridor,
        "infer_corridor_metadata",
        lambda *args, **kwargs: DummyMetadata(),
    )
    monkeypatch.setattr(evaluator, "_build_environment_for_options", fake_build_env)
    monkeypatch.setattr(
        evaluator,
        "_build_candidates",
        lambda metadata, *, start, goal, static_obstacles, geometry_config, **kwargs: [
            _candidate("TERMINAL", float(np.asarray(goal)[0]), direction="BYPASS")
        ],
    )
    monkeypatch.setattr(evaluator, "_initialize_runtime_waypoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(evaluator, "run_execution_episode", lambda **kwargs: {})

    evaluator.run_step1(model=object(), settings=settings, seed_count=seed_count)

    assert len(captured) == seed_count * 4
    for offset in range(seed_count):
        seed = int(settings["seed_base"]) + offset
        calls = [row for row in captured if row["seed"] == seed]
        assert len(calls) == 4
        single_calls = [row for row in calls if row["mode"] == "blind"]
        multi_calls = [row for row in calls if row["mode"] == settings["observation_mode"]]
        assert len(single_calls) == 3
        assert len(multi_calls) == 1
        expected = generated[seed]
        np.testing.assert_array_equal(multi_calls[0]["options"]["starts"], expected["starts"])
        np.testing.assert_array_equal(multi_calls[0]["options"]["goals"], expected["goals"])
        for route_id, row in enumerate(single_calls):
            np.testing.assert_array_equal(
                row["options"]["starts"], expected["starts"][[route_id]]
            )
            np.testing.assert_array_equal(
                row["options"]["goals"], expected["goals"][[route_id]]
            )


def test_step1_config_records_shared_protocol_and_rollout_gate() -> None:
    settings = _settings()
    assert settings["groups"] == [evaluator.E0_GROUP, evaluator.E1_GROUP]
    assert settings["seed_base"] == 202608050
    assert settings["smoke_seed_count"] == 10
    assert settings["max_steps"] == 200
    assert settings["dt"] == pytest.approx(0.1)
    assert settings["peer_radius"] == pytest.approx(0.3)
    assert settings["inter_agent_collision_distance"] == pytest.approx(0.6)
    assert settings["rollout"]["enabled"] is False
    assert settings["rollout"]["peer_prediction_mode"] == (
        "maintain_current_active_waypoint"
    )
    assert 20 <= settings["rollout"]["horizon_steps"] <= 50


@pytest.mark.skip(reason=STEP1_GATE_REASON)
def test_e3_and_e4_configs_differ_only_in_rollout_conditioning() -> None:
    """Enabled after Step 1 passes and both E3/E4 configurations exist."""
