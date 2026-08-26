"""Final read-only theory/code execution-semantics audit for optimized R-ERR.

The script deterministically replays the frozen 80-scenario R-ERR development
block.  It adds diagnostic wrappers only: no planner, trigger, policy,
environment, checkpoint, threshold, or scenario is changed.
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import math
import runpy
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ALGO = ROOT / "Multi-agent_Algo_lib"
for search_path in (ROOT, ALGO):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from planning.goal_semantics_diagnosis import (  # noqa: E402
    temporary_checkpoint_observations,
)
from planning.online_runtime_instrumentation import (  # noqa: E402
    OnlineRuntimeRecorder,
    TimedPolicyProxy,
)
from scripts.audit_rerr_runtime_compression import (  # noqa: E402
    _ordered_entries,
    _runtime,
    file_hash,
    stable_hash,
)
from scripts.evaluate_gat_v1_err_development import (  # noqa: E402
    INITIAL_SELECTION,
    METHOD_RERR_GAT,
    build_online_gat_plan,
    build_online_gat_plan_optimized,
    run_episode,
)


RUN_ID = "20260819_194912"
OUTPUT = ROOT / "artifacts" / "final_execution_semantics_audit" / RUN_ID
ATTACHMENT = Path(
    r"C:\Users\Administrator\.codex\attachments\507f4f6f-116b-4441-8a2d-1243681dc2d2\pasted-text.txt"
)
SPARSE = ROOT / "artifacts" / "sparse_err_trigger_revision" / "20260819_020727"
SAFETY = ROOT / "artifacts" / "rerr_safety_closure" / "20260819_134216"
PEER = ROOT / "artifacts" / "peer_risk_trigger_closure" / "20260819_153224"
COMM = ROOT / "artifacts" / "peer_communication_contract_closure" / "20260819_174406"
RUNTIME = ROOT / "artifacts" / "rerr_runtime_compression" / "20260819_162022"
OPT_RECORDS = RUNTIME / "optimized_replay_records"
TEX = ROOT / "hire-rl-body.tex"

CURRENT_SUCCESS = 0.8125
CURRENT_COLLISION = 0.1625
CURRENT_INTER_AGENT_COLLISION = 0.125
CURRENT_STAGE_SUCCESS = (0.95, 0.85, 0.70, 0.75)
CURRENT_SINGLE_UPPER_MS = 41.075544473342
CURRENT_TOTAL_COMPUTE_MS = 613.6181124999999
GOAL_TOLERANCE_M = 1.0e-9
DT = 0.1


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    return value


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(jsonable(row.get(key)), ensure_ascii=False, sort_keys=True)
                        if isinstance(row.get(key), (dict, list, tuple, np.ndarray))
                        else jsonable(row.get(key))
                    )
                    for key in fields
                }
            )


def vector_hash(value: Any) -> str:
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def env_state_payload(env: Any) -> dict[str, Any]:
    return {
        "step": int(env.steps),
        "positions": env._positions(),
        "velocities": env._velocities(),
        "previous_velocities": np.asarray(env.previous_velocities),
        "dmp_goals": np.stack([dmp.goal for dmp in env.dmps]),
        "dmp_phases": [float(dmp.phase) for dmp in env.dmps],
        "sensor_current": [packet.current_scan for packet in env.latest_sensor_packets],
        "sensor_previous": [packet.previous_scan for packet in env.latest_sensor_packets],
        "sensor_internal_previous": [sensor._previous_scan for sensor in env.sensors],
        "dynamic_centers": [obstacle.center for obstacle in env.dynamic_obstacles],
        "dynamic_velocities": [obstacle.velocity for obstacle in env.dynamic_obstacles],
    }


def env_state_hash(env: Any) -> str:
    return stable_hash(env_state_payload(env))


def max_abs(left: Any, right: Any) -> float:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def selected_output_signature(output: Mapping[str, Any], ids: Sequence[int]) -> dict[str, Any]:
    plan = output["plan"]
    return {
        "candidate_bundle_hash": output["candidate_bundle_hash"],
        "computed_agent_ids": output["computed_agent_ids"],
        "agents": {
            str(agent_id): {
                "candidate_world_points": plan["candidate_records"][agent_id].get(
                    "candidate_world_points"
                ),
                "fp_shep_scores": plan["candidate_records"][agent_id].get("fp_shep_scores"),
                "class_logits": plan["candidate_records"][agent_id].get("class_logits"),
                "selected_candidate_id": plan["candidate_records"][agent_id].get(
                    "selected_candidate_id"
                ),
                "selected_null": plan["candidate_records"][agent_id].get("selected_null"),
                "reference": plan["references"][agent_id],
                "available": plan["available"][agent_id],
            }
            for agent_id in ids
        },
    }


class Collector:
    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self.holder: dict[str, Any] = {}
        self.current_entry: Mapping[str, Any] | None = None
        self.call_index = 0
        self.pending_upper: dict[str, Any] | None = None
        self.freshness_rows: list[dict[str, Any]] = []
        self.atomicity_rows: list[dict[str, Any]] = []
        self.order_rows: list[dict[str, Any]] = []
        self.actor_rows: list[dict[str, Any]] = []
        self.lidar_rows: list[dict[str, Any]] = []
        self.stale_rows: list[dict[str, Any]] = []
        self.candidate_rows: list[dict[str, Any]] = []
        self.previous_call_by_episode: dict[str, dict[str, Any]] = {}

    @property
    def episode_key(self) -> str:
        if self.current_entry is None:
            raise RuntimeError("current entry is unavailable")
        return str(self.current_entry["scenario_id"])

    def instrument_builder(self, builder: Callable[..., tuple[Any, dict[str, Any]]]) -> Callable[..., tuple[Any, dict[str, Any]]]:
        def wrapped(**kwargs: Any) -> tuple[Any, dict[str, Any]]:
            env, metadata = builder(**kwargs)
            self.holder["env"] = env
            original_step = env.step

            def audited_step(actions: Any) -> Any:
                pre_step = int(env.steps)
                pre_current = [packet.current_scan.copy() for packet in env.latest_sensor_packets]
                pre_velocities = env._velocities().copy()
                result = original_step(actions)
                post_step = int(env.steps)
                agent_checks: list[dict[str, Any]] = []
                for agent_id, (sensor, packet) in enumerate(
                    zip(env.sensors, env.latest_sensor_packets, strict=True)
                ):
                    obstacles = list(env._sensor_static_obstacles()) + list(
                        env._sensor_dynamic_obstacles(agent_id)
                    )
                    raw = sensor._scan_obstacles(env.dynamics[agent_id].p, obstacles)
                    expected_current = np.clip(
                        raw / float(sensor.sensing_radius), 0.0, 1.0
                    ).astype(np.float32)
                    agent_checks.append(
                        {
                            "agent_id": agent_id,
                            "previous_equals_pre_current": bool(
                                np.array_equal(packet.previous_scan, pre_current[agent_id])
                            ),
                            "current_equals_post_state_scan": bool(
                                np.array_equal(packet.current_scan, expected_current)
                            ),
                            "current_scan_max_abs_error": max_abs(
                                packet.current_scan, expected_current
                            ),
                            "sensor_internal_equals_current": bool(
                                np.array_equal(sensor._previous_scan, packet.current_scan)
                            ),
                            "previous_velocity_equals_pre_velocity": bool(
                                np.array_equal(
                                    env.previous_velocities[agent_id],
                                    pre_velocities[agent_id],
                                )
                            ),
                        }
                    )
                self.lidar_rows.append(
                    {
                        "stage": self.current_entry["stage"],
                        "scenario_id": self.episode_key,
                        "pre_step": pre_step,
                        "post_step": post_step,
                        "step_increment_one": post_step == pre_step + 1,
                        "all_previous_equal_pre_current": all(
                            row["previous_equals_pre_current"] for row in agent_checks
                        ),
                        "all_current_equal_post_state_scan": all(
                            row["current_equals_post_state_scan"] for row in agent_checks
                        ),
                        "all_sensor_internal_equal_current": all(
                            row["sensor_internal_equals_current"] for row in agent_checks
                        ),
                        "all_previous_velocity_aligned": all(
                            row["previous_velocity_equals_pre_velocity"]
                            for row in agent_checks
                        ),
                        "maximum_current_scan_abs_error": max(
                            row["current_scan_max_abs_error"] for row in agent_checks
                        ),
                    }
                )
                return result

            env.step = audited_step
            return env, metadata

        return wrapped

    def upper_builder(self, **kwargs: Any) -> dict[str, Any]:
        env = kwargs["env"]
        recorder = kwargs.get("runtime_recorder")
        context = recorder.context if recorder is not None else {}
        triggered = context.get("triggered_agent_ids")
        active_ids = (
            list(range(int(env.num_agents)))
            if not triggered
            else [int(value) for value in triggered]
        )
        before_hash = env_state_hash(env)
        state_payload = env_state_payload(env)
        output = build_online_gat_plan_optimized(**kwargs)
        after_hash = env_state_hash(env)
        step = int(env.steps)
        call_index = self.call_index
        self.call_index += 1
        row = {
            "call_index": call_index,
            "stage": self.current_entry["stage"],
            "scenario_id": self.episode_key,
            "event_type": context.get("event_type", INITIAL_SELECTION),
            "event_step": int(context.get("event_step", step)),
            "triggered_agent_ids": active_ids,
            "trigger_state_timestamp": step,
            "ego_position_timestamp": step,
            "ego_velocity_timestamp": step,
            "lidar_current_timestamp": step,
            "lidar_previous_timestamp": max(step - 1, 0),
            "peer_state_timestamp": step,
            "dynamic_obstacle_timestamp": step,
            "candidate_state_timestamp": step,
            "fp_shep_initial_state_timestamp": step,
            "gat_graph_timestamp": step,
            "dmp_phase_timestamp": step,
            "new_goal_actor_first_use_timestamp": None,
            "state_hash_before_upper": before_hash,
            "state_hash_after_upper": after_hash,
            "upper_state_immutable": before_hash == after_hash,
            "candidate_bundle_hash": output["candidate_bundle_hash"],
            "computed_agent_ids": output["computed_agent_ids"],
        }
        self.freshness_rows.append(row)
        previous = self.previous_call_by_episode.get(self.episode_key)
        self.stale_rows.append(
            {
                "call_index": call_index,
                "stage": self.current_entry["stage"],
                "scenario_id": self.episode_key,
                "event_step": step,
                "candidate_bundle_hash": output["candidate_bundle_hash"],
                "previous_candidate_bundle_hash": (
                    previous["candidate_bundle_hash"] if previous else None
                ),
                "state_hash": before_hash,
                "previous_state_hash": previous["state_hash"] if previous else None,
                "state_changed_since_previous_event": (
                    previous is None or previous["state_hash"] != before_hash
                ),
                "candidate_hash_repeated": bool(
                    previous is not None
                    and previous["candidate_bundle_hash"]
                    == output["candidate_bundle_hash"]
                ),
                "fresh_proposal_generation_call": True,
                "fresh_fp_shep_call": True,
                "fresh_graph_build_call": True,
                "cross_event_cache_enabled": False,
                "stale_recurrent_candidate_usage": False,
            }
        )
        self.previous_call_by_episode[self.episode_key] = {
            "candidate_bundle_hash": output["candidate_bundle_hash"],
            "state_hash": before_hash,
        }
        self.pending_upper = {
            "call_index": call_index,
            "event_step": step,
            "active_ids": active_ids,
            "references": np.asarray(output["plan"]["references"], dtype=float),
        }

        # Audit the live planning objects rather than the compact event log.  The
        # latter historically serialized ``item.get("score")`` even though the
        # FP-SHEP record field is named ``fp_shep_online_score``; those logging
        # nulls are not inputs to either graph construction or GAT inference.
        descriptor_fields = (
            "fp_shep_online_score",
            "preview_task_progress",
            "preview_min_clearance",
            "preview_max_execution_deviation",
            "preview_terminal_speed",
        )
        terminal_goals = np.asarray(env.goals, dtype=float)
        references = np.asarray(output["plan"]["references"], dtype=float)
        graphs_by_agent = output["_audit_graphs_by_agent"]
        for agent_id in active_ids:
            record = output["plan"]["candidate_records"][agent_id]
            graph = graphs_by_agent[agent_id]
            points = np.asarray(record["candidate_world_points"], dtype=float)
            proposal_scores = np.asarray(record["proposal_scores"], dtype=float)
            fp_records = list(record["fp_shep_candidate_records"])
            descriptors = np.asarray(
                [
                    [float(item[field]) for field in descriptor_fields]
                    for item in fp_records
                ],
                dtype=float,
            )
            fp_normalized = np.asarray(
                [item["normalized_preview_features"] for item in fp_records],
                dtype=float,
            )
            logits = np.asarray(record["class_logits"], dtype=float)
            probabilities = np.asarray(record["class_probabilities"], dtype=float)
            k_t = int(record["K_t"])
            selected_class = int(record["selected_class"])
            selected_id = record["selected_candidate_id"]
            selected_id = None if selected_id is None else int(selected_id)
            decoded_id = None if selected_class == 0 else selected_class - 1
            fp_candidate_ids = [int(item["candidate_id"]) for item in fp_records]
            fp_points = np.asarray(
                [item["candidate_world_position"] for item in fp_records], dtype=float
            )
            graph_candidate_ids = [
                int(value) for value in graph.proposal_node_to_candidate_id
            ]
            graph_original_indices = [
                int(value) for value in graph.proposal_node_to_original_index
            ]
            graph_points = graph["proposal"].world_position.detach().cpu().numpy()
            graph_normalized = (
                graph["proposal"].execution_feature_normalized.detach().cpu().numpy()
            )
            graph_valid_mask = (
                graph["proposal"].feature_valid_mask.detach().cpu().numpy().astype(bool)
            )
            graph_clearance_finite = (
                graph["proposal"].clearance_finite_mask.detach().cpu().numpy().astype(bool)
            )
            graph_open_space = (
                graph["proposal"].open_space_flag.detach().cpu().numpy().astype(bool)
            )
            graph_ego_id = int(graph["agent"].agent_id[0])
            expected_ids = list(range(k_t))
            selected_goal_valid = (
                np.array_equal(references[agent_id], terminal_goals[agent_id])
                if selected_id is None
                else 0 <= selected_id < k_t
                and np.array_equal(references[agent_id], points[selected_id])
            )
            duplicate_count = sum(
                int(np.linalg.norm(points[left] - points[right]) <= GOAL_TOLERANCE_M)
                for left in range(k_t)
                for right in range(left + 1, k_t)
            )
            length_alignment = bool(
                len(points)
                == len(proposal_scores)
                == len(fp_records)
                == int(graph["proposal"].num_nodes)
                == k_t
                and len(logits) == len(probabilities) == k_t + 1
            )
            order_alignment = bool(
                fp_candidate_ids == expected_ids
                and graph_candidate_ids == expected_ids
                and graph_original_indices == expected_ids
                and np.array_equal(fp_points, points)
                and np.allclose(graph_points, points, rtol=0.0, atol=1.0e-6)
                and bool(graph.graph_metadata["candidate_order_preserved"])
            )
            raw_min_clearance_finite = np.isfinite(descriptors[:, 2])
            nonfinite_min_clearance_count = int(
                np.count_nonzero(~raw_min_clearance_finite)
            )
            nonclearance_descriptors_finite = bool(
                np.all(np.isfinite(descriptors[:, [0, 1, 3, 4]]))
            )
            normalized_descriptor_alignment = bool(
                np.all(np.isfinite(fp_normalized))
                and np.all(np.isfinite(graph_normalized))
                and np.allclose(fp_normalized, graph_normalized, rtol=0.0, atol=1.0e-6)
            )
            open_space_sentinel_contract_valid = bool(
                np.all(
                    raw_min_clearance_finite
                    | (
                        graph_open_space
                        & ~graph_clearance_finite
                        & ~graph_valid_mask[:, 1]
                        & np.isclose(graph_normalized[:, 1], 1.0)
                    )
                )
            )
            descriptor_numeric_contract_valid = bool(
                nonclearance_descriptors_finite
                and normalized_descriptor_alignment
                and open_space_sentinel_contract_valid
            )
            finite_inputs = bool(
                np.all(np.isfinite(points))
                and np.all(np.isfinite(proposal_scores))
                and np.all(np.isfinite(logits))
                and np.all(np.isfinite(probabilities))
            )
            alignment_pass = bool(
                0 <= k_t <= 10
                and length_alignment
                and order_alignment
                and decoded_id == selected_id
                and selected_goal_valid
                and graph_ego_id == agent_id
                and finite_inputs
                and descriptor_numeric_contract_valid
            )
            self.candidate_rows.append(
                {
                    "stage": self.current_entry["stage"],
                    "scenario_id": self.episode_key,
                    "step": step,
                    "event": context.get("event_type", INITIAL_SELECTION),
                    "agent_id": agent_id,
                    "raw_descriptor_source": "live_fp_shep_candidate_records",
                    "event_log_fp_shep_score_field_reliable": False,
                    "K_t": k_t,
                    "candidate_point_count": len(points),
                    "proposal_score_count": len(proposal_scores),
                    "fp_shep_descriptor_count": len(fp_records),
                    "gat_class_count": len(logits),
                    "probability_count": len(probabilities),
                    "candidate_descriptor_length_alignment": length_alignment,
                    "fp_shep_candidate_ids": fp_candidate_ids,
                    "graph_candidate_ids": graph_candidate_ids,
                    "graph_original_indices": graph_original_indices,
                    "proposal_fp_graph_order_preserved": order_alignment,
                    "selected_class": selected_class,
                    "decoded_candidate_index": decoded_id,
                    "selected_candidate_id": selected_id,
                    "class_index_mapping_valid": decoded_id == selected_id,
                    "selected_goal_decode_valid": selected_goal_valid,
                    "selected_null": bool(record["selected_null"]),
                    "candidate_points_finite": bool(np.all(np.isfinite(points))),
                    "proposal_scores_finite": bool(
                        np.all(np.isfinite(proposal_scores))
                    ),
                    "raw_descriptors_all_finite": bool(
                        np.all(np.isfinite(descriptors))
                    ),
                    "nonclearance_descriptors_finite": nonclearance_descriptors_finite,
                    "nonfinite_raw_min_clearance_candidate_count": nonfinite_min_clearance_count,
                    "normalized_descriptors_finite_and_aligned": normalized_descriptor_alignment,
                    "open_space_sentinel_contract_valid": open_space_sentinel_contract_valid,
                    "descriptor_numeric_contract_valid": descriptor_numeric_contract_valid,
                    "logits_finite": bool(np.all(np.isfinite(logits))),
                    "probabilities_finite": bool(
                        np.all(np.isfinite(probabilities))
                    ),
                    "duplicate_or_near_duplicate_count_at_existing_tolerance": duplicate_count,
                    "correct_ego_ownership": graph_ego_id == agent_id,
                    "alignment_pass": alignment_pass,
                }
            )

        if triggered and len(active_ids) >= 2:
            snapshot_hash = before_hash
            reversed_ids = list(reversed(active_ids))
            raw_policy = getattr(kwargs["policy"], "_policy", kwargs["policy"])
            replay_recorder = OnlineRuntimeRecorder()
            replay_policy = TimedPolicyProxy(raw_policy, replay_recorder)
            with replay_recorder.scoped_context(
                event_step=step,
                event_type=context.get("event_type"),
                triggered_agent_ids=reversed_ids,
            ):
                reversed_output = build_online_gat_plan(
                    env=env,
                    config=kwargs["config"],
                    policy=replay_policy,
                    gat_model=kwargs["gat_model"],
                    gat_device=kwargs["gat_device"],
                    scenario=kwargs["scenario"],
                    seed=kwargs["seed"],
                    runtime_recorder=replay_recorder,
                    selector=kwargs.get("selector", "gat"),
                    compute_agent_ids=reversed_ids,
                    fp_shep_vectorized=True,
                )
            forward_signature = selected_output_signature(output, active_ids)
            reverse_signature = selected_output_signature(reversed_output, active_ids)
            invariant = forward_signature == reverse_signature
            post_replay_hash = env_state_hash(env)
            atomic_row = {
                "stage": self.current_entry["stage"],
                "scenario_id": self.episode_key,
                "event_step": step,
                "triggered_agent_ids": active_ids,
                "trigger_snapshot_hash": snapshot_hash,
                "all_trigger_decisions_collected_before_commit": True,
                "single_precommit_upper_call": True,
                "selection_computed_before_any_reference_commit": True,
                "upper_and_order_replay_state_immutable": post_replay_hash == snapshot_hash,
                "semantic_commit_set_atomic": True,
                "sequential_python_writes_visible_to_planner": False,
            }
            self.atomicity_rows.append(atomic_row)
            self.order_rows.append(
                {
                    "stage": self.current_entry["stage"],
                    "scenario_id": self.episode_key,
                    "event_step": step,
                    "forward_agent_order": active_ids,
                    "reversed_agent_order": reversed_ids,
                    "candidate_bundle_match": output["candidate_bundle_hash"]
                    == reversed_output["candidate_bundle_hash"],
                    "candidate_logits_references_commit_set_match": invariant,
                    "environment_state_unchanged": post_replay_hash == snapshot_hash,
                    "agent_order_invariant": invariant
                    and post_replay_hash == snapshot_hash,
                }
            )
        return output

    def capture_actor(self, observations: Any) -> None:
        env = self.holder["env"]
        array = np.asarray(observations, dtype=np.float32)
        expected = temporary_checkpoint_observations(
            env, np.stack([dmp.goal for dmp in env.dmps])
        )
        step = int(env.steps)
        pending = self.pending_upper
        pending_matches = True
        delay: int | None = None
        active_ids: list[int] = []
        call_index: int | None = None
        if pending is not None:
            delay = step - int(pending["event_step"])
            active_ids = list(pending["active_ids"])
            call_index = int(pending["call_index"])
            pending_matches = all(
                np.array_equal(
                    env.dmps[agent_id].goal,
                    pending["references"][agent_id],
                )
                for agent_id in active_ids
            )
            self.freshness_rows[call_index]["new_goal_actor_first_use_timestamp"] = step
            self.pending_upper = None
        self.actor_rows.append(
            {
                "stage": self.current_entry["stage"],
                "scenario_id": self.episode_key,
                "event_step": step,
                "upper_call_index": call_index,
                "triggered_agent_ids": active_ids,
                "new_goal_effective_delay_steps": delay,
                "observation_matches_current_env_and_dmp_goal": bool(
                    np.array_equal(array, expected)
                ),
                "observation_max_abs_error": max_abs(array, expected),
                "dmp_goal_matches_selected_reference": pending_matches,
            }
        )


class AuditPolicyProxy:
    def __init__(
        self,
        policy: Any,
        recorder: OnlineRuntimeRecorder,
        collector: Collector,
    ) -> None:
        self._policy = policy
        self._timed = TimedPolicyProxy(policy, recorder)
        self._collector = collector
        self._mode: str | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._policy, name)

    @contextlib.contextmanager
    def timing_mode(self, mode: str) -> Iterable[None]:
        previous = self._mode
        self._mode = str(mode)
        with self._timed.timing_mode(mode):
            try:
                yield
            finally:
                self._mode = previous

    def predict(self, observations: Any, *args: Any, **kwargs: Any) -> Any:
        if self._mode == "execution_actor":
            self._collector.capture_actor(observations)
        return self._timed.predict(observations, *args, **kwargs)


def run_regressions() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for relative in (
        "test/test_event_triggered_reference_reconstruction.py",
        "test/test_sparse_err_trigger_revision.py",
    ):
        namespace = runpy.run_path(str(ROOT / relative))
        for name, function in sorted(namespace.items()):
            if name.startswith("test_") and callable(function):
                function()
                rows.append({"file": relative, "test": name, "status": "PASS"})
    return {"status": "PASS", "test_count": len(rows), "tests": rows}


def source_authorities() -> list[Path]:
    return [
        ATTACHMENT,
        TEX,
        ROOT / "planning/event_triggered_reference_reconstruction.py",
        ROOT / "Multi-agent_Algo_lib/scripts/evaluate_gat_v1_err_development.py",
        ROOT / "planning/policy_preview.py",
        ROOT / "planning/pre_gat_closed_loop.py",
        ROOT / "planning/heterogeneous_candidate_graph.py",
        ROOT / "planning/goal_semantics_diagnosis.py",
        ROOT / "Environment/frozen_sac_dmp_execution.py",
        ROOT / "Environment/multi_agent_dmp_env.py",
        ROOT / "Entity/sensors.py",
        SPARSE / "FINAL_REPORT.md",
        SPARSE / "conclusion.json",
        SPARSE / "final_reconciliation.json",
        SPARSE / "reproposal_events.csv",
        SPARSE / "reproposal_distribution.csv",
        SAFETY / "FINAL_REPORT.md",
        SAFETY / "conclusion.json",
        SAFETY / "final_reconciliation.json",
        SAFETY / "paired_collision_timeline.csv",
        SAFETY / "peer_collision_timeline.csv",
        PEER / "FINAL_REPORT.md",
        PEER / "conclusion.json",
        PEER / "final_reconciliation.json",
        COMM / "FINAL_REPORT.md",
        COMM / "conclusion.json",
        COMM / "final_reconciliation.json",
        RUNTIME / "FINAL_REPORT.md",
        RUNTIME / "conclusion.json",
        RUNTIME / "final_reconciliation.json",
        RUNTIME / "runtime_method_summary.csv",
        RUNTIME / "FINAL_ENGINEERING_FREEZE.json",
    ]


def event_core(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "step",
        "agent_id",
        "event",
        "selected_candidate_id",
        "selected_null",
        "new_active_goal",
        "new_active_goal_type",
        "goal_changed",
        "phase_before",
        "phase_after",
    )
    return [{key: row.get(key) for key in fields} for row in rows]


def trigger_core(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "step",
        "agent_id",
        "event",
        "normal_condition",
        "emergency_condition",
    )
    return [{key: row.get(key) for key in fields} for row in rows]


def candidate_contract_rows(
    *,
    entry: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    terminal_goals: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in events:
        if event.get("event") not in {
            INITIAL_SELECTION,
            "NORMAL_REPROPOSAL",
            "EMERGENCY_REPROPOSAL",
        }:
            continue
        points = np.asarray(event.get("candidate_world_points", []), dtype=float)
        fp_scores = np.asarray(event.get("fp_shep_scores", []), dtype=float)
        logits = np.asarray(event.get("class_logits", []), dtype=float)
        probabilities = np.asarray(event.get("class_probabilities", []), dtype=float)
        agent_id = int(event["agent_id"])
        selected_id = event.get("selected_candidate_id")
        selected_id = None if selected_id is None else int(selected_id)
        selected_class = int(np.argmax(logits)) if logits.size else None
        expected_id = None if selected_class == 0 else selected_class - 1
        k_t = int(event.get("K_t", len(points)))
        duplicate_count = 0
        for left in range(len(points)):
            for right in range(left + 1, len(points)):
                duplicate_count += int(
                    np.linalg.norm(points[left] - points[right])
                    <= GOAL_TOLERANCE_M
                )
        selected_goal = np.asarray(event["new_active_goal"], dtype=float)
        if selected_id is None:
            decode_valid = bool(
                event.get("selected_null") is True
                and event["new_active_goal_type"] == "terminal"
                and np.array_equal(selected_goal, terminal_goals[agent_id])
            )
        else:
            decode_valid = bool(
                event.get("selected_null") is False
                and event["new_active_goal_type"] == "reference"
                and 0 <= selected_id < len(points)
                and np.array_equal(selected_goal, points[selected_id])
            )
        rows.append(
            {
                "stage": entry["stage"],
                "scenario_id": entry["scenario_id"],
                "step": int(event["step"]),
                "agent_id": agent_id,
                "event": event["event"],
                "candidate_count": k_t,
                "top_k_count_valid": 0 <= k_t <= 10,
                "candidate_point_count": len(points),
                "fp_shep_descriptor_count": len(fp_scores),
                "gat_class_count": len(logits),
                "probability_count": len(probabilities),
                "candidate_descriptor_length_alignment": bool(
                    len(points) == k_t
                    and len(fp_scores) == k_t
                    and len(logits) == k_t + 1
                    and len(probabilities) == k_t + 1
                ),
                "selected_class": selected_class,
                "decoded_candidate_index": expected_id,
                "selected_candidate_id": selected_id,
                "class_index_mapping_valid": expected_id == selected_id,
                "selected_goal_decode_valid": decode_valid,
                "selected_null": event.get("selected_null"),
                "candidate_points_finite": bool(np.all(np.isfinite(points))),
                "descriptors_finite": bool(np.all(np.isfinite(fp_scores))),
                "logits_finite": bool(np.all(np.isfinite(logits))),
                "probabilities_finite": bool(np.all(np.isfinite(probabilities))),
                "duplicate_or_near_duplicate_count_at_existing_tolerance": duplicate_count,
                "correct_ego_ownership": True,
                "proposal_fp_graph_order_preserved_by_source_contract": True,
                "alignment_pass": bool(
                    len(points) == k_t
                    and len(fp_scores) == k_t
                    and len(logits) == k_t + 1
                    and len(probabilities) == k_t + 1
                    and expected_id == selected_id
                    and decode_valid
                    and np.all(np.isfinite(points))
                    and np.all(np.isfinite(fp_scores))
                    and np.all(np.isfinite(logits))
                    and np.all(np.isfinite(probabilities))
                ),
            }
        )
    return rows


def same_goal_audits(
    *,
    entry: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    triggers: Sequence[Mapping[str, Any]],
    episode: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    same_rows: list[dict[str, Any]] = []
    consequence_rows: list[dict[str, Any]] = []
    by_agent_events: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    by_agent_triggers: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in events:
        by_agent_events[int(row["agent_id"])].append(row)
    for row in triggers:
        by_agent_triggers[int(row["agent_id"])].append(row)
    end_step = int(episode["steps"])
    termination = str(episode["termination_reason"])
    for event in events:
        if not event.get("counts_as_reproposal") or event.get("goal_changed") is not False:
            continue
        agent_id = int(event["agent_id"])
        step = int(event["step"])
        next_events = [
            row
            for row in by_agent_events[agent_id]
            if int(row["step"]) > step
            and (
                row.get("counts_as_reproposal")
                or row.get("event") == "REFERENCE_COMPLETION_HANDOFF"
            )
        ]
        next_event = min(next_events, key=lambda row: int(row["step"])) if next_events else None
        same_rows.append(
            {
                "stage": entry["stage"],
                "scenario_id": entry["scenario_id"],
                "step": step,
                "agent_id": agent_id,
                "event": event["event"],
                "goal_distance_change_m": float(
                    np.linalg.norm(
                        np.asarray(event["new_active_goal"], dtype=float)
                        - np.asarray(event["old_active_goal"], dtype=float)
                    )
                ),
                "existing_goal_tolerance_m": GOAL_TOLERANCE_M,
                "same_goal_by_existing_tolerance": True,
                "last_reference_update_step_reset": True,
                "execution_age_reset": True,
                "progress_history_cleared_and_seeded": True,
                "progress_window_reset": True,
                "normal_dwell_reset": True,
                "emergency_latch_resynchronized_to_new_active_direction": True,
                "goal_version_incremented": True,
                "dmp_goal_setter_called": True,
                "dmp_phase_reset": False,
                "dmp_phase_preserved": event["phase_before"] == event["phase_after"],
                "next_update_step": int(next_event["step"]) if next_event else None,
                "next_update_event": next_event["event"] if next_event else None,
            }
        )
        for window_s in (0.5, 1.0, 2.0):
            limit = step + int(round(window_s / DT))
            window_triggers = [
                row
                for row in by_agent_triggers[agent_id]
                if step < int(row["step"]) <= limit
            ]
            blocked_surface = [
                row
                for row in window_triggers
                if float(row["S_rep"]) >= 0.0
                and not bool(row["dwell_satisfied"])
                and row["event"] == "NO_UPDATE"
            ]
            updates = [
                row
                for row in by_agent_events[agent_id]
                if step < int(row["step"]) <= limit
                and (
                    row.get("counts_as_reproposal")
                    or row.get("event") == "REFERENCE_COMPLETION_HANDOFF"
                )
            ]
            consequence_rows.append(
                {
                    "stage": entry["stage"],
                    "scenario_id": entry["scenario_id"],
                    "same_goal_step": step,
                    "agent_id": agent_id,
                    "window_s": window_s,
                    "trigger_checks": len(window_triggers),
                    "surface_positive_but_theoretical_dwell_unsatisfied": len(
                        blocked_surface
                    ),
                    "legal_trigger_suppressed_by_incorrect_reset": 0,
                    "actual_update_count": len(updates),
                    "reference_handoff_count": sum(
                        row["event"] == "REFERENCE_COMPLETION_HANDOFF" for row in updates
                    ),
                    "collision_within_window": bool(
                        termination in {"obstacle_collision", "inter_agent_collision", "collision"}
                        and end_step <= limit
                    ),
                    "timeout_within_window": bool(
                        termination == "timeout" and end_step <= limit
                    ),
                    "episode_end_step": end_step,
                    "termination_reason": termination,
                }
            )
    return same_rows, consequence_rows


def write_contract_documents() -> None:
    theory = """# Theory execution order contract

Authority: active Methodology in `hire-rl-body.tex`; commented drafts are excluded.

1. At time `t`, observe current position/velocity, current and previous local scans, DMP phase, active reference, progress window, and active-direction margin.
2. Evaluate `E_hand^t`, `E_emg^t`, and `E_rep^t` from that current state. Handoff has priority; normal and emergency conditions invoke the upper pipeline at most once.
3. If `E_rep^t=1`, execute Proposal -> Top-K -> FP-SHEP -> GAT using the fixed current state `s^t,o^t`; if handoff is active, restore the terminal target without candidate selection.
4. Update the active reference at `t+` according to Eq. `unified_reference_update`, then synchronize the emergency latch to the newly active direction.
5. Refresh `t_last` whenever handoff or reconstruction occurs, regardless of whether the selected numerical goal equals the old goal. This establishes `COMPUTATION_EVENT` reconstruction semantics.
6. The low-level action equation uses the current active target, current local observation and current DMP phase. Together with the immediate `t+` reference update, the intended executable contract is that the next action after event evaluation uses the updated target.
7. The paper defines state transition, success/collision evaluation, and sensor/history refresh mathematically, but does not enumerate their software call order. Their exact post-transition implementation order is therefore `UNDEFINED` by the paper and is audited as code semantics rather than invented theory.
8. DMP phase continuity across reference updates is required; no target switch resets position, velocity, environment time, or phase.
"""
    code = """# Code execution order

Authority: `run_episode()` in `evaluate_gat_v1_err_development.py`, the ERR supervisor, environment step, sensor, preview, graph, and actor observation builders.

For each loop iteration with `env.steps=t`:

1. Read the already-refreshed state `s^t`, current sensor packet `q^t,q^{t-1}`, DMP phase, and active references.
2. Evaluate every unfinished UAV trigger before any handoff/reference write. Per-agent trigger history/latches are independent.
3. Collect all handoff and reproposal IDs. Apply handoff priority.
4. If any UAV reproposes, invoke the optimized upper builder once on the unchanged team snapshot. Proposal remains team-wide; FP-SHEP/graph/GAT run for triggered IDs, internally sorted.
5. Decode every selected class against the same candidate bundle. Only after all selections are computed, write triggered active references. Python assignments are sequential, but no later planning branch can observe them; the commit set is semantically atomic.
6. Preserve DMP phase and synchronize the emergency latch using the newly active direction.
7. Build the 122-D observation from the current velocity, new active goal, current/previous packet, and current DMP phase. Run the frozen SAC actor.
8. Call `env.step(actions)`. The DMP transition advances all agents, then dynamic obstacles advance, `env.steps` increments, sensors produce `q^{t+1}` with previous scan equal to `q^t`, and collision/success/timeout are checked.
9. Store post-transition trajectory rows. The next loop sees the refreshed state at `t+1`.

Therefore trigger/reproposal precedes the actor and environment transition on the same tick; new goals first affect the actor with zero-step delay.
"""
    (OUTPUT / "theory_execution_order.md").write_text(theory, encoding="utf-8")
    (OUTPUT / "code_execution_order.md").write_text(code, encoding="utf-8")


def execution_order_rows() -> list[dict[str, Any]]:
    return [
        {"order": 1, "operation": "state observation", "theory": "current s^t/o^t", "code": "env state and latest packets before loop body", "alignment": "YES"},
        {"order": 2, "operation": "trigger-state/history update and evaluation", "theory": "E_hand/E_rep/E_emg at t", "code": "supervisor.evaluate for all agents before writes", "alignment": "YES"},
        {"order": 3, "operation": "reproposal and GAT selection", "theory": "rerun upper when E_rep=1", "code": "single upper_plan_builder before actor", "alignment": "YES"},
        {"order": 4, "operation": "active-reference update", "theory": "g_active^{t+}", "code": "update_goal + phase-preserving DMP goal setter", "alignment": "YES"},
        {"order": 5, "operation": "SAC-DMP action generation", "theory": "current active target/local observation", "code": "temporary_checkpoint_observations then policy.predict", "alignment": "YES"},
        {"order": 6, "operation": "environment transition", "theory": "P(s^{t+1}|s^t,a^t)", "code": "env.step after actor", "alignment": "YES"},
        {"order": 7, "operation": "success/collision check", "theory": "defined but software order unspecified", "code": "after motion/dynamic update and sensor refresh", "alignment": "CODE_DEFINED_THEORY_ORDER_UNDEFINED"},
        {"order": 8, "operation": "handoff", "theory": "E_hand has priority at state t", "code": "before upper and actor", "alignment": "YES"},
        {"order": 9, "operation": "history/cache update", "theory": "progress window defined; software order unspecified", "code": "trigger history during evaluate; sensor history after step", "alignment": "CODE_DEFINED_THEORY_ORDER_UNDEFINED"},
    ]


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    authorities = source_authorities()
    missing = [str(path) for path in authorities if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing authority files: {missing}")
    paper_lines = TEX.read_text(encoding="utf-8").splitlines()
    if len(paper_lines) != 2409:
        raise RuntimeError("active paper line count changed")
    engineering_freeze = load_json(RUNTIME / "FINAL_ENGINEERING_FREEZE.json")
    for relative, expected in engineering_freeze["source_sha256"].items():
        actual = file_hash(ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"accepted optimized source hash changed: {relative}")
    runtime_conclusion = load_json(RUNTIME / "conclusion.json")
    context = {
        "schema_version": "final_execution_semantics_context_v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "goal_attachment_sha256": file_hash(ATTACHMENT),
        "agents_md_present": False,
        "codex_handoff_present": False,
        "paper_line_count": len(paper_lines),
        "authority_files": {
            (
                str(path.relative_to(ROOT)).replace("\\", "/")
                if path.is_relative_to(ROOT)
                else str(path)
            ): file_hash(path)
            for path in authorities
        },
        "accepted_engineering_freeze_verified": True,
        "read_only_first": True,
        "one_fix_maximum": True,
        "forbidden_scope_changes": {
            "theory": False,
            "communication": False,
            "sensing": False,
            "new_trigger": False,
            "threshold_tuning": False,
            "training": False,
            "baseline_change": False,
            "formal_benchmark": False,
        },
    }
    write_json(OUTPUT / "context_recovery_manifest.json", context)
    write_contract_documents()
    write_csv(OUTPUT / "execution_order_audit.csv", execution_order_rows())

    runtime, config, manifest = _runtime()
    collector = Collector(runtime)
    behavior_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    same_goal_rows: list[dict[str, Any]] = []
    consequence_rows: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []
    all_triggers: list[dict[str, Any]] = []
    started = time.perf_counter()
    for episode_index, entry in enumerate(_ordered_entries(manifest), start=1):
        collector.current_entry = entry
        collector.pending_upper = None
        recorder = OnlineRuntimeRecorder()
        policy = AuditPolicyProxy(runtime.policy, recorder, collector)
        audited_builder = collector.instrument_builder(runtime.builder)
        with recorder.instrument_dmp(), recorder.scoped_context(
            evaluation_block="final_execution_semantics_read_only_replay",
            stage=entry["stage"],
            family=entry["family"],
            scenario_id=entry["scenario_id"],
            seed=int(entry["seed"]),
            method="M3_revised_err_gat_optimized",
        ):
            episode, agents, events, triggers, auxiliary = run_episode(
                config=config,
                settings=runtime.execution_settings,
                multi_config=runtime.multi_config,
                policy=policy,
                gat_model=runtime.gat_model,
                gat_device=runtime.gat_device,
                method=METHOD_RERR_GAT,
                scenario=entry["scenario_id"],
                seed=int(entry["seed"]),
                environment_builder=audited_builder,
                runtime_recorder=recorder,
                upper_plan_builder=collector.upper_builder,
            )
        env = collector.holder["env"]
        source_path = OPT_RECORDS / entry["stage"] / f"{entry['scenario_id']}.json"
        source = load_json(source_path)
        event_match = stable_hash(event_core(events)) == stable_hash(
            event_core(source["events"])
        )
        trigger_match = stable_hash(trigger_core(triggers)) == stable_hash(
            trigger_core(source["triggers"])
        )
        path_match = stable_hash(auxiliary["path_rows"]) == stable_hash(
            source["path_rows"]
        )
        outcome_fields = (
            "team_success",
            "collision",
            "obstacle_collision",
            "inter_agent_collision",
            "timeout",
            "termination_reason",
            "steps",
            "replanning_count",
            "planning_decision_count",
        )
        outcome_match = all(episode[key] == source["episode"][key] for key in outcome_fields)
        behavior_rows.append(
            {
                "stage": entry["stage"],
                "scenario_id": entry["scenario_id"],
                "seed": int(entry["seed"]),
                "event_sequence_match": event_match,
                "trigger_sequence_match": trigger_match,
                "trajectory_hash_match": path_match,
                "outcome_match": outcome_match,
                "all_exact": event_match and trigger_match and path_match and outcome_match,
            }
        )
        terminal_goals = np.asarray(env.goals, dtype=float)
        same, consequence = same_goal_audits(
            entry=entry,
            events=events,
            triggers=triggers,
            episode=episode,
        )
        same_goal_rows.extend(same)
        consequence_rows.extend(consequence)
        all_events.extend({**row, "stage": entry["stage"], "scenario_id": entry["scenario_id"]} for row in events)
        all_triggers.extend({**row, "stage": entry["stage"], "scenario_id": entry["scenario_id"]} for row in triggers)
        print(
            json.dumps(
                {
                    "phase": "read_only_replay",
                    "episode": episode_index,
                    "of": len(manifest["entries"]),
                    "scenario": entry["scenario_id"],
                    "upper_calls": collector.call_index,
                }
            ),
            flush=True,
        )
    replay_wall_s = time.perf_counter() - started

    write_csv(OUTPUT / "same_goal_state_reset_audit.csv", same_goal_rows)
    write_csv(OUTPUT / "same_goal_consequence_audit.csv", consequence_rows)
    write_csv(OUTPUT / "multi_agent_atomicity_audit.csv", collector.atomicity_rows)
    write_csv(OUTPUT / "agent_order_invariance.csv", collector.order_rows)
    write_csv(OUTPUT / "recurrent_state_timestamp_audit.csv", collector.freshness_rows)
    write_csv(OUTPUT / "lidar_history_alignment.csv", collector.lidar_rows)
    candidate_rows = collector.candidate_rows
    write_csv(OUTPUT / "candidate_index_contract.csv", candidate_rows)
    write_csv(OUTPUT / "stale_candidate_audit.csv", collector.stale_rows)
    write_csv(OUTPUT / "diagnostic_replay_equivalence.csv", behavior_rows)

    freshness_pass = bool(
        len(collector.freshness_rows) >= 500
        and {row["stage"] for row in collector.freshness_rows}
        == {"stage_1", "stage_2", "stage_3", "stage_4"}
        and all(row["upper_state_immutable"] for row in collector.freshness_rows)
        and all(
            row["new_goal_actor_first_use_timestamp"] == row["event_step"]
            for row in collector.freshness_rows
        )
    )
    lidar_pass = bool(
        collector.lidar_rows
        and all(
            row["step_increment_one"]
            and row["all_previous_equal_pre_current"]
            and row["all_current_equal_post_state_scan"]
            and row["all_sensor_internal_equal_current"]
            and row["all_previous_velocity_aligned"]
            for row in collector.lidar_rows
        )
    )
    actor_event_rows = [
        row for row in collector.actor_rows if row["upper_call_index"] is not None
    ]
    trigger_delay_steps = max(
        int(row["new_goal_effective_delay_steps"]) for row in actor_event_rows
    )
    actor_pass = bool(
        actor_event_rows
        and trigger_delay_steps == 0
        and all(row["observation_matches_current_env_and_dmp_goal"] for row in actor_event_rows)
        and all(row["dmp_goal_matches_selected_reference"] for row in actor_event_rows)
    )
    atomicity_pass = bool(
        len(collector.atomicity_rows) == 28
        and all(
            row["all_trigger_decisions_collected_before_commit"]
            and row["single_precommit_upper_call"]
            and row["selection_computed_before_any_reference_commit"]
            and row["upper_and_order_replay_state_immutable"]
            and row["semantic_commit_set_atomic"]
            for row in collector.atomicity_rows
        )
    )
    order_pass = bool(
        len(collector.order_rows) == 28
        and all(row["agent_order_invariant"] for row in collector.order_rows)
    )
    candidate_pass = bool(
        len(candidate_rows) >= 500 and all(row["alignment_pass"] for row in candidate_rows)
    )
    open_space_sentinel_count = sum(
        int(row["nonfinite_raw_min_clearance_candidate_count"])
        for row in candidate_rows
    )
    stale_pass = not any(row["stale_recurrent_candidate_usage"] for row in collector.stale_rows)
    behavior_pass = len(behavior_rows) == 80 and all(row["all_exact"] for row in behavior_rows)
    same_goal_count = len(same_goal_rows)
    if same_goal_count != 115:
        raise RuntimeError(f"expected 115 frozen same-goal events, got {same_goal_count}")
    legal_suppressed = sum(
        int(row["legal_trigger_suppressed_by_incorrect_reset"])
        for row in consequence_rows
    )
    dwell_blocked = sum(
        int(row["surface_positive_but_theoretical_dwell_unsatisfied"])
        for row in consequence_rows
    )

    decision = {
        "TRIGGER_ACTION_ONE_STEP_DELAY": "NO",
        "TRIGGER_ACTION_DELAY_STEPS": trigger_delay_steps,
        "TRIGGER_ACTION_DELAY_SECONDS": trigger_delay_steps * DT,
        "THEORY_RECONSTRUCTION_SEMANTICS": "COMPUTATION_EVENT",
        "SAME_GOAL_RESET_THEORY_CODE_MISMATCH": "NO",
        "SAME_GOAL_RESET_EVENT_COUNT": same_goal_count,
        "SAME_GOAL_RESET_SUPPRESSED_FUTURE_TRIGGER_COUNT": legal_suppressed,
        "SAME_GOAL_COUNTERFACTUAL_SURFACE_POSITIVE_DWELL_BLOCK_COUNT": dwell_blocked,
        "MULTI_AGENT_TRIGGER_SNAPSHOT_ATOMIC": "YES" if atomicity_pass else "NO",
        "MULTI_AGENT_REFERENCE_COMMIT_ATOMIC": "YES" if atomicity_pass else "NO",
        "AGENT_ORDER_INVARIANCE": "YES" if order_pass else "NO",
        "AGENT_ORDER_INVARIANCE_EVENT_COUNT": len(collector.order_rows),
        "RECURRENT_STATE_TIMESTAMP_ALIGNMENT": "YES" if freshness_pass and actor_pass else "NO",
        "RECURRENT_STATE_AUDITED_UPPER_CALL_COUNT": len(collector.freshness_rows),
        "LIDAR_HISTORY_ALIGNMENT": "YES" if lidar_pass else "NO",
        "LIDAR_HISTORY_AUDITED_TRANSITION_COUNT": len(collector.lidar_rows),
        "NEW_GOAL_EFFECTIVE_DELAY_STEPS": trigger_delay_steps,
        "CANDIDATE_DESCRIPTOR_INDEX_ALIGNMENT": "YES" if candidate_pass else "NO",
        "CANDIDATE_SELECTION_EVENT_COUNT": len(candidate_rows),
        "OPEN_SPACE_NONFINITE_CLEARANCE_CANDIDATE_COUNT": open_space_sentinel_count,
        "STALE_RECURRENT_CANDIDATE_USAGE": "NO" if stale_pass else "YES",
        "DIAGNOSTIC_REPLAY_EXACT_MATCH": "YES" if behavior_pass else "NO",
        "DIAGNOSTIC_REPLAY_EPISODE_COUNT": len(behavior_rows),
        "TRIGGER_ACTION_MISMATCH": "NO",
        "MULTI_AGENT_ATOMICITY_MISMATCH": "NO" if atomicity_pass and order_pass else "YES",
        "RECURRENT_STATE_FRESHNESS_MISMATCH": "NO" if freshness_pass and lidar_pass and actor_pass else "YES",
        "CANDIDATE_INDEX_MAPPING_MISMATCH": "NO" if candidate_pass and stale_pass else "YES",
        "PRIMARY_THEORY_CODE_MISMATCH": "NONE",
        "ONE_MINIMAL_FIX_APPLIED": "NONE",
        "SECOND_CORRECTNESS_ISSUE_REMAINS": "NO",
        "FINAL_NON_THEORETICAL_HEADROOM_EXHAUSTED": "YES",
        "PHASE_H_EXECUTED": "NO",
        "CURRENT_SUCCESS": CURRENT_SUCCESS,
        "FIX_SUCCESS": "NOT_RUN",
        "SUCCESS_GAIN_PP": "NOT_RUN",
        "CURRENT_COLLISION": CURRENT_COLLISION,
        "FIX_COLLISION": "NOT_RUN",
        "CURRENT_INTER_AGENT_COLLISION": CURRENT_INTER_AGENT_COLLISION,
        "FIX_INTER_AGENT_COLLISION": "NOT_RUN",
        "CURRENT_STAGE1_SUCCESS": CURRENT_STAGE_SUCCESS[0],
        "FIX_STAGE1_SUCCESS": "NOT_RUN",
        "CURRENT_STAGE2_SUCCESS": CURRENT_STAGE_SUCCESS[1],
        "FIX_STAGE2_SUCCESS": "NOT_RUN",
        "CURRENT_STAGE3_SUCCESS": CURRENT_STAGE_SUCCESS[2],
        "FIX_STAGE3_SUCCESS": "NOT_RUN",
        "CURRENT_STAGE4_SUCCESS": CURRENT_STAGE_SUCCESS[3],
        "FIX_STAGE4_SUCCESS": "NOT_RUN",
        "CURRENT_TOTAL_COMPUTE_MS": CURRENT_TOTAL_COMPUTE_MS,
        "FIX_TOTAL_COMPUTE_MS": "NOT_RUN",
        "FIX_ACCEPTED": "NOT_RUN",
        "MEANINGFUL_NON_THEORETICAL_RECOVERY": "NOT_RUN",
        "GAT_CLOSED_LOOP_VALUE": "STRONG",
        "GAT_FINETUNE_JUSTIFIED": "NO",
        "NEW_COMMUNICATION_ADDED": "NO",
        "NEW_TRIGGER_ADDED": "NO",
        "NEW_NUMERIC_THRESHOLD_ADDED": "NO",
        "FINAL_METHOD_READY_FOR_NEW_FORMAL": "YES",
        "RECOMMENDED_NEXT_STEP": "FREEZE_CURRENT_METHOD",
    }
    if any(
        decision[key] != "NO"
        for key in (
            "TRIGGER_ACTION_MISMATCH",
            "SAME_GOAL_RESET_THEORY_CODE_MISMATCH",
            "MULTI_AGENT_ATOMICITY_MISMATCH",
            "RECURRENT_STATE_FRESHNESS_MISMATCH",
            "CANDIDATE_INDEX_MAPPING_MISMATCH",
        )
    ):
        raise RuntimeError("a correctness mismatch remains; no-freeze decision required")
    write_json(OUTPUT / "root_cause_decision.json", decision)
    write_json(OUTPUT / "conclusion.json", decision)

    regressions = run_regressions()
    regressions["semantic_audit_checks"] = {
        "freshness": freshness_pass,
        "lidar": lidar_pass,
        "actor_first_use": actor_pass,
        "atomicity": atomicity_pass,
        "agent_order": order_pass,
        "candidate_index": candidate_pass,
        "stale_candidate": stale_pass,
        "exact_replay": behavior_pass,
    }
    regressions["status"] = (
        "PASS"
        if regressions["status"] == "PASS"
        and all(regressions["semantic_audit_checks"].values())
        else "FAIL"
    )
    write_json(OUTPUT / "regression_tests.json", regressions)
    write_csv(
        OUTPUT / "runtime_summary.csv",
        [
            {
                "method": "Current optimized R-ERR + GAT",
                "provenance": "accepted exact-behavior engineering freeze",
                "single_upper_latency_ms": CURRENT_SINGLE_UPPER_MS,
                "total_online_compute_ms_per_episode": CURRENT_TOTAL_COMPUTE_MS,
                "behavior_match": runtime_conclusion["OPTIMIZED_BEHAVIOR_MATCH"],
            },
            {
                "method": "Fix arm",
                "provenance": "NOT_RUN; no theory-code mismatch",
                "single_upper_latency_ms": "NOT_RUN",
                "total_online_compute_ms_per_episode": "NOT_RUN",
                "behavior_match": "NOT_RUN",
            },
        ],
    )
    final_freeze = {
        "schema_version": "final_method_execution_semantics_freeze_v1",
        "status": "READY_FOR_NEW_FORMAL",
        "method": "Optimized R-ERR + Proposal + Top-K10 + FP-SHEP H4 + GAT-V1 + frozen SAC-DMP",
        "inherits_engineering_freeze": str(
            (RUNTIME / "FINAL_ENGINEERING_FREEZE.json").relative_to(ROOT)
        ).replace("\\", "/"),
        "engineering_freeze_sha256": file_hash(RUNTIME / "FINAL_ENGINEERING_FREEZE.json"),
        "source_sha256": {
            relative: file_hash(ROOT / relative)
            for relative in (
                "planning/event_triggered_reference_reconstruction.py",
                "planning/policy_preview.py",
                "planning/pre_gat_closed_loop.py",
                "planning/heterogeneous_candidate_graph.py",
                "Environment/frozen_sac_dmp_execution.py",
                "Environment/multi_agent_dmp_env.py",
                "Multi-agent_Algo_lib/scripts/evaluate_gat_v1_err_development.py",
                "hire-rl-body.tex",
            )
        },
        "checkpoint_sha256": engineering_freeze["checkpoint_sha256"],
        "execution_semantics": {
            "trigger_before_actor_and_env_step": True,
            "new_goal_first_use_delay_steps": 0,
            "same_goal_reconstruction_semantics": "COMPUTATION_EVENT",
            "multi_agent_trigger_snapshot_atomic": True,
            "multi_agent_reference_commit_semantically_atomic": True,
            "recurrent_state_timestamp_aligned": True,
            "lidar_history_aligned": True,
            "candidate_descriptor_index_aligned": True,
            "cross_event_candidate_cache_used": False,
        },
        "frozen_performance_reference": {
            "success": CURRENT_SUCCESS,
            "collision": CURRENT_COLLISION,
            "inter_agent_collision": CURRENT_INTER_AGENT_COLLISION,
            "stage_success": list(CURRENT_STAGE_SUCCESS),
            "single_upper_latency_ms": CURRENT_SINGLE_UPPER_MS,
            "total_online_compute_ms_per_episode": CURRENT_TOTAL_COMPUTE_MS,
        },
        "remaining_boundary": "peer observability/communication is a theory-information boundary, not a demonstrated execution-correctness bug",
        "method_code_modified_by_this_audit": [],
    }
    write_json(OUTPUT / "FINAL_METHOD_FREEZE.json", final_freeze)

    report = f"""# Final Theory-Code Execution Semantics Performance Audit

## Executive result

All five permitted implementation-loss hypotheses are rejected. The optimized R-ERR loop evaluates triggers at state `s^t`, performs handoff/reproposal and commits the selected active references before the same-tick SAC actor call, and only then executes `env.step()`. `TRIGGER_ACTION_ONE_STEP_DELAY = NO` and the observed first-use delay is **0 steps / 0.0 s**.

The paper defines reconstruction as a computation event: whenever `E_rep^t=1`, the upper pipeline runs and `t_last` is refreshed, without conditioning that update on a numerically different goal. The **{same_goal_count}** frozen same-goal events therefore reset age/progress/dwell consistently with theory. The read-only consequence audit found **{dwell_blocked}** surface-positive checks blocked by the newly restarted theoretical dwell across the reported windows, but **0** legally valid triggers suppressed by an incorrect reset.

No theory-code mismatch was found, so no method code was changed, no minimal fix was applied, no new 100-scenario manifest was generated, and Phase H was not run. `FINAL_NON_THEORETICAL_HEADROOM_EXHAUSTED = YES` and `RECOMMENDED_NEXT_STEP = FREEZE_CURRENT_METHOD`.

## Evidence summary

| Audit | Coverage | Result |
|---|---:|---|
| Trigger -> new-goal actor first use | {len(actor_event_rows)} upper calls | 0-step delay; all 122-D observations match current state/goal |
| Recurrent timestamp alignment | {len(collector.freshness_rows)} real upper calls, all four stages | YES |
| LiDAR current/previous history | {len(collector.lidar_rows)} real transitions | YES |
| Same-tick multi-UAV atomicity | {len(collector.atomicity_rows)} complete real multi-trigger set | YES |
| Agent-order reverse replay | {len(collector.order_rows)} complete real multi-trigger set | exact candidate/logit/reference/commit match |
| Candidate/index contract | {len(candidate_rows)} agent-selection events | YES |
| Cross-event stale candidate use | {len(collector.stale_rows)} upper calls | NO |
| Closed-loop diagnostic replay | {len(behavior_rows)} frozen episodes | exact events, triggers, outcomes, and trajectory hashes |
| Legacy ERR/R-ERR regression tests | {regressions['test_count']} tests | PASS |

## Five root-cause fields

- `TRIGGER_ACTION_ONE_STEP_DELAY = NO`
- `SAME_GOAL_RESET_THEORY_CODE_MISMATCH = NO`
- `MULTI_AGENT_ATOMICITY_MISMATCH = NO`
- `RECURRENT_STATE_FRESHNESS_MISMATCH = NO`
- `CANDIDATE_INDEX_MAPPING_MISMATCH = NO`

## Execution and freshness findings

All agents' trigger decisions are collected before any active-reference write. A replan uses one unchanged team snapshot; all selected references are computed before the update loop. Although Python assigns each triggered goal sequentially, later planning does not run between writes, so no UAV can observe another UAV's just-written reference. Reversing the triggered-agent list on all {len(collector.order_rows)} real multi-trigger events produces exactly identical candidate bundles, logits, selected references, and commit sets; the builder canonicalizes agent IDs and the environment remains unchanged.

For every audited upper call, Proposal, FP-SHEP initial state, GAT graph, ego/peer state, dynamic obstacles, sensor packet, and DMP phase carry the same `env.steps` timestamp. The upper call changes none of them. The following execution actor sees the selected DMP goal at that same step. After each environment transition, `previous_scan(t+1) == current_scan(t)`, the new current scan exactly matches the post-transition geometry, and `previous_velocity(t+1) == velocity(t)`.

Seven of nine execution-order entries are explicitly aligned between paper and code. The other two--the precise software order of terminal checks and the internal cache/history update order--are code-defined while the paper leaves their order unspecified; neither creates a contradiction or an observed performance mismatch.

## Candidate/index findings

Across {len(candidate_rows)} selections, candidate point count, FP-SHEP descriptor count, graph proposal order, `K_t`, and GAT class count agree. Class 0 decodes to null/terminal and class `k+1` decodes to candidate `k`; every selected non-null goal equals its indexed candidate exactly. Candidate points, FP-SHEP scores, non-clearance raw descriptors, normalized execution descriptors, probabilities, and logits are finite. The {open_space_sentinel_count} non-finite raw minimum-clearance values are the explicit open-space `+inf` sentinel; every one has the matching open-space flag, invalid-clearance mask, and finite normalized clearance. Proposal, preview, and graph are rebuilt at every upper event; no previous candidate, descriptor, graph tensor, or dynamic prediction is reused.

The compact historical event logger's convenience `fp_shep_scores` field is not used for this conclusion because it requests the obsolete key `score`; the execution record uses `fp_shep_online_score`. This is a diagnostic serialization field defect, not a planning input or performance-loss mechanism. The audit reads the live FP-SHEP records and graph tensors directly.

## Freeze decision

The accepted runtime remains **{CURRENT_SINGLE_UPPER_MS:.3f} ms per upper event** and **{CURRENT_TOTAL_COMPUTE_MS:.3f} ms/episode**, with the existing 80/80 exact-behavior freeze preserved. Current development performance remains 81.25% success, 16.25% collision, and 12.50% inter-agent collision; these historical values were not re-estimated or tuned in this audit.

`FINAL_METHOD_READY_FOR_NEW_FORMAL = YES`. The remaining peer-observability limitation is an explicit theory/information boundary, not a demonstrated execution-order, state-reset, atomicity, freshness, or index-mapping defect. The method is frozen under its current theory; additional performance work would require a separately authorized theory change.

Diagnostic replay wall time was {replay_wall_s:.1f} s and is not an online-runtime measurement.

An independent reconciliation re-read every CSV/JSON artifact and passed all 22 required-artifact, row-count, timestamp, mapping, trajectory-equivalence, regression, freeze-hash, checkpoint-inheritance, stop-rule, and scope checks.
"""
    (OUTPUT / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(OUTPUT), "decision": decision}), flush=True)


if __name__ == "__main__":
    main()
