"""Read-only Peer-Risk Trigger Closure recoverability audit.

The native multi-agent observation contains a normalized/clipped ally block at
every environment step, although the frozen 122-D SAC checkpoint does not
consume it.  This script reconstructs only that already-generated observation
block from the frozen 80-record diagnostic replay, evaluates the preregistered
P1/P2/P3 candidates, and applies the support gate before any implementation.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts" / "peer_risk_trigger_closure" / "20260819_153224"
SPARSE = ROOT / "artifacts" / "sparse_err_trigger_revision" / "20260819_020727"
SAFETY = ROOT / "artifacts" / "rerr_safety_closure" / "20260819_134216"
EQUAL = ROOT / "artifacts" / "equal_information_baseline_audit" / "20260819_012339"
PREVIOUS = ROOT / "artifacts" / "peer_risk_trigger_closure" / "20260819_143612"
RECORD_ROOT = SAFETY / "diagnostic_records"

DT = 0.1
D_SAFE = 0.6
HORIZON_S = 0.4
POSITION_SCALE = 1.2
VELOCITY_SCALE = 4.0
SIGNALS = ("P1", "P2")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: list[str] | None = None) -> None:
    materialized = [dict(row) for row in rows]
    if fields is None:
        if not materialized:
            raise ValueError(f"fields required for empty CSV: {path}")
        fields = list(materialized[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)


def source_entry(path: Path, role: str) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "role": role,
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256(path) if path.is_file() else None,
    }


def vector(row: Mapping[str, Any], fields: tuple[str, str, str]) -> np.ndarray:
    return np.asarray([float(row[name]) for name in fields], dtype=float)


def state_maps(record: Mapping[str, Any]) -> dict[int, dict[int, tuple[np.ndarray, np.ndarray]]]:
    result: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]] = defaultdict(dict)
    for row in record["path_rows"]:
        step = int(row["step"])
        agent = int(row["agent_id"])
        result[step][agent] = (
            vector(row, ("x_m", "y_m", "z_m")),
            vector(row, ("vx_mps", "vy_mps", "vz_mps")),
        )
    return dict(result)


def native_ally_decode(
    ego: tuple[np.ndarray, np.ndarray],
    peer: tuple[np.ndarray, np.ndarray],
) -> dict[str, Any]:
    """Reconstruct and decode only the native clipped ally observation block."""

    ego_position, ego_velocity = ego
    peer_position, peer_velocity = peer
    exact_r = peer_position - ego_position
    exact_v = peer_velocity - ego_velocity
    exact_distance = float(np.linalg.norm(exact_r))
    relative_position_feature = np.clip(exact_r / POSITION_SCALE, -1.0, 1.0)
    relative_velocity_feature = np.clip(exact_v / VELOCITY_SCALE, -1.0, 1.0)
    distance_feature = float(np.clip(exact_distance / POSITION_SCALE, 0.0, 1.0))
    return {
        "relative_position": relative_position_feature * POSITION_SCALE,
        "relative_velocity": relative_velocity_feature * VELOCITY_SCALE,
        "distance": distance_feature * POSITION_SCALE,
        "position_saturated": bool(np.any(np.abs(relative_position_feature) >= 1.0)),
        "velocity_saturated": bool(np.any(np.abs(relative_velocity_feature) >= 1.0)),
    }


def pair_signal(
    ego: tuple[np.ndarray, np.ndarray],
    peer: tuple[np.ndarray, np.ndarray],
) -> dict[str, Any]:
    observed = native_ally_decode(ego, peer)
    relative_position = observed["relative_position"]
    relative_velocity = observed["relative_velocity"]
    velocity_norm_sq = float(relative_velocity @ relative_velocity)
    if velocity_norm_sq <= 1.0e-12:
        t_cpa = 0.0
    else:
        t_cpa = float(
            np.clip(
                -float(relative_position @ relative_velocity) / velocity_norm_sq,
                0.0,
                HORIZON_S,
            )
        )
    d_cpa = float(np.linalg.norm(relative_position + t_cpa * relative_velocity))
    return {
        "P1": bool(float(observed["distance"]) < D_SAFE),
        "P2": bool(d_cpa < D_SAFE),
        "current_distance_m": float(observed["distance"]),
        "t_cpa_s": t_cpa,
        "d_cpa_m": d_cpa,
        "position_saturated": observed["position_saturated"],
        "velocity_saturated": observed["velocity_saturated"],
    }


def load_records() -> list[dict[str, Any]]:
    paths = sorted(RECORD_ROOT.glob("stage_*/SR*/gat_v1_rerr.json"))
    if len(paths) != 80:
        raise RuntimeError(f"expected 80 frozen M3 records, got {len(paths)}")
    return [read_json(path) for path in paths]


def analyze_record(record: Mapping[str, Any]) -> dict[str, Any]:
    states = state_maps(record)
    steps = sorted(states)
    if not steps or steps[0] != 0:
        raise RuntimeError("diagnostic record must begin at step zero")
    for step in steps:
        if len(states[step]) != 3:
            raise RuntimeError("every frozen record must retain all three agents")

    pair_conditions: dict[str, dict[tuple[int, int], bool]] = {
        signal: {} for signal in SIGNALS
    }
    agent_conditions: dict[str, dict[int, bool]] = {
        signal: {agent: False for agent in range(3)} for signal in SIGNALS
    }
    pair_entries: dict[str, list[dict[str, Any]]] = {signal: [] for signal in SIGNALS}
    agent_entries: dict[str, list[dict[str, Any]]] = {signal: [] for signal in SIGNALS}

    for step_index, step in enumerate(steps):
        current_pair: dict[str, dict[tuple[int, int], bool]] = {
            signal: {} for signal in SIGNALS
        }
        current_agent: dict[str, dict[int, bool]] = {
            signal: {agent: False for agent in range(3)} for signal in SIGNALS
        }
        pair_diagnostics: dict[tuple[int, int], dict[str, Any]] = {}
        for i in range(3):
            for j in range(i + 1, 3):
                diagnostic = pair_signal(states[step][i], states[step][j])
                pair_diagnostics[(i, j)] = diagnostic
                for signal in SIGNALS:
                    current_pair[signal][(i, j)] = bool(diagnostic[signal])
                    current_agent[signal][i] |= bool(diagnostic[signal])
                    current_agent[signal][j] |= bool(diagnostic[signal])

        if step_index > 0:
            for signal in SIGNALS:
                for pair, condition in current_pair[signal].items():
                    if condition and not pair_conditions[signal][pair]:
                        diagnostic = pair_diagnostics[pair]
                        pair_entries[signal].append(
                            {
                                "signal": signal,
                                "scenario_id": record["entry"]["scenario_id"],
                                "stage": record["entry"]["stage"],
                                "step": step,
                                "time_s": step * DT,
                                "agent_i": pair[0],
                                "agent_j": pair[1],
                                "current_distance_m": diagnostic["current_distance_m"],
                                "t_cpa_s": diagnostic["t_cpa_s"],
                                "d_cpa_m": diagnostic["d_cpa_m"],
                                "position_saturated": diagnostic["position_saturated"],
                                "velocity_saturated": diagnostic["velocity_saturated"],
                                "team_success": bool(record["episode"]["team_success"]),
                                "inter_agent_collision": bool(record["episode"]["inter_agent_collision"]),
                                "mode": "READ_ONLY_NO_INTERVENTION",
                            }
                        )
                for agent, condition in current_agent[signal].items():
                    if condition and not agent_conditions[signal][agent]:
                        agent_entries[signal].append(
                            {
                                "signal": signal,
                                "scenario_id": record["entry"]["scenario_id"],
                                "stage": record["entry"]["stage"],
                                "step": step,
                                "time_s": step * DT,
                                "agent_id": agent,
                                "team_success": bool(record["episode"]["team_success"]),
                                "inter_agent_collision": bool(record["episode"]["inter_agent_collision"]),
                                "mode": "READ_ONLY_NO_INTERVENTION",
                            }
                        )
        pair_conditions = current_pair
        agent_conditions = current_agent

    collision_steps = sorted(
        {
            int(row["step"])
            for row in record["path_rows"]
            if bool(row.get("inter_agent_collision", False))
        }
    )
    collision_step = collision_steps[0] if collision_steps else None
    colliding_pairs: list[tuple[int, int]] = []
    if collision_step is not None:
        collision_states = states[collision_step]
        for i in range(3):
            for j in range(i + 1, 3):
                distance = float(np.linalg.norm(collision_states[i][0] - collision_states[j][0]))
                if distance <= D_SAFE + 1.0e-9:
                    colliding_pairs.append((i, j))

    return {
        "scenario_id": record["entry"]["scenario_id"],
        "stage": record["entry"]["stage"],
        "team_success": bool(record["episode"]["team_success"]),
        "inter_agent_collision": bool(record["episode"]["inter_agent_collision"]),
        "collision_step": collision_step,
        "colliding_pairs": colliding_pairs,
        "pair_entries": pair_entries,
        "agent_entries": agent_entries,
    }


def quantile(values: list[float], q: float) -> float | None:
    return float(np.quantile(np.asarray(values, dtype=float), q)) if values else None


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = load_records()
    analyses = [analyze_record(record) for record in records]

    source_files = [
        (SPARSE / "FINAL_REPORT.md", "frozen M3 development authority"),
        (SPARSE / "conclusion.json", "frozen M3 decision fields"),
        (SPARSE / "final_reconciliation.json", "frozen M3 reconciliation"),
        (SPARSE / "reproposal_events.csv", "frozen planning events"),
        (SPARSE / "runtime_method_summary.csv", "historical runtime reference"),
        (SAFETY / "FINAL_REPORT.md", "peer-risk root-cause authority"),
        (SAFETY / "conclusion.json", "peer-risk root-cause fields"),
        (SAFETY / "final_reconciliation.json", "safety audit reconciliation"),
        (SAFETY / "diagnostic_replay_reconciliation.json", "80-record replay integrity"),
        (SAFETY / "paired_collision_timeline.csv", "adverse paired timeline"),
        (SAFETY / "peer_collision_timeline.csv", "ten collision windows"),
        (SAFETY / "peer_collision_classification.csv", "ten collision classes"),
        (SAFETY / "normal_trigger_utility.csv", "normal trigger utility"),
        (SAFETY / "replanning_scope_audit.csv", "replanning scope"),
        (EQUAL / "FINAL_REPORT.md", "equal-information authority"),
        (EQUAL / "conclusion.json", "equal-information fields"),
        (EQUAL / "proposed_online_information_contract.csv", "prior Proposed information contract"),
        (PREVIOUS / "FINAL_REPORT.md", "prior conservative peer-trigger gate"),
        (PREVIOUS / "conclusion.json", "prior peer-trigger conclusion"),
        (ROOT / "Environment" / "multi_agent_dmp_env.py", "native ally observation implementation"),
        (ROOT / "Entity" / "sensors.py", "LiDAR packet implementation"),
        (ROOT / "planning" / "event_triggered_reference_reconstruction.py", "active R-ERR supervisor"),
        (ROOT / "planning" / "heterogeneous_candidate_graph.py", "upper-event peer graph"),
        (ROOT / "planning" / "candidate_execution_interface.py", "existing interaction diagnostic"),
        (ROOT / "planning" / "goal_semantics_diagnosis.py", "122-D frozen actor gate"),
        (ROOT / "hire-rl-body.tex", "active Methodology"),
    ]
    record_hashes = {
        path.relative_to(ROOT).as_posix(): sha256(path)
        for path in sorted(RECORD_ROOT.glob("stage_*/SR*/gat_v1_rerr.json"))
    }
    aggregate = hashlib.sha256()
    for path, digest in record_hashes.items():
        aggregate.update(path.encode("utf-8"))
        aggregate.update(digest.encode("ascii"))
    context = {
        "audit": "Peer-Risk Trigger Closure",
        "timestamp": "20260819_153224",
        "mode": "READ_ONLY_SIGNAL_GATE",
        "AGENTS_md_exists": (ROOT / "AGENTS.md").exists(),
        "CODEX_HANDOFF_md_exists": (ROOT / "CODEX_HANDOFF.md").exists(),
        "authorities": [source_entry(path, role) for path, role in source_files],
        "frozen_M3_record_count": len(record_hashes),
        "frozen_M3_record_aggregate_sha256": aggregate.hexdigest(),
        "frozen_M3_record_hashes": record_hashes,
        "frozen_facts": {
            "success": 0.8125,
            "collision": 0.1625,
            "inter_agent_collision": 0.125,
            "timeout": 0.025,
            "stage_success": [0.95, 0.85, 0.70, 0.75],
            "mean_reproposals": 8.9625,
            "total_online_compute_ms": 2271.5911549999996,
            "single_upper_latency_ms": 213.89047152145642,
        },
    }
    write_json(OUTPUT / "context_recovery_manifest.json", context)

    information_rows = [
        {
            "signal": "peer_relative_position_native_ally_block",
            "category": "C_CHEAP_BEFORE_GAT",
            "available_each_execution_step": "YES",
            "available_only_when_GAT_called": "NO",
            "representation": "relative position / 1.2 m, component-wise clipped to [-1,1]",
            "exactness": "PARTIAL_CLIPPED",
            "identity": "NO_STABLE_ID_IN_FLAT_BLOCK",
            "range_or_membership": "nearest_agent_observation_count; current config exposes all other agents",
            "source": "MultiAgentDMPEnv._compose_inter_agent_observation; get_observation called by every step",
            "legal_for_trigger": "YES_WITH_CLIPPING_DISCLOSED",
        },
        {
            "signal": "peer_relative_velocity_native_ally_block",
            "category": "C_CHEAP_BEFORE_GAT",
            "available_each_execution_step": "YES",
            "available_only_when_GAT_called": "NO",
            "representation": "relative velocity / 4 m/s, component-wise clipped to [-1,1]",
            "exactness": "PARTIAL_CLIPPED",
            "identity": "NO_STABLE_ID_IN_FLAT_BLOCK",
            "range_or_membership": "same ordered ally slots as relative position",
            "source": "MultiAgentDMPEnv._compose_inter_agent_observation; latest_observation",
            "legal_for_trigger": "YES_WITH_CLIPPING_DISCLOSED",
        },
        {
            "signal": "peer_absolute_position",
            "category": "A_PRIVATE_EXACT / B_UPPER_EVENT_EXACT / C_DERIVED_PARTIAL",
            "available_each_execution_step": "PARTIAL_ONLY_VIA_EGO_PLUS_CLIPPED_RELATIVE",
            "available_only_when_GAT_called": "EXACT_COMPANION_INTERFACE_USED_BY_GRAPH",
            "representation": "exact private dynamics; exact upper event; partial C derivation",
            "exactness": "PARTIAL_LEGAL_CONTINUOUS",
            "identity": "NO_IN_C",
            "range_or_membership": "same ally slots",
            "source": "observable_neighbor_states for B; native ally block for C",
            "legal_for_trigger": "NO_EXACT / YES_PARTIAL_RELATIVE_ONLY",
        },
        {
            "signal": "peer_absolute_velocity",
            "category": "A_PRIVATE_EXACT / B_UPPER_EVENT_EXACT / C_DERIVED_PARTIAL",
            "available_each_execution_step": "PARTIAL_ONLY_VIA_EGO_PLUS_CLIPPED_RELATIVE",
            "available_only_when_GAT_called": "EXACT_COMPANION_INTERFACE_USED_BY_GRAPH",
            "representation": "exact private dynamics; exact upper event; partial C derivation",
            "exactness": "PARTIAL_LEGAL_CONTINUOUS",
            "identity": "NO_IN_C",
            "range_or_membership": "same ally slots",
            "source": "observable_neighbor_states for B; native ally block for C",
            "legal_for_trigger": "NO_EXACT / YES_PARTIAL_RELATIVE_ONLY",
        },
        {
            "signal": "peer_identity",
            "category": "B_UPPER_EVENT_ONLY",
            "available_each_execution_step": "NO_IN_FLAT_ALLY_BLOCK",
            "available_only_when_GAT_called": "YES",
            "representation": "agent id in ObservableNeighborState / graph mapping",
            "exactness": "EXACT_AT_UPPER_EVENT",
            "identity": "YES_ONLY_IN_B",
            "range_or_membership": "event-local",
            "source": "observable_neighbor_states; graph.neighbor_node_to_agent_id",
            "legal_for_trigger": "NO",
        },
        {
            "signal": "NeighborGraphState",
            "category": "B_UPPER_EVENT",
            "available_each_execution_step": "NOT_CONSTRUCTED_BY_ACTIVE_EXECUTION_LOOP",
            "available_only_when_GAT_called": "YES",
            "representation": "exact position, velocity, id",
            "exactness": "EXACT_CURRENT_NO_FUTURE",
            "identity": "YES",
            "range_or_membership": "existing ally membership",
            "source": "build_heterogeneous_candidate_graph_from_env",
            "legal_for_trigger": "NO_AS_EXACT_B_INTERFACE",
        },
        {
            "signal": "current_peer_lidar_surface_return",
            "category": "C_CHEAP_BEFORE_GAT",
            "available_each_execution_step": "YES_IF_PEER_IS_NEAREST_HIT",
            "available_only_when_GAT_called": "NO",
            "representation": "56-ray nearest surface shared by static/dynamic/peer",
            "exactness": "EXACT_RANGE_RETURN_UNTYPED",
            "identity": "NO",
            "range_or_membership": "4.5 m",
            "source": "SensorPacket.current_scan",
            "legal_for_trigger": "GENERIC_ONLY_NOT_RELIABLE_P3_ASSOCIATION",
        },
        {
            "signal": "previous_peer_lidar_surface_return",
            "category": "C_CHEAP_BEFORE_GAT",
            "available_each_execution_step": "YES_AFTER_HISTORY",
            "available_only_when_GAT_called": "NO",
            "representation": "previous 56-ray nearest surface, untyped and unassociated",
            "exactness": "EXACT_PREVIOUS_RANGE_RETURN_UNTYPED",
            "identity": "NO",
            "range_or_membership": "4.5 m",
            "source": "SensorPacket.previous_scan",
            "legal_for_trigger": "NO_RELIABLE_OBJECT_ASSOCIATION",
        },
        {
            "signal": "peer_communication_state",
            "category": "UNAVAILABLE",
            "available_each_execution_step": "NO_FORMAL_CHANNEL",
            "available_only_when_GAT_called": "NO",
            "representation": "none",
            "exactness": "UNAVAILABLE",
            "identity": "UNAVAILABLE",
            "range_or_membership": "N/A",
            "source": "no execution feature or Methodology channel",
            "legal_for_trigger": "NO",
        },
        {
            "signal": "historical_typed_peer_observation",
            "category": "UNAVAILABLE",
            "available_each_execution_step": "NO",
            "available_only_when_GAT_called": "NO",
            "representation": "no typed peer track; only current ally block and untyped two-frame LiDAR",
            "exactness": "UNAVAILABLE",
            "identity": "NO",
            "range_or_membership": "N/A",
            "source": "no retained typed execution history",
            "legal_for_trigger": "NO",
        },
    ]
    write_csv(OUTPUT / "peer_online_information_contract.csv", information_rows)
    write_json(
        OUTPUT / "peer_information_legality.json",
        {
            "CONTINUOUS_PEER_POSITION_LEGAL": "PARTIAL",
            "CONTINUOUS_PEER_VELOCITY_LEGAL": "PARTIAL",
            "PEER_IDENTITY_LEGAL": "NO",
            "LEGAL_PEER_TRIGGER_SIGNAL_AVAILABLE": "YES",
            "legal_C_interface": "native normalized/clipped ally observation block computed in every get_observation call",
            "exact_peer_state_polling_legal": "NO",
            "upper_event_ObservableNeighborState_reclassified_as_C": "NO",
            "equal_information_alignment": "The 122-D SAC actor remains LiDAR-only. The trigger audit uses only the separately existing native ally observation block and never the exact upper-event companion interface.",
        },
    )

    collision_analyses = [item for item in analyses if item["inter_agent_collision"]]
    success_analyses = [item for item in analyses if item["team_success"]]
    if len(collision_analyses) != 10 or len(success_analyses) != 65:
        raise RuntimeError("frozen M3 categorical counts changed")

    lead_rows: list[dict[str, Any]] = []
    signal_summaries: list[dict[str, Any]] = []
    false_rows: list[dict[str, Any]] = []
    diagnostic_event_rows: list[dict[str, Any]] = []
    for signal in SIGNALS:
        leads: list[float] = []
        for item in collision_analyses:
            collision_step = int(item["collision_step"])
            relevant = [
                event
                for event in item["pair_entries"][signal]
                if tuple(sorted((int(event["agent_i"]), int(event["agent_j"]))))
                in item["colliding_pairs"]
                and int(event["step"]) <= collision_step
            ]
            first = min(relevant, key=lambda row: int(row["step"])) if relevant else None
            lead = (collision_step - int(first["step"])) * DT if first else None
            if lead is not None:
                leads.append(float(lead))
            lead_rows.append(
                {
                    "signal": signal,
                    "scenario_id": item["scenario_id"],
                    "stage": item["stage"],
                    "colliding_pairs": json.dumps(item["colliding_pairs"]),
                    "collision_step": collision_step,
                    "first_risk_entry_step": int(first["step"]) if first else "",
                    "lead_time_s": lead if lead is not None else "",
                    "detected": first is not None,
                    "ge_0_3s": bool(lead is not None and lead >= 0.3 - 1.0e-12),
                    "ge_0_5s": bool(lead is not None and lead >= 0.5 - 1.0e-12),
                    "ge_1_0s": bool(lead is not None and lead >= 1.0 - 1.0e-12),
                    "ge_2_0s": bool(lead is not None and lead >= 2.0 - 1.0e-12),
                    "mode": "READ_ONLY_NO_INTERVENTION",
                }
            )

        success_counts = [len(item["agent_entries"][signal]) for item in success_analyses]
        success_any = sum(count > 0 for count in success_counts)
        detected = len(leads)
        ge_03 = sum(value >= 0.3 - 1.0e-12 for value in leads)
        ge_05 = sum(value >= 0.5 - 1.0e-12 for value in leads)
        ge_10 = sum(value >= 1.0 - 1.0e-12 for value in leads)
        ge_20 = sum(value >= 2.0 - 1.0e-12 for value in leads)
        signal_summaries.append(
            {
                "signal": signal,
                "legal": "YES",
                "definition": (
                    "min observed current peer distance < existing d_safe"
                    if signal == "P1"
                    else "native ally relative state constant-velocity d_cpa < existing d_safe over existing H4=0.4s"
                ),
                "threshold_source": "MultiAgentEnvConfig.inter_agent_safe_distance=0.6m",
                "horizon_source": "existing FP-SHEP/GAT H4=0.4s" if signal == "P2" else "CURRENT_STATE",
                "collision_count": 10,
                "detected_count": detected,
                "detection_rate": detected / 10.0,
                "ge_0_3s_count": ge_03,
                "ge_0_5s_count": ge_05,
                "ge_1_0s_count": ge_10,
                "ge_2_0s_count": ge_20,
                "ge_0_5s_rate": ge_05 / 10.0,
                "lead_p25_s": quantile(leads, 0.25),
                "lead_median_s": quantile(leads, 0.5),
                "lead_p75_s": quantile(leads, 0.75),
                "lead_p90_s": quantile(leads, 0.90),
                "success_episode_count": 65,
                "success_episodes_with_event": success_any,
                "success_false_wakeup_rate": success_any / 65.0,
                "success_mean_agent_events_per_episode": float(np.mean(success_counts)),
                "success_median_agent_events_per_episode": float(np.median(success_counts)),
                "success_p90_agent_events_per_episode": float(np.quantile(success_counts, 0.9)),
                "success_max_agent_events_per_episode": max(success_counts),
                "support_gate": "NO" if ge_05 / 10.0 < 0.70 else "YES",
                "support_gate_failure": "GE_0_5S_LEAD_RATE_BELOW_0_70" if ge_05 / 10.0 < 0.70 else "NONE",
            }
        )
        false_rows.append(
            {
                "signal": signal,
                "success_episode_count": 65,
                "episodes_with_any_peer_wakeup": success_any,
                "false_wakeup_episode_rate": success_any / 65.0,
                "mean_agent_events_per_episode": float(np.mean(success_counts)),
                "median_agent_events_per_episode": float(np.median(success_counts)),
                "p90_agent_events_per_episode": float(np.quantile(success_counts, 0.9)),
                "max_agent_events_per_episode": max(success_counts),
                "mode": "READ_ONLY_NO_INTERVENTION",
            }
        )
        for item in analyses:
            diagnostic_event_rows.extend(item["agent_entries"][signal])

    signal_summaries.append(
        {
            "signal": "P3",
            "legal": "NO_RELIABLE_ASSOCIATION",
            "definition": "two-frame LiDAR peer range-rate proxy",
            "threshold_source": "existing d_safe",
            "horizon_source": "N/A",
            "collision_count": 10,
            "detected_count": "NOT_AVAILABLE",
            "detection_rate": "NOT_AVAILABLE",
            "ge_0_3s_count": "NOT_AVAILABLE",
            "ge_0_5s_count": "NOT_AVAILABLE",
            "ge_1_0s_count": "NOT_AVAILABLE",
            "ge_2_0s_count": "NOT_AVAILABLE",
            "ge_0_5s_rate": "NOT_AVAILABLE",
            "lead_p25_s": "NOT_AVAILABLE",
            "lead_median_s": "NOT_AVAILABLE",
            "lead_p75_s": "NOT_AVAILABLE",
            "lead_p90_s": "NOT_AVAILABLE",
            "success_episode_count": 65,
            "success_episodes_with_event": "NOT_AVAILABLE",
            "success_false_wakeup_rate": "NOT_AVAILABLE",
            "success_mean_agent_events_per_episode": "NOT_AVAILABLE",
            "success_median_agent_events_per_episode": "NOT_AVAILABLE",
            "success_p90_agent_events_per_episode": "NOT_AVAILABLE",
            "success_max_agent_events_per_episode": "NOT_AVAILABLE",
            "support_gate": "NO",
            "support_gate_failure": "UNTYPED_UNASSOCIATED_LIDAR_CANNOT_FORM_PEER_SPECIFIC_HISTORY",
        }
    )
    false_rows.append(
        {
            "signal": "P3",
            "success_episode_count": 65,
            "episodes_with_any_peer_wakeup": "NOT_AVAILABLE",
            "false_wakeup_episode_rate": "NOT_AVAILABLE",
            "mean_agent_events_per_episode": "NOT_AVAILABLE",
            "median_agent_events_per_episode": "NOT_AVAILABLE",
            "p90_agent_events_per_episode": "NOT_AVAILABLE",
            "max_agent_events_per_episode": "NOT_AVAILABLE",
            "mode": "NOT_COMPUTABLE_NO_RELIABLE_OBJECT_ASSOCIATION",
        }
    )
    write_csv(OUTPUT / "peer_risk_signal_audit.csv", signal_summaries)
    write_csv(OUTPUT / "peer_collision_lead_time.csv", lead_rows)
    write_csv(OUTPUT / "successful_false_wakeup.csv", false_rows)
    write_csv(OUTPUT / "peer_event_log.csv", diagnostic_event_rows)

    p2 = next(row for row in signal_summaries if row["signal"] == "P2")
    selection = {
        "LEGAL_PEER_TRIGGER_SIGNAL_AVAILABLE": "YES",
        "best_legal_candidate": "P2",
        "SELECTED_PEER_RISK_SIGNAL": "NONE",
        "PEER_TRIGGER_SIGNAL_SUPPORTED": "NO",
        "selection_rule": "support requires detection>=0.80 and >=0.70 of collisions with lead>=0.5s, no sustained repeated events, and no new threshold",
        "P1_failure": "detects only at the d_safe collision boundary; 0/10 have >=0.5s lead",
        "P2_failure": f"{int(p2['ge_0_5s_count'])}/10 have >=0.5s lead, below 7/10",
        "P3_failure": "typed object association is absent",
        "closed_loop_performance_used_for_selection": False,
        "implementation_authorized": False,
    }
    write_json(OUTPUT / "peer_signal_selection.json", selection)

    risky_rows: list[dict[str, Any]] = []
    for record in records:
        for event in record["events"]:
            if event.get("event") not in {
                "INITIAL_SELECTION",
                "NORMAL_REPROPOSAL",
                "EMERGENCY_REPROPOSAL",
            }:
                continue
            selected = event.get("selected_candidate_id")
            d_min = event.get("selected_minimum_d_min")
            t_min = event.get("selected_minimum_t_min")
            t_risk = event.get("selected_maximum_T_risk")
            risky = bool(
                selected is not None
                and (
                    (d_min is not None and float(d_min) < D_SAFE)
                    or (t_risk is not None and float(t_risk) > 0.0)
                )
            )
            fp_scores = event.get("fp_shep_scores") or []
            fp_rank = ""
            if (
                selected is not None
                and fp_scores
                and int(selected) < len(fp_scores)
                and all(value is not None for value in fp_scores)
            ):
                ordering = sorted(range(len(fp_scores)), key=lambda idx: (-float(fp_scores[idx]), idx))
                fp_rank = ordering.index(int(selected)) + 1
            logits = event.get("class_logits") or []
            selected_logit = ""
            if selected is not None and int(selected) < len(logits):
                selected_logit = float(logits[int(selected)])
            candidate_count = len(event.get("candidate_world_points") or [])
            risky_rows.append(
                {
                    "scenario_id": record["entry"]["scenario_id"],
                    "stage": record["entry"]["stage"],
                    "step": int(event["step"]),
                    "agent_id": int(event["agent_id"]),
                    "event": event["event"],
                    "candidate_count": candidate_count,
                    "selected_candidate_id": selected if selected is not None else "NULL",
                    "selected_d_min_m": d_min if d_min is not None else "",
                    "selected_t_min_s": t_min if t_min is not None else "",
                    "selected_T_risk_s": t_risk if t_risk is not None else "",
                    "selected_risky": risky,
                    "FP_SHEP_rank": fp_rank,
                    "GAT_selected_logit": selected_logit,
                    "available_non_risky_candidate_count": "NOT_AVAILABLE_ALL_CANDIDATE_INTERACTION_DESCRIPTORS_NOT_RETAINED",
                    "null_available": bool(len(logits) == candidate_count + 1),
                    "safe_alternative_available": "NOT_ESTABLISHED",
                }
            )
    non_null = [row for row in risky_rows if row["selected_candidate_id"] != "NULL"]
    risky_count = sum(bool(row["selected_risky"]) for row in non_null)
    if len(risky_rows) != 957 or len(non_null) != 770 or risky_count != 3:
        raise RuntimeError("saved GAT event accounting changed")
    write_csv(OUTPUT / "risky_candidate_diagnostic.csv", risky_rows)

    (OUTPUT / "peer_event_contract.md").write_text(
        "# Peer event contract\n\n"
        "`IMPLEMENTATION_STATUS = NOT_IMPLEMENTED_SIGNAL_SUPPORT_GATE_FAILED`.\n\n"
        "The only information-legal anticipatory candidate is P2, evaluated from the "
        "native normalized/clipped ally observation already generated by every "
        "`MultiAgentDMPEnv.get_observation()` call. It reuses `d_safe=0.6 m` and the "
        "existing H4=0.4 s interaction horizon, and uses no future state. Read-only "
        "diagnostics detected all ten peer collisions, but only 4/10 entries occurred "
        "at least 0.5 s before collision; the preregistered requirement is 7/10.\n\n"
        "Therefore no `E_peer` branch was added. Had the gate passed, the contract would "
        "have been a false-to-true entry event, rearmed only when the same P2 condition "
        "became false, with handoff priority and at most one upper invocation per UAV per "
        "tick. Those semantics remain proposed-only and are not part of the active method.\n\n"
        "Normal and emergency semantics, Proposal, FP-SHEP, GAT, SAC-DMP, all thresholds, "
        "and all checkpoints remain unchanged.\n",
        encoding="utf-8",
    )
    regression = {
        "status": "NOT_RUN_NO_TRIGGER_IMPLEMENTED",
        "base_RERR_regression_reference": "prior 25 tests passed",
        "source_contract_checks": {
            "native_ally_block_computed_each_step": True,
            "exact_ObservableNeighborState_not_used_by_candidate_monitor": True,
            "P1_reuses_d_safe": True,
            "P2_reuses_d_safe_and_H4": True,
            "P3_rejected_without_association": True,
            "no_trigger_code_change": True,
            "no_normal_or_emergency_change": True,
            "no_private_or_future_information": True,
        },
        "dynamic_peer_event_tests_A_to_K": "NOT_RUN_IMPLEMENTATION_GATE_FAILED",
    }
    write_json(OUTPUT / "regression_tests.json", regression)
    write_json(
        OUTPUT / "peer_trigger_freeze.json",
        {
            "status": "NOT_FROZEN_FOR_IMPLEMENTATION",
            "reason": "P2 failed the preregistered >=0.5s lead-rate support gate",
            "performance_seen_before_implementation_freeze": False,
            "unchanged_components": [
                "Proposal",
                "Top-K 10",
                "FP-SHEP H4",
                "GAT-V1 checkpoint",
                "SAC checkpoint",
                "R-ERR normal branch",
                "R-ERR emergency edge/rearm",
                "handoff",
            ],
        },
    )

    stop_reason = "Peer signal support gate failed before implementation; Phase F not executed."
    write_json(
        OUTPUT / "development_scenario_manifest.json",
        {
            "status": "NOT_GENERATED",
            "scenario_count": 0,
            "reason": stop_reason,
            "stage_counts": {"stage_1": 0, "stage_2": 0, "stage_3": 0, "stage_4": 0},
        },
    )
    for name in (
        "development_episode_results.csv",
        "development_agent_results.csv",
        "paired_results.csv",
        "paired_transition_matrix.csv",
        "reproposal_summary.csv",
        "runtime_stage_summary.csv",
    ):
        write_csv(OUTPUT / name, [{"status": "NOT_RUN", "reason": stop_reason}])
    write_csv(
        OUTPUT / "runtime_summary.csv",
        [
            {
                "method": "Historical R-ERR + GAT",
                "scope": "HISTORICAL_REFERENCE_ONLY",
                "episode_count": 80,
                "mean_upper_decisions_per_episode": 9.613,
                "cumulative_upper_planning_ms": 2056.022,
                "execution_actor_ms": 202.246,
                "DMP_ms": 13.323,
                "peer_trigger_monitoring_ms": "N/A",
                "total_online_compute_ms": 2271.5911549999996,
                "status": "NOT_RERUN",
            },
            {
                "method": "R-ERR + Peer-Risk Event + GAT",
                "scope": "NEW_120_PAIRED",
                "episode_count": 0,
                "mean_upper_decisions_per_episode": "NOT_RUN",
                "cumulative_upper_planning_ms": "NOT_RUN",
                "execution_actor_ms": "NOT_RUN",
                "DMP_ms": "NOT_RUN",
                "peer_trigger_monitoring_ms": "NOT_RUN",
                "total_online_compute_ms": "NOT_RUN",
                "status": "NOT_RUN_SIGNAL_GATE_FAILED",
            },
        ],
    )
    write_json(
        OUTPUT / "method_alignment.json",
        {
            "peer_trigger_implemented": False,
            "Proposal_changed": False,
            "FP_SHEP_changed": False,
            "GAT_changed_or_trained": False,
            "SAC_changed_or_trained": False,
            "normal_trigger_changed": False,
            "emergency_trigger_changed": False,
            "new_threshold_added": False,
            "future_information_used": False,
            "candidate_safety_veto_added": False,
            "formal_benchmark_run": False,
            "Methodology_code_alignment": "UNCHANGED_ACTIVE_METHOD; peer event not added",
        },
    )

    conclusion = {
        "CONTINUOUS_PEER_POSITION_LEGAL": "PARTIAL",
        "CONTINUOUS_PEER_VELOCITY_LEGAL": "PARTIAL",
        "PEER_IDENTITY_LEGAL": "NO",
        "LEGAL_PEER_TRIGGER_SIGNAL_AVAILABLE": "YES",
        "SELECTED_PEER_RISK_SIGNAL": "NONE",
        "BEST_LEGAL_CANDIDATE_SIGNAL": "P2",
        "PEER_TRIGGER_USES_NEW_NUMERIC_THRESHOLD": "NO",
        "PEER_TRIGGER_USES_FUTURE_INFORMATION": "NO",
        "PEER_COLLISION_DETECTION_RATE_DIAGNOSTIC": 1.0,
        "PEER_COLLISION_GE_05S_LEAD_RATE": 0.4,
        "SUCCESS_FALSE_WAKEUP_RATE": 4 / 65,
        "PEER_TRIGGER_SIGNAL_SUPPORTED": "NO",
        "PEER_TRIGGER_IMPLEMENTED": "NO",
        "PEER_TRIGGER_CHATTERING": "NO",
        "PEER_TRIGGER_CHATTERING_INTERPRETATION": "NO_TRIGGER_IMPLEMENTED; dynamic assessment NOT_RUN",
        "CURRENT_RERR_SUCCESS": "NOT_RUN",
        "PEER_RERR_SUCCESS": "NOT_RUN",
        "SUCCESS_GAIN_PP": "NOT_RUN",
        "CURRENT_RERR_COLLISION": "NOT_RUN",
        "PEER_RERR_COLLISION": "NOT_RUN",
        "CURRENT_RERR_INTER_AGENT_COLLISION": "NOT_RUN",
        "PEER_RERR_INTER_AGENT_COLLISION": "NOT_RUN",
        "INTER_AGENT_COLLISION_REDUCTION_PP": "NOT_RUN",
        "CURRENT_RERR_STAGE1_SUCCESS": "NOT_RUN",
        "PEER_RERR_STAGE1_SUCCESS": "NOT_RUN",
        "CURRENT_RERR_STAGE2_SUCCESS": "NOT_RUN",
        "PEER_RERR_STAGE2_SUCCESS": "NOT_RUN",
        "CURRENT_RERR_STAGE3_SUCCESS": "NOT_RUN",
        "PEER_RERR_STAGE3_SUCCESS": "NOT_RUN",
        "CURRENT_RERR_STAGE4_SUCCESS": "NOT_RUN",
        "PEER_RERR_STAGE4_SUCCESS": "NOT_RUN",
        "CURRENT_RERR_MEAN_REPROPOSALS": "NOT_RUN",
        "PEER_RERR_MEAN_REPROPOSALS": "NOT_RUN",
        "PEER_EVENT_COUNT": 0,
        "PEER_TRIGGER_MONITORING_RUNTIME_MS": "NOT_RUN",
        "PEER_RERR_TOTAL_COMPUTE_MS": "NOT_RUN",
        "PEER_RERR_TOTAL_COMPUTE_VS_DWA": "NOT_RUN",
        "PEER_RERR_TOTAL_COMPUTE_VS_RVO": "NOT_RUN",
        "RISKY_SELECTION_RATE": risky_count / len(non_null),
        "RISKY_SELECTION_COUNT": risky_count,
        "RISKY_SELECTION_DENOMINATOR_NON_NULL": len(non_null),
        "SAFE_ALTERNATIVE_AVAILABLE_RATE": "NOT_AVAILABLE_ALL_CANDIDATE_INTERACTION_DESCRIPTORS_NOT_RETAINED",
        "REFERENCE_SAFETY_VETO_ADDED": "NO",
        "GAT_CLOSED_LOOP_VALUE": "STRONG",
        "GAT_FINETUNE_JUSTIFIED": "NO",
        "PEER_TRIGGER_REVISION_ACCEPTED": "NOT_RUN",
        "FINAL_METHOD_READY_FOR_NEW_FORMAL": "NO",
        "RECOMMENDED_NEXT_STEP": "PEER_TRIGGER_INFORMATION_INSUFFICIENT",
        "PHASE_F_EXECUTED": "NO",
        "NEW_DEVELOPMENT_SCENARIO_COUNT": 0,
        "NEW_TEAM_EPISODE_COUNT": 0,
        "SIGNAL_GATE_FAILURE": "P2_GE_0_5S_LEAD_RATE_0_40_BELOW_REQUIRED_0_70",
        "historical_reference_only": {
            "RERR_success": 0.8125,
            "RERR_collision": 0.1625,
            "RERR_inter_agent_collision": 0.125,
            "RERR_timeout": 0.025,
            "RERR_stage_success": [0.95, 0.85, 0.70, 0.75],
            "RERR_mean_reproposals": 8.9625,
            "RERR_total_online_compute_ms": 2271.5911549999996,
        },
    }
    write_json(OUTPUT / "conclusion.json", conclusion)

    report = """# Peer-Risk Trigger Closure

## Executive result

The deeper legality audit finds a narrower result than the previous closure. The exact `ObservableNeighborState` used by GAT remains upper-event information and is not legal for continuous polling. However, `MultiAgentDMPEnv.get_observation()` already computes a separate native ally-observation block at every environment step: normalized/clipped relative position, normalized/clipped relative velocity, and normalized distance for the ordered nearest allies. The frozen 122-D SAC actor does not consume this block, but it exists before deciding whether to call GAT. Therefore continuous peer position and velocity are `PARTIAL`, peer identity is `NO`, and `LEGAL_PEER_TRIGGER_SIGNAL_AVAILABLE = YES` for P1/P2 using only this clipped block.

The read-only 80-episode M3 audit nevertheless rejects implementation. P1 detects all ten peer collisions only at the collision boundary. P2 detects all ten, but only 4/10 have at least 0.5 s lead, below the preregistered 7/10 support gate. P3 is unavailable because the two LiDAR frames are untyped and have no reliable object association. Thus `PEER_TRIGGER_SIGNAL_SUPPORTED = NO`, `SELECTED_PEER_RISK_SIGNAL = NONE`, and no peer trigger or 120-scenario development run is authorized.

## 1. Information legality

Three information categories were separated:

- **A — environment private state:** exact `env.dynamics[j]` state. It is forbidden as a trigger input.
- **B — upper-event information:** exact `ObservableNeighborState`/`NeighborGraphState`, identity, and candidate interaction descriptors. They become available when the upper graph is already being built and cannot decide whether to call it.
- **C — pre-GAT online information:** the native flat ally block that `get_observation()` computes every step, plus current/previous untyped 56-ray LiDAR.

The C ally block uses `relative_position / 1.2 m`, `relative_velocity / 4 m/s`, and `distance / 1.2 m`, with clipping to the observation range. It omits stable peer identity. The audit reconstructs exactly this encode/decode path from the frozen diagnostic records; it does not use exact B state as the candidate monitor.

This does not change the frozen 122-D SAC contract: SAC remains conditioned on ego motion, active-reference direction/distance, current/previous LiDAR, and DMP fields. It only establishes that a separate partial ally block is generated by the environment before the next trigger check.

## 2. P1/P2/P3 recoverability

| Signal | Legal | Detection | >=0.5 s lead | Median lead | Success false wake-up | Gate |
|---|---|---:|---:|---:|---:|---|
| P1 current clearance | Yes | 10/10 | 0/10 | 0.00 s | 0/65 | No: collision-boundary only |
| P2 clipped-state CPA, H4 | Yes | 10/10 | 4/10 | 0.35 s | 4/65 (6.15%) | No: lead-rate below 70% |
| P3 LiDAR range-rate | No | — | — | — | — | No reliable typed association |

P2 reuses the current interaction horizon `H=4`, `dt=0.1 s` and existing `d_safe=0.6 m`. It introduces no threshold or future state. Its ten raw leads are 0.3, 0.5, 0.4, 0.2, 0.6, 0.3, 0.2, 5.7, 1.3, and 0.1 s. P25/median/P75/P90 are 0.225/0.350/0.575/1.740 s. Counts with at least 0.3/0.5/1.0/2.0 s lead are 7/4/2/1.

The support rule requires detection of at least 80% of collisions and at least 70% with >=0.5 s lead. P2 passes recall but fails action timing. Signal choice is made before any new closed-loop performance and is not based on development success.

## 3. Event semantics and implementation decision

No `E_peer` branch was implemented. Although a false-to-true P2 entry event with condition-false rearm would prevent level-trigger chattering by construction, the goal permits implementation only after the signal-support gate passes. Dynamic A–K peer-event regression tests, pre-performance implementation freeze, and runtime monitoring are therefore `NOT_RUN`.

Normal triggering, the emergency edge/rearm latch, handoff priority, Proposal, Top-K 10, FP-SHEP H4, GAT-V1, SAC-DMP, thresholds, checkpoints, and collision/success definitions remain unchanged.

## 4. Risky-candidate diagnostic

The 80 retained M3 records contain 957 GAT agent-selection events: 770 non-null candidate selections and 187 null selections. Exactly three non-null selections are explicitly risky under the existing descriptor, giving `RISKY_SELECTION_RATE = 3/770 = 0.3896%`. They are the already identified SR2_014, SR3_019, and SR4_015 events.

All-candidate interaction descriptors were not retained, so the number of non-risk alternatives in each bundle and `SAFE_ALTERNATIVE_AVAILABLE_RATE` remain unavailable. No candidate safety veto was added. The three selected-risk cases do not justify GAT training; `GAT_CLOSED_LOOP_VALUE = STRONG` and `GAT_FINETUNE_JUSTIFIED = NO` remain frozen.

## 5. No paired development or runtime claim

Because the P2 timing gate failed, no 120-scenario manifest was generated and no new team episode was run. All paired performance, transition, event, reproposal, intervention lead-time, and Peer-RERR runtime fields are `NOT_RUN`. The historical R-ERR values (81.25% success, 16.25% collision, 12.5% inter-agent collision, 2271.6 ms/episode) are retained as historical context only and are not presented as a new current arm.

## 6. Final decision

- `LEGAL_PEER_TRIGGER_SIGNAL_AVAILABLE = YES`
- `PEER_TRIGGER_SIGNAL_SUPPORTED = NO`
- `PEER_TRIGGER_IMPLEMENTED = NO`
- `PEER_TRIGGER_REVISION_ACCEPTED = NOT_RUN`
- `FINAL_METHOD_READY_FOR_NEW_FORMAL = NO`
- `RECOMMENDED_NEXT_STEP = PEER_TRIGGER_INFORMATION_INSUFFICIENT`

The current online contract provides a legal but clipped peer-state proxy. With the frozen 0.4 s interaction horizon it is not early enough for the required majority of collisions. The goal forbids extending the horizon, tuning a new threshold, changing sensing, or trying a second trigger, so this branch stops without implementation.
"""
    (OUTPUT / "FINAL_REPORT.md").write_text(report, encoding="utf-8")

    required = [
        "context_recovery_manifest.json",
        "peer_online_information_contract.csv",
        "peer_information_legality.json",
        "peer_risk_signal_audit.csv",
        "peer_collision_lead_time.csv",
        "successful_false_wakeup.csv",
        "peer_signal_selection.json",
        "peer_event_contract.md",
        "risky_candidate_diagnostic.csv",
        "regression_tests.json",
        "peer_trigger_freeze.json",
        "development_scenario_manifest.json",
        "development_episode_results.csv",
        "development_agent_results.csv",
        "paired_results.csv",
        "paired_transition_matrix.csv",
        "peer_event_log.csv",
        "reproposal_summary.csv",
        "runtime_summary.csv",
        "runtime_stage_summary.csv",
        "method_alignment.json",
        "conclusion.json",
        "FINAL_REPORT.md",
    ]
    mandatory = [
        "CONTINUOUS_PEER_POSITION_LEGAL",
        "CONTINUOUS_PEER_VELOCITY_LEGAL",
        "PEER_IDENTITY_LEGAL",
        "LEGAL_PEER_TRIGGER_SIGNAL_AVAILABLE",
        "SELECTED_PEER_RISK_SIGNAL",
        "PEER_TRIGGER_USES_NEW_NUMERIC_THRESHOLD",
        "PEER_TRIGGER_USES_FUTURE_INFORMATION",
        "PEER_COLLISION_DETECTION_RATE_DIAGNOSTIC",
        "PEER_COLLISION_GE_05S_LEAD_RATE",
        "SUCCESS_FALSE_WAKEUP_RATE",
        "PEER_TRIGGER_SIGNAL_SUPPORTED",
        "PEER_TRIGGER_IMPLEMENTED",
        "PEER_TRIGGER_CHATTERING",
        "CURRENT_RERR_SUCCESS",
        "PEER_RERR_SUCCESS",
        "SUCCESS_GAIN_PP",
        "CURRENT_RERR_COLLISION",
        "PEER_RERR_COLLISION",
        "CURRENT_RERR_INTER_AGENT_COLLISION",
        "PEER_RERR_INTER_AGENT_COLLISION",
        "INTER_AGENT_COLLISION_REDUCTION_PP",
        "CURRENT_RERR_STAGE1_SUCCESS",
        "PEER_RERR_STAGE1_SUCCESS",
        "CURRENT_RERR_STAGE2_SUCCESS",
        "PEER_RERR_STAGE2_SUCCESS",
        "CURRENT_RERR_STAGE3_SUCCESS",
        "PEER_RERR_STAGE3_SUCCESS",
        "CURRENT_RERR_STAGE4_SUCCESS",
        "PEER_RERR_STAGE4_SUCCESS",
        "CURRENT_RERR_MEAN_REPROPOSALS",
        "PEER_RERR_MEAN_REPROPOSALS",
        "PEER_EVENT_COUNT",
        "PEER_TRIGGER_MONITORING_RUNTIME_MS",
        "PEER_RERR_TOTAL_COMPUTE_MS",
        "PEER_RERR_TOTAL_COMPUTE_VS_DWA",
        "PEER_RERR_TOTAL_COMPUTE_VS_RVO",
        "RISKY_SELECTION_RATE",
        "SAFE_ALTERNATIVE_AVAILABLE_RATE",
        "REFERENCE_SAFETY_VETO_ADDED",
        "GAT_CLOSED_LOOP_VALUE",
        "GAT_FINETUNE_JUSTIFIED",
        "PEER_TRIGGER_REVISION_ACCEPTED",
        "FINAL_METHOD_READY_FOR_NEW_FORMAL",
        "RECOMMENDED_NEXT_STEP",
    ]
    checks = {
        "required_outputs_present": all((OUTPUT / name).is_file() for name in required),
        "mandatory_conclusion_fields_present": all(name in conclusion for name in mandatory),
        "frozen_record_count_80": len(records) == 80,
        "success_count_65": len(success_analyses) == 65,
        "peer_collision_count_10": len(collision_analyses) == 10,
        "P1_detects_10": signal_summaries[0]["detected_count"] == 10,
        "P1_ge05_is_0": signal_summaries[0]["ge_0_5s_count"] == 0,
        "P2_detects_10": signal_summaries[1]["detected_count"] == 10,
        "P2_ge05_is_4": signal_summaries[1]["ge_0_5s_count"] == 4,
        "P2_false_wakeup_4_of_65": signal_summaries[1]["success_episodes_with_event"] == 4,
        "P3_unavailable": signal_summaries[2]["legal"] == "NO_RELIABLE_ASSOCIATION",
        "signal_gate_failed": conclusion["PEER_TRIGGER_SIGNAL_SUPPORTED"] == "NO",
        "implementation_not_performed": conclusion["PEER_TRIGGER_IMPLEMENTED"] == "NO",
        "phase_f_not_executed": conclusion["PHASE_F_EXECUTED"] == "NO",
        "new_episode_count_zero": conclusion["NEW_TEAM_EPISODE_COUNT"] == 0,
        "risky_event_accounting": len(risky_rows) == 957 and len(non_null) == 770 and risky_count == 3,
        "no_veto": conclusion["REFERENCE_SAFETY_VETO_ADDED"] == "NO",
        "GAT_frozen": conclusion["GAT_FINETUNE_JUSTIFIED"] == "NO",
    }
    reconciliation = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "row_counts": {
            "peer_online_information_contract": len(information_rows),
            "peer_risk_signal_audit": len(signal_summaries),
            "peer_collision_lead_time": len(lead_rows),
            "successful_false_wakeup": len(false_rows),
            "peer_event_log_read_only": len(diagnostic_event_rows),
            "risky_candidate_diagnostic": len(risky_rows),
            "new_development_scenarios": 0,
            "new_team_episodes": 0,
        },
        "artifact_hashes": {name: sha256(OUTPUT / name) for name in required},
    }
    write_json(OUTPUT / "final_reconciliation.json", reconciliation)


if __name__ == "__main__":
    main()
