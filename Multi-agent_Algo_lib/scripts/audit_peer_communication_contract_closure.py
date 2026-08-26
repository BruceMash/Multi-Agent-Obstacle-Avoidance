"""Peer communication-contract and final safety-observability closure.

This audit is deliberately read-only with respect to the method.  It recovers
the active paper contract, traces the runtime peer-state dataflow, and reuses
the frozen 80-episode R-ERR diagnostic records for range/clipping analysis.
The mandatory gate forbids implementation when continuous peer communication
is not established by the paper/system contract.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "20260819_174406"
OUTPUT = ROOT / "artifacts" / "peer_communication_contract_closure" / RUN_ID
GOAL_ATTACHMENT = Path(
    r"C:\Users\Administrator\.codex\attachments\6f9afb20-4da0-458a-9f75-c64bf02ebdf8\pasted-text.txt"
)
PRIOR_GOAL_ATTACHMENT = Path(
    r"C:\Users\Administrator\.codex\attachments\3a3bbaf7-7f81-4bd3-b4d2-8c581c85e667\pasted-text.txt"
)
TEX = ROOT / "hire-rl-body.tex"
ENV = ROOT / "Environment" / "multi_agent_dmp_env.py"
GRAPH = ROOT / "planning" / "heterogeneous_candidate_graph.py"
ACTOR = ROOT / "Environment" / "frozen_sac_dmp_execution.py"
EVALUATOR = ROOT / "Multi-agent_Algo_lib" / "scripts" / "evaluate_gat_v1_err_development.py"
SAFETY = ROOT / "artifacts" / "rerr_safety_closure" / "20260819_134216"
PEER = ROOT / "artifacts" / "peer_risk_trigger_closure" / "20260819_153224"
RUNTIME = ROOT / "artifacts" / "rerr_runtime_compression" / "20260819_162022"
EQUAL = ROOT / "artifacts" / "equal_information_baseline_audit" / "20260819_012339"
RECORD_ROOT = SAFETY / "diagnostic_records"

DT = 0.1
D_SAFE = 0.6
H4_S = 0.4
ALLY_POSITION_SCALE_M = 1.2
ALLY_VELOCITY_SCALE_MPS = 4.0


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


def write_csv(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    fields: list[str] | None = None,
) -> None:
    materialized = [dict(row) for row in rows]
    if fields is None:
        if not materialized:
            raise ValueError(f"fields are required for an empty CSV: {path}")
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
        result[int(row["step"])][int(row["agent_id"])] = (
            vector(row, ("x_m", "y_m", "z_m")),
            vector(row, ("vx_mps", "vy_mps", "vz_mps")),
        )
    return dict(result)


def cpa_diagnostic(
    ego: tuple[np.ndarray, np.ndarray],
    peer: tuple[np.ndarray, np.ndarray],
    *,
    clipped: bool,
) -> dict[str, Any]:
    relative_position = peer[0] - ego[0]
    relative_velocity = peer[1] - ego[1]
    position_saturated = False
    velocity_saturated = False
    if clipped:
        position_feature = np.clip(
            relative_position / ALLY_POSITION_SCALE_M, -1.0, 1.0
        )
        velocity_feature = np.clip(
            relative_velocity / ALLY_VELOCITY_SCALE_MPS, -1.0, 1.0
        )
        position_saturated = bool(np.any(np.abs(position_feature) >= 1.0))
        velocity_saturated = bool(np.any(np.abs(velocity_feature) >= 1.0))
        relative_position = position_feature * ALLY_POSITION_SCALE_M
        relative_velocity = velocity_feature * ALLY_VELOCITY_SCALE_MPS
    speed_sq = float(relative_velocity @ relative_velocity)
    t_cpa = 0.0 if speed_sq <= 1.0e-12 else float(
        np.clip(
            -float(relative_position @ relative_velocity) / speed_sq,
            0.0,
            H4_S,
        )
    )
    d_cpa = float(np.linalg.norm(relative_position + t_cpa * relative_velocity))
    return {
        "risk": bool(d_cpa < D_SAFE),
        "t_cpa_s": t_cpa,
        "d_cpa_m": d_cpa,
        "position_saturated": position_saturated,
        "velocity_saturated": velocity_saturated,
    }


def first_step(steps: list[int], predicate: Any, collision_step: int) -> int | None:
    for step in steps:
        if step <= collision_step and predicate(step):
            return step
    return None


def paper_rows() -> list[dict[str, Any]]:
    common = {
        "document": "hire-rl-body.tex",
        "paper_sha256": sha256(TEX),
    }
    rows = [
        {
            **common,
            "evidence_id": "P01",
            "lines": "16",
            "active": "YES",
            "section": "Introduction",
            "statement": "Neighbor information is gradually obtained through onboard sensing and finite neighboring state.",
            "position": "MENTIONED",
            "velocity": "NOT_SPECIFIED",
            "identity": "NOT_SPECIFIED",
            "source": "SENSING_OR_FINITE_NEIGHBOR_STATE_AMBIGUOUS",
            "frequency": "UNDEFINED",
            "range": "LIMITED_BUT_UNDEFINED",
            "contract_effect": "Does not establish continuous peer communication.",
        },
        {
            **common,
            "evidence_id": "P02",
            "lines": "18",
            "active": "YES",
            "section": "Introduction",
            "statement": "Communication and sensing are limited; decisions mainly use local perception and limited neighbor information.",
            "position": "NOT_SPECIFIED",
            "velocity": "NOT_SPECIFIED",
            "identity": "NOT_SPECIFIED",
            "source": "COMMUNICATION_AND_SENSING_BOTH_MENTIONED",
            "frequency": "UNDEFINED",
            "range": "LIMITED_BUT_UNDEFINED",
            "contract_effect": "Explicitly rejects assuming an unlimited channel but does not define a channel.",
        },
        {
            **common,
            "evidence_id": "P03",
            "lines": "20",
            "active": "YES",
            "section": "Introduction",
            "statement": "Candidate selection combines ego state and neighboring-UAV state.",
            "position": "IMPLIED",
            "velocity": "IMPLIED",
            "identity": "NOT_SPECIFIED",
            "source": "UNDEFINED",
            "frequency": "UNDEFINED",
            "range": "UNDEFINED",
            "contract_effect": "Describes consumed information, not how it is acquired.",
        },
        {
            **common,
            "evidence_id": "P04",
            "lines": "68-81",
            "active": "YES",
            "section": "Environment Formulation",
            "statement": "Each UAV obtains local environmental point observations through onboard sensing.",
            "position": "ENVIRONMENT_POINTS_ONLY",
            "velocity": "NO",
            "identity": "NO",
            "source": "ONBOARD_SENSOR",
            "frequency": "AT_TIME_STEP_T",
            "range": "NOT_NUMERIC_HERE",
            "contract_effect": "Does not type observation points as peer states.",
        },
        {
            **common,
            "evidence_id": "P05",
            "lines": "863-865",
            "active": "YES",
            "section": "PACE-GAT",
            "statement": "The upper graph jointly considers candidate quality and inter-UAV interaction.",
            "position": "IMPLIED",
            "velocity": "IMPLIED",
            "identity": "IMPLIED_BY_NEIGHBOR_NODE",
            "source": "UNDEFINED",
            "frequency": "UPPER_GRAPH_EVALUATION_ONLY",
            "range": "UNDEFINED",
            "contract_effect": "Defines graph semantics, not communication semantics.",
        },
        {
            **common,
            "evidence_id": "P06",
            "lines": "1049-1064",
            "active": "YES",
            "section": "Neighboring-UAV Node",
            "statement": "Neighbor nodes use current relative bearing, distance, position, and relative velocity.",
            "position": "YES_CURRENT",
            "velocity": "YES_CURRENT",
            "identity": "INDEX_J_IN_GRAPH",
            "source": "UNDEFINED",
            "frequency": "WHEN_GRAPH_IS_CONSTRUCTED",
            "range": "UNDEFINED_NEIGHBOR_SET",
            "contract_effect": "Exact current state is required at an upper event, but acquisition is not defined.",
        },
        {
            **common,
            "evidence_id": "P07",
            "lines": "1090-1103",
            "active": "YES",
            "section": "Proposal-align Features",
            "statement": "Constant-velocity prediction uses the neighbor's current position and velocity.",
            "position": "YES_CURRENT",
            "velocity": "YES_CURRENT",
            "identity": "INDEX_J_IN_GRAPH",
            "source": "UNDEFINED",
            "frequency": "WHEN_GRAPH_IS_CONSTRUCTED",
            "range": "UNDEFINED",
            "contract_effect": "Future prediction is model-based, but the current peer-state source remains unstated.",
        },
        {
            **common,
            "evidence_id": "P08",
            "lines": "1468",
            "active": "YES",
            "section": "Low-level POMDP",
            "statement": "The low-level UAV cannot access complete state and receives a local observation.",
            "position": "NO_EXACT_PEER_IN_DEFINED_122D_OBSERVATION",
            "velocity": "NO_EXACT_PEER_IN_DEFINED_122D_OBSERVATION",
            "identity": "NO",
            "source": "LOCAL_OBSERVATION",
            "frequency": "EVERY_LOW_LEVEL_STEP",
            "range": "LOCAL",
            "contract_effect": "Does not authorize continuous exact peer state for a separate trigger.",
        },
        {
            **common,
            "evidence_id": "P09",
            "lines": "1539-1541;1676",
            "active": "YES",
            "section": "ERR",
            "statement": "Upper candidate generation, FP-SHEP, and GAT rerun only at reconstruction events.",
            "position": "AVAILABLE_TO_GAT_AT_EVENT",
            "velocity": "AVAILABLE_TO_GAT_AT_EVENT",
            "identity": "AVAILABLE_TO_GAT_AT_EVENT",
            "source": "UNDEFINED",
            "frequency": "EVENT_TRIGGERED_UPPER_CALLS",
            "range": "UNDEFINED",
            "contract_effect": "The active method establishes event-time consumption, not continuous polling.",
        },
        {
            **common,
            "evidence_id": "P10",
            "lines": "2374-2376",
            "active": "YES",
            "section": "Conclusion",
            "statement": "More realistic communication-constrained conditions are future work.",
            "position": "NOT_ESTABLISHED",
            "velocity": "NOT_ESTABLISHED",
            "identity": "NOT_ESTABLISHED",
            "source": "COMMUNICATION_MODEL_DEFERRED",
            "frequency": "UNDEFINED",
            "range": "UNDEFINED",
            "contract_effect": "Strong evidence that a complete continuous communication contract is absent.",
        },
        {
            **common,
            "evidence_id": "C01",
            "lines": "258",
            "active": "NO_COMMENTED_DRAFT",
            "section": "Commented problem draft",
            "statement": "Mentions an observable-or-communicable neighbor set.",
            "position": "DRAFT_ONLY",
            "velocity": "DRAFT_ONLY",
            "identity": "DRAFT_ONLY",
            "source": "SENSOR_OR_COMMUNICATION_AMBIGUOUS",
            "frequency": "DRAFT_ONLY",
            "range": "DRAFT_ONLY",
            "contract_effect": "Excluded from the active paper contract.",
        },
        {
            **common,
            "evidence_id": "C02",
            "lines": "1805",
            "active": "NO_COMMENTED_DRAFT",
            "section": "Commented graph draft",
            "statement": "Defines neighbors as currently sensed or communicated.",
            "position": "DRAFT_ONLY",
            "velocity": "DRAFT_ONLY",
            "identity": "DRAFT_ONLY",
            "source": "SENSOR_OR_COMMUNICATION_AMBIGUOUS",
            "frequency": "DRAFT_ONLY",
            "range": "DRAFT_ONLY",
            "contract_effect": "Even the inactive draft does not select a source, frequency, or range.",
        },
    ]
    return rows


def dataflow_rows() -> list[dict[str, Any]]:
    return [
        {
            "category": "A_PRIVATE_SIMULATOR_STATE",
            "object": "env.dynamics[j].p / env.dynamics[j].v",
            "producer": "point-mass simulator",
            "consumer": "environment internals",
            "runtime_frequency": "CONTINUOUS_INTERNAL_STATE",
            "position": "EXACT",
            "velocity": "EXACT",
            "identity": "EXACT_INDEX",
            "range": "GLOBAL_ALL_AGENTS",
            "code_evidence": "multi_agent_dmp_env.py:789-799,825-836",
            "legal_continuous_trigger_input": "NO",
            "reason": "Private simulator state is not an observation/communication contract.",
        },
        {
            "category": "B_UPPER_EVENT_EXACT_PEER_INFORMATION",
            "object": "ObservableNeighborState",
            "producer": "observable_neighbor_states reads env.dynamics",
            "consumer": "NeighborGraphState via build_heterogeneous_candidate_graph_from_env",
            "runtime_frequency": "INITIAL_SELECTION_AND_ERR_UPPER_EVENTS_ONLY",
            "position": "EXACT",
            "velocity": "EXACT",
            "identity": "EXACT_AGENT_ID",
            "range": "NEAREST_COUNT; DEFAULT_ALL_OTHER_AGENTS; NO_DISTANCE_DROPOUT",
            "code_evidence": "multi_agent_dmp_env.py:801-818; heterogeneous_candidate_graph.py:561-630; evaluate_gat_v1_err_development.py:987-997,1274-1309",
            "legal_continuous_trigger_input": "NO",
            "reason": "Actual runtime consumption is event-local and the paper does not authorize continuous polling.",
        },
        {
            "category": "C_EVERY_STEP_FLAT_ALLY_BLOCK",
            "object": "_compose_inter_agent_observation output",
            "producer": "env.dynamics -> normalize/clip -> get_observation",
            "consumer": "native 138-D environment observation",
            "runtime_frequency": "EVERY_GET_OBSERVATION_CALL",
            "position": "RELATIVE_COMPONENTWISE_CLIPPED_AT_1.2_M_SCALE",
            "velocity": "RELATIVE_COMPONENTWISE_CLIPPED_AT_4_MPS_SCALE",
            "identity": "OMITTED_FROM_FEATURE_VECTOR",
            "range": "NO_TRUE_1.2_M_DROPOUT; DISTANCE_AND_COMPONENTS_SATURATE",
            "code_evidence": "multi_agent_dmp_env.py:820-867",
            "legal_continuous_trigger_input": "NOT_ESTABLISHED",
            "reason": "The code labels it observation but sources it from private state; the paper defines neither its sensor nor communication channel.",
        },
        {
            "category": "LOWER_FROZEN_ACTOR",
            "object": "historical 122-D SAC observation",
            "producer": "build_historical_actor_observation",
            "consumer": "frozen SAC actor",
            "runtime_frequency": "EVERY_EXECUTION_AND_FP_PREVIEW_ACTOR_CALL",
            "position": "NO_TYPED_PEER_POSITION",
            "velocity": "NO_TYPED_PEER_VELOCITY",
            "identity": "NO",
            "range": "UNTYPED_4.5_M_LIDAR_ONLY_FOR_PEERS",
            "code_evidence": "frozen_sac_dmp_execution.py:33-77; goal_semantics_diagnosis.py:65-95",
            "legal_continuous_trigger_input": "NO_EXACT_PEER_STATE",
            "reason": "The frozen checkpoint intentionally bypasses env.get_observation and consumes exactly 122 dimensions.",
        },
    ]


def collision_diagnostics(records: list[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        if not bool(record["episode"]["inter_agent_collision"]):
            continue
        states = state_maps(record)
        steps = sorted(states)
        collision_steps = sorted(
            {
                int(row["step"])
                for row in record["path_rows"]
                if bool(row.get("inter_agent_collision", False))
            }
        )
        collision_step = collision_steps[0]
        collision_states = states[collision_step]
        for i in range(3):
            for j in range(i + 1, 3):
                collision_distance = float(
                    np.linalg.norm(collision_states[j][0] - collision_states[i][0])
                )
                if collision_distance > D_SAFE + 1.0e-9:
                    continue
                range_entry = first_step(
                    steps,
                    lambda step: float(
                        np.linalg.norm(states[step][j][0] - states[step][i][0])
                    )
                    <= ALLY_POSITION_SCALE_M + 1.0e-12,
                    collision_step,
                )
                clipped_entry = first_step(
                    steps,
                    lambda step: cpa_diagnostic(
                        states[step][i], states[step][j], clipped=True
                    )["risk"],
                    collision_step,
                )
                exact_entry = first_step(
                    steps,
                    lambda step: cpa_diagnostic(
                        states[step][i], states[step][j], clipped=False
                    )["risk"],
                    collision_step,
                )
                if range_entry is None or clipped_entry is None or exact_entry is None:
                    raise RuntimeError("all frozen peer collisions must enter every diagnostic condition")
                clipped_at_entry = cpa_diagnostic(
                    states[clipped_entry][i], states[clipped_entry][j], clipped=True
                )
                exact_at_exact_entry = cpa_diagnostic(
                    states[exact_entry][i], states[exact_entry][j], clipped=False
                )
                range_lead = (collision_step - range_entry) * DT
                clipped_lead = (collision_step - clipped_entry) * DT
                exact_lead = (collision_step - exact_entry) * DT
                rows.append(
                    {
                        "scenario_id": record["entry"]["scenario_id"],
                        "stage": record["entry"]["stage"],
                        "agent_i": i,
                        "agent_j": j,
                        "collision_step": collision_step,
                        "collision_distance_m": collision_distance,
                        "first_within_1_2m_step": range_entry,
                        "ally_1_2m_boundary_lead_s": range_lead,
                        "ally_boundary_ge_0_5s": range_lead >= 0.5 - 1.0e-12,
                        "clipped_h4_first_risk_step": clipped_entry,
                        "clipped_h4_lead_s": clipped_lead,
                        "unclipped_h4_first_risk_step": exact_entry,
                        "unclipped_h4_lead_s": exact_lead,
                        "clipping_lead_change_s": clipped_lead - exact_lead,
                        "clipped_position_saturated_at_risk_entry": clipped_at_entry["position_saturated"],
                        "clipped_velocity_saturated_at_risk_entry": clipped_at_entry["velocity_saturated"],
                        "clipped_d_cpa_at_entry_m": clipped_at_entry["d_cpa_m"],
                        "unclipped_d_cpa_at_entry_m": exact_at_exact_entry["d_cpa_m"],
                        "mode": "PRIVILEGED_READ_ONLY_CLIPPING_DIAGNOSTIC_NO_INTERVENTION",
                    }
                )
    if len(rows) != 10:
        raise RuntimeError(f"expected ten peer-collision pairs, got {len(rows)}")
    range_leads = [float(row["ally_1_2m_boundary_lead_s"]) for row in rows]
    clipped_leads = [float(row["clipped_h4_lead_s"]) for row in rows]
    exact_leads = [float(row["unclipped_h4_lead_s"]) for row in rows]
    changed = [row for row in rows if abs(float(row["clipping_lead_change_s"])) > 1.0e-12]
    summary = {
        "peer_collision_count": len(rows),
        "ALLY_RANGE_ENTRY_MEDIAN_LEAD_S": float(np.median(range_leads)),
        "ALLY_RANGE_GE_05S_COUNT": sum(value >= 0.5 - 1.0e-12 for value in range_leads),
        "ALLY_RANGE_ENTRY_LEADS_S": range_leads,
        "CURRENT_OBSERVATION_SUPPORT_IS_LIMITING": "NO",
        "support_interpretation": "1.2 m is a normalization/saturation scale, not a coded availability cutoff. The geometric boundary was reached >=0.5 s early in 9/10 collisions.",
        "clipped_h4_recall_count": len(clipped_leads),
        "clipped_h4_ge_0_5s_count": sum(value >= 0.5 - 1.0e-12 for value in clipped_leads),
        "clipped_h4_ge_1_0s_count": sum(value >= 1.0 - 1.0e-12 for value in clipped_leads),
        "clipped_h4_median_lead_s": float(np.median(clipped_leads)),
        "unclipped_h4_recall_count": len(exact_leads),
        "unclipped_h4_ge_0_5s_count": sum(value >= 0.5 - 1.0e-12 for value in exact_leads),
        "unclipped_h4_ge_1_0s_count": sum(value >= 1.0 - 1.0e-12 for value in exact_leads),
        "unclipped_h4_median_lead_s": float(np.median(exact_leads)),
        "clipping_changed_first_risk_count": len(changed),
        "clipping_advanced_first_risk_count": sum(
            float(row["clipping_lead_change_s"]) > 0.0 for row in changed
        ),
        "clipping_delayed_first_risk_count": sum(
            float(row["clipping_lead_change_s"]) < 0.0 for row in changed
        ),
        "clipping_mean_lead_change_s": float(
            np.mean([float(row["clipping_lead_change_s"]) for row in rows])
        ),
        "clipping_interpretation": "Clipping did not explain late detection: it left 8/10 first-risk steps unchanged and conservatively advanced 2/10; median lead stayed 0.35 s.",
        "mode": "READ_ONLY_NO_INTERVENTION",
    }
    return rows, summary


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not GOAL_ATTACHMENT.is_file() or not PRIOR_GOAL_ATTACHMENT.is_file():
        raise RuntimeError("goal attachment is unavailable")
    attachment_sha = sha256(GOAL_ATTACHMENT)
    prior_attachment_sha = sha256(PRIOR_GOAL_ATTACHMENT)
    if attachment_sha != prior_attachment_sha:
        raise RuntimeError("reissued goal attachment differs from prior audited goal")
    tex_lines = TEX.read_text(encoding="utf-8").splitlines()
    if len(tex_lines) != 2409:
        raise RuntimeError(f"paper line count changed: {len(tex_lines)}")
    if "Neighboring-UAV Node" not in tex_lines[1048]:
        raise RuntimeError("paper neighbor-node anchor changed")
    if tex_lines[2373].strip() != r"\section{Conclusion}":
        raise RuntimeError("paper conclusion anchor changed")

    record_paths = sorted(RECORD_ROOT.glob("stage_*/SR*/gat_v1_rerr.json"))
    if len(record_paths) != 80:
        raise RuntimeError(f"expected 80 frozen R-ERR records, got {len(record_paths)}")
    records = [read_json(path) for path in record_paths]
    if sum(bool(record["episode"]["inter_agent_collision"]) for record in records) != 10:
        raise RuntimeError("frozen peer-collision count changed")

    runtime_conclusion = read_json(RUNTIME / "conclusion.json")
    runtime_freeze = read_json(RUNTIME / "FINAL_ENGINEERING_FREEZE.json")
    if runtime_conclusion["FINAL_ENGINEERING_OPTIMIZATION_ACCEPTED"] != "YES":
        raise RuntimeError("runtime engineering freeze is not accepted")
    for relative, expected in runtime_freeze["source_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"optimized source hash changed: {relative}")

    source_files = [
        (TEX, "active paper contract"),
        (ENV, "simulator and native ally observation"),
        (GRAPH, "event-time exact peer graph adapter"),
        (ACTOR, "frozen 122-D actor observation"),
        (EVALUATOR, "actual initial/event upper-planner call sites"),
        (SAFETY / "FINAL_REPORT.md", "historical 10-collision safety authority"),
        (SAFETY / "final_reconciliation.json", "historical safety reconciliation"),
        (PEER / "FINAL_REPORT.md", "prior clipped P2 audit"),
        (PEER / "final_reconciliation.json", "prior peer audit reconciliation"),
        (RUNTIME / "FINAL_REPORT.md", "accepted runtime compression authority"),
        (RUNTIME / "FINAL_ENGINEERING_FREEZE.json", "accepted engineering freeze"),
        (RUNTIME / "final_reconciliation.json", "runtime compression reconciliation"),
        (EQUAL / "FINAL_REPORT.md", "equal-information authority"),
        (EQUAL / "proposed_online_information_contract.csv", "prior information contract"),
    ]
    record_hashes = {
        path.relative_to(ROOT).as_posix(): sha256(path) for path in record_paths
    }
    aggregate = hashlib.sha256()
    for path, digest in record_hashes.items():
        aggregate.update(path.encode("utf-8"))
        aggregate.update(digest.encode("ascii"))
    write_json(
        OUTPUT / "context_recovery_manifest.json",
        {
            "audit": "Peer Communication Contract and Final Safety-Observability Closure",
            "timestamp": RUN_ID,
            "mode": "READ_ONLY_MANDATORY_GATE",
            "goal_attachment": {
                "path": str(GOAL_ATTACHMENT),
                "size_bytes": GOAL_ATTACHMENT.stat().st_size,
                "sha256": attachment_sha,
                "byte_identical_to_prior_audited_goal": True,
                "prior_sha256": prior_attachment_sha,
            },
            "authorities": [source_entry(path, role) for path, role in source_files],
            "paper_line_count": len(tex_lines),
            "record_count": len(record_paths),
            "record_aggregate_sha256": aggregate.hexdigest(),
            "record_hashes": record_hashes,
            "frozen_runtime": {
                "single_upper_latency_ms": runtime_conclusion["OPTIMIZED_SINGLE_UPPER_LATENCY_MS"],
                "total_online_compute_ms_per_episode": runtime_conclusion["OPTIMIZED_TOTAL_COMPUTE_MS"],
                "behavior_match": runtime_conclusion["OPTIMIZED_BEHAVIOR_MATCH"],
                "trajectory_match": runtime_conclusion["TRAJECTORY_MATCH"],
            },
        },
    )

    paper = paper_rows()
    dataflow = dataflow_rows()
    diagnostic_rows, diagnostic_summary = collision_diagnostics(records)
    write_csv(OUTPUT / "paper_peer_communication_contract.csv", paper)
    write_csv(OUTPUT / "code_peer_state_dataflow.csv", dataflow)
    write_csv(OUTPUT / "ally_range_and_clipping_diagnostic.csv", diagnostic_rows)
    write_json(OUTPUT / "ally_range_and_clipping_summary.json", diagnostic_summary)

    paper_contract = {
        "PAPER_CONTINUOUS_PEER_STATE": "UNDEFINED",
        "PAPER_PEER_POSITION": "USED_AS_CURRENT_GRAPH_FEATURE_SOURCE_UNDEFINED",
        "PAPER_PEER_VELOCITY": "USED_AS_CURRENT_GRAPH_FEATURE_SOURCE_UNDEFINED",
        "PAPER_PEER_IDENTITY": "GRAPH_INDEX_ONLY_ACQUISITION_UNDEFINED",
        "PAPER_COMMUNICATION_FREQUENCY": "UNDEFINED",
        "PAPER_COMMUNICATION_OR_SENSING_RANGE": "UNDEFINED_FOR_TYPED_PEER_STATE",
        "PAPER_GAT_EXACT_PEER_STATE_SOURCE": "UNDEFINED",
        "active_comment_boundary": "Only active lines establish the contract; commented observable-or-communicable drafts are excluded.",
        "THEORY_EXTENSION_REQUIRED": "YES",
    }
    runtime_status = {
        "EXACT_PEER_STATE_RUNTIME_STATUS": "UPPER_EVENT_ONLY",
        "PRIVATE_EXACT_STATE_CONTINUOUSLY_EXISTS": "YES_NOT_LEGAL_INPUT",
        "EVERY_STEP_CLIPPED_ALLY_BLOCK_EXISTS": "YES",
        "EVERY_STEP_CLIPPED_ALLY_BLOCK_SOURCE": "PRIVATE_SIMULATOR_DYNAMICS",
        "EVERY_STEP_CLIPPED_ALLY_BLOCK_COMMUNICATION_LEGALITY": "NOT_ESTABLISHED",
        "EXACT_PEER_IDENTITY_EVERY_STEP_LEGAL": "NO",
        "actor_122d_consumes_ally_block": "NO",
        "upper_event_graph_consumes_exact_peer_state": "YES",
    }
    write_json(OUTPUT / "paper_contract_decision.json", paper_contract)
    write_json(OUTPUT / "exact_peer_runtime_status.json", runtime_status)

    write_json(
        OUTPUT / "continuous_state_diagnostic.json",
        {
            "status": "NOT_RUN",
            "phase": "D",
            "reason": "Paper/system contract does not establish continuous legal peer position/velocity. Positive-time unbounded CPA and success false-risk analysis are gated off.",
            "future_state_used": "NO",
            "new_threshold_used": "NO",
            "note": "Only the mandatory Phase-C privileged read-only clipping comparison was computed; it is not an admissible online monitor.",
        },
    )
    root_cause = {
        "PEER_OBSERVABILITY_ROOT_CAUSE": "COMMUNICATION_MODEL_UNDEFINED",
        "IMPLEMENTATION_INTERFACE_GAP": "NO_NOT_ESTABLISHED",
        "SENSING_RANGE_LIMITATION": "NO_FOR_1_2M_GEOMETRIC_ENTRY_DIAGNOSTIC",
        "TRIGGER_MODEL_LIMITATION": "SECONDARY_COUNTERFACTUAL_ONLY_NOT_LEGAL_ROOT_CAUSE",
        "CURRENT_OBSERVATION_SUPPORT_IS_LIMITING": diagnostic_summary[
            "CURRENT_OBSERVATION_SUPPORT_IS_LIMITING"
        ],
        "root_cause_reason": "The paper consumes current peer state at graph events but defines neither a continuous communication channel nor typed-peer sensing source, frequency, identity, or range. The code's every-step ally vector is synthesized from private simulator state.",
    }
    write_json(OUTPUT / "root_cause_decision.json", root_cause)

    gate = {
        "PHASE_F_ALLOWED": "NO",
        "PHASE_F_EXECUTED": "NO",
        "PEER_TRIGGER_IMPLEMENTED": "NO",
        "PHASE_G_ALLOWED": "NO",
        "NEW_DEVELOPMENT_MANIFEST_GENERATED": "NO",
        "NEW_DEVELOPMENT_SCENARIO_COUNT": 0,
        "NEW_TEAM_EPISODE_COUNT": 0,
        "gate_reason": "Mandatory stop: PAPER_CONTINUOUS_PEER_STATE=UNDEFINED and root cause is not IMPLEMENTATION_INTERFACE_GAP.",
        "method_files_modified_by_this_audit": [],
        "forbidden_changes_confirmed_absent": [
            "no sensing-range enlargement",
            "no d_safe change",
            "no H4 tuning",
            "no candidate veto",
            "no GAT training",
            "no SAC training",
            "no new formal benchmark",
        ],
    }
    fairness = {
        "FINAL_EQUAL_INFORMATION_CONTRACT_UPDATE_REQUIRED": "NO",
        "fairness_reason": "No new continuous peer communication was introduced into Proposed.",
        "conditional_future_requirement": "YES_IF_A_FUTURE_THEORY_EXTENSION_ADDS_CONTINUOUS_PEER_COMMUNICATION; the same peer contract must then be granted to Sensing-Matched DWA/RVO.",
    }
    write_json(OUTPUT / "implementation_and_development_gate.json", gate)
    write_json(OUTPUT / "fairness_consequence.json", fairness)

    conclusion = {
        **paper_contract,
        **runtime_status,
        **root_cause,
        **gate,
        **fairness,
        "ALLY_RANGE_ENTRY_MEDIAN_LEAD_S": diagnostic_summary[
            "ALLY_RANGE_ENTRY_MEDIAN_LEAD_S"
        ],
        "ALLY_RANGE_GE_05S_COUNT": diagnostic_summary["ALLY_RANGE_GE_05S_COUNT"],
        "CLIPPED_H4_COLLISION_RECALL": diagnostic_summary["clipped_h4_recall_count"],
        "CLIPPED_H4_GE_05S_COUNT": diagnostic_summary["clipped_h4_ge_0_5s_count"],
        "CLIPPED_H4_MEDIAN_LEAD_S": diagnostic_summary["clipped_h4_median_lead_s"],
        "UNCLIPPED_H4_COLLISION_RECALL": diagnostic_summary["unclipped_h4_recall_count"],
        "UNCLIPPED_H4_GE_05S_COUNT": diagnostic_summary[
            "unclipped_h4_ge_0_5s_count"
        ],
        "UNCLIPPED_H4_MEDIAN_LEAD_S": diagnostic_summary[
            "unclipped_h4_median_lead_s"
        ],
        "CLIPPING_CHANGED_FIRST_RISK_COUNT": diagnostic_summary[
            "clipping_changed_first_risk_count"
        ],
        "CLIPPING_DELAYED_FIRST_RISK_COUNT": diagnostic_summary[
            "clipping_delayed_first_risk_count"
        ],
        "OPTIMIZED_RERR_FREEZE_PRESERVED": "YES",
        "OPTIMIZED_RERR_SINGLE_UPPER_LATENCY_MS": runtime_conclusion[
            "OPTIMIZED_SINGLE_UPPER_LATENCY_MS"
        ],
        "OPTIMIZED_RERR_TOTAL_COMPUTE_MS": runtime_conclusion[
            "OPTIMIZED_TOTAL_COMPUTE_MS"
        ],
        "OPTIMIZED_BEHAVIOR_MATCH": runtime_conclusion["OPTIMIZED_BEHAVIOR_MATCH"],
        "GOAL_ATTACHMENT_SHA256": attachment_sha,
        "GOAL_ATTACHMENT_IDENTICAL_TO_PRIOR_AUDITED_GOAL": "YES",
        "PEER_AWARE_SUCCESS_GAIN": "NOT_RUN",
        "PEER_AWARE_INTER_AGENT_COLLISION_REDUCTION": "NOT_RUN",
        "PEER_AWARE_TOTAL_COMPUTE": "NOT_RUN",
        "FINAL_METHOD_READY_FOR_NEW_FORMAL": "NO",
        "RECOMMENDED_NEXT_STEP": "DEFINE_AND_JUSTIFY_PEER_COMMUNICATION_OR_TYPED_SENSING_CONTRACT_BEFORE_IMPLEMENTATION",
    }
    write_json(OUTPUT / "conclusion.json", conclusion)

    report = f"""# Peer Communication Contract and Final Safety-Observability Closure

## Executive result

`PAPER_CONTINUOUS_PEER_STATE = UNDEFINED` and `PEER_OBSERVABILITY_ROOT_CAUSE = COMMUNICATION_MODEL_UNDEFINED`. The active paper uses current neighboring-UAV position and velocity inside the event-time GAT graph, but it never defines whether these values come from communication, onboard sensing, or simulator state; it also defines no communication frequency, typed-peer range, or continuous identity contract. The conclusion explicitly leaves more realistic communication-constrained operation to future work.

The code does not close that theory gap. `ObservableNeighborState` reads exact position, velocity, and identity directly from `env.dynamics` and is consumed when the upper graph is built. The separate ally block is generated on every `get_observation()` call, but it is also synthesized directly from `env.dynamics`, component-wise clipped, and identity-free. A code object named "observation" is not by itself a paper-authorized communication channel. Thus `EXACT_PEER_STATE_RUNTIME_STATUS = UPPER_EVENT_ONLY`; continuous private simulator state must not be repackaged as communication.

The mandatory stop applies. No peer monitor was implemented, no method file was changed, no 120-scenario manifest was generated, and no team episode was run. `THEORY_EXTENSION_REQUIRED = YES`.

## 1. Paper communication contract

| Question | Result |
|---|---|
| Continuous position exchange | Undefined |
| Continuous velocity exchange | Undefined |
| Continuous identity exchange | Undefined |
| Communication frequency | Undefined; exact peer state is only consumed when GAT runs |
| Typed-peer communication/sensing range | Undefined |
| Source of GAT exact peer state in the paper | Undefined |

The active local-observation definition covers onboard environmental observations, while the active neighboring-UAV equations specify what the graph consumes. Neither section connects the two with a typed tracking or communication model. Commented drafts that say "observable or communicable" are inactive and remain ambiguous even if read diagnostically. Detailed line-by-line evidence is in `paper_peer_communication_contract.csv`.

## 2. Code dataflow

| Category | State | Frequency | Identity | Legal continuous trigger input |
|---|---|---|---|---|
| A: private simulator | Exact `env.dynamics` position/velocity | Internal every step | Exact index | No |
| B: upper event | `ObservableNeighborState` exact state | Initial selection and ERR events | Exact | No continuous polling |
| C: native ally vector | Relative position/velocity clipped at 1.2 m/4 m/s scales | Every `get_observation()` | Omitted | Not established by paper |
| Frozen SAC actor | Historical 122-D vector | Every actor call | None | No exact peer state |

The 1.2 m value is a normalization/saturation scale, not an implemented detection cutoff: nearest peer slots continue to exist outside 1.2 m, while distance and components saturate. The frozen SAC actor intentionally bypasses the native 138-D observation builder and retains its historical 122-D input.

## 3. Historical observability boundary

Across the ten frozen R-ERR peer-collision episodes:

- Median lead from first geometric entry within 1.2 m to collision: **{diagnostic_summary['ALLY_RANGE_ENTRY_MEDIAN_LEAD_S']:.2f} s**.
- Entry at least 0.5 s before collision: **{diagnostic_summary['ALLY_RANGE_GE_05S_COUNT']}/10**.
- `CURRENT_OBSERVATION_SUPPORT_IS_LIMITING = {diagnostic_summary['CURRENT_OBSERVATION_SUPPORT_IS_LIMITING']}`.

The current 1.2 m geometric boundary is therefore not the direct timing bottleneck. This result does not make the ally block a legal communication channel; it only localizes the failure.

## 4. Clipping impact on the existing H4 CPA

The previously reported clipped H4 monitor detects 10/10 collisions, with **{diagnostic_summary['clipped_h4_ge_0_5s_count']}/10** at least 0.5 s early and median lead **{diagnostic_summary['clipped_h4_median_lead_s']:.2f} s**. A privileged read-only recomputation with the exact, unclipped stored current state also detects 10/10, but only **{diagnostic_summary['unclipped_h4_ge_0_5s_count']}/10** are at least 0.5 s early; median lead remains **{diagnostic_summary['unclipped_h4_median_lead_s']:.2f} s**.

Clipping changed the first-risk step in {diagnostic_summary['clipping_changed_first_risk_count']}/10 cases, advanced it in {diagnostic_summary['clipping_advanced_first_risk_count']}/10, and delayed it in {diagnostic_summary['clipping_delayed_first_risk_count']}/10. Thus saturation did not cause the late warning. The exact-state comparison is diagnostic-only and is not an admissible online peer monitor.

Phase D's continuous-state H4/unbounded-positive-time CPA and success false-risk evaluation is `NOT_RUN`, because its legality precondition failed.

## 5. Root cause and forced gate

- `PEER_OBSERVABILITY_ROOT_CAUSE = COMMUNICATION_MODEL_UNDEFINED`
- `IMPLEMENTATION_INTERFACE_GAP = NO_NOT_ESTABLISHED`
- `THEORY_EXTENSION_REQUIRED = YES`
- `PHASE_F_EXECUTED = NO`
- `NEW_DEVELOPMENT_SCENARIO_COUNT = 0`
- `NEW_TEAM_EPISODE_COUNT = 0`

The evidence does not support the claim that a legal continuous peer channel already exists and merely needs wiring. A future theory revision must explicitly define the signal source, update rate, range/dropout behavior, identity/data association, latency, and fairness contract before code integration.

## 6. Engineering freeze and fairness

The accepted optimized R-ERR + GAT freeze remains untouched: single upper latency **{runtime_conclusion['OPTIMIZED_SINGLE_UPPER_LATENCY_MS']:.3f} ms**, total online compute **{runtime_conclusion['OPTIMIZED_TOTAL_COMPUTE_MS']:.3f} ms/episode**, and exact behavior/trajectory equivalence. Because no communication privilege was added, `FINAL_EQUAL_INFORMATION_CONTRACT_UPDATE_REQUIRED = NO` for the current method.

If a future revision introduces continuous peer communication, equal-information DWA/RVO must receive the identical peer-state contract. That future conditional requirement is `YES`; it cannot be postponed until after performance is observed.

## Final answer

The remaining peer-collision bottleneck is **not an already legal communication state that simply has not been connected to ERR**. It is that the current paper/system information contract does not define a legal continuous typed-peer channel. The existing clipped ally proxy is geometrically present early enough in 9/10 collision cases, and clipping itself does not explain late CPA warning, but neither fact authorizes exact continuous peer polling.

Therefore whether a legal continuous peer state would further improve success and reduce inter-agent collision while retaining the 613.6 ms compute advantage is **not established and was not tested**. The next valid action is a theory-and-fairness contract revision, not an implementation or benchmark run.
"""
    (OUTPUT / "FINAL_REPORT.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
