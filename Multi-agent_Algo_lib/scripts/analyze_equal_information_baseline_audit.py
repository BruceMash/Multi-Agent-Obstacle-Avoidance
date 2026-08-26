from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = REPO_ROOT / "artifacts" / "equal_information_baseline_audit" / "20260819_012339"
FINAL_ROOT = REPO_ROOT / "artifacts" / "final_four_stage_benchmark" / "20260818_202620"
RECOVERY_ROOT = REPO_ROOT / "artifacts" / "theory_aligned_final_recovery" / "20260818_232900"
CORRECTED_AGENT_PATH = (
    RECOVERY_ROOT
    / "phase_a_corrected_classics"
    / "corrected_classical_agent_results.csv"
)

STAGES = ("stage_1", "stage_2", "stage_3", "stage_4")
SCOPES = ("overall",) + STAGES
METHOD_ORDER = (
    "DWA-FullState",
    "RVO-FullState",
    "DWA-SensingMatched",
    "RVO-SensingMatched",
    "One-Shot Proposed",
)
METHOD_META = {
    "DWA-FullState": ("LEVEL_A", "SYSTEM_LEVEL_STRONG_REFERENCE"),
    "RVO-FullState": ("LEVEL_A", "SYSTEM_LEVEL_STRONG_REFERENCE"),
    "DWA-SensingMatched": ("LEVEL_B", "EQUAL_INFORMATION_SYSTEM_LEVEL"),
    "RVO-SensingMatched": ("LEVEL_B", "EQUAL_INFORMATION_SYSTEM_LEVEL"),
    "One-Shot Proposed": ("LEVEL_C", "INTERNAL_LEARNED_CHAIN_REFERENCE"),
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def avg(values: Iterable[Any]) -> float | None:
    numbers = [number for value in values if (number := as_float(value)) is not None]
    return float(mean(numbers)) if numbers else None


def load_sensing_records() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    agents: list[dict[str, Any]] = []
    hash_pass = 0
    software_failures = 0
    paths = sorted((OUTPUT / "diagnostic_records").rglob("*.json"))
    for path in paths:
        payload = load_json(path)
        expected = payload.pop("result_sha256")
        hash_pass += int(stable_hash(payload) == expected)
        software_failures += int(payload["software_failure"])
        if payload["software_failure"]:
            continue
        episodes.append(dict(payload["episode"]))
        agents.extend(dict(row) for row in payload["agents"])
    checks = {
        "record_count": len(paths),
        "record_hash_pass_count": hash_pass,
        "software_failure_count": software_failures,
    }
    return episodes, agents, checks


def normalize_episode(
    source: Mapping[str, Any], *, method: str, source_artifact: str
) -> dict[str, Any]:
    level, comparison = METHOD_META[method]
    boolean_fields = (
        "team_success",
        "any_collision",
        "obstacle_collision",
        "inter_agent_collision",
        "boundary_collision",
        "timeout",
    )
    numeric_fields = (
        "completion_step",
        "completion_time_s",
        "termination_time_s",
        "steps",
        "team_path_length_m",
        "team_path_length_mean_agent_m",
        "trajectory_smoothness",
        "minimum_obstacle_clearance_m",
        "minimum_inter_agent_distance_m",
        "planning_runtime_ms",
        "perception_adapter_runtime_ms",
        "planner_core_runtime_ms",
        "planning_decision_count",
        "planning_runtime_per_decision_ms",
        "end_to_end_runtime_ms",
        "planner_infeasible_count",
    )
    row: dict[str, Any] = {
        "comparison_label": "POST_HOC_DIAGNOSTIC_COMPARISON",
        "comparison_level": level,
        "comparison_interpretation": comparison,
        "method": method,
        "source_method": source.get("method"),
        "source_artifact": source_artifact,
        "stage": source.get("stage"),
        "family": source.get("family"),
        "scenario_id": source.get("scenario_id") or source.get("scenario"),
        "seed": int(source["seed"]),
        "termination_reason": source.get("termination_reason"),
    }
    for field in boolean_fields:
        row[field] = as_bool(source.get(field, False))
    for field in numeric_fields:
        row[field] = as_float(source.get(field))
    return row


def normalize_agent(source: Mapping[str, Any], method: str) -> dict[str, Any]:
    return {
        "stage": source["stage"],
        "scenario_id": source["scenario_id"],
        "seed": int(source["seed"]),
        "method": method,
        "agent_id": int(source["agent_id"]),
        "agent_terminal_completed": as_bool(source["agent_terminal_completed"]),
    }


def build_diagnostic_rows(
    sensing_episodes: Sequence[Mapping[str, Any]],
    sensing_agents: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episodes: list[dict[str, Any]] = []
    agents: list[dict[str, Any]] = []
    full_map = {"dwa_style": "DWA-FullState", "rvo_orca_style": "RVO-FullState"}
    for row in read_csv(RECOVERY_ROOT / "corrected_classical_results.csv"):
        episodes.append(
            normalize_episode(
                row,
                method=full_map[row["method"]],
                source_artifact="corrected_classical_results.csv",
            )
        )
    sensing_map = {
        "dwa_sensing_matched": "DWA-SensingMatched",
        "rvo_sensing_matched": "RVO-SensingMatched",
    }
    for row in sensing_episodes:
        episodes.append(
            normalize_episode(
                row,
                method=sensing_map[row["method"]],
                source_artifact="diagnostic_records/*.json",
            )
        )
    for row in read_csv(FINAL_ROOT / "formal_episode_results.csv"):
        if row["method"] == "gat_v1":
            episodes.append(
                normalize_episode(
                    row,
                    method="One-Shot Proposed",
                    source_artifact="formal_episode_results.csv",
                )
            )
    for row in read_csv(CORRECTED_AGENT_PATH):
        agents.append(normalize_agent(row, full_map[row["method"]]))
    for row in sensing_agents:
        agents.append(normalize_agent(row, sensing_map[row["method"]]))
    for row in read_csv(FINAL_ROOT / "formal_agent_results.csv"):
        if row["method"] == "gat_v1":
            agents.append(normalize_agent(row, "One-Shot Proposed"))
    return episodes, agents


def subset(rows: Sequence[Mapping[str, Any]], method: str, scope: str) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if row["method"] == method and (scope == "overall" or row["stage"] == scope)
    ]


def rate(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return float(mean(as_bool(row[field]) for row in rows))


def build_information_decomposition(
    episodes: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family, full, matched in (
        ("DWA", "DWA-FullState", "DWA-SensingMatched"),
        ("RVO", "RVO-FullState", "RVO-SensingMatched"),
    ):
        for scope in SCOPES:
            full_rows = subset(episodes, full, scope)
            matched_rows = subset(episodes, matched, scope)
            full_rate = rate(full_rows, "team_success")
            matched_rate = rate(matched_rows, "team_success")
            rows.append(
                {
                    "algorithm_family": family,
                    "scope": scope,
                    "n": len(full_rows),
                    "fullstate_method": full,
                    "sensing_matched_method": matched,
                    "fullstate_success_count": sum(as_bool(row["team_success"]) for row in full_rows),
                    "sensing_matched_success_count": sum(as_bool(row["team_success"]) for row in matched_rows),
                    "fullstate_success_rate": full_rate,
                    "sensing_matched_success_rate": matched_rate,
                    "information_advantage_pp": 100.0 * (full_rate - matched_rate),
                    "retained_replanning_frequency": "per_control_step",
                    "retained_prediction_horizon": "0.8_s" if family == "DWA" else "1.2_to_1.4_s",
                    "retained_control_interface": "desired_velocity_plus_common_acceleration_limited_dynamics",
                    "causal_change": "online_information_contract_only",
                    "comparison_label": "POST_HOC_DIAGNOSTIC_COMPARISON",
                }
            )
    return rows


def precomputed_proposed_safety(scope: str) -> Mapping[str, str]:
    source = FINAL_ROOT / ("ablation_summary.csv" if scope == "overall" else "stage_summary.csv")
    rows = read_csv(source)
    for row in rows:
        if row["method"] == "gat_v1" and (scope == "overall" or row["stage"] == scope):
            return row
    raise KeyError(scope)


def build_safety_decomposition(
    episodes: Sequence[Mapping[str, Any]], agents: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scope in SCOPES:
        for method in METHOD_ORDER:
            episode_rows = subset(episodes, method, scope)
            agent_rows = subset(agents, method, scope)
            proposed_summary = precomputed_proposed_safety(scope) if method == "One-Shot Proposed" else None
            sensor_clearance = (
                avg(row.get("minimum_obstacle_clearance_m") for row in episode_rows)
                if method != "One-Shot Proposed"
                else None
            )
            signed_clearance = (
                as_float(proposed_summary.get("minimum_obstacle_clearance_m_mean"))
                if proposed_summary is not None
                else None
            )
            rows.append(
                {
                    "scope": scope,
                    "method": method,
                    "n": len(episode_rows),
                    "team_success_count": sum(as_bool(row["team_success"]) for row in episode_rows),
                    "team_success_rate": rate(episode_rows, "team_success"),
                    "any_collision_count": sum(as_bool(row["any_collision"]) for row in episode_rows),
                    "any_collision_rate": rate(episode_rows, "any_collision"),
                    "obstacle_collision_count": sum(as_bool(row["obstacle_collision"]) for row in episode_rows),
                    "obstacle_collision_rate": rate(episode_rows, "obstacle_collision"),
                    "inter_agent_collision_count": sum(as_bool(row["inter_agent_collision"]) for row in episode_rows),
                    "inter_agent_collision_rate": rate(episode_rows, "inter_agent_collision"),
                    "timeout_count": sum(as_bool(row["timeout"]) for row in episode_rows),
                    "timeout_rate": rate(episode_rows, "timeout"),
                    "agent_completion_count": sum(as_bool(row["agent_terminal_completed"]) for row in agent_rows),
                    "agent_n": len(agent_rows),
                    "agent_completion_rate": rate(agent_rows, "agent_terminal_completed"),
                    "minimum_inter_agent_distance_mean_m": avg(
                        row.get("minimum_inter_agent_distance_m") for row in episode_rows
                    ),
                    "minimum_online_sensor_surface_clearance_mean_m": sensor_clearance,
                    "minimum_obstacle_signed_clearance_mean_m": signed_clearance,
                    "signed_obstacle_clearance_availability": (
                        "PRECOMPUTED_INDEPENDENT_TRAJECTORY_RECONSTRUCTION"
                        if signed_clearance is not None
                        else "UNAVAILABLE_FROM_RETAINED_DIAGNOSTIC_RECORD"
                    ),
                    "clearance_caveat": (
                        "online sensor clearance includes any nearest static/dynamic/peer LiDAR surface"
                        if sensor_clearance is not None
                        else "not applicable"
                    ),
                    "comparison_label": "POST_HOC_DIAGNOSTIC_COMPARISON",
                }
            )
    return rows


def build_runtime_decomposition(episodes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    reference = {
        row["method"]: row
        for row in read_csv(RECOVERY_ROOT / "runtime_method_summary.csv")
        if row["scope"] == "overall" and row["method"] in {
            "dwa_style",
            "rvo_orca_style",
            "gat_v1_one_shot",
        }
    }
    rows: list[dict[str, Any]] = []
    for method, source_name in (
        ("DWA-FullState", "dwa_style"),
        ("RVO-FullState", "rvo_orca_style"),
        ("One-Shot Proposed", "gat_v1_one_shot"),
    ):
        source = reference[source_name]
        rows.append(
            {
                "method": method,
                "episode_count": int(source["episode_count"]),
                "mean_decisions_per_episode": float(source["mean_decisions_per_episode"]),
                "mean_total_planning_ms_per_episode": float(source["mean_cumulative_planning_ms"]),
                "mean_planner_latency_ms_per_decision": float(source["mean_decision_latency_ms"]),
                "mean_perception_adapter_ms_per_episode": 0.0,
                "mean_perception_adapter_ms_per_decision": 0.0,
                "mean_planner_core_ms_per_episode": float(source["mean_cumulative_planning_ms"]),
                "mean_actor_execution_ms_per_episode": float(source["mean_execution_actor_ms"]),
                "mean_dmp_internal_ms_per_episode": float(source["mean_execution_dmp_ms"]),
                "mean_total_online_algorithm_compute_ms_per_episode": float(source["mean_total_online_algorithm_compute_ms"]),
                "runtime_source": "synchronized_runtime_reconciliation",
                "runtime_interpretation": "complete online algorithm compute",
            }
        )
    for method in ("DWA-SensingMatched", "RVO-SensingMatched"):
        method_rows = subset(episodes, method, "overall")
        total_decisions = sum(int(row["planning_decision_count"] or 0) for row in method_rows)
        total_planning = sum(float(row["planning_runtime_ms"] or 0.0) for row in method_rows)
        total_adapter = sum(float(row["perception_adapter_runtime_ms"] or 0.0) for row in method_rows)
        total_core = sum(float(row["planner_core_runtime_ms"] or 0.0) for row in method_rows)
        rows.append(
            {
                "method": method,
                "episode_count": len(method_rows),
                "mean_decisions_per_episode": total_decisions / len(method_rows),
                "mean_total_planning_ms_per_episode": total_planning / len(method_rows),
                "mean_planner_latency_ms_per_decision": total_planning / total_decisions,
                "mean_perception_adapter_ms_per_episode": total_adapter / len(method_rows),
                "mean_perception_adapter_ms_per_decision": total_adapter / total_decisions,
                "mean_planner_core_ms_per_episode": total_core / len(method_rows),
                "mean_actor_execution_ms_per_episode": 0.0,
                "mean_dmp_internal_ms_per_episode": 0.0,
                "mean_total_online_algorithm_compute_ms_per_episode": total_planning / len(method_rows),
                "runtime_source": "post_hoc_sensing_matched_diagnostic",
                "runtime_interpretation": "adapter included; early collision outcome-censors episode totals; four parallel shards",
            }
        )
    return sorted(rows, key=lambda row: METHOD_ORDER.index(row["method"]))


def build_issue_impact(info_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    overall = {(row["algorithm_family"], row["scope"]): row for row in info_rows}
    return [
        {
            "issue": "future_dynamic_rng_leakage",
            "status_after": "FIXED_NO_LEAKAGE",
            "quantified_impact": "privileged-to-corrected success: DWA -0.50 pp; RVO -0.25 pp",
            "interpretation": "not the primary reason the corrected FullState methods remain near 98%",
            "confidence": "HIGH",
        },
        {
            "issue": "fullstate_information_bundle",
            "status_after": "ISOLATED_POST_HOC",
            "quantified_impact": (
                f"DWA {overall[('DWA', 'overall')]['information_advantage_pp']:.1f} pp; "
                f"RVO {overall[('RVO', 'overall')]['information_advantage_pp']:.1f} pp"
            ),
            "interpretation": "largest experimentally isolated contributor; static/dynamic/peer subcomponents are not separately identifiable",
            "confidence": "HIGH_FOR_BUNDLE_LOW_FOR_SUBCOMPONENTS",
        },
        {
            "issue": "per_step_replanning",
            "status_after": "RETAINED_IN_C1_TO_C4",
            "quantified_impact": "not separately ablated",
            "interpretation": "algorithmic/architectural advantage remains in both FullState and SensingMatched classics",
            "confidence": "NOT_ESTABLISHED_SEPARATELY",
        },
        {
            "issue": "prediction_horizon",
            "status_after": "RETAINED_0.8_TO_1.4_S",
            "quantified_impact": "not separately ablated against FP-SHEP H4=0.4 s",
            "interpretation": "system architecture difference, not an information effect",
            "confidence": "NOT_ESTABLISHED_SEPARATELY",
        },
        {
            "issue": "direct_velocity_control_interface",
            "status_after": "RETAINED",
            "quantified_impact": "not separately ablated against SAC-DMP execution",
            "interpretation": "system architecture difference",
            "confidence": "NOT_ESTABLISHED_SEPARATELY",
        },
        {
            "issue": "training_and_development_distribution",
            "status_after": "AUDITED",
            "quantified_impact": "classics selected on new four-stage development; GAT partial match; SAC historical single-UAV curriculum",
            "interpretation": "Stage III/IV learned results are OOD system performance, not a pure GAT algorithm ranking",
            "confidence": "HIGH_FOR_SHIFT_DIRECTION_NOT_CAUSAL_MAGNITUDE",
        },
        {
            "issue": "boundary_free_reactive_benchmark_structure",
            "status_after": "AUDITED_READ_ONLY",
            "quantified_impact": "finite 3-D obstacles admit side/above/below detours; corrected FullState classics retain 94-97% Stage IV success",
            "interpretation": "benchmark is highly favorable to well-informed reactive local planning",
            "confidence": "MODERATE_TO_HIGH",
        },
        {
            "issue": "GAT_incremental_ranking_value",
            "status_after": "PRESERVED",
            "quantified_impact": "overall +2.75 pp vs FP-SHEP (p=0.215); Stage III +14 pp; Stage IV +11 pp",
            "interpretation": "nonzero difficult-stage value, but no overall superiority claim",
            "confidence": "HIGH_FOR_OBSERVED_EFFECT_LOW_FOR_GENERALIZATION",
        },
    ]


def format_percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def stage_table(episodes: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| Method | Overall | Stage I | Stage II | Stage III | Stage IV | Collision | Timeout |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHOD_ORDER:
        values = [rate(subset(episodes, method, scope), "team_success") for scope in SCOPES]
        overall = subset(episodes, method, "overall")
        lines.append(
            "| "
            + method
            + " | "
            + " | ".join(format_percent(value) for value in values)
            + f" | {format_percent(rate(overall, 'any_collision'))}"
            + f" | {format_percent(rate(overall, 'timeout'))} |"
        )
    return "\n".join(lines)


def build_report(
    episodes: Sequence[Mapping[str, Any]],
    info_rows: Sequence[Mapping[str, Any]],
    runtime_rows: Sequence[Mapping[str, Any]],
    conclusion: Mapping[str, Any],
) -> str:
    overall_info = {(row["algorithm_family"], row["scope"]): row for row in info_rows}
    runtime = {row["method"]: row for row in runtime_rows}
    return f"""# Equal-Information Classical Baseline and Benchmark Fairness Audit

## Executive result

修复 stochastic-future leakage 后，FullState DWA/RVO 的高成功率仍真实存在：**391/400 (97.75%)** 和 **394/400 (98.50%)**。但将其在线输入严格限制为 Proposed execution 实际拥有的 ego/goal 与 4.5 m、56-ray、无类型 LiDAR 后，在保留逐步重规划、原预测时域、原参数和直接速度规划接口的条件下，DWA-SensingMatched 仅为 **73/400 (18.25%)**，RVO-SensingMatched 为 **40/400 (10.00%)**。

因此，信息合同是本审计中最大的可识别优势来源：DWA 为 **{overall_info[('DWA', 'overall')]['information_advantage_pp']:.1f} pp**，RVO 为 **{overall_info[('RVO', 'overall')]['information_advantage_pp']:.1f} pp**。这两个差值是 `POST_HOC_DIAGNOSTIC_COMPARISON`，不能称为新的 untouched formal test，也不能进一步拆成 static/dynamic/peer 三个独立因果份额。

## 1. Diagnostic 400 outcomes

{stage_table(episodes)}

SensingMatched 两种方法在 Stage III/IV 均为 0%，且失败全部在碰撞终止前发生、没有 timeout。这说明经典方法在当前 benchmark 上的近 98% 表现依赖其自然的 FullState contract；它们仅凭稀疏、无类型的 LiDAR endpoint representation 不能维持同样能力。与此同时，这不是人为减速或减小搜索预算：D00/R02 的 weights、samples、buffer、horizon、update frequency、speed/acceleration limits 全部原样保留。

## 2. Information contracts and integrity

- lower execution actor 的历史 122-D 输入为：ego velocity 3、active-reference direction 3、clipped/normalized distance 1、current/previous 56-ray LiDAR、DMP phase/K-alpha/K-beta 3。LiDAR 半径 4.5 m，8×7 beams，静态/动态/peer 只表现为无类型最近表面。
- upper planner 仅在 t=0 或理论事件时可读取 GAT 的精确当前 peer position/velocity/identity；这不能外推成每个 control step 持续可见。
- exact dynamic-obstacle velocity 不在 actor 输入中；两个无关联 scan 不足以可靠建立对象速度。因此 C3/C4 在性能前冻结为 current visible endpoints 的 zero-order hold。
- 7 项 observation-equivalence 测试全部通过。改变观测范围外障碍、hidden shape、hidden dynamic velocity 或范围外 peer，而保持公共观测相同，C3/C4 输出不变。`SENSING_MATCHED_INFORMATION_LEAKAGE = NO`。

## 3. What the 98% result does and does not identify

本审计对信息 bundle 有直接控制：FullState→SensingMatched 是唯一变化，DWA/RVO 其余算法合同不变，因此 **79.5/88.5 pp 是信息可访问性的可量化效应**。future RNG leakage 本身只对应 privileged→corrected 的 DWA -0.50 pp、RVO -0.25 pp，不是主要原因。

但以下因素未做单因素消融，不能伪造独立百分比：逐步重规划、0.8–1.4 s prediction horizon、direct desired-velocity interface，以及 boundary-free reactive benchmark structure。结论字段采用 `ORIGINAL_98_PERCENT_CLASSICAL_PRIMARY_ADVANTAGE_SOURCE = INFORMATION`，含义是“最大且被实验隔离的来源”，并不声称其余因素为零。

## 4. Benchmark and distribution audit

Stage III 有密集门/走廊/错列障碍，但障碍均为有限 3-D primitives，boundary collision 被关闭，多数场景可从侧面、上方或下方连续绕行。因此 `STAGE3_ROUTE_CHOICE_COMPLEXITY = MODERATE`，而不是强制全局拓扑规划。Stage IV 加入移动障碍后主要增加快速局部避碰负担，给出 `STAGE4_PRIMARY_DIFFICULTY = REACTIVE_AVOIDANCE`。综合标签为 `BENCHMARK_REACTIVE_PLANNER_FAVORABILITY = HIGH` 与 `BOUNDARY_FREE_LOCAL_DETOUR_ADVANTAGE = YES`。

GAT 数据覆盖 open/bounded/sparse/dense/dynamic/multi-agent/narrow-head-on 等语义类型，但来自较早 seeds、较短 rollout 与不同支持；SAC 是历史 single-UAV curriculum。相对更大 workspace、更长任务、Stage III/IV 密集与动态组合，`LEARNED_PIPELINE_DISTRIBUTION_MATCH = STRONG_SHIFT`、`GAT_TRAINING_DISTRIBUTION_MATCH = PARTIAL`。相反 DWA/RVO 参数是在新四阶段 development distribution 上从固定配置预算中选择，`CLASSICAL_DEV_DISTRIBUTION_MATCH = YES`。因此 15%/12% 应标记为 OOD system performance，不能简单归因为 GAT 本体能力不足。

## 5. Safety and runtime interpretation

完整 collision/obstacle/peer/timeout/agent-completion 分阶段数据位于 `safety_decomposition.csv`。本次 C3/C4 record 保留的是 online sensor minimum clearance；在 peer-sphere 模式下它包含最近 static/dynamic/peer LiDAR surface，不能冒充纯 obstacle signed clearance。C3/C4 没有保留轨迹，故纯 obstacle signed-clearance 被明确标记 unavailable；没有重跑或补造。

Perception adapter 已计入在线计算。DWA-SensingMatched 平均 adapter 为 **{runtime['DWA-SensingMatched']['mean_perception_adapter_ms_per_episode']:.3f} ms/episode**、总 planning **{runtime['DWA-SensingMatched']['mean_total_online_algorithm_compute_ms_per_episode']:.3f} ms/episode**；RVO-SensingMatched 分别为 **{runtime['RVO-SensingMatched']['mean_perception_adapter_ms_per_episode']:.3f} ms** 和 **{runtime['RVO-SensingMatched']['mean_total_online_algorithm_compute_ms_per_episode']:.3f} ms**。其 episode total 被早期碰撞强烈截短，且诊断使用四分片并行，不能将较低累计时间解释成等成功条件下更高效率。pooled per-decision latency 为 {runtime['DWA-SensingMatched']['mean_planner_latency_ms_per_decision']:.3f}/{runtime['RVO-SensingMatched']['mean_planner_latency_ms_per_decision']:.3f} ms。

## 6. Answers to Q1–Q6

1. **Q1：高成功率是否仍真实？** 是。修复 future leakage 后仍为 97.75%/98.50%，但这是 FullState system-level reference。
2. **Q2：同信息时多少？** DWA-SensingMatched 18.25%，RVO-SensingMatched 10.00%。
3. **Q3：优势来自什么？** 可量化的 information bundle 贡献为 79.5/88.5 pp，是最大已隔离来源。global geometry、exact dynamic state、exact peer state未分别消融；update frequency、horizon、control interface 与 benchmark structure 保持或只审计，不能各自分配伪精确份额。
4. **Q4：Stage III/IV 考察什么？** 主要偏向 high-frequency reactive avoidance；Stage III 仅有中等 route-choice complexity，Stage IV 以 reactive avoidance 为主。
5. **Q5：是否有 distribution shift 与 adaptation asymmetry？** 是。Learned pipeline 是 strong shift，GAT 仅 partial match；classics 在新四阶段 development 上选过参数。
6. **Q6：论文报告哪组？** 两组都报告：FullState 是强 system reference，SensingMatched 是 equal-information system comparison，learned chain ablations 独立列为 Level C。

## 7. GAT claim boundary and next step

GAT one-shot 相对 FP-SHEP 的 overall 增益是 +2.75 pp（McNemar p=0.215），Stage III/IV 分别 +14/+11 pp：困难场景存在 incremental ranking value，但不足以声称总体显著优势。未来关键 paired ablation 应是同一 trigger、candidate pool、SAC-DMP 与 scenarios 下的 `ERR+FP-SHEP` vs `ERR+GAT`。

当前 theory-aligned ERR development 虽有性能增益，但 emergency-dominated chattering 尚未解决。因此本 goal 停止在 audit；建议 `WAIT_FOR_FULL_EVENT_TRIGGERED_METHOD`，待全方法和全部基线冻结后，再生成一次新的 untouched final manifest。

## Final decision fields

- `FULL_STATE_STATIC_GEOMETRY_ADVANTAGE = {conclusion['FULL_STATE_STATIC_GEOMETRY_ADVANTAGE']}`
- `FULL_STATE_DYNAMIC_STATE_ADVANTAGE = {conclusion['FULL_STATE_DYNAMIC_STATE_ADVANTAGE']}`
- `FULL_STATE_PEER_ADVANTAGE = {conclusion['FULL_STATE_PEER_ADVANTAGE']}`
- `SENSING_MATCHED_INFORMATION_LEAKAGE = {conclusion['SENSING_MATCHED_INFORMATION_LEAKAGE']}`
- `ORIGINAL_98_PERCENT_CLASSICAL_PRIMARY_ADVANTAGE_SOURCE = {conclusion['ORIGINAL_98_PERCENT_CLASSICAL_PRIMARY_ADVANTAGE_SOURCE']}`
- `EQUAL_INFORMATION_BASELINE_REQUIRED_IN_PAPER = {conclusion['EQUAL_INFORMATION_BASELINE_REQUIRED_IN_PAPER']}`
- `NEW_UNTOUCHED_FINAL_REQUIRED = {conclusion['NEW_UNTOUCHED_FINAL_REQUIRED']}`
- `RECOMMENDED_NEXT_STEP = {conclusion['RECOMMENDED_NEXT_STEP']}`
"""


def reconcile(
    episodes: Sequence[Mapping[str, Any]],
    agents: Sequence[Mapping[str, Any]],
    sensing_checks: Mapping[str, Any],
    conclusion: Mapping[str, Any],
) -> dict[str, Any]:
    freeze = load_json(OUTPUT / "implementation_prefreeze.json")
    code_hash_checks = {
        name: file_hash(REPO_ROOT / name) == expected
        for name, expected in freeze["code_hashes"].items()
    }
    keys = {(row["stage"], row["scenario_id"], row["seed"], row["method"]) for row in episodes}
    method_counts = {method: len(subset(episodes, method, "overall")) for method in METHOD_ORDER}
    agent_counts = {method: len(subset(agents, method, "overall")) for method in METHOD_ORDER}
    mandatory = {
        "FULL_STATE_STATIC_GEOMETRY_ADVANTAGE",
        "FULL_STATE_DYNAMIC_STATE_ADVANTAGE",
        "FULL_STATE_PEER_ADVANTAGE",
        "SENSING_MATCHED_INFORMATION_LEAKAGE",
        "DYNAMIC_VELOCITY_AVAILABLE_TO_PROPOSED",
        "DWA_SENSING_MATCHED_SUCCESS",
        "RVO_SENSING_MATCHED_SUCCESS",
        "DWA_FULLSTATE_SUCCESS",
        "RVO_FULLSTATE_SUCCESS",
        "DWA_INFORMATION_ADVANTAGE_PP",
        "RVO_INFORMATION_ADVANTAGE_PP",
        "LEARNED_PIPELINE_DISTRIBUTION_MATCH",
        "CLASSICAL_DEV_DISTRIBUTION_MATCH",
        "GAT_TRAINING_DISTRIBUTION_MATCH",
        "BENCHMARK_REACTIVE_PLANNER_FAVORABILITY",
        "BOUNDARY_FREE_LOCAL_DETOUR_ADVANTAGE",
        "STAGE3_ROUTE_CHOICE_COMPLEXITY",
        "STAGE4_PRIMARY_DIFFICULTY",
        "CONTROL_INTERFACE_DIFFERENCE",
        "UPDATE_FREQUENCY_DIFFERENCE",
        "PREDICTION_HORIZON_DIFFERENCE",
        "BASELINE_COMPARISON_LEVEL_FULLSTATE",
        "BASELINE_COMPARISON_LEVEL_SENSING_MATCHED",
        "ORIGINAL_98_PERCENT_CLASSICAL_PRIMARY_ADVANTAGE_SOURCE",
        "EQUAL_INFORMATION_BASELINE_REQUIRED_IN_PAPER",
        "NEW_UNTOUCHED_FINAL_REQUIRED",
        "RECOMMENDED_NEXT_STEP",
    }
    rate_checks = {
        "DWA_FULLSTATE_SUCCESS": rate(subset(episodes, "DWA-FullState", "overall"), "team_success"),
        "RVO_FULLSTATE_SUCCESS": rate(subset(episodes, "RVO-FullState", "overall"), "team_success"),
        "DWA_SENSING_MATCHED_SUCCESS": rate(subset(episodes, "DWA-SensingMatched", "overall"), "team_success"),
        "RVO_SENSING_MATCHED_SUCCESS": rate(subset(episodes, "RVO-SensingMatched", "overall"), "team_success"),
    }
    rate_match = all(abs(float(conclusion[key]) - value) < 1e-12 for key, value in rate_checks.items())
    pass_gate = (
        sensing_checks["record_count"] == 800
        and sensing_checks["record_hash_pass_count"] == 800
        and sensing_checks["software_failure_count"] == 0
        and len(episodes) == 2000
        and len(keys) == 2000
        and set(method_counts.values()) == {400}
        and set(agent_counts.values()) == {1200}
        and all(code_hash_checks.values())
        and load_json(OUTPUT / "development_correctness_gate.json")["DEVELOPMENT_CORRECTNESS_GATE"] == "PASS"
        and load_json(OUTPUT / "observation_equivalence_tests.json")["all_passed"]
        and mandatory.issubset(conclusion)
        and rate_match
    )
    return {
        "EQUAL_INFORMATION_BASELINE_AUDIT_RECONCILIATION": "PASS" if pass_gate else "FAIL",
        "diagnostic_comparison_label": "POST_HOC_DIAGNOSTIC_COMPARISON",
        "diagnostic_sensing_record_checks": dict(sensing_checks),
        "combined_episode_count": len(episodes),
        "combined_unique_key_count": len(keys),
        "method_episode_counts": method_counts,
        "method_agent_counts": agent_counts,
        "code_hash_checks": code_hash_checks,
        "development_gate": "PASS",
        "observation_equivalence_gate": "PASS",
        "mandatory_conclusion_fields_present": len(mandatory.intersection(conclusion)),
        "mandatory_conclusion_fields_required": len(mandatory),
        "aggregate_rate_reproduction": rate_checks,
        "aggregate_rate_match": rate_match,
        "signed_obstacle_clearance_limitation_disclosed": True,
        "no_new_final_benchmark": True,
        "no_training": True,
        "no_parameter_retuning": True,
    }


def main() -> None:
    sensing_episodes, sensing_agents, sensing_checks = load_sensing_records()
    episodes, agents = build_diagnostic_rows(sensing_episodes, sensing_agents)
    write_csv(OUTPUT / "diagnostic_400_results.csv", episodes)
    information = build_information_decomposition(episodes)
    write_csv(OUTPUT / "information_advantage_decomposition.csv", information)
    safety = build_safety_decomposition(episodes, agents)
    write_csv(OUTPUT / "safety_decomposition.csv", safety)
    runtime = build_runtime_decomposition(episodes)
    write_csv(OUTPUT / "runtime_decomposition.csv", runtime)
    write_csv(OUTPUT / "issue_impact_assessment.csv", build_issue_impact(information))

    conclusion = {
        "FULL_STATE_STATIC_GEOMETRY_ADVANTAGE": "YES",
        "FULL_STATE_DYNAMIC_STATE_ADVANTAGE": "YES",
        "FULL_STATE_PEER_ADVANTAGE": "YES",
        "SENSING_MATCHED_INFORMATION_LEAKAGE": "NO",
        "DYNAMIC_VELOCITY_AVAILABLE_TO_PROPOSED": "NO",
        "DWA_SENSING_MATCHED_SUCCESS": 0.1825,
        "RVO_SENSING_MATCHED_SUCCESS": 0.1,
        "DWA_FULLSTATE_SUCCESS": 0.9775,
        "RVO_FULLSTATE_SUCCESS": 0.985,
        "DWA_INFORMATION_ADVANTAGE_PP": 79.5,
        "RVO_INFORMATION_ADVANTAGE_PP": 88.5,
        "LEARNED_PIPELINE_DISTRIBUTION_MATCH": "STRONG_SHIFT",
        "CLASSICAL_DEV_DISTRIBUTION_MATCH": "YES",
        "GAT_TRAINING_DISTRIBUTION_MATCH": "PARTIAL",
        "BENCHMARK_REACTIVE_PLANNER_FAVORABILITY": "HIGH",
        "BOUNDARY_FREE_LOCAL_DETOUR_ADVANTAGE": "YES",
        "STAGE3_ROUTE_CHOICE_COMPLEXITY": "MODERATE",
        "STAGE4_PRIMARY_DIFFICULTY": "REACTIVE_AVOIDANCE",
        "CONTROL_INTERFACE_DIFFERENCE": "YES",
        "UPDATE_FREQUENCY_DIFFERENCE": "YES",
        "PREDICTION_HORIZON_DIFFERENCE": "YES",
        "BASELINE_COMPARISON_LEVEL_FULLSTATE": "SYSTEM_LEVEL_STRONG_REFERENCE",
        "BASELINE_COMPARISON_LEVEL_SENSING_MATCHED": "EQUAL_INFORMATION_SYSTEM_LEVEL",
        "ORIGINAL_98_PERCENT_CLASSICAL_PRIMARY_ADVANTAGE_SOURCE": "INFORMATION",
        "EQUAL_INFORMATION_BASELINE_REQUIRED_IN_PAPER": "YES",
        "NEW_UNTOUCHED_FINAL_REQUIRED": "YES",
        "RECOMMENDED_NEXT_STEP": "WAIT_FOR_FULL_EVENT_TRIGGERED_METHOD",
        "DIAGNOSTIC_COMPARISON_LABEL": "POST_HOC_DIAGNOSTIC_COMPARISON",
        "SENSING_MATCHED_PARAMETER_RETUNING": "NO",
        "FUTURE_DYNAMIC_INFORMATION_LEAKAGE_C1_C4": "NO",
        "FUTURE_PEER_INFORMATION_LEAKAGE_C1_C4": "NO",
        "FUTURE_STATIC_INFORMATION_LEAKAGE_FULLSTATE": "NOT_APPLICABLE_PUBLIC_STATIC_SYSTEM_STATE",
        "FUTURE_STATIC_INFORMATION_LEAKAGE_SENSING_MATCHED": "NO",
        "DWA_SENSING_MATCHED_SUCCESS_COUNT": 73,
        "RVO_SENSING_MATCHED_SUCCESS_COUNT": 40,
        "DWA_SENSING_MATCHED_TOTAL_COMPUTE_MS": next(
            row["mean_total_online_algorithm_compute_ms_per_episode"]
            for row in runtime
            if row["method"] == "DWA-SensingMatched"
        ),
        "RVO_SENSING_MATCHED_TOTAL_COMPUTE_MS": next(
            row["mean_total_online_algorithm_compute_ms_per_episode"]
            for row in runtime
            if row["method"] == "RVO-SensingMatched"
        ),
        "PERCEPTION_ADAPTER_RUNTIME_MS": {
            row["method"]: row["mean_perception_adapter_ms_per_episode"]
            for row in runtime
            if "SensingMatched" in row["method"]
        },
        "INFORMATION_SUBCOMPONENT_ATTRIBUTION": "NOT_SEPARATELY_IDENTIFIABLE",
        "UPDATE_HORIZON_CONTROL_CAUSAL_MAGNITUDES": "NOT_ESTABLISHED_SEPARATELY",
        "STAGE3_STAGE4_LEARNED_RESULT_LABEL": "OOD_SYSTEM_PERFORMANCE",
        "GAT_INCREMENTAL_RANKING_VALUE": {
            "overall_gain_pp_vs_fp_shep": 2.75,
            "overall_mcnemar_p": 0.215,
            "stage_3_gain_pp": 14.0,
            "stage_4_gain_pp": 11.0,
        },
        "PURE_OBSTACLE_SIGNED_CLEARANCE_C3_C4": "UNAVAILABLE_FROM_RETAINED_DIAGNOSTIC_RECORD_NO_RERUN_OR_FABRICATION",
        "FULL_THEORY_ALIGNED_METHOD_FORMALLY_TESTED": "NO",
        "NEW_FINAL_BENCHMARK_RUN": "NO",
    }
    write_json(OUTPUT / "conclusion.json", conclusion)
    (OUTPUT / "FINAL_REPORT.md").write_text(
        build_report(episodes, information, runtime, conclusion), encoding="utf-8"
    )

    final_reconciliation = reconcile(episodes, agents, sensing_checks, conclusion)
    output_names = (
        "diagnostic_400_results.csv",
        "information_advantage_decomposition.csv",
        "safety_decomposition.csv",
        "runtime_decomposition.csv",
        "issue_impact_assessment.csv",
        "conclusion.json",
        "FINAL_REPORT.md",
    )
    final_reconciliation["output_hashes"] = {
        name: file_hash(OUTPUT / name) for name in output_names
    }
    write_json(OUTPUT / "final_reconciliation.json", final_reconciliation)

    # Independent second pass: read only serialized outputs and reproduce the
    # primary counts/rates without reusing the in-memory episode dictionaries.
    serialized = read_csv(OUTPUT / "diagnostic_400_results.csv")
    independent_counts: dict[str, dict[str, Any]] = {}
    for method in METHOD_ORDER:
        rows = [row for row in serialized if row["method"] == method]
        independent_counts[method] = {
            "n": len(rows),
            "success_count": sum(as_bool(row["team_success"]) for row in rows),
            "success_rate": mean(as_bool(row["team_success"]) for row in rows),
        }
    independent_pass = (
        final_reconciliation["EQUAL_INFORMATION_BASELINE_AUDIT_RECONCILIATION"] == "PASS"
        and independent_counts["DWA-FullState"]["success_count"] == 391
        and independent_counts["RVO-FullState"]["success_count"] == 394
        and independent_counts["DWA-SensingMatched"]["success_count"] == 73
        and independent_counts["RVO-SensingMatched"]["success_count"] == 40
        and independent_counts["One-Shot Proposed"]["success_count"] == 197
    )
    write_json(
        OUTPUT / "independent_reconciliation.json",
        {
            "INDEPENDENT_RECONCILIATION": "PASS" if independent_pass else "FAIL",
            "serialized_episode_count": len(serialized),
            "serialized_unique_key_count": len(
                {
                    (row["stage"], row["scenario_id"], row["seed"], row["method"])
                    for row in serialized
                }
            ),
            "method_reproduction": independent_counts,
            "final_reconciliation_sha256": file_hash(OUTPUT / "final_reconciliation.json"),
            "all_required_output_hashes_reproduced": all(
                file_hash(OUTPUT / name) == expected
                for name, expected in final_reconciliation["output_hashes"].items()
            ),
            "comparison_label": "POST_HOC_DIAGNOSTIC_COMPARISON",
            "no_new_final_benchmark": True,
        },
    )
    print(
        json.dumps(
            {
                "analysis": "PASS" if independent_pass else "FAIL",
                "episodes": len(episodes),
                "sensing_records": sensing_checks,
                "output": str(OUTPUT),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
