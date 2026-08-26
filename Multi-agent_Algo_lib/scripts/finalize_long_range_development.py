"""Close long-range development and materialize the pre-formal audit trail.

This script reads development/training artifacts only.  It refuses to run once
the untouched formal manifest or any formal episode exists.  Its output is the
single auditable hand-off from development to the short-range final-method
check and, subsequently, the untouched long-range formal benchmark.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT / "artifacts/semi_structured_long_range_main_benchmark/20260820_193228"
DEVELOPMENT_DIR = ARTIFACT_ROOT / "08_development"
BASELINE_DIR = ARTIFACT_ROOT / "09_baseline_tuning"
TRAINING_DIR = ARTIFACT_ROOT / "07_training/sac_long_range_256_encoder_adapt_run1"
FREEZE_DIR = ARTIFACT_ROOT / "10_final_freeze"
FORMAL_MANIFEST = ARTIFACT_ROOT / "11_formal_manifest/FINAL_LONG_RANGE_MANIFEST.json"
FORMAL_RECORD_DIR = ARTIFACT_ROOT / "12_formal_records"

FINAL_PROPOSED_DIR = "D05_interaction_feasibility_mask_full200"
STRICT_RERR_FP_DIR = "RERR_FP_F00_base_common_rerr_full200"
AUXILIARY_RERR_FP_DIR = "RERR_FP_F02_long_rep_long_dwell_full200"
FINAL_DWA_FS_DIR = "DWA_FS_FULL_S00"
FINAL_DWA_SM_DIR = "DWA_SM_FULL_S05"

REQUIRED_FULL_DEVELOPMENT = {
    "DWA-FullState": BASELINE_DIR / FINAL_DWA_FS_DIR / "development_reconciliation.json",
    "DWA-SensingMatched": BASELINE_DIR / FINAL_DWA_SM_DIR / "development_reconciliation.json",
    "Direct SAC-DMP": DEVELOPMENT_DIR / "ABL_M4_Direct_SAC_DMP_full200/development_reconciliation.json",
    "Proposal + SAC-DMP": DEVELOPMENT_DIR / "ABL_M5_Proposal_SAC_DMP_full200/development_reconciliation.json",
    "FP-SHEP + SAC-DMP": DEVELOPMENT_DIR / "ABL_M6_FP_SHEP_SAC_DMP_full200/development_reconciliation.json",
    "One-Shot GAT + SAC-DMP": DEVELOPMENT_DIR / "ABL_M7_OneShot_GAT_SAC_DMP_full200/development_reconciliation.json",
    "R-ERR + FP-SHEP + SAC-DMP": DEVELOPMENT_DIR / STRICT_RERR_FP_DIR / "development_reconciliation.json",
    "Proposed": DEVELOPMENT_DIR / FINAL_PROPOSED_DIR / "development_reconciliation.json",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"refusing to write empty audit table: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if str(key) not in fields:
                fields.append(str(key))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(json_ready(row.get(key)), ensure_ascii=False, separators=(",", ":"))
                    if isinstance(row.get(key), (dict, list, tuple))
                    else row.get(key)
                    for key in fields
                }
            )
    temporary.replace(path)


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def assert_formal_unopened() -> None:
    if FORMAL_MANIFEST.exists():
        raise RuntimeError("formal manifest already exists; development closure cannot be regenerated")
    if FORMAL_RECORD_DIR.exists() and any(FORMAL_RECORD_DIR.rglob("*.json")):
        raise RuntimeError("formal records already exist; development closure cannot be regenerated")


def config_index() -> dict[str, tuple[Path, dict[str, Any]]]:
    index: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in sorted((ROOT / "configs").rglob("*.json")):
        try:
            payload = load_json(path)
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        output_subdir = payload.get("output_subdir")
        if isinstance(output_subdir, str):
            index[output_subdir.replace("\\", "/").rstrip("/")] = (path, payload)
    return index


def read_summary(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return {str(row["scope"]): dict(row) for row in csv.DictReader(handle)}


def family_success_rates(path: Path) -> dict[str, float]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[str, list[bool]] = {}
    for row in rows:
        grouped.setdefault(str(row["family"]), []).append(
            str(row["team_success"]).strip().lower() in {"1", "true", "yes"}
        )
    return {
        family: sum(values) / len(values)
        for family, values in sorted(grouped.items())
        if values
    }


def selected_role(parent: Path) -> str:
    name = parent.name
    if name == FINAL_PROPOSED_DIR:
        return "FINAL_PROPOSED"
    if name == STRICT_RERR_FP_DIR:
        return "FORMAL_GAT_ABLATION_IDENTICAL_RERR"
    if name == AUXILIARY_RERR_FP_DIR:
        return "AUXILIARY_BEST_RERR_FP_NOT_FORMAL"
    if name == FINAL_DWA_FS_DIR:
        return "FINAL_FULLSTATE_REFERENCE"
    if name == FINAL_DWA_SM_DIR:
        return "FINAL_MATCHED_BASELINE"
    if name.startswith("ABL_M"):
        return "FINAL_FIXED_ABLATION"
    return "NOT_SELECTED"


def method_family(parent: Path, freeze: Mapping[str, Any]) -> str:
    name = parent.name
    if name.startswith("DWA_FS"):
        return "DWA-FullState"
    if name.startswith("DWA_SM"):
        return "DWA-SensingMatched"
    if name.startswith("RERR_FP"):
        return "R-ERR + FP-SHEP + SAC-DMP"
    if name.startswith("D0"):
        return "Proposed"
    if name.startswith("ABL_M4"):
        return "Direct SAC-DMP"
    if name.startswith("ABL_M5"):
        return "Proposal + SAC-DMP"
    if name.startswith("ABL_M6"):
        return "FP-SHEP + SAC-DMP"
    if name.startswith("ABL_M7"):
        return "One-Shot GAT + SAC-DMP"
    return str(freeze.get("method", name))


def configuration_parameters(config: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "configuration_id",
        "method",
        "top_k",
        "H_preview",
        "max_steps",
        "dt",
        "maximum_speed_norm_mps",
        "sensor",
        "proposal_config",
        "fp_shep",
        "graph",
        "err",
        "planner",
    )
    return {key: config[key] for key in keys if key in config}


def collect_history() -> list[dict[str, Any]]:
    index = config_index()
    rows: list[dict[str, Any]] = []
    search_roots = (DEVELOPMENT_DIR, BASELINE_DIR)
    for search_root in search_roots:
        for reconciliation_path in sorted(search_root.glob("*/development_reconciliation.json")):
            parent = reconciliation_path.parent
            reconciliation = load_json(reconciliation_path)
            freeze_path = parent / "development_freeze.json"
            freeze = load_json(freeze_path) if freeze_path.is_file() else {}
            output_key = relative(parent).split("/", 4)[-1]
            if output_key.startswith("08_development/") or output_key.startswith("09_baseline_tuning/"):
                config_key = output_key
            else:
                config_key = f"{search_root.name}/{parent.name}"
            config_entry = index.get(config_key)
            config_path, config = config_entry if config_entry else (None, {})
            summary = read_summary(parent / "development_summary.csv")
            overall = dict(reconciliation.get("overall") or summary.get("overall") or {})
            row: dict[str, Any] = {
                "evaluation_directory": parent.name,
                "method_family": method_family(parent, freeze),
                "configuration_id": config.get("configuration_id", freeze.get("configuration_id", parent.name)),
                "evaluation_scope": "full200" if int(reconciliation.get("expected_scenarios", 0)) == 200 else "panel40",
                "scenario_count": int(reconciliation.get("completed_scenarios", overall.get("n", 0))),
                "status": reconciliation.get("status"),
                "software_error_count": int(reconciliation.get("software_error_count", 0)),
                "team_success_rate": overall.get("team_success_rate"),
                "collision_rate": overall.get("collision_rate"),
                "obstacle_collision_rate": overall.get("obstacle_collision_rate"),
                "inter_agent_collision_rate": overall.get("inter_agent_collision_rate"),
                "timeout_rate": overall.get("timeout_rate"),
                "agent_completion_rate": overall.get("agent_completion_rate"),
                "mean_total_compute_ms": overall.get("mean_total_compute_ms", overall.get("mean_compute_ms")),
                "stage_1_success": (summary.get("stage_1") or {}).get("team_success_rate"),
                "stage_2_success": (summary.get("stage_2") or {}).get("team_success_rate"),
                "stage_3_success": (summary.get("stage_3") or {}).get("team_success_rate"),
                "stage_4_success": (summary.get("stage_4") or {}).get("team_success_rate"),
                "selection_role": selected_role(parent),
                "configuration_path": relative(config_path) if config_path else "UNAVAILABLE",
                "configuration_sha256": sha256_file(config_path) if config_path else freeze.get("configuration_sha256"),
                "parameters": configuration_parameters(config),
                "reconciliation_path": relative(reconciliation_path),
                "reconciliation_sha256": sha256_file(reconciliation_path),
                "formal_data_used": reconciliation.get("formal_data_used", freeze.get("formal_data_used", False)),
            }
            rows.append(row)
    rows.sort(key=lambda row: (str(row["method_family"]), str(row["configuration_id"]), str(row["evaluation_directory"])))
    return rows


def checkpoint_history() -> list[dict[str, Any]]:
    result_path = TRAINING_DIR / "training_result.json"
    result = load_json(result_path)
    best_timestep = int(load_json(TRAINING_DIR / "post_training_freeze_reconciliation.json")["best_validation_timestep"])
    rows: list[dict[str, Any]] = []
    for validation in result["validation_history"]:
        timestep = int(validation["training_timestep"])
        checkpoint_name = str(validation["checkpoint"])
        if timestep == 0:
            checkpoint_path = ROOT / result["source_checkpoint"]
            checkpoint_role = "SOURCE_56_RAY_PROVENANCE"
        else:
            checkpoint_path = TRAINING_DIR / "checkpoints" / f"checkpoint_{timestep:07d}.pt"
            checkpoint_role = "SELECTED_BEST_VALIDATION" if timestep == best_timestep else "NOT_SELECTED"
        rows.append(
            {
                "model": "adapted 256-ray SAC-DMP actor",
                "training_timestep": timestep,
                "checkpoint_label": checkpoint_name,
                "checkpoint_path": relative(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "validation_episode_count": validation["episode_count"],
                "validation_success_rate": validation["success_rate"],
                "validation_collision_rate": validation["collision_rate"],
                "validation_boundary_collision_rate": validation["boundary_collision_rate"],
                "validation_timeout_rate": validation["timeout_rate"],
                "selection_role": checkpoint_role,
                "selection_rule": "max success, then min collision/timeout, earliest exact tie",
                "development_data_used": False,
                "formal_data_used": False,
            }
        )
    best_path = TRAINING_DIR / "best_validation.pt"
    rows.append(
        {
            "model": "adapted 256-ray SAC-DMP actor",
            "training_timestep": best_timestep,
            "checkpoint_label": "deployment_copy_best_validation.pt",
            "checkpoint_path": relative(best_path),
            "checkpoint_sha256": sha256_file(best_path),
            "validation_episode_count": next(row["episode_count"] for row in result["validation_history"] if int(row["training_timestep"]) == best_timestep),
            "validation_success_rate": next(row["success_rate"] for row in result["validation_history"] if int(row["training_timestep"]) == best_timestep),
            "validation_collision_rate": next(row["collision_rate"] for row in result["validation_history"] if int(row["training_timestep"]) == best_timestep),
            "validation_boundary_collision_rate": next(row["boundary_collision_rate"] for row in result["validation_history"] if int(row["training_timestep"]) == best_timestep),
            "validation_timeout_rate": next(row["timeout_rate"] for row in result["validation_history"] if int(row["training_timestep"]) == best_timestep),
            "selection_role": "FINAL_DEPLOYMENT_CHECKPOINT",
            "selection_rule": "frozen copy of selected validation checkpoint",
            "development_data_used": False,
            "formal_data_used": False,
        }
    )
    gat = ROOT / "artifacts/gat_stage1_training/20260815_230509/checkpoints/best_validation.pt"
    rows.append(
        {
            "model": "GAT-V1",
            "training_timestep": "historical",
            "checkpoint_label": "historical_gat_v1_best_validation",
            "checkpoint_path": relative(gat),
            "checkpoint_sha256": sha256_file(gat),
            "validation_episode_count": "historical",
            "validation_success_rate": "historical",
            "validation_collision_rate": "historical",
            "validation_boundary_collision_rate": "historical",
            "validation_timeout_rate": "historical",
            "selection_role": "FINAL_FROZEN_GAT_CHECKPOINT",
            "selection_rule": "GAT retraining was not required; preserve validated V1",
            "development_data_used": False,
            "formal_data_used": False,
        }
    )
    return rows


def freeze_classical(name: str, source_dir: Path, display_name: str, information_contract: str) -> Path:
    reconciliation_path = source_dir / "development_reconciliation.json"
    development_freeze_path = source_dir / "development_freeze.json"
    reconciliation = load_json(reconciliation_path)
    development_freeze = load_json(development_freeze_path)
    config_path = ROOT / next(
        relative(path)
        for path in (ROOT / "configs/evaluation").glob("*.json")
        if load_json(path).get("output_subdir") == f"09_baseline_tuning/{source_dir.name}"
    )
    output_path = FREEZE_DIR / name
    atomic_json(
        output_path,
        {
            "schema_version": "long_range_classical_final_freeze_v1",
            "status": "FROZEN_AFTER_DEVELOPMENT_BEFORE_FORMAL",
            "created_at": datetime.now().astimezone().isoformat(),
            "method": display_name,
            "information_contract": information_contract,
            "configuration_path": relative(config_path),
            "configuration_sha256": sha256_file(config_path),
            "resolved_configuration": load_json(config_path),
            "development_evidence_path": relative(reconciliation_path),
            "development_evidence_sha256": sha256_file(reconciliation_path),
            "development_overall": reconciliation["overall"],
            "development_freeze_path": relative(development_freeze_path),
            "development_freeze_sha256": sha256_file(development_freeze_path),
            "future_dynamic_information_used": False,
            "formal_data_used": False,
            "post_freeze_tuning_allowed": False,
        },
    )
    return output_path


def main() -> None:
    assert_formal_unopened()
    full_evidence: dict[str, Any] = {}
    for method, path in REQUIRED_FULL_DEVELOPMENT.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing required full development result: {relative(path)}")
        payload = load_json(path)
        if payload.get("status") != "PASS":
            raise RuntimeError(f"development result is not PASS: {method}")
        if int(payload.get("completed_scenarios", 0)) != 200:
            raise RuntimeError(f"development result is not complete 200: {method}")
        if int(payload.get("software_error_count", 0)) != 0:
            raise RuntimeError(f"development result contains software errors: {method}")
        if bool(payload.get("formal_data_used", False)):
            raise RuntimeError(f"formal data contamination in development: {method}")
        full_evidence[method] = {
            "path": relative(path),
            "sha256": sha256_file(path),
            "overall": payload["overall"],
        }

    history = collect_history()
    if not history or any(bool(row["formal_data_used"]) for row in history):
        raise RuntimeError("development history is empty or contaminated by formal data")
    long_history_path = DEVELOPMENT_DIR / "LONG_RANGE_DEV_HISTORY.csv"
    tuning_history_path = BASELINE_DIR / "TUNING_HISTORY.csv"
    checkpoint_history_path = ARTIFACT_ROOT / "07_training/CHECKPOINT_SELECTION_HISTORY.csv"
    write_csv(long_history_path, history)
    write_csv(
        tuning_history_path,
        [row for row in history if row["method_family"] in {"Proposed", "R-ERR + FP-SHEP + SAC-DMP", "DWA-FullState", "DWA-SensingMatched"}],
    )
    write_csv(checkpoint_history_path, checkpoint_history())

    dwa_fs_path = freeze_classical(
        "DWA_FULLSTATE_FREEZE.json",
        BASELINE_DIR / FINAL_DWA_FS_DIR,
        "DWA-FullState",
        "exact current static/dynamic/peer state; constant-velocity prediction; no future trajectory",
    )
    dwa_sm_path = freeze_classical(
        "DWA_SENSING_MATCHED_FREEZE.json",
        BASELINE_DIR / FINAL_DWA_SM_DIR,
        "DWA-SensingMatched",
        "ego state + task goal + same 4.5 m untyped 256-direction current/previous range observations",
    )

    proposed_summary = read_summary(DEVELOPMENT_DIR / FINAL_PROPOSED_DIR / "development_summary.csv")
    proposed = proposed_summary["overall"]
    stage_rates = [float(proposed_summary[f"stage_{index}"]["team_success_rate"]) for index in range(1, 5)]
    summarized_family_rates = [
        float(row["team_success_rate"])
        for scope, row in proposed_summary.items()
        if scope.startswith("family:")
    ]
    family_by_name = family_success_rates(
        DEVELOPMENT_DIR / FINAL_PROPOSED_DIR / "development_team_results.csv"
    )
    family_rates = summarized_family_rates or list(family_by_name.values())
    matched = full_evidence["DWA-SensingMatched"]["overall"]
    training_result_path = TRAINING_DIR / "training_result.json"
    training_result = load_json(training_result_path)
    selected_sac = ROOT / training_result["best_validation_checkpoint"] if "/" in training_result["best_validation_checkpoint"] else TRAINING_DIR / training_result["best_validation_checkpoint"]
    selected_gat = ROOT / "artifacts/gat_stage1_training/20260815_230509/checkpoints/best_validation.pt"
    selected_configs = {
        "Proposed": ROOT / "configs/evaluation/semi_structured_long_range_development_d05.json",
        "R-ERR + FP-SHEP formal ablation": ROOT / "configs/evaluation/semi_structured_long_range_rerr_fp_f00_full200.json",
        "R-ERR + FP-SHEP auxiliary best": ROOT / "configs/evaluation/semi_structured_long_range_rerr_fp_f02_full200.json",
        "DWA-FullState": ROOT / "configs/evaluation/long_range_dwa_fs_full_s00.json",
        "DWA-SensingMatched": ROOT / "configs/evaluation/long_range_dwa_sm_full_s05.json",
    }
    selection_freeze = {
        "schema_version": "long_range_development_selection_freeze_v1",
        "status": "DEVELOPMENT_CLOSED_BEFORE_SHORT_AND_FORMAL",
        "created_at": datetime.now().astimezone().isoformat(),
        "development_scenario_count": 200,
        "development_manifest": relative(DEVELOPMENT_DIR / "development_manifest.json"),
        "development_manifest_sha256": sha256_file(DEVELOPMENT_DIR / "development_manifest.json"),
        "formal_manifest_exists_at_close": False,
        "formal_episode_count_at_close": 0,
        "formal_data_used_for_training_tuning_or_selection": False,
        "final_proposed_configuration_id": "D05",
        "final_proposed_development_success": float(proposed["team_success_rate"]),
        "final_proposed_development_collision": float(proposed["collision_rate"]),
        "final_proposed_stage_success": stage_rates,
        "final_proposed_family_success": family_by_name,
        "final_proposed_min_family_success": min(family_rates) if family_rates else None,
        "no_catastrophic_family_failure": bool(family_rates) and min(family_rates) >= 0.80,
        "development_90_percent_target_met": float(proposed["team_success_rate"]) >= 0.90,
        "strongest_matched_baseline": "DWA-SensingMatched",
        "strongest_matched_baseline_development_success": float(matched["team_success_rate"]),
        "proposed_matches_or_exceeds_strongest_matched_baseline": float(proposed["team_success_rate"]) >= float(matched["team_success_rate"]),
        "strict_gat_ablation_configuration_id": "RERR_FP_F00",
        "strict_gat_ablation_reason": "identical R-ERR parameters to D05; GAT selection is the only intended selector difference",
        "auxiliary_rerr_fp_configuration_id": "RERR_FP_F02",
        "auxiliary_rerr_fp_formal_status": "NOT_USED_AS_FORMAL_GAT_ABLATION",
        "selected_configuration_sha256": {name: sha256_file(path) for name, path in selected_configs.items()},
        "selected_sac_checkpoint": relative(selected_sac),
        "selected_sac_checkpoint_sha256": sha256_file(selected_sac),
        "selected_gat_checkpoint": relative(selected_gat),
        "selected_gat_checkpoint_sha256": sha256_file(selected_gat),
        "actor_trainable_scope": training_result["actor_trainable_scope"],
        "actor_trunk_and_action_heads_frozen_exactly": training_result["frozen_actor_exact_match"],
        "gat_retrained": False,
        "sac_sensor_encoder_adapted": True,
        "core_theory_changed": False,
        "required_full_development_evidence": full_evidence,
        "audit_artifacts": {
            "long_range_development_history": {"path": relative(long_history_path), "sha256": sha256_file(long_history_path)},
            "tuning_history": {"path": relative(tuning_history_path), "sha256": sha256_file(tuning_history_path)},
            "checkpoint_selection_history": {"path": relative(checkpoint_history_path), "sha256": sha256_file(checkpoint_history_path)},
            "method_tuning_budget": {"path": relative(BASELINE_DIR / "METHOD_TUNING_BUDGET.csv"), "sha256": sha256_file(BASELINE_DIR / "METHOD_TUNING_BUDGET.csv")},
            "hyperparameter_search_space": {"path": relative(BASELINE_DIR / "HYPERPARAMETER_SEARCH_SPACE.json"), "sha256": sha256_file(BASELINE_DIR / "HYPERPARAMETER_SEARCH_SPACE.json")},
            "ppo_baseline_audit": {"path": relative(BASELINE_DIR / "PPO_BASELINE_AUDIT.json"), "sha256": sha256_file(BASELINE_DIR / "PPO_BASELINE_AUDIT.json")},
            "dwa_fullstate_freeze": {"path": relative(dwa_fs_path), "sha256": sha256_file(dwa_fs_path)},
            "dwa_sensing_matched_freeze": {"path": relative(dwa_sm_path), "sha256": sha256_file(dwa_sm_path)},
            "training_result": {"path": relative(training_result_path), "sha256": sha256_file(training_result_path)},
        },
        "post_close_training_or_tuning_allowed": False,
        "next_authorized_experiment": "frozen final-method short-horizon coordination revalidation, then untouched formal prepare",
    }
    selection_path = FREEZE_DIR / "DEVELOPMENT_SELECTION_FREEZE.json"
    atomic_json(selection_path, selection_freeze)
    atomic_json(
        FREEZE_DIR / "development_closure_reconciliation.json",
        {
            "status": "PASS",
            "formal_manifest_absent": not FORMAL_MANIFEST.exists(),
            "formal_record_count": 0,
            "required_full_method_count": len(full_evidence),
            "required_full_development_all_pass": True,
            "development_history_row_count": len(history),
            "development_90_percent_target_met": selection_freeze["development_90_percent_target_met"],
            "matched_baseline_target_met": selection_freeze["proposed_matches_or_exceeds_strongest_matched_baseline"],
            "no_catastrophic_family_failure": selection_freeze["no_catastrophic_family_failure"],
            "selected_checkpoint_hashes_match": sha256_file(selected_sac) == training_result["best_validation_checkpoint_sha256"],
            "selection_freeze_path": relative(selection_path),
            "selection_freeze_sha256": sha256_file(selection_path),
        },
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "development_history_rows": len(history),
                "proposed_success": float(proposed["team_success_rate"]),
                "matched_baseline_success": float(matched["team_success_rate"]),
                "formal_manifest_exists": False,
            }
        )
    )


if __name__ == "__main__":
    main()
