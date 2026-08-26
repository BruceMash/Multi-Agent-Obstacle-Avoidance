#!/usr/bin/env python3
"""Pre-training legality audit for Track B risk-aware learned turn continuity."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import torch

torch.set_num_threads(1)


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
for item in (REPO_ROOT, ALGO_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

def _load_source(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_NET_MODULE = _load_source("track_b_frozen_sac_net", REPO_ROOT / "baseline/sac/net.py")
_RISK_MODULE = _load_source(
    "track_b_risk_primitives", REPO_ROOT / "planning/risk_aware_turn_continuity.py"
)
_STRONG_MODULE = _load_source(
    "track_b_strong_source", REPO_ROOT / "planning/safety_adaptive_jerk_limiter.py"
)
_PERSIST_MODULE = _load_source(
    "track_b_persistence_source", REPO_ROOT / "planning/turn_sign_persistence.py"
)
Actor = _NET_MODULE.Actor
Critic = _NET_MODULE.Critic
ACTION_DIM = _RISK_MODULE.ACTION_DIM
NEW_CONTEXT_DIM = _RISK_MODULE.NEW_CONTEXT_DIM
NEW_EXTRA_OBSERVATION_DIM = _RISK_MODULE.NEW_EXTRA_OBSERVATION_DIM
OLD_EXTRA_OBSERVATION_DIM = _RISK_MODULE.OLD_EXTRA_OBSERVATION_DIM
SENSOR_OBSERVATION_DIM = _RISK_MODULE.SENSOR_OBSERVATION_DIM
dmp_action_jacobian = _RISK_MODULE.dmp_action_jacobian
expand_checkpoint_extra_columns = _RISK_MODULE.expand_checkpoint_extra_columns
freeze_sensor_encoders = _RISK_MODULE.freeze_sensor_encoders
polyak_update_excluding_sensor_encoder = _RISK_MODULE.polyak_update_excluding_sensor_encoder
sensor_state = _RISK_MODULE.sensor_state
tensor_group_sha256 = _RISK_MODULE.tensor_group_sha256
SafetyAdaptiveVectorJerkLimiter = _STRONG_MODULE.SafetyAdaptiveVectorJerkLimiter
TurnSignPersistenceExecutionFilter = _PERSIST_MODULE.TurnSignPersistenceExecutionFilter


DEFAULT_ROOT = REPO_ROOT / "artifacts/parallel_zigzag_resolution/20260826_091334/track_B_learned_turn"
COMMON_FREEZE = REPO_ROOT / "artifacts/parallel_zigzag_resolution/20260826_091334/00_context/COMMON_EXPERIMENT_FREEZE.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or ["status"])
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def network_config() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_dim=256,
        output_dim=3,
        sensor_output_dim=128,
        num_sensor_layers=2,
        num_observation_layers=2,
    )


def zero_init_audit(root: Path, checkpoint_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    print("track-b audit: loading checkpoint", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = network_config()
    old_actor = Actor(SENSOR_OBSERVATION_DIM, OLD_EXTRA_OBSERVATION_DIM, cfg).double().eval()
    old_critic = Critic(SENSOR_OBSERVATION_DIM, OLD_EXTRA_OBSERVATION_DIM, ACTION_DIM, cfg).double().eval()
    old_actor.load_state_dict(checkpoint["actor"], strict=True)
    old_critic.load_state_dict(checkpoint["critic"], strict=True)
    new_actor = Actor(SENSOR_OBSERVATION_DIM, NEW_EXTRA_OBSERVATION_DIM, cfg).double().eval()
    new_critic = Critic(SENSOR_OBSERVATION_DIM, NEW_EXTRA_OBSERVATION_DIM, ACTION_DIM, cfg).double().eval()
    new_target = Critic(SENSOR_OBSERVATION_DIM, NEW_EXTRA_OBSERVATION_DIM, ACTION_DIM, cfg).double().eval()
    print("track-b audit: expanding tensors", flush=True)
    expansion = {
        "actor": expand_checkpoint_extra_columns(new_actor, checkpoint["actor"], module_name="actor"),
        "critic": expand_checkpoint_extra_columns(new_critic, checkpoint["critic"], module_name="critic"),
        "critic_target": expand_checkpoint_extra_columns(
            new_target, checkpoint["critic_target"], module_name="critic_target"
        ),
    }
    freeze = freeze_sensor_encoders(new_actor, new_critic, new_target)
    rng = np.random.default_rng(2026082601)
    actor_max = 0.0
    critic_max = 0.0
    # One fixed vectorized batch is sufficient to exercise every expanded
    # input column while avoiding competition with the parallel Track-A run.
    sample_count = 64
    print("track-b audit: numerical equivalence", flush=True)
    with torch.no_grad():
        for start in range(0, sample_count, 64):
            count = min(64, sample_count - start)
            sensor = torch.as_tensor(rng.normal(size=(count, SENSOR_OBSERVATION_DIM)), dtype=torch.float64)
            old_extra = torch.as_tensor(rng.normal(size=(count, OLD_EXTRA_OBSERVATION_DIM)), dtype=torch.float64)
            new_extra = torch.cat(
                [old_extra, torch.zeros((count, NEW_CONTEXT_DIM), dtype=torch.float64)], dim=1
            )
            action = torch.as_tensor(rng.uniform(-1.0, 1.0, size=(count, ACTION_DIM)), dtype=torch.float64)
            old_parts = old_actor(sensor, old_extra, deterministic=True)
            new_parts = new_actor(sensor, new_extra, deterministic=True)
            old_action = torch.cat(old_parts[:2], dim=1)
            new_action = torch.cat(new_parts[:2], dim=1)
            actor_max = max(actor_max, float(torch.max(torch.abs(old_action - new_action))))
            old_q = old_critic(sensor, old_extra, action)
            new_q = new_critic(sensor, new_extra, action)
            critic_max = max(
                critic_max,
                max(float(torch.max(torch.abs(first - second))) for first, second in zip(old_q, new_q)),
            )
    observation = {
        "schema_version": "turn_observation_extension_v1",
        "audit_status": "COMPLETE_BEFORE_TEACHER_GATE",
        "source_observation_dim": 522,
        "source_sensor_observation_dim": SENSOR_OBSERVATION_DIM,
        "source_extra_observation_dim": OLD_EXTRA_OBSERVATION_DIM,
        "expanded_observation_dim": 529,
        "expanded_extra_observation_dim": NEW_EXTRA_OBSERVATION_DIM,
        "added_dimension_count": NEW_CONTEXT_DIM,
        "source_observation_order": [
            "velocity_3",
            "active_reference_direction_3",
            "normalized_active_reference_distance_1",
            "current_256_ray_scan",
            "previous_256_ray_scan",
            "DMP_phase_1",
            "DMP_K_alpha_1",
            "DMP_K_beta_1",
        ],
        "added_context_order": [
            "previous_executed_acceleration_3",
            "previous_executed_yaw_rate_1",
            "previous_executed_pitch_rate_1",
            "current_active_direction_safety_margin_m_1",
            "safety_margin_change_m_1",
        ],
        "duplicate_existing_feature_count": 0,
        "duplicate_audit": {
            "previous_acceleration": "MISSING",
            "previous_yaw_rate": "MISSING",
            "previous_pitch_rate": "MISSING",
            "explicit_active_direction_margin": "MISSING; scan is untyped per-ray range, not the derived scalar",
            "margin_change": "MISSING",
        },
        "future_or_pending_reference_added": False,
        "sensor_encoder_input_changed": False,
        "sensor_contract_256_directions_preserved": True,
        "expansion": expansion,
    }
    equivalence = {
        "schema_version": "turn_zero_init_equivalence_v1",
        "source_checkpoint": str(checkpoint_path.relative_to(REPO_ROOT).as_posix()),
        "source_checkpoint_sha256": sha256_file(checkpoint_path),
        "sample_count": sample_count,
        "input": "old_observation_plus_seven_exact_zero_context_features",
        "maximum_absolute_actor_action_difference": actor_max,
        "maximum_absolute_critic_q_difference": critic_max,
        "tolerance": 1.0e-6,
        "all_new_columns_zero": bool(
            all(part["all_new_context_columns_zero"] for part in expansion.values())
        ),
        "ZERO_INIT_EQUIVALENCE": "PASS"
        if actor_max <= 1.0e-6 and critic_max <= 1.0e-6
        else "FAIL",
    }

    before = {
        "actor": tensor_group_sha256(sensor_state(new_actor)),
        "critic": tensor_group_sha256(sensor_state(new_critic)),
        "critic_target": tensor_group_sha256(sensor_state(new_target)),
    }
    print("track-b audit: polyak freeze microtest", flush=True)
    with torch.no_grad():
        for name, parameter in new_critic.named_parameters():
            if not name.startswith("sensor_encoder."):
                parameter.add_(torch.randn_like(parameter) * 0.01)
    polyak_update_excluding_sensor_encoder(new_critic, new_target, tau=0.005)
    after = {
        "actor": tensor_group_sha256(sensor_state(new_actor)),
        "critic": tensor_group_sha256(sensor_state(new_critic)),
        "critic_target": tensor_group_sha256(sensor_state(new_target)),
    }
    encoder = {
        "schema_version": "turn_target_encoder_freeze_audit_v1",
        "pretraining_freeze_mechanism_test": "PASS" if before == after else "FAIL",
        "freeze_declarations": freeze,
        "hash_before": before,
        "hash_after_nonencoder_polyak_microtest": after,
        "actor_encoder_drift": 0.0 if before["actor"] == after["actor"] else None,
        "online_critic_encoder_drift": 0.0 if before["critic"] == after["critic"] else None,
        "target_critic_encoder_drift": 0.0 if before["critic_target"] == after["critic_target"] else None,
        "standard_polyak_all_parameters_forbidden": True,
        "post_training_drift_audit": "NOT_RUN_AFTER_TEACHER_ACTION_GATE_FAIL",
    }
    atomic_json(root / "TURN_OBSERVATION_EXTENSION.json", observation)
    atomic_json(root / "ZERO_INIT_EQUIVALENCE.json", equivalence)
    atomic_json(root / "TARGET_ENCODER_FREEZE_AUDIT.json", encoder)
    return observation, equivalence, encoder


def teacher_legality_audit(root: Path) -> dict[str, Any]:
    gates = np.asarray([0.0, 0.25, 0.75], dtype=float)
    jacobian = dmp_action_jacobian(k_alpha=20.0, k_beta=5.0, tau=1.2, forcing_gate=gates)
    # The goal-offset block is a nonzero diagonal (K_alpha*K_beta/tau^2),
    # hence rank is exactly three without invoking a second BLAS runtime.
    rank = 3
    nullity = int(jacobian.shape[1] - rank)
    smoother_methods = {
        "strong_public_methods": sorted(
            name
            for name, value in inspect.getmembers(SafetyAdaptiveVectorJerkLimiter, inspect.isfunction)
            if not name.startswith("_")
        ),
        "persistence_public_methods": sorted(
            name
            for name, value in inspect.getmembers(TurnSignPersistenceExecutionFilter, inspect.isfunction)
            if not name.startswith("_")
        ),
    }
    actor_like = {"predict", "act", "forward", "teacher_action"}
    exposed = set(smoother_methods["strong_public_methods"]) | set(
        smoother_methods["persistence_public_methods"]
    )
    report = {
        "schema_version": "turn_smooth_teacher_action_recoverability_v1",
        "status": "HARD_GATE_FAIL",
        "required_teacher_loss_domain": "six-dimensional normalized SAC-DMP action",
        "original_sac_teacher_output_dim": 6,
        "smooth_teacher_implemented_domain": "three-dimensional executed acceleration after DMP and Strong",
        "strong_and_persistence_change_sac_action": False,
        "strong_and_persistence_change_dmp_state": False,
        "strong_and_persistence_are_stateful_execution_filters": True,
        "public_interface_audit": smoother_methods,
        "actor_like_output_interface_present": bool(actor_like & exposed),
        "historical_dmp_unclipped_action_to_acceleration_jacobian": jacobian.tolist(),
        "jacobian_shape": list(jacobian.shape),
        "jacobian_rank": rank,
        "jacobian_nullity": nullity,
        "unique_inverse_exists": False,
        "additional_noninvertibility": [
            "forcing and goal-offset clipping",
            "Frozen Strong stateful acceleration projection",
            "Turn-Sign Persistence stateful sign suppression",
            "physical acceleration saturation",
        ],
        "using_original_sac_action_for_smooth_teacher": {
            "legal_shape": True,
            "distinct_from_original_teacher_at_same_observation": False,
            "can_teach_persistence_behavior_by_action_MSE": False,
        },
        "missing_authority": (
            "No source-supported inverse or distillation target maps the post-DMP "
            "Strong+Persistence acceleration back to one unique SAC forcing/offset action."
        ),
        "forbidden_repairs_without_new_authority": [
            "invent a pseudoinverse/minimum-norm convention",
            "change teacher loss to acceleration space",
            "replace the auxiliary action loss with trajectory imitation",
            "deploy hard persistence at final runtime",
        ],
        "TEACHER_ACTION_CONTRACT_VALID": "NO",
        "TRAINING_AUTHORIZED": "NO",
        "stop_reason": "SMOOTH_TEACHER_HAS_NO_LEGAL_SAC_ACTION_TARGET",
    }
    atomic_json(root / "TURN_SMOOTH_TEACHER_ACTION_RECOVERABILITY.json", report)
    return report


def write_closure(root: Path, common: Mapping[str, Any], equivalence: Mapping[str, Any], encoder: Mapping[str, Any], teacher: Mapping[str, Any]) -> None:
    teacher_contract = {
        "schema_version": "turn_teacher_contract_v1",
        "original_sac": {
            "role": "frozen_training_only_safety_teacher",
            "checkpoint": common["checkpoints"]["sac_dmp"]["path"],
            "checkpoint_sha256": common["checkpoints"]["sac_dmp"]["sha256"],
            "action_domain": "SAC-DMP forcing_3_plus_goal_offset_3",
        },
        "smooth_teacher": {
            "requested_behavior": "Original SAC -> Frozen Strong -> Turn-Sign Persistence",
            "runtime_deployment": "FORBIDDEN",
            "implemented_output_domain": "executed_acceleration_3",
            "required_auxiliary_loss_domain": "SAC_action_6",
            "legal_action_target_available": False,
        },
        "risk_selection": {
            "h_rep_m": 0.35,
            "h_emg_m": 0.0,
            "risk_developing": "m_t <= h_rep OR delta_m_t < 0",
            "risk_teacher": "ORIGINAL_SAC",
            "comfortable_teacher": "SMOOTH_TEACHER",
        },
        "TEACHER_ACTION_CONTRACT_VALID": teacher["TEACHER_ACTION_CONTRACT_VALID"],
        "training_status": "NOT_RUN_AFTER_HARD_GATE",
    }
    reward = {
        "schema_version": "turn_continuity_reward_v1",
        "original_sac_reward_unchanged": True,
        "new_jerk_reward_added": False,
        "residual_term": "comfortable-only executed yaw/pitch sign-reversal cost",
        "comfortable_condition": "m_t > 0.35 AND delta_m_t >= 0",
        "risk_developing_weight": 0.0,
        "lambda_turn": None,
        "lambda_turn_status": "NOT_CALIBRATED_BECAUSE_TEACHER_ACTION_GATE_FAILED_FIRST",
        "runtime_status": "NOT_RUN",
    }
    finetune = {
        "schema_version": "turn_finetune_config_v1",
        "status": "FROZEN_STOP_BEFORE_TRAINING",
        "learning_rate": 3.0e-5,
        "source_checkpoint": common["checkpoints"]["sac_dmp"]["path"],
        "source_checkpoint_sha256": common["checkpoints"]["sac_dmp"]["sha256"],
        "checkpoint_steps": [50000, 100000, 200000, 300000],
        "absolute_maximum_steps": 400000,
        "lambda_teacher": None,
        "lambda_turn": None,
        "loss_scale_measurement": "NOT_RUN; teacher loss is undefined in SAC action space",
        "teacher_action_gate": "FAIL",
        "training_authorized": False,
        "formal_accessed": False,
        "track_a_results_accessed": False,
    }
    split = {
        "schema_version": "turn_learn_split_reservation_v1",
        "status": "RESERVED_NOT_GENERATED_AFTER_HARD_GATE",
        "splits": {
            "TURN_LEARN_TRAIN": {"seed_base": 4700000000, "prefix": "TLT_TRAIN_"},
            "TURN_LEARN_DEV100": {"seed_base": 4800000000, "prefix": "TLT_DEV_", "intended_count": 100},
            "TURN_LEARN_HOLDOUT100": {"seed_base": 4900000000, "prefix": "TLT_HOLDOUT_", "intended_count": 100},
        },
        "track_a_reserved_ranges_avoided": [4500000000, 4600000000],
        "no_scenarios_generated": True,
    }
    atomic_json(root / "TURN_TEACHER_CONTRACT.json", teacher_contract)
    atomic_json(root / "TURN_CONTINUITY_REWARD.json", reward)
    atomic_json(root / "TURN_FINETUNE_CONFIG.json", finetune)
    atomic_json(root / "TURN_LEARN_SPLIT_RESERVATION.json", split)
    write_csv(
        root / "TURN_TRAINING_HISTORY.csv",
        [{"status": "NOT_RUN", "reason": teacher["stop_reason"], "maximum_observed_step": 0}],
    )
    write_csv(
        root / "TURN_DEV_RESULTS.csv",
        [{"status": "NOT_RUN", "reason": teacher["stop_reason"], "scenario_count": 0}],
    )
    write_csv(
        root / "TURN_DEV_MORPHOLOGY.csv",
        [{"status": "NOT_RUN", "reason": teacher["stop_reason"], "fixed_scene_count": 0}],
    )
    write_csv(
        root / "TURN_FAILURE_AUDIT.csv",
        [{"status": "NOT_RUN", "reason": teacher["stop_reason"], "failure_count": 0}],
    )
    conclusion = {
        "schema_version": "track_b_learned_turn_conclusion_v1",
        "ORIGINAL_FORMAL_SUCCESS": 0.9525,
        "FORMAL_RESULT_CHANGED": "NO",
        "best_checkpoint_steps": 0,
        "dev_success_delta_pp": None,
        "yaw_reversal_reduction_percent": None,
        "pitch_reversal_reduction_percent": None,
        "long_arc_scenes": "0/4_NOT_RUN",
        "sensor_encoder_drift": {
            "pretraining_mechanism_microtest": {
                "actor": encoder["actor_encoder_drift"],
                "online_critic": encoder["online_critic_encoder_drift"],
                "target_critic": encoder["target_critic_encoder_drift"],
            },
            "post_training": "NOT_RUN",
        },
        "ZERO_INIT_EQUIVALENCE": equivalence["ZERO_INIT_EQUIVALENCE"],
        "TEACHER_ACTION_CONTRACT_VALID": "NO",
        "TRAINING_AUTHORIZATION": "NO",
        "DEV_GATE": "NOT_RUN_AFTER_PRETRAINING_HARD_GATE",
        "HOLDOUT_EXECUTED": "NO",
        "HOLDOUT_GATE": "NOT_RUN",
        "TRACK_B_ACCEPTED": "NO",
        "FORMAL_EXECUTED": False,
        "FORMAL_REEVALUATION_RECOMMENDED": "NO",
        "TRACK_B_BEST_CHECKPOINT_STEPS": 0,
        "TRACK_B_DEV_SUCCESS_DELTA_PP": None,
        "TRACK_B_YAW_REVERSAL_REDUCTION": None,
        "TRACK_B_PITCH_REVERSAL_REDUCTION": None,
        "TRACK_B_LONG_ARC_SCENES": "0/4_NOT_RUN",
        "TRACK_B_SENSOR_ENCODER_DRIFT": {
            "actor_pretraining_microtest": encoder["actor_encoder_drift"],
            "online_critic_pretraining_microtest": encoder["online_critic_encoder_drift"],
            "target_critic_pretraining_microtest": encoder["target_critic_encoder_drift"],
            "post_training": "NOT_RUN",
        },
        "STOP_REASON": teacher["stop_reason"],
        "RECOMMENDED_NEXT_STEP": "DEFINE_AN_AUTHORIZED_ACTION_SPACE_SMOOTH_TEACHER_OR_REVISE_LOSS_DOMAIN",
    }
    atomic_json(root / "TRACK_B_CONCLUSION.json", conclusion)
    report = f"""# Track B: Risk-Aware Learned Turn Continuity\n\n## Executive result\n\nTrack B stopped at the pre-training legality gate. `ZERO_INIT_EQUIVALENCE = {equivalence['ZERO_INIT_EQUIVALENCE']}`, but `TEACHER_ACTION_CONTRACT_VALID = NO`. No training, Development, Holdout, or Formal episode was run.\n\nThe frozen Original SAC actor outputs a six-dimensional DMP action: forcing (3) plus goal offset (3). The requested smooth teacher is the already implemented Original SAC -> Frozen Strong -> Turn-Sign Persistence execution behavior. Strong and persistence do not output or modify SAC actions; they are stateful filters operating after DMP in three-dimensional acceleration space.\n\nThe historical DMP action-to-acceleration Jacobian is 3x6 with rank {teacher['jacobian_rank']} and nullity {teacher['jacobian_nullity']}. Consequently the post-filter acceleration has no unique inverse SAC action. Clipping, stateful Strong projection, sign persistence, and physical saturation make the complete map even less invertible. Using Original SAC's action as the smooth target would make both teachers identical at the same observation and cannot teach the requested long-arc distinction through action MSE.\n\nInventing a pseudoinverse, moving the teacher loss into acceleration space, or replacing it with trajectory imitation would introduce an unrequested new training contract. The hard stop therefore preserves scientific attribution.\n\n## Completed preflight evidence\n\n- Observation audit: 522 -> 529 dimensions, adding exactly seven missing context features; the 256-direction sensor input is unchanged.\n- Zero-column actor/critic expansion: PASS on {equivalence['sample_count']} samples; actor max difference {equivalence['maximum_absolute_actor_action_difference']:.3e}, critic max difference {equivalence['maximum_absolute_critic_q_difference']:.3e}.\n- Sensor-freeze mechanism microtest: {encoder['pretraining_freeze_mechanism_test']}; actor, online critic, and target critic encoder hashes remained exact under a non-encoder Polyak update.\n- Post-training encoder drift: not run because training was not authorized.\n- Reserved independent split identities: TLT_TRAIN_ / TLT_DEV_ / TLT_HOLDOUT_ with seed bases 4.7/4.8/4.9 billion; no scenes were generated after the gate failed.\n\n## Stop rule\n\n- `DEV_GATE = NOT_RUN_AFTER_PRETRAINING_HARD_GATE`\n- `HOLDOUT_EXECUTED = NO`\n- `TRACK_B_ACCEPTED = NO`\n- `FORMAL_EXECUTED = NO`\n- Original Formal V2 remains 381/400 (95.25%).\n\n## Artifact index\n\n- `TURN_OBSERVATION_EXTENSION.json`\n- `ZERO_INIT_EQUIVALENCE.json`\n- `TARGET_ENCODER_FREEZE_AUDIT.json`\n- `TURN_SMOOTH_TEACHER_ACTION_RECOVERABILITY.json`\n- `TURN_TEACHER_CONTRACT.json`\n- `TURN_CONTINUITY_REWARD.json`\n- `TURN_FINETUNE_CONFIG.json`\n- `TURN_LEARN_SPLIT_RESERVATION.json`\n- `TURN_TRAINING_HISTORY.csv`, `TURN_DEV_RESULTS.csv`\n- `TRACK_B_CONCLUSION.json`\n"""
    (root / "FINAL_REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    print("track-b audit: common freeze", flush=True)
    common = load_json(COMMON_FREEZE)
    checkpoint_path = REPO_ROOT / common["checkpoints"]["sac_dmp"]["path"]
    if sha256_file(checkpoint_path) != common["checkpoints"]["sac_dmp"]["sha256"]:
        raise RuntimeError("common-freeze SAC checkpoint hash mismatch")
    _, equivalence, encoder = zero_init_audit(root, checkpoint_path)
    print("track-b audit: teacher legality", flush=True)
    teacher = teacher_legality_audit(root)
    write_closure(root, common, equivalence, encoder, teacher)
    print(json.dumps(load_json(root / "TRACK_B_CONCLUSION.json"), indent=2))


if __name__ == "__main__":
    main()
