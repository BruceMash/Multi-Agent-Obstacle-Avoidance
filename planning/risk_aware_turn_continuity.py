"""Auditable primitives for the Track-B learned-turn preflight.

This module intentionally stops short of defining a new teacher inverse.  The
existing smooth teacher operates in executed-acceleration space while SAC
operates in a six-dimensional DMP-action space.  Utilities here cover only the
authorized zero-column expansion and a sensor-excluding target update.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch


SENSOR_OBSERVATION_DIM = 519
OLD_EXTRA_OBSERVATION_DIM = 3
NEW_CONTEXT_DIM = 7
NEW_EXTRA_OBSERVATION_DIM = OLD_EXTRA_OBSERVATION_DIM + NEW_CONTEXT_DIM
ACTION_DIM = 6
SENSOR_FEATURE_DIM = 128
SENSOR_ENCODER_PREFIX = "sensor_encoder."


def tensor_group_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash a tensor mapping with names, shapes, dtypes, and exact bytes."""

    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def sensor_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return an exact detached copy of one module's sensor encoder."""

    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in module.state_dict().items()
        if name.startswith(SENSOR_ENCODER_PREFIX)
    }


def expand_checkpoint_extra_columns(
    module: torch.nn.Module,
    source: Mapping[str, torch.Tensor],
    *,
    module_name: str,
) -> dict[str, Any]:
    """Copy a 522-D checkpoint into a 529-D model with seven zero columns."""

    target = module.state_dict()
    expanded: dict[str, torch.Tensor] = {}
    zero_columns: dict[str, bool] = {}
    copied_exact: list[str] = []
    for name, target_tensor in target.items():
        source_tensor = source[name]
        if tuple(target_tensor.shape) == tuple(source_tensor.shape):
            expanded[name] = source_tensor.detach().clone()
            copied_exact.append(name)
            continue
        result = torch.zeros_like(target_tensor)
        if module_name == "actor" and name == "layers.0.weight":
            expected_old = (256, SENSOR_FEATURE_DIM + OLD_EXTRA_OBSERVATION_DIM)
            expected_new = (256, SENSOR_FEATURE_DIM + NEW_EXTRA_OBSERVATION_DIM)
            if tuple(source_tensor.shape) != expected_old or tuple(target_tensor.shape) != expected_new:
                raise RuntimeError(f"unexpected actor expansion {source_tensor.shape} -> {target_tensor.shape}")
            result[:, : expected_old[1]] = source_tensor
            zero_columns[f"{module_name}.{name}"] = bool(
                torch.count_nonzero(result[:, expected_old[1] : expected_new[1]]) == 0
            )
        elif module_name in {"critic", "critic_target"} and name in {
            "q1_net.layers.0.weight",
            "q2_net.layers.0.weight",
        }:
            old_feature = SENSOR_FEATURE_DIM + OLD_EXTRA_OBSERVATION_DIM
            new_feature = SENSOR_FEATURE_DIM + NEW_EXTRA_OBSERVATION_DIM
            expected_old = (256, old_feature + ACTION_DIM)
            expected_new = (256, new_feature + ACTION_DIM)
            if tuple(source_tensor.shape) != expected_old or tuple(target_tensor.shape) != expected_new:
                raise RuntimeError(f"unexpected critic expansion {source_tensor.shape} -> {target_tensor.shape}")
            result[:, :old_feature] = source_tensor[:, :old_feature]
            result[:, new_feature : new_feature + ACTION_DIM] = source_tensor[
                :, old_feature : old_feature + ACTION_DIM
            ]
            zero_columns[f"{module_name}.{name}"] = bool(
                torch.count_nonzero(result[:, old_feature:new_feature]) == 0
            )
        else:
            raise RuntimeError(
                f"unauthorized shape change {module_name}.{name}: "
                f"{tuple(source_tensor.shape)} -> {tuple(target_tensor.shape)}"
            )
        expanded[name] = result
    module.load_state_dict(expanded, strict=True)
    return {
        "zero_new_context_columns": zero_columns,
        "all_new_context_columns_zero": bool(zero_columns and all(zero_columns.values())),
        "copied_exact_tensor_count": len(copied_exact),
    }


def freeze_sensor_encoders(actor: Any, critic: Any, critic_target: Any) -> dict[str, Any]:
    """Disable gradients for all three encoder copies."""

    result: dict[str, Any] = {}
    for label, module in (
        ("actor", actor),
        ("critic", critic),
        ("critic_target", critic_target),
    ):
        names: list[str] = []
        for name, parameter in module.named_parameters():
            if name.startswith(SENSOR_ENCODER_PREFIX):
                parameter.requires_grad_(False)
                names.append(name)
        result[label] = {
            "frozen_parameter_names": names,
            "frozen_parameter_count": int(
                sum(
                    parameter.numel()
                    for name, parameter in module.named_parameters()
                    if name.startswith(SENSOR_ENCODER_PREFIX)
                )
            ),
            "all_require_grad_false": bool(
                names
                and all(
                    not parameter.requires_grad
                    for name, parameter in module.named_parameters()
                    if name.startswith(SENSOR_ENCODER_PREFIX)
                )
            ),
        }
    return result


@torch.no_grad()
def polyak_update_excluding_sensor_encoder(
    critic: torch.nn.Module,
    critic_target: torch.nn.Module,
    tau: float,
) -> None:
    """Polyak-update matching named tensors while keeping target sensor exact."""

    tau = float(tau)
    if not 0.0 <= tau <= 1.0:
        raise ValueError("tau must lie in [0,1]")
    source = dict(critic.named_parameters())
    target = dict(critic_target.named_parameters())
    if source.keys() != target.keys():
        raise RuntimeError("critic and target parameter names differ")
    for name in source:
        if name.startswith(SENSOR_ENCODER_PREFIX):
            continue
        target[name].mul_(1.0 - tau).add_(source[name], alpha=tau)


def dmp_action_jacobian(*, k_alpha: float, k_beta: float, tau: float, forcing_gate: np.ndarray) -> np.ndarray:
    """Jacobian of unclipped historical DMP acceleration with respect to action."""

    gate = np.asarray(forcing_gate, dtype=float)
    if gate.shape != (3,) or not np.all(np.isfinite(gate)):
        raise ValueError("forcing_gate must be a finite 3-vector")
    jacobian = np.zeros((3, 6), dtype=float)
    jacobian[:, :3] = np.diag(gate / float(tau) ** 2)
    jacobian[:, 3:] = np.eye(3) * float(k_alpha) * float(k_beta) / float(tau) ** 2
    return jacobian


__all__ = [
    "ACTION_DIM",
    "NEW_CONTEXT_DIM",
    "NEW_EXTRA_OBSERVATION_DIM",
    "OLD_EXTRA_OBSERVATION_DIM",
    "SENSOR_OBSERVATION_DIM",
    "dmp_action_jacobian",
    "expand_checkpoint_extra_columns",
    "freeze_sensor_encoders",
    "polyak_update_excluding_sensor_encoder",
    "sensor_state",
    "tensor_group_sha256",
]
