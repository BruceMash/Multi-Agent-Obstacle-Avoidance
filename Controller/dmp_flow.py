from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - NumPy-only deployments remain supported.
    torch = None


def _is_torch_tensor(value: Any) -> bool:
    return torch is not None and isinstance(value, torch.Tensor)


def compute_dmp_drives(
    goal_delta,
    velocity,
    residual_forcing,
    *,
    k_alpha: float,
    k_beta: float,
    tau: float,
):
    """Return the numerator-level nominal, residual, and closed-loop DMP drives.

    The controller divides all three terms by ``tau ** 2``. That common positive
    scale does not affect cosine consistency, so keeping numerator-level drives
    avoids unnecessary divisions while exactly preserving the executed flow
    direction. The residual includes the controller's per-axis distance gate.
    """

    if _is_torch_tensor(goal_delta):
        if not (_is_torch_tensor(velocity) and _is_torch_tensor(residual_forcing)):
            raise TypeError("all DMP drive inputs must use the same tensor backend")
        nominal = float(k_alpha) * (
            float(k_beta) * goal_delta - float(tau) * velocity
        )
        gated_residual = residual_forcing * torch.tanh(torch.abs(goal_delta))
    else:
        goal_delta = np.asarray(goal_delta)
        velocity = np.asarray(velocity)
        residual_forcing = np.asarray(residual_forcing)
        nominal = float(k_alpha) * (
            float(k_beta) * goal_delta - float(tau) * velocity
        )
        gated_residual = residual_forcing * np.tanh(np.abs(goal_delta))

    if nominal.shape != gated_residual.shape:
        raise ValueError(
            "nominal and residual DMP drives must have identical shapes, "
            f"got {nominal.shape} and {gated_residual.shape}"
        )
    return nominal, gated_residual, nominal + gated_residual


def compute_dmp_flow_consistency(
    nominal_drive,
    closed_loop_drive,
    *,
    zero_threshold: float = 1.0e-4,
    epsilon: float = 1.0e-8,
):
    """Compute last-axis cosine consistency with stable equilibrium handling.

    Inputs may have shape ``[dims]``, ``[batch, dims]`` or
    ``[batch, agents, dims]``. PyTorch inputs preserve gradients with respect to
    ``closed_loop_drive`` and support CPU/CUDA; NumPy inputs are used by the
    environment controller.
    """

    zero_threshold = float(zero_threshold)
    epsilon = float(epsilon)
    if zero_threshold < 0.0:
        raise ValueError("zero_threshold must be non-negative")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")

    if _is_torch_tensor(nominal_drive):
        if not _is_torch_tensor(closed_loop_drive):
            raise TypeError("both flow-consistency inputs must be torch tensors")
        if nominal_drive.shape != closed_loop_drive.shape:
            raise ValueError("flow-consistency inputs must have identical shapes")
        if nominal_drive.ndim < 1:
            raise ValueError("flow-consistency inputs must include a vector dimension")

        nominal_norm = torch.linalg.vector_norm(nominal_drive, dim=-1)
        closed_norm = torch.linalg.vector_norm(closed_loop_drive, dim=-1)
        dot = torch.sum(nominal_drive * closed_loop_drive, dim=-1)
        cosine = dot / (nominal_norm * closed_norm + epsilon)
        cosine = cosine.clamp(-1.0, 1.0)
        nominal_zero = nominal_norm < zero_threshold
        closed_zero = closed_norm < zero_threshold
        consistency = torch.where(nominal_zero, torch.zeros_like(cosine), cosine)
        return torch.where(
            nominal_zero & closed_zero,
            torch.ones_like(consistency),
            consistency,
        )

    nominal_array = np.asarray(nominal_drive)
    closed_array = np.asarray(closed_loop_drive)
    if nominal_array.shape != closed_array.shape:
        raise ValueError("flow-consistency inputs must have identical shapes")
    if nominal_array.ndim < 1:
        raise ValueError("flow-consistency inputs must include a vector dimension")

    nominal_norm = np.linalg.norm(nominal_array, axis=-1)
    closed_norm = np.linalg.norm(closed_array, axis=-1)
    dot = np.sum(nominal_array * closed_array, axis=-1)
    cosine = np.clip(
        dot / (nominal_norm * closed_norm + epsilon),
        -1.0,
        1.0,
    )
    nominal_zero = nominal_norm < zero_threshold
    closed_zero = closed_norm < zero_threshold
    consistency = np.where(nominal_zero, 0.0, cosine)
    return np.where(nominal_zero & closed_zero, 1.0, consistency)


def update_dmp_phase(
    phase,
    consistency,
    *,
    alpha_s: float,
    dt: float,
    tau: float,
    phase_min: float,
    phase_mode: str,
    phase_integrator: str,
):
    """Advance classic or FCEP phase without cross-step autograd state."""

    phase_mode = str(phase_mode).lower()
    phase_integrator = str(phase_integrator).lower()
    if phase_mode not in {"classic", "fcep"}:
        raise ValueError("phase_mode must be 'classic' or 'fcep'")
    if phase_integrator not in {"legacy_euler", "exponential"}:
        raise ValueError("phase_integrator must be 'legacy_euler' or 'exponential'")
    if float(alpha_s) < 0.0 or float(dt) <= 0.0 or float(tau) <= 0.0:
        raise ValueError("alpha_s must be non-negative and dt/tau must be positive")
    if not 0.0 <= float(phase_min) <= 1.0:
        raise ValueError("phase_min must lie in [0, 1]")

    if _is_torch_tensor(phase):
        phase_value = phase.detach()
        if phase_mode == "classic":
            phase_rate = torch.ones_like(phase_value)
        else:
            consistency_tensor = torch.as_tensor(
                consistency,
                device=phase_value.device,
                dtype=phase_value.dtype,
            ).detach()
            phase_rate = torch.relu(consistency_tensor)
        decay = float(alpha_s) * phase_rate * float(dt) / float(tau)
        if phase_integrator == "exponential":
            next_phase = phase_value * torch.exp(-decay)
        else:
            next_phase = phase_value * (1.0 - decay)
        return next_phase.clamp(float(phase_min), 1.0).detach(), phase_rate.detach()

    phase_value = np.asarray(phase)
    if phase_mode == "classic":
        phase_rate = np.ones_like(phase_value)
    else:
        phase_rate = np.maximum(np.asarray(consistency), 0.0)
    decay = float(alpha_s) * phase_rate * float(dt) / float(tau)
    if phase_integrator == "exponential":
        next_phase = phase_value * np.exp(-decay)
    else:
        next_phase = phase_value * (1.0 - decay)
    return np.clip(next_phase, float(phase_min), 1.0), phase_rate
