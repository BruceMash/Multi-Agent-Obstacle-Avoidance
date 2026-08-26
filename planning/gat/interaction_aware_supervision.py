"""Deterministic supervision-only risk-consistency projection for GAT V1.

The module consumes only the frozen V1 soft target and risk descriptors already
stored in the V1 graph.  It does not alter graph features, null semantics,
model architecture, loss, or inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


SAFE = "SAFE"
RISKY = "RISKY"
NEUTRAL = "NEUTRAL"


@dataclass(frozen=True)
class CandidateRiskDescriptor:
    proposal_node: int
    candidate_id: int
    status: str
    d_min: float | None
    t_risk: float | None
    edge_count: int


@dataclass(frozen=True)
class ProjectionResult:
    target: tuple[float, ...]
    proposal_conditional: tuple[float, ...]
    original_proposal_conditional: tuple[float, ...]
    changed: bool
    threshold: float | None
    safe_count: int
    risky_count: int
    neutral_count: int
    active_safe_count: int
    active_risky_count: int
    squared_l2_shift: float


def candidate_risk_descriptors(graph: Any) -> tuple[CandidateRiskDescriptor, ...]:
    """Recover per-candidate SAFE/RISKY/NEUTRAL state from existing ST edges."""

    candidate_ids = np.asarray(graph["proposal"].candidate_id.detach().cpu(), dtype=int)
    proposal_count = int(candidate_ids.size)
    d_safe = float(graph.graph_metadata["d_safe"])
    edge_store = graph["align", "spatiotemporal", "proposal"]
    edge_index = np.asarray(edge_store.edge_index.detach().cpu(), dtype=int)
    edge_attr = np.asarray(edge_store.edge_attr.detach().cpu(), dtype=float)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("spatiotemporal edge_index must have shape [2, E]")
    if edge_attr.ndim != 2 or edge_attr.shape[0] != edge_index.shape[1] or edge_attr.shape[1] < 3:
        raise ValueError("spatiotemporal edge_attr must have shape [E, >=3]")

    records: list[CandidateRiskDescriptor] = []
    for proposal_node, candidate_id in enumerate(candidate_ids.tolist()):
        mask = edge_index[1] == proposal_node
        values = edge_attr[mask]
        valid = (
            values.shape[0] > 0
            and np.isfinite(values[:, 1]).all()
            and np.isfinite(values[:, 2]).all()
        )
        if not valid:
            records.append(
                CandidateRiskDescriptor(
                    proposal_node=proposal_node,
                    candidate_id=int(candidate_id),
                    status=NEUTRAL,
                    d_min=None,
                    t_risk=None,
                    edge_count=int(values.shape[0]),
                )
            )
            continue
        d_min = float(np.min(values[:, 1]))
        t_risk = float(np.max(values[:, 2]))
        status = RISKY if (d_min < d_safe or t_risk > 0.0) else SAFE
        records.append(
            CandidateRiskDescriptor(
                proposal_node=proposal_node,
                candidate_id=int(candidate_id),
                status=status,
                d_min=d_min,
                t_risk=t_risk,
                edge_count=int(values.shape[0]),
            )
        )
    if len({item.candidate_id for item in records}) != proposal_count:
        raise ValueError("candidate_id values must be unique within a graph")
    return tuple(records)


def minimum_l2_risk_consistency_projection(
    proposal_conditional: Sequence[float],
    statuses: Sequence[str],
    candidate_ids: Sequence[int],
) -> tuple[np.ndarray, float | None, int, int]:
    """Project q onto ``q_safe >= q_risky`` using the exact Euclidean solution.

    Neutral coordinates remain fixed.  Violating low-safe and high-risk values
    are pooled at their common mean.  Canonical candidate-id tie-breaking makes
    the implementation independent of input proposal order.
    """

    q = np.asarray(proposal_conditional, dtype=float)
    status = np.asarray(tuple(statuses), dtype=object)
    ids = np.asarray(candidate_ids, dtype=int)
    if q.ndim != 1 or status.shape != q.shape or ids.shape != q.shape:
        raise ValueError("q, statuses, and candidate_ids must be one-dimensional and aligned")
    if not np.isfinite(q).all() or np.any(q < 0.0) or not np.isclose(q.sum(), 1.0, atol=1e-12):
        raise ValueError("proposal conditional must be a finite probability vector")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("candidate_ids must be unique")
    if not set(status.tolist()).issubset({SAFE, RISKY, NEUTRAL}):
        raise ValueError("unknown proposal risk status")

    safe = np.flatnonzero(status == SAFE)
    risky = np.flatnonzero(status == RISKY)
    if safe.size == 0 or risky.size == 0:
        return q.copy(), None, 0, 0
    if float(np.min(q[safe])) >= float(np.max(q[risky])):
        return q.copy(), None, 0, 0

    safe_order = safe[np.lexsort((ids[safe], q[safe]))]
    risky_order = risky[np.lexsort((ids[risky], -q[risky]))]
    safe_values = q[safe_order]
    risky_values = q[risky_order]
    tolerance = 1e-15
    solution: tuple[float, int, int] | None = None
    for safe_count in range(1, len(safe_values) + 1):
        for risky_count in range(1, len(risky_values) + 1):
            threshold = float(
                (
                    safe_values[:safe_count].sum()
                    + risky_values[:risky_count].sum()
                )
                / (safe_count + risky_count)
            )
            active_safe_ok = bool(np.all(safe_values[:safe_count] <= threshold + tolerance))
            inactive_safe_ok = safe_count == len(safe_values) or safe_values[safe_count] >= threshold - tolerance
            active_risky_ok = bool(np.all(risky_values[:risky_count] >= threshold - tolerance))
            inactive_risky_ok = risky_count == len(risky_values) or risky_values[risky_count] <= threshold + tolerance
            if active_safe_ok and inactive_safe_ok and active_risky_ok and inactive_risky_ok:
                solution = (threshold, safe_count, risky_count)
                break
        if solution is not None:
            break
    if solution is None:
        raise RuntimeError("failed to solve the SAFE/RISKY Euclidean projection")

    threshold, safe_count, risky_count = solution
    projected = q.copy()
    projected[safe_order[:safe_count]] = threshold
    projected[risky_order[:risky_count]] = threshold
    if not np.isclose(projected.sum(), q.sum(), atol=1e-12):
        raise AssertionError("projection changed proposal probability mass")
    if float(np.min(projected[safe])) + 1e-12 < float(np.max(projected[risky])):
        raise AssertionError("projection did not satisfy SAFE >= RISKY")
    return projected, threshold, safe_count, risky_count


def project_v1_soft_target(
    soft_target: Sequence[float],
    descriptors: Sequence[CandidateRiskDescriptor],
) -> ProjectionResult:
    """Project proposal probabilities while preserving the V1 null mass exactly."""

    original = np.asarray(soft_target, dtype=float)
    if original.ndim != 1 or len(original) != len(descriptors) + 1:
        raise ValueError("soft target must contain null plus one class per descriptor")
    if not np.isfinite(original).all() or np.any(original < 0.0) or not np.isclose(original.sum(), 1.0, atol=1e-12):
        raise ValueError("soft target must be a finite probability vector")
    statuses = tuple(item.status for item in descriptors)
    safe_count = statuses.count(SAFE)
    risky_count = statuses.count(RISKY)
    neutral_count = statuses.count(NEUTRAL)
    proposal_mass = float(1.0 - original[0])
    unchanged = safe_count == 0 or risky_count == 0 or proposal_mass <= 0.0
    if unchanged:
        conditional = original[1:] / proposal_mass if proposal_mass > 0.0 else np.zeros(len(descriptors))
        return ProjectionResult(
            target=tuple(float(value) for value in original),
            proposal_conditional=tuple(float(value) for value in conditional),
            original_proposal_conditional=tuple(float(value) for value in conditional),
            changed=False,
            threshold=None,
            safe_count=safe_count,
            risky_count=risky_count,
            neutral_count=neutral_count,
            active_safe_count=0,
            active_risky_count=0,
            squared_l2_shift=0.0,
        )

    conditional = original[1:] / proposal_mass
    projected, threshold, active_safe, active_risky = minimum_l2_risk_consistency_projection(
        conditional,
        statuses,
        [item.candidate_id for item in descriptors],
    )
    if np.array_equal(projected, conditional):
        return ProjectionResult(
            target=tuple(float(value) for value in original),
            proposal_conditional=tuple(float(value) for value in conditional),
            original_proposal_conditional=tuple(float(value) for value in conditional),
            changed=False,
            threshold=None,
            safe_count=safe_count,
            risky_count=risky_count,
            neutral_count=neutral_count,
            active_safe_count=0,
            active_risky_count=0,
            squared_l2_shift=0.0,
        )
    target = original.copy()
    target[0] = original[0]
    changed_proposals = projected != conditional
    target[1:][changed_proposals] = projected[changed_proposals] * proposal_mass
    mass_residual = 1.0 - float(target.sum())
    if abs(mass_residual) > 1e-12:
        raise AssertionError("projected target lost probability mass")
    if target.size > 1 and mass_residual != 0.0:
        # The correction is below roundoff and touches proposals only; class 0
        # remains bit-exact to the source target.
        target[1] += mass_residual
    return ProjectionResult(
        target=tuple(float(value) for value in target),
        proposal_conditional=tuple(float(value) for value in projected),
        original_proposal_conditional=tuple(float(value) for value in conditional),
        changed=not np.array_equal(target, original),
        threshold=threshold,
        safe_count=safe_count,
        risky_count=risky_count,
        neutral_count=neutral_count,
        active_safe_count=active_safe,
        active_risky_count=active_risky,
        squared_l2_shift=float(np.sum(np.square(projected - conditional))),
    )
