"""Type-specific finite feature encoders for the candidate graph."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn


NODE_INPUT_DIMS = {
    "null": 5,
    "agent": 66,
    "proposal": 13,
    "align": 7,
}
EDGE_INPUT_DIMS = {
    "smooth": 1,
    "spatiotemporal": 3,
}


def build_activation(name: str) -> nn.Module:
    normalized = str(name).strip().lower()
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "elu":
        return nn.ELU()
    raise ValueError(f"unsupported activation: {name}")


class _TypedMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        activation: str,
        dropout: float,
        layer_norm: bool,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(int(input_dim), int(hidden_dim)),
            build_activation(activation),
        ]
        if float(dropout) > 0.0:
            layers.append(nn.Dropout(float(dropout)))
        layers.append(nn.Linear(int(hidden_dim), int(output_dim)))
        if layer_norm:
            layers.append(nn.LayerNorm(int(output_dim)))
        self.network = nn.Sequential(*layers)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2 or values.shape[1] != self.input_dim:
            raise ValueError(
                f"encoder expected [N, {self.input_dim}], got {tuple(values.shape)}"
            )
        if not values.is_floating_point() or not torch.isfinite(values).all():
            raise ValueError("encoder inputs must be finite floating-point tensors")
        return self.network(values)


class NullNodeEncoder(_TypedMLP):
    pass


class AgentNodeEncoder(_TypedMLP):
    pass


class ProposalNodeEncoder(_TypedMLP):
    pass


class AlignNodeEncoder(_TypedMLP):
    pass


class SmoothEdgeEncoder(_TypedMLP):
    pass


class SpatiotemporalEdgeEncoder(_TypedMLP):
    pass


class TypeSpecificNodeEncoders(nn.Module):
    """Independent MLPs for the four physically distinct node types."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        encoder_hidden_dim: int,
        activation: str,
        dropout: float,
        layer_norm: bool,
    ) -> None:
        super().__init__()
        kwargs = {
            "hidden_dim": int(encoder_hidden_dim),
            "output_dim": int(hidden_dim),
            "activation": activation,
            "dropout": float(dropout),
            "layer_norm": bool(layer_norm),
        }
        self.encoders = nn.ModuleDict(
            {
                "null": NullNodeEncoder(NODE_INPUT_DIMS["null"], **kwargs),
                "agent": AgentNodeEncoder(NODE_INPUT_DIMS["agent"], **kwargs),
                "proposal": ProposalNodeEncoder(
                    NODE_INPUT_DIMS["proposal"], **kwargs
                ),
                "align": AlignNodeEncoder(NODE_INPUT_DIMS["align"], **kwargs),
            }
        )

    def forward(self, values: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        missing = set(NODE_INPUT_DIMS) - set(values)
        if missing:
            raise ValueError(f"missing node features for: {sorted(missing)}")
        return {
            node_type: encoder(values[node_type])
            for node_type, encoder in self.encoders.items()
        }


class TypeSpecificEdgeEncoders(nn.Module):
    """Independent MLPs for smooth and spatiotemporal edge semantics."""

    def __init__(
        self,
        *,
        edge_dim: int,
        encoder_hidden_dim: int,
        activation: str,
        dropout: float,
        layer_norm: bool,
    ) -> None:
        super().__init__()
        kwargs = {
            "hidden_dim": int(encoder_hidden_dim),
            "output_dim": int(edge_dim),
            "activation": activation,
            "dropout": float(dropout),
            "layer_norm": bool(layer_norm),
        }
        self.encoders = nn.ModuleDict(
            {
                "smooth": SmoothEdgeEncoder(EDGE_INPUT_DIMS["smooth"], **kwargs),
                "spatiotemporal": SpatiotemporalEdgeEncoder(
                    EDGE_INPUT_DIMS["spatiotemporal"], **kwargs
                ),
            }
        )

    def forward(self, values: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        missing = set(EDGE_INPUT_DIMS) - set(values)
        if missing:
            raise ValueError(f"missing edge features for: {sorted(missing)}")
        return {
            edge_type: encoder(values[edge_type])
            for edge_type, encoder in self.encoders.items()
        }
