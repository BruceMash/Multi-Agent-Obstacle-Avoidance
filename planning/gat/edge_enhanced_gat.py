"""Relation-aware edge-enhanced attention for proposal receiver nodes.

Unlike PyG's standard GAT layers, the encoded edge feature participates in
both the attention score and the message content.  No self-loop or reverse
relation is synthesized; proposal self-information is retained by an explicit
residual update outside the graph edge set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import softmax as pyg_softmax

from planning.gat.typed_encoders import build_activation


RELATION_SOURCES = {
    "smooth": "agent",
    "spatiotemporal": "align",
}
RELATION_ORDER = tuple(RELATION_SOURCES)


@dataclass(frozen=True)
class RelationAttentionDebug:
    layer_index: int
    relation: str
    source_node_index: torch.Tensor
    target_proposal_index: torch.Tensor
    target_graph_index: torch.Tensor
    raw_attention_logits: torch.Tensor
    normalized_alpha: torch.Tensor
    message_content: torch.Tensor


class _RelationParameters(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        edge_dim: int,
        num_heads: int,
        head_dim: int,
        negative_slope: float,
    ) -> None:
        super().__init__()
        projected_dim = int(num_heads) * int(head_dim)
        self.source_attention = nn.Linear(hidden_dim, projected_dim, bias=False)
        self.target_attention = nn.Linear(hidden_dim, projected_dim, bias=False)
        self.edge_attention = nn.Linear(edge_dim, projected_dim, bias=False)
        self.node_message = nn.Linear(hidden_dim, projected_dim, bias=False)
        self.edge_message = nn.Linear(edge_dim, projected_dim, bias=False)
        self.attention_vector = nn.Parameter(
            torch.empty(num_heads, 3 * head_dim)
        )
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.negative_slope = float(negative_slope)
        nn.init.xavier_uniform_(self.attention_vector)

    def forward(
        self,
        *,
        source_hidden: torch.Tensor,
        proposal_hidden: torch.Tensor,
        edge_index: torch.Tensor,
        edge_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        edge_count = int(edge_index.shape[1])
        if edge_embedding.ndim != 2 or edge_embedding.shape[0] != edge_count:
            raise ValueError("edge embedding and edge_index must share edge count")
        if edge_count == 0:
            empty_logits = proposal_hidden.new_empty((0, self.num_heads))
            empty_messages = proposal_hidden.new_empty(
                (0, self.num_heads, self.head_dim)
            )
            return empty_logits, empty_messages
        source_index, target_index = edge_index
        if int(source_index.min()) < 0 or int(source_index.max()) >= source_hidden.shape[0]:
            raise IndexError("relation source index is out of range")
        if int(target_index.min()) < 0 or int(target_index.max()) >= proposal_hidden.shape[0]:
            raise IndexError("relation proposal target index is out of range")
        source = source_hidden[source_index]
        target = proposal_hidden[target_index]
        source_key = self.source_attention(source).view(
            edge_count, self.num_heads, self.head_dim
        )
        target_key = self.target_attention(target).view(
            edge_count, self.num_heads, self.head_dim
        )
        edge_key = self.edge_attention(edge_embedding).view(
            edge_count, self.num_heads, self.head_dim
        )
        attention_input = torch.cat([source_key, target_key, edge_key], dim=-1)
        raw_logits = F.leaky_relu(
            (attention_input * self.attention_vector.unsqueeze(0)).sum(dim=-1),
            negative_slope=self.negative_slope,
        )
        node_message = self.node_message(source).view(
            edge_count, self.num_heads, self.head_dim
        )
        edge_message = self.edge_message(edge_embedding).view(
            edge_count, self.num_heads, self.head_dim
        )
        return raw_logits, node_message + edge_message


class EdgeEnhancedGATLayer(nn.Module):
    """Aggregate both heterogeneous relations with one target-wise softmax."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        edge_dim: int,
        num_heads: int,
        head_aggregation: str = "concat",
        activation: str = "gelu",
        dropout: float = 0.0,
        negative_slope: float = 0.2,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        num_heads = int(num_heads)
        aggregation = str(head_aggregation).lower()
        if aggregation not in {"concat", "mean"}:
            raise ValueError("head_aggregation must be 'concat' or 'mean'")
        if aggregation == "concat" and hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads for concat")
        self.head_dim = hidden_dim // num_heads if aggregation == "concat" else hidden_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_aggregation = aggregation
        self.dropout = float(dropout)
        self.relations = nn.ModuleDict(
            {
                relation: _RelationParameters(
                    hidden_dim=hidden_dim,
                    edge_dim=int(edge_dim),
                    num_heads=num_heads,
                    head_dim=self.head_dim,
                    negative_slope=negative_slope,
                )
                for relation in RELATION_ORDER
            }
        )
        self.activation = build_activation(activation)
        self.normalization = nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity()

    def forward(
        self,
        *,
        node_hidden: Mapping[str, torch.Tensor],
        edge_index: Mapping[str, torch.Tensor],
        edge_embedding: Mapping[str, torch.Tensor],
        proposal_batch: torch.Tensor,
        layer_index: int,
        return_attention_debug: bool,
    ) -> tuple[torch.Tensor, tuple[RelationAttentionDebug, ...]]:
        proposal_hidden = node_hidden["proposal"]
        proposal_count = int(proposal_hidden.shape[0])
        relation_logits: list[torch.Tensor] = []
        relation_messages: list[torch.Tensor] = []
        relation_targets: list[torch.Tensor] = []
        relation_slices: dict[str, slice] = {}
        offset = 0
        for relation in RELATION_ORDER:
            indices = edge_index[relation]
            logits, messages = self.relations[relation](
                source_hidden=node_hidden[RELATION_SOURCES[relation]],
                proposal_hidden=proposal_hidden,
                edge_index=indices,
                edge_embedding=edge_embedding[relation],
            )
            edge_count = int(indices.shape[1])
            relation_slices[relation] = slice(offset, offset + edge_count)
            offset += edge_count
            relation_logits.append(logits)
            relation_messages.append(messages)
            relation_targets.append(indices[1])

        if offset == 0:
            aggregated_heads = proposal_hidden.new_zeros(
                (proposal_count, self.num_heads, self.head_dim)
            )
            all_alpha = proposal_hidden.new_empty((0, self.num_heads))
            all_logits = proposal_hidden.new_empty((0, self.num_heads))
            all_messages = proposal_hidden.new_empty(
                (0, self.num_heads, self.head_dim)
            )
        else:
            all_logits = torch.cat(relation_logits, dim=0)
            all_messages = torch.cat(relation_messages, dim=0)
            all_targets = torch.cat(relation_targets, dim=0)
            all_alpha = pyg_softmax(
                all_logits,
                index=all_targets,
                num_nodes=proposal_count,
            )
            applied_alpha = F.dropout(
                all_alpha,
                p=self.dropout,
                training=self.training,
            )
            weighted = all_messages * applied_alpha.unsqueeze(-1)
            aggregated_heads = proposal_hidden.new_zeros(
                (proposal_count, self.num_heads, self.head_dim)
            ).index_add(0, all_targets, weighted)

        if self.head_aggregation == "concat":
            aggregated = aggregated_heads.reshape(proposal_count, self.hidden_dim)
        else:
            aggregated = aggregated_heads.mean(dim=1)
        updated = self.normalization(
            proposal_hidden
            + F.dropout(
                self.activation(aggregated),
                p=self.dropout,
                training=self.training,
            )
        )

        debug: list[RelationAttentionDebug] = []
        if return_attention_debug:
            for relation in RELATION_ORDER:
                indices = edge_index[relation]
                part = relation_slices[relation]
                targets = indices[1]
                target_graph = (
                    proposal_batch[targets]
                    if targets.numel()
                    else proposal_batch.new_empty((0,))
                )
                debug.append(
                    RelationAttentionDebug(
                        layer_index=int(layer_index),
                        relation=relation,
                        source_node_index=indices[0],
                        target_proposal_index=targets,
                        target_graph_index=target_graph,
                        raw_attention_logits=all_logits[part],
                        normalized_alpha=all_alpha[part],
                        message_content=all_messages[part],
                    )
                )
        return updated, tuple(debug)
