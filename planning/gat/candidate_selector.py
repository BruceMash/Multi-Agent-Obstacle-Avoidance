"""Pure forward selector for null and proposal candidate classes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.utils import softmax as pyg_softmax

from planning.gat.edge_enhanced_gat import (
    EdgeEnhancedGATLayer,
    RelationAttentionDebug,
)
from planning.gat.typed_encoders import (
    EDGE_INPUT_DIMS,
    NODE_INPUT_DIMS,
    TypeSpecificEdgeEncoders,
    TypeSpecificNodeEncoders,
    build_activation,
)


# These graph-global fields are Python/debug representations, not network
# tensors.  Their authoritative batched equivalents already exist on proposal,
# align, null, and agent node stores.
BATCH_EXCLUDED_GLOBAL_KEYS = (
    "graph_metadata",
    "proposal_node_to_candidate_id",
    "proposal_node_to_original_index",
    "candidate_id_to_proposal_node",
    "original_proposals",
    "neighbor_node_to_agent_id",
    "candidate_preview_positions",
    "candidate_preview_velocities",
)


def batch_candidate_graphs(graphs: list[Any] | tuple[Any, ...]) -> Batch:
    """Batch network tensors while excluding non-collatable debug metadata."""

    if not graphs:
        raise ValueError("at least one candidate graph is required")
    batch = Batch.from_data_list(
        list(graphs),
        exclude_keys=list(BATCH_EXCLUDED_GLOBAL_KEYS),
    )
    batch.input_graph_metadata = tuple(
        getattr(graph, "graph_metadata", None) for graph in graphs
    )
    return batch


@dataclass(frozen=True)
class EdgeEnhancedGATConfig:
    hidden_dim: int = 64
    node_encoder_hidden_dim: int = 64
    edge_dim: int = 32
    edge_encoder_hidden_dim: int = 32
    num_heads: int = 4
    num_layers: int = 2
    head_aggregation: str = "concat"
    activation: str = "gelu"
    dropout: float = 0.0
    negative_slope: float = 0.2
    layer_norm: bool = True

    def __post_init__(self) -> None:
        positive_ints = (
            self.hidden_dim,
            self.node_encoder_hidden_dim,
            self.edge_dim,
            self.edge_encoder_hidden_dim,
            self.num_heads,
            self.num_layers,
        )
        if any(int(value) <= 0 for value in positive_ints):
            raise ValueError("all network dimensions/counts must be positive")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if float(self.negative_slope) <= 0.0:
            raise ValueError("negative_slope must be positive")
        if self.head_aggregation == "concat" and self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        build_activation(self.activation)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "EdgeEnhancedGATConfig":
        field_names = cls.__dataclass_fields__.keys()
        return cls(**{name: values[name] for name in field_names if name in values})


@dataclass(frozen=True)
class CandidateClassMapping:
    graph_index: int
    class_index: int
    kind: str
    proposal_node_index: int | None
    candidate_id: int | None
    original_index: int | None
    world_goal: tuple[float, float, float]


@dataclass(frozen=True)
class GATSelectorOutput:
    candidate_logits: torch.Tensor
    candidate_probabilities: torch.Tensor
    candidate_batch: torch.Tensor
    candidate_ptr: torch.Tensor
    selected_class_index: torch.Tensor
    selected_candidate_id: tuple[int | None, ...]
    selected_original_index: tuple[int | None, ...]
    selected_world_goal: torch.Tensor
    class_mapping: tuple[CandidateClassMapping, ...]
    proposal_hidden_states: torch.Tensor
    null_hidden_state: torch.Tensor
    node_embeddings: Mapping[str, torch.Tensor]
    edge_embeddings: Mapping[str, torch.Tensor]
    attention_debug: tuple[RelationAttentionDebug, ...]
    graph_metadata: Any

    @property
    def graph_count(self) -> int:
        return int(self.candidate_ptr.numel() - 1)

    def logits_for_graph(self, graph_index: int) -> torch.Tensor:
        start = int(self.candidate_ptr[graph_index])
        stop = int(self.candidate_ptr[graph_index + 1])
        return self.candidate_logits[start:stop]

    def probabilities_for_graph(self, graph_index: int) -> torch.Tensor:
        start = int(self.candidate_ptr[graph_index])
        stop = int(self.candidate_ptr[graph_index + 1])
        return self.candidate_probabilities[start:stop]


class OutputScoringMLP(nn.Module):
    def __init__(self, hidden_dim: int, activation: str) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            build_activation(activation),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.network(hidden).squeeze(-1)


def _node_batch(store: Any, node_count: int, device: torch.device) -> torch.Tensor:
    batch = getattr(store, "batch", None)
    if batch is None:
        return torch.zeros(node_count, dtype=torch.long, device=device)
    return batch.to(device=device, dtype=torch.long)


class PolicyPreviewEdgeEnhancedGATSelector(nn.Module):
    """Map one HeteroData graph or PyG batch to local candidate distributions."""

    def __init__(self, config: EdgeEnhancedGATConfig | None = None) -> None:
        super().__init__()
        self.config = config or EdgeEnhancedGATConfig()
        self.node_encoders = TypeSpecificNodeEncoders(
            hidden_dim=self.config.hidden_dim,
            encoder_hidden_dim=self.config.node_encoder_hidden_dim,
            activation=self.config.activation,
            dropout=self.config.dropout,
            layer_norm=self.config.layer_norm,
        )
        self.edge_encoders = TypeSpecificEdgeEncoders(
            edge_dim=self.config.edge_dim,
            encoder_hidden_dim=self.config.edge_encoder_hidden_dim,
            activation=self.config.activation,
            dropout=self.config.dropout,
            layer_norm=self.config.layer_norm,
        )
        self.gat_layers = nn.ModuleList(
            [
                EdgeEnhancedGATLayer(
                    hidden_dim=self.config.hidden_dim,
                    edge_dim=self.config.edge_dim,
                    num_heads=self.config.num_heads,
                    head_aggregation=self.config.head_aggregation,
                    activation=self.config.activation,
                    dropout=self.config.dropout,
                    negative_slope=self.config.negative_slope,
                    layer_norm=self.config.layer_norm,
                )
                for _ in range(self.config.num_layers)
            ]
        )
        self.output_scoring = OutputScoringMLP(
            self.config.hidden_dim, self.config.activation
        )

    def _validated_inputs(
        self, data: Any
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        expected_nodes = set(NODE_INPUT_DIMS)
        expected_edges = {
            ("agent", "smooth", "proposal"),
            ("align", "spatiotemporal", "proposal"),
        }
        if set(data.node_types) != expected_nodes:
            raise ValueError(f"unexpected node types: {data.node_types}")
        if set(data.edge_types) != expected_edges:
            raise ValueError(f"unexpected edge types: {data.edge_types}")
        node_values = {node_type: data[node_type].x for node_type in NODE_INPUT_DIMS}
        smooth_store = data["agent", "smooth", "proposal"]
        spatiotemporal_store = data["align", "spatiotemporal", "proposal"]
        edge_values = {
            "smooth": smooth_store.edge_attr,
            "spatiotemporal": getattr(
                spatiotemporal_store,
                "edge_attr_normalized",
                spatiotemporal_store.edge_attr,
            ),
        }
        edge_indices = {
            "smooth": smooth_store.edge_index,
            "spatiotemporal": spatiotemporal_store.edge_index,
        }
        for node_type, values in node_values.items():
            if values.ndim != 2 or values.shape[1] != NODE_INPUT_DIMS[node_type]:
                raise ValueError(f"invalid {node_type} x shape: {tuple(values.shape)}")
            if not torch.isfinite(values).all():
                raise ValueError(f"{node_type}.x must be finite")
        for relation, values in edge_values.items():
            if values.ndim != 2 or values.shape[1] != EDGE_INPUT_DIMS[relation]:
                raise ValueError(f"invalid {relation} edge shape: {tuple(values.shape)}")
            if not torch.isfinite(values).all():
                raise ValueError(f"{relation} edge input must be finite")
        return node_values, edge_values, edge_indices

    def forward(
        self,
        data: Any,
        *,
        return_attention_debug: bool = False,
    ) -> GATSelectorOutput:
        node_values, edge_values, edge_indices = self._validated_inputs(data)
        node_hidden = self.node_encoders(node_values)
        edge_hidden = self.edge_encoders(edge_values)
        device = node_hidden["null"].device
        null_batch = _node_batch(data["null"], len(node_hidden["null"]), device)
        agent_batch = _node_batch(data["agent"], len(node_hidden["agent"]), device)
        proposal_batch = _node_batch(
            data["proposal"], len(node_hidden["proposal"]), device
        )
        graph_count = int(null_batch.max().item() + 1) if null_batch.numel() else 0
        if graph_count <= 0:
            raise ValueError("selector input must contain at least one null node")
        for name, batch in (("null", null_batch), ("agent", agent_batch)):
            counts = torch.bincount(batch, minlength=graph_count)
            if counts.numel() != graph_count or not torch.equal(
                counts, torch.ones_like(counts)
            ):
                raise ValueError(f"each graph must contain exactly one {name} node")

        all_attention: list[RelationAttentionDebug] = []
        for layer_index, layer in enumerate(self.gat_layers):
            updated, debug = layer(
                node_hidden=node_hidden,
                edge_index=edge_indices,
                edge_embedding=edge_hidden,
                proposal_batch=proposal_batch,
                layer_index=layer_index,
                return_attention_debug=return_attention_debug,
            )
            node_hidden = {**node_hidden, "proposal": updated}
            all_attention.extend(debug)

        class_hidden: list[torch.Tensor] = []
        class_batch: list[torch.Tensor] = []
        mappings: list[CandidateClassMapping] = []
        ptr = [0]
        task_goals = data["agent"].task_goal
        proposal_ids = data["proposal"].candidate_id
        original_indices = data["proposal"].original_index
        world_positions = data["proposal"].world_position
        for graph_index in range(graph_count):
            null_indices = torch.nonzero(null_batch == graph_index, as_tuple=False).flatten()
            proposal_indices = torch.nonzero(
                proposal_batch == graph_index, as_tuple=False
            ).flatten()
            local_hidden = torch.cat(
                [node_hidden["null"][null_indices], node_hidden["proposal"][proposal_indices]],
                dim=0,
            )
            class_hidden.append(local_hidden)
            class_batch.append(
                torch.full(
                    (local_hidden.shape[0],),
                    graph_index,
                    dtype=torch.long,
                    device=device,
                )
            )
            goal = task_goals[torch.nonzero(agent_batch == graph_index, as_tuple=False)[0, 0]]
            mappings.append(
                CandidateClassMapping(
                    graph_index=graph_index,
                    class_index=0,
                    kind="null",
                    proposal_node_index=None,
                    candidate_id=None,
                    original_index=None,
                    world_goal=tuple(float(value) for value in goal.detach().cpu()),
                )
            )
            for class_index, proposal_node in enumerate(proposal_indices.tolist(), start=1):
                world = world_positions[proposal_node]
                mappings.append(
                    CandidateClassMapping(
                        graph_index=graph_index,
                        class_index=class_index,
                        kind="proposal",
                        proposal_node_index=int(proposal_node),
                        candidate_id=int(proposal_ids[proposal_node]),
                        original_index=int(original_indices[proposal_node]),
                        world_goal=tuple(float(value) for value in world.detach().cpu()),
                    )
                )
            ptr.append(ptr[-1] + int(local_hidden.shape[0]))

        stacked_hidden = torch.cat(class_hidden, dim=0)
        candidate_batch = torch.cat(class_batch, dim=0)
        logits = self.output_scoring(stacked_hidden)
        probabilities = pyg_softmax(
            logits, index=candidate_batch, num_nodes=graph_count
        )
        candidate_ptr = torch.as_tensor(ptr, dtype=torch.long, device=device)
        selected_local: list[int] = []
        selected_candidate_ids: list[int | None] = []
        selected_original_indices: list[int | None] = []
        selected_goals: list[torch.Tensor] = []
        mapping_by_graph = [
            [item for item in mappings if item.graph_index == graph_index]
            for graph_index in range(graph_count)
        ]
        for graph_index in range(graph_count):
            start, stop = ptr[graph_index], ptr[graph_index + 1]
            local_index = int(torch.argmax(probabilities[start:stop]).item())
            selected_local.append(local_index)
            mapping = mapping_by_graph[graph_index][local_index]
            selected_candidate_ids.append(mapping.candidate_id)
            selected_original_indices.append(mapping.original_index)
            if mapping.proposal_node_index is None:
                agent_index = torch.nonzero(
                    agent_batch == graph_index, as_tuple=False
                )[0, 0]
                selected_goals.append(task_goals[agent_index])
            else:
                selected_goals.append(world_positions[mapping.proposal_node_index])

        return GATSelectorOutput(
            candidate_logits=logits,
            candidate_probabilities=probabilities,
            candidate_batch=candidate_batch,
            candidate_ptr=candidate_ptr,
            selected_class_index=torch.as_tensor(
                selected_local, dtype=torch.long, device=device
            ),
            selected_candidate_id=tuple(selected_candidate_ids),
            selected_original_index=tuple(selected_original_indices),
            selected_world_goal=torch.stack(selected_goals, dim=0),
            class_mapping=tuple(mappings),
            proposal_hidden_states=node_hidden["proposal"],
            null_hidden_state=node_hidden["null"],
            node_embeddings=node_hidden,
            edge_embeddings=edge_hidden,
            attention_debug=tuple(all_attention),
            graph_metadata=getattr(
                data,
                "input_graph_metadata",
                getattr(data, "graph_metadata", None),
            ),
        )

    def parameter_counts(self) -> dict[str, int]:
        components = {
            "node_encoders": self.node_encoders,
            "edge_encoders": self.edge_encoders,
            "gat": self.gat_layers,
            "output_head": self.output_scoring,
        }
        counts = {
            name: int(sum(parameter.numel() for parameter in module.parameters()))
            for name, module in components.items()
        }
        counts.update(
            {
                f"node_encoder_{name}": int(
                    sum(parameter.numel() for parameter in module.parameters())
                )
                for name, module in self.node_encoders.encoders.items()
            }
        )
        counts.update(
            {
                f"edge_encoder_{name}": int(
                    sum(parameter.numel() for parameter in module.parameters())
                )
                for name, module in self.edge_encoders.encoders.items()
            }
        )
        counts["total_trainable"] = int(
            sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        )
        return counts
