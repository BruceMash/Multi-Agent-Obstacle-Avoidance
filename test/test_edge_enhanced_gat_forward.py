from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from planning.gat import (
    EdgeEnhancedGATConfig,
    EdgeEnhancedGATLayer,
    PolicyPreviewEdgeEnhancedGATSelector,
    batch_candidate_graphs,
)
from planning.gat.typed_encoders import NODE_INPUT_DIMS
from test.test_heterogeneous_candidate_graph import (
    _build,
    _execution,
    _proposal,
    NeighborGraphState,
)


def _model(*, num_layers: int = 2) -> PolicyPreviewEdgeEnhancedGATSelector:
    torch.manual_seed(20260813)
    model = PolicyPreviewEdgeEnhancedGATSelector(
        EdgeEnhancedGATConfig(
            hidden_dim=32,
            node_encoder_hidden_dim=24,
            edge_dim=16,
            edge_encoder_hidden_dim=12,
            num_heads=4,
            num_layers=num_layers,
            dropout=0.0,
        )
    )
    model.eval()
    return model


def _graph(
    candidate_count: int,
    neighbor_count: int,
    *,
    conflict: bool = False,
):
    proposals = [
        _proposal(index, (1.0 + 0.1 * index, 0.05 * index, 0.0))
        for index in range(candidate_count)
    ]
    trajectory = np.asarray([[1.0, 0.0, 0.0]] * 4)
    executions = [
        _execution(
            index,
            tuple(proposal.point),
            trajectory=trajectory if conflict else None,
        )
        for index, proposal in enumerate(proposals)
    ]
    if conflict:
        offsets = [0.2, -0.35, 0.5]
        neighbors = [
            NeighborGraphState(
                index + 1,
                [1.0, offsets[index], 0.0],
                [0.0, 0.0, 0.0],
            )
            for index in range(neighbor_count)
        ]
    else:
        neighbors = [
            NeighborGraphState(index + 1, [5.0 + index, 3.0, 0.0], [0.0, 0.0, 0.0])
            for index in range(neighbor_count)
        ]
    return _build(proposals, executions, neighbors=neighbors)


@pytest.mark.parametrize("candidate_count", [0, 1, 2, 6, 10])
@pytest.mark.parametrize("neighbor_count", [0, 1, 3])
def test_variable_graph_sizes_forward_are_finite(candidate_count, neighbor_count):
    graph = _graph(candidate_count, neighbor_count)
    output = _model()(graph, return_attention_debug=True)
    assert output.candidate_logits.shape == (candidate_count + 1,)
    assert output.candidate_probabilities.shape == (candidate_count + 1,)
    assert torch.isfinite(output.candidate_logits).all()
    assert torch.isfinite(output.candidate_probabilities).all()
    assert output.candidate_probabilities.sum().item() == pytest.approx(1.0)
    assert output.proposal_hidden_states.shape == (candidate_count, 32)
    assert output.null_hidden_state.shape == (1, 32)
    for values in output.node_embeddings.values():
        assert torch.isfinite(values).all()
    for values in output.edge_embeddings.values():
        assert torch.isfinite(values).all()
    for debug in output.attention_debug:
        assert torch.isfinite(debug.raw_attention_logits).all()
        assert torch.isfinite(debug.normalized_alpha).all()
        assert torch.isfinite(debug.message_content).all()


def test_zero_proposal_outputs_only_null_with_terminal_goal():
    graph = _graph(0, 2)
    output = _model()(graph)
    assert output.candidate_ptr.tolist() == [0, 1]
    assert output.candidate_probabilities.tolist() == [1.0]
    assert output.selected_class_index.tolist() == [0]
    assert output.selected_candidate_id == (None,)
    assert output.class_mapping[0].kind == "null"
    torch.testing.assert_close(
        output.selected_world_goal[0], graph["agent"].task_goal[0]
    )


def test_type_specific_node_features_reach_the_expected_outputs():
    graph = _graph(1, 1, conflict=True)
    model = _model(num_layers=1)
    baseline = model(graph)
    effects = {}
    for node_type in NODE_INPUT_DIMS:
        changed = copy.deepcopy(graph)
        changed[node_type].x = changed[node_type].x.clone()
        changed[node_type].x[0, 0] += 0.37
        result = model(changed)
        if node_type == "null":
            effects[node_type] = not torch.allclose(
                baseline.candidate_logits[0], result.candidate_logits[0]
            )
        else:
            effects[node_type] = not torch.allclose(
                baseline.proposal_hidden_states, result.proposal_hidden_states
            )
    assert all(effects.values()), effects
    assert len({id(module) for module in model.node_encoders.encoders.values()}) == 4


def test_spatiotemporal_edge_changes_embedding_attention_alpha_hidden_and_logit():
    graph_a = _graph(1, 2, conflict=True)
    graph_b = copy.deepcopy(graph_a)
    store = graph_b["align", "spatiotemporal", "proposal"]
    assert store.edge_index.shape[1] == 2
    store.edge_attr_normalized = store.edge_attr_normalized.clone()
    store.edge_attr_normalized[0] = torch.tensor([1.0, 0.95, 0.0])
    model = _model(num_layers=1)
    output_a = model(graph_a, return_attention_debug=True)
    output_b = model(graph_b, return_attention_debug=True)
    debug_a = next(item for item in output_a.attention_debug if item.relation == "spatiotemporal")
    debug_b = next(item for item in output_b.attention_debug if item.relation == "spatiotemporal")
    assert not torch.allclose(
        output_a.edge_embeddings["spatiotemporal"],
        output_b.edge_embeddings["spatiotemporal"],
    )
    assert not torch.allclose(debug_a.raw_attention_logits, debug_b.raw_attention_logits)
    assert not torch.allclose(debug_a.normalized_alpha, debug_b.normalized_alpha)
    assert not torch.allclose(output_a.proposal_hidden_states, output_b.proposal_hidden_states)
    assert not torch.allclose(output_a.candidate_logits, output_b.candidate_logits)


def test_edge_feature_changes_message_when_single_edge_attention_is_one():
    torch.manual_seed(9)
    layer = EdgeEnhancedGATLayer(
        hidden_dim=16,
        edge_dim=8,
        num_heads=2,
        dropout=0.0,
    ).eval()
    nodes = {
        "proposal": torch.randn(1, 16),
        "agent": torch.empty(0, 16),
        "align": torch.randn(1, 16),
    }
    indices = {
        "smooth": torch.empty((2, 0), dtype=torch.long),
        "spatiotemporal": torch.tensor([[0], [0]], dtype=torch.long),
    }
    edge_a = {
        "smooth": torch.empty((0, 8)),
        "spatiotemporal": torch.zeros((1, 8)),
    }
    edge_b = {
        "smooth": torch.empty((0, 8)),
        "spatiotemporal": torch.ones((1, 8)),
    }
    _, debug_a = layer(
        node_hidden=nodes,
        edge_index=indices,
        edge_embedding=edge_a,
        proposal_batch=torch.zeros(1, dtype=torch.long),
        layer_index=0,
        return_attention_debug=True,
    )
    _, debug_b = layer(
        node_hidden=nodes,
        edge_index=indices,
        edge_embedding=edge_b,
        proposal_batch=torch.zeros(1, dtype=torch.long),
        layer_index=0,
        return_attention_debug=True,
    )
    relation_a = next(item for item in debug_a if item.relation == "spatiotemporal")
    relation_b = next(item for item in debug_b if item.relation == "spatiotemporal")
    torch.testing.assert_close(relation_a.normalized_alpha, torch.ones_like(relation_a.normalized_alpha))
    torch.testing.assert_close(relation_b.normalized_alpha, torch.ones_like(relation_b.normalized_alpha))
    assert not torch.allclose(relation_a.message_content, relation_b.message_content)


def test_smooth_edge_enters_attention_and_message():
    graph_a = _graph(1, 0)
    graph_b = copy.deepcopy(graph_a)
    graph_a["agent", "smooth", "proposal"].edge_attr.fill_(1.0)
    graph_b["agent", "smooth", "proposal"].edge_attr.fill_(-1.0)
    model = _model(num_layers=1)
    output_a = model(graph_a, return_attention_debug=True)
    output_b = model(graph_b, return_attention_debug=True)
    debug_a = next(item for item in output_a.attention_debug if item.relation == "smooth")
    debug_b = next(item for item in output_b.attention_debug if item.relation == "smooth")
    assert not torch.allclose(output_a.edge_embeddings["smooth"], output_b.edge_embeddings["smooth"])
    assert not torch.allclose(debug_a.raw_attention_logits, debug_b.raw_attention_logits)
    # A single incoming edge has alpha=1, so this isolates W_z z in the message.
    torch.testing.assert_close(debug_a.normalized_alpha, debug_b.normalized_alpha)
    assert not torch.allclose(debug_a.message_content, debug_b.message_content)
    assert not torch.allclose(output_a.proposal_hidden_states, output_b.proposal_hidden_states)


def test_no_align_edge_is_stable_and_align_relation_changes_representation_when_present():
    without = _graph(1, 1, conflict=False)
    with_edge = _graph(1, 1, conflict=True)
    model = _model(num_layers=1)
    output_without = model(without, return_attention_debug=True)
    output_with = model(with_edge, return_attention_debug=True)
    assert without["align", "spatiotemporal", "proposal"].edge_index.shape[1] == 0
    assert with_edge["align", "spatiotemporal", "proposal"].edge_index.shape[1] == 1
    assert torch.isfinite(output_without.candidate_logits).all()
    assert not torch.allclose(
        output_without.proposal_hidden_states, output_with.proposal_hidden_states
    )


def test_zero_spatiotemporal_edge_graph_supports_backward():
    graph = _graph(2, 2, conflict=False)
    model = _model(num_layers=2)
    output = model(graph, return_attention_debug=True)
    conflict_debug = [
        item for item in output.attention_debug if item.relation == "spatiotemporal"
    ]
    assert graph["align", "spatiotemporal", "proposal"].edge_index.shape[1] == 0
    assert len(conflict_debug) == 2
    assert all(item.raw_attention_logits.shape == (0, 4) for item in conflict_debug)
    output.candidate_logits.square().mean().backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_attention_normalizes_jointly_across_relations_for_each_target_and_head():
    graph = _graph(1, 2, conflict=True)
    output = _model(num_layers=1)(graph, return_attention_debug=True)
    smooth = next(item for item in output.attention_debug if item.relation == "smooth")
    conflict = next(
        item for item in output.attention_debug if item.relation == "spatiotemporal"
    )
    assert smooth.normalized_alpha.shape == (1, 4)
    assert conflict.normalized_alpha.shape == (2, 4)
    combined = torch.cat([smooth.normalized_alpha, conflict.normalized_alpha], dim=0)
    torch.testing.assert_close(combined.sum(dim=0), torch.ones(4))


def test_batch_softmax_and_candidate_mapping_remain_graph_local():
    graphs = [_graph(0, 0), _graph(1, 1), _graph(3, 2, conflict=True)]
    batch = batch_candidate_graphs(graphs)
    output = _model()(batch, return_attention_debug=True)
    assert isinstance(output.graph_metadata, tuple)
    assert len(output.graph_metadata) == 3
    assert all(item["feature_schema_version"] == "heterogeneous_candidate_graph_v1" for item in output.graph_metadata)
    assert output.candidate_ptr.tolist() == [0, 1, 3, 7]
    assert output.candidate_batch.tolist() == [0, 1, 1, 2, 2, 2, 2]
    for graph_index, candidate_count in enumerate((0, 1, 3)):
        probabilities = output.probabilities_for_graph(graph_index)
        assert probabilities.shape == (candidate_count + 1,)
        assert probabilities.sum().item() == pytest.approx(1.0)
        local_mapping = [
            item for item in output.class_mapping if item.graph_index == graph_index
        ]
        assert [item.class_index for item in local_mapping] == list(
            range(candidate_count + 1)
        )
        assert local_mapping[0].kind == "null"
        for proposal_index, item in enumerate(local_mapping[1:]):
            assert item.kind == "proposal"
            assert item.candidate_id == proposal_index
            assert item.original_index == proposal_index
            np.testing.assert_allclose(
                item.world_goal,
                graphs[graph_index]["proposal"].world_position[proposal_index].numpy(),
            )
        selected = local_mapping[int(output.selected_class_index[graph_index])]
        assert output.selected_candidate_id[graph_index] == selected.candidate_id
        assert output.selected_original_index[graph_index] == selected.original_index
        np.testing.assert_allclose(
            output.selected_world_goal[graph_index].detach().numpy(),
            selected.world_goal,
        )


def test_batch_preserves_nonsequential_candidate_ids_without_resorting():
    identifiers = ([9, 2], [17, 4, 12])
    graphs = []
    for graph_index, graph_ids in enumerate(identifiers):
        proposals = [
            _proposal(
                candidate_id,
                (1.0 + 0.1 * local_index, 0.2 * graph_index, 0.0),
                score=float(candidate_id),
            )
            for local_index, candidate_id in enumerate(graph_ids)
        ]
        executions = [
            _execution(candidate_id, tuple(proposal.point))
            for candidate_id, proposal in zip(graph_ids, proposals)
        ]
        graphs.append(_build(proposals, executions))
    output = _model()(batch_candidate_graphs(graphs))
    for graph_index, graph_ids in enumerate(identifiers):
        mapping = [
            item
            for item in output.class_mapping
            if item.graph_index == graph_index and item.kind == "proposal"
        ]
        assert [item.candidate_id for item in mapping] == list(graph_ids)
        assert [item.original_index for item in mapping] == list(range(len(graph_ids)))
        for local_index, item in enumerate(mapping):
            np.testing.assert_allclose(
                item.world_goal,
                graphs[graph_index]["proposal"].world_position[local_index].numpy(),
            )


def test_raw_inf_is_debug_only_and_network_consumes_finite_x():
    proposal = _proposal(0, (1.0, 0.0, 0.0))
    graph = _build(
        [proposal],
        [_execution(0, tuple(proposal.point), clearance=np.inf)],
    )
    assert torch.isinf(graph["proposal"].x_raw).any()
    assert torch.isfinite(graph["proposal"].x).all()
    output = _model()(graph)
    assert torch.isfinite(output.candidate_logits).all()
    changed_raw = copy.deepcopy(graph)
    changed_raw["proposal"].x_raw.fill_(float("inf"))
    unchanged_output = _model()(changed_raw)
    # Same seed gives identical model parameters; x_raw is not a forward input.
    torch.testing.assert_close(output.candidate_logits, unchanged_output.candidate_logits)


def test_backward_reaches_all_network_components_but_not_graph_inputs():
    graph = _graph(2, 2, conflict=True)
    model = _model(num_layers=2)
    output = model(graph)
    loss = output.candidate_logits.square().mean()
    loss.backward()
    groups = {
        "null": model.node_encoders.encoders["null"],
        "agent": model.node_encoders.encoders["agent"],
        "proposal": model.node_encoders.encoders["proposal"],
        "align": model.node_encoders.encoders["align"],
        "smooth": model.edge_encoders.encoders["smooth"],
        "spatiotemporal": model.edge_encoders.encoders["spatiotemporal"],
        "gat": model.gat_layers,
        "output": model.output_scoring,
    }
    for name, module in groups.items():
        gradients = [parameter.grad for parameter in module.parameters()]
        assert gradients and all(gradient is not None for gradient in gradients), name
        assert all(torch.isfinite(gradient).all() for gradient in gradients), name
        assert any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients), name
    for node_type in graph.node_types:
        assert graph[node_type].x.requires_grad is False
        assert graph[node_type].x.grad is None
    for edge_type in graph.edge_types:
        assert graph[edge_type].edge_attr.requires_grad is False
        assert graph[edge_type].edge_attr.grad is None


def test_parameter_count_is_independent_of_candidate_count_and_relations_are_distinct():
    model = _model()
    before = model.parameter_counts()
    model(_graph(1, 0))
    model(_graph(10, 3, conflict=True))
    assert model.parameter_counts() == before
    assert before["total_trainable"] == sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    assert model.edge_encoders.encoders["smooth"] is not model.edge_encoders.encoders["spatiotemporal"]
    assert model.gat_layers[0].relations["smooth"] is not model.gat_layers[0].relations["spatiotemporal"]
