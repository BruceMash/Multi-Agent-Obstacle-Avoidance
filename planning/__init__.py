"""Planning-side execution preview utilities."""

from planning.policy_preview import (
    CandidatePreview,
    PreviewInitialState,
    PreviewLocalContext,
    PreviewPerformance,
    PreviewTrajectory,
    adapt_candidate_proposals,
    build_preview_inputs_from_env,
    preview_candidate,
    preview_candidates,
)
from planning.candidate_execution_interface import (
    EXECUTION_FEATURE_ORDER,
    ConstantVelocityConflictDiagnostic,
    ExecutionNormalizationSpec,
    GraphReadyCandidateExecution,
    NormalizedExecutionFeatures,
    constant_velocity_conflict_diagnostic,
    graph_ready_candidate_execution,
    normalize_execution_features,
)

# Graph construction requires torch_geometric.  Keeping these imports lazy
# preserves the established preview-only environment where PyG is optional.
_GRAPH_EXPORTS = {
    "EgoGraphState",
    "GRAPH_SCHEMA_VERSION",
    "HeterogeneousCandidateGraphConfig",
    "LOCAL_FRAME_CONVENTION",
    "NeighborGraphState",
    "build_heterogeneous_candidate_graph",
    "build_heterogeneous_candidate_graph_from_env",
    "feature_schema_table",
    "graph_debug_summary",
    "world_to_ego_local",
}


def __getattr__(name: str):
    if name not in _GRAPH_EXPORTS:
        raise AttributeError(name)
    from planning import heterogeneous_candidate_graph as graph_module

    return getattr(graph_module, name)

__all__ = [
    "CandidatePreview",
    "PreviewInitialState",
    "PreviewLocalContext",
    "PreviewPerformance",
    "PreviewTrajectory",
    "adapt_candidate_proposals",
    "build_preview_inputs_from_env",
    "preview_candidate",
    "preview_candidates",
    "EXECUTION_FEATURE_ORDER",
    "ConstantVelocityConflictDiagnostic",
    "ExecutionNormalizationSpec",
    "GraphReadyCandidateExecution",
    "NormalizedExecutionFeatures",
    "constant_velocity_conflict_diagnostic",
    "graph_ready_candidate_execution",
    "normalize_execution_features",
    *_GRAPH_EXPORTS,
]
