"""Lightweight names shared by Pre-GAT execution and artifact tooling."""

from __future__ import annotations


CLOSED_LOOP_SCHEMA_VERSION = "pre_gat_closed_loop_v1"
FORMAL_PREVIEW_HORIZON = 4

METHOD_FROZEN = "frozen_sac_dmp"
METHOD_PROPOSAL = "proposal_sac_dmp"
METHOD_FP_SHEP = "fp_shep_sac_dmp"
METHOD_ORDER = (METHOD_FROZEN, METHOD_PROPOSAL, METHOD_FP_SHEP)
METHOD_DISPLAY_NAMES = {
    METHOD_FROZEN: "Frozen SAC-DMP",
    METHOD_PROPOSAL: "Proposal + SAC-DMP",
    METHOD_FP_SHEP: "FP-SHEP + SAC-DMP",
}

SELECTION_BASELINE_TERMINAL = "baseline_terminal_goal"
SELECTION_PROPOSAL = "proposal_selection"
SELECTION_FP_SHEP = "fp_shep_selection"
SELECTION_NO_CANDIDATE_FALLBACK = "no_candidate_fallback"
