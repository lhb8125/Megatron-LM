"""Depth-routing primitives shared by local and distributed MoR execution."""

from .depth_router import (
    DepthRouter,
    DepthRouterOutput,
    ExpertChoiceSelection,
    apply_recurrent_update,
    globally_normalized_bce_with_logits,
    replay_selected_gates,
    select_expert_choice_per_sample,
    stable_expert_choice_indices,
)
from .plan import ROUTE_PLAN_SCHEMA_VERSION, RoutePlan

__all__ = [
    "ROUTE_PLAN_SCHEMA_VERSION",
    "DepthRouter",
    "DepthRouterOutput",
    "ExpertChoiceSelection",
    "RoutePlan",
    "apply_recurrent_update",
    "globally_normalized_bce_with_logits",
    "replay_selected_gates",
    "select_expert_choice_per_sample",
    "stable_expert_choice_indices",
]
