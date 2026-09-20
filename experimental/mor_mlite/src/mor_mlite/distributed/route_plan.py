"""Adapters between routing.RoutePlan and distributed token batches."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .all_to_all import BalancedTargetPlan
from .group import TPxCPRouteGroup
from .layout import ActiveTokenBatch, ActiveTokenLayout

if TYPE_CHECKING:
    from mor_mlite.routing.plan import RoutePlan


def _flatten_rank(
    tp_ranks: torch.Tensor,
    cp_ranks: torch.Tensor,
    route_group: TPxCPRouteGroup,
) -> torch.Tensor:
    if tp_ranks.shape != cp_ranks.shape:
        raise ValueError("TP and CP rank fields must have equal shapes")
    if tp_ranks.numel() and (
        int(tp_ranks.min().item()) < 0 or int(tp_ranks.max().item()) >= route_group.tp_size
    ):
        raise ValueError("RoutePlan contains a TP rank outside the route group")
    if cp_ranks.numel() and (
        int(cp_ranks.min().item()) < 0 or int(cp_ranks.max().item()) >= route_group.cp_size
    ):
        raise ValueError("RoutePlan contains a CP rank outside the route group")
    return cp_ranks.to(dtype=torch.long) * route_group.tp_size + tp_ranks.to(dtype=torch.long)


def batch_from_route_plan(
    hidden: torch.Tensor,
    plan: RoutePlan,
    route_group: TPxCPRouteGroup,
    *,
    differentiable_gates: torch.Tensor | None = None,
    replicated: bool = False,
) -> ActiveTokenBatch:
    """Attach a RoutePlan to selected hidden rows without detaching gates.

    ``hidden`` and ``differentiable_gates`` must already be aligned with the
    rows in ``plan``.  Use ``DepthRouterOutput.selected_indices`` to select the
    hidden rows and ``DepthRouterOutput.selected_gates`` for the live gate path;
    ``plan.selected_gates`` is intentionally detached for serialization.
    """

    token_count = len(plan)
    if hidden.size(0) != token_count:
        raise ValueError("hidden rows must be aligned with RoutePlan rows")
    if differentiable_gates is None:
        differentiable_gates = plan.selected_gates.to(device=hidden.device, dtype=hidden.dtype)
    if differentiable_gates.size(0) != token_count:
        raise ValueError("differentiable_gates must be aligned with RoutePlan rows")
    source_ranks = _flatten_rank(
        plan.source_tp_ranks.to(hidden.device),
        plan.source_cp_ranks.to(hidden.device),
        route_group,
    )
    current_ranks = source_ranks.clone()
    if (
        not replicated
        and token_count
        and not bool((current_ranks == route_group.rank).all().item())
    ):
        raise ValueError("local RoutePlan contains tokens sourced by another route rank")
    destination_slots = plan.target_local_rows.to(device=hidden.device, dtype=torch.long)
    layout = ActiveTokenLayout(
        sample_ids=plan.sample_ids.to(device=hidden.device, dtype=torch.long),
        position_ids=plan.original_positions.to(device=hidden.device, dtype=torch.long),
        global_token_ids=plan.global_token_ids.to(device=hidden.device, dtype=torch.long),
        source_route_ranks=source_ranks,
        source_local_rows=plan.source_local_rows.to(device=hidden.device, dtype=torch.long),
        current_route_ranks=current_ranks,
        padding_mask=plan.padding_mask.to(device=hidden.device, dtype=torch.bool),
        destination_slots=(
            destination_slots if bool((destination_slots >= 0).all().item()) else None
        ),
        round_index=plan.round_index,
        layout_kind="route_plan",
        replicated=replicated,
    )
    return ActiveTokenBatch(hidden=hidden, gates=differentiable_gates, layout=layout)


def target_plan_from_route_plan(
    plan: RoutePlan, route_group: TPxCPRouteGroup, *, device: torch.device | str
) -> BalancedTargetPlan:
    target_ranks = _flatten_rank(
        plan.target_tp_ranks.to(device), plan.target_cp_ranks.to(device), route_group
    )
    destination_slots = plan.target_local_rows.to(device=device, dtype=torch.long)
    if destination_slots.numel() and int(destination_slots.min().item()) < 0:
        raise ValueError(
            "RoutePlan target_local_rows are unresolved; run a layout planner before dispatch"
        )
    if bool(plan.padding_mask.any().item()):
        raise ValueError("padding RoutePlan rows must be excluded before active dispatch")
    return BalancedTargetPlan(target_ranks, destination_slots)


__all__ = ["batch_from_route_plan", "target_plan_from_route_plan"]
