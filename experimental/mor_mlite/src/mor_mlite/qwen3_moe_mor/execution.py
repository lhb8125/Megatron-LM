"""Active-layout execution helpers for the native Qwen3-MoE MoR model.

This module contains no model parameters. It translates a globally identical
depth RoutePlan into the topology-specific token layout required by either TE
sequence parallelism (CP=1) or MagiAttention (CP>1), while keeping hidden and
gate movement differentiable.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch
from megatron.lite.primitive.utils.packed_seq import PackedSeqParams

from mor_mlite.distributed.all_to_all import ActiveTokenDispatcher
from mor_mlite.distributed.group import TPxCPRouteGroup
from mor_mlite.distributed.layout import ActiveTokenBatch, ActiveTokenLayout
from mor_mlite.distributed.packing import CanonicalActivePacking
from mor_mlite.routing import RoutePlan


def _placement_by_id(
    global_ids: torch.Tensor,
    target_ranks: torch.Tensor,
    destination_slots: torch.Tensor,
) -> dict[int, tuple[int, int]]:
    if not (
        global_ids.ndim == target_ranks.ndim == destination_slots.ndim == 1
        and global_ids.numel() == target_ranks.numel() == destination_slots.numel()
    ):
        raise ValueError("global placement vectors must be equally-sized and one-dimensional")
    result = {
        int(token_id): (int(target), int(slot))
        for token_id, target, slot in zip(
            global_ids.detach().cpu().tolist(),
            target_ranks.detach().cpu().tolist(),
            destination_slots.detach().cpu().tolist(),
            strict=True,
        )
    }
    if len(result) != global_ids.numel():
        raise ValueError("global placement token IDs must be unique")
    return result


@dataclass(frozen=True, slots=True)
class RegularDirectPlan:
    """Certified contiguous TP-SP placement for a CP=1 active buffer."""

    target_ranks: torch.Tensor
    destination_slots: torch.Tensor
    local_sample_ids: torch.Tensor
    local_position_ids: torch.Tensor
    local_global_token_ids: torch.Tensor
    local_padding_mask: torch.Tensor
    attention_position_ids: torch.Tensor
    attention_padding_mask: torch.Tensor
    global_real_token_ids: torch.Tensor
    global_target_ranks: torch.Tensor
    global_destination_slots: torch.Tensor

    @property
    def local_num_slots(self) -> int:
        return int(self.local_global_token_ids.numel())


def build_regular_direct_plan(
    batch: ActiveTokenBatch,
    *,
    packing: CanonicalActivePacking,
    route_group: TPxCPRouteGroup,
) -> RegularDirectPlan:
    """Map real selected tokens into contiguous equal TP sequence shards."""

    if route_group.cp_size != 1:
        raise ValueError("regular direct planning is valid only when CP=1")
    if packing.num_rows % route_group.tp_size:
        raise ValueError("canonical active rows must be divisible by TP")
    rows_per_rank = packing.num_rows // route_group.tp_size
    device = batch.hidden.device
    canonical_rows = torch.arange(packing.num_rows, dtype=torch.long, device=device)
    canonical_targets = torch.div(canonical_rows, rows_per_rank, rounding_mode="floor")
    canonical_slots = canonical_rows.remainder(rows_per_rank)
    real_mask = ~packing.padding_mask
    real_ids = packing.global_token_ids[real_mask]
    real_targets = canonical_targets[real_mask]
    real_slots = canonical_slots[real_mask]
    placement = _placement_by_id(real_ids, real_targets, real_slots)
    try:
        local_placements = [
            placement[int(token_id)]
            for token_id in batch.layout.global_token_ids.detach().cpu().tolist()
        ]
    except KeyError as exc:
        raise ValueError(
            f"selected token ID {exc.args[0]} is absent from canonical active metadata"
        ) from exc
    target_ranks = torch.tensor(
        [value[0] for value in local_placements], dtype=torch.long, device=device
    )
    destination_slots = torch.tensor(
        [value[1] for value in local_placements], dtype=torch.long, device=device
    )
    begin = route_group.rank * rows_per_rank
    local_rows = canonical_rows.narrow(0, begin, rows_per_rank)
    return RegularDirectPlan(
        target_ranks=target_ranks,
        destination_slots=destination_slots,
        local_sample_ids=packing.sample_ids.index_select(0, local_rows),
        local_position_ids=packing.position_ids.index_select(0, local_rows),
        local_global_token_ids=packing.global_token_ids.index_select(0, local_rows),
        local_padding_mask=packing.padding_mask.index_select(0, local_rows),
        attention_position_ids=packing.position_ids,
        attention_padding_mask=packing.padding_mask,
        global_real_token_ids=real_ids,
        global_target_ranks=real_targets,
        global_destination_slots=real_slots,
    )


def _expand_compact_result(
    compact: ActiveTokenBatch,
    *,
    route_rank: int,
    local_sample_ids: torch.Tensor,
    local_position_ids: torch.Tensor,
    local_global_token_ids: torch.Tensor,
    local_padding_mask: torch.Tensor,
) -> ActiveTokenBatch:
    slots = compact.layout.destination_slots
    if slots is None:
        raise RuntimeError("direct dispatch lost destination slots")
    local_count = int(local_global_token_ids.numel())
    if slots.numel() and int(slots.max().item()) >= local_count:
        raise ValueError("received destination slot is outside the local active buffer")
    if not torch.equal(
        local_global_token_ids.index_select(0, slots),
        compact.layout.global_token_ids,
    ):
        raise RuntimeError("direct result does not match the certified token order")
    hidden = compact.hidden.new_zeros((local_count, *compact.hidden.shape[1:])).index_copy(
        0, slots, compact.hidden
    )
    gates = None
    if compact.gates is not None:
        gates = compact.gates.new_zeros((local_count, *compact.gates.shape[1:])).index_copy(
            0, slots, compact.gates
        )
    source_ranks = torch.full(
        (local_count,), -1, dtype=torch.long, device=compact.hidden.device
    ).index_copy(0, slots, compact.layout.source_route_ranks)
    source_rows = torch.full_like(source_ranks, -1).index_copy(
        0, slots, compact.layout.source_local_rows
    )
    layout = ActiveTokenLayout(
        sample_ids=local_sample_ids,
        position_ids=local_position_ids,
        global_token_ids=local_global_token_ids,
        source_route_ranks=source_ranks,
        source_local_rows=source_rows,
        current_route_ranks=torch.full_like(source_ranks, int(route_rank)),
        padding_mask=local_padding_mask,
        destination_slots=torch.arange(local_count, dtype=torch.long, device=compact.hidden.device),
        round_index=compact.layout.round_index,
        layout_kind="regular_direct",
        replicated=False,
    )
    if not torch.equal(layout.padding_mask, source_ranks < 0):
        raise RuntimeError("dummy slots and real receives do not form an exact partition")
    return ActiveTokenBatch(hidden=hidden, gates=gates, layout=layout)


class RegularDirectBackend:
    """One differentiable A2A into a padded CP=1 TP-SP active layout."""

    name = "regular_direct"

    def rebalance(
        self,
        batch: ActiveTokenBatch,
        dispatcher: ActiveTokenDispatcher,
        *,
        target_ranks: torch.Tensor | None = None,
        destination_slots: torch.Tensor | None = None,
        context: dict[str, Any] | None = None,
    ) -> ActiveTokenBatch:
        plan = None if context is None else context.get("direct_plan")
        if not isinstance(plan, RegularDirectPlan):
            raise TypeError("regular direct backend requires a RegularDirectPlan")
        if target_ranks is not None or destination_slots is not None:
            raise ValueError("placement is owned by the RegularDirectPlan")
        compact = dispatcher.dispatch(
            batch,
            plan.target_ranks,
            destination_slots=plan.destination_slots,
            layout_kind=self.name,
        )
        return _expand_compact_result(
            compact,
            route_rank=dispatcher.route_group.rank,
            local_sample_ids=plan.local_sample_ids,
            local_position_ids=plan.local_position_ids,
            local_global_token_ids=plan.local_global_token_ids,
            local_padding_mask=plan.local_padding_mask,
        )


def rewrite_route_targets(
    plan: RoutePlan,
    *,
    real_token_ids: torch.Tensor,
    target_ranks: torch.Tensor,
    destination_slots: torch.Tensor,
    route_group: TPxCPRouteGroup,
) -> RoutePlan:
    """Attach topology-specific destination ownership to a global RoutePlan."""

    placement = _placement_by_id(real_token_ids, target_ranks, destination_slots)
    try:
        values = [
            placement[int(token_id)] for token_id in plan.global_token_ids.detach().cpu().tolist()
        ]
    except KeyError as exc:
        raise ValueError(f"RoutePlan token ID {exc.args[0]} has no destination placement") from exc
    device = plan.global_token_ids.device
    flat_ranks = torch.tensor([value[0] for value in values], dtype=torch.long, device=device)
    slots = torch.tensor([value[1] for value in values], dtype=torch.long, device=device)
    return replace(
        plan,
        target_tp_ranks=flat_ranks.remainder(route_group.tp_size),
        target_cp_ranks=torch.div(flat_ranks, route_group.tp_size, rounding_mode="floor"),
        target_local_rows=slots,
    )


def active_packed_seq_params(
    packing: CanonicalActivePacking,
    *,
    cp_group=None,
    cp_rank: int = 0,
    cp_size: int = 1,
    runtime_key: Any | None = None,
) -> PackedSeqParams:
    """Build one runtime object shared by all physical recurrent layers."""

    lengths = packing.cu_seqlens[1:] - packing.cu_seqlens[:-1]
    active_max = int(lengths.max().item()) if lengths.numel() else 0
    if runtime_key is None:
        return PackedSeqParams.from_cu_seqlens(packing.cu_seqlens, max_seqlen=active_max)
    max_position = int(packing.position_ids.max().item()) + 1 if packing.position_ids.numel() else 0
    rope_length = max(active_max, max_position)
    return PackedSeqParams(
        qkv_format="magi",
        cu_seqlens_q=packing.cu_seqlens,
        cu_seqlens_kv=packing.cu_seqlens,
        max_seqlen_q=rope_length,
        max_seqlen_kv=rope_length,
        local_cp_size=cp_size,
        cp_group=cp_group,
        cp_rank=cp_rank,
        magi_runtime_key=runtime_key,
    )


__all__ = [
    "RegularDirectBackend",
    "RegularDirectPlan",
    "active_packed_seq_params",
    "build_regular_direct_plan",
    "rewrite_route_targets",
]
