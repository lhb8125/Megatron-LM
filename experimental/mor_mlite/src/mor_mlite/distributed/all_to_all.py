"""Differentiable variable-size collectives for active token movement."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.distributed as dist

from .counters import CommunicationCounters
from .group import TPxCPRouteGroup
from .layout import ActiveTokenBatch, ActiveTokenLayout


def _split_tuple(values: torch.Tensor | tuple[int, ...] | list[int]) -> tuple[int, ...]:
    if isinstance(values, torch.Tensor):
        values = values.detach().to(device="cpu", dtype=torch.long).tolist()
    result = tuple(int(value) for value in values)
    if any(value < 0 for value in result):
        raise ValueError("all-to-all split sizes must be non-negative")
    return result


def _raw_variable_all_to_all(
    tensor: torch.Tensor,
    *,
    send_splits: tuple[int, ...],
    receive_splits: tuple[int, ...],
    process_group: dist.ProcessGroup,
) -> torch.Tensor:
    if sum(send_splits) != tensor.size(0):
        raise ValueError(
            f"send splits total {sum(send_splits)}, but tensor has {tensor.size(0)} rows"
        )
    output_shape = (sum(receive_splits), *tensor.shape[1:])
    output = tensor.new_empty(output_shape)
    dist.all_to_all_single(
        output,
        tensor.contiguous(),
        output_split_sizes=list(receive_splits),
        input_split_sizes=list(send_splits),
        group=process_group,
    )
    return output


class _VariableAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, send_splits, receive_splits, process_group):
        send = _split_tuple(send_splits)
        receive = _split_tuple(receive_splits)
        ctx.send_splits = send
        ctx.receive_splits = receive
        ctx.process_group = process_group
        return _raw_variable_all_to_all(
            tensor,
            send_splits=send,
            receive_splits=receive,
            process_group=process_group,
        )

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = _raw_variable_all_to_all(
            grad_output,
            send_splits=ctx.receive_splits,
            receive_splits=ctx.send_splits,
            process_group=ctx.process_group,
        )
        return grad_input, None, None, None


def variable_all_to_all(
    tensor: torch.Tensor,
    *,
    send_splits: torch.Tensor | tuple[int, ...] | list[int],
    receive_splits: torch.Tensor | tuple[int, ...] | list[int],
    route_group: TPxCPRouteGroup,
) -> torch.Tensor:
    """Move variable first-dimension rows with an exact inverse backward."""

    send = _split_tuple(send_splits)
    receive = _split_tuple(receive_splits)
    if len(send) != route_group.world_size or len(receive) != route_group.world_size:
        raise ValueError("split vectors must have one element per route-group rank")
    if route_group.world_size == 1:
        if send != receive or sum(send) != tensor.size(0):
            raise ValueError("single-rank all-to-all splits must describe an identity transfer")
        return tensor
    assert route_group.process_group is not None
    return _VariableAllToAll.apply(tensor, send, receive, route_group.process_group)


def _stable_order(*keys: torch.Tensor) -> torch.Tensor:
    if not keys:
        raise ValueError("at least one ordering key is required")
    order = torch.arange(keys[0].numel(), device=keys[0].device)
    for key in reversed(keys):
        order = order[torch.argsort(key.index_select(0, order), stable=True)]
    return order


def _exchange_counts(
    send_counts: torch.Tensor,
    route_group: TPxCPRouteGroup,
    counters: CommunicationCounters,
) -> torch.Tensor:
    if route_group.world_size == 1:
        return send_counts.clone()
    receive_counts = torch.empty_like(send_counts)
    assert route_group.process_group is not None
    dist.all_to_all_single(receive_counts, send_counts, group=route_group.process_group)
    counters.count_exchanges += 1
    counters.collective_calls += 1
    return receive_counts


def _all_gather_variable_rows(
    rows: torch.Tensor,
    route_group: TPxCPRouteGroup,
    counters: CommunicationCounters,
) -> torch.Tensor:
    if rows.dim() != 2:
        raise ValueError("rows must be a two-dimensional tensor")
    if route_group.world_size == 1:
        return rows
    assert route_group.process_group is not None
    local_count = torch.tensor([rows.size(0)], dtype=torch.long, device=rows.device)
    counts = [torch.empty_like(local_count) for _ in range(route_group.world_size)]
    dist.all_gather(counts, local_count, group=route_group.process_group)
    counters.collective_calls += 1
    maximum = max(int(count.item()) for count in counts)
    if maximum == 0:
        counters.route_gathers += 1
        return rows.new_empty((0, rows.size(1)))
    padded = rows.new_full((maximum, rows.size(1)), -1)
    if rows.numel():
        padded[: rows.size(0)] = rows
    gathered = [torch.empty_like(padded) for _ in range(route_group.world_size)]
    dist.all_gather(gathered, padded, group=route_group.process_group)
    counters.collective_calls += 1
    counters.route_gathers += 1
    return torch.cat(
        [value[: int(count.item())] for value, count in zip(gathered, counts, strict=True)],
        dim=0,
    )


@dataclass(frozen=True, slots=True)
class BalancedTargetPlan:
    """Per-local-token deterministic target rank and local destination slot."""

    target_ranks: torch.Tensor
    destination_slots: torch.Tensor


def build_balanced_target_plan(
    layout: ActiveTokenLayout,
    route_group: TPxCPRouteGroup,
    counters: CommunicationCounters,
) -> BalancedTargetPlan:
    """Balance every sample independently in canonical causal order.

    The plan is reproduced on every rank from gathered integer metadata.  It
    deliberately does not approximate or round the active-token count.
    """

    layout.assert_compute_ready(local_route_rank=route_group.rank)
    local = torch.stack([layout.global_token_ids, layout.sample_ids, layout.position_ids], dim=1)
    global_rows = _all_gather_variable_rows(local, route_group, counters)
    total = global_rows.size(0)
    if total == 0:
        empty = torch.empty(0, dtype=torch.long, device=layout.device)
        return BalancedTargetPlan(empty, empty)
    global_ids = global_rows[:, 0]
    if torch.unique(global_ids).numel() != total:
        raise ValueError("global_token_ids must be unique across the TPxCP route group")
    canonical = _stable_order(global_rows[:, 1], global_rows[:, 2], global_ids)
    global_targets = torch.empty(total, dtype=torch.long, device=layout.device)
    global_slots = torch.empty_like(global_targets)
    per_rank_next_slot = torch.zeros(route_group.world_size, dtype=torch.long, device=layout.device)
    ordered_samples = global_rows[canonical, 1]
    cursor = 0
    while cursor < total:
        sample_id = ordered_samples[cursor]
        end = cursor + 1
        while end < total and bool((ordered_samples[end] == sample_id).item()):
            end += 1
        sample_order = canonical[cursor:end]
        count = end - cursor
        base, extra = divmod(count, route_group.world_size)
        offset = 0
        for target in range(route_group.world_size):
            take = base + int(target < extra)
            indices = sample_order[offset : offset + take]
            global_targets[indices] = target
            if take:
                start_slot = int(per_rank_next_slot[target].item())
                global_slots[indices] = torch.arange(
                    start_slot,
                    start_slot + take,
                    dtype=torch.long,
                    device=layout.device,
                )
                per_rank_next_slot[target] += take
            offset += take
        cursor = end

    ids_sorted, id_order = torch.sort(global_ids)
    lookup = torch.searchsorted(ids_sorted, layout.global_token_ids)
    if layout.num_tokens and (
        bool((lookup >= ids_sorted.numel()).any().item())
        or not torch.equal(ids_sorted.index_select(0, lookup), layout.global_token_ids)
    ):
        raise RuntimeError("failed to map local token IDs into the global route plan")
    source_indices = id_order.index_select(0, lookup)
    return BalancedTargetPlan(
        target_ranks=global_targets.index_select(0, source_indices),
        destination_slots=global_slots.index_select(0, source_indices),
    )


class ActiveTokenDispatcher:
    """Move active hidden/gate rows while preserving inverse ownership metadata."""

    def __init__(
        self,
        route_group: TPxCPRouteGroup | None = None,
        *,
        counters: CommunicationCounters | None = None,
    ) -> None:
        self.route_group = route_group or TPxCPRouteGroup.local()
        self.counters = counters or CommunicationCounters()

    def dispatch(
        self,
        batch: ActiveTokenBatch,
        target_ranks: torch.Tensor,
        *,
        destination_slots: torch.Tensor | None = None,
        inverse: bool = False,
        layout_kind: str = "route_sharded",
    ) -> ActiveTokenBatch:
        layout = batch.layout
        layout.assert_compute_ready(local_route_rank=self.route_group.rank)
        # Preserve the placement *mode*, not merely whether this rank happens
        # to receive rows.  In a variable A2A an explicitly slotted direct
        # plan may legitimately produce an empty receive shard; that shard
        # still needs an empty destination_slots tensor so the backend can
        # expand it into its certified padded layout without deadlocking peers.
        explicit_destination_slots = destination_slots is not None
        if target_ranks.shape != (batch.num_tokens,) or target_ranks.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise ValueError("target_ranks must be a one-dimensional integer tensor per token")
        target_ranks = target_ranks.to(device=layout.device, dtype=torch.long)
        if target_ranks.numel() and (
            int(target_ranks.min().item()) < 0
            or int(target_ranks.max().item()) >= self.route_group.world_size
        ):
            raise ValueError("target_ranks contains a rank outside the TPxCP route group")
        if destination_slots is not None:
            if destination_slots.shape != (batch.num_tokens,):
                raise ValueError("destination_slots must contain one value per token")
            destination_slots = destination_slots.to(device=layout.device, dtype=torch.long)
            if destination_slots.numel() and int(destination_slots.min().item()) < 0:
                raise ValueError("destination slots must be non-negative")
            order = _stable_order(target_ranks, destination_slots, layout.global_token_ids)
        else:
            order = _stable_order(
                target_ranks, layout.sample_ids, layout.position_ids, layout.global_token_ids
            )
        target_ranks = target_ranks.index_select(0, order)
        send_batch = batch.index_select(order)
        ordered_slots = (
            None if destination_slots is None else destination_slots.index_select(0, order)
        )
        send_counts = torch.bincount(target_ranks, minlength=self.route_group.world_size).to(
            dtype=torch.long
        )
        receive_counts = _exchange_counts(send_counts, self.route_group, self.counters)
        send_splits = _split_tuple(send_counts)
        receive_splits = _split_tuple(receive_counts)

        hidden = variable_all_to_all(
            send_batch.hidden,
            send_splits=send_splits,
            receive_splits=receive_splits,
            route_group=self.route_group,
        )
        if self.route_group.world_size > 1:
            self.counters.hidden_all_to_all += 1
            self.counters.collective_calls += 1
        gates = None
        if send_batch.gates is not None:
            gates = variable_all_to_all(
                send_batch.gates,
                send_splits=send_splits,
                receive_splits=receive_splits,
                route_group=self.route_group,
            )
            if self.route_group.world_size > 1:
                self.counters.gate_all_to_all += 1
                self.counters.collective_calls += 1

        sent_layout = send_batch.layout
        slot_column = (
            torch.full((batch.num_tokens,), -1, dtype=torch.long, device=layout.device)
            if ordered_slots is None
            else ordered_slots
        )
        metadata = torch.stack(
            [
                sent_layout.sample_ids,
                sent_layout.position_ids,
                sent_layout.global_token_ids,
                sent_layout.source_route_ranks,
                sent_layout.source_local_rows,
                sent_layout.padding_mask.to(dtype=torch.long),
                slot_column,
            ],
            dim=1,
        )
        received_metadata = variable_all_to_all(
            metadata,
            send_splits=send_splits,
            receive_splits=receive_splits,
            route_group=self.route_group,
        )
        if self.route_group.world_size > 1:
            self.counters.metadata_all_to_all += 1
            self.counters.collective_calls += 1
        num_received = received_metadata.size(0)
        received_slots = received_metadata[:, 6]
        have_slots = explicit_destination_slots
        if have_slots and num_received and not bool((received_slots >= 0).all().item()):
            raise RuntimeError(
                "explicit destination-slot dispatch received an invalid negative slot"
            )
        received_layout = ActiveTokenLayout(
            sample_ids=received_metadata[:, 0],
            position_ids=received_metadata[:, 1],
            global_token_ids=received_metadata[:, 2],
            source_route_ranks=received_metadata[:, 3],
            source_local_rows=received_metadata[:, 4],
            current_route_ranks=torch.full(
                (num_received,),
                self.route_group.rank,
                dtype=torch.long,
                device=layout.device,
            ),
            padding_mask=received_metadata[:, 5].bool(),
            destination_slots=received_slots if have_slots else None,
            round_index=layout.round_index,
            layout_kind=layout_kind,
            replicated=False,
        )
        received = ActiveTokenBatch(hidden=hidden, gates=gates, layout=received_layout)
        receive_order = (
            received.layout.destination_order() if have_slots else received.layout.canonical_order()
        )
        received = received.index_select(receive_order)
        # Count only after hidden/gate/metadata have all completed and the
        # received layout has been validated.  This is deliberately distinct
        # from the transition state machine's active-set-change counter.
        self.counters.record_token_dispatch(inverse=inverse)
        return received

    def restore_to_sources(self, batch: ActiveTokenBatch) -> ActiveTokenBatch:
        """Return all rows to their immutable source owner and canonical local row."""

        restore_layout = replace(
            batch.layout,
            destination_slots=batch.layout.source_local_rows,
            layout_kind="restoring",
            replicated=False,
        )
        return self.dispatch(
            replace(batch, layout=restore_layout),
            restore_layout.source_route_ranks,
            destination_slots=restore_layout.source_local_rows,
            inverse=True,
            layout_kind="source_local",
        )


__all__ = [
    "ActiveTokenDispatcher",
    "BalancedTargetPlan",
    "build_balanced_target_plan",
    "variable_all_to_all",
]
