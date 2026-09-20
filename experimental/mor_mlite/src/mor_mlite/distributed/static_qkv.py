"""Autograd-safe full active Q/K/V gather for the static CP oracle.

This path intentionally duplicates the tiny attention working set on every CP
rank.  It is a correctness/imbalance diagnostic, not the production CP backend;
MagiAttention remains responsible for BF16 Qwen CP execution.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from .group import TPxCPRouteGroup


class _VariableAllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, counts, process_group):
        counts = tuple(int(count) for count in counts)
        rank = dist.get_rank(process_group)
        maximum = max(counts, default=0)
        padded = value.new_zeros((maximum, *value.shape[1:]))
        if value.size(0):
            padded[: value.size(0)] = value
        peers = [torch.empty_like(padded) for _ in counts]
        dist.all_gather(peers, padded, group=process_group)
        ctx.counts = counts
        ctx.rank = rank
        ctx.process_group = process_group
        return torch.cat([peer[:count] for peer, count in zip(peers, counts, strict=True)], dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        # Every CP rank owns a disjoint set of query losses but all ranks consume
        # every gathered K/V row.  Sum those contributions before returning the
        # gradient slice to the source owner.
        grad_global = grad_output.contiguous()
        dist.all_reduce(grad_global, group=ctx.process_group)
        begin = sum(ctx.counts[: ctx.rank])
        local = grad_global.narrow(0, begin, ctx.counts[ctx.rank]).contiguous()
        return local, None, None


def _gather_counts(
    local_count: int, group: TPxCPRouteGroup, device: torch.device
) -> tuple[int, ...]:
    count = torch.tensor([local_count], dtype=torch.int64, device=device)
    peers = [torch.empty_like(count) for _ in range(group.world_size)]
    assert group.process_group is not None
    dist.all_gather(peers, count, group=group.process_group)
    return tuple(int(peer.item()) for peer in peers)


def _all_gather_metadata(
    value: torch.Tensor,
    *,
    counts: tuple[int, ...],
    group: TPxCPRouteGroup,
) -> torch.Tensor:
    maximum = max(counts, default=0)
    padded = value.new_full((maximum, *value.shape[1:]), -1)
    if value.size(0):
        padded[: value.size(0)] = value
    peers = [torch.empty_like(padded) for _ in counts]
    assert group.process_group is not None
    dist.all_gather(peers, padded, group=group.process_group)
    return torch.cat([peer[:count] for peer, count in zip(peers, counts, strict=True)], dim=0)


@dataclass(frozen=True, slots=True)
class StaticGatheredQKV:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    sample_ids: torch.Tensor
    position_ids: torch.Tensor
    global_token_ids: torch.Tensor
    local_rows: torch.Tensor


def gather_static_active_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    sample_ids: torch.Tensor,
    position_ids: torch.Tensor,
    global_token_ids: torch.Tensor,
    route_group: TPxCPRouteGroup,
) -> StaticGatheredQKV:
    """Gather an imbalanced CP shard and return canonical global active rows.

    The function is restricted to TP=1 because the tiny reference keeps the
    hidden dimension intact.  Output rows are ordered by sample, original
    position, then global token ID.  ``local_rows`` maps the caller's unchanged
    local token order into that canonical buffer.
    """

    local_count = q.size(0)
    if route_group.tp_size != 1:
        raise ValueError("static active Q/K/V gather is a TP=1 tiny diagnostic")
    if not (q.ndim >= 2 and k.ndim >= 2 and v.ndim >= 2):
        raise ValueError("q, k, and v need a leading token dimension")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same device")
    vectors = (sample_ids, position_ids, global_token_ids)
    if any(
        vector.ndim != 1 or vector.numel() != local_count or vector.device != q.device
        for vector in vectors
    ):
        raise ValueError("static Q/K/V metadata must be local-token-aligned")

    if route_group.world_size == 1:
        counts = (local_count,)
        gathered_q, gathered_k, gathered_v = q, k, v
        metadata = torch.stack(vectors, dim=1)
    else:
        if not dist.is_initialized() or route_group.process_group is None:
            raise RuntimeError("distributed static Q/K/V gather needs an initialized CP group")
        counts = _gather_counts(local_count, route_group, q.device)
        gathered_q = _VariableAllGather.apply(q, counts, route_group.process_group)
        gathered_k = _VariableAllGather.apply(k, counts, route_group.process_group)
        gathered_v = _VariableAllGather.apply(v, counts, route_group.process_group)
        metadata = _all_gather_metadata(
            torch.stack(vectors, dim=1), counts=counts, group=route_group
        )

    if metadata.numel() and torch.unique(metadata[:, 2]).numel() != metadata.size(0):
        raise ValueError("static CP shards contain duplicate global token IDs")
    order = torch.arange(metadata.size(0), device=q.device)
    for column in (2, 1, 0):
        order = order[torch.argsort(metadata[order, column], stable=True)]
    metadata = metadata.index_select(0, order)
    gathered_q = gathered_q.index_select(0, order)
    gathered_k = gathered_k.index_select(0, order)
    gathered_v = gathered_v.index_select(0, order)

    row_by_id = {
        int(token_id): row for row, token_id in enumerate(metadata[:, 2].detach().cpu().tolist())
    }
    try:
        local_rows = torch.tensor(
            [row_by_id[int(token_id)] for token_id in global_token_ids.detach().cpu().tolist()],
            dtype=torch.long,
            device=q.device,
        )
    except KeyError as exc:  # pragma: no cover - guarded by the gathered metadata.
        raise RuntimeError(f"local token {exc.args[0]} disappeared during static gather") from exc
    return StaticGatheredQKV(
        q=gathered_q,
        k=gathered_k,
        v=gathered_v,
        sample_ids=metadata[:, 0],
        position_ids=metadata[:, 1],
        global_token_ids=metadata[:, 2],
        local_rows=local_rows,
    )


__all__ = ["StaticGatheredQKV", "gather_static_active_qkv"]
