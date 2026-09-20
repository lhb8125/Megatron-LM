"""Dense-DP-local depth routing across tensor- and context-parallel shards.

The recurrent router is evaluated on the local TP sequence-parallel shard.  Only
detached FP32 scalar scores and integer token metadata are gathered across the
current dense-DP replica's ``TP x CP`` rectangle.  Hidden states never take
part in this collective; they move later through an active-token dispatcher.

Every peer reconstructs the same per-sample expert-choice decision.  The
returned global :class:`~mor_mlite.routing.RoutePlan` is therefore suitable for
logging/replay, while ``selected_batch.gates`` remains connected to the local
router logits for backpropagation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

from mor_mlite.routing import (
    DepthRouter,
    RoutePlan,
    replay_selected_gates,
    select_expert_choice_per_sample,
)
from mor_mlite.routing.depth_router import validate_replay_capacity

from .counters import CommunicationCounters
from .group import TPxCPRouteGroup
from .layout import ActiveTokenBatch

_INT_FIELDS = 9


def _dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _assert_dense_dp_local_membership(route_group: TPxCPRouteGroup) -> None:
    """Validate MLite PP=1's contiguous rank block for one dense-DP replica."""

    width = route_group.world_size
    first = route_group.global_ranks[0]
    base = (first // width) * width
    expected = tuple(range(base, base + width))
    if route_group.global_ranks != expected:
        raise ValueError(
            "the TPxCP route group must be exactly one dense-DP replica's "
            f"contiguous MLite rank block; got {route_group.global_ranks}, expected {expected}"
        )


def create_dense_dp_route_group(parallel_state: Any) -> TPxCPRouteGroup:
    """Collectively create one ``TP x CP`` route group per dense-DP replica.

    This helper follows MLite's pinned dense rank decomposition

    ``global_rank = ((dp_rank * cp + cp_rank) * tp + tp_rank)``

    for the first-release ``PP=1`` configuration.  All world ranks must call
    this function once, in the same order.  EP is deliberately absent: it is a
    second view over these ranks and must not enlarge the depth-routing group.
    """

    if not _dist_ready():
        tp_size = int(getattr(parallel_state, "tp_size", 1))
        cp_size = int(getattr(parallel_state, "cp_size", 1))
        dp_size = int(getattr(parallel_state, "dp_size", 1))
        pp_size = int(getattr(parallel_state, "pp_size", 1))
        if (tp_size, cp_size, dp_size, pp_size) == (1, 1, 1, 1):
            return TPxCPRouteGroup.local()
        raise RuntimeError("torch.distributed must be initialized before route groups")

    tp_size = int(getattr(parallel_state, "tp_size", 1))
    cp_size = int(getattr(parallel_state, "cp_size", 1))
    dp_size = int(getattr(parallel_state, "dp_size", 1))
    pp_size = int(getattr(parallel_state, "pp_size", 1))
    if min(tp_size, cp_size, dp_size, pp_size) < 1:
        raise ValueError("parallel-state sizes must all be positive")
    if pp_size != 1:
        raise ValueError("MoR v1 route-group construction requires PP=1")
    expected_world = tp_size * cp_size * dp_size
    if dist.get_world_size() != expected_world:
        raise ValueError(
            "WORLD_SIZE must equal tp_size * cp_size * dense dp_size for PP=1: "
            f"{dist.get_world_size()} != {tp_size} * {cp_size} * {dp_size}"
        )

    global_rank = dist.get_rank()
    own_group: dist.ProcessGroup | None = None
    own_ranks: tuple[int, ...] | None = None
    for dp_rank in range(dp_size):
        ranks = tuple(
            (dp_rank * cp_size + cp_rank) * tp_size + tp_rank
            for cp_rank in range(cp_size)
            for tp_rank in range(tp_size)
        )
        group = dist.new_group(list(ranks))
        if global_rank in ranks:
            own_group = group
            own_ranks = ranks

    if own_group is None or own_ranks is None:
        raise RuntimeError("failed to create the current dense-DP route group")
    result = TPxCPRouteGroup.from_process_group(
        own_group,
        tp_size=tp_size,
        cp_size=cp_size,
        global_ranks=own_ranks,
    )
    expected_dp_rank = global_rank // (tp_size * cp_size)
    configured_dp_rank = int(getattr(parallel_state, "dp_rank", expected_dp_rank))
    if configured_dp_rank != expected_dp_rank:
        raise ValueError(
            "parallel_state.dp_rank disagrees with MLite's dense rank decomposition: "
            f"{configured_dp_rank} != {expected_dp_rank}"
        )
    return result


def _all_gather_variable(
    value: torch.Tensor,
    route_group: TPxCPRouteGroup,
    *,
    known_counts: tuple[int, ...] | None = None,
) -> tuple[torch.Tensor, tuple[int, ...], int]:
    """Gather variable first-dimension tensors in route-rank order."""

    if value.ndim < 1:
        raise ValueError("variable gather needs a leading row dimension")
    if route_group.world_size == 1:
        return value.detach(), (value.size(0),), 0
    if not _dist_ready() or route_group.process_group is None:
        raise RuntimeError("distributed route gather requires an initialized route group")

    if known_counts is None:
        local_count = torch.tensor([value.size(0)], dtype=torch.int64, device=value.device)
        gathered_counts = [torch.empty_like(local_count) for _ in range(route_group.world_size)]
        dist.all_gather(gathered_counts, local_count, group=route_group.process_group)
        counts = tuple(int(item.item()) for item in gathered_counts)
        collective_calls = 1
    else:
        counts = tuple(int(count) for count in known_counts)
        if len(counts) != route_group.world_size or any(count < 0 for count in counts):
            raise ValueError("known_counts must have one non-negative value per route rank")
        if counts[route_group.rank] != value.size(0):
            raise ValueError("known_counts does not match this rank's local row count")
        collective_calls = 0
    maximum = max(counts, default=0)
    if maximum == 0:
        return value.detach().new_empty((0, *value.shape[1:])), counts, collective_calls

    padded = value.detach().new_zeros((maximum, *value.shape[1:]))
    if value.size(0):
        padded[: value.size(0)].copy_(value.detach())
    gathered = [torch.empty_like(padded) for _ in range(route_group.world_size)]
    dist.all_gather(gathered, padded, group=route_group.process_group)
    return (
        torch.cat(
            [item[:count] for item, count in zip(gathered, counts, strict=True)],
            dim=0,
        ),
        counts,
        collective_calls + 1,
    )


class _GlobalValueWithLocalGradient(torch.autograd.Function):
    """Expose one collective value while differentiating a local surrogate.

    ``local + (global - local.detach())`` has the right derivative, but its
    floating-point cancellation can leave TP/CP peers one ULP apart because
    ``local`` differs by shard. Returning the all-reduced value directly makes
    the replicated scalar bitwise identical; backward still follows the
    correctly scaled local numerator.
    """

    @staticmethod
    def forward(
        _ctx: Any,
        differentiable_local_value: torch.Tensor,
        detached_global_value: torch.Tensor,
    ) -> torch.Tensor:
        del differentiable_local_value
        return detached_global_value.clone()

    @staticmethod
    def backward(_ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output, None


def _global_bce_for_mlite(
    local_logits: torch.Tensor,
    local_targets: torch.Tensor,
    route_group: TPxCPRouteGroup,
) -> torch.Tensor:
    """Return the global route mean with MLite-correct local gradients.

    MLite first SUM-reduces replicated sequence-parallel parameters over TP,
    then the distributed optimizer averages dense gradients over ``DP x CP``.
    Consequently each local differentiable numerator must be multiplied by
    ``cp_size`` (and *not* by ``tp_size``).  After TP SUM and DPxCP mean, this
    recovers exactly ``sum(all token losses) / num(all active tokens)`` for a
    dense-DP replica. A value/gradient autograd bridge exposes the all-reduced
    forward scalar bitwise on every peer without putting a collective in the
    autograd graph.
    """

    if local_logits.shape != local_targets.shape:
        raise ValueError("local_logits and local_targets must have the same shape")
    local_sum = F.binary_cross_entropy_with_logits(
        local_logits.float(), local_targets.float(), reduction="sum"
    )
    local_count = torch.tensor(local_logits.numel(), dtype=torch.int64, device=local_logits.device)
    if route_group.world_size == 1:
        return local_sum / local_count.clamp_min(1).to(local_sum.dtype)
    assert route_group.process_group is not None
    global_count = local_count.clone()
    global_sum = local_sum.detach().clone()
    dist.all_reduce(global_count, op=dist.ReduceOp.SUM, group=route_group.process_group)
    dist.all_reduce(global_sum, op=dist.ReduceOp.SUM, group=route_group.process_group)
    denominator = global_count.clamp_min(1).to(local_sum.dtype)
    differentiable = local_sum * route_group.cp_size / denominator
    global_value = global_sum / denominator
    return _GlobalValueWithLocalGradient.apply(differentiable, global_value)


def _route_fingerprint(plan: RoutePlan) -> bytes:
    encoded = json.dumps(
        plan.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).digest()


def assert_route_plan_consistent(
    plan: RoutePlan,
    route_group: TPxCPRouteGroup,
    *,
    device: torch.device | None = None,
) -> None:
    """Raise when TPxCP peers did not reconstruct the exact same RoutePlan."""

    if route_group.world_size == 1:
        return
    assert route_group.process_group is not None
    if device is None:
        backend = dist.get_backend(route_group.process_group)
        device = torch.device("cuda") if backend == "nccl" else torch.device("cpu")
    digest = torch.tensor(list(_route_fingerprint(plan)), dtype=torch.uint8, device=device)
    peer_digests = [torch.empty_like(digest) for _ in range(route_group.world_size)]
    dist.all_gather(peer_digests, digest, group=route_group.process_group)
    if any(not torch.equal(digest, other) for other in peer_digests):
        raise RuntimeError(
            "TPxCP peers reconstructed different depth RoutePlans; check router "
            "metadata, original lengths, and replay inputs"
        )


@dataclass(frozen=True)
class DistributedDepthRouterOutput:
    """Global discrete route plus the current rank's differentiable rows."""

    raw_logits: torch.Tensor
    scores: torch.Tensor
    selected_local_indices: torch.Tensor
    selected_local_mask: torch.Tensor
    selected_local_gates: torch.Tensor
    selected_batch: ActiveTokenBatch
    aux_loss: torch.Tensor
    weighted_aux_loss: torch.Tensor
    plan: RoutePlan
    route_gathers: int
    route_collective_calls: int


def _resolve_round(router: DepthRouter, round_index: int | None) -> int:
    resolved = router.round_index if round_index is None else round_index
    if resolved is None:
        raise ValueError("round_index must be supplied to the router or route call")
    router.architecture.capacity_for_round(resolved)
    return int(resolved)


def _local_original_lengths(
    sample_ids: torch.Tensor,
    original_lengths: Mapping[int, int] | torch.Tensor,
) -> torch.Tensor:
    values: list[int] = []
    for sample_id_value in sample_ids.detach().cpu().tolist():
        sample_id = int(sample_id_value)
        if isinstance(original_lengths, torch.Tensor):
            if original_lengths.ndim != 1 or not 0 <= sample_id < original_lengths.numel():
                raise ValueError(f"no original length is available for sample {sample_id}")
            length = int(original_lengths[sample_id].item())
        else:
            if sample_id not in original_lengths:
                raise ValueError(f"no original length is available for sample {sample_id}")
            length = original_lengths[sample_id]
        if isinstance(length, bool) or not isinstance(length, int) or length < 0:
            raise ValueError(f"original length for sample {sample_id} must be non-negative")
        values.append(length)
    return torch.tensor(values, dtype=torch.int64, device=sample_ids.device)


def _selected_local_rows(
    local_global_ids: torch.Tensor,
    selected_global_ids: torch.Tensor,
) -> torch.Tensor:
    """Map selected IDs to rows while preserving the physical local layout."""

    selected = set(selected_global_ids.detach().cpu().tolist())
    selected_rows = [
        row
        for row, token_id in enumerate(local_global_ids.detach().cpu().tolist())
        if int(token_id) in selected
    ]
    return torch.tensor(selected_rows, dtype=torch.int64, device=local_global_ids.device)


def distributed_depth_route(
    router: DepthRouter,
    batch: ActiveTokenBatch,
    *,
    original_lengths: Mapping[int, int] | torch.Tensor,
    route_group: TPxCPRouteGroup | None = None,
    round_index: int | None = None,
    replay_plan: RoutePlan | None = None,
    verify_peer_consistency: bool = True,
    counters: CommunicationCounters | None = None,
    logit_bias: torch.Tensor | None = None,
) -> DistributedDepthRouterOutput:
    """Route one active set within exactly one dense-DP replica.

    ``selected_local_indices`` and ``selected_local_mask`` are coordinates in
    the *input batch*, including any padding rows.  ``batch`` may contain
    synthetic tail padding; padding is excluded from the score gather, BCE
    denominator, selection, and returned active batch.
    Candidates should be the previous round's selected batch, which makes the
    learned active sets strictly nested.  Replay similarly fails if any oracle
    token is absent from this candidate set.
    """

    route_group = route_group or TPxCPRouteGroup.local()
    _assert_dense_dp_local_membership(route_group)
    if batch.hidden.ndim != 2 or batch.hidden.size(-1) != router.hidden_size:
        raise ValueError(
            f"batch.hidden must have shape [tokens, {router.hidden_size}], "
            f"got {tuple(batch.hidden.shape)}"
        )
    if batch.layout.replicated and route_group.world_size > 1:
        raise ValueError("distributed depth routing requires route-sharded, not replicated, tokens")
    real_mask = ~batch.layout.padding_mask
    real_layout = batch.layout.index_select(real_mask)
    real_layout.assert_compute_ready(local_route_rank=route_group.rank)
    if real_layout.num_tokens and torch.any(real_layout.global_token_ids < 0):
        raise ValueError(
            "real global token IDs must be non-negative; negatives are reserved for padding"
        )

    resolved_round = _resolve_round(router, round_index)
    # Preserve the autograd connection even if model.to(BF16) cast the resident
    # parameter.  Selection uses the detached FP32 scores gathered below.
    raw_logits = F.linear(batch.hidden.float(), router.proj.weight.float()).squeeze(-1)
    if logit_bias is not None:
        if logit_bias.shape != (batch.num_tokens,):
            raise ValueError("logit_bias must align with the local active-token buffer")
        logit_bias = logit_bias.to(device=batch.hidden.device, dtype=torch.float32)
        if not torch.isfinite(logit_bias[real_mask]).all():
            raise ValueError("real-token logit_bias values must be finite")
        raw_logits = raw_logits + logit_bias
    decision_logits = raw_logits / router.config.temperature
    scores = torch.sigmoid(decision_logits) * router.config.alpha
    local_real_rows = torch.nonzero(real_mask, as_tuple=False).flatten()
    local_scores = scores.index_select(0, local_real_rows)
    local_lengths = _local_original_lengths(real_layout.sample_ids, original_lengths)

    gathered_scores, counts, score_gather_calls = _all_gather_variable(
        local_scores.detach().to(dtype=torch.float32), route_group
    )
    route_rank_column = torch.full(
        (real_layout.num_tokens,),
        route_group.rank,
        dtype=torch.int64,
        device=batch.hidden.device,
    )
    current_local_rows = local_real_rows.to(dtype=torch.int64)
    metadata = torch.stack(
        [
            real_layout.sample_ids.to(dtype=torch.int64),
            real_layout.position_ids.to(dtype=torch.int64),
            real_layout.global_token_ids.to(dtype=torch.int64),
            real_layout.source_route_ranks.to(dtype=torch.int64),
            real_layout.source_local_rows.to(dtype=torch.int64),
            route_rank_column,
            current_local_rows,
            local_lengths,
            torch.full_like(route_rank_column, resolved_round),
        ],
        dim=1,
    )
    gathered_metadata, metadata_counts, metadata_gather_calls = _all_gather_variable(
        metadata, route_group, known_counts=counts
    )
    if counts != metadata_counts or gathered_metadata.shape != (
        gathered_scores.numel(),
        _INT_FIELDS,
    ):
        raise RuntimeError("score and metadata route gathers produced inconsistent row counts")
    if gathered_metadata.numel() and not bool(
        (gathered_metadata[:, 8] == resolved_round).all().item()
    ):
        raise RuntimeError("TPxCP peers entered different recurrent rounds")

    global_sample_ids = gathered_metadata[:, 0]
    global_positions = gathered_metadata[:, 1]
    global_ids = gathered_metadata[:, 2]
    if torch.unique(global_ids).numel() != global_ids.numel():
        raise ValueError(
            "global token IDs must be unique inside one dense-DP TPxCP route group; "
            "a replicated or cross-DP group was likely supplied"
        )
    if gathered_metadata.numel() and (
        int(gathered_metadata[:, 3].min().item()) < 0
        or int(gathered_metadata[:, 3].max().item()) >= route_group.world_size
        or int(gathered_metadata[:, 4].min().item()) < 0
    ):
        raise ValueError("real tokens contain invalid original source ownership metadata")
    global_original_lengths: dict[int, int] = {}
    for sample_id_value in sorted(set(global_sample_ids.detach().cpu().tolist())):
        sample_id = int(sample_id_value)
        lengths = torch.unique(gathered_metadata[global_sample_ids == sample_id, 7])
        if lengths.numel() != 1:
            raise ValueError(
                f"TPxCP peers supplied inconsistent original lengths for sample {sample_id}"
            )
        global_original_lengths[sample_id] = int(lengths.item())

    mode = "learned"
    if replay_plan is None:
        selection = select_expert_choice_per_sample(
            gathered_scores,
            sample_ids=global_sample_ids,
            original_positions=global_positions,
            global_token_ids=global_ids,
            original_lengths=global_original_lengths,
            architecture=router.architecture,
            round_index=resolved_round,
        )
        chosen_global_rows = selection.selected_indices
        active_cu_seqlens = selection.active_cu_seqlens
        margins = dict(selection.cutoff_score_margins)
    else:
        if replay_plan.round_index != resolved_round:
            raise ValueError(
                f"replay round {replay_plan.round_index} does not match {resolved_round}"
            )
        chosen_global_rows = replay_plan.replay_indices(global_ids)
        expected_real = ~replay_plan.padding_mask.to(device=global_ids.device)
        replay_ids = replay_plan.global_token_ids.to(global_ids.device)[expected_real]
        replay_samples = replay_plan.sample_ids.to(global_ids.device)[expected_real]
        replay_positions = replay_plan.original_positions.to(global_ids.device)[expected_real]
        if not torch.equal(global_ids[chosen_global_rows], replay_ids):
            raise RuntimeError("replay token order did not map exactly onto gathered candidates")
        samples_match = torch.equal(global_sample_ids[chosen_global_rows], replay_samples)
        positions_match = torch.equal(global_positions[chosen_global_rows], replay_positions)
        if not samples_match or not positions_match:
            raise ValueError("replay sample/position metadata differs from the oracle RoutePlan")
        validate_replay_capacity(
            replay_plan,
            candidate_samples=global_sample_ids,
            selected_samples=replay_samples,
            original_lengths=global_original_lengths,
            architecture=router.architecture,
            round_index=resolved_round,
        )
        active_cu_seqlens = replay_plan.active_cu_seqlens.to(global_ids.device)
        margins = dict(replay_plan.cutoff_score_margins)
        mode = "replay"

    selected_global_ids = global_ids.index_select(0, chosen_global_rows)
    selected_local_indices = _selected_local_rows(
        batch.layout.global_token_ids[real_mask], selected_global_ids
    )
    # The lookup above is relative to the compact real shard; map it back to the
    # original possibly padded local buffer.
    selected_local_indices = local_real_rows.index_select(0, selected_local_indices)
    selected_local_mask = torch.zeros(
        batch.num_tokens, dtype=torch.bool, device=batch.hidden.device
    )
    selected_local_mask[selected_local_indices] = True
    local_targets = selected_local_mask[real_mask].to(dtype=raw_logits.dtype)
    aux_loss = _global_bce_for_mlite(decision_logits[real_mask], local_targets, route_group)
    selected_local_gates = scores.index_select(0, selected_local_indices)
    if replay_plan is not None:
        local_selected_ids = batch.layout.global_token_ids.index_select(0, selected_local_indices)
        replay_gates = replay_plan.replay_gates(local_selected_ids)
        selected_local_gates = replay_selected_gates(selected_local_gates, replay_gates)

    selected_local = batch.index_select(selected_local_indices).with_round(resolved_round)
    selected_batch = ActiveTokenBatch(
        hidden=selected_local.hidden,
        gates=selected_local_gates,
        layout=selected_local.layout,
    )

    selected_metadata = gathered_metadata.index_select(0, chosen_global_rows)
    source_route_ranks = selected_metadata[:, 3]
    source_tp_ranks = source_route_ranks.remainder(route_group.tp_size)
    source_cp_ranks = torch.div(source_route_ranks, route_group.tp_size, rounding_mode="floor")
    current_route_ranks = selected_metadata[:, 5]
    target_tp_ranks = current_route_ranks.remainder(route_group.tp_size)
    target_cp_ranks = torch.div(current_route_ranks, route_group.tp_size, rounding_mode="floor")
    plan = RoutePlan(
        round_index=resolved_round,
        mode=mode,
        sample_ids=selected_metadata[:, 0],
        original_positions=selected_metadata[:, 1],
        global_token_ids=selected_metadata[:, 2],
        source_tp_ranks=source_tp_ranks,
        source_cp_ranks=source_cp_ranks,
        source_local_rows=selected_metadata[:, 4],
        # Until a CP/Magi layout planner rewrites them, targets describe the
        # current physical ownership at this routing boundary.
        target_tp_ranks=target_tp_ranks,
        target_cp_ranks=target_cp_ranks,
        target_local_rows=selected_metadata[:, 6],
        selected_gates=(
            replay_plan.replay_gates(selected_global_ids).detach()
            if replay_plan is not None
            else gathered_scores.index_select(0, chosen_global_rows).detach()
        ),
        active_cu_seqlens=active_cu_seqlens,
        padding_mask=torch.zeros(
            chosen_global_rows.numel(), dtype=torch.bool, device=batch.hidden.device
        ),
        cutoff_score_margins=margins,
    )
    if verify_peer_consistency:
        assert_route_plan_consistent(plan, route_group, device=batch.hidden.device)

    route_gathers = int(route_group.world_size > 1)
    route_collective_calls = 0
    if route_group.world_size > 1:
        # Two scalar all-reduces form the value/count BCE normalization; the
        # optional peer digest adds one final all-gather.
        route_collective_calls = (
            score_gather_calls + metadata_gather_calls + 2 + int(verify_peer_consistency)
        )
    if counters is not None:
        counters.route_gathers += route_gathers
        counters.collective_calls += route_collective_calls

    return DistributedDepthRouterOutput(
        raw_logits=raw_logits,
        scores=scores,
        selected_local_indices=selected_local_indices,
        selected_local_mask=selected_local_mask,
        selected_local_gates=selected_local_gates,
        selected_batch=selected_batch,
        aux_loss=aux_loss,
        weighted_aux_loss=aux_loss * router.config.aux_loss_coef,
        plan=plan,
        route_gathers=route_gathers,
        route_collective_calls=route_collective_calls,
    )


__all__ = [
    "DistributedDepthRouterOutput",
    "assert_route_plan_consistent",
    "create_dense_dp_route_group",
    "distributed_depth_route",
]
