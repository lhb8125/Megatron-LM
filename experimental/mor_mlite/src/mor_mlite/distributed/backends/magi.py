"""Explicit adapters for the two valid MagiAttention integration contracts.

MagiAttention's public ``dispatch`` API consumes a canonical, replicated packed
batch and produces CP-local shards.  It does not accept arbitrary TP-SP x CP
local active shards.  ``MagiCanonicalBackend`` models that public contract.

``MagiDirectBackend`` is the optimized extension seam: an integration layer may
translate a Magi runtime plan into exact per-token destination ranks and slots,
then this package performs one differentiable variable all-to-all directly.
The public Magi API currently does not expose that translation, so it is never
guessed here.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import torch

from ..all_to_all import ActiveTokenDispatcher
from ..layout import ActiveTokenBatch, ActiveTokenLayout

_MAGI_MODULE = "megatron.lite.primitive.modules.attention.magi"


def _load_magi_adapter():
    try:
        module = importlib.import_module(_MAGI_MODULE)
    except (ImportError, OSError) as exc:
        raise ImportError(
            "The requested Magi backend requires Megatron-LM Lite pinned to the "
            "framework's recorded commit and a working SandAI-org/MagiAttention "
            "installation. Install MagiAttention for this CUDA architecture, or "
            "use the tiny-only static_reference diagnostic."
        ) from exc
    magi_package = importlib.import_module("magi_attention")
    installed_version = getattr(magi_package, "__version__", None)
    if installed_version != "1.1.1":
        raise RuntimeError(
            "magi_direct's layout decoder is pinned to MagiAttention v1.1.1; "
            f"found {installed_version!r}"
        )
    required = (
        "dispatch_magi_attention_tensor",
        "undispatch_magi_attention_tensor",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise RuntimeError(
            f"The installed MLite Magi adapter is incompatible; missing {missing}. "
            "Use the repository-pinned Megatron-LM commit."
        )
    # Force the optional extension check only when a Magi backend is selected.
    loader = getattr(module, "_load_magi_attention_api", None)
    if loader is not None:
        loader()
    return module


def _load_magi_runtime_manager(runtime_key: Any):
    """Return v1.1.1's cached manager for a public runtime key."""

    _load_magi_adapter()
    try:
        interface = importlib.import_module("magi_attention.api.magi_attn_interface")
        manager = interface.dist_attn_runtime_dict_mgr.get(runtime_key)
    except (AttributeError, ImportError) as exc:
        raise RuntimeError(
            "MagiAttention v1.1.1 runtime-manager layout changed; cannot decode "
            "the direct token placement safely"
        ) from exc
    if manager is None or not hasattr(manager, "dispatch_meta_q"):
        raise ValueError("the Magi runtime key is not present in this process's runtime cache")
    return manager


def _runtime_key(context: dict[str, Any] | None) -> Any:
    runtime_key = None if context is None else context.get("runtime_key")
    if runtime_key is None:
        raise ValueError(
            "Magi dispatch needs context={'runtime_key': key} built from the active "
            "round's exact padded cu_seqlens. A key from a previous active set is invalid."
        )
    pad_size = int(getattr(runtime_key, "pad_size", 0))
    if pad_size:
        raise ValueError(
            f"Magi runtime_key adds pad_size={pad_size}. Exact MoR routing requires "
            "padding-free active dispatch; rebuild the key with a compatible chunk size."
        )
    return runtime_key


class MagiCanonicalBackend:
    """Use Magi's public replicated-canonical-batch dispatch contract."""

    name = "magi_canonical"

    def rebalance(
        self,
        batch: ActiveTokenBatch,
        dispatcher: ActiveTokenDispatcher,
        *,
        target_ranks: torch.Tensor | None = None,
        destination_slots: torch.Tensor | None = None,
        context: dict[str, Any] | None = None,
    ) -> ActiveTokenBatch:
        if target_ranks is not None or destination_slots is not None:
            raise ValueError("magi_canonical obtains placement from its runtime key")
        if not batch.layout.replicated:
            raise ValueError(
                "magi_canonical requires every CP rank to hold the same full, canonical "
                "active packed batch. It cannot consume an arbitrary active local shard. "
                "Canonicalize/gather before this backend or use magi_direct with an "
                "explicit certified plan."
            )
        group = dispatcher.route_group
        if group.tp_size != 1 or group.cp_size <= 1:
            raise ValueError(
                "magi_canonical represents Magi's pre-embedding CP dispatch and therefore "
                "requires a CP-only route group (tp_size=1, cp_size>1). With TP-SP, first "
                "gather the TP token shard and scatter it again after Magi dispatch."
            )
        runtime_key = _runtime_key(context)
        module = _load_magi_adapter()
        dispatch = module.dispatch_magi_attention_tensor
        hidden = dispatch(batch.hidden, runtime_key, pad_value=0.0)
        gates = None if batch.gates is None else dispatch(batch.gates, runtime_key, pad_value=0.0)
        metadata = torch.stack(
            [
                batch.layout.sample_ids,
                batch.layout.position_ids,
                batch.layout.global_token_ids,
                batch.layout.source_route_ranks,
                batch.layout.source_local_rows,
                batch.layout.padding_mask.to(dtype=torch.long),
            ],
            dim=1,
        )
        metadata = dispatch(metadata, runtime_key, pad_value=-1)
        num_tokens = metadata.size(0)
        layout = ActiveTokenLayout(
            sample_ids=metadata[:, 0],
            position_ids=metadata[:, 1],
            global_token_ids=metadata[:, 2],
            source_route_ranks=metadata[:, 3],
            source_local_rows=metadata[:, 4],
            current_route_ranks=torch.full(
                (num_tokens,), group.rank, dtype=torch.long, device=metadata.device
            ),
            padding_mask=metadata[:, 5].bool(),
            destination_slots=torch.arange(num_tokens, dtype=torch.long, device=metadata.device),
            round_index=batch.layout.round_index,
            layout_kind=self.name,
            replicated=False,
        )
        if not bool(layout.padding_mask.any().item()):
            layout.assert_compute_ready(local_route_rank=group.rank)
        dispatcher.counters.magi_dispatches += 1
        dispatcher.counters.record_backend_hidden_rebalance()
        # Magi owns its internal collective schedule; do not pretend it is one
        # torch.distributed collective in the physical counter.
        return ActiveTokenBatch(hidden=hidden, gates=gates, layout=layout)


@dataclass(frozen=True, slots=True)
class MagiDirectPlan:
    """Certified local slice of a Magi v1.1.1 token placement plan."""

    target_ranks: torch.Tensor
    destination_slots: torch.Tensor
    runtime_key: Any
    local_sample_ids: torch.Tensor
    local_position_ids: torch.Tensor
    local_global_token_ids: torch.Tensor
    local_padding_mask: torch.Tensor
    # QKV's sequence-parallel column projection gathers the token dimension
    # across TP before attention.  Consequently every TP peer in one CP shard
    # must pass the *full CP-local* original positions to GQA, while the
    # active hidden/layout vectors above remain TP-SP local.
    attention_position_ids: torch.Tensor
    attention_padding_mask: torch.Tensor
    # Topology-specific placement for every real canonical token.  These
    # vectors let callers materialize an identical, serializable RoutePlan on
    # all TPxCP peers without gathering hidden states.
    global_real_token_ids: torch.Tensor
    global_target_ranks: torch.Tensor
    global_destination_slots: torch.Tensor

    @property
    def local_num_slots(self) -> int:
        return int(self.local_global_token_ids.numel())


def _partition_token_indices(meta: Any, cp_rank: int) -> list[int]:
    indices: list[int] = []
    chunk_size = int(meta.chunk_size)
    actual_sizes = getattr(meta, "chunk_actual_sizes", None)
    for chunk_id in meta.partitions[cp_rank]:
        size = chunk_size if actual_sizes is None else int(actual_sizes[chunk_id])
        start = int(chunk_id) * chunk_size
        indices.extend(range(start, start + size))
    return indices


def decode_magi_direct_plan(
    batch: ActiveTokenBatch,
    *,
    runtime_key: Any,
    route_group,
    canonical_sample_ids: torch.Tensor,
    canonical_position_ids: torch.Tensor,
    canonical_global_token_ids: torch.Tensor,
    canonical_padding_mask: torch.Tensor,
) -> MagiDirectPlan:
    """Decode Magi v1.1.1 chunk partitions into one TP-SP x CP A2A plan.

    The canonical vectors include any per-sequence tail padding used to make
    active packed sequences compatible with TP/CP.  ``batch`` contains real
    selected tokens only.  The decoder reads Magi's cached, deterministic
    ``DispatchMeta.partitions`` and performs no data collective.
    """

    vectors = (
        canonical_sample_ids,
        canonical_position_ids,
        canonical_global_token_ids,
        canonical_padding_mask,
    )
    total = canonical_global_token_ids.numel()
    if any(value.ndim != 1 or value.numel() != total for value in vectors):
        raise ValueError("canonical Magi metadata must be equally-sized 1-D tensors")
    if canonical_padding_mask.dtype != torch.bool:
        raise TypeError("canonical_padding_mask must be boolean")
    if torch.unique(canonical_global_token_ids).numel() != total:
        raise ValueError("canonical global token IDs, including dummy IDs, must be unique")
    manager = _load_magi_runtime_manager(runtime_key)
    meta = manager.dispatch_meta_q
    if int(meta.cp_size) != route_group.cp_size:
        raise ValueError(
            f"Magi runtime CP={meta.cp_size} does not match route CP={route_group.cp_size}"
        )
    if int(meta.total_seqlen) != total:
        raise ValueError(
            f"Magi runtime contains {meta.total_seqlen} tokens, canonical metadata has {total}"
        )

    target_by_id: dict[int, tuple[int, int]] = {}
    # Decode CPU-owned Magi partitions with one bulk copy, never one CUDA
    # scalar read (and synchronization) for each token on every peer.
    canonical_ids_cpu = canonical_global_token_ids.detach().cpu().tolist()
    local_indices: list[int] | None = None
    attention_indices: list[int] | None = None
    for cp_rank in range(route_group.cp_size):
        cp_indices = _partition_token_indices(meta, cp_rank)
        if len(cp_indices) % route_group.tp_size:
            raise ValueError(
                f"Magi CP rank {cp_rank} has {len(cp_indices)} tokens, not divisible "
                f"by TP={route_group.tp_size}"
            )
        tp_tokens = len(cp_indices) // route_group.tp_size
        if cp_rank == route_group.cp_rank:
            attention_indices = cp_indices
        for tp_rank in range(route_group.tp_size):
            route_rank = cp_rank * route_group.tp_size + tp_rank
            begin = tp_rank * tp_tokens
            indices = cp_indices[begin : begin + tp_tokens]
            if route_rank == route_group.rank:
                local_indices = indices
            for destination_slot, canonical_row in enumerate(indices):
                token_id = int(canonical_ids_cpu[canonical_row])
                target_by_id[token_id] = (route_rank, destination_slot)
    if local_indices is None or attention_indices is None:
        raise RuntimeError("failed to decode the current TPxCP route rank")

    targets: list[int] = []
    slots: list[int] = []
    for token_id in batch.layout.global_token_ids.detach().cpu().tolist():
        try:
            target, slot = target_by_id[int(token_id)]
        except KeyError as exc:
            raise ValueError(f"active token ID {token_id} is absent from Magi metadata") from exc
        targets.append(target)
        slots.append(slot)
    device = batch.hidden.device
    local_index_tensor = torch.tensor(local_indices, dtype=torch.long, device=device)
    attention_index_tensor = torch.tensor(attention_indices, dtype=torch.long, device=device)
    real_ids = canonical_global_token_ids[~canonical_padding_mask]
    global_targets = torch.tensor(
        [target_by_id[int(token_id)][0] for token_id in real_ids.detach().cpu().tolist()],
        dtype=torch.long,
        device=device,
    )
    global_slots = torch.tensor(
        [target_by_id[int(token_id)][1] for token_id in real_ids.detach().cpu().tolist()],
        dtype=torch.long,
        device=device,
    )
    return MagiDirectPlan(
        target_ranks=torch.tensor(targets, dtype=torch.long, device=device),
        destination_slots=torch.tensor(slots, dtype=torch.long, device=device),
        runtime_key=runtime_key,
        local_sample_ids=canonical_sample_ids.to(device).index_select(0, local_index_tensor),
        local_position_ids=canonical_position_ids.to(device).index_select(0, local_index_tensor),
        local_global_token_ids=canonical_global_token_ids.to(device).index_select(
            0, local_index_tensor
        ),
        local_padding_mask=canonical_padding_mask.to(device).index_select(0, local_index_tensor),
        attention_position_ids=canonical_position_ids.to(device).index_select(
            0, attention_index_tensor
        ),
        attention_padding_mask=canonical_padding_mask.to(device).index_select(
            0, attention_index_tensor
        ),
        global_real_token_ids=real_ids.to(device),
        global_target_ranks=global_targets,
        global_destination_slots=global_slots,
    )


def _expand_direct_result(
    compact: ActiveTokenBatch,
    direct_plan: MagiDirectPlan,
    *,
    route_rank: int,
) -> ActiveTokenBatch:
    """Insert zero dummy rows into the exact local Magi/TP-SP layout."""

    slots = compact.layout.destination_slots
    if slots is None:
        raise RuntimeError("magi_direct dispatch lost destination slots")
    local_count = direct_plan.local_num_slots
    if slots.numel() and int(slots.max().item()) >= local_count:
        raise ValueError("received Magi destination slot is outside the local buffer")
    expected_ids = direct_plan.local_global_token_ids.index_select(0, slots)
    if not torch.equal(expected_ids, compact.layout.global_token_ids):
        raise RuntimeError("direct A2A result does not match decoded Magi token order")
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
    current_ranks = torch.full_like(source_ranks, int(route_rank))
    layout = ActiveTokenLayout(
        sample_ids=direct_plan.local_sample_ids,
        position_ids=direct_plan.local_position_ids,
        global_token_ids=direct_plan.local_global_token_ids,
        source_route_ranks=source_ranks,
        source_local_rows=source_rows,
        current_route_ranks=current_ranks,
        padding_mask=direct_plan.local_padding_mask,
        destination_slots=torch.arange(local_count, dtype=torch.long, device=compact.hidden.device),
        round_index=compact.layout.round_index,
        layout_kind="magi_direct",
        replicated=False,
    )
    if not torch.equal(layout.padding_mask, source_ranks < 0):
        raise RuntimeError("Magi dummy slots and real A2A receives do not form an exact partition")
    return ActiveTokenBatch(hidden=hidden, gates=gates, layout=layout)


class MagiDirectBackend:
    """One-A2A active transition using an externally decoded Magi plan."""

    name = "magi_direct"

    def rebalance(
        self,
        batch: ActiveTokenBatch,
        dispatcher: ActiveTokenDispatcher,
        *,
        target_ranks: torch.Tensor | None = None,
        destination_slots: torch.Tensor | None = None,
        context: dict[str, Any] | None = None,
    ) -> ActiveTokenBatch:
        direct_plan = None if context is None else context.get("direct_plan")
        if direct_plan is not None:
            if not isinstance(direct_plan, MagiDirectPlan):
                raise TypeError("context['direct_plan'] must be a MagiDirectPlan")
            if target_ranks is not None or destination_slots is not None:
                raise ValueError("pass either direct_plan or explicit placement, not both")
            target_ranks = direct_plan.target_ranks
            destination_slots = direct_plan.destination_slots
            runtime_key = direct_plan.runtime_key
        else:
            runtime_key = _runtime_key(context)
        if target_ranks is None or destination_slots is None:
            raise ValueError(
                "magi_direct requires a certified per-token target_ranks and "
                "destination_slots plan. Magi's public API does not expose this mapping; "
                "use magi_canonical until a pinned-version plan decoder is supplied."
            )
        if target_ranks.shape != (batch.num_tokens,) or destination_slots.shape != (
            batch.num_tokens,
        ):
            raise ValueError("Magi direct placement must contain one entry per local token")
        _runtime_key({"runtime_key": runtime_key})
        _load_magi_adapter()
        compact = dispatcher.dispatch(
            batch,
            target_ranks,
            destination_slots=destination_slots,
            layout_kind=self.name,
        )
        if direct_plan is None:
            return compact
        return _expand_direct_result(compact, direct_plan, route_rank=dispatcher.route_group.rank)


__all__ = [
    "MagiCanonicalBackend",
    "MagiDirectBackend",
    "MagiDirectPlan",
    "decode_magi_direct_plan",
]
