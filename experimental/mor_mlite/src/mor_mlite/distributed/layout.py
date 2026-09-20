"""Tensor-only metadata carried with active recurrent tokens."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace

import torch

_INT_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}


def _as_index(index: torch.Tensor, *, length: int) -> torch.Tensor:
    if index.dtype == torch.bool:
        if index.dim() != 1 or index.numel() != length:
            raise ValueError(f"boolean index must have shape [{length}]")
        return torch.where(index)[0]
    if index.dtype not in _INT_DTYPES or index.dim() != 1:
        raise TypeError("token index must be a one-dimensional integer or boolean tensor")
    return index.to(dtype=torch.long)


def _stable_lexicographic_order(*keys: torch.Tensor) -> torch.Tensor:
    """Return stable lexicographic order for equally-sized one-dimensional keys."""

    if not keys:
        raise ValueError("at least one sort key is required")
    length = keys[0].numel()
    if any(key.dim() != 1 or key.numel() != length for key in keys):
        raise ValueError("all lexicographic keys must be one-dimensional and equally sized")
    order = torch.arange(length, device=keys[0].device)
    for key in reversed(keys):
        order = order[torch.argsort(key.index_select(0, order), stable=True)]
    return order


@dataclass(frozen=True, slots=True)
class ActiveTokenLayout:
    """Ownership and semantic identity of a flat active-token tensor.

    ``padding_mask=True`` always means a synthetic/padding row.  Active compute
    and dispatch reject such rows; padding may exist only in a full canonical
    source tensor before ``without_padding`` is called.

    ``source_route_ranks`` and ``source_local_rows`` never change.  They are the
    inverse route used by the final merge. ``current_route_ranks`` describes the
    placement tensor placement and must equal the local route rank for a normal
    sharded batch.  A Magi canonical batch is explicitly marked ``replicated``.
    """

    sample_ids: torch.Tensor
    position_ids: torch.Tensor
    global_token_ids: torch.Tensor
    source_route_ranks: torch.Tensor
    source_local_rows: torch.Tensor
    current_route_ranks: torch.Tensor
    padding_mask: torch.Tensor
    destination_slots: torch.Tensor | None = None
    round_index: int = 0
    layout_kind: str = "route_sharded"
    replicated: bool = False

    def __post_init__(self) -> None:
        tensors = (
            self.sample_ids,
            self.position_ids,
            self.global_token_ids,
            self.source_route_ranks,
            self.source_local_rows,
            self.current_route_ranks,
            self.padding_mask,
        )
        length = self.sample_ids.numel()
        for tensor in tensors:
            if tensor.dim() != 1 or tensor.numel() != length:
                raise ValueError("all ActiveTokenLayout fields must be equally-sized 1-D tensors")
            if tensor.device != self.sample_ids.device:
                raise ValueError("all ActiveTokenLayout tensors must be on the same device")
        for tensor in tensors[:-1]:
            if tensor.dtype not in _INT_DTYPES:
                raise TypeError("identity and rank metadata must use integer tensors")
        if self.padding_mask.dtype != torch.bool:
            raise TypeError("padding_mask must be boolean, with True denoting padding")
        if self.destination_slots is not None and (
            self.destination_slots.dim() != 1
            or self.destination_slots.numel() != length
            or self.destination_slots.dtype not in _INT_DTYPES
            or self.destination_slots.device != self.sample_ids.device
        ):
            raise ValueError("destination_slots must be an equally-sized integer tensor")
        if self.round_index < 0:
            raise ValueError("round_index must be non-negative")
        if not self.layout_kind:
            raise ValueError("layout_kind must be non-empty")

    @classmethod
    def from_local(
        cls,
        *,
        sample_ids: torch.Tensor,
        position_ids: torch.Tensor,
        route_rank: int = 0,
        global_token_ids: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        round_index: int = 0,
        drop_padding: bool = True,
    ) -> ActiveTokenLayout:
        if sample_ids.dim() != 1 or position_ids.shape != sample_ids.shape:
            raise ValueError("sample_ids and position_ids must be equally-sized 1-D tensors")
        device = sample_ids.device
        length = sample_ids.numel()
        local_rows = torch.arange(length, dtype=torch.long, device=device)
        if global_token_ids is None:
            # Unique for every source row in any practical route-group size.
            global_token_ids = local_rows + (int(route_rank) << 32)
        if padding_mask is None:
            padding_mask = torch.zeros(length, dtype=torch.bool, device=device)
        rank_vector = torch.full((length,), int(route_rank), dtype=torch.long, device=device)
        result = cls(
            sample_ids=sample_ids.to(dtype=torch.long),
            position_ids=position_ids.to(dtype=torch.long),
            global_token_ids=global_token_ids.to(device=device, dtype=torch.long),
            source_route_ranks=rank_vector,
            source_local_rows=local_rows,
            current_route_ranks=rank_vector.clone(),
            padding_mask=padding_mask.to(device=device, dtype=torch.bool),
            round_index=round_index,
        )
        return result.without_padding() if drop_padding else result

    @property
    def num_tokens(self) -> int:
        return self.sample_ids.numel()

    @property
    def device(self) -> torch.device:
        return self.sample_ids.device

    def index_select(self, index: torch.Tensor) -> ActiveTokenLayout:
        index = _as_index(index, length=self.num_tokens)
        return replace(
            self,
            sample_ids=self.sample_ids.index_select(0, index),
            position_ids=self.position_ids.index_select(0, index),
            global_token_ids=self.global_token_ids.index_select(0, index),
            source_route_ranks=self.source_route_ranks.index_select(0, index),
            source_local_rows=self.source_local_rows.index_select(0, index),
            current_route_ranks=self.current_route_ranks.index_select(0, index),
            padding_mask=self.padding_mask.index_select(0, index),
            destination_slots=(
                None
                if self.destination_slots is None
                else self.destination_slots.index_select(0, index)
            ),
        )

    def without_padding(self) -> ActiveTokenLayout:
        return self.index_select(~self.padding_mask)

    def canonical_order(self) -> torch.Tensor:
        return _stable_lexicographic_order(
            self.sample_ids, self.position_ids, self.global_token_ids
        )

    def destination_order(self) -> torch.Tensor:
        if self.destination_slots is None:
            return self.canonical_order()
        return _stable_lexicographic_order(self.destination_slots, self.global_token_ids)

    def with_round(self, round_index: int) -> ActiveTokenLayout:
        return replace(self, round_index=round_index)

    def assert_compute_ready(self, *, local_route_rank: int | None = None) -> None:
        if bool(self.padding_mask.any().item()):
            raise ValueError(
                "padding rows cannot enter active recurrent compute or dispatch; "
                "filter with without_padding() first"
            )
        if not self.replicated and local_route_rank is not None:
            expected = torch.full_like(self.current_route_ranks, int(local_route_rank))
            if not torch.equal(self.current_route_ranks, expected):
                raise ValueError("active batch contains tokens not owned by this route rank")
        if self.num_tokens and torch.unique(self.global_token_ids).numel() != self.num_tokens:
            raise ValueError("global_token_ids must be unique within an active shard")

    @classmethod
    def cat(cls, layouts: Iterable[ActiveTokenLayout]) -> ActiveTokenLayout:
        values = list(layouts)
        if not values:
            raise ValueError("cannot concatenate zero layouts")
        device = values[0].device
        if any(layout.device != device for layout in values):
            raise ValueError("all layouts must use the same device")
        have_slots = [layout.destination_slots is not None for layout in values]
        if any(have_slots) and not all(have_slots):
            raise ValueError("destination_slots must be present on all or none of the layouts")

        def concatenate(name: str) -> torch.Tensor:
            return torch.cat([getattr(layout, name) for layout in values], dim=0)

        kinds = {layout.layout_kind for layout in values}
        rounds = {layout.round_index for layout in values}
        replicated_values = {layout.replicated for layout in values}
        return cls(
            sample_ids=concatenate("sample_ids"),
            position_ids=concatenate("position_ids"),
            global_token_ids=concatenate("global_token_ids"),
            source_route_ranks=concatenate("source_route_ranks"),
            source_local_rows=concatenate("source_local_rows"),
            current_route_ranks=concatenate("current_route_ranks"),
            padding_mask=concatenate("padding_mask"),
            destination_slots=(
                None
                if not all(have_slots)
                else torch.cat([layout.destination_slots for layout in values], dim=0)
            ),
            round_index=next(iter(rounds)) if len(rounds) == 1 else max(rounds),
            layout_kind=next(iter(kinds)) if len(kinds) == 1 else "mixed",
            replicated=next(iter(replicated_values)) if len(replicated_values) == 1 else False,
        )


@dataclass(frozen=True, slots=True)
class ActiveTokenBatch:
    """Active hidden states, optional gates, and their inseparable layout."""

    hidden: torch.Tensor
    layout: ActiveTokenLayout
    gates: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.hidden.dim() < 2:
            raise ValueError("hidden must have a leading token axis and at least one feature axis")
        if self.hidden.size(0) != self.layout.num_tokens:
            raise ValueError("hidden token dimension does not match layout")
        if self.hidden.device != self.layout.device:
            raise ValueError("hidden and layout must be on the same device")
        if self.gates is not None:
            if self.gates.dim() < 1 or self.gates.size(0) != self.layout.num_tokens:
                raise ValueError("gates token dimension does not match layout")
            if self.gates.device != self.hidden.device:
                raise ValueError("gates and hidden must be on the same device")

    @property
    def num_tokens(self) -> int:
        return self.layout.num_tokens

    def index_select(self, index: torch.Tensor) -> ActiveTokenBatch:
        index = _as_index(index, length=self.num_tokens)
        return ActiveTokenBatch(
            hidden=self.hidden.index_select(0, index),
            gates=None if self.gates is None else self.gates.index_select(0, index),
            layout=self.layout.index_select(index),
        )

    def canonicalized(self) -> ActiveTokenBatch:
        return self.index_select(self.layout.canonical_order())

    def with_round(self, round_index: int) -> ActiveTokenBatch:
        return replace(self, layout=self.layout.with_round(round_index))

    def split(self, keep_mask: torch.Tensor) -> tuple[ActiveTokenBatch, ActiveTokenBatch]:
        if keep_mask.dtype != torch.bool or keep_mask.shape != (self.num_tokens,):
            raise ValueError(f"keep_mask must be boolean with shape [{self.num_tokens}]")
        return self.index_select(keep_mask), self.index_select(~keep_mask)

    @classmethod
    def cat(cls, batches: Iterable[ActiveTokenBatch]) -> ActiveTokenBatch:
        values = list(batches)
        if not values:
            raise ValueError("cannot concatenate zero batches")
        have_gates = [batch.gates is not None for batch in values]
        if any(have_gates) and not all(have_gates):
            raise ValueError("gates must be present on all or none of the batches")
        return cls(
            hidden=torch.cat([batch.hidden for batch in values], dim=0),
            gates=(
                None if not all(have_gates) else torch.cat([batch.gates for batch in values], dim=0)
            ),
            layout=ActiveTokenLayout.cat([batch.layout for batch in values]),
        )


__all__ = ["ActiveTokenBatch", "ActiveTokenLayout"]
