"""Canonical active-sequence metadata and deterministic tail padding.

Filtering preserves every real token's original position.  Synthetic rows are
added only at the end of each active sample so tensor sequence parallelism and
MagiAttention can split the packed token axis without changing causal order.
The rows in this module are metadata only; hidden-state padding is materialized
after the differentiable active-token dispatch.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from mor_mlite.routing import RoutePlan

_INT_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}
_DUMMY_ID_STRIDE = 1 << 48


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def active_sequence_alignment(
    *,
    tp_size: int,
    cp_size: int,
    use_magi: bool | None = None,
) -> int:
    """Return the per-sample recurrent-buffer alignment.

    CP=1 needs only TP sequence-parallel divisibility.  The supported CP>1
    path is MagiAttention, whose head/tail chunk layout requires ``2 * CP`` in
    addition to the TP split.
    """

    tp_size = _positive_int("tp_size", tp_size)
    cp_size = _positive_int("cp_size", cp_size)
    if use_magi is not None and not isinstance(use_magi, bool):
        raise TypeError("use_magi must be a bool or None")
    if use_magi is None:
        use_magi = cp_size > 1
    if cp_size > 1 and not use_magi:
        raise ValueError("CP>1 active packing requires the MagiAttention layout")
    if cp_size == 1:
        return tp_size
    return tp_size * 2 * cp_size


def _stable_canonical_order(
    sample_ids: torch.Tensor,
    position_ids: torch.Tensor,
    global_token_ids: torch.Tensor,
) -> torch.Tensor:
    order = torch.argsort(global_token_ids, stable=True)
    order = order[torch.argsort(position_ids[order], stable=True)]
    return order[torch.argsort(sample_ids[order], stable=True)]


def _original_length(
    sample_id: int,
    original_lengths: Mapping[int, int] | torch.Tensor | None,
    positions: torch.Tensor,
) -> int:
    inferred = int(positions.max().item()) + 1 if positions.numel() else 0
    if original_lengths is None:
        return inferred
    if isinstance(original_lengths, torch.Tensor):
        if original_lengths.ndim != 1 or not 0 <= sample_id < original_lengths.numel():
            raise ValueError(f"no original length is available for sample {sample_id}")
        result = int(original_lengths[sample_id].item())
    else:
        if sample_id not in original_lengths:
            raise ValueError(f"no original length is available for sample {sample_id}")
        result = original_lengths[sample_id]
    if isinstance(result, bool) or not isinstance(result, int) or result < 0:
        raise ValueError(f"original length for sample {sample_id} must be non-negative")
    if inferred > result:
        raise ValueError(
            f"sample {sample_id} has original position {inferred - 1} outside length {result}"
        )
    return result


@dataclass(frozen=True, slots=True)
class CanonicalActivePacking:
    """Padded canonical metadata for one recurrent round.

    ``source_rows`` maps every real canonical row back to the input metadata
    row and is ``-1`` for dummies.  ``cu_seqlens`` describes padded segments;
    ``active_cu_seqlens`` describes the same samples before padding.
    """

    sample_ids: torch.Tensor
    position_ids: torch.Tensor
    global_token_ids: torch.Tensor
    padding_mask: torch.Tensor
    source_rows: torch.Tensor
    cu_seqlens: torch.Tensor
    active_cu_seqlens: torch.Tensor
    alignment: int
    round_index: int

    def __post_init__(self) -> None:
        vectors = (
            self.sample_ids,
            self.position_ids,
            self.global_token_ids,
            self.padding_mask,
            self.source_rows,
        )
        length = self.sample_ids.numel()
        if any(value.ndim != 1 or value.numel() != length for value in vectors):
            raise ValueError("canonical active metadata must be equally-sized 1-D tensors")
        if any(value.device != self.sample_ids.device for value in vectors):
            raise ValueError("canonical active metadata must reside on one device")
        for value in (self.sample_ids, self.position_ids, self.global_token_ids, self.source_rows):
            if value.dtype not in _INT_DTYPES:
                raise TypeError("canonical identity metadata must use integer tensors")
        if self.padding_mask.dtype != torch.bool:
            raise TypeError("padding_mask must be boolean")
        if self.cu_seqlens.dtype not in {torch.int32, torch.int64} or self.cu_seqlens.ndim != 1:
            raise TypeError("cu_seqlens must be a one-dimensional int tensor")
        if (
            self.active_cu_seqlens.dtype not in {torch.int32, torch.int64}
            or self.active_cu_seqlens.ndim != 1
        ):
            raise TypeError("active_cu_seqlens must be a one-dimensional int tensor")
        if (
            self.cu_seqlens.device != self.sample_ids.device
            or self.active_cu_seqlens.device != self.sample_ids.device
        ):
            raise ValueError("cu_seqlens tensors must share the metadata device")
        if self.cu_seqlens.numel() != self.active_cu_seqlens.numel():
            raise ValueError("padded and active cu_seqlens must describe the same samples")
        for name, cu, expected in (
            ("cu_seqlens", self.cu_seqlens, length),
            ("active_cu_seqlens", self.active_cu_seqlens, int((~self.padding_mask).sum().item())),
        ):
            if cu.numel() == 0 or int(cu[0].item()) != 0:
                raise ValueError(f"{name} must start at zero")
            if bool((cu[1:] < cu[:-1]).any().item()):
                raise ValueError(f"{name} must be monotonically non-decreasing")
            if int(cu[-1].item()) != expected:
                raise ValueError(f"{name} has an incorrect final offset")
        _positive_int("alignment", self.alignment)
        if (
            isinstance(self.round_index, bool)
            or not isinstance(self.round_index, int)
            or self.round_index < 0
        ):
            raise ValueError("round_index must be a non-negative integer")
        if torch.unique(self.global_token_ids).numel() != length:
            raise ValueError("real and dummy global token IDs must all be unique")
        if bool((self.global_token_ids[self.padding_mask] >= 0).any().item()):
            raise ValueError("dummy global token IDs must be negative")
        if bool((self.global_token_ids[~self.padding_mask] < 0).any().item()):
            raise ValueError("real global token IDs must be non-negative")
        if bool((self.source_rows[self.padding_mask] != -1).any().item()):
            raise ValueError("dummy source_rows must be -1")
        if bool((self.source_rows[~self.padding_mask] < 0).any().item()):
            raise ValueError("real source_rows must be non-negative")
        real_source_rows = self.source_rows[~self.padding_mask]
        if torch.unique(real_source_rows).numel() != real_source_rows.numel():
            raise ValueError("real source_rows must map one-to-one onto input rows")

        previous_sample_id: int | None = None
        for segment in range(self.cu_seqlens.numel() - 1):
            begin = int(self.cu_seqlens[segment].item())
            end = int(self.cu_seqlens[segment + 1].item())
            if (end - begin) % self.alignment:
                raise ValueError("every padded active sequence must satisfy alignment")
            segment_padding = self.padding_mask[begin:end]
            if segment_padding.numel():
                first_padding = torch.nonzero(segment_padding, as_tuple=False)
                if first_padding.numel():
                    first = int(first_padding[0].item())
                    if not bool(segment_padding[first:].all().item()):
                        raise ValueError("dummy padding may appear only at a sequence tail")
            segment_samples = self.sample_ids[begin:end]
            if segment_samples.numel() and not bool(
                (segment_samples == segment_samples[0]).all().item()
            ):
                raise ValueError("a padded sequence segment must contain one sample ID")
            if segment_samples.numel():
                sample_id = int(segment_samples[0].item())
                if previous_sample_id is not None and sample_id <= previous_sample_id:
                    raise ValueError("canonical sample segments must be strictly ID-ordered")
                previous_sample_id = sample_id
            segment_positions = self.position_ids[begin:end]
            if segment_positions.numel() > 1 and not bool(
                (segment_positions[1:] > segment_positions[:-1]).all().item()
            ):
                raise ValueError("positions within a canonical sample must be strictly increasing")
            active_begin = int(self.active_cu_seqlens[segment].item())
            active_end = int(self.active_cu_seqlens[segment + 1].item())
            if active_end - active_begin != int((~segment_padding).sum().item()):
                raise ValueError("active_cu_seqlens disagrees with per-sample padding")

    @property
    def num_rows(self) -> int:
        return self.global_token_ids.numel()

    @property
    def num_active_tokens(self) -> int:
        return int((~self.padding_mask).sum().item())

    @property
    def canonical_sample_ids(self) -> torch.Tensor:
        return self.sample_ids

    @property
    def canonical_position_ids(self) -> torch.Tensor:
        return self.position_ids

    @property
    def canonical_global_token_ids(self) -> torch.Tensor:
        return self.global_token_ids

    @property
    def canonical_padding_mask(self) -> torch.Tensor:
        return self.padding_mask

    @property
    def real_rows(self) -> torch.Tensor:
        return torch.nonzero(~self.padding_mask, as_tuple=False).flatten()

    def strip_padding(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim < 1 or value.size(0) != self.num_rows:
            raise ValueError("value must have the canonical padded token axis first")
        return value.index_select(0, self.real_rows)

    def magi_decode_kwargs(self) -> dict[str, torch.Tensor]:
        """Return keyword arguments accepted by ``decode_magi_direct_plan``."""

        return {
            "canonical_sample_ids": self.sample_ids,
            "canonical_position_ids": self.position_ids,
            "canonical_global_token_ids": self.global_token_ids,
            "canonical_padding_mask": self.padding_mask,
        }

    def to(self, device: torch.device | str) -> CanonicalActivePacking:
        return type(self)(
            sample_ids=self.sample_ids.to(device),
            position_ids=self.position_ids.to(device),
            global_token_ids=self.global_token_ids.to(device),
            padding_mask=self.padding_mask.to(device),
            source_rows=self.source_rows.to(device),
            cu_seqlens=self.cu_seqlens.to(device),
            active_cu_seqlens=self.active_cu_seqlens.to(device),
            alignment=self.alignment,
            round_index=self.round_index,
        )


def pack_active_sequences(
    *,
    sample_ids: torch.Tensor,
    original_positions: torch.Tensor,
    global_token_ids: torch.Tensor,
    tp_size: int,
    cp_size: int,
    use_magi: bool | None = None,
    original_lengths: Mapping[int, int] | torch.Tensor | None = None,
    round_index: int = 0,
) -> CanonicalActivePacking:
    """Canonicalize real active tokens and append deterministic tail dummies."""

    vectors = (sample_ids, original_positions, global_token_ids)
    length = sample_ids.numel()
    if any(value.ndim != 1 or value.numel() != length for value in vectors):
        raise ValueError("active identity metadata must be equally-sized 1-D tensors")
    if any(value.device != sample_ids.device for value in vectors):
        raise ValueError("active identity metadata must reside on one device")
    if any(value.dtype not in _INT_DTYPES for value in vectors):
        raise TypeError("active identity metadata must use integer tensors")
    if isinstance(round_index, bool) or not isinstance(round_index, int) or round_index < 0:
        raise ValueError("round_index must be a non-negative integer")
    if bool((sample_ids < 0).any().item()) or bool((original_positions < 0).any().item()):
        raise ValueError("real sample IDs and original positions must be non-negative")
    if bool((global_token_ids < 0).any().item()):
        raise ValueError("real global token IDs must be non-negative")
    if torch.unique(global_token_ids).numel() != length:
        raise ValueError("real global token IDs must be unique")

    alignment = active_sequence_alignment(tp_size=tp_size, cp_size=cp_size, use_magi=use_magi)
    order = _stable_canonical_order(sample_ids, original_positions, global_token_ids)
    ordered_samples = sample_ids.to(dtype=torch.int64).index_select(0, order)
    ordered_positions = original_positions.to(dtype=torch.int64).index_select(0, order)
    ordered_ids = global_token_ids.to(dtype=torch.int64).index_select(0, order)

    sample_chunks: list[torch.Tensor] = []
    position_chunks: list[torch.Tensor] = []
    id_chunks: list[torch.Tensor] = []
    padding_chunks: list[torch.Tensor] = []
    source_chunks: list[torch.Tensor] = []
    padded_cu = [0]
    active_cu = [0]
    dummy_cursor = 0
    device = sample_ids.device
    sample_order = sorted(set(ordered_samples.detach().cpu().tolist()))
    for sample_id_value in sample_order:
        sample_id = int(sample_id_value)
        rows = torch.nonzero(ordered_samples == sample_id, as_tuple=False).flatten()
        positions = ordered_positions.index_select(0, rows)
        if torch.unique(positions).numel() != positions.numel():
            raise ValueError(f"sample {sample_id} contains duplicate original positions")
        count = rows.numel()
        pad_count = (-count) % alignment
        original_length = _original_length(sample_id, original_lengths, positions)

        sample_chunks.append(ordered_samples.index_select(0, rows))
        position_chunks.append(positions)
        id_chunks.append(ordered_ids.index_select(0, rows))
        padding_chunks.append(torch.zeros(count, dtype=torch.bool, device=device))
        source_chunks.append(order.index_select(0, rows).to(dtype=torch.int64))
        if pad_count:
            sample_chunks.append(
                torch.full((pad_count,), sample_id, dtype=torch.int64, device=device)
            )
            position_chunks.append(
                torch.arange(
                    original_length,
                    original_length + pad_count,
                    dtype=torch.int64,
                    device=device,
                )
            )
            # Rounds receive disjoint deterministic negative ID ranges.  This
            # makes trace comparisons independent of the input shard order.
            first_dummy = -1 - round_index * _DUMMY_ID_STRIDE - dummy_cursor
            id_chunks.append(
                torch.arange(
                    first_dummy,
                    first_dummy - pad_count,
                    -1,
                    dtype=torch.int64,
                    device=device,
                )
            )
            padding_chunks.append(torch.ones(pad_count, dtype=torch.bool, device=device))
            source_chunks.append(torch.full((pad_count,), -1, dtype=torch.int64, device=device))
            dummy_cursor += pad_count
        active_cu.append(active_cu[-1] + count)
        padded_cu.append(padded_cu[-1] + count + pad_count)

    def concatenate(chunks: list[torch.Tensor], *, dtype: torch.dtype) -> torch.Tensor:
        if chunks:
            return torch.cat(chunks, dim=0)
        return torch.empty(0, dtype=dtype, device=device)

    return CanonicalActivePacking(
        sample_ids=concatenate(sample_chunks, dtype=torch.int64),
        position_ids=concatenate(position_chunks, dtype=torch.int64),
        global_token_ids=concatenate(id_chunks, dtype=torch.int64),
        padding_mask=concatenate(padding_chunks, dtype=torch.bool),
        source_rows=concatenate(source_chunks, dtype=torch.int64),
        cu_seqlens=torch.tensor(padded_cu, dtype=torch.int32, device=device),
        active_cu_seqlens=torch.tensor(active_cu, dtype=torch.int32, device=device),
        alignment=alignment,
        round_index=round_index,
    )


def pack_route_plan_canonical(
    plan: RoutePlan,
    *,
    tp_size: int,
    cp_size: int,
    use_magi: bool | None = None,
    original_lengths: Mapping[int, int] | torch.Tensor | None = None,
) -> CanonicalActivePacking:
    """Create canonical recurrent metadata from the real rows of a RoutePlan."""

    real = ~plan.padding_mask
    result = pack_active_sequences(
        sample_ids=plan.sample_ids[real],
        original_positions=plan.original_positions[real],
        global_token_ids=plan.global_token_ids[real],
        tp_size=tp_size,
        cp_size=cp_size,
        use_magi=use_magi,
        original_lengths=original_lengths,
        round_index=plan.round_index,
    )
    expected_cu = plan.active_cu_seqlens.to(result.active_cu_seqlens.device)
    if not torch.equal(result.active_cu_seqlens, expected_cu):
        raise ValueError("RoutePlan active_cu_seqlens disagrees with its sample/token metadata")
    return result


__all__ = [
    "CanonicalActivePacking",
    "active_sequence_alignment",
    "pack_active_sequences",
    "pack_route_plan_canonical",
]
