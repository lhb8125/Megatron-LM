"""Deterministic packed batches shared by the reference and MLite paths."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(slots=True)
class PackedBatch:
    """Minimal source-compatible form of MLite's ``PackedBatch``.

    The CLI converts this object to MLite's native contract at the runtime
    boundary.  Keeping the reference contract local makes the numerical oracle
    independent from a Megatron checkout.
    """

    input_ids: torch.Tensor
    labels: torch.Tensor
    seq_lens: torch.Tensor
    loss_mask: torch.Tensor | None = None
    position_ids: torch.Tensor | None = None
    routed_experts: torch.Tensor | None = None
    r3_replay_mask: torch.Tensor | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.seq_lens.numel())

    def sizes(self) -> torch.Tensor:
        return self.seq_lens

    @property
    def total_tokens(self) -> int:
        return int(self.seq_lens.sum().item())

    @property
    def cu_seqlens(self) -> torch.Tensor:
        zero = torch.zeros(1, dtype=torch.int32, device=self.seq_lens.device)
        return torch.cat((zero, self.seq_lens.to(torch.int32).cumsum(0)))

    def make_position_ids(self) -> torch.Tensor:
        if self.position_ids is not None:
            return self.position_ids
        pieces = [
            torch.arange(int(length), device=self.seq_lens.device, dtype=torch.long)
            for length in self.seq_lens.tolist()
        ]
        return torch.cat(pieces) if pieces else torch.empty(0, dtype=torch.long)

    def to(self, device: torch.device | str) -> PackedBatch:
        def move(value: Any) -> Any:
            return value.to(device) if isinstance(value, torch.Tensor) else value

        return PackedBatch(
            input_ids=move(self.input_ids),
            labels=move(self.labels),
            seq_lens=move(self.seq_lens),
            loss_mask=move(self.loss_mask),
            position_ids=move(self.position_ids),
            routed_experts=move(self.routed_experts),
            r3_replay_mask=move(self.r3_replay_mask),
            extras={key: move(value) for key, value in self.extras.items()},
        )


def original_sample_lengths(seq_lens: torch.Tensor, sample_ids: torch.Tensor) -> dict[int, int]:
    """Resolve stable sample identities without treating them as sequence indices."""
    lengths = seq_lens.detach().cpu().tolist()
    samples = sample_ids.detach().cpu().tolist()
    if any(length <= 0 for length in lengths) or sum(lengths) != len(samples):
        raise ValueError("positive sequence lengths must cover every sample ID")
    result: dict[int, int] = {}
    offset = 0
    for length in lengths:
        ids = set(samples[offset : offset + length])
        if len(ids) != 1:
            raise ValueError("one packed sequence must map to exactly one sample ID")
        sample_id = ids.pop()
        if sample_id < 0 or sample_id in result:
            raise ValueError("sample IDs must be non-negative and unique across packed sequences")
        result[int(sample_id)] = int(length)
        offset += length
    return result


def packed_lm_targets(batch: PackedBatch) -> tuple[torch.Tensor, torch.Tensor]:
    """Shift unshifted labels/masks within sequences, following MLite's THD contract.

    An absent source mask includes the synthetic last target (label zero).
    Explicit masks shift once and exclude that target. Dummy tails are never targets.
    """
    labels = batch.labels.clone()
    mask = (
        torch.ones_like(labels, dtype=torch.float32)
        if batch.loss_mask is None
        else batch.loss_mask.float().clone()
    )
    offset = 0
    for length in batch.seq_lens.detach().cpu().tolist():
        stop = offset + int(length)
        labels[offset:stop] = torch.roll(labels[offset:stop], -1, 0)
        labels[stop - 1] = 0
        if batch.loss_mask is not None:
            mask[offset:stop] = torch.roll(mask[offset:stop], -1, 0)
            mask[stop - 1] = 0.0
        offset = stop
    padding = batch.extras.get("padding_mask")
    if padding is not None:
        mask = mask.masked_fill(padding.to(device=mask.device), 0.0)
    return labels, mask


def _metadata(seq_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sample_ids: list[torch.Tensor] = []
    positions: list[torch.Tensor] = []
    for sample_id, length in enumerate(seq_lens.tolist()):
        sample_ids.append(torch.full((length,), sample_id, dtype=torch.long))
        positions.append(torch.arange(length, dtype=torch.long))
    total = int(seq_lens.sum().item())
    return (
        torch.cat(sample_ids) if sample_ids else torch.empty(0, dtype=torch.long),
        torch.cat(positions) if positions else torch.empty(0, dtype=torch.long),
        torch.arange(total, dtype=torch.long),
    )


def make_synthetic_batch(
    *,
    seq_lens: Iterable[int] = (11, 7, 3),
    vocab_size: int = 257,
    seed: int = 1234,
    eos_token_id: int = 2,
    mask_last_token: bool = True,
    extreme_routing: bool = False,
) -> PackedBatch:
    """Create a deterministic variable-length LM batch.

    ``routing_bias`` is diagnostic metadata.  Reference tests can add it to
    router logits to force a strongly skewed cross-CP selection without
    changing token identities or labels.
    """

    lengths = torch.tensor(tuple(int(x) for x in seq_lens), dtype=torch.int32)
    if lengths.numel() == 0 or bool(torch.any(lengths <= 0)):
        raise ValueError("seq_lens must contain at least one positive length")
    if vocab_size <= max(eos_token_id, 3):
        raise ValueError("vocab_size must leave room for ordinary and EOS tokens")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    total = int(lengths.sum().item())
    input_ids = torch.randint(3, vocab_size, (total,), generator=generator, dtype=torch.long)
    # MLite's public PackedBatch contract carries token-aligned, unshifted
    # labels. The model protocol rolls labels and loss masks once, inside each
    # packed sequence, while constructing THD input. Keep the local oracle on
    # that same contract so parity never compares a once-shifted target with a
    # twice-shifted one.
    loss_mask = torch.ones(total, dtype=torch.float32)

    offset = 0
    for length in lengths.tolist():
        stop = offset + length
        input_ids[stop - 1] = eos_token_id
        offset = stop
    labels = input_ids.clone()
    if not mask_last_token:
        # Retained as a diagnostic escape hatch. The protocol still excludes
        # THD dummy padding; callers opting out are explicitly asking to train
        # the synthetic post-EOS target as well.
        loss_mask = None

    sample_ids, positions, global_ids = _metadata(lengths)
    routing_bias = torch.zeros(total, dtype=torch.float32)
    if extreme_routing:
        # Within every sample, strongly prefer its first positional half.  The
        # expert-choice budget then concentrates active rows in a contiguous
        # region instead of merely shifting every score in a sample equally.
        midpoints = torch.div(lengths.to(torch.long)[sample_ids] + 1, 2, rounding_mode="floor")
        routing_bias.copy_(torch.where(positions < midpoints, 20.0, -20.0))

    return PackedBatch(
        input_ids=input_ids,
        labels=labels,
        seq_lens=lengths,
        loss_mask=loss_mask,
        position_ids=positions,
        extras={
            "sample_ids": sample_ids,
            "original_position_ids": positions,
            "global_token_ids": global_ids,
            "padding_mask": torch.zeros(total, dtype=torch.bool),
            "routing_bias": routing_bias,
            "apply_routing_bias": bool(extreme_routing),
        },
    )


def as_mlite_packed_batch(batch: PackedBatch):
    """Convert to MLite's public data contract without importing it eagerly."""

    try:
        from megatron.lite.runtime.contracts.data import PackedBatch as MLitePackedBatch
    except ImportError as exc:  # pragma: no cover - exercised in EOS environment
        raise RuntimeError(
            "Megatron-Lite is not importable. Add the pinned "
            "Megatron-LM/experimental/lite directory to PYTHONPATH."
        ) from exc
    return MLitePackedBatch(
        input_ids=batch.input_ids,
        labels=batch.labels,
        seq_lens=batch.seq_lens,
        loss_mask=batch.loss_mask,
        position_ids=batch.position_ids,
        routed_experts=batch.routed_experts,
        r3_replay_mask=batch.r3_replay_mask,
        extras=dict(batch.extras),
    )


__all__ = ["PackedBatch", "as_mlite_packed_batch", "make_synthetic_batch"]
