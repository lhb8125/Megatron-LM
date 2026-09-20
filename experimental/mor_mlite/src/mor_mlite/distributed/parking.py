"""Park early-exit activations and merge them exactly once at model exit."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from .all_to_all import ActiveTokenDispatcher
from .layout import ActiveTokenBatch


@dataclass(frozen=True, slots=True)
class ParkingTicket:
    """Validated, side-effect-free early-exit state awaiting commit."""

    batch: ActiveTokenBatch | None


class EarlyExitParking:
    """Autograd-preserving storage for tokens that stop recurring.

    Exited tokens remain on the rank where they exit.  They do not participate
    in later recurrent attention or KV exchange.  ``finalize`` concatenates all
    parked rounds with the final active shard, performs one inverse ownership
    transfer, and writes every real token into its original local row.
    """

    def __init__(
        self,
        base_hidden: torch.Tensor,
        *,
        padding_mask: torch.Tensor | None = None,
    ) -> None:
        if base_hidden.dim() < 2:
            raise ValueError("base_hidden must have a leading token axis")
        if padding_mask is None:
            padding_mask = torch.zeros(
                base_hidden.size(0), dtype=torch.bool, device=base_hidden.device
            )
        if padding_mask.shape != (base_hidden.size(0),) or padding_mask.dtype != torch.bool:
            raise ValueError("padding_mask must be boolean with one entry per base token")
        if padding_mask.device != base_hidden.device:
            raise ValueError("base_hidden and padding_mask must share a device")
        self.base_hidden = base_hidden
        self.padding_mask = padding_mask
        self._parked: list[ActiveTokenBatch] = []
        self._finalized = False

    @property
    def expected_local_rows(self) -> torch.Tensor:
        return torch.where(~self.padding_mask)[0]

    @property
    def num_parked_local(self) -> int:
        return sum(batch.num_tokens for batch in self._parked)

    def prepare(self, exited: ActiveTokenBatch) -> ParkingTicket:
        """Validate an exit batch without mutating parking state."""

        if self._finalized:
            raise RuntimeError("cannot park tokens after final merge")
        exited.layout.assert_compute_ready()
        if not exited.num_tokens:
            return ParkingTicket(None)
        # Destination slots describe a transient compute layout and are not
        # meaningful when batches from multiple rounds are concatenated.
        layout = replace(exited.layout, destination_slots=None, layout_kind="parked")
        # A gate belongs to the update at the exit boundary; final merge only
        # transports the resulting hidden state. Dropping it allows rounds to
        # use different gate shapes or omit gates entirely.
        return ParkingTicket(ActiveTokenBatch(hidden=exited.hidden, layout=layout))

    def commit(self, ticket: ParkingTicket) -> None:
        """Commit a validated ticket; this is the transaction's only mutation."""

        if self._finalized:
            raise RuntimeError("cannot park tokens after final merge")
        if not isinstance(ticket, ParkingTicket):
            raise TypeError("commit expects a ParkingTicket returned by prepare")
        if ticket.batch is not None:
            self._parked.append(ticket.batch)

    def park(self, exited: ActiveTokenBatch) -> None:
        """Immediate prepare+commit helper for non-transactional callers."""

        self.commit(self.prepare(exited))

    def assert_initial_full_shard(self, batch: ActiveTokenBatch, *, local_route_rank: int) -> None:
        """Verify that the communication-free first round covers every real local row."""

        batch.layout.assert_compute_ready(local_route_rank=local_route_rank)
        local_source = batch.layout.source_route_ranks == local_route_rank
        rows = torch.sort(batch.layout.source_local_rows[local_source]).values
        expected = self.expected_local_rows
        if not torch.equal(rows, expected):
            raise AssertionError(
                "the first 100% recurrent round must contain every non-padding "
                "source-local token exactly once"
            )

    def finalize(
        self,
        final_active: ActiveTokenBatch,
        dispatcher: ActiveTokenDispatcher,
    ) -> torch.Tensor:
        if self._finalized:
            raise RuntimeError("EarlyExitParking.finalize may only be called once")
        final_active.layout.assert_compute_ready()
        final_layout = replace(
            final_active.layout, destination_slots=None, layout_kind="final_active"
        )
        batches = [
            *self._parked,
            ActiveTokenBatch(hidden=final_active.hidden, layout=final_layout),
        ]
        combined = ActiveTokenBatch.cat(batches)
        restored = dispatcher.restore_to_sources(combined)
        local_rank = dispatcher.route_group.rank
        if restored.num_tokens and not bool(
            (restored.layout.source_route_ranks == local_rank).all().item()
        ):
            raise RuntimeError("inverse dispatch returned a token to the wrong source rank")
        rows = restored.layout.source_local_rows.to(dtype=torch.long)
        if rows.numel() and (
            int(rows.min().item()) < 0 or int(rows.max().item()) >= self.base_hidden.size(0)
        ):
            raise RuntimeError("inverse dispatch returned an invalid source-local row")
        if torch.unique(rows).numel() != rows.numel():
            raise RuntimeError("a token was parked or finalized more than once")
        sorted_rows, order = torch.sort(rows)
        if not torch.equal(sorted_rows, self.expected_local_rows):
            missing = sorted(
                set(self.expected_local_rows.detach().cpu().tolist())
                - set(sorted_rows.detach().cpu().tolist())
            )
            extra = sorted(
                set(sorted_rows.detach().cpu().tolist())
                - set(self.expected_local_rows.detach().cpu().tolist())
            )
            raise RuntimeError(
                "final merge does not cover every non-padding token; "
                f"missing={missing}, extra={extra}"
            )
        source = restored.hidden.index_select(0, order)
        output = torch.index_copy(self.base_hidden, 0, sorted_rows, source)
        self._finalized = True
        return output


__all__ = ["EarlyExitParking", "ParkingTicket"]
