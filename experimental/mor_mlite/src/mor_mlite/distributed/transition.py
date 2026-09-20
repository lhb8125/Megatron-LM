"""One-dispatch-per-boundary state machine for active recurrent tokens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .all_to_all import ActiveTokenDispatcher
from .backends import ActiveDispatchBackend, get_dispatch_backend
from .layout import ActiveTokenBatch
from .parking import EarlyExitParking


@dataclass(frozen=True, slots=True)
class TransitionResult:
    active: ActiveTokenBatch
    exited: ActiveTokenBatch
    dispatched: bool


class ActiveTokenTransition:
    """Enforce MoR communication invariants around recurrent boundaries."""

    def __init__(
        self,
        dispatcher: ActiveTokenDispatcher,
        parking: EarlyExitParking,
        *,
        backend: str | ActiveDispatchBackend = "static_reference",
    ) -> None:
        self.dispatcher = dispatcher
        self.parking = parking
        self.backend = get_dispatch_backend(backend) if isinstance(backend, str) else backend
        self._entered_first_round = False

    def enter_first_round(self, batch: ActiveTokenBatch) -> ActiveTokenBatch:
        """Enter the required 100%-token first round without any dispatch."""

        if self._entered_first_round:
            raise RuntimeError("enter_first_round may only be called once")
        before = self.dispatcher.counters.snapshot()
        self.parking.assert_initial_full_shard(
            batch, local_route_rank=self.dispatcher.route_group.rank
        )
        self._entered_first_round = True
        self.dispatcher.counters.skipped_full_first_round += 1
        self.dispatcher.counters.assert_hidden_rebalance_delta(
            before, 0, context="first 100% recurrent round"
        )
        return batch.with_round(0)

    def advance(
        self,
        batch: ActiveTokenBatch,
        keep_mask: torch.Tensor,
        *,
        round_index: int,
        target_ranks: torch.Tensor | None = None,
        destination_slots: torch.Tensor | None = None,
        backend_context: dict[str, Any] | None = None,
    ) -> TransitionResult:
        """Park exiting tokens and rebalance the changed active set exactly once.

        Explicit placement tensors, when supplied, are aligned with the tokens
        selected by ``keep_mask`` rather than with the pre-selection batch.
        """

        if not self._entered_first_round:
            raise RuntimeError("call enter_first_round before the first routed boundary")
        if round_index <= batch.layout.round_index:
            raise ValueError("round_index must increase at every recurrent boundary")
        batch.layout.assert_compute_ready(local_route_rank=self.dispatcher.route_group.rank)
        continuing, exited = batch.split(keep_mask)
        # Preparation performs every parking validation but does not mutate
        # state. Commit happens only after the transition fully succeeds.
        parking_ticket = self.parking.prepare(exited)
        group = self.dispatcher.route_group
        global_changed = group.any(exited.num_tokens > 0, device=batch.hidden.device)
        if group.world_size > 1:
            self.dispatcher.counters.collective_calls += 1
        if not global_changed:
            self.parking.commit(parking_ticket)
            self.dispatcher.counters.skipped_unchanged_boundaries += 1
            return TransitionResult(
                active=continuing.with_round(round_index),
                exited=exited,
                dispatched=False,
            )
        if target_ranks is not None and target_ranks.shape != (continuing.num_tokens,):
            raise ValueError("target_ranks must align with keep_mask-selected tokens")
        if destination_slots is not None and destination_slots.shape != (continuing.num_tokens,):
            raise ValueError("destination_slots must align with keep_mask-selected tokens")
        before = self.dispatcher.counters.snapshot()
        continuing = continuing.with_round(round_index)
        active = self.backend.rebalance(
            continuing,
            self.dispatcher,
            target_ranks=target_ranks,
            destination_slots=destination_slots,
            context=backend_context,
        )
        self.dispatcher.counters.assert_hidden_rebalance_delta(
            before, 1, context=f"active boundary entering round {round_index}"
        )
        # The backend call and communication-contract assertion are both part
        # of the transaction. A failure above leaves parking unchanged, so the
        # exact same boundary can be retried without duplicate exits.
        self.parking.commit(parking_ticket)
        self.dispatcher.counters.record_active_set_change()
        return TransitionResult(active=active, exited=exited, dispatched=True)

    def finalize(self, final_active: ActiveTokenBatch) -> torch.Tensor:
        return self.parking.finalize(final_active, self.dispatcher)


__all__ = ["ActiveTokenTransition", "TransitionResult"]
