"""Reference transition backends used by tiny distributed diagnostics."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from ..all_to_all import ActiveTokenDispatcher, build_balanced_target_plan
from ..layout import ActiveTokenBatch


class StaticReferenceBackend:
    """Keep every selected token on its current TP/CP owner.

    This is the deliberately imbalanced comparison path from the v1 design.
    Attention must pair it with the PyTorch active-Q/K/V gather in
    :mod:`mor_mlite.distributed.static_qkv`.  This transition backend issues
    no hidden, gate, or metadata All-to-All at a recurrent boundary; the
    separate static Q/K/V oracle still uses gather/reduction collectives.
    """

    name = "static_reference"

    def rebalance(
        self,
        batch: ActiveTokenBatch,
        dispatcher: ActiveTokenDispatcher,
        *,
        target_ranks: torch.Tensor | None = None,
        destination_slots: torch.Tensor | None = None,
        context: dict[str, Any] | None = None,
    ) -> ActiveTokenBatch:
        del context
        if target_ranks is not None or destination_slots is not None:
            raise ValueError("static_reference owns no target-placement plan")
        batch.layout.assert_compute_ready(local_route_rank=dispatcher.route_group.rank)
        result = replace(
            batch,
            layout=replace(
                batch.layout,
                destination_slots=None,
                layout_kind=self.name,
                replicated=False,
            ),
        )
        # This backend deliberately retains token ownership, but it still owns
        # and completes the hidden-layout transition.  Record that execution
        # separately from ActiveTokenTransition's state-change observation;
        # this backend does not increment a transition-collective counter.
        dispatcher.counters.record_backend_hidden_rebalance()
        return result


class BalancedReferenceBackend:
    """Deterministic A2A layout oracle retained for dispatcher unit tests."""

    name = "balanced_reference"

    def rebalance(
        self,
        batch: ActiveTokenBatch,
        dispatcher: ActiveTokenDispatcher,
        *,
        target_ranks: torch.Tensor | None = None,
        destination_slots: torch.Tensor | None = None,
        context: dict[str, Any] | None = None,
    ) -> ActiveTokenBatch:
        del context
        if target_ranks is None:
            plan = build_balanced_target_plan(
                batch.layout, dispatcher.route_group, dispatcher.counters
            )
            target_ranks = plan.target_ranks
            destination_slots = plan.destination_slots
        return dispatcher.dispatch(
            batch,
            target_ranks,
            destination_slots=destination_slots,
            layout_kind=self.name,
        )


__all__ = ["BalancedReferenceBackend", "StaticReferenceBackend"]
