"""Backend contract for one active-token layout transition."""

from __future__ import annotations

from typing import Any, Protocol

import torch

from ..all_to_all import ActiveTokenDispatcher
from ..layout import ActiveTokenBatch


class ActiveDispatchBackend(Protocol):
    name: str

    def rebalance(
        self,
        batch: ActiveTokenBatch,
        dispatcher: ActiveTokenDispatcher,
        *,
        target_ranks: torch.Tensor | None = None,
        destination_slots: torch.Tensor | None = None,
        context: dict[str, Any] | None = None,
    ) -> ActiveTokenBatch:
        """Perform exactly one logical active-layout transition."""


__all__ = ["ActiveDispatchBackend"]
