"""Active dispatch backend registry."""

from __future__ import annotations

from .base import ActiveDispatchBackend
from .magi import (
    MagiCanonicalBackend,
    MagiDirectBackend,
    MagiDirectPlan,
    decode_magi_direct_plan,
)
from .static_reference import BalancedReferenceBackend, StaticReferenceBackend


def get_dispatch_backend(name: str) -> ActiveDispatchBackend:
    if name == "static_reference":
        return StaticReferenceBackend()
    if name == "balanced_reference":
        return BalancedReferenceBackend()
    if name in {"magi_canonical", "magi_direct"}:
        # Importing this module is cheap and never imports MagiAttention itself;
        # the optional extension is loaded only by ``rebalance``.
        return MagiCanonicalBackend() if name == "magi_canonical" else MagiDirectBackend()
    raise ValueError(
        f"unknown active dispatch backend {name!r}; expected static_reference, "
        "balanced_reference, magi_canonical, or magi_direct"
    )


__all__ = [
    "ActiveDispatchBackend",
    "BalancedReferenceBackend",
    "MagiCanonicalBackend",
    "MagiDirectBackend",
    "MagiDirectPlan",
    "StaticReferenceBackend",
    "decode_magi_direct_plan",
    "get_dispatch_backend",
]
