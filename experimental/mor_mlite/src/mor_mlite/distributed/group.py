"""Process-group description for MoR routing across TP-SP and CP ranks.

The depth router treats tensor sequence parallelism and context parallelism as
one logical token-placement grid.  This module intentionally does *not* create
process groups: ``torch.distributed.new_group`` must be called collectively by
the application while all ranks agree on group creation order.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist


def _dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


@dataclass(frozen=True, slots=True)
class TPxCPRouteGroup:
    """A dense-DP-local TP(sequence-parallel) x CP routing group.

    ``rank`` and all token ownership fields in the distributed package are
    ranks *within this group*, not global process ranks.  The canonical local
    rank mapping is ``cp_rank * tp_size + tp_rank``.

    A world-size-one instance is valid without initializing torch.distributed.
    For a distributed instance, callers should construct one composite process
    group for exactly one dense-DP replica and pass it here.
    """

    process_group: dist.ProcessGroup | None = None
    global_ranks: tuple[int, ...] = (0,)
    tp_size: int = 1
    cp_size: int = 1
    rank: int = 0

    def __post_init__(self) -> None:
        expected = self.tp_size * self.cp_size
        if self.tp_size < 1 or self.cp_size < 1:
            raise ValueError("tp_size and cp_size must both be positive")
        if len(self.global_ranks) != expected:
            raise ValueError(
                "TPxCPRouteGroup global_ranks must contain exactly "
                f"tp_size * cp_size={expected} ranks, got {len(self.global_ranks)}"
            )
        if not 0 <= self.rank < expected:
            raise ValueError(f"route-group rank {self.rank} is outside [0, {expected})")
        if expected > 1:
            if not _dist_ready():
                raise RuntimeError(
                    "A distributed TPxCPRouteGroup requires an initialized "
                    "torch.distributed process group"
                )
            if self.process_group is None:
                raise ValueError(
                    "A TPxCP route group larger than one needs an explicit composite "
                    "process_group. Create it collectively for the TP-SP x CP rectangle."
                )
            actual_world = dist.get_world_size(self.process_group)
            actual_rank = dist.get_rank(self.process_group)
            if actual_world != expected:
                raise ValueError(
                    f"route process group has size {actual_world}, expected {expected}"
                )
            if actual_rank != self.rank:
                raise ValueError(
                    f"route process-group rank is {actual_rank}, configured rank is {self.rank}"
                )

    @classmethod
    def local(cls) -> TPxCPRouteGroup:
        """Return the no-distributed-initialization single-rank route group."""

        return cls()

    @classmethod
    def from_process_group(
        cls,
        process_group: dist.ProcessGroup,
        *,
        tp_size: int,
        cp_size: int,
        global_ranks: Iterable[int] | None = None,
    ) -> TPxCPRouteGroup:
        if not _dist_ready():
            raise RuntimeError("torch.distributed must be initialized first")
        world_size = dist.get_world_size(process_group)
        if global_ranks is None:
            getter = getattr(dist, "get_process_group_ranks", None)
            if getter is None:
                raise ValueError(
                    "This PyTorch version cannot discover process-group members. "
                    "Pass global_ranks explicitly."
                )
            global_ranks = getter(process_group)
        ranks = tuple(int(value) for value in global_ranks)
        if len(ranks) != world_size:
            raise ValueError("global_ranks does not match the process-group world size")
        return cls(
            process_group=process_group,
            global_ranks=ranks,
            tp_size=tp_size,
            cp_size=cp_size,
            rank=dist.get_rank(process_group),
        )

    @classmethod
    def from_parallel_state(
        cls,
        parallel_state: Any,
        *,
        process_group: dist.ProcessGroup | None = None,
        global_ranks: Iterable[int] | None = None,
    ) -> TPxCPRouteGroup:
        """Adapt an MLite ``ParallelState`` without importing MLite.

        MLite currently exposes separate TP and CP groups, not their Cartesian
        product.  Consequently a composite group is mandatory whenever
        ``tp * cp > 1``.  Keeping this requirement explicit prevents accidental
        routing across DP or EP ranks.
        """

        tp_size = int(getattr(parallel_state, "tp_size", 1))
        cp_size = int(getattr(parallel_state, "cp_size", 1))
        if tp_size * cp_size == 1:
            return cls.local()
        if process_group is None:
            raise ValueError(
                "MLite ParallelState has no TPxCP group. Create a dense-DP-local "
                "TP-SP x CP group collectively and pass process_group/global_ranks."
            )
        return cls.from_process_group(
            process_group,
            tp_size=tp_size,
            cp_size=cp_size,
            global_ranks=global_ranks,
        )

    @property
    def world_size(self) -> int:
        return self.tp_size * self.cp_size

    @property
    def tp_rank(self) -> int:
        return self.rank % self.tp_size

    @property
    def cp_rank(self) -> int:
        return self.rank // self.tp_size

    def route_rank(self, *, tp_rank: int, cp_rank: int) -> int:
        if not 0 <= tp_rank < self.tp_size:
            raise ValueError(f"tp_rank {tp_rank} is outside [0, {self.tp_size})")
        if not 0 <= cp_rank < self.cp_size:
            raise ValueError(f"cp_rank {cp_rank} is outside [0, {self.cp_size})")
        return cp_rank * self.tp_size + tp_rank

    def coordinates(self, route_rank: int) -> tuple[int, int]:
        if not 0 <= route_rank < self.world_size:
            raise ValueError(f"route rank {route_rank} is outside this group")
        return route_rank % self.tp_size, route_rank // self.tp_size

    def any(self, local_value: bool, *, device: torch.device | None = None) -> bool:
        """Return a group-wide OR while remaining a no-op in local mode."""

        if self.world_size == 1:
            return bool(local_value)
        if device is None:
            backend = dist.get_backend(self.process_group)
            device = torch.device("cuda") if backend == "nccl" else torch.device("cpu")
        value = torch.tensor(int(local_value), dtype=torch.int32, device=device)
        dist.all_reduce(value, op=dist.ReduceOp.MAX, group=self.process_group)
        return bool(value.item())


__all__ = ["TPxCPRouteGroup"]
