"""Logical and physical communication accounting for recurrent routing."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch


def count_unexpected_real_token_ids(
    global_token_ids: torch.Tensor,
    padding_mask: torch.Tensor,
    expected_active_global_token_ids: torch.Tensor,
) -> tuple[int, int]:
    """Count real compute rows and rows absent from the selected RoutePlan."""

    if global_token_ids.ndim != 1 or padding_mask.shape != global_token_ids.shape:
        raise ValueError("QKV token IDs and padding mask must be aligned one-dimensional tensors")
    if padding_mask.dtype != torch.bool:
        raise TypeError("QKV padding mask must be boolean")
    expected = expected_active_global_token_ids.reshape(-1).to(
        device=global_token_ids.device, dtype=torch.long
    )
    if torch.unique(expected).numel() != expected.numel():
        raise ValueError("expected recurrent active token IDs must be unique")
    real_ids = global_token_ids[~padding_mask]
    unexpected = ~torch.isin(real_ids, expected)
    return int(real_ids.numel()), int(unexpected.sum().item())


@dataclass(frozen=True, slots=True)
class CommunicationSnapshot:
    active_set_changes: int
    hidden_rebalances: int
    recurrent_inner_dispatches: int
    early_exit_qkv_tokens: int
    recurrent_block_calls: int
    recurrent_qkv_checks: int
    recurrent_qkv_real_tokens: int
    rebalance_calls: int
    inverse_calls: int
    collective_calls: int
    count_exchanges: int
    metadata_all_to_all: int
    hidden_all_to_all: int
    gate_all_to_all: int
    route_gathers: int
    magi_dispatches: int
    skipped_full_first_round: int
    skipped_unchanged_boundaries: int


@dataclass(slots=True)
class CommunicationCounters:
    """MoR-orchestration counters and executable communication contracts.

    These counters cover routing and active-layout transitions, not collectives
    internal to attention/MoE kernels or the static Q/K/V correctness oracle.

    Active-set changes are recorded by the transition state machine, while
    hidden rebalances are recorded only by the backend that actually moves the
    hidden tensor.  They intentionally have independent sources so the
    one-rebalance-per-changed-boundary contract cannot pass by construction.
    World-size-one dispatches still count as logical hidden rebalances, but do
    not increment physical collective counters.
    """

    active_set_changes: int = 0
    hidden_rebalances: int = 0
    recurrent_inner_dispatches: int = 0
    early_exit_qkv_tokens: int = 0
    recurrent_block_calls: int = 0
    recurrent_qkv_checks: int = 0
    recurrent_qkv_real_tokens: int = 0
    rebalance_calls: int = 0
    inverse_calls: int = 0
    collective_calls: int = 0
    count_exchanges: int = 0
    metadata_all_to_all: int = 0
    hidden_all_to_all: int = 0
    gate_all_to_all: int = 0
    route_gathers: int = 0
    magi_dispatches: int = 0
    skipped_full_first_round: int = 0
    skipped_unchanged_boundaries: int = 0
    _recurrent_block_depth: int = field(default=0, init=False, repr=False)

    def snapshot(self) -> CommunicationSnapshot:
        return CommunicationSnapshot(
            active_set_changes=self.active_set_changes,
            hidden_rebalances=self.hidden_rebalances,
            recurrent_inner_dispatches=self.recurrent_inner_dispatches,
            early_exit_qkv_tokens=self.early_exit_qkv_tokens,
            recurrent_block_calls=self.recurrent_block_calls,
            recurrent_qkv_checks=self.recurrent_qkv_checks,
            recurrent_qkv_real_tokens=self.recurrent_qkv_real_tokens,
            rebalance_calls=self.rebalance_calls,
            inverse_calls=self.inverse_calls,
            collective_calls=self.collective_calls,
            count_exchanges=self.count_exchanges,
            metadata_all_to_all=self.metadata_all_to_all,
            hidden_all_to_all=self.hidden_all_to_all,
            gate_all_to_all=self.gate_all_to_all,
            route_gathers=self.route_gathers,
            magi_dispatches=self.magi_dispatches,
            skipped_full_first_round=self.skipped_full_first_round,
            skipped_unchanged_boundaries=self.skipped_unchanged_boundaries,
        )

    def record_active_set_change(self) -> None:
        """Record one globally committed shrink of the recurrent active set."""

        self.active_set_changes += 1

    def record_token_dispatch(self, *, inverse: bool) -> None:
        """Record a completed active-token dispatcher operation."""

        if self._recurrent_block_depth:
            self.recurrent_inner_dispatches += 1
        if inverse:
            self.inverse_calls += 1
            return
        self.hidden_rebalances += 1
        # Retain the original field for low-level compatibility and existing
        # profiler consumers. Contract checks use ``hidden_rebalances``.
        self.rebalance_calls += 1

    def record_backend_hidden_rebalance(self) -> None:
        """Record a backend-owned hidden migration that bypasses the dispatcher."""

        if self._recurrent_block_depth:
            self.recurrent_inner_dispatches += 1
        self.hidden_rebalances += 1
        self.rebalance_calls += 1

    @contextmanager
    def recurrent_block_scope(self) -> Iterator[None]:
        """Mark the exact interval in which recurrent physical layers execute."""

        if self._recurrent_block_depth:
            raise RuntimeError("recurrent block instrumentation cannot be nested")
        self._recurrent_block_depth = 1
        self.recurrent_block_calls += 1
        try:
            yield
        finally:
            self._recurrent_block_depth = 0

    def record_recurrent_qkv(self, *, real_token_count: int, early_exit_token_count: int) -> None:
        """Record one actual recurrent QKV projection pre-hook invocation."""

        if not self._recurrent_block_depth:
            raise AssertionError("recurrent QKV executed outside its recurrent block scope")
        if real_token_count < 0 or early_exit_token_count < 0:
            raise ValueError("QKV token counts must be non-negative")
        if early_exit_token_count > real_token_count:
            raise ValueError("early-exit QKV count cannot exceed the real QKV token count")
        self.recurrent_qkv_checks += 1
        self.recurrent_qkv_real_tokens += int(real_token_count)
        self.early_exit_qkv_tokens += int(early_exit_token_count)
        if early_exit_token_count:
            raise AssertionError(
                f"{early_exit_token_count} early-exit token(s) entered recurrent QKV"
            )

    @contextmanager
    def recurrent_qkv_scope(
        self,
        qkv_module: Any,
        *,
        expected_input_rows: int,
        real_token_count: int,
        early_exit_token_count: int,
        context: str,
    ) -> Iterator[None]:
        """Observe one real QKV module invocation within a recurrent layer."""

        register_hook = getattr(qkv_module, "register_forward_pre_hook", None)
        if not callable(register_hook):
            raise TypeError(f"{context} exposes no instrumentable QKV module")
        checks_before = self.recurrent_qkv_checks

        def record_qkv_execution(_module: Any, inputs: tuple[Any, ...]) -> None:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise AssertionError(f"{context} QKV received no tensor input")
            observed_rows = int(inputs[0].size(0)) if inputs[0].ndim else 0
            if observed_rows != expected_input_rows:
                raise AssertionError(
                    f"{context} QKV rows disagree with active layout: "
                    f"{observed_rows} != {expected_input_rows}"
                )
            self.record_recurrent_qkv(
                real_token_count=real_token_count,
                early_exit_token_count=early_exit_token_count,
            )

        handle = register_hook(record_qkv_execution)
        try:
            yield
        finally:
            handle.remove()
        observed_checks = self.recurrent_qkv_checks - checks_before
        if observed_checks != 1:
            raise AssertionError(f"{context} executed QKV {observed_checks} times; expected 1")

    def assert_hidden_rebalance_delta(
        self, before: CommunicationSnapshot, expected: int, *, context: str
    ) -> None:
        actual = self.hidden_rebalances - before.hidden_rebalances
        if actual != expected:
            raise AssertionError(
                f"{context}: expected {expected} hidden rebalance(s), observed {actual}"
            )

    def assert_execution_contract(
        self,
        *,
        expected_recurrent_blocks: int,
        expected_recurrent_qkv_checks: int,
    ) -> None:
        """Validate independently sourced end-to-end recurrent invariants."""

        errors: list[str] = []
        if self.active_set_changes != self.hidden_rebalances:
            errors.append(
                f"active-set changes {self.active_set_changes} != hidden rebalances "
                f"{self.hidden_rebalances}"
            )
        if self.recurrent_inner_dispatches:
            errors.append(
                f"recurrent block performed {self.recurrent_inner_dispatches} inner dispatch(es)"
            )
        if self.early_exit_qkv_tokens:
            errors.append(f"{self.early_exit_qkv_tokens} early-exit token(s) entered recurrent QKV")
        if self.recurrent_block_calls != expected_recurrent_blocks:
            errors.append(
                f"recurrent block calls {self.recurrent_block_calls} != expected "
                f"{expected_recurrent_blocks}"
            )
        if self.recurrent_qkv_checks != expected_recurrent_qkv_checks:
            errors.append(
                f"recurrent QKV checks {self.recurrent_qkv_checks} != expected "
                f"{expected_recurrent_qkv_checks}"
            )
        if self._recurrent_block_depth:
            errors.append("recurrent block instrumentation scope was not closed")
        if errors:
            raise AssertionError("; ".join(errors))

    def assert_rebalance_delta(
        self, before: CommunicationSnapshot, expected: int, *, context: str
    ) -> None:
        actual = self.rebalance_calls - before.rebalance_calls
        if actual != expected:
            raise AssertionError(
                f"{context}: expected {expected} logical rebalance(s), observed {actual}"
            )

    def assert_collective_delta(
        self, before: CommunicationSnapshot, expected: int, *, context: str
    ) -> None:
        actual = self.collective_calls - before.collective_calls
        if actual != expected:
            raise AssertionError(
                f"{context}: expected {expected} orchestration collective(s), observed {actual}"
            )


__all__ = [
    "CommunicationCounters",
    "CommunicationSnapshot",
    "count_unexpected_real_token_ids",
]
