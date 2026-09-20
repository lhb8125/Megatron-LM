from __future__ import annotations

import pytest
import torch
from torch import nn

from mor_mlite.distributed import (
    ActiveTokenBatch,
    ActiveTokenDispatcher,
    ActiveTokenLayout,
    ActiveTokenTransition,
    CommunicationCounters,
    EarlyExitParking,
    count_unexpected_real_token_ids,
)


def _batch(token_ids: list[int]) -> ActiveTokenBatch:
    count = len(token_ids)
    return ActiveTokenBatch(
        hidden=torch.arange(count * 2, dtype=torch.float32).reshape(count, 2),
        layout=ActiveTokenLayout.from_local(
            sample_ids=torch.zeros(count, dtype=torch.long),
            position_ids=torch.arange(count, dtype=torch.long),
            global_token_ids=torch.tensor(token_ids, dtype=torch.long),
        ),
    )


def test_active_change_and_backend_rebalance_have_independent_execution_sources() -> None:
    batch = _batch([10, 11, 12])
    dispatcher = ActiveTokenDispatcher()
    transition = ActiveTokenTransition(
        dispatcher,
        EarlyExitParking(batch.hidden),
    )
    first = transition.enter_first_round(batch)
    assert dispatcher.counters.active_set_changes == 0
    assert dispatcher.counters.hidden_rebalances == 0

    changed = transition.advance(
        first,
        torch.tensor([True, False, True]),
        round_index=1,
    )
    assert changed.dispatched
    assert dispatcher.counters.active_set_changes == 1
    assert dispatcher.counters.hidden_rebalances == 1

    unchanged = transition.advance(
        changed.active,
        torch.ones(changed.active.num_tokens, dtype=torch.bool),
        round_index=2,
    )
    assert not unchanged.dispatched
    transition.finalize(unchanged.active)
    assert dispatcher.counters.active_set_changes == 1
    assert dispatcher.counters.hidden_rebalances == 1
    assert dispatcher.counters.inverse_calls == 1


class _LegacyOnlyBackend:
    name = "legacy_only"

    def rebalance(self, batch, dispatcher, **_kwargs):
        dispatcher.counters.rebalance_calls += 1
        return batch


def test_transition_rejects_backend_that_does_not_record_a_real_hidden_transition() -> None:
    batch = _batch([10, 11])
    dispatcher = ActiveTokenDispatcher()
    transition = ActiveTokenTransition(
        dispatcher,
        EarlyExitParking(batch.hidden),
        backend=_LegacyOnlyBackend(),
    )
    first = transition.enter_first_round(batch)

    with pytest.raises(AssertionError, match="expected 1 hidden rebalance"):
        transition.advance(first, torch.tensor([True, False]), round_index=1)

    assert dispatcher.counters.active_set_changes == 0
    assert dispatcher.counters.hidden_rebalances == 0


def test_dispatcher_inside_recurrent_scope_is_detected() -> None:
    batch = _batch([10, 11])
    dispatcher = ActiveTokenDispatcher()

    with dispatcher.counters.recurrent_block_scope():
        dispatcher.dispatch(batch, torch.zeros(batch.num_tokens, dtype=torch.long))

    assert dispatcher.counters.recurrent_inner_dispatches == 1
    with pytest.raises(AssertionError, match="inner dispatch"):
        dispatcher.counters.assert_execution_contract(
            expected_recurrent_blocks=1,
            expected_recurrent_qkv_checks=0,
        )


def test_real_qkv_hook_counts_checks_and_excludes_dummy_tail_ids() -> None:
    counters = CommunicationCounters()
    qkv = nn.Linear(4, 6, bias=False)
    real_count, leaked_count = count_unexpected_real_token_ids(
        torch.tensor([10, 11, -1]),
        torch.tensor([False, False, True]),
        torch.tensor([10, 11]),
    )

    with (
        counters.recurrent_block_scope(),
        counters.recurrent_qkv_scope(
            qkv,
            expected_input_rows=3,
            real_token_count=real_count,
            early_exit_token_count=leaked_count,
            context="test recurrent layer",
        ),
    ):
        qkv(torch.ones(3, 4))

    assert counters.recurrent_qkv_checks == 1
    assert counters.recurrent_qkv_real_tokens == 2
    assert counters.early_exit_qkv_tokens == 0
    counters.assert_execution_contract(
        expected_recurrent_blocks=1,
        expected_recurrent_qkv_checks=1,
    )


def test_real_qkv_hook_fails_when_an_exited_token_reaches_attention() -> None:
    counters = CommunicationCounters()
    qkv = nn.Linear(4, 6, bias=False)
    real_count, leaked_count = count_unexpected_real_token_ids(
        torch.tensor([10, 99, -1]),
        torch.tensor([False, False, True]),
        torch.tensor([10, 11]),
    )

    with (
        pytest.raises(AssertionError, match="early-exit token"),
        counters.recurrent_block_scope(),
        counters.recurrent_qkv_scope(
            qkv,
            expected_input_rows=3,
            real_token_count=real_count,
            early_exit_token_count=leaked_count,
            context="test recurrent layer",
        ),
    ):
        qkv(torch.ones(3, 4))

    assert counters.recurrent_qkv_checks == 1
    assert counters.early_exit_qkv_tokens == 1
