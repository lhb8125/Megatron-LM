from __future__ import annotations

import multiprocessing
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from mor_mlite.distributed import (
    ActiveTokenBatch,
    ActiveTokenDispatcher,
    ActiveTokenLayout,
    ActiveTokenTransition,
    EarlyExitParking,
    TPxCPRouteGroup,
    get_dispatch_backend,
)


def _local_batch(
    hidden: torch.Tensor,
    *,
    positions: list[int],
    global_ids: list[int] | None = None,
    gates: torch.Tensor | None = None,
    padding_mask: torch.Tensor | None = None,
    drop_padding: bool = True,
) -> ActiveTokenBatch:
    sample_ids = torch.zeros(len(positions), dtype=torch.long, device=hidden.device)
    position_ids = torch.tensor(positions, dtype=torch.long, device=hidden.device)
    ids = (
        None
        if global_ids is None
        else torch.tensor(global_ids, dtype=torch.long, device=hidden.device)
    )
    layout = ActiveTokenLayout.from_local(
        sample_ids=sample_ids,
        position_ids=position_ids,
        global_token_ids=ids,
        padding_mask=padding_mask,
        drop_padding=drop_padding,
    )
    if drop_padding and padding_mask is not None:
        hidden = hidden[~padding_mask]
        gates = None if gates is None else gates[~padding_mask]
    return ActiveTokenBatch(hidden=hidden, gates=gates, layout=layout)


def test_world_size_one_needs_no_distributed_initialization_and_preserves_gradients():
    assert not dist.is_initialized()
    hidden = torch.tensor([[2.0], [0.0], [1.0]], requires_grad=True)
    gates = torch.tensor([0.2, 0.0, 0.1], requires_grad=True)
    batch = _local_batch(
        hidden,
        positions=[7, 0, 3],
        global_ids=[2, 0, 1],
        gates=gates,
    )
    dispatcher = ActiveTokenDispatcher()
    backend = get_dispatch_backend("balanced_reference")

    result = backend.rebalance(batch, dispatcher)

    assert result.layout.position_ids.tolist() == [0, 3, 7]
    assert result.layout.global_token_ids.tolist() == [0, 1, 2]
    assert result.hidden.flatten().tolist() == [0.0, 1.0, 2.0]
    assert result.gates.tolist() == pytest.approx([0.0, 0.1, 0.2])
    result.hidden.square().sum().add(result.gates.square().sum()).backward()
    torch.testing.assert_close(hidden.grad, 2 * hidden)
    torch.testing.assert_close(gates.grad, 2 * gates)
    assert dispatcher.counters.rebalance_calls == 1
    assert dispatcher.counters.collective_calls == 0

    restored = dispatcher.restore_to_sources(result)
    assert restored.layout.source_local_rows.tolist() == [0, 1, 2]
    torch.testing.assert_close(restored.hidden, hidden.detach())
    assert dispatcher.counters.inverse_calls == 1
    assert dispatcher.counters.collective_calls == 0


def test_static_reference_keeps_local_shard_without_transition_collectives():
    hidden = torch.tensor([[2.0], [0.0], [1.0]], requires_grad=True)
    batch = _local_batch(
        hidden,
        positions=[7, 0, 3],
        global_ids=[2, 0, 1],
    )
    dispatcher = ActiveTokenDispatcher()

    result = get_dispatch_backend("static_reference").rebalance(batch, dispatcher)

    assert result.hidden is hidden
    assert result.layout.position_ids.tolist() == [7, 0, 3]
    assert result.layout.global_token_ids.tolist() == [2, 0, 1]
    assert result.layout.layout_kind == "static_reference"
    assert dispatcher.counters.rebalance_calls == 1
    assert dispatcher.counters.hidden_all_to_all == 0
    assert dispatcher.counters.metadata_all_to_all == 0
    assert dispatcher.counters.collective_calls == 0


def test_first_full_round_parks_exits_and_rebalances_once_per_changed_boundary():
    base = torch.arange(10, dtype=torch.float32).view(5, 2).requires_grad_()
    padding = torch.tensor([False, False, False, False, True])
    batch = _local_batch(
        base,
        positions=[0, 1, 2, 3, 4],
        global_ids=[0, 1, 2, 3, 4],
        padding_mask=padding,
    )
    dispatcher = ActiveTokenDispatcher(TPxCPRouteGroup.local())
    parking = EarlyExitParking(base, padding_mask=padding)
    transition = ActiveTokenTransition(dispatcher, parking)

    first = transition.enter_first_round(batch)
    assert dispatcher.counters.rebalance_calls == 0
    first_output = ActiveTokenBatch(
        hidden=first.hidden + 10,
        gates=first.gates,
        layout=first.layout,
    )
    routed = transition.advance(
        first_output,
        torch.tensor([True, False, True, False]),
        round_index=1,
    )
    assert routed.dispatched
    assert routed.exited.layout.position_ids.tolist() == [1, 3]
    assert routed.active.layout.position_ids.tolist() == [0, 2]
    assert dispatcher.counters.rebalance_calls == 1

    second_output = ActiveTokenBatch(
        hidden=routed.active.hidden + 100,
        gates=routed.active.gates,
        layout=routed.active.layout,
    )
    unchanged = transition.advance(
        second_output,
        torch.ones(second_output.num_tokens, dtype=torch.bool),
        round_index=2,
    )
    assert not unchanged.dispatched
    assert dispatcher.counters.rebalance_calls == 1

    merged = transition.finalize(unchanged.active)
    expected = base.detach().clone()
    expected[[0, 2]] += 110
    expected[[1, 3]] += 10
    torch.testing.assert_close(merged, expected)
    merged.sum().backward()
    torch.testing.assert_close(base.grad, torch.ones_like(base))
    assert dispatcher.counters.skipped_full_first_round == 1
    assert dispatcher.counters.skipped_unchanged_boundaries == 1
    assert dispatcher.counters.inverse_calls == 1
    assert dispatcher.counters.collective_calls == 0


def test_final_merge_ignores_gate_presence_and_shape_across_parked_rounds():
    base = torch.arange(8, dtype=torch.float32).view(4, 2).requires_grad_()
    full = _local_batch(
        base,
        positions=[0, 1, 2, 3],
        gates=torch.ones(4),
    )
    first_exit = full.index_select(torch.tensor([0]))
    first_exit = ActiveTokenBatch(
        hidden=first_exit.hidden + 10,
        gates=first_exit.gates,
        layout=first_exit.layout,
    )
    second_layout = full.index_select(torch.tensor([1])).layout.with_round(1)
    second_exit = ActiveTokenBatch(
        hidden=base[[1]] + 20,
        # A later router may expose multiple diagnostics/gates per token.
        gates=torch.ones(1, 2),
        layout=second_layout,
    )
    final_layout = full.index_select(torch.tensor([2, 3])).layout.with_round(2)
    final_active = ActiveTokenBatch(
        hidden=base[[2, 3]] + 30,
        gates=None,
        layout=final_layout,
    )
    dispatcher = ActiveTokenDispatcher()
    parking = EarlyExitParking(base)
    parking.park(first_exit)
    parking.park(second_exit)

    merged = parking.finalize(final_active, dispatcher)

    expected = base.detach().clone()
    expected[[0]] += 10
    expected[[1]] += 20
    expected[[2, 3]] += 30
    torch.testing.assert_close(merged, expected)
    merged.sum().backward()
    torch.testing.assert_close(base.grad, torch.ones_like(base))


class _FailAfterDispatchOnce:
    name = "fail_after_dispatch_once"

    def __init__(self) -> None:
        self.calls = 0
        self.delegate = get_dispatch_backend("balanced_reference")

    def rebalance(
        self,
        batch,
        dispatcher,
        *,
        target_ranks=None,
        destination_slots=None,
        context=None,
    ):
        self.calls += 1
        result = self.delegate.rebalance(
            batch,
            dispatcher,
            target_ranks=target_ranks,
            destination_slots=destination_slots,
            context=context,
        )
        if self.calls == 1:
            raise RuntimeError("injected backend failure after transport")
        return result


def test_failed_backend_does_not_commit_exits_and_boundary_can_be_retried():
    base = torch.arange(6, dtype=torch.float32).view(3, 2)
    batch = _local_batch(base, positions=[0, 1, 2])
    dispatcher = ActiveTokenDispatcher()
    parking = EarlyExitParking(base)
    backend = _FailAfterDispatchOnce()
    transition = ActiveTokenTransition(dispatcher, parking, backend=backend)
    first = transition.enter_first_round(batch)
    keep = torch.tensor([True, False, True])

    with pytest.raises(RuntimeError, match="injected backend failure"):
        transition.advance(first, keep, round_index=1)
    assert parking.num_parked_local == 0

    retried = transition.advance(first, keep, round_index=1)
    assert retried.dispatched
    assert parking.num_parked_local == 1
    assert backend.calls == 2
    merged = transition.finalize(retried.active)
    torch.testing.assert_close(merged, base)


def test_padding_is_rejected_from_active_dispatch():
    hidden = torch.arange(6, dtype=torch.float32).view(3, 2)
    batch = _local_batch(
        hidden,
        positions=[0, 1, 2],
        padding_mask=torch.tensor([False, True, False]),
        drop_padding=False,
    )
    dispatcher = ActiveTokenDispatcher()
    with pytest.raises(ValueError, match="padding rows cannot enter"):
        dispatcher.dispatch(batch, torch.zeros(3, dtype=torch.long))


def test_first_round_must_cover_all_non_padding_source_rows():
    base = torch.arange(8, dtype=torch.float32).view(4, 2)
    parking = EarlyExitParking(base)
    dispatcher = ActiveTokenDispatcher()
    incomplete = _local_batch(base[[0, 2, 3]], positions=[0, 2, 3])
    transition = ActiveTokenTransition(dispatcher, parking)
    with pytest.raises(AssertionError, match="first 100% recurrent round"):
        transition.enter_first_round(incomplete)


def test_magi_backends_are_lazy_and_reject_an_arbitrary_local_shard():
    backend = get_dispatch_backend("magi_canonical")
    batch = _local_batch(torch.ones(2, 3), positions=[0, 1])
    with pytest.raises(ValueError, match="cannot consume an arbitrary active local shard"):
        backend.rebalance(batch, ActiveTokenDispatcher())


def _two_rank_worker(rank: int, init_path: str, queue: multiprocessing.Queue) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=2,
    )
    try:
        group = TPxCPRouteGroup.from_process_group(
            dist.group.WORLD,
            tp_size=1,
            cp_size=2,
            global_ranks=(0, 1),
        )
        positions = [rank, rank + 2]
        hidden = torch.tensor(positions, dtype=torch.float32).view(2, 1).requires_grad_()
        gates = (torch.tensor(positions, dtype=torch.float32) / 10).requires_grad_()
        batch = ActiveTokenBatch(
            hidden=hidden,
            gates=gates,
            layout=ActiveTokenLayout.from_local(
                sample_ids=torch.zeros(2, dtype=torch.long),
                position_ids=torch.tensor(positions, dtype=torch.long),
                global_token_ids=torch.tensor(positions, dtype=torch.long),
                route_rank=rank,
            ),
        )
        dispatcher = ActiveTokenDispatcher(group)
        result = get_dispatch_backend("balanced_reference").rebalance(batch, dispatcher)
        (result.hidden.sum() + result.gates.sum()).backward()
        queue.put(
            (
                rank,
                result.layout.position_ids.tolist(),
                result.hidden.detach().flatten().tolist(),
                hidden.grad.flatten().tolist(),
                gates.grad.flatten().tolist(),
                dispatcher.counters.rebalance_calls,
                dispatcher.counters.collective_calls,
            )
        )
    finally:
        dist.destroy_process_group()


def _empty_receive_worker(rank: int, init_path: str, queue: multiprocessing.Queue) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=2,
    )
    try:
        group = TPxCPRouteGroup.from_process_group(
            dist.group.WORLD,
            tp_size=1,
            cp_size=2,
            global_ranks=(0, 1),
        )
        hidden = torch.tensor([[float(rank + 1)]], requires_grad=True)
        batch = ActiveTokenBatch(
            hidden=hidden,
            layout=ActiveTokenLayout.from_local(
                sample_ids=torch.tensor([rank], dtype=torch.long),
                position_ids=torch.tensor([0], dtype=torch.long),
                global_token_ids=torch.tensor([rank], dtype=torch.long),
                route_rank=rank,
            ),
        )
        dispatcher = ActiveTokenDispatcher(group)
        result = dispatcher.dispatch(
            batch,
            torch.tensor([0], dtype=torch.long),
            destination_slots=torch.tensor([rank], dtype=torch.long),
        )
        result.hidden.sum().backward()
        queue.put(
            (
                rank,
                result.num_tokens,
                result.layout.destination_slots is not None,
                result.layout.destination_slots.tolist()
                if result.layout.destination_slots is not None
                else None,
                hidden.grad.flatten().tolist(),
            )
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not getattr(dist, "is_gloo_available", lambda: False)(),
    reason="Gloo distributed backend is unavailable",
)
def test_two_rank_variable_all_to_all_is_balanced_and_differentiable(tmp_path: Path):
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    init_path = str(tmp_path / "gloo-init")
    processes = [
        context.Process(target=_two_rank_worker, args=(rank, init_path, queue)) for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            pytest.fail("two-rank active dispatch timed out")
        assert process.exitcode == 0
    results = sorted([queue.get(timeout=5) for _ in range(2)])
    assert results[0][1:3] == ([0, 1], [0.0, 1.0])
    assert results[1][1:3] == ([2, 3], [2.0, 3.0])
    for result in results:
        assert result[3] == [1.0, 1.0]
        assert result[4] == [1.0, 1.0]
        assert result[5] == 1
        assert result[6] >= 6


@pytest.mark.skipif(
    not dist.is_available() or not getattr(dist, "is_gloo_available", lambda: False)(),
    reason="Gloo distributed backend is unavailable",
)
def test_explicit_destination_slots_survive_an_empty_receive_rank(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    init_path = str(tmp_path / "gloo-empty-receive")
    processes = [
        context.Process(
            target=_empty_receive_worker,
            args=(rank, init_path, queue),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            pytest.fail("two-rank empty-receive dispatch timed out")
        assert process.exitcode == 0

    rank0, rank1 = sorted(queue.get(timeout=5) for _ in range(2))
    assert rank0[1:4] == (2, True, [0, 1])
    assert rank1[1:4] == (0, True, [])
    assert rank0[4] == [1.0]
    assert rank1[4] == [1.0]
