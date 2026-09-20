from __future__ import annotations

import multiprocessing
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
dist = pytest.importorskip("torch.distributed")

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from mor_mlite.config import MoRArchitectureConfig
from mor_mlite.distributed import (
    ActiveTokenBatch,
    ActiveTokenLayout,
    TPxCPRouteGroup,
    create_dense_dp_route_group,
    distributed_depth_route,
)
from mor_mlite.routing import DepthRouter


def _router(round_index: int) -> DepthRouter:
    router = DepthRouter(
        hidden_size=2,
        architecture=MoRArchitectureConfig.tiny(),
        round_index=round_index,
    )
    with torch.no_grad():
        router.proj.weight.copy_(torch.tensor([[1.0, 0.0]]))
    return router


def _batch(
    hidden: torch.Tensor,
    *,
    sample_ids: list[int],
    positions: list[int],
    global_ids: list[int],
    route_rank: int = 0,
    padding_mask: torch.Tensor | None = None,
) -> ActiveTokenBatch:
    return ActiveTokenBatch(
        hidden=hidden,
        layout=ActiveTokenLayout.from_local(
            sample_ids=torch.tensor(sample_ids, dtype=torch.long, device=hidden.device),
            position_ids=torch.tensor(positions, dtype=torch.long, device=hidden.device),
            global_token_ids=torch.tensor(global_ids, dtype=torch.long, device=hidden.device),
            route_rank=route_rank,
            padding_mask=padding_mask,
            drop_padding=False,
        ),
    )


def test_full_first_round_keeps_current_local_layout_for_zero_dispatch() -> None:
    batch = _batch(
        torch.tensor([[2.0, 0.0], [0.0, 0.0], [1.0, 0.0]]),
        sample_ids=[0, 0, 0],
        positions=[2, 0, 1],
        global_ids=[12, 10, 11],
    )
    result = distributed_depth_route(_router(0), batch, original_lengths={0: 3})
    assert result.plan.global_token_ids.tolist() == [10, 11, 12]
    assert result.selected_local_indices.tolist() == [0, 1, 2]
    assert result.selected_batch.layout.global_token_ids.tolist() == [12, 10, 11]


def test_local_distributed_router_excludes_padding_and_returns_live_gates() -> None:
    hidden = torch.tensor(
        [[0.0, 0.0], [4.0, 0.0], [3.0, 0.0], [100.0, 0.0]],
        requires_grad=True,
    )
    batch = _batch(
        hidden,
        sample_ids=[0, 0, 0, -1],
        positions=[0, 1, 2, -1],
        global_ids=[10, 11, 12, -1],
        padding_mask=torch.tensor([False, False, False, True]),
    )
    result = distributed_depth_route(
        _router(1),
        batch,
        original_lengths={0: 3},
    )

    assert result.plan.global_token_ids.tolist() == [11, 12]
    assert result.selected_local_indices.tolist() == [1, 2]
    assert result.selected_local_mask.tolist() == [False, True, True, False]
    assert result.selected_batch.layout.global_token_ids.tolist() == [11, 12]
    assert result.selected_local_gates.requires_grad
    assert result.route_gathers == 0
    assert result.route_collective_calls == 0
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        result.raw_logits[:3], torch.tensor([0.0, 1.0, 1.0])
    )
    torch.testing.assert_close(result.aux_loss, expected)


def test_candidates_from_previous_round_make_selection_nested_and_replay_exact() -> None:
    hidden = torch.tensor([[0.0, 0.0], [5.0, 0.0], [2.0, 0.0], [4.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    initial = _batch(
        hidden,
        sample_ids=[0] * 6,
        positions=list(range(6)),
        global_ids=list(range(100, 106)),
    )
    second = distributed_depth_route(_router(1), initial, original_lengths={0: 6})
    third = distributed_depth_route(_router(2), second.selected_batch, original_lengths={0: 6})
    assert set(third.plan.global_token_ids.tolist()) < set(second.plan.global_token_ids.tolist())

    permutation = torch.tensor([2, 0, 1, 3])
    permuted = second.selected_batch.index_select(permutation)
    replay_router = _router(2)
    with torch.no_grad():
        replay_router.proj.weight.copy_(torch.tensor([[-1.0, 0.0]]))
    replayed = distributed_depth_route(
        replay_router,
        permuted,
        original_lengths={0: 6},
        replay_plan=third.plan,
    )
    assert replayed.plan.mode == "replay"
    assert replayed.plan.global_token_ids.tolist() == third.plan.global_token_ids.tolist()
    torch.testing.assert_close(
        replayed.plan.selected_gates, third.plan.selected_gates, rtol=0, atol=0
    )
    local_ids = replayed.selected_batch.layout.global_token_ids
    torch.testing.assert_close(
        replayed.selected_local_gates,
        third.plan.replay_gates(local_ids),
        rtol=0,
        atol=0,
    )
    replayed.selected_local_gates.sum().backward()
    assert replay_router.proj.weight.grad is not None
    assert torch.count_nonzero(replay_router.proj.weight.grad) > 0


def test_distributed_replay_rejects_wrong_budget_even_when_ids_are_valid():
    batch = _batch(
        torch.ones(6, 2),
        sample_ids=[0] * 6,
        positions=list(range(6)),
        global_ids=list(range(6)),
    )
    too_large = distributed_depth_route(_router(0), batch, original_lengths={0: 6}).plan
    with pytest.raises(ValueError, match="replay capacity mismatch"):
        distributed_depth_route(
            _router(1),
            batch,
            original_lengths={0: 6},
            replay_plan=replace(too_large, round_index=1),
        )


def _two_rank_route_worker(
    rank: int,
    init_path: str,
    queue: multiprocessing.Queue,
) -> None:
    dist.init_process_group("gloo", init_method=f"file://{init_path}", rank=rank, world_size=2)
    try:
        route_group = TPxCPRouteGroup.from_process_group(
            dist.group.WORLD,
            tp_size=1,
            cp_size=2,
            global_ranks=(0, 1),
        )
        ids = [0, 2] if rank == 0 else [1, 3]
        values = [0.0, 4.0] if rank == 0 else [4.0, 1.0]
        hidden = torch.tensor(
            [[value, 0.0] for value in values], dtype=torch.float32, requires_grad=True
        )
        router = _router(1)
        output = distributed_depth_route(
            router,
            _batch(
                hidden,
                sample_ids=[0, 0],
                positions=ids,
                global_ids=ids,
                route_rank=rank,
            ),
            original_lengths={0: 4},
            route_group=route_group,
        )
        output.aux_loss.backward()
        queue.put(
            (
                rank,
                output.plan.to_dict(),
                output.selected_batch.layout.global_token_ids.tolist(),
                float(output.aux_loss.detach()),
                float(router.proj.weight.grad[0, 0]),
                output.selected_local_gates.requires_grad,
                output.route_gathers,
                output.route_collective_calls,
            )
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not getattr(dist, "is_gloo_available", lambda: False)(),
    reason="Gloo distributed backend is unavailable",
)
def test_two_rank_plan_is_identical_and_cp_gradient_scale_matches_global_bce(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    init_path = str(tmp_path / "route-gloo-init")
    processes = [
        context.Process(target=_two_rank_route_worker, args=(rank, init_path, queue))
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            pytest.fail("two-rank distributed router timed out")
        assert process.exitcode == 0
    results = sorted(queue.get(timeout=5) for _ in range(2))

    assert results[0][1] == results[1][1]
    assert results[0][1]["global_token_ids"] == [1, 2]
    assert results[0][2] == [2]
    assert results[1][2] == [1]
    # The replicated forward scalar comes directly from the same all-reduce,
    # not a shard-dependent cancellation surrogate, so peers are bitwise equal.
    assert results[0][3] == results[1][3]
    assert results[0][5] and results[1][5]
    assert results[0][6:] == (1, 6)
    assert results[1][6:] == (1, 6)

    # MLite TP finalize is a SUM and dense DPxCP synchronization is a mean.
    # With TP=1, CP=2, averaging the two locally backpropagated gradients must
    # equal the derivative of the single-process global BCE mean.
    averaged_distributed_gradient = (results[0][4] + results[1][4]) / 2.0
    logits = torch.tensor([0.0, 4.0, 4.0, 1.0], requires_grad=True)
    targets = torch.tensor([0.0, 1.0, 1.0, 0.0])
    features = torch.tensor([0.0, 4.0, 4.0, 1.0])
    expected = ((torch.sigmoid(logits.detach()) - targets) * features).sum() / logits.numel()
    assert averaged_distributed_gradient == pytest.approx(float(expected), abs=1e-6)


def _four_rank_group_worker(
    rank: int,
    init_path: str,
    queue: multiprocessing.Queue,
) -> None:
    dist.init_process_group("gloo", init_method=f"file://{init_path}", rank=rank, world_size=4)
    try:
        ps = SimpleNamespace(
            tp_size=1,
            cp_size=2,
            dp_size=2,
            pp_size=1,
            dp_rank=rank // 2,
        )
        route_group = create_dense_dp_route_group(ps)
        queue.put((rank, route_group.global_ranks, route_group.rank))
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not getattr(dist, "is_gloo_available", lambda: False)(),
    reason="Gloo distributed backend is unavailable",
)
def test_group_builder_never_crosses_dense_dp_replicas(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    init_path = str(tmp_path / "group-gloo-init")
    processes = [
        context.Process(target=_four_rank_group_worker, args=(rank, init_path, queue))
        for rank in range(4)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            pytest.fail("four-rank route-group construction timed out")
        assert process.exitcode == 0
    results = sorted(queue.get(timeout=5) for _ in range(4))
    assert results == [
        (0, (0, 1), 0),
        (1, (0, 1), 1),
        (2, (2, 3), 0),
        (3, (2, 3), 1),
    ]
