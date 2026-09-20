from __future__ import annotations

import multiprocessing
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from mor_mlite.data import make_synthetic_batch
from mor_mlite.distributed import TPxCPRouteGroup
from mor_mlite.tiny.model import PositionAwareGQA, TinyMoRConfig, TinyMoRModel


def _hidden() -> torch.Tensor:
    return torch.arange(4 * 32, dtype=torch.float32).reshape(4, 32) / 127.0


def _static_cp_worker(
    rank: int,
    init_path: str,
    queue: multiprocessing.Queue,
) -> None:
    dist.init_process_group("gloo", init_method=f"file://{init_path}", rank=rank, world_size=2)
    try:
        group = TPxCPRouteGroup.from_process_group(
            dist.group.WORLD,
            tp_size=1,
            cp_size=2,
            global_ranks=(0, 1),
        )
        torch.manual_seed(77)
        attention = PositionAwareGQA(TinyMoRConfig())
        local_rows = torch.tensor([0, 3] if rank == 0 else [1, 2], dtype=torch.long)
        local_hidden = _hidden().index_select(0, local_rows).clone().requires_grad_(True)
        output = attention(
            local_hidden,
            sample_ids=torch.zeros(2, dtype=torch.long),
            original_positions=local_rows,
            global_token_ids=local_rows + 100,
            static_cp_group=group,
        )
        output.square().sum().backward()
        queue.put(
            {
                "rank": rank,
                "ids": (local_rows + 100).tolist(),
                "output": output.detach().tolist(),
                "input_grad": local_hidden.grad.tolist(),
                "parameter_grads": {
                    name: parameter.grad.detach().tolist()
                    for name, parameter in attention.named_parameters()
                },
            }
        )
    finally:
        dist.destroy_process_group()


def _static_cp_model_worker(
    rank: int,
    init_path: str,
    queue: multiprocessing.Queue,
) -> None:
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{init_path}", rank=rank, world_size=2)
    try:
        group = TPxCPRouteGroup.from_process_group(
            dist.group.WORLD,
            tp_size=1,
            cp_size=2,
            global_ranks=(0, 1),
        )
        batch = make_synthetic_batch(
            seq_lens=(9, 6, 3),
            seed=919,
            extreme_routing=True,
        )
        model = TinyMoRModel(seed=313)
        output = model(batch, static_cp_group=group)
        assert output.total_loss is not None
        output.total_loss.backward()

        averaged_gradients: dict[str, torch.Tensor | None] = {}
        for name, parameter in model.named_parameters():
            globally_present = torch.tensor(int(parameter.grad is not None), dtype=torch.int32)
            dist.all_reduce(globally_present, op=dist.ReduceOp.MAX)
            if not int(globally_present.item()):
                averaged_gradients[name] = None
                continue
            gradient = torch.zeros_like(parameter) if parameter.grad is None else parameter.grad
            dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
            gradient.div_(2.0)
            averaged_gradients[name] = gradient.detach().clone()

        # The serial oracle must run after destroying WORLD: DepthRouter treats
        # an initialized default group as its auxiliary-loss reduction group.
        dist.destroy_process_group()

        baseline_model = TinyMoRModel(seed=313)
        baseline = baseline_model(batch)
        assert baseline.total_loss is not None
        baseline.total_loss.backward()

        sorted_ids, id_order = torch.sort(baseline.global_token_ids)
        lookup = torch.searchsorted(sorted_ids, output.global_token_ids)
        baseline_rows = id_order.index_select(0, lookup)
        torch.testing.assert_close(
            output.logits,
            baseline.logits.index_select(0, baseline_rows),
            rtol=2e-5,
            atol=2e-6,
        )
        torch.testing.assert_close(output.lm_loss, baseline.lm_loss, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(output.aux_loss, baseline.aux_loss, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(output.total_loss, baseline.total_loss, rtol=2e-5, atol=2e-6)
        for actual_hidden, expected_hidden in zip(
            output.hidden_by_round, baseline.hidden_by_round, strict=True
        ):
            torch.testing.assert_close(
                actual_hidden,
                expected_hidden.index_select(0, baseline_rows),
                rtol=2e-5,
                atol=2e-6,
            )
        for actual_plan, expected_plan in zip(
            output.route_plans, baseline.route_plans, strict=True
        ):
            assert torch.equal(actual_plan.sample_ids, expected_plan.sample_ids)
            assert torch.equal(actual_plan.original_positions, expected_plan.original_positions)
            assert torch.equal(actual_plan.global_token_ids, expected_plan.global_token_ids)
            assert torch.equal(actual_plan.active_cu_seqlens, expected_plan.active_cu_seqlens)
            assert torch.equal(actual_plan.padding_mask, expected_plan.padding_mask)
            torch.testing.assert_close(
                actual_plan.selected_gates,
                expected_plan.selected_gates,
                rtol=2e-5,
                atol=2e-6,
            )
        for name, expected_parameter in baseline_model.named_parameters():
            actual_gradient = averaged_gradients[name]
            if expected_parameter.grad is None:
                assert actual_gradient is None
            else:
                assert actual_gradient is not None
                torch.testing.assert_close(
                    actual_gradient,
                    expected_parameter.grad,
                    rtol=5e-5,
                    atol=5e-6,
                )
        queue.put(
            {
                "rank": rank,
                "global_token_ids": output.global_token_ids.tolist(),
                "last_round_local_active": int(
                    (output.route_plans[-1].source_cp_ranks == rank).sum().item()
                ),
                "communication": output.communication,
            }
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not getattr(dist, "is_gloo_available", lambda: False)(),
    reason="Gloo distributed backend is unavailable",
)
def test_static_cp_active_qkv_gather_matches_single_rank_forward_and_backward(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    init_path = str(tmp_path / "static-cp-gloo-init")
    processes = [
        context.Process(target=_static_cp_worker, args=(rank, init_path, queue))
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=45)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            pytest.fail("two-rank static CP attention timed out")
        assert process.exitcode == 0
    results = sorted((queue.get(timeout=5) for _ in range(2)), key=lambda item: item["rank"])

    torch.manual_seed(77)
    baseline = PositionAwareGQA(TinyMoRConfig())
    full_hidden = _hidden().requires_grad_(True)
    expected_output = baseline(
        full_hidden,
        sample_ids=torch.zeros(4, dtype=torch.long),
        original_positions=torch.arange(4),
        global_token_ids=torch.arange(4) + 100,
    )
    expected_output.square().sum().backward()

    output_by_id: dict[int, torch.Tensor] = {}
    grad_by_id: dict[int, torch.Tensor] = {}
    for result in results:
        for token_id, output, gradient in zip(
            result["ids"], result["output"], result["input_grad"], strict=True
        ):
            output_by_id[int(token_id)] = torch.tensor(output)
            grad_by_id[int(token_id)] = torch.tensor(gradient)
    actual_output = torch.stack([output_by_id[token_id] for token_id in range(100, 104)])
    actual_input_grad = torch.stack([grad_by_id[token_id] for token_id in range(100, 104)])
    torch.testing.assert_close(actual_output, expected_output.detach(), rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual_input_grad, full_hidden.grad, rtol=5e-5, atol=5e-6)

    for name, parameter in baseline.named_parameters():
        summed = sum(
            (torch.tensor(result["parameter_grads"][name]) for result in results),
            start=torch.zeros_like(parameter),
        )
        torch.testing.assert_close(summed, parameter.grad, rtol=5e-5, atol=5e-6)


@pytest.mark.skipif(
    not dist.is_available() or not getattr(dist, "is_gloo_available", lambda: False)(),
    reason="Gloo distributed backend is unavailable",
)
def test_static_cp_full_tiny_mor_matches_single_rank_forward_and_backward(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    init_path = str(tmp_path / "static-cp-model-gloo-init")
    processes = [
        context.Process(target=_static_cp_model_worker, args=(rank, init_path, queue))
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=60)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            pytest.fail("two-rank static CP full-model diagnostic timed out")
        assert process.exitcode == 0
    results = sorted((queue.get(timeout=5) for _ in range(2)), key=lambda item: item["rank"])
    all_ids = [token_id for result in results for token_id in result["global_token_ids"]]
    assert sorted(all_ids) == list(range(18))
    assert results[1]["last_round_local_active"] == 0
    for result in results:
        communication = result["communication"]
        assert communication["active_set_changes"] == 2
        assert communication["hidden_rebalances"] == 2
        assert communication["hidden_all_to_all"] == 0
        assert communication["gate_all_to_all"] == 0
        assert communication["metadata_all_to_all"] == 0
        assert communication["route_gathers"] == 3
        assert communication["skipped_full_first_round"] == 1
        assert communication["recurrent_inner_dispatches"] == 0
        assert communication["early_exit_qkv_tokens"] == 0
