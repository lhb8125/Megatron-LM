from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from mor_mlite.parity.mlite import (
    _canonical_optimizer_step,
    _distributed_optimizer_fingerprint,
    _optimizer_fingerprint_payload,
)


def _handle_for(optimizer: torch.optim.Optimizer) -> SimpleNamespace:
    model = nn.Module()
    group_map = {}
    param_map = {}
    parameter_index = 0
    for group_index, group in enumerate(optimizer.param_groups):
        for group_order, main_parameter in enumerate(group["params"]):
            name = f"weight_{parameter_index}"
            model.register_parameter(
                name,
                nn.Parameter(torch.zeros_like(main_parameter, dtype=torch.bfloat16)),
            )
            model_parameter = getattr(model, name)
            group_map[model_parameter] = (group_index, group_order)
            param_map[model_parameter] = {
                "param": SimpleNamespace(start=0, end=model_parameter.numel())
            }
            parameter_index += 1
    leaf = SimpleNamespace(
        optimizer=optimizer,
        model_param_group_index_map=group_map,
        gbuf_ranges=[
            {
                (torch.bfloat16, torch.float32): [
                    {
                        "param_map": param_map,
                    }
                ]
            }
        ],
    )
    return SimpleNamespace(
        _optimizer=SimpleNamespace(chained_optimizers=[leaf]),
        _model=(model,),
        _parallel_state=SimpleNamespace(
            tp_rank=0,
            tp_size=1,
            cp_rank=0,
            dp_rank=0,
            ep_rank=0,
            ep_size=1,
            etp_rank=0,
            expert_dp_rank=0,
        ),
        _extras={
            "model_chunks": (model,),
            "model_cfg": SimpleNamespace(num_experts=4, vocab_size=257),
        },
    )


def _stepped_adam() -> tuple[torch.nn.Parameter, torch.optim.Adam]:
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float32))
    optimizer = torch.optim.Adam([parameter], lr=1.0e-3)
    parameter.grad = torch.tensor([0.25, -0.5], dtype=torch.float32)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return parameter, optimizer


def test_optimizer_fingerprint_covers_master_weights_and_adam_moments() -> None:
    parameter, optimizer = _stepped_adam()
    handle = _handle_for(optimizer)

    baseline = _distributed_optimizer_fingerprint(handle)

    assert baseline["rank_count"] == 1
    assert baseline["optimizer_leaf_counts"] == [1]
    assert baseline["master_parameter_counts"] == [1]
    assert baseline["master_parameter_bytes"] == [parameter.numel() * parameter.element_size()]
    assert baseline["adam_moment_tensor_counts"] == [2]
    assert baseline["adam_moment_tensor_bytes"] == [
        2 * parameter.numel() * parameter.element_size()
    ]

    with torch.no_grad():
        parameter[0] += 1.0
    master_mutated = _distributed_optimizer_fingerprint(handle)
    assert master_mutated["sha256"] != baseline["sha256"]

    with torch.no_grad():
        parameter[0] -= 1.0
        optimizer.state[parameter]["exp_avg"][0] += 1.0
    moment_mutated = _distributed_optimizer_fingerprint(handle)
    assert moment_mutated["sha256"] != baseline["sha256"]

    with torch.no_grad():
        optimizer.state[parameter]["exp_avg"][0] -= 1.0
        optimizer.state[parameter]["exp_avg_sq"][0] += 1.0
    second_moment_mutated = _distributed_optimizer_fingerprint(handle)
    assert second_moment_mutated["sha256"] != baseline["sha256"]

    with torch.no_grad():
        optimizer.state[parameter]["exp_avg_sq"][0] -= 1.0
        optimizer.state[parameter]["step"] += 1.0
    step_mutated = _distributed_optimizer_fingerprint(handle)
    assert step_mutated["sha256"] != baseline["sha256"]


def test_optimizer_fingerprint_is_stable_across_fresh_optimizer_load() -> None:
    parameter, optimizer = _stepped_adam()
    baseline = _distributed_optimizer_fingerprint(_handle_for(optimizer))

    fresh_parameter = torch.nn.Parameter(parameter.detach().clone())
    fresh_optimizer = torch.optim.Adam([fresh_parameter], lr=1.0e-3)
    fresh_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))

    assert _distributed_optimizer_fingerprint(_handle_for(fresh_optimizer)) == baseline


def test_optimizer_fingerprint_normalizes_parameter_step_representation() -> None:
    parameter, optimizer = _stepped_adam()
    baseline = _distributed_optimizer_fingerprint(_handle_for(optimizer))

    numeric_step = int(optimizer.state[parameter]["step"].item())
    optimizer.state[parameter]["step"] = numeric_step
    assert _distributed_optimizer_fingerprint(_handle_for(optimizer)) == baseline

    optimizer.state[parameter]["step"] = float(numeric_step)
    assert _distributed_optimizer_fingerprint(_handle_for(optimizer)) == baseline

    optimizer.state[parameter]["step"] = numeric_step + 1
    assert _distributed_optimizer_fingerprint(_handle_for(optimizer)) != baseline


def test_optimizer_fingerprint_normalizes_group_step_representation() -> None:
    parameter, optimizer = _stepped_adam()
    optimizer.param_groups[0]["step"] = optimizer.state[parameter].pop("step")
    baseline = _distributed_optimizer_fingerprint(_handle_for(optimizer))

    numeric_step = int(optimizer.param_groups[0]["step"].item())
    optimizer.param_groups[0]["step"] = numeric_step
    assert _distributed_optimizer_fingerprint(_handle_for(optimizer)) == baseline

    optimizer.param_groups[0]["step"] = numeric_step + 1
    assert _distributed_optimizer_fingerprint(_handle_for(optimizer)) != baseline


def test_optimizer_fingerprint_ignores_restored_empty_group_step_count() -> None:
    parameter, optimizer = _stepped_adam()
    baseline = _distributed_optimizer_fingerprint(_handle_for(optimizer))

    empty_group = {
        key: copy.deepcopy(value)
        for key, value in optimizer.param_groups[0].items()
        if key != "params"
    }
    empty_group["params"] = []
    empty_group["step"] = int(optimizer.state[parameter]["step"].item())
    optimizer.param_groups.append(empty_group)

    assert _distributed_optimizer_fingerprint(_handle_for(optimizer)) == baseline


def test_group_step_counts_each_owned_parameter_once_and_ignores_empty_groups() -> None:
    first = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float32))
    second = torch.nn.Parameter(torch.tensor([3.0, -4.0], dtype=torch.float32))
    optimizer = torch.optim.Adam([first, second], lr=1.0e-3)
    first.grad = torch.tensor([0.25, -0.5])
    second.grad = torch.tensor([-0.75, 1.0])
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    group_step = int(optimizer.state[first].pop("step").item())
    assert int(optimizer.state[second].pop("step").item()) == group_step
    optimizer.param_groups[0]["step"] = group_step

    baseline = _distributed_optimizer_fingerprint(_handle_for(optimizer))
    assert baseline["optimizer_step_counts"] == [2]

    empty_group = {
        key: copy.deepcopy(value)
        for key, value in optimizer.param_groups[0].items()
        if key != "params"
    }
    empty_group["params"] = []
    optimizer.param_groups.append(empty_group)
    assert _distributed_optimizer_fingerprint(_handle_for(optimizer)) == baseline

    optimizer.param_groups[0]["step"] = group_step + 1
    assert _distributed_optimizer_fingerprint(_handle_for(optimizer)) != baseline


def test_group_step_counts_each_owned_parameter_and_still_hashes_its_value() -> None:
    first = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float32))
    second = torch.nn.Parameter(torch.tensor([3.0, -4.0], dtype=torch.float32))
    optimizer = torch.optim.Adam([first, second], lr=1.0e-3)
    first.grad = torch.tensor([0.25, -0.5])
    second.grad = torch.tensor([-0.75, 1.0])
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    step = int(optimizer.state[first].pop("step").item())
    assert int(optimizer.state[second].pop("step").item()) == step
    optimizer.param_groups[0]["step"] = step

    handle = _handle_for(optimizer)
    baseline = _distributed_optimizer_fingerprint(handle)
    assert baseline["optimizer_step_counts"] == [2]

    optimizer.param_groups[0]["step"] = step + 1
    assert _distributed_optimizer_fingerprint(handle) != baseline


@pytest.mark.parametrize(
    "value",
    [
        True,
        -1,
        1.5,
        float("nan"),
        float("inf"),
        torch.tensor([1, 2]),
        torch.tensor(1 + 0j),
    ],
)
def test_optimizer_step_canonicalization_rejects_invalid_values(value: object) -> None:
    with pytest.raises(TypeError, match="optimizer step"):
        _canonical_optimizer_step(value, location="test")


def test_optimizer_fingerprint_is_stable_when_mcore_reorders_groups() -> None:
    first = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float32))
    second = torch.nn.Parameter(torch.tensor([3.0, -4.0], dtype=torch.float32))
    optimizer = torch.optim.Adam(
        [
            {"params": [first], "lr": 1.0e-3},
            {"params": [second], "lr": 2.0e-3},
        ]
    )
    first.grad = torch.tensor([0.25, -0.5])
    second.grad = torch.tensor([-0.75, 1.0])
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    handle = _handle_for(optimizer)
    baseline = _distributed_optimizer_fingerprint(handle)

    leaf = handle._optimizer.chained_optimizers[0]
    named_parameters = dict(handle._model[0].named_parameters())
    optimizer.param_groups.reverse()
    leaf.model_param_group_index_map = {
        named_parameters["weight_0"]: (1, 0),
        named_parameters["weight_1"]: (0, 0),
    }

    assert _distributed_optimizer_fingerprint(handle) == baseline


def test_optimizer_fingerprint_fails_closed_before_adam_state_exists() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = torch.optim.Adam([parameter], lr=1.0e-3)

    with pytest.raises(RuntimeError, match="no initialized Adam moment tensors"):
        _optimizer_fingerprint_payload(_handle_for(optimizer))


def test_optimizer_fingerprint_fails_closed_when_adam_step_is_missing() -> None:
    parameter, optimizer = _stepped_adam()
    del optimizer.state[parameter]["step"]

    with pytest.raises(RuntimeError, match="has no optimizer step"):
        _optimizer_fingerprint_payload(_handle_for(optimizer))


def test_optimizer_fingerprint_rejects_non_fp32_master_parameters() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.bfloat16))
    optimizer = torch.optim.Adam([parameter], lr=1.0e-3)
    parameter.grad = torch.tensor([0.25], dtype=torch.bfloat16)
    optimizer.step()

    with pytest.raises(RuntimeError, match="expected torch.float32"):
        _optimizer_fingerprint_payload(_handle_for(optimizer))
