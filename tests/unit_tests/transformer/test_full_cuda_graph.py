# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch
from pytest_mock import mocker

import megatron.core.pipeline_parallel.schedules as schedule
from megatron.core import ModelParallelConfig
from megatron.core.full_cuda_graph import (
    FullCudaGraphWrapper,
    _has_v2_fsdp_modules,
    _stop_v2_fsdp_communication,
    _synchronize_fsdp_param_gathers,
)
from megatron.core.tensor_parallel.random import (
    HAVE_TE,
    initialize_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.utils import is_te_min_version
from tests.unit_tests.test_utilities import Utils

rank = Utils.rank


class _FSDPStopModule(torch.nn.Module):
    def __init__(self, use_megatron_fsdp_v2=True):
        super().__init__()
        self.ddp_config = SimpleNamespace(use_megatron_fsdp_v2=use_megatron_fsdp_v2)
        self.stop_calls = 0
        self.gather_calls = 0

    def stop_communication(self):
        self.stop_calls += 1

    def synchronize_param_gather(self):
        self.gather_calls += 1


def test_synchronize_fsdp_param_gathers_walks_unique_modules():
    root = torch.nn.Module()
    child = _FSDPStopModule()
    root.add_module("child", child)

    synchronized = _synchronize_fsdp_param_gathers([root, child])

    assert synchronized == 1
    assert child.stop_calls == 0
    assert child.gather_calls == 1


def test_has_v2_fsdp_modules_walks_nested_model_list():
    root = torch.nn.Module()
    v1_child = _FSDPStopModule(use_megatron_fsdp_v2=False)
    v2_child = _FSDPStopModule()
    root.add_module("v1_child", v1_child)

    assert not _has_v2_fsdp_modules([root, v1_child])
    assert _has_v2_fsdp_modules([root, v2_child])


def test_v2_fsdp_forward_only_stays_eager(mocker):
    eager_result = object()
    forward_backward_func = mocker.Mock(return_value=eager_result)
    wrapper = FullCudaGraphWrapper(forward_backward_func)
    model = [_FSDPStopModule()]

    result = wrapper(
        model=model,
        data_iterator=[iter(())],
        num_microbatches=1,
        seq_length=None,
        forward_only=True,
    )

    assert result is eager_result
    forward_backward_func.assert_called_once()


def test_stop_v2_fsdp_communication_skips_v1_modules():
    root = torch.nn.Module()
    v2_child = _FSDPStopModule()
    v1_child = _FSDPStopModule(use_megatron_fsdp_v2=False)
    root.add_module("v2_child", v2_child)
    root.add_module("v1_child", v1_child)

    stopped = _stop_v2_fsdp_communication(root)

    assert stopped == 1
    assert v2_child.stop_calls == 1
    assert v1_child.stop_calls == 0


@pytest.mark.skipif(
    not (HAVE_TE and is_te_min_version("1.5.0")),
    reason="use_te_rng_tracker requires TransformerEngine version >= 1.5",
)
def test_forward_backward_func_with_full_cuda_graph(mocker):
    from megatron.core.pipeline_parallel import get_forward_backward_func

    initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
    Utils.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=1)

    def forward_step_func(data_iterator, model):
        import os

        rank = int(os.environ['LOCAL_RANK'])
        dummy_data = torch.ones(1, 4)

        def loss_func(output_tensor):
            return rank, {'loss_reduced': rank}

        return model(dummy_data), loss_func

    model = torch.nn.Linear(4, 1)

    model.model_type = 'unit-test'

    def set_input_tensor(input_tensor):
        return None

    model.set_input_tensor = set_input_tensor

    forward_backward_func = get_forward_backward_func()
    assert schedule.get_forward_backward_func() == schedule.forward_backward_no_pipelining

    # Wrapping the forward_backward_func with FullCudaGraphWrapper enables full iteration CUDA graphs.
    forward_backward_func = FullCudaGraphWrapper(forward_backward_func)
    mocker.patch("megatron.core.pipeline_parallel.schedules.custom_backward", return_value=2)
    config = ModelParallelConfig(pipeline_model_parallel_size=1)
    model.config = config

    num_microbatches = 4

    # CUDA graph warmup
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=[iter([{'input': torch.ones(1, 4)}] * num_microbatches)],
        model=[model],
        num_microbatches=num_microbatches,
        seq_length=None,
        micro_batch_size=None,
        forward_only=True,
    )
    # CUDA graph capture and replay
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=[iter([{'input': torch.ones(1, 4)}] * num_microbatches)],
        model=[model],
        num_microbatches=num_microbatches,
        seq_length=None,
        micro_batch_size=None,
        forward_only=True,
    )
    loss_reduced_expected = [
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
    ]

    for i, j in zip(losses_reduced, loss_reduced_expected):
        print(losses_reduced)
        assert i['loss_reduced'] == j['loss_reduced']
    Utils.destroy_model_parallel()
