# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch
from pytest_mock import mocker

import megatron.core.pipeline_parallel.schedules as schedule
from megatron.core import ModelParallelConfig
from megatron.core.full_cuda_graph import FullCudaGraphWrapper
from megatron.core.tensor_parallel.random import (
    HAVE_TE,
    initialize_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.utils import is_te_min_version
from tests.unit_tests.test_utilities import Utils

rank = Utils.rank


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


def test_full_cuda_graph_replay_has_nvtx_range(mocker):
    graph = mocker.Mock()
    mocker.patch.object(FullCudaGraphWrapper, 'curr_iteration', {'training': 2, 'validation': 0})
    mocker.patch.object(FullCudaGraphWrapper, 'cuda_graph', {'training': graph, 'validation': None})
    mocker.patch.object(
        FullCudaGraphWrapper, 'result', {'training': 'captured-result', 'validation': None}
    )
    push = mocker.patch('megatron.core.full_cuda_graph.nvtx_range_push')
    pop = mocker.patch('megatron.core.full_cuda_graph.nvtx_range_pop')

    wrapper = FullCudaGraphWrapper.__new__(FullCudaGraphWrapper)
    wrapper.cuda_graph_warmup_steps = 1
    wrapper.data_read = mocker.Mock(return_value=[])

    result = wrapper(
        model=[], data_iterator=[], num_microbatches=1, seq_length=1, forward_only=False
    )

    assert result == 'captured-result'
    graph.replay.assert_called_once_with()
    push.assert_called_once_with('megatron.core.full_cuda_graph.replay.training')
    pop.assert_called_once_with('megatron.core.full_cuda_graph.replay.training')


def test_full_cuda_graph_capture_disables_autograd_multithreading(mocker):
    graph = mocker.Mock()
    mocker.patch.object(FullCudaGraphWrapper, 'curr_iteration', {'training': 1, 'validation': 0})
    mocker.patch.object(FullCudaGraphWrapper, 'cuda_graph', {'training': None, 'validation': None})
    mocker.patch.object(FullCudaGraphWrapper, 'result', {'training': None, 'validation': None})
    mocker.patch('megatron.core.full_cuda_graph.get_all_rng_states', return_value={})
    mocker.patch('megatron.core.full_cuda_graph.get_shared_capture_stream', return_value='stream')
    mocker.patch('megatron.core.full_cuda_graph.get_graph_pool', return_value='pool')
    mocker.patch('megatron.core.full_cuda_graph.torch.cuda.CUDAGraph', return_value=graph)
    mocker.patch('megatron.core.full_cuda_graph.torch.cuda.graph', return_value=mocker.MagicMock())
    mocker.patch('megatron.core.full_cuda_graph.torch.cuda.synchronize')
    mocker.patch('megatron.core.full_cuda_graph.torch.distributed.barrier')
    mocker.patch('megatron.core.full_cuda_graph.is_nvtx_profiling_enabled', return_value=True)
    multithreading_context = mocker.patch(
        'megatron.core.full_cuda_graph.torch.autograd.set_multithreading_enabled',
        return_value=mocker.MagicMock(),
    )

    wrapper = FullCudaGraphWrapper.__new__(FullCudaGraphWrapper)
    wrapper.forward_backward_func = mocker.Mock(return_value='captured-result')
    wrapper.cuda_graph_warmup_steps = 1
    wrapper.use_single_mempool = True
    wrapper.disable_autograd_multithreading = False
    wrapper.capture_error_mode = 'thread_local'
    wrapper.data_read = mocker.Mock(return_value=[])

    result = wrapper(
        model=[], data_iterator=[], num_microbatches=1, seq_length=1, forward_only=False
    )

    assert result == 'captured-result'
    multithreading_context.assert_called_once_with(False)
    wrapper.forward_backward_func.assert_called_once_with(
        model=[], data_iterator=[], num_microbatches=1, seq_length=1, forward_only=False
    )
