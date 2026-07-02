# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Full iteration CUDA graph for training."""

import gc
import logging
from contextlib import nullcontext

import torch

from megatron.core.tensor_parallel.random import get_all_rng_states

logger = logging.getLogger(__name__)

# Process-wide handle so full-iter and optimizer graph captures share one pool and one
# non-default stream (per-stream alloc segments can inflate memory_reserved; see
# tools/debug_cuda_graph_pool_memory*.py).
_shared_graph_pool = None
_shared_capture_stream = None


def _walk_unique_modules(model):
    """Yield unique modules from a model or model list."""
    seen = set()
    stack = list(model) if isinstance(model, (list, tuple)) else [model]
    while stack:
        module = stack.pop()
        if module is None:
            continue
        module_id = id(module)
        if module_id in seen:
            continue
        seen.add(module_id)
        yield module
        modules = getattr(module, "modules", None)
        if callable(modules):
            stack.extend(child for child in modules() if id(child) not in seen)


def _has_v2_fsdp_modules(model):
    """Return whether the model contains a Megatron-FSDP v2 wrapper."""
    return any(
        getattr(getattr(module, "ddp_config", None), "use_megatron_fsdp_v2", False)
        for module in _walk_unique_modules(model)
    )


def _synchronize_fsdp_param_gathers(model):
    """Drain Megatron-FSDP parameter all-gathers before full-iteration capture."""
    synchronized = 0
    for module in _walk_unique_modules(model):
        synchronize_param_gather = getattr(module, "synchronize_param_gather", None)
        if callable(synchronize_param_gather):
            synchronize_param_gather()
            synchronized += 1
    return synchronized


def _stop_v2_fsdp_communication(model):
    """Join Megatron-FSDP v2 side streams before full-iteration capture exits."""
    stopped = 0
    for module in _walk_unique_modules(model):
        ddp_config = getattr(module, "ddp_config", None)
        if not getattr(ddp_config, "use_megatron_fsdp_v2", False):
            continue
        stop_communication = getattr(module, "stop_communication", None)
        if callable(stop_communication):
            stop_communication()
            stopped += 1
    return stopped


def _clear_v2_fsdp_uncaptured_unshard_events(model):
    """Drop pre-capture FSDP v2 unshard events after their streams are drained."""
    cleared = 0
    seen_contexts = set()
    for module in _walk_unique_modules(model):
        ddp_config = getattr(module, "ddp_config", None)
        fsdp_module = getattr(module, "module", None)
        if not getattr(ddp_config, "use_megatron_fsdp_v2", False):
            fsdp_module = module

        ctx = getattr(fsdp_module, "_fsdp_root_context", None)
        if ctx is None or id(ctx) in seen_contexts:
            continue
        seen_contexts.add(id(ctx))

        unshard_done_events = getattr(ctx, "unshard_done_events", None)
        if not unshard_done_events:
            continue
        for module_id, event in list(unshard_done_events.items()):
            if event is not None:
                unshard_done_events[module_id] = None
                cleared += 1
    return cleared


def _reset_fsdp_full_iteration_grad_buffers(model):
    """Restore FSDP grad buffers to the pre-capture zero-grad state before first replay."""
    reset_groups = 0
    for module in _walk_unique_modules(model):
        param_groups = getattr(module, "_fsdp_param_groups", None)
        if not param_groups:
            continue
        for param_group in param_groups:
            if getattr(param_group, "enable_full_iteration_cuda_graph", False):
                param_group.zero_grad(set_to_none=True)
                reset_groups += 1
    return reset_groups


def get_shared_capture_stream():
    """Return one `torch.cuda.Stream` for all full-iter and optimizer graph captures.

    Call after the target CUDA device is selected.
    """
    global _shared_capture_stream
    if _shared_capture_stream is None:
        _shared_capture_stream = torch.cuda.Stream()
    return _shared_capture_stream


def get_shared_graph_pool():
    """Return a process-wide handle so all call sites share one graph memory pool.

    `torch.cuda.graph_pool_handle()` returns a new pool each time; this lazy singleton
    ensures e.g. full-iteration and optimizer captures reuse the same pool.
    """
    global _shared_graph_pool
    if _shared_graph_pool is None:
        _shared_graph_pool = torch.cuda.graph_pool_handle()
    return _shared_graph_pool


def get_graph_pool(use_single_mempool):
    """Return graph pool handle for full-iter/optimizer graph capture.

    When `use_single_mempool` is True, train/eval and optimizer captures reuse one
    process-wide pool. Otherwise, each capture call gets a new pool handle.
    """
    if use_single_mempool:
        return get_shared_graph_pool()
    return torch.cuda.graph_pool_handle()


# The below functions traverse through nested data structures (tuples, lists, dicts)
# present in src and creates a deep copy where all PyTorch tensors are cloned,
# detached from the computation graph, and moved to CUDA device. Non-tensor objects
# are returned as-is.


def copy_tensors_in_struct(src):
    """Copy src to new tensors."""
    if isinstance(src, tuple):
        return tuple(copy_tensors_in_struct(i) for i in src)
    elif isinstance(src, list):
        return list(copy_tensors_in_struct(i) for i in src)
    elif isinstance(src, dict):
        return {k: copy_tensors_in_struct(src[k]) for k in src}
    elif isinstance(src, torch.Tensor):
        return src.clone().detach().cuda()
    else:
        return src


def clone_tensors_in_struct(tgt, src):
    """Copy src to pre-existing tensors in tgt."""
    if isinstance(src, tuple):
        raise Exception(f"Unsupported copy for tuple yet: {type(src)}")
    elif isinstance(src, list):
        for i in range(len(src)):
            if isinstance(src[i], (tuple, list, dict, torch.Tensor)):
                clone_tensors_in_struct(tgt[i], src[i])
            else:
                tgt[i] = src[i]
    elif isinstance(src, dict):
        for k in src:
            if isinstance(src[k], (tuple, list, dict, torch.Tensor)):
                clone_tensors_in_struct(tgt[k], src[k])
            else:
                tgt[k] = src[k]
    elif isinstance(src, torch.Tensor):
        tgt.copy_(src, non_blocking=True)
    else:
        raise Exception(f"Expect top-level as container type but got: {type(src)}")


# Class to copy dataloader output to static CUDA tensors for CUDA graph input. This
# maintains separate static buffers for training and validation CUDA graphs.
class StaticBufferLoader:
    """Load data to static buffers."""

    static_buffers: dict = {'training': [], 'validation': []}

    def __init__(self):
        self.stream = torch.cuda.Stream()

    def __call__(self, inputs, stage, microbatch):
        assert stage in ['training', 'validation']
        assert microbatch <= len(StaticBufferLoader.static_buffers[stage])
        if isinstance(inputs, tuple) and isinstance(inputs[0], dict):
            inputs = inputs[0]

        assert isinstance(inputs, dict)
        if microbatch == len(StaticBufferLoader.static_buffers[stage]):
            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream):
                StaticBufferLoader.static_buffers[stage].append(copy_tensors_in_struct(inputs))
        else:

            for k in inputs.keys():
                if k not in StaticBufferLoader.static_buffers[stage][microbatch]:
                    if isinstance(inputs[k], torch.Tensor):
                        StaticBufferLoader.static_buffers[stage][microbatch][k] = torch.empty_like(
                            inputs[k], device="cuda"
                        )
                    else:
                        StaticBufferLoader.static_buffers[stage][microbatch][k] = inputs[k]

            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream):
                clone_tensors_in_struct(
                    StaticBufferLoader.static_buffers[stage][microbatch], inputs
                )
        torch.cuda.current_stream().wait_stream(self.stream)
        return StaticBufferLoader.static_buffers[stage][microbatch]


class FullCudaGraphWrapper:
    """Wrapper class to enable FullIterationCUDAgraph."""

    curr_iteration = {'training': 0, 'validation': 0}
    cuda_graph = {'training': None, 'validation': None}
    result = {'training': None, 'validation': None}

    def __init__(self, forward_backward_func, cuda_graph_warmup_steps=1, use_single_mempool=False):
        self.forward_backward_func = forward_backward_func
        self.static_loader = StaticBufferLoader()
        self.cuda_graph_warmup_steps = cuda_graph_warmup_steps
        self.use_single_mempool = use_single_mempool

    def _forward_backward_on_capture_stream(self, *args, **kwargs):
        """Run eager warmup on the same stream that will later be captured."""
        capture_stream = get_shared_capture_stream()
        current_stream = torch.cuda.current_stream()
        capture_stream.wait_stream(current_stream)
        with torch.cuda.stream(capture_stream):
            result = self.forward_backward_func(*args, **kwargs)
        current_stream.wait_stream(capture_stream)
        return result

    def data_read(self, data_iterator, model, training, num_microbatches):
        """Read all microbatch inputs from Dataloader and copy to static buffers."""
        if not isinstance(model, list) or len(model) == 1:
            assert not isinstance(data_iterator, list) or len(data_iterator) == 1
            iterator0 = data_iterator if not isinstance(data_iterator, list) else data_iterator[0]
            data_list = []
            if iterator0 is not None:
                for b in range(num_microbatches):
                    data_list.append(
                        self.static_loader(
                            next(iterator0), 'training' if training else 'validation', b
                        )
                    )
                data_list = [iter(data_list)]
            else:
                data_list.append(None)
        else:
            assert isinstance(data_iterator, list) and len(data_iterator) == len(model)
            data_list = []
            for i in range(len(model)):
                if data_iterator[i] is not None:
                    data_list_i = []
                    for b in range(num_microbatches):
                        data_list_i.append(
                            self.static_loader(
                                next(data_iterator[i]), 'training' if training else 'validation', b
                            )
                        )
                    data_list.append(iter(data_list_i))
                else:
                    data_list.append(None)
        return data_list

    def __call__(self, *args, **kwargs):
        assert len(args) == 0, 'forward_backward_func does not accept positional args'
        assert all(
            [
                kwarg in kwargs
                for kwarg in [
                    'model',
                    'data_iterator',
                    'num_microbatches',
                    'seq_length',
                    'forward_only',
                ]
            ]
        )
        model = kwargs['model']
        num_microbatches = kwargs['num_microbatches']

        training = not kwargs['forward_only']
        training_str = 'training' if training else 'validation'
        use_v2_fsdp = _has_v2_fsdp_modules(model)
        if not training and use_v2_fsdp:
            # Validation warmup may leave v2 FSDP side-stream work that cannot be
            # joined from a new CUDA graph capture. Keep this path eager while
            # preserving the existing validation graph behavior for non-v2 models.
            return self.forward_backward_func(*args, **kwargs)
        curr_iteration = self.curr_iter(training_str)
        capture_iteration = curr_iteration == self.cuda_graph_warmup_steps
        data_iterator = kwargs['data_iterator']
        data_list = self.data_read(data_iterator, model, training, num_microbatches)
        kwargs['data_iterator'] = data_list

        if capture_iteration:
            _synchronize_fsdp_param_gathers(model)
            torch.cuda.synchronize()
            if use_v2_fsdp:
                _clear_v2_fsdp_uncaptured_unshard_events(model)
            logger.info(f'Capture CUDA graph for {training_str}!!!')
            torch.distributed.barrier()
            assert FullCudaGraphWrapper.cuda_graph[training_str] is None
            # Drop eager warmup outputs before capture. Replacing them from inside the
            # capture context can release tensors while CUDA stream capture is active.
            FullCudaGraphWrapper.result[training_str] = None
            gc.collect()
            torch.cuda.empty_cache()
            FullCudaGraphWrapper.cuda_graph[training_str] = torch.cuda.CUDAGraph()
            for _, state in get_all_rng_states().items():
                FullCudaGraphWrapper.cuda_graph[training_str].register_generator_state(state)
            torch.cuda.synchronize()
            capture_stream = get_shared_capture_stream()
            captured_result = None
            autograd_context = (
                torch.autograd.set_multithreading_enabled(False) if use_v2_fsdp else nullcontext()
            )
            capture_error_mode = "relaxed" if use_v2_fsdp else "thread_local"
            with autograd_context:
                with torch.cuda.graph(
                    FullCudaGraphWrapper.cuda_graph[training_str],
                    stream=capture_stream,
                    pool=get_graph_pool(self.use_single_mempool),
                    capture_error_mode=capture_error_mode,
                ):
                    captured_result = self.forward_backward_func(*args, **kwargs)
                    if use_v2_fsdp:
                        _stop_v2_fsdp_communication(model)
            FullCudaGraphWrapper.result[training_str] = captured_result
            torch.cuda.synchronize()
            _reset_fsdp_full_iteration_grad_buffers(model)
            torch.cuda.synchronize()
            torch.distributed.barrier()
            logger.info(f'CUDA graph capture done for {training_str}!!!')
        if FullCudaGraphWrapper.cuda_graph[training_str] is None:
            if use_v2_fsdp:
                result = self._forward_backward_on_capture_stream(*args, **kwargs)
            else:
                result = self.forward_backward_func(*args, **kwargs)
        else:
            FullCudaGraphWrapper.cuda_graph[training_str].replay()
            torch.cuda.current_stream().wait_stream(get_shared_capture_stream())
            result = FullCudaGraphWrapper.result[training_str]
        self.next_iter(training_str)
        return result

    def curr_iter(self, stage):
        """Return current training/validation iteration."""
        return FullCudaGraphWrapper.curr_iteration[stage]

    def next_iter(self, stage):
        """Increment current training/validation iteration."""
        FullCudaGraphWrapper.curr_iteration[stage] += 1

    def reset_cuda_graph(self, stage=None):
        """Reset CUDA graph."""
        if stage is None or stage == 'training':
            if FullCudaGraphWrapper.cuda_graph['training'] is not None:
                del FullCudaGraphWrapper.cuda_graph['training']
                FullCudaGraphWrapper.cuda_graph['training'] = None
            FullCudaGraphWrapper.result['training'] = None
            FullCudaGraphWrapper.curr_iteration['training'] = 0
        if stage is None or stage == 'validation':
            if FullCudaGraphWrapper.cuda_graph['validation'] is not None:
                del FullCudaGraphWrapper.cuda_graph['validation']
                FullCudaGraphWrapper.cuda_graph['validation'] = None
            FullCudaGraphWrapper.result['validation'] = None
            FullCudaGraphWrapper.curr_iteration['validation'] = 0
        gc.collect()
