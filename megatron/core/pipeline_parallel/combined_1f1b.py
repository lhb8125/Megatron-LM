from typing import Callable, Union, Any, Tuple, Iterator, List
import torch
from torch import Tensor

from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllPerBatchState
from megatron.core.transformer.module import Float16Module
from megatron.core.distributed import DistributedDataParallel



import contextlib
from torch.autograd.variable import Variable
from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.pipeline_parallel import p2p_communication
from megatron.core.transformer.cuda_graphs import create_cudagraphs
from megatron.core.transformer.moe.router import MoEAuxLossAutoScaler
from megatron.core.utils import (
    drain_embedding_wgrad_compute,
    get_attr_wrapped_model,
    get_model_config,
    get_model_type,
    get_model_xattn,
)

from .schedules import (
    clear_embedding_activation_buffer,
    check_first_val_step,
    set_current_microbatch,
    finish_embedding_wgrad_compute,
    deallocate_output_tensor
)
from megatron.core.utils import make_viewless_tensor

# Types
Shape = Union[List[int], torch.Size]


def make_viewless(e):
    e = make_viewless_tensor(inp=e, requires_grad=e.requires_grad, keep_graph=True)
    return e


class StreamRelease(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, event, stream, *inputs) -> Union[Tensor, Tuple[Tensor]]:
        ctx.event = event
        ctx.stream = stream
        ctx.event.record(stream)
        return inputs if len(inputs) > 1 else inputs[0]

    @staticmethod
    def backward(ctx: Any, *grad_outputs) -> Union[Tensor, Tuple[Tensor]]:
        event = ctx.event
        stream = ctx.stream
        event.wait(stream)
        return (None, None) + grad_outputs


class StreamAcquire(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, event, stream, *inputs: Tensor) -> Union[Tensor, Tuple[Tensor]]:
        ctx.event = event
        ctx.stream = stream
        ctx.event.wait(stream)
        # multiple in, multiple out
        return inputs if len(inputs) > 1 else inputs[0]

    @staticmethod
    def backward(ctx: Any, *grad_outputs) -> Union[Tensor, Tuple[Tensor]]:
        event = ctx.event
        stream = ctx.stream
        event.record(stream)
        return (None, None) + grad_outputs


class ScheduleNode:

    def __init__(self, forward_func, stream, event, name="schedule_node"):
        self.name = name
        self.forward_func = forward_func
        self.stream = stream
        self.event = event
        self.inputs = None
        self.outputs = None

    def forward(self, inputs=()):

        if not isinstance(inputs, tuple):
            inputs = (inputs,)
        return self._forward(*inputs)

    def _forward(self, *inputs):
        torch.cuda.nvtx.range_push(f"{self.name} forward")
        self.inputs = [make_viewless(e.detach()) if e is not None else None for e in inputs]
        for i, input in enumerate(self.inputs):
            if input is not None:
                input.requires_grad = inputs[i].requires_grad
        if len(inputs):
            data = StreamAcquire.apply(self.event, self.stream, *self.inputs)
        else:
            # empty input, alse need wait stream
            data = inputs
            self.event.wait(self.stream)
        # pack args to tuple
        if not isinstance(data, tuple):
            data = (data,)

        with torch.cuda.stream(self.stream):
            data = self.forward_func(*data)

        # pack args to tuple
        if not isinstance(data, tuple):
            data = (data,)

        data = StreamRelease.apply(self.event, self.stream, *data)
        
        if not isinstance(data, tuple):
            data = make_viewless(data)
        else:    
            data = tuple([make_viewless(e) if isinstance(e, Tensor) else e for e in data])
        
        self.output = data
        torch.cuda.nvtx.range_pop()
        return self.output

    def get_output(self):
        return self.output

    def backward(self, output_grad):
        if not isinstance(output_grad, tuple):
            output_grad = (output_grad,)
        return self._backward(*output_grad)

    def _backward(self, *output_grad):

        torch.cuda.nvtx.range_push(f"{self.name} backward")
        """
        # not multiple input
        if len(output_grad) == 1:
            output_grad = output_grad[0]
        """
        outputs = self.output
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        assert len(outputs) == len(output_grad), f"{len(outputs)} of {type(outputs[0])} vs {len(output_grad)} of {type(output_grad[0])}"
        with torch.cuda.stream(self.stream):
            #torch.autograd.backward(self.output, grad_tensors=output_grad)
            Variable._execution_engine.run_backward(
                tensors=outputs,
                grad_tensors=output_grad,
                keep_graph=False,
                create_graph=False,
                inputs=tuple(),
                allow_unreachable=True,
                accumulate_grad=True,
            )
        if not len(self.inputs):
            # empty input, alse need record stream
            self.event.record(self.stream)

        torch.cuda.nvtx.range_pop()
        return self.get_grad()

    def get_grad(self):
        grad = tuple([e.grad if e is not None else None for e in self.inputs])
        # multiple in, multiple out
        if len(grad) == 1:
            grad = grad[0]
        return grad


class TransformerLayerState(MoEAlltoAllPerBatchState):
    pass


class ModelChunkSate:
    pass


class TransformerLayerSchedulePlan:

    def __init__(self, attn, dispatch, mlp, combine, post_combine):
        self.attn = attn
        self.dispatch = dispatch
        self.mlp = mlp
        self.combine = combine
        self.post_combine = post_combine


class ModelChunkSchedulePlan:
    def __init__(self):
        self._pre_process = None
        self._post_process = None
        self._model_chunk_state = ModelChunkSate()
        self._transformer_layers = []
        self._event = torch.cuda.Event()

    @property
    def event(self):
        return self._event

    @property
    def pre_process(self):
        return self._pre_process

    @pre_process.setter
    def pre_process(self, value):
        self._pre_process = value

    @property
    def post_process(self):
        return self._post_process

    @post_process.setter
    def post_process(self, value):
        self._post_process = value

    def get_layer(self, i):
        assert i < self.num_layers()
        return self._transformer_layers[i]

    def num_layers(self):
        return len(self._transformer_layers)

    def add_layer(self, layer):
        self._transformer_layers.append(layer)

    @property
    def state(self):
        return self._model_chunk_state


def schedule_layer_1f1b(
    f_layer, b_layer, f_input=None, b_grad=None, pre_forward=None, pre_backward=None, f_context=None, b_context=None,
):
    f_context = f_context if f_context is not None else contextlib.nullcontext()
    b_context = b_context if b_context is not None else contextlib.nullcontext()

    if pre_backward is not None:
        # attn backward from last iter
        assert b_grad is None
        b_grad = pre_backward()

    if b_layer is not None:
        with b_context:
            b_grad = b_layer.post_combine.backward(b_grad)
            b_grad = b_layer.combine.backward(b_grad)

    if pre_forward is not None:
        assert f_input is None
        # post combine from last iter
        f_input = pre_forward()
        del pre_forward

    if f_layer is not None:
        with f_context:
            f_input = f_layer.attn.forward(f_input)
            f_input = f_layer.dispatch.forward(f_input)

    if b_layer is not None:
        with b_context:
            b_grad = b_layer.mlp.backward(b_grad)
            b_grad = b_layer.dispatch.backward(b_grad)

    if f_layer is not None:
        with f_context:
            f_input = f_layer.mlp.forward(f_input)
            f_input = f_layer.combine.forward(f_input)

    def next_iter_pre_forward():
        if f_layer is not None:
            with f_context:
                output = f_layer.post_combine.forward(f_input)
                return output

    def next_iter_pre_backward():
        if b_layer is not None:
            with b_context:
                grad = b_layer.attn.backward(b_grad)
                return grad

    if f_layer and b_layer:
        return next_iter_pre_forward, next_iter_pre_backward
    else:
        return next_iter_pre_forward(), next_iter_pre_backward()

def schedule_chunk_1f1b(f_schedule_plan, b_schedule_plan, grad=None, f_context=None, b_context=None):
    f_context = f_context if f_context is not None else contextlib.nullcontext()
    b_context = b_context if b_context is not None else contextlib.nullcontext()

    f_input = None
    if f_schedule_plan is not None:
        with f_context:
            f_input = f_schedule_plan.pre_process.forward()

    if b_schedule_plan is not None:
        assert grad is not None
        if b_schedule_plan.post_process is not None:
            with b_context:
                grad = b_schedule_plan.post_process.backward(grad)

    f_num_layers = f_schedule_plan.num_layers() if f_schedule_plan is not None else 0
    b_num_layers = b_schedule_plan.num_layers() if b_schedule_plan is not None else 0
    overlaped_layers = min(f_num_layers, b_num_layers)

    pre_forward = lambda: f_input
    pre_backward = lambda: grad

    for i in range(overlaped_layers):
        f_layer = f_schedule_plan.get_layer(i)
        b_layer = b_schedule_plan.get_layer(b_num_layers - 1 - i)
        torch.cuda.nvtx.range_push(f"layer_{i}f-layer_{b_num_layers - 1 - i}b")
        pre_forward, pre_backward = schedule_layer_1f1b(
            f_layer, b_layer, pre_forward=pre_forward, pre_backward=pre_backward, f_context=f_context, b_context=b_context
        )
        torch.cuda.nvtx.range_pop()

    # tail forward
    f_input = pre_forward()
    with f_context:
        for i in range(overlaped_layers, f_num_layers):
            f_layer = f_schedule_plan.get_layer(i)
            torch.cuda.nvtx.range_push(f"layer_{i}f")
            f_input, tmp = schedule_layer_1f1b(f_layer, None, f_input=f_input)
            torch.cuda.nvtx.range_pop()

        if f_schedule_plan is not None and f_schedule_plan.post_process is not None:
            f_input = f_schedule_plan.post_process.forward(f_input)

    # tail backward
    grad = pre_backward()
    with b_context:
        for i in range(overlaped_layers, b_num_layers):
            b_layer = b_schedule_plan.get_layer(b_num_layers - 1 - i)
            torch.cuda.nvtx.range_push(f"layer_{b_num_layers - 1 - i}b")
            tmp, grad = schedule_layer_1f1b(None, b_layer, b_grad=grad)
            torch.cuda.nvtx.range_pop()

        if b_schedule_plan is not None:
            b_schedule_plan.pre_process.backward(grad)

    return f_input

def schedule_chunk_forward(schedule_plan):
    f_input = schedule_chunk_1f1b(schedule_plan, None, None)
    return f_input


def schedule_chunk_backward(schedule_plan, grad):
    tmp = schedule_chunk_1f1b(None, schedule_plan, grad)

_COMP_STREAM = None
_COM_STREAM = None

def set_streams(comp_stream=None, com_stream=None):
    if comp_stream is None:
        comp_stream = torch.cuda.current_stream()
    if com_stream is None:
        com_stream = torch.cuda.Stream(device="cuda")

    global _COMP_STREAM
    global _COM_STREAM
    assert _COMP_STREAM is None
    assert _COM_STREAM is None
    _COMP_STREAM = comp_stream
    _COM_STREAM = com_stream


def get_comp_stream():
    global _COMP_STREAM
    return _COMP_STREAM

def get_com_stream():
    global _COM_STREAM
    return _COM_STREAM


class VppContextManager():
    
    def __init__(self, vpp_rank):
        self.vpp_rank = vpp_rank
        
    def __enter__(self):
        self.origin_vpp_rank = parallel_state.get_virtual_pipeline_model_parallel_rank()
        parallel_state.set_virtual_pipeline_model_parallel_rank(self.vpp_rank)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        parallel_state.set_virtual_pipeline_model_parallel_rank(self.origin_vpp_rank)



def forward_backward_step(
        forward_step_func,
        data_iterator,
        f_model,
        num_microbatches,
        input_tensor,
        forward_data_store,
        b_model,
        b_input_tensor,
        b_output_tensor,
        b_output_tensor_grad,
        config,
        f_context=None,
        b_context=None,
        collect_non_loss_data=False,
        checkpoint_activations_microbatch=None,
        is_first_microbatch=False,
        current_microbatch=None,
        encoder_decoder_xattn=False,
):
    """Forward step for passed-in model.

    If it is the first stage, the input tensor is obtained from the data_iterator.
    Otherwise, the passed-in input_tensor is used.

    Args:
        forward_step_func (callable):
            The forward step function for the model that takes the
            data iterator as the first argument, and model as the second.
            This user's forward step is expected to output a tuple of two elements:

                1. The output object from the forward step. This output object needs to be a
                    tensor or some kind of collection of tensors. The only hard requirement
                    for this object is that it needs to be acceptible as input into the second
                    function.
                2. A function to reduce (optionally) the output from the forward step. This
                    could be a reduction over the loss from the model, it could be a function that
                    grabs the output from the model and reformats, it could be a function that just
                    passes through the model output. This function must have one of the following
                    patterns, and depending on the pattern different things happen internally:

                        a. A tuple of reduced loss and some other data. Note that in this case
                            the first argument is divided by the number of global microbatches,
                            assuming it is a loss, so that the loss is stable as a function of
                            the number of devices the step is split across.
                        b. A triple of reduced loss, number of tokens, and some other data. This
                            is similar to case (a), but the loss is further averaged across the
                            number of tokens in the batch. If the user is not already averaging
                            across the number of tokens, this pattern is useful to use.
                        c. Any arbitrary data the user wants (eg a dictionary of tensors, a list
                            of tensors, etc in the case of inference). To trigger case 3 you need
                            to specify `collect_non_loss_data=True` and you may also want to
                            specify `forward_only=True` in the call to the parent forward_backward
                            function.
        data_iterator (iterator):
            The data iterator.
        model (nn.Module):
            The model to perform the forward step on.
        num_microbatches (int):
            The number of microbatches.
        input_tensor (Tensor or list[Tensor]):
            The input tensor(s) for the forward step.
        forward_data_store (list):
            The list to store the forward data. If you go down path 2.a or
            2.b for the return of your forward reduction function then this will store only the
            final dimension of the output, for example the metadata output by the loss function.
            If you go down the path of 2.c then this will store the entire output of the forward
            reduction function applied to the model output.
        config (object):
            The configuration object.
        collect_non_loss_data (bool, optional):
            Whether to collect non-loss data. Defaults to False.
            This is the path to use if you want to collect arbitrary output from the model forward,
            such as with inference use cases. Defaults to False.
        checkpoint_activations_microbatch (int, optional):
            The microbatch to checkpoint activations.
            Defaults to None.
        is_first_microbatch (bool, optional):
            Whether it is the first microbatch. Defaults to False.
        current_microbatch (int, optional):
            The current microbatch. Defaults to None.

    Returns:
        Tensor or list[Tensor]: The output object(s) from the forward step.
        Tensor: The number of tokens.
    """
    if config.timers is not None:
        config.timers('forward-compute', log_level=2).start()

    if config.enable_autocast:
        context_manager = torch.autocast("cuda", dtype=config.autocast_dtype)
    else:
        context_manager = contextlib.nullcontext()

    # forward preprocess
    if f_model is not None:
        with f_context:
            if is_first_microbatch and hasattr(f_model, 'set_is_first_microbatch'):
                f_model.set_is_first_microbatch()
            if current_microbatch is not None:
                set_current_microbatch(f_model, current_microbatch)

            unwrap_output_tensor = False
            if not isinstance(input_tensor, list):
                input_tensor = [input_tensor]
                unwrap_output_tensor = True

            set_input_tensor = get_attr_wrapped_model(f_model, "set_input_tensor")
            set_input_tensor(input_tensor)

            with context_manager:
                if checkpoint_activations_microbatch is None:
                    output_tensor, loss_func = forward_step_func(data_iterator, f_model)
                else:
                    output_tensor, loss_func = forward_step_func(
                    data_iterator, f_model, checkpoint_activations_microbatch
                )

    # backward preprocess
    unwrap_input_tensor_grad = False
    b_schedule_plan = None
    if b_model is not None:
        # Retain the grad on the input_tensor.
        if not isinstance(b_input_tensor, list):
            b_input_tensor = [b_input_tensor]
            unwrap_input_tensor_grad = True
        for x in b_input_tensor:
            if x is not None:
                x.retain_grad()

        if not isinstance(b_output_tensor, list):
            b_output_tensor = [b_output_tensor]
        if not isinstance(b_output_tensor_grad, list):
            b_output_tensor_grad = [b_output_tensor_grad]

        # Backward pass for loss function
        b_schedule_plan = b_output_tensor[0].schedule_plan
        b_output_tensor[0].schedule_plan = None
        if b_output_tensor_grad[0] is None and config.grad_scale_func is not None:
            #backward schedule plan
            loss_node = b_output_tensor[0].loss_func
            b_output_tensor[0].loss_func = None
            b_output_tensor[0] = config.grad_scale_func(b_output_tensor[0])
            torch.autograd.backward(b_output_tensor[0], grad_tensors=b_output_tensor_grad[0])
            b_output_tensor_grad[0] = loss_node.get_grad()

    f_schedule_plan = output_tensor if f_model else None
    grad = b_output_tensor_grad[0] if b_model else None
    # schedule forward and backward
    output_tensor = schedule_chunk_1f1b(f_schedule_plan, b_schedule_plan, grad, f_context=f_context, b_context=b_context)
    # forward post process
    num_tokens = None
    if f_model:
        with f_context:
            num_tokens = torch.tensor(0, dtype=torch.int)
            if parallel_state.is_pipeline_last_stage():
                if not collect_non_loss_data:
                    loss_node = ScheduleNode(loss_func, torch.cuda.current_stream(), f_schedule_plan.event, "loss_func")
                    loss_func = loss_node.forward
                    outputs = loss_func(output_tensor)
                    if len(outputs) == 3:
                        output_tensor, num_tokens, loss_reduced = outputs
                        if not config.calculate_per_token_loss:
                            output_tensor /= num_tokens
                            output_tensor /= num_microbatches
                    else:
                        # preserve legacy loss averaging behavior (ie, over the number of microbatches)
                        assert len(outputs) == 2
                        output_tensor, loss_reduced = outputs
                        output_tensor = output_tensor / num_microbatches
                    forward_data_store.append(loss_reduced)
                    
                    # attach loss_func on output_tensor
                    output_tensor.loss_func = loss_node
                else:
                    data = loss_func(output_tensor, non_loss_data=True)
                    forward_data_store.append(data)
            # attach schedule plan on output tensor
            output_tensor.schedule_plan = f_schedule_plan
            if config.timers is not None:
                config.timers('forward-compute').stop()

            # Set the loss scale for the auxiliary loss of the MoE layer.
            # Since we use a trick to do backward on the auxiliary loss, we need to set the scale
            # explicitly.
            if hasattr(config, 'num_moe_experts') and config.num_moe_experts is not None:
                # Calculate the loss scale based on the grad_scale_func if available, else default to 1.
                loss_scale = (
                    config.grad_scale_func(torch.ones(1, device=output_tensor.device))
                    if config.grad_scale_func is not None
                    else torch.tensor(1.0)
                )
                # Set the loss scale
                MoEAuxLossAutoScaler.set_loss_scale(loss_scale / num_microbatches)


            if not unwrap_output_tensor:
                output_tensor, num_tokens = [output_tensor], num_tokens
    # backward post process            
    input_tensor_grad = None
    if b_model is not None:
        input_tensor_grad = [None]
        if b_input_tensor is not None:
            input_tensor_grad = []
            for x in b_input_tensor:
                if x is None:
                    input_tensor_grad.append(None)
                else:
                    input_tensor_grad.append(x.grad)


        if unwrap_input_tensor_grad:
            input_tensor_grad = input_tensor_grad[0]

    return output_tensor, num_tokens, input_tensor_grad


def unwrap_model(model, module_instances=(DistributedDataParallel, Float16Module)):
    return_list = True
    if not isinstance(model, list):
        model = [model]
        return_list = False
    unwrapped_model = []
    for model_module in model:
        while isinstance(model_module, module_instances):
            model_module = model_module.module
        unwrapped_model.append(model_module)
    if not return_list:
        return unwrapped_model[0]
    return unwrapped_model

def wrap_forward_func(forward_step_func):
    
    def wrapped_func(data_iterator, model):
        return forward_step_func(data_iterator, unwrap_model(model).build_schedule_plan)

    return wrapped_func

def forward_backward_pipelining_with_interleaving(
        *,
        forward_step_func,
        data_iterator: Union[Iterator, List[Iterator]],
        model: Union[torch.nn.Module, List[torch.nn.Module]],
        num_microbatches: int,
        seq_length: int,
        micro_batch_size: int,
        decoder_seq_length: int = None,
        forward_only: bool = False,
        collect_non_loss_data: bool = False,
        first_val_step: bool = None,
):
    """Run interleaved 1F1B schedule (model split into model chunks), with
    communication between pipeline stages as needed.

    Returns dictionary with losses if the last stage, empty dict otherwise."""

    # Convention used in this function:
    # num_microbatches for number of microbatches per pipeline stage;
    # num_model_chunks for virtual pipeline size;
    # then total_num_microbatches = num_microbatches * num_model_chunks.
    # Their corresponding index variables are
    # microbatch_id in [0, num_microbatches)
    # model_chunk_id in [0, num_model_chunks)
    # virtual_microbatch_id in [0, total_num_microbatches)

    # decorate forward step
    forward_step_func = wrap_forward_func(forward_step_func)

    assert isinstance(model, list), "interleaved pipeline parallelism expected model chunking"
    assert all(isinstance(chunk, torch.nn.Module) for chunk in model), "invalid model chunking"
    assert isinstance(
        data_iterator, list
    ), "interleaved pipeline parallelism expected each model chunk to have a data iterator"

    config = get_model_config(model[0])
    if config.overlap_p2p_comm and config.batch_p2p_comm:
        raise ValueError("Can not use both overlap_p2p_comm and batch_p2p_comm")

    # Needed only when gradients are finalized in M-Core
    if config.finalize_model_grads_func is not None and not forward_only:
        embedding_module = clear_embedding_activation_buffer(config, model)

    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    # Disable async grad reductions
    no_sync_func = config.no_sync_func
    if isinstance(no_sync_func, list):

        def multi_no_sync():
            stack = contextlib.ExitStack()
            for model_chunk_no_sync_func in config.no_sync_func:
                stack.enter_context(model_chunk_no_sync_func())
            return stack

        no_sync_func = multi_no_sync
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    if config.grad_sync_func is not None and not isinstance(config.grad_sync_func, list):
        config.grad_sync_func = [config.grad_sync_func for _ in model]

    if config.param_sync_func is not None and not isinstance(config.param_sync_func, list):
        config.param_sync_func = [config.param_sync_func for _ in model]

    # Disable config.grad_sync_func and config.param_sync_func if only running forward passes.
    # They will be re-enabled at the end of this function.
    grad_sync_func, param_sync_func = None, None
    if forward_only:
        grad_sync_func, param_sync_func = config.grad_sync_func, config.param_sync_func
        config.grad_sync_func, config.param_sync_func = None, None

    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    disable_grad_sync()

    # Model chunk IDs with synchronized grads
    synchronized_model_chunks = set()

    input_tensors = [[] for _ in range(len(model))]
    output_tensors = [[] for _ in range(len(model))]
    total_num_tokens = torch.tensor(0, dtype=torch.int).cuda()

    forward_data_store = []
    if not forward_only:
        output_tensor_grads = [[] for _ in range(len(model))]

    pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    pipeline_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()

    if (
            config.microbatch_group_size_per_vp_stage > num_microbatches
            or config.microbatch_group_size_per_vp_stage < pipeline_parallel_size
    ):
        msg = (
            'The number of contiguous micro-batches in a virtual pipeline stage'
            f'should range in [PP={pipeline_parallel_size} , M={num_microbatches}]'
        )
        raise ValueError(msg)

    # If the final micro-batch group has fewer micro-batches than pipeline-parallel size,
    # the pipeline will have dependency bubbles.
    final_microbatch_group_size = num_microbatches % config.microbatch_group_size_per_vp_stage
    if 0 < final_microbatch_group_size < pipeline_parallel_size:
        msg = 'The remainder of M (the total micro-batches) divided by N (number of '
        msg += 'contiguous micro-batches in a virtual pipeline stage) should be 0, '
        msg += 'or larger than or equal to the pipeline-parallel size, but it is '
        msg += f'{final_microbatch_group_size}. '
        msg += 'Otherwise, it introduces dependency bubbles in the pipeline '
        msg += 'and reduces throughput.'
        raise RuntimeError(msg)

    model_type = get_model_type(model[0])
    if model_type == ModelType.encoder_and_decoder:
        raise RuntimeError("Interleaving is not supported with an encoder and decoder model.")

    if decoder_seq_length is not None and decoder_seq_length != seq_length:
        raise RuntimeError(
            "Interleaving is not supported with a different decoder sequence length."
        )

    tensor_shape = [seq_length, micro_batch_size, config.hidden_size]
    tensor_shape[0] = tensor_shape[0] // parallel_state.get_context_parallel_world_size()
    if config.sequence_parallel:
        tensor_shape[0] = tensor_shape[0] // parallel_state.get_tensor_model_parallel_world_size()

    # Compute number of warmup and remaining microbatches.
    num_model_chunks = len(model)
    total_num_microbatches = num_microbatches * num_model_chunks
    all_warmup_microbatches = False
    if forward_only:
        num_warmup_microbatches = total_num_microbatches
    else:
        # Run (num_model_chunks-1)*config.microbatch_group_size_per_vp_stage on
        # all workers, followed by more microbatches after depending on
        # stage ID (more forward passes for earlier stages, later stages can
        # immediately start with 1F1B).
        num_warmup_microbatches = (pipeline_parallel_size - pipeline_parallel_rank - 1) * 2
        num_warmup_microbatches += (
                                           num_model_chunks - 1
                                   ) * config.microbatch_group_size_per_vp_stage
        num_warmup_microbatches = num_warmup_microbatches + 1
        if num_warmup_microbatches >= total_num_microbatches:
            num_warmup_microbatches = total_num_microbatches
            all_warmup_microbatches = True
    num_microbatches_remaining = total_num_microbatches - num_warmup_microbatches
    # Checkpoint the activations of partial Transformer layers in a number of micro-batches
    # within the maximum outstanding micro-batch backpropagations.
    # Micro-batches with the ids less than 'num_microbatches_with_partial_activation_checkpoints'
    # checkpoint partial Transformer layers (or skip checkpointing) and
    # the rest of micro-batches within a window of micro-batches checkpoint
    # all Transformer layers. The window of micro-batches is set by the maximum
    # outstanding backpropagations and becomes smaller at later pipeline stages.
    # Please refer the appendix C in https://arxiv.org/pdf/2205.05198.pdf
    max_outstanding_backprops = None
    if config.num_microbatches_with_partial_activation_checkpoints is not None:
        max_outstanding_backprops = num_warmup_microbatches + 1

    # Synchronize params for first two model chunks
    if config.param_sync_func is not None:
        config.param_sync_func[0](model[0].parameters())
        config.param_sync_func[1](model[1].parameters())

    # Create a tunable schedule lookup table.
    # The schedule lookup table uses the virtual_microbatch_id to find the corresponding
    # microbatch_id and model_chunk_id. For example, the tunable schedule table for
    # PP2 N3M5 with VP2 is constructed as below:
    # virtual_microbatch_id | 0 1 2 3 4 5 6 7 8 9
    # microbatch_id         | 0 1 2 0 1 2 3 4 3 4
    # model_chunk_id        | 0 0 0 1 1 1 0 0 1 1
    schedule_table = []
    for min_microbatch_id_in_group in range(
            0, num_microbatches, config.microbatch_group_size_per_vp_stage
    ):
        if (
                min_microbatch_id_in_group + config.microbatch_group_size_per_vp_stage
                >= num_microbatches
        ):
            # Construct schedule for the last microbatch group
            schedule_table.extend(
                [
                    (microbatch_id, model_chunk_id)
                    for model_chunk_id in range(len(model))
                    for microbatch_id in range(min_microbatch_id_in_group, num_microbatches)
                ]
            )
        else:
            # Construct schedule for other microbatch groups
            schedule_table.extend(
                [
                    (microbatch_id, model_chunk_id)
                    for model_chunk_id in range(len(model))
                    for microbatch_id in range(
                    min_microbatch_id_in_group,
                    min_microbatch_id_in_group + config.microbatch_group_size_per_vp_stage,
                    )
                ]
            )

    # Decouple individual lookup table for microbatch_id and model_chunk_id.
    # For example, the micro-batch table for PP2 N3M5 with VP2 is
    # virtual_microbatch_id | 0 1 2 3 4 5 6 7 8 9
    # microbatch_id         | 0 1 2 0 1 2 3 4 3 4
    # Similarly, the model chunk table is
    # virtual_microbatch_id | 0 1 2 3 4 5 6 7 8 9
    # model_chunk_id        | 0 0 0 1 1 1 0 0 1 1
    # Both tables are indexed with virtual_microbatch_id.
    microbatch_id_table, model_chunk_id_table = zip(*schedule_table)

    def get_model_chunk_id(virtual_microbatch_id, forward):
        """Helper method to get the model chunk ID given the iteration number."""
        model_chunk_id = model_chunk_id_table[virtual_microbatch_id % total_num_microbatches]
        if not forward:
            model_chunk_id = num_model_chunks - model_chunk_id - 1
        return model_chunk_id

    def get_microbatch_id_in_model_chunk(iteration_id, forward):
        """Helper method to get the microbatch_id within model chunk given the iteration number."""
        assert forward
        microbatch_id_in_model_chunk = microbatch_id_table[iteration_id]
        return microbatch_id_in_model_chunk

    def num_released_microbatches(virtual_microbatch_id, model_chunk_id):
        """Helper method to count number of released (i.e. popped from input_tensors)
        microbatches for a model chunk."""
        if forward_only:  # Micro-batch is released after forward prop.
            return model_chunk_id_table[:virtual_microbatch_id].count(model_chunk_id)
        else:  # Micro-batch is released after backward prop.
            # Zero backward prop in warmup.
            if virtual_microbatch_id < num_warmup_microbatches:
                return 0
            else:
                backward_microbatch_id = virtual_microbatch_id - num_warmup_microbatches
                model_chunk_id = num_model_chunks - model_chunk_id - 1
                return model_chunk_id_table[:backward_microbatch_id].count(model_chunk_id)

    def is_first_microbatch_for_model_chunk(virtual_microbatch_id: int) -> bool:
        """Check if an iteration is the first for a model chunk."""
        if virtual_microbatch_id < total_num_microbatches:
            return microbatch_id_table[virtual_microbatch_id] == 0
        else:
            return False

    def is_last_microbatch_for_model_chunk(virtual_microbatch_id: int) -> bool:
        """Check if an iteration is the last for a model chunk."""
        if virtual_microbatch_id < total_num_microbatches:
            return microbatch_id_table[virtual_microbatch_id] == num_microbatches - 1
        else:
            return False

    def recv_tensor_from_previous_stage(virtual_microbatch_id, forward):
        """Determine if peers are sending, and where in data structure
        to put received tensors.
        Return a boolean if the pipeline stage expects to recv from peers, and the
        corresponding model_chunk_id for the received tensor.
        """
        recv = True
        # The leading pipeline stage is the first rank in fwd and the last rank in bwd.
        is_leading_pipeline_stage = (
            parallel_state.is_pipeline_first_stage(ignore_virtual=True)
            if forward
            else parallel_state.is_pipeline_last_stage(ignore_virtual=True)
        )

        last_model_chunk = (num_model_chunks - 1) if forward else 0

        if is_leading_pipeline_stage:
            # The leading pipeline stage is ahead of the ending pipeline stage
            # (i.e. last rank in fwd and first rank in bwd) by (pipeline_parallel_size - 1).
            # Let's consider bwd as an example with PP 4:
            #       0 1 2 3 ...
            #     0 1 2 3 ...
            #   0 1 2 3 ...
            # 0 1 2 3 ...
            if virtual_microbatch_id < (pipeline_parallel_size - 1):
                # The ending stage has not produced any tensors, so no recv will be initiated.
                recv = False
                next_model_chunk_id = get_model_chunk_id(virtual_microbatch_id + 1, forward)
            else:
                # Find the model chunk of the aligned microbatches in the ending stage.
                # For example, microbatch 0 in the ending stage is aligned with microbatch 3
                # in the leading stage.
                next_model_chunk_id = get_model_chunk_id(
                    virtual_microbatch_id - (pipeline_parallel_size - 1), forward
                )
            # Last model chunk in the final stage does not produce tensors.
            if next_model_chunk_id == last_model_chunk:
                recv = False
            if forward:
                # Model chunk id increases in forward.
                next_model_chunk_id += 1
            else:
                # Model chunk id decreases in backward.
                next_model_chunk_id -= 1
        else:
            next_model_chunk_id = get_model_chunk_id(virtual_microbatch_id + 1, forward)

        return recv, next_model_chunk_id


    def forward_backward_helper(f_virtual_microbatch_id=None, f_microbatch_id=None, b_virtual_microbatch_id=None):
        """Helper method to run forward step with model split into chunks
        (run set_virtual_pipeline_model_parallel_rank() before calling
        forward_step())."""
        # forward prepare
        f_model_chunk_id = None
        f_context = contextlib.nullcontext()
        input_tensor = None
        if f_virtual_microbatch_id is not None:
            model_chunk_id = get_model_chunk_id(f_virtual_microbatch_id, forward=True)
            f_model_chunk_id = model_chunk_id
            f_context = VppContextManager(f_model_chunk_id)
            with f_context:
                # launch param synchronization for next model chunk
                # Note: Asynchronous communication tends to slow down compute.
                # To reduce idling from mismatched microbatch times, we launch
                # asynchronous communication at the same time across the
                # pipeline-parallel group.
                if config.param_sync_func is not None:
                    param_sync_virtual_microbatch_id = f_virtual_microbatch_id + pipeline_parallel_rank
                    if (
                            param_sync_virtual_microbatch_id < total_num_microbatches
                            and is_first_microbatch_for_model_chunk(param_sync_virtual_microbatch_id)
                    ):
                        param_sync_chunk_id = (
                                get_model_chunk_id(param_sync_virtual_microbatch_id, forward=True) + 1
                        )
                        if 1 < param_sync_chunk_id < num_model_chunks:
                            config.param_sync_func[param_sync_chunk_id](
                                model[param_sync_chunk_id].parameters()
                            )

                # forward step
                if parallel_state.is_pipeline_first_stage():
                    if len(input_tensors[model_chunk_id]) == len(output_tensors[model_chunk_id]):
                        input_tensors[model_chunk_id].append(None)

                # For non-depth-first pipeline schedules, the first rank would buffer multiple received
                # activation tensors for a model chunk until accessed during warmup.
                # This input buffering is needed to overlap the computation with the receipt of
                # the next inputs. To index the proper buffered inputs for forword_step, we use
                # microbatch_id offset with number of released microbatches that have completed backprop.
                offset = num_released_microbatches(f_virtual_microbatch_id, model_chunk_id)
                input_tensor = input_tensors[model_chunk_id][f_microbatch_id - offset]


        # backward prepare
        b_model_chunk_id = None
        b_context = contextlib.nullcontext()
        b_input_tensor = None
        b_output_tensor =  None
        b_output_tensor_grad = None
        if b_virtual_microbatch_id is not None:
            model_chunk_id = get_model_chunk_id(b_virtual_microbatch_id, forward=False)
            b_model_chunk_id = model_chunk_id
            b_context = VppContextManager(b_model_chunk_id)
            with b_context:
                # launch grad synchronization (default)
                if config.grad_sync_func is None and is_last_microbatch_for_model_chunk(
                        b_virtual_microbatch_id
                ):
                    enable_grad_sync()
                    synchronized_model_chunks.add(model_chunk_id)

                if parallel_state.is_pipeline_last_stage():
                    if len(output_tensor_grads[model_chunk_id]) == 0:
                        output_tensor_grads[model_chunk_id].append(None)
                b_input_tensor = input_tensors[model_chunk_id].pop(0)
                b_output_tensor = output_tensors[model_chunk_id].pop(0)
                b_output_tensor_grad = output_tensor_grads[model_chunk_id].pop(0)



        output_tensor, num_tokens, input_tensor_grad = forward_backward_step(
            forward_step_func,
            data_iterator[f_model_chunk_id] if f_model_chunk_id is not None else None,
            model[f_model_chunk_id] if f_model_chunk_id is not  None else  None,
            num_microbatches,
            input_tensor,
            forward_data_store,
            model[b_model_chunk_id] if b_model_chunk_id is not  None else  None,
            b_input_tensor,
            b_output_tensor,
            b_output_tensor_grad,
            config,
            f_context=f_context,
            b_context=b_context,
            collect_non_loss_data=collect_non_loss_data,
            checkpoint_activations_microbatch= None,
            is_first_microbatch = check_first_val_step(
                first_val_step,
                forward_only,
                is_first_microbatch_for_model_chunk(f_virtual_microbatch_id) if f_virtual_microbatch_id is not None else None,
            ),
            current_microbatch=f_microbatch_id,
        )

        # forward post process
        if f_model_chunk_id is not None:
            with f_context:
                output_tensors[f_model_chunk_id].append(output_tensor)
                nonlocal total_num_tokens
                total_num_tokens += num_tokens.item()
                # If forward-only, no need to save tensors for a backward pass.
                if forward_only:
                    # Release the tensor that have completed forward step.
                    input_tensors[f_model_chunk_id].pop(0)
                    output_tensors[f_model_chunk_id].pop()

        # backward post process
        if b_model_chunk_id:
            with b_context:
                # launch grad synchronization (custom grad sync)
                # Note: Asynchronous communication tends to slow down compute.
                # To reduce idling from mismatched microbatch times, we launch
                # asynchronous communication at the same time across the
                # pipeline-parallel group.
                if config.grad_sync_func is not None:
                    grad_sync_virtual_microbatch_id = b_virtual_microbatch_id - pipeline_parallel_rank
                    if grad_sync_virtual_microbatch_id >= 0 and is_last_microbatch_for_model_chunk(
                            grad_sync_virtual_microbatch_id
                    ):
                        grad_sync_chunk_id = get_model_chunk_id(
                            grad_sync_virtual_microbatch_id, forward=False
                        )
                        enable_grad_sync()
                        config.grad_sync_func[grad_sync_chunk_id](model[grad_sync_chunk_id].parameters())
                        synchronized_model_chunks.add(grad_sync_chunk_id)
                disable_grad_sync()     
                if input_tensor is not None:
                    assert input_tensor_grad is not None

        return output_tensor, input_tensor_grad


    ### ==============================================================================================

    # Run warmup forward passes.
    parallel_state.set_virtual_pipeline_model_parallel_rank(0)
    input_tensors[0].append(p2p_communication.recv_forward(tensor_shape, config))

    fwd_wait_handles = None
    fwd_wait_recv_handles = None
    bwd_wait_handles = None
    bwd_wait_recv_handles = None
    if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
        fwd_recv_buffer_size = (
                config.microbatch_group_size_per_vp_stage - pipeline_parallel_size + 1
        )
    else:
        fwd_recv_buffer_size = 1
    if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
        bwd_recv_buffer_size = (
                config.microbatch_group_size_per_vp_stage - pipeline_parallel_size + 1
        )
    else:
        bwd_recv_buffer_size = 1
    fwd_recv_buffer = [None] * fwd_recv_buffer_size
    bwd_recv_buffer = [None] * bwd_recv_buffer_size
    recv_prev_wait_handles = []
    send_next_wait_handle = None
    send_prev_wait_handle = None
    recv_next_wait_handles = []

    for k in range(num_warmup_microbatches):
        cur_model_chunk_id = get_model_chunk_id(k, forward=True)
        parallel_state.set_virtual_pipeline_model_parallel_rank(cur_model_chunk_id)

        if config.overlap_p2p_comm_warmup_flush:
            if not parallel_state.is_pipeline_first_stage() and k != 0:
                assert recv_prev_wait_handles, (
                    f'pp rank {pipeline_parallel_rank}, iteration {k},'
                    'should have registered recv handle'
                )
                recv_prev_wait_handle = recv_prev_wait_handles.pop(0)
                recv_prev_wait_handle.wait()

        # Determine if tensor should be received from previous stage.
        recv_prev, next_forward_model_chunk_id = recv_tensor_from_previous_stage(k, forward=True)

        # No receive in last iteration when recv iteration k+1.
        if k == (total_num_microbatches - 1):
            recv_prev = False

        # Prefetch recv for iteration k+1 for non-first ranks.
        if config.overlap_p2p_comm_warmup_flush and not parallel_state.is_pipeline_first_stage(
                ignore_virtual=True
        ):
            fwd_recv_buffer[k % fwd_recv_buffer_size], fwd_wait_recv_handles = (
                p2p_communication.send_forward_recv_forward(
                    output_tensor=None,  # No output_tensor to send.
                    recv_prev=recv_prev,
                    tensor_shape=tensor_shape,
                    config=config,
                    overlap_p2p_comm=True,
                )
            )

            if fwd_wait_recv_handles:
                recv_prev_wait_handles.append(fwd_wait_recv_handles.pop("recv_prev"))

        # Decide to checkpoint all layers' activations of the current micro-batch.
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                    k % max_outstanding_backprops
                    >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        microbatch_id = get_microbatch_id_in_model_chunk(k, forward=True)
        output_tensor, _ = forward_backward_helper(f_virtual_microbatch_id=k, f_microbatch_id=microbatch_id)
        # Don't send tensor downstream if on last stage.
        if parallel_state.is_pipeline_last_stage():
            output_tensor = None

        # Send and receive tensors as appropriate (send tensors computed
        # in this iteration; receive tensors for next iteration).
        if not config.overlap_p2p_comm_warmup_flush:
            if (
                    k == (num_warmup_microbatches - 1)
                    and not config.overlap_p2p_comm
                    and not forward_only
                    and not all_warmup_microbatches
            ):
                input_tensor_grad = None
                recv_next = True
                if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    recv_next = False
                
                (input_tensor, output_tensor_grad) = (
                    p2p_communication.send_forward_backward_recv_forward_backward(
                        output_tensor,
                        input_tensor_grad,
                        recv_prev=recv_prev,
                        recv_next=recv_next,
                        tensor_shape=tensor_shape,
                        config=config,
                    )
                )
                
                output_tensor_grads[num_model_chunks - 1].append(output_tensor_grad)
            else:
                input_tensor = p2p_communication.send_forward_recv_forward(
                    output_tensor, recv_prev=recv_prev, tensor_shape=tensor_shape, config=config
                )
            if recv_prev:
                input_tensors[next_forward_model_chunk_id].append(input_tensor)
            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
        else:
            if not parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                # Send only since recv prefetched.
                _, fwd_wait_handles = p2p_communication.send_forward_recv_forward(
                    output_tensor,
                    recv_prev=False,
                    tensor_shape=tensor_shape,
                    config=config,
                    overlap_p2p_comm=True,
                )
            else:  # No prefetch for first rank, so both send and recv initiated.
                fwd_recv_buffer[k % fwd_recv_buffer_size], fwd_wait_handles = (
                    p2p_communication.send_forward_recv_forward(
                        output_tensor,
                        recv_prev=recv_prev,
                        tensor_shape=tensor_shape,
                        config=config,
                        overlap_p2p_comm=True,
                    )
                )
            if send_next_wait_handle is not None:
                send_next_wait_handle.wait()
            if fwd_wait_handles is not None:
                send_next_wait_handle = (
                    fwd_wait_handles.pop("send_next") if "send_next" in fwd_wait_handles else None
                )
                if "recv_prev" in fwd_wait_handles:
                    recv_prev_wait_handles.append(fwd_wait_handles.pop("recv_prev"))

            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
            if recv_prev:
                input_tensors[next_forward_model_chunk_id].append(
                    fwd_recv_buffer[k % fwd_recv_buffer_size]
                )
                fwd_recv_buffer[(k + 1) % fwd_recv_buffer_size] = None

        if config.overlap_p2p_comm:
            if (
                    k == (num_warmup_microbatches - 1)
                    and not forward_only
                    and not all_warmup_microbatches
            ):
                input_tensor_grad = None
                recv_next = True
                if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    recv_next = False

                (bwd_recv_buffer[-1], bwd_wait_handles) = (
                    p2p_communication.send_backward_recv_backward(
                        input_tensor_grad,
                        recv_next=recv_next,
                        tensor_shape=tensor_shape,
                        config=config,
                        overlap_p2p_comm=True,
                    )
                )
                if send_prev_wait_handle is not None:
                    send_prev_wait_handle.wait()
                if bwd_wait_handles is not None:
                    send_prev_wait_handle = (
                        bwd_wait_handles.pop("send_prev")
                        if "send_prev" in bwd_wait_handles
                        else None
                    )
                    if "recv_next" in bwd_wait_handles:
                        recv_next_wait_handles.append(bwd_wait_handles.pop("recv_next"))

                if recv_next:
                    output_tensor_grads[num_model_chunks - 1].append(bwd_recv_buffer[-1])
    
    
    # Run 1F1B in steady state.
    for k in range(num_microbatches_remaining):
        # Forward pass.
        forward_k = k + num_warmup_microbatches

        # Decide to checkpoint all layers' activations of the current micro-batch.
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                    forward_k % max_outstanding_backprops
                    >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        cur_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
        parallel_state.set_virtual_pipeline_model_parallel_rank(cur_model_chunk_id)
        microbatch_id = get_microbatch_id_in_model_chunk(forward_k, forward=True)
        if config.overlap_p2p_comm:
            assert False, "not supported yet"
            if not parallel_state.is_pipeline_first_stage():
                if config.overlap_p2p_comm_warmup_flush:
                    assert recv_prev_wait_handles, (
                        f'pp rank {pipeline_parallel_rank}, fwd iteration {forward_k}, '
                        'should have registered recv handle'
                    )
                    recv_prev_wait_handle = recv_prev_wait_handles.pop(0)
                    recv_prev_wait_handle.wait()
                else:
                    if recv_prev_wait_handles is not None and recv_prev_wait_handles:
                        recv_prev_wait_handle = recv_prev_wait_handles.pop(0)
                        recv_prev_wait_handle.wait()

            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
            output_tensor, _ = forward_step_helper(
                forward_k, microbatch_id, checkpoint_activations_microbatch
            )

            # Determine if current stage has anything to send in either direction,
            # otherwise set tensor to None.
            forward_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
            parallel_state.set_virtual_pipeline_model_parallel_rank(forward_model_chunk_id)

            # Last virtual stage no activation tensor to send.
            if parallel_state.is_pipeline_last_stage():
                output_tensor = None

            recv_prev, next_forward_model_chunk_id = recv_tensor_from_previous_stage(
                forward_k, forward=True
            )

            # If last iteration, don't receive; we already received one extra
            # before the start of the for loop.
            if k == (num_microbatches_remaining - 1):
                recv_prev = False

            # Send activation tensor to the next stage and receive activation tensor from the
            # previous stage
            fwd_recv_buffer[forward_k % fwd_recv_buffer_size], fwd_wait_handles = (
                p2p_communication.send_forward_recv_forward(
                    output_tensor,
                    recv_prev=recv_prev,
                    tensor_shape=tensor_shape,
                    config=config,
                    overlap_p2p_comm=True,
                )
            )
            if send_next_wait_handle is not None:
                send_next_wait_handle.wait()
            if fwd_wait_handles is not None:
                send_next_wait_handle = (
                    fwd_wait_handles.pop("send_next") if "send_next" in fwd_wait_handles else None
                )
                if "recv_prev" in fwd_wait_handles:
                    recv_prev_wait_handles.append(fwd_wait_handles.pop("recv_prev"))
            # assert fwd_wait_handles is not None

            # Backward pass.
            backward_k = k
            backward_model_chunk_id = get_model_chunk_id(backward_k, forward=False)
            parallel_state.set_virtual_pipeline_model_parallel_rank(backward_model_chunk_id)
            if not parallel_state.is_pipeline_last_stage():
                if config.overlap_p2p_comm_warmup_flush:
                    assert recv_next_wait_handles, (
                        f'pp rank {pipeline_parallel_rank}, bwd iteration {backward_k}, '
                        'should have registered recv next handle'
                    )
                    recv_next_wait_handle = recv_next_wait_handles.pop(0)
                    recv_next_wait_handle.wait()
                else:
                    if recv_next_wait_handles is not None and recv_next_wait_handles:
                        recv_next_wait_handle = recv_next_wait_handles.pop(0)
                        recv_next_wait_handle.wait()

            input_tensor_grad = backward_step_helper(backward_k)

            # First virtual stage no activation gradient tensor to send.
            if parallel_state.is_pipeline_first_stage():
                input_tensor_grad = None

            recv_next, next_backward_model_chunk_id = recv_tensor_from_previous_stage(
                backward_k, forward=False
            )

            (bwd_recv_buffer[backward_k % bwd_recv_buffer_size], bwd_wait_handles) = (
                p2p_communication.send_backward_recv_backward(
                    input_tensor_grad,
                    recv_next=recv_next,
                    tensor_shape=tensor_shape,
                    config=config,
                    overlap_p2p_comm=True,
                )
            )
            if send_prev_wait_handle is not None:
                send_prev_wait_handle.wait()
            if bwd_wait_handles is not None:
                send_prev_wait_handle = (
                    bwd_wait_handles.pop("send_prev") if "send_prev" in bwd_wait_handles else None
                )
                if "recv_next" in bwd_wait_handles:
                    recv_next_wait_handles.append(bwd_wait_handles.pop("recv_next"))

            # Put input_tensor and output_tensor_grad in data structures in the
            # right location.
            if recv_prev:
                input_tensors[next_forward_model_chunk_id].append(
                    fwd_recv_buffer[forward_k % fwd_recv_buffer_size]
                )
                fwd_recv_buffer[(forward_k + 1) % fwd_recv_buffer_size] = None
            if recv_next:
                output_tensor_grads[next_backward_model_chunk_id].append(
                    bwd_recv_buffer[backward_k % bwd_recv_buffer_size]
                )
                bwd_recv_buffer[(backward_k + 1) % bwd_recv_buffer_size] = None
        else:  # No p2p overlap.

            # fused schedule
            backward_k = k
            output_tensor, input_tensor_grad = forward_backward_helper(f_virtual_microbatch_id=forward_k, f_microbatch_id=microbatch_id, b_virtual_microbatch_id=backward_k)

            # Send output_tensor and input_tensor_grad, receive input_tensor
            # and output_tensor_grad.

            # Determine if current stage has anything to send in either direction,
            # otherwise set tensor to None.
            forward_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
            parallel_state.set_virtual_pipeline_model_parallel_rank(forward_model_chunk_id)
            if parallel_state.is_pipeline_last_stage():
                output_tensor = None

            backward_model_chunk_id = get_model_chunk_id(backward_k, forward=False)
            parallel_state.set_virtual_pipeline_model_parallel_rank(backward_model_chunk_id)
            if parallel_state.is_pipeline_first_stage():
                input_tensor_grad = None

            recv_prev, next_forward_model_chunk_id = recv_tensor_from_previous_stage(
                forward_k, forward=True
            )

            recv_next, next_backward_model_chunk_id = recv_tensor_from_previous_stage(
                backward_k, forward=False
            )

            # If last iteration, don't receive; we already received one extra
            # before the start of the for loop.
            if k == (num_microbatches_remaining - 1):
                recv_prev = False

            # Communicate tensors.
                
            (input_tensor, output_tensor_grad) = (
                p2p_communication.send_forward_backward_recv_forward_backward(
                    output_tensor,
                    input_tensor_grad,
                    recv_prev=recv_prev,
                    recv_next=recv_next,
                    tensor_shape=tensor_shape,
                    config=config,
                )
            )
            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

            # Put input_tensor and output_tensor_grad in data structures in the
            # right location.
            if recv_prev:
                input_tensors[next_forward_model_chunk_id].append(input_tensor)
            if recv_next:
                output_tensor_grads[next_backward_model_chunk_id].append(output_tensor_grad)

    deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

    # Run cooldown backward passes (flush out pipeline).
    if not forward_only:
        if bwd_wait_handles is not None:
            for bwd_wait_handle in bwd_wait_handles.values():
                bwd_wait_handle.wait()

        if all_warmup_microbatches:
            output_tensor_grads[num_model_chunks - 1].append(
                p2p_communication.recv_backward(tensor_shape, config=config)
            )
        for k in range(num_microbatches_remaining, total_num_microbatches):
            cur_model_chunk_id = get_model_chunk_id(k, forward=False)
            parallel_state.set_virtual_pipeline_model_parallel_rank(cur_model_chunk_id)
            if not parallel_state.is_pipeline_last_stage() and k != 0:
                if config.overlap_p2p_comm_warmup_flush:
                    assert recv_next_wait_handles, (
                        f'pp rank {pipeline_parallel_rank}, backward iteration {k}, '
                        'should have registered recv next handle'
                    )
                    recv_next_wait_handle = recv_next_wait_handles.pop(0)
                    recv_next_wait_handle.wait()
                else:
                    if recv_next_wait_handles is not None and recv_next_wait_handles:
                        recv_next_wait_handle = recv_next_wait_handles.pop(0)
                        recv_next_wait_handle.wait()

            recv_next, next_backward_model_chunk_id = recv_tensor_from_previous_stage(
                k, forward=False
            )

            if k == (total_num_microbatches - 1):
                recv_next = False

            # Prefetch recv for backward iteration k+1 for non last ranks.
            if config.overlap_p2p_comm_warmup_flush and not parallel_state.is_pipeline_last_stage(
                    ignore_virtual=True
            ):
                bwd_recv_buffer[k % bwd_recv_buffer_size], bwd_wait_recv_handles = (
                    p2p_communication.send_backward_recv_backward(
                        input_tensor_grad=None,  # No input_tensor_grad to send.
                        recv_next=recv_next,
                        tensor_shape=tensor_shape,
                        config=config,
                        overlap_p2p_comm=True,
                    )
                )

                if bwd_wait_recv_handles:
                    recv_next_wait_handles.append(bwd_wait_recv_handles.pop("recv_next"))

            tmp, input_tensor_grad = forward_backward_helper(b_virtual_microbatch_id=k)

            # First virtual stage no activation gradient tensor to send.
            if parallel_state.is_pipeline_first_stage():
                input_tensor_grad = None

            if config.overlap_p2p_comm_warmup_flush:
                if not parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    _, bwd_wait_handles = p2p_communication.send_backward_recv_backward(
                        input_tensor_grad,
                        recv_next=False,
                        tensor_shape=tensor_shape,
                        config=config,
                        overlap_p2p_comm=True,
                    )
                else:
                    bwd_recv_buffer[k % bwd_recv_buffer_size], bwd_wait_handles = (
                        p2p_communication.send_backward_recv_backward(
                            input_tensor_grad,
                            recv_next=recv_next,
                            tensor_shape=tensor_shape,
                            config=config,
                            overlap_p2p_comm=True,
                        )
                    )

                if send_prev_wait_handle is not None:
                    send_prev_wait_handle.wait()
                if bwd_wait_handles is not None:
                    send_prev_wait_handle = (
                        bwd_wait_handles.pop("send_prev")
                        if "send_prev" in bwd_wait_handles
                        else None
                    )
                    if "recv_next" in bwd_wait_handles:
                        recv_next_wait_handles.append(bwd_wait_handles.pop("recv_next"))
                if recv_next:
                    output_tensor_grads[next_backward_model_chunk_id].append(
                        bwd_recv_buffer[k % bwd_recv_buffer_size]
                    )
                    bwd_recv_buffer[(k + 1) % bwd_recv_buffer_size] = None

            else:
                output_tensor_grad = p2p_communication.send_backward_recv_backward(
                    input_tensor_grad, recv_next=recv_next, tensor_shape=tensor_shape, config=config
                )

                if recv_next:
                    output_tensor_grads[next_backward_model_chunk_id].append(output_tensor_grad)

        if send_prev_wait_handle is not None:
            send_prev_wait_handle.wait()

        # Launch any remaining grad reductions.
        enable_grad_sync()
        if config.grad_sync_func is not None:
            for model_chunk_id in range(num_model_chunks):
                if model_chunk_id not in synchronized_model_chunks:
                    config.grad_sync_func[model_chunk_id](model[model_chunk_id].parameters())
                    synchronized_model_chunks.add(model_chunk_id)

    assert (
        not recv_prev_wait_handles
    ), 'recv_prev_wait_handles should be cleared at the end of a step'
    assert (
        not recv_next_wait_handles
    ), 'recv_next_wait_handles should be cleared at the end of a step'

    if config.finalize_model_grads_func is not None and not forward_only:

        # If defer_embedding_wgrad_compute is enabled we need to do the
        # weight gradient GEMM's here.
        finish_embedding_wgrad_compute(config, embedding_module)

        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism, layernorm all-reduce for sequence parallelism, and
        # embedding all-reduce for pipeline parallelism).
        config.finalize_model_grads_func(
            model, total_num_tokens if config.calculate_per_token_loss else None
        )

    # Restore config.grad_sync_func and config.param_sync_func.
    if forward_only:
        config.grad_sync_func, config.param_sync_func = grad_sync_func, param_sync_func

    if config.timers is not None:
        config.timers('forward-backward').stop()

    if hasattr(config, 'enable_cuda_graph') and config.enable_cuda_graph:
        create_cudagraphs()

    return forward_data_store





