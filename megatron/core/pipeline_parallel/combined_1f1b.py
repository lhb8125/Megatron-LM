from typing import Callable, Union, Any, Tuple
import torch
from torch import Tensor

from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllPerBatchState


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
        self.inputs = [e.detach() if e is not None else None for e in inputs]
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
        # not multiple input
        if len(output_grad) == 1:
            output_grad = output_grad[0]
        with torch.cuda.stream(self.stream):
            torch.autograd.backward(self.output, grad_tensors=output_grad)
        if not len(self.inputs):
            # empty input, alse need record stream
            self.event.record(self.stream)

        torch.cuda.nvtx.range_pop()
        return self.get_grad()

    def get_grad(self):
        grad = [e.grad if e is not None else None for e in self.inputs]
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
    f_layer, b_layer, f_input=None, b_grad=None, pre_forward=None, pre_backward=None
):

    if pre_backward is not None:
        # attn backward from last iter
        assert b_grad is None
        b_grad = pre_backward()

    if b_layer is not None:
        b_grad = b_layer.post_combine.backward(b_grad)

    if b_layer is not None:
        b_grad = b_layer.combine.backward(b_grad)

    if pre_forward is not None:
        assert f_input is None
        # post combine from last iter
        f_input = pre_forward()
        del pre_forward
    if f_layer is not None:
        f_input = f_layer.attn.forward(f_input)

    if f_layer is not None:
        f_input = f_layer.dispatch.forward(f_input)

    if b_layer is not None:
        b_grad = b_layer.mlp.backward(b_grad)
    if b_layer is not None:
        b_grad = b_layer.dispatch.backward(b_grad)

    if f_layer is not None:
        f_input = f_layer.mlp.forward(f_input)
    if f_layer is not None:
        f_input = f_layer.combine.forward(f_input)

    def next_iter_pre_forward():
        if f_layer is not None:
            output = f_layer.post_combine.forward(f_input)
            return output

    def next_iter_pre_backward():
        if b_layer is not None:
            grad = b_layer.attn.backward(b_grad)
            return grad

    if f_layer and b_layer:
        return next_iter_pre_forward, next_iter_pre_backward
    else:
        return next_iter_pre_forward(), next_iter_pre_backward()

def schedule_chunk_1f1b(f_schedule_plan, b_schedule_plan, grad=None):

    f_input = None
    if f_schedule_plan is not None:
        f_input = f_schedule_plan.pre_process.forward()

    if b_schedule_plan is not None:
        assert grad is not None
        if b_schedule_plan.post_process is not None:
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
            f_layer, b_layer, pre_forward=pre_forward, pre_backward=pre_backward
        )
        torch.cuda.nvtx.range_pop()

    # tail forward
    f_input = pre_forward()
    for i in range(overlaped_layers, f_num_layers):
        f_layer = f_schedule_plan.get_layer(i)
        torch.cuda.nvtx.range_push(f"layer_{i}f")
        f_input, tmp = schedule_layer_1f1b(f_layer, None, f_input=f_input)
        torch.cuda.nvtx.range_pop()

    if f_schedule_plan is not None and f_schedule_plan.post_process is not None:
        f_input = f_schedule_plan.post_process.forward(f_input)

    # tail backward
    grad = pre_backward()
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


global _COMP_STREAM = None
global _COM_STREAM = None

def set_streams(comp_stream, com_stream):
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



