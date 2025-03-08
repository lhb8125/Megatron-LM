import weakref
from typing import Any, Callable, Optional, Tuple, Union
import contextlib

import torch
from torch import Tensor

from megatron.core.pipeline_parallel.combined_1f1b import (
    AbstractSchedulePlan,
    ScheduleNode,
    get_com_stream,
    get_comp_stream,
)
from megatron.core.transformer import transformer_layer
from megatron.core.transformer.module import float16_to_fp32
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllPerBatchState



def weak_method(method):
    method_ref = weakref.WeakMethod(method)
    del method

    def wrapped_func(*args, **kwarg):
        # nonlocal object_ref
        return method_ref()(*args, **kwarg)

    return wrapped_func


class PreProcessNode(ScheduleNode):

    def __init__(self, gpt_model, model_chunk_state, event, stream):
        super().__init__(weak_method(self.forward_impl), stream, event)
        self.gpt_model = gpt_model
        self.model_chunk_state = model_chunk_state

    def forward_impl(self):

        gpt_model = self.gpt_model
        decoder_input = self.model_chunk_state.decoder_input
        input_ids = self.model_chunk_state.input_ids
        position_ids = self.model_chunk_state.position_ids
        inference_params = self.model_chunk_state.inference_params
        packed_seq_params = self.model_chunk_state.packed_seq_params

        # Decoder embedding.
        if decoder_input is not None:
            pass
        elif gpt_model.pre_process:
            decoder_input = gpt_model.embedding(input_ids=input_ids, position_ids=position_ids)
        else:
            # intermediate stage of pipeline
            # decoder will get hidden_states from encoder.input_tensor
            decoder_input = gpt_model.decoder.input_tensor

        # Rotary positional embeddings (embedding is None for PP intermediate devices)
        rotary_pos_emb = None
        rotary_pos_cos = None
        rotary_pos_sin = None
        if (
            gpt_model.position_embedding_type == 'rope'
            and not gpt_model.config.multi_latent_attention
        ):
            if not gpt_model.training and gpt_model.config.flash_decode and inference_params:
                # Flash decoding uses precomputed cos and sin for RoPE
                rotary_pos_cos, rotary_pos_sin = gpt_model.rotary_pos_emb_cache.setdefault(
                    inference_params.max_sequence_length,
                    gpt_model.rotary_pos_emb.get_cos_sin(inference_params.max_sequence_length),
                )
            else:
                rotary_seq_len = gpt_model.rotary_pos_emb.get_rotary_seq_len(
                    inference_params,
                    gpt_model.decoder,
                    decoder_input,
                    gpt_model.config,
                    packed_seq_params,
                )
                rotary_pos_emb = gpt_model.rotary_pos_emb(
                    rotary_seq_len,
                    packed_seq=packed_seq_params is not None
                    and packed_seq_params.qkv_format == 'thd',
                )
        if (
            (gpt_model.config.enable_cuda_graph or gpt_model.config.flash_decode)
            and rotary_pos_cos is not None
            and inference_params
        ):
            sequence_len_offset = torch.tensor(
                [inference_params.sequence_len_offset] * inference_params.current_batch_size,
                dtype=torch.int32,
                device=rotary_pos_cos.device,  # Co-locate this with the rotary tensors
            )
        else:
            sequence_len_offset = None

        # saved for later use
        self.model_chunk_state.rotary_pos_emb = rotary_pos_emb
        self.model_chunk_state.rotary_pos_cos = rotary_pos_cos
        self.model_chunk_state.rotary_pos_sin = rotary_pos_sin
        self.model_chunk_state.sequence_len_offset = sequence_len_offset
        return decoder_input


class PostProcessNode(ScheduleNode):

    def __init__(self, gpt_model, model_chunk_state, event, stream):
        super().__init__(weak_method(self.forward_impl), stream, event)
        self.gpt_model = gpt_model
        self.model_chunk_state = model_chunk_state

    def forward_impl(self, hidden_states):

        # Final layer norm.
        if self.gpt_model.decoder.final_layernorm is not None:
            hidden_states = self.gpt_model.decoder.final_layernorm(hidden_states)
            # TENorm produces a "viewed" tensor. This will result in schedule.py's
            # deallocate_output_tensor() throwing an error, so a viewless tensor is
            # created to prevent this.
            hidden_states = transformer_layer.make_viewless_tensor(
                inp=hidden_states, requires_grad=True, keep_graph=True
            )

        gpt_model = self.gpt_model
        runtime_gather_output = self.model_chunk_state.runtime_gather_output
        labels = self.model_chunk_state.labels
        output_weight = None
        if gpt_model.share_embeddings_and_output_weights:
            output_weight = gpt_model.shared_embedding_or_output_weight()
        logits, _ = gpt_model.output_layer(
            hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output
        )

        if labels is None:
            # [s b h] => [b s h]
            return float16_to_fp32(logits.transpose(0, 1).contiguous())
        loss = float16_to_fp32(gpt_model.compute_language_model_loss(labels, logits))
        return loss


class TransformerLayerNode(ScheduleNode):

    def __init__(self, chunk_state, common_state, layer, stream, event):
        super().__init__(weak_method(self.forward_impl), stream, event)
        # layer state
        self.common_state = common_state
        # model chunk state
        self.chunk_state = chunk_state
        self.layer = layer


class AttnNode(TransformerLayerNode):

    def forward_impl(self, hidden_states):
        attention_mask = self.chunk_state.attention_mask
        context = self.chunk_state.context
        context_mask = self.chunk_state.context_mask
        rotary_pos_emb = self.chunk_state.rotary_pos_emb
        rotary_pos_cos = self.chunk_state.rotary_pos_cos
        rotary_pos_sin = self.chunk_state.rotary_pos_sin
        attention_bias = self.chunk_state.attention_bias
        inference_params = self.chunk_state.inference_params
        packed_seq_params = self.chunk_state.packed_seq_params

        # Residual connection.
        residual = hidden_states

        # Optional Input Layer norm
        input_layernorm_output = self.layer.input_layernorm(hidden_states)

        # Self attention.
        attention_output_with_bias = self.layer.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
        )

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.layer.bias_dropout_add_exec_handler():
            hidden_states = self.layer.self_attn_bda(
                self.layer.training, self.layer.config.bias_dropout_fusion
            )(attention_output_with_bias, residual, self.layer.hidden_dropout)

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm after self-attention
        pre_cross_attn_layernorm_output = self.layer.pre_cross_attn_layernorm(hidden_states)

        # Cross attention.
        attention_output_with_bias = self.layer.cross_attention(
            pre_cross_attn_layernorm_output,
            attention_mask=context_mask,
            key_value_states=context,
            inference_params=inference_params,
        )

        if isinstance(attention_output_with_bias, dict) and "context" in attention_output_with_bias:
            context = attention_output_with_bias["context"]

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.layer.bias_dropout_add_exec_handler():
            hidden_states = self.layer.cross_attn_bda(
                self.layer.training, self.layer.config.bias_dropout_fusion
            )(attention_output_with_bias, residual, self.layer.hidden_dropout)

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm post the cross-attention.
        pre_mlp_layernorm_output = self.layer.pre_mlp_layernorm(hidden_states)

        # residual, pre_mlp_layernorm_output
        # MLP.
        probs, routing_map = self.layer.mlp.router(pre_mlp_layernorm_output)

        self.common_state.probs = probs
        token_dispatcher = self.layer.mlp.token_dispatcher
        with token_dispatcher.per_batch_state_context(self.common_state):
            tokens_per_expert = token_dispatcher.meta_prepare(
                pre_mlp_layernorm_output, probs, routing_map
            )
            permutated_local_input_tokens = token_dispatcher.dispatch_preprocess(
                pre_mlp_layernorm_output, routing_map
            )
        self.common_state.tokens_per_expert = tokens_per_expert
        return residual, pre_mlp_layernorm_output, permutated_local_input_tokens, probs


class DispatchNode(TransformerLayerNode):

    def forward_impl(
        self, residual, pre_mlp_layernorm_output, permutated_local_input_tokens, probs
    ):

        self.common_state.probs = probs
        token_dispatcher = self.layer.mlp.token_dispatcher
        with token_dispatcher.per_batch_state_context(self.common_state):
            inputs = permutated_local_input_tokens
            dispatched_input = token_dispatcher.dispatch_all_to_all(permutated_local_input_tokens)
            # release tensor not used by backward
            inputs.untyped_storage().resize_(0)
        return residual, pre_mlp_layernorm_output, dispatched_input, probs


class MlPNode(TransformerLayerNode):
    def forward_impl(self, residual, pre_mlp_layernorm_output, dispatched_input, probs):
        self.common_state.probs = probs
        token_dispatcher = self.layer.mlp.token_dispatcher
        with token_dispatcher.per_batch_state_context(self.common_state):
            inputs = dispatched_input
            dispatched_input = token_dispatcher.dispatch_postprocess(dispatched_input)
            # release tensor not used by backward
            inputs.untyped_storage().resize_(0)
            expert_output, mlp_bias = self.layer.mlp.experts(
                dispatched_input, self.common_state.tokens_per_expert
            )
            assert mlp_bias is None
            permutated_local_input_tokens = token_dispatcher.combine_preprocess(expert_output)
        shared_output = self.layer.mlp.shared_experts(pre_mlp_layernorm_output)
        return residual, permutated_local_input_tokens, shared_output, probs


class CombineNode(TransformerLayerNode):
    def forward_impl(self, residual, permutated_local_input_tokens, shared_output, probs):
        token_dispatcher = self.layer.mlp.token_dispatcher
        self.common_state.probs = probs
        with token_dispatcher.per_batch_state_context(self.common_state):
            # release tensor not used by backward
            inputs = permutated_local_input_tokens
            permutated_local_input_tokens = token_dispatcher.combine_all_to_all(
                permutated_local_input_tokens
            )
            inputs.untyped_storage().resize_(0)
        return residual, permutated_local_input_tokens, shared_output, probs


class CombinePostProcessNode(TransformerLayerNode):
    def forward_impl(self, residual, permutated_local_input_tokens, shared_output, probs):
        token_dispatcher = self.layer.mlp.token_dispatcher
        self.common_state.probs = probs
        with token_dispatcher.per_batch_state_context(self.common_state):
            output = token_dispatcher.combine_postprocess(permutated_local_input_tokens)
        output = output.type_as(residual)
        output += shared_output
        # release tensor not used by backward
        shared_output.untyped_storage().resize_(0)
        mlp_output_with_bias = (output, None)

        with self.layer.bias_dropout_add_exec_handler():
            hidden_states = self.layer.mlp_bda(
                self.layer.training, self.layer.config.bias_dropout_fusion
            )(mlp_output_with_bias, residual, self.layer.hidden_dropout)
        output = transformer_layer.make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )
        return output


class AttnNodeNoMoe(TransformerLayerNode):

    def forward_impl(self, hidden_states):
        attention_mask = self.chunk_state.attention_mask
        context = self.chunk_state.context
        context_mask = self.chunk_state.context_mask
        rotary_pos_emb = self.chunk_state.rotary_pos_emb
        rotary_pos_cos = self.chunk_state.rotary_pos_cos
        rotary_pos_sin = self.chunk_state.rotary_pos_sin
        attention_bias = self.chunk_state.attention_bias
        inference_params = self.chunk_state.inference_params
        packed_seq_params = self.chunk_state.packed_seq_params

        # Residual connection.
        residual = hidden_states

        # Optional Input Layer norm
        input_layernorm_output = self.layer.input_layernorm(hidden_states)

        # Self attention.
        attention_output_with_bias = self.layer.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
        )

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.layer.bias_dropout_add_exec_handler():
            hidden_states = self.layer.self_attn_bda(
                self.layer.training, self.layer.config.bias_dropout_fusion
            )(attention_output_with_bias, residual, self.layer.hidden_dropout)

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm after self-attention
        pre_cross_attn_layernorm_output = self.layer.pre_cross_attn_layernorm(hidden_states)

        # Cross attention.
        attention_output_with_bias = self.layer.cross_attention(
            pre_cross_attn_layernorm_output,
            attention_mask=context_mask,
            key_value_states=context,
            inference_params=inference_params,
        )

        if isinstance(attention_output_with_bias, dict) and "context" in attention_output_with_bias:
            context = attention_output_with_bias["context"]

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.layer.bias_dropout_add_exec_handler():
            hidden_states = self.layer.cross_attn_bda(
                self.layer.training, self.layer.config.bias_dropout_fusion
            )(attention_output_with_bias, residual, self.layer.hidden_dropout)

        return hidden_states

        # Residual connection.
        residual = hidden_states


class FakeScheduleNode:

    def forward(self, inputs):
        return inputs

    def backward(self, outgrads):
        return outgrads


class MlpNoMoe(TransformerLayerNode):
    def forward_impl(self, hidden_states):
        # Residual connection.
        residual = hidden_states

        # Optional Layer norm post the cross-attention.
        pre_mlp_layernorm_output = self.layer.pre_mlp_layernorm(hidden_states)

        # MLP.
        mlp_output_with_bias = self.layer.mlp(pre_mlp_layernorm_output)

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.layer.bias_dropout_add_exec_handler():
            hidden_states = self.layer.mlp_bda(
                self.layer.training, self.layer.config.bias_dropout_fusion
            )(mlp_output_with_bias, residual, self.layer.hidden_dropout)

        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        output = transformer_layer.make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )

        # CUDA graph requires returned values to be Tensors
        if self.layer.config.external_cuda_graph and self.layer.training:
            return output
        return output


def build_non_moe_layer_plan(layer, event, chunk_state, comp_stream, com_stream):
    common_state = TransformerLayerState()
    attn = AttnNodeNoMoe(chunk_state, common_state, layer, comp_stream, event)
    attn.name = "attn"
    dispatch = FakeScheduleNode()
    mlp = MlpNoMoe(chunk_state, common_state, layer, comp_stream, event)
    combine = FakeScheduleNode()
    post_combine = FakeScheduleNode()
    return TransformerLayerSchedulePlan(attn, dispatch, mlp, combine, post_combine)


def build_layer_schedule_plan(layer, event, chunk_state, comp_stream, com_stream):
    if not isinstance(layer.mlp, MoELayer):
        return build_non_moe_layer_plan(layer, event, chunk_state, comp_stream, com_stream)
    common_state = TransformerLayerState()
    attn = AttnNode(chunk_state, common_state, layer, comp_stream, event)
    attn.name = "attn"
    dispatch = DispatchNode(chunk_state, common_state, layer, com_stream, event)
    dispatch.name = "dispatch"
    mlp = MlPNode(chunk_state, common_state, layer, comp_stream, event)
    mlp.name = "mlp"
    combine = CombineNode(chunk_state, common_state, layer, com_stream, event)
    combine.name = "combine"
    post_combine = CombinePostProcessNode(chunk_state, common_state, layer, comp_stream, event)
    post_combine.name = "post_combine"
    return TransformerLayerSchedulePlan(attn, dispatch, mlp, combine, post_combine)


def build_model_chunk_schedule_plan(
    model,
    input_ids: Tensor,
    position_ids: Tensor,
    attention_mask: Tensor,
    decoder_input: Tensor = None,
    labels: Tensor = None,
    inference_params=None,
    packed_seq_params=None,
    extra_block_kwargs=None,
    runtime_gather_output: Optional[bool] = None,
):

    comp_stream = get_comp_stream()
    com_stream = get_com_stream()
    model_chunk_schedule_plan = ModelChunkSchedulePlan()
    event = model_chunk_schedule_plan.event
    state = model_chunk_schedule_plan.state
    # save for later use
    state.input_ids = input_ids
    state.position_ids = position_ids
    state.attention_mask = attention_mask
    state.decoder_input = decoder_input
    state.labels = labels
    state.inference_params = inference_params
    state.packed_seq_params = packed_seq_params
    state.extra_block_kwargs = extra_block_kwargs
    state.runtime_gather_output = runtime_gather_output
    state.context = None
    state.context_mask = None
    state.attention_bias = None

    # build preprocess
    model_chunk_schedule_plan.pre_process = PreProcessNode(model, state, event, comp_stream)
    model_chunk_schedule_plan.pre_process.name = "pre_process"
    # build for layers
    for layer_idx in range(model.decoder.num_layers_per_pipeline_rank):
        layer = model.decoder._get_layer(layer_idx)
        layer_plan = build_layer_schedule_plan(layer, event, state, comp_stream, com_stream)
        model_chunk_schedule_plan.add_layer(layer_plan)
    # build post process
    if model.post_process:

        model_chunk_schedule_plan.post_process = PostProcessNode(model, state, event, comp_stream)
        model_chunk_schedule_plan.post_process.name = "post_process"

    return model_chunk_schedule_plan


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


class ModelChunkSchedulePlan(AbstractSchedulePlan):
    def __init__(self):
        super().__init__()
        self._pre_process = None
        self._post_process = None
        self._model_chunk_state = ModelChunkSate()
        self._transformer_layers = []

    @classmethod
    def forward_backward(
        cls,
        f_schedule_plan,
        b_schedule_plan,
        grad=None,
        f_context=None,
        b_context=None,
        pre_forward=None,
        pre_backward=None,
        post_forward=None,
        post_backward=None,
    ):

        return schedule_chunk_1f1b(
            f_schedule_plan,
            b_schedule_plan,
            grad=grad,
            f_context=f_context,
            b_context=b_context,
            pre_forward=pre_forward,
            pre_backward=pre_backward,
            post_forward=post_forward,
            post_backward=post_backward,
        )

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
    f_layer,
    b_layer,
    f_input=None,
    b_grad=None,
    pre_forward=None,
    pre_backward=None,
    f_context=None,
    b_context=None,
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


def schedule_chunk_1f1b(
    f_schedule_plan,
    b_schedule_plan,
    grad=None,
    f_context=None,
    b_context=None,
    pre_forward=None,
    pre_backward=None,
    post_forward=None,
    post_backward=None,
):
    f_context = f_context if f_context is not None else contextlib.nullcontext()
    b_context = b_context if b_context is not None else contextlib.nullcontext()

    if f_schedule_plan:
        # TODO(liuzhenhai93): defered until first layer_pre_forward when integrated with deepep
        # output send/receive sync
        if pre_forward is not None:
            with f_context:
                pre_forward()
        f_schedule_plan.record_current_stream()

    if b_schedule_plan:
        # TODO(liuzhenhai93): defered until first layer_pre_backward  when integrated with deepep
        # grad send/receive sync
        if pre_backward is not None:
            with b_context:
                pre_backward()
        b_schedule_plan.record_current_stream()

    f_input = None

    def layer_pre_forward():
        tmp = f_input
        if f_schedule_plan is not None:
            tmp = f_schedule_plan.pre_process.forward()
        return tmp

    def layer_pre_backward():
        tmp = grad
        if b_schedule_plan is not None:
            assert grad is not None
            if b_schedule_plan.post_process is not None:
                with b_context:
                    tmp = b_schedule_plan.post_process.backward(grad)
        return tmp

    f_num_layers = f_schedule_plan.num_layers() if f_schedule_plan is not None else 0
    b_num_layers = b_schedule_plan.num_layers() if b_schedule_plan is not None else 0
    overlaped_layers = min(f_num_layers, b_num_layers)

    for i in range(overlaped_layers):
        f_layer = f_schedule_plan.get_layer(i)
        b_layer = b_schedule_plan.get_layer(b_num_layers - 1 - i)
        torch.cuda.nvtx.range_push(f"layer_{i}f-layer_{b_num_layers - 1 - i}b")
        layer_pre_forward, layer_pre_backward = schedule_layer_1f1b(
            f_layer,
            b_layer,
            pre_forward=layer_pre_forward,
            pre_backward=layer_pre_backward,
            f_context=f_context,
            b_context=b_context,
        )
        torch.cuda.nvtx.range_pop()

    # tail forward
    f_input = layer_pre_forward()
    with f_context:
        for i in range(overlaped_layers, f_num_layers):
            f_layer = f_schedule_plan.get_layer(i)
            torch.cuda.nvtx.range_push(f"layer_{i}f")
            f_input, tmp = schedule_layer_1f1b(f_layer, None, f_input=f_input)
            torch.cuda.nvtx.range_pop()

        if f_schedule_plan is not None and f_schedule_plan.post_process is not None:
            f_input = f_schedule_plan.post_process.forward(f_input)

    # output pp send receive
    if f_schedule_plan is not None and post_forward is not None:
        with f_context:
            f_schedule_plan.wait_current_stream()
            post_forward(f_input)

    # tail backward
    grad = layer_pre_backward()
    with b_context:
        for i in range(overlaped_layers, b_num_layers):
            b_layer = b_schedule_plan.get_layer(b_num_layers - 1 - i)
            torch.cuda.nvtx.range_push(f"layer_{b_num_layers - 1 - i}b")
            tmp, grad = schedule_layer_1f1b(None, b_layer, b_grad=grad)
            torch.cuda.nvtx.range_pop()

        if b_schedule_plan is not None:
            b_schedule_plan.pre_process.backward(grad)

    # grad send receive
    if b_schedule_plan is not None and post_backward is not None:
        with b_context:
            b_schedule_plan.wait_current_stream()
            post_backward(grad)

    if f_schedule_plan:
        f_schedule_plan.wait_current_stream()
    if b_schedule_plan:
        b_schedule_plan.wait_current_stream()

    return f_input


def build_model_chunk_schedule_plan(
    model,
    input_ids: Tensor,
    position_ids: Tensor,
    attention_mask: Tensor,
    decoder_input: Tensor = None,
    labels: Tensor = None,
    inference_params=None,
    packed_seq_params=None,
    extra_block_kwargs=None,
    runtime_gather_output: Optional[bool] = None,
):

    comp_stream = get_comp_stream()
    com_stream = get_com_stream()
    model_chunk_schedule_plan = ModelChunkSchedulePlan()
    event = model_chunk_schedule_plan.event
    state = model_chunk_schedule_plan.state
    # save for later use
    state.input_ids = input_ids
    state.position_ids = position_ids
    state.attention_mask = attention_mask
    state.decoder_input = decoder_input
    state.labels = labels
    state.inference_params = inference_params
    state.packed_seq_params = packed_seq_params
    state.extra_block_kwargs = extra_block_kwargs
    state.runtime_gather_output = runtime_gather_output
    state.context = None
    state.context_mask = None
    state.attention_bias = None

    # build preprocess
    model_chunk_schedule_plan.pre_process = PreProcessNode(model, state, event, comp_stream)
    model_chunk_schedule_plan.pre_process.name = "pre_process"
    # build for layers
    for layer_idx in range(model.decoder.num_layers_per_pipeline_rank):
        layer = model.decoder._get_layer(layer_idx)
        layer_plan = build_layer_schedule_plan(layer, event, state, comp_stream, com_stream)
        model_chunk_schedule_plan.add_layer(layer_plan)
    # build post process
    if model.post_process:

        model_chunk_schedule_plan.post_process = PostProcessNode(model, state, event, comp_stream)
        model_chunk_schedule_plan.post_process.name = "post_process"

    return model_chunk_schedule_plan
