# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain GPT."""

import logging
import os
import sys
from megatron.training import get_args
from megatron.training.initialize import initialize_megatron

from typing import Callable, Union, Any, Tuple, Optional

import torch
from torch import Tensor
from megatron.core.transformer import transformer_layer
from megatron.core.pipeline_parallel.combined_1f1b import ScheduleNode, StreamRelease, StreamAcquire
from megatron.training.arguments import core_transformer_config_from_args
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllPerBatchState
from functools import partial
from megatron.core.models.gpt import GPTModel
from megatron.core.transformer.module import Float16Module
from megatron.training.utils import unwrap_model
import weakref


class TransformerLayerState(MoEAlltoAllPerBatchState):
    pass

class ModelChunkSate:
    pass


class TransformerSchedulePlan:

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


class PreProcessNode(ScheduleNode):

    def __init__(self, gpt_model, model_chunk_state, event, stream):
        super().__init__(self.forward_impl, stream, event)
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
        super().__init__(self.forward_impl, stream, event)
        self.gpt_model = gpt_model
        self.model_chunk_state = model_chunk_state

    def forward_impl(self, hidden_states):
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
            return logits.transpose(0, 1).contiguous()
        loss = gpt_model.compute_language_model_loss(labels, logits)
        return loss


class TransformerNode(ScheduleNode):

    def __init__(self, common_state, layer, stream, event):
        super().__init__(self.forward_impl, stream, event)
        self.common_state = common_state
        self.layer = layer


class AttnNode(TransformerNode):

    def forward_impl(self, hidden_states):
        attention_mask = None
        context = None
        context_mask = None
        rotary_pos_emb = None
        rotary_pos_cos = None
        rotary_pos_sin = None
        attention_bias = None
        inference_params = None
        packed_seq_params = None

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


class DispatchNode(TransformerNode):

    def forward_impl(
        self, residual, pre_mlp_layernorm_output, permutated_local_input_tokens, probs
    ):

        self.common_state.probs = probs
        token_dispatcher = self.layer.mlp.token_dispatcher
        with token_dispatcher.per_batch_state_context(self.common_state):
            dispatched_input = token_dispatcher.dispatch_all_to_all(permutated_local_input_tokens)
        return residual, pre_mlp_layernorm_output, dispatched_input, probs


class MlPNode(TransformerNode):
    def forward_impl(self, residual, pre_mlp_layernorm_output, dispatched_input, probs):
        self.common_state.probs = probs
        token_dispatcher = self.layer.mlp.token_dispatcher
        with token_dispatcher.per_batch_state_context(self.common_state):
            dispatched_input = token_dispatcher.dispatch_postprocess(dispatched_input)
            expert_output, mlp_bias = self.layer.mlp.experts(
                dispatched_input, self.common_state.tokens_per_expert
            )
            assert mlp_bias is None
            permutated_local_input_tokens = token_dispatcher.combine_preprocess(expert_output)
        shared_output = self.layer.mlp.shared_experts(pre_mlp_layernorm_output)
        return residual, permutated_local_input_tokens, shared_output, probs


class CombineNode(TransformerNode):
    def forward_impl(self, residual, permutated_local_input_tokens, shared_output, probs):
        token_dispatcher = self.layer.mlp.token_dispatcher
        self.common_state.probs = probs
        with token_dispatcher.per_batch_state_context(self.common_state):
            permutated_local_input_tokens = token_dispatcher.combine_all_to_all(
                permutated_local_input_tokens
            )
        return residual, permutated_local_input_tokens, shared_output, probs


class CombinePostProcessNode(TransformerNode):
    def forward_impl(self, residual, permutated_local_input_tokens, shared_output, probs):
        token_dispatcher = self.layer.mlp.token_dispatcher
        self.common_state.probs = probs
        with token_dispatcher.per_batch_state_context(self.common_state):
            output = token_dispatcher.combine_postprocess(permutated_local_input_tokens)
        output = output.type_as(residual)
        output += shared_output
        mlp_output_with_bias = (output, None)
        with self.layer.bias_dropout_add_exec_handler():
            hidden_states = self.layer.mlp_bda(
                self.layer.training, self.layer.config.bias_dropout_fusion
            )(mlp_output_with_bias, residual, self.layer.hidden_dropout)
        output = transformer_layer.make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )
        return output


def build_layer_schedule_plan(layer, event, comp_stream, com_stream):
    common_state = TransformerLayerState()
    attn = AttnNode(common_state, layer, comp_stream, event)
    attn.name = "attn"
    dispatch = DispatchNode(common_state, layer, com_stream, event)
    dispatch.name = "dispatch"
    mlp = MlPNode(common_state, layer, comp_stream, event)
    mlp.name = "mlp"
    combine = CombineNode(common_state, layer, com_stream, event)
    combine.name = "combine"
    post_combine = CombinePostProcessNode(common_state, layer, comp_stream, event)
    post_combine.name = "post_combine"
    return TransformerSchedulePlan(attn, dispatch, mlp, combine, post_combine)


def build_model_chunk_schedule_plan(
    model,
    comp_stream,
    com_stream,
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

    # build preprocess
    model_chunk_schedule_plan.pre_process = PreProcessNode(
        model, state, event, comp_stream
    )
    model_chunk_schedule_plan.pre_process.name = "pre_process"
    # build for layers
    for layer_idx in range(model.decoder.num_layers_per_pipeline_rank):
        layer = model.decoder._get_layer(layer_idx)
        layer_plan = build_layer_schedule_plan(
            layer, event, comp_stream, com_stream
        )
        model_chunk_schedule_plan.add_layer(layer_plan)
    # build post process
    if model.post_process:

        model_chunk_schedule_plan.post_process = PostProcessNode(
            model, state, event, comp_stream
        )
        model_chunk_schedule_plan.post_process.name = "post_process"
        
    return model_chunk_schedule_plan


def schedule_layer_1f1b(f_layer, b_layer, f_input, b_grad):
    if b_layer is not None:
        b_grad = b_layer.post_combine.backward(b_grad)
    if f_layer is not None:
        f_input = f_layer.attn.forward(f_input)
    if b_layer is not None:
        b_grad = b_layer.combine.backward(b_grad)
    if f_layer is not None:
        f_input = f_layer.dispatch.forward(f_input)
    if b_layer is not None:
        b_grad = b_layer.mlp.backward(b_grad)
    if f_layer is not None:
        f_input = f_layer.mlp.forward(f_input)
    if b_layer is not None:
        b_grad = b_layer.dispatch.backward(b_grad)
    if f_layer is not None:
        f_input = f_layer.combine.forward(f_input)
    if b_layer is not None:
        b_grad = b_layer.attn.backward(b_grad)
    if f_layer is not None:
        f_input = f_layer.post_combine.forward(f_input)
    return f_input, b_grad


def schedule_chunk_forward(schedule_plan):
    f_input = schedule_chunk_1f1b(schedule_plan, None, None)
    return f_input


def schedule_chunk_backward(schedule_plan, grad):
    tmp = schedule_chunk_1f1b(None, schedule_plan, grad)

def schedule_chunk_1f1b(f_schedule_plan, b_schedule_plan, grad):
    f_input = None
    if f_schedule_plan is not None:
        f_input = f_schedule_plan.pre_process.forward()
    if b_schedule_plan is not None and b_schedule_plan.post_process is not None:
        grad = b_schedule_plan.post_process.backward(grad)

    f_num_layers = f_schedule_plan.num_layers() if f_schedule_plan is not None else 0
    b_num_layers = b_schedule_plan.num_layers() if b_schedule_plan is not None else 0
    overlaped_layers = min(f_num_layers, b_num_layers)

    for i in range(overlaped_layers):
        f_layer = f_schedule_plan.get_layer(i)
        b_layer = b_schedule_plan.get_layer(b_num_layers - 1 - i)
        torch.cuda.nvtx.range_push(f"layer_{i}f-layer_{b_num_layers - 1 - i}b")
        f_input, grad = schedule_layer_1f1b(f_layer, b_layer, f_input, grad)
        torch.cuda.nvtx.range_pop()

    for i in range(overlaped_layers, b_num_layers):
        b_layer = b_schedule_plan.get_layer(b_num_layers - 1 - i)
        torch.cuda.nvtx.range_push(f"layer_{b_num_layers - 1 - i}b")
        tmp, grad = schedule_layer_1f1b(None, b_layer, None, grad)
        torch.cuda.nvtx.range_pop()

    if b_schedule_plan is not None:
        b_schedule_plan.pre_process.backward(grad)

    for i in range(overlaped_layers, f_num_layers):
        f_layer = f_schedule_plan.get_layer(i)
        torch.cuda.nvtx.range_push(f"layer_{i}f")
        f_input, tmp = schedule_layer_1f1b(f_layer, None, f_input, None)
        torch.cuda.nvtx.range_pop()

    if f_schedule_plan is not None and f_schedule_plan.post_process is not None:
        f_input = f_schedule_plan.post_process.forward(f_input)

    return  f_input



def schedule_1f1b_overlap(datas, comp_stream, com_stream, model):
    l = len(datas)
    # first f
    pre_stream = torch.cuda.current_stream()

    build_plan_func = partial(
        build_model_chunk_schedule_plan, model, comp_stream, com_stream, None, None, None
    )
    pre_schedule_plan = build_plan_func(decoder_input=datas[0])
    print("schedule_chunk_forward")
    torch.cuda.nvtx.range_push(f"forward schudule")
    pre_output = schedule_chunk_forward(pre_schedule_plan)
    torch.cuda.nvtx.range_pop()
    # 1f1b
    for i in range(1, l):
        print("schedule_chunk_1f1b")
        grad = torch.ones_like(pre_output)
        grad = StreamRelease.apply(pre_schedule_plan.event, pre_stream, grad)
        schedule_plan = build_plan_func(decoder_input=datas[i])
        torch.cuda.nvtx.range_push(f"1f1b schudule")
        pre_output = schedule_chunk_1f1b(schedule_plan, pre_schedule_plan, grad)
        pre_schedule_plan = schedule_plan
        torch.cuda.nvtx.range_pop()

    # last b
    grad = torch.ones_like(pre_output)
    grad = StreamRelease.apply(pre_schedule_plan.event, pre_stream, grad)
    torch.cuda.nvtx.range_push(f"backward schudule")
    print("schedule_chunk_backward")
    schedule_chunk_backward(pre_schedule_plan, grad)
    torch.cuda.nvtx.range_pop()
    pre_schedule_plan.event.wait(torch.cuda.current_stream())
    torch.cuda.synchronize()
    print("finish")


def set_deterministic():
    torch.use_deterministic_algorithms(True)


def build_data(args):
    s = args.seq_length
    if args.sequence_parallel:
        s = s // args.tensor_model_parallel_size
    b = 1
    h = args.hidden_size

    hidden_states = torch.randn(*(s, b, h), dtype=torch.bfloat16, device="cuda") * h
    hidden_states.requires_grad = True
    return hidden_states


def build_gpt_model(args):
    config = core_transformer_config_from_args(args)
    extra_args = {}
    import inspect

    signature = inspect.signature(get_gpt_layer_with_transformer_engine_spec)
    if "multi_latent_attention" in signature.parameters:
        extra_args["multi_latent_attention"] = args.multi_latent_attention

    if "qk_layernorm" in signature.parameters:
        extra_args["qk_layernorm"] = args.qk_layernorm

    if "moe_use_legacy_grouped_gemm" in signature.parameters:
        extra_args["moe_use_legacy_grouped_gemm"] = True

    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
        args.num_experts,
        args.moe_grouped_gemm,
        **extra_args,
    )


    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=True,
        post_process=True,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rope_theta,
    )

    if args.fp16 or args.bf16:
        model = Float16Module(config, model)
    model = unwrap_model(model, (Float16Module,))
    return model


def test_1f1b_overlap(args):
    model = build_gpt_model(args)
    datas = [build_data(args) for _ in range(16)]
    com_stream = torch.cuda.Stream(device="cuda")
    comp_stream = torch.cuda.Stream(device="cuda")  # torch.cuda.current_stream()
    schedule_1f1b_overlap(datas, comp_stream, com_stream, model)

def main():
    initialize_megatron()
    args = get_args()
    torch.cuda.cudart().cudaProfilerStart()
    test_1f1b_overlap(args)
    torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
