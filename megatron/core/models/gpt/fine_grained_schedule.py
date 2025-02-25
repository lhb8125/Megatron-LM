import weakref
from typing import Any, Callable, Optional, Tuple, Union

from torch import Tensor

from megatron.core.pipeline_parallel.combined_1f1b import (
    ModelChunkSchedulePlan,
    ScheduleNode,
    TransformerLayerSchedulePlan,
    TransformerLayerState,
    get_com_stream,
    get_comp_stream,
)
from megatron.core.transformer import transformer_layer


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
        super().__init__(weak_method(self.forward_impl), stream, event)
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

    # build preprocess
    model_chunk_schedule_plan.pre_process = PreProcessNode(model, state, event, comp_stream)
    model_chunk_schedule_plan.pre_process.name = "pre_process"
    # build for layers
    for layer_idx in range(model.decoder.num_layers_per_pipeline_rank):
        layer = model.decoder._get_layer(layer_idx)
        layer_plan = build_layer_schedule_plan(layer, event, comp_stream, com_stream)
        model_chunk_schedule_plan.add_layer(layer_plan)
    # build post process
    if model.post_process:

        model_chunk_schedule_plan.post_process = PostProcessNode(model, state, event, comp_stream)
        model_chunk_schedule_plan.post_process.name = "post_process"

    return model_chunk_schedule_plan
