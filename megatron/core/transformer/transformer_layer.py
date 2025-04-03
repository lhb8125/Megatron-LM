# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import warnings
from abc import ABC
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Optional, Union

import torch
import torch.distributed
from torch import Tensor

from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import apply_prefix_mapping
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.cuda_graphs import CudaGraphManager
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import SubmoduleCallables, TransformerLayerSubmoduleCallables

# from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllPerBatchState, per_batch_state_context
from megatron.core.utils import deprecate_inference_params, make_viewless_tensor


def get_transformer_layer_offset(config: TransformerConfig):
    """Get the index offset of current pipeline stage, given the level of pipelining."""
    pipeline_rank = parallel_state.get_pipeline_model_parallel_rank()
    if not parallel_state.is_inside_encoder():
        pp_decoder_start = parallel_state.get_pipeline_model_parallel_decoder_start()
        if pp_decoder_start is not None:
            pipeline_rank = pipeline_rank - pp_decoder_start

    if config.pipeline_model_parallel_size > 1:

        if (
            config.num_layers_in_first_pipeline_stage is not None
            or config.num_layers_in_last_pipeline_stage is not None
        ):
            # Calculate number of pipeline stages to distribute the remaining Transformer
            # layers after deducting the Transformer layers in the first or the last stages
            middle_pipeline_stages = config.pipeline_model_parallel_size
            middle_pipeline_stages -= sum(
                [
                    1 if x is not None else 0
                    for x in (
                        config.num_layers_in_first_pipeline_stage,
                        config.num_layers_in_last_pipeline_stage,
                    )
                ]
            )

            # Calculate layers to distribute in each pipeline stage. If the
            # num_layers_in_first_pipeline_stage and num_layers_in_last_pipeline_stage
            # are not set, we will not enable uneven pipeline. All layers will be treated
            # as middle layers.
            num_layers_in_first_pipeline_stage = (
                0
                if config.num_layers_in_first_pipeline_stage is None
                else config.num_layers_in_first_pipeline_stage
            )
            num_layers_in_last_pipeline_stage = (
                0
                if config.num_layers_in_last_pipeline_stage is None
                else config.num_layers_in_last_pipeline_stage
            )

            middle_num_layers = (
                config.num_layers
                - num_layers_in_first_pipeline_stage
                - num_layers_in_last_pipeline_stage
            )

            if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
                vp_rank = parallel_state.get_virtual_pipeline_model_parallel_rank()
                vp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()

                # Calculate number of layers in each virtual model chunk
                # If the num_layers_in_first_pipeline_stage and
                # num_layers_in_last_pipeline_stage are not set, all pipeline stages
                # will be treated as middle pipeline stages in the calculation
                num_layers_per_virtual_model_chunk_in_first_pipeline_stage = (
                    0
                    if config.num_layers_in_first_pipeline_stage is None
                    else config.num_layers_in_first_pipeline_stage // vp_size
                )

                num_layers_per_virtual_model_chunk_in_last_pipeline_stage = (
                    0
                    if config.num_layers_in_last_pipeline_stage is None
                    else config.num_layers_in_last_pipeline_stage // vp_size
                )

                num_layers_per_vritual_model_chunk_in_middle_pipeline_stage = (
                    middle_num_layers // vp_size
                )

                # First stage + middle stage + last stage
                total_virtual_chunks = (
                    num_layers_per_virtual_model_chunk_in_first_pipeline_stage
                    + num_layers_per_vritual_model_chunk_in_middle_pipeline_stage
                    + num_layers_per_virtual_model_chunk_in_last_pipeline_stage
                )

                # Calculate the layer offset with interleaved uneven pipeline parallelism
                if pipeline_rank == 0:
                    offset = vp_rank * total_virtual_chunks
                else:
                    offset = (
                        vp_rank * total_virtual_chunks
                        + num_layers_per_virtual_model_chunk_in_first_pipeline_stage
                        + (pipeline_rank - 1)
                        * (
                            num_layers_per_vritual_model_chunk_in_middle_pipeline_stage
                            // middle_pipeline_stages
                        )
                    )
            else:
                if middle_pipeline_stages > 0:
                    num_layers_per_pipeline_rank = middle_num_layers // middle_pipeline_stages
                else:
                    num_layers_per_pipeline_rank = 0

                middle_pipeline_rank = (
                    pipeline_rank
                    if config.num_layers_in_first_pipeline_stage is None
                    else pipeline_rank - 1
                )

                if pipeline_rank == 0:
                    offset = 0
                else:
                    offset = (
                        middle_pipeline_rank * num_layers_per_pipeline_rank
                    ) + num_layers_in_first_pipeline_stage
        else:
            num_layers = config.num_layers

            # Increase the number of layers by one if we include the embedding (loss)
            # layer into pipeline parallelism partition and placement
            if config.account_for_embedding_in_pipeline_split:
                num_layers += 1

            if config.account_for_loss_in_pipeline_split:
                num_layers += 1

            num_layers_per_pipeline_rank = num_layers // config.pipeline_model_parallel_size

            if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
                vp_rank = parallel_state.get_virtual_pipeline_model_parallel_rank()
                vp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()

                num_layers_per_virtual_rank = num_layers_per_pipeline_rank // vp_size
                total_virtual_chunks = num_layers // vp_size
                offset = vp_rank * total_virtual_chunks + (
                    pipeline_rank * num_layers_per_virtual_rank
                )

                # Reduce the offset of embedding layer from the total layer number
                if (
                    config.account_for_embedding_in_pipeline_split
                    and not parallel_state.is_pipeline_first_stage()
                ):
                    offset -= 1
            else:
                offset = pipeline_rank * num_layers_per_pipeline_rank

                # Reduce the offset of embedding layer from the total layer number
                if (
                    config.account_for_embedding_in_pipeline_split
                    and not parallel_state.is_pipeline_first_stage()
                ):
                    offset -= 1
    else:
        offset = 0
    return offset


@dataclass
class TransformerLayerSubmodules:
    """
    Configuration class for specifying the submodules of a transformer layer.

    This class defines the structure and default implementations for various
    components of a transformer layer, allowing for flexible customization
    of the layer's architecture.

    Args:
        input_layernorm (Union[ModuleSpec, type]): Specification for the input layer normalization.
        self_attention (Union[ModuleSpec, type]): Specification for the self-attention mechanism.
        self_attn_bda (Union[ModuleSpec, type]): Specification for the bias-dropout-add operation
            after self-attention.
        pre_cross_attn_layernorm (Union[ModuleSpec, type]): Specification for the layer
            normalization before cross-attention.
        cross_attention (Union[ModuleSpec, type]): Specification for the cross-attention mechanism.
        cross_attn_bda (Union[ModuleSpec, type]): Specification for the bias-dropout-add operation
            after cross-attention.
        pre_mlp_layernorm (Union[ModuleSpec, type]): Specification for the layer normalization
            before the MLP.
        mlp (Union[ModuleSpec, type]): Specification for the MLP in Dense layer.
        mlp_bda (Union[ModuleSpec, type]): Specification for the bias-dropout-add operation
            after the MLP.
        sharded_state_dict_keys_map (Dict[str, str]): Mapping for sharded tensor keys to be applied
            in the `sharded_state_dict` method.
    """

    input_layernorm: Union[ModuleSpec, type] = IdentityOp
    self_attention: Union[ModuleSpec, type] = IdentityOp
    self_attn_bda: Union[ModuleSpec, type] = IdentityFuncOp

    pre_cross_attn_layernorm: Union[ModuleSpec, type] = IdentityOp
    cross_attention: Union[ModuleSpec, type] = IdentityOp
    cross_attn_bda: Union[ModuleSpec, type] = IdentityFuncOp

    pre_mlp_layernorm: Union[ModuleSpec, type] = IdentityOp
    mlp: Union[ModuleSpec, type] = IdentityOp
    mlp_bda: Union[ModuleSpec, type] = IdentityFuncOp

    # Mapping for sharded tensor keys to be applied in `sharded_state_dict` method
    sharded_state_dict_keys_map: Dict[str, str] = field(default_factory=dict)


class BaseTransformerLayer(ABC):
    """A common parent class for `TransformerLayer` like implementations.

    A dummy class that is subclassed by similar `TransformerLayer`s e.g. the
    `TransformerLayer` in this file and possibly other `TransformerLayer`
    implementations that aim to use `TransformerBlock` as the base module.
    The main purpose is to check if any layer (or module) provided in the spec
    is a subclass of this class to allow fanning-out of that spec for all the
    layers in the `TransformerBlock`. See `_get_block_submodules` method
    implementation in `transformer_block.py` file for more details.
    """

    def __init__(self):
        pass


class TransformerLayer(MegatronModule, BaseTransformerLayer):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: float = None,
    ):
        super().__init__(config=config)

        # Enable cuda graphs.
        if config.enable_cuda_graph:
            self.cudagraph_manager = CudaGraphManager(config)

        self.submodules_config = submodules
        self.layer_number = layer_number + get_transformer_layer_offset(self.config)
        self.hidden_dropout = config.hidden_dropout if hidden_dropout is None else hidden_dropout

        # [Module 1: Input Layernorm] Optional Layernorm on the input data
        # TODO: add pytorch only layernorm
        self.input_layernorm = build_module(
            submodules.input_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        attention_optional_kwargs = {}
        if config.context_parallel_size > 1 and config.cp_comm_type is not None:
            if isinstance(config.cp_comm_type, list):
                attention_optional_kwargs["cp_comm_type"] = config.cp_comm_type[self.layer_number]
            else:
                attention_optional_kwargs["cp_comm_type"] = config.cp_comm_type

        # [Module 2: SelfAttention]
        self.self_attention = build_module(
            submodules.self_attention,
            config=self.config,
            layer_number=layer_number,
            **attention_optional_kwargs,
        )

        # [Module 3: BiasDropoutFusion]
        self.self_attn_bda = build_module(submodules.self_attn_bda)

        # [Module 4: Post SelfAttention] Optional Layernorm after self-attn
        self.pre_cross_attn_layernorm = build_module(
            submodules.pre_cross_attn_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        # [Module 5: CrossAttention]
        self.cross_attention = build_module(
            submodules.cross_attention,
            config=self.config,
            layer_number=layer_number,
            **attention_optional_kwargs,
        )

        # [Module 6: BiasDropoutFusion]
        self.cross_attn_bda = build_module(submodules.cross_attn_bda, config=self.config)

        # [Module 7: Pre MLP] Optional Layernorm before MLP
        self.pre_mlp_layernorm = build_module(
            submodules.pre_mlp_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        # [Module 8: MLP block]
        self.mlp = build_module(submodules.mlp, config=self.config)
        if hasattr(self.mlp, 'set_layer_number'):
            self.mlp.set_layer_number(self.layer_number)

        # [Module 9: BiasDropoutFusion]
        self.mlp_bda = build_module(submodules.mlp_bda)

        # @jcasper how should we handle nvfuser?
        # Set bias+dropout+add fusion grad_enable execution handler.
        # TORCH_MAJOR = int(torch.__version__.split('.')[0])
        # TORCH_MINOR = int(torch.__version__.split('.')[1])
        # use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        # self.bias_dropout_add_exec_handler = nullcontext if use_nvfuser else torch.enable_grad
        self.bias_dropout_add_exec_handler = torch.enable_grad

    @staticmethod
    def _get_layer_offset(config: TransformerConfig):
        """
        Get the layer offset for the current pipeline stage.

        Deprecated: please use `get_transformer_layer_offset` instead.
        """

        warnings.warn(
            "TransformerLayer._get_layer_offset is deprecated."
            "Please use get_transformer_layer_offset instead."
        )
        return get_transformer_layer_offset(config)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ):
        """
        Perform a forward pass through the transformer layer.

        This method implements the core computation of a transformer layer, including
        self-attention, cross-attention (if applicable), and feed-forward operations.

        Args:
            hidden_states (Tensor): Input tensor of shape [s, b, h] where s is sequence length,
                b is batch size, and h is hidden size.
            attention_mask (Tensor): Mask tensor for self-attention.
            context (Tensor, optional): Context tensor for cross-attention.
            context_mask (Tensor, optional): Mask tensor for cross-attention.
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
            attention_bias (Tensor, optional): Bias tensor for Q * K.T.
            inference_context (object, optional): Parameters for inference-time optimizations.
            packed_seq_params (object, optional): Parameters for packed sequence processing.
            sequence_len_offset (Tensor, optional): Offset along sequence dimension
                during inference.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing:
                output (Tensor): Transformed hidden states of shape [s, b, h].
                context (Tensor): Updated context tensor if cross-attention is used,
                otherwise None.
        """

        inference_context = deprecate_inference_params(inference_context, inference_params)

        # Residual connection.
        residual = hidden_states

        # Optional Input Layer norm
        input_layernorm_output = self.input_layernorm(hidden_states)

        # Self attention.
        attention_output_with_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm after self-attention
        pre_cross_attn_layernorm_output = self.pre_cross_attn_layernorm(hidden_states)

        # Cross attention.
        attention_output_with_bias = self.cross_attention(
            pre_cross_attn_layernorm_output,
            attention_mask=context_mask,
            key_value_states=context,
            inference_context=inference_context,
        )

        if isinstance(attention_output_with_bias, dict) and "context" in attention_output_with_bias:
            context = attention_output_with_bias["context"]

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.cross_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm post the cross-attention.
        pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)

        # MLP.
        mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, self.hidden_dropout
            )

        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        output = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )

        # CUDA graph requires returned values to be Tensors
        if self.config.external_cuda_graph and self.training:
            return output

        return output, context

    def _submodule_attn_forward(
        self, 
        hidden_states,
        attention_mask=None,
        inference_params=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        state=None,
    ):
        # Residual connection.
        residual = hidden_states

        # Optional Input Layer norm
        input_layernorm_output = self.input_layernorm(hidden_states)

        # Self attention.
        attention_output_with_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )
        return hidden_states

    def _submodule_attn_router_forward(
            self, 
            hidden_states,
            attention_mask=None,
            inference_params=None,
            rotary_pos_emb=None,
            rotary_pos_cos=None,
            rotary_pos_sin=None,
            attention_bias=None,
            packed_seq_params=None,
            sequence_len_offset=None,
            state=None,
        ):
        """
        Performs a combined forward pass that includes self-attention and MLP routing logic.
        """
        hidden_states = self._submodule_attn_forward(
            hidden_states,
            attention_mask,
            inference_params,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            attention_bias,
            packed_seq_params,
            sequence_len_offset)

        pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)
        probs, routing_map = self.mlp.router(pre_mlp_layernorm_output)
        tokens_per_expert = self.mlp.token_dispatcher.meta_prepare(pre_mlp_layernorm_output, probs, routing_map)
        permutated_local_input_tokens = self.mlp.token_dispatcher.dispatch_preprocess(pre_mlp_layernorm_output, routing_map)

        return hidden_states, pre_mlp_layernorm_output, tokens_per_expert, permutated_local_input_tokens
    
    def _submodule_dispatch_forward(self, tokens, state=None):
        """
        Dispatches tokens to the appropriate experts based on the router output.
        """
        output_tokens = self.mlp.token_dispatcher.dispatch_all_to_all(tokens)
        return output_tokens

    def _submodule_dense_forward(self, hidden_states, state=None):
        residual = hidden_states
        pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)
        mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, self.hidden_dropout
            )
        output = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )

        return output
    
    def _submodule_moe_forward(self, dispatched_input, state=None):
        """
        Performs a forward pass for the MLP submodule, including both expert-based
        and optional shared-expert computations.
        """
        hidden_states, tokens_per_expert = state.pre_mlp_layernorm_output, state.tokens_per_expert
        shared_expert_output = None
        dispatched_input = self.mlp.token_dispatcher.dispatch_postprocess(dispatched_input)
        expert_output, mlp_bias = self.mlp.experts(dispatched_input, tokens_per_expert)
        expert_output = self.mlp.token_dispatcher.combine_preprocess(expert_output)
        if self.mlp.use_shared_expert and not self.mlp.shared_expert_overlap:
            shared_expert_output = self.mlp.shared_experts(hidden_states)
        return expert_output, shared_expert_output, mlp_bias

    def _submodule_combine_forward(self, output, shared_expert_output, state=None):
        residual = state.residual
        output = self.mlp.token_dispatcher.combine_all_to_all(output)
        output = self.mlp.token_dispatcher.combine_postprocess(output)
        if shared_expert_output is not None:
            output = output + shared_expert_output
        mlp_output_with_bias = (output, None)
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, self.hidden_dropout
            )
        output = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )

        return output

    def _submodule_attn_router_dw(self):
        self.self_attention.backward_dw()

    def _submodule_mlp_dw(self):
        self.mlp.backward_dw()

    def _submodule_attn_router_postprocess(self, node, residual, pre_mlp_layernorm_output, tokens_per_expert, local_tokens):
        node.common_state.tokens_per_expert = tokens_per_expert
        node.common_state.probs = node.detach(node.common_state.probs)
        node.common_state.residual = node.detach(residual)
        node.common_state.pre_mlp_layernorm_output = node.detach(pre_mlp_layernorm_output)
        return local_tokens
    
    def _submodule_dispatch_postprocess(self, node, dispatched_input):
        return dispatched_input, 
    
    def _submodule_mlp_postprocess(self, node, expert_output, shared_expert_output, mlp_bias):
        assert mlp_bias is None
        node.common_state.probs = node.detach(node.common_state.probs)
        node.common_state.pre_mlp_layernorm_output = None
        return expert_output, shared_expert_output
    
    def _submodule_combine_postprocess(self, node, output):
        cur_stream = torch.cuda.current_stream()
        node.common_state.residual.record_stream(cur_stream)
        node.common_state.probs.record_stream(cur_stream)
        node.common_state.residual = None
        node.common_state.probs = None
        return output
    
    def _submodule_non_moe_postprocess(self, node, hidden_states):
        return hidden_states
    
    def _submodule_not_implemented(self, *args):
        raise NotImplementedError("This callable is not implemented.")

    def get_submodule_callables(self, chunk_state):
        """
        The forward callables take 2 parts of inputs:
        1. The ScheduleNode object.
        2. The input tensors.
        """
        from megatron.core.transformer.moe.moe_layer import MoELayer
        is_moe = isinstance(self.mlp, MoELayer)
        def get_func_with_default(func, default_func):
            if is_moe:
                return func
            return default_func
        
        def callable_wrapper(forward_func, postprocess_func, node, *args):
            with node.forward_ctx():
                state = getattr(node, 'common_state', None)
                callable_outputs = forward_func(*args, state=state)
            if isinstance(callable_outputs, tuple):
                outputs = postprocess_func(node, *callable_outputs)
            else:
                outputs = postprocess_func(node, callable_outputs)
            return outputs

        attn_func = get_func_with_default(self._submodule_attn_router_forward, self._submodule_attn_forward)
        def attn_wrapper(hidden_states, state=None):
            return attn_func(
                hidden_states, 
                attention_mask = chunk_state.attention_mask,
                attention_bias = chunk_state.attention_bias,
                inference_params = chunk_state.inference_params,
                packed_seq_params = chunk_state.packed_seq_params,
                sequence_len_offset = chunk_state.sequence_len_offset,
                rotary_pos_emb = chunk_state.rotary_pos_emb,
                rotary_pos_cos = chunk_state.rotary_pos_cos,
                rotary_pos_sin = chunk_state.rotary_pos_sin,
                state = state
            )
        attn_postprocess_func = get_func_with_default(self._submodule_attn_router_postprocess, self._submodule_non_moe_postprocess)
        
        dispatch_func = get_func_with_default(self._submodule_dispatch_forward, self._submodule_not_implemented)
        dispatch_postprocess_func = get_func_with_default(self._submodule_dispatch_postprocess, self._submodule_non_moe_postprocess)
        
        mlp_func = get_func_with_default(self._submodule_moe_forward, self._submodule_dense_forward)
        mlp_postprocess_func = get_func_with_default(self._submodule_mlp_postprocess, self._submodule_non_moe_postprocess)
        
        combine_func = get_func_with_default(self._submodule_combine_forward, self._submodule_not_implemented)
        combine_postprocess_func = get_func_with_default(self._submodule_combine_postprocess, self._submodule_non_moe_postprocess)
        
        attn_forward = partial(callable_wrapper, attn_wrapper, attn_postprocess_func)
        dispatch_forward = partial(callable_wrapper, dispatch_func, dispatch_postprocess_func)
        mlp_forward = partial(callable_wrapper, mlp_func, mlp_postprocess_func)
        combine_forward = partial(callable_wrapper, combine_func, combine_postprocess_func)

        callables = TransformerLayerSubmoduleCallables(
            attention=SubmoduleCallables(
                forward=attn_forward,
                dw=self._submodule_attn_router_dw,
            ),
            dispatch=SubmoduleCallables(forward=dispatch_forward),
            mlp=SubmoduleCallables(
                forward=mlp_forward,
                dw=self._submodule_mlp_dw,
            ),
            combine=SubmoduleCallables(forward=combine_forward),
        )
        return callables

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """
        Generate a sharded state dictionary for the transformer layer.

        Args:
            prefix (str, optional): Prefix to be added to all keys in the state dict.
            sharded_offsets (tuple, optional): Tuple of sharding offsets.
            metadata (Optional[dict], optional): Additional metadata for sharding.

        Returns:
            ShardedStateDict: A dictionary containing the sharded state of the transformer layer.
        """
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        prefixed_map = {
            f'{prefix}{k}': f'{prefix}{v}'
            for k, v in self.submodules_config.sharded_state_dict_keys_map.items()
        }
        if prefixed_map:
            apply_prefix_mapping(sharded_state_dict, prefixed_map)
        return sharded_state_dict

    def __call__(self, *args, **kwargs):
        # Training and validation mode CUDA graphs
        if hasattr(self, 'cudagraph_manager') and kwargs.get('inference_context') is None:
            return self.cudagraph_manager(self, args, kwargs)
        # Inference mode. CUDA graphs are used in the decode phase only, when attn mask is None
        elif not self.training and (
            hasattr(self, 'cudagraph_manager')
            and kwargs['attention_mask'] is None
            and (
                kwargs.get('inference_context') is not None
                and kwargs['inference_context'].is_decode_only()
                or kwargs.get('inference_params') is not None
                and kwargs['inference_params'].is_decode_only()
            )
        ):
            assert (
                kwargs.get('attention_mask') is None
            ), f"Attention mask must not be set when using CUDA graphs for decode"
            return self.cudagraph_manager(self, args, kwargs)
        return super(MegatronModule, self).__call__(*args, **kwargs)
