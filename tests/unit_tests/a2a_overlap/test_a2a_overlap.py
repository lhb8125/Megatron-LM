# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain GPT."""

import gc
import logging
import os
import sys
import weakref
from functools import partial
from typing import Any, Callable, Optional, Tuple, Union

import torch
from torch import Tensor

from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.transformer import transformer_layer
from megatron.core.transformer.module import Float16Module
from megatron.training import get_args
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.initialize import initialize_megatron
from megatron.training.utils import unwrap_model

def build_data(args):
    s = args.seq_length
    if args.sequence_parallel:
        s = s // args.tensor_model_parallel_size
    b = 1
    h = args.hidden_size

    # Create input tensor
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
        args.num_experts, args.moe_grouped_gemm, **extra_args
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
        rotary_base=args.rotary_base,
    )

    if args.fp16 or args.bf16:
        model = Float16Module(config, model)
    model = unwrap_model(model, (Float16Module,))
    return model

def copy_weights_distributed(model_source, model_target):
    for name, param in model_source.named_parameters():
        if name in dict(model_target.named_parameters()):
            target_param = model_target.get_parameter(name)
            target_param.data.copy_(param.data)
    print("Weights have been successfully copied.")

def run_model_ref_with_capture(args, model, iterations):
    for module in model.modules():
        if hasattr(module, 'fuse_wgrad_accumulation'):
            module.fuse_wgrad_accumulation = False
    
    input_tensors = []
    output_tensors = []
    for _ in range(iterations):
        input_tensor = build_data(args)
        input_tensors.append(input_tensor)
        output = model.decoder(input_tensor, None)
        output_tensors.append(output)
        output.backward(torch.ones_like(output))
    
    capture = {
        "outputs": output_tensors,
    }
    for name, param in model.named_parameters():
        capture[name] = param.grad

    for i in range(len(input_tensors)):
        input_tensors[i].grad.zero_()
    return capture, input_tensors


def run_model_a2a_overlap_with_capture(model, input_tensors, microbatches):
    for module in model.modules():
        if hasattr(module, 'fuse_wgrad_accumulation'):
            module.fuse_wgrad_accumulation = False
    events = [torch.cuda.Event() for _ in range(microbatches)]
    
    # Create streams for computation and communication
    comp_stream = torch.cuda.Stream(device="cuda")
    comm_stream = torch.cuda.Stream(device="cuda")
    
    # Get the callables, only 1 layer in the decoder
    callables = model.decoder.get_layer_callables(0)
    
    # Initialize tensors to store intermediate results
    attention_outputs = []
    dispatch_outputs = []
    mlp_outputs = []
    combine_outputs = []
    attention_detached_outputs = []
    dispatch_detached_outputs = []
    mlp_detached_outputs = []
    
    
    # Run the first microbatch forward pass
    input_tensor = input_tensors[0]
    
    # First microbatch forward pass
    # f1.Attention + Router forward
    attention_output, detached_outputs = callables.attention.forward(
        comp_stream, events[0],
        input_tensor, None, None, None, None, None, None, None, None
    )
    hidden_states, pre_mlp_layernorm_output, probs, routing_map = detached_outputs
    attention_outputs.append(attention_output)
    attention_detached_outputs.append(detached_outputs)
    
    # f2. Token dispatch forward
    dispatch_output, detached_outputs = callables.dispatch.forward(
        comm_stream, events[0],
        pre_mlp_layernorm_output, probs, routing_map
    )
    dispatched_input, tokens_per_expert = detached_outputs
    dispatch_outputs.append(dispatch_output)
    dispatch_detached_outputs.append(detached_outputs)
    
    # f3. MLP (experts) forward
    mlp_output, detached_outputs = callables.mlp.forward(
        comp_stream, events[0],
        dispatched_input, tokens_per_expert, pre_mlp_layernorm_output, 
    )
    expert_output, shared_expert_output, mlp_bias = detached_outputs
    mlp_outputs.append(mlp_output)
    mlp_detached_outputs.append(detached_outputs)
    
    # f4. Combine outputs forward
    output, detached_outputs = callables.combine.forward(
        comm_stream, events[0],
        expert_output, shared_expert_output, mlp_bias, hidden_states
    )
    combine_outputs.append(output)
    
    # Run the overlapped 1F1B schedule for the remaining microbatches
    for i in range(1, microbatches):
        # Current microbatch input
        input_tensor = input_tensors[i]
        
        # Previous microbatch index
        prev_idx = i-1
        
        # 1F1B interleaved schedule (following the reference pattern)
        # f1. Attention forward for current microbatch
        attention_output, detached_outputs = callables.attention.forward(
            comp_stream, events[i],
            input_tensor, None, None, None, None, None, None, None, None
        )
        hidden_states, pre_mlp_layernorm_output, probs, routing_map = detached_outputs
        attention_outputs.append(attention_output)
        attention_detached_outputs.append(detached_outputs)
        
        # Gradient for previous microbatch output
        prev_output_grad = torch.ones_like(combine_outputs[prev_idx])
        # b1. Combine backward for previous microbatch
        callables.combine.backward(
            comm_stream, events[prev_idx],
            combine_outputs[prev_idx], prev_output_grad
        )

        # f2. Dispatch forward for current microbatch
        dispatch_output, detached_outputs = callables.dispatch.forward(
            comm_stream, events[i],
            pre_mlp_layernorm_output, probs, routing_map
        )
        dispatched_input, tokens_per_expert = detached_outputs
        dispatch_outputs.append(dispatch_output)
        dispatch_detached_outputs.append(detached_outputs)

        # b2. MLP backward for previous microbatch
        callables.mlp.backward(
            comp_stream, events[prev_idx],
            *mlp_outputs[prev_idx], mlp_detached_outputs[prev_idx]
        )

         # f3. MLP forward for current microbatch
        mlp_output, detached_outputs = callables.mlp.forward(
            comp_stream, events[i],
            dispatched_input, tokens_per_expert, pre_mlp_layernorm_output, 
        )
        expert_output, shared_expert_output, mlp_bias = detached_outputs
        mlp_outputs.append(mlp_output)
        mlp_detached_outputs.append(detached_outputs)

        # b3. Dispatch backward for previous microbatch
        callables.dispatch.backward(
            comm_stream, events[prev_idx],
            *dispatch_outputs[prev_idx], dispatch_detached_outputs[prev_idx]
        )

        # f4. Combine forward for current microbatch
        output, detached_outputs = callables.combine.forward(
            comm_stream, events[i],
            expert_output, shared_expert_output, mlp_bias, hidden_states
        )
        combine_outputs.append(output)
        
        # b4. Attention backward for previous microbatch
        callables.attention.backward(
            comp_stream, events[prev_idx],
            *attention_outputs[prev_idx], attention_detached_outputs[prev_idx]
        )
    
    #Last microbatch backward pass
    # b1. Combine backward for last microbatch
    callables.combine.backward(
        comm_stream, events[microbatches-1],
        combine_outputs[microbatches-1], torch.ones_like(combine_outputs[microbatches-1])
    )   

    # b2. MLP backward for last microbatch
    callables.mlp.backward(
        comp_stream, events[microbatches-1],
        *mlp_outputs[microbatches-1], mlp_detached_outputs[microbatches-1]
    )
    
    # b3. Dispatch backward for last microbatch
    callables.dispatch.backward(
        comm_stream, events[microbatches-1],
        *dispatch_outputs[microbatches-1], dispatch_detached_outputs[microbatches-1]
    )

    # b4. Attention backward for last microbatch
    callables.attention.backward(
        comp_stream, events[microbatches-1],
        *attention_outputs[microbatches-1], attention_detached_outputs[microbatches-1]
    )

    capture = {
        "outputs": combine_outputs,
    }
    for name, param in model.named_parameters():
        capture[name] = param.grad  
    
    return capture
    
def test_1f1b_overlap(args):
    microbatches = 2
    model_ref = build_gpt_model(args)
    model_a2a_overlap = build_gpt_model(args)

    copy_weights_distributed(model_ref, model_a2a_overlap)
    
    capture_ref, input_tensors= run_model_ref_with_capture(args, model_ref, microbatches)

    input_tensors_copy = []
    for input_tensor in input_tensors:
        input_tensors_copy.append(input_tensor.clone())
    capture_a2a_overlap = run_model_a2a_overlap_with_capture(model_a2a_overlap, input_tensors_copy, microbatches)
    
    # for name, value in capture_ref.items():
    #     if value is None:
    #         continue
    #     assert torch.allclose(value, capture_a2a_overlap[name])


def main():
    initialize_megatron()
    args = get_args()
    torch.cuda.cudart().cudaProfilerStart()
    test_1f1b_overlap(args)
    torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
    print("done")