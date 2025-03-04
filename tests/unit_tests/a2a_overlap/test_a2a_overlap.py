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
    events = [torch.cuda.Event() for _ in range(microbatches)]
    
    # Create streams for computation and communication
    comp_stream = torch.cuda.Stream(device="cuda")
    comm_stream = torch.cuda.Stream(device="cuda")
    
    # Get the callables for the layer
    callables = model.decoder.get_layer_callables(0)
    
    # Initialize tensors to store intermediate results
    attention_outputs = []
    dispatch_outputs = []
    mlp_outputs = []
    final_outputs = []
    
    # Run the first microbatch forward pass
    input_tensor = input_tensors[0]
    
    # First microbatch forward pass (all on comp_stream)
    # Attention + Router forward
    probs, routing_map = callables.attention.forward(
        comp_stream, events[0],
        input_tensor, None, None, None, None, None, None, None, None
    )
    attention_outputs.append((probs, routing_map))
    
    # Token dispatch forward
    dispatched_input, tokens_per_expert = callables.dispatch.forward(
        comm_stream, events[0],
        input_tensor, probs, routing_map
    )
    dispatch_outputs.append((dispatched_input, tokens_per_expert))
    
    # MLP (experts) forward
    expert_output, shared_expert_output, mlp_bias = callables.mlp.forward(
        comp_stream, events[0],
        input_tensor, dispatched_input, tokens_per_expert
    )
    mlp_outputs.append((expert_output, shared_expert_output, mlp_bias))
    
    # Combine outputs forward
    output = callables.combine.forward(
        comm_stream, events[0],
        input_tensor, expert_output, mlp_bias, shared_expert_output
    )
    final_outputs.append(output)
    
    # Run the overlapped 1F1B schedule for the remaining microbatches
    for i in range(1, microbatches):
        # Current microbatch input
        input_tensor = input_tensors[i]
        
        # Previous microbatch index
        prev_idx = i-1
        
        # 1F1B interleaved schedule (following the reference pattern)
        # f1. Attention forward for current microbatch
        probs, routing_map = callables.attention.forward(
            comp_stream, events[i],
            input_tensor, None, None, None, None, None, None, None, None
        )
        attention_outputs.append((probs, routing_map))
        
        # Gradient for previous microbatch output
        prev_output_grad = torch.ones_like(final_outputs[prev_idx])
        # b1. Combine backward for previous microbatch
        callables.combine.backward(
            comm_stream, events[prev_idx],
            final_outputs[prev_idx], prev_output_grad
        )

        # f2. Dispatch forward for current microbatch
        dispatched_input, tokens_per_expert = callables.dispatch.forward(
            comm_stream, events[i],
            input_tensor, probs, routing_map
        )
        dispatch_outputs.append((dispatched_input, tokens_per_expert))
        # b2. MLP backward for previous microbatch
        prev_expert_output, prev_shared_expert_output, prev_mlp_bias = mlp_outputs[prev_idx]
        callables.mlp.backward(
            comm_stream, events[prev_idx],
            prev_expert_output, prev_shared_expert_output, prev_mlp_bias,
            prev_expert_output.grad,
            prev_shared_expert_output.grad if prev_shared_expert_output is not None else None,
            prev_mlp_bias.grad
        )

         # f3. MLP backward for current microbatch
        prev_expert_output, prev_shared_expert_output, prev_mlp_bias = mlp_outputs[prev_idx]
        callables.mlp.backward(
            comp_stream, events[i],
            prev_expert_output, prev_shared_expert_output, prev_mlp_bias,
            torch.ones_like(prev_expert_output),
            torch.ones_like(prev_shared_expert_output) if prev_shared_expert_output is not None else None,
            torch.ones_like(prev_mlp_bias)
        )
        # b3. Dispatch backward for previous microbatch
        prev_dispatched_input, prev_tokens_per_expert = dispatch_outputs[prev_idx]
        callables.dispatch.backward(
            comm_stream, events[prev_idx],
            prev_dispatched_input, prev_tokens_per_expert,
            prev_dispatched_input.grad
        )

        # f4. Combine forward for current microbatch
        output = callables.combine.forward(
            comm_stream, events[i],
            input_tensor, prev_expert_output, prev_mlp_bias, prev_shared_expert_output
        )
        final_outputs.append(output)
        # b4. Attention backward for previous microbatch
        prev_probs, prev_routing_map = attention_outputs[prev_idx]
        callables.attention.backward(
            comm_stream, events[prev_idx],
            prev_probs, prev_routing_map,
            prev_probs.grad
        )
    
    #Last microbatch backward pass
    # b1. Combine backward for last microbatch
    callables.combine.backward(
        comm_stream, events[microbatches-1],
        final_outputs[microbatches-1], torch.ones_like(final_outputs[microbatches-1])
    )   

    # b2. MLP backward for last microbatch
    prev_expert_output, prev_shared_expert_output, prev_mlp_bias = mlp_outputs[microbatches-1]
    callables.mlp.backward(
        comp_stream, events[microbatches-1],
        prev_expert_output, prev_shared_expert_output, prev_mlp_bias,
        prev_expert_output.grad,
        prev_shared_expert_output.grad if prev_shared_expert_output is not None else None,
        prev_mlp_bias.grad
    )
    
    # b3. Dispatch backward for last microbatch
    prev_dispatched_input, prev_tokens_per_expert = dispatch_outputs[microbatches-1]
    callables.dispatch.backward(
        comm_stream, events[microbatches-1],
        prev_dispatched_input, prev_tokens_per_expert,
        prev_dispatched_input.grad
    )           

    # b4. Attention backward for last microbatch
    prev_probs, prev_routing_map = attention_outputs[microbatches-1]
    callables.attention.backward(
        comp_stream, events[microbatches-1],
        prev_probs, prev_routing_map,
        prev_probs.grad
    )

    capture = {
        "outputs": final_outputs,
    }
    for name, param in model.named_parameters():
        capture[name] = param.grad  
    
    return capture
    
def test_1f1b_overlap(args):
    microbatches = 16
    model_ref = build_gpt_model(args)
    model_a2a_overlap = build_gpt_model(args)

    copy_weights_distributed(model_ref, model_a2a_overlap)
    
    capture_ref, input_tensors= run_model_ref_with_capture(args, model_ref, microbatches)

    input_tensors_copy = []
    for input_tensor in input_tensors:
        input_tensors_copy.append(input_tensor.clone())
    # capture_a2a_overlap = run_model_a2a_overlap_with_capture(model_a2a_overlap, input_tensors_copy, microbatches)
    
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