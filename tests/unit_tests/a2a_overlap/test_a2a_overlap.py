# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
import os
import time
import random

import torch

from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.transformer import transformer_layer
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllPerBatchState
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
    hidden_states = torch.randn(*(s, b, h), dtype=torch.bfloat16, device="cuda") * 100
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
    # Recursively copy weights from source to target model
    def _copy_weights_recursive(source_module, target_module):
        for name, child in source_module.named_children():
            if name in dict(target_module.named_modules()):
                target_child = target_module.get_submodule(name)
                _copy_weights_recursive(child, target_child)
            else:
                print(f"[debug] child_prefix not in target_module: {name}")
        
        # Copy parameters at current level
        for name, param in source_module.named_parameters(recurse=False):
            if name in dict(target_module.named_parameters()):
                target_param = target_module.get_parameter(name)
                target_param.data.copy_(param.data)
            else:
                print(f"[debug] param_name not in target_module: {name}")
    
    # Start recursive copy from the root modules
    _copy_weights_recursive(model_source, model_target)
    print("Weights have been successfully copied.")

def run_model_ref_with_capture(model, input_tensors, iterations):
    for module in model.modules():
        if hasattr(module, 'fuse_wgrad_accumulation'):
            module.fuse_wgrad_accumulation = False
    
    output_tensors = []
    for i in range(iterations):
        output = model(input_tensors[i].clone())[0]
        output_tensors.append(output)
        output.backward(torch.ones_like(output))

    capture = {
        "outputs": output_tensors,
    }
    for name, param in model.named_parameters():
        capture[name] = param.grad

    return capture


def run_model_a2a_overlap_with_capture(model, input_tensors, microbatches):
    for module in model.modules():
        if hasattr(module, 'fuse_wgrad_accumulation'):
            module.fuse_wgrad_accumulation = False
    events = [torch.cuda.Event() for _ in range(microbatches)]
    states = [MoEAlltoAllPerBatchState() for _ in range(microbatches)]
    # Create streams for computation and communication
    comp_stream = torch.cuda.current_stream()
    comm_stream = torch.cuda.Stream(device="cuda")
    
    # Get the callables, only 1 layer in the decoder
    callables = model.get_submodule_callables()
    
    # Initialize tensors to store intermediate results
    attention_outputs = []
    dispatch_outputs = []
    mlp_outputs = []
    combine_outputs = []
    attention_detached_outputs = []
    dispatch_detached_outputs = []
    mlp_detached_outputs = []
    
    
    # Run the first microbatch forward pass
    input_tensor = input_tensors[0].clone()
    events[0].record(comp_stream)
    # First microbatch forward pass
    # f1.Attention + Router forward
    attention_output, detached_outputs = callables.attention.forward(
        comp_stream, events[0],
        input_tensor, None, None, None, None, None, None, None, None, states[0]
    )
    hidden_states, pre_mlp_layernorm_output, probs, routing_map = detached_outputs
    attention_outputs.append(attention_output)
    attention_detached_outputs.append(detached_outputs)
    
    # f2. Token dispatch forward
    dispatch_output, detached_outputs = callables.dispatch.forward(
        comm_stream, events[0],
        pre_mlp_layernorm_output, probs, routing_map, states[0]
    )
    dispatched_input, tokens_per_expert = detached_outputs
    dispatch_outputs.append(dispatch_output)
    dispatch_detached_outputs.append(detached_outputs)
    
    # f3. MLP (experts) forward
    mlp_output, detached_outputs = callables.mlp.forward(
        comp_stream, events[0],
        dispatched_input, tokens_per_expert, pre_mlp_layernorm_output, states[0]
    )
    expert_output, shared_expert_output, mlp_bias = detached_outputs
    mlp_outputs.append(mlp_output)
    mlp_detached_outputs.append(detached_outputs)
    
    # f4. Combine outputs forward
    output, detached_outputs = callables.combine.forward(
        comm_stream, events[0],
        expert_output, shared_expert_output, mlp_bias, hidden_states, states[0]
    )
    combine_outputs.append(output)
    
    # Run the overlapped 1F1B schedule for the remaining microbatches
    for i in range(1, microbatches):
        events[i].record(comp_stream)

        # Current microbatch input
        input_tensor = input_tensors[i].clone()
        
        # Previous microbatch index
        prev_idx = i-1
        
        # 1F1B interleaved schedule (following the reference pattern)
        # f1. Attention forward for current microbatch
        attention_output, detached_outputs = callables.attention.forward(
            comp_stream, events[i],
            input_tensor, None, None, None, None, None, None, None, None, states[i]
        )
        hidden_states, pre_mlp_layernorm_output, probs, routing_map = detached_outputs
        attention_outputs.append(attention_output)
        attention_detached_outputs.append(detached_outputs)
        
        # Gradient for previous microbatch output
        prev_output_grad = torch.ones_like(combine_outputs[prev_idx])
        # b1. Combine backward for previous microbatch
        callables.combine.backward(
            comm_stream, events[prev_idx],
            combine_outputs[prev_idx], prev_output_grad, states[prev_idx]
        )

        # f2. Dispatch forward for current microbatch
        dispatch_output, detached_outputs = callables.dispatch.forward(
            comm_stream, events[i],
            pre_mlp_layernorm_output, probs, routing_map, states[i]
        )
        dispatched_input, tokens_per_expert = detached_outputs
        dispatch_outputs.append(dispatch_output)
        dispatch_detached_outputs.append(detached_outputs)

        # b2. MLP backward for previous microbatch
        callables.mlp.backward(
            comp_stream, events[prev_idx],
            *mlp_outputs[prev_idx], mlp_detached_outputs[prev_idx], states[prev_idx]
        )

         # f3. MLP forward for current microbatch
        mlp_output, detached_outputs = callables.mlp.forward(
            comp_stream, events[i],
            dispatched_input, tokens_per_expert, pre_mlp_layernorm_output, states[i]
        )
        expert_output, shared_expert_output, mlp_bias = detached_outputs
        mlp_outputs.append(mlp_output)
        mlp_detached_outputs.append(detached_outputs)

        # b3. Dispatch backward for previous microbatch
        callables.dispatch.backward(
            comm_stream, events[prev_idx],
            *dispatch_outputs[prev_idx], dispatch_detached_outputs[prev_idx], states[prev_idx]
        )

        # f4. Combine forward for current microbatch
        output, detached_outputs = callables.combine.forward(
            comm_stream, events[i],
            expert_output, shared_expert_output, mlp_bias, hidden_states, states[i]
        )
        combine_outputs.append(output)
        
        # b4. Attention backward for previous microbatch
        callables.attention.backward(
            comp_stream, events[prev_idx],
            *attention_outputs[prev_idx], attention_detached_outputs[prev_idx], states[prev_idx]
        )

    #Last microbatch backward pass
    # b1. Combine backward for last microbatch
    callables.combine.backward(
        comm_stream, events[prev_idx],
        combine_outputs[microbatches-1], torch.ones_like(combine_outputs[microbatches-1]), states[microbatches-1]
    )   

    # b2. MLP backward for last microbatch
    callables.mlp.backward(
        comp_stream, events[prev_idx],
        *mlp_outputs[microbatches-1], mlp_detached_outputs[microbatches-1], states[microbatches-1]
    )
    
    # b3. Dispatch backward for last microbatch
    callables.dispatch.backward(
        comm_stream, events[prev_idx],
        *dispatch_outputs[microbatches-1], dispatch_detached_outputs[microbatches-1], states[microbatches-1]
    )

    # b4. Attention backward for last microbatch
    callables.attention.backward(
        comp_stream, events[prev_idx],
        *attention_outputs[microbatches-1], attention_detached_outputs[microbatches-1], states[microbatches-1]
    )
    for event in events:
        event.wait(comp_stream)
        event.wait(comm_stream)

    capture = {
        "outputs": combine_outputs,
    }
    for name, param in model.named_parameters():
        capture[name] = param.grad  
    
    return capture

def reset_model(model, params=None):
    model.zero_grad()
    if params is None:
        params = {}
        for name, param in model.named_parameters():
            params[name] = param.data.clone()
        return params
    else:
        for name, param in model.named_parameters():
            param.data.copy_(params[name])

def compare_captures(capture_a2a_overlap, capture_ref):
    for name, value in capture_ref.items():
        print(name, end='\0')
        assert name in capture_a2a_overlap, f"gradient name mismatch, '{name}' not in capture_a2a_overlap.keys()"
        if value is None:
            assert capture_a2a_overlap[name] is None
            print(": PASS")
        elif name == "outputs":
            assert len(value) == len(capture_a2a_overlap[name]), "outputs length mismatch"
            for i in range(len(value)):
                assert value[i].shape == capture_a2a_overlap[name][i].shape, "outputs shape mismatch"
                assert torch.allclose(value[i], capture_a2a_overlap[name][i]), f"outputs value mismatch at index {i}."
            print(": PASS")
        else:
            try:
                assert value.shape == capture_a2a_overlap[name].shape, f"gradient shape mismatch: '{name}'"
                assert torch.allclose(value, capture_a2a_overlap[name]), f"gradient mismatch: '{name}'"
                print(": PASS")
            except Exception as e:
                max_diff = torch.abs(value - capture_a2a_overlap[name])
                max_diff_value = torch.max(max_diff)
                max_diff_index = torch.argmax(max_diff.view(-1))
                flat_original = value.view(-1)
                flat_a2a = capture_a2a_overlap[name].view(-1)
                print(": FAIL")
                print(f"max diff: {max_diff_value} at index {max_diff_index}, original/a2a_overlap value at max diff: {flat_original[max_diff_index]}/{flat_a2a[max_diff_index]}")
                # print(f"original {name}: ", value)
                # print(f"a2a_overlap {name}: ", capture_a2a_overlap[name])

def test_1f1b_overlap(args):
    microbatches = 3
    model = build_gpt_model(args).decoder.layers[0]
    params = reset_model(model)
    input_tensors = [build_data(args) for _ in range(microbatches)]

    # capture_a2a_overlap = run_model_a2a_overlap_with_capture(model, input_tensors, microbatches)
    capture_ref = run_model_ref_with_capture(model, input_tensors, microbatches)
    reset_model(model, params)
    capture_a2a_overlap = run_model_a2a_overlap_with_capture(model, input_tensors, microbatches)
    for i in range(8):
        if torch.distributed.get_rank() == i:
            print(f"########## rank {i} result ##########")
            compare_captures(capture_ref, capture_a2a_overlap)
            break
        time.sleep(3)

def main():
    initialize_megatron()
    args = get_args()
    torch.cuda.cudart().cudaProfilerStart()
    test_1f1b_overlap(args)
    time.sleep(20)
    torch.cuda.cudart().cudaProfilerStop()

torch.use_deterministic_algorithms(True)
torch.manual_seed(0)
random.seed(0)

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

debug_name="mlp.experts.weight1"
debug_rank=0

if __name__ == "__main__":
    main()