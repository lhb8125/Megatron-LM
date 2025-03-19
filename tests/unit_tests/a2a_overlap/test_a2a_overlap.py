# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
import os
import time
import random

import torch

from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.transformer.spec_utils import build_module
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
    hidden_states = torch.randn(*(s, b, h), dtype=torch.bfloat16, device="cuda")
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

    # if "moe_use_legacy_grouped_gemm" in signature.parameters:
    #     extra_args["moe_use_legacy_grouped_gemm"] = True

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

    for param in model.parameters():
        param.grad = None
        param.main_grad = torch.zeros_like(param, dtype=torch.float32)
    
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

def print_grad(model, str):
    if torch.distributed.get_rank() == 6:
        for name, param in model.named_parameters():
            print(f"[debug] {str} {name} grad: {param.grad}")

def run_model_ref_with_capture(model, input_tensors, iterations):
    for module in model.modules():
        if hasattr(module, 'fuse_wgrad_accumulation'):
            module.fuse_wgrad_accumulation = False
    
    output_tensors = []
    dgrads = []
    for i in range(iterations):
        input_tensor = input_tensors[i].clone().detach().requires_grad_(True)
        output = model(input_tensor)[0]
        output_tensors.append(output)
        output.backward(torch.ones_like(output))
        dgrads.append(input_tensor.grad)
    capture = {
        "@outputs": output_tensors,
        "@dgrads": dgrads,
    }
    for name, param in model.named_parameters():
        capture[name] = param.grad

    return capture

def run_model_a2a_overlap_with_capture(model, input_tensors, microbatches):
    if model.is_deepep_dispatcher():
        if torch.distributed.get_rank() == 0:
            print("Using DeepEP dispatcher")
        return run_model_a2a_overlap_with_capture_deepep(model, input_tensors, microbatches)
    else:
        if torch.distributed.get_rank() == 0:
            print("Using AlltoAll dispatcher")
        return run_model_a2a_overlap_with_capture_all2all(model, input_tensors, microbatches)

def run_model_a2a_overlap_with_capture_deepep(model, input_tensors, microbatches):
    for i in range(len(input_tensors)):
        input_tensors[i] = input_tensors[i].clone()
    for module in model.modules():
        if hasattr(module, 'fuse_wgrad_accumulation'):
            module.fuse_wgrad_accumulation = False
    events = [torch.cuda.Event() for _ in range(microbatches)]
    # Create streams for computation and communication
    comp_stream = torch.cuda.Stream(device="cuda")
    comm_stream = torch.cuda.Stream(device="cuda")
    
    # Get the callables, only 1 layer in the decoder
    callables = model.get_submodule_callables()
    
    # Initialize tensors to store intermediate results
    attention_outputs = []
    dispatch_outputs = []
    mlp_outputs = []
    combine_outputs = []
    attention_detached_inputs = []
    dispatch_detached_inputs = []
    mlp_detached_inputs = []
    combine_detached_inputs = []
    dgrads = []
    
    # Run the first microbatch forward pass
    input_tensor = input_tensors[0]
    events[0].record(comp_stream)
    # First microbatch forward pass
    # f1.Attention + Router forward
    attention_output, detached_inputs = callables.attention.forward(
        comp_stream, events[0], input_tensor
    )
    hidden_states, pre_mlp_layernorm_output, deepep_hidden_states, probs = attention_output
    attention_outputs.append(attention_output)
    attention_detached_inputs.append(detached_inputs)
    
    # f2. Token dispatch forward
    dispatch_output, detached_inputs = callables.dispatch.forward(
        comm_stream, events[0],
        deepep_hidden_states, probs
    )
    deepep_hidden_states, probs = dispatch_output
    dispatch_outputs.append(dispatch_output)
    dispatch_detached_inputs.append(detached_inputs)
    
    # f3. MLP (experts) forward
    mlp_output, detached_inputs = callables.mlp.forward(
        comp_stream, events[0],
        deepep_hidden_states, pre_mlp_layernorm_output, probs,
    )
    expert_output, shared_expert_output, _, mlp_bias = mlp_output
    mlp_outputs.append(mlp_output)
    mlp_detached_inputs.append(detached_inputs)
    
    # f4. Combine forward
    combine_output, detached_inputs = callables.combine.forward(
        comm_stream, events[0],
        expert_output, shared_expert_output, mlp_bias, None, hidden_states
    )
    combine_outputs.append(combine_output)
    combine_detached_inputs.append(detached_inputs)

    prev_idx = 0
    # Run the overlapped 1F1B schedule for the remaining microbatches
    for i in range(1, microbatches):
        # print_grad(model, f"before {i}th backward")
        torch.cuda.nvtx.range_push(f"1f1b loop {i}")
        # Current microbatch input
        input_tensor = input_tensors[i]
        
        # Previous microbatch index
        prev_idx = i-1
        
        # 1F1B interleaved schedule (following the reference pattern)
        # Gradient for previous microbatch output
        # b1. Combine backward for previous microbatch
        grads = callables.combine.backward(
            comm_stream, events[prev_idx],
            combine_outputs[prev_idx], 
            tuple([torch.ones_like(combine_outputs[prev_idx][0])]),
            combine_detached_inputs[prev_idx],
        )   
        output_grad, shared_expert_output_grad, mlp_bias_grad, _, residual_grad = grads
        
        # f1. Attention forward for current microbatch
        attention_output, detached_inputs = callables.attention.forward(
            comp_stream, events[i], input_tensor
        )
        hidden_states, pre_mlp_layernorm_output, deepep_hidden_states, probs = attention_output
        attention_outputs.append(attention_output)
        attention_detached_inputs.append(detached_inputs)

        # f2. Dispatch forward for current microbatch
        dispatch_output, detached_inputs = callables.dispatch.forward(
            comm_stream, events[i],
            deepep_hidden_states, probs
        )
        deepep_hidden_states, probs = dispatch_output
        dispatch_outputs.append(dispatch_output)
        dispatch_detached_inputs.append(detached_inputs)

        # b2. MLP backward for previous microbatch
        grads = callables.mlp.backward(
            comp_stream, events[prev_idx],
            mlp_outputs[prev_idx], 
            tuple([output_grad, shared_expert_output_grad, None, mlp_bias_grad]),
            mlp_detached_inputs[prev_idx],
        )
        dispatched_input_grad, hidden_states_grad, probs_grad = grads

        # b3. Dispatch backward for previous microbatch
        grads = callables.dispatch.backward(
            comm_stream, events[prev_idx],
            dispatch_outputs[prev_idx], 
            tuple([dispatched_input_grad, probs_grad]),
            dispatch_detached_inputs[prev_idx],
        )
        dispatched_input_grad, probs_grad = grads

         # f3. MLP forward for current microbatch
        mlp_output, detached_inputs = callables.mlp.forward(
            comp_stream, events[i],
            deepep_hidden_states, pre_mlp_layernorm_output, probs,
        )
        expert_output, shared_expert_output, _, mlp_bias = mlp_output
        mlp_outputs.append(mlp_output)
        mlp_detached_inputs.append(detached_inputs)
        
        # f4. Combine forward for current microbatch
        combine_output, detached_inputs = callables.combine.forward(
            comm_stream, events[i],
            expert_output, shared_expert_output, mlp_bias, None, hidden_states
        )
        combine_outputs.append(combine_output)
        combine_detached_inputs.append(detached_inputs)

        # b4. Attention backward for previous microbatch
        grads = callables.attention.backward(
            comp_stream, events[prev_idx],
            attention_outputs[prev_idx], 
            tuple([residual_grad, hidden_states_grad, dispatched_input_grad, probs_grad]), 
            attention_detached_inputs[prev_idx],
        )
        dgrads.append(grads)
        torch.cuda.nvtx.range_pop()

    #Last microbatch backward pass
    # b1. Combine backward for last microbatch
    grads = callables.combine.backward(
        comm_stream, events[prev_idx],
        combine_outputs[microbatches-1], 
        tuple([torch.ones_like(combine_outputs[microbatches-1][0])]),
        combine_detached_inputs[microbatches-1],
    )   
    output_grad, shared_expert_output_grad, mlp_bias_grad, _, residual_grad = grads

    # b2. MLP backward for last microbatch
    grads = callables.mlp.backward(
        comp_stream, events[prev_idx],
        mlp_outputs[microbatches-1], 
        tuple([output_grad, shared_expert_output_grad, None, mlp_bias_grad]),
        mlp_detached_inputs[microbatches-1],
    )
    dispatched_input_grad, hidden_states_grad, probs_grad = grads
    
    # b3. Dispatch backward for last microbatch
    grads = callables.dispatch.backward(
        comm_stream, events[prev_idx],
        dispatch_outputs[microbatches-1], 
        tuple([dispatched_input_grad, probs_grad]),
        dispatch_detached_inputs[microbatches-1],
    )
    dispatched_input_grad, probs_grad = grads

    # b4. Attention backward for last microbatch
    grads = callables.attention.backward(
        comp_stream, events[prev_idx],
        attention_outputs[microbatches-1], 
        tuple([residual_grad, hidden_states_grad, dispatched_input_grad, probs_grad]), 
        attention_detached_inputs[microbatches-1],
    )
    dgrads.append(grads)
    torch.cuda.synchronize()
    combine_outputs = list(map(lambda x: x[0], combine_outputs))
    dgrads = list(map(lambda x: x[0], dgrads))
    capture = {
        "@outputs": combine_outputs,
        "@dgrads": dgrads,
    }
    for name, param in model.named_parameters():
        capture[name] = param.grad  
    
    return capture

def run_model_a2a_overlap_with_capture_all2all(model, input_tensors, microbatches):
    for i in range(len(input_tensors)):
        input_tensors[i] = input_tensors[i].clone()
    for module in model.modules():
        if hasattr(module, 'fuse_wgrad_accumulation'):
            module.fuse_wgrad_accumulation = False
    events = [torch.cuda.Event() for _ in range(microbatches)]
    # Create streams for computation and communication
    comp_stream = torch.cuda.Stream(device="cuda")
    comm_stream = torch.cuda.Stream(device="cuda")
    
    # Get the callables, only 1 layer in the decoder
    callables = model.get_submodule_callables()
    
    # Initialize tensors to store intermediate results
    attention_outputs = []
    dispatch_outputs = []
    mlp_outputs = []
    combine_outputs = []
    attention_detached_inputs = []
    dispatch_detached_inputs = []
    mlp_detached_inputs = []
    combine_detached_inputs = []
    dgrads = []
    
    
    # Run the first microbatch forward pass
    input_tensor = input_tensors[0]
    events[0].record(comp_stream)
    # First microbatch forward pass
    # f1.Attention + Router forward
    attention_output, detached_inputs = callables.attention.forward(
        comp_stream, events[0], input_tensor
    )
    hidden_states, pre_mlp_layernorm_output, tokens_per_expert, permutated_local_input_tokens, probs = attention_output
    attention_outputs.append(attention_output)
    attention_detached_inputs.append(detached_inputs)
    
    # f2. Token dispatch forward
    dispatch_output, detached_inputs = callables.dispatch.forward(
        comm_stream, events[0],
        permutated_local_input_tokens
    )
    global_input_tokens = dispatch_output[0]
    dispatch_outputs.append(dispatch_output)
    dispatch_detached_inputs.append(detached_inputs)
    
    # f3. MLP (experts) forward
    mlp_output, detached_inputs = callables.mlp.forward(
        comp_stream, events[0],
        global_input_tokens, pre_mlp_layernorm_output, probs, tokens_per_expert
    )
    expert_output, shared_expert_output, probs, mlp_bias = mlp_output
    mlp_outputs.append(mlp_output)
    mlp_detached_inputs.append(detached_inputs)
    
    # f4. Combine forward
    combine_output, detached_inputs = callables.combine.forward(
        comm_stream, events[0],
        expert_output, shared_expert_output, mlp_bias, probs, hidden_states
    )
    combine_outputs.append(combine_output)
    combine_detached_inputs.append(detached_inputs)

    # Run the overlapped 1F1B schedule for the remaining microbatches
    for i in range(1, microbatches):
        # print_grad(model, f"before {i}th backward")
        torch.cuda.nvtx.range_push(f"1f1b loop {i}")
        # Current microbatch input
        input_tensor = input_tensors[i]
        
        # Previous microbatch index
        prev_idx = i-1
        
        # 1F1B interleaved schedule (following the reference pattern)
        # Gradient for previous microbatch output
        # b1. Combine backward for previous microbatch
        grads = callables.combine.backward(
            comm_stream, events[prev_idx],
            combine_outputs[prev_idx], 
            tuple([torch.ones_like(combine_outputs[prev_idx][0])]),
            combine_detached_inputs[prev_idx],
        )
        output_grad, shared_expert_output_grad, mlp_bias_grad, probs_grad, residual_grad = grads
        
        # f1. Attention forward for current microbatch
        attention_output, detached_inputs = callables.attention.forward(
            comp_stream, events[i], input_tensor
        )
        hidden_states, pre_mlp_layernorm_output, tokens_per_expert, permutated_local_input_tokens, probs = attention_output
        attention_outputs.append(attention_output)
        attention_detached_inputs.append(detached_inputs)

        # f2. Dispatch forward for current microbatch
        dispatch_output, detached_inputs = callables.dispatch.forward(
            comm_stream, events[i],
            permutated_local_input_tokens
        )
        global_input_tokens = dispatch_output[0]
        dispatch_outputs.append(dispatch_output)
        dispatch_detached_inputs.append(detached_inputs)

        # b2. MLP backward for previous microbatch
        grads = callables.mlp.backward(
            comp_stream, events[prev_idx],
            mlp_outputs[prev_idx], 
            tuple([output_grad, shared_expert_output_grad, probs_grad, mlp_bias_grad]),
            mlp_detached_inputs[prev_idx],
        )
        dispatched_input_grad, hidden_states_grad, probs_grad, _ = grads

        # b3. Dispatch backward for previous microbatch
        grads = callables.dispatch.backward(
            comm_stream, events[prev_idx],
            dispatch_outputs[prev_idx], 
            tuple([dispatched_input_grad]),
            dispatch_detached_inputs[prev_idx],
        )
        dispatched_input_grad = grads[0]

         # f3. MLP forward for current microbatch
        mlp_output, detached_inputs = callables.mlp.forward(
            comp_stream, events[i],
            global_input_tokens, pre_mlp_layernorm_output, probs, tokens_per_expert
        )
        expert_output, shared_expert_output, probs, mlp_bias = mlp_output
        mlp_outputs.append(mlp_output)
        mlp_detached_inputs.append(detached_inputs)
        
        # f4. Combine forward for current microbatch
        combine_output, detached_inputs = callables.combine.forward(
            comm_stream, events[i],
            expert_output, shared_expert_output, mlp_bias, probs, hidden_states
        )
        combine_outputs.append(combine_output)
        combine_detached_inputs.append(detached_inputs)

        # b4. Attention backward for previous microbatch
        grads = callables.attention.backward(
            comp_stream, events[prev_idx],
            attention_outputs[prev_idx], 
            tuple([residual_grad, hidden_states_grad,  None, dispatched_input_grad, probs_grad]), 
            attention_detached_inputs[prev_idx],
        )
        dgrads.append(grads)
        torch.cuda.nvtx.range_pop()

    #Last microbatch backward pass
    # b1. Combine backward for last microbatch
    grads = callables.combine.backward(
        comm_stream, events[prev_idx],
        combine_outputs[microbatches-1], 
        tuple([torch.ones_like(combine_outputs[microbatches-1][0])]),
        combine_detached_inputs[microbatches-1],
    )   
    output_grad, shared_expert_output_grad, mlp_bias_grad, probs_grad, residual_grad = grads

    # b2. MLP backward for last microbatch
    grads = callables.mlp.backward(
        comp_stream, events[prev_idx],
        mlp_outputs[microbatches-1], 
        tuple([output_grad, shared_expert_output_grad, probs_grad, mlp_bias_grad]),
        mlp_detached_inputs[microbatches-1],
    )
    dispatched_input_grad, hidden_states_grad, probs_grad, _ = grads
    
    # b3. Dispatch backward for last microbatch
    grads = callables.dispatch.backward(
        comm_stream, events[prev_idx],
        dispatch_outputs[microbatches-1], 
        tuple([dispatched_input_grad]),
        dispatch_detached_inputs[microbatches-1],
    )
    dispatched_input_grad = grads[0]

    # b4. Attention backward for last microbatch
    grads = callables.attention.backward(
        comp_stream, events[prev_idx],
        attention_outputs[microbatches-1], 
        tuple([residual_grad, hidden_states_grad,  None, dispatched_input_grad, probs_grad]), 
        attention_detached_inputs[microbatches-1],
    )
    dgrads.append(grads)
    torch.cuda.synchronize()

    # record wgrad
    for name, param in model.named_parameters():
        capture[name] = param.grad  
    
    # record activation grad
    if torch.distributed.get_rank() == 0:
        print("debug:", len(dgrads[0]))
    dgrads = list(map(lambda x: x[0], dgrads))

    # record forward outputs
    combine_outputs = list(map(lambda x: x[0], combine_outputs))
    
    capture = {
        "@outputs": combine_outputs,
        "@dgrads": dgrads
    }

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

def compare_captures(capture_ref, capture_a2a_overlap):
    def bit_same(a, b):
        assert a.dtype == b.dtype, "dtype mismatch"
        if a.dtype in [torch.bfloat16, torch.half]:
            return torch.all(a.view(torch.int16)==b.view(torch.int16))
        else:
            return torch.all(a.view(torch.int32)==b.view(torch.int32))
    for name, value in capture_ref.items():
        print(name, end='\0')
        assert name in capture_a2a_overlap, f"gradient name mismatch, '{name}' not in capture_a2a_overlap.keys()"
        if value is None:
            assert capture_a2a_overlap[name] is None
            print(": PASS")
        elif name in ["@outputs", "@dgrads"]:
            assert len(value) == len(capture_a2a_overlap[name]), f"'{name}' outputs length mismatch"
            for i in range(len(value)):
                assert capture_a2a_overlap[name][i] is not None, f"'{name}' outputs at index {i} is None, {capture_a2a_overlap[name]}"
                assert value[i].shape == capture_a2a_overlap[name][i].shape, f"'{name}' outputs shape mismatch"
                # assert torch.allclose(value[i], capture_a2a_overlap[name][i]), f"outputs value mismatch at index {i}."
                assert bit_same(value[i], capture_a2a_overlap[name][i]), f"'{name}'outputs value mismatch at index {i}."
            print(": PASS")
        else:
            try:
                assert value.shape == capture_a2a_overlap[name].shape, f"gradient shape mismatch: '{name}'"
                assert bit_same(value, capture_a2a_overlap[name]), f"gradient mismatch: '{name}'"
                # assert torch.allclose(value, capture_a2a_overlap[name]), f"gradient mismatch: '{name}'"
                print(": PASS")
            except Exception as e:
                max_diff = torch.abs(value - capture_a2a_overlap[name])
                max_diff_value = torch.max(max_diff)
                max_diff_index = torch.argmax(max_diff.view(-1))
                flat_original = value.view(-1)
                flat_a2a = capture_a2a_overlap[name].view(-1)
                print(": FAIL")
                print(f"\tmax diff: {max_diff_value} at index {max_diff_index}, original/a2a_overlap value at max diff: {flat_original[max_diff_index]}/{flat_a2a[max_diff_index]}")

def build_transformer_layer(args):
    config = core_transformer_config_from_args(args)
    model_spec = get_gpt_layer_with_transformer_engine_spec(
        args.num_experts,
        args.moe_grouped_gemm,
        multi_latent_attention=args.multi_latent_attention,
        moe_use_legacy_grouped_gemm=True,
    )
    transformer_layer = build_module(model_spec, config=config, layer_number=1)
    return transformer_layer

def test_1f1b_overlap(args):
    microbatches = 16
    model = build_transformer_layer(args)
    # model = build_gpt_model(args).decoder.layers[0]
    params = reset_model(model)
    input_tensors = [build_data(args) for _ in range(microbatches)]

    capture_ref = run_model_ref_with_capture(model, input_tensors, microbatches)
    reset_model(model, params)
    if torch.distributed.get_rank() == 0:
        torch.cuda.cudart().cudaProfilerStart()
        # torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()
    capture_a2a_overlap = run_model_a2a_overlap_with_capture(model, input_tensors, microbatches)
    if torch.distributed.get_rank() == 0:
        torch.cuda.cudart().cudaProfilerStop()
    for i in range(8):
        if torch.distributed.get_rank() == i:
            print(f"########## rank {i} result ##########")
            compare_captures(capture_ref, capture_a2a_overlap)
            break
        time.sleep(2)

def main():
    initialize_megatron()
    args = get_args()
    test_1f1b_overlap(args)

torch.use_deterministic_algorithms(True)
torch.manual_seed(0)
random.seed(0)

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

debug_name="mlp.experts.weight1"
debug_rank=0

if __name__ == "__main__":
    main()