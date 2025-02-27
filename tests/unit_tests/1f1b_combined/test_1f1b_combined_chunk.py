# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain GPT."""

from functools import partial

import torch

from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.fine_grained_schedule import build_model_chunk_schedule_plan
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.pipeline_parallel.combined_1f1b import (
    StreamRelease,
    schedule_chunk_1f1b,
    schedule_chunk_backward,
    schedule_chunk_forward,
    set_streams,
)
from megatron.core.transformer.module import Float16Module
from megatron.training import get_args
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.initialize import initialize_megatron
from megatron.training.utils import unwrap_model
import numpy as np
import random


def schedule_1f1b_overlap(datas, model):
    outputs = []
    l = len(datas)
    # first f
    pre_stream = torch.cuda.current_stream()

    build_plan_func = partial(build_model_chunk_schedule_plan, model, None, None, None)
    data = datas[0]
    pre_schedule_plan = build_plan_func(decoder_input=data)
    print("schedule_chunk_forward")
    torch.cuda.nvtx.range_push(f"forward schudule")
    pre_output = schedule_chunk_forward(pre_schedule_plan)
    outputs.append(pre_output)
    torch.cuda.nvtx.range_pop()
    # 1f1b
    for i in range(1, l):
        print("schedule_chunk_1f1b")
        grad = torch.ones_like(pre_output)
        grad = StreamRelease.apply(pre_schedule_plan.event, pre_stream, grad)
        data = datas[i]
        schedule_plan = build_plan_func(decoder_input=data)
        torch.cuda.nvtx.range_push(f"1f1b schudule")
        pre_output = schedule_chunk_1f1b(schedule_plan, pre_schedule_plan, grad)
        outputs.append(pre_output)
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
    return outputs
    print("finish")
    
    
def schedule_f_and_b(datas, model):
    outputs = []
    for e in datas:
        print("schedule_f_and_b")
        output = model(None, None, None, decoder_input= e)
        grad = torch.ones_like(output)
        torch.autograd.backward(output, grad_tensors=grad)
        outputs.append(output)
    return  outputs   
        
    
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

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

def check_eq(a, b):
    np.testing.assert_equal(a.detach().float().cpu().numpy(), b.detach().float().cpu().numpy())
        
def test_1f1b_overlap(args):
    model1 = build_gpt_model(args)
    model2 = build_gpt_model(args)
    model2.load_state_dict(model1.state_dict())
    
    data1 = [build_data(args) for e in range(2)]
    for e in data1:
        e.requires_grad = True
        
    data2 = [e.detach() for e in data1]
    for e in data2:
        e.requires_grad = True
    set_streams()
    set_seed()
    outputs1 = schedule_1f1b_overlap(data1, model1)
    set_seed()
    outputs2 = schedule_f_and_b(data2, model2)
    
    
    # check output
    for (e1, e2) in zip(outputs1, outputs2):
        check_eq(e1, e2)
        
    # check data1, data2 grad
    for (e1, e2) in zip(data1, data2):
        check_eq(e1.grad, e2.grad)
        
    # check parameter weight grad
    for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
        if p1.grad is None:
            assert p2.grad is None
        else:    
            #print(f"check {n1}")
            check_eq(p1.grad, p2.grad)    
        


def main():
    initialize_megatron()
    set_deterministic()
    args = get_args()
    from megatron.core.transformer.enums import AttnBackend
    #args.attention_backend = AttnBackend.flash
    torch.cuda.cudart().cudaProfilerStart()
    test_1f1b_overlap(args)
    torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
