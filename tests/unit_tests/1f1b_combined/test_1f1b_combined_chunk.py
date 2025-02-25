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


def schedule_1f1b_overlap(args, model):
    l = 16
    # first f
    pre_stream = torch.cuda.current_stream()

    build_plan_func = partial(build_model_chunk_schedule_plan, model, None, None, None)
    data = build_data(args)
    pre_schedule_plan = build_plan_func(decoder_input=data)
    print("schedule_chunk_forward")
    torch.cuda.nvtx.range_push(f"forward schudule")
    pre_output = schedule_chunk_forward(pre_schedule_plan)
    torch.cuda.nvtx.range_pop()
    # 1f1b
    for i in range(1, l):
        print("schedule_chunk_1f1b")
        grad = torch.ones_like(pre_output)
        grad = StreamRelease.apply(pre_schedule_plan.event, pre_stream, grad)
        data = build_data(args)
        schedule_plan = build_plan_func(decoder_input=data)
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
        rotary_base=args.rope_theta,
    )

    if args.fp16 or args.bf16:
        model = Float16Module(config, model)
    model = unwrap_model(model, (Float16Module,))
    return model


def test_1f1b_overlap(args):
    model = build_gpt_model(args)
    set_streams()
    schedule_1f1b_overlap(args, model)


def main():
    initialize_megatron()
    args = get_args()
    torch.cuda.cudart().cudaProfilerStart()
    test_1f1b_overlap(args)
    torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
