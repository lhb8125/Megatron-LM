# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import math

import torch

from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import (
    BucketingPolicy,
    _get_parameter_groups,
)


class _ExpertTestModule(torch.nn.Module):
    """
    Mock module whose params are routed under `.experts.` to trigger
    is_expert_param=True. The outer `layer` attribute puts a dot before
    `experts` in the parameter path (e.g. `layer.experts.linear_fc1`).
    """

    def __init__(self, shapes):
        super().__init__()
        self.layer = torch.nn.Module()
        self.layer.experts = torch.nn.ParameterDict(
            {name: torch.nn.Parameter(torch.empty(shape)) for name, shape in shapes.items()}
        )


class _MultiUseEmbeddingTestModule(torch.nn.Module):
    """Mock module with an MTP-style multi-use embedding and a regular root parameter."""

    def __init__(self):
        super().__init__()
        self.embedding_weight = torch.nn.Parameter(torch.empty(32, 16))
        self.embedding_weight.zero_out_wgrad = True
        self.final_norm_weight = torch.nn.Parameter(torch.empty(16))


def _get_bucket_signatures(module):
    bucket_groups, _, _ = _get_parameter_groups(
        module, BucketingPolicy(suggested_bucket_size=None), meta_device_init_fp8_params={}
    )
    param_to_name = {param: name for name, param in module.named_parameters()}
    return [
        {
            "chunk_size_factor": group.chunk_size_factor,
            "params": [(param_to_name[param], tuple(param.shape)) for param in group.params],
        }
        for group in bucket_groups
    ]


def test_multi_use_embedding_gets_separate_bucket_when_sharded():
    """Root-handled multi-use embeddings must not share a bucket with regular-hook params."""
    module = _MultiUseEmbeddingTestModule()
    bucket_groups, param_to_group, _ = _get_parameter_groups(
        module,
        BucketingPolicy(
            suggested_bucket_size=None, data_parallel_sharding_strategy="optim_grads_params"
        ),
        meta_device_init_fp8_params={},
    )

    embedding_group = param_to_group[module.embedding_weight]
    final_norm_group = param_to_group[module.final_norm_weight]
    assert embedding_group != final_norm_group
    assert bucket_groups[embedding_group].params == [module.embedding_weight]


def test_grouped_expert_weights_split_when_chunk_size_factors_differ():
    """Grouped expert weights with mismatched chunk size factors get routed to separate buckets."""
    num_local_experts = 4
    hidden_size = 12
    moe_ffn_hidden_size = 8
    shapes = {
        "linear_fc1": (num_local_experts, 2 * moe_ffn_hidden_size, hidden_size),
        "linear_fc2": (num_local_experts, hidden_size, moe_ffn_hidden_size),
    }
    module = _ExpertTestModule(shapes)

    assert _get_bucket_signatures(module) == [
        {
            "chunk_size_factor": torch.Size(shapes["linear_fc1"])[1:].numel(),
            "params": [("layer.experts.linear_fc1", shapes["linear_fc1"])],
        },
        {
            "chunk_size_factor": torch.Size(shapes["linear_fc2"])[1:].numel(),
            "params": [("layer.experts.linear_fc2", shapes["linear_fc2"])],
        },
    ]


def test_per_expert_2d_weights_merge_via_lcm():
    """Per-expert 2D weights merge into a single bucket via LCM chunk size factor."""
    hidden_size = 12
    moe_ffn_hidden_size = 8
    shapes = {
        "linear_fc1": (2 * moe_ffn_hidden_size, hidden_size),
        "linear_fc2": (hidden_size, moe_ffn_hidden_size),
    }
    module = _ExpertTestModule(shapes)

    assert _get_bucket_signatures(module) == [
        {
            "chunk_size_factor": math.lcm(
                torch.Size(shapes["linear_fc1"])[1:].numel(),
                torch.Size(shapes["linear_fc2"])[1:].numel(),
            ),
            "params": [
                ("layer.experts.linear_fc1", shapes["linear_fc1"]),
                ("layer.experts.linear_fc2", shapes["linear_fc2"]),
            ],
        }
    ]
