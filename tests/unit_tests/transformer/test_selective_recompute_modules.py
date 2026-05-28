# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from tests.unit_tests.test_utilities import Utils


def _base_config(**kwargs):
    return TransformerConfig(
        num_layers=2, hidden_size=128, num_attention_heads=4, use_cpu_initialization=True, **kwargs
    )


def test_config_accepts_new_selective_recompute_modules():
    config = _base_config(
        recompute_granularity="selective",
        recompute_modules=["attn_norm", "mlp_norm", "qkv_linear", "attn_proj"],
    )

    assert config.recompute_modules == ["attn_norm", "mlp_norm", "qkv_linear", "attn_proj"]


def test_config_accepts_self_attn_selective_recompute_module():
    config = _base_config(
        recompute_granularity="selective",
        recompute_modules=["self_attn", "mlp_norm"],
    )

    assert config.recompute_modules == ["self_attn", "mlp_norm"]


def test_self_attn_recompute_rejects_nested_attention_modules():
    with pytest.raises(ValueError, match="self_attn.*nested"):
        _base_config(
            recompute_granularity="selective",
            recompute_modules=["self_attn", "core_attn"],
        )


def test_self_attn_recompute_conflicts_with_consumed_output_offload():
    with pytest.raises(AssertionError, match="self_attn"):
        _base_config(
            recompute_granularity="selective",
            recompute_modules=["self_attn"],
            fine_grained_activation_offloading=True,
            offload_modules=["mlp_norm"],
        )


def test_expert_fc1_recompute_requires_grouped_gemm():
    with pytest.raises(ValueError, match="expert_fc1.*moe_grouped_gemm"):
        _base_config(
            num_moe_experts=2, recompute_granularity="selective", recompute_modules=["expert_fc1"]
        )

    config = _base_config(
        num_moe_experts=2,
        moe_grouped_gemm=True,
        recompute_granularity="selective",
        recompute_modules=["expert_fc1"],
    )
    assert config.recompute_modules == ["expert_fc1"]


def test_expert_fc1_recompute_conflicts_with_moe_act_offload():
    with pytest.raises(AssertionError, match="moe_act.*expert_fc1"):
        _base_config(
            num_moe_experts=2,
            moe_grouped_gemm=True,
            recompute_granularity="selective",
            recompute_modules=["expert_fc1"],
            fine_grained_activation_offloading=True,
            offload_modules=["moe_act"],
        )


def test_expert_fc1_recompute_allows_expert_fc1_offload():
    config = _base_config(
        num_moe_experts=2,
        moe_grouped_gemm=True,
        recompute_granularity="selective",
        recompute_modules=["expert_fc1"],
        fine_grained_activation_offloading=True,
        offload_modules=["expert_fc1"],
    )

    assert config.recompute_modules == ["expert_fc1"]
    assert config.offload_modules == ["expert_fc1"]


def test_moe_act_recompute_allows_moe_act_offload():
    config = _base_config(
        num_moe_experts=2,
        moe_grouped_gemm=True,
        recompute_granularity="selective",
        recompute_modules=["moe_act"],
        fine_grained_activation_offloading=True,
        offload_modules=["moe_act"],
    )

    assert config.recompute_modules == ["moe_act"]
    assert config.offload_modules == ["moe_act"]


def test_attention_linear_recompute_is_standard_attention_only():
    with pytest.raises(ValueError, match="standard attention"):
        _base_config(
            multi_latent_attention=True,
            recompute_granularity="selective",
            recompute_modules=["qkv_linear"],
        )


@pytest.mark.parametrize(
    "modules,recompute_input_norm,recompute_pre_mlp_norm",
    [(["attn_norm"], True, False), (["mlp_norm"], False, True), (["layernorm"], True, True)],
)
def test_granular_layernorm_recompute_flags(modules, recompute_input_norm, recompute_pre_mlp_norm):
    Utils.initialize_model_parallel(1, 1)
    try:
        model_parallel_cuda_manual_seed(123)
        config = _base_config(recompute_granularity="selective", recompute_modules=modules)
        layer = TransformerLayer(config, get_gpt_layer_local_submodules())

        assert layer.recompute_input_layernorm is recompute_input_norm
        assert layer.recompute_pre_mlp_layernorm is recompute_pre_mlp_norm
    finally:
        Utils.destroy_model_parallel()


def test_self_attn_recompute_covers_attn_norm_flag():
    Utils.initialize_model_parallel(1, 1)
    try:
        model_parallel_cuda_manual_seed(123)
        config = _base_config(
            recompute_granularity="selective", recompute_modules=["self_attn", "layernorm"]
        )
        layer = TransformerLayer(config, get_gpt_layer_local_submodules())

        assert layer.recompute_self_attn
        assert not layer.recompute_input_layernorm
        assert layer.recompute_pre_mlp_layernorm
    finally:
        Utils.destroy_model_parallel()


def test_attention_linear_recompute_flags():
    Utils.initialize_model_parallel(1, 1)
    try:
        model_parallel_cuda_manual_seed(123)
        config = _base_config(
            recompute_granularity="selective", recompute_modules=["qkv_linear", "attn_proj"]
        )
        attention = SelfAttention(
            config, get_gpt_layer_local_submodules().self_attention.submodules, layer_number=1
        )

        assert attention.recompute_qkv_linear
        assert attention.recompute_attn_proj
    finally:
        Utils.destroy_model_parallel()
