# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.transformer.moe.fused_aux_loss import fused_calculate_aux_loss
from megatron.core.transformer.moe.fused_router import fused_topk_softmax_with_capacity
from megatron.core.transformer.moe.moe_utils import (
    switch_load_balancing_loss_func,
    topk_softmax_with_capacity,
)
from megatron.core.utils import is_triton_min_version


class TestRouterFusion:
    def setup_method(self):
        torch.manual_seed(1234)

    def teardown_method(self, _):
        pass

    @pytest.mark.skipif(
        not is_triton_min_version("2.2.0"),
        reason="Only triton>=2.2.0 supports MoE router aux loss fusion.",
    )
    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("num_tokens", [1023, 2049])
    @pytest.mark.parametrize("num_experts", [7, 9])
    @pytest.mark.parametrize("topk", [2])
    @pytest.mark.parametrize("capacity_factor", [1.0, 0.5])
    @pytest.mark.parametrize("pad_to_capacity", [True, False])
    @pytest.mark.parametrize("drop_policy", ["probs", "position"])
    @pytest.mark.parametrize("use_pre_softmax", [True, False])
    def test_token_drop_router(
        self,
        dtype,
        num_tokens,
        num_experts,
        topk,
        capacity_factor,
        pad_to_capacity,
        drop_policy,
        use_pre_softmax,
    ):
        logits = torch.rand((num_tokens, num_experts), dtype=dtype, device="cuda") - torch.rand(
            (num_tokens, num_experts), dtype=dtype, device="cuda"
        )
        logits.requires_grad_()

        probs0, routing_map0, tokens_per_expert0 = topk_softmax_with_capacity(
            logits,
            topk,
            capacity_factor=capacity_factor,
            pad_to_capacity=pad_to_capacity,
            drop_policy=drop_policy,
            use_pre_softmax=use_pre_softmax,
        )
        scores0 = torch.softmax(logits, dim=-1, dtype=torch.float32)
        loss0 = torch.sum(probs0 * scores0)
        loss0.backward()
        logits_grad0 = logits.grad.detach().clone()

        logits.grad = None
        probs1, scores1, routing_map1, tokens_per_expert1 = fused_topk_softmax_with_capacity(
            logits,
            topk,
            capacity_factor=capacity_factor,
            pad_to_capacity=pad_to_capacity,
            drop_policy=drop_policy,
            use_pre_softmax=use_pre_softmax,
        )
        loss1 = torch.sum(probs1 * scores1)
        loss1.backward()
        logits_grad1 = logits.grad.detach().clone()

        torch.testing.assert_close(probs0, probs1)
        torch.testing.assert_close(scores0, scores1)
        assert torch.equal(routing_map0, routing_map1)
        assert torch.equal(tokens_per_expert0, tokens_per_expert1)
        torch.testing.assert_close(logits_grad0, logits_grad1)

    @pytest.mark.skipif(
        not is_triton_min_version("2.2.0"),
        reason="Only triton>=2.2.0 supports MoE router aux loss fusion.",
    )
    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("num_tokens", [1023, 2049])
    @pytest.mark.parametrize("num_experts", [8])
    @pytest.mark.parametrize("topk", [2])
    @pytest.mark.parametrize("use_pre_softmax", [True, False])
    def test_dropless_router(self, dtype, num_tokens, num_experts, topk, use_pre_softmax):
        logits = torch.rand((num_tokens, num_experts), dtype=dtype, device="cuda") - torch.rand(
            (num_tokens, num_experts), dtype=dtype, device="cuda"
        )
        logits.requires_grad_()

        probs0, routing_map0, tokens_per_expert0 = topk_softmax_with_capacity(
            logits, topk, use_pre_softmax=use_pre_softmax
        )
        scores0 = torch.softmax(logits, dim=-1, dtype=torch.float32)
        loss0 = torch.sum(probs0 * scores0)
        loss0.backward()
        logits_grad0 = logits.grad.detach().clone()

        logits.grad = None
        probs1, scores1, routing_map1, tokens_per_expert1 = fused_topk_softmax_with_capacity(
            logits, topk, use_pre_softmax=use_pre_softmax
        )
        loss1 = torch.sum(probs1 * scores1)
        loss1.backward()
        logits_grad1 = logits.grad.detach().clone()

        torch.testing.assert_close(probs0, probs1)
        torch.testing.assert_close(scores0, scores1)
        assert torch.equal(routing_map0, routing_map1)
        assert torch.equal(tokens_per_expert0, tokens_per_expert1)
        torch.testing.assert_close(logits_grad0, logits_grad1)

    @pytest.mark.skipif(
        not is_triton_min_version("2.2.0"),
        reason="Only triton>=2.2.0 supports MoE router aux loss fusion.",
    )
    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("num_tokens", [1023, 2049])
    @pytest.mark.parametrize("num_experts", [8])
    @pytest.mark.parametrize("topk", [2])
    def test_fused_aux_loss(self, dtype, num_tokens, num_experts, topk):
        logits = torch.rand((num_tokens, num_experts), dtype=dtype, device="cuda") - torch.rand(
            (num_tokens, num_experts), dtype=dtype, device="cuda"
        )
        probs0, routing_map0, tokens_per_expert0 = topk_softmax_with_capacity(logits, topk)
        scores0 = torch.softmax(logits, dim=-1, dtype=torch.float32)
        scores1 = scores0.detach().clone()
        tokens_per_expert1 = tokens_per_expert0.detach().clone()

        # unfused
        scores0.requires_grad_()
        loss0, _ = switch_load_balancing_loss_func(scores0, tokens_per_expert0, 2, 1e-2)
        loss0.backward()
        scores_grad0 = scores0.grad.detach().clone()

        # fused
        scores1.requires_grad_()
        loss1, _ = fused_calculate_aux_loss(scores1, tokens_per_expert1, 2, 1e-2, 1)
        loss1.backward()
        scores_grad1 = scores1.grad.detach().clone()

        torch.testing.assert_close(scores0, scores1)
        torch.testing.assert_close(scores_grad0, scores_grad1)
