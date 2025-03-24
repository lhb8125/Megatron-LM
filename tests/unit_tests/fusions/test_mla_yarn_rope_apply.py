# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.fusions.fused_mla_yarn_rope_apply import (
    fused_apply_mla_rope_for_kv,
    fused_apply_mla_rope_for_q,
)
from megatron.core.models.common.embeddings import apply_rotary_pos_emb
from megatron.core.models.common.embeddings.yarn_rotary_pos_embedding import YarnRotaryEmbedding
from megatron.core.transformer.transformer_config import TransformerConfig


def dtype_tols(dtype_str):
    if dtype_str == 'float32':
        return dict(rtol=1.0e-6, atol=1.0e-6)
    elif dtype_str == 'float16':
        return dict(rtol=3.0e-3, atol=1.0e-5)
    elif dtype_str == 'bfloat16':
        return dict(rtol=2.0e-2, atol=1.0e-3)
    else:
        raise ValueError(f"Unsuppored dtype ({dtype_str})")


def _test_fused_apply_mla_rope_for_q():
    seqlen = 1024
    batch_size = 2
    num_heads = 32
    q_dim = 128
    emb_dim = 64
    transformer_config = TransformerConfig(
        num_attention_heads=num_heads,
        num_layers=1,
        rotary_interleaved=False,
        multi_latent_attention=True,
    )
    yarn_rope = YarnRotaryEmbedding(emb_dim, original_max_position_embeddings=seqlen)
    freqs, mscale = yarn_rope(seqlen, 0)
    cos = (torch.cos(freqs) * mscale).to(torch.bfloat16)
    sin = (torch.sin(freqs) * mscale).to(torch.bfloat16)

    pytorch_fwd_input = torch.rand(
        (seqlen, batch_size, num_heads, q_dim + emb_dim), dtype=torch.bfloat16, device='cuda'
    )
    pytorch_fwd_input.requires_grad_(True)
    pytorch_bwd_input = torch.rand(
        (seqlen, batch_size, num_heads, q_dim + emb_dim), dtype=torch.bfloat16, device='cuda'
    )

    fused_fwd_input = pytorch_fwd_input.detach()
    fused_fwd_input.requires_grad_(True)
    fused_bwd_input = pytorch_bwd_input.detach()

    no_pe, pe = torch.split(pytorch_fwd_input, [q_dim, emb_dim], dim=-1)
    pe_output = apply_rotary_pos_emb(pe, freqs, transformer_config, mscale=mscale)
    pytorch_output = torch.concat([no_pe, pe_output], dim=-1)
    pytorch_output.backward(pytorch_bwd_input, retain_graph=True)

    fused_output = fused_apply_mla_rope_for_q(fused_fwd_input, cos, sin, q_dim)
    fused_output.backward(fused_bwd_input, retain_graph=True)

    tols = dtype_tols('bfloat16')
    torch.testing.assert_close(
        pytorch_output.float(), fused_output.float(), msg=f"Mismatch in fwd", **tols
    )
    torch.testing.assert_close(
        pytorch_fwd_input.grad.float(), fused_fwd_input.grad.float(), msg=f"Mismatch in bwd", **tols
    )


def _test_fused_apply_mla_rope_for_kv():
    seqlen = 1024
    batch_size = 2
    num_heads = 32
    k_dim = 128
    v_dim = 128
    emb_dim = 64
    transformer_config = TransformerConfig(
        num_attention_heads=num_heads,
        num_layers=1,
        rotary_interleaved=False,
        multi_latent_attention=True,
    )
    yarn_rope = YarnRotaryEmbedding(emb_dim, original_max_position_embeddings=seqlen)
    freqs, mscale = yarn_rope(seqlen, 0)
    cos = (torch.cos(freqs) * mscale).to(torch.bfloat16)
    sin = (torch.sin(freqs) * mscale).to(torch.bfloat16)

    pytorch_fwd_kv_input = torch.rand(
        (seqlen, batch_size, num_heads, k_dim + v_dim), dtype=torch.bfloat16, device='cuda'
    )
    pytorch_fwd_kv_input.requires_grad_(True)
    pytorch_fwd_emb_input = torch.rand(
        (seqlen, batch_size, 1, emb_dim), dtype=torch.bfloat16, device='cuda'
    )
    pytorch_fwd_emb_input.requires_grad_(True)
    pytorch_bwd_k_input = torch.rand(
        (seqlen, batch_size, num_heads, k_dim + emb_dim), dtype=torch.bfloat16, device='cuda'
    )
    pytorch_bwd_v_input = torch.rand(
        (seqlen, batch_size, num_heads, v_dim), dtype=torch.bfloat16, device='cuda'
    )

    fused_fwd_kv_input = pytorch_fwd_kv_input.detach()
    fused_fwd_kv_input.requires_grad_(True)
    fused_fwd_emb_input = pytorch_fwd_emb_input.detach()
    fused_fwd_emb_input.requires_grad_(True)
    fused_bwd_k_input = pytorch_bwd_k_input.detach()
    fused_bwd_v_input = pytorch_bwd_v_input.detach()

    pe_output = apply_rotary_pos_emb(
        pytorch_fwd_emb_input, freqs, transformer_config, mscale=mscale
    )
    pe_output = pe_output.expand(-1, -1, num_heads, -1)
    k, pytorch_v_output = torch.split(pytorch_fwd_kv_input, [k_dim, v_dim], dim=-1)
    pytorch_k_output = torch.concat([k, pe_output], dim=-1)
    torch.autograd.backward(
        (pytorch_k_output, pytorch_v_output), (pytorch_bwd_k_input, pytorch_bwd_v_input)
    )

    fused_k_output, fused_v_output = fused_apply_mla_rope_for_kv(
        fused_fwd_kv_input, fused_fwd_emb_input, cos, sin, emb_dim, k_dim, v_dim
    )
    torch.autograd.backward(
        (fused_k_output, fused_v_output), (fused_bwd_k_input, fused_bwd_v_input)
    )

    tols = dtype_tols('bfloat16')
    torch.testing.assert_close(
        pytorch_k_output.float(), fused_k_output.float(), msg=f"Mismatch in k fwd", **tols
    )
    torch.testing.assert_close(
        pytorch_v_output.float(), fused_v_output.float(), msg=f"Mismatch in v fwd", **tols
    )
    torch.testing.assert_close(
        pytorch_fwd_kv_input.grad.float(),
        fused_fwd_kv_input.grad.float(),
        msg=f"Mismatch in kv bwd",
        **tols,
    )
    torch.testing.assert_close(
        pytorch_fwd_emb_input.grad.float(),
        fused_fwd_emb_input.grad.float(),
        msg=f"Mismatch in emb bwd",
        **tols,
    )


@pytest.mark.experimental
class TestFusedApplyMLARope:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    def test_forward_backward_for_q(self):
        _test_fused_apply_mla_rope_for_q()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    def test_forward_backward_for_kv(self):
        _test_fused_apply_mla_rope_for_kv()
