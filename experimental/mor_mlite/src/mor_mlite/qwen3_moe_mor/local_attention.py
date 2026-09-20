"""Native BF16 FFA core for regular (CP=1) packed Qwen attention.

Magi's public functional API explicitly supports local execution without its
distributed dispatch/runtime. Keep the upstream distributed CP>1 guard intact.
QKV, QK norm, original-position RoPE and output projection remain native MLite.
"""

from __future__ import annotations

import torch
from torch import nn


def _local_ffa(*args, **kwargs):
    from magi_attention.functional import flex_flash_attn_func

    return flex_flash_attn_func(*args, **kwargs)


class LocalMagiAttention(nn.Module):
    """Parameter-free packed causal self-attention; never used for CP>1."""

    def __init__(self, *, cp_size: int, deterministic: bool):
        super().__init__()
        if cp_size != 1:
            raise ValueError("local FFA requires CP=1; use distributed Magi for CP>1")
        self.deterministic = deterministic

    def forward(
        self,
        q,
        k,
        v,
        *,
        qkv_format="thd",
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        cu_seqlens_q_padded=None,
        cu_seqlens_kv_padded=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        attn_mask_type="padding_causal",
        core_attention_bias_type="no_bias",
    ):
        if (
            qkv_format != "thd"
            or attn_mask_type != "padding_causal"
            or core_attention_bias_type != "no_bias"
        ):
            raise ValueError("local FFA supports THD causal self-attention without bias only")
        if any(x.ndim != 3 or x.dtype != torch.bfloat16 or x.device != q.device for x in (q, k, v)):
            raise ValueError("local FFA requires same-device BF16 THD Q/K/V")
        if (
            k.shape != v.shape
            or q.shape[0] != k.shape[0]
            or q.shape[2] != k.shape[2]
            or not k.shape[1]
            or q.shape[1] % k.shape[1]
        ):
            raise ValueError(
                "local FFA requires aligned self-attention Q/K/V and divisible GQA heads"
            )
        cu = cu_seqlens_q
        if (
            cu is None
            or cu_seqlens_kv is None
            or cu.ndim != 1
            or cu.numel() < 2
            or cu.dtype != torch.int32
            or not torch.equal(cu, cu_seqlens_kv)
        ):
            raise ValueError("local FFA requires equal int32 self-attention cumulative lengths")
        padded = cu if cu_seqlens_q_padded is None else cu_seqlens_q_padded
        kv_padded = cu if cu_seqlens_kv_padded is None else cu_seqlens_kv_padded
        if (
            padded.shape != cu.shape
            or padded.dtype != torch.int32
            or not torch.equal(padded, kv_padded)
        ):
            raise ValueError("local FFA requires equal Q/KV padded cumulative lengths")
        # One metadata transfer per packed batch, never one CUDA read per token.
        lengths = (cu[1:] - cu[:-1]).detach().cpu().tolist()
        starts = padded.detach().cpu().tolist()
        if (
            int(cu[0]) != 0
            or starts[0] != 0
            or starts[-1] != q.shape[0]
            or any(
                length < 0 or starts[i + 1] - starts[i] < length for i, length in enumerate(lengths)
            )
        ):
            raise ValueError("local FFA cumulative lengths do not cover the physical token buffer")
        max_length = max(lengths, default=0)
        if any(value is not None and value < max_length for value in (max_seqlen_q, max_seqlen_kv)):
            raise ValueError("local FFA max sequence length is smaller than a real sequence")
        if not sum(lengths):
            return q * 0 + (k.sum() + v.sum()) * 0
        compact = starts != cu.detach().cpu().tolist()
        if compact:
            indices = torch.cat(
                [
                    torch.arange(start, start + length, device=q.device)
                    for start, length in zip(starts, lengths)
                ]
            )
            inputs = [x.index_select(0, indices) for x in (q, k, v)]
        else:
            inputs = [q, k, v]
        ranges = torch.stack((cu[:-1], cu[1:]), dim=1).to(q.device)
        ranges = ranges[(cu[1:] > cu[:-1]).to(q.device)].contiguous()
        output, _ = _local_ffa(
            *(x.contiguous() for x in inputs),
            q_ranges=ranges,
            k_ranges=ranges,
            attn_type_map=torch.ones(ranges.shape[0], dtype=torch.int32, device=q.device),
            deterministic=self.deterministic,
            max_seqlen_q=max_length,
        )
        if compact:
            output = torch.zeros_like(q).index_copy(0, indices, output)
        return output


__all__ = ["LocalMagiAttention"]
