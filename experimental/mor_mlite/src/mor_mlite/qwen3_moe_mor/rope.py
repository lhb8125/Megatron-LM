"""Original-position RoPE adapter for active-token Qwen3 GQA.

Pinned MLite's regular packed-Qwen GQA constructs contiguous positions from
``cu_seqlens`` and ignores explicit ``position_ids``.  Active MoR tokens are
causally sorted but sparse in the original sequence, so this subclass indexes
the same rotary table with their original positions before calling the native
TE core attention.  MagiAttention already consumes dispatched position IDs and
is delegated unchanged.
"""

from __future__ import annotations

import torch
from megatron.lite.primitive.modules.gqa import GQAttention
from megatron.lite.primitive.parallel import all_gather_last_dim_with_grad_reduce
from megatron.lite.primitive.utils.rope import _apply_rotary_pos_emb_bshd

_KEPT_PSP_FIELDS = (
    "qkv_format",
    "cu_seqlens_q",
    "cu_seqlens_kv",
    "cu_seqlens_q_padded",
    "cu_seqlens_kv_padded",
    "max_seqlen_q",
    "max_seqlen_kv",
)


class OriginalPositionGQAttention(GQAttention):
    """GQA that honors sparse original positions for regular packed THD."""

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        packed_seq_params=None,
    ) -> torch.Tensor:
        # Native MLite already has the correct position-indexed Magi branch and
        # the Qwen3-MoE integration never enables MRoPE.  Dense/non-packed
        # execution also remains on the native implementation.
        if (
            position_ids is None
            or packed_seq_params is None
            or self.attention_backend == "magi"
            or self._mrope_section is not None
        ):
            return super().forward(
                x, position_ids=position_ids, packed_seq_params=packed_seq_params
            )

        qkv = self.qkv(x)
        if self.qkv_lora is not None:
            qkv = qkv + self.qkv_lora(self._qkv_lora_input(x))
        if self._replicate_kv:
            qkv = all_gather_last_dim_with_grad_reduce(qkv, self.ps.tp_group)
        q, gate, k, v = self._split_qkv(qkv)
        q, k, v = q.squeeze(1), k.squeeze(1), v.squeeze(1)
        q, k = self.q_norm(q), self.k_norm(k)

        original_positions = position_ids.reshape(-1).to(device=q.device, dtype=torch.long)
        if original_positions.numel() != q.size(0):
            raise ValueError(
                "original position_ids must have one entry per active THD token: "
                f"{original_positions.numel()} != {q.size(0)}"
            )
        if original_positions.numel() and int(original_positions.min().item()) < 0:
            raise ValueError("active original position_ids must be non-negative")

        if self._use_fp32_rope:
            original_dtype = q.dtype
            q, k = q.float(), k.float()
        table_length = int(original_positions.max().item()) + 1 if original_positions.numel() else 0
        frequencies = self.rotary(table_length, packed_seq=True)
        if isinstance(frequencies, tuple):
            frequencies, mscale = frequencies
        else:
            mscale = 1.0
        local_frequencies = frequencies.index_select(0, original_positions)
        q = _apply_rotary_pos_emb_bshd(
            q[:, None],
            local_frequencies,
            rotary_interleaved=False,
            mscale=mscale,
        ).squeeze(1)
        k = _apply_rotary_pos_emb_bshd(
            k[:, None],
            local_frequencies,
            rotary_interleaved=False,
            mscale=mscale,
        ).squeeze(1)
        if self._use_fp32_rope:
            q, k = q.to(original_dtype), k.to(original_dtype)

        psp_kwargs = {
            key: getattr(packed_seq_params, key)
            for key in _KEPT_PSP_FIELDS
            if getattr(packed_seq_params, key, None) is not None
        }
        attention_output = self.core_attn(
            q,
            k,
            v,
            core_attention_bias_type="no_bias",
            attn_mask_type="padding_causal",
            **psp_kwargs,
        ).reshape(q.size(0), 1, -1)
        if gate is not None:
            gate_fp32 = gate.reshape(attention_output.shape).float().sigmoid()
            attention_output = (attention_output.float() * gate_fp32).to(attention_output.dtype)
        output = self.proj(attention_output)
        if self.proj_lora is not None:
            output = output + self.proj_lora(attention_output)
        return output


def install_original_position_rope(attention: GQAttention) -> None:
    """Upgrade an already-built native GQA module without reallocating weights."""

    if isinstance(attention, OriginalPositionGQAttention):
        return
    if not isinstance(attention, GQAttention):
        raise TypeError("original-position RoPE can only be installed on MLite GQAttention")
    # Both classes have the same Python object layout.  Rebinding the class
    # preserves every registered QKV/projection parameter and its state-dict key.
    attention.__class__ = OriginalPositionGQAttention


__all__ = ["OriginalPositionGQAttention", "install_original_position_rope"]
