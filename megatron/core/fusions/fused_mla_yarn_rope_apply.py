import torch
import triton
import triton.language as tl

from megatron.core.utils import experimental_fn


@triton.jit
def rotary_fwd_q_kernel(
    Q, COS, SIN, offset, emb_dim: tl.constexpr, stride_x_seqlen, stride_x_batch, stride_x_nheads
):  # pylint: disable=missing-function-docstring
    pid_m = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)
    pid_head = tl.program_id(axis=2)

    cos_left = tl.load(COS + pid_m * emb_dim + tl.arange(0, emb_dim // 2))
    sin_left = tl.load(SIN + pid_m * emb_dim + tl.arange(0, emb_dim // 2))
    cos_right = tl.load(COS + pid_m * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2))
    sin_right = tl.load(SIN + pid_m * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2))

    Q = Q + pid_m * stride_x_seqlen + pid_batch * stride_x_batch + pid_head * stride_x_nheads

    X = Q + offset
    x = tl.load(X + tl.arange(0, emb_dim))
    # x1 = t[..., 0::2], x2 = t[..., 1::2]
    x_1, x_2 = x.reshape(emb_dim // 2, 2).split()

    x_left = x_1 * cos_left - x_2 * sin_left
    x_right = x_2 * cos_right + x_1 * sin_right
    tl.store(X + tl.arange(0, emb_dim // 2), x_left)
    tl.store(X + emb_dim // 2 + tl.arange(0, emb_dim // 2), x_right)


@triton.jit
def rotary_bwd_q_kernel(
    DO, COS, SIN, offset, emb_dim: tl.constexpr, stride_x_seqlen, stride_x_batch, stride_x_nheads
):  # pylint: disable=missing-function-docstring
    pid_m = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)
    pid_head = tl.program_id(axis=2)

    cos_left = tl.load(COS + pid_m * emb_dim + tl.arange(0, emb_dim // 2))
    sin_left = tl.load(SIN + pid_m * emb_dim + tl.arange(0, emb_dim // 2))
    cos_right = tl.load(COS + pid_m * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2))
    sin_right = tl.load(SIN + pid_m * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2))

    DO = DO + pid_m * stride_x_seqlen + pid_batch * stride_x_batch + pid_head * stride_x_nheads

    X = DO + offset
    x_left = tl.load(X + tl.arange(0, emb_dim // 2))
    x_right = tl.load(X + emb_dim // 2 + tl.arange(0, emb_dim // 2))

    x_1 = x_left * cos_left + x_right * sin_right
    x_2 = -x_left * sin_left + x_right * cos_right
    x_out = tl.interleave(x_1, x_2)
    tl.store(X + tl.arange(0, emb_dim), x_out)


class ApplyMLARotaryEmbQ(torch.autograd.Function):  # pylint: disable=missing-class-docstring
    @staticmethod
    def forward(ctx, q, cos, sin, offset, rotary_interleaved=False):
        """
        Arguments:
            x: [sbhd]
            cos/sin: [s11d]
        """
        assert not rotary_interleaved
        seqlen, batch, nheads, headdim = q.shape
        assert cos.is_contiguous()
        assert sin.is_contiguous()
        emb_dim = headdim - offset
        assert emb_dim == 64

        grid = (seqlen, batch, nheads)
        rotary_fwd_q_kernel[grid](
            q, cos, sin, offset, emb_dim, q.stride(0), q.stride(1), q.stride(2)
        )
        ctx.save_for_backward(cos, sin)
        ctx.offset = offset
        ctx.rotary_interleaved = rotary_interleaved
        return q

    @staticmethod
    def backward(ctx, grad):
        cos, sin = ctx.saved_tensors
        seqlen, batch, nheads, headdim = grad.shape
        grid = (seqlen, batch, nheads)
        rotary_bwd_q_kernel[grid](
            grad,
            cos,
            sin,
            ctx.offset,
            headdim - ctx.offset,
            grad.stride(0),
            grad.stride(1),
            grad.stride(2),
        )
        return grad, None, None, None, None


@experimental_fn(introduced_with_version="0.12.0")
def fused_apply_mla_rope_for_q(t, cos, sin, offset, rotary_interleaved=False):
    """Fused apply YARN RoPE for MLA's query."""
    return ApplyMLARotaryEmbQ.apply(t, cos, sin, offset, rotary_interleaved)


@triton.jit
def rotary_fwd_kv_kernel(
    KV,
    K_POS_EMB,
    O_KEY,
    O_VALUE,
    COS,
    SIN,
    emb_dim: tl.constexpr,
    k_dim: tl.constexpr,
    v_dim: tl.constexpr,
    stride_kv_seqlen,
    stride_kv_batch,
    stride_kv_nheads,
    stride_emb_seqlen,
    stride_emb_batch,
    stride_k_seqlen,
    stride_k_batch,
    stride_k_nheads,
    stride_v_seqlen,
    stride_v_batch,
    stride_v_nheads,
):  # pylint: disable=missing-function-docstring
    pid_m = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)
    pid_head = tl.program_id(axis=2)

    cos_left = tl.load(COS + pid_m * emb_dim + tl.arange(0, emb_dim // 2))
    sin_left = tl.load(SIN + pid_m * emb_dim + tl.arange(0, emb_dim // 2))
    cos_right = tl.load(COS + pid_m * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2))
    sin_right = tl.load(SIN + pid_m * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2))

    EMB = K_POS_EMB + pid_m * stride_emb_seqlen + pid_batch * stride_emb_batch
    emb = tl.load(EMB + tl.arange(0, emb_dim))
    # x1 = t[..., 0::2], x2 = t[..., 1::2]
    x_1, x_2 = emb.reshape(emb_dim // 2, 2).split()

    x_left = x_1 * cos_left - x_2 * sin_left
    x_right = x_2 * cos_right + x_1 * sin_right

    KV_ptr = (
        KV + pid_m * stride_kv_seqlen + pid_batch * stride_kv_batch + pid_head * stride_kv_nheads
    )
    k = tl.load(KV_ptr + tl.arange(0, k_dim))
    v = tl.load(KV_ptr + k_dim + tl.arange(0, v_dim))

    K_ptr = (
        O_KEY + pid_m * stride_k_seqlen + pid_batch * stride_k_batch + pid_head * stride_k_nheads
    )
    V_ptr = (
        O_VALUE + pid_m * stride_v_seqlen + pid_batch * stride_v_batch + pid_head * stride_v_nheads
    )

    tl.store(K_ptr + k_dim + tl.arange(0, emb_dim // 2), x_left)
    tl.store(K_ptr + k_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2), x_right)
    tl.store(K_ptr + tl.arange(0, k_dim), k)
    tl.store(V_ptr + tl.arange(0, v_dim), v)


@triton.jit
def rotary_bwd_kv_kernel(
    dK,
    dV,
    dKV,
    dEMB,
    COS,
    SIN,
    nheads: tl.constexpr,
    emb_dim: tl.constexpr,
    k_dim: tl.constexpr,
    v_dim: tl.constexpr,
    stride_dk_seqlen,
    stride_dk_batch,
    stride_dk_nheads,
    stride_dv_seqlen,
    stride_dv_batch,
    stride_dv_nheads,
    stride_dkv_seqlen,
    stride_dkv_batch,
    stride_dkv_nheads,
    stride_demb_seqlen,
    stride_demb_batch,
):  # pylint: disable=missing-function-docstring
    pid_m = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)
    pid_head = tl.program_id(axis=2)

    dK_ptr = (
        dK + pid_m * stride_dk_seqlen + pid_batch * stride_dk_batch + pid_head * stride_dk_nheads
    )
    dk = tl.load(dK_ptr + tl.arange(0, k_dim))
    dV_ptr = (
        dV + pid_m * stride_dv_seqlen + pid_batch * stride_dv_batch + pid_head * stride_dv_nheads
    )
    dv = tl.load(dV_ptr + tl.arange(0, v_dim))
    dKV_ptr = (
        dKV
        + pid_m * stride_dkv_seqlen
        + pid_batch * stride_dkv_batch
        + pid_head * stride_dkv_nheads
    )
    tl.store(dKV_ptr + tl.arange(0, k_dim), dk)
    tl.store(dKV_ptr + k_dim + tl.arange(0, v_dim), dv)

    cos_left = tl.load(COS + pid_m * emb_dim + tl.arange(0, emb_dim // 2))
    sin_left = tl.load(SIN + pid_m * emb_dim + tl.arange(0, emb_dim // 2))
    cos_right = tl.load(COS + pid_m * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2))
    sin_right = tl.load(SIN + pid_m * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2))

    if pid_head == 0:
        x_left_accum = tl.zeros((emb_dim // 2,), dtype=tl.float32)
        x_right_accum = tl.zeros((emb_dim // 2,), dtype=tl.float32)
        for i in tl.static_range(nheads):
            dK_ptr = (
                dK + pid_m * stride_dk_seqlen + pid_batch * stride_dk_batch + i * stride_dk_nheads
            )
            x_left = tl.load(dK_ptr + k_dim + tl.arange(0, emb_dim // 2))
            x_right = tl.load(dK_ptr + k_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2))
            x_left_accum += x_left
            x_right_accum += x_right
        x_left_accum = x_left_accum.to(dEMB.dtype.element_ty)
        x_right_accum = x_right_accum.to(dEMB.dtype.element_ty)
        x_1 = x_left_accum * cos_left + x_right_accum * sin_right
        x_2 = -x_left_accum * sin_left + x_right_accum * cos_right
        x_out = tl.interleave(x_1, x_2)
        dEMB_ptr = dEMB + pid_m * stride_demb_seqlen + pid_batch * stride_demb_batch
        tl.store(dEMB_ptr + tl.arange(0, emb_dim), x_out)


class ApplyMLARotaryEmbKV(torch.autograd.Function):  # pylint: disable=missing-class-docstring
    @staticmethod
    def forward(ctx, kv, k_pos_emb, cos, sin, emb_dim, k_dim, v_dim, rotary_interleaved=False):
        """
        Arguments:
            x: [sbhd]
            cos/sin: [s11d]
        """
        assert not rotary_interleaved
        seqlen, batch, nheads, _ = kv.shape
        assert kv.stride(-1) == 1
        assert k_pos_emb.stride(-1) == 1
        assert cos.is_contiguous()
        assert sin.is_contiguous()

        o_key = kv.new_empty(seqlen, batch, nheads, emb_dim + k_dim)
        o_value = kv.new_empty(seqlen, batch, nheads, v_dim)

        grid = (seqlen, batch, nheads)
        rotary_fwd_kv_kernel[grid](
            kv,
            k_pos_emb,
            o_key,
            o_value,
            cos,
            sin,
            emb_dim,
            k_dim,
            v_dim,
            kv.stride(0),
            kv.stride(1),
            kv.stride(2),
            k_pos_emb.stride(0),
            k_pos_emb.stride(1),
            o_key.stride(0),
            o_key.stride(1),
            o_key.stride(2),
            o_value.stride(0),
            o_value.stride(1),
            o_value.stride(2),
        )
        ctx.save_for_backward(cos, sin)
        ctx.rotary_interleaved = rotary_interleaved
        ctx.emb_dim = emb_dim
        ctx.k_dim = k_dim
        ctx.v_dim = v_dim
        return o_key, o_value

    @staticmethod
    def backward(ctx, dk, dv):
        cos, sin = ctx.saved_tensors
        seqlen, batch, nheads, _ = dk.shape

        d_kv = dk.new_empty(seqlen, batch, nheads, ctx.k_dim + ctx.v_dim)
        d_emb = dk.new_empty(seqlen, batch, 1, ctx.emb_dim)

        grid = (seqlen, batch, nheads)
        rotary_bwd_kv_kernel[grid](
            dk,
            dv,
            d_kv,
            d_emb,
            cos,
            sin,
            nheads,
            ctx.emb_dim,
            ctx.k_dim,
            ctx.v_dim,
            dk.stride(0),
            dk.stride(1),
            dk.stride(2),
            dv.stride(0),
            dv.stride(1),
            dv.stride(2),
            d_kv.stride(0),
            d_kv.stride(1),
            d_kv.stride(2),
            d_emb.stride(0),
            d_emb.stride(1),
        )
        return d_kv, d_emb, None, None, None, None, None, None


@experimental_fn(introduced_with_version="0.12.0")
def fused_apply_mla_rope_for_kv(
    kv, k_pos_emb, cos, sin, pos_dim, k_dim, v_dim, rotary_interleaved=False
):
    """Fused apply YARN RoPE for MLA's key and value."""
    return ApplyMLARotaryEmbKV.apply(
        kv, k_pos_emb, cos, sin, pos_dim, k_dim, v_dim, rotary_interleaved
    )
