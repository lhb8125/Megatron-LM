"""Adapter contracts; real FFA forward/backward parity runs separately on EOS."""

import pytest

torch = pytest.importorskip("torch")

from mor_mlite.qwen3_moe_mor import local_attention


def _naive(q, k, v):
    q, k, v = (x.transpose(0, 1).float() for x in (q, k, v))
    repeats = q.shape[0] // k.shape[0]
    k, v = k.repeat_interleave(repeats, 0), v.repeat_interleave(repeats, 0)
    scores = q @ k.transpose(-1, -2) * q.shape[-1] ** -0.5
    mask = torch.ones(q.shape[1], k.shape[1], dtype=torch.bool).triu(1)
    return (scores.masked_fill(mask, -torch.inf).softmax(-1) @ v).transpose(0, 1).bfloat16()


@pytest.mark.parametrize("padded", [False, True])
def test_local_ffa_packing_output_and_all_input_gradients(monkeypatch, padded):
    calls = []

    def fake_ffa(q, k, v, **kwargs):
        calls.append(kwargs)
        assert torch.equal(kwargs["q_ranges"], kwargs["k_ranges"])
        outputs = [
            _naive(*(x[start:end] for x in (q, k, v))) for start, end in kwargs["q_ranges"].tolist()
        ]
        return torch.cat(outputs), None

    monkeypatch.setattr(local_attention, "_local_ffa", fake_ffa)
    torch.manual_seed(812)
    total = 8 if padded else 5
    native = [
        torch.randn(total, heads, 16, dtype=torch.bfloat16).requires_grad_() for heads in (4, 2, 2)
    ]
    reference = [x.detach().clone().requires_grad_() for x in native]
    cu = torch.tensor([0, 3, 3, 5], dtype=torch.int32)
    pad = torch.tensor([0, 4, 4, 8], dtype=torch.int32) if padded else cu
    module = local_attention.LocalMagiAttention(cp_size=1, deterministic=True)
    assert list(module.parameters()) == []
    assert module.state_dict() == {}
    output = module(
        *native,
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        cu_seqlens_q_padded=pad,
        cu_seqlens_kv_padded=pad,
        max_seqlen_q=3,
        max_seqlen_kv=3,
    )
    starts = [0, 4 if padded else 3]
    expected = torch.zeros_like(output)
    for start, length in zip(starts, [3, 2], strict=True):
        expected[start : start + length] = _naive(*(x[start : start + length] for x in reference))
    assert torch.equal(output, expected)
    grad = torch.randn_like(output)
    output.backward(grad)
    expected.backward(grad)
    assert all(torch.equal(x.grad, y.grad) for x, y in zip(native, reference, strict=True))
    assert len(calls) == 1
    assert calls[0]["deterministic"] is True
    assert calls[0]["max_seqlen_q"] == 3
    assert torch.equal(calls[0]["q_ranges"], torch.tensor([[0, 3], [3, 5]], dtype=torch.int32))


def test_local_ffa_rejects_distributed_scope():
    with pytest.raises(ValueError, match="requires CP=1"):
        local_attention.LocalMagiAttention(cp_size=2, deterministic=True)


@pytest.mark.parametrize(
    "failure",
    [
        "dtype",
        "mask",
        "qkv_format",
        "bias",
        "kv_lengths",
        "physical_end",
        "max_length",
        "nonmonotonic",
    ],
)
def test_local_ffa_rejects_invalid_contract_before_kernel(monkeypatch, failure):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid metadata reached the FFA kernel")

    monkeypatch.setattr(local_attention, "_local_ffa", forbidden)
    q = torch.ones(5, 2, 16, dtype=torch.bfloat16)
    cu = torch.tensor([0, 3, 5], dtype=torch.int32)
    kwargs = {"cu_seqlens_q": cu, "cu_seqlens_kv": cu}
    if failure == "dtype":
        q = q.float()
    if failure == "mask":
        kwargs["attn_mask_type"] = "no_mask"
    if failure == "qkv_format":
        kwargs["qkv_format"] = "sbhd"
    if failure == "bias":
        kwargs["core_attention_bias_type"] = "post_scale_bias"
    if failure == "kv_lengths":
        kwargs["cu_seqlens_kv"] = torch.tensor([0, 2, 5], dtype=torch.int32)
    if failure in {"physical_end", "nonmonotonic"}:
        bad = torch.tensor([0, 6, 5] if failure == "nonmonotonic" else [0, 3, 6], dtype=torch.int32)
        kwargs.update(cu_seqlens_q=bad, cu_seqlens_kv=bad)
    if failure == "max_length":
        kwargs["max_seqlen_q"] = 2
    with pytest.raises(ValueError):
        local_attention.LocalMagiAttention(cp_size=1, deterministic=True)(q, q, q, **kwargs)


def test_local_ffa_empty_real_batch_keeps_zero_gradients(monkeypatch):
    monkeypatch.setattr(local_attention, "_local_ffa", lambda *a, **kw: pytest.fail("empty kernel"))
    inputs = [torch.randn(4, 2, 16, dtype=torch.bfloat16).requires_grad_() for _ in range(3)]
    cu = torch.tensor([0, 0], dtype=torch.int32)
    pad = torch.tensor([0, 4], dtype=torch.int32)
    output = local_attention.LocalMagiAttention(cp_size=1, deterministic=True)(
        *inputs,
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        cu_seqlens_q_padded=pad,
        cu_seqlens_kv_padded=pad,
    )
    output.sum().backward()
    assert torch.count_nonzero(output) == 0
    assert all(x.grad is not None and torch.count_nonzero(x.grad) == 0 for x in inputs)
