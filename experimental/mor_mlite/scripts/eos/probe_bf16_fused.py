"""Independent attention forward/backward and CUDA-library provenance diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import transformer_engine.pytorch as te
from probe_attention_boundaries import metrics


def naive_attention(q, k, v):
    """Qwen GQA math evaluated in FP32, not a BF16 eager-rounding replica.

    Formula source: transformers v4.57.1, Qwen team/HuggingFace,
    src/transformers/models/qwen3_moe/modeling_qwen3_moe.py,
    eager_attention_forward and repeat_kv. Dropout is disabled and the caller
    supplies each packed sequence separately with a causal mask.
    https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py
    """
    q, k, v = (x.transpose(0, 1) for x in (q, k, v))
    repeats = q.shape[0] // k.shape[0]
    k, v = k.repeat_interleave(repeats, 0), v.repeat_interleave(repeats, 0)
    scores = (q @ k.transpose(-1, -2)) * (q.shape[-1] ** -0.5)
    mask = torch.ones(q.shape[1], k.shape[1], device=q.device, dtype=torch.bool).triu(1)
    return (scores.masked_fill(mask, -torch.inf).softmax(-1) @ v).transpose(0, 1)


def similarity(a, b):
    result = metrics(a, b)
    a, b = a.double(), b.double()
    denominator = (a.square() + b.square()).sum()
    result["tensor_similarity"] = (2 * (a * b).sum() / denominator).item() if denominator else 1.0
    return result


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--backend", choices=("te_fused", "magi_local", "magi_adapter"), default="te_fused"
    )
    args = parser.parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    import atexit

    # cuDNN lazily loads engines on first execution. Preserve both initial and
    # exit library maps, including when its native runtime guard rejects them.
    def save_maps():
        maps = Path("/proc/self/maps").read_text().splitlines()
        libraries = sorted({line.split()[-1] for line in maps if ".so" in line})
        args.report.with_suffix(".libraries.json").write_text(json.dumps(libraries, indent=2))

    atexit.register(save_maps)
    torch.cuda.init()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(1234)
    maps = Path("/proc/self/maps").read_text().splitlines()
    libraries = sorted({line.split()[-1] for line in maps if "libcudart.so" in line})
    if not libraries or any("libcudart.so.13" in path for path in libraries):
        raise RuntimeError(
            f"CUDA12-only fused probe has conflicting runtime libraries: {libraries}"
        )
    report = {
        "cuda_libraries": libraries,
        "dtype": "torch.bfloat16",
        "backend": args.backend,
        "stages": {},
    }
    canonical = torch.load(args.traces, map_location="cpu", weights_only=True)["canonical"]
    for layer in range(3):
        inputs = [canonical[f"end_{layer}/{name}"] for name in ("rope_q", "rope_k", "core_v")]
        module = te.DotProductAttention(
            num_attention_heads=inputs[0].shape[1],
            kv_channels=inputs[0].shape[2],
            num_gqa_groups=inputs[1].shape[1],
            attention_dropout=0.0,
            attn_mask_type="causal",
            qkv_format="thd",
        ).cuda()
        assert not list(module.parameters()), "core attention is expected to be parameter-free"
        native = [x.cuda().detach().requires_grad_(True) for x in inputs]
        reference = [x.float().cuda().detach().requires_grad_(True) for x in inputs]
        cu = torch.tensor([0, 128, 256], device="cuda", dtype=torch.int32)
        if args.backend == "magi_adapter":
            from mor_mlite.qwen3_moe_mor.local_attention import LocalMagiAttention

            module = LocalMagiAttention(cp_size=1, deterministic=True)
            assert not list(module.parameters())
            output = module(
                *native, cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=128, max_seqlen_kv=128
            )
        elif args.backend == "magi_local":
            from magi_attention.functional import flex_flash_attn_func

            ranges = torch.stack((cu[:-1], cu[1:]), dim=1).contiguous()
            output, _ = flex_flash_attn_func(
                *native,
                q_ranges=ranges,
                k_ranges=ranges,
                attn_type_map=torch.ones(2, device="cuda", dtype=torch.int32),
                deterministic=True,
                max_seqlen_q=128,
            )
        else:
            output = module(
                *native,
                qkv_format="thd",
                cu_seqlens_q=cu,
                cu_seqlens_kv=cu,
                max_seqlen_q=128,
                max_seqlen_kv=128,
                attn_mask_type="padding_causal",
                core_attention_bias_type="no_bias",
            ).reshape_as(native[0])
        expected = torch.cat(
            [naive_attention(*(x[start : start + 128] for x in reference)) for start in (0, 128)]
        )
        grad = torch.randn_like(output)
        output.backward(grad)
        expected.backward(grad.float())
        values = {"output": similarity(expected, output)}
        values.update(
            {
                f"grad_{name}": similarity(ref.grad, actual.grad)
                for name, ref, actual in zip(("q", "k", "v"), reference, native, strict=True)
            }
        )
        report["stages"][f"end_{layer}"] = values
    report["passed"] = all(
        value["cosine"] > 0.999 and value["tensor_similarity"] > 0.999
        for stage in report["stages"].values()
        for value in stage.values()
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(run())
