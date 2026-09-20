"""Forward-only precision isolation on native weights; never a training backend."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from types import MethodType

import torch
import torch.distributed as dist
import torch.nn.functional as F

from mor_mlite.parity import mlite
from mor_mlite.parity.__main__ import main


def residual_forward(self, x, position_ids=None, packed_seq_params=None):
    residual = x.float()
    h = self.attn(
        x.to(torch.bfloat16), position_ids=position_ids, packed_seq_params=packed_seq_params
    )
    x = residual + h.float()
    h = self.mlp_norm(x.to(torch.bfloat16))
    return x + self.moe(h).float()


def final_norm_forward(self, x):
    return F.rms_norm(x.float(), (x.shape[-1],), self.weight.float(), self.eps).to(
        self.weight.dtype
    )


def full_norm_forward(self, x):
    return F.rms_norm(x.float(), (x.shape[-1],), self.weight.float(), self.eps)


def full_experts_forward(
    self, x, tokens_per_expert, permuted_probs=None, tokens_per_expert_list=None
):
    if torch.is_grad_enabled():
        raise RuntimeError("FP32 expert isolation is forward-only")
    if (
        self.fp8
        or self.etp_group is not None
        or self.fc1_lora is not None
        or self.fc2_lora is not None
        or self.swiglu_limit
    ):
        raise RuntimeError(
            "FP32 expert isolation only implements this pinned Qwen BF16/ETP1/no-LoRA profile"
        )
    counts = (
        tokens_per_expert.tolist() if tokens_per_expert_list is None else tokens_per_expert_list
    )
    if len(counts) != self.num_local_experts or sum(counts) != x.shape[0]:
        raise ValueError("expert token splits do not cover dispatched inputs")
    outputs = []
    offset = 0
    for index, count in enumerate(counts):
        stop = offset + count
        if count:
            projected = F.linear(
                x[offset:stop].float(), getattr(self.fc1, f"weight{index}").float()
            )
            gate, up = projected.chunk(2, dim=-1)
            hidden = F.silu(gate) * up
            if permuted_probs is not None:
                hidden = hidden * permuted_probs[offset:stop].float().reshape(-1, 1)
            outputs.append(F.linear(hidden, getattr(self.fc2, f"weight{index}").float()))
        offset = stop
    return torch.cat(outputs, dim=0) if outputs else x.new_empty(x.shape, dtype=torch.float32)


def stable_residual_forward(self, x, position_ids=None, packed_seq_params=None):
    residual = x.float()
    h = self.attn(residual, position_ids=position_ids, packed_seq_params=packed_seq_params)
    x = residual + h.float()
    return x + self.moe(self.mlp_norm(x)).float()


def stable_qkv_forward(self, x):
    if torch.is_grad_enabled():
        raise RuntimeError("stable QKV probe is forward-only")
    normalized = F.rms_norm(
        x.float(), (x.shape[-1],), self.linear.layer_norm_weight.float(), self.linear.eps
    )
    if self.tp_size > 1:
        if not self.use_sp:
            raise RuntimeError("stable QKV probe requires sequence parallelism with TP")
        gathered = normalized.new_empty((normalized.shape[0] * self.tp_size, *normalized.shape[1:]))
        dist.all_gather_into_tensor(gathered, normalized.contiguous(), group=self.tp_group)
        normalized = gathered
    projected = F.linear(normalized, self.linear.weight.float())
    return (
        projected
        if getattr(self, "_probe_keep_output_fp32", False)
        else projected.to(self.linear.weight.dtype)
    )


def projection_forward(self, x):
    if torch.is_grad_enabled():
        raise RuntimeError("projection probe is forward-only")
    y = F.linear(x.float(), self.linear.weight.float())
    if self.tp_size > 1:
        shape = (y.shape[0] // self.tp_size, *y.shape[1:])
        output = torch.empty(shape, dtype=torch.float32, device=y.device)
        dist.reduce_scatter_tensor(output, y.contiguous(), group=self.tp_group)
        y = output
    return y if getattr(self, "_probe_keep_output_fp32", False) else y.to(x.dtype)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=(
            "residual_fp32",
            "projection_fp32",
            "both",
            "stable_fp32",
            "full_fp32",
            "fp16_attention",
        ),
        required=True,
    )
    args, remaining = parser.parse_known_args()
    if "--forward-only" not in remaining:
        parser.error("this diagnostic requires --forward-only")
    original_build = mlite.build_runtime_session

    def build(config):
        session = original_build(config)
        chunks = session.handle._extras["model_chunks"]
        for wrapped in chunks:
            model = wrapped
            while hasattr(model, "module"):
                model = model.module
            for layer in model.layers:
                if args.mode in {"residual_fp32", "both"}:
                    layer.forward = MethodType(residual_forward, layer)
                if args.mode in {"stable_fp32", "full_fp32", "fp16_attention"}:
                    layer.forward = MethodType(stable_residual_forward, layer)
                    layer.attn.qkv.forward = MethodType(stable_qkv_forward, layer.attn.qkv)
                    layer.mlp_norm.forward = MethodType(final_norm_forward, layer.mlp_norm)
                if args.mode == "full_fp32":
                    layer.mlp_norm.forward = MethodType(full_norm_forward, layer.mlp_norm)
                    layer.moe.experts.forward = MethodType(full_experts_forward, layer.moe.experts)
                    layer.attn.proj._probe_keep_output_fp32 = True
                if args.mode == "fp16_attention":
                    layer.attn.qkv._probe_keep_output_fp32 = True
                    layer.attn.q_norm.forward = MethodType(full_norm_forward, layer.attn.q_norm)
                    layer.attn.k_norm.forward = MethodType(full_norm_forward, layer.attn.k_norm)
                    layer.attn.proj._probe_keep_output_fp32 = True

                    def attention_inputs(_module, inputs):
                        return tuple(value.to(torch.float16) for value in inputs)

                    layer.attn.core_attn.register_forward_pre_hook(attention_inputs)
                if args.mode in {
                    "projection_fp32",
                    "both",
                    "stable_fp32",
                    "full_fp32",
                    "fp16_attention",
                }:
                    layer.attn.proj.forward = MethodType(projection_forward, layer.attn.proj)
            if args.mode in {"residual_fp32", "both", "stable_fp32", "full_fp32", "fp16_attention"}:
                model.norm.forward = MethodType(final_norm_forward, model.norm)
        return session

    def save(directory, *, metadata, tensors, routes):
        metadata = {
            **metadata,
            "precision_probe": {
                "mode": args.mode,
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            },
        }
        return original_save(directory, metadata=metadata, tensors=tensors, routes=routes)

    # The runner imports save_artifact inside run_mlite, so install provenance
    # at the shared artifact API before invoking it.
    from mor_mlite.parity import artifacts

    original_save = artifacts.save_artifact
    artifacts.save_artifact = save
    mlite.build_runtime_session = build
    return main(["run", *remaining])


if __name__ == "__main__":
    sys.exit(run())
