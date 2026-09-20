"""Forward-only BF16 attention attribution, with explicit diagnostic input injection."""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path
from types import MethodType

import torch
import torch.distributed as dist

from mor_mlite.parity import artifacts, mlite
from mor_mlite.parity.__main__ import main
from mor_mlite.provenance import file_sha256


def assemble(records):
    """Reconstruct every stage by original token ID and global head index."""
    result = {}
    for record in records:
        ids = record["token_ids"]
        real = ids >= 0
        for stage, value in record["tensors"].items():
            key = f"end_{record['layer']}/{stage}"
            value = value[real]
            selected_ids = ids[real]
            offset = record["tp_rank"] * value.shape[1]
            target = result.setdefault(key, {})
            for token_id, token_value in zip(selected_ids.tolist(), value, strict=True):
                for head, vector in enumerate(token_value):
                    identity = (token_id, offset + head)
                    if identity in target and not torch.equal(target[identity], vector):
                        raise ValueError(
                            f"inconsistent duplicated boundary tensor: {key} {identity}"
                        )
                    target[identity] = vector.clone()
    canonical = {}
    for key, values in result.items():
        num_tokens = max(token for token, _head in values) + 1
        num_heads = max(head for _token, head in values) + 1
        if len(values) != num_tokens * num_heads:
            raise ValueError(f"incomplete token/head coverage: {key}")
        canonical[key] = torch.stack(
            [values[token, head] for token in range(num_tokens) for head in range(num_heads)]
        ).reshape(num_tokens, num_heads, -1)
    return canonical


def metrics(reference, candidate):
    if reference.shape != candidate.shape:
        raise ValueError(f"boundary shape mismatch: {reference.shape} != {candidate.shape}")
    a, b = reference.double().reshape(-1), candidate.double().reshape(-1)
    return {
        "relative_l2": ((a - b).norm() / a.norm().clamp_min(1e-30)).item(),
        "cosine": torch.nn.functional.cosine_similarity(a, b, dim=0).item(),
        "max_abs": (a - b).abs().max().item(),
        "exact": torch.equal(reference, candidate),
    }


def math_attention(q, k, v, lengths):
    """Independent unfused FP32 math, without intermediate BF16 score rounding."""
    outputs = []
    offset = 0
    for length in lengths:
        qs, ks, vs = (x[offset : offset + length].float().transpose(0, 1) for x in (q, k, v))
        repeats = qs.shape[0] // ks.shape[0]
        ks, vs = ks.repeat_interleave(repeats, 0), vs.repeat_interleave(repeats, 0)
        scores = (qs @ ks.transpose(-1, -2)) * (qs.shape[-1] ** -0.5)
        mask = torch.ones(length, length, dtype=torch.bool).triu(1)
        outputs.append((scores.masked_fill(mask, -torch.inf).softmax(-1) @ vs).transpose(0, 1))
        offset += length
    if offset != q.shape[0]:
        raise ValueError("reference attention lengths do not cover captured tokens")
    return torch.cat(outputs)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--compare-traces", type=Path)
    parser.add_argument("--inject-core", action="store_true")
    args, remaining = parser.parse_known_args()
    if "--forward-only" not in remaining:
        parser.error("attention boundary injection is diagnostic only and cannot train")
    if args.inject_core and args.compare_traces is None:
        parser.error("--inject-core requires --compare-traces")
    metadata, oracle, _routes = artifacts.load_artifact(args.oracle)
    if metadata["steps"] != 1 or metadata["num_microbatches"] != 1:
        parser.error("this diagnostic requires exactly one step and microbatch")
    prefix = "forward/step_000/mb_000/"
    baseline_trace = (
        torch.load(args.compare_traces, map_location="cpu", weights_only=True)["canonical"]
        if args.compare_traces is not None
        else None
    )
    records = []
    original_build, original_save = mlite.build_runtime_session, artifacts.save_artifact

    def build(config):
        session = original_build(config)
        for wrapped in session.handle._extras["model_chunks"]:
            model = wrapped
            while hasattr(model, "module"):
                model = model.module
            original_scope = model._moe_probe_scope

            @contextmanager
            def scope(self, layer, _original_scope=original_scope, **context):
                handles = []
                if context["stage"] == "end":
                    index = context["stage_layer_index"]
                    ids = context["global_token_ids"].reshape(-1).long().clone()
                    ids[context["padding_mask"].reshape(-1)] = -1
                    full_ids = ids
                    if layer.attn.ps.tp_size > 1:
                        full_ids = ids.new_empty(ids.numel() * layer.attn.ps.tp_size)
                        dist.all_gather_into_tensor(full_ids, ids, group=layer.attn.ps.tp_group)
                    record = {
                        "layer": index,
                        "tp_rank": layer.attn.ps.tp_rank,
                        "token_ids": full_ids.cpu(),
                        "tensors": {},
                    }
                    records.append(record)
                    baseline = oracle[
                        prefix + ("post_merge_hidden" if index == 0 else f"end_hidden_{index - 1}")
                    ]

                    def inject_block(_module, inputs):
                        x = inputs[0]
                        fixed = baseline.to(x.device).index_select(0, ids.clamp_min(0))
                        fixed[ids < 0] = 0
                        return (fixed.reshape_as(x).to(x.dtype), *inputs[1:])

                    def capture(stage, value):
                        if value.shape[0] != full_ids.numel():
                            raise ValueError(f"stage {stage} is not TP-gathered token aligned")
                        record["tensors"][stage] = (
                            value.detach()
                            .reshape(value.shape[0], -1, layer.attn.head_dim)
                            .cpu()
                            .clone()
                        )

                    def qkv_hook(_module, _inputs, output):
                        q, gate, k, v = layer.attn._split_qkv(output)
                        if gate is not None or layer.attn._replicate_kv:
                            raise ValueError("this probe requires ungated, non-replicated Qwen KV")
                        for stage, value in (("qkv_q", q), ("qkv_k", k), ("qkv_v", v)):
                            capture(stage, value)

                    def norm_hook(stage):
                        def hook(_module, _inputs, output):
                            capture(stage, output)

                        return hook

                    def core_pre(_module, inputs):
                        if args.inject_core:
                            changed = []
                            for name, x in zip(("rope_q", "rope_k", "core_v"), inputs, strict=True):
                                table = baseline_trace[f"end_{index}/{name}"]
                                offset = layer.attn.ps.tp_rank * x.shape[1]
                                fixed = table[:, offset : offset + x.shape[1]].to(x.device)
                                fixed = fixed.index_select(0, full_ids.clamp_min(0))
                                fixed[full_ids < 0] = 0
                                changed.append(fixed.to(x.dtype))
                            inputs = tuple(changed)
                        for name, value in zip(("rope_q", "rope_k", "core_v"), inputs, strict=True):
                            capture(name, value)
                        return inputs

                    handles.extend(
                        (
                            layer.register_forward_pre_hook(inject_block, prepend=True),
                            layer.attn.qkv.register_forward_hook(qkv_hook),
                            layer.attn.q_norm.register_forward_hook(norm_hook("norm_q")),
                            layer.attn.k_norm.register_forward_hook(norm_hook("norm_k")),
                            layer.attn.core_attn.register_forward_pre_hook(core_pre),
                            layer.attn.core_attn.register_forward_hook(norm_hook("core_output")),
                        )
                    )
                try:
                    with _original_scope(layer, **context):
                        yield
                finally:
                    for handle in handles:
                        handle.remove()

            model._moe_probe_scope = MethodType(scope, model)
        return session

    def save(directory, *, metadata, tensors, routes):
        return original_save(
            directory,
            metadata={
                **metadata,
                "operator_probe": {
                    "kind": "attention-boundaries",
                    "inject_core": args.inject_core,
                    "acceptance_eligible": False,
                    "script_sha256": file_sha256(__file__),
                    "oracle_manifest_sha256": file_sha256(args.oracle / "manifest.json"),
                    "oracle_tensors_sha256": file_sha256(args.oracle / "tensors.pt"),
                },
            },
            tensors=tensors,
            routes=routes,
        )

    mlite.build_runtime_session, artifacts.save_artifact = build, save
    result = main(["run", *remaining])
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, records)
    if dist.get_rank() == 0:
        canonical = assemble([record for group in gathered for record in group])
        args.traces.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"canonical": canonical}, args.traces)
        report = {
            "status": "diagnostic",
            "acceptance_eligible": False,
            "boundaries": {},
            "math_attention": {},
        }
        if baseline_trace is not None:
            if set(canonical) != set(baseline_trace):
                raise ValueError("attention boundary stages differ from baseline")
            report["boundaries"] = {
                key: metrics(baseline_trace[key], value) for key, value in canonical.items()
            }
        for index in range(metadata["architecture"]["n_end_layers"]):
            ref = math_attention(
                *(canonical[f"end_{index}/{stage}"] for stage in ("rope_q", "rope_k", "core_v")),
                metadata["global_batch"]["sequence_lengths"],
            )
            report["math_attention"][f"end_{index}"] = metrics(
                ref, canonical[f"end_{index}/core_output"]
            )
        args.traces.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(run())
