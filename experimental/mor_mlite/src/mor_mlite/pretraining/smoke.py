"""Real MLite/EP16 tiny-width smoke. Evidence, never a full acceptance certificate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def fingerprint(model):
    import torch

    return {
        name: {
            "shape": list(p.shape),
            "numel": p.numel(),
            "sha256": hashlib.sha256(
                p.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
            ).hexdigest(),
        }
        for name, p in model.named_parameters()
        if not name.startswith("depth_routers.")
    }


def training_state(handle):
    """Fingerprint real model/master weights, Adam moments and all RNG streams."""
    from mor_mlite.parity.mlite import (
        _distributed_optimizer_fingerprint,
        _distributed_rng_fingerprint,
        _distributed_state_fingerprint,
    )
    from mor_mlite.pretraining.train import models

    return {
        "model": _distributed_state_fingerprint(models(handle)[0].state_dict(), label="model"),
        "optimizer": _distributed_optimizer_fingerprint(handle),
        "rng": _distributed_rng_fingerprint(),
    }


def next_update(runtime, handle, config):
    import numpy as np
    import torch.distributed as dist

    from mor_mlite.pretraining.train import packed, set_learning_rate

    # A distinct second global input detects accidental reuse of step one's data.
    tokens = (np.arange(64)[None] + dist.get_rank() + 7) % 256
    losses = []

    def loss_fn(output, *_):
        losses.append(float(output["loss"].detach()))
        return output["loss"], {}

    with runtime.train_mode(handle):
        runtime.zero_grad(handle)
        set_learning_rate(
            handle, config.learning_rate(2 * config.tokens_per_step, 100 * config.tokens_per_step)
        )
        runtime.forward_backward(handle, iter([packed(tokens)]), loss_fn, num_microbatches=1)
        updated, norm, _ = runtime.optimizer_step(handle)
    if not updated or not np.isfinite(float(norm)) or not all(np.isfinite(losses)):
        raise AssertionError("second optimizer update failed")
    return {
        "state": training_state(handle),
        "input_sha256": hashlib.sha256(tokens.astype("<i4").tobytes()).hexdigest(),
        "losses": losses,
        "cursor": 128 * config.world_size,
    }


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--arm", choices=list("ABCD"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument(
        "--resume", type=Path, help="Completed smoke directory from another process"
    )
    args = parser.parse_args()
    import numpy as np
    import torch
    import torch.distributed as dist

    from mor_mlite.pretraining.config import Experiment
    from mor_mlite.pretraining.runtime import build, distributed_environment, topology_evidence
    from mor_mlite.pretraining.train import models, packed, save, set_learning_rate
    from mor_mlite.provenance import source_snapshot

    source_before = source_snapshot()
    distributed_environment()
    config = Experiment(args.arm, world_size=int(os.environ["WORLD_SIZE"]))
    dist.init_process_group("nccl")
    if dist.get_rank() == 0:
        args.output.mkdir(parents=True, exist_ok=False)
        fixture = (
            json.loads((args.resume / "config.json").read_text())
            if args.resume
            else {
                "model_type": "qwen3_moe",
                "num_hidden_layers": 48,
                "hidden_size": 128,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 32,
                "vocab_size": 256,
                "num_experts": 128,
                "num_experts_per_tok": 8,
                "moe_intermediate_size": 64,
                "max_position_embeddings": 4096,
            }
        )
        (args.output / "config.json").write_text(json.dumps(fixture))
    dist.barrier()
    runtime, handle = build(config, hf_config_dir=args.output, total_steps=100)
    model = models(handle)[0]
    topology = topology_evidence(handle, args.topology.read_text())
    if args.resume:
        state = json.loads((args.resume / "checkpoint/pretraining-state.json").read_text())
        if state.get("source_sha256") != source_before["sha256"]:
            raise ValueError("resume source does not match the original smoke checkpoint")
        if state["arm"] != args.arm or state["cursor"] != 64 * config.world_size:
            raise ValueError("resume arm/data cursor differs from the saved smoke contract")
        restored = runtime.load_checkpoint(
            handle, str(args.resume / "checkpoint"), load_rng=True, load_optimizer=True
        )
        if restored != 1:
            raise AssertionError("checkpoint did not restore step one")
        expected = json.loads((args.resume / f"continuity-rank-{dist.get_rank()}.json").read_text())
        if training_state(handle) != expected["at_save"]:
            raise AssertionError("cross-process model/optimizer/RNG restoration is not bitwise")
        if next_update(runtime, handle, config) != expected["next_update"]:
            raise AssertionError("resumed next update differs from uninterrupted reference")
        (args.output / f"resume-rank-{dist.get_rank()}.json").write_text(
            json.dumps(
                {
                    "passed": True,
                    "scope": "tiny model, synthetic two-step continuity",
                    "topology": topology,
                }
            )
            + "\n"
        )
        dist.barrier()
        dist.destroy_process_group()
        return
    initial = fingerprint(model)
    (args.output / f"initial-rank-{dist.get_rank()}.json").write_text(json.dumps(initial))
    if len(model.layers) != config.physical_layers:
        raise AssertionError("physical layer count mismatch")
    if args.arm != "D" and len(model.depth_routers):
        raise AssertionError("ordinary/fixed arms must not own depth gates")
    tokens = (np.arange(64)[None] + dist.get_rank()) % 256
    observed = []

    def loss_fn(output, *_):
        observed.append(float(output["loss"].detach()))
        return output["loss"], {}

    with runtime.train_mode(handle):
        runtime.zero_grad(handle)
        set_learning_rate(handle, config.lr)
        runtime.forward_backward(handle, iter([packed(tokens)]), loss_fn, num_microbatches=1)
        updated, norm, _ = runtime.optimizer_step(handle)
    if not updated or not np.isfinite(float(norm)) or not all(np.isfinite(observed)):
        raise AssertionError("tiny optimizer step failed")
    # The auxiliary-only backward must reach both native MoE routers and (D)
    # depth routers. Keep zero*LM connected to exercise AutoScaler injection.
    with runtime.train_mode(handle):
        runtime.zero_grad(handle)
        runtime.forward_backward(
            handle,
            iter([packed(tokens)]),
            lambda out, *_: (out["loss"] * 0 + out["mor_router_aux_loss"], {}),
            num_microbatches=1,
        )
    auxiliary_gradients = {}
    for name, parameter in model.named_parameters():
        if ".moe.router.gate.weight" in name or name.startswith("depth_routers."):
            grad = getattr(parameter, "main_grad", None)
            if grad is None:
                grad = parameter.grad
            if grad is None or not torch.isfinite(grad).all():
                raise AssertionError(f"missing/nonfinite auxiliary gradient: {name}")
            auxiliary_gradients[name] = float(grad.detach().float().norm())
    if not any(v > 0 for k, v in auxiliary_gradients.items() if ".moe.router." in k):
        raise AssertionError("MoE auxiliary has no gradient")
    if args.arm == "D" and not any(
        v > 0 for k, v in auxiliary_gradients.items() if k.startswith("depth_routers.")
    ):
        raise AssertionError("depth auxiliary has no gradient")
    with runtime.eval_mode(handle), torch.no_grad():
        runtime.forward_backward(
            handle, iter([packed(tokens)]), loss_fn, num_microbatches=1, forward_only=True
        )
    from mor_mlite.pretraining.causal_probe import check as check_causal

    causal = check_causal(runtime, handle)
    runtime.zero_grad(handle)
    at_save = training_state(handle)
    save(
        runtime,
        handle,
        args.output / "checkpoint",
        {
            "step": 1,
            "cursor": 64 * config.world_size,
            "arm": args.arm,
            "source_sha256": source_before["sha256"],
        },
    )
    if training_state(handle) != at_save:
        raise AssertionError("checkpoint saving unexpectedly changed training state")
    continuity = {"at_save": at_save, "next_update": next_update(runtime, handle, config)}
    (args.output / f"continuity-rank-{dist.get_rank()}.json").write_text(
        json.dumps(continuity) + "\n"
    )
    if source_snapshot() != source_before:
        raise RuntimeError("source changed while the smoke was running")
    report = {
        "scope": "tiny-width real MLite smoke, not full acceptance",
        "arm": args.arm,
        "physical_layers": len(model.layers),
        "logical_layers": config.logical_layers,
        "optimizer_update": bool(updated),
        "grad_norm": float(norm),
        "losses": observed,
        "auxiliary_gradient_norms": auxiliary_gradients,
        "causality": causal,
        "topology": topology,
        "source": source_before,
    }
    (args.output / f"result-rank-{dist.get_rank()}.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    if dist.get_rank() == 0:
        print(
            json.dumps(
                {
                    k: v
                    for k, v in report.items()
                    if k not in ("source", "topology", "auxiliary_gradient_norms")
                }
            )
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
