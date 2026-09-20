"""Token-budgeted training and causal held-out evaluation through MLite APIs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

from mor_mlite.pretraining.config import (
    Experiment,
    milestones,
    validate_base_model,
    validate_data_purpose,
)
from mor_mlite.pretraining.cost import FLOP_METHOD, observed_flops, parameter_counts
from mor_mlite.pretraining.data import TokenStream, global_input_digest, sha256_file, verify_sources


def packed(tokens):
    import torch

    from mor_mlite.data import PackedBatch, as_mlite_packed_batch

    ids = torch.as_tensor(tokens.copy(), dtype=torch.long, device=torch.cuda.current_device())
    count, length = ids.shape
    ids = ids.reshape(-1)
    return as_mlite_packed_batch(
        PackedBatch(
            input_ids=ids,
            labels=ids.clone(),
            seq_lens=torch.full((count,), length, dtype=torch.int32, device=ids.device),
            loss_mask=torch.ones_like(ids, dtype=torch.float32),
        )
    )


def models(handle):
    from mor_mlite.parity.mlite import _unwrapped_model_chunks

    return _unwrapped_model_chunks(handle)


def evaluate(runtime, handle, stream, config, *, control=None):
    import numpy as np
    import torch
    import torch.distributed as dist

    chunks = math.ceil(len(stream) / config.seq_length)
    if len(stream) < 2:
        raise ValueError("validation stream contains no next-token target")
    local_chunks = math.ceil(chunks / config.world_size)
    stats = torch.zeros(6, device=torch.cuda.current_device(), dtype=torch.float64)
    with runtime.eval_mode(handle), torch.no_grad():
        for index in range(local_chunks):
            chunk_id = index * config.world_size + handle.dp_rank
            valid = chunk_id < chunks
            length = (
                min(config.seq_length, len(stream) - chunk_id * config.seq_length) if valid else 1
            )
            tokens = (
                stream.read(chunk_id * config.seq_length, length)
                if valid
                else np.zeros(1, dtype=np.int64)
            )

            def loss_fn(output, *_, valid=valid, length=length):
                lm = output["loss"] - output["mor_router_aux_loss"]
                targets = (length - 1) if valid else 0
                stats[0].add_(lm.detach().double() * targets)
                stats[1].add_(targets)
                stats[4].add_(observed_flops(models(handle)[0], config.arm, [length]))
                if config.arm == "D" and valid:
                    traces = models(handle)[0].causal_route_traces
                    stats[2].add_(sum(t["selected_rows"].numel() for t in traces))
                    stats[3].add_(length)
                stats[5].add_(length if valid else 0)
                return lm, {}

            runtime.forward_backward(
                handle, iter([packed(tokens[None])]), loss_fn, num_microbatches=1, forward_only=True
            )
    if control is None:
        dist.all_reduce(stats)
    else:
        stats = control.reduce(stats)
    nll = float(stats[0] / stats[1])
    if not math.isfinite(nll):
        raise RuntimeError("non-finite held-out NLL")
    return {
        "nll": nll,
        "ppl": math.exp(nll),
        "targets": int(stats[1]),
        "mean_recursions": float(stats[2] / stats[3]) if stats[3] > 0 else None,
        "policy": "strict-causal-threshold" if config.arm == "D" else "causal-full-depth",
        "estimated_forward_flops": float(stats[4]),
        "flop_method": FLOP_METHOD,
        "validation_input_tokens": int(stats[5]),
        "excluded_chunk_boundary_targets": chunks,
        "discarded_validation_tail_tokens": 0,
    }


def set_learning_rate(handle, lr):
    from mor_mlite.parity.mlite import _optimizer_leaves

    for opt in _optimizer_leaves(handle._optimizer):
        for group in opt.param_groups:
            group["lr"] = lr * group.get("lr_mult", 1.0)


def save(runtime, handle, path, state, *, control=None):
    import torch.distributed as dist

    exists = [path.exists() if dist.get_rank() == 0 else None]
    if control is None:
        dist.broadcast_object_list(exists, src=0)
    else:
        control.broadcast(exists)
    if exists[0]:
        raise FileExistsError(path)
    runtime.save_checkpoint(
        handle, str(path), step=state["step"], save_rng=True, save_optimizer=True, use_dcp=True
    )
    barrier = dist.barrier if control is None else control.barrier
    barrier()
    if dist.get_rank() == 0:
        # Last write is the completeness marker; an interrupted save is never resumed.
        (path / "pretraining-state.json").write_text(json.dumps(state, indent=2) + "\n")
    barrier()


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--arm", choices=list("ABCD"), required=True)
    parser.add_argument("--world-size", type=int, choices=[32, 64], default=64)
    parser.add_argument("--mbs", type=int, choices=[1, 2, 4, 8], default=1)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--hf-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["train", "tune"], default="train")
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument(
        "--topology", type=Path, required=True, help="Host-side scontrol show topology snapshot"
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-tokens", type=int)
    parser.add_argument(
        "--acceptance",
        type=Path,
        help="GPU acceptance certificate bound to source/data/config",
    )
    args = parser.parse_args()
    config = Experiment(args.arm, world_size=args.world_size, micro_batch_size=args.mbs)
    import torch
    import torch.distributed as dist

    from mor_mlite.pretraining.control import HostControl
    from mor_mlite.pretraining.memory import MemoryMonitor
    from mor_mlite.pretraining.probe import environment
    from mor_mlite.pretraining.runtime import build, distributed_environment, topology_evidence
    from mor_mlite.provenance import source_snapshot

    source = source_snapshot()["sha256"]
    data_hash = sha256_file(args.data / "manifest.json")
    model_config_hash = sha256_file(args.hf_config / "config.json")
    validate_data_purpose(json.loads((args.data / "manifest.json").read_text()), args.mode)
    validate_base_model(
        json.loads((args.hf_config / "config.json").read_text()),
        json.loads((args.data / "manifest.json").read_text()),
    )
    required = (
        "causality",
        "initialization",
        "shared_gradients",
        "auxiliary_gradients",
        "data_integrity",
        "checkpoint_resume",
        "gb200_multinode",
    )
    if args.mode == "train":
        if args.acceptance is None:
            parser.error("formal training requires --acceptance")
        acceptance = json.loads(args.acceptance.read_text())
        if acceptance.get("source_sha256") != source or acceptance.get("data_sha256") != data_hash:
            raise ValueError("acceptance source/data mismatch")
        if not all(acceptance.get("checks", {}).get(name) is True for name in required):
            raise ValueError("formal training requires all acceptance checks")
        if acceptance.get("experiment") != config.to_dict():
            raise ValueError("acceptance does not bind this arm/topology/MBS/config")
        if acceptance.get("environment_sha256") != sha256_file(args.environment):
            raise ValueError("acceptance does not bind the candidate environment")
        if acceptance.get("model_config_sha256") != model_config_hash:
            raise ValueError("acceptance does not bind this model configuration")
    elif args.resume is not None or args.stop_tokens is not None:
        parser.error("tuning is exactly 50 updates from initialization; no resume/stop override")
    train = TokenStream(args.data, "train")
    validation = TokenStream(args.data, "validation")
    total_steps = len(train) // config.tokens_per_step
    if total_steps < 1:
        raise ValueError("training data does not fill one global batch")
    total_tokens = total_steps * config.tokens_per_step
    if args.mode == "tune" and total_steps < 50:
        raise ValueError("tuning requires 50 complete global batches")
    distributed_environment()
    actual_environment = environment(args.native_root)
    expected_environment = json.loads(args.environment.read_text())
    if not expected_environment.get("te_forward_backward") or any(
        expected_environment.get(k) != v for k, v in actual_environment.items()
    ):
        raise ValueError("candidate environment no longer matches the CUDA canary")
    memory = MemoryMonitor()
    memory.start()  # Includes model build, optimizer initialization, eval, and save.
    run_started = time.monotonic()
    runtime, handle = build(config, hf_config_dir=args.hf_config, total_steps=total_steps)
    control = HostControl()
    topology = topology_evidence(handle, args.topology.read_text())
    audit = [None]
    if dist.get_rank() == 0:
        try:
            audit[0] = verify_sources(args.data)
        except (ValueError, OSError) as exc:
            audit[0] = {"error": str(exc)}
    control.broadcast(audit)
    if "error" in audit[0]:
        raise ValueError(audit[0]["error"])
    contract = {
        "experiment": config.to_dict(),
        "data_sha256": data_hash,
        "source_sha256": source,
        "model_config_sha256": model_config_hash,
        "total_steps": total_steps,
        "total_tokens": total_tokens,
        "environment_sha256": sha256_file(args.environment),
        "mode": args.mode,
        "discarded_training_tokens": len(train) - total_tokens,
        "parameters": parameter_counts(models(handle)[0]),
    }
    step = 0
    if args.resume:
        state = json.loads((args.resume / "pretraining-state.json").read_text())
        if (
            state["contract"] != contract
            or state["cursor"] != state["step"] * config.tokens_per_step
        ):
            raise ValueError("checkpoint run contract/data cursor mismatch")
        step = runtime.load_checkpoint(handle, str(args.resume), load_rng=True, load_optimizer=True)
        if step != state["step"]:
            raise ValueError("native checkpoint step mismatch")
    exists = [args.output.exists() if dist.get_rank() == 0 else None]
    control.broadcast(exists)
    if exists[0]:
        raise FileExistsError("use a new output directory for each continuation")
    if dist.get_rank() == 0:
        args.output.mkdir(parents=True)
        (args.output / "manifest.json").write_text(
            json.dumps({**contract, "topology": topology}, indent=2) + "\n"
        )
    control.barrier()
    stop = (
        total_steps
        if args.stop_tokens is None
        else min(total_steps, math.ceil(args.stop_tokens / config.tokens_per_step))
    )
    if args.mode == "tune":
        stop = 50
    if stop <= step:
        raise ValueError("stop token target is not ahead of the restored cursor")
    checkpoints = (
        {stop}
        if args.mode == "tune"
        else set(milestones(total_steps, config.tokens_per_step)) | {stop}
    )
    measured_seconds = []

    def log(name, value):
        if dist.get_rank() == 0:
            with (args.output / name).open("a") as f:
                f.write(json.dumps(value) + "\n")
            print(json.dumps(value), flush=True)

    if step == 0:
        log(
            "evaluation.jsonl",
            {
                "step": 0,
                "tokens": 0,
                **evaluate(runtime, handle, validation, config, control=control),
            },
        )
    while step < stop:
        started = time.monotonic()
        observed = []
        digest = hashlib.sha256()
        samples = []

        def batches(step=step, digest=digest, samples=samples):
            for microstep in range(config.accumulation_steps):
                tokens = train.microbatch(
                    step,
                    microstep,
                    dp_rank=handle.dp_rank,
                    dp_size=config.world_size,
                    mbs=config.micro_batch_size,
                    gbs=config.global_batch_size,
                    seq_length=config.seq_length,
                )
                digest.update(tokens.astype("<i4").tobytes())
                first = (
                    step * config.global_batch_size
                    + microstep * config.world_size * config.micro_batch_size
                    + handle.dp_rank * config.micro_batch_size
                )
                samples.extend(
                    (first + i, hashlib.sha256(row.astype("<i4").tobytes()).hexdigest())
                    for i, row in enumerate(tokens)
                )
                yield packed(tokens)

        def loss_fn(output, *_, observed=observed):
            aux = output["mor_router_aux_loss"]
            observed.append(
                torch.stack(
                    (
                        (output["loss"] - aux).detach().double(),
                        aux.detach().double(),
                        aux.new_tensor(
                            observed_flops(
                                models(handle)[0],
                                config.arm,
                                [config.seq_length] * config.micro_batch_size,
                            ),
                            dtype=torch.float64,
                        ),
                    )
                )
            )
            return output["loss"], {}

        lr = config.learning_rate((step + 1) * config.tokens_per_step, total_tokens)
        set_learning_rate(handle, lr)
        with runtime.train_mode(handle):
            runtime.zero_grad(handle)
            runtime.forward_backward(
                handle,
                batches(),
                loss_fn,
                num_microbatches=config.accumulation_steps,
                forward_only=False,
            )
            values = control.reduce(torch.stack(observed).mean(0))
            values /= config.world_size
            if not torch.isfinite(values).all():
                raise RuntimeError(f"non-finite loss before optimizer update at {step + 1}")
            updated, grad_norm, _ = runtime.optimizer_step(handle)
        if not updated or not math.isfinite(float(grad_norm)) or not torch.isfinite(values).all():
            raise RuntimeError(f"invalid optimizer update at {step + 1}")
        torch.cuda.synchronize()
        elapsed = control.reduce(torch.tensor(time.monotonic() - started), maximum=True)
        if step >= 20:
            measured_seconds.append(float(elapsed))
        sample_records = control.gather(samples)
        input_hash = global_input_digest(
            [item for records in sample_records for item in records],
            first_sample=step * config.global_batch_size,
            batch_size=config.global_batch_size,
        )
        step += 1
        log(
            "loss.jsonl",
            {
                "step": step,
                "tokens": step * config.tokens_per_step,
                "lr": lr,
                "lm_loss": float(values[0]),
                "depth_aux": float(values[1]),
                "grad_norm": float(grad_norm),
                "seconds": float(elapsed),
                "input_sha256": input_hash,
                "estimated_training_flops": float(values[2])
                * config.accumulation_steps
                * config.world_size
                * 3,
            },
        )
        # Per-rank evidence covers every sample, independent of logging rank.
        with (args.output / f"input-rank-{dist.get_rank()}.jsonl").open("a") as f:
            f.write(json.dumps({"step": step, "sha256": digest.hexdigest()}) + "\n")
        if step in checkpoints or step % 100 == 0:
            save(
                runtime,
                handle,
                args.output / f"checkpoint-{step:06d}",
                {
                    "contract": contract,
                    "step": step,
                    "cursor": step * config.tokens_per_step,
                    "scheduler": "pure-token-function-v1",
                },
                control=control,
            )
        if step in checkpoints:
            log(
                "evaluation.jsonl",
                {
                    "step": step,
                    "tokens": step * config.tokens_per_step,
                    **evaluate(runtime, handle, validation, config, control=control),
                },
            )
    if source_snapshot()["sha256"] != source:
        raise RuntimeError("source changed during training")
    local_memory = {"rank": dist.get_rank(), **memory.finish()}
    memory_records = control.gather(local_memory)
    log("memory.jsonl", {"ranks": memory_records})
    log(
        "cost.jsonl",
        {
            "wall_seconds": time.monotonic() - run_started,
            "gpu_hours": (time.monotonic() - run_started) * config.world_size / 3600,
        },
    )
    if args.mode == "tune":
        from dataclasses import asdict

        from mor_mlite.pretraining.tuning import Trial

        trial = Trial(
            config.micro_batch_size,
            30 * config.tokens_per_step / sum(measured_seconds),
            max(r["peak_device_fraction"] for r in memory_records),
            20,
            30,
            True,
            True,
            True,
        )
        log(
            "trial.jsonl",
            {"trial": asdict(trial), "contract": contract, "formal_training_tokens": 0},
        )
    log("complete.jsonl", {"step": step, "full_epoch": step == total_steps, "contract": contract})
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
