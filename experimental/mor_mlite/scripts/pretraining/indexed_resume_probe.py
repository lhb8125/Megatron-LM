"""Two real indexed-data updates and external-process continuity on a tiny backbone.

Uses production sequence length, global batch, data partition and token schedule.
This is not a 30B acceptance certificate and never consumes formal run tokens.
"""

import argparse
import hashlib
import json
import os
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--arm", choices=list("ABCD"), required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--full-width",
        action="store_true",
        help="Verify official 30B widths; never a formal training run",
    )
    parser.add_argument("--mbs", type=int, choices=[1, 2, 4, 8], default=1)
    args = parser.parse_args()
    import numpy as np
    import torch
    import torch.distributed as dist

    from mor_mlite.parity.mlite import _distributed_rng_fingerprint, _optimizer_leaves
    from mor_mlite.pretraining.config import Experiment, validate_base_model
    from mor_mlite.pretraining.control import HostControl
    from mor_mlite.pretraining.data import TokenStream, global_input_digest, sha256_file
    from mor_mlite.pretraining.runtime import build, distributed_environment
    from mor_mlite.pretraining.smoke import training_state
    from mor_mlite.pretraining.train import packed, save, set_learning_rate
    from mor_mlite.provenance import source_snapshot

    distributed_environment()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    config = Experiment(
        args.arm, world_size=int(os.environ["WORLD_SIZE"]), micro_batch_size=args.mbs
    )
    stream = TokenStream(args.data, "train")
    total_steps = len(stream) // config.tokens_per_step
    if total_steps < 2:
        raise ValueError("two complete real global batches required")
    fixture = json.loads((args.fixture / "config.json").read_text())
    if args.full_width:
        validate_base_model(fixture, stream.manifest)
    else:
        if fixture["hidden_size"] != 128 or fixture["num_hidden_layers"] != 48:
            raise ValueError("tiny probe requires hidden128; use --full-width explicitly for 30B")
        fixture["vocab_size"] = 151936
    contract = {
        "scope": ("full-width" if args.full_width else "tiny")
        + " backbone; real indexed data, seq4096, GBS2048; not formal training",
        "experiment": config.to_dict(),
        "fixture": fixture,
        "data_sha256": sha256_file(args.data / "manifest.json"),
        "source_sha256": source_snapshot()["sha256"],
        "probe_sha256": sha256_file(Path(__file__)),
        "scheduler": "pure-token-function-v1",
        "total_steps": total_steps,
        "total_tokens": total_steps * config.tokens_per_step,
    }
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "config.json").write_text(json.dumps(fixture))
    dist.barrier()
    runtime, handle = build(config, hf_config_dir=args.output, total_steps=total_steps)
    control = HostControl()

    def learning_rates():
        return [
            float(g["lr"]) for o in _optimizer_leaves(handle._optimizer) for g in o.param_groups
        ]

    def update(step):
        records, losses = [], []

        def batches():
            for micro in range(config.accumulation_steps):
                if rank == 0 and micro % 16 == 0:
                    print(
                        json.dumps(
                            {
                                "arm": args.arm,
                                "update": step + 1,
                                "microbatch": micro,
                                "microbatches": config.accumulation_steps,
                            }
                        ),
                        flush=True,
                    )
                tokens = stream.microbatch(
                    step,
                    micro,
                    dp_rank=handle.dp_rank,
                    dp_size=config.world_size,
                    mbs=config.micro_batch_size,
                    gbs=config.global_batch_size,
                    seq_length=config.seq_length,
                )
                first = (
                    step * config.global_batch_size
                    + (micro * config.world_size + handle.dp_rank) * config.micro_batch_size
                )
                records.extend(
                    (first + i, hashlib.sha256(row.astype("<i4").tobytes()).hexdigest())
                    for i, row in enumerate(tokens)
                )
                yield packed(tokens)

        def loss_fn(out, *_):
            losses.append(float(out["loss"].detach()))
            return out["loss"], {}

        lr = config.learning_rate((step + 1) * config.tokens_per_step, contract["total_tokens"])
        set_learning_rate(handle, lr)
        with runtime.train_mode(handle):
            runtime.zero_grad(handle)
            runtime.forward_backward(
                handle, batches(), loss_fn, num_microbatches=config.accumulation_steps
            )
            updated, norm, _ = runtime.optimizer_step(handle)
        if not updated or not np.isfinite(float(norm)) or not np.isfinite(losses).all():
            raise AssertionError("invalid indexed-data update")
        gathered = control.gather(records)
        delivered = global_input_digest(
            [item for items in gathered for item in items],
            first_sample=step * config.global_batch_size,
            batch_size=config.global_batch_size,
        )
        result = {
            "state": training_state(handle),
            "input_sha256": delivered,
            "cursor": (step + 1) * config.tokens_per_step,
            "step": step + 1,
            "losses": losses,
            "lr": lr,
            "optimizer_lrs": learning_rates(),
        }
        if rank == 0:
            print(
                json.dumps({k: v for k, v in result.items() if k not in ("state", "losses")}),
                flush=True,
            )
        return result

    if args.resume:
        saved = json.loads((args.resume / "checkpoint/pretraining-state.json").read_text())
        if (
            saved["contract"] != contract
            or saved["step"] != 1
            or saved["cursor"] != config.tokens_per_step
        ):
            raise ValueError("resume contract/scheduler/data cursor mismatch")
        expected = json.loads((args.resume / f"reference-rank-{rank}.json").read_text())
        if _distributed_rng_fingerprint() == expected["at_save"]["rng"]:
            raise AssertionError("RNG test must start from a different state")
        restored = runtime.load_checkpoint(
            handle, str(args.resume / "checkpoint"), load_rng=True, load_optimizer=True
        )
        if restored != 1 or training_state(handle) != expected["at_save"]:
            raise AssertionError("indexed checkpoint model/optimizer/RNG is not bitwise")
        if learning_rates() != expected["optimizer_lrs"]:
            raise AssertionError("optimizer learning rate not restored")
        result = update(saved["step"])
        if result != expected["next_update"]:
            raise AssertionError("resumed indexed next update differs from uninterrupted reference")
        report = {
            "passed": True,
            "contract": contract,
            "next_input_sha256": result["input_sha256"],
            "cursor": result["cursor"],
        }
        (args.output / f"resume-rank-{rank}.json").write_text(json.dumps(report) + "\n")
    else:
        first = update(0)
        # Deliberately advance every RNG stream. A no-op RNG loader must fail.
        for _ in range(rank + 3):
            random.random()
        np.random.random(rank + 5)
        torch.rand(rank + 7)
        torch.rand(rank + 11, device=torch.cuda.current_device())
        at_save = training_state(handle)
        lrs = learning_rates()
        save(
            runtime,
            handle,
            args.output / "checkpoint",
            {"contract": contract, "step": 1, "cursor": config.tokens_per_step},
            control=control,
        )
        if training_state(handle) != at_save:
            raise AssertionError("checkpoint saving changed training state")
        second = update(1)
        if first["input_sha256"] == second["input_sha256"]:
            raise AssertionError("indexed cursor reused the previous global batch")
        reference = {
            "at_save": at_save,
            "optimizer_lrs": lrs,
            "first_input_sha256": first["input_sha256"],
            "next_update": second,
        }
        (args.output / f"reference-rank-{rank}.json").write_text(json.dumps(reference) + "\n")
    control.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
