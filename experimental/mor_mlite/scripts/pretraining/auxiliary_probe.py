"""Independent raw-gradient reference for native DP/EP auxiliary normalization."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def validate_ownership(peers, required):
    """Each independently stored tensor has exactly one owner per element."""
    coverage = {}
    for peer in peers:
        for name, expert_rank, start, end, size in peer:
            coverage.setdefault((name, expert_rank, size), []).append((start, end))
    for (name, _, size), ranges in coverage.items():
        cursor = 0
        for start, end in sorted(ranges):
            if start != cursor or end <= start:
                raise AssertionError(f"duplicate or missing optimizer ownership: {name}")
            cursor = end
        if cursor != size:
            raise AssertionError(f"incomplete optimizer ownership: {name}")
    for key in required:
        if key not in coverage:
            raise AssertionError(f"parameter absent from optimizer: {key[0]}")


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--arm", choices=list("ABCD"), required=True)
    parser.add_argument("--hf-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    import torch.distributed as dist
    from megatron.lite.primitive.train_step import run_microbatch_loop

    from mor_mlite.parity.mlite import _local_gradient_fragments, _optimizer_leaves
    from mor_mlite.pretraining.config import Experiment
    from mor_mlite.pretraining.protocol import EXPERT_CLASSIFIER
    from mor_mlite.pretraining.runtime import build, distributed_environment
    from mor_mlite.pretraining.train import models, packed
    from mor_mlite.provenance import source_snapshot

    distributed_environment()
    config = Experiment(args.arm, world_size=int(os.environ["WORLD_SIZE"]))
    runtime, handle = build(config, hf_config_dir=args.hf_config, total_steps=100)
    exists = [args.output.exists() if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(exists, src=0)
    if exists[0]:
        raise FileExistsError(args.output)
    if dist.get_rank() == 0:
        args.output.mkdir(parents=True)
    dist.barrier()
    model = models(handle)[0]
    model.train()
    tokens = (np.arange(64)[None] + dist.get_rank()) % model.config.vocab_size

    def auxiliary(out, *_):
        return out["loss"] * 0 + out["mor_router_aux_loss"], {}

    runtime.zero_grad(handle)
    # The same native microbatch loop as Runtime.forward_backward, stopping
    # before finalization so the reference can observe unreduced gradients.
    run_microbatch_loop(
        handle._model,
        iter([packed(tokens)]),
        1,
        handle._extras["forward_step"],
        optimizer=handle._optimizer,
        dist_opt=True,
        pre_forward_hook=handle._extras["pre_forward_hook"],
        loss_fn=auxiliary,
    )
    expected = {}
    for expert in (False, True):
        parameters = [
            p for name, p in model.named_parameters() if EXPERT_CLASSIFIER(name) == expert
        ]
        if any(getattr(p, "main_grad", None) is None for p in parameters):
            raise AssertionError("unreduced gradient capture is incomplete")
        flat = torch.cat([p.main_grad.detach().reshape(-1).double() for p in parameters])
        # Experts already receive EP-dispatched token contributions; their
        # replica sum is still divided by dense-DP world, not expert-DP size.
        group = model.ps.ep_dp_group if expert else model.ps.dp_group
        dist.all_reduce(flat, group=group)
        flat /= config.world_size
        offset = 0
        for parameter in parameters:
            expected[id(parameter)] = flat[offset : offset + parameter.numel()].clone()
            offset += parameter.numel()
    handle._extras["finalize_grads"]()
    error_sq, reference_sq, elements = 0.0, 0.0, 0
    names = {id(p): name for name, p in model.named_parameters()}
    ownership = []
    for optimizer in _optimizer_leaves(handle._optimizer):
        for ranges in optimizer.gbuf_ranges:
            for buckets in ranges.values():
                for bucket in buckets:
                    for parameter, bounds in bucket["param_map"].items():
                        start, end = bounds["param"].start, bounds["param"].end
                        reference = expected[id(parameter)][start:end]
                        actual = parameter.main_grad.reshape(-1)[start:end].double()
                        torch.testing.assert_close(actual, reference, atol=1e-8, rtol=1e-4)
                        error_sq += float((actual - reference).square().sum())
                        reference_sq += float(reference.square().sum())
                        elements += actual.numel()
                        name = names[id(parameter)]
                        ownership.append(
                            (
                                name,
                                model.ps.ep_rank if EXPERT_CLASSIFIER(name) else -1,
                                start,
                                end,
                                parameter.numel(),
                            )
                        )
    if not elements or reference_sq == 0:
        raise AssertionError("auxiliary reference has no nonzero owned gradients")
    peers = [None] * config.world_size
    dist.all_gather_object(peers, ownership)
    required = [
        (name, model.ps.ep_rank if EXPERT_CLASSIFIER(name) else -1, parameter.numel())
        for name, parameter in model.named_parameters()
    ]
    validate_ownership(peers, required)
    one = _local_gradient_fragments(handle)["fragments"]
    runtime.zero_grad(handle)
    runtime.forward_backward(
        handle,
        iter([packed(tokens), packed(tokens)]),
        auxiliary,
        num_microbatches=2,
    )
    two = _local_gradient_fragments(handle)["fragments"]
    if len(one) != len(two):
        raise AssertionError("optimizer gradient ownership changed with accumulation")
    for left, right in zip(one, two, strict=True):
        for key in ("name", "start", "end", "shape"):
            if left[key] != right[key]:
                raise AssertionError("gradient identity mismatch")
        torch.testing.assert_close(left["value"], right["value"], atol=1e-7, rtol=0.005)
    report = {
        "scope": "tiny native auxiliary-only gradients; not full training acceptance",
        "arm": args.arm,
        "passed": True,
        "owned_elements": elements,
        "dp_reference_relative_l2": (error_sq / reference_sq) ** 0.5,
        "reference": "FP64 replica sum of raw pre-finalize gradients / dense-DP world",
        "accumulation": "one microbatch vs two identical microbatches",
        "optimizer_ownership": "exact nonoverlapping global coverage of every parameter element",
        "source": source_snapshot(),
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.output / f"rank-{dist.get_rank()}.json").write_text(json.dumps(report) + "\n")
    dist.barrier()
    if dist.get_rank() == 0:
        print(json.dumps({k: v for k, v in report.items() if k != "source"}), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
