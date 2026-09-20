"""Check weight-tying chain rule against an explicitly untied native layer stack.

Not an independent Transformer-kernel reference: the tested claim is that the
shared parameter gradient equals the sum of the independent per-use gradients.
The unrolled stack uses distinct Parameter objects and the same native kernels.
"""

import argparse
import json
import os
from pathlib import Path


def shared_name(name):
    if not name.startswith("layers."):
        return name
    _, index, suffix = name.split(".", 2)
    index = int(index)
    if not 0 <= index < 48:
        raise ValueError("unrolled layer index outside the 48-layer reference")
    # Explicit mathematical 3 + 14*3 + 3 mapping, not execution.layer_indices.
    physical = index if index < 3 else 3 + (index - 3) % 14 if index < 45 else index - 28
    return f"layers.{physical}.{suffix}"


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    import torch.distributed as dist
    from megatron.lite.primitive.train_step import run_microbatch_loop

    from mor_mlite.pretraining.config import Experiment
    from mor_mlite.pretraining.data import sha256_file
    from mor_mlite.pretraining.runtime import build, distributed_environment
    from mor_mlite.pretraining.train import models, packed
    from mor_mlite.provenance import source_snapshot

    distributed_environment()
    config = Experiment("B", world_size=int(os.environ["WORLD_SIZE"]))
    runtime, handle = build(config, hf_config_dir=args.fixture, total_steps=100)
    model = models(handle)[0]
    weights = {name: p.detach().clone() for name, p in model.named_parameters()}
    tokens = (np.arange(64)[None] + dist.get_rank()) % model.config.vocab_size

    def gradients(rt, h):
        with rt.train_mode(h):
            rt.zero_grad(h)
            out = run_microbatch_loop(
                h._model,
                iter([packed(tokens)]),
                1,
                h._extras["forward_step"],
                optimizer=h._optimizer,
                dist_opt=True,
                pre_forward_hook=h._extras["pre_forward_hook"],
            )
        grads = {}
        for name, parameter in models(h)[0].named_parameters():
            if getattr(parameter, "main_grad", None) is None:
                raise AssertionError(f"missing raw gradient: {name}")
            grads[name] = parameter.main_grad.detach().double().clone()
        return float(out["loss"].detach()), grads

    loss, shared = gradients(runtime, handle)
    reference_config = Experiment("A", world_size=config.world_size)
    reference_rt, reference_handle = build(
        reference_config, hf_config_dir=args.fixture, total_steps=100
    )
    reference = models(reference_handle)[0]
    objects = [id(p) for _, p in reference.named_parameters(remove_duplicate=False)]
    if len(objects) != len(set(objects)):
        raise AssertionError("unrolled reference unexpectedly shares Parameter objects")
    with torch.no_grad():
        for name, parameter in reference.named_parameters():
            value = weights[shared_name(name)]
            if value.shape != parameter.shape:
                raise AssertionError(f"reference parameter shape mismatch: {name}")
            parameter.copy_(value)
    # No optimizer update is performed: compare pre-finalize gradients directly.
    reference_loss, untied = gradients(reference_rt, reference_handle)
    if any(not torch.equal(p, weights[shared_name(n)]) for n, p in reference.named_parameters()):
        raise AssertionError("reference forward changed or regathered the copied weights")
    torch.testing.assert_close(
        torch.tensor(loss), torch.tensor(reference_loss), atol=1e-6, rtol=1e-6
    )
    summed = {}
    multiplicity = {}
    for name, value in untied.items():
        target = shared_name(name)
        if target not in summed:
            summed[target] = torch.zeros_like(value)
        summed[target].add_(value)
        multiplicity[target] = multiplicity.get(target, 0) + 1
    if set(summed) != set(shared):
        raise AssertionError("reference omitted or added shared parameters")
    metrics = []
    for name, actual in shared.items():
        expected = summed[name]
        inner = (actual * expected).sum()
        a2, b2 = actual.square().sum(), expected.square().sum()
        if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
            raise AssertionError(f"nonfinite gradient: {name}")
        similarity = float(2 * inner / (a2 + b2)) if a2 + b2 > 0 else 1.0
        cosine = float(inner / (a2 * b2).sqrt()) if a2 * b2 > 0 else float(a2 == b2)
        if min(similarity, cosine) < 0.999:
            raise AssertionError(
                f"shared gradient is not the summed per-use gradient: {name}: {similarity}, {cosine}"
            )
        expected_uses = 3 if name.startswith("layers.") and 3 <= int(name.split(".")[1]) < 17 else 1
        if multiplicity[name] != expected_uses:
            raise AssertionError(f"incorrect sharing multiplicity: {name}")
        metrics.append(
            {
                "name": name,
                "uses": multiplicity[name],
                "tensor_similarity": similarity,
                "cosine": cosine,
            }
        )
    if dist.get_rank() == 0:
        args.output.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    report = {
        "passed": True,
        "scope": __doc__,
        "source": source_snapshot(),
        "probe_sha256": sha256_file(Path(__file__)),
        "loss": loss,
        "reference_loss": reference_loss,
        "metrics": metrics,
    }
    (args.output / f"rank-{dist.get_rank()}.json").write_text(json.dumps(report) + "\n")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
