"""Native prefix-prediction check; fixed tolerances, no fitted routing threshold."""

from __future__ import annotations


def check(runtime, handle):
    import numpy as np
    import torch
    import torch.distributed as dist

    from mor_mlite.pretraining.train import models, packed
    from mor_mlite.qwen3_moe_mor.protocol import unpack_forward_output

    model = models(handle)[0]
    tokens = (np.arange(8)[None] * 13 + dist.get_rank()) % model.config.vocab_size
    zero_rounds = 0
    comparisons = 0
    max_error = 0.0
    # These are small-model BF16 prediction tolerances, not full-model parity gates.
    atol, rtol = 0.005, 0.01

    def forward(ids):
        nonlocal zero_rounds
        batch = packed(ids)
        batch.extras["mor_return_full_logits"] = True
        values = []

        def capture(output, *_):
            logits = unpack_forward_output(model, batch, output["logits"])
            if getattr(logits, "is_nested", False):
                logits = torch.cat(list(logits.unbind()), dim=0)
            values.append(logits.reshape(-1, model.config.vocab_size).detach().float())
            return output["loss"], {}

        runtime.forward_backward(
            handle, iter([batch]), capture, num_microbatches=1, forward_only=True
        )
        routes = [t["selected_rows"].detach().clone() for t in model.causal_route_traces]
        zero_rounds += sum(rows.numel() == 0 for rows in routes)
        if len(values) != 1 or values[0].shape[0] != ids.size:
            raise AssertionError("prediction unpacking did not preserve real token count")
        return values[0], routes

    def compare(reference, candidate, length):
        nonlocal comparisons, max_error
        expected, routes = reference
        actual, other_routes = candidate
        a, b = expected[:length], actual[:length]
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise AssertionError("nonfinite causal prediction")
        max_error = max(max_error, float((a - b).abs().max()))
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
        if len(routes) != len(other_routes):
            raise AssertionError("recursion count changed with suffix")
        for left, right in zip(routes, other_routes, strict=True):
            if not torch.equal(left[left < length], right[right < length]):
                raise AssertionError("future text changed prefix routing")
        comparisons += 1

    with runtime.eval_mode(handle), torch.no_grad():
        full = forward(tokens)
        modified = tokens.copy()
        modified[:, 4:] = (modified[:, 4:] + 53) % model.config.vocab_size
        compare(full, forward(modified), 4)
        # All prefixes, including length one; each rank follows identical call order.
        for length in range(1, 8):
            compare(full, forward(tokens[:, :length]), length)
    zeros = torch.tensor(zero_rounds, device=torch.cuda.current_device(), dtype=torch.int64)
    dist.all_reduce(zeros)
    if model.experiment_arm == "D" and zeros == 0:
        raise AssertionError("native test did not exercise zero-active rank; evidence incomplete")
    return {
        "passed": True,
        "scope": "tiny-model eight-token native causal predictions and exact prefix routes",
        "comparisons_per_rank": comparisons,
        "atol": atol,
        "rtol": rtol,
        "max_abs_error_local": max_error,
        "zero_active_rank_rounds": int(zeros),
    }
