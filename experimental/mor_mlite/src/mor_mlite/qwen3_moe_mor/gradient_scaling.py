"""Gradient-scaling adapters for native MLite Qwen3-MoE modules."""

from __future__ import annotations

from typing import Any

import torch


def install_replicated_expert_tp_gradient_average(model: Any) -> int:
    """Turn the native replicated-expert TP reduction into an average.

    MLite's native ``Experts`` installs a TP ``SUM`` parameter hook when
    ``TP>1, EP=ETP=1``.  That hook is necessary because the full expert is
    replicated while its input tokens are sequence-parallel, but the LM loss
    used by this protocol is already a mean over the TP-local token rows.  A
    sum therefore leaves the expert gradient larger by ``TP`` than the
    equivalent single-rank gradient.  Registering this hook after native model
    construction preserves the native collective and converts its result to
    the required mean.  Dense TP-sharded parameters and EP-sharded experts are
    deliberately untouched.

    The marker makes adaptation idempotent and is also an executable contract
    for the parity tests.
    """

    ps = getattr(model, "ps", None)
    tp_size = int(getattr(ps, "tp_size", 1) or 1)
    ep_size = int(getattr(ps, "ep_size", 1) or 1)
    etp_size = int(getattr(ps, "etp_size", 1) or 1)
    if tp_size <= 1 or ep_size != 1 or etp_size != 1:
        return 0

    installed = 0
    scale = 1.0 / float(tp_size)
    for layer in getattr(model, "layers", ()):
        experts = getattr(getattr(layer, "moe", None), "experts", None)
        if experts is None:
            continue
        for parameter in experts.parameters():
            if bool(getattr(parameter, "_mor_tp_expert_gradient_average", False)):
                continue

            def _average_after_native_sum(gradient: torch.Tensor, factor: float = scale):
                return gradient.mul(factor)

            parameter.register_hook(_average_after_native_sum)
            parameter._mor_tp_expert_gradient_average = True
            installed += 1
    return installed


__all__ = ["install_replicated_expert_tp_gradient_average"]
