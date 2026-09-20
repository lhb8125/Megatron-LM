from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from mor_mlite.qwen3_moe_mor.gradient_scaling import (
    install_replicated_expert_tp_gradient_average,
)


def _model(*, tp: int, ep: int = 1, etp: int = 1):
    experts = nn.Linear(3, 2, bias=False)
    return SimpleNamespace(
        ps=SimpleNamespace(tp_size=tp, ep_size=ep, etp_size=etp),
        layers=[SimpleNamespace(moe=SimpleNamespace(experts=experts))],
    ), experts


def test_replicated_expert_tp_sum_is_converted_to_average() -> None:
    model, experts = _model(tp=2)
    # MLite native Experts installs this collective hook before MoR adapts the
    # model.  Multiplication is a single-process stand-in for a two-rank SUM.
    experts.weight.register_hook(lambda gradient: gradient.mul(2.0))

    assert install_replicated_expert_tp_gradient_average(model) == 1
    assert install_replicated_expert_tp_gradient_average(model) == 0

    experts.weight.sum().backward()
    assert torch.equal(experts.weight.grad, torch.ones_like(experts.weight))
    assert experts.weight._mor_tp_expert_gradient_average is True


@pytest.mark.parametrize("tp,ep,etp", [(1, 1, 1), (2, 2, 1), (2, 1, 2)])
def test_tp_expert_average_skips_non_replicated_native_case(tp: int, ep: int, etp: int) -> None:
    model, experts = _model(tp=tp, ep=ep, etp=etp)

    assert install_replicated_expert_tp_gradient_average(model) == 0
    assert not hasattr(experts.weight, "_mor_tp_expert_gradient_average")
