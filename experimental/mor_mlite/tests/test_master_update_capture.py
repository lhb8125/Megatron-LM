from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from mor_mlite.parity.mlite import _local_master_weight_fragments


def _fake_handle(*, master_dtype: torch.dtype = torch.float32):
    model = nn.Module()
    model.register_parameter(
        "weight",
        nn.Parameter(torch.tensor([10.0, 20.0, 30.0, 40.0], dtype=torch.bfloat16)),
    )
    parameter = model.weight
    master = torch.tensor([20.25, 30.25], dtype=master_dtype)
    leaf = SimpleNamespace(
        gbuf_ranges=[
            {
                (torch.bfloat16, torch.float32): [
                    {"param_map": {parameter: {"param": SimpleNamespace(start=1, end=3)}}}
                ]
            }
        ],
        model_param_group_index_map={parameter: (0, 0)},
        optimizer=SimpleNamespace(param_groups=[{"params": [master]}]),
    )
    handle = SimpleNamespace(
        _model=(model,),
        _optimizer=leaf,
        _parallel_state=SimpleNamespace(
            tp_rank=0,
            tp_size=1,
            cp_rank=0,
            dp_rank=0,
            ep_rank=0,
            ep_size=1,
            etp_rank=0,
            expert_dp_rank=0,
        ),
        _extras={
            "model_chunks": (model,),
            "model_cfg": SimpleNamespace(num_experts=4, vocab_size=257),
        },
    )
    return handle, master


def test_master_weight_fragments_read_the_owned_fp32_optimizer_shard() -> None:
    handle, master = _fake_handle()

    captured = _local_master_weight_fragments(handle)

    assert len(captured["fragments"]) == 1
    fragment = captured["fragments"][0]
    assert fragment["name"] == "weight"
    assert fragment["shape"] == (4,)
    assert fragment["numel"] == 4
    assert (fragment["start"], fragment["end"]) == (1, 3)
    assert fragment["value"].dtype == torch.float32
    assert torch.equal(fragment["value"], master)


def test_master_weight_capture_rejects_non_fp32_optimizer_shards() -> None:
    handle, _ = _fake_handle(master_dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="expected torch.float32"):
        _local_master_weight_fragments(handle)
