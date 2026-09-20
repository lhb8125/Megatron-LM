"""Regression checks for the diagnostic's token/head reconstruction and math oracle."""

import importlib.util
from pathlib import Path

import pytest
import torch

_path = Path(__file__).resolve().parents[1] / "scripts/eos/probe_attention_boundaries.py"
_spec = importlib.util.spec_from_file_location("attention_boundaries", _path)
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)


def test_assemble_uses_token_ids_and_tp_heads():
    records = [
        {
            "layer": 0,
            "tp_rank": rank,
            "token_ids": torch.tensor([1, -1, 0]),
            "tensors": {"norm_q": torch.tensor([[[10.0 + rank]], [[99.0]], [[float(rank)]]])},
        }
        for rank in range(2)
    ]
    torch.testing.assert_close(
        probe.assemble(records)["end_0/norm_q"],
        torch.tensor([[[0.0], [1.0]], [[10.0], [11.0]]]),
    )


def test_assemble_rejects_missing_heads():
    records = [
        {
            "layer": 0,
            "tp_rank": 1,
            "token_ids": torch.tensor([0]),
            "tensors": {"q": torch.ones(1, 1, 2)},
        }
    ]
    with pytest.raises(ValueError, match="incomplete"):
        probe.assemble(records)


def test_math_attention_respects_sample_boundaries_and_causality():
    q, k = torch.zeros(3, 2, 2), torch.zeros(3, 1, 2)
    v = torch.tensor([[[2.0, 4.0]], [[6.0, 8.0]], [[20.0, 30.0]]])
    expected = torch.tensor([[2.0, 4.0], [4.0, 6.0], [20.0, 30.0]])
    actual = probe.math_attention(q, k, v, [2, 1])
    torch.testing.assert_close(actual, expected[:, None, :].expand(-1, 2, -1))
