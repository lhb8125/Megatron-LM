from __future__ import annotations

import torch

from mor_mlite.config_loader import load_preset_config
from mor_mlite.parity.mlite import _tiny_hf_weights


def test_default_tiny_native_moe_routes_have_a_deterministic_cutoff_margin() -> None:
    preset = load_preset_config("tiny")
    weights = _tiny_hf_weights(preset.architecture, seed=1234, model=preset.model)
    embedding = weights["model.embed_tokens.weight"].float()

    assert torch.equal(embedding[:, 0], torch.full_like(embedding[:, 0], 16.0))
    assert torch.equal(
        embedding[:, 1],
        torch.where(
            torch.arange(embedding.shape[0]).remainder(2) == 0,
            torch.tensor(16.0),
            torch.tensor(-16.0),
        ),
    )

    hidden = embedding * torch.rsqrt(
        embedding.square().mean(dim=-1, keepdim=True) + float(preset.model["rms_norm_eps"])
    )
    for logical_layer in range(preset.architecture.logical_num_layers):
        prefix = f"model.layers.{logical_layer}"
        gate = weights[f"{prefix}.mlp.gate.weight"].float()
        logits = hidden @ gate.t()
        values, indices = torch.topk(logits, k=3, dim=-1)
        cutoff_margin = values[:, 1] - values[:, 2]

        assert float(cutoff_margin.min()) > 0.5
        assert torch.equal(
            torch.sort(indices[::2, :2], dim=-1).values,
            torch.tensor([[0, 2]]).expand(indices[::2].shape[0], -1),
        )
        assert torch.equal(
            torch.sort(indices[1::2, :2], dim=-1).values,
            torch.tensor([[1, 3]]).expand(indices[1::2].shape[0], -1),
        )
        assert torch.count_nonzero(weights[f"{prefix}.self_attn.o_proj.weight"][:2]) == 0
        for expert in range(int(preset.model["num_experts"])):
            assert (
                torch.count_nonzero(weights[f"{prefix}.mlp.experts.{expert}.down_proj.weight"][:2])
                == 0
            )


def test_custom_tiny_model_does_not_silently_receive_default_margin_profile() -> None:
    preset = load_preset_config("tiny")
    custom = dict(preset.model)
    custom["vocab_size"] = int(custom["vocab_size"]) + 1
    weights = _tiny_hf_weights(preset.architecture, seed=1234, model=custom)

    embedding = weights["model.embed_tokens.weight"]
    assert not torch.equal(embedding[:, 0], torch.full_like(embedding[:, 0], 16.0))
