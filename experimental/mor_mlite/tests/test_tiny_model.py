from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from mor_mlite.data import make_synthetic_batch
from mor_mlite.optim import MasterWeightAdamW
from mor_mlite.reference_checkpoint import (
    load_reference_checkpoint,
    save_reference_checkpoint,
)
from mor_mlite.tiny import TinyMoRConfig, TinyMoRModel
from mor_mlite.tiny.model import PositionAwareGQA


def _small_config() -> TinyMoRConfig:
    return TinyMoRConfig(
        vocab_size=31,
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=12,
        num_experts=2,
        num_experts_per_tok=2,
        max_position_embeddings=32,
    )


def _small_batch(seed: int = 11):
    return make_synthetic_batch(seq_lens=(4, 3), vocab_size=31, seed=seed, mask_last_token=True)


def _assert_nested_equal(lhs, rhs) -> None:
    if isinstance(lhs, torch.Tensor):
        torch.testing.assert_close(lhs, rhs, rtol=0.0, atol=0.0)
    elif isinstance(lhs, dict):
        assert lhs.keys() == rhs.keys()
        for key in lhs:
            _assert_nested_equal(lhs[key], rhs[key])
    elif isinstance(lhs, (list, tuple)):
        assert len(lhs) == len(rhs)
        for left_item, right_item in zip(lhs, rhs, strict=True):
            _assert_nested_equal(left_item, right_item)
    else:
        assert lhs == rhs


def test_synthetic_batch_uses_mlite_unshifted_label_contract() -> None:
    batch = _small_batch(seed=13)
    assert torch.equal(batch.labels, batch.input_ids)
    offset = 0
    for length_value in batch.seq_lens.tolist():
        stop = offset + int(length_value)
        assert int(batch.input_ids[stop - 1]) == 2
        offset = stop

    labels, mask = TinyMoRModel._lm_targets(batch)
    assert labels.tolist() == [
        *batch.input_ids[1:4].tolist(),
        0,
        *batch.input_ids[5:7].tolist(),
        0,
    ]
    assert mask.tolist() == [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0]


def test_recurrent_layers_are_registered_once_but_called_every_round() -> None:
    model = TinyMoRModel(_small_config(), seed=5)
    architecture = model.config.architecture

    assert len(model.recurrent_layers) == architecture.n_recurrent_layers
    assert model.physical_num_layers == 4
    assert model.logical_num_layers == 8
    recurrent_indices = {
        name.split(".")[1] for name in model.state_dict() if name.startswith("recurrent_layers.")
    }
    assert recurrent_indices == {"0", "1"}

    call_counts = [0] * architecture.n_recurrent_layers
    handles = []
    for layer_index, layer in enumerate(model.recurrent_layers):

        def count_call(_module, _inputs, _output, *, index=layer_index):
            call_counts[index] += 1

        handles.append(layer.register_forward_hook(count_call))
    try:
        model(_small_batch())
    finally:
        for handle in handles:
            handle.remove()

    assert call_counts == [architecture.num_recursions] * architecture.n_recurrent_layers


def test_forward_route_budgets_are_per_original_sequence_and_nested() -> None:
    model = TinyMoRModel(_small_config(), seed=7)
    batch = _small_batch()
    output = model(batch)

    # For original lengths 4 and 3, exact linear budgets are 4/3, 2/2, 1/1.
    expected_per_sample = ((4, 3), (2, 2), (1, 1))
    previous_ids = set(batch.extras["global_token_ids"].tolist())
    for round_index, (plan, expected) in enumerate(
        zip(output.route_plans, expected_per_sample, strict=True)
    ):
        counts = tuple(int((plan.sample_ids == sample_id).sum()) for sample_id in range(2))
        assert counts == expected
        assert plan.active_cu_seqlens.tolist() == [0, expected[0], sum(expected)]
        current_ids = set(plan.global_token_ids.tolist())
        assert current_ids.issubset(previous_ids)
        previous_ids = current_ids
        for sample_id in range(2):
            positions = plan.original_positions[plan.sample_ids == sample_id]
            assert positions.tolist() == sorted(positions.tolist())
        assert plan.round_index == round_index

    assert len(output.hidden_by_round) == model.config.architecture.num_recursions
    assert all(hidden.shape[0] == batch.total_tokens for hidden in output.hidden_by_round)
    assert output.communication == {
        "active_set_changes": 2,
        "hidden_rebalances": 2,
        "recurrent_inner_dispatches": 0,
        "early_exit_qkv_tokens": 0,
        "recurrent_block_calls": 3,
        "recurrent_qkv_checks": 6,
        "recurrent_qkv_real_tokens": 26,
    }


def test_extreme_synthetic_batch_actually_concentrates_each_sample_route() -> None:
    model = TinyMoRModel(_small_config(), seed=7)
    batch = make_synthetic_batch(
        seq_lens=(4, 3),
        vocab_size=31,
        seed=11,
        extreme_routing=True,
    )

    assert batch.extras["apply_routing_bias"] is True
    output = model(batch)
    second_round = output.route_plans[1]
    for sample_id, midpoint in ((0, 2), (1, 2)):
        positions = second_round.original_positions[second_round.sample_ids == sample_id]
        assert positions.tolist() == list(range(midpoint))


def test_position_aware_gqa_uses_sparse_original_rope_positions() -> None:
    torch.manual_seed(19)
    attention = PositionAwareGQA(_small_config()).eval()
    hidden = torch.randn(3, 8)
    sample_ids = torch.zeros(3, dtype=torch.long)

    sparse = attention(
        hidden,
        sample_ids=sample_ids,
        original_positions=torch.tensor([0, 3, 7]),
    )
    compressed = attention(
        hidden,
        sample_ids=sample_ids,
        original_positions=torch.tensor([0, 1, 2]),
    )

    # The first causal query sees only itself; later queries must change because
    # RoPE uses the original gaps instead of compressing [0, 3, 7] to [0, 1, 2].
    torch.testing.assert_close(sparse[0], compressed[0], rtol=1e-6, atol=1e-6)
    assert not torch.allclose(sparse[1:], compressed[1:], rtol=1e-5, atol=1e-6)


def test_loss_backward_accumulates_into_the_single_physical_block() -> None:
    model = TinyMoRModel(_small_config(), seed=23)
    backward_calls = [0] * len(model.recurrent_layers)
    handles = []

    for layer_index, layer in enumerate(model.recurrent_layers):

        def observe_output(_module, _inputs, output, *, index=layer_index):
            output.register_hook(
                lambda gradient, index=index: (
                    backward_calls.__setitem__(index, backward_calls[index] + 1) or gradient
                )
            )

        handles.append(layer.register_forward_hook(observe_output))
    try:
        output = model(_small_batch())
        assert output.total_loss is not None and torch.isfinite(output.total_loss)
        assert output.lm_loss is not None and torch.isfinite(output.lm_loss)
        assert torch.isfinite(output.aux_loss)
        output.total_loss.backward()
    finally:
        for handle in handles:
            handle.remove()

    expected_calls = model.config.architecture.num_recursions
    assert backward_calls == [expected_calls] * len(model.recurrent_layers)
    recurrent_parameter = model.recurrent_layers[0].attn.q_proj.weight
    assert recurrent_parameter.grad is not None
    assert torch.isfinite(recurrent_parameter.grad).all()
    assert float(recurrent_parameter.grad.norm()) > 0.0
    for router in model.depth_routers:
        assert router.proj.weight.grad is not None
        assert torch.isfinite(router.proj.weight.grad).all()


def test_two_microbatch_gradient_equals_the_mean_of_independent_gradients() -> None:
    batches = (_small_batch(seed=31), _small_batch(seed=32))

    combined = TinyMoRModel(_small_config(), seed=29)
    for batch in batches:
        (combined(batch).total_loss / len(batches)).backward()

    independent_gradients: list[dict[str, torch.Tensor | None]] = []
    for batch in batches:
        model = TinyMoRModel(_small_config(), seed=29)
        model(batch).total_loss.backward()
        independent_gradients.append(
            {
                name: None if parameter.grad is None else parameter.grad.detach().clone()
                for name, parameter in model.named_parameters()
            }
        )

    for name, parameter in combined.named_parameters():
        pieces = [gradients[name] for gradients in independent_gradients]
        if parameter.grad is None:
            assert pieces == [None, None]
            continue
        assert all(piece is not None for piece in pieces)
        expected = (pieces[0] + pieces[1]) / len(pieces)
        torch.testing.assert_close(parameter.grad, expected, rtol=2e-5, atol=2e-6)


def test_master_weight_optimizer_keeps_fp32_master_for_bf16_model() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.bfloat16))
    optimizer = MasterWeightAdamW([parameter], lr=0.01)
    assert optimizer.master_parameters[0].dtype == torch.float32
    before = optimizer.master_parameters[0].detach().clone()
    parameter.grad = torch.tensor([0.5, -0.25], dtype=torch.bfloat16)

    norm = optimizer.step()

    assert float(norm) > 0.0
    assert not torch.equal(before, optimizer.master_parameters[0])
    torch.testing.assert_close(
        parameter.float(),
        optimizer.master_parameters[0].detach().to(torch.bfloat16).float(),
        rtol=0.0,
        atol=0.0,
    )


def test_checkpoint_restores_model_master_weights_and_next_update(tmp_path: Path) -> None:
    config = _small_config()
    model = TinyMoRModel(config, seed=29)
    optimizer = MasterWeightAdamW(model.parameters(), lr=3e-4)
    first = model(_small_batch(seed=31))
    assert first.total_loss is not None
    first.total_loss.backward()
    optimizer.step()

    checkpoint = save_reference_checkpoint(
        tmp_path / "tiny.pt",
        model=model,
        optimizer=optimizer,
        step=1,
        metadata={"case": "tiny-roundtrip", "model": config.to_dict()},
    )
    resumed = TinyMoRModel(config, seed=999)
    resumed_optimizer = MasterWeightAdamW(resumed.parameters(), lr=3e-4)
    step, metadata = load_reference_checkpoint(
        checkpoint, model=resumed, optimizer=resumed_optimizer
    )

    assert step == 1
    assert metadata == {"case": "tiny-roundtrip", "model": config.to_dict()}
    _assert_nested_equal(model.state_dict(), resumed.state_dict())
    _assert_nested_equal(optimizer.state_dict(), resumed_optimizer.state_dict())

    # Optimizer moments/master weights must make the first post-resume update exact.
    optimizer.zero_grad()
    resumed_optimizer.zero_grad()
    batch = _small_batch(seed=37)
    original_loss = model(batch).total_loss
    resumed_loss = resumed(batch).total_loss
    assert original_loss is not None and resumed_loss is not None
    original_loss.backward()
    resumed_loss.backward()
    optimizer.step()
    resumed_optimizer.step()
    _assert_nested_equal(model.state_dict(), resumed.state_dict())
