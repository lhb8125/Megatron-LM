from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from mor_mlite.config import DepthRouterConfig, MoRArchitectureConfig
from mor_mlite.routing import (
    DepthRouter,
    RoutePlan,
    apply_recurrent_update,
    globally_normalized_bce_with_logits,
    stable_expert_choice_indices,
)


def _router(round_index: int) -> DepthRouter:
    router = DepthRouter(
        2,
        MoRArchitectureConfig.tiny(),
        DepthRouterConfig(temperature=1.0, alpha=0.1, aux_loss_coef=0.001),
        round_index=round_index,
    )
    with torch.no_grad():
        router.proj.weight.copy_(torch.tensor([[1.0, 0.0]]))
    return router


def test_stable_choice_breaks_exact_ties_by_global_token_id() -> None:
    scores = torch.tensor([0.5, 0.7, 0.7, 0.1])
    token_ids = torch.tensor([40, 30, 20, 10])
    chosen = stable_expert_choice_indices(scores, token_ids, 2)
    assert chosen.tolist() == [2, 1]


def test_router_is_per_sample_and_restores_causal_order() -> None:
    router = _router(round_index=1)
    hidden = torch.tensor([[0.1, 0.0], [4.0, 0.0], [3.0, 0.0], [2.0, 0.0], [0.5, 0.0], [5.0, 0.0]])
    output = router(
        hidden,
        sample_ids=torch.tensor([0, 0, 0, 1, 1, 1]),
        original_positions=torch.tensor([0, 1, 2, 0, 1, 2]),
        global_token_ids=torch.tensor([0, 1, 2, 3, 4, 5]),
        original_lengths={0: 3, 1: 3},
    )
    # Capacity 2/3 chooses two tokens from each sample, then sorts by position.
    assert output.selected_indices.tolist() == [1, 2, 3, 5]
    assert output.plan.active_cu_seqlens.tolist() == [0, 2, 4]
    assert output.plan.mode == "learned"
    assert output.selected_gates.requires_grad


def test_active_sets_are_nested_when_previous_selection_is_next_input() -> None:
    hidden = torch.tensor([[0.0, 0.0], [1.0, 0.0], [4.0, 0.0], [3.0, 0.0], [2.0, 0.0], [5.0, 0.0]])
    sample_ids = torch.zeros(6, dtype=torch.long)
    positions = torch.arange(6)
    ids = torch.arange(100, 106)

    first = _router(0)(
        hidden,
        sample_ids=sample_ids,
        original_positions=positions,
        global_token_ids=ids,
        original_lengths={0: 6},
    )
    assert first.selected_indices.numel() == 6

    active = first.selected_indices
    second = _router(1)(
        hidden[active],
        sample_ids=sample_ids[active],
        original_positions=positions[active],
        global_token_ids=ids[active],
        original_lengths={0: 6},
    )
    active = active[second.selected_indices]
    third = _router(2)(
        hidden[active],
        sample_ids=sample_ids[active],
        original_positions=positions[active],
        global_token_ids=ids[active],
        original_lengths={0: 6},
    )
    final_ids = set(ids[active[third.selected_indices]].tolist())
    assert final_ids < set(ids.tolist())
    assert len(final_ids) == 2


def test_padding_is_excluded_from_route_and_auxiliary_loss() -> None:
    router = _router(0)
    hidden = torch.tensor([[1.0, 0.0], [2.0, 0.0], [100.0, 0.0]])
    output = router(
        hidden,
        sample_ids=torch.tensor([0, 0, -1]),
        original_positions=torch.tensor([0, 1, -1]),
        global_token_ids=torch.tensor([0, 1, -1]),
        original_lengths={0: 2},
        padding_mask=torch.tensor([False, False, True]),
    )
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        output.raw_logits[:2], torch.ones(2), reduction="mean"
    )
    torch.testing.assert_close(output.aux_loss, expected)
    assert output.selected_indices.tolist() == [0, 1]


def test_auxiliary_loss_uses_raw_logits_and_has_gradient() -> None:
    logits = torch.tensor([-1.0, 2.0], requires_grad=True)
    target = torch.tensor([0.0, 1.0])
    loss = globally_normalized_bce_with_logits(logits, target)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad) == 2


def test_router_temperature_scales_the_shared_score_and_bce_logits() -> None:
    router = DepthRouter(
        2,
        MoRArchitectureConfig.tiny(),
        DepthRouterConfig(temperature=2.0, alpha=0.1, aux_loss_coef=0.001),
        round_index=0,
    )
    with torch.no_grad():
        router.proj.weight.copy_(torch.tensor([[1.0, 0.0]]))
    output = router(
        torch.tensor([[2.0, 0.0], [-2.0, 0.0]]),
        sample_ids=torch.zeros(2, dtype=torch.long),
        original_positions=torch.arange(2),
        global_token_ids=torch.arange(2),
        original_lengths={0: 2},
    )

    decision_logits = output.raw_logits / 2.0
    torch.testing.assert_close(output.scores, torch.sigmoid(decision_logits) * 0.1)
    torch.testing.assert_close(
        output.aux_loss,
        torch.nn.functional.binary_cross_entropy_with_logits(
            decision_logits, torch.ones_like(decision_logits)
        ),
    )


def test_synthetic_logit_bias_drives_expert_choice_without_hiding_router_gradient() -> None:
    router = _router(round_index=1)
    hidden = torch.zeros((4, 2), requires_grad=True)
    output = router(
        hidden,
        sample_ids=torch.zeros(4, dtype=torch.long),
        original_positions=torch.arange(4),
        global_token_ids=torch.arange(4),
        original_lengths={0: 4},
        logit_bias=torch.tensor([20.0, 20.0, -20.0, -20.0]),
    )

    assert output.selected_indices.tolist() == [0, 1]
    output.weighted_aux_loss.backward()
    assert router.proj.weight.grad is not None


def test_route_plan_json_round_trip_and_replay(tmp_path: Path) -> None:
    output = _router(2)(
        torch.tensor([[2.0, 0.0], [1.0, 0.0], [3.0, 0.0]]),
        sample_ids=torch.zeros(3, dtype=torch.long),
        original_positions=torch.arange(3),
        global_token_ids=torch.tensor([12, 10, 11]),
        original_lengths={0: 3},
    )
    path = tmp_path / "round-2.json"
    output.plan.save(path)
    restored = RoutePlan.load(path)
    assert restored.to_dict() == output.plan.to_dict()

    # The current physical layout is different, but replay follows global IDs.
    current_ids = torch.tensor([11, 12, 10])
    rows = restored.replay_indices(current_ids)
    assert current_ids[rows].tolist() == restored.global_token_ids.tolist()

    replay_router = _router(2)
    with torch.no_grad():
        replay_router.proj.weight.copy_(torch.tensor([[-1.0, 0.0]]))
    replayed = replay_router(
        torch.tensor([[3.0, 0.0], [2.0, 0.0], [1.0, 0.0]], requires_grad=True),
        sample_ids=torch.zeros(3, dtype=torch.long),
        original_positions=torch.tensor([2, 0, 1]),
        global_token_ids=current_ids,
        original_lengths={0: 3},
        replay_plan=restored,
    )
    assert replayed.plan.mode == "replay"
    assert replayed.plan.global_token_ids.tolist() == restored.global_token_ids.tolist()
    torch.testing.assert_close(replayed.selected_gates, restored.selected_gates, rtol=0, atol=0)
    torch.testing.assert_close(
        replayed.plan.selected_gates, restored.selected_gates, rtol=0, atol=0
    )
    replayed.selected_gates.sum().backward()
    assert replay_router.proj.weight.grad is not None
    assert torch.count_nonzero(replay_router.proj.weight.grad) > 0


def test_full_capacity_cutoff_uses_standard_json_null_sentinel(tmp_path: Path) -> None:
    output = _router(0)(
        torch.tensor([[2.0, 0.0], [1.0, 0.0]]),
        sample_ids=torch.zeros(2, dtype=torch.long),
        original_positions=torch.arange(2),
        global_token_ids=torch.arange(2),
        original_lengths={0: 2},
    )
    assert math.isinf(output.plan.cutoff_score_margins[0])
    assert output.plan.to_dict()["cutoff_score_margins"]["0"] is None

    path = tmp_path / "full-capacity.json"
    output.plan.save(path)
    assert "Infinity" not in path.read_text(encoding="utf-8")
    restored = RoutePlan.load(path)
    assert math.isinf(restored.cutoff_score_margins[0])


def test_cutoff_margin_records_boundary_gap() -> None:
    output = _router(2)(
        torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        sample_ids=torch.zeros(3, dtype=torch.long),
        original_positions=torch.arange(3),
        global_token_ids=torch.arange(3),
        original_lengths={0: 3},
    )
    assert output.plan.cutoff_score_margins[0] > 0.0
    assert not math.isinf(output.plan.cutoff_score_margins[0])


def test_recurrent_update_is_literal_residual_gate() -> None:
    before = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    block = torch.tensor([[2.0, 4.0], [6.0, 8.0]])
    gate = torch.tensor([0.5, 0.25])
    torch.testing.assert_close(
        apply_recurrent_update(before, block, gate),
        torch.tensor([[2.0, 4.0], [4.5, 6.0]]),
    )


def test_recurrent_update_preserves_bf16_model_dtype_and_router_gradient() -> None:
    before = torch.ones((2, 3), dtype=torch.bfloat16)
    block = torch.full((2, 3), 2.0, dtype=torch.bfloat16)
    gate = torch.tensor([0.25, 0.75], dtype=torch.float32, requires_grad=True)

    updated = apply_recurrent_update(before, block, gate)
    assert updated.dtype == torch.bfloat16
    updated.float().sum().backward()
    assert gate.grad is not None
    assert torch.all(gate.grad != 0)
