from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from mor_mlite.distributed import (
    active_sequence_alignment,
    pack_active_sequences,
    pack_route_plan_canonical,
)
from mor_mlite.routing import RoutePlan


def test_cp1_pads_each_sample_tail_to_tp_and_preserves_sparse_positions() -> None:
    packed = pack_active_sequences(
        sample_ids=torch.tensor([1, 0, 1, 0, 0]),
        original_positions=torch.tensor([4, 7, 1, 0, 3]),
        global_token_ids=torch.tensor([14, 7, 11, 0, 3]),
        original_lengths={0: 8, 1: 5},
        tp_size=2,
        cp_size=1,
        round_index=1,
    )

    assert packed.alignment == 2
    assert packed.active_cu_seqlens.tolist() == [0, 3, 5]
    assert packed.cu_seqlens.tolist() == [0, 4, 6]
    assert packed.padding_mask.tolist() == [False, False, False, True, False, False]
    assert packed.sample_ids.tolist() == [0, 0, 0, 0, 1, 1]
    assert packed.position_ids.tolist() == [0, 3, 7, 8, 1, 4]
    assert packed.global_token_ids[packed.padding_mask].tolist()[0] < 0
    assert packed.global_token_ids[~packed.padding_mask].tolist() == [0, 3, 7, 11, 14]
    assert packed.source_rows.tolist() == [3, 4, 1, -1, 2, 0]

    values = torch.arange(packed.num_rows)
    assert packed.strip_padding(values).tolist() == [0, 1, 2, 4, 5]


def test_magi_alignment_is_tp_times_two_cp_and_dummy_ids_are_round_unique() -> None:
    kwargs = {
        "sample_ids": torch.tensor([0, 0, 0, 1, 1]),
        "original_positions": torch.tensor([0, 3, 7, 1, 4]),
        "global_token_ids": torch.tensor([0, 3, 7, 11, 14]),
        "original_lengths": {0: 8, 1: 5},
        "tp_size": 2,
        "cp_size": 2,
        "use_magi": True,
    }
    first = pack_active_sequences(**kwargs, round_index=0)
    second = pack_active_sequences(**kwargs, round_index=1)

    assert active_sequence_alignment(tp_size=2, cp_size=2) == 8
    assert first.alignment == 8
    assert first.cu_seqlens.tolist() == [0, 8, 16]
    assert first.num_active_tokens == 5
    first_dummies = set(first.global_token_ids[first.padding_mask].tolist())
    second_dummies = set(second.global_token_ids[second.padding_mask].tolist())
    assert len(first_dummies) == 11
    assert all(token_id < 0 for token_id in first_dummies)
    assert first_dummies.isdisjoint(second_dummies)
    assert first.magi_decode_kwargs()["canonical_padding_mask"] is first.padding_mask


def test_canonical_packing_is_independent_of_input_order() -> None:
    common = {"tp_size": 1, "cp_size": 1, "original_lengths": {0: 6}}
    first = pack_active_sequences(
        sample_ids=torch.zeros(3, dtype=torch.long),
        original_positions=torch.tensor([5, 0, 2]),
        global_token_ids=torch.tensor([15, 10, 12]),
        **common,
    )
    second = pack_active_sequences(
        sample_ids=torch.zeros(3, dtype=torch.long),
        original_positions=torch.tensor([2, 5, 0]),
        global_token_ids=torch.tensor([12, 15, 10]),
        **common,
    )
    assert first.sample_ids.tolist() == second.sample_ids.tolist()
    assert first.position_ids.tolist() == second.position_ids.tolist()
    assert first.global_token_ids.tolist() == second.global_token_ids.tolist()
    assert first.padding_mask.tolist() == second.padding_mask.tolist()


def _plan() -> RoutePlan:
    size = 5
    zeros = torch.zeros(size, dtype=torch.long)
    return RoutePlan(
        round_index=1,
        mode="learned",
        sample_ids=torch.tensor([0, 0, 0, 1, 1]),
        original_positions=torch.tensor([0, 3, 7, 1, 4]),
        global_token_ids=torch.tensor([0, 3, 7, 11, 14]),
        source_tp_ranks=zeros,
        source_cp_ranks=zeros,
        source_local_rows=torch.arange(size),
        target_tp_ranks=zeros,
        target_cp_ranks=zeros,
        target_local_rows=torch.arange(size),
        selected_gates=torch.linspace(0.01, 0.05, size),
        active_cu_seqlens=torch.tensor([0, 3, 5], dtype=torch.int32),
        padding_mask=torch.zeros(size, dtype=torch.bool),
        cutoff_score_margins={0: 0.1, 1: 0.2},
    )


def test_route_plan_wrapper_preserves_real_cu_seqlens() -> None:
    packed = pack_route_plan_canonical(
        _plan(),
        tp_size=2,
        cp_size=2,
        use_magi=True,
        original_lengths={0: 8, 1: 5},
    )
    assert packed.active_cu_seqlens.tolist() == [0, 3, 5]
    assert packed.cu_seqlens.tolist() == [0, 8, 16]


def test_cp_without_magi_and_duplicate_positions_fail_closed() -> None:
    with pytest.raises(ValueError, match="requires the MagiAttention layout"):
        active_sequence_alignment(tp_size=1, cp_size=2, use_magi=False)
    with pytest.raises(ValueError, match="duplicate original positions"):
        pack_active_sequences(
            sample_ids=torch.tensor([0, 0]),
            original_positions=torch.tensor([1, 1]),
            global_token_ids=torch.tensor([1, 2]),
            tp_size=1,
            cp_size=1,
        )
