# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Focused tests for Megatron-FSDP v2 mixed-precision helpers."""

from megatron.core.distributed.fsdp.src.megatron_fsdp.v2 import fsdp_module, mixed_precision
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.fsdp_module import FSDPModule
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.mixed_precision import (
    FP8WeightUpdate,
    apply_fp8_weight_updates,
)


def _update(group, value):
    """Build a compact synthetic update request."""
    return FP8WeightUpdate(
        model_params=[f"model-{value}"],
        main_params=[f"main-{value}"],
        start_offsets=[value],
        data_parallel_group=group,
        fsdp_shard_model_params=[(f"row-{value}", f"column-{value}")],
    )


def test_apply_fp8_weight_updates_batches_by_process_group_identity(monkeypatch):
    """Requests sharing the exact process group should use one quantize call."""
    first_group = object()
    second_group = object()
    calls = []

    monkeypatch.setattr(
        mixed_precision, "quantize_main_weights_to_fp8", lambda *args: calls.append(args)
    )

    apply_fp8_weight_updates(
        [_update(first_group, 0), _update(second_group, 1), _update(first_group, 2)]
    )

    assert len(calls) == 2
    assert calls[0] == (
        ["model-0", "model-2"],
        ["main-0", "main-2"],
        [0, 2],
        first_group,
        [("row-0", "column-0"), ("row-2", "column-2")],
    )
    assert calls[1] == (["model-1"], ["main-1"], [1], second_group, [("row-1", "column-1")])


def test_apply_fp8_weight_updates_accepts_no_requests(monkeypatch):
    """An all-BF16 model should not issue an empty quantize call."""
    calls = []
    monkeypatch.setattr(
        mixed_precision, "quantize_main_weights_to_fp8", lambda *args: calls.append(args)
    )

    apply_fp8_weight_updates([])

    assert calls == []


def test_root_weight_refresh_bounds_fp8_batches(monkeypatch):
    """The root should flush same-group FP8 updates in bounded batches."""
    group = object()
    updates = [_update(group, value) for value in range(18)]

    class FakeParameterGroup:
        """Minimal parameter group that contributes one deferred update."""

        def __init__(self, update):
            self.update = update

        def copy_main_weights_to_model_weights(self, queue):
            """Append this group's update to the root-owned queue."""
            queue.append(self.update)

    class FakeFSDPModule(FSDPModule):
        """Minimal FSDP module tree used by the root refresh helper."""

        def __init__(self, param_groups, children=()):
            self._fsdp_param_groups = param_groups
            self.children = children

        def modules(self):
            """Yield this module and its direct synthetic children."""
            return iter((self, *self.children))

    root = FakeFSDPModule([FakeParameterGroup(update) for update in updates])
    applied = []
    monkeypatch.setattr(
        fsdp_module, "apply_fp8_weight_updates", lambda batch: applied.append(list(batch))
    )

    root._copy_main_weights_to_model_weights()

    assert applied == [updates[:8], updates[8:16], updates[16:]]
