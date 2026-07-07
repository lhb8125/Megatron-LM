# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Focused tests for Megatron-FSDP v2 hook dispatch."""

import weakref
from types import SimpleNamespace
from unittest.mock import Mock, call

import torch.nn as nn

from megatron.core.distributed.fsdp.src.megatron_fsdp.v2 import hooks


class _HookTarget:
    pass


def _make_hook_target(*, backward_phase):
    target = _HookTarget()
    target._fsdp_root_context = SimpleNamespace(
        backward_phase=backward_phase,
        cuda_graph_active=False,
        enable_unshard_prefetch=False,
        enable_cuda_graph=False,
    )
    target._fsdp_state = SimpleNamespace(_is_root=False, enable_cuda_graph=False)
    target._fsdp_param_groups = []
    target.unshard = Mock()
    target.unshard_for_submodule = Mock()

    child = nn.Identity()
    child._fsdp_parent_module = weakref.ref(target)
    return target, child


def test_recompute_forward_uses_targeted_unshard(monkeypatch):
    target, child = _make_hook_target(backward_phase=True)
    monkeypatch.setattr(hooks, "is_recomputing", lambda: True)

    hooks.mfsdp_forward_pre_hook(child, (), {})

    target.unshard.assert_called_once_with(async_op=False, bwd_pass=True)
    target.unshard_for_submodule.assert_called_once_with(child, async_op=False)


def test_overlapped_normal_forward_keeps_full_unshard(monkeypatch):
    target, child = _make_hook_target(backward_phase=True)
    monkeypatch.setattr(hooks, "is_recomputing", lambda: False)

    hooks.mfsdp_forward_pre_hook(child, (), {})

    assert target.unshard.call_args_list == [
        call(async_op=False, bwd_pass=True),
        call(async_op=False, bwd_pass=False),
    ]
    target.unshard_for_submodule.assert_not_called()
