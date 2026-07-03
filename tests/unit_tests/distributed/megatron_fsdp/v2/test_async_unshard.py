# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from contextlib import contextmanager
from types import SimpleNamespace

from megatron.core.distributed.fsdp.src.megatron_fsdp.v2 import fsdp_module as fsdp_module_impl


def test_async_unshard_orders_ag_stream_after_caller(monkeypatch):
    caller_stream = object()
    waits = []

    class FakeAGStream:
        def wait_stream(self, stream):
            waits.append(stream)

    ag_stream = FakeAGStream()
    monkeypatch.setattr(fsdp_module_impl.torch.cuda, "current_stream", lambda: caller_stream)

    selected = fsdp_module_impl._select_unshard_stream(
        SimpleNamespace(ag_stream=ag_stream), async_op=True
    )

    assert selected is ag_stream
    assert waits == [caller_stream]


def test_sync_unshard_stays_on_caller_stream(monkeypatch):
    caller_stream = object()
    monkeypatch.setattr(fsdp_module_impl.torch.cuda, "current_stream", lambda: caller_stream)

    selected = fsdp_module_impl._select_unshard_stream(
        SimpleNamespace(ag_stream=object()), async_op=False
    )

    assert selected is caller_stream


def test_async_coalesced_unshard_waits_for_work(monkeypatch):
    order = []
    dp_group = object()

    class FakeCoalescingManager:
        def wait(self):
            order.append("wait")

    class FakeWeightBuffer:
        def __init__(self, name):
            self.name = name

        def unshard(self, bind_params=False):
            assert bind_params
            order.append(f"unshard:{self.name}")

    @contextmanager
    def fake_coalescing_manager(group, async_ops=False):
        assert group is dp_group
        assert async_ops
        yield FakeCoalescingManager()
        order.append("launch")

    monkeypatch.setattr(fsdp_module_impl, "_coalescing_manager", fake_coalescing_manager)
    fsdp_module_impl._unshard_weight_buffers(
        dp_group, [FakeWeightBuffer("a"), FakeWeightBuffer("b")], async_op=True
    )

    assert order == ["unshard:a", "unshard:b", "launch", "wait"]


def test_single_buffer_unshard_skips_coalescing_manager(monkeypatch):
    order = []

    class FakeWeightBuffer:
        def unshard(self, bind_params=False):
            assert bind_params
            order.append("unshard")

    def unexpected_coalescing_manager(*args, **kwargs):
        raise AssertionError("A single buffer must not enter a coalescing manager")

    monkeypatch.setattr(fsdp_module_impl, "_coalescing_manager", unexpected_coalescing_manager)
    fsdp_module_impl._unshard_weight_buffers(object(), [FakeWeightBuffer()], async_op=False)

    assert order == ["unshard"]
