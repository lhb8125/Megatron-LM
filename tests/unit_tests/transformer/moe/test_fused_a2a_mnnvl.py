from types import SimpleNamespace

import pytest

from megatron.core.transformer.moe import fused_a2a


class _FakeGroup:
    def size(self):
        return 16


class _FakeConfig:
    def get_nvl_buffer_size_hint(self, hidden_bytes, group_size):
        return hidden_bytes * group_size

    def get_rdma_buffer_size_hint(self, hidden_bytes, group_size):
        return hidden_bytes * group_size * 2


class _FakeBufferBase:
    calls = []

    @staticmethod
    def get_dispatch_config(group_size):
        return _FakeConfig()

    @staticmethod
    def get_combine_config(group_size):
        return _FakeConfig()

    def _record_init(self, group, num_nvl_bytes, num_rdma_bytes, **kwargs):
        self.group = group
        self.num_nvl_bytes = num_nvl_bytes
        self.num_rdma_bytes = num_rdma_bytes
        self.calls.append(SimpleNamespace(kwargs=kwargs))


class _LegacyBuffer(_FakeBufferBase):
    def __init__(self, group, num_nvl_bytes, num_rdma_bytes, allow_mnnvl=False, use_fabric=False):
        kwargs = {}
        if allow_mnnvl:
            kwargs["allow_mnnvl"] = True
        if use_fabric:
            kwargs["use_fabric"] = True
        self._record_init(group, num_nvl_bytes, num_rdma_bytes, **kwargs)


class _CurrentBuffer(_FakeBufferBase):
    def __init__(self, group, num_nvl_bytes, num_rdma_bytes, allow_mnnvl=False):
        kwargs = {"allow_mnnvl": True} if allow_mnnvl else {}
        self._record_init(group, num_nvl_bytes, num_rdma_bytes, **kwargs)


@pytest.fixture(autouse=True)
def _reset_buffer(monkeypatch):
    monkeypatch.setattr(fused_a2a, "Buffer", _LegacyBuffer)
    monkeypatch.setattr(fused_a2a.torch.cuda, "device_count", lambda: 4)
    monkeypatch.delenv("USE_MNNVL", raising=False)
    fused_a2a._buffer = None
    _FakeBufferBase.calls.clear()
    yield
    fused_a2a._buffer = None


def test_get_buffer_uses_deepep_defaults_without_mnnvl():
    fused_a2a.get_buffer(_FakeGroup(), hidden_bytes=128)

    assert _FakeBufferBase.calls[-1].kwargs == {}


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_get_buffer_enables_legacy_fabric_handles_for_mnnvl(monkeypatch, value):
    monkeypatch.setenv("USE_MNNVL", value)

    fused_a2a.get_buffer(_FakeGroup(), hidden_bytes=128)

    assert _FakeBufferBase.calls[-1].kwargs == {"allow_mnnvl": True, "use_fabric": True}


def test_get_buffer_uses_current_mnnvl_api(monkeypatch):
    monkeypatch.setattr(fused_a2a, "Buffer", _CurrentBuffer)
    monkeypatch.setenv("USE_MNNVL", "1")

    fused_a2a.get_buffer(_FakeGroup(), hidden_bytes=128)

    assert _FakeBufferBase.calls[-1].kwargs == {"allow_mnnvl": True}
