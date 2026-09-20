from __future__ import annotations

import importlib
import sys
import types


def test_model_package_is_importable_without_mlite() -> None:
    package = importlib.import_module("mor_mlite.qwen3_moe_mor")

    assert package.MODEL_NAME == "qwen3_moe_mor"
    assert package.MEGATRON_LM_PINNED_SHA == ("5c8315f12a64a7279eec58896af9e74ee3351b74")
    assert "mor_mlite.qwen3_moe_mor.protocol" not in sys.modules


def test_registration_uses_mlite_public_registry_lazily(monkeypatch) -> None:
    calls: list[tuple[tuple, dict]] = []

    megatron = types.ModuleType("megatron")
    lite = types.ModuleType("megatron.lite")
    model = types.ModuleType("megatron.lite.model")
    registry = types.ModuleType("megatron.lite.model.registry")

    def register_model(*args, **kwargs) -> None:
        calls.append((args, kwargs))

    registry.register_model = register_model
    megatron.lite = lite
    lite.model = model
    model.registry = registry
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.lite", lite)
    monkeypatch.setitem(sys.modules, "megatron.lite.model", model)
    monkeypatch.setitem(sys.modules, "megatron.lite.model.registry", registry)

    registration = importlib.import_module("mor_mlite.register")
    registration.register_with_mlite()

    assert calls == [
        (
            ("qwen3_moe_mor",),
            {
                "package": "mor_mlite.qwen3_moe_mor",
                "hf_model_types": ["qwen3_moe_mor"],
                "impls": {"lite": "mor_mlite.qwen3_moe_mor.protocol"},
            },
        )
    ]
    assert "mor_mlite.qwen3_moe_mor.protocol" not in sys.modules
