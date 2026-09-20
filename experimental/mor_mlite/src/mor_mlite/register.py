"""Explicit registration of the external model with Megatron-Lite."""

from __future__ import annotations

MODEL_NAME = "qwen3_moe_mor"
MODEL_PACKAGE = "mor_mlite.qwen3_moe_mor"
PROTOCOL_MODULE = "mor_mlite.qwen3_moe_mor.protocol"


def register_with_mlite() -> None:
    """Register ``qwen3_moe_mor`` using MLite's public ``register_model``.

    Megatron-Lite remains an optional dependency: importing :mod:`mor_mlite`
    or this module does not import it.  The dependency is resolved only when
    the caller explicitly requests registration.
    """

    try:
        from megatron.lite.model.registry import register_model
    except (ImportError, OSError) as exc:
        raise ImportError(
            "qwen3_moe_mor registration requires the pinned Megatron-LM "
            "experimental/lite package on PYTHONPATH"
        ) from exc

    register_model(
        MODEL_NAME,
        package=MODEL_PACKAGE,
        hf_model_types=[MODEL_NAME],
        impls={"lite": PROTOCOL_MODULE},
    )


__all__ = [
    "MODEL_NAME",
    "MODEL_PACKAGE",
    "PROTOCOL_MODULE",
    "register_with_mlite",
]
