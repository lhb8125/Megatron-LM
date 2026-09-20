"""Out-of-tree Qwen3-MoE Mixture-of-Recursions model package.

The package root intentionally contains no Megatron-Lite imports.  MLite only
loads :mod:`mor_mlite.qwen3_moe_mor.protocol` after the caller explicitly
registers the model and asks the runtime to build it.
"""

from __future__ import annotations

from .metadata import (
    HF_FOLDING_METADATA_FILENAME,
    MEGATRON_LM_PINNED_SHA,
    MoRCheckpointMetadata,
    physical_to_logical_layer_map,
)

MODEL_NAME = "qwen3_moe_mor"
MODEL_PACKAGE = "mor_mlite.qwen3_moe_mor"
PROTOCOL_MODULE = "mor_mlite.qwen3_moe_mor.protocol"


def register_with_mlite() -> None:
    """Register this external model through MLite's public registry API."""

    from mor_mlite.register import register_with_mlite as _register

    _register()


__all__ = [
    "HF_FOLDING_METADATA_FILENAME",
    "MEGATRON_LM_PINNED_SHA",
    "MODEL_NAME",
    "MODEL_PACKAGE",
    "PROTOCOL_MODULE",
    "MoRCheckpointMetadata",
    "physical_to_logical_layer_map",
    "register_with_mlite",
]
