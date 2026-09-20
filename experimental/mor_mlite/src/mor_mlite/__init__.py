"""Mixture-of-Recursions extensions for Megatron-Lite.

Importing this package is side-effect free.  Call :func:`register_with_mlite`
before constructing a Megatron-Lite runtime, or use the provided CLI which
does that automatically.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mor-mlite")
except PackageNotFoundError:  # editable source tree without installation
    __version__ = "0.1.0"


def register_with_mlite() -> None:
    """Register the external ``qwen3_moe_mor`` model with MLite."""

    from .register import register_with_mlite as _register

    _register()


__all__ = ["__version__", "register_with_mlite"]
