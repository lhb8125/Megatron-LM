"""Strict numerical-mode helpers used by parity runs."""

from __future__ import annotations

import os
import random

import torch


def configure_determinism(seed: int, *, strict: bool = True) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("MEGATRON_LITE_DETERMINISTIC", "1")
    if strict:
        torch.use_deterministic_algorithms(True, warn_only=False)
        if hasattr(torch.backends, "cuda"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)


__all__ = ["configure_determinism"]
