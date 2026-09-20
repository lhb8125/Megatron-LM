"""Checkpoint helpers for the tiny single-rank oracle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .optim import MasterWeightAdamW


def save_reference_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: MasterWeightAdamW,
    step: int,
    metadata: dict[str, Any],
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "mor_mlite.reference.v1",
            "step": int(step),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metadata": metadata,
        },
        target,
    )
    return target


def load_reference_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: MasterWeightAdamW,
) -> tuple[int, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "mor_mlite.reference.v1":
        raise ValueError("unsupported reference checkpoint format")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    return int(payload["step"]), dict(payload.get("metadata", {}))


__all__ = ["load_reference_checkpoint", "save_reference_checkpoint"]
