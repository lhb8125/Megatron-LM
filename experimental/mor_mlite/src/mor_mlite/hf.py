"""Resolve and validate local Hugging Face safetensors checkpoints.

The pinned MLite config loader accepts either a local directory or a Hub ID,
whereas its :class:`SafeTensorReader` consumes a local directory.  Keeping this
boundary in the out-of-tree package prevents a late, memory-expensive failure
after the Qwen model has already been allocated.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_HF_WEIGHT_PATTERNS = (
    "config.json",
    "model.safetensors",
    "model.safetensors.index.json",
    "model-*.safetensors",
)


@dataclass(frozen=True, slots=True)
class ResolvedHFCheckpoint:
    """The user-visible source and the concrete directory MLite must read."""

    source: str
    local_path: Path
    downloaded: bool


def _validate_index(index_path: Path) -> None:
    try:
        payload: Any = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = payload["weight_map"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"invalid HF safetensors index: {index_path}") from exc
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"HF safetensors index has an empty weight_map: {index_path}")

    unsafe: list[object] = []
    for filename in weight_map.values():
        # The index is untrusted input.  A shard may itself be a symlink, but
        # the index entry must remain a basename below the checkpoint root so
        # SafeTensorReader cannot be directed to an absolute/parent path.
        if (
            not isinstance(filename, str)
            or not filename
            or filename in {".", ".."}
            or Path(filename).is_absolute()
            or Path(filename).name != filename
            or "/" in filename
            or "\\" in filename
        ):
            unsafe.append(filename)
    if unsafe:
        preview = ", ".join(repr(filename) for filename in unsafe[:4])
        suffix = " ..." if len(unsafe) > 4 else ""
        raise ValueError(
            f"HF safetensors index shard paths must be safe relative basenames: {preview}{suffix}"
        )

    missing = sorted(
        {
            filename
            for filename in weight_map.values()
            if not (index_path.parent / filename).is_file()
        }
    )
    if missing:
        preview = ", ".join(missing[:4])
        suffix = " ..." if len(missing) > 4 else ""
        raise FileNotFoundError(
            f"HF safetensors index references missing shard(s): {preview}{suffix}"
        )


def validate_local_hf_checkpoint(
    path: str | os.PathLike[str], *, require_weights: bool = True
) -> Path:
    """Return an absolute checkpoint directory after fail-fast validation."""

    candidate = Path(path).expanduser()
    if candidate.is_file() and candidate.name == "config.json":
        candidate = candidate.parent
    if not candidate.is_dir():
        raise FileNotFoundError(f"HF checkpoint directory does not exist: {candidate}")
    candidate = candidate.resolve()
    config_path = candidate / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"HF checkpoint has no config.json: {candidate}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid HF config.json: {config_path}") from exc
    if not isinstance(config, dict):
        raise TypeError(f"HF config.json must contain a JSON object: {config_path}")

    if require_weights:
        single = candidate / "model.safetensors"
        index = candidate / "model.safetensors.index.json"
        if index.is_file():
            _validate_index(index)
        elif not single.is_file():
            raise FileNotFoundError(
                "HF checkpoint has neither model.safetensors nor "
                f"model.safetensors.index.json: {candidate}"
            )
    return candidate


def resolve_hf_checkpoint(
    source: str | os.PathLike[str],
    *,
    require_weights: bool = True,
    downloader: Callable[..., str] | None = None,
) -> ResolvedHFCheckpoint:
    """Resolve ``source`` to the validated local snapshot required by MLite.

    ``snapshot_download`` is process-safe through the Hugging Face cache lock,
    so this function is safe when every local torchrun worker reaches it before
    MLite initializes NCCL.  EOS launchers still resolve once before torchrun to
    avoid eight processes waiting on that lock.
    """

    source_text = os.fspath(source)
    if not source_text:
        raise ValueError("HF checkpoint source must not be empty")
    local_candidate = Path(source_text).expanduser()
    if local_candidate.exists():
        return ResolvedHFCheckpoint(
            source=source_text,
            local_path=validate_local_hf_checkpoint(
                local_candidate, require_weights=require_weights
            ),
            downloaded=False,
        )

    if downloader is None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                f"{source_text!r} is not a local checkpoint and huggingface-hub is unavailable"
            ) from exc
        downloader = snapshot_download
    resolved = downloader(repo_id=source_text, allow_patterns=_HF_WEIGHT_PATTERNS)
    return ResolvedHFCheckpoint(
        source=source_text,
        local_path=validate_local_hf_checkpoint(resolved, require_weights=require_weights),
        downloaded=True,
    )


__all__ = [
    "ResolvedHFCheckpoint",
    "resolve_hf_checkpoint",
    "validate_local_hf_checkpoint",
]
