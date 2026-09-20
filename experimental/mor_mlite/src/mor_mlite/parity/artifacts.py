"""On-disk artifacts for baseline/candidate parity comparisons."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from mor_mlite.provenance import source_snapshot

FORMAT_VERSION = 1


def require_fresh_artifact_directory(directory: str | Path) -> Path:
    """Reject stale files instead of letting checkpoint step discovery reuse them."""

    root = Path(directory)
    if root.exists() and not root.is_dir():
        raise FileExistsError(f"artifact output exists and is not a directory: {root}")
    if root.is_dir():
        existing = sorted(path.name for path in root.iterdir())
        if existing:
            preview = ", ".join(existing[:5])
            suffix = " ..." if len(existing) > 5 else ""
            raise FileExistsError(
                f"artifact output must be empty to prevent stale-step reuse: "
                f"{root} contains {preview}{suffix}"
            )
    return root


def tensor_sha256(tensor: torch.Tensor) -> str:
    # Reinterpret after flattening so zero-dimensional loss/grad-norm tensors
    # have a byte-addressable dimension as well.
    value = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def save_artifact(
    directory: str | Path,
    *,
    metadata: Mapping[str, Any],
    tensors: Mapping[str, torch.Tensor],
    routes: list[dict[str, Any]],
) -> Path:
    current_source = source_snapshot()
    started_source = metadata.get("source_snapshot", current_source)
    if started_source != current_source:
        raise RuntimeError("package source changed while this experiment was running")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    cpu_tensors = {name: tensor.detach().cpu() for name, tensor in tensors.items()}
    manifest = {
        "format_version": FORMAT_VERSION,
        **dict(metadata),
        "source_snapshot": current_source,
        "tensor_index": {
            name: {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "sha256": tensor_sha256(tensor),
            }
            for name, tensor in sorted(cpu_tensors.items())
        },
        "route_count": len(routes),
    }
    torch.save(cpu_tensors, root / "tensors.pt")
    (root / "routes.json").write_text(
        json.dumps(routes, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return root


def load_artifact(directory: str | Path) -> tuple[dict[str, Any], dict[str, torch.Tensor], list]:
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError("parity manifest must contain a JSON object")
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"unsupported parity artifact format {manifest.get('format_version')}; "
            f"expected {FORMAT_VERSION}"
        )
    tensors = torch.load(root / "tensors.pt", map_location="cpu", weights_only=True)
    if not isinstance(tensors, dict) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in tensors.items()
    ):
        raise TypeError("tensors.pt is not a string-to-tensor mapping")
    tensor_index = manifest.get("tensor_index")
    if not isinstance(tensor_index, dict) or set(tensor_index) != set(tensors):
        raise ValueError("manifest tensor_index does not exactly cover tensors.pt")
    for name, tensor in tensors.items():
        record = tensor_index[name]
        if not isinstance(record, dict):
            raise TypeError(f"tensor_index entry {name!r} must be a mapping")
        expected = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "sha256": tensor_sha256(tensor),
        }
        if record != expected:
            raise ValueError(f"tensor_index integrity mismatch for {name!r}")
    routes = json.loads((root / "routes.json").read_text(encoding="utf-8"))
    if not isinstance(routes, list) or not all(isinstance(route, dict) for route in routes):
        raise TypeError("routes.json must contain a list of RoutePlan mappings")
    if manifest.get("route_count") != len(routes):
        raise ValueError("manifest route_count does not match routes.json")
    from mor_mlite.routing import RoutePlan

    seen_route_keys: set[tuple[str, int, int, int]] = set()
    for route in routes:
        plan = RoutePlan.from_dict(route)
        key = (
            str(route.get("phase", "train")),
            int(route.get("step", 0)),
            int(route.get("microbatch", 0)),
            plan.round_index,
        )
        if key in seen_route_keys:
            raise ValueError(f"duplicate RoutePlan key in artifact: {key}")
        seen_route_keys.add(key)
    return manifest, tensors, routes


__all__ = [
    "FORMAT_VERSION",
    "load_artifact",
    "require_fresh_artifact_directory",
    "save_artifact",
    "tensor_sha256",
]
