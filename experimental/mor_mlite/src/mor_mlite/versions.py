"""Dependency manifest collection and production-version validation."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sysconfig
from pathlib import Path
from typing import Any

MEGATRON_SHA = "5c8315f12a64a7279eec58896af9e74ee3351b74"
MAGI_VERSION = "1.1.1"
TORCH_VERSION = "2.10.0+cu129"
TORCH_CUDA_VERSION = "12.9"
CUDA_PYTHON_VERSION = "12.9.4"
CUDA_BINDINGS_VERSION = "12.9.4"
TRITON_VERSION = "3.6.0"
NVRX_VERSION = "0.6.0"
TRANSFORMER_ENGINE_VERSION = "2.13.0"
NVCC_VERSION = "13.1"
MAGI_ENVIRONMENT = {
    "MAGI_ATTENTION_KERNEL_BACKEND": "ffa",
    "MAGI_ATTENTION_NATIVE_GRPCOLL": "0",
    "MAGI_ATTENTION_HIERARCHICAL_COMM": "0",
    "MAGI_ATTENTION_QO_COMM": "0",
    "MAGI_ATTENTION_DETERMINISTIC_MODE": "0",
    "MAGI_ATTENTION_FORWARD_HIGH_PRECISION_REDUCE": "0",
    "MAGI_ATTENTION_BACKWARD_HIGH_PRECISION_REDUCE": "0",
    "MAGI_ATTENTION_BWD_HIDE_TAIL_REDUCE": "0",
    "MAGI_ATTENTION_SDPA_BACKEND": None,
    "MAGI_ATTENTION_FA4_BACKEND": None,
}

PINNED_DISTRIBUTIONS = {
    "torch": TORCH_VERSION,
    "cuda-python": CUDA_PYTHON_VERSION,
    "cuda-bindings": CUDA_BINDINGS_VERSION,
    "triton": TRITON_VERSION,
    "nvidia-resiliency-ext": NVRX_VERSION,
    "transformer-engine": TRANSFORMER_ENGINE_VERSION,
    "transformer-engine-torch": TRANSFORMER_ENGINE_VERSION,
    "transformer-engine-cu12": TRANSFORMER_ENGINE_VERSION,
    "magi-attention": MAGI_VERSION,
}


def _package_version(*names: str) -> str | None:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _distribution_record(name: str) -> dict[str, str | None]:
    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return {"version": None, "path": None, "error": "not installed"}
    try:
        path = str(Path(distribution.locate_file("")).resolve())
    except (OSError, RuntimeError) as exc:
        return {
            "version": distribution.version,
            "path": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {"version": distribution.version, "path": path, "error": None}


def _nvcc_record() -> dict[str, str | None]:
    executable = shutil.which("nvcc")
    if not executable:
        return {"path": None, "version": None, "output": None, "error": "nvcc not found"}
    try:
        output = subprocess.check_output(
            [executable, "--version"], text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return {
            "path": str(Path(executable).resolve()),
            "version": None,
            "output": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    match = re.search(r"\brelease\s+([0-9]+\.[0-9]+)", output)
    return {
        "path": str(Path(executable).resolve()),
        "version": match.group(1) if match else None,
        "output": output,
        "error": None if match else "could not parse nvcc release",
    }


def _git_sha(path: str | os.PathLike[str] | None) -> str | None:
    if not path:
        return None
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def collect_version_manifest(*, megatron_root: str | None = None) -> dict[str, Any]:
    torch_cxx11_abi = None
    try:
        import torch

        torch_version = torch.__version__
        cuda_version = torch.version.cuda
        nccl_version = (
            ".".join(str(x) for x in torch.cuda.nccl.version())
            if torch.cuda.is_available() and torch.distributed.is_nccl_available()
            else None
        )
        cuda_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name() if cuda_available else None
        torch_cxx11_abi = getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", None)
        if torch_cxx11_abi is not None:
            torch_cxx11_abi = bool(torch_cxx11_abi)
    except ImportError:
        torch_version = cuda_version = nccl_version = gpu_name = None
        cuda_available = False

    magi_import = None
    for module_name in ("magi_attention", "magi_attention_interface"):
        try:
            module = importlib.import_module(module_name)
            magi_import = getattr(module, "__version__", "importable")
            break
        except (ImportError, OSError):
            continue

    root = megatron_root or os.environ.get("MEGATRON_LM_ROOT")
    distributions = {name: _distribution_record(name) for name in PINNED_DISTRIBUTIONS}
    return {
        "torch": torch_version,
        "cuda": cuda_version,
        "nccl": nccl_version,
        "cuda_available": cuda_available,
        "gpu": gpu_name,
        "torch_cxx11_abi": torch_cxx11_abi,
        "python_soabi": sysconfig.get_config_var("SOABI"),
        "nvcc": _nvcc_record(),
        "distributions": distributions,
        "transformer_engine": _package_version("transformer-engine"),
        "megatron_core": _package_version("megatron-core"),
        "megatron_git_sha": _git_sha(root),
        "magi_attention": _package_version("MagiAttention", "magi-attention") or magi_import,
        "magi_environment": {name: os.environ.get(name) for name in MAGI_ENVIRONMENT},
        "mor_mlite": _package_version("mor-mlite"),
    }


def validate_production_manifest(
    manifest: dict[str, Any],
    *,
    require_cuda: bool = True,
    require_magi: bool = True,
) -> list[str]:
    errors: list[str] = []
    torch_version = str(manifest.get("torch") or "")
    if torch_version != TORCH_VERSION:
        errors.append(f"expected torch {TORCH_VERSION}, got {torch_version or 'missing'}")
    cuda_version = str(manifest.get("cuda") or "")
    if require_cuda and cuda_version != TORCH_CUDA_VERSION:
        errors.append(f"expected Torch CUDA {TORCH_CUDA_VERSION}, got {cuda_version or 'missing'}")
    if require_cuda and not manifest.get("cuda_available"):
        errors.append("CUDA is not available")
    gpu = str(manifest.get("gpu") or "")
    if require_cuda and "H100" not in gpu:
        errors.append(f"expected an H100 GPU, got {gpu or 'missing'}")
    if manifest.get("torch_cxx11_abi") not in (True, False):
        errors.append("Torch C++11 ABI setting could not be determined")
    if require_cuda:
        nvcc_version = str((manifest.get("nvcc") or {}).get("version") or "")
        if nvcc_version != NVCC_VERSION:
            errors.append(
                f"expected bootstrap nvcc {NVCC_VERSION}, got {nvcc_version or 'missing'}"
            )
    if require_magi:
        actual_magi_environment = manifest.get("magi_environment") or {}
        for name, expected in MAGI_ENVIRONMENT.items():
            actual = actual_magi_environment.get(name)
            if actual != expected:
                errors.append(f"expected {name}={expected!r}, got {actual!r}")

    distributions = manifest.get("distributions") or {}
    for name, expected in PINNED_DISTRIBUTIONS.items():
        if name == "magi-attention" and not require_magi:
            continue
        actual = str((distributions.get(name) or {}).get("version") or "")
        if actual != expected:
            errors.append(f"expected {name} {expected}, got {actual or 'missing'}")

    sha = manifest.get("megatron_git_sha")
    if sha != MEGATRON_SHA:
        errors.append(f"expected Megatron-LM {MEGATRON_SHA}, got {sha or 'missing'}")
    return errors


def write_manifest(path: str | os.PathLike[str], manifest: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


__all__ = [
    "CUDA_BINDINGS_VERSION",
    "CUDA_PYTHON_VERSION",
    "MAGI_ENVIRONMENT",
    "MAGI_VERSION",
    "MEGATRON_SHA",
    "NVCC_VERSION",
    "NVRX_VERSION",
    "PINNED_DISTRIBUTIONS",
    "TORCH_CUDA_VERSION",
    "TORCH_VERSION",
    "TRANSFORMER_ENGINE_VERSION",
    "TRITON_VERSION",
    "collect_version_manifest",
    "validate_production_manifest",
    "write_manifest",
]
