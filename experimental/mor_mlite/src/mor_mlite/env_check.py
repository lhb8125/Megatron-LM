"""Fail-closed production environment check for EOS/H100 jobs.

The checker deliberately prints its complete, non-secret manifest before it
returns a failure.  That makes a rejected Slurm job diagnosable without ever
starting model construction or distributed collectives.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

from mor_mlite.versions import (
    MAGI_VERSION,
    MEGATRON_SHA,
    PINNED_DISTRIBUTIONS,
    collect_version_manifest,
    validate_production_manifest,
    write_manifest,
)

EOS_ACCOUNT = "coreai_devtech_all"
EOS_PARTITION = "batch"
CONTAINER_IMAGE = "nvcr.io/nvidia/pytorch:26.01-py3"
MAX_WORLD_SIZE = 8
MAGI_TAG = "v1.1.1"


def _import_record(module_name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - extension imports raise backend-specific errors.
        return {
            "importable": False,
            "error": f"{type(exc).__name__}: {exc}",
            "file": None,
            "paths": [],
        }
    module_paths = [str(Path(path).resolve()) for path in getattr(module, "__path__", [])]
    return {
        "importable": True,
        "error": None,
        "file": str(Path(module.__file__).resolve()) if getattr(module, "__file__", None) else "",
        "paths": module_paths,
    }


def _git_record(path: str | None) -> dict[str, Any]:
    if not path:
        return {"path": None, "sha": None, "exact_tag": None}
    record: dict[str, Any] = {"path": str(Path(path).resolve())}
    for field, command in (
        ("sha", ("rev-parse", "HEAD")),
        ("exact_tag", ("describe", "--tags", "--exact-match", "HEAD")),
    ):
        try:
            record[field] = subprocess.check_output(
                ["git", "-C", path, *command],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            record[field] = None
    return record


def _runtime_details() -> dict[str, Any]:
    details: dict[str, Any] = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "python_prefix": sys.prefix,
        "python_base_prefix": sys.base_prefix,
        "mor_venv": os.environ.get("MOR_VENV"),
        "configured_project_root": os.environ.get("MOR_PROJECT_ROOT"),
        "loaded_project_root": str(Path(__file__).resolve().parents[2]),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "account": os.environ.get("SLURM_JOB_ACCOUNT"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "num_nodes": os.environ.get("SLURM_JOB_NUM_NODES"),
            "node_list": os.environ.get("SLURM_JOB_NODELIST"),
        },
        "expected_container_image": CONTAINER_IMAGE,
        "configured_container_image": os.environ.get("MOR_CONTAINER_IMAGE_SOURCE"),
        "imports": {
            "megatron.lite": _import_record("megatron.lite"),
            "megatron.core": _import_record("megatron.core"),
            "transformer_engine": _import_record("transformer_engine"),
            "transformer_engine.pytorch": _import_record("transformer_engine.pytorch"),
            "transformer_engine_torch": _import_record("transformer_engine_torch"),
            "torch": _import_record("torch"),
            "triton": _import_record("triton"),
            "cuda.bindings": _import_record("cuda.bindings"),
            "nvidia_resiliency_ext": _import_record("nvidia_resiliency_ext"),
            "magi_attention": _import_record("magi_attention"),
            "magi_attention.magi_attn_ext": _import_record("magi_attention.magi_attn_ext"),
        },
    }

    try:
        import torch

        devices = []
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                devices.append(
                    {
                        "index": index,
                        "name": torch.cuda.get_device_name(index),
                        "capability": list(torch.cuda.get_device_capability(index)),
                    }
                )
        details["cuda_devices"] = devices
        details["cuda_device_count"] = len(devices)
    except Exception as exc:  # noqa: BLE001 - device probing is reported in the manifest.
        details["cuda_devices"] = []
        details["cuda_device_count"] = 0
        details["torch_probe_error"] = f"{type(exc).__name__}: {exc}"
    return details


def build_manifest(*, megatron_root: str | None = None) -> dict[str, Any]:
    try:
        manifest = collect_version_manifest(megatron_root=megatron_root)
    except Exception as exc:  # noqa: BLE001 - fail-closed validation reports the root error.
        manifest = {
            "torch": None,
            "cuda": None,
            "nccl": None,
            "cuda_available": False,
            "gpu": None,
            "transformer_engine": None,
            "megatron_core": None,
            "megatron_git_sha": None,
            "magi_attention": None,
            "mor_mlite": None,
            "version_collection_error": f"{type(exc).__name__}: {exc}",
        }
    manifest.update(_runtime_details())
    manifest["mlite_source"] = _git_record(megatron_root)
    manifest["magi_source"] = _git_record(os.environ.get("MAGI_ATTENTION_SOURCE"))
    manifest["pins"] = {
        "megatron_git_sha": MEGATRON_SHA,
        "magi_attention": MAGI_VERSION,
        "container_image": CONTAINER_IMAGE,
        "distributions": PINNED_DISTRIBUTIONS,
    }
    return manifest


def validate_manifest(
    manifest: dict[str, Any],
    *,
    expected_world_size: int,
    require_cuda: bool = True,
    require_magi: bool = True,
    enforce_eos_job: bool = True,
) -> list[str]:
    errors = validate_production_manifest(
        manifest,
        require_cuda=require_cuda,
        require_magi=require_magi,
    )
    if not 1 <= expected_world_size <= MAX_WORLD_SIZE:
        errors.append(
            f"expected world size must be in [1, {MAX_WORLD_SIZE}], got {expected_world_size}"
        )

    imports = manifest.get("imports", {})
    if not imports.get("megatron.lite", {}).get("importable"):
        errors.append("megatron.lite is not importable from the pinned checkout")
    if not imports.get("megatron.core", {}).get("importable"):
        errors.append("megatron.core is not importable from the pinned checkout")
    if not imports.get("transformer_engine", {}).get("importable"):
        errors.append("Transformer Engine is not importable")

    venv_value = str(manifest.get("mor_venv") or "")
    venv = Path(venv_value).resolve() if venv_value else None
    if enforce_eos_job and venv is None:
        errors.append("MOR_VENV is not configured")
    if venv is not None:
        prefix = Path(str(manifest.get("python_prefix") or "")).resolve()
        if prefix != venv:
            errors.append(f"Python prefix resolved outside MOR_VENV: {prefix}")
        required_modules = {
            "torch",
            "triton",
            "cuda.bindings",
            "nvidia_resiliency_ext",
            "transformer_engine",
            "transformer_engine.pytorch",
            "transformer_engine_torch",
        }
        if require_magi:
            required_modules.update({"magi_attention", "magi_attention.magi_attn_ext"})
        for module_name in sorted(required_modules):
            record = imports.get(module_name, {})
            if not record.get("importable"):
                errors.append(f"{module_name} is not importable from MOR_VENV")
                continue
            origins = [record.get("file"), *(record.get("paths") or [])]
            origins = [Path(str(path)).resolve() for path in origins if path]
            if not origins or any(not path.is_relative_to(venv) for path in origins):
                errors.append(f"{module_name} resolved outside MOR_VENV: {origins or 'missing'}")
        distributions = manifest.get("distributions") or {}
        for name in PINNED_DISTRIBUTIONS:
            if name == "magi-attention" and not require_magi:
                continue
            path_value = (distributions.get(name) or {}).get("path")
            path = Path(str(path_value)).resolve() if path_value else None
            if path is None or not path.is_relative_to(venv):
                errors.append(
                    f"{name} distribution resolved outside MOR_VENV: {path_value or 'missing'}"
                )
    if require_magi:
        if not imports.get("magi_attention", {}).get("importable"):
            errors.append("magi_attention is not importable")
        if not imports.get("magi_attention.magi_attn_ext", {}).get("importable"):
            errors.append("MagiAttention sm90 CUDA extension is not importable")
        if manifest.get("magi_source", {}).get("exact_tag") != MAGI_TAG:
            errors.append(
                f"expected MagiAttention source tag {MAGI_TAG}, got "
                f"{manifest.get('magi_source', {}).get('exact_tag') or 'missing'}"
            )

    if require_cuda:
        devices = manifest.get("cuda_devices") or []
        if len(devices) < expected_world_size:
            errors.append(f"need {expected_world_size} visible GPUs, found {len(devices)}")
        for device in devices[:expected_world_size]:
            name = str(device.get("name") or "")
            capability = device.get("capability")
            if "H100" not in name:
                errors.append(f"CUDA device {device.get('index')} is not H100: {name or 'unknown'}")
            if capability != [9, 0]:
                errors.append(f"CUDA device {device.get('index')} is not sm90: {capability!r}")

    if require_cuda and not manifest.get("nccl"):
        errors.append("NCCL version could not be determined")
    if not manifest.get("transformer_engine"):
        errors.append("Transformer Engine package version could not be determined")

    mlite_file = str(imports.get("megatron.lite", {}).get("file") or "")
    if mlite_file and "/experimental/lite/megatron/lite/" not in mlite_file:
        errors.append(f"megatron.lite resolved outside experimental/lite: {mlite_file}")
    megatron_root = str(manifest.get("mlite_source", {}).get("path") or "")
    core_file = str(imports.get("megatron.core", {}).get("file") or "")
    if megatron_root and core_file:
        expected_core_root = str(Path(megatron_root) / "megatron" / "core")
        if not Path(core_file).resolve().is_relative_to(Path(expected_core_root)):
            errors.append(f"megatron.core resolved outside pinned source: {core_file}")

    configured_image = manifest.get("configured_container_image")
    if configured_image and configured_image != CONTAINER_IMAGE:
        errors.append(f"expected container image {CONTAINER_IMAGE}, got {configured_image}")

    slurm = manifest.get("slurm", {})
    if enforce_eos_job:
        if slurm.get("job_id"):
            if slurm.get("account") != EOS_ACCOUNT:
                errors.append(
                    f"expected Slurm account {EOS_ACCOUNT}, got {slurm.get('account') or 'missing'}"
                )
            if slurm.get("partition") != EOS_PARTITION:
                errors.append(
                    f"expected Slurm partition {EOS_PARTITION}, got "
                    f"{slurm.get('partition') or 'missing'}"
                )
            if str(slurm.get("num_nodes") or "") != "1":
                errors.append(
                    f"v1 is single-node only, got SLURM_JOB_NUM_NODES="
                    f"{slurm.get('num_nodes') or 'missing'}"
                )
        configured_root = manifest.get("configured_project_root")
        loaded_root = manifest.get("loaded_project_root")
        if not configured_root or not Path(configured_root).is_absolute():
            errors.append("MOR_PROJECT_ROOT must specify an absolute project directory")
        elif not loaded_root or Path(loaded_root).resolve() != Path(configured_root).resolve():
            errors.append(
                f"expected project at {configured_root}, resolved package root is {loaded_root}"
            )
    return errors


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mor_mlite.env_check",
        description="print and validate the pinned EOS/H100 runtime manifest",
    )
    parser.add_argument(
        "--megatron-root",
        default=os.environ.get("MEGATRON_LM_ROOT"),
        help="pinned Megatron-LM checkout (defaults to MEGATRON_LM_ROOT)",
    )
    parser.add_argument("--output", type=Path, help="optional JSON manifest output")
    parser.add_argument("--expected-world-size", type=int, default=1)
    parser.add_argument("--no-cuda", action="store_true", help="developer-only CPU probe")
    parser.add_argument("--no-magi", action="store_true", help="skip Magi import/version checks")
    parser.add_argument(
        "--allow-non-eos",
        action="store_true",
        help="skip EOS source-root/account/partition checks for local diagnostics",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = build_manifest(megatron_root=args.megatron_root)
    errors = validate_manifest(
        manifest,
        expected_world_size=args.expected_world_size,
        require_cuda=not args.no_cuda,
        require_magi=not args.no_magi,
        enforce_eos_job=not args.allow_non_eos,
    )
    manifest["validation"] = {"passed": not errors, "errors": errors}
    if args.output:
        write_manifest(args.output, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if errors:
        print("environment validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
