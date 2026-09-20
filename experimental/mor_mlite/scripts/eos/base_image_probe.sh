#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_eos_project
export PYTHONPATH="${EOS_WORKDIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

python - <<'PY'
import importlib.metadata
import json

import torch


def version(*names):
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return None


manifest = {
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "nccl": (
        ".".join(map(str, torch.cuda.nccl.version()))
        if torch.cuda.is_available() and torch.distributed.is_nccl_available()
        else None
    ),
    "transformer_engine": version("transformer-engine"),
    "devices": [
        {
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "capability": list(torch.cuda.get_device_capability(index)),
        }
        for index in range(torch.cuda.device_count())
    ],
}
print(json.dumps(manifest, indent=2, sort_keys=True))

errors = []
if not str(manifest["torch"]).startswith("2.10."):
    errors.append(f"expected Torch 2.10.x, got {manifest['torch']}")
# This probe validates the immutable *bootstrap* image before the project venv
# overlays the requested torch==2.10.0+cu129 runtime.  Keep the two manifests
# distinct so the image's cu131 prerelease can never be mistaken for the final
# training runtime checked by ``mor_mlite.env_check``.
if not str(manifest["cuda"]).startswith("13.1"):
    errors.append(f"expected 26.01 bootstrap CUDA 13.1, got {manifest['cuda']}")
if manifest["cuda_device_count"] < 8:
    errors.append(f"expected 8 visible H100 GPUs, got {manifest['cuda_device_count']}")
for device in manifest["devices"][:8]:
    if "H100" not in device["name"] or device["capability"] != [9, 0]:
        errors.append(f"unexpected device: {device}")
if not manifest["nccl"]:
    errors.append("NCCL is unavailable")
if not manifest["transformer_engine"]:
    errors.append("Transformer Engine is unavailable")
if errors:
    raise SystemExit("base image validation failed:\n  - " + "\n  - ".join(errors))
PY
