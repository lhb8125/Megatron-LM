#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_eos_project
require_command git
require_command python
require_command flock

# All jobs share one persistent dependency tree. Serialize first-time clone,
# venv creation, and the Magi sm90 build so two queued jobs cannot corrupt the
# same checkout or wheel installation.
exec 9>"${EOS_WORKDIR}/.setup.lock"
flock 9

mkdir -p "${DEPS_ROOT}" "${EOS_WORKDIR}/.cache/huggingface" "${EOS_WORKDIR}/artifacts/eos"

ensure_repository() {
    local url="$1"
    local target="$2"
    if [[ -e "${target}" && ! -d "${target}/.git" ]]; then
        die "${target} exists but is not a git checkout"
    fi
    if [[ ! -d "${target}/.git" ]]; then
        # Materialize the default branch before the clean-tree checks in
        # checkout_commit. A fresh --no-checkout clone otherwise looks like a
        # worktree full of deletions and cannot bootstrap.
        git clone --filter=blob:none "${url}" "${target}"
    fi
}

checkout_commit() {
    local target="$1"
    local commit="$2"
    git -C "${target}" diff --quiet || die "tracked changes in ${target}"
    git -C "${target}" diff --cached --quiet || die "staged changes in ${target}"
    if ! git -C "${target}" cat-file -e "${commit}^{commit}" 2>/dev/null; then
        git -C "${target}" fetch --depth=1 origin "${commit}"
    fi
    if [[ "$(git -C "${target}" rev-parse HEAD 2>/dev/null || true)" != "${commit}" ]]; then
        git -C "${target}" checkout --detach "${commit}"
    fi
    [[ "$(git -C "${target}" rev-parse HEAD)" == "${commit}" ]] || \
        die "failed to pin ${target} to ${commit}"
}

checkout_tag() {
    local target="$1"
    local tag="$2"
    if ! git -C "${target}" show-ref --verify --quiet "refs/tags/${tag}"; then
        git -C "${target}" fetch --depth=1 origin "refs/tags/${tag}:refs/tags/${tag}"
    fi
    local commit
    commit="$(git -C "${target}" rev-list -n 1 "refs/tags/${tag}")"
    checkout_commit "${target}" "${commit}"
    [[ "$(git -C "${target}" describe --tags --exact-match HEAD)" == "${tag}" ]] || \
        die "failed to pin ${target} to ${tag}"
}

ensure_repository "https://github.com/NVIDIA/Megatron-LM.git" "${MEGATRON_LM_ROOT}"
checkout_commit "${MEGATRON_LM_ROOT}" "${MEGATRON_SHA}"

ensure_repository "https://github.com/SandAI-org/MagiAttention.git" \
    "${MAGI_ATTENTION_SOURCE}"
checkout_tag "${MAGI_ATTENTION_SOURCE}" "${MAGI_TAG}"
git -C "${MAGI_ATTENTION_SOURCE}" submodule sync --recursive
git -C "${MAGI_ATTENTION_SOURCE}" submodule update --init --recursive \
    --jobs "${GIT_SUBMODULE_JOBS:-8}"
if git -C "${MAGI_ATTENTION_SOURCE}" submodule status --recursive | \
    grep -Eq '^[+-]'; then
    die "MagiAttention submodules are missing or not at their pinned revisions"
fi

if [[ ! -x "${MOR_VENV}/bin/python" ]]; then
    python -m venv --system-site-packages "${MOR_VENV}"
fi

activate_mor_environment

[[ -f "${MOR_PIP_CONSTRAINT}" ]] || \
    die "missing pip constraint file: ${MOR_PIP_CONSTRAINT}"
# MLite's official Magi builder invokes pip internally.  Environment-level
# constraints are the only way to keep those nested resolver calls on the
# requested Torch/cu129 stack without modifying either upstream checkout.
export PIP_CONSTRAINT="${MOR_PIP_CONSTRAINT}"
export PIP_EXTRA_INDEX_URL="${TORCH_WHEEL_INDEX}"

# The fixed 26.01 container is the compiler/driver bootstrap, but its bundled
# PyTorch reports CUDA 13.1.  Install the requested final cu129 runtime into a
# version-keyed venv before importing MLite.  The exact checks make this block
# idempotent and prevent pip from silently accepting the container prerelease.
python -m pip install --disable-pip-version-check --upgrade \
    pip setuptools wheel "packaging==25.0" ninja
if ! python - "${TORCH_WHEEL_VERSION}" <<'PY'
import sys

import torch

expected = sys.argv[1]
raise SystemExit(0 if torch.__version__ == expected and torch.version.cuda == "12.9" else 1)
PY
then
    # Keep PyPI as the dependency index; the PyTorch index only carries the
    # local-version cu129 wheel and cannot satisfy general dependencies such
    # as setuptools.
    python -m pip install --disable-pip-version-check --upgrade --force-reinstall \
        --index-url "https://pypi.org/simple" \
        --extra-index-url "${TORCH_WHEEL_INDEX}" \
        "torch==${TORCH_WHEEL_VERSION}"
fi

# Transformer Engine's PyTorch extension is ABI-specific.  Do not reuse the
# system extension compiled for the container's prerelease Torch/cu131 pair.
# The version in common.sh deliberately stops at TE 2.13: it has MLite's fused
# permute-and-pad API and its published PyTorch sdist still selects a cu12 core.
if ! python - "${TRANSFORMER_ENGINE_VERSION}" "${MOR_VENV}" <<'PY'
import importlib.metadata
from pathlib import Path
import sys

import torch
import transformer_engine
import transformer_engine.pytorch  # noqa: F401
from transformer_engine.pytorch.permutation import moe_permute_and_pad_with_probs

expected = sys.argv[1]
venv = Path(sys.argv[2]).resolve()
version = importlib.metadata.version("transformer-engine")
module_path = Path(transformer_engine.__file__).resolve()
wheel_path = Path(
    importlib.metadata.distribution("transformer-engine-torch").locate_file("")
).resolve()
core_path = Path(
    importlib.metadata.distribution("transformer-engine-cu12").locate_file("")
).resolve()
raise SystemExit(
    0
    if version == expected
    and torch.__version__ == "2.10.0+cu129"
    and torch.version.cuda == "12.9"
    and module_path.is_relative_to(venv)
    and wheel_path.is_relative_to(venv)
    and core_path.is_relative_to(venv)
    else 1
)
PY
then
    env -u NVIDIA_PRODUCT_NAME -u NVIDIA_PYTORCH_VERSION \
        python -m pip install --disable-pip-version-check --upgrade --force-reinstall \
        --no-build-isolation \
        "transformer-engine[pytorch]==${TRANSFORMER_ENGINE_VERSION}"
fi

python - "${TORCH_WHEEL_VERSION}" "${TRANSFORMER_ENGINE_VERSION}" "${MOR_VENV}" <<'PY'
import importlib.metadata
from pathlib import Path
import sys

import torch
import transformer_engine
import transformer_engine.pytorch  # noqa: F401
from transformer_engine.pytorch.permutation import moe_permute_and_pad_with_probs

if torch.__version__ != sys.argv[1] or torch.version.cuda != "12.9":
    raise SystemExit(
        f"pinned Torch runtime was replaced: {torch.__version__=} {torch.version.cuda=}"
    )
if importlib.metadata.version("transformer-engine") != sys.argv[2]:
    raise SystemExit("unexpected Transformer Engine version")
if not callable(moe_permute_and_pad_with_probs):
    raise SystemExit("Transformer Engine is missing MLite's fused MoE permutation API")
venv = Path(sys.argv[3]).resolve()
module_path = Path(transformer_engine.__file__).resolve()
torch_distribution = importlib.metadata.distribution("transformer-engine-torch")
wheel_path = Path(torch_distribution.locate_file("")).resolve()
core_path = Path(
    importlib.metadata.distribution("transformer-engine-cu12").locate_file("")
).resolve()
torch_requirements = [
    requirement.lower().replace("_", "-")
    for requirement in (torch_distribution.requires or ())
]
if not any(
    requirement.startswith(f"transformer-engine-cu12=={sys.argv[2]}")
    for requirement in torch_requirements
):
    raise SystemExit(
        "Transformer Engine PyTorch extension does not declare the pinned cu12 core"
    )
if any(requirement.startswith("transformer-engine-cu13") for requirement in torch_requirements):
    raise SystemExit("Transformer Engine PyTorch extension unexpectedly selects a cu13 core")
if (
    not module_path.is_relative_to(venv)
    or not wheel_path.is_relative_to(venv)
    or not core_path.is_relative_to(venv)
):
    raise SystemExit(
        "Transformer Engine was inherited from the base image instead of "
        f"the pinned venv: module={module_path}, wheel={wheel_path}, core={core_path}"
    )
PY

# MCore distributed checkpointing imports the NVRx strategy module eagerly and
# the pinned SHA requires 0.6.0.  Install the small set of runtime packages used
# by MLite rather than the repository's broad development extra (which also
# pulls unrelated model families and another CUDA stack).
python -m pip install --disable-pip-version-check \
    "nvidia-resiliency-ext==${NVRX_VERSION}" \
    "einops~=0.8" \
    "tensorstore~=0.1,!=0.1.46,!=0.1.72" \
    "nvtx~=0.2" \
    "pyyaml>=6"

# system-site-packages is retained for image utilities, but these ABI-sensitive
# runtime distributions must physically live in the cu129 venv.  Reinstall
# only when an exact version/path probe says an inherited copy won resolution.
if ! python - "${MOR_VENV}" <<'PY'
import importlib.metadata
from pathlib import Path
import sys

venv = Path(sys.argv[1]).resolve()
pins = {
    "triton": "3.6.0",
    "cuda-python": "12.9.4",
    "cuda-bindings": "12.9.4",
    "nvidia-resiliency-ext": "0.6.0",
}
for name, expected in pins.items():
    dist = importlib.metadata.distribution(name)
    if dist.version != expected or not Path(dist.locate_file("")).resolve().is_relative_to(venv):
        raise SystemExit(1)
PY
then
    python -m pip install --disable-pip-version-check --force-reinstall --no-deps \
        "triton==3.6.0" \
        "cuda-python==12.9.4" \
        "cuda-bindings==12.9.4" \
        "nvidia-resiliency-ext==${NVRX_VERSION}"
fi

python -m mor_mlite.runtime_canary te

# Reject a wrong base image or GPU allocation before paying the Magi build
# cost.  The complete check (including Magi) runs again after installation.
python -m mor_mlite.env_check \
    --megatron-root "${MEGATRON_LM_ROOT}" \
    --expected-world-size "${MOR_EXPECTED_WORLD_SIZE:-1}" \
    --no-magi

export MAGI_ATTENTION_SOURCE
export MAGI_ATTENTION_VENV="${MOR_VENV}"
export MAGI_ATTENTION_BUILD_COMPUTE_CAPABILITY=90
export MAGI_ATTENTION_ALLOW_BUILD_WITH_CUDA12=1
# MLite's backend uses Magi's attention kernels plus torch.distributed
# collectives.  The optional NVSHMEM/group-collective extension is outside the
# v1 standard-All-to-All scope and is not required by the official import test.
export MAGI_ATTENTION_SKIP_MAGI_ATTN_COMM_BUILD=1
export MAX_JOBS="${MAX_JOBS:-32}"

# Use the pinned MLite-provided builder.  Its completion marker makes repeated
# setup calls cheap, while its import check prevents a stale marker from hiding
# an ABI or CUDA-extension failure.
bash "${MEGATRON_LM_ROOT}/experimental/lite/tests/setup_magi_attention_env.sh"

activate_mor_environment
# Fail immediately if any nested Magi dependency replaced Torch or its CUDA
# Python bindings.  The complete environment manifest runs after installation,
# but this gives a precise resolver error before project packages are touched.
python - "${TORCH_WHEEL_VERSION}" <<'PY'
import importlib.metadata
import sys

import torch

expected = sys.argv[1]
cuda_python = importlib.metadata.version("cuda-python")
cuda_bindings = importlib.metadata.version("cuda-bindings")
if (
    torch.__version__ != expected
    or torch.version.cuda != "12.9"
    or cuda_python != "12.9.4"
    or cuda_bindings != "12.9.4"
):
    raise SystemExit(
        "Magi dependency resolution escaped the cu129 lock: "
        f"torch={torch.__version__}, torch_cuda={torch.version.cuda}, "
        f"cuda-python={cuda_python}, cuda-bindings={cuda_bindings}"
    )
PY
python -m pip install --disable-pip-version-check \
    "numpy>=2.0" "safetensors>=0.5" "huggingface-hub>=0.27" \
    "pytest>=8.3" "pytest-xdist>=3.6" "ruff>=0.9"
python -m pip install --disable-pip-version-check --no-deps -e "${EOS_WORKDIR}[dev]"

echo "EOS environment is installed at ${MOR_VENV}."
