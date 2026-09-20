from __future__ import annotations

from copy import deepcopy

from mor_mlite.versions import (
    MAGI_ENVIRONMENT,
    MEGATRON_SHA,
    NVCC_VERSION,
    PINNED_DISTRIBUTIONS,
    TORCH_CUDA_VERSION,
    TORCH_VERSION,
    validate_production_manifest,
)


def _valid_manifest() -> dict:
    return {
        "torch": TORCH_VERSION,
        "cuda": TORCH_CUDA_VERSION,
        "cuda_available": True,
        "gpu": "NVIDIA H100 80GB HBM3",
        "torch_cxx11_abi": True,
        "nvcc": {"version": NVCC_VERSION},
        "distributions": {
            name: {"version": version, "path": f"/venv/{name}"}
            for name, version in PINNED_DISTRIBUTIONS.items()
        },
        "megatron_git_sha": MEGATRON_SHA,
        "magi_environment": dict(MAGI_ENVIRONMENT),
    }


def test_production_manifest_requires_exact_versions() -> None:
    manifest = _valid_manifest()
    assert validate_production_manifest(manifest) == []
    manifest["torch"] = "2.10.1+cu129"
    manifest["distributions"]["triton"]["version"] = "3.6.1"
    errors = validate_production_manifest(manifest)
    assert any("expected torch 2.10.0+cu129" in error for error in errors)
    assert any("expected triton 3.6.0" in error for error in errors)


def test_no_magi_skips_only_the_magi_distribution() -> None:
    manifest = deepcopy(_valid_manifest())
    del manifest["distributions"]["magi-attention"]
    assert validate_production_manifest(manifest, require_magi=False) == []


def test_production_manifest_rejects_inherited_magi_switches() -> None:
    manifest = deepcopy(_valid_manifest())
    manifest["magi_environment"]["MAGI_ATTENTION_QO_COMM"] = "1"
    errors = validate_production_manifest(manifest)
    assert any("MAGI_ATTENTION_QO_COMM" in error for error in errors)


def test_eos_source_guard_accepts_worktrees_and_rejects_wrong_imports() -> None:
    from mor_mlite.env_check import validate_manifest

    manifest = _valid_manifest()
    manifest["configured_project_root"] = "/scratch/isolated/worktree/mor_mlite"
    manifest["loaded_project_root"] = manifest["configured_project_root"]
    errors = validate_manifest(manifest, expected_world_size=1)
    assert not any("project" in error.lower() for error in errors)
    manifest["loaded_project_root"] = "/scratch/stale/mor_mlite"
    errors = validate_manifest(manifest, expected_world_size=1)
    assert any("expected project at" in error for error in errors)
    del manifest["configured_project_root"]
    errors = validate_manifest(manifest, expected_world_size=1)
    assert any("MOR_PROJECT_ROOT" in error for error in errors)
