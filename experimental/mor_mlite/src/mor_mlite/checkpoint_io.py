"""MoR-aware wrappers around MLite distributed checkpoint save/load.

MLite owns tensor and optimizer serialization.  This module owns the semantic
sidecar that is needed to rebuild a folded recurrent model without consulting
the original Hugging Face weight shards.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from mor_mlite.config import (
    DepthRouterConfig,
    MoRArchitectureConfig,
    MoRParallelConfig,
)
from mor_mlite.config_loader import load_json
from mor_mlite.qwen3_moe_mor.metadata import (
    HF_FOLDING_METADATA_FILENAME,
    MoRCheckpointMetadata,
)


def build_checkpoint_metadata(
    *,
    architecture: MoRArchitectureConfig,
    depth_router: DepthRouterConfig,
    depth_router_seed: int,
    hf_source: str,
    parallel: MoRParallelConfig,
    folding_policy: str = "mean",
    cp_transition: str = "magi_direct",
) -> MoRCheckpointMetadata:
    return MoRCheckpointMetadata(
        architecture=architecture,
        folding_policy=folding_policy,
        depth_router=depth_router,
        depth_router_seed=depth_router_seed,
        hf_source=hf_source,
        cp_transition=cp_transition,
        parallel=parallel,
    )


def _distributed_context() -> tuple[int, bool, Any | None]:
    try:
        import torch.distributed as dist
    except (ImportError, OSError):
        return 0, False, None
    initialized = bool(dist.is_available() and dist.is_initialized())
    return (dist.get_rank() if initialized else 0), initialized, dist


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(dict(payload), stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def run_rank_zero_io(action: Callable[[], None], *, label: str) -> None:
    """Run rank-zero filesystem work and propagate failure to every rank.

    A plain rank-zero write followed by a barrier strands the remaining ranks
    until the process-group timeout when Lustre or JSON serialization fails.
    Broadcasting the outcome gives every rank the same fail-closed result.
    """

    rank, initialized, dist = _distributed_context()
    root_error: Exception | None = None
    failure: str | None = None
    if rank == 0:
        try:
            action()
        except Exception as error:  # noqa: BLE001 - synchronize arbitrary I/O failures
            root_error = error
            failure = f"{type(error).__name__}: {error}"
    if initialized:
        assert dist is not None
        envelope: list[str | None] = [failure]
        dist.broadcast_object_list(envelope, src=0)
        failure = envelope[0]
    if failure is not None:
        message = f"rank-zero {label} failed: {failure}"
        if root_error is not None:
            raise RuntimeError(message) from root_error
        raise RuntimeError(message)


def _read_base_config(source: str | os.PathLike[str]) -> dict[str, Any]:
    path = Path(source)
    if path.is_dir():
        path = path / "config.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"a local HF config.json is required to make the MoR checkpoint self-describing: {path}"
        )
    value = load_json(path, expected_type=dict)
    assert isinstance(value, dict)
    return value


def write_mor_sidecars(
    checkpoint: str | os.PathLike[str],
    *,
    metadata: MoRCheckpointMetadata,
    base_hf_path: str | os.PathLike[str],
) -> None:
    """Atomically write semantic metadata and the base architecture snapshot."""

    root = Path(checkpoint)
    base_config = _read_base_config(base_hf_path)
    raw_num_experts = base_config.get("num_experts")
    metadata.parallel.validate_world_size(
        metadata.parallel.expected_world_size,
        num_experts=None if raw_num_experts is None else int(raw_num_experts),
    )
    _atomic_json(root / HF_FOLDING_METADATA_FILENAME, metadata.to_dict())
    _atomic_json(root / "config.json", base_config)


def read_mor_sidecar(
    checkpoint: str | os.PathLike[str],
) -> MoRCheckpointMetadata:
    path = Path(checkpoint) / HF_FOLDING_METADATA_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"MoR checkpoint sidecar is missing: {path}")
    value = load_json(path, expected_type=dict)
    assert isinstance(value, dict)
    return MoRCheckpointMetadata.from_dict(value)


def validate_checkpoint_metadata(
    actual: MoRCheckpointMetadata,
    expected: MoRCheckpointMetadata,
    *,
    strict_versions: bool = True,
    strict_runtime: bool = True,
) -> None:
    """Fail before tensor loading if checkpoint execution semantics differ."""

    comparisons = {
        "model_type": (actual.model_type, expected.model_type),
        "base_model_type": (actual.base_model_type, expected.base_model_type),
        "architecture": (actual.architecture, expected.architecture),
        "folding_policy": (actual.folding_policy, expected.folding_policy),
        "depth_router": (actual.depth_router, expected.depth_router),
        "depth_router_seed": (actual.depth_router_seed, expected.depth_router_seed),
    }
    if strict_runtime:
        comparisons.update(
            {
                "cp_transition": (actual.cp_transition, expected.cp_transition),
                "parallel": (actual.parallel, expected.parallel),
            }
        )
    else:
        # Qwen GroupedLinear parameters are keyed by EP-local expert index in
        # the pinned MLite DCP bridge. Dense DP/TP/CP axes can be resharded, but
        # changing EP would reinterpret those keys as different global experts.
        comparisons.update(
            {
                "expert_parallel_size": (actual.parallel.ep, expected.parallel.ep),
                "expert_tensor_parallel_size": (
                    actual.parallel.etp,
                    expected.parallel.etp,
                ),
            }
        )
    if strict_versions:
        comparisons.update(
            {
                "megatron_lm_sha": (
                    actual.megatron_lm_sha,
                    expected.megatron_lm_sha,
                ),
                "magi_attention_version": (
                    actual.magi_attention_version,
                    expected.magi_attention_version,
                ),
            }
        )
    mismatches = [
        f"{name}: checkpoint={left!r}, runtime={right!r}"
        for name, (left, right) in comparisons.items()
        if left != right
    ]
    if mismatches:
        raise ValueError("incompatible MoR checkpoint metadata; " + "; ".join(mismatches))


def save_mor_checkpoint(
    runtime: Any,
    handle: Any,
    checkpoint: str | os.PathLike[str],
    *,
    step: int,
    metadata: MoRCheckpointMetadata,
    base_hf_path: str | os.PathLike[str],
    save_rng: bool = True,
    save_model: bool = True,
    save_optimizer: bool = True,
) -> None:
    """Collectively save MLite state, then publish rank-zero sidecars."""

    runtime.save_checkpoint(
        handle,
        str(checkpoint),
        step=step,
        use_dcp=True,
        save_rng=save_rng,
        save_model=save_model,
        save_optimizer=save_optimizer,
    )
    run_rank_zero_io(
        lambda: write_mor_sidecars(
            checkpoint,
            metadata=metadata,
            base_hf_path=base_hf_path,
        ),
        label="MoR checkpoint sidecar write",
    )


def load_mor_checkpoint(
    runtime: Any,
    handle: Any,
    checkpoint: str | os.PathLike[str],
    *,
    expected_metadata: MoRCheckpointMetadata,
    strict_versions: bool = True,
    strict_runtime: bool = True,
    load_rng: bool = True,
    load_model: bool = True,
    load_optimizer: bool = True,
) -> int:
    """Validate the semantic sidecar before collectively restoring MLite state."""

    actual = read_mor_sidecar(checkpoint)
    validate_checkpoint_metadata(
        actual,
        expected_metadata,
        strict_versions=strict_versions,
        strict_runtime=strict_runtime,
    )
    return int(
        runtime.load_checkpoint(
            handle,
            str(checkpoint),
            use_dcp=True,
            load_rng=load_rng,
            load_model=load_model,
            load_optimizer=load_optimizer,
        )
    )


__all__ = [
    "build_checkpoint_metadata",
    "load_mor_checkpoint",
    "read_mor_sidecar",
    "run_rank_zero_io",
    "save_mor_checkpoint",
    "validate_checkpoint_metadata",
    "write_mor_sidecars",
]
