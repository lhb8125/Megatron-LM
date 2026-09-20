"""Checkpoint-continuation evidence shared by the MLite runner and comparator.

The checkpoint acceptance contract is stronger than "load succeeded": the
saved training state must reproduce an uninterrupted next step in a freshly
constructed runtime session.  This module is intentionally independent of
MLite/CUDA so its comparison and fingerprinting rules are unit-testable on a
CPU-only host.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from .metrics import compare_tensor

CHECKPOINT_CONTINUITY_SCHEMA_VERSION = 1
_PHASES = ("uninterrupted", "resume")
_ROUTE_FIELDS = (
    "round",
    "mode",
    "sample_ids",
    "original_positions",
    "global_token_ids",
    "active_cu_seqlens",
    "padding_mask",
)


def state_fingerprint(value: Any) -> dict[str, Any]:
    """Return a deterministic content fingerprint for a nested training state.

    Tensor payloads are streamed into the digest in their native dtype.  The
    function accepts the value shapes used by PyTorch/Megatron optimizer state
    dictionaries and by the Python/NumPy/Torch RNG snapshots, while failing
    closed on opaque objects whose identity would not survive a fresh session.
    """

    digest = hashlib.sha256()
    statistics = {
        "tensor_count": 0,
        "tensor_numel": 0,
        "tensor_bytes": 0,
        "leaf_count": 0,
    }

    def emit(payload: bytes) -> None:
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)

    def key_digest(key: Any) -> bytes:
        nested = state_fingerprint(key)
        return bytes.fromhex(str(nested["sha256"]))

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            if item.layout != torch.strided:
                raise TypeError(f"cannot fingerprint non-strided tensor layout {item.layout}")
            cpu = item.detach().cpu().contiguous()
            byte_view = cpu.reshape(-1).view(torch.uint8)
            emit(b"tensor")
            emit(str(cpu.dtype).encode("utf-8"))
            emit(repr(tuple(cpu.shape)).encode("ascii"))
            emit(byte_view.numpy().tobytes())
            statistics["tensor_count"] += 1
            statistics["tensor_numel"] += int(cpu.numel())
            statistics["tensor_bytes"] += int(cpu.numel() * cpu.element_size())
            statistics["leaf_count"] += 1
            return
        if isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            emit(b"ndarray")
            emit(array.dtype.str.encode("ascii"))
            emit(repr(tuple(array.shape)).encode("ascii"))
            emit(array.tobytes())
            statistics["tensor_count"] += 1
            statistics["tensor_numel"] += int(array.size)
            statistics["tensor_bytes"] += int(array.nbytes)
            statistics["leaf_count"] += 1
            return
        if isinstance(item, np.generic):
            visit(item.item())
            return
        if isinstance(item, Mapping):
            emit(b"mapping")
            entries = sorted(
                ((key_digest(key), key, child) for key, child in item.items()),
                key=lambda entry: entry[0],
            )
            for serialized_key, key, child in entries:
                emit(serialized_key)
                visit(key)
                visit(child)
            emit(b"mapping-end")
            return
        if isinstance(item, tuple):
            emit(b"tuple")
            for child in item:
                visit(child)
            emit(b"tuple-end")
            return
        if isinstance(item, list):
            emit(b"list")
            for child in item:
                visit(child)
            emit(b"list-end")
            return
        if item is None:
            emit(b"none")
        elif isinstance(item, bool):
            emit(b"bool:true" if item else b"bool:false")
        elif isinstance(item, int):
            emit(b"int")
            emit(str(item).encode("ascii"))
        elif isinstance(item, float):
            emit(b"float")
            emit(item.hex().encode("ascii"))
        elif isinstance(item, str):
            emit(b"str")
            emit(item.encode("utf-8"))
        elif isinstance(item, bytes):
            emit(b"bytes")
            emit(item)
        elif isinstance(item, (torch.dtype, torch.device)):
            emit(type(item).__name__.encode("ascii"))
            emit(str(item).encode("ascii"))
        else:
            raise TypeError(
                "cannot fingerprint opaque training-state value "
                f"of type {type(item).__module__}.{type(item).__qualname__}"
            )
        statistics["leaf_count"] += 1

    visit(value)
    return {"sha256": digest.hexdigest(), **statistics}


def capture_rng_state() -> dict[str, Any]:
    """Mirror the RNG families persisted by pinned MLite distributed checkpoints."""

    cuda_rng_state = None
    tracker_states: dict[str, torch.Tensor] = {}
    if torch.cuda.is_initialized():
        cuda_rng_state = torch.cuda.get_rng_state().detach().cpu().clone()
        from megatron.core import tensor_parallel

        tracker = tensor_parallel.get_cuda_rng_tracker()
        tracker_states = {
            str(name): state.detach().cpu().clone()
            for name, state in tracker.get_states().items()
            if state is not None
        }
    return {
        "random_rng_state": random.getstate(),
        "np_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state().detach().cpu().clone(),
        "cuda_rng_state": cuda_rng_state,
        "rng_tracker_states": tracker_states,
    }


def _phase_tensors(
    tensors: Mapping[str, torch.Tensor], *, phase: str, step: int
) -> dict[str, tuple[str, torch.Tensor]]:
    step_name = f"step_{step:03d}"
    result: dict[str, tuple[str, torch.Tensor]] = {}
    for name, tensor in tensors.items():
        parts = name.split("/")
        if len(parts) < 3 or parts[1] != phase or parts[2] != step_name:
            continue
        canonical = "/".join((parts[0], "checkpoint_next", *parts[2:]))
        result[canonical] = (name, tensor)
    return result


def _tensor_kind(name: str) -> str:
    namespace = name.split("/", 1)[0]
    if namespace == "loss":
        return "loss"
    if namespace == "gradient":
        return "gradient"
    if namespace == "update":
        return "update"
    if namespace == "post_step":
        return "post_step"
    return "forward"


def _phase_routes(
    routes: Sequence[Mapping[str, Any]], *, phase: str, step: int
) -> dict[tuple[int, int], dict[str, Any]]:
    result: dict[tuple[int, int], dict[str, Any]] = {}
    for route in routes:
        if str(route.get("phase", "train")) != phase or int(route.get("step", -1)) != step:
            continue
        key = (int(route.get("microbatch", -1)), int(route.get("round", -1)))
        if key in result:
            raise ValueError(f"duplicate {phase} RoutePlan for key {key}")
        result[key] = {field: route.get(field) for field in _ROUTE_FIELDS}
    return result


def _phase_optimizer_step(
    optimizer_steps: Sequence[Mapping[str, Any]], *, phase: str, step: int
) -> Mapping[str, Any] | None:
    matches = [
        item
        for item in optimizer_steps
        if str(item.get("phase", "train")) == phase and int(item.get("step", -1)) == step
    ]
    if len(matches) > 1:
        raise ValueError(f"duplicate {phase} optimizer-step evidence for step {step}")
    return matches[0] if matches else None


def _fingerprint_result(
    left: Mapping[str, Any] | None, right: Mapping[str, Any] | None
) -> dict[str, Any]:
    left_sha = None if left is None else left.get("sha256")
    right_sha = None if right is None else right.get("sha256")
    return {
        "passed": bool(left_sha and right_sha and left_sha == right_sha),
        "saved_sha256": left_sha,
        "restored_sha256": right_sha,
    }


def build_checkpoint_continuity_report(
    *,
    step: int,
    precision: str,
    tensors: Mapping[str, torch.Tensor],
    routes: Sequence[Mapping[str, Any]],
    optimizer_steps: Sequence[Mapping[str, Any]],
    saved_fingerprints: Mapping[str, Mapping[str, Any]],
    restored_fingerprints: Mapping[str, Mapping[str, Any]],
    uninterrupted_fingerprints: Mapping[str, Mapping[str, Any]],
    resumed_fingerprints: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare fresh-session resume against the uninterrupted next step."""

    if precision not in {"fp32", "bf16"}:
        raise ValueError(f"unsupported checkpoint-continuity precision {precision!r}")
    details: list[str] = []
    restored_state = {
        name: _fingerprint_result(saved_fingerprints.get(name), restored_fingerprints.get(name))
        for name in ("parameters", "optimizer", "rng")
    }
    for name, result in restored_state.items():
        if not result["passed"]:
            details.append(f"restored {name} fingerprint differs from checkpoint save point")

    uninterrupted = _phase_tensors(tensors, phase=_PHASES[0], step=step)
    resumed = _phase_tensors(tensors, phase=_PHASES[1], step=step)
    missing_resume = sorted(uninterrupted.keys() - resumed.keys())
    missing_uninterrupted = sorted(resumed.keys() - uninterrupted.keys())
    if missing_resume:
        details.append(f"resume is missing {len(missing_resume)} next-step tensor(s)")
    if missing_uninterrupted:
        details.append(
            f"uninterrupted branch is missing {len(missing_uninterrupted)} next-step tensor(s)"
        )

    tensor_results = []
    coverage = {
        "forward": 0,
        "loss": 0,
        "gradient": 0,
        "update": 0,
        "post_step": 0,
    }
    for canonical in sorted(uninterrupted.keys() & resumed.keys()):
        left_name, left = uninterrupted[canonical]
        _, right = resumed[canonical]
        kind = _tensor_kind(canonical)
        coverage[kind] += 1
        result = compare_tensor(
            left_name,
            left,
            right,
            precision=precision,
            kind=kind,
        ).to_dict()
        result["canonical_name"] = canonical
        tensor_results.append(result)
        if not result["passed"]:
            details.append(f"next-step {kind} differs: {canonical}")
    for required_kind, count in coverage.items():
        if count == 0:
            details.append(f"next-step comparison recorded no {required_kind} tensors")

    uninterrupted_routes = _phase_routes(routes, phase=_PHASES[0], step=step)
    resumed_routes = _phase_routes(routes, phase=_PHASES[1], step=step)
    routes_exact = bool(uninterrupted_routes) and uninterrupted_routes == resumed_routes
    if not routes_exact:
        details.append("next-step canonical RoutePlans differ or are missing")

    uninterrupted_step = _phase_optimizer_step(optimizer_steps, phase=_PHASES[0], step=step)
    resumed_step = _phase_optimizer_step(optimizer_steps, phase=_PHASES[1], step=step)
    optimizer_step_result: dict[str, Any] = {
        "passed": False,
        "updated": None,
        "num_zeros_exact": False,
        "grad_norm": None,
    }
    if uninterrupted_step is None or resumed_step is None:
        details.append("next-step optimizer result is missing")
    else:
        updated = bool(uninterrupted_step.get("updated")) and bool(resumed_step.get("updated"))
        updated_exact = uninterrupted_step.get("updated") == resumed_step.get("updated")
        zeros_exact = uninterrupted_step.get("num_zeros") == resumed_step.get("num_zeros")
        grad_norm = compare_tensor(
            "optimizer_step/global_norm",
            torch.tensor(float(uninterrupted_step["grad_norm"]), dtype=torch.float32),
            torch.tensor(float(resumed_step["grad_norm"]), dtype=torch.float32),
            precision=precision,
            kind="gradient",
        ).to_dict()
        optimizer_step_result = {
            "passed": bool(updated and updated_exact and zeros_exact and grad_norm["passed"]),
            "updated": updated,
            "updated_exact": updated_exact,
            "num_zeros_exact": zeros_exact,
            "grad_norm": grad_norm,
        }
        if not optimizer_step_result["passed"]:
            details.append("next-step optimizer result differs or did not update")

    state_after_step = {
        name: _fingerprint_result(
            uninterrupted_fingerprints.get(name), resumed_fingerprints.get(name)
        )
        for name in ("parameters", "optimizer", "rng")
    }
    for name, result in state_after_step.items():
        if not result["passed"]:
            details.append(f"post-next-step {name} fingerprint differs")

    next_step_passed = (
        not missing_resume
        and not missing_uninterrupted
        and all(result["passed"] for result in tensor_results)
        and all(count > 0 for count in coverage.values())
        and routes_exact
        and optimizer_step_result["passed"]
        and all(result["passed"] for result in state_after_step.values())
    )
    passed = all(result["passed"] for result in restored_state.values()) and next_step_passed
    return {
        "schema_version": CHECKPOINT_CONTINUITY_SCHEMA_VERSION,
        "required": True,
        "status": "passed" if passed else "failed",
        "step": int(step),
        "precision": precision,
        "restored_state": restored_state,
        "next_step": {
            "passed": next_step_passed,
            "tensor_coverage": coverage,
            "tensor_results": tensor_results,
            "missing_resume_tensors": missing_resume,
            "missing_uninterrupted_tensors": missing_uninterrupted,
            "route_count": len(uninterrupted_routes),
            "routes_exact": routes_exact,
            "optimizer_step": optimizer_step_result,
            "state_after_step": state_after_step,
        },
        "details": details,
    }


def check_checkpoint_continuity(raw: Any, *, required: bool) -> dict[str, Any]:
    """Validate the compact manifest contract without recomputing tensors."""

    if not required and raw is None:
        return {"passed": True, "required": False, "details": ["not required"]}
    if not isinstance(raw, Mapping):
        return {
            "passed": False,
            "required": required,
            "details": ["checkpoint-continuity report is missing"],
        }
    details: list[str] = []
    if raw.get("schema_version") != CHECKPOINT_CONTINUITY_SCHEMA_VERSION:
        details.append("unsupported checkpoint-continuity schema")
    if raw.get("status") != "passed":
        details.append(f"checkpoint-continuity status is {raw.get('status')!r}")
    restored = raw.get("restored_state")
    next_step = raw.get("next_step")
    for name in ("parameters", "optimizer", "rng"):
        if not isinstance(restored, Mapping) or not bool(
            isinstance(restored.get(name), Mapping) and restored[name].get("passed")
        ):
            details.append(f"restored {name} fingerprint was not verified")
    required_coverage = ("forward", "loss", "gradient", "update", "post_step")
    coverage = next_step.get("tensor_coverage") if isinstance(next_step, Mapping) else None
    for name in required_coverage:
        if not isinstance(coverage, Mapping) or int(coverage.get(name, 0)) <= 0:
            details.append(f"next-step {name} coverage is missing")
    if not isinstance(next_step, Mapping) or not bool(next_step.get("passed")):
        details.append("uninterrupted/resume next-step comparison did not pass")
    if not isinstance(next_step, Mapping) or not bool(next_step.get("routes_exact")):
        details.append("uninterrupted/resume RoutePlans were not exact")
    optimizer_step = next_step.get("optimizer_step") if isinstance(next_step, Mapping) else None
    if not isinstance(optimizer_step, Mapping) or not bool(optimizer_step.get("passed")):
        details.append("uninterrupted/resume optimizer-step comparison did not pass")
    post_state = next_step.get("state_after_step") if isinstance(next_step, Mapping) else None
    for name in ("parameters", "optimizer", "rng"):
        if not isinstance(post_state, Mapping) or not bool(
            isinstance(post_state.get(name), Mapping) and post_state[name].get("passed")
        ):
            details.append(f"post-next-step {name} fingerprint was not verified")
    return {"passed": not details, "required": required, "details": details}


__all__ = [
    "CHECKPOINT_CONTINUITY_SCHEMA_VERSION",
    "build_checkpoint_continuity_report",
    "capture_rng_state",
    "check_checkpoint_continuity",
    "state_fingerprint",
]
