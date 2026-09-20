"""Process-isolated checkpoint save/resume receipts for large-model parity."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mor_mlite.config_loader import load_json
from mor_mlite.parity.artifacts import load_artifact

RECEIPT_FILENAME = "mor_parity_checkpoint.json"
_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "saved_step",
        "topology",
        "architecture",
        "depth_router",
        "optimizer",
        "run_contract",
        "parameter_fingerprint",
        "optimizer_fingerprint",
        "rng_fingerprint",
        "rng_sidecars",
        "uninterrupted_next_step",
        "optimizer_saved",
        "rng_saved",
    }
)
_RUN_CONTRACT_FIELDS = frozenset(
    {
        "preset",
        "precision",
        "strict",
        "attention_policy",
        "seed",
        "seq_lens",
        "num_microbatches",
        "route_mode",
        "replay_routes_sha256",
        "cp_transition",
    }
)
_RNG_SIDECAR_FIELDS = frozenset({"rank", "path", "num_bytes", "sha256"})
_NEXT_STEP_FIELDS = frozenset(
    {
        "step",
        "parameter_fingerprint",
        "optimizer_fingerprint",
        "rng_fingerprint",
        "optimizer_step",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_fingerprint(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("checkpoint parameter fingerprint must be a mapping")
    result = dict(value)
    digest = result.get("sha256")
    rank_count = result.get("rank_count")
    manifests = result.get("rank_manifests")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("checkpoint parameter fingerprint has an invalid sha256")
    if isinstance(rank_count, bool) or not isinstance(rank_count, int) or rank_count < 1:
        raise ValueError("checkpoint parameter fingerprint has an invalid rank_count")
    if not isinstance(manifests, list) or len(manifests) != rank_count:
        raise ValueError("checkpoint parameter fingerprint rank manifests are incomplete")
    return result


def _validated_state_fingerprint(
    value: Any,
    *,
    label: str,
    require_optimizer_payload: bool = False,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"checkpoint {label} fingerprint must be a mapping")
    result = dict(value)
    digest = result.get("sha256")
    rank_count = result.get("rank_count")
    rank_fingerprints = result.get("rank_fingerprints")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"checkpoint {label} fingerprint has an invalid sha256")
    if isinstance(rank_count, bool) or not isinstance(rank_count, int) or rank_count < 1:
        raise ValueError(f"checkpoint {label} fingerprint has an invalid rank_count")
    if not isinstance(rank_fingerprints, list) or len(rank_fingerprints) != rank_count:
        raise ValueError(f"checkpoint {label} fingerprint rank records are incomplete")
    for vector_name, peer_name in (
        ("tensor_counts", "tensor_count"),
        ("tensor_bytes", "tensor_bytes"),
    ):
        values = result.get(vector_name)
        if (
            not isinstance(values, list)
            or len(values) != rank_count
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in values
            )
        ):
            raise ValueError(f"checkpoint {label} fingerprint has invalid positive {vector_name}")
        for rank, (item, peer) in enumerate(zip(values, rank_fingerprints, strict=True)):
            if not isinstance(peer, Mapping) or peer.get(peer_name) != item:
                raise ValueError(
                    f"checkpoint {label} fingerprint rank {rank} disagrees on {peer_name}"
                )
    if require_optimizer_payload:
        optimizer_vectors = (
            ("optimizer_leaf_counts", "optimizer_leaf_count"),
            ("master_parameter_counts", "master_parameter_count"),
            ("master_parameter_bytes", "master_parameter_bytes"),
            ("adam_moment_tensor_counts", "adam_moment_tensor_count"),
            ("adam_moment_tensor_bytes", "adam_moment_tensor_bytes"),
            ("optimizer_step_counts", "optimizer_step_count"),
        )
        for vector_name, peer_name in optimizer_vectors:
            values = result.get(vector_name)
            if (
                not isinstance(values, list)
                or len(values) != rank_count
                or any(
                    isinstance(item, bool) or not isinstance(item, int) or item <= 0
                    for item in values
                )
            ):
                raise ValueError(
                    f"checkpoint {label} fingerprint has invalid positive {vector_name}"
                )
            for rank, (item, peer) in enumerate(zip(values, rank_fingerprints, strict=True)):
                if not isinstance(peer, Mapping) or peer.get(peer_name) != item:
                    raise ValueError(
                        f"checkpoint {label} fingerprint rank {rank} disagrees on {peer_name}"
                    )
        for rank in range(rank_count):
            required_tensor_count = (
                result["master_parameter_counts"][rank] + result["adam_moment_tensor_counts"][rank]
            )
            required_tensor_bytes = (
                result["master_parameter_bytes"][rank] + result["adam_moment_tensor_bytes"][rank]
            )
            if result["tensor_counts"][rank] < required_tensor_count:
                raise ValueError(
                    f"checkpoint {label} fingerprint rank {rank} tensor count "
                    "does not cover master parameters and Adam moments"
                )
            if result["tensor_bytes"][rank] < required_tensor_bytes:
                raise ValueError(
                    f"checkpoint {label} fingerprint rank {rank} tensor bytes "
                    "do not cover master parameters and Adam moments"
                )
    return result


def _validated_next_step(value: Any, *, saved_step: int) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("checkpoint uninterrupted_next_step must be a mapping")
    result = dict(value)
    fields = frozenset(result)
    if fields != _NEXT_STEP_FIELDS:
        missing = sorted(_NEXT_STEP_FIELDS - fields)
        unknown = sorted(fields - _NEXT_STEP_FIELDS)
        raise ValueError(
            "checkpoint uninterrupted_next_step fields differ: "
            f"missing={missing}, unknown={unknown}"
        )
    if result["step"] != saved_step:
        raise ValueError("checkpoint uninterrupted next step must start at saved_step")
    result["parameter_fingerprint"] = _validated_fingerprint(result["parameter_fingerprint"])
    result["optimizer_fingerprint"] = _validated_state_fingerprint(
        result["optimizer_fingerprint"],
        label="next-step optimizer",
        require_optimizer_payload=True,
    )
    result["rng_fingerprint"] = _validated_state_fingerprint(
        result["rng_fingerprint"], label="next-step RNG"
    )
    optimizer_step = result["optimizer_step"]
    if not isinstance(optimizer_step, Mapping):
        raise TypeError("checkpoint uninterrupted next-step optimizer record must be a mapping")
    optimizer_step = dict(optimizer_step)
    if (
        optimizer_step.get("phase") != "uninterrupted"
        or optimizer_step.get("step") != saved_step
        or optimizer_step.get("updated") is not True
        or not math.isfinite(float(optimizer_step.get("grad_norm", float("nan"))))
    ):
        raise ValueError("checkpoint uninterrupted next-step optimizer record is invalid")
    result["optimizer_step"] = optimizer_step
    return result


def _validated_run_contract(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("checkpoint run_contract must be a mapping")
    result = dict(value)
    fields = frozenset(result)
    if fields != _RUN_CONTRACT_FIELDS:
        missing = sorted(_RUN_CONTRACT_FIELDS - fields)
        unknown = sorted(fields - _RUN_CONTRACT_FIELDS)
        raise ValueError(
            f"checkpoint run_contract fields differ: missing={missing}, unknown={unknown}"
        )
    if not isinstance(result["preset"], str) or not result["preset"]:
        raise ValueError("checkpoint run_contract preset must be a non-empty string")
    if result["precision"] not in {"bf16", "fp32"}:
        raise ValueError("checkpoint run_contract precision is unsupported")
    if not isinstance(result["strict"], bool):
        raise TypeError("checkpoint run_contract strict flag must be boolean")
    if not isinstance(result["attention_policy"], str) or not result["attention_policy"]:
        raise ValueError("checkpoint run_contract attention_policy must be a non-empty string")
    if isinstance(result["seed"], bool) or not isinstance(result["seed"], int):
        raise TypeError("checkpoint run_contract seed must be an integer")
    seq_lens = result["seq_lens"]
    if (
        not isinstance(seq_lens, list)
        or not seq_lens
        or any(
            isinstance(length, bool) or not isinstance(length, int) or length < 1
            for length in seq_lens
        )
    ):
        raise ValueError("checkpoint run_contract seq_lens must contain positive integers")
    if (
        isinstance(result["num_microbatches"], bool)
        or not isinstance(result["num_microbatches"], int)
        or result["num_microbatches"] < 1
    ):
        raise ValueError("checkpoint run_contract num_microbatches must be positive")
    if result["route_mode"] not in {"learned", "replay"}:
        raise ValueError("checkpoint run_contract route_mode is unsupported")
    replay_digest = result["replay_routes_sha256"]
    if result["route_mode"] == "replay":
        if (
            not isinstance(replay_digest, str)
            or len(replay_digest) != 64
            or any(character not in "0123456789abcdef" for character in replay_digest)
        ):
            raise ValueError("checkpoint replay run_contract requires a routes sha256")
    elif replay_digest is not None:
        raise ValueError("learned run_contract cannot name a replay routes sha256")
    if result["cp_transition"] not in {
        "magi_direct",
        "magi_canonical",
        "static_reference",
    }:
        raise ValueError("checkpoint run_contract CP transition is unsupported")
    return result


def _validated_rng_sidecars(
    value: Any, *, saved_step: int, rank_count: int
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != rank_count:
        raise ValueError("checkpoint RNG sidecar manifest does not cover every rank")
    records: list[dict[str, Any]] = []
    for expected_rank, raw in enumerate(value):
        if not isinstance(raw, Mapping) or frozenset(raw) != _RNG_SIDECAR_FIELDS:
            raise ValueError("checkpoint RNG sidecar record has invalid fields")
        record = dict(raw)
        expected_path = f"step_{saved_step}/rng_state_rank_{expected_rank:05d}.pt"
        digest = record.get("sha256")
        if record.get("rank") != expected_rank or record.get("path") != expected_path:
            raise ValueError("checkpoint RNG sidecar rank/path mapping is invalid")
        if (
            isinstance(record.get("num_bytes"), bool)
            or not isinstance(record.get("num_bytes"), int)
            or record["num_bytes"] < 1
        ):
            raise ValueError("checkpoint RNG sidecar size is invalid")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("checkpoint RNG sidecar sha256 is invalid")
        records.append(record)
    return records


def _distributed_rank_world() -> tuple[int, int, Any | None]:
    try:
        import torch.distributed as dist
    except (ImportError, OSError):
        return 0, 1, None
    if not dist.is_available() or not dist.is_initialized():
        return 0, 1, None
    return dist.get_rank(), dist.get_world_size(), dist


def collect_rng_sidecar_manifest(
    checkpoint: str | Path, *, saved_step: int
) -> list[dict[str, Any]]:
    """Collectively require and hash the RNG file written by every rank."""

    root = Path(checkpoint)
    rank, world_size, dist = _distributed_rank_world()
    relative = Path(f"step_{saved_step}") / f"rng_state_rank_{rank:05d}.pt"
    path = root / relative
    try:
        local: dict[str, Any] = {
            "rank": rank,
            "path": relative.as_posix(),
            "num_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    except Exception as error:  # noqa: BLE001 - report arbitrary file failures collectively
        local = {"rank": rank, "error": f"{type(error).__name__}: {error}"}
    if dist is None:
        peers = [local]
    else:
        peers: list[dict[str, Any] | None] = [None] * world_size
        dist.all_gather_object(peers, local)
    failures = [peer for peer in peers if not isinstance(peer, dict) or "error" in peer]
    if failures:
        raise RuntimeError(f"checkpoint RNG sidecar collection failed: {failures}")
    return _validated_rng_sidecars(peers, saved_step=saved_step, rank_count=world_size)


def verify_rng_sidecar_manifest(
    checkpoint: str | Path,
    manifest: Any,
    *,
    saved_step: int,
    rank_count: int,
) -> None:
    """Collectively verify all receipt-bound RNG files before MLite loading."""

    records = _validated_rng_sidecars(
        manifest,
        saved_step=saved_step,
        rank_count=rank_count,
    )
    rank, world_size, dist = _distributed_rank_world()
    failure: str | None = None
    if world_size != rank_count:
        failure = f"runtime world size {world_size} differs from RNG manifest {rank_count}"
    else:
        try:
            root = Path(checkpoint)
            for record in records:
                path = root / record["path"]
                if path.stat().st_size != record["num_bytes"] or _sha256(path) != record["sha256"]:
                    raise RuntimeError(f"RNG sidecar integrity mismatch: {path}")
        except Exception as error:  # noqa: BLE001 - report arbitrary file failures collectively
            failure = f"rank {rank}: {type(error).__name__}: {error}"
    if dist is None:
        failures = [failure] if failure is not None else []
    else:
        peers: list[str | None] = [None] * world_size
        dist.all_gather_object(peers, failure)
        failures = [item for item in peers if item is not None]
    if failures:
        raise RuntimeError(f"checkpoint RNG sidecar verification failed: {failures}")


def write_checkpoint_save_receipt(
    checkpoint: str | Path,
    *,
    saved_step: int,
    topology: Mapping[str, Any],
    architecture: Mapping[str, Any],
    depth_router: Mapping[str, Any],
    optimizer: Mapping[str, Any],
    run_contract: Mapping[str, Any],
    parameter_fingerprint: Mapping[str, Any],
    optimizer_fingerprint: Mapping[str, Any],
    rng_fingerprint: Mapping[str, Any],
    rng_sidecars: list[Mapping[str, Any]],
    uninterrupted_next_step: Mapping[str, Any],
) -> Path:
    """Atomically bind a DCP save point to its full distributed model hash."""

    root = Path(checkpoint)
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist after save: {root}")
    if isinstance(saved_step, bool) or not isinstance(saved_step, int) or saved_step < 1:
        raise ValueError("saved_step must be a positive integer")
    validated_parameter_fingerprint = _validated_fingerprint(parameter_fingerprint)
    payload = {
        "schema_version": 1,
        "kind": "mor_mlite_external_checkpoint_save",
        "saved_step": saved_step,
        "topology": dict(topology),
        "architecture": dict(architecture),
        "depth_router": dict(depth_router),
        "optimizer": dict(optimizer),
        "run_contract": _validated_run_contract(run_contract),
        "parameter_fingerprint": validated_parameter_fingerprint,
        "optimizer_fingerprint": _validated_state_fingerprint(
            optimizer_fingerprint,
            label="optimizer",
            require_optimizer_payload=True,
        ),
        "rng_fingerprint": _validated_state_fingerprint(rng_fingerprint, label="RNG"),
        "rng_sidecars": _validated_rng_sidecars(
            rng_sidecars,
            saved_step=saved_step,
            rank_count=validated_parameter_fingerprint["rank_count"],
        ),
        "uninterrupted_next_step": _validated_next_step(
            uninterrupted_next_step, saved_step=saved_step
        ),
        "optimizer_saved": True,
        "rng_saved": True,
    }
    destination = root / RECEIPT_FILENAME
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=root,
        prefix=f".{RECEIPT_FILENAME}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    return destination


def read_checkpoint_save_receipt(checkpoint: str | Path) -> dict[str, Any]:
    """Read and strictly validate a process-isolated checkpoint receipt."""

    path = Path(checkpoint) / RECEIPT_FILENAME
    payload = load_json(path, expected_type=dict)
    assert isinstance(payload, dict)
    fields = frozenset(payload)
    if fields != _RECEIPT_FIELDS:
        missing = sorted(_RECEIPT_FIELDS - fields)
        unknown = sorted(fields - _RECEIPT_FIELDS)
        raise ValueError(f"checkpoint receipt fields differ: missing={missing}, unknown={unknown}")
    if payload["schema_version"] != 1:
        raise ValueError("unsupported checkpoint receipt schema")
    if payload["kind"] != "mor_mlite_external_checkpoint_save":
        raise ValueError("checkpoint receipt has an unexpected kind")
    saved_step = payload["saved_step"]
    if isinstance(saved_step, bool) or not isinstance(saved_step, int) or saved_step < 1:
        raise ValueError("checkpoint receipt saved_step must be a positive integer")
    for key in ("topology", "architecture", "depth_router", "optimizer", "run_contract"):
        if not isinstance(payload[key], dict):
            raise TypeError(f"checkpoint receipt {key} must be a mapping")
    if payload["optimizer_saved"] is not True or payload["rng_saved"] is not True:
        raise ValueError("checkpoint receipt must attest optimizer and RNG state")
    payload["parameter_fingerprint"] = _validated_fingerprint(payload["parameter_fingerprint"])
    payload["optimizer_fingerprint"] = _validated_state_fingerprint(
        payload["optimizer_fingerprint"],
        label="optimizer",
        require_optimizer_payload=True,
    )
    payload["rng_fingerprint"] = _validated_state_fingerprint(
        payload["rng_fingerprint"], label="RNG"
    )
    payload["run_contract"] = _validated_run_contract(payload["run_contract"])
    payload["rng_sidecars"] = _validated_rng_sidecars(
        payload["rng_sidecars"],
        saved_step=saved_step,
        rank_count=payload["parameter_fingerprint"]["rank_count"],
    )
    payload["uninterrupted_next_step"] = _validated_next_step(
        payload["uninterrupted_next_step"], saved_step=saved_step
    )
    return payload


def checkpoint_save_receipt_sha256(checkpoint: str | Path) -> str:
    path = Path(checkpoint) / RECEIPT_FILENAME
    read_checkpoint_save_receipt(checkpoint)
    return _sha256(path)


def certify_external_checkpoint_resume(
    checkpoint: str | Path,
    resume_artifact: str | Path,
    *,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Certify a fresh-process full-state restore and its next optimizer step."""

    checkpoint = Path(checkpoint).resolve()
    receipt = read_checkpoint_save_receipt(checkpoint)
    receipt_sha256 = checkpoint_save_receipt_sha256(checkpoint)
    metadata, _, _ = load_artifact(resume_artifact)
    resume = metadata.get("external_checkpoint_resume")
    details: list[str] = []
    if not isinstance(resume, dict):
        details.append("resume artifact has no external checkpoint evidence")
        resume = {}
    if resume.get("status") != "passed":
        details.append("fresh-process checkpoint restore did not pass")
    if resume.get("checkpoint") != str(checkpoint):
        details.append("resume artifact names a different checkpoint")
    if resume.get("save_receipt_sha256") != receipt_sha256:
        details.append("resume artifact was not bound to the current save receipt")
    if resume.get("restored_step") != receipt["saved_step"]:
        details.append("restored step differs from the saved step")
    expected_fingerprint = receipt["parameter_fingerprint"]
    if resume.get("expected_parameter_sha256") != expected_fingerprint["sha256"]:
        details.append("resume artifact expected a different parameter fingerprint")
    if resume.get("restored_parameter_sha256") != expected_fingerprint["sha256"]:
        details.append("restored distributed parameters differ from the save point")
    if resume.get("restored_optimizer_sha256") != receipt["optimizer_fingerprint"]["sha256"]:
        details.append("restored optimizer state differs from the save point")
    if resume.get("restored_rng_sha256") != receipt["rng_fingerprint"]["sha256"]:
        details.append("restored RNG state differs from the save point")
    if resume.get("optimizer_loaded") is not True or resume.get("rng_loaded") is not True:
        details.append("resume artifact does not attest full optimizer and RNG loading")
    if metadata.get("topology") != receipt["topology"]:
        details.append("resume topology differs from the checkpoint topology")
    if metadata.get("architecture") != receipt["architecture"]:
        details.append("resume architecture differs from the checkpoint architecture")
    if metadata.get("depth_router") != receipt["depth_router"]:
        details.append("resume depth-router contract differs from the checkpoint")
    if metadata.get("optimizer") != receipt["optimizer"]:
        details.append("resume optimizer contract differs from the checkpoint")
    if resume.get("run_contract") != receipt["run_contract"]:
        details.append("resume data/routing contract differs from the checkpoint")

    expected_next = receipt["uninterrupted_next_step"]
    if resume.get("continued_step") != expected_next["step"]:
        details.append("fresh-process continuation executed an unexpected step")
    if resume.get("continued_parameter_sha256") != expected_next["parameter_fingerprint"]["sha256"]:
        details.append("fresh-process next-step parameters differ from uninterrupted training")
    if resume.get("continued_optimizer_sha256") != expected_next["optimizer_fingerprint"]["sha256"]:
        details.append("fresh-process next-step optimizer differs from uninterrupted training")
    if resume.get("continued_rng_sha256") != expected_next["rng_fingerprint"]["sha256"]:
        details.append("fresh-process next-step RNG differs from uninterrupted training")

    optimizer_steps = metadata.get("optimizer_steps")
    matches = (
        [
            item
            for item in optimizer_steps
            if isinstance(item, dict)
            and item.get("phase") == "resume"
            and item.get("step") == receipt["saved_step"]
        ]
        if isinstance(optimizer_steps, list)
        else []
    )
    if len(matches) != 1:
        details.append("resume artifact must contain exactly one next optimizer step")
    elif not bool(matches[0].get("updated")):
        details.append("resumed optimizer did not update")
    elif not math.isfinite(float(matches[0].get("grad_norm", float("nan")))):
        details.append("resumed optimizer step has a non-finite grad norm")

    report = {
        "passed": not details,
        "artifact_identity": {
            "resume": {
                "source_sha256": metadata.get("source_snapshot", {}).get("sha256"),
                "manifest_sha256": _sha256(Path(resume_artifact) / "manifest.json"),
                "routes_sha256": _sha256(Path(resume_artifact) / "routes.json"),
                "tensors_sha256": _sha256(Path(resume_artifact) / "tensors.pt"),
            }
        },
        "checkpoint": str(checkpoint),
        "resume_artifact": str(Path(resume_artifact).resolve()),
        "saved_step": receipt["saved_step"],
        "save_receipt_sha256": receipt_sha256,
        "parameter_sha256": expected_fingerprint["sha256"],
        "optimizer_sha256": receipt["optimizer_fingerprint"]["sha256"],
        "rng_sha256": receipt["rng_fingerprint"]["sha256"],
        "continued_parameter_sha256": expected_next["parameter_fingerprint"]["sha256"],
        "continued_optimizer_sha256": expected_next["optimizer_fingerprint"]["sha256"],
        "continued_rng_sha256": expected_next["rng_fingerprint"]["sha256"],
        "details": details,
    }
    if report_path is not None:
        destination = Path(report_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return report


__all__ = [
    "RECEIPT_FILENAME",
    "certify_external_checkpoint_resume",
    "checkpoint_save_receipt_sha256",
    "collect_rng_sidecar_manifest",
    "read_checkpoint_save_receipt",
    "verify_rng_sidecar_manifest",
    "write_checkpoint_save_receipt",
]
