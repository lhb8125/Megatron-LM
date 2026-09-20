"""Fail-closed completion receipts for EOS acceptance runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .topologies import get_topology

TINY_TOPOLOGIES = (
    "zero1",
    "tp",
    "cp",
    "ep",
    "tp_cp_ep",
    "tp_dp_ep",
    "cp_dp_ep",
    "all",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_passed_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required acceptance report is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("passed") is not True:
        raise RuntimeError(f"acceptance report did not pass: {path}")
    return payload


def _bound_artifact(report: dict, side: str) -> tuple[dict, dict]:
    identity = report.get("artifact_identity", {}).get(side)
    if not isinstance(identity, dict):
        raise TypeError(f"missing {side} artifact identity")
    path = Path(report["resume_artifact"] if side == "resume" else report[side])
    for filename, field in (
        ("manifest.json", "manifest_sha256"),
        ("routes.json", "routes_sha256"),
        ("tensors.pt", "tensors_sha256"),
    ):
        if _sha256(path / filename) != identity.get(field):
            raise RuntimeError(f"{side} {filename} changed since comparison")
    metadata = json.loads((path / "manifest.json").read_text())
    source = metadata.get("source_snapshot", {}).get("sha256")
    if not isinstance(source, str) or len(source) != 64 or source != identity.get("source_sha256"):
        raise RuntimeError(f"{side} source snapshot is missing or inconsistent")
    for field in ("topology", "route_mode", "cp_transition", "backend"):
        if field in identity and metadata.get(field) != identity[field]:
            raise RuntimeError(f"{side} {field} differs from compared artifact")
    # A matrix must describe one experiment per backend, not merely a set of
    # individually passing pairs with unrelated seeds, masks or initial states.
    contract = {
        field: metadata.get(field)
        for field in (
            "backend",
            "precision",
            "strict",
            "attention_policy",
            "seed",
            "steps",
            "num_microbatches",
            "model_config",
            "architecture",
            "depth_router",
            "optimizer",
            "versions",
            "precision_probe",
        )
    }
    contract["sequence_lengths"] = metadata.get("global_batch", {}).get("sequence_lengths")
    contract["initial_parameters"] = {
        key: value
        for key, value in metadata.get("tensor_index", {}).items()
        if key.startswith("initial/")
    }
    identity = {
        **identity,
        "experiment_sha256": hashlib.sha256(
            json.dumps(contract, sort_keys=True).encode()
        ).hexdigest(),
        "backend": metadata.get("backend"),
    }
    return metadata, identity


def _validate_report_identity(name: str, report: dict) -> list[dict]:
    if name == "checkpoint_external_resume":
        metadata, identity = _bound_artifact(report, "resume")
        if metadata.get("topology") != get_topology("baseline").to_dict():
            raise RuntimeError("checkpoint certificate does not describe the baseline topology")
        if _sha256(Path(report["checkpoint"]) / "mor_parity_checkpoint.json") != report.get(
            "save_receipt_sha256"
        ):
            raise RuntimeError("checkpoint save receipt changed since certification")
        return [identity]
    if report.get("acceptance_complete") is not True or report.get("scope") != "all":
        raise RuntimeError(f"{name} is not a complete training acceptance report")
    baseline, baseline_id = _bound_artifact(report, "baseline")
    candidate, candidate_id = _bound_artifact(report, "candidate")
    if name == "reference_fp32_replay":
        if baseline.get("backend") != "reference" or candidate.get("backend") != "reference":
            raise RuntimeError("reference report has the wrong backend")
        expected_mode = "replay"
    else:
        topology = "cp" if name.startswith("cp_canonical") else name.removesuffix("_learned")
        expected_baseline = "cp" if name == "cp_canonical_vs_direct" else "baseline"
        for label, metadata, expected in (
            ("baseline", baseline, expected_baseline),
            ("candidate", candidate, topology),
        ):
            if (
                metadata.get("backend") != "mlite"
                or metadata.get("topology") != get_topology(expected).to_dict()
            ):
                raise RuntimeError(f"{name} {label} has the wrong topology/backend")
        if name == "cp_canonical" and candidate.get("cp_transition") != "magi_canonical":
            raise RuntimeError("canonical report did not use magi_canonical")
        if name == "cp_canonical_vs_direct" and (
            baseline.get("cp_transition") != "magi_canonical"
            or candidate.get("cp_transition") != "magi_direct"
        ):
            raise RuntimeError("canonical/direct report has the wrong transition backends")
        expected_mode = "learned" if name.endswith("_learned") else "replay"
    if candidate.get("route_mode") != expected_mode:
        raise RuntimeError(f"{name} candidate has the wrong route mode")
    return [baseline_id, candidate_id]


def write_tiny_matrix_receipt(artifact_root: str | Path, *, job_id: str) -> Path:
    """Verify every required report and atomically write the full-matrix receipt."""

    root = Path(artifact_root)
    reports_root = root / "reports"
    names = ["reference_fp32_replay"]
    for topology in TINY_TOPOLOGIES:
        names.extend((f"{topology}_learned", topology))
    names.extend(("cp_canonical", "cp_canonical_vs_direct", "checkpoint_external_resume"))

    reports: dict[str, dict[str, str]] = {}
    source_hashes: set[str] = set()
    experiments: dict[str, set[str]] = {}
    for name in names:
        path = reports_root / f"{name}.json"
        payload = _load_passed_report(path)
        identities = _validate_report_identity(name, payload)
        source_hashes.update(identity["source_sha256"] for identity in identities)
        if name != "checkpoint_external_resume":
            for identity in identities:
                experiments.setdefault(identity["backend"], set()).add(
                    identity["experiment_sha256"]
                )
        reports[name] = {
            "path": str(path.relative_to(root)),
            "sha256": _sha256(path),
        }

    if len(source_hashes) != 1:
        raise RuntimeError("matrix reports were produced from different source snapshots")
    if any(len(values) != 1 for values in experiments.values()):
        raise RuntimeError(
            "matrix reports describe different input/config/initial-state experiments"
        )

    receipt = {
        "schema_version": 2,
        "source_sha256": next(iter(source_hashes)),
        "experiments": {backend: next(iter(values)) for backend, values in experiments.items()},
        "kind": "tiny_topology_matrix",
        "status": "passed",
        "slurm_job_id": str(job_id),
        "topologies": list(TINY_TOPOLOGIES),
        "reports": reports,
    }
    destination = reports_root / "matrix_complete.json"
    reports_root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=reports_root,
        prefix=".matrix_complete.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m mor_mlite.parity.receipt")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = write_tiny_matrix_receipt(args.artifact_root, job_id=args.job_id)
    print(json.dumps({"receipt": str(receipt)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["TINY_TOPOLOGIES", "write_tiny_matrix_receipt"]
