from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mor_mlite.parity.receipt import TINY_TOPOLOGIES, write_tiny_matrix_receipt
from mor_mlite.parity.topologies import get_topology


def _write_reports(root: Path) -> None:
    reports = root / "reports"
    reports.mkdir(parents=True)
    names = ["reference_fp32_replay"]
    for topology in TINY_TOPOLOGIES:
        names.extend((f"{topology}_learned", topology))
    names.extend(("cp_canonical", "cp_canonical_vs_direct", "checkpoint_external_resume"))

    def artifact(
        name, *, backend="mlite", topology="baseline", mode="replay", transition="magi_direct"
    ):
        path = root / name
        path.mkdir(exist_ok=True)
        metadata = {
            "backend": backend,
            "topology": get_topology(topology).to_dict(),
            "route_mode": mode,
            "cp_transition": transition,
            "source_snapshot": {"sha256": "a" * 64},
        }
        (path / "manifest.json").write_text(json.dumps(metadata))
        (path / "routes.json").write_text("[]")
        (path / "tensors.pt").write_bytes(b"fixture")
        identity = {
            "source_sha256": "a" * 64,
            "manifest_sha256": hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest(),
            "routes_sha256": hashlib.sha256(b"[]").hexdigest(),
            "tensors_sha256": hashlib.sha256(b"fixture").hexdigest(),
        }
        return str(path), identity

    for name in names:
        if name == "checkpoint_external_resume":
            path, identity = artifact("checkpoint_resume")
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "mor_parity_checkpoint.json").write_text("{}")
            payload = {
                "passed": True,
                "resume_artifact": path,
                "checkpoint": str(checkpoint),
                "save_receipt_sha256": hashlib.sha256(b"{}").hexdigest(),
                "artifact_identity": {"resume": identity},
            }
        else:
            backend = "reference" if name == "reference_fp32_replay" else "mlite"
            topology = (
                "baseline"
                if backend == "reference"
                else "cp"
                if name.startswith("cp_canonical")
                else name.removesuffix("_learned")
            )
            baseline, lhs = artifact(
                name + "_baseline",
                backend=backend,
                topology="cp" if name == "cp_canonical_vs_direct" else "baseline",
                transition="magi_canonical" if name == "cp_canonical_vs_direct" else "magi_direct",
            )
            candidate, rhs = artifact(
                name + "_candidate",
                backend=backend,
                topology=topology,
                mode="learned" if name.endswith("_learned") else "replay",
                transition="magi_canonical" if name == "cp_canonical" else "magi_direct",
            )
            payload = {
                "passed": True,
                "acceptance_complete": True,
                "scope": "all",
                "baseline": baseline,
                "candidate": candidate,
                "artifact_identity": {"baseline": lhs, "candidate": rhs},
            }
        (reports / f"{name}.json").write_text(
            json.dumps(payload) + "\n",
            encoding="utf-8",
        )


def test_full_matrix_receipt_hashes_every_required_passed_report(tmp_path: Path) -> None:
    _write_reports(tmp_path)

    path = write_tiny_matrix_receipt(tmp_path, job_id="12345")
    receipt = json.loads(path.read_text(encoding="utf-8"))

    assert receipt["status"] == "passed"
    assert receipt["slurm_job_id"] == "12345"
    assert receipt["topologies"] == list(TINY_TOPOLOGIES)
    assert len(receipt["reports"]) == 20
    assert all(len(item["sha256"]) == 64 for item in receipt["reports"].values())
    assert receipt["source_sha256"] == "a" * 64


@pytest.mark.parametrize(
    "mutation",
    [
        "partial",
        "scope",
        "unbound",
        "changed_manifest",
        "wrong_topology",
        "mixed_source",
        "mixed_seed",
        "changed_tensors",
    ],
)
def test_receipt_rejects_unbound_or_mislabelled_reports(tmp_path, mutation):
    _write_reports(tmp_path)
    target = tmp_path / "reports" / "all.json"
    report = json.loads(target.read_text())
    if mutation == "partial":
        report["acceptance_complete"] = False
    elif mutation == "scope":
        report["scope"] = "forward"
    elif mutation == "unbound":
        del report["artifact_identity"]
    elif mutation == "changed_tensors":
        (Path(report["candidate"]) / "tensors.pt").write_bytes(b"changed")
    else:
        manifest = Path(report["candidate"]) / "manifest.json"
        data = json.loads(manifest.read_text())
        if mutation == "wrong_topology":
            data["topology"] = get_topology("tp").to_dict()
        elif mutation == "mixed_source":
            data["source_snapshot"]["sha256"] = "b" * 64
            report["artifact_identity"]["candidate"]["source_sha256"] = "b" * 64
        elif mutation == "mixed_seed":
            data["seed"] = 5678
        else:
            data["changed"] = True
        manifest.write_text(json.dumps(data))
        if mutation != "changed_manifest":
            report["artifact_identity"]["candidate"]["manifest_sha256"] = hashlib.sha256(
                manifest.read_bytes()
            ).hexdigest()
    target.write_text(json.dumps(report))
    with pytest.raises((RuntimeError, TypeError)):
        write_tiny_matrix_receipt(tmp_path, job_id="12345")


def test_full_matrix_receipt_fails_when_a_report_is_missing_or_failed(tmp_path: Path) -> None:
    _write_reports(tmp_path)
    missing = tmp_path / "reports" / "all.json"
    missing.unlink()
    with pytest.raises(FileNotFoundError, match="all.json"):
        write_tiny_matrix_receipt(tmp_path, job_id="12345")

    missing.write_text(json.dumps({"passed": False}) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="did not pass"):
        write_tiny_matrix_receipt(tmp_path, job_id="12345")
