from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from mor_mlite.parity.artifacts import save_artifact
from mor_mlite.parity.external_checkpoint import (
    RECEIPT_FILENAME,
    certify_external_checkpoint_resume,
    checkpoint_save_receipt_sha256,
    collect_rng_sidecar_manifest,
    read_checkpoint_save_receipt,
    verify_rng_sidecar_manifest,
    write_checkpoint_save_receipt,
)


def _parameter_fingerprint(digest: str = "a" * 64) -> dict:
    return {
        "sha256": digest,
        "rank_count": 1,
        "parameter_counts": [1],
        "num_bytes": [2],
        "rank_manifests": [{"rank": 0, "parameters": []}],
    }


def _state_fingerprint(digest: str, *, optimizer: bool = False) -> dict:
    rank_fingerprint = {
        "rank": 0,
        "sha256": digest,
        "tensor_count": 1,
        "tensor_bytes": 4,
    }
    result = {
        "sha256": digest,
        "rank_count": 1,
        "tensor_counts": [1],
        "tensor_bytes": [4],
        "rank_fingerprints": [rank_fingerprint],
    }
    if optimizer:
        optimizer_statistics = {
            "optimizer_leaf_count": 1,
            "master_parameter_count": 1,
            "master_parameter_bytes": 4,
            "adam_moment_tensor_count": 2,
            "adam_moment_tensor_bytes": 8,
            "optimizer_step_count": 1,
        }
        rank_fingerprint.update(optimizer_statistics)
        result.update(
            {
                "tensor_counts": [4],
                "tensor_bytes": [16],
                "optimizer_leaf_counts": [1],
                "master_parameter_counts": [1],
                "master_parameter_bytes": [4],
                "adam_moment_tensor_counts": [2],
                "adam_moment_tensor_bytes": [8],
                "optimizer_step_counts": [1],
            }
        )
        rank_fingerprint["tensor_count"] = 4
        rank_fingerprint["tensor_bytes"] = 16
    return result


def _contracts() -> dict:
    return {
        "topology": {
            "name": "all",
            "world_size": 8,
            "tp": 2,
            "cp": 2,
            "dp": 2,
            "ep": 4,
            "etp": 1,
        },
        "architecture": {
            "n_start_layers": 3,
            "n_recurrent_layers": 14,
            "num_recursions": 3,
            "n_end_layers": 3,
            "capacity_schedule": "linear",
        },
        "depth_router": {"temperature": 1.0, "alpha": 0.1, "aux_loss_coef": 0.001},
        "optimizer": {"name": "adam", "lr": 0.001, "adam_eps": 1e-6, "clip_grad": 1.0},
        "run_contract": {
            "preset": "qwen3-30b",
            "precision": "bf16",
            "strict": True,
            "attention_policy": "native-bf16-local-ffa-strict-cp1-magi-cp-v1",
            "seed": 1234,
            "seq_lens": [128, 128],
            "num_microbatches": 1,
            "route_mode": "replay",
            "replay_routes_sha256": "9" * 64,
            "cp_transition": "magi_direct",
        },
    }


def _write_save_receipt(checkpoint: Path) -> dict:
    contracts = _contracts()
    rng_root = checkpoint / "step_1"
    rng_root.mkdir()
    (rng_root / "rng_state_rank_00000.pt").write_bytes(b"rng-state")
    write_checkpoint_save_receipt(
        checkpoint,
        saved_step=1,
        **contracts,
        parameter_fingerprint=_parameter_fingerprint(),
        optimizer_fingerprint=_state_fingerprint("b" * 64, optimizer=True),
        rng_fingerprint=_state_fingerprint("c" * 64),
        rng_sidecars=collect_rng_sidecar_manifest(checkpoint, saved_step=1),
        uninterrupted_next_step={
            "step": 1,
            "parameter_fingerprint": _parameter_fingerprint("d" * 64),
            "optimizer_fingerprint": _state_fingerprint("e" * 64, optimizer=True),
            "rng_fingerprint": _state_fingerprint("f" * 64),
            "optimizer_step": {
                "phase": "uninterrupted",
                "step": 1,
                "updated": True,
                "grad_norm": 1.25,
                "num_zeros": 0,
            },
        },
    )
    return contracts


def test_runtime_generated_contract_is_accepted_by_checkpoint_schema(tmp_path):
    from mor_mlite.parity.external_checkpoint import _validated_run_contract
    from mor_mlite.parity.mlite import MLiteRunConfig, _external_run_contract

    config = MLiteRunConfig(output=tmp_path)
    contract = _external_run_contract(config, cp_transition="magi_direct")
    assert _validated_run_contract(contract) == contract
    del contract["attention_policy"]
    with pytest.raises(ValueError, match="attention_policy"):
        _validated_run_contract(contract)


@pytest.mark.parametrize("policy", [None, "", False, 1])
def test_checkpoint_rejects_invalid_attention_policy(policy):
    from mor_mlite.parity.external_checkpoint import _validated_run_contract

    contract = _contracts()["run_contract"]
    contract["attention_policy"] = policy
    with pytest.raises(ValueError, match="attention_policy"):
        _validated_run_contract(contract)


def _write_resume_artifact(root: Path, checkpoint: Path, contracts: dict, **overrides) -> None:
    external = {
        "status": "passed",
        "checkpoint": str(checkpoint.resolve()),
        "save_receipt_sha256": checkpoint_save_receipt_sha256(checkpoint),
        "restored_step": 1,
        "expected_parameter_sha256": "a" * 64,
        "restored_parameter_sha256": "a" * 64,
        "restored_optimizer_sha256": "b" * 64,
        "restored_rng_sha256": "c" * 64,
        "optimizer_loaded": True,
        "rng_loaded": True,
        "run_contract": contracts["run_contract"],
        "continued_step": 1,
        "continued_parameter_sha256": "d" * 64,
        "continued_optimizer_sha256": "e" * 64,
        "continued_rng_sha256": "f" * 64,
        **overrides,
    }
    save_artifact(
        root,
        metadata={
            **contracts,
            "backend": "mlite",
            "external_checkpoint_resume": external,
            "optimizer_steps": [
                {
                    "phase": "resume",
                    "step": 1,
                    "updated": True,
                    "grad_norm": 1.25,
                    "num_zeros": 0,
                }
            ],
        },
        tensors={"loss/resume/step_001/microbatch_000/lm": torch.tensor(1.0)},
        routes=[],
    )


def test_external_checkpoint_receipt_and_fresh_resume_certificate(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    contracts = _write_save_receipt(checkpoint)
    receipt = read_checkpoint_save_receipt(checkpoint)
    assert receipt["saved_step"] == 1
    assert receipt["optimizer_saved"] is True
    assert receipt["rng_saved"] is True

    resumed = tmp_path / "resumed"
    _write_resume_artifact(resumed, checkpoint, contracts)
    report_path = tmp_path / "report.json"
    report = certify_external_checkpoint_resume(checkpoint, resumed, report_path=report_path)
    assert report["passed"] is True
    assert report["details"] == []
    assert json.loads(report_path.read_text(encoding="utf-8")) == report


def test_external_checkpoint_certificate_fails_on_optimizer_drift(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    contracts = _write_save_receipt(checkpoint)
    resumed = tmp_path / "resumed"
    _write_resume_artifact(
        resumed,
        checkpoint,
        contracts,
        restored_optimizer_sha256="d" * 64,
    )
    report = certify_external_checkpoint_resume(checkpoint, resumed)
    assert report["passed"] is False
    assert "restored optimizer state differs from the save point" in report["details"]


def test_external_checkpoint_certificate_fails_on_next_step_drift(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    contracts = _write_save_receipt(checkpoint)
    resumed = tmp_path / "resumed"
    _write_resume_artifact(
        resumed,
        checkpoint,
        contracts,
        continued_parameter_sha256="0" * 64,
    )
    report = certify_external_checkpoint_resume(checkpoint, resumed)
    assert report["passed"] is False
    assert (
        "fresh-process next-step parameters differ from uninterrupted training" in report["details"]
    )


def test_external_checkpoint_receipt_rejects_unknown_fields(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    _write_save_receipt(checkpoint)
    path = checkpoint / RECEIPT_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["untrusted_extension"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        read_checkpoint_save_receipt(checkpoint)
    except ValueError as error:
        assert "fields differ" in str(error)
    else:
        raise AssertionError("unknown checkpoint receipt fields were accepted")


def test_external_checkpoint_receipt_rejects_metadata_only_optimizer_hash(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    _write_save_receipt(checkpoint)
    path = checkpoint / RECEIPT_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    fingerprint = payload["optimizer_fingerprint"]
    del fingerprint["master_parameter_counts"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="master_parameter_counts"):
        read_checkpoint_save_receipt(checkpoint)


def test_external_checkpoint_receipt_rejects_optimizer_tensor_undercoverage(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    _write_save_receipt(checkpoint)
    path = checkpoint / RECEIPT_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    fingerprint = payload["optimizer_fingerprint"]
    fingerprint["tensor_counts"] = [1]
    fingerprint["rank_fingerprints"][0]["tensor_count"] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="does not cover master parameters"):
        read_checkpoint_save_receipt(checkpoint)


def test_rng_sidecar_manifest_rejects_missing_or_changed_rank_file(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    _write_save_receipt(checkpoint)
    receipt = read_checkpoint_save_receipt(checkpoint)
    verify_rng_sidecar_manifest(
        checkpoint,
        receipt["rng_sidecars"],
        saved_step=1,
        rank_count=1,
    )

    (checkpoint / "step_1" / "rng_state_rank_00000.pt").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="RNG sidecar.*verification failed"):
        verify_rng_sidecar_manifest(
            checkpoint,
            receipt["rng_sidecars"],
            saved_step=1,
            rank_count=1,
        )
