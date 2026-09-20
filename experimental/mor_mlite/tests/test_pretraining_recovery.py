"""Recovery preflight must not turn partial logs into completed allocations."""

import json
from types import SimpleNamespace

import pytest
from test_pretraining_probes import load_probe


def test_prefix_ignores_only_suffix_damage(tmp_path):
    module = load_probe("inspect_recovery")
    path = tmp_path / "loss.jsonl"
    path.write_text('{"step": 121}\n{"step": 122}\n{"step":')
    assert len(module.prefix_rows(path, 121, 122)) == 2
    with pytest.raises(json.JSONDecodeError):
        module.prefix_rows(path, 121, 123)
    path.write_text('{"step": 121}\n{"step": 123}\n')
    with pytest.raises(ValueError, match="discontinuous"):
        module.prefix_rows(path, 121, 122)
    path.write_text('{"step": 121}\n')
    with pytest.raises(ValueError, match="missing"):
        module.prefix_rows(path, 121, 122)


def metadata(name="__0_0.distcp", offset=0, length=100):
    return SimpleNamespace(
        storage_data={0: SimpleNamespace(relative_path=name, offset=offset, length=length)},
        state_dict_metadata={f"optimizer.state.{key}.weight": None for key in ("param", "exp_avg", "exp_avg_sq")},
    )


def test_recovery_storage_missing_truncated_and_optimizer():
    module = load_probe("inspect_recovery")
    files = {"__0_0.distcp": {"size": 100}}
    assert module.validate_storage(metadata(), files) == {"__0_0.distcp": 100}
    for values in ({}, {"__0_0.distcp": {"size": 99}}):
        with pytest.raises(ValueError, match="missing/truncated"):
            module.validate_storage(metadata(), values)
    for broken in (metadata("../bad"), metadata(offset=-1), metadata(length=0)):
        with pytest.raises(ValueError, match="unsafe or invalid"):
            module.validate_storage(broken, files)
    broken = metadata()
    del broken.state_dict_metadata["optimizer.state.exp_avg_sq.weight"]
    with pytest.raises(ValueError, match="missing optimizer"):
        module.validate_storage(broken, files)


def test_recovery_checks_last_extent_not_first():
    module = load_probe("inspect_recovery")
    value = metadata()
    value.storage_data[1] = SimpleNamespace(relative_path="__0_0.distcp", offset=100, length=200)
    with pytest.raises(ValueError, match="truncated"):
        module.validate_storage(value, {"__0_0.distcp": {"size": 299}})


@pytest.fixture
def reconciliation(tmp_path, monkeypatch):
    module = load_probe("reconcile_campaign")
    import drive_campaign

    audits = []
    monkeypatch.setattr(drive_campaign, "audit_history", lambda *args: audits.append("history"))
    monkeypatch.setattr(drive_campaign, "audit_record", lambda *args: audits.append(args[1]["arm"]))
    state = {"schema_version": 2, "status": "failed", "error": "quota", "runs": [],
             "completed": dict.fromkeys("ABCD", 120), "active": {}}
    statuses = {}
    for rank, arm in enumerate("ABCD"):
        job = str(rank + 10)
        state["active"][arm] = {"arm": arm, "from_step": 120, "to_step": 520,
                                "stop_tokens": 520 * 8388608, "scheduler_id": job,
                                "registry_id": arm, "output": f"/output/{arm}"}
        statuses[job] = ("FAILED", "143:0") if arm in "AB" else (
            ("COMPLETED", "0:0") if arm == "C" else ("RUNNING", "0:0"))
        (tmp_path / job).mkdir()
        (tmp_path / job / "recovery-preflight-200.json").write_text("{}")
    receipts = dict.fromkeys("ABCD", {})
    return module, state, statuses, tmp_path, receipts, audits


def test_reconcile_keeps_live_job_and_records_only_saved_prefix(reconciliation):
    module, state, statuses, evidence, receipts, audits = reconciliation
    result = module.reconcile(state, statuses, evidence, receipts, 5954, {"A": 200, "B": 200})
    assert state["status"] == "failed" and len(state["active"]) == 4
    assert result["completed"] == {"A": 200, "B": 200, "C": 520, "D": 120}
    assert result["active"] == {"D": state["active"]["D"]}
    assert result["status"] == "watching" and result["reconciled_error"] == "quota"
    assert audits == ["history", "A", "B", "C", "history"]
    assert result["runs"][0]["recovery"]["original_plan"]["to_step"] == 520
    assert result["runs"][0]["to_step"] == 200


@pytest.mark.parametrize("change", ["intent", "live", "missing", "cancelled", "zero-exit"])
def test_reconcile_rejects_ambiguous_or_unrequested_failure(reconciliation, change):
    module, state, statuses, evidence, receipts, _ = reconciliation
    requests = {"A": 200, "B": 200}
    if change == "intent":
        state["intent"] = {}
    elif change == "live":
        statuses["10"] = ("RUNNING", "0:0")
    elif change == "missing":
        del requests["B"]
    elif change == "cancelled":
        statuses["10"] = ("CANCELLED", "0:15")
    elif change == "zero-exit":
        statuses["10"] = ("FAILED", "0:0")
    with pytest.raises(ValueError):
        module.reconcile(state, statuses, evidence, receipts, 5954, requests)


def test_reconcile_does_not_override_failed_audit(reconciliation, monkeypatch):
    module, state, statuses, evidence, receipts, _ = reconciliation
    import drive_campaign

    def reject(*args):
        raise ValueError("changed checkpoint")

    monkeypatch.setattr(drive_campaign, "audit_record", reject)
    with pytest.raises(ValueError, match="changed checkpoint"):
        module.reconcile(state, statuses, evidence, receipts, 5954, {"A": 200, "B": 200})
    assert len(state["active"]) == 4
