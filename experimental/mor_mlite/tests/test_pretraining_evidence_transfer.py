import subprocess
from unittest.mock import Mock

import pytest
from test_pretraining_probes import load_probe


@pytest.mark.parametrize("failure", [subprocess.TimeoutExpired(["rsync"], 180),
                                    subprocess.CalledProcessError(255, ["rsync"])])
def test_evidence_retry_is_bounded(monkeypatch, tmp_path, failure):
    driver = load_probe("drive_campaign")
    active = {"scheduler_id": "123", "output": "/formal/d", "to_step": 920}
    run = Mock(side_effect=[failure, subprocess.CompletedProcess([], 0)])
    monkeypatch.setattr(driver.subprocess, "run", run)
    monkeypatch.setattr(driver.time, "sleep", Mock())
    assert driver.fetch_evidence("host", active, tmp_path) == tmp_path / "evidence/123"
    assert [c.kwargs["timeout"] for c in run.call_args_list] == [180, 240]
    cmd = run.call_args.args[0]
    assert cmd[-2:] == ["host:/formal/d/", str(tmp_path / "evidence/123") + "/"]
    assert "--delete" not in cmd and "--remove-source-files" not in cmd
    run.reset_mock()
    run.side_effect = failure
    with pytest.raises(type(failure)):
        driver.fetch_evidence("host", active, tmp_path)
    assert run.call_count == 3


@pytest.mark.parametrize("code", [1, 12, 23, 24])
def test_evidence_does_not_retry_nontransport_failure(monkeypatch, tmp_path, code):
    driver = load_probe("drive_campaign")
    run = Mock(side_effect=subprocess.CalledProcessError(code, ["rsync"]))
    monkeypatch.setattr(driver.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        driver.fetch_evidence("host", {"scheduler_id": "123", "output": "/d", "to_step": 920}, tmp_path)
    assert run.call_count == 1


def test_exact_evidence_recovery_preserves_jobs_and_cursors(monkeypatch, tmp_path):
    driver = load_probe("drive_campaign")
    active = {"scheduler_id": "123", "output": "/formal/d", "to_step": 920}
    state = {"schema_version": 2, "status": "failed", "binding": {"host": "host"},
             "runs": [], "active": {"D": active}, "completed": {"D": 520}}
    command = driver.evidence_command("host", active, tmp_path)
    state["error"] = f"CalledProcessError: {subprocess.CalledProcessError(255, command)}"
    fetch = Mock()
    monkeypatch.setattr(driver, "fetch_evidence", fetch)
    result = driver.recover_evidence_transfer(state, tmp_path, "123")
    assert result["status"] == "watching" and state["status"] == "failed"
    assert result["active"] == state["active"] and result["completed"] == state["completed"]
    assert result["evidence_transfer_recoveries"] == [state["error"]]
    fetch.assert_called_once_with("host", active, tmp_path)
    for changes in ({"intent": {}}, {"status": "watching"},
                    {"error": "CalledProcessError: sbatch"},
                    {"error": "TimeoutExpired: prune --apply"},
                    {"error": f"CalledProcessError: {subprocess.CalledProcessError(23, command)}"}):
        with pytest.raises(ValueError):
            driver.recover_evidence_transfer({**state, **changes}, tmp_path, "123")
    with pytest.raises(ValueError):
        driver.recover_evidence_transfer(state, tmp_path, "999")
    with pytest.raises(ValueError):
        driver.recover_evidence_transfer(state, tmp_path / "different", "123")
    fetch.side_effect = subprocess.CalledProcessError(255, command)
    with pytest.raises(subprocess.CalledProcessError):
        driver.recover_evidence_transfer(state, tmp_path, "123")
    assert state["status"] == "failed" and state["completed"] == {"D": 520}
