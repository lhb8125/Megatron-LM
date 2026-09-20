import subprocess
import sys
from unittest.mock import Mock

import pytest
from test_pretraining_probes import load_probe


@pytest.mark.parametrize("failure", [subprocess.TimeoutExpired(["ssh"], 60),
                                    subprocess.CalledProcessError(255, ["ssh"])])
def test_readonly_transport_retry_bounded(monkeypatch, failure):
    module = load_probe("readonly_probe")
    success = subprocess.CompletedProcess(["ssh"], 0, "{}")
    run = Mock(side_effect=[failure, success])
    monkeypatch.setattr(module.subprocess, "run", run)
    monkeypatch.setattr(module.time, "sleep", Mock())
    assert module.run_readonly(["ssh"]) is success
    assert [c.kwargs["timeout"] for c in run.call_args_list] == [60, 120]
    run.reset_mock()
    run.side_effect = failure
    with pytest.raises(type(failure)):
        module.run_readonly(["ssh"])
    assert run.call_count == 3


def test_readonly_does_not_retry_checkpoint_validation(monkeypatch):
    module = load_probe("readonly_probe")
    run = Mock(side_effect=subprocess.CalledProcessError(1, ["ssh"], stderr="corrupt RNG archive"))
    monkeypatch.setattr(module.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        module.run_readonly(["ssh"])
    assert run.call_count == 1


def test_exact_checkpoint_read_recovery(monkeypatch):
    probe = load_probe("inspect_recovery")
    monkeypatch.setitem(sys.modules, "inspect_recovery", probe)
    read = Mock(return_value={})
    monkeypatch.setattr(probe, "remote_files", read)
    driver = load_probe("drive_campaign")
    native = "/formal/d/checkpoint-000800/step_800"
    state = {"schema_version": 2, "status": "failed", "binding": {"host": "host"},
             "runs": [], "active": {"D": {"output": "/formal/d", "scheduler_id": "123"}},
             "completed": {"D": 520}}
    cmd = probe.remote_files_command("host", native)
    state["error"] = f"CalledProcessError: {subprocess.CalledProcessError(255, cmd)}"
    result = driver.recover_checkpoint_read(state, native)
    assert result["status"] == "watching" and state["status"] == "failed"
    assert result["active"] == state["active"] and result["completed"] == state["completed"]
    assert result["checkpoint_read_recoveries"] == [state["error"]]
    read.assert_called_once_with("host", native)
    for changes in ({"intent": {}}, {"error": "TimeoutExpired: prune --apply"},
                    {"error": "CalledProcessError: sbatch"}, {"status": "watching"}):
        with pytest.raises(ValueError):
            driver.recover_checkpoint_read({**state, **changes}, native)
    with pytest.raises(ValueError):
        driver.recover_checkpoint_read(state, "/other/checkpoint-000800/step_800")
    read.side_effect = subprocess.CalledProcessError(1, cmd, stderr="corrupt RNG archive")
    with pytest.raises(subprocess.CalledProcessError):
        driver.recover_checkpoint_read(state, native)
    assert state["status"] == "failed"
