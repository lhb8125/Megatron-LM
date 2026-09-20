import copy
import io
import json
import subprocess
import sys
from unittest.mock import Mock

import pytest
from test_pretraining_probes import load_probe


PREAUTH = "ssh_exchange_identification: Connection closed by remote host\n"


@pytest.mark.parametrize(
    "free_values,expected_calls,admit",
    [([1000], 0, True), ([999, 1500], 2, True), ([999, 900], 2, False)],
)
def test_pressure_only_prunes_after_failed_budget(
    monkeypatch, tmp_path, free_values, expected_calls, admit
):
    module = load_probe("maintain_checkpoints")
    inventory = load_probe("plan_checkpoint_retention")
    readonly = load_probe("readonly_probe")
    monkeypatch.setitem(sys.modules, "plan_checkpoint_retention", inventory)
    monkeypatch.setitem(sys.modules, "readonly_probe", readonly)
    monkeypatch.setattr(inventory, "remote_inventory", Mock(return_value=[]))
    monkeypatch.setattr(
        inventory,
        "build_plan",
        Mock(return_value={"candidates": [{"files": [{"size": 10}]}], "kept_paths": []}),
    )
    monkeypatch.setattr(module, "verify_retained", Mock())
    monkeypatch.setattr(module, "write_budget", Mock(return_value=1000))
    read = Mock(side_effect=[subprocess.CompletedProcess([], 0, str(n), "") for n in free_values])
    monkeypatch.setattr(readonly, "run_readonly", read)
    prune = Mock(
        return_value=subprocess.CompletedProcess(
            [], 0, '{"applied": true, "files": 1, "bytes": 10}', ""
        )
    )
    monkeypatch.setattr(module, "run_prune", prune)
    state = {
        "binding": {"host": "host", "remote_root": "/formal", "prune_approved": "True"},
        "runs": [],
        "active": {},
    }
    result = module.maintain(state, tmp_path, 5954, prune_on_pressure=True)
    assert prune.call_count == expected_calls
    assert result["admit"] == admit
    assert result["removed_bytes"] == (10 if expected_calls else 0)
    assert read.call_count == len(free_values)


@pytest.mark.parametrize("approved,retain", [("False", False), ("True", True)])
def test_pressure_requires_authorization(monkeypatch, tmp_path, approved, retain):
    module = load_probe("maintain_checkpoints")
    monkeypatch.setitem(
        sys.modules, "plan_checkpoint_retention", load_probe("plan_checkpoint_retention")
    )
    monkeypatch.setitem(sys.modules, "readonly_probe", load_probe("readonly_probe"))
    with pytest.raises(ValueError, match="explicit authorization"):
        module.maintain(
            {"binding": {"prune_approved": approved}},
            tmp_path,
            5954,
            retain_all=retain,
            prune_on_pressure=True,
        )


@pytest.mark.parametrize("free,admit", [(1000, True), (999, False)])
def test_retain_all_never_deletes_but_checks_budget(monkeypatch, tmp_path, free, admit):
    module = load_probe("maintain_checkpoints")
    inventory = load_probe("plan_checkpoint_retention")
    readonly = load_probe("readonly_probe")
    monkeypatch.setitem(sys.modules, "plan_checkpoint_retention", inventory)
    monkeypatch.setitem(sys.modules, "readonly_probe", readonly)
    monkeypatch.setattr(inventory, "remote_inventory", Mock(return_value=[]))
    monkeypatch.setattr(inventory, "build_plan", Mock(return_value={"candidates": [{}]}))
    prune = Mock(side_effect=AssertionError("must not invoke prune"))
    monkeypatch.setattr(module, "run_prune", prune)
    monkeypatch.setattr(
        module, "verify_retained", Mock(side_effect=AssertionError("no deletion path"))
    )
    budget = Mock(return_value=1000)
    monkeypatch.setattr(module, "write_budget", budget)
    monkeypatch.setattr(
        readonly,
        "run_readonly",
        Mock(return_value=subprocess.CompletedProcess([], 0, str(free), "")),
    )
    state = {
        "binding": {"host": "host", "remote_root": "/formal"},
        "runs": [],
        "active": {"A": {"output": "/formal/a"}},
    }
    next_plan = {"output": "/formal/b"}
    result = module.maintain(state, tmp_path, 5954, next_plan=next_plan, retain_all=True)
    assert result == {
        "removed_bytes": 0,
        "free_bytes": free,
        "required_bytes": 1000,
        "admit": admit,
    }
    assert budget.call_args.args[1] == [state["active"]["A"], next_plan]
    prune.assert_not_called()
    assert list(tmp_path.iterdir()) == []


def test_only_proven_preauth_is_retried(monkeypatch):
    module = load_probe("maintain_checkpoints")
    raw = b'{"files": []}'
    command = module.prune_command("host", raw, apply=True)
    failed = subprocess.CompletedProcess(command, 255, "", PREAUTH)
    success = subprocess.CompletedProcess(command, 0, '{"applied": true}\n', "")
    run = Mock(side_effect=[failed, success])
    sleep = Mock()
    monkeypatch.setattr(module.subprocess, "run", run)
    monkeypatch.setattr(module.time, "sleep", sleep)
    log = io.BytesIO()
    assert module.run_prune("host", raw, apply=True, stream=log) == success
    assert run.call_count == 2
    assert all(
        c.args[0] == command and c.kwargs["input"] == raw.decode() for c in run.call_args_list
    )
    sleep.assert_called_once_with(5)
    assert log.getvalue().decode() == PREAUTH + success.stdout
    run.reset_mock()
    run.side_effect = None
    run.return_value = failed
    with pytest.raises(subprocess.CalledProcessError):
        module.run_prune("host", raw, apply=True)
    assert run.call_count == 3


@pytest.mark.parametrize(
    "stdout,stderr,code",
    [
        ('{"deleted":"file"}\n', PREAUTH, 255),
        ("", "Connection closed by remote host", 255),
        ("", "Permission denied", 255),
        ("", PREAUTH + "remote validation failed\n", 255),
        ("", PREAUTH, 1),
    ],
)
def test_ambiguous_or_remote_failures_not_retried(monkeypatch, stdout, stderr, code):
    module = load_probe("maintain_checkpoints")
    result = subprocess.CompletedProcess([], code, stdout, stderr)
    run = Mock(return_value=result)
    monkeypatch.setattr(module.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        module.run_prune("host", b"{}", apply=True)
    assert run.call_count == 1


def test_prune_timeout_not_retried(monkeypatch):
    module = load_probe("maintain_checkpoints")
    run = Mock(side_effect=subprocess.TimeoutExpired("ssh", 180))
    monkeypatch.setattr(module.subprocess, "run", run)
    with pytest.raises(subprocess.TimeoutExpired):
        module.run_prune("host", b"{}", apply=True)
    assert run.call_count == 1


@pytest.fixture
def recovery(monkeypatch, tmp_path):
    module = load_probe("maintain_checkpoints")
    inventory = load_probe("plan_checkpoint_retention")
    monkeypatch.setitem(sys.modules, "plan_checkpoint_retention", inventory)
    state = {
        "schema_version": 2,
        "status": "failed",
        "runs": [],
        "active": {"B": {"output": "/formal/b", "scheduler_id": "123"}},
        "completed": {"B": 600},
        "binding": {"host": "host", "prune_approved": "True"},
    }
    item = {
        "path": "/formal/b/checkpoint-000500/step_500/__0_0.distcp",
        "size": 10,
        "inode": 22,
        "mtime_ns": 33,
    }
    manifest = {"allowed_run_roots": ["/formal/b"], "files": [item]}
    raw = json.dumps(manifest).encode()
    path = tmp_path / "retention-123.json"
    path.write_bytes(raw)
    path.with_suffix(".log").write_text(PREAUTH)
    command = module.prune_command("host", raw, apply=True)
    state["error"] = f"CalledProcessError: {subprocess.CalledProcessError(255, command)}"
    monkeypatch.setattr(inventory, "remote_inventory", Mock(return_value=[]))
    plan = {"candidates": [{"files": [item]}]}
    monkeypatch.setattr(inventory, "build_plan", Mock(return_value=plan))
    check = Mock(
        return_value=subprocess.CompletedProcess(
            [], 0, json.dumps({"applied": False, "files": 1, "bytes": 10})
        )
    )
    monkeypatch.setattr(module, "run_prune", check)
    return module, state, path, plan, check


def test_recovery_is_readonly_and_preserves_jobs(recovery):
    module, state, path, _, check = recovery
    before = copy.deepcopy(state)
    result = module.recover_prune_preauth(state, path.parent, path.name, 5954)
    assert state == before and result["status"] == "watching"
    assert result["active"] == state["active"] and result["completed"] == state["completed"]
    assert result["prune_preauth_recoveries"][0]["error"] == state["error"]
    check.assert_called_once_with("host", path.read_bytes())


@pytest.mark.parametrize(
    "change",
    [
        "intent",
        "unapproved",
        "wrong-error",
        "changed-file",
        "missing-file",
        "partial-log",
        "empty-log",
        "bad-result",
    ],
)
def test_recovery_refuses_ambiguous_or_changed_state(recovery, change):
    module, state, path, plan, check = recovery
    if change == "intent":
        state["intent"] = {}
    elif change == "unapproved":
        state["binding"]["prune_approved"] = "False"
    elif change == "wrong-error":
        state["error"] = "TimeoutExpired: prune"
    elif change == "changed-file":
        plan["candidates"][0]["files"][0]["inode"] += 1
    elif change == "missing-file":
        plan["candidates"] = []
    elif change == "partial-log":
        path.with_suffix(".log").write_text('{"deleted":"x"}\n' + PREAUTH)
    elif change == "empty-log":
        path.with_suffix(".log").write_text("")
    elif change == "bad-result":
        check.return_value.stdout = '{"applied":true}'
    with pytest.raises(ValueError):
        module.recover_prune_preauth(state, path.parent, path.name, 5954)
    assert state["status"] == "failed"


def test_recovery_requires_successful_fresh_validation(recovery):
    module, state, path, _, check = recovery
    check.side_effect = subprocess.CalledProcessError(1, ["ssh"])
    with pytest.raises(subprocess.CalledProcessError):
        module.recover_prune_preauth(state, path.parent, path.name, 5954)
    assert state["status"] == "failed"
