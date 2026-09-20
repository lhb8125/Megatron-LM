import subprocess
import sys
from unittest.mock import Mock

import pytest
from test_pretraining_probes import load_probe


@pytest.mark.parametrize("transport", ["timeout", "ssh255"])
def test_inventory_timeout_retry_is_bounded(monkeypatch, transport):
    module = load_probe("plan_checkpoint_retention")
    probe = load_probe("readonly_probe")
    monkeypatch.setitem(sys.modules, "readonly_probe", probe)
    command = module.inventory_command("host", [])
    failure = (subprocess.TimeoutExpired(command, 60) if transport == "timeout"
               else subprocess.CalledProcessError(255, command))
    run = Mock(side_effect=[failure,
                           subprocess.CompletedProcess(command, 0, "[]")])
    sleep = Mock()
    monkeypatch.setattr(probe.subprocess, "run", run)
    monkeypatch.setattr(probe.time, "sleep", sleep)
    assert module.remote_inventory("host", []) == []
    assert [c.kwargs["timeout"] for c in run.call_args_list] == [60, 120]
    sleep.assert_called_once_with(5)
    run.reset_mock()
    run.side_effect = subprocess.TimeoutExpired(command, 180)
    with pytest.raises(subprocess.TimeoutExpired):
        module.remote_inventory("host", [])
    assert run.call_count == 3


def test_inventory_does_not_retry_remote_validation_or_invalid_json(monkeypatch):
    module = load_probe("plan_checkpoint_retention")
    probe = load_probe("readonly_probe")
    monkeypatch.setitem(sys.modules, "readonly_probe", probe)
    run = Mock(side_effect=subprocess.CalledProcessError(1, "ssh", stderr="unsafe shard"))
    monkeypatch.setattr(probe.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        module.remote_inventory("host", [])
    assert run.call_count == 1
    run.reset_mock()
    run.side_effect = None
    run.return_value = subprocess.CompletedProcess("ssh", 0, "not json")
    with pytest.raises(ValueError):
        module.remote_inventory("host", [])
    assert run.call_count == 1


@pytest.mark.parametrize("transport", ["timeout", "ssh255"])
def test_inventory_timeout_recovery_preserves_active_jobs(monkeypatch, transport):
    inventory = load_probe("plan_checkpoint_retention")
    monkeypatch.setitem(sys.modules, "plan_checkpoint_retention", inventory)
    driver = load_probe("drive_campaign")
    active = {"C": {"arm": "C", "output": "/formal/c", "scheduler_id": "123"}}
    state = {"schema_version": 2, "status": "failed", "binding": {"host": "host"},
             "runs": [], "active": active, "completed": {"C": 920}}
    command = inventory.inventory_command("host", list(active.values()))
    state["error"] = (f"TimeoutExpired: {subprocess.TimeoutExpired(command, 60)}"
                      if transport == "timeout" else
                      f"CalledProcessError: {subprocess.CalledProcessError(255, command)}")
    result = driver.recover_inventory_timeout(state)
    assert result["status"] == "watching" and state["status"] == "failed"
    assert result["active"] == active and result["completed"] == {"C": 920}
    assert result["inventory_timeout_recoveries"] == [state["error"]]
    for changed in ({"intent": {}}, {"error": "TimeoutExpired: sbatch"},
                    {"error": "TimeoutExpired: prune --apply"}, {"status": "watching"}):
        with pytest.raises(ValueError):
            driver.recover_inventory_timeout({**state, **changed})


def test_keep_milestones_two_latest_and_incomplete():
    module = load_probe("plan_checkpoint_retention")
    rows = [{"arm": "A", "path": f"/a/{s}", "step": s, "complete": s != 400}
            for s in (100, 120, 200, 300, 400)]
    candidates, kept = module.select_candidates(rows, {120, 1193}, set())
    assert [r["step"] for r in candidates] == [100]
    assert set(kept) == {"/a/120", "/a/200", "/a/300", "/a/400"}


def test_keep_active_restore_source_and_copies():
    module = load_probe("plan_checkpoint_retention")
    rows = [{"arm": "B", "path": f"/b/{s}", "step": s, "complete": True}
            for s in (200, 300, 400, 500)]
    rows.append({"arm": "B", "path": "/b-copy/500", "step": 500, "complete": True})
    candidates, kept = module.select_candidates(rows, set(), {"/b/200"})
    assert [r["step"] for r in candidates] == [300]
    assert "/b-copy/500" in kept


def test_keep_each_arm_independently():
    module = load_probe("plan_checkpoint_retention")
    rows = [{"arm": a, "path": f"/{a}/{s}", "step": s, "complete": True}
            for a, steps in (("A", [100, 200]), ("B", [300, 400, 500])) for s in steps]
    candidates, _ = module.select_candidates(rows, set(), set())
    assert [(r["arm"], r["step"]) for r in candidates] == [("B", 300)]
    with pytest.raises(ValueError, match="duplicate"):
        module.select_candidates(rows + rows[:1], set(), set())


@pytest.fixture
def inventory():
    state = {"binding": {"source": "source"}, "runs": [
        {"arm": "A", "output": "/formal/a", "from_step": 0, "to_step": 120}],
        "active": {"A": {"arm": "A", "output": "/formal/a-next", "from_step": 120, "to_step": 520}}}
    records = []
    for step in (100, 120, 200, 300):
        run = "/formal/a" if step <= 120 else "/formal/a-next"
        path = f"{run}/checkpoint-{step:06d}"
        records.append({"arm": "A", "path": path, "step": step,
                        "metadata_present": True,
                        "files": [{"path": f"{path}/step_{step}/__0_0.distcp", "size": 100}],
                        "marker": {"step": step, "cursor": step * 8388608,
                                   "scheduler": "pure-token-function-v1",
                                   "contract": {"mode": "train", "source_sha256": "source",
                                                "total_steps": 5954, "experiment": {"arm": "A"}}}})
    return state, records


def test_plan_is_read_only_and_protects_active_source(inventory):
    state, records = inventory
    module = load_probe("plan_checkpoint_retention")
    plan = module.build_plan(state, records, 5954)
    assert plan["read_only"] and not plan["deletion_authorized"]
    assert not plan["native_integrity_verified"]
    assert [r["step"] for r in plan["candidates"]] == [100]
    assert plan["candidate_bytes"] == 100
    assert "/formal/a/checkpoint-000120" in plan["kept_paths"]


@pytest.mark.parametrize("failure", ["outside", "cursor", "source", "not-shard", "ambiguous-resume"])
def test_plan_rejects_invalid_inventory(inventory, failure):
    state, records = inventory
    module = load_probe("plan_checkpoint_retention")
    if failure == "outside":
        records[0]["path"] = "/someone-else/checkpoint-000100"
    elif failure == "cursor":
        records[0]["marker"]["cursor"] += 1
    elif failure == "source":
        records[0]["marker"]["contract"]["source_sha256"] = "changed"
    elif failure == "not-shard":
        records[0]["files"][0]["path"] = "/formal/a/checkpoint-000100/pretraining-state.json"
    else:
        state["runs"] = []
    with pytest.raises(ValueError):
        module.build_plan(state, records, 5954)
