"""Formal orchestration guards; these tests do not issue GPU acceptance."""

import json

import pytest
from test_pretraining_probes import load_probe


def test_campaign_order_and_bounded_continuations():
    driver = load_probe("drive_campaign")
    first = driver.next_allocation({}, 5954, 1000)
    assert first == {"arm": "A", "from_step": 0, "to_step": 120, "stop_tokens": 1006632960}
    assert driver.next_allocation({"A": 120}, 5954, 1000)["arm"] == "B"
    completed = dict.fromkeys("ABCD", 120)
    assert driver.next_allocation(completed, 5954, 1000)["to_step"] == 1120
    completed["A"] = 1120
    assert driver.next_allocation(completed, 5954, 1000) == {
        "arm": "A",
        "from_step": 1120,
        "to_step": 1193,
        "stop_tokens": 1193 * 8388608,
    }
    completed["A"] = 1193
    assert driver.next_allocation(completed, 5954, 1000)["arm"] == "B"
    assert driver.next_allocation(dict.fromkeys("ABCD", 5954), 5954, 1000) is None
    with pytest.raises(ValueError):
        driver.next_allocation({}, 5954, 0)


def test_campaign_submission_is_unambiguous():
    driver = load_probe("drive_campaign")
    record = '{"job_id": "registered", "scheduler_id": "123"}'
    assert driver.submission_record("warnings\n" + record)["scheduler_id"] == "123"
    for content in ("", "{incomplete", record + record):
        with pytest.raises(ValueError, match="ambiguous"):
            driver.submission_record(content)


@pytest.fixture
def formal_run(tmp_path):
    from mor_mlite.pretraining.config import Experiment

    config = Experiment("A", micro_batch_size=2)
    receipt = {
        "experiment": config.to_dict(),
        "source_sha256": "source",
        "data_sha256": "data",
        "environment_sha256": "env",
        "model_config_sha256": "model",
    }
    contract = {
        **receipt,
        "mode": "train",
        "total_steps": 5954,
        "total_tokens": 5954 * config.tokens_per_step,
    }
    plan = {"arm": "A", "from_step": 0, "to_step": 2}

    def write(name, value, jsonl=False):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(
            "\n".join(json.dumps(row) for row in value) + "\n" if jsonl else json.dumps(value)
        )

    write("manifest.json", {**contract, "topology": {"passed": True, "world_size": 64}})
    write("complete.jsonl", [{"contract": contract, "step": 2}], True)
    write(
        "checkpoint-000002/pretraining-state.json",
        {
            "contract": contract,
            "step": 2,
            "cursor": 2 * config.tokens_per_step,
            "scheduler": "pure-token-function-v1",
        },
    )
    updates = [
        {
            "step": step,
            "tokens": step * config.tokens_per_step,
            "lr": config.learning_rate(step * config.tokens_per_step, contract["total_tokens"]),
            "lm_loss": 10.0,
            "depth_aux": 0.0,
            "grad_norm": 1.0,
            "seconds": 20.0,
            "input_sha256": str(step) * 64,
        }
        for step in (1, 2)
    ]
    write("loss.jsonl", updates, True)
    write("evaluation.jsonl", [{"step": s, "nll": 10.0, "targets": 100} for s in (0, 2)], True)
    write("memory.jsonl", [{"ranks": [{"rank": rank} for rank in range(64)]}], True)
    return tmp_path, plan, receipt, updates, write


def test_campaign_audits_real_cursor_schedule_and_artifacts(formal_run):
    root, plan, receipt, _, _ = formal_run
    assert load_probe("drive_campaign").audit_run(root, plan, receipt, 5954) == {
        1: "1" * 64,
        2: "2" * 64,
    }


@pytest.mark.parametrize(
    "damage", ["duplicate", "schedule", "nonfinite", "missing_checkpoint", "missing_eval"]
)
def test_campaign_rejects_incomplete_or_reset_continuations(formal_run, damage):
    root, plan, receipt, updates, write = formal_run
    if damage == "duplicate":
        updates.append(updates[-1])
        write("loss.jsonl", updates, True)
    elif damage == "schedule":
        updates[1]["lr"] = 0.0003
        write("loss.jsonl", updates, True)
    elif damage == "nonfinite":
        updates[1]["grad_norm"] = float("nan")
        write("loss.jsonl", updates, True)
    elif damage == "missing_checkpoint":
        (root / "checkpoint-000002/pretraining-state.json").unlink()
    else:
        write("evaluation.jsonl", [{"step": 0, "nll": 10.0, "targets": 100}], True)
    with pytest.raises((ValueError, FileNotFoundError)):
        load_probe("drive_campaign").audit_run(root, plan, receipt, 5954)


def test_campaign_requires_four_issued_receipts(tmp_path):
    from pathlib import Path

    from mor_mlite.pretraining.config import Experiment
    from mor_mlite.pretraining.data import sha256_file

    driver = load_probe("drive_campaign")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"stored_tokens": 50_000_000_000, "tokens": {"train": 49_949_241_482}})
    )
    checks = dict.fromkeys(
        [
            "causality",
            "initialization",
            "shared_gradients",
            "auxiliary_gradients",
            "data_integrity",
            "checkpoint_resume",
            "gb200_multinode",
        ],
        True,
    )
    for arm in "ABCD":
        receipt = {
            "experiment": Experiment(arm, micro_batch_size=2).to_dict(),
            "source_sha256": "source",
            "data_sha256": sha256_file(manifest),
            "issuer_sha256": sha256_file(Path(driver.__file__).with_name("issue_acceptance.py")),
            "startup_wrapper_sha256": sha256_file(
                Path(driver.__file__).with_name("fused_python.sh")
            ),
            "checks": checks,
            "evidence_sha256": {"gpu-receipt": "digest"},
            "environment_sha256": "env",
            "model_config_sha256": "model",
        }
        (tmp_path / f"{arm.lower()}.json").write_text(json.dumps(receipt))
    receipts, steps = driver.acceptance_contract(tmp_path, manifest, source="source", world_size=64)
    assert set(receipts) == set("ABCD") and steps == 5954
    receipts["D"]["startup_wrapper_sha256"] = "changed"
    (tmp_path / "d.json").write_text(json.dumps(receipts["D"]))
    with pytest.raises(ValueError, match="arm D"):
        driver.acceptance_contract(tmp_path, manifest, source="source", world_size=64)
    receipts["D"]["startup_wrapper_sha256"] = receipts["A"]["startup_wrapper_sha256"]
    receipts["D"]["checks"]["checkpoint_resume"] = False
    (tmp_path / "d.json").write_text(json.dumps(receipts["D"]))
    with pytest.raises(ValueError, match="arm D"):
        driver.acceptance_contract(tmp_path, manifest, source="source", world_size=64)


def test_campaign_requires_explicit_startup_python():
    from pathlib import Path

    driver = load_probe("drive_campaign")
    package = Path(driver.__file__).resolve().parents[2]
    path = "${PROJECT_ROOT}/scripts/pretraining/fused_python.sh"
    assert driver.startup_python_path(path, package, "/remote") == (
        "/remote/scripts/pretraining/fused_python.sh"
    )
    with pytest.raises(ValueError, match="startup wrapper"):
        driver.startup_python_path("/venv/bin/python", package, "/remote")


def test_parallel_campaign_submits_other_arms_while_b_pending():
    driver = load_probe("drive_campaign")
    completed = {"A": 120}
    active = {"B": {"arm": "B", "from_step": 0, "to_step": 120}}
    for expected in "CD":
        plan = driver.available_allocation(completed, active, 5954, 400, 4)
        assert plan["arm"] == expected and plan["to_step"] == 120
        active[expected] = plan
    assert driver.available_allocation(completed, active, 5954, 400, 4) is None
    # C finishes before B and D: it cannot cross the 1B barrier.
    completed["C"] = 120
    del active["C"]
    assert driver.available_allocation(completed, active, 5954, 400, 4) is None
    # Restarting with the same in-flight state does not duplicate a submission.
    assert driver.available_allocation(completed, active, 5954, 400, 4) is None


def test_parallel_campaign_bounded_resume_and_final_completion():
    driver = load_probe("drive_campaign")
    completed = dict.fromkeys("ABCD", 120)
    active = {}
    for arm in "ABCD":
        plan = driver.available_allocation(completed, active, 5954, 400, 4)
        assert (plan["arm"], plan["from_step"], plan["to_step"]) == (arm, 120, 520)
        active[arm] = plan
    assert driver.available_allocation(completed, active, 5954, 400, 4) is None
    completed["C"] = 520
    del active["C"]
    assert driver.available_allocation(completed, active, 5954, 400, 4)["arm"] == "C"
    completed["C"] = 1193
    assert driver.available_allocation(completed, active, 5954, 400, 4) is None
    assert driver.available_allocation(dict.fromkeys("ABCD", 5954), {}, 5954, 400, 4) is None


def test_sequential_policy_remains_default_compatible():
    driver = load_probe("drive_campaign")
    active = {"A": {"arm": "A", "from_step": 0, "to_step": 120}}
    assert driver.available_allocation({}, active, 5954, 400, 1) is None
    assert driver.available_allocation({}, {}, 5954, 400, 1) == driver.next_allocation(
        {}, 5954, 400
    )


@pytest.mark.parametrize("damage", ["cursor", "barrier", "arm", "capacity"])
def test_parallel_campaign_rejects_invalid_active_state(damage):
    driver = load_probe("drive_campaign")
    active = {"B": {"arm": "B", "from_step": 0, "to_step": 120}}
    capacity = 4
    if damage == "cursor":
        active["B"]["from_step"] = 1
    elif damage == "barrier":
        active["B"]["to_step"] = 520
    elif damage == "arm":
        active["B"]["arm"] = "C"
    else:
        capacity = 1
        active["C"] = {"arm": "C", "from_step": 0, "to_step": 120}
    with pytest.raises(ValueError):
        driver.available_allocation({"A": 120}, active, 5954, 400, capacity)


def test_state_migration_preserves_b_id_a_evidence_and_scientific_binding():
    driver = load_probe("drive_campaign")
    state = {
        "binding": {"source": "fixed", "world_size": "64"},
        "status": "watching",
        "completed": {"A": 120},
        "runs": [{"arm": "A"}],
        "active": {"arm": "B", "scheduler_id": "7237775", "from_step": 0, "to_step": 120},
    }
    binding = {**state["binding"], "parallel_arms": "4", "first_stage_time_limit": "1:30:00"}
    upgraded = driver.upgrade_state(state, binding)
    assert upgraded["active"]["B"] == state["active"]
    assert upgraded["completed"] == state["completed"]
    assert upgraded["runs"] == state["runs"]
    assert "schema_version" not in state  # No in-place edit of the archived input.
    assert driver.upgrade_state(upgraded, binding) == upgraded
    with pytest.raises(ValueError, match="binding"):
        driver.upgrade_state(state, {**binding, "source": "changed"})
    for status in ("failed", "submitting"):
        with pytest.raises(ValueError, match="ambiguous"):
            driver.upgrade_state({**state, "status": status}, binding)
    with pytest.raises(ValueError, match="ambiguous"):
        driver.upgrade_state({**state, "intent": {}}, binding)


def test_parallel_launch_uses_same_arm_resume_and_90min_initial_recipe(tmp_path):
    from types import SimpleNamespace

    import yaml

    driver = load_probe("drive_campaign")
    recipe = tmp_path / "recipe.yaml"
    original = {"SLURM": {"segment": 4, "time_limit": "4:00:00"}, "EXPERIMENT": {"seed": 1234}}
    recipe.write_text(yaml.safe_dump(original))
    args = SimpleNamespace(
        world_size=64,
        recipe=recipe,
        state_dir=tmp_path,
        first_stage_time_limit="1:30:00",
        native_root="native",
        python="wrapper",
        data="data",
        hf_config="hf",
        environment="env",
        container_sqsh="sqsh",
    )
    receipt = {"experiment": {"micro_batch_size": 4}}
    plan = {"arm": "C", "from_step": 0, "to_step": 120, "stop_tokens": 120 * 8388608}
    command = driver.submission_command(args, plan, receipt, "receipt-c", "output-c", [])
    assert "--resume" not in command and "--submit" not in command
    assert command[command.index("--mbs") + 1] == "4"
    actual = yaml.safe_load((tmp_path / "first-stage-recipe.yaml").read_text())
    assert actual == {**original, "SLURM": {"segment": 4, "time_limit": "1:30:00"}}
    assert yaml.safe_load(recipe.read_text()) == original
    plan.update(from_step=120, to_step=520, stop_tokens=520 * 8388608)
    runs = [
        {"arm": "C", "to_step": 120, "output": "/c"},
        {"arm": "B", "to_step": 120, "output": "/b"},
    ]
    command = driver.submission_command(args, plan, receipt, "receipt-c", "output-c2", runs)
    assert command[command.index("--resume") + 1] == "/c/checkpoint-000120"
    assert command[command.index("--recipe") + 1] == str(recipe)
    with pytest.raises(ValueError, match="same-arm"):
        driver.submission_command(args, plan, receipt, "receipt-c", "output-c2", runs[1:])


def test_history_migration_reaudits_artifacts_and_rejects_invented_progress(tmp_path, monkeypatch):
    driver = load_probe("drive_campaign")
    calls = []

    def audit(root, plan, receipt, total):
        calls.append((root.name, plan["arm"]))
        return {1: "same-input"}

    monkeypatch.setattr(driver, "audit_run", audit)
    run = {"arm": "A", "from_step": 0, "to_step": 120, "scheduler_id": "a-job"}
    state = {"runs": [run], "completed": {"A": 120}, "active": {"B": {"scheduler_id": "b-job"}}}
    driver.audit_history(state, tmp_path, {"A": {}}, 5954)
    assert calls == [("a-job", "A")]
    with pytest.raises(ValueError, match="audited history"):
        driver.audit_history(
            {**state, "completed": {"A": 120, "C": 120}}, tmp_path, {"A": {}}, 5954
        )
    with pytest.raises(ValueError, match="duplicate"):
        driver.audit_history({**state, "runs": [run, run]}, tmp_path, {"A": {}}, 5954)
    with pytest.raises(ValueError, match="duplicate active"):
        driver.audit_history(
            {**state, "active": {"B": {"scheduler_id": "a-job"}}}, tmp_path, {"A": {}}, 5954
        )


@pytest.mark.parametrize("storage_blocks_a", [False, True])
def test_main_adopts_pending_b_and_submits_c_d_without_waiting(tmp_path, monkeypatch, storage_blocks_a):
    """Exercise the actual event loop with isolated fake SLURM/registry, no GPU submits."""
    import sys
    from types import SimpleNamespace

    import mor_mlite.pretraining.snapshot as snapshot
    import mor_mlite.provenance as provenance
    from mor_mlite.pretraining.data import sha256_file

    driver = load_probe("drive_campaign")
    receipts = {}
    for arm in "ABCD":
        (tmp_path / f"{arm.lower()}.json").write_text(arm)
        receipts[arm] = {
            "experiment": {"micro_batch_size": 4 if arm == "C" else 2},
            "startup_wrapper_sha256": "wrapper",
        }
    values = {
        "project_root": str(tmp_path),
        "recipe": str(tmp_path / "recipe.yaml"),
        "native_root": "native",
        "receipts": str(tmp_path),
        "data_manifest": "data.json",
        "host": "host",
        "remote_root": "/remote",
        "remote_receipts": "/receipts",
        "data": "data",
        "hf_config": "hf",
        "environment": "env",
        "container_sqsh": "sqsh",
        "python": "wrapper",
        "output_prefix": "/outputs/run",
        "world_size": "64",
        "max_updates": "400",
    }
    binding = {
        **values,
        "source": "fixed",
        "bundle": "bundle",
        "execute": "True",
        "receipts": {a: sha256_file(tmp_path / f"{a.lower()}.json") for a in "ABCD"},
    }
    state = {
        "binding": binding,
        "status": "watching",
        "completed": {} if storage_blocks_a else {"A": 120},
        "runs": [],
        "active": {"arm": "B", "from_step": 0, "to_step": 120, "scheduler_id": "b-job"},
    }
    (tmp_path / "state.json").write_text(json.dumps(state))
    argv = ["drive_campaign", "--state-dir", str(tmp_path), "--parallel-arms", "4", "--execute"]
    checked = []
    if storage_blocks_a:
        argv.append("--retain-all-checkpoints")

        def maintain(state, state_dir, total, *, next_plan=None, **kwargs):
            assert state["active"]["B"]["scheduler_id"] == "b-job"
            assert kwargs["retain_all"]
            if next_plan:
                checked.append(next_plan["arm"])
            return {"admit": next_plan is None or next_plan["arm"] != "A"}

        monkeypatch.setitem(sys.modules, "maintain_checkpoints", SimpleNamespace(maintain=maintain))
    for key, value in values.items():
        argv += ["--" + key.replace("_", "-"), value]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(driver, "acceptance_contract", lambda *a, **kw: (receipts, 5954))
    monkeypatch.setattr(driver, "startup_python_path", lambda *a: "/wrapper")
    monkeypatch.setattr(driver, "audit_history", lambda *a: None)
    monkeypatch.setattr(provenance, "source_snapshot", lambda: {"sha256": "fixed"})
    monkeypatch.setattr(snapshot, "freeze", lambda *a, **kw: tmp_path / "bundle")
    records, submitted = {}, []
    monkeypatch.setitem(
        sys.modules,
        "mcore_devtoolkit.cluster_run.registry",
        SimpleNamespace(default_registry=lambda: SimpleNamespace(get=records.get)),
    )

    def run(command, **kwargs):
        if command[0] == "ssh":
            if "sacct" in command[-1]:
                job = command[-1].split("-j ")[1].split()[0]
                return SimpleNamespace(stdout=f"{job}|PENDING|0:0\n")
            arm = "C" if "/receipts/c.json" in command[-1] else "D"
            return SimpleNamespace(stdout=f"{binding['receipts'][arm]} receipt\nwrapper /wrapper\n")
        assert "--submit" in command
        arm = command[command.index("--arm") + 1]
        submitted.append(arm)
        job = str(100 + len(submitted))
        records[job] = SimpleNamespace(job_id=job, scheduler_id=job, command=command)
        kwargs["stdout"].write(json.dumps({"job_id": job, "scheduler_id": job}))
        return SimpleNamespace(returncode=0)

    class PollFinished(BaseException):
        pass

    def stop_poll(_):
        raise PollFinished

    monkeypatch.setattr(driver.subprocess, "run", run)
    monkeypatch.setattr(driver.time, "sleep", stop_poll)
    with pytest.raises(PollFinished):
        driver.main()
    result = json.loads((tmp_path / "state.json").read_text())
    assert submitted == ["C", "D"]
    assert result["active"]["B"]["scheduler_id"] == "b-job"
    assert set(result["active"]) == set("BCD")
    assert result["completed"] == ({} if storage_blocks_a else {"A": 120})
    assert result["status"] == "watching" and "intent" not in result
    if storage_blocks_a:
        assert checked == ["A", "C", "A", "D", "A"]


def test_storage_selection_defers_a_without_changing_cursor_or_stage():
    driver = load_probe("drive_campaign")
    completed = {"A": 1993, "B": 1593, "C": 1593, "D": 1593}
    active = {a: {"arm": a, "from_step": 1593, "to_step": 1993} for a in "BC"}
    plans = driver.available_allocations(completed, active, 5954, 400, 4)
    assert [p["arm"] for p in plans] == ["A", "D"]
    calls = []

    def check(plan):
        calls.append(plan["arm"])
        return plan["arm"] == "D"

    plan = driver.admitted_allocation(plans, check)
    assert calls == ["A", "D"]
    assert plan == {"arm": "D", "from_step": 1593, "to_step": 1993,
                    "stop_tokens": 1993 * 8388608}
    assert driver.admitted_allocation(plans, lambda _: False) is None
    assert driver.admitted_allocation(plans, lambda _: True) == plans[0]
    assert completed == {"A": 1993, "B": 1593, "C": 1593, "D": 1593}
    assert set(active) == set("BC")


def test_storage_selection_does_not_swallow_integrity_errors():
    driver = load_probe("drive_campaign")
    calls = []

    def check(plan):
        calls.append(plan["arm"])
        raise ValueError("damaged retained checkpoint")

    with pytest.raises(ValueError, match="damaged retained"):
        driver.admitted_allocation([{"arm": "A"}, {"arm": "D"}], check)
    assert calls == ["A"]
