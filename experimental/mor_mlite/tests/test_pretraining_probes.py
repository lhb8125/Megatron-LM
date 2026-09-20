"""Fault injection for GPU-probe evidence guards; no GPU completion implied."""

import importlib.util
from pathlib import Path

import pytest


def load_probe(name):
    import sys

    path = Path(__file__).resolve().parents[1] / "scripts/pretraining" / f"{name}.py"
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_probe_launcher_full_width_selected_mbs_and_fresh_smoke():
    from argparse import Namespace

    launcher = load_probe("launch_probes")
    args = Namespace(
        kind="indexed",
        arms=["A"],
        world_size=64,
        mbs=2,
        fixture="official",
        data="final",
        python="python",
        output="fresh",
    )
    command = launcher.suite_command(args, "suite.py", "topology.txt")
    assert command[-3:] == ["--full-width", "--mbs", "2"]
    assert command[command.index("--data") + 1] == "final"
    args.kind, args.world_size, args.mbs = "smoke", 32, 1
    args.fixture, args.data = None, None
    command = launcher.suite_command(args, "suite.py", "topology.txt")
    assert command[-2:] == ["--topology", "topology.txt"]
    assert "--fixture" not in command
    args.debugger = "/opt/cuda-gdb"
    command = launcher.suite_command(args, "suite.py", "topology.txt")
    assert command[-2:] == ["--debugger", "/opt/cuda-gdb"]


def test_probe_debugger_preserves_program_arguments_and_child_exit_status():
    suite = load_probe("probe_suite")
    command = ["python", "probe.py", "--arm", "D", "--output", "with spaces"]
    assert suite.diagnostic_command(command) == command
    wrapped = suite.diagnostic_command(command, "/opt/cuda-gdb")
    assert wrapped[0] == "/opt/cuda-gdb"
    assert "--return-child-result" in wrapped
    assert "thread apply all bt 40" in wrapped
    assert wrapped.index("set disable-randomization off") < wrapped.index("run")
    assert "show disable-randomization" in wrapped
    assert wrapped[wrapped.index("--args") + 1 :] == command


def test_probe_attention_environment_is_configured_before_child_start(monkeypatch):
    from mor_mlite.pretraining import runtime

    launcher = load_probe("launch_probes")
    assert launcher.attention_startup_environment() == {
        "NVTE_FLASH_ATTN": "0",
        "NVTE_FUSED_ATTN": "1",
        "NVTE_UNFUSED_ATTN": "0",
    }
    monkeypatch.setattr(runtime, "ATTENTION_BACKEND", "flash")
    with pytest.raises(ValueError, match="attention backend"):
        launcher.attention_startup_environment()


def test_fused_python_startup_preserves_arguments_and_child_status(tmp_path):
    import json
    import os
    import subprocess
    import sys

    wrapper = Path(__file__).resolve().parents[1] / "scripts/pretraining/fused_python.sh"
    interpreter = tmp_path / "runtime/mor-pretraining/gb200-env-nvrx060/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(sys.executable)
    names = ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN")
    env = {key: value for key, value in os.environ.items() if key not in names}
    env["PROJECT_ROOT"] = str(tmp_path)
    result = subprocess.run(
        [
            "sh",
            str(wrapper),
            "-c",
            (
                "import json,os,sys; print(json.dumps([sys.argv[1:],"
                "[os.environ[n] for n in sys.argv[1:4]]]))"
            ),
            *names,
            "literal spaces; $not_expanded",
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == [
        [*names, "literal spaces; $not_expanded"],
        ["0", "1", "0"],
    ]
    command = ["sh", str(wrapper), "-c", "raise SystemExit(7)"]
    assert subprocess.run(command, env=env, check=False).returncode == 7
    assert (
        subprocess.run(command, env={**env, "NVTE_FLASH_ATTN": "1"}, check=False).returncode == 2
    )
    env.pop("PROJECT_ROOT")
    assert subprocess.run(command, env=env, check=False).returncode != 0


@pytest.mark.parametrize(
    "changes",
    [
        {"world_size": 32},
        {"fixture": None},
        {"data": None},
        {"arms": ["A", "A"]},
        {"kind": "initialization", "mbs": 2},
        {"kind": "shared", "world_size": 32, "arms": ["A"]},
    ],
)
def test_probe_launcher_rejects_wrong_acceptance_scope(changes):
    from argparse import Namespace

    args = {
        "kind": "indexed",
        "arms": ["A"],
        "world_size": 64,
        "mbs": 1,
        "fixture": "official",
        "data": "final",
        "python": "python",
        "output": "fresh",
    }
    args.update(changes)
    with pytest.raises(ValueError):
        load_probe("launch_probes").suite_command(Namespace(**args), "suite.py", "topology.txt")


def test_screening_seeds_final_candidate_but_does_not_freeze_it():
    from mor_mlite.pretraining.tuning import Trial

    policy = load_probe("tuning_evidence")
    screened = [
        Trial(1, 235, 0.48, 20, 30, True, True, True),
        Trial(2, 369, 0.77, 20, 30, True, True, True),
    ]
    assert policy.next_candidate(screened, []) == 2
    final = [Trial(2, 370, 0.78, 20, 30, True, True, True)]
    assert policy.next_candidate(screened, final) == 4
    final.append(Trial(4, 380, 0.92, 20, 30, True, True, True))
    assert policy.next_candidate(screened, final) is None
    # A slower final-data candidate cannot freeze a faster pilot-only size.
    final[0] = Trial(2, 100, 0.78, 20, 30, True, True, True)
    assert policy.next_candidate(screened, final) == 1


def test_cuda_oom_rejection_requires_terminal_job_and_exception():
    audit = load_probe("inspect_rejected_trial")
    log = "5: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 128.00 MiB"
    assert audit.cuda_oom_terminal("123|FAILED|1:0\n", "123", log) == "FAILED"
    for accounting in ("123|RUNNING|0:0", "123|CANCELLED|0:15", "123|COMPLETED|0:0", ""):
        with pytest.raises(ValueError):
            audit.cuda_oom_terminal(accounting, "123", log)
    for other in (
        "NCCL watchdog timeout",
        "slurmstepd: host Out Of Memory",
        "CUDA out of memory warning",
    ):
        with pytest.raises(ValueError):
            audit.cuda_oom_terminal("123|FAILED|1:0", "123", other)


def test_rejected_candidate_has_no_fabricated_metrics_and_cannot_be_selected():
    from mor_mlite.pretraining.tuning import Trial, select_mbs

    audit = load_probe("inspect_rejected_trial")
    policy = load_probe("tuning_evidence")
    one = Trial(1, 235, 0.48, 20, 30, True, True, True)
    two = Trial(2, 369, 0.77, 20, 30, True, True, True)
    rejected = audit.RejectedTrial(4)
    assert rejected.eligible is False
    assert not hasattr(rejected, "peak_device_fraction")
    assert not hasattr(rejected, "tokens_per_second")
    assert select_mbs([one, two, rejected]) == 2
    assert policy.next_candidate([one, two], [two, rejected]) is None
    with pytest.raises(ValueError):
        select_mbs([audit.RejectedTrial(1)])


def test_rejected_evidence_rechecks_original_hashes_and_conflicts(tmp_path):
    import json

    from mor_mlite.pretraining.config import Experiment
    from mor_mlite.pretraining.data import sha256_file

    audit = load_probe("inspect_rejected_trial")
    contract = {"mode": "tune", "experiment": Experiment("A", micro_batch_size=4).to_dict()}
    (tmp_path / "manifest.json").write_text(
        json.dumps({**contract, "topology": {"passed": True, "world_size": 64}})
    )
    (tmp_path / "scheduler.txt").write_text("123|FAILED|1:0\n")
    (tmp_path / "stderr.log").write_text("torch.OutOfMemoryError: CUDA out of memory.\n")
    (tmp_path / "submission.json").write_text(
        json.dumps(
            {
                "scheduler_id": "123",
                "job_id": "registered",
                "command_json": json.dumps(
                    [
                        "python",
                        "-m",
                        "mor_mlite.pretraining.train",
                        "--mode",
                        "tune",
                        "--arm",
                        "A",
                        "--mbs",
                        "4",
                        "--world-size",
                        "64",
                    ]
                ),
            }
        )
    )
    report = {
        "outcome": "rejected_cuda_oom",
        "formal_training_tokens": 0,
        "scheduler_id": "123",
        "registry_id": "registered",
        "contract": contract,
        "measured_trial": None,
        "files_sha256": {
            name: sha256_file(tmp_path / name)
            for name in ("manifest.json", "scheduler.txt", "stderr.log", "submission.json")
        },
    }
    (tmp_path / "rejection.json").write_text(json.dumps(report))
    trial, actual, hashes = audit.inspect_attempt(tmp_path)
    assert trial.mbs == 4 and not trial.eligible and actual == contract and hashes is None
    (tmp_path / "complete.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="conflicting"):
        audit.inspect_attempt(tmp_path)
    (tmp_path / "stderr.log").write_text("changed")
    with pytest.raises(ValueError, match="evidence changed"):
        audit.inspect(tmp_path)


def test_screening_does_not_override_final_memory_stop_or_allow_skips():
    from mor_mlite.pretraining.tuning import Trial

    policy = load_probe("tuning_evidence")
    screened = [
        Trial(1, 100, 0.48, 20, 30, True, True, True),
        Trial(2, 200, 0.77, 20, 30, True, True, True),
    ]
    assert policy.next_candidate(screened, [Trial(2, 210, 0.81, 20, 30, True, True, True)]) is None
    with pytest.raises(ValueError):
        policy.next_candidate([], [Trial(2, 200, 0.77, 20, 30, True, True, True)])
    with pytest.raises(ValueError, match="duplicate"):
        policy.combined_trials(screened, [screened[1], screened[1]])


def test_screening_identity_only_relaxes_data_budget_not_scientific_contract():
    from copy import deepcopy

    policy = load_probe("tuning_evidence")
    baseline = {
        "experiment": {"arm": "A", "micro_batch_size": 2, "seq_length": 4096},
        "source_sha256": "source",
        "environment_sha256": "env",
        "mode": "tune",
        "parameters": {"count": 30},
        "data_sha256": "pilot",
        "total_steps": 283,
        "total_tokens": 2373976064,
        "discarded_training_tokens": 1,
    }
    final = {
        **deepcopy(baseline),
        "data_sha256": "full",
        "total_steps": 5954,
        "total_tokens": 49945772032,
        "discarded_training_tokens": 2,
    }
    policy.compatible(baseline, final)
    for key in ("source_sha256", "environment_sha256", "mode", "parameters"):
        with pytest.raises(ValueError, match="screening changed"):
            policy.compatible(baseline, {**final, key: "changed"})
    final["experiment"]["seq_length"] = 8192
    with pytest.raises(ValueError):
        policy.compatible(baseline, final)


def test_indexed_resume_uses_production_host_control():
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(load_probe("indexed_resume_probe")))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert any(ast.unparse(call.func) == "HostControl" for call in calls)
    assert any(ast.unparse(call.func) == "control.gather" for call in calls)
    assert not any(ast.unparse(call.func) == "dist.all_gather_object" for call in calls)
    saves = [call for call in calls if ast.unparse(call.func) == "save"]
    assert len(saves) == 1
    assert any(k.arg == "control" and ast.unparse(k.value) == "control" for k in saves[0].keywords)


@pytest.mark.parametrize("status", ["PENDING", "RUNNING", "CONFIGURING", "COMPLETING"])
def test_tuning_driver_waits_for_scheduler_completion(status):
    driver = load_probe("drive_tuning")
    assert driver.scheduler_result(f"123|{status}|0:0\n123.batch|FAILED|1:0\n", "123") == (
        False,
        status,
    )


def test_tuning_driver_rejects_failures_and_ambiguous_accounting():
    driver = load_probe("drive_tuning")
    assert driver.scheduler_result("123|COMPLETED|0:0\n", "123") == (True, "COMPLETED")
    for status in ("FAILED", "OUT_OF_MEMORY", "TIMEOUT", "CANCELLED by 10", "COMPLETED"):
        with pytest.raises(RuntimeError):
            driver.scheduler_result(f"123|{status}|1:0\n", "123")
    for text in ("", "124|COMPLETED|0:0", "123|RUNNING|0:0\n123|RUNNING|0:0"):
        with pytest.raises(ValueError):
            driver.scheduler_result(text, "123")


def test_tuning_driver_sequential_memory_policy():
    from mor_mlite.pretraining.tuning import Trial

    driver = load_probe("drive_tuning")

    def trial(mbs, fraction):
        return Trial(mbs, 100, fraction, 20, 30, True, True, True)

    assert driver.next_trial({}) == ("A", 1)
    assert driver.next_trial({"A": [trial(1, 0.7)]}) == ("A", 2)
    assert driver.next_trial({"A": [trial(1, 0.81)]}) == ("B", 1)
    assert driver.next_trial({arm: [trial(1, 0.81)] for arm in "ABCD"}) is None
    with pytest.raises(ValueError):
        driver.next_trial({"A": [trial(1, 0.7), trial(4, 0.7)]})
    with pytest.raises(ValueError):
        driver.next_trial({"A": [trial(1, 0.95)]})


def test_tuning_driver_rejects_adopting_formal_training_or_different_output():
    from types import SimpleNamespace

    driver = load_probe("drive_tuning")
    command = [
        "python",
        "-m",
        "mor_mlite.pretraining.train",
        "--arm",
        "A",
        "--mbs",
        "1",
        "--world-size",
        "64",
        "--mode",
        "tune",
        "--output",
        "/pilot",
    ]
    record = SimpleNamespace(command=command, scheduler_id="123")
    arguments = {"arm": "A", "mbs": 1, "world_size": 64, "output": "/pilot"}
    driver.validate_job(record, **arguments)
    with pytest.raises(ValueError):
        driver.validate_job(record, **{**arguments, "output": "/different"})
    command[command.index("tune")] = "train"
    with pytest.raises(ValueError):
        driver.validate_job(record, **arguments)
    with pytest.raises(ValueError):
        driver.validate_job(None, **arguments)


def test_tuning_driver_state_replace_is_readable(tmp_path):
    import json

    driver = load_probe("drive_tuning")
    path = tmp_path / "state.json"
    driver.write_state(path, {"status": "submitting"})
    driver.write_state(path, {"status": "watching", "active": "123"})
    assert json.loads(path.read_text()) == {"status": "watching", "active": "123"}
    assert not path.with_suffix(".tmp").exists()


def test_tuning_driver_audits_cross_arm_inputs_and_source(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace

    from mor_mlite.pretraining.config import Experiment
    from mor_mlite.pretraining.tuning import Trial

    driver = load_probe("drive_tuning")
    jobs = [{"arm": arm, "mbs": 1, "scheduler_id": str(i)} for i, arm in enumerate("AB")]
    contracts = [
        {
            "experiment": Experiment(arm, world_size=64).to_dict(),
            "source_sha256": "source",
            "data_sha256": "data",
            "parameters": {"count": i},
        }
        for i, arm in enumerate("AB")
    ]
    hashes = [["input"] for _ in jobs]

    def inspect(path):
        index = int(path.name)
        return Trial(1, 100, 0.81, 20, 30, True, True, True), contracts[index], hashes[index]

    monkeypatch.setitem(sys.modules, "inspect_trials", SimpleNamespace(inspect=inspect))
    assert set(driver.audit_completed(jobs, tmp_path, world_size=64, source="source")) == {"A", "B"}
    hashes[1] = ["changed"]
    with pytest.raises(ValueError, match="global inputs"):
        driver.audit_completed(jobs, tmp_path, world_size=64, source="source")
    hashes[1] = ["input"]
    contracts[1]["source_sha256"] = "changed"
    with pytest.raises(ValueError, match="model/source/parallelism"):
        driver.audit_completed(jobs, tmp_path, world_size=64, source="source")


def test_driver_keeps_rejected_attempt_but_requires_successful_candidate(monkeypatch, tmp_path):
    import inspect_rejected_trial

    from mor_mlite.pretraining.config import Experiment
    from mor_mlite.pretraining.tuning import Trial

    driver = load_probe("drive_tuning")

    def contract(arm, mbs, data):
        return {
            "experiment": Experiment(arm, micro_batch_size=mbs).to_dict(),
            "source_sha256": "source",
            "data_sha256": data,
            "mode": "tune",
        }

    one = Trial(1, 100, 0.48, 20, 30, True, True, True)
    two = Trial(2, 200, 0.77, 20, 30, True, True, True)
    screening = {
        "A": [
            (one, contract("A", 1, "pilot"), ["pilot"]),
            (two, contract("A", 2, "pilot"), ["pilot"]),
        ]
    }
    jobs = [
        {"arm": a, "mbs": m, "scheduler_id": str(i)}
        for i, (a, m) in enumerate([("A", 2), ("A", 4), ("B", 1)])
    ]
    results = [
        (two, contract("A", 2, "final"), ["input"]),
        (inspect_rejected_trial.RejectedTrial(4), contract("A", 4, "final"), None),
        (Trial(1, 300, 0.81, 20, 30, True, True, True), contract("B", 1, "final"), ["input"]),
    ]
    monkeypatch.setattr(inspect_rejected_trial, "inspect_attempt", lambda p: results[int(p.name)])
    completed = driver.audit_completed(
        jobs, tmp_path, world_size=64, source="source", screening=screening
    )
    assert driver.next_trial(completed, screening) == ("C", 1)
    results[1][1]["data_sha256"] = "other"
    with pytest.raises(ValueError, match="changed data"):
        driver.audit_completed(jobs, tmp_path, world_size=64, source="source", screening=screening)


def test_dist_opt_exact_ownership_dense_and_experts():
    probe = load_probe("auxiliary_probe")
    peers = [
        [("norm", -1, 0, 2, 4), ("experts", 0, 0, 8, 8)],
        [("norm", -1, 2, 4, 4), ("experts", 1, 0, 8, 8)],
    ]
    probe.validate_ownership(peers, [("norm", -1, 4), ("experts", 0, 8), ("experts", 1, 8)])


@pytest.mark.parametrize(
    "peers,required",
    [
        ([[("p", -1, 0, 3, 4)], [("p", -1, 2, 4, 4)]], [("p", -1, 4)]),
        ([[("p", -1, 0, 2, 4)], [("p", -1, 3, 4, 4)]], [("p", -1, 4)]),
        ([[("p", -1, 0, 2, 4)]], [("p", -1, 4)]),
        ([[("p", -1, 0, 4, 4)]], [("p", -1, 4), ("missing", -1, 4)]),
        ([[("experts", 0, 0, 4, 4)]], [("experts", 0, 4), ("experts", 1, 4)]),
    ],
)
def test_dist_opt_ownership_rejects_missing_and_duplicate(peers, required):
    with pytest.raises(AssertionError):
        load_probe("auxiliary_probe").validate_ownership(peers, required)


def test_probe_suite_uses_fresh_processes_and_distinct_ports(monkeypatch):
    suite = load_probe("probe_suite")
    calls = []
    monkeypatch.setenv("SLURM_JOB_ID", "7190000")
    monkeypatch.setattr(
        suite.sys,
        "argv",
        [
            "probe_suite.py",
            "--kind",
            "indexed",
            "--arms",
            "B",
            "D",
            "--fixture",
            "/fixture",
            "--data",
            "/data",
            "--output",
            "/probe",
        ],
    )
    monkeypatch.setattr(suite.subprocess, "run", lambda command, **kw: calls.append((command, kw)))
    suite.main()
    assert len(calls) == 4
    assert all(kw["check"] is True for _, kw in calls)
    assert len({kw["env"]["MASTER_PORT"] for _, kw in calls}) == 4
    assert ["--resume" in command for command, _ in calls] == [False, True, False, True]
    assert calls[1][0][-2:] == ["--resume", "/probe/b"]
    assert calls[3][0][-2:] == ["--resume", "/probe/d"]


def test_probe_suite_stops_on_first_failure(monkeypatch):
    suite = load_probe("probe_suite")
    monkeypatch.setenv("SLURM_JOB_ID", "7190000")
    monkeypatch.setattr(
        suite.sys,
        "argv",
        [
            "probe_suite.py",
            "--kind",
            "auxiliary",
            "--arms",
            "A",
            "B",
            "--fixture",
            "/fixture",
            "--output",
            "/probe",
        ],
    )
    calls = []

    def fail(command, **kwargs):
        calls.append(command)
        raise suite.subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(suite.subprocess, "run", fail)
    with pytest.raises(suite.subprocess.CalledProcessError):
        suite.main()
    assert len(calls) == 1


def test_probe_suite_smokes_keep_arm_topology_and_fresh_processes(monkeypatch):
    suite = load_probe("probe_suite")
    calls = []
    monkeypatch.setenv("SLURM_JOB_ID", "7213156")
    monkeypatch.setattr(
        suite.sys,
        "argv",
        [
            "probe_suite.py",
            "--kind",
            "smoke",
            "--arms",
            "A",
            "B",
            "C",
            "D",
            "--topology",
            "/inventory",
            "--output",
            "/smokes",
        ],
    )
    monkeypatch.setattr(suite.subprocess, "run", lambda cmd, **kw: calls.append((cmd, kw)))
    suite.main()
    assert len(calls) == 4
    assert len({kw["env"]["MASTER_PORT"] for _, kw in calls}) == 4
    for arm, (cmd, kw) in zip("ABCD", calls, strict=True):
        assert cmd[1:3] == ["-m", "mor_mlite.pretraining.smoke"]
        assert cmd[cmd.index("--arm") + 1] == arm
        assert cmd[cmd.index("--topology") + 1] == "/inventory"
        assert cmd[-2:] == ["--output", f"/smokes/{arm.lower()}"]
        assert kw["check"] is True


def test_probe_suite_smokes_require_real_inventory(monkeypatch):
    suite = load_probe("probe_suite")
    monkeypatch.setattr(
        suite.sys,
        "argv",
        ["probe_suite.py", "--kind", "smoke", "--arms", "A", "--output", "/smokes"],
    )
    with pytest.raises(SystemExit):
        suite.main()


def test_unrolled_reference_mapping_has_exact_threefold_middle():
    from collections import Counter

    probe = load_probe("shared_gradient_probe")
    mapping = Counter(probe.shared_name(f"layers.{i}.weight") for i in range(48))
    assert len(mapping) == 20
    assert mapping == {f"layers.{i}.weight": 3 if 3 <= i < 17 else 1 for i in range(20)}
    assert probe.shared_name("embedding.weight") == "embedding.weight"
    with pytest.raises(ValueError):
        probe.shared_name("layers.48.weight")


def test_probe_rendezvous_ports_do_not_overlap_ephemeral_listeners():
    suite = load_probe("probe_suite")
    for job in (7193819, 1, 99999999):
        ports = [suite.rendezvous_port(job, i) for i in range(8)]
        assert len(set(ports)) == 8
        assert all(7000 <= port < 30000 for port in ports)
    assert suite.rendezvous_port(7193819, 3, ephemeral_start=20000) < 20000
    assert suite.rendezvous_port(7193819, 3, ephemeral_start=9000) < 9000
    with pytest.raises(ValueError):
        suite.rendezvous_port(7193819, 0, ephemeral_start=7000)


def trial_fixture(root, *, fraction=0.7):
    import json
    from dataclasses import asdict

    from mor_mlite.pretraining.config import Experiment
    from mor_mlite.pretraining.tuning import Trial

    config = Experiment("A", world_size=32)
    contract = {"experiment": config.to_dict(), "mode": "tune"}

    def write(name, rows):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    write("manifest.json", [contract])
    write(
        "trial.jsonl",
        [
            {
                "trial": asdict(
                    Trial(1, config.tokens_per_step, fraction, 20, 30, True, True, True)
                ),
                "contract": contract,
                "formal_training_tokens": 0,
            }
        ],
    )
    write("complete.jsonl", [{"step": 50, "contract": contract}])
    write(
        "checkpoint-000050/pretraining-state.json",
        [{"step": 50, "cursor": 50 * config.tokens_per_step, "contract": contract}],
    )
    write(
        "loss.jsonl",
        [
            {
                "step": step,
                "tokens": step * config.tokens_per_step,
                "lm_loss": 8.0,
                "depth_aux": 0.0,
                "grad_norm": 1.0,
                "seconds": 1.0,
                "input_sha256": "a" * 64,
            }
            for step in range(1, 51)
        ],
    )
    write("evaluation.jsonl", [{"step": step, "nll": 8.0, "targets": 100} for step in (0, 50)])
    write(
        "memory.jsonl",
        [
            {
                "ranks": [
                    {
                        "rank": rank,
                        "samples": 100,
                        "capacity_bytes": 1000,
                        "whole_device_peak_bytes": fraction * 1000,
                        "peak_device_fraction": fraction,
                        "allocated_peak_bytes": 400,
                        "reserved_peak_bytes": 500,
                    }
                    for rank in range(32)
                ]
            }
        ],
    )
    return write


def test_trial_audit_checks_completed_evidence_not_only_summary(tmp_path):
    audit = load_probe("inspect_trials")
    write = trial_fixture(tmp_path)
    trial, _, hashes = audit.inspect(tmp_path)
    assert trial.eligible and len(hashes) == 50
    write("evaluation.jsonl", [{"step": 0, "nll": 8.0, "targets": 100}])
    with pytest.raises(ValueError, match="validation"):
        audit.inspect(tmp_path)


def test_trial_audit_over_memory_limit_is_recorded_but_ineligible(tmp_path):
    audit = load_probe("inspect_trials")
    trial_fixture(tmp_path, fraction=0.95)
    trial, _, _ = audit.inspect(tmp_path)
    assert not trial.eligible


def test_trial_audit_does_not_freeze_after_only_low_memory_mbs1(tmp_path, monkeypatch, capsys):
    import json

    audit = load_probe("inspect_trials")
    trial_fixture(tmp_path)
    monkeypatch.setattr("sys.argv", ["inspect_trials", "--run", str(tmp_path)])
    audit.main()
    result = json.loads(capsys.readouterr().out)
    assert result["next_mbs"] == 2
    assert result["selected_mbs"] is None
    assert result["sweep_complete"] is False


def test_initialization_audit_rejects_accidentally_cloned_experts():
    audit = load_probe("inspect_smokes")
    initial = [
        {
            "norm.weight": {"numel": 4, "sha256": "common"},
            "layers.0.moe.experts.fc1.weight0": {"numel": 8, "sha256": f"expert-{rank % 16}"},
        }
        for rank in range(32)
    ]
    assert audit.inspect_parameters(initial, 32) == 4 + 16 * 8
    for rank in range(32):
        initial[rank]["layers.0.moe.experts.fc1.weight0"]["sha256"] = "same-expert"
    with pytest.raises(ValueError, match="expert matrices"):
        audit.inspect_parameters(initial, 32)


def test_acceptance_requires_all_arms_and_matching_code():
    issuer = load_probe("issue_acceptance")
    issuer.require_four(dict.fromkeys("ABCD"), "test")
    with pytest.raises(ValueError, match="exactly"):
        issuer.require_four(dict.fromkeys("ABC"), "test")
    issuer.checked_source([{"source": {"sha256": "frozen"}}], "frozen")
    with pytest.raises(ValueError, match="another source"):
        issuer.checked_source([{"source": {"sha256": "old"}}], "frozen")


def test_acceptance_requires_every_rank_receipt(tmp_path):
    import json

    issuer = load_probe("issue_acceptance")
    (tmp_path / "rank-0.json").write_text(json.dumps({"passed": True}))
    with pytest.raises(FileNotFoundError):
        issuer.rank_reports(tmp_path, 32)


def test_acceptance_rejects_pilot_before_issuing_anything(tmp_path):
    issuer = load_probe("issue_acceptance")
    with pytest.raises(ValueError, match="pilot"):
        issuer.check_data_provenance(
            {"provenance": {"purpose": "performance-pilot-only"}}, tmp_path
        )


def test_acceptance_pins_dataset_tokenizer_and_rechecks_tokenizer_bytes(tmp_path):
    import json

    from mor_mlite.pretraining.data import sha256_file
    from mor_mlite.pretraining.prepare_raw import DATASET, REVISION, TOKENIZER, TOKENIZER_REVISION

    issuer = load_probe("issue_acceptance")
    tokenizer_file = tmp_path / "tokenizer.json"
    tokenizer_file.write_text("{}")
    contract = {
        "dataset": DATASET,
        "revision": REVISION,
        "tokenizer": TOKENIZER,
        "tokenizer_revision": TOKENIZER_REVISION,
        "target_tokens": 50_000_000_000,
        "eos": 151643,
        "chat_template": False,
        "sampling": "sorted-train-shards-whole-document-prefix-v1",
    }
    manifest = {
        "provenance": {
            "dataset_source": json.dumps(contract),
            "tokenizer_source": f"{TOKENIZER}@{TOKENIZER_REVISION}",
        },
        "stored_tokens": 50_000_000_001,
        "seed": 1234,
        "split": "sha256(token-content)-mod1000-zero-validation",
        "tokenizer_files": {"tokenizer.json": sha256_file(tokenizer_file)},
    }
    issuer.check_data_provenance(manifest, tmp_path)
    tokenizer_file.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="tokenizer files"):
        issuer.check_data_provenance(manifest, tmp_path)
    contract["revision"] = "unpinned"
    manifest["provenance"]["dataset_source"] = json.dumps(contract)
    with pytest.raises(ValueError, match="pinned provenance"):
        issuer.check_data_provenance(manifest, tmp_path)
