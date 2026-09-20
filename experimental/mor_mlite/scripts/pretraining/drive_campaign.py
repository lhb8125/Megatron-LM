"""Bounded formal campaign, requiring issued receipts before any submission.

All arms reach 1B before any reaches 10B. Long stages are split into bounded
allocations, restoring the actual checkpoint and keeping the full token horizon.
Submission ambiguity or failed evidence stops the driver, never auto-resubmits.
"""

import argparse
import copy
import fcntl
import json
import math
import shlex
import subprocess
import sys
import time
from pathlib import Path


def next_allocation(completed, total_steps, max_updates):
    from mor_mlite.pretraining.campaign import next_stage

    if max_updates < 1:
        raise ValueError("max updates must be positive")
    stage = next_stage(completed, total_steps=total_steps)
    if stage is None:
        return None
    end = min(stage["to_step"], stage["from_step"] + max_updates)
    return {**stage, "to_step": end, "stop_tokens": end * 8_388_608}


def available_allocation(completed, active, total_steps, max_updates, parallel_arms):
    plans = available_allocations(completed, active, total_steps, max_updates, parallel_arms)
    return plans[0] if plans else None


def available_allocations(completed, active, total_steps, max_updates, parallel_arms):
    """Pending jobs occupy a slot, but never count as completed at a stage barrier."""
    from mor_mlite.pretraining.campaign import next_stage

    if parallel_arms not in (1, 4) or max_updates < 1:
        raise ValueError("invalid campaign concurrency/update bound")
    stage = next_stage(completed, total_steps=total_steps)
    if set(active) - set("ABCD") or len(active) > parallel_arms:
        raise ValueError("invalid active-arm set")
    for arm, run in active.items():
        if (
            run["arm"] != arm
            or run["from_step"] != completed.get(arm, 0)
            or stage is None
            or not run["from_step"] < run["to_step"] <= stage["to_step"]
        ):
            raise ValueError("active allocation violates cursor/stage barrier")
    if stage is None or len(active) == parallel_arms:
        return []
    plans = []
    for arm in "ABCD":
        start = completed.get(arm, 0)
        if arm not in active and start < stage["to_step"]:
            end = min(stage["to_step"], start + max_updates)
            plans.append({
                "arm": arm,
                "from_step": start,
                "to_step": end,
                "stop_tokens": end * 8_388_608,
            })
    return plans


def admitted_allocation(plans, check):
    """A storage denial defers only that arm; integrity errors still propagate."""
    for plan in plans:
        if check(plan):
            return plan
    return None


def upgrade_state(state, binding):
    """Explicit scheduling-only migration, retaining jobs and all science bindings."""
    if state["status"] not in {"ready", "watching", "complete"} or "intent" in state:
        raise ValueError("ambiguous previous failure; manual reconciliation required")
    scheduling = {"parallel_arms", "first_stage_time_limit", "prune_approved", "retain_all_checkpoints", "prune_on_pressure"}
    if {k: v for k, v in state["binding"].items() if k not in scheduling} != {
        k: v for k, v in binding.items() if k not in scheduling
    }:
        raise ValueError("changed campaign scientific/launch binding")
    result = copy.deepcopy(state)
    if result.get("schema_version", 1) == 1:
        current = result["active"]
        result["active"] = {current["arm"]: current} if current else {}
    elif result["schema_version"] != 2:
        raise ValueError("unsupported campaign state schema")
    result.update(schema_version=2, binding=binding)
    return result


def recover_inventory_timeout(state):
    """Explicit recovery only of the exact read-only inventory transport failure."""
    from plan_checkpoint_retention import inventory_command

    if (state.get("status") != "failed" or "intent" in state
            or state.get("schema_version") != 2 or not state.get("active")):
        raise ValueError("not a recoverable inventory timeout")
    runs = {r["output"]: r for r in state["runs"] + list(state["active"].values())}
    command = inventory_command(state["binding"]["host"], list(runs.values()))
    expected = {f"TimeoutExpired: {subprocess.TimeoutExpired(command, t)}" for t in (60, 120, 180)}
    expected.add(f"CalledProcessError: {subprocess.CalledProcessError(255, command)}")
    if state.get("error") not in expected:
        raise ValueError("failure was not the exact read-only inventory operation")
    result = copy.deepcopy(state)
    result.setdefault("inventory_timeout_recoveries", []).append(result.pop("error"))
    result["status"] = "watching"
    return result


def recover_checkpoint_read(state, native):
    """Recheck the exact failed read; no submission/deletion failure is eligible."""
    from inspect_recovery import remote_files, remote_files_command

    if (state.get("status") != "failed" or "intent" in state
            or state.get("schema_version") != 2 or not state.get("active")):
        raise ValueError("not a recoverable checkpoint read")
    path = Path(native)
    runs = {r["output"] for r in state["runs"] + list(state["active"].values())}
    if str(path.parent.parent) not in runs:
        raise ValueError("checkpoint read escaped campaign outputs")
    try:
        step = int(path.name.removeprefix("step_"))
    except ValueError as exc:
        raise ValueError("invalid native checkpoint path") from exc
    if step < 0 or path.name != f"step_{step}" or path.parent.name != f"checkpoint-{step:06d}":
        raise ValueError("invalid native checkpoint path")
    command = remote_files_command(state["binding"]["host"], native)
    expected = {f"CalledProcessError: {subprocess.CalledProcessError(255, command)}"}
    expected.update(f"TimeoutExpired: {subprocess.TimeoutExpired(command, t)}" for t in (60, 120, 180))
    if state.get("error") not in expected:
        raise ValueError("failure was not the exact checkpoint read operation")
    remote_files(state["binding"]["host"], native)  # Must succeed before state recovery.
    result = copy.deepcopy(state)
    result.setdefault("checkpoint_read_recoveries", []).append(result.pop("error"))
    result["status"] = "watching"
    return result


def evidence_command(host, active, state_dir):
    """Download only completed-run evidence; never upload or delete remote files."""
    job = active["scheduler_id"]
    if not str(job).isdigit() or not Path(active["output"]).is_absolute():
        raise ValueError("invalid evidence source")
    checkpoint = f"checkpoint-{active['to_step']:06d}"
    destination = Path(state_dir) / "evidence" / str(job)
    return [
        "rsync", "-a", f"--include=/{checkpoint}/",
        f"--include=/{checkpoint}/pretraining-state.json",
        "--include=/*.json", "--include=/*.jsonl", "--exclude=*",
        f"{host}:{active['output']}/", str(destination) + "/",
    ]


def fetch_evidence(host, active, state_dir):
    """Idempotent local mirror transfer; validation remains outside the retry."""
    command = evidence_command(host, active, state_dir)
    destination = Path(state_dir) / "evidence" / str(active["scheduler_id"])
    destination.mkdir(parents=True, exist_ok=True)
    for attempt, timeout in enumerate((180, 240, 300)):
        try:
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=timeout)
            return destination
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
            transient = isinstance(exc, subprocess.TimeoutExpired) or exc.returncode == 255
            if not transient or attempt == 2:
                raise
            print(json.dumps({"evidence_transfer_retry": attempt + 2,
                              "job": active["scheduler_id"],
                              "error_type": type(exc).__name__}), flush=True)
            time.sleep(5 * (attempt + 1))


def recover_evidence_transfer(state, state_dir, job):
    """Resume only an exact failed evidence download, preserving all job cursors."""
    if (state.get("status") != "failed" or "intent" in state
            or state.get("schema_version") != 2):
        raise ValueError("not a recoverable evidence transfer")
    matches = [r for r in state["active"].values() if r["scheduler_id"] == job]
    if len(matches) != 1:
        raise ValueError("evidence job is not uniquely active")
    active = matches[0]
    command = evidence_command(state["binding"]["host"], active, state_dir)
    expected = {f"CalledProcessError: {subprocess.CalledProcessError(255, command)}"}
    expected.update(f"TimeoutExpired: {subprocess.TimeoutExpired(command, t)}"
                    for t in (180, 240, 300))
    if state.get("error") not in expected:
        raise ValueError("failure was not the exact evidence download")
    fetch_evidence(state["binding"]["host"], active, state_dir)
    result = copy.deepcopy(state)
    result.setdefault("evidence_transfer_recoveries", []).append(result.pop("error"))
    result["status"] = "watching"
    return result


def submission_command(args, plan, receipt, acceptance, output, runs):
    """Build one arm's command; only that arm's audited checkpoint can be resumed."""
    command = [
        sys.executable,
        "-m",
        "mor_mlite.pretraining.launch",
        "--arm",
        plan["arm"],
        "--mbs",
        str(receipt["experiment"]["micro_batch_size"]),
        "--world-size",
        str(args.world_size),
        "--mode",
        "train",
        "--acceptance",
        acceptance,
        "--output",
        output,
        "--stop-tokens",
        str(plan["stop_tokens"]),
    ]
    recipe = args.recipe
    if args.first_stage_time_limit and plan["from_step"] == 0:
        import yaml

        content = yaml.safe_load(recipe.read_text())
        content["SLURM"]["time_limit"] = args.first_stage_time_limit
        recipe = args.state_dir / "first-stage-recipe.yaml"
        rendered = yaml.safe_dump(content, sort_keys=False)
        if recipe.exists():
            if recipe.read_text() != rendered:
                raise ValueError("first-stage recipe changed")
        else:
            with recipe.open("x") as stream:
                stream.write(rendered)
    command += ["--recipe", str(recipe)]
    for name in ("native_root", "python", "data", "hf_config", "environment", "container_sqsh"):
        command += ["--" + name.replace("_", "-"), str(getattr(args, name))]
    if plan["from_step"]:
        prior = [run for run in runs if run["arm"] == plan["arm"]]
        if not prior or prior[-1]["to_step"] != plan["from_step"]:
            raise ValueError("missing same-arm audited checkpoint")
        command += ["--resume", f"{prior[-1]['output']}/checkpoint-{plan['from_step']:06d}"]
    return command


def acceptance_contract(root, data_manifest, *, source, world_size):
    from mor_mlite.pretraining.config import Experiment, validate_data_purpose
    from mor_mlite.pretraining.data import sha256_file

    manifest = json.loads(data_manifest.read_text())
    validate_data_purpose(manifest, "train")
    if manifest["stored_tokens"] < 50_000_000_000:
        raise ValueError("formal campaign requires the completed 50B corpus")
    data_sha = sha256_file(data_manifest)
    issuer_sha = sha256_file(Path(__file__).with_name("issue_acceptance.py"))
    startup_sha = sha256_file(Path(__file__).with_name("fused_python.sh"))
    required = {
        "causality",
        "initialization",
        "shared_gradients",
        "auxiliary_gradients",
        "data_integrity",
        "checkpoint_resume",
        "gb200_multinode",
    }
    receipts = {}
    common = None
    for arm in "ABCD":
        receipt = json.loads((root / f"{arm.lower()}.json").read_text())
        config = Experiment(
            arm, world_size=world_size, micro_batch_size=receipt["experiment"]["micro_batch_size"]
        )
        if (
            receipt["source_sha256"] != source
            or receipt["data_sha256"] != data_sha
            or receipt["experiment"] != config.to_dict()
            or receipt.get("issuer_sha256") != issuer_sha
            or receipt.get("startup_wrapper_sha256") != startup_sha
            or not receipt.get("evidence_sha256")
            or any(receipt.get("checks", {}).get(key) is not True for key in required)
        ):
            raise ValueError(f"missing/incompatible issued acceptance for arm {arm}")
        identity = receipt["environment_sha256"], receipt["model_config_sha256"]
        if common is not None and identity != common:
            raise ValueError("arm receipts changed environment or model configuration")
        common = identity
        receipts[arm] = receipt
    total_steps = manifest["tokens"]["train"] // 8_388_608
    return receipts, total_steps


def startup_python_path(python, project_root, remote_root):
    relative = (
        Path(__file__)
        .with_name("fused_python.sh")
        .resolve()
        .relative_to(Path(project_root).resolve())
    )
    expected = str(Path(remote_root) / relative)
    if python.replace("${PROJECT_ROOT}", str(remote_root)) != expected:
        raise ValueError("formal campaign requires the verified fused Python startup wrapper")
    return expected


def audit_run(root, plan, receipt, total_steps):
    from inspect_trials import rows, single

    from mor_mlite.pretraining.config import Experiment, milestones

    contract = json.loads((root / "manifest.json").read_text())
    experiment = receipt["experiment"]
    config = Experiment(**experiment)
    if (
        contract["mode"] != "train"
        or contract["experiment"] != experiment
        or contract["total_steps"] != total_steps
        or contract["total_tokens"] != total_steps * config.tokens_per_step
        or any(
            contract[key] != receipt[key]
            for key in ("source_sha256", "data_sha256", "environment_sha256", "model_config_sha256")
        )
        or not contract["topology"]["passed"]
    ):
        raise ValueError("formal run changed the accepted experiment")
    scientific = {key: value for key, value in contract.items() if key != "topology"}
    done = single(root / "complete.jsonl")
    checkpoint = json.loads(
        (root / f"checkpoint-{plan['to_step']:06d}/pretraining-state.json").read_text()
    )
    if (
        done["contract"] != scientific
        or done["step"] != plan["to_step"]
        or checkpoint["contract"] != scientific
        or checkpoint["step"] != plan["to_step"]
        or checkpoint["cursor"] != plan["to_step"] * config.tokens_per_step
        or checkpoint["scheduler"] != "pure-token-function-v1"
    ):
        raise ValueError("formal completion/checkpoint cursor mismatch")
    updates = rows(root / "loss.jsonl")
    if [row["step"] for row in updates] != list(range(plan["from_step"] + 1, plan["to_step"] + 1)):
        raise ValueError("formal continuation duplicated or omitted updates")
    for row in updates:
        if (
            row["tokens"] != row["step"] * config.tokens_per_step
            or not math.isclose(
                row["lr"],
                config.learning_rate(row["tokens"], scientific["total_tokens"]),
                rel_tol=1e-12,
            )
            or any(
                not math.isfinite(row[key])
                for key in ("lm_loss", "depth_aux", "grad_norm", "seconds")
            )
            or row["seconds"] <= 0
            or len(bytes.fromhex(row["input_sha256"])) != 32
        ):
            raise ValueError("invalid formal update/token schedule")
    expected = {
        s
        for s in milestones(total_steps, config.tokens_per_step)
        if plan["from_step"] < s <= plan["to_step"]
    } | {plan["to_step"]}
    if plan["from_step"] == 0:
        expected.add(0)
    evaluations = rows(root / "evaluation.jsonl")
    if [row["step"] for row in evaluations] != sorted(expected) or any(
        not math.isfinite(row["nll"]) or row["targets"] <= 0 for row in evaluations
    ):
        raise ValueError("formal milestone evaluation missing/invalid")
    memory = single(root / "memory.jsonl")["ranks"]
    if sorted(row["rank"] for row in memory) != list(range(config.world_size)):
        raise ValueError("formal memory evidence is incomplete")
    return {row["step"]: row["input_sha256"] for row in updates}


def submission_record(content):
    found = []
    for offset, char in enumerate(content):
        if char != "{":
            continue
        try:
            value, _ = json.JSONDecoder().raw_decode(content[offset:])
        except ValueError:
            continue
        if isinstance(value, dict) and "job_id" in value and "scheduler_id" in value:
            found.append(value)
    if len(found) != 1:
        raise ValueError("ambiguous submission; reconcile registry before restarting")
    return found[0]


def audit_record(root, run, receipt, total):
    if "recovery" not in run:
        return audit_run(root, run, receipt, total)
    from inspect_recovery import inspect

    from mor_mlite.pretraining.data import sha256_file

    recovery = run["recovery"]
    original = recovery["original_plan"]
    path = root / f"recovery-preflight-{run['to_step']}.json"
    if (
        sha256_file(path) != recovery["report_sha256"]
        or recovery["scheduler_state"] not in {"FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"}
        or recovery["exit_code"] == "0:0"
        or any(run[k] != original[k] for k in ("arm", "from_step", "output", "registry_id", "scheduler_id"))
        or run["stop_tokens"] != run["to_step"] * 8_388_608
    ):
        raise ValueError("invalid failed-run recovery lineage")
    report = json.loads(path.read_text())
    checked = inspect(root, original, receipt, run["to_step"], report["remote_files"])
    if checked != {k: v for k, v in report.items() if k != "remote_files"}:
        raise ValueError("recovery evidence changed")
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["total_steps"] != total:
        raise ValueError("recovery changed full token horizon")
    return {int(k): v for k, v in checked["input_hashes"].items()}


def audit_history(state, evidence, receipts, total):
    completed, inputs, jobs = {}, {}, set()
    for run in state["runs"]:
        arm, job = run["arm"], run["scheduler_id"]
        if job in jobs or run["from_step"] != completed.get(arm, 0):
            raise ValueError("duplicate or discontinuous campaign history")
        hashes = audit_record(evidence / job, run, receipts[arm], total)
        for step, digest in hashes.items():
            if step in inputs and inputs[step] != digest:
                raise ValueError("formal arms consumed different global inputs")
            inputs[step] = digest
        completed[arm] = run["to_step"]
        jobs.add(job)
    if completed != state["completed"]:
        raise ValueError("completed cursor is not supported by audited history")
    for run in state["active"].values():
        if run["scheduler_id"] in jobs:
            raise ValueError("duplicate active scheduler job")
        jobs.add(run["scheduler_id"])


def main():
    from drive_tuning import scheduler_result, write_state
    from readonly_probe import run_readonly
    from mcore_devtoolkit.cluster_run.registry import default_registry

    from mor_mlite.pretraining.data import sha256_file
    from mor_mlite.pretraining.snapshot import freeze
    from mor_mlite.provenance import source_snapshot

    parser = argparse.ArgumentParser(allow_abbrev=False)
    for name in ("state-dir", "project-root", "recipe", "native-root", "receipts", "data-manifest"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in (
        "host",
        "remote-root",
        "remote-receipts",
        "data",
        "hf-config",
        "environment",
        "container-sqsh",
        "python",
        "output-prefix",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--world-size", type=int, choices=[32, 64], default=64)
    parser.add_argument("--max-updates", type=int, default=1000)
    parser.add_argument("--parallel-arms", type=int, choices=[1, 4], default=1)
    parser.add_argument("--first-stage-time-limit", choices=["1:30:00"])
    parser.add_argument("--prune-approved", action="store_true",
                        help="User authorized milestone/latest-two tensor retention")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--retain-all-checkpoints", action="store_true",
                        help="Disable tensor deletion while retaining storage admission checks")
    parser.add_argument("--prune-on-pressure", action="store_true",
                        help="Only prune when a fresh storage budget fails; requires --prune-approved")
    parser.add_argument("--recover-inventory-timeout", "--recover-inventory-read", action="store_true",
                        help="Reaudit and resume an exact read-only inventory transport failure; never a submission/deletion failure")
    parser.add_argument("--recover-checkpoint-read", metavar="NATIVE_PATH",
                        help="Recheck and reaudit an exact failed read-only checkpoint probe")
    parser.add_argument("--recover-evidence-transfer", metavar="JOB_ID",
                        help="Refetch an exact failed evidence download; recheck scheduler and audit before completion")
    parser.add_argument("--recover-prune-preauth", metavar="MANIFEST_NAME",
                        help="Reconcile a proven pre-auth prune failure without deleting; retain original jobs")
    args = parser.parse_args()
    if args.prune_on_pressure and (not args.prune_approved or args.retain_all_checkpoints):
        parser.error("--prune-on-pressure requires --prune-approved and forbids --retain-all-checkpoints")
    source = source_snapshot()["sha256"]
    receipts, total = acceptance_contract(
        args.receipts, args.data_manifest, source=source, world_size=args.world_size
    )
    startup_python = startup_python_path(args.python, args.project_root, args.remote_root)
    if not args.execute:
        print(
            json.dumps(
                {
                    "next": available_allocation(
                        {}, {}, total, args.max_updates, args.parallel_arms
                    ),
                    "parallel_arms": args.parallel_arms,
                    "selected_mbs": {
                        a: r["experiment"]["micro_batch_size"] for a, r in receipts.items()
                    },
                }
            )
        )
        return
    package = Path(__file__).resolve().parents[2]
    bundle = freeze(
        package, args.project_root / "runtime/mor-pretraining/sources", write=False
    ).name
    binding = {
        **{k: str(v) for k, v in vars(args).items()
           if k not in {"state_dir", "recover_inventory_timeout", "recover_checkpoint_read",
                        "recover_evidence_transfer", "recover_prune_preauth"}},
        "source": source,
        "bundle": bundle,
        "receipts": {a: sha256_file(args.receipts / f"{a.lower()}.json") for a in "ABCD"},
    }
    args.state_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.state_dir / "driver.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = args.state_dir / "state.json"
    state = (
        json.loads(path.read_text())
        if path.exists()
        else {
            "schema_version": 2,
            "binding": binding,
            "status": "ready",
            "active": {},
            "completed": {},
            "runs": [],
        }
    )
    previous_state = state
    if sum(bool(v) for v in (args.recover_inventory_timeout, args.recover_checkpoint_read,
                            args.recover_evidence_transfer, args.recover_prune_preauth)) > 1:
        raise ValueError("choose only one exact failure recovery")
    if args.recover_prune_preauth:
        from maintain_checkpoints import recover_prune_preauth

        state = recover_prune_preauth(state, args.state_dir, args.recover_prune_preauth, total)
    if args.recover_evidence_transfer:
        state = recover_evidence_transfer(state, args.state_dir, args.recover_evidence_transfer)
    if args.recover_checkpoint_read:
        state = recover_checkpoint_read(state, args.recover_checkpoint_read)
    if args.recover_inventory_timeout:
        state = recover_inventory_timeout(state)
    state = upgrade_state(state, binding)
    available_allocation(
        state["completed"], state["active"], total, args.max_updates, args.parallel_arms
    )
    audit_history(state, args.state_dir / "evidence", receipts, total)
    if state != previous_state:
        archive = args.state_dir / f"state-before-upgrade-{time.time_ns()}.json"
        with archive.open("x") as stream:
            stream.write(json.dumps(previous_state, indent=2) + "\n")
        write_state(path, state)
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", args.host]
    registry = default_registry()
    failures = {}
    maintained_at = 0.0
    try:
        while state["status"] != "complete":
            if (
                source_snapshot()["sha256"] != source
                or freeze(
                    package, args.project_root / "runtime/mor-pretraining/sources", write=False
                ).name
                != bundle
            ):
                raise ValueError("production source/config changed during campaign")
            poll_failed = False
            for active_arm, active in list(state["active"].items()):
                job = active["scheduler_id"]
                try:
                    result = subprocess.run(
                        ssh + [f"sacct -n -P -j {job} --format=JobID,State,ExitCode"],
                        text=True,
                        capture_output=True,
                        check=True,
                        timeout=60,
                    )
                    done, _ = scheduler_result(result.stdout, job)
                    failures[job] = 0
                except (subprocess.SubprocessError, ValueError):
                    failures[job] = failures.get(job, 0) + 1
                    if failures[job] >= 5:
                        raise
                    poll_failed = True
                    break
                if not done:
                    continue
                destination = fetch_evidence(args.host, active, args.state_dir)
                hashes = audit_run(destination, active, receipts[active["arm"]], total)
                for previous in state["runs"]:
                    if previous["arm"] == active["arm"]:
                        continue
                    previous_hashes = audit_record(
                        args.state_dir / "evidence" / previous["scheduler_id"],
                        previous, receipts[previous["arm"]], total,
                    )
                    for step, digest in previous_hashes.items():
                        if step in hashes and hashes[step] != digest:
                            raise ValueError("formal arms consumed different global inputs")
                state["completed"][active["arm"]] = active["to_step"]
                state["runs"].append(active)
                del state["active"][active_arm]
                state["status"] = "watching" if state["active"] else "ready"
                write_state(path, state)
                print(json.dumps({"audited": job, "completed": state["completed"]}), flush=True)
            if poll_failed:
                time.sleep(30)
                continue
            if (args.prune_approved or args.retain_all_checkpoints) and time.monotonic() - maintained_at >= 300:
                from maintain_checkpoints import maintain

                storage = maintain(state, args.state_dir, total, retain_all=args.retain_all_checkpoints,
                                   prune_on_pressure=args.prune_on_pressure)
                maintained_at = time.monotonic()
                print(json.dumps({"retention": storage}), flush=True)
            plans = available_allocations(
                state["completed"], state["active"], total, args.max_updates, args.parallel_arms
            )
            if not plans:
                if state["active"]:
                    time.sleep(30)
                    continue
                state["status"] = "complete"
                write_state(path, state)
                break
            def output_path(candidate):
                return (
                    f"{args.output_prefix}-{candidate['arm'].lower()}-"
                    f"{candidate['from_step']:06d}-{candidate['to_step']:06d}"
                ).replace("${PROJECT_ROOT}", args.remote_root)

            plan = plans[0]
            if args.prune_approved or args.retain_all_checkpoints:
                from maintain_checkpoints import maintain

                def storage_admits(candidate):
                    nonlocal maintained_at
                    storage = maintain(
                        state, args.state_dir, total,
                        next_plan={**candidate, "output": output_path(candidate)},
                        retain_all=args.retain_all_checkpoints,
                        prune_on_pressure=args.prune_on_pressure,
                    )
                    maintained_at = time.monotonic()
                    if not storage["admit"]:
                        print(json.dumps({"waiting_for_storage": storage, "plan": candidate}), flush=True)
                    return storage["admit"]

                plan = admitted_allocation(plans, storage_admits)
                if plan is None:
                    time.sleep(30)
                    continue
            arm = plan["arm"]
            output = output_path(plan)
            acceptance = f"{args.remote_receipts}/{arm.lower()}.json".replace(
                "${PROJECT_ROOT}", args.remote_root
            )
            remote_hashes = run_readonly(
                ssh
                + [
                    "test -x "
                    + shlex.quote(startup_python)
                    + " && sha256sum "
                    + shlex.quote(acceptance)
                    + " "
                    + shlex.quote(startup_python)
                ],
            ).stdout.splitlines()
            if (
                len(remote_hashes) != 2
                or remote_hashes[0].split()[0] != binding["receipts"][arm]
                or remote_hashes[1].split()[0] != receipts[arm]["startup_wrapper_sha256"]
            ):
                raise ValueError("remote acceptance/startup wrapper differs from verified evidence")
            command = submission_command(
                args, plan, receipts[arm], acceptance, output, state["runs"]
            )
            command.append("--submit")
            state.update(status="submitting", intent={**plan, "output": output})
            write_state(path, state)
            log = args.state_dir / f"submit-{arm.lower()}-{plan['from_step']:06d}.log"
            with log.open("x") as stream:
                subprocess.run(
                    command,
                    cwd=args.project_root,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=1200,
                )
            submitted = submission_record(log.read_text())
            record = registry.get(submitted["job_id"])
            if record is None or not str(record.scheduler_id).isdigit():
                raise ValueError("submission lacks an authoritative registry record")
            for flag, value in {
                "--mode": "train",
                "--arm": arm,
                "--output": output,
                "--acceptance": acceptance,
                "--stop-tokens": str(plan["stop_tokens"]),
                "--mbs": str(receipts[arm]["experiment"]["micro_batch_size"]),
                "--world-size": str(args.world_size),
            }.items():
                if (
                    record.command.count(flag) != 1
                    or record.command[record.command.index(flag) + 1] != value
                ):
                    raise ValueError("submitted formal job differs from intended plan")
            state["active"][arm] = {
                **plan,
                "output": output,
                "registry_id": record.job_id,
                "scheduler_id": record.scheduler_id,
            }
            state["status"] = "watching"
            state.pop("intent")
            write_state(path, state)
            print(json.dumps({"submitted": state["active"][arm]}), flush=True)
    except Exception as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_state(path, state)
        raise


if __name__ == "__main__":
    main()
