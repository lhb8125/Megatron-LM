"""Sequential MBS workflow on one fixed dataset; never submit formal training.

One owned SLURM job at a time. Submission ambiguity, failed jobs or incomplete
evidence stop progression. State and logs remain readable without this process.
"""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def write_state(path, state):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def scheduler_result(output, job_id):
    records = [line.split("|") for line in output.splitlines() if line.split("|")[0] == job_id]
    if len(records) != 1 or len(records[0]) < 3:
        raise ValueError("missing or ambiguous SLURM accounting record")
    _, state, code, *_ = records[0]
    state = state.split()[0]
    if state in {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "REQUEUED", "SUSPENDED"}:
        return False, state
    if state == "COMPLETED" and code == "0:0":
        return True, state
    raise RuntimeError(f"SLURM job {job_id} stopped: {state}, exit={code}")


def next_trial(completed, screening=None):
    from tuning_evidence import next_candidate

    for arm in "ABCD":
        trials = completed.get(arm, [])
        screened = [row[0] for row in (screening or {}).get(arm, [])]
        following = next_candidate(screened, trials)
        if following is not None:
            return arm, following
    return None


def audit_completed(jobs, evidence, *, world_size, source, screening=None):
    from inspect_rejected_trial import inspect_attempt as inspect
    from tuning_evidence import check_screening_contract

    from mor_mlite.pretraining.config import Experiment

    completed = {}
    common = None
    common_hashes = None
    for job in jobs:
        trial, contract, hashes = inspect(evidence / job["scheduler_id"])
        if (job["arm"], job["mbs"]) != next_trial(completed, screening):
            raise ValueError("completed jobs do not follow the sequential MBS policy")
        if (
            trial.mbs != job["mbs"]
            or contract["experiment"]
            != Experiment(job["arm"], world_size=world_size, micro_batch_size=trial.mbs).to_dict()
            or contract["source_sha256"] != source
        ):
            raise ValueError("trial does not match the intended model/source/parallelism")
        check_screening_contract((screening or {}).get(job["arm"], []), contract)
        identity = {k: v for k, v in contract.items() if k not in {"experiment", "parameters"}}
        if common is not None and identity != common:
            raise ValueError("MBS/arm trials changed data, environment, schedule or global inputs")
        common = identity
        if hashes is not None:
            if common_hashes is not None and hashes != common_hashes:
                raise ValueError("MBS/arm trials changed global inputs")
            common_hashes = hashes
        completed.setdefault(job["arm"], []).append(trial)
    return completed


def validate_job(record, *, arm, mbs, world_size, output):
    if record is None:
        raise ValueError("job is missing from the registry")
    command = record.command
    for flag, value in {
        "--arm": arm,
        "--mbs": str(mbs),
        "--world-size": str(world_size),
        "--mode": "tune",
        "--output": output,
    }.items():
        if command.count(flag) != 1 or command[command.index(flag) + 1] != value:
            raise ValueError(f"registry job differs from intended tuning job: {flag}")
    if "mor_mlite.pretraining.train" not in command or not str(record.scheduler_id).isdigit():
        raise ValueError("not a registered SLURM pretraining job")


def main():
    from mcore_devtoolkit.cluster_run.registry import default_registry
    from tuning_evidence import combined_trials, load_screening

    from mor_mlite.pretraining.snapshot import freeze
    from mor_mlite.pretraining.tuning import select_mbs
    from mor_mlite.provenance import source_snapshot

    parser = argparse.ArgumentParser(allow_abbrev=False)
    for name in ("state-dir", "project-root", "recipe", "native-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in (
        "host",
        "remote-root",
        "data",
        "hf-config",
        "environment",
        "container-sqsh",
        "python",
        "output-prefix",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--world-size", type=int, choices=[32, 64], default=64)
    parser.add_argument(
        "--adopt-job", required=True, help="Existing registry ID for first candidate"
    )
    parser.add_argument("--screening-run", action="append", type=Path, default=[])
    parser.add_argument(
        "--reconcile-rejected", help="Explicit terminal job ID with sealed CUDA-OOM evidence"
    )
    parser.add_argument(
        "--adopt-submitted", help="Reconcile a separately submitted registry ID with saved intent"
    )
    args = parser.parse_args()
    if args.reconcile_rejected and args.adopt_submitted:
        parser.error("only one explicit reconciliation may be requested")
    package = Path(__file__).resolve().parents[2]
    args.state_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.state_dir / "driver.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = args.state_dir / "state.json"
    source = source_snapshot()["sha256"]
    screening = load_screening(args.screening_run, world_size=args.world_size, source=source)
    bundle = freeze(
        package, args.project_root / "runtime/mor-pretraining/sources", write=False
    ).name
    contract = {
        **{
            k: str(v)
            for k, v in vars(args).items()
            if k not in {"state_dir", "reconcile_rejected", "adopt_submitted"}
        },
        "source": source,
        "bundle": bundle,
    }
    registry = default_registry()

    def output_for(arm, mbs):
        return f"{args.output_prefix}-{arm.lower()}{args.world_size}-mbs{mbs}-v1"

    def job_info(record, arm, mbs):
        output = output_for(arm, mbs)
        validate_job(
            record,
            arm=arm,
            mbs=mbs,
            world_size=args.world_size,
            output=output.replace("${PROJECT_ROOT}", args.remote_root),
        )
        return {
            "registry_id": record.job_id,
            "scheduler_id": record.scheduler_id,
            "arm": arm,
            "mbs": mbs,
            "output": output,
            "log_path": record.log_path,
        }

    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["contract"] != contract:
            raise ValueError("driver source/config changed; explicit investigation required")
        if args.adopt_submitted:
            if state["status"] not in {"failed", "submitting"} or state.get("active"):
                raise ValueError("adoption requires an unresolved submission, not an active job")
            completed = audit_completed(
                state["completed"],
                args.state_dir / "evidence",
                world_size=args.world_size,
                source=source,
                screening=screening,
            )
            arm, mbs = next_trial(completed, screening)
            expected = {"arm": arm, "mbs": mbs, "output": output_for(arm, mbs)}
            if state.get("intent") != expected:
                raise ValueError("saved submission intent differs from the audited next candidate")
            adopted = job_info(registry.get(args.adopt_submitted), arm, mbs)
            archived = args.state_dir / f"failed-submit-{arm.lower()}-mbs{mbs}.json"
            if not archived.exists():
                write_state(archived, state)
            state.update(status="watching", active=adopted)
            state.pop("intent")
            state.pop("error", None)
            write_state(state_path, state)
        if args.reconcile_rejected:
            from inspect_rejected_trial import inspect as inspect_rejection

            active = state.get("active")
            if (
                state["status"] != "failed"
                or not active
                or active["scheduler_id"] != args.reconcile_rejected
            ):
                raise ValueError("reconciliation does not identify the stopped active attempt")
            root = args.state_dir / "evidence" / active["scheduler_id"]
            rejected, _, _ = inspect_rejection(root)
            if rejected.mbs != active["mbs"]:
                raise ValueError("rejection differs from the stopped MBS")
            audit_completed(
                state["completed"] + [active],
                args.state_dir / "evidence",
                world_size=args.world_size,
                source=source,
                screening=screening,
            )
            archived = args.state_dir / f"failed-{active['scheduler_id']}.json"
            if not archived.exists():
                write_state(archived, state)
            active["outcome"] = "rejected_cuda_oom"
            state.update(status="watching")
            state.pop("error", None)
            write_state(state_path, state)
        if state["status"] in {"submitting", "failed"}:
            raise ValueError("previous failure/submission ambiguity requires manual reconciliation")
    else:
        if args.reconcile_rejected or args.adopt_submitted:
            raise ValueError("no stopped driver to reconcile")
        state = {
            "contract": contract,
            "status": "watching",
            "active": job_info(registry.get(args.adopt_job), *next_trial({}, screening)),
            "completed": [],
        }
        write_state(state_path, state)
    print(
        json.dumps(
            {"driver_pid": os.getpid(), "state": str(state_path), "status": state["status"]}
        ),
        flush=True,
    )
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", args.host]
    previous = None
    failures = 0
    try:
        while state["status"] != "complete":
            if (
                source_snapshot()["sha256"] != source
                or freeze(
                    package, args.project_root / "runtime/mor-pretraining/sources", write=False
                ).name
                != bundle
            ):
                raise ValueError("live source/config changed; no further jobs will be submitted")
            active = state["active"]
            job_id = active["scheduler_id"]
            try:
                result = subprocess.run(
                    ssh + [f"sacct -n -P -j {job_id} --format=JobID,State,ExitCode"],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=60,
                )
                if active.get("outcome") == "rejected_cuda_oom":
                    from inspect_rejected_trial import cuda_oom_terminal

                    cuda_oom_terminal(
                        result.stdout,
                        job_id,
                        (args.state_dir / "evidence" / job_id / "stderr.log").read_text(),
                    )
                    done, status = True, "REJECTED_CUDA_OOM"
                else:
                    done, status = scheduler_result(result.stdout, job_id)
                failures = 0
            except (subprocess.SubprocessError, ValueError) as exc:
                failures += 1
                if failures >= 5:
                    raise
                print(json.dumps({"read_retry": failures, "error": str(exc)}), flush=True)
                time.sleep(30)
                continue
            if (job_id, status) != previous:
                print(
                    json.dumps(
                        {
                            "job": job_id,
                            "arm": active["arm"],
                            "mbs": active["mbs"],
                            "status": status,
                        }
                    ),
                    flush=True,
                )
                previous = job_id, status
            if not done:
                time.sleep(30)
                continue
            destination = args.state_dir / "evidence" / job_id
            destination.mkdir(parents=True, exist_ok=True)
            remote = active["output"].replace("${PROJECT_ROOT}", args.remote_root)
            subprocess.run(
                [
                    "rsync",
                    "-a",
                    "-e",
                    "ssh -o BatchMode=yes -o ConnectTimeout=15",
                    "--include=/checkpoint-000050/",
                    "--include=/checkpoint-000050/pretraining-state.json",
                    "--include=/*.json",
                    "--include=/*.jsonl",
                    "--exclude=*",
                    f"{args.host}:{remote}/",
                    str(destination) + "/",
                ],
                check=True,
                timeout=180,
            )
            candidates = state["completed"] + [active]
            completed = audit_completed(
                candidates,
                args.state_dir / "evidence",
                world_size=args.world_size,
                source=source,
                screening=screening,
            )
            following = next_trial(completed, screening)
            state.update(completed=candidates, active=None)
            state["best_so_far"] = {
                arm: select_mbs(combined_trials([row[0] for row in screening.get(arm, [])], trials))
                for arm, trials in completed.items()
            }
            print(
                json.dumps(
                    {"audited": job_id, "next": following, "best_so_far": state["best_so_far"]}
                ),
                flush=True,
            )
            if following is None:
                state["status"] = "complete"
                state["scope"] = "MBS tuning only; not formal acceptance or formal training"
                write_state(state_path, state)
                break
            arm, mbs = following
            # Persist intent BEFORE a potentially successful sbatch call. Never retry
            # an ambiguous submission automatically, including after a process crash.
            state.update(
                status="submitting", intent={"arm": arm, "mbs": mbs, "output": output_for(arm, mbs)}
            )
            write_state(state_path, state)
            command = [
                sys.executable,
                "-m",
                "mor_mlite.pretraining.launch",
                "--arm",
                arm,
                "--mbs",
                str(mbs),
                "--world-size",
                str(args.world_size),
                "--mode",
                "tune",
            ]
            for name in (
                "recipe",
                "native_root",
                "python",
                "data",
                "hf_config",
                "environment",
                "container_sqsh",
            ):
                command += ["--" + name.replace("_", "-"), str(getattr(args, name))]
            command += ["--output", output_for(arm, mbs), "--submit"]
            log = args.state_dir / f"submit-{arm.lower()}-mbs{mbs}.log"
            with log.open("x") as file:
                subprocess.run(
                    command,
                    cwd=args.project_root,
                    stdout=file,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=600,
                )
            # The launcher emits a final public registry record after transport logs.
            content = log.read_text()
            records = []
            for offset, char in enumerate(content):
                if char == "{":
                    try:
                        value, _ = json.JSONDecoder().raw_decode(content[offset:])
                    except ValueError:
                        continue
                    if isinstance(value, dict) and "job_id" in value and "scheduler_id" in value:
                        records.append(value)
            if len(records) != 1:
                raise ValueError("submission result ambiguous; inspect registry, do not resubmit")
            state.update(
                status="watching", active=job_info(registry.get(records[0]["job_id"]), arm, mbs)
            )
            state.pop("intent")
            write_state(state_path, state)
    except Exception as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_state(state_path, state)
        raise


if __name__ == "__main__":
    main()
