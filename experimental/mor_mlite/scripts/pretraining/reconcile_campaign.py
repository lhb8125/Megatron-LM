"""Explicit, audited reconciliation after a diagnosed terminal failure.

Default is read-only. Never submits jobs, deletes checkpoints, invents a final
evaluation, or treats work after the selected checkpoint as durable progress.
"""

import argparse
import copy
import fcntl
import json
import subprocess
import time
from pathlib import Path


def reconcile(state, statuses, evidence, receipts, total, recovery_steps):
    from drive_campaign import audit_history, audit_record

    from mor_mlite.pretraining.data import sha256_file

    if state["status"] != "failed" or "intent" in state or state.get("schema_version") != 2:
        raise ValueError("reconciliation requires an unambiguous failed schema-v2 campaign")
    audit_history(state, evidence, receipts, total)
    result = copy.deepcopy(state)
    used = set()
    for arm, run in state["active"].items():
        scheduler_state, exit_code = statuses[run["scheduler_id"]]
        if scheduler_state in {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "REQUEUED", "SUSPENDED"}:
            if arm in recovery_steps:
                raise ValueError("cannot recover a live job")
            continue
        if scheduler_state == "COMPLETED" and exit_code == "0:0":
            if arm in recovery_steps:
                raise ValueError("completed job must use full completion audit")
            accepted = copy.deepcopy(run)
        elif scheduler_state in {"FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"} and exit_code != "0:0":
            if arm not in recovery_steps:
                raise ValueError("terminal failure needs an explicit checked recovery point")
            step = recovery_steps[arm]
            report = evidence / run["scheduler_id"] / f"recovery-preflight-{step}.json"
            accepted = {
                **run, "to_step": step, "stop_tokens": step * 8_388_608,
                "recovery": {
                    "original_plan": copy.deepcopy(run), "scheduler_state": scheduler_state,
                    "exit_code": exit_code, "report_sha256": sha256_file(report),
                },
            }
            used.add(arm)
        else:
            raise ValueError(f"unsupported or ambiguous job state: {scheduler_state}, {exit_code}")
        audit_record(evidence / run["scheduler_id"], accepted, receipts[arm], total)
        result["runs"].append(accepted)
        result["completed"][arm] = accepted["to_step"]
        del result["active"][arm]
    if used != set(recovery_steps):
        raise ValueError("unused recovery request")
    # Rechecks continuity and cross-arm global input hashes over accepted prefixes.
    audit_history(result, evidence, receipts, total)
    result["status"] = "watching" if result["active"] else "ready"
    result["reconciled_error"] = result.pop("error", None)
    return result


def main():
    from drive_campaign import acceptance_contract
    from drive_tuning import write_state
    from inspect_recovery import remote_files
    from mcore_devtoolkit.cluster_run.registry import default_registry

    from mor_mlite.pretraining.data import sha256_file
    from mor_mlite.provenance import source_snapshot

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--recover", action="append", default=[], help="ARM:STEP")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    requests = {}
    for value in args.recover:
        arm, step = value.split(":")
        if arm not in "ABCD" or len(arm) != 1 or arm in requests:
            raise ValueError("duplicate/unknown recovery arm")
        requests[arm] = int(step)
    with (args.state_dir / "driver.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = args.state_dir / "state.json"
        state = json.loads(path.read_text())
        binding = state["binding"]
        source = source_snapshot()["sha256"]
        if binding["source"] != source:
            raise ValueError("production source changed")
        receipts, total = acceptance_contract(
            args.receipts, Path(binding["data_manifest"]), source=source,
            world_size=int(binding["world_size"]),
        )
        if binding["receipts"] != {a: sha256_file(args.receipts / f"{a.lower()}.json") for a in "ABCD"}:
            raise ValueError("campaign receipt identity changed")
        registry = default_registry()
        statuses = {}
        for arm, run in state["active"].items():
            record = registry.get(run["registry_id"])
            if record is None or str(record.scheduler_id) != run["scheduler_id"]:
                raise ValueError("recovery job absent from authoritative registry")
            for flag, value in {"--arm": arm, "--output": run["output"], "--mode": "train",
                                "--stop-tokens": str(run["stop_tokens"]),
                                "--world-size": binding["world_size"],
                                "--mbs": str(receipts[arm]["experiment"]["micro_batch_size"])}.items():
                if record.command.count(flag) != 1 or record.command[record.command.index(flag) + 1] != value:
                    raise ValueError("registry command differs from failed allocation")
            job = run["scheduler_id"]
            if not job.isdigit():
                raise ValueError("invalid scheduler ID")
            output = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", binding["host"],
                 f"sacct -n -P -X -j {job} --format=JobID,State,ExitCode"],
                check=True, capture_output=True, text=True, timeout=60,
            ).stdout
            found = [line.split("|") for line in output.splitlines() if line.split("|")[0] == job]
            if len(found) != 1 or len(found[0]) != 3:
                raise ValueError("ambiguous scheduler response")
            statuses[job] = tuple(found[0][1:])
            if arm in requests:
                step = requests[arm]
                report = json.loads((args.state_dir / "evidence" / job / f"recovery-preflight-{step}.json").read_text())
                actual = remote_files(binding["host"], f"{run['output']}/checkpoint-{step:06d}/step_{step}")
                if actual != report["remote_files"]:
                    raise ValueError("remote recovery checkpoint changed since preflight")
        result = reconcile(state, statuses, args.state_dir / "evidence", receipts, total, requests)
        if args.apply:
            with (args.state_dir / f"state-before-recovery-{time.time_ns()}.json").open("x") as stream:
                stream.write(json.dumps(state, indent=2) + "\n")
            write_state(path, result)
        print(json.dumps({"applied": args.apply, "completed": result["completed"],
                          "active": result["active"], "statuses": statuses}, indent=2))


if __name__ == "__main__":
    main()
