"""Reconcile proven pre-registry sync or pre-auth SSH failure, never sbatch."""

import argparse
import copy
import fcntl
import hashlib
import json
import shlex
import sqlite3
import subprocess
import time
from pathlib import Path


def validate_failure(state, log, local_matches, remote_matches, output_exists):
    from drive_campaign import available_allocation, submission_record

    if state["status"] != "failed" or "intent" not in state:
        raise ValueError("not a stopped submission")
    intent = state["intent"]
    binding = state["binding"]
    manifest = json.loads(Path(binding["data_manifest"]).read_text())
    plan = available_allocation(state["completed"], state["active"],
                                manifest["tokens"]["train"] // 8_388_608,
                                int(binding["max_updates"]), int(binding["parallel_arms"]))
    expected_output = (f"{binding['output_prefix']}-{plan['arm'].lower()}-"
                       f"{plan['from_step']:06d}-{plan['to_step']:06d}").replace("${PROJECT_ROOT}", binding["remote_root"])
    if intent != {**plan, "output": expected_output}:
        raise ValueError("failed intent differs from audited next allocation")
    pre_registry = (
        "in launch_remote_slurm" in log and "sync_with_retry(" in log
        and "RemoteSyncError: sync_project" in log
        and '"scheduler_id"' not in log and '"job_id"' not in log and not local_matches
    )
    preauth = False
    if len(local_matches) == 1 and isinstance(local_matches[0], dict):
        row = local_matches[0]
        try:
            submitted = submission_record(log)
            metadata = json.loads(row["metadata_json"])
            preauth = (
                submitted["job_id"] == row["job_id"] and row["status"] == "failed"
                and row["scheduler_id"] is None
                and not metadata.get("remote_job_id") and not metadata.get("remote_scheduler_id")
                and metadata.get("launch_error") == (
                    "remote command exited with status 255\n"
                    "ssh_exchange_identification: Connection closed by remote host\n"
                    "No JSON envelope found in remote output:\n"
                )
            )
        except (KeyError, ValueError, TypeError):
            preauth = False
    if not (pre_registry or preauth) or remote_matches or output_exists:
        raise ValueError("not a proven pre-registry sync failure; no retry allowed")
    return plan


def main():
    from drive_campaign import acceptance_contract, audit_history
    from drive_tuning import write_state

    from mor_mlite.pretraining.data import sha256_file
    from mor_mlite.provenance import source_snapshot

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    with (args.state_dir / "driver.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = args.state_dir / "state.json"
        state = json.loads(path.read_text())
        binding = state["binding"]
        if source_snapshot()["sha256"] != binding["source"]:
            raise ValueError("production source changed")
        receipts, total = acceptance_contract(args.receipts, Path(binding["data_manifest"]),
                                             source=binding["source"], world_size=int(binding["world_size"]))
        if binding["receipts"] != {a: sha256_file(args.receipts / f"{a.lower()}.json") for a in "ABCD"}:
            raise ValueError("receipt identities changed")
        audit_history(state, args.state_dir / "evidence", receipts, total)
        intent = state["intent"]
        output = intent["output"]
        needle = Path(output).name
        db = Path(binding["project_root"]) / "runtime/job_launch/jobs.db"
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as connection:
            connection.row_factory = sqlite3.Row
            local = [dict(row) for row in connection.execute(
                "SELECT job_id,status,scheduler_id,metadata_json FROM jobs WHERE instr(command_json, ?) > 0", (needle,))]
        code = """import json,sqlite3,sys
from pathlib import Path
with sqlite3.connect('file:'+sys.argv[1]+'?mode=ro',uri=True) as c:
 rows=c.execute('SELECT job_id FROM jobs WHERE instr(command_json, ?) > 0',(sys.argv[2],)).fetchall()
print(json.dumps({'matches':rows,'output_exists':Path(sys.argv[3]).exists()}))
"""
        remote = json.loads(subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", binding["host"],
             "python3 -c " + shlex.quote(code) + " " + shlex.join([
                 binding["remote_root"] + "/runtime/job_launch/jobs.db", needle, output])],
            check=True, capture_output=True, text=True, timeout=60,
        ).stdout)
        log = args.state_dir / f"submit-{intent['arm'].lower()}-{intent['from_step']:06d}.log"
        validate_failure(state, log.read_text(), local, remote["matches"], remote["output_exists"])
        result = copy.deepcopy(state)
        stamp = time.time_ns()
        archive_log = log.with_name(log.stem + f"-presubmit-failed-{stamp}.log")
        result.setdefault("presubmit_recoveries", []).append({
            "intent": result.pop("intent"), "error": result.pop("error"),
            "local_matches": local, "remote_matches": remote["matches"],
            "output_exists": remote["output_exists"], "log": str(archive_log),
            "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
        })
        result["status"] = "watching" if result["active"] else "ready"
        if args.apply:
            with (args.state_dir / f"state-before-presubmit-recovery-{stamp}.json").open("x") as stream:
                stream.write(json.dumps(state, indent=2) + "\n")
            log.rename(archive_log)
            write_state(path, result)
        print(json.dumps({"applied": args.apply,
                          "failure_kind": "preauth_ssh" if local else "pre_registry_sync",
                          "local_matches": local, "remote": remote, "retry_plan": intent}))


if __name__ == "__main__":
    main()
