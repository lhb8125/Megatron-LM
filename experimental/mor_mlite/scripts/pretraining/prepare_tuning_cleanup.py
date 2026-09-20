"""Prepare the explicitly approved completed-tuning cleanup; does not delete."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    from inspect_trials import inspect
    from plan_checkpoint_retention import remote_inventory

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    runs, contracts = [], {}
    host = remote_root = None
    for name in ("pilot-driver-fix1", "finaldata-driver-v2"):
        state = json.loads((args.runtime / name / "state.json").read_text())
        binding = state["contract"]
        host, remote_root = binding["host"], binding["remote_root"]
        for run in state["completed"]:
            if run.get("outcome") == "rejected_cuda_oom":
                continue
            evidence = args.runtime / name / "evidence" / run["scheduler_id"]
            inspect(evidence)
            run = {**run, "output": run["output"].replace("${PROJECT_ROOT}", remote_root)}
            contracts[run["output"]] = json.loads((evidence / "checkpoint-000050/pretraining-state.json").read_text())
            runs.append(run)
    if len(runs) != 10 or len(contracts) != 10:
        raise ValueError("cleanup differs from the approved ten completed tuning runs")
    jobs = ",".join(r["scheduler_id"] for r in runs)
    if any(not r["scheduler_id"].isdigit() for r in runs):
        raise ValueError("unsafe job ID")
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host,
         f"sacct -n -P -X -j {jobs} --format=JobID,State,ExitCode"],
        check=True, capture_output=True, text=True, timeout=60,
    )
    states = {row.split("|")[0]: row.split("|")[1:] for row in result.stdout.splitlines() if row}
    if any(states.get(r["scheduler_id"]) != ["COMPLETED", "0:0"] for r in runs):
        raise ValueError("tuning job not terminal-successful")
    records = remote_inventory(host, runs)
    files = []
    for row in records:
        if row["step"] != 50 or row["marker"] != contracts[str(Path(row["path"]).parent)]:
            raise ValueError("remote tuning checkpoint changed")
        if len(row["files"]) != 16 or not row["metadata_present"]:
            raise ValueError("incomplete tuning checkpoint")
        files.extend(row["files"])
    if len(files) != 160:
        raise ValueError("unexpected cleanup file count")
    plan = {"scope": "user-approved-ten-completed-tuning-weight-checkpoints",
            "host": host, "allowed_run_roots": [r["output"] for r in runs],
            "files": files, "scheduler_states": states}
    with args.report.open("x") as stream:
        stream.write(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({"files": len(files), "bytes": sum(f["size"] for f in files), "report": str(args.report)}))


if __name__ == "__main__":
    main()
