"""Read-only formal checkpoint retention candidates. No deletion capability.

Keeps scientific milestones, two newest complete-marker checkpoints per arm,
active jobs' restore sources, and every incomplete checkpoint. A candidate plan
is not authorization to delete and is not a native checkpoint-load certificate.
"""

import argparse
import json
import shlex
from pathlib import Path


def select_candidates(records, milestone_steps, protected_paths):
    paths = [r["path"] for r in records]
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate checkpoint inventory")
    kept = set(protected_paths)
    for arm in "ABCD":
        complete = [r for r in records if r["arm"] == arm and r["complete"]]
        # Preserve all copies at either of the latest two step numbers.
        latest = set(sorted({r["step"] for r in complete}, reverse=True)[:2])
        kept.update(r["path"] for r in complete if r["step"] in latest)
    candidates = []
    for row in records:
        if row["arm"] not in "ABCD" or len(row["arm"]) != 1 or row["step"] < 0:
            raise ValueError("invalid checkpoint identity")
        if not row["complete"] or row["step"] in milestone_steps:
            kept.add(row["path"])
        if row["path"] not in kept:
            candidates.append(row)
    return candidates, sorted(kept)


def inventory_command(host, runs):
    # Explicit campaign output paths only; never recurse across a shared root.
    code = """import json,sys
from pathlib import Path
runs=json.loads(sys.argv[1]); records=[]
for run in runs:
 root=Path(run['output'])
 if root.is_symlink() or not root.is_absolute(): raise ValueError('unsafe run root')
 for p in sorted(root.glob('checkpoint-*')):
  if p.is_symlink() or not p.is_dir(): raise ValueError('unsafe checkpoint entry')
  step=int(p.name.removeprefix('checkpoint-'))
  native=p/f'step_{step}'
  if native.is_symlink(): raise ValueError('unsafe native directory')
  marker=p/'pretraining-state.json'
  if marker.is_symlink(): raise ValueError('unsafe checkpoint marker')
  state=json.loads(marker.read_text()) if marker.is_file() else None
  files=[]
  for f in sorted(native.glob('*.distcp')):
   if f.is_symlink() or not f.is_file(): raise ValueError('unsafe shard')
   s=f.stat()
   files.append({'path':str(f),'size':s.st_size,'inode':s.st_ino,'mtime_ns':s.st_mtime_ns})
  records.append({'arm':run['arm'],'path':str(p),'step':step,'marker':state,'files':files,
                  'metadata_present':(native/'.metadata').is_file()})
print(json.dumps(records))
"""
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host,
            "python3 -c " + shlex.quote(code) + " " + shlex.quote(json.dumps(runs))]


def remote_inventory(host, runs):
    # Only this read-only operation may be repeated. Never reuse this retry
    # around pruning or sbatch: a timeout there has ambiguous side effects.
    from readonly_probe import run_readonly

    return json.loads(run_readonly(inventory_command(host, runs)).stdout)


def build_plan(state, records, total_steps):
    from mor_mlite.pretraining.config import milestones

    binding = state["binding"]
    known = {r["output"]: r["arm"] for r in state["runs"] + list(state["active"].values())}
    protected = set()
    for active in state["active"].values():
        if active["from_step"]:
            prior = [r for r in state["runs"] if r["arm"] == active["arm"] and r["to_step"] == active["from_step"]]
            if len(prior) != 1:
                raise ValueError("active restore source is ambiguous")
            protected.add(f"{prior[0]['output']}/checkpoint-{active['from_step']:06d}")
    checked = []
    milestone_steps = set(milestones(total_steps, 8_388_608))
    for row in records:
        path = Path(row["path"])
        if (
            str(path.parent) not in known or known[str(path.parent)] != row["arm"]
            or path.name != f"checkpoint-{row['step']:06d}"
        ):
            raise ValueError("inventory escaped explicit campaign outputs")
        marker = row["marker"]
        if marker and (row["step"] in milestone_steps or row["path"] in protected) and not row["files"]:
            raise ValueError("protected checkpoint tensor files are missing")
        if marker:
            contract = marker["contract"]
            if (
                marker["step"] != row["step"] or marker["cursor"] != row["step"] * 8_388_608
                or marker["scheduler"] != "pure-token-function-v1"
                or contract["mode"] != "train" or contract["experiment"]["arm"] != row["arm"]
                or contract["source_sha256"] != binding["source"]
                or contract["total_steps"] != total_steps
            ):
                raise ValueError("checkpoint contract/cursor mismatch")
        for f in row["files"]:
            p = Path(f["path"])
            if p.parent != path / f"step_{row['step']}" or p.suffix != ".distcp" or f["size"] < 0:
                raise ValueError("candidate is not an explicitly inventoried tensor shard")
        checked.append({**row, "complete": bool(marker and row["metadata_present"] and row["files"])
                        and all(f["size"] > 0 for f in row["files"])})
    candidates, kept = select_candidates(checked, milestone_steps, protected)
    return {
        "policy": "milestones-and-two-latest-complete-plus-active-resume-v1",
        "read_only": True, "deletion_authorized": False, "native_integrity_verified": False,
        "kept_paths": kept,
        "candidates": [{"checkpoint": r["path"], "arm": r["arm"], "step": r["step"], "files": r["files"]}
                       for r in candidates],
        "candidate_bytes": sum(f["size"] for r in candidates for f in r["files"]),
    }


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    state = json.loads(args.state.read_text())
    if "intent" in state or state.get("schema_version") != 2:
        raise ValueError("ambiguous campaign state")
    unique = {r["output"]: r for r in state["runs"] + list(state["active"].values())}
    records = remote_inventory(state["binding"]["host"], list(unique.values()))
    manifest = json.loads(Path(state["binding"]["data_manifest"]).read_text())
    report = build_plan(state, records, manifest["tokens"]["train"] // 8_388_608)
    with args.report.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"read_only": True, "candidate_bytes": report["candidate_bytes"],
                      "candidate_checkpoints": len(report["candidates"]), "report": str(args.report)}))


if __name__ == "__main__":
    main()
