"""Exact-manifest tensor deletion; metadata, logs and datasets are never targets."""

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path


def validate(plan):
    roots = [Path(p) for p in plan["allowed_run_roots"]]
    if not roots or len(set(roots)) != len(roots):
        raise ValueError("missing or duplicate scoped run roots")
    for root in roots:
        if not root.is_absolute() or root.resolve() != root or len(root.parts) < 5:
            raise ValueError("unsafe run root")
        if root.parent.name != "mor-pretraining":
            raise ValueError("run root is outside this experiment")
    paths = []
    for item in plan["files"]:
        path = Path(item["path"])
        if path in paths or path.resolve() != path:
            raise ValueError("duplicate or symlink target")
        if path.parents[2] not in roots:
            raise ValueError("target escaped approved run root")
        checkpoint = path.parents[1]
        if not re.fullmatch(r"checkpoint-\d{6}", checkpoint.name):
            raise ValueError("not a checkpoint path")
        step = int(checkpoint.name.split("-")[1])
        if path.parent.name != f"step_{step}" or not re.fullmatch(r"__\d+_\d+\.distcp", path.name):
            raise ValueError("only native tensor shards may be deleted")
        marker = checkpoint / "pretraining-state.json"
        if marker.is_symlink() or json.loads(marker.read_text())["step"] != step:
            raise ValueError("checkpoint is incomplete")
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise ValueError("target is not an owned regular file")
        for key, actual in (("size", info.st_size), ("inode", info.st_ino), ("mtime_ns", info.st_mtime_ns)):
            if item[key] != actual:
                raise ValueError(f"target changed after inventory: {path}")
        paths.append(path)
    return paths


def execute(plan, apply=False):
    paths = validate(plan)  # Validate the ENTIRE manifest before the first unlink.
    if apply:
        for path in paths:
            path.unlink()
            print(json.dumps({"deleted": str(path)}), flush=True)
    return {"applied": apply, "files": len(paths), "bytes": sum(f["size"] for f in plan["files"])}


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--approved-sha256", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    raw = sys.stdin.buffer.read()
    if hashlib.sha256(raw).hexdigest() != args.approved_sha256:
        raise ValueError("manifest hash does not match approval")
    print(json.dumps(execute(json.loads(raw), args.apply)), flush=True)


if __name__ == "__main__":
    main()
