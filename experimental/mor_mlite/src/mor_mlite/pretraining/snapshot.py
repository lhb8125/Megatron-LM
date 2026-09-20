"""Content-addressed job source/config bundles, independent of live worktrees."""

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from mor_mlite.pretraining.data import sha256_file


def inventory(package):
    package = Path(package)
    files = {}
    for relative in ("src/mor_mlite", "configs"):
        tree = package / relative
        if not tree.is_dir():
            raise ValueError(f"missing required package tree: {relative}")
        for path in sorted(tree.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                files[str(path.relative_to(package))] = sha256_file(path)
    if "configs/topologies.json" not in files:
        raise ValueError("package snapshot is missing the imported topology configuration")
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"sha256": digest, "files": files}


def freeze(package, store, *, write=True):
    package, store = Path(package), Path(store)
    manifest = inventory(package)
    target = store / manifest["sha256"]
    if target.exists():
        if inventory(target) != manifest:
            raise ValueError("previously frozen source/config bundle changed")
        if json.loads((target / "snapshot.json").read_text()) != manifest:
            raise ValueError("source bundle completeness marker changed")
        return target
    if not write:
        return target
    store.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="partial-", dir=store))
    for relative in manifest["files"]:
        destination = staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(package / relative, destination)
    if inventory(staging) != manifest:
        raise ValueError("source/config changed while freezing the job bundle")
    (staging / "snapshot.json").write_text(json.dumps(manifest, indent=2) + "\n")
    staging.rename(target)
    return target
