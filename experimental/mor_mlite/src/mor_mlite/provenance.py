"""Content identity of the imported package, including uncommitted source changes."""

from __future__ import annotations

import hashlib
from pathlib import Path


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_snapshot() -> dict:
    root = Path(__file__).resolve().parent
    files = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*.py"))
    }
    digest = hashlib.sha256()
    for name, value in files.items():
        digest.update(name.encode() + b"\0" + value.encode() + b"\n")
    return {"algorithm": "sha256-python-tree-v1", "sha256": digest.hexdigest(), "files": files}
