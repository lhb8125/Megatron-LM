"""Backport NVIDIA/TransformerEngine#2896's cu12 wheel metadata correction.

TE's Python-agnostic C library wheel filename is py3-none, but releases through
2.13.0 embedded cp310-cp310. This does not retag the ABI-specific torch extension.
Run only inside the explicitly isolated diagnostic environment, never the source.
"""

import argparse
import base64
import csv
import hashlib
import io
import json
from importlib.metadata import distribution
from pathlib import Path


def repair(venv: Path):
    dist = distribution("transformer-engine-cu12")
    root = Path(dist.locate_file("")).resolve()
    if not root.is_relative_to(venv.resolve()) or dist.version != "2.13.0":
        raise RuntimeError("metadata repair is restricted to isolated TE cu12 2.13.0")
    metadata = root / "transformer_engine_cu12-2.13.0.dist-info"
    wheel, record = metadata / "WHEEL", metadata / "RECORD"
    before = wheel.read_bytes()
    old = b"Tag: cp310-cp310-manylinux_2_28_x86_64"
    new = b"Tag: py3-none-manylinux_2_28_x86_64"
    if new in before and old not in before:
        return
    if before.count(old) != 1 or sum(line.startswith(b"Tag:") for line in before.splitlines()) != 1:
        raise RuntimeError("unexpected wheel metadata; refusing to modify its tags")
    after = before.replace(old, new)
    rows = list(csv.reader(io.StringIO(record.read_text())))
    key = str(wheel.relative_to(root))
    matching = [row for row in rows if row[0] == key]
    if len(matching) != 1:
        raise RuntimeError("WHEEL must have exactly one RECORD entry")
    matching[0][1] = "sha256=" + base64.urlsafe_b64encode(
        hashlib.sha256(after).digest()
    ).decode().rstrip("=")
    matching[0][2] = str(len(after))
    payload = io.StringIO()
    csv.writer(payload).writerows(rows)
    provenance = metadata / "mor_wheel_metadata_repair.json"
    provenance.write_text(
        json.dumps(
            {
                "upstream_issue": "https://github.com/NVIDIA/TransformerEngine/issues/2896",
                "package": "transformer-engine-cu12",
                "version": dist.version,
                "old_wheel_sha256": hashlib.sha256(before).hexdigest(),
                "new_wheel_sha256": hashlib.sha256(after).hexdigest(),
                "changed_files": ["WHEEL", "RECORD"],
                "binary_payload_modified": False,
            },
            indent=2,
        )
        + "\n"
    )
    wheel.write_bytes(after)
    record.write_text(payload.getvalue())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--venv", type=Path, required=True)
    repair(parser.parse_args().venv)
