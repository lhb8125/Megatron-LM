"""The isolated TE metadata repair must not touch its C/Python ABI binaries."""

import csv
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_path = Path(__file__).resolve().parents[1] / "scripts/eos/repair_te_wheel_tag.py"
_spec = importlib.util.spec_from_file_location("repair_te_tag", _path)
repair_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(repair_module)


def test_repairs_known_tag_and_record_without_changing_binary(tmp_path, monkeypatch):
    site = tmp_path / "site-packages"
    info = site / "transformer_engine_cu12-2.13.0.dist-info"
    info.mkdir(parents=True)
    wheel = info / "WHEEL"
    wheel.write_text("Wheel-Version: 1.0\nTag: cp310-cp310-manylinux_2_28_x86_64\n")
    (info / "RECORD").write_text(f"{wheel.relative_to(site)},,\n")
    binary = site / "libtransformer_engine.so"
    binary.write_bytes(b"unchanged C library payload")
    monkeypatch.setattr(
        repair_module,
        "distribution",
        lambda _name: SimpleNamespace(
            version="2.13.0",
            locate_file=lambda _path: site,
        ),
    )
    repair_module.repair(tmp_path)
    assert "Tag: py3-none-manylinux_2_28_x86_64" in wheel.read_text()
    assert binary.read_bytes() == b"unchanged C library payload"
    rows = list(csv.reader((info / "RECORD").read_text().splitlines()))
    assert rows[0][1].startswith("sha256=")
    assert int(rows[0][2]) == len(wheel.read_bytes())
    repair_module.repair(tmp_path)


def test_metadata_repair_rejects_source_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(
        repair_module,
        "distribution",
        lambda _name: SimpleNamespace(
            version="2.13.0",
            locate_file=lambda _path: tmp_path / "source",
        ),
    )
    with pytest.raises(RuntimeError, match="restricted"):
        repair_module.repair(tmp_path / "isolated")
