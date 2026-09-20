import json

import pytest
from test_pretraining_probes import load_probe


@pytest.fixture
def plan(tmp_path):
    root = tmp_path / "mor-pretraining" / "run"
    checkpoint = root / "checkpoint-000050"
    native = checkpoint / "step_50"
    native.mkdir(parents=True)
    (checkpoint / "pretraining-state.json").write_text(json.dumps({"step": 50}))
    path = native / "__0_0.distcp"
    path.write_bytes(b"test")
    s = path.stat()
    return {"allowed_run_roots": [str(root)], "files": [{"path": str(path), "size": s.st_size,
             "inode": s.st_ino, "mtime_ns": s.st_mtime_ns}]}


def test_prune_only_exact_owned_tensor_and_preserves_marker(plan):
    from pathlib import Path

    module = load_probe("prune_weight_files")
    file = Path(plan["files"][0]["path"])
    assert module.execute(plan)["applied"] is False
    assert file.exists()
    assert module.execute(plan, True)["files"] == 1
    assert not file.exists()
    assert (file.parents[1] / "pretraining-state.json").exists()


@pytest.mark.parametrize("change", ["size", "mtime_ns", "inode", "outside", "marker", "duplicate", "symlink"])
def test_prune_validates_all_before_deleting(plan, change, tmp_path):
    from pathlib import Path

    module = load_probe("prune_weight_files")
    original = Path(plan["files"][0]["path"])
    bad = dict(plan["files"][0])
    if change in {"size", "mtime_ns", "inode"}:
        second = original.parent / "__1_0.distcp"
        second.write_bytes(b"second")
        info = second.stat()
        bad = {"path": str(second), "size": info.st_size, "inode": info.st_ino, "mtime_ns": info.st_mtime_ns}
        bad[change] += 1
    elif change == "outside":
        bad["path"] = str(tmp_path / "other.distcp")
    elif change == "marker":
        bad["path"] = str(original.parents[1] / "pretraining-state.json")
    elif change == "symlink":
        link = original.parent / "__1_0.distcp"
        link.symlink_to(original)
        bad["path"] = str(link)
    plan["files"].append(bad)
    with pytest.raises((ValueError, FileNotFoundError)):
        module.execute(plan, True)
    assert original.exists()
