import pytest
from test_pretraining_probes import load_probe


def record(arm, output, step, size=100):
    return {"arm": arm, "path": output + f"/checkpoint-{step:06d}", "step": step,
            "marker": {"step": step}, "files": [{"size": size}]}


def test_disk_budget_counts_active_pending_endpoints_and_headroom():
    module = load_probe("maintain_checkpoints")
    rows = [record("A", "/old-a", 120), record("B", "/old-b", 120, 50),
            record("A", "/new-a", 200)]
    runs = [{"arm": "A", "output": "/new-a", "from_step": 120, "to_step": 520},
            {"arm": "B", "output": "/new-b", "from_step": 120, "to_step": 520}]
    # A: 300/400/500/520, B pending: 200/300/400/500/520; +10%, margin.
    assert module.write_budget(rows, runs, margin=10) == 10 + int(4 * 100 * 1.1) + int(5 * 50 * 1.1)


def test_disk_budget_includes_interior_scientific_milestone():
    module = load_probe("maintain_checkpoints")
    rows = [record("A", "/old", 2000)]
    run = {"arm": "A", "output": "/new", "from_step": 2000, "to_step": 2400}
    assert module.write_budget(rows, [run], [2385], margin=0) == int(5 * 100 * 1.1)


def test_disk_budget_does_not_assume_missing_or_partial_checkpoints_complete():
    module = load_probe("maintain_checkpoints")
    run = {"arm": "A", "output": "/new", "from_step": 200, "to_step": 300}
    with pytest.raises(ValueError, match="no measured"):
        module.write_budget([], [run])
    partial = record("A", "/new", 300)
    partial["marker"] = None
    assert module.write_budget([record("A", "/old", 200), partial], [run], margin=0) == 110
