"""BF16 reference kernel policy is separate from strict deterministic execution."""

import pytest

from mor_mlite.parity.mlite import _attention_backend
from mor_mlite.parity.topologies import get_topology


@pytest.mark.parametrize("topology", ["baseline", "tp", "ep", "zero1", "tp_dp_ep"])
def test_strict_cp1_uses_native_local_attention(topology):
    assert _attention_backend(get_topology(topology), strict=True) == "local"


@pytest.mark.parametrize("topology", ["cp", "all", "tp_cp_ep", "cp_dp_ep"])
@pytest.mark.parametrize("strict", [True, False])
def test_cp_keeps_magi(topology, strict):
    assert _attention_backend(get_topology(topology), strict=strict) == "magi"


def test_nonstrict_cp1_keeps_existing_flash_policy():
    assert _attention_backend(get_topology("baseline"), strict=False) == "flash"


@pytest.mark.parametrize("field,value", [("strict", True), ("attention_policy", "policy-v1")])
@pytest.mark.parametrize("changed", [False, "different-policy", None])
def test_attention_contract_rejects_changed_or_one_sided_missing_metadata(field, value, changed):
    pytest.importorskip("torch")
    from mor_mlite.parity.compare import _check_metadata_compatibility

    report = _check_metadata_compatibility({field: value}, {field: changed})
    assert not report["passed"]
    assert f"{field} differs between baseline and candidate" in report["details"]


def test_actual_backend_may_differ_under_the_same_policy():
    pytest.importorskip("torch")
    from mor_mlite.parity.compare import _check_metadata_compatibility

    common = {"strict": True, "attention_policy": "policy-v1"}
    assert _check_metadata_compatibility(
        {**common, "attention_backend": "fused"}, {**common, "attention_backend": "magi"}
    )["passed"]
