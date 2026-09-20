"""Cross-topology correctness artifacts and comparisons."""

from .topologies import TOPOLOGY_MATRIX, Topology


def __getattr__(name: str):
    """Keep parser/runtime-adapter imports usable before PyTorch is installed."""

    if name in {"ComparisonResult", "compare_tensor"}:
        from .metrics import ComparisonResult, compare_tensor

        return {
            "ComparisonResult": ComparisonResult,
            "compare_tensor": compare_tensor,
        }[name]
    raise AttributeError(name)


__all__ = ["TOPOLOGY_MATRIX", "ComparisonResult", "Topology", "compare_tensor"]
