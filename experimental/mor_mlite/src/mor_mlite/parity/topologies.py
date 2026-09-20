"""The acceptance topology matrix.

EP is an overlapping expert view of the same world, not an additional world
dimension. Therefore ``world_size == dp * tp * cp``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mor_mlite.config import MoRParallelConfig
from mor_mlite.config_loader import load_json, topology_config_path


@dataclass(frozen=True, slots=True)
class Topology:
    name: str
    world_size: int
    tp: int = 1
    cp: int = 1
    dp: int = 1
    ep: int = 1
    etp: int = 1

    def to_parallel_config(self, *, cp_transition: str = "magi_direct") -> MoRParallelConfig:
        return MoRParallelConfig(
            dp=self.dp,
            tp=self.tp,
            cp=self.cp,
            ep=self.ep,
            etp=self.etp,
            zero_stage=1,
            cp_transition=cp_transition,
        )

    def validate(self, *, num_experts: int | None = None) -> None:
        self.to_parallel_config().validate_world_size(self.world_size, num_experts=num_experts)

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


_TOPOLOGY_FIELDS = frozenset({"name", "world_size", "tp", "cp", "dp", "ep"})
_OPTIONAL_TOPOLOGY_FIELDS = frozenset({"etp"})


def _topology_from_dict(values: Any, *, index: int, source: Path) -> Topology:
    if not isinstance(values, dict):
        raise TypeError(f"topology entry {index} in {source} must be a JSON object")
    fields = frozenset(values)
    missing = _TOPOLOGY_FIELDS - fields
    unknown = fields - _TOPOLOGY_FIELDS - _OPTIONAL_TOPOLOGY_FIELDS
    if missing:
        raise ValueError(f"topology entry {index} in {source} is missing fields: {sorted(missing)}")
    if unknown:
        raise ValueError(
            f"topology entry {index} in {source} has unknown fields: {sorted(unknown)}"
        )
    name = values["name"]
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"topology entry {index} in {source} needs a non-empty name")
    integer_fields = ("world_size", "tp", "cp", "dp", "ep", "etp")
    parsed: dict[str, int] = {}
    for field in integer_fields:
        value = values.get(field, 1)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"topology {name!r} field {field} in {source} must be a positive integer"
            )
        parsed[field] = value
    topology = Topology(name=name, **parsed)
    topology.validate()
    return topology


def load_topology_matrix(path: Path | None = None) -> tuple[Topology, ...]:
    """Load the acceptance matrix from its checked-in JSON source of truth."""

    source = topology_config_path() if path is None else Path(path)
    raw = load_json(source, expected_type=list)
    assert isinstance(raw, list)
    if not raw:
        raise ValueError(f"topology matrix must not be empty: {source}")
    matrix = tuple(
        _topology_from_dict(values, index=index, source=source) for index, values in enumerate(raw)
    )
    names = [topology.name for topology in matrix]
    if len(names) != len(set(names)):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(f"duplicate topology names in {source}: {duplicates}")
    return matrix


TOPOLOGY_MATRIX: tuple[Topology, ...] = load_topology_matrix()


def get_topology(name: str) -> Topology:
    for topology in TOPOLOGY_MATRIX:
        if topology.name == name:
            return topology
    choices = ", ".join(item.name for item in TOPOLOGY_MATRIX)
    raise ValueError(f"unknown topology {name!r}; choose one of: {choices}")


__all__ = ["TOPOLOGY_MATRIX", "Topology", "get_topology", "load_topology_matrix"]
