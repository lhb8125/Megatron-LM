"""Serializable routing decisions used for learned execution and replay."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

ROUTE_PLAN_SCHEMA_VERSION = 1


def _as_cpu_tensor(value: Any, *, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(value, dtype=dtype, device="cpu").contiguous()


@dataclass(frozen=True)
class RoutePlan:
    """One round's selected tokens and their source/target ownership.

    Vector fields are aligned row-for-row.  They normally contain real selected
    tokens only; a transition backend may append dummy rows and mark them in
    ``padding_mask``.  Consequently ``active_cu_seqlens[-1]`` counts non-padding
    rows and need not equal the total vector length.
    """

    round_index: int
    mode: str
    sample_ids: torch.Tensor
    original_positions: torch.Tensor
    global_token_ids: torch.Tensor
    source_tp_ranks: torch.Tensor
    source_cp_ranks: torch.Tensor
    source_local_rows: torch.Tensor
    target_tp_ranks: torch.Tensor
    target_cp_ranks: torch.Tensor
    target_local_rows: torch.Tensor
    selected_gates: torch.Tensor
    active_cu_seqlens: torch.Tensor
    padding_mask: torch.Tensor
    cutoff_score_margins: Mapping[int, float]

    def __post_init__(self) -> None:
        if isinstance(self.round_index, bool) or self.round_index < 0:
            raise ValueError("round_index must be a non-negative integer")
        if self.mode not in {"learned", "replay"}:
            raise ValueError("RoutePlan mode must be 'learned' or 'replay'")

        integer_fields = (
            "sample_ids",
            "original_positions",
            "global_token_ids",
            "source_tp_ranks",
            "source_cp_ranks",
            "source_local_rows",
            "target_tp_ranks",
            "target_cp_ranks",
            "target_local_rows",
        )
        expected_length: int | None = None
        for name in integer_fields:
            tensor = getattr(self, name)
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != 1:
                raise ValueError(f"{name} must be a one-dimensional torch.Tensor")
            if tensor.dtype not in {
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            }:
                raise ValueError(f"{name} must have an integer dtype")
            expected_length = tensor.numel() if expected_length is None else expected_length
            if tensor.numel() != expected_length:
                raise ValueError("all token-aligned RoutePlan fields must have equal length")

        if self.selected_gates.ndim != 1 or self.selected_gates.numel() != expected_length:
            raise ValueError("selected_gates must be one-dimensional and token-aligned")
        if self.padding_mask.dtype != torch.bool or self.padding_mask.ndim != 1:
            raise ValueError("padding_mask must be a one-dimensional bool tensor")
        if self.padding_mask.numel() != expected_length:
            raise ValueError("padding_mask must be token-aligned")
        if self.active_cu_seqlens.ndim != 1 or self.active_cu_seqlens.dtype not in {
            torch.int32,
            torch.int64,
        }:
            raise ValueError("active_cu_seqlens must be a one-dimensional int tensor")
        if self.active_cu_seqlens.numel() == 0 or self.active_cu_seqlens[0].item() != 0:
            raise ValueError("active_cu_seqlens must start at zero")
        if torch.any(self.active_cu_seqlens[1:] < self.active_cu_seqlens[:-1]):
            raise ValueError("active_cu_seqlens must be monotonically non-decreasing")
        real_count = int((~self.padding_mask).sum().item())
        if self.active_cu_seqlens[-1].item() != real_count:
            raise ValueError("active_cu_seqlens[-1] must equal the number of non-padding rows")
        if len(set(self.global_token_ids[~self.padding_mask].tolist())) != real_count:
            raise ValueError("real global_token_ids must be unique within a RoutePlan")
        if any(int(sample_id) < 0 for sample_id in self.cutoff_score_margins):
            raise ValueError("cutoff_score_margins keys must be non-negative sample IDs")
        margins = [float(value) for value in self.cutoff_score_margins.values()]
        if any(math.isnan(value) or value < 0.0 for value in margins):
            raise ValueError("cutoff_score_margins must be non-negative and not NaN")

    def __len__(self) -> int:
        return self.global_token_ids.numel()

    @property
    def num_active_tokens(self) -> int:
        return int((~self.padding_mask).sum().item())

    def with_mode(self, mode: str) -> RoutePlan:
        return replace(self, mode=mode)

    def to(self, device: torch.device | str) -> RoutePlan:
        values: dict[str, Any] = {}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            values[name] = value.to(device) if isinstance(value, torch.Tensor) else value
        return type(self)(**values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ROUTE_PLAN_SCHEMA_VERSION,
            "round": self.round_index,
            "mode": self.mode,
            "sample_ids": self.sample_ids.detach().cpu().tolist(),
            "original_positions": self.original_positions.detach().cpu().tolist(),
            "global_token_ids": self.global_token_ids.detach().cpu().tolist(),
            "source_tp_ranks": self.source_tp_ranks.detach().cpu().tolist(),
            "source_cp_ranks": self.source_cp_ranks.detach().cpu().tolist(),
            "source_local_rows": self.source_local_rows.detach().cpu().tolist(),
            "target_tp_ranks": self.target_tp_ranks.detach().cpu().tolist(),
            "target_cp_ranks": self.target_cp_ranks.detach().cpu().tolist(),
            "target_local_rows": self.target_local_rows.detach().cpu().tolist(),
            "selected_gates": self.selected_gates.detach().float().cpu().tolist(),
            "active_cu_seqlens": self.active_cu_seqlens.detach().cpu().tolist(),
            "padding_mask": self.padding_mask.detach().cpu().tolist(),
            "cutoff_score_margins": {
                str(sample_id): (None if math.isinf(float(margin)) else float(margin))
                for sample_id, margin in self.cutoff_score_margins.items()
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RoutePlan:
        schema_version = value.get("schema_version")
        if schema_version != ROUTE_PLAN_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported RoutePlan schema_version {schema_version!r}; "
                f"expected {ROUTE_PLAN_SCHEMA_VERSION}"
            )
        return cls(
            round_index=int(value["round"]),
            mode=str(value["mode"]),
            sample_ids=_as_cpu_tensor(value["sample_ids"], dtype=torch.int64),
            original_positions=_as_cpu_tensor(value["original_positions"], dtype=torch.int64),
            global_token_ids=_as_cpu_tensor(value["global_token_ids"], dtype=torch.int64),
            source_tp_ranks=_as_cpu_tensor(value["source_tp_ranks"], dtype=torch.int64),
            source_cp_ranks=_as_cpu_tensor(value["source_cp_ranks"], dtype=torch.int64),
            source_local_rows=_as_cpu_tensor(value["source_local_rows"], dtype=torch.int64),
            target_tp_ranks=_as_cpu_tensor(value["target_tp_ranks"], dtype=torch.int64),
            target_cp_ranks=_as_cpu_tensor(value["target_cp_ranks"], dtype=torch.int64),
            target_local_rows=_as_cpu_tensor(value["target_local_rows"], dtype=torch.int64),
            selected_gates=_as_cpu_tensor(value["selected_gates"], dtype=torch.float32),
            active_cu_seqlens=_as_cpu_tensor(value["active_cu_seqlens"], dtype=torch.int32),
            padding_mask=_as_cpu_tensor(value["padding_mask"], dtype=torch.bool),
            cutoff_score_margins={
                int(sample_id): (math.inf if margin is None else float(margin))
                for sample_id, margin in value["cutoff_score_margins"].items()
            },
        )

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> RoutePlan:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def replay_indices(
        self,
        global_token_ids: torch.Tensor,
        *,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Map replay token IDs to rows in a possibly different local layout."""

        if global_token_ids.ndim != 1:
            raise ValueError("global_token_ids must be one-dimensional")
        if padding_mask is None:
            padding_mask = torch.zeros_like(global_token_ids, dtype=torch.bool)
        if padding_mask.shape != global_token_ids.shape or padding_mask.dtype != torch.bool:
            raise ValueError("padding_mask must be a bool tensor shaped like global_token_ids")

        candidate_ids = global_token_ids[~padding_mask].detach().cpu().tolist()
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate global_token_ids must be unique")
        row_by_id = {
            int(token_id): row
            for row, token_id in enumerate(global_token_ids.detach().cpu().tolist())
            if not bool(padding_mask[row].item())
        }
        selected_ids = self.global_token_ids[~self.padding_mask].detach().cpu().tolist()
        missing = [int(token_id) for token_id in selected_ids if int(token_id) not in row_by_id]
        if missing:
            raise ValueError(f"replay candidates are missing selected token IDs: {missing}")
        return torch.tensor(
            [row_by_id[int(token_id)] for token_id in selected_ids],
            dtype=torch.int64,
            device=global_token_ids.device,
        )

    def replay_gates(self, selected_global_token_ids: torch.Tensor) -> torch.Tensor:
        """Return stored gate values in the caller's selected-token order.

        Route replay is a numerical oracle as well as a discrete-routing oracle.
        The caller may own an arbitrary topology-local subset of selected tokens,
        so gate values are resolved by immutable global token ID rather than by
        the RoutePlan's serialized row position.
        """

        if selected_global_token_ids.ndim != 1:
            raise ValueError("selected_global_token_ids must be one-dimensional")
        requested_ids = [int(value) for value in selected_global_token_ids.detach().cpu().tolist()]
        if len(requested_ids) != len(set(requested_ids)):
            raise ValueError("selected_global_token_ids must be unique")

        real_mask = ~self.padding_mask
        plan_ids = [
            int(value) for value in self.global_token_ids[real_mask].detach().cpu().tolist()
        ]
        plan_gates = self.selected_gates[real_mask].detach().float().cpu().tolist()
        gate_by_id = dict(zip(plan_ids, plan_gates, strict=True))
        missing = [token_id for token_id in requested_ids if token_id not in gate_by_id]
        if missing:
            raise ValueError(f"replay gate plan is missing selected token IDs: {missing}")
        return torch.tensor(
            [gate_by_id[token_id] for token_id in requested_ids],
            dtype=self.selected_gates.dtype,
            device=selected_global_token_ids.device,
        )
