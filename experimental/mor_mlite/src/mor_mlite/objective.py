"""Public token-weighted objectives for MLite DP and microbatch averaging.

Counts describe a complete global step, indexed [microbatch][dense-DP rank].
The model already normalizes within CP; never include TP/CP replica copies in
these counts. MLite averages microbatch losses and dense-DP gradients.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ObjectiveScale:
    lm: float
    auxiliary: tuple[float, ...]


def nonzero_weight_denominator(count):
    """Protect empty masks without rounding fractional positive weights up to one."""
    import torch

    return torch.where(count > 0, count, torch.ones_like(count))


def objective_scales(lm_counts, router_counts) -> tuple[tuple[ObjectiveScale, ...], ...]:
    """Return scales for global token means, including unequal microbatch counts.

    LM counts may be fractional for weighted masks. Router counts have an extra
    final dimension for each recursion's *candidate* count, before selection.
    """
    if not lm_counts or not lm_counts[0] or len(router_counts) != len(lm_counts):
        raise ValueError("counts must contain aligned nonempty microbatch and DP axes")
    microbatches, dp = len(lm_counts), len(lm_counts[0])
    if any(len(row) != dp for row in lm_counts) or any(len(row) != dp for row in router_counts):
        raise ValueError("counts must have a rectangular dense-DP axis")
    rounds = len(router_counts[0][0])
    if not rounds or any(len(value) != rounds for row in router_counts for value in row):
        raise ValueError("router candidate counts must have a consistent recursion axis")
    values = [value for row in lm_counts for value in row]
    values += [value for row in router_counts for counts in row for value in counts]
    if any(
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
        or value < 0
        for value in values
    ):
        raise ValueError("objective counts must be finite non-negative numbers")
    lm_total = sum(sum(row) for row in lm_counts)
    aux_totals = tuple(
        sum(counts[r] for row in router_counts for counts in row) for r in range(rounds)
    )
    if lm_total <= 0 or any(value <= 0 for value in aux_totals):
        raise ValueError("global objective counts must be positive")
    compensation = dp * microbatches
    return tuple(
        tuple(
            ObjectiveScale(
                compensation * lm_counts[mb][rank] / lm_total,
                tuple(
                    compensation * router_counts[mb][rank][r] / aux_totals[r] for r in range(rounds)
                ),
            )
            for rank in range(dp)
        )
        for mb in range(microbatches)
    )


def apply_objective(output: dict[str, Any], scale: ObjectiveScale):
    """Adapt native MoR model output to MLite's loss_fn without parity metadata."""
    loss = output["loss"]
    auxiliary = output["mor_router_aux_losses"].reshape(-1)
    if auxiliary.numel() != len(scale.auxiliary):
        raise ValueError("router auxiliary loss count differs from objective scales")
    lm = loss - output["mor_router_aux_loss"].to(dtype=loss.dtype)
    return lm * scale.lm + sum(
        value.to(dtype=loss.dtype) * factor
        for value, factor in zip(auxiliary, scale.auxiliary, strict=True)
    )
