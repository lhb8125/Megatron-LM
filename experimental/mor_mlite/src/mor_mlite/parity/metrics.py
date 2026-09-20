"""Numerical and routing comparison rules from the acceptance contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import torch


@dataclass(slots=True)
class ComparisonResult:
    name: str
    passed: bool
    max_abs: float
    relative_l2: float
    cosine_similarity: float
    detail: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _flat64(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu", dtype=torch.float64).reshape(-1)


def compare_tensor(
    name: str,
    baseline: torch.Tensor,
    candidate: torch.Tensor,
    *,
    precision: Literal["fp32", "bf16"],
    kind: Literal["forward", "loss", "gradient", "update", "post_step"],
) -> ComparisonResult:
    """Compare one tensor using the plan's FP32 or BF16 contract."""

    if baseline.shape != candidate.shape:
        return ComparisonResult(
            name=name,
            passed=False,
            max_abs=float("inf"),
            relative_l2=float("inf"),
            cosine_similarity=-1.0,
            detail=f"shape mismatch: {tuple(baseline.shape)} != {tuple(candidate.shape)}",
        )
    lhs = _flat64(baseline)
    rhs = _flat64(candidate)
    if lhs.numel() == 0:
        return ComparisonResult(name, True, 0.0, 0.0, 1.0)
    if not bool(torch.isfinite(lhs).all() and torch.isfinite(rhs).all()):
        return ComparisonResult(
            name=name,
            passed=False,
            max_abs=float("inf"),
            relative_l2=float("inf"),
            cosine_similarity=-1.0,
            detail="baseline or candidate contains non-finite values",
        )
    diff = rhs - lhs
    max_abs = float(diff.abs().max().item())
    lhs_norm = torch.linalg.vector_norm(lhs)
    diff_norm = torch.linalg.vector_norm(diff)
    relative_l2 = float((diff_norm / lhs_norm.clamp_min(1e-30)).item())
    if float(lhs_norm) == 0.0 and float(torch.linalg.vector_norm(rhs)) == 0.0:
        cosine = 1.0
    else:
        cosine = float(
            torch.nn.functional.cosine_similarity(lhs.unsqueeze(0), rhs.unsqueeze(0), dim=1).item()
        )

    if precision == "fp32":
        rtol, atol = (2e-5, 2e-6) if kind in {"forward", "loss"} else (5e-5, 5e-6)
        passed = bool(torch.allclose(lhs, rhs, rtol=rtol, atol=atol))
        detail = f"allclose(rtol={rtol:g}, atol={atol:g})"
    elif precision == "bf16":
        if kind == "loss":
            passed = max_abs <= 1e-2
            detail = "absolute loss error <= 1e-2"
        else:
            limit = 0.02 if kind == "forward" else 0.03
            passed = relative_l2 <= limit and cosine >= 0.999
            detail = f"relative_l2 <= {limit:g} and cosine >= 0.999"
    else:
        raise ValueError(f"unknown precision: {precision}")
    return ComparisonResult(name, passed, max_abs, relative_l2, cosine, detail)


@dataclass(frozen=True, slots=True)
class NearTieResult:
    near_tie: bool
    margin: float
    score_error: float
    ambiguity_ids: tuple[int, ...]


def classify_cutoff(
    baseline_scores: torch.Tensor,
    candidate_scores: torch.Tensor,
    global_token_ids: torch.Tensor,
    top_k: int,
    *,
    epsilon: float = 1e-7,
) -> NearTieResult:
    """Classify a learned-router cutoff and return its ambiguity set."""

    if baseline_scores.shape != candidate_scores.shape:
        raise ValueError("baseline and candidate score shapes differ")
    if baseline_scores.ndim != 1 or global_token_ids.shape != baseline_scores.shape:
        raise ValueError("scores and global_token_ids must be equally sized 1-D tensors")
    count = baseline_scores.numel()
    if not 1 <= top_k <= count:
        raise ValueError(f"top_k must be in [1, {count}]")
    sorted_scores = torch.sort(baseline_scores.detach().float(), descending=True).values
    margin = (
        float("inf") if top_k == count else float(sorted_scores[top_k - 1] - sorted_scores[top_k])
    )
    error = float(
        (baseline_scores.detach().float() - candidate_scores.detach().float()).abs().max()
    )
    near_tie = margin <= 2.0 * error + epsilon
    if top_k == count:
        ambiguity = ()
    else:
        cutoff = 0.5 * (sorted_scores[top_k - 1] + sorted_scores[top_k])
        radius = error + epsilon
        mask = (baseline_scores.detach().float() - cutoff).abs() <= radius
        ambiguity = tuple(int(x) for x in global_token_ids[mask].tolist())
    return NearTieResult(near_tie, margin, error, ambiguity)


__all__ = [
    "ComparisonResult",
    "NearTieResult",
    "classify_cutoff",
    "compare_tensor",
]
