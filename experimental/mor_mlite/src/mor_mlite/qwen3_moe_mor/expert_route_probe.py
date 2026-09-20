"""Opt-in parity probe for native Qwen MoE expert routing.

The production model deliberately does not expose native MoE router decisions.
This helper attaches a short-lived forward hook only while a parity caller has
explicitly enabled the probe and entered a logical layer scope.  The hook sees
the exact ``(scores, indices)`` tuple consumed by ``TokenDispatcher``.  An
independent, forward-scoped replay mode can replace the expert *set* through
MLite's native ``router_replay.select_indices`` seam.  It also replays the
dispatch-visible score values with a straight-through autograd bridge: forward
uses the baseline score while backward follows the live router score.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True, slots=True)
class ExpertRouteContext:
    """Stable identity of one logical invocation of a physical MoE layer."""

    stage: str
    round_index: int
    stage_layer_index: int
    physical_layer_index: int
    logical_layer_index: int

    def validate(self) -> None:
        if self.stage not in {"start", "recurrent", "end"}:
            raise ValueError(f"unsupported MoE probe stage: {self.stage!r}")
        if self.stage == "recurrent" and self.round_index < 0:
            raise ValueError("a recurrent MoE probe context requires a non-negative round")
        if self.stage != "recurrent" and self.round_index != -1:
            raise ValueError("start/end MoE probe contexts use round_index=-1")
        for name in ("stage_layer_index", "physical_layer_index", "logical_layer_index"):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")

    def to_dict(self) -> dict[str, int | str]:
        self.validate()
        return {
            "stage": self.stage,
            "round_index": self.round_index,
            "stage_layer_index": self.stage_layer_index,
            "physical_layer_index": self.physical_layer_index,
            "logical_layer_index": self.logical_layer_index,
        }


@dataclass(frozen=True, slots=True)
class ExpertRouteReplayPlan:
    """Canonical expert choices and consumed scores for one logical layer.

    ``global_token_ids`` contains only real tokens.  Padding is intentionally
    absent: a live padding row always retains its native expert choices.
    """

    global_token_ids: torch.Tensor
    topk_indices: torch.Tensor
    selected_scores: torch.Tensor

    def validate(self) -> None:
        token_ids = self.global_token_ids
        indices = self.topk_indices
        scores = self.selected_scores
        if not all(isinstance(value, torch.Tensor) for value in (token_ids, indices, scores)):
            raise TypeError("expert replay IDs and scores must be tensors")
        if token_ids.dtype != torch.long or indices.dtype != torch.long:
            raise TypeError("expert replay token and expert IDs must use torch.long")
        if not scores.dtype.is_floating_point:
            raise TypeError("expert replay selected scores must use a floating dtype")
        if token_ids.ndim != 1:
            raise ValueError("expert replay global token IDs must be one-dimensional")
        if indices.ndim != 2 or indices.size(0) != token_ids.numel():
            raise ValueError("expert replay Top-K rows must align with global token IDs")
        if scores.shape != indices.shape:
            raise ValueError("expert replay selected scores must align with Top-K expert IDs")
        if indices.size(1) <= 0:
            raise ValueError("expert replay Top-K width must be positive")
        if token_ids.numel() and bool((token_ids < 0).any().item()):
            raise ValueError("expert replay global token IDs must be non-negative")
        if torch.unique(token_ids).numel() != token_ids.numel():
            raise ValueError("expert replay has duplicate token IDs")
        if indices.numel() and bool((indices < 0).any().item()):
            raise ValueError("expert replay expert IDs must be non-negative")
        if scores.numel() and not bool(torch.isfinite(scores).all().item()):
            raise ValueError("expert replay selected scores must be finite")
        if scores.numel() and bool((scores < 0).any().item()):
            raise ValueError("expert replay selected scores must be non-negative")
        if indices.size(1) > 1:
            ordered = torch.sort(indices, dim=1).values
            if bool((ordered[:, 1:] == ordered[:, :-1]).any().item()):
                raise ValueError("expert replay selected one expert twice for a token")

    def to(self, device: torch.device | str | int) -> ExpertRouteReplayPlan:
        """Move replay tensors without weakening their validation contract."""

        self.validate()
        return ExpertRouteReplayPlan(
            global_token_ids=self.global_token_ids.to(device=device),
            topk_indices=self.topk_indices.to(device=device),
            selected_scores=self.selected_scores.to(device=device),
        )


@dataclass(frozen=True, slots=True)
class _PreparedReplayPlan:
    topk: int
    expert_rows: Mapping[int, tuple[int, ...]]
    score_rows: Mapping[int, tuple[float, ...]]


class _ReplaySelectedScores(torch.autograd.Function):
    """Use exact replay values in forward and the live-router Jacobian in backward."""

    @staticmethod
    def forward(
        ctx: Any,
        live_scores: torch.Tensor,
        target_scores: torch.Tensor,
        real_rows: torch.Tensor,
    ) -> torch.Tensor:
        del ctx
        output = live_scores.clone()
        output.index_copy_(0, real_rows, target_scores)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        del ctx
        return grad_output, None, None


class _ScopedReplaySelector:
    """Duck-typed adapter for MLite ``TopKRouter.router_replay``."""

    # MLite only treats its own RouterReplayAction values specially.  Keeping
    # this unset lets the router attach its ordinary auxiliary-loss gradient;
    # ExpertRouteProbe rejects differentiable replay when that native auxiliary
    # objective is enabled because its load statistic follows native indices.
    router_replay_action = None

    def __init__(
        self,
        *,
        target_rows: torch.Tensor,
        real_rows: torch.Tensor,
        expected_shape: tuple[int, int],
    ) -> None:
        self._target_rows = target_rows
        self._real_rows = real_rows
        self._expected_shape = expected_shape
        self.calls = 0

    def select_indices(self, native_indices: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        if self.calls != 1:
            raise RuntimeError("native MoE router requested expert replay more than once")
        if tuple(native_indices.shape) != self._expected_shape:
            raise RuntimeError(
                "native MoE router shape changed before expert replay: "
                f"{tuple(native_indices.shape)} != {self._expected_shape}"
            )
        selected = native_indices.clone()
        real_rows = self._real_rows.to(device=selected.device, dtype=torch.long)
        target_rows = self._target_rows.to(device=selected.device, dtype=torch.long)
        selected.index_copy_(0, real_rows, target_rows)
        # Return a distinct tensor even when all selected experts happen to be
        # native.  MLite then follows its replay branch and gathers scores from
        # this invocation's live logits rather than accepting stored scores.
        return selected


def _default_gating_linear(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    router_dtype: torch.dtype,
) -> torch.Tensor:
    """Torch fallback used by unit tests outside a Megatron installation."""

    return F.linear(
        inputs.to(dtype=router_dtype),
        weight.to(dtype=router_dtype),
        None if bias is None else bias.to(dtype=router_dtype),
    )


class ExpertRouteProbe:
    """Capture or replay dispatch-visible expert choices for parity."""

    def __init__(
        self,
        *,
        gating_linear: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor | None, torch.dtype], torch.Tensor
        ] = _default_gating_linear,
    ) -> None:
        self._gating_linear = gating_linear
        self.enabled = False
        self._active_scope = False
        self._traces: list[dict[str, Any]] = []
        self._replay_plans: dict[int, _PreparedReplayPlan] | None = None
        self._replay_consumed_layers: set[int] = set()

    @property
    def replay_active(self) -> bool:
        """Whether the current forward has an expert replay contract."""

        return self._replay_plans is not None

    def set_enabled(self, enabled: bool) -> None:
        if self._active_scope:
            raise RuntimeError("cannot change the expert-route probe inside a layer scope")
        self.enabled = bool(enabled)
        self._traces.clear()

    @staticmethod
    def _prepare_replay_plans(
        replay_plans: Mapping[int, ExpertRouteReplayPlan],
    ) -> dict[int, _PreparedReplayPlan]:
        prepared: dict[int, _PreparedReplayPlan] = {}
        for logical_layer_raw, plan in replay_plans.items():
            if isinstance(logical_layer_raw, bool) or not isinstance(logical_layer_raw, int):
                raise TypeError("expert replay logical layer indices must be non-negative ints")
            logical_layer = logical_layer_raw
            if logical_layer < 0:
                raise ValueError("expert replay logical layer indices must be non-negative ints")
            if not isinstance(plan, ExpertRouteReplayPlan):
                raise TypeError("expert replay values must be ExpertRouteReplayPlan instances")
            plan.validate()
            token_ids = plan.global_token_ids.detach()
            indices = plan.topk_indices.detach()
            scores = plan.selected_scores.detach().float()
            expert_rows = {
                int(token_id): tuple(int(expert_id) for expert_id in row)
                for token_id, row in zip(
                    token_ids.cpu().tolist(), indices.cpu().tolist(), strict=True
                )
            }
            score_rows = {
                int(token_id): tuple(float(score) for score in row)
                for token_id, row in zip(
                    token_ids.cpu().tolist(), scores.cpu().tolist(), strict=True
                )
            }
            prepared[logical_layer] = _PreparedReplayPlan(
                topk=int(indices.size(1)),
                expert_rows=expert_rows,
                score_rows=score_rows,
            )
        return prepared

    def begin_forward(
        self,
        replay_plans: Mapping[int, ExpertRouteReplayPlan] | None = None,
    ) -> None:
        if self._active_scope:
            raise RuntimeError("cannot begin an expert-route probe inside a layer scope")
        if self.replay_active:
            raise RuntimeError("previous expert replay forward was not finished")
        self._traces.clear()
        self._replay_plans = (
            None if replay_plans is None else self._prepare_replay_plans(replay_plans)
        )
        self._replay_consumed_layers.clear()

    def finish_forward(self) -> tuple[dict[str, Any], ...]:
        if self._active_scope:
            raise RuntimeError("cannot finish an expert-route probe inside a layer scope")
        traces = tuple(self._traces)
        replay_plans = self._replay_plans
        consumed = set(self._replay_consumed_layers)
        self._replay_plans = None
        self._replay_consumed_layers.clear()
        if replay_plans is not None and consumed != set(replay_plans):
            missing = sorted(set(replay_plans) - consumed)
            extra = sorted(consumed - set(replay_plans))
            raise RuntimeError(
                "expert replay logical-layer coverage differs: "
                f"missing={missing[:8]}, extra={extra[:8]}"
            )
        return traces

    @contextmanager
    def capture(
        self,
        router: nn.Module,
        *,
        context: ExpertRouteContext,
        global_token_ids: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> Iterator[None]:
        """Capture, and optionally replay, one native router invocation."""

        if not self.enabled and not self.replay_active:
            yield
            return
        if self._active_scope:
            raise RuntimeError("nested expert-route probe scopes are not supported")
        context.validate()
        token_ids = global_token_ids.detach().reshape(-1).to(dtype=torch.long)
        padding = padding_mask.detach().reshape(-1).to(dtype=torch.bool)
        if token_ids.shape != padding.shape:
            raise ValueError("expert-route probe token IDs and padding mask must be aligned")
        real_rows = torch.nonzero(~padding, as_tuple=False).reshape(-1)
        real_ids = token_ids.index_select(0, real_rows)
        if real_ids.numel() and int(real_ids.min().item()) < 0:
            raise ValueError("non-padding expert-route probe token IDs must be non-negative")
        if torch.unique(real_ids).numel() != real_ids.numel():
            raise ValueError("expert-route probe token IDs must be locally unique")

        replay_selector: _ScopedReplaySelector | None = None
        previous_router_replay: Any = None
        if self.replay_active:
            assert self._replay_plans is not None
            logical_layer = context.logical_layer_index
            if logical_layer in self._replay_consumed_layers:
                raise RuntimeError(
                    f"expert replay logical layer {logical_layer} ran more than once"
                )
            plan = self._replay_plans.get(logical_layer)
            if plan is None:
                raise RuntimeError(f"expert replay has no plan for logical layer {logical_layer}")
            topk = int(getattr(router, "topk", 0))
            num_experts = int(getattr(router, "num_experts", 0))
            if topk <= 0 or num_experts <= 0:
                raise TypeError("native MoE router exposes no Top-K/expert count")
            if plan.topk != topk:
                raise ValueError(
                    f"expert replay Top-K mismatch at logical layer {logical_layer}: "
                    f"{plan.topk} != {topk}"
                )
            real_id_values = [int(token_id) for token_id in real_ids.cpu().tolist()]
            missing_ids = [
                token_id for token_id in real_id_values if token_id not in plan.expert_rows
            ]
            if missing_ids:
                raise RuntimeError(
                    f"expert replay is missing token IDs at logical layer {logical_layer}: "
                    f"{missing_ids[:8]}"
                )
            replay_rows = [plan.expert_rows[token_id] for token_id in real_id_values]
            replay_scores = [plan.score_rows[token_id] for token_id in real_id_values]
            target_rows = torch.tensor(replay_rows, dtype=torch.long).reshape(-1, topk)
            target_scores = torch.tensor(replay_scores, dtype=torch.float32).reshape(-1, topk)
            if target_rows.numel() and bool((target_rows >= num_experts).any().item()):
                raise ValueError(
                    f"expert replay has out-of-range expert IDs at logical layer {logical_layer}"
                )
            if not hasattr(router, "router_replay"):
                raise TypeError("native MoE router exposes no router_replay seam")
            previous_router_replay = router.router_replay  # type: ignore[attr-defined]
            if previous_router_replay is not None:
                raise RuntimeError("expert replay conflicts with an existing native router replay")
            if (
                router.training
                and torch.is_grad_enabled()
                and bool(getattr(router, "compute_aux_loss", False))
                and bool(getattr(router, "aux_loss_coeff", 0.0))
            ):
                raise RuntimeError(
                    "differentiable expert replay requires native router auxiliary loss "
                    "to be disabled"
                )
            replay_selector = _ScopedReplaySelector(
                target_rows=target_rows,
                real_rows=real_rows,
                expected_shape=(token_ids.numel(), topk),
            )
            router.router_replay = replay_selector  # type: ignore[attr-defined]
        calls = 0

        def capture_output(
            module: nn.Module, inputs: tuple[Any, ...], output: Any
        ) -> tuple[torch.Tensor, torch.Tensor] | None:
            nonlocal calls
            calls += 1
            if calls != 1:
                raise RuntimeError("native MoE router ran more than once in one layer scope")
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise TypeError("native MoE router input is not a tensor")
            if (
                not isinstance(output, (tuple, list))
                or len(output) != 2
                or not all(isinstance(value, torch.Tensor) for value in output)
            ):
                raise TypeError("native MoE router must return tensor (scores, indices)")
            scores, indices = output
            router_input = inputs[0].reshape(-1, inputs[0].size(-1))
            indices = indices.reshape(router_input.size(0), -1)
            scores = scores.reshape_as(indices)
            live_scores = scores
            if token_ids.numel() != router_input.size(0):
                raise RuntimeError(
                    "expert-route probe metadata does not match router rows: "
                    f"{token_ids.numel()} != {router_input.size(0)}"
                )
            if indices.dtype != torch.long:
                indices = indices.to(dtype=torch.long)
            topk = int(getattr(module, "topk", indices.size(1)))
            num_experts = int(getattr(module, "num_experts", 0))
            gate = getattr(module, "gate", None)
            weight = getattr(gate, "weight", None)
            if not isinstance(weight, torch.Tensor) or num_experts <= 0:
                raise TypeError("native MoE router exposes no gate weight/num_experts")
            if indices.shape != (router_input.size(0), topk):
                raise RuntimeError("native MoE router returned an unexpected Top-K shape")
            if bool(((indices < 0) | (indices >= num_experts)).any().item()):
                raise RuntimeError("native MoE router returned an out-of-range expert ID")
            if topk > 1:
                ordered_indices = torch.sort(indices, dim=1).values
                if bool((ordered_indices[:, 1:] == ordered_indices[:, :-1]).any().item()):
                    raise RuntimeError("native MoE router selected one expert twice for a token")

            if replay_selector is not None:
                # The native replay seam fixes expert identity.  Pin the
                # accompanying baseline forward scores as well: otherwise a
                # small BF16 router difference is repeatedly amplified by
                # weighted MoE combines.  Backward still follows the exact
                # live-router Jacobian through the custom identity gradient.
                target = target_scores.to(device=scores.device, dtype=scores.dtype)
                replay_rows = real_rows.to(device=scores.device, dtype=torch.long)
                scores = _ReplaySelectedScores.apply(scores, target, replay_rows)

            # Recompute only the gate projection, with the same primitive and
            # dtype as TopKRouter, to expose the K/(K+1) cutoff margin.  The
            # dispatch-visible ``indices`` above remain the source of truth for
            # expert identity; this diagnostic projection is detached and
            # cannot affect RNG, autograd, dispatch, or combine.
            router_dtype = getattr(module, "router_dtype", None) or router_input.dtype
            with torch.no_grad():
                logits = self._gating_linear(
                    router_input.detach(), weight.detach(), None, router_dtype
                ).reshape(router_input.size(0), num_experts)
                logits = logits.float()
                selected_logits = torch.gather(logits, 1, indices)
                selected_floor = selected_logits.min(dim=1).values
                if topk < num_experts:
                    selected_mask = torch.zeros_like(logits, dtype=torch.bool)
                    selected_mask.scatter_(1, indices, True)
                    unselected_ceiling = (
                        logits.masked_fill(selected_mask, -torch.inf).max(dim=1).values
                    )
                    cutoff_margins = selected_floor - unselected_ceiling
                    has_unselected_expert = True
                else:
                    # There is no K+1 expert.  Keep the tensor finite so generic
                    # artifact validation remains meaningful and expose the
                    # condition explicitly in metadata.
                    cutoff_margins = torch.zeros_like(selected_floor)
                    has_unselected_expert = False

            capture_real_rows = real_rows
            capture_real_ids = real_ids
            order = torch.argsort(capture_real_ids, stable=True)
            capture_real_rows = capture_real_rows.index_select(0, order)
            capture_real_ids = capture_real_ids.index_select(0, order)
            self._traces.append(
                {
                    **context.to_dict(),
                    "topk": topk,
                    "num_experts": num_experts,
                    "has_unselected_expert": has_unselected_expert,
                    "global_token_ids": capture_real_ids.detach().clone(),
                    "topk_indices": indices.index_select(0, capture_real_rows).detach().clone(),
                    "selected_scores": (
                        scores.index_select(0, capture_real_rows).detach().float().clone()
                    ),
                    "live_selected_scores": (
                        live_scores.index_select(0, capture_real_rows).detach().float().clone()
                    ),
                    "cutoff_logit_margins": (
                        cutoff_margins.index_select(0, capture_real_rows).detach().clone()
                    ),
                }
            )
            if replay_selector is not None:
                return scores, indices
            return None

        self._active_scope = True
        handle = router.register_forward_hook(capture_output)
        try:
            yield
        finally:
            handle.remove()
            if replay_selector is not None:
                router.router_replay = previous_router_replay  # type: ignore[attr-defined]
            self._active_scope = False
        if calls != 1:
            raise RuntimeError("native MoE router did not run in the declared layer scope")
        if replay_selector is not None:
            if replay_selector.calls != 1:
                raise RuntimeError("native MoE router did not consume the expert replay")
            self._replay_consumed_layers.add(context.logical_layer_index)


__all__ = ["ExpertRouteContext", "ExpertRouteProbe", "ExpertRouteReplayPlan"]
