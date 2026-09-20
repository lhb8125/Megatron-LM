from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from mor_mlite.qwen3_moe_mor.expert_route_probe import (
    ExpertRouteContext,
    ExpertRouteProbe,
    ExpertRouteReplayPlan,
)


class _FakeTopKRouter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.topk = 2
        self.num_experts = 4
        self.router_dtype = None
        self.router_replay = None
        self.compute_aux_loss = False
        self.aux_loss_coeff = 0.0
        self.gate = nn.Linear(2, 4, bias=False)
        with torch.no_grad():
            self.gate.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [-1.0, 0.0],
                        [0.0, -1.0],
                    ]
                )
            )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.gate(inputs)
        probabilities = torch.softmax(logits, dim=-1)
        _, indices = torch.topk(probabilities, self.topk, dim=-1)
        # Pinned native Qwen returns selected experts in expert-ID order.
        indices = torch.sort(indices, dim=-1).values
        scores = torch.gather(probabilities, 1, indices)
        if self.router_replay is not None:
            selected = self.router_replay.select_indices(indices)
            if selected is not indices:
                indices = selected
                # This fake uses pre-softmax probabilities.  Like MLite's
                # gather_replayed_router_scores, values always come from the
                # current differentiable router invocation.
                scores = torch.gather(probabilities, 1, indices)
        return scores, indices


def _context(*, logical_layer: int = 3) -> ExpertRouteContext:
    return ExpertRouteContext(
        stage="recurrent",
        round_index=1,
        stage_layer_index=0,
        physical_layer_index=1,
        logical_layer_index=logical_layer,
    )


def test_probe_captures_dispatch_indices_and_filters_padding() -> None:
    router = _FakeTopKRouter()
    inputs = torch.tensor([[3.0, 1.0], [0.0, -3.0], [1.0, 3.0]])
    token_ids = torch.tensor([12, -2, 10])
    padding = torch.tensor([False, True, False])
    probe = ExpertRouteProbe()
    probe.set_enabled(True)
    probe.begin_forward()

    expected_scores, expected_indices = router(inputs)
    with probe.capture(
        router,
        context=_context(),
        global_token_ids=token_ids,
        padding_mask=padding,
    ):
        actual_scores, actual_indices = router(inputs)

    assert torch.equal(actual_scores, expected_scores)
    assert torch.equal(actual_indices, expected_indices)
    (trace,) = probe.finish_forward()
    assert trace["global_token_ids"].tolist() == [10, 12]
    assert torch.equal(trace["topk_indices"], expected_indices[[2, 0]])
    assert torch.equal(trace["selected_scores"], expected_scores[[2, 0]].float())
    # Rows 2 and 0 have logits [1,3,-1,-3] and [3,1,-3,-1].  The weakest
    # selected logit is 1 and the strongest unselected logit is -1.
    assert torch.equal(trace["cutoff_logit_margins"], torch.full((2,), 2.0))
    assert trace["logical_layer_index"] == 3
    assert trace["physical_layer_index"] == 1


def test_disabled_probe_adds_no_hook_or_gate_projection() -> None:
    calls = 0

    def gating_linear(inputs, weight, bias, router_dtype):
        nonlocal calls
        calls += 1
        return torch.nn.functional.linear(inputs, weight, bias)

    router = _FakeTopKRouter()
    probe = ExpertRouteProbe(gating_linear=gating_linear)
    inputs = torch.tensor([[1.0, 2.0]])
    before = tuple(router._forward_hooks)
    with probe.capture(
        router,
        context=_context(),
        global_token_ids=torch.tensor([7]),
        padding_mask=torch.tensor([False]),
    ):
        router(inputs)
    assert tuple(router._forward_hooks) == before
    assert calls == 0
    assert probe.finish_forward() == ()


def _replay_plan(
    token_ids: list[int],
    indices: list[list[int]],
    scores: list[list[float]] | None = None,
) -> ExpertRouteReplayPlan:
    if scores is None:
        scores = [[1.0 / len(row) for _ in row] for row in indices]
    return ExpertRouteReplayPlan(
        global_token_ids=torch.tensor(token_ids, dtype=torch.long),
        topk_indices=torch.tensor(indices, dtype=torch.long),
        selected_scores=torch.tensor(scores, dtype=torch.float32),
    )


def test_replay_uses_logical_token_identity_keeps_dummy_native_and_is_differentiable() -> None:
    router = _FakeTopKRouter()
    inputs = torch.tensor([[3.0, 1.0], [0.0, -3.0], [1.0, 3.0]], requires_grad=True)
    token_ids = torch.tensor([12, -2, 10])
    padding = torch.tensor([False, True, False])
    with torch.no_grad():
        _, native_indices = router(inputs.detach())

    probe = ExpertRouteProbe()
    probe.begin_forward(
        replay_plans={
            3: _replay_plan(
                [10, 12],
                [[1, 2], [2, 3]],
                [[0.25, 0.75], [0.6, 0.4]],
            )
        }
    )
    assert probe.replay_active
    with probe.capture(
        router,
        context=_context(),
        global_token_ids=token_ids,
        padding_mask=padding,
    ):
        scores, indices = router(inputs)

    assert indices.tolist() == [[2, 3], native_indices[1].tolist(), [1, 2]]
    live_scores = torch.softmax(router.gate(inputs), dim=-1).gather(1, indices)
    expected_scores = live_scores.detach().clone()
    expected_scores[0] = torch.tensor([0.6, 0.4])
    expected_scores[2] = torch.tensor([0.25, 0.75])
    assert torch.equal(scores, expected_scores)
    expected_input_grad, expected_weight_grad = torch.autograd.grad(
        live_scores.sum(),
        (inputs, router.gate.weight),
        retain_graph=True,
    )
    scores.sum().backward()
    assert inputs.grad is not None and bool((inputs.grad != 0).any().item())
    assert router.gate.weight.grad is not None
    assert bool((router.gate.weight.grad != 0).any().item())
    torch.testing.assert_close(inputs.grad, expected_input_grad)
    torch.testing.assert_close(router.gate.weight.grad, expected_weight_grad)
    assert router.router_replay is None

    (trace,) = probe.finish_forward()
    assert not probe.replay_active
    assert trace["global_token_ids"].tolist() == [10, 12]
    assert trace["topk_indices"].tolist() == [[1, 2], [2, 3]]
    assert torch.equal(trace["selected_scores"], scores.detach()[[2, 0]].float())
    assert torch.equal(trace["live_selected_scores"], live_scores.detach()[[2, 0]].float())


def test_replay_missing_live_token_fails_before_router_dispatch() -> None:
    router = _FakeTopKRouter()
    probe = ExpertRouteProbe()
    probe.begin_forward(replay_plans={3: _replay_plan([10], [[0, 1]])})

    with (
        pytest.raises(RuntimeError, match="missing token IDs.*12"),
        probe.capture(
            router,
            context=_context(),
            global_token_ids=torch.tensor([12]),
            padding_mask=torch.tensor([False]),
        ),
    ):
        pytest.fail("capture must fail before the native router runs")
    with pytest.raises(RuntimeError, match="coverage differs"):
        probe.finish_forward()
    assert not probe.replay_active


@pytest.mark.parametrize(
    ("plan", "message"),
    [
        (_replay_plan([10, 10], [[0, 1], [1, 2]]), "duplicate token IDs"),
        (_replay_plan([10], [[1, 1]]), "selected one expert twice"),
    ],
)
def test_replay_duplicate_identity_fails_closed(plan: ExpertRouteReplayPlan, message: str) -> None:
    probe = ExpertRouteProbe()
    with pytest.raises(ValueError, match=message):
        probe.begin_forward(replay_plans={3: plan})


@pytest.mark.parametrize(
    ("scores", "message"),
    [
        (torch.tensor([[0.5]], dtype=torch.float32), "align with Top-K"),
        (torch.tensor([[float("nan"), 0.5]]), "must be finite"),
        (torch.tensor([[-0.1, 1.1]]), "must be non-negative"),
        (torch.tensor([[1, 0]], dtype=torch.long), "floating dtype"),
    ],
)
def test_replay_score_contract_fails_closed(scores: torch.Tensor, message: str) -> None:
    plan = ExpertRouteReplayPlan(
        global_token_ids=torch.tensor([10], dtype=torch.long),
        topk_indices=torch.tensor([[0, 1]], dtype=torch.long),
        selected_scores=scores,
    )
    with pytest.raises((TypeError, ValueError), match=message):
        plan.validate()


def test_replay_requires_every_declared_logical_context_exactly_once() -> None:
    router = _FakeTopKRouter()
    probe = ExpertRouteProbe()
    probe.begin_forward(
        replay_plans={
            3: _replay_plan([10], [[0, 2]]),
            4: _replay_plan([10], [[1, 3]]),
        }
    )
    with probe.capture(
        router,
        context=_context(logical_layer=3),
        global_token_ids=torch.tensor([10]),
        padding_mask=torch.tensor([False]),
    ):
        router(torch.tensor([[1.0, 3.0]]))

    with pytest.raises(RuntimeError, match=r"missing=\[4\]"):
        probe.finish_forward()


def test_replay_rejects_duplicate_logical_invocation() -> None:
    router = _FakeTopKRouter()
    probe = ExpertRouteProbe()
    probe.begin_forward(replay_plans={3: _replay_plan([10], [[0, 2]])})
    kwargs = {
        "context": _context(),
        "global_token_ids": torch.tensor([10]),
        "padding_mask": torch.tensor([False]),
    }
    with probe.capture(router, **kwargs):
        router(torch.tensor([[1.0, 3.0]]))
    with pytest.raises(RuntimeError, match="ran more than once"), probe.capture(router, **kwargs):
        pytest.fail("duplicate logical context must fail before dispatch")
    probe.finish_forward()


def test_replay_distinguishes_logical_invocations_of_one_physical_router() -> None:
    router = _FakeTopKRouter()
    probe = ExpertRouteProbe()
    probe.begin_forward(
        replay_plans={
            3: _replay_plan([10], [[0, 2]]),
            4: _replay_plan([10], [[1, 3]]),
        }
    )
    observed = []
    for logical_layer in (3, 4):
        with probe.capture(
            router,
            context=_context(logical_layer=logical_layer),
            global_token_ids=torch.tensor([10]),
            padding_mask=torch.tensor([False]),
        ):
            _, indices = router(torch.tensor([[1.0, 3.0]]))
        observed.append(indices.tolist())

    assert observed == [[[0, 2]], [[1, 3]]]
    assert [trace["logical_layer_index"] for trace in probe.finish_forward()] == [3, 4]


def test_replay_plan_validate_and_to_preserve_integer_identity() -> None:
    plan = _replay_plan([12, 10], [[0, 3], [1, 2]])
    plan.validate()
    moved = plan.to(torch.device("cpu"))
    assert moved is not plan
    assert torch.equal(moved.global_token_ids, plan.global_token_ids)
    assert torch.equal(moved.topk_indices, plan.topk_indices)
    assert torch.equal(moved.selected_scores, plan.selected_scores)


def test_replay_rejects_topk_range_and_existing_native_replay() -> None:
    router = _FakeTopKRouter()
    probe = ExpertRouteProbe()
    probe.begin_forward(replay_plans={3: _replay_plan([10], [[0, 1, 2]])})
    with (
        pytest.raises(ValueError, match="Top-K mismatch"),
        probe.capture(
            router,
            context=_context(),
            global_token_ids=torch.tensor([10]),
            padding_mask=torch.tensor([False]),
        ),
    ):
        pytest.fail("shape validation must precede dispatch")
    with pytest.raises(RuntimeError, match="coverage differs"):
        probe.finish_forward()

    probe.begin_forward(replay_plans={3: _replay_plan([10], [[0, 4]])})
    with (
        pytest.raises(ValueError, match="out-of-range"),
        probe.capture(
            router,
            context=_context(),
            global_token_ids=torch.tensor([10]),
            padding_mask=torch.tensor([False]),
        ),
    ):
        pytest.fail("range validation must precede dispatch")
    with pytest.raises(RuntimeError, match="coverage differs"):
        probe.finish_forward()

    router.router_replay = object()
    probe.begin_forward(replay_plans={3: _replay_plan([10], [[0, 2]])})
    with (
        pytest.raises(RuntimeError, match="conflicts with an existing"),
        probe.capture(
            router,
            context=_context(),
            global_token_ids=torch.tensor([10]),
            padding_mask=torch.tensor([False]),
        ),
    ):
        pytest.fail("replay conflict must precede dispatch")
    assert router.router_replay is not None
    with pytest.raises(RuntimeError, match="coverage differs"):
        probe.finish_forward()


def test_replay_restores_router_state_when_native_forward_raises() -> None:
    class _FailingRouter(_FakeTopKRouter):
        def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            assert self.router_replay is not None
            raise LookupError("injected router failure")

    router = _FailingRouter()
    probe = ExpertRouteProbe()
    probe.begin_forward(replay_plans={3: _replay_plan([10], [[0, 2]])})
    with (
        pytest.raises(LookupError, match="injected"),
        probe.capture(
            router,
            context=_context(),
            global_token_ids=torch.tensor([10]),
            padding_mask=torch.tensor([False]),
        ),
    ):
        router(torch.tensor([[1.0, 3.0]]))
    assert router.router_replay is None
    with pytest.raises(RuntimeError, match="coverage differs"):
        probe.finish_forward()


def _trace(
    token_ids: list[int],
    indices: list[list[int]],
    *,
    logical_layer: int = 0,
) -> dict:
    return {
        "stage": "start",
        "round_index": -1,
        "stage_layer_index": 0,
        "physical_layer_index": 0,
        "logical_layer_index": logical_layer,
        "topk": 2,
        "num_experts": 4,
        "global_token_ids": torch.tensor(token_ids),
        "topk_indices": torch.tensor(indices),
        "selected_scores": torch.ones((len(token_ids), 2)),
        "live_selected_scores": torch.ones((len(token_ids), 2)),
        "cutoff_logit_margins": torch.full((len(token_ids),), 0.25),
    }


def _peer(batch_ids: list[int], trace: dict) -> dict:
    return {
        "batch_global_token_ids": torch.tensor(batch_ids),
        "expert_route_traces": [trace],
    }


def test_merge_expert_routes_is_canonical_across_topology_peers() -> None:
    from mor_mlite.parity.mlite import _merge_expert_route_traces

    peers = [
        _peer([12], _trace([12], [[1, 3]])),
        _peer([10], _trace([10], [[0, 2]])),
    ]
    (merged,) = _merge_expert_route_traces(peers, active_traces=[])
    assert merged["global_token_ids"].tolist() == [10, 12]
    assert merged["topk_indices"].tolist() == [[0, 2], [1, 3]]
    assert merged["live_selected_scores"].tolist() == [[1.0, 1.0], [1.0, 1.0]]


def test_merge_expert_routes_rejects_peer_disagreement() -> None:
    from mor_mlite.parity.mlite import _merge_expert_route_traces

    peers = [
        _peer([10], _trace([10], [[0, 2]])),
        _peer([10], _trace([10], [[1, 2]])),
    ]
    with pytest.raises(RuntimeError, match="disagree on native expert route"):
        _merge_expert_route_traces(peers, active_traces=[])


def test_record_capture_uses_phase_step_microbatch_and_logical_context() -> None:
    from mor_mlite.parity.mlite import _record_capture

    tensors: dict[str, torch.Tensor] = {}
    _record_capture(
        {
            "output": {},
            "expert_route_traces": [
                {
                    **_trace([10], [[0, 2]], logical_layer=4),
                    "stage": "recurrent",
                    "round_index": 1,
                    "physical_layer_index": 1,
                }
            ],
        },
        step=2,
        microbatch=3,
        prefix="resume/",
        tensors=tensors,
        routes=[],
    )
    root = "expert_route/resume/step_002/mb_003/logical_004/recurrent/round_001/physical_001"
    assert set(tensors) == {
        f"{root}/global_token_ids",
        f"{root}/topk_indices",
        f"{root}/selected_scores",
        f"{root}/live_selected_scores",
        f"{root}/cutoff_logit_margins",
    }


def test_record_capture_keeps_end_and_final_hidden_diagnostics_in_forward_namespace() -> None:
    from mor_mlite.parity.mlite import _record_capture

    final_hidden = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    hidden_for_head = final_hidden + 10
    attention_output = final_hidden + 20
    tensors: dict[str, torch.Tensor] = {}
    _record_capture(
        {
            "output": {
                "mor_diagnostic_final_hidden": final_hidden,
                "mor_diagnostic_hidden_for_head": hidden_for_head,
                "mor_diagnostic_end_sublayer_0_attention_output": attention_output,
            }
        },
        step=1,
        microbatch=2,
        tensors=tensors,
        routes=[],
    )

    assert tensors == {
        "forward/step_001/mb_002/final_hidden": final_hidden,
        "forward/step_001/mb_002/hidden_for_head": hidden_for_head,
        "forward/step_001/mb_002/end_sublayer_0_attention_output": attention_output,
    }


def test_serial_virtual_dp_peers_merge_back_to_global_token_order() -> None:
    from mor_mlite.config import DepthRouterConfig
    from mor_mlite.parity.mlite import _merge_diagnostics

    def peer(dp_rank: int, token_id: int, value: float) -> dict:
        return {
            "dp_rank": dp_rank,
            "dp_size": 2,
            "tp_rank": 0,
            "cp_rank": 0,
            "output": {"logits": torch.tensor([[value]], dtype=torch.float32)},
            "plans": [],
            "traces": [],
            "expert_route_traces": [],
            "communication": {},
            "batch_global_token_ids": torch.tensor([token_id]),
            "batch_sample_ids": torch.tensor([dp_rank]),
            "batch_original_positions": torch.tensor([0]),
            "lm_scale": 1.0,
            "aux_scales": (),
            "local_valid_tokens": 1,
            "global_valid_tokens": 2,
            "global_input_tokens": 2,
            "expected_global_token_ids": torch.tensor([3, 10]),
        }

    merged = _merge_diagnostics(
        [peer(0, 10, 1.0), peer(1, 3, 2.0)],
        router=DepthRouterConfig(),
        training=False,
    )

    assert torch.equal(merged["output"]["logits"], torch.tensor([[2.0], [1.0]]))

    overlapping = [peer(0, 3, 2.0), peer(1, 3, 2.0)]
    overlapping[1]["batch_sample_ids"] = torch.tensor([0])
    with pytest.raises(RuntimeError, match="token shards overlap"):
        _merge_diagnostics(
            overlapping,
            router=DepthRouterConfig(),
            training=False,
        )

    rogue = [peer(0, 10, 1.0), peer(1, 99, 2.0)]
    with pytest.raises(RuntimeError, match="expected global token-ID universe"):
        _merge_diagnostics(
            rogue,
            router=DepthRouterConfig(),
            training=False,
        )


def test_mlite_probe_configuration_is_explicit() -> None:
    from mor_mlite.parity.mlite import _set_moe_expert_route_probe

    calls: list[bool] = []
    model = SimpleNamespace(
        mor_architecture=object(),
        set_moe_expert_route_probe=lambda enabled: calls.append(bool(enabled)),
    )
    handle = SimpleNamespace(_extras={"model_chunks": [model]})
    assert _set_moe_expert_route_probe(handle, enabled=True)
    assert calls == [True]


def test_mlite_round_diagnostic_capture_is_explicit() -> None:
    from mor_mlite.parity.mlite import _set_mor_diagnostic_capture

    calls: list[bool] = []
    model = SimpleNamespace(
        mor_architecture=object(),
        set_mor_diagnostic_capture=lambda enabled: calls.append(bool(enabled)),
    )
    handle = SimpleNamespace(_extras={"model_chunks": [model]})
    assert _set_mor_diagnostic_capture(handle, enabled=True)
    assert calls == [True]


def test_active_trace_merge_accepts_an_empty_topology_shard() -> None:
    from mor_mlite.config import DepthRouterConfig
    from mor_mlite.parity.mlite import _merge_active_traces

    def trace(token_ids: list[int], hidden: torch.Tensor) -> dict:
        return {
            "global_token_ids": torch.tensor(token_ids, dtype=torch.long),
            "hidden": hidden,
            "sample_ids": torch.zeros(len(token_ids), dtype=torch.long),
            "original_positions": torch.tensor(token_ids, dtype=torch.long),
            "candidate_global_token_ids": torch.tensor(token_ids, dtype=torch.long),
            "router_logits": torch.ones(len(token_ids)),
            "selected_global_token_ids": torch.tensor(token_ids, dtype=torch.long),
            "selected_gates": torch.full((len(token_ids),), 0.1),
        }

    peers = [
        {
            "batch_global_token_ids": torch.tensor([7]),
            "batch_sample_ids": torch.tensor([0]),
            "batch_original_positions": torch.tensor([7]),
            "traces": [trace([], torch.empty((0, 4)))],
        },
        {
            "batch_global_token_ids": torch.tensor([7]),
            "batch_sample_ids": torch.tensor([0]),
            "batch_original_positions": torch.tensor([7]),
            "traces": [trace([7], torch.arange(4, dtype=torch.float32).reshape(1, 4))],
        },
    ]

    (merged,) = _merge_active_traces(peers, DepthRouterConfig())
    assert merged["global_token_ids"].tolist() == [7]
    assert merged["hidden"].tolist() == [[0.0, 1.0, 2.0, 3.0]]
