"""Execution policies independently testable without Megatron or CUDA."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def run_fixed(hidden, layers, layer_indices, *, position_ids, packed_seq_params):
    for index in layer_indices:
        hidden = layers[index](
            hidden, position_ids=position_ids, packed_seq_params=packed_seq_params
        )
    return hidden


def run_causal(
    hidden,
    *,
    layers,
    routers,
    sample_ids,
    positions,
    padding_mask,
    position_ids,
    packed_seq_params,
    make_packed,
    start=3,
    recurrent=14,
    end=3,
):
    """Threshold-only inference. No length-dependent Top-K or minimum capacity.

    Every rank executes every layer, even when no tokens remain locally:
    a discarded dummy row preserves EP collective order without retaining a
    real token or introducing a future-dependent capacity decision.
    """
    if torch.is_grad_enabled():
        raise ValueError("causal threshold policy is evaluation-only")
    h = run_fixed(
        hidden, layers, range(start), position_ids=position_ids, packed_seq_params=packed_seq_params
    )
    shape = h.shape
    flat = h.reshape(-1, h.shape[-1])
    rows = torch.nonzero(~padding_mask, as_tuple=False).flatten()
    traces = []
    for round_index, router in enumerate(routers):
        candidates = rows
        logits = F.linear(flat[rows].float(), router.proj.weight.float()).squeeze(-1)
        logits = logits / router.config.temperature
        if not torch.isfinite(logits).all():
            raise ValueError("nonfinite causal router logits")
        keep = torch.ones_like(logits, dtype=torch.bool) if round_index == 0 else logits >= 0
        rows = rows[keep]
        gates = router.config.alpha * torch.sigmoid(logits[keep])
        traces.append(
            {
                "round": round_index,
                "candidate_rows": candidates.detach().clone(),
                "selected_rows": rows.detach().clone(),
            }
        )
        if rows.numel():
            # Protocol metadata is in sample/position order at TP=CP=1.
            ids = sample_ids[rows]
            _, counts = torch.unique_consecutive(ids, return_counts=True)
            params = make_packed(counts)
            active = flat[rows].unsqueeze(1)
            active_positions = positions[rows].unsqueeze(0)
        else:
            params = make_packed(torch.ones(1, dtype=torch.int64, device=flat.device))
            active = flat.new_zeros((1, 1, flat.shape[-1]))
            active_positions = torch.zeros((1, 1), device=flat.device, dtype=torch.long)
        out = run_fixed(
            active,
            layers,
            range(start, start + recurrent),
            position_ids=active_positions,
            packed_seq_params=params,
        )
        if rows.numel():
            update = flat[rows] + gates.to(flat.dtype).unsqueeze(-1) * out[:, 0]
            flat = flat.index_copy(0, rows, update)
    merged = flat.reshape(shape)
    h = run_fixed(
        merged,
        layers,
        range(start + recurrent, start + recurrent + end),
        position_ids=position_ids,
        packed_seq_params=packed_seq_params,
    )
    return h, traces
