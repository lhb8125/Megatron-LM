"""Required evidence for complete acceptance, independent of observed tensor names."""

from __future__ import annotations

from typing import Any


def check_evidence(metadata: dict, tensors: dict, routes: list, *, scope: str) -> dict[str, Any]:
    details: list[str] = []
    if metadata.get("operator_probe") is not None:
        details.append(
            "injected operator-oracle artifacts are diagnostic, never acceptance evidence"
        )
    source = metadata.get("source_snapshot", {}).get("sha256")
    if not isinstance(source, str) or len(source) != 64:
        details.append("source snapshot identity is missing")
    counts = [
        metadata.get("steps"),
        metadata.get("num_microbatches"),
        metadata.get("architecture", {}).get("num_recursions"),
    ]
    if any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in counts):
        return {
            "passed": False,
            "details": [
                "steps, num_microbatches and num_recursions are required positive integers"
            ],
        }
    steps, microbatches, rounds = counts
    phase_steps = [("train", step) for step in range(steps)]
    if metadata.get("checkpoint_next_step"):
        phase_steps.append(("resume", steps))
    if metadata.get("checkpoint_uninterrupted_step"):
        phase_steps.append(("uninterrupted", steps))
    expected_routes = {
        (phase, step, mb, r)
        for phase, step in phase_steps
        for mb in range(microbatches)
        for r in range(rounds)
    }
    actual_routes = {
        (r.get("phase", "train"), r.get("step", 0), r.get("microbatch", 0), r.get("round"))
        for r in routes
    }
    if actual_routes != expected_routes or len(routes) != len(expected_routes):
        details.append(
            f"RoutePlan coverage differs: missing={sorted(expected_routes - actual_routes)}, unexpected={sorted(actual_routes - expected_routes)}"
        )
    required: set[str] = set()
    for phase, step in phase_steps:
        prefix = "" if phase == "train" else f"{phase}/"
        base = f"{prefix}step_{step:03d}"
        for mb in range(microbatches):
            path = f"{base}/mb_{mb:03d}"
            required.add(f"forward/{path}/logits")
            # labels=None inference has no LM loss, but must retain all forward evidence.
            if not metadata.get("forward_only"):
                required.update(f"loss/{path}/{kind}" for kind in ("lm", "aux", "total"))
            for r in range(rounds):
                required.update(
                    f"forward/{path}/{kind}_{r}"
                    for kind in ("hidden_round", "router_scores_round", "selected_gates_round")
                )
    training = scope == "all"
    if training:
        if metadata.get("forward_only"):
            details.append("full training acceptance cannot use a forward-only artifact")
        parameters = metadata.get("initialized_parameters")
        if (
            not isinstance(parameters, list)
            or not parameters
            or any(not isinstance(p, str) for p in parameters)
            or len(set(parameters)) != len(parameters)
        ):
            details.append("initialized physical parameter inventory is missing or invalid")
            parameters = []
        required.update(f"initial/{name}" for name in parameters)
        initial = {name.removeprefix("initial/") for name in tensors if name.startswith("initial/")}
        if initial != set(parameters):
            details.append("initial parameter tensors differ from initialized parameter inventory")
        for phase, step in phase_steps:
            prefix = "" if phase == "train" else f"{phase}/"
            path = f"{prefix}step_{step:03d}"
            for namespace in ("gradient", "update", "post_step"):
                required.update(f"{namespace}/{path}/{name}" for name in parameters)
            required.add(f"gradient/{path}/global_norm")
    missing = sorted(required - tensors.keys())
    if missing:
        details.append(f"missing {len(missing)} required tensors: {missing[:12]}")
    empty = sorted(name for name in required & tensors.keys() if tensors[name].numel() == 0)
    if empty:
        details.append(f"empty required tensors: {empty[:12]}")
    return {
        "passed": not details,
        "details": details,
        "required_tensor_count": len(required),
        "required_route_count": len(expected_routes),
    }
