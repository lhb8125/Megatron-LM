"""Compare one distributed artifact against the single-rank oracle."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from mor_mlite.provenance import file_sha256

from .artifacts import load_artifact
from .continuity import check_checkpoint_continuity
from .evidence import check_evidence
from .metrics import ComparisonResult, classify_cutoff, compare_tensor

_NAMESPACES = (
    "initial",
    "forward",
    "loss",
    "expert_route",
    "gradient",
    "update",
    "post_step",
)
_STEP_PREFIX = re.compile(r"^(?:(?P<phase>[^/]+)/)?step_(?P<step>\d+)(?:/|$)")
_EXPERT_ROUTE_FIELDS = frozenset(
    {
        "global_token_ids",
        "topk_indices",
        "selected_scores",
        "live_selected_scores",
        "cutoff_logit_margins",
    }
)
_EXPERT_ROUTE_IDENTITY_FIELDS = frozenset({"global_token_ids", "topk_indices"})


def _namespace(name: str) -> str:
    """Return the semantic artifact namespace, retaining legacy aliases."""

    prefix = name.split("/", 1)[0].lower()
    if prefix == "grad":
        return "gradient"
    if prefix == "weight_after":
        return "post_step"
    if prefix in _NAMESPACES:
        return prefix
    # Preserve the historical behavior for extension tensors: an unknown
    # namespace describes an intermediate unless it is explicitly registered.
    return "forward"


def _kind(name: str) -> str:
    namespace = _namespace(name)
    if namespace == "initial":
        # Initial tensors have a dedicated bitwise comparison path.
        return "forward"
    return namespace


def _phase_step(name: str) -> tuple[str, int | None]:
    """Parse ``namespace/[phase/]step_NNN/...`` into an aggregation key."""

    tail = name.split("/", 1)[1] if "/" in name else ""
    match = _STEP_PREFIX.match(tail)
    if match is None:
        return "unscoped", None
    return match.group("phase") or "train", int(match.group("step"))


def _is_finite(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor.detach()).all())


def _bitwise_tensor_result(
    name: str,
    baseline: torch.Tensor,
    candidate: torch.Tensor,
    *,
    subject: str,
) -> dict[str, Any]:
    """Compare a tensor's shape, dtype, finiteness, and raw bytes."""

    if baseline.shape != candidate.shape:
        return ComparisonResult(
            name=name,
            passed=False,
            max_abs=float("inf"),
            relative_l2=float("inf"),
            cosine_similarity=-1.0,
            detail=f"shape mismatch: {tuple(baseline.shape)} != {tuple(candidate.shape)}",
        ).to_dict()
    if baseline.dtype != candidate.dtype:
        return ComparisonResult(
            name=name,
            passed=False,
            max_abs=float("inf"),
            relative_l2=float("inf"),
            cosine_similarity=-1.0,
            detail=f"dtype mismatch: {baseline.dtype} != {candidate.dtype}",
        ).to_dict()
    if not (_is_finite(baseline) and _is_finite(candidate)):
        return ComparisonResult(
            name=name,
            passed=False,
            max_abs=float("inf"),
            relative_l2=float("inf"),
            cosine_similarity=-1.0,
            detail=f"{subject} tensor contains non-finite values",
        ).to_dict()
    lhs_bytes = baseline.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    rhs_bytes = candidate.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    exact = bool(torch.equal(lhs_bytes, rhs_bytes))
    if exact:
        return ComparisonResult(
            name=name,
            passed=True,
            max_abs=0.0,
            relative_l2=0.0,
            cosine_similarity=1.0,
            detail=f"{subject} bitwise exact (shape, dtype, and tensor bytes)",
        ).to_dict()
    result = compare_tensor(name, baseline, candidate, precision="fp32", kind="forward").to_dict()
    result["passed"] = False
    result["detail"] = f"{subject} tensor bytes differ"
    return result


def _bitwise_initial_result(
    name: str, baseline: torch.Tensor, candidate: torch.Tensor
) -> dict[str, Any]:
    """Initial state must match in shape, dtype, and raw tensor bytes."""

    return _bitwise_tensor_result(name, baseline, candidate, subject="initial")


def _is_aggregate_member(name: str, namespace: str) -> bool:
    # global_norm is a derived scalar, not part of the reconstructed parameter
    # gradient vector promised by the acceptance contract.
    return namespace in {"gradient", "update"} and not (
        namespace == "gradient" and name.rsplit("/", 1)[-1] == "global_norm"
    )


def _aggregate_full_vector(
    *,
    namespace: str,
    phase: str,
    step: int | None,
    pairs: list[tuple[str, torch.Tensor, torch.Tensor]],
) -> dict[str, Any]:
    """Compute full-vector metrics without concatenating all model tensors."""

    lhs_sq = 0.0
    rhs_sq = 0.0
    diff_sq = 0.0
    dot = 0.0
    max_abs = 0.0
    numel = 0
    invalid: list[str] = []
    for name, baseline, candidate in pairs:
        if baseline.shape != candidate.shape:
            invalid.append(f"{name}: shape mismatch")
            continue
        if not (_is_finite(baseline) and _is_finite(candidate)):
            invalid.append(f"{name}: non-finite values")
            continue
        lhs = baseline.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
        rhs = candidate.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
        difference = rhs - lhs
        if lhs.numel():
            max_abs = max(max_abs, float(difference.abs().max().item()))
            lhs_sq += float(torch.dot(lhs, lhs).item())
            rhs_sq += float(torch.dot(rhs, rhs).item())
            diff_sq += float(torch.dot(difference, difference).item())
            dot += float(torch.dot(lhs, rhs).item())
            numel += lhs.numel()

    lhs_norm = math.sqrt(max(lhs_sq, 0.0))
    rhs_norm = math.sqrt(max(rhs_sq, 0.0))
    relative_l2 = math.sqrt(max(diff_sq, 0.0)) / max(lhs_norm, 1e-30)
    if lhs_norm == 0.0 and rhs_norm == 0.0:
        cosine = 1.0
    elif lhs_norm == 0.0 or rhs_norm == 0.0:
        cosine = 0.0
    else:
        cosine = dot / (lhs_norm * rhs_norm)
        # Roundoff in the streaming dot products can exceed the mathematical
        # range by a few ulps.
        cosine = min(1.0, max(-1.0, cosine))
    passed = not invalid and numel > 0 and relative_l2 <= 0.03 and cosine >= 0.999
    step_label = "unscoped" if step is None else f"step_{step:03d}"
    detail = "aggregate relative_l2 <= 0.03 and cosine >= 0.999"
    if invalid:
        detail += "; " + "; ".join(invalid)
    elif numel == 0:
        detail += "; aggregate has no parameter elements"
    return {
        "name": f"{namespace}/{phase}/{step_label}",
        "namespace": namespace,
        "phase": phase,
        "step": step,
        "passed": passed,
        "hard_gate": True,
        "tensor_count": len(pairs),
        "numel": numel,
        "max_abs": max_abs if not invalid else float("inf"),
        "relative_l2": relative_l2 if not invalid else float("inf"),
        "cosine_similarity": cosine if not invalid else -1.0,
        "detail": detail,
    }


def _namespace_summary(
    tensor_results: list[dict[str, Any]], aggregate_results: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for namespace in _NAMESPACES:
        items = [item for item in tensor_results if item["namespace"] == namespace]
        aggregates = [item for item in aggregate_results if item["namespace"] == namespace]
        hard_failures = [item["name"] for item in items if item["hard_gate"] and not item["passed"]]
        diagnostic_outliers = [
            item["name"] for item in items if not item["hard_gate"] and not item["passed"]
        ]
        aggregate_failures = [item["name"] for item in aggregates if not item["passed"]]
        summaries[namespace] = {
            "tensor_count": len(items),
            "hard_failures": hard_failures,
            "diagnostic_outliers": diagnostic_outliers,
            "aggregate_count": len(aggregates),
            "aggregate_failures": aggregate_failures,
            "passed": not hard_failures and not aggregate_failures,
        }
    return summaries


def _expert_route_contexts(
    tensors: dict[str, torch.Tensor],
) -> dict[str, set[str]]:
    contexts: dict[str, set[str]] = defaultdict(set)
    for name in tensors:
        if not name.startswith("expert_route/") or "/" not in name:
            continue
        context, field = name.rsplit("/", 1)
        contexts[context].add(field)
    return dict(contexts)


def _default_tiny_expert_routes_required(metadata: dict[str, Any]) -> bool:
    parameter_capture = metadata.get("parameter_capture")
    return bool(
        metadata.get("backend") == "mlite"
        and metadata.get("preset") == "tiny"
        and isinstance(parameter_capture, dict)
        and parameter_capture.get("enabled") is True
    )


def _check_expert_routes(
    baseline_meta: dict[str, Any],
    candidate_meta: dict[str, Any],
    baseline_tensors: dict[str, torch.Tensor],
    candidate_tensors: dict[str, torch.Tensor],
    tensor_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate native MoE routing identity and probe provenance."""

    baseline_probe_raw = baseline_meta.get("expert_route_probe")
    candidate_probe_raw = candidate_meta.get("expert_route_probe")
    baseline_probe = baseline_probe_raw if isinstance(baseline_probe_raw, dict) else {}
    candidate_probe = candidate_probe_raw if isinstance(candidate_probe_raw, dict) else {}
    baseline_enabled = baseline_probe.get("enabled") is True
    candidate_enabled = candidate_probe.get("enabled") is True
    baseline_required = _default_tiny_expert_routes_required(baseline_meta)
    candidate_required = _default_tiny_expert_routes_required(candidate_meta)
    required = baseline_required or candidate_required
    candidate_replay_raw = candidate_meta.get("expert_route_replay")
    replay_required = bool(
        candidate_meta.get("backend") == "mlite"
        and candidate_meta.get("route_mode") == "replay"
        and candidate_meta.get("forward_only") is True
    )
    replay_enabled = bool(
        isinstance(candidate_replay_raw, dict) and candidate_replay_raw.get("enabled") is True
    )

    baseline_contexts = _expert_route_contexts(baseline_tensors)
    candidate_contexts = _expert_route_contexts(candidate_tensors)
    has_probe_evidence = bool(
        required
        or baseline_probe_raw is not None
        or candidate_probe_raw is not None
        or baseline_contexts
        or candidate_contexts
    )
    metadata_details: list[str] = []
    details: list[str] = []
    if replay_required:
        if not replay_enabled:
            metadata_details.append("forward-only MLite replay requires native expert-route replay")
        else:
            expected_replay_contract = {
                "scope": "forward-only-cross-topology-oracle",
                "identity_key": "logical-layer-plus-global-token-id",
                "expert_ids": "baseline-bitwise-exact",
                "selected_scores": "baseline-forward-values-with-live-gradient-ste",
                "live_selected_scores": "diagnostic-only-live-router-values",
                "training_native_router_unchanged": True,
            }
            for field, expected in expected_replay_contract.items():
                if candidate_replay_raw.get(field) != expected:
                    metadata_details.append(f"native expert-route replay {field} contract differs")
    if has_probe_evidence:
        if not isinstance(baseline_probe_raw, dict):
            metadata_details.append("baseline expert_route_probe metadata is missing")
        if not isinstance(candidate_probe_raw, dict):
            metadata_details.append("candidate expert_route_probe metadata is missing")
        for label, raw in (
            ("baseline", baseline_probe_raw),
            ("candidate", candidate_probe_raw),
        ):
            if isinstance(raw, dict) and type(raw.get("enabled")) is not bool:
                metadata_details.append(f"{label} expert-route enabled flag is not boolean")
        if baseline_required != candidate_required:
            metadata_details.append("default-tiny expert-route requirement differs")
        if baseline_enabled != candidate_enabled:
            metadata_details.append("expert-route probe enabled state differs")
        if required and not (baseline_enabled and candidate_enabled):
            metadata_details.append("default tiny acceptance requires expert-route probes")
        for field in (
            "scope",
            "identity_contract",
            "topk_indices",
            "cutoff_margin",
            "dummy_padding_excluded",
        ):
            if baseline_probe.get(field) != candidate_probe.get(field):
                metadata_details.append(f"expert-route probe {field} metadata differs")

    for label, probe, contexts, enabled in (
        ("baseline", baseline_probe, baseline_contexts, baseline_enabled),
        ("candidate", candidate_probe, candidate_contexts, candidate_enabled),
    ):
        captured = probe.get("captured_contexts")
        if enabled:
            if type(captured) is not int or captured != len(contexts):
                metadata_details.append(
                    f"{label} captured_contexts does not match artifact contexts"
                )
            if not contexts:
                details.append(f"{label} expert-route probe captured no contexts")
        elif contexts:
            details.append(f"{label} has expert-route tensors while its probe is disabled")

    if baseline_contexts.keys() != candidate_contexts.keys():
        details.append("baseline and candidate expert-route contexts differ")
    for label, contexts in (("baseline", baseline_contexts), ("candidate", candidate_contexts)):
        for context, fields in sorted(contexts.items()):
            if fields != _EXPERT_ROUTE_FIELDS:
                missing_fields = sorted(_EXPERT_ROUTE_FIELDS - fields)
                extra_fields = sorted(fields - _EXPERT_ROUTE_FIELDS)
                details.append(
                    f"{label} context {context} has incomplete fields: "
                    f"missing={missing_fields}, extra={extra_fields}"
                )

    expert_results = [item for item in tensor_results if item["namespace"] == "expert_route"]
    results_by_name = {str(item["name"]): item for item in expert_results}
    exact_identity_count = 0
    exact_identity_tensor_count = 0
    for context in sorted(baseline_contexts.keys() & candidate_contexts.keys()):
        context_exact = True
        for field in sorted(_EXPERT_ROUTE_IDENTITY_FIELDS):
            result = results_by_name.get(f"{context}/{field}")
            exact = result is not None and bool(result["passed"])
            exact_identity_tensor_count += int(exact)
            context_exact = context_exact and exact
        exact_identity_count += int(context_exact)

    cutoff_values: list[float] = []
    for name, tensor in baseline_tensors.items():
        if not (name.startswith("expert_route/") and name.endswith("/cutoff_logit_margins")):
            continue
        if tensor.numel() == 0:
            details.append(f"baseline cutoff margin tensor is empty: {name}")
        elif not _is_finite(tensor):
            details.append(f"baseline cutoff margin tensor is non-finite: {name}")
        else:
            cutoff_values.append(float(tensor.detach().float().min().item()))
    baseline_minimum_cutoff_margin = min(cutoff_values) if cutoff_values else None
    if required and (
        baseline_minimum_cutoff_margin is None or baseline_minimum_cutoff_margin <= 0.5
    ):
        details.append("default tiny baseline minimum expert cutoff-logit margin must be > 0.5")

    metadata_compatible = not metadata_details
    details = [*metadata_details, *details]
    tensor_checks_passed = all(
        bool(item["passed"]) for item in expert_results if bool(item["hard_gate"])
    )
    return {
        "required": required,
        "baseline_required": baseline_required,
        "candidate_required": candidate_required,
        "enabled": baseline_enabled and candidate_enabled,
        "baseline_enabled": baseline_enabled,
        "candidate_enabled": candidate_enabled,
        "enabled_compatible": baseline_enabled == candidate_enabled,
        "metadata_compatible": metadata_compatible,
        "context_count": len(baseline_contexts),
        "baseline_context_count": len(baseline_contexts),
        "candidate_context_count": len(candidate_contexts),
        "exact_identity_count": exact_identity_count,
        "exact_identity_tensor_count": exact_identity_tensor_count,
        "baseline_minimum_cutoff_margin": baseline_minimum_cutoff_margin,
        "minimum_cutoff_margin_required": 0.5 if required else None,
        "replay_required": replay_required,
        "replay_enabled": replay_enabled,
        "passed": metadata_compatible and not details and tensor_checks_passed,
        "details": details,
    }


def _route_key(route: dict[str, Any]) -> tuple[str, int, int, int]:
    return (
        str(route.get("phase", "train")),
        int(route.get("step", 0)),
        int(route.get("microbatch", 0)),
        int(route.get("round", route.get("round_idx", -1))),
    )


_CANONICAL_ROUTE_FIELDS = (
    "round",
    "sample_ids",
    "original_positions",
    "global_token_ids",
    "active_cu_seqlens",
    "padding_mask",
)


def _canonical_route(
    route: dict[str, Any], *, include_selected_gates: bool = True
) -> dict[str, Any]:
    """Strip topology-specific ownership from a serialized RoutePlan.

    Replay is a bitwise oracle, so its stored gate values are part of the
    canonical plan. Learned routing compares gates through the ordinary
    forward-tensor tolerance; requiring their JSON floats to be bitwise equal
    would incorrectly reject harmless topology-dependent BF16 roundoff even
    when the selected IDs and all discrete metadata agree.
    """

    canonical = {field: route.get(field) for field in _CANONICAL_ROUTE_FIELDS}
    if include_selected_gates:
        canonical["selected_gates"] = route.get("selected_gates")
    return canonical


def _check_metadata_compatibility(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Reject artifacts that cannot describe the same numerical experiment."""

    details: list[str] = []
    if baseline.get("precision") != candidate.get("precision"):
        details.append(
            "precision differs: "
            f"baseline={baseline.get('precision')!r}, candidate={candidate.get('precision')!r}"
        )
    for field in (
        "backend",
        "source_snapshot",
        "precision_probe",
        "strict",
        "attention_policy",
        "preset",
        "seed",
        "steps",
        "num_microbatches",
        "architecture",
        "diagnostic_capture",
        "model_config",
        "optimizer",
        "synthetic_weight_profile",
        "hf_source",
        "megatron_lm_sha",
    ):
        lhs = baseline.get(field)
        rhs = candidate.get(field)
        if lhs != rhs and (
            field in {"source_snapshot", "precision_probe", "strict", "attention_policy"}
            or (lhs is not None and rhs is not None)
        ):
            details.append(f"{field} differs between baseline and candidate")
    lhs_batch = baseline.get("global_batch")
    rhs_batch = candidate.get("global_batch")
    if (
        isinstance(lhs_batch, dict)
        and isinstance(rhs_batch, dict)
        and lhs_batch.get("sequence_lengths") != rhs_batch.get("sequence_lengths")
    ):
        details.append("global batch sequence lengths differ")
    reference_dp_shards = baseline.get("reference_dp_shards", 1)
    if (
        isinstance(reference_dp_shards, bool)
        or not isinstance(reference_dp_shards, int)
        or reference_dp_shards < 1
    ):
        details.append("baseline reference_dp_shards metadata is invalid")
    elif reference_dp_shards > 1:
        candidate_topology = candidate.get("topology")
        candidate_dp = (
            candidate_topology.get("dp") if isinstance(candidate_topology, dict) else None
        )
        if candidate_dp != reference_dp_shards:
            details.append(
                "serial baseline reference DP differs from candidate dense-DP: "
                f"{reference_dp_shards} != {candidate_dp}"
            )
        expected_policy = "deterministic-longest-first-whole-sequence"
        if not isinstance(lhs_batch, dict) or lhs_batch.get("partition_policy") != expected_policy:
            details.append("serial baseline partition policy is incompatible")
        if not isinstance(rhs_batch, dict) or rhs_batch.get("partition_policy") != expected_policy:
            details.append("candidate partition policy is incompatible with serial baseline")
        if (
            isinstance(lhs_batch, dict)
            and isinstance(rhs_batch, dict)
            and lhs_batch.get("sample_partitions") != rhs_batch.get("sample_partitions")
        ):
            details.append("serial baseline and candidate sample partitions differ")
    return {"passed": not details, "details": details}


def compare_artifacts(
    baseline_path: str | Path,
    candidate_path: str | Path,
    *,
    report_path: str | Path | None = None,
    scope: str = "all",
    diagnostic: bool = False,
) -> dict[str, Any]:
    if scope not in {"all", "forward"}:
        raise ValueError("comparison scope must be 'all' or 'forward'")
    baseline_meta, baseline_tensors, baseline_routes = load_artifact(baseline_path)
    candidate_meta, candidate_tensors, candidate_routes = load_artifact(candidate_path)
    evidence = {
        "baseline": check_evidence(baseline_meta, baseline_tensors, baseline_routes, scope=scope),
        "candidate": check_evidence(
            candidate_meta, candidate_tensors, candidate_routes, scope=scope
        ),
    }
    evidence["passed"] = all(item["passed"] for item in evidence.values())
    compatibility = _check_metadata_compatibility(baseline_meta, candidate_meta)
    precision = str(baseline_meta.get("precision", "fp32")).lower()
    if precision not in {"fp32", "bf16"}:
        raise ValueError(f"unsupported precision in baseline manifest: {precision}")

    def in_scope(name: str) -> bool:
        # ``_kind`` intentionally treats unknown tensor namespaces as forward
        # for tolerance selection.  A forward-only smoke comparison must be
        # narrower: initial weights, gradients, updates, and checkpoint-resume
        # diagnostics from a training candidate are not part of the one-card
        # forward baseline contract. Native expert routes are forward evidence.
        return scope == "all" or name.startswith(("forward/", "loss/", "expert_route/"))

    baseline_names = {name for name in baseline_tensors if in_scope(name)}
    candidate_names = {name for name in candidate_tensors if in_scope(name)}
    missing = sorted(baseline_names - candidate_names)
    unexpected = sorted(candidate_names - baseline_names) if scope == "all" else []
    ignored_candidate_tensors = [] if scope == "all" else sorted(candidate_names - baseline_names)
    candidate_expert_replay = candidate_meta.get("expert_route_replay")
    expert_score_replay_enabled = bool(
        isinstance(candidate_expert_replay, dict) and candidate_expert_replay.get("enabled") is True
    )
    tensor_results: list[dict[str, Any]] = []
    aggregate_groups: dict[
        tuple[str, str, int | None], list[tuple[str, torch.Tensor, torch.Tensor]]
    ] = defaultdict(list)
    for name in sorted(baseline_names & candidate_names):
        namespace = _namespace(name)
        baseline_tensor = baseline_tensors[name]
        candidate_tensor = candidate_tensors[name]
        if namespace == "initial":
            result = _bitwise_initial_result(name, baseline_tensor, candidate_tensor)
        elif (
            namespace == "expert_route" and name.rsplit("/", 1)[-1] in _EXPERT_ROUTE_IDENTITY_FIELDS
        ):
            result = _bitwise_tensor_result(
                name,
                baseline_tensor,
                candidate_tensor,
                subject="expert-route identity",
            )
        elif (
            namespace == "expert_route"
            and name.endswith("/selected_scores")
            and expert_score_replay_enabled
        ):
            result = _bitwise_tensor_result(
                name,
                baseline_tensor,
                candidate_tensor,
                subject="expert-route replay score",
            )
        else:
            result = compare_tensor(
                name,
                baseline_tensor,
                candidate_tensor,
                precision=precision,
                kind="forward" if namespace == "expert_route" else _kind(name),
            ).to_dict()
        aggregate_member = _is_aggregate_member(name, namespace)
        structurally_valid = (
            baseline_tensor.shape == candidate_tensor.shape
            and _is_finite(baseline_tensor)
            and _is_finite(candidate_tensor)
        )
        # BF16 training parity is judged over the complete reconstructed
        # parameter vector. Individual parameter excursions remain visible,
        # while malformed tensors still fail closed.
        diagnostic_only = bool(
            structurally_valid
            and (
                (precision == "bf16" and aggregate_member)
                # Native-MoE cutoff margins explain discrete Top-K
                # sensitivity but do not participate in model output.  In a
                # replay run expert identity is the hard contract and the
                # replayed scores are a hard forward contract.  The separately
                # captured live scores and near-zero K/(K+1) margins explain
                # sensitivity but do not participate in replayed execution.
                or (
                    namespace == "expert_route"
                    and name.endswith(("/live_selected_scores", "/cutoff_logit_margins"))
                )
            )
        )
        result.update(
            {
                "namespace": namespace,
                "hard_gate": not diagnostic_only,
                "diagnostic_only": diagnostic_only,
            }
        )
        tensor_results.append(result)
        if precision == "bf16" and aggregate_member:
            phase, step = _phase_step(name)
            aggregate_groups[(namespace, phase, step)].append(
                (name, baseline_tensor, candidate_tensor)
            )

    aggregate_results = [
        _aggregate_full_vector(
            namespace=namespace,
            phase=phase,
            step=step,
            pairs=pairs,
        )
        for (namespace, phase, step), pairs in sorted(
            aggregate_groups.items(),
            key=lambda item: (
                item[0][0],
                item[0][1],
                -1 if item[0][2] is None else item[0][2],
            ),
        )
    ]
    namespace_summary = _namespace_summary(tensor_results, aggregate_results)
    expert_routes = _check_expert_routes(
        baseline_meta,
        candidate_meta,
        baseline_tensors,
        candidate_tensors,
        tensor_results,
    )

    if scope == "forward":
        baseline_route_keys = {_route_key(route) for route in baseline_routes}
        candidate_routes = [
            route for route in candidate_routes if _route_key(route) in baseline_route_keys
        ]
    route_mode = str(candidate_meta.get("route_mode", baseline_meta.get("route_mode", "replay")))
    if route_mode == "replay":
        baseline_by_key = {_route_key(route): _canonical_route(route) for route in baseline_routes}
        candidate_by_key = {
            _route_key(route): _canonical_route(route) for route in candidate_routes
        }
        routes_equal = baseline_by_key == candidate_by_key
        route_result = {
            "passed": routes_equal,
            "mode": "replay",
            "near_ties": 0,
            "total": len(baseline_routes),
            "details": [] if routes_equal else ["canonical RoutePlan metadata differs"],
        }
    else:
        route_result = _compare_learned_routes(baseline_routes, candidate_routes)

    communication = _check_communication(
        candidate_meta.get("communication", {}),
        forward_only=bool(candidate_meta.get("forward_only", False)),
        require_gradient_probe=str(candidate_meta.get("backend", "")) == "mlite",
        architecture=candidate_meta.get("architecture"),
        expected_steps=(
            int(candidate_meta["steps"])
            + int(bool(candidate_meta.get("checkpoint_next_step", False)))
            + int(bool(candidate_meta.get("checkpoint_uninterrupted_step", False)))
            if "steps" in candidate_meta
            else None
        ),
    )
    model_structure = _check_model_structure(
        candidate_meta.get("model_structure"),
        required=str(candidate_meta.get("backend", "")) == "mlite",
    )
    baseline_continuity = check_checkpoint_continuity(
        baseline_meta.get("checkpoint_continuity"),
        required=_requires_checkpoint_continuity(baseline_meta),
    )
    candidate_continuity = check_checkpoint_continuity(
        candidate_meta.get("checkpoint_continuity"),
        required=_requires_checkpoint_continuity(candidate_meta),
    )
    checkpoint_continuity = {
        "passed": bool(baseline_continuity["passed"] and candidate_continuity["passed"]),
        "baseline": baseline_continuity,
        "candidate": candidate_continuity,
    }
    checkpoint_roundtrip = _check_checkpoint_roundtrip(candidate_meta)
    passed = (
        (diagnostic or bool(evidence["passed"]))
        and not missing
        and not unexpected
        and all(bool(item["passed"]) for item in tensor_results if item["hard_gate"])
        and all(bool(item["passed"]) for item in aggregate_results)
        and bool(expert_routes["passed"])
        and bool(route_result["passed"])
        and bool(communication["passed"])
        and bool(model_structure["passed"])
        and bool(checkpoint_continuity["passed"])
        and bool(checkpoint_roundtrip["passed"])
        and bool(compatibility["passed"])
    )
    report = {
        "passed": passed,
        "status": "partial" if diagnostic else ("passed" if passed else "failed"),
        "acceptance_complete": bool(passed and not diagnostic),
        "evidence": evidence,
        "artifact_identity": {
            side: {
                "source_sha256": meta.get("source_snapshot", {}).get("sha256"),
                "manifest_sha256": file_sha256(Path(path) / "manifest.json"),
                "routes_sha256": file_sha256(Path(path) / "routes.json"),
                "tensors_sha256": file_sha256(Path(path) / "tensors.pt"),
                "topology": meta.get("topology"),
                "route_mode": meta.get("route_mode"),
                "cp_transition": meta.get("cp_transition"),
                "backend": meta.get("backend"),
            }
            for side, path, meta in (
                ("baseline", baseline_path, baseline_meta),
                ("candidate", candidate_path, candidate_meta),
            )
        },
        "baseline": str(Path(baseline_path)),
        "candidate": str(Path(candidate_path)),
        "precision": precision,
        "scope": scope,
        "missing_tensors": missing,
        "unexpected_tensors": unexpected,
        "ignored_candidate_tensors": ignored_candidate_tensors,
        "tensor_results": tensor_results,
        "aggregate_results": aggregate_results,
        "namespace_summary": namespace_summary,
        "expert_routes": expert_routes,
        "routes": route_result,
        "communication": communication,
        "model_structure": model_structure,
        "checkpoint_continuity": checkpoint_continuity,
        "checkpoint_roundtrip": checkpoint_roundtrip,
        "metadata_compatibility": compatibility,
    }
    if report_path is not None:
        target = Path(report_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _requires_checkpoint_continuity(metadata: dict[str, Any]) -> bool:
    """Tiny MLite training roundtrips must prove true optimizer continuation."""

    return bool(
        metadata.get("backend") == "mlite"
        and metadata.get("preset") == "tiny"
        and not metadata.get("forward_only", False)
        and metadata.get("checkpoint_next_step", False)
    )


def _check_checkpoint_roundtrip(metadata: dict[str, Any]) -> dict[str, Any]:
    """Require lightweight restore/update evidence for every MLite checkpoint run."""

    required = bool(
        metadata.get("backend") == "mlite" and metadata.get("checkpoint_next_step", False)
    )
    if not required:
        return {"required": False, "passed": True, "details": ["not required"]}
    details: list[str] = []
    expected_step = int(metadata.get("steps", -1))
    if metadata.get("checkpoint_restored_step") != expected_step:
        details.append("checkpoint restored step does not match the saved global step")
    fingerprint = metadata.get("checkpoint_parameter_fingerprint")
    if not isinstance(fingerprint, dict) or fingerprint.get("status") != "passed":
        details.append("checkpoint parameter fingerprint is missing or failed")
    if not metadata.get("forward_only", False):
        optimizer_steps = metadata.get("optimizer_steps")
        if not isinstance(optimizer_steps, list):
            details.append("checkpoint run is missing optimizer-step evidence")
        else:
            for phase in ("train", "resume"):
                matches = [
                    item
                    for item in optimizer_steps
                    if isinstance(item, dict) and item.get("phase") == phase
                ]
                if not matches:
                    details.append(f"checkpoint run is missing {phase} optimizer step")
                    continue
                if any(not bool(item.get("updated")) for item in matches):
                    details.append(f"{phase} optimizer step did not update")
                if any(
                    not math.isfinite(float(item.get("grad_norm", float("nan"))))
                    for item in matches
                ):
                    details.append(f"{phase} optimizer step has a non-finite grad norm")
    return {"required": True, "passed": not details, "details": details}


def _as_long(route: dict[str, Any], key: str) -> torch.Tensor:
    return torch.tensor(route.get(key, []), dtype=torch.long)


def _as_float(route: dict[str, Any], key: str) -> torch.Tensor:
    return torch.tensor(route.get(key, []), dtype=torch.float64)


def _compare_learned_routes(
    baseline_routes: list[dict[str, Any]], candidate_routes: list[dict[str, Any]]
) -> dict[str, Any]:
    baseline = {_route_key(route): route for route in baseline_routes}
    candidate = {_route_key(route): route for route in candidate_routes}
    details: list[str] = []
    near_ties = 0
    changed_near_ties = 0
    passed = baseline.keys() == candidate.keys()
    if not passed:
        details.append("round/sample RoutePlan keys differ")
    for key in sorted(baseline.keys() & candidate.keys()):
        lhs = baseline[key]
        rhs = candidate[key]
        peer_routes = rhs.get("peer_selected_global_token_ids", [])
        if peer_routes and any(list(peer) != list(peer_routes[0]) for peer in peer_routes[1:]):
            passed = False
            details.append(f"{key}: TPxCP peers disagree on selected IDs")
        if lhs.get("global_token_ids") == rhs.get("global_token_ids") and _canonical_route(
            lhs, include_selected_gates=False
        ) != _canonical_route(rhs, include_selected_gates=False):
            passed = False
            details.append(f"{key}: token IDs match but canonical RoutePlan metadata differs")
            continue
        lhs_candidates = _as_long(lhs, "candidate_global_token_ids")
        rhs_candidates = _as_long(rhs, "candidate_global_token_ids")
        lhs_candidate_samples = _as_long(lhs, "candidate_sample_ids")
        rhs_candidate_samples = _as_long(rhs, "candidate_sample_ids")
        lhs_candidate_positions = _as_long(lhs, "candidate_original_positions")
        rhs_candidate_positions = _as_long(rhs, "candidate_original_positions")
        lhs_scores_all = _as_float(lhs, "candidate_scores")
        rhs_scores_all = _as_float(rhs, "candidate_scores")
        if not (
            lhs_candidates.numel()
            == rhs_candidates.numel()
            == lhs_candidate_samples.numel()
            == rhs_candidate_samples.numel()
            == lhs_candidate_positions.numel()
            == rhs_candidate_positions.numel()
            == lhs_scores_all.numel()
            == rhs_scores_all.numel()
        ):
            passed = False
            details.append(f"{key}: candidate score metadata sizes differ")
            continue
        if (
            torch.unique(lhs_candidates).numel() != lhs_candidates.numel()
            or torch.unique(rhs_candidates).numel() != rhs_candidates.numel()
        ):
            passed = False
            details.append(f"{key}: candidate token IDs are not unique")
            continue
        if set(lhs_candidates.tolist()) != set(rhs_candidates.tolist()):
            passed = False
            details.append(f"{key}: learned route has different candidate active sets")
            continue
        rhs_row = {int(token_id): row for row, token_id in enumerate(rhs_candidates.tolist())}
        rhs_order = torch.tensor([rhs_row[int(token_id)] for token_id in lhs_candidates.tolist()])
        rhs_scores_all = rhs_scores_all[rhs_order]
        rhs_candidate_samples = rhs_candidate_samples[rhs_order]
        rhs_candidate_positions = rhs_candidate_positions[rhs_order]
        if not torch.equal(lhs_candidate_samples, rhs_candidate_samples):
            passed = False
            details.append(f"{key}: candidate sample IDs differ")
            continue
        if not torch.equal(lhs_candidate_positions, rhs_candidate_positions):
            passed = False
            details.append(f"{key}: candidate original positions differ")
            continue

        lhs_selected_ids = _as_long(lhs, "global_token_ids")
        rhs_selected_ids = _as_long(rhs, "global_token_ids")
        lhs_selected_samples = _as_long(lhs, "sample_ids")
        rhs_selected_samples = _as_long(rhs, "sample_ids")
        lhs_selected_positions = _as_long(lhs, "original_positions")
        rhs_selected_positions = _as_long(rhs, "original_positions")
        lhs_padding = torch.tensor(lhs.get("padding_mask", []), dtype=torch.bool)
        rhs_padding = torch.tensor(rhs.get("padding_mask", []), dtype=torch.bool)
        selected_size = lhs_selected_ids.numel()
        if not (
            selected_size
            == rhs_selected_ids.numel()
            == lhs_selected_samples.numel()
            == rhs_selected_samples.numel()
            == lhs_selected_positions.numel()
            == rhs_selected_positions.numel()
            == lhs_padding.numel()
            == rhs_padding.numel()
        ):
            passed = False
            details.append(f"{key}: selected RoutePlan metadata sizes differ")
            continue
        if lhs.get("active_cu_seqlens") != rhs.get("active_cu_seqlens"):
            passed = False
            details.append(f"{key}: selected per-sample cardinalities differ")
            continue
        if not torch.equal(lhs_padding, rhs_padding):
            passed = False
            details.append(f"{key}: selected padding masks differ")
            continue
        if not torch.equal(lhs_selected_samples, rhs_selected_samples):
            passed = False
            details.append(f"{key}: selected sample layout differs")
            continue
        if torch.unique(lhs_selected_ids[~lhs_padding]).numel() != int(
            (~lhs_padding).sum().item()
        ) or torch.unique(rhs_selected_ids[~rhs_padding]).numel() != int(
            (~rhs_padding).sum().item()
        ):
            passed = False
            details.append(f"{key}: selected real token IDs are not unique")
            continue

        candidate_metadata = {
            int(token_id): (int(sample_id), int(position))
            for token_id, sample_id, position in zip(
                lhs_candidates.tolist(),
                lhs_candidate_samples.tolist(),
                lhs_candidate_positions.tolist(),
                strict=True,
            )
        }
        selected_metadata_valid = True
        for side, ids, samples, positions, padding in (
            (
                "baseline",
                lhs_selected_ids,
                lhs_selected_samples,
                lhs_selected_positions,
                lhs_padding,
            ),
            (
                "candidate",
                rhs_selected_ids,
                rhs_selected_samples,
                rhs_selected_positions,
                rhs_padding,
            ),
        ):
            for token_id, sample_id, position in zip(
                ids[~padding].tolist(),
                samples[~padding].tolist(),
                positions[~padding].tolist(),
                strict=True,
            ):
                expected = candidate_metadata.get(int(token_id))
                if expected != (int(sample_id), int(position)):
                    passed = False
                    selected_metadata_valid = False
                    details.append(
                        f"{key}: {side} token {token_id} has inconsistent sample/position metadata"
                    )
                    break
            if not selected_metadata_valid:
                break
        if not selected_metadata_valid:
            continue

        sample_values = sorted(set(lhs_candidate_samples.tolist()))
        for sample_id in sample_values:
            candidate_mask = lhs_candidate_samples == sample_id
            selected_lhs = lhs_selected_ids[(lhs_selected_samples == sample_id) & ~lhs_padding]
            selected_rhs = rhs_selected_ids[(rhs_selected_samples == sample_id) & ~rhs_padding]
            if selected_lhs.numel() != selected_rhs.numel():
                passed = False
                details.append(
                    f"{key}/sample={sample_id}: selected cardinality differs "
                    f"({selected_lhs.numel()} != {selected_rhs.numel()})"
                )
                continue
            all_ids = lhs_candidates[candidate_mask]
            lhs_scores = lhs_scores_all[candidate_mask]
            rhs_scores = rhs_scores_all[candidate_mask]
            top_k = selected_lhs.numel()
            if top_k <= 0:
                passed = False
                details.append(f"{key}/sample={sample_id}: no selected tokens")
                continue
            cutoff = classify_cutoff(lhs_scores, rhs_scores, all_ids, top_k)
            near_ties += int(cutoff.near_tie)
            if set(selected_lhs.tolist()) == set(selected_rhs.tolist()):
                continue
            if not cutoff.near_tie:
                passed = False
                details.append(
                    f"{key}/sample={sample_id}: non-near-tie route differs "
                    f"(margin={cutoff.margin:g}, error={cutoff.score_error:g})"
                )
                continue
            changed_near_ties += 1
            changed = set(selected_lhs.tolist()) ^ set(selected_rhs.tolist())
            if not changed.issubset(set(cutoff.ambiguity_ids)):
                passed = False
                details.append(f"{key}/sample={sample_id}: changed IDs escape cutoff ambiguity set")
    total = sum(len(set(route.get("candidate_sample_ids", []))) for route in baseline.values())
    return {
        "passed": passed,
        "mode": "learned",
        "near_ties": near_ties,
        "changed_near_ties": changed_near_ties,
        "total": total,
        "near_tie_rate": near_ties / total if total else 0.0,
        "details": details,
    }


def _check_communication(
    raw: Any,
    *,
    forward_only: bool = False,
    require_gradient_probe: bool = True,
    expected_steps: int | None = None,
    architecture: Any = None,
) -> dict[str, Any]:
    details: list[str] = []
    if not isinstance(raw, dict):
        details.append("artifact communication metadata is not a mapping")
        raw = {}
    if not raw:
        details.append("artifact recorded no communication counters")
    required_route_fields = {
        "active_set_changes",
        "hidden_rebalances",
        "recurrent_inner_dispatches",
        "early_exit_qkv_tokens",
        "recurrent_qkv_checks",
    }
    missing_route_fields = sorted(required_route_fields - raw.keys())
    if missing_route_fields:
        details.append(
            "missing recurrent communication instrumentation: " + ", ".join(missing_route_fields)
        )

    changed = raw.get("active_set_changes")
    rebalances = raw.get("hidden_rebalances")
    if isinstance(changed, int) and isinstance(rebalances, int) and rebalances != changed:
        details.append(f"hidden rebalances {rebalances} != active-set changes {changed}")
    if raw.get("recurrent_inner_dispatches") not in {None, 0}:
        details.append("recurrent block performed an inner token dispatch")
    if raw.get("early_exit_qkv_tokens") not in {None, 0}:
        details.append("early-exit tokens entered later Q/K/V communication")
    qkv_checks = raw.get("recurrent_qkv_checks")
    if isinstance(qkv_checks, int) and qkv_checks <= 0:
        details.append("recurrent QKV instrumentation recorded no checks")
    if isinstance(architecture, dict) and {
        "n_recurrent_layers",
        "num_recursions",
    }.issubset(architecture):
        expected_qkv_checks = int(architecture["n_recurrent_layers"]) * int(
            architecture["num_recursions"]
        )
        if isinstance(qkv_checks, int) and qkv_checks != expected_qkv_checks:
            details.append(f"recurrent QKV checks {qkv_checks} != expected {expected_qkv_checks}")
    probe = raw.get("grad_sync_probe")
    if forward_only or not require_gradient_probe:
        # A forward-only baseline builds no optimizer/finalizer and therefore
        # has no gradient collective to instrument.  If a probe is present it
        # may report the explicit not-required state, but its absence is not a
        # parity failure.
        if (
            forward_only
            and isinstance(probe, dict)
            and probe.get("status")
            not in {
                "not_required",
                "available",
            }
        ):
            details.append(
                f"forward-only gradient-sync probe has an invalid status: {probe.get('status')!r}"
            )
        return {"passed": not details, "details": details}

    if not isinstance(probe, dict):
        details.append("training artifact is missing the real gradient-sync probe")
        return {"passed": False, "details": details}
    if probe.get("status") != "available":
        reasons = [
            str(reason) for step in probe.get("steps", []) for reason in step.get("reasons", [])
        ]
        suffix = f": {'; '.join(reasons)}" if reasons else ""
        details.append(
            f"gradient-sync instrumentation is {probe.get('status', 'unavailable')}{suffix}"
        )

    steps = probe.get("steps")
    if not isinstance(steps, list) or not steps:
        details.append("gradient-sync probe recorded no global training steps")
        steps = []
    if expected_steps is not None and len(steps) != expected_steps:
        details.append(
            f"gradient-sync probe recorded {len(steps)} steps, expected {expected_steps}"
        )
    labels = [str(step.get("global_step", "<unknown>")) for step in steps]
    if len(labels) != len(set(labels)):
        details.append("gradient-sync probe contains duplicate global-step labels")
    for step in steps:
        label = str(step.get("global_step", "<unknown>"))
        finalize_min = step.get("finalize_grads_calls_min")
        finalize_max = step.get("finalize_grads_calls_max")
        if finalize_min != 1 or finalize_max != 1:
            details.append(
                f"{label}: finalize_grads calls must be exactly one on every rank "
                f"(min={finalize_min!r}, max={finalize_max!r})"
            )
        bucket_min = step.get("physical_bucket_sync_calls_min")
        bucket_max = step.get("physical_bucket_sync_calls_max")
        if bucket_min != 1:
            details.append(
                f"{label}: at least one physical parameter bucket did not synchronize "
                f"exactly once (min={bucket_min!r})"
            )
        if bucket_max is None:
            details.append(f"{label}: physical bucket synchronization is unavailable")
        elif int(bucket_max) > 1:
            details.append(f"{label}: a physical parameter bucket synchronized {bucket_max} times")
        for rank in step.get("ranks", []):
            if int(rank.get("out_of_step_finalize_calls", 0)) != 0:
                details.append(
                    f"{label}/rank={rank.get('rank')}: finalize_grads ran outside a global step"
                )
            if int(rank.get("out_of_step_bucket_calls", 0)) != 0:
                details.append(
                    f"{label}/rank={rank.get('rank')}: a bucket synchronized outside a global step"
                )

    flat_bucket_max = raw.get("physical_bucket_sync_dispatch_max")
    measured_bucket_max = probe.get("physical_bucket_sync_calls_max")
    if flat_bucket_max is None or measured_bucket_max is None:
        details.append("physical bucket sync-dispatch maximum is unavailable")
    elif int(flat_bucket_max) != int(measured_bucket_max):
        details.append(
            "flat physical bucket counter disagrees with the instrumented probe: "
            f"{flat_bucket_max} != {measured_bucket_max}"
        )
    elif int(flat_bucket_max) > 1:
        details.append("a physical parameter bucket dispatched sync more than once per step")
    return {"passed": not details, "details": details}


def _check_model_structure(raw: Any, *, required: bool) -> dict[str, Any]:
    if not required and raw is None:
        return {"passed": True, "details": ["non-MLite artifact has no structure probe"]}
    if not isinstance(raw, dict):
        return {
            "passed": False,
            "details": ["MLite artifact is missing recurrent parameter registration evidence"],
        }
    details = []
    if raw.get("status") != "passed":
        details.append(f"recurrent parameter registration status is {raw.get('status')!r}")
    if raw.get("max_registrations_per_recurrent_parameter") != 1:
        details.append("a recurrent physical parameter is registered more than once")
    if int(raw.get("recurrent_physical_parameters", 0)) <= 0:
        details.append("no recurrent physical parameters were recorded")
    return {"passed": not details, "details": details}


__all__ = ["compare_artifacts"]
