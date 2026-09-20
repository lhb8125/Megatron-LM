"""Import a vanilla Qwen3-MoE HF checkpoint into folded MLite MoR state.

This command is intentionally one-way.  It can inspect the physical/logical
folding plan without importing PyTorch or Megatron-Lite, and its execution path
uses the public MLite Runtime checkpoint API.  Reverse HF export is not exposed
because a folded MoR checkpoint contains independent depth-router parameters
that vanilla Qwen has no representation for.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from mor_mlite.checkpoint_io import (
    build_checkpoint_metadata,
    save_mor_checkpoint,
)
from mor_mlite.config import DepthRouterConfig, MoRArchitectureConfig, MoRParallelConfig
from mor_mlite.config_loader import load_json
from mor_mlite.hf import resolve_hf_checkpoint
from mor_mlite.parity.topologies import Topology
from mor_mlite.qwen3_moe_mor.metadata import (
    MEGATRON_LM_PINNED_SHA,
    physical_to_logical_layer_map,
    validate_folding_policy,
)
from mor_mlite.versions import MAGI_VERSION

REVERSE_EXPORT_SUPPORTED = False


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mor-mlite-convert-hf",
        description="Import HF Qwen3-MoE weights into an MLite MoR checkpoint.",
    )
    parser.add_argument(
        "--direction",
        choices=("import",),
        default="import",
        help="conversion direction; reverse HF export is intentionally unsupported",
    )
    parser.add_argument("--hf-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-start-layers", type=int, default=3)
    parser.add_argument("--n-recurrent-layers", type=int, default=14)
    parser.add_argument("--num-recursions", type=int, default=3)
    parser.add_argument("--n-end-layers", type=int, default=3)
    parser.add_argument(
        "--capacity-schedule",
        default="linear",
        help="linear or comma-separated per-round fractions",
    )
    parser.add_argument(
        "--folding-policy",
        choices=("mean",),
        default="mean",
    )
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--etp", type=int, default=1)
    parser.add_argument(
        "--cp-transition",
        choices=("magi_direct", "magi_canonical", "static_reference"),
        default="magi_direct",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--depth-router-temperature", type=float, default=1.0)
    parser.add_argument("--depth-router-alpha", type=float, default=0.1)
    parser.add_argument("--depth-router-aux-loss-coef", type=float, default=0.001)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _capacity_schedule(raw: str, *, num_recursions: int) -> str | tuple[float, ...]:
    if raw == "linear":
        return raw
    try:
        values = tuple(float(item.strip()) for item in raw.split(","))
    except ValueError as exc:
        raise ValueError("capacity schedule must be 'linear' or comma-separated fractions") from exc
    if len(values) != num_recursions:
        raise ValueError("capacity schedule must contain exactly num_recursions fractions")
    return values


def _architecture(args: argparse.Namespace) -> MoRArchitectureConfig:
    return MoRArchitectureConfig(
        n_start_layers=args.n_start_layers,
        n_recurrent_layers=args.n_recurrent_layers,
        num_recursions=args.num_recursions,
        n_end_layers=args.n_end_layers,
        capacity_schedule=_capacity_schedule(
            args.capacity_schedule, num_recursions=args.num_recursions
        ),
    )


def _local_hf_config(hf_path: str) -> dict[str, Any] | None:
    source = Path(hf_path)
    config_path = source / "config.json" if source.is_dir() else source
    if not config_path.is_file() or config_path.name != "config.json":
        return None
    value = load_json(config_path, expected_type=dict)
    assert isinstance(value, dict)
    return value


def build_import_plan(
    *,
    hf_path: str,
    output: Path,
    architecture: MoRArchitectureConfig,
    folding_policy: str,
    topology: dict[str, int],
    cp_transition: str,
    seed: int,
    depth_router: DepthRouterConfig | None = None,
) -> dict[str, Any]:
    """Return the side-effect-free import plan printed by ``--dry-run``."""

    if not hf_path:
        raise ValueError("hf_path must not be empty")
    policy = validate_folding_policy(folding_policy)
    source_candidate = Path(hf_path)
    resolved_source = source_candidate.resolve() if source_candidate.exists() else None
    if resolved_source is not None and resolved_source.is_file():
        resolved_source = resolved_source.parent
    resolved_output = output.resolve()
    if resolved_source is not None and (
        resolved_output == resolved_source or resolved_source in resolved_output.parents
    ):
        raise ValueError("output must not overwrite or be nested in the source HF checkpoint")
    if output.exists() and not output.is_dir():
        raise FileExistsError(f"checkpoint output exists and is not a directory: {output}")
    if output.is_dir() and any(output.iterdir()):
        raise FileExistsError(
            f"checkpoint output must be empty to prevent stale step_* reuse: {output}"
        )
    source_config = _local_hf_config(hf_path)
    source_depth = None if source_config is None else source_config.get("num_hidden_layers")
    if source_depth is not None and int(source_depth) != architecture.logical_num_layers:
        raise ValueError(
            f"HF num_hidden_layers={source_depth} does not match MoR logical depth "
            f"{architecture.logical_num_layers}"
        )
    required_topology = {"world_size", "tp", "cp", "dp", "ep", "etp"}
    missing_topology = required_topology - topology.keys()
    if missing_topology:
        raise ValueError(f"topology is missing required fields: {sorted(missing_topology)}")
    topology_values = {key: int(topology[key]) for key in sorted(required_topology)}
    validated_topology = Topology(
        "hf-import",
        world_size=topology_values["world_size"],
        tp=topology_values["tp"],
        cp=topology_values["cp"],
        dp=topology_values["dp"],
        ep=topology_values["ep"],
        etp=topology_values["etp"],
    )
    source_experts = None if source_config is None else source_config.get("num_experts")
    validated_topology.validate(num_experts=None if source_experts is None else int(source_experts))
    if cp_transition not in {
        "magi_direct",
        "magi_canonical",
        "static_reference",
    }:
        raise ValueError(f"unsupported CP transition backend {cp_transition!r}")
    if validated_topology.cp > 1 and cp_transition == "static_reference":
        raise ValueError("static_reference is not a distributed Qwen CP backend")
    if validated_topology.tp > 1 and cp_transition == "magi_canonical":
        raise ValueError("magi_canonical is defined only for TP=1")
    layer_map = physical_to_logical_layer_map(architecture)
    router = depth_router or DepthRouterConfig()
    parallel = validated_topology.to_parallel_config(cp_transition=cp_transition)
    return {
        "direction": "hf-to-mlite",
        "reverse_hf_export_supported": REVERSE_EXPORT_SUPPORTED,
        "model_name": "qwen3_moe_mor",
        "hf_path": hf_path,
        "output": str(output),
        "output_format": "mlite-distributed-checkpoint",
        "megatron_lm_sha": MEGATRON_LM_PINNED_SHA,
        "magi_attention_version": MAGI_VERSION,
        "hf_source": hf_path,
        "architecture": architecture.to_dict(),
        "logical_num_layers": architecture.logical_num_layers,
        "physical_num_layers": architecture.physical_num_layers,
        "folding_policy": policy,
        "physical_to_logical_layers": {
            str(physical): list(logical) for physical, logical in layer_map.items()
        },
        "source_num_hidden_layers": source_depth,
        "topology": topology_values,
        "cp_transition": cp_transition,
        "parallel": parallel.to_dict(),
        "depth_router": router.to_dict(),
        "depth_router_seed": int(seed),
        "depth_router_source": "deterministic-initialization",
    }


def _run_import(args: argparse.Namespace, plan: dict[str, Any]) -> None:
    try:
        import torch
        import torch.distributed as dist
    except ImportError as exc:  # pragma: no cover - exercised on login nodes.
        raise RuntimeError("HF import requires the pinned CUDA/PyTorch environment") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("HF import through the pinned MLite Runtime requires CUDA")

    from mor_mlite.parity.mlite import (
        MLiteRuntimeBuildConfig,
        build_runtime_session,
    )

    topology_values = plan["topology"]
    topology = Topology(
        "hf-import",
        world_size=topology_values["world_size"],
        tp=topology_values["tp"],
        cp=topology_values["cp"],
        dp=topology_values["dp"],
        ep=topology_values["ep"],
        etp=topology_values["etp"],
    )
    architecture = MoRArchitectureConfig.from_dict(plan["architecture"])
    depth_router = DepthRouterConfig.from_dict(plan["depth_router"])
    parallel = MoRParallelConfig.from_dict(plan["parallel"])
    resolved_hf = resolve_hf_checkpoint(args.hf_path, require_weights=True)
    local_hf_path = str(resolved_hf.local_path)
    session = build_runtime_session(
        MLiteRuntimeBuildConfig(
            hf_path=local_hf_path,
            topology=topology,
            architecture=architecture,
            depth_router=depth_router,
            load_hf_weights=True,
            build_optimizer=False,
            seed=args.seed,
            total_training_steps=1,
            cp_transition=args.cp_transition,
            route_mode="learned",
            folding_policy=args.folding_policy,
            strict=True,
        )
    )
    metadata = build_checkpoint_metadata(
        architecture=architecture,
        depth_router=depth_router,
        depth_router_seed=args.seed,
        hf_source=args.hf_path,
        parallel=parallel,
        folding_policy=args.folding_policy,
        cp_transition=args.cp_transition,
    )
    save_mor_checkpoint(
        session.runtime,
        session.handle,
        args.output,
        step=0,
        metadata=metadata,
        base_hf_path=local_hf_path,
        save_rng=False,
        save_model=True,
        save_optimizer=False,
    )
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "hf_import_plan.json").write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if dist.is_initialized():
        dist.barrier()


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.direction != "import":  # argparse guards this; retain a hard API invariant.
        parser.error("reverse HF export is unsupported")
    architecture = _architecture(args)
    topology = {
        "world_size": args.dp * args.tp * args.cp,
        "tp": args.tp,
        "cp": args.cp,
        "dp": args.dp,
        "ep": args.ep,
        "etp": args.etp,
    }
    plan = build_import_plan(
        hf_path=args.hf_path,
        output=args.output,
        architecture=architecture,
        folding_policy=args.folding_policy,
        topology=topology,
        cp_transition=args.cp_transition,
        seed=args.seed,
        depth_router=DepthRouterConfig(
            temperature=args.depth_router_temperature,
            alpha=args.depth_router_alpha,
            aux_loss_coef=args.depth_router_aux_loss_coef,
        ),
    )
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    _run_import(args, plan)

    # torchrun launches one CLI process per rank; emit the machine-readable
    # completion record once without importing torch on the dry-run path.
    rank = int(__import__("os").environ.get("RANK", "0"))
    if rank == 0:
        print(json.dumps({"checkpoint": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "REVERSE_EXPORT_SUPPORTED",
    "build_import_plan",
    "main",
]
