"""Training entry point for the tiny oracle and MLite backends."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from mor_mlite.cli_config import add_mor_config_arguments, resolve_cli_preset
from mor_mlite.parity.__main__ import _seq_lens


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m mor_mlite.train")
    parser.add_argument("--backend", choices=("reference", "mlite"), default="reference")
    parser.add_argument("--preset", choices=("tiny", "qwen3-30b"), default="tiny")
    add_mor_config_arguments(parser)
    parser.add_argument("--output", type=Path, default=Path("artifacts/train"))
    parser.add_argument("--hf-path", default="")
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help="initialize model weights from a self-describing MoR DCP; optimizer/RNG start fresh",
    )
    parser.add_argument("--topology", default="baseline")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--num-microbatches", type=int, default=2)
    parser.add_argument("--seq-lens", type=_seq_lens, default=(9, 6, 3))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--adam-eps", type=float, default=1e-6)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--route-mode", choices=("learned", "replay"), default="learned")
    parser.add_argument("--replay-from", type=Path)
    parser.add_argument(
        "--cp-transition",
        choices=("magi_direct", "magi_canonical", "static_reference"),
        default=None,
        help="override the checkpoint CP backend; defaults to its saved value or magi_direct",
    )
    parser.add_argument("--no-checkpoint-roundtrip", action="store_true")
    parser.add_argument("--non-strict", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _validate_optimizer_arguments(args: argparse.Namespace) -> None:
    values = (args.lr, args.adam_eps, args.clip_grad)
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("lr, adam_eps, and clip_grad must be finite and positive")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    from mor_mlite.parity.topologies import get_topology
    from mor_mlite.versions import collect_version_manifest

    preset = resolve_cli_preset(args)
    cp_transition = (
        args.cp_transition
        if args.init_checkpoint is not None
        else (args.cp_transition or preset.parallel.cp_transition)
    )
    topology = get_topology(args.topology)
    topology.validate(num_experts=preset.num_experts)
    _validate_optimizer_arguments(args)
    if args.hf_path and args.init_checkpoint is not None:
        raise ValueError("--hf-path and --init-checkpoint are mutually exclusive")
    if args.backend == "reference" and args.init_checkpoint is not None:
        raise ValueError("--init-checkpoint is supported only by the MLite backend")
    if args.dry_run and args.backend == "mlite" and args.init_checkpoint is not None:
        # A dry-run must not advertise architecture/router overrides that the
        # self-describing checkpoint would replace at execution time.
        from mor_mlite.parity.mlite import MLiteRunConfig

        MLiteRunConfig(
            output=args.output,
            preset=args.preset,
            topology=args.topology,
            precision=args.precision,
            lr=args.lr,
            adam_eps=args.adam_eps,
            clip_grad=args.clip_grad,
            hf_path=args.hf_path,
            init_checkpoint=args.init_checkpoint,
            cp_transition=cp_transition,
            preset_config=args.preset_config,
            architecture=preset.architecture,
            depth_router=preset.depth_router,
        ).validate()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "backend": args.backend,
                    "preset": args.preset,
                    "preset_config_source": str(preset.source),
                    "architecture": preset.architecture.to_dict(),
                    "depth_router": preset.depth_router.to_dict(),
                    "optimizer": {
                        "name": "adam" if args.backend == "mlite" else "adamw",
                        "lr": args.lr,
                        "adam_eps": args.adam_eps,
                        "clip_grad": args.clip_grad,
                    },
                    "topology": topology.to_dict(),
                    "init_checkpoint": (
                        str(args.init_checkpoint) if args.init_checkpoint is not None else None
                    ),
                    "versions": collect_version_manifest(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.backend == "reference":
        if args.preset != "tiny" or topology.world_size != 1:
            raise ValueError("reference training supports tiny on one rank only")
        from mor_mlite.parity.reference import ReferenceRunConfig, run_reference

        output = run_reference(
            ReferenceRunConfig(
                output=args.output,
                precision=args.precision,
                device=args.device,
                seed=args.seed,
                steps=args.steps,
                num_microbatches=args.num_microbatches,
                route_mode=args.route_mode,
                replay_from=args.replay_from,
                checkpoint_roundtrip=not args.no_checkpoint_roundtrip,
                strict=not args.non_strict,
                lr=args.lr,
                adam_eps=args.adam_eps,
                clip_grad=args.clip_grad,
                seq_lens=args.seq_lens,
                preset_config=args.preset_config,
                architecture=preset.architecture,
                depth_router=preset.depth_router,
            )
        )
    else:
        from mor_mlite.parity.mlite import MLiteRunConfig, run_mlite

        output = run_mlite(
            MLiteRunConfig(
                output=args.output,
                preset=args.preset,
                topology=args.topology,
                precision=args.precision,
                seed=args.seed,
                steps=args.steps,
                num_microbatches=args.num_microbatches,
                route_mode=args.route_mode,
                replay_from=args.replay_from,
                checkpoint_roundtrip=not args.no_checkpoint_roundtrip,
                strict=not args.non_strict,
                seq_lens=args.seq_lens,
                hf_path=args.hf_path,
                init_checkpoint=args.init_checkpoint,
                cp_transition=cp_transition,
                lr=args.lr,
                adam_eps=args.adam_eps,
                clip_grad=args.clip_grad,
                preset_config=args.preset_config,
                architecture=preset.architecture,
                depth_router=preset.depth_router,
            )
        )
    print(json.dumps({"artifact": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
