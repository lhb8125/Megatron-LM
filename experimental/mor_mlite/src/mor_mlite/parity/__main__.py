"""CLI for generating and comparing correctness artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mor_mlite.cli_config import add_mor_config_arguments, resolve_cli_preset


def _seq_lens(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(part) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seq lengths must be comma-separated integers") from exc
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("seq lengths must be positive")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m mor_mlite.parity")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="generate one parity artifact")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--backend", choices=("reference", "mlite"), default="reference")
    run.add_argument("--preset", choices=("tiny", "qwen3-30b"), default="tiny")
    add_mor_config_arguments(run)
    run.add_argument("--topology", default="baseline")
    run.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    run.add_argument("--device", default="auto")
    run.add_argument("--seed", type=int, default=1234)
    run.add_argument("--steps", type=int, default=1)
    run.add_argument("--num-microbatches", type=int, default=2)
    run.add_argument("--lr", type=float, default=1e-3)
    run.add_argument("--adam-eps", type=float, default=1e-6)
    run.add_argument("--clip-grad", type=float, default=1.0)
    run.add_argument("--seq-lens", type=_seq_lens, default=(9, 6, 3))
    run.add_argument("--route-mode", choices=("learned", "replay"), default="learned")
    run.add_argument("--replay-from", type=Path)
    run.add_argument("--hf-path", default="")
    run.add_argument(
        "--init-checkpoint",
        type=Path,
        help="initialize model weights from a self-describing MoR DCP",
    )
    run.add_argument(
        "--resume-checkpoint",
        type=Path,
        help="restore model, optimizer, and RNG from a process-isolated MoR DCP",
    )
    run.add_argument(
        "--forward-only",
        action="store_true",
        help="run forward parity without building or stepping an optimizer",
    )
    run.add_argument(
        "--reference-dp-shards",
        type=int,
        default=1,
        help=("serially partition a one-rank forward baseline like the target dense-DP degree"),
    )
    run.add_argument(
        "--cp-transition",
        choices=("magi_direct", "magi_canonical", "static_reference"),
        default=None,
    )
    run.add_argument("--no-checkpoint-roundtrip", action="store_true")
    run.add_argument(
        "--checkpoint-save-only",
        action="store_true",
        help="save a full-state DCP plus receipt, then let this torchrun process exit",
    )
    run.add_argument("--non-strict", action="store_true")

    compare = subparsers.add_parser("compare", help="compare candidate with baseline")
    compare.add_argument("baseline", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("--report", type=Path)
    compare.add_argument("--scope", choices=("all", "forward"), default="all")
    compare.add_argument(
        "--diagnostic",
        action="store_true",
        help="Compare partial evidence; never eligible for an acceptance receipt",
    )

    certify = subparsers.add_parser(
        "certify-checkpoint",
        help="certify a fresh-process full-state restore and next optimizer step",
    )
    certify.add_argument("checkpoint", type=Path)
    certify.add_argument("resume_artifact", type=Path)
    certify.add_argument("--report", type=Path)
    return parser


def _run(args: argparse.Namespace) -> int:
    preset = resolve_cli_preset(args)
    cp_transition = (
        args.cp_transition
        if args.init_checkpoint is not None or args.resume_checkpoint is not None
        else (args.cp_transition or preset.parallel.cp_transition)
    )
    initialization_sources = sum(
        (bool(args.hf_path), args.init_checkpoint is not None, args.resume_checkpoint is not None)
    )
    if initialization_sources > 1:
        raise ValueError(
            "--hf-path, --init-checkpoint, and --resume-checkpoint are mutually exclusive"
        )
    if args.backend == "reference":
        if args.preset != "tiny":
            raise ValueError("the reference backend supports only the tiny preset")
        if args.topology != "baseline":
            raise ValueError("the reference backend is the single-rank baseline")
        if args.reference_dp_shards != 1:
            raise ValueError("--reference-dp-shards is supported only by the MLite backend")
        if args.init_checkpoint is not None:
            raise ValueError("--init-checkpoint is supported only by the MLite backend")
        if args.resume_checkpoint is not None or args.checkpoint_save_only:
            raise ValueError("process-isolated checkpoints are supported only by MLite")
        from .reference import ReferenceRunConfig, run_reference

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
        from .mlite import MLiteRunConfig, run_mlite

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
                checkpoint_roundtrip=(
                    not args.no_checkpoint_roundtrip
                    and not args.checkpoint_save_only
                    and args.resume_checkpoint is None
                ),
                checkpoint_save_only=args.checkpoint_save_only,
                strict=not args.non_strict,
                lr=args.lr,
                adam_eps=args.adam_eps,
                clip_grad=args.clip_grad,
                seq_lens=args.seq_lens,
                hf_path=args.hf_path,
                init_checkpoint=args.init_checkpoint,
                resume_checkpoint=args.resume_checkpoint,
                cp_transition=cp_transition,
                forward_only=args.forward_only,
                reference_dp_shards=args.reference_dp_shards,
                preset_config=args.preset_config,
                architecture=preset.architecture,
                depth_router=preset.depth_router,
            )
        )
    print(json.dumps({"artifact": str(output)}, sort_keys=True))
    return 0


def _compare(args: argparse.Namespace) -> int:
    from .compare import compare_artifacts

    report = compare_artifacts(
        args.baseline,
        args.candidate,
        report_path=args.report,
        scope=args.scope,
        diagnostic=args.diagnostic,
    )
    summary = {
        "passed": report["passed"],
        "failed_tensors": [
            item["name"]
            for item in report["tensor_results"]
            if item.get("hard_gate", True) and not item["passed"]
        ],
        "diagnostic_tensor_outliers": [
            item["name"]
            for item in report["tensor_results"]
            if not item.get("hard_gate", True) and not item["passed"]
        ],
        "failed_aggregates": [
            item["name"] for item in report.get("aggregate_results", []) if not item["passed"]
        ],
        "expert_routes": report["expert_routes"],
        "routes": report["routes"],
        "communication": report["communication"],
        "checkpoint_continuity": report["checkpoint_continuity"],
        "checkpoint_roundtrip": report["checkpoint_roundtrip"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


def _certify_checkpoint(args: argparse.Namespace) -> int:
    from .external_checkpoint import certify_external_checkpoint_resume

    report = certify_external_checkpoint_resume(
        args.checkpoint,
        args.resume_artifact,
        report_path=args.report,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run":
        return _run(args)
    if args.command == "compare":
        return _compare(args)
    return _certify_checkpoint(args)


if __name__ == "__main__":
    raise SystemExit(main())
