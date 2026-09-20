"""Independent GB200 candidate-environment canary, not a training certificate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path


def native_tree(root):
    root = Path(root).resolve()
    digest = hashlib.sha256()
    count = 0
    for subtree in ("megatron", "experimental/lite/megatron"):
        if not (root / subtree).is_dir():
            raise ValueError(f"native source subtree missing: {subtree}")
        for path in sorted((root / subtree).rglob("*.py")):
            digest.update(str(path.relative_to(root)).encode() + b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
            count += 1
    return {"sha256": digest.hexdigest(), "python_files": count}


def environment(native_root):
    import torch
    import transformer_engine
    from megatron.lite.model.qwen3_moe.lite import protocol

    # Native Qwen import alone does not touch the distributed checkpoint/
    # optimizer stack. Import the actual training protocol to catch NVRX/API drift.
    from mor_mlite.pretraining import protocol as experiment_protocol

    root = Path(native_root).resolve()
    if not Path(protocol.__file__).resolve().is_relative_to(root):
        raise ValueError("imported MLite is not from the selected native source snapshot")
    if not callable(experiment_protocol.build_model):
        raise TypeError("experiment training protocol is not importable")
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformer_engine": transformer_engine.__version__,
        "capability": list(torch.cuda.get_device_capability()),
        "gpu": torch.cuda.get_device_name(),
        "native": native_tree(root),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "triton",
                "transformers",
                "huggingface-hub",
                "nvidia-resiliency-ext",
                "protobuf",
                "grpcio",
                "grpcio-tools",
                "nvidia-cutlass-dsl",
                "nvidia-cutlass-dsl-libs-base",
                "nvidia-cutlass-dsl-libs-cu13",
                "quack-kernels",
            )
        },
    }


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch
    import transformer_engine.pytorch as te

    from mor_mlite.runtime_canary import require_te_mlite_api

    torch.cuda.set_device(0)
    report = environment(args.native_root)
    if report["capability"] != [10, 0]:
        raise ValueError("this environment probe requires GB200/sm100")
    report["mlite_api"] = require_te_mlite_api()
    torch.manual_seed(1234)
    layer = te.Linear(128, 256, bias=False, params_dtype=torch.bfloat16, device="cuda")
    x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y = layer(x)
    y.float().square().mean().backward()
    if any(
        t is None or not torch.isfinite(t).all() or not t.abs().max() > 0
        for t in (y, x.grad, layer.weight.grad)
    ):
        raise ValueError("TE forward/backward canary failed")
    reference = torch.nn.functional.linear(x.detach().float(), layer.weight.detach().float())
    relative_l2 = float((y.detach().float() - reference).norm() / reference.norm())
    if relative_l2 > 0.02:
        raise ValueError(f"TE BF16 canary disagrees with FP32 reference: {relative_l2}")
    report["te_forward_backward"] = True
    report["relative_l2"] = relative_l2
    report["scope"] = "candidate environment only; not four-arm acceptance"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
