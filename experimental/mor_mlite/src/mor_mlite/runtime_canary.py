"""Small CUDA kernel canaries used before EOS training starts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def require_te_mlite_api() -> dict[str, Any]:
    """Fail closed when Transformer Engine cannot satisfy pinned MLite imports."""

    from transformer_engine.pytorch.permutation import (
        moe_permute_and_pad_with_probs,
    )

    if not callable(moe_permute_and_pad_with_probs):
        raise TypeError("Transformer Engine moe_permute_and_pad_with_probs is not callable")

    # Import the real pinned MLite consumer as well.  This catches additional
    # import-time TE API drift without adding compatibility shims to MLite.
    from megatron.lite.primitive.utils import moe as mlite_moe

    if mlite_moe.fused_permute_and_pad_with_probs is not moe_permute_and_pad_with_probs:
        raise RuntimeError("MLite did not bind Transformer Engine moe_permute_and_pad_with_probs")
    return {
        "api": "transformer_engine.pytorch.permutation.moe_permute_and_pad_with_probs",
        "callable": True,
        "mlite_import": "megatron.lite.primitive.utils.moe",
    }


def run_te_bf16_canary() -> dict[str, Any]:
    """Exercise the pinned Transform Engine PyTorch extension through backward."""

    import torch
    import transformer_engine.pytorch as te

    api = require_te_mlite_api()

    if not torch.cuda.is_available():
        raise RuntimeError("Transformer Engine canary requires CUDA")
    device_index = torch.cuda.current_device()
    if torch.cuda.get_device_capability(device_index) != (9, 0):
        raise RuntimeError("Transformer Engine production canary requires sm90")
    device = torch.device("cuda", device_index)
    torch.manual_seed(20260909)
    torch.cuda.manual_seed_all(20260909)
    layer = te.Linear(
        32,
        48,
        bias=True,
        params_dtype=torch.bfloat16,
        device=device,
    )
    x = torch.randn(17, 32, device=device, dtype=torch.bfloat16, requires_grad=True)
    output = layer(x)
    loss = output.float().square().mean()
    loss.backward()
    torch.cuda.synchronize(device_index)

    tensors = {"output": output, "input_grad": x.grad}
    tensors.update(
        {f"parameter_grad:{name}": parameter.grad for name, parameter in layer.named_parameters()}
    )
    for name, tensor in tensors.items():
        if tensor is None:
            raise RuntimeError(f"Transformer Engine canary did not produce {name}")
        if not torch.isfinite(tensor.float()).all().item():
            raise RuntimeError(f"Transformer Engine canary produced non-finite {name}")
        if tensor.float().abs().max().item() == 0.0:
            raise RuntimeError(f"Transformer Engine canary produced all-zero {name}")
    return {
        "canary": "transformer_engine_bf16_linear_fwd_bwd",
        "passed": True,
        "device": torch.cuda.get_device_name(device_index),
        "capability": list(torch.cuda.get_device_capability(device_index)),
        "dtype": str(output.dtype),
        "output_shape": list(output.shape),
        "loss": float(loss.detach()),
        "mlite_api": api,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="run fail-closed EOS CUDA extension canaries")
    parser.add_argument("canary", choices=("te",))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = run_te_bf16_canary()
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
