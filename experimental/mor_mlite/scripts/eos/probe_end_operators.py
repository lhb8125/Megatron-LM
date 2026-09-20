"""Diagnostic only: feed native end blocks identical canonical baseline inputs."""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path
from types import MethodType

import torch
import torch.distributed as dist

from mor_mlite.parity import artifacts, mlite
from mor_mlite.parity.__main__ import main
from mor_mlite.parity.compare import compare_artifacts
from mor_mlite.provenance import file_sha256


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--submodules", action="store_true")
    parser.add_argument("--projection-fp32", action="store_true")
    args, remaining = parser.parse_known_args()
    if "--forward-only" not in remaining:
        parser.error(
            "operator injection is only a forward diagnostic, never training or acceptance"
        )
    metadata, tensors, routes = artifacts.load_artifact(args.oracle)
    if metadata["steps"] != 1 or metadata["num_microbatches"] != 1:
        parser.error("operator probe requires one step and one microbatch")
    route = next(r for r in routes if r["round"] == 0)
    ids = route["global_token_ids"]
    if ids != list(range(len(ids))):
        parser.error("this diagnostic expects contiguous synthetic global token IDs")
    prefix = "forward/step_000/mb_000/"
    inputs = [tensors[prefix + "post_merge_hidden"]]
    inputs += [
        tensors[prefix + f"end_hidden_{i}"]
        for i in range(metadata["architecture"]["n_end_layers"] - 1)
    ]
    original_build = mlite.build_runtime_session

    def install(model):
        original_scope = model._moe_probe_scope

        @contextmanager
        def scope(self, layer, **context):
            handles = []
            if context["stage"] == "end":
                index = context["stage_layer_index"]
                token_ids = context["global_token_ids"].reshape(-1).long()
                padding = context["padding_mask"].reshape(-1)
                if torch.any(token_ids[~padding] < 0) or torch.any(token_ids[~padding] >= len(ids)):
                    raise ValueError("operator probe token identity is outside its oracle")

                def injection(baseline_input):
                    def inject(_module, values):
                        x = values[0]
                        if x.shape[0] != token_ids.numel():
                            raise ValueError("operator probe input layout is not token aligned")
                        fixed = baseline_input.to(device=x.device).index_select(
                            0, token_ids.clamp_min(0)
                        )
                        fixed[padding] = 0
                        return (fixed.reshape_as(x).to(dtype=x.dtype), *values[1:])

                    return inject

                handles.append(
                    layer.register_forward_pre_hook(injection(inputs[index]), prepend=True)
                )
                if args.submodules:
                    for module, key in (
                        (layer.mlp_norm, "post_attention_residual"),
                        (layer.moe, "mlp_norm_output"),
                    ):
                        oracle_input = tensors[prefix + f"end_sublayer_{index}_{key}"]
                        handles.append(
                            module.register_forward_pre_hook(injection(oracle_input), prepend=True)
                        )
            try:
                with original_scope(layer, **context):
                    yield
            finally:
                for handle in handles:
                    handle.remove()

        model._moe_probe_scope = MethodType(scope, model)

    def build(config):
        session = original_build(config)
        for wrapped in session.handle._extras["model_chunks"]:
            model = wrapped
            while hasattr(model, "module"):
                model = model.module
            install(model)
            if args.projection_fp32:
                from probe_precision import projection_forward

                for layer in model.end_layers:
                    layer.attn.proj.forward = MethodType(projection_forward, layer.attn.proj)
        return session

    original_save = artifacts.save_artifact

    def save(directory, *, metadata, tensors, routes):
        metadata = {
            **metadata,
            "operator_probe": {
                "kind": "identical-input-native-end-blocks",
                "submodules": args.submodules,
                "projection_fp32": args.projection_fp32,
                "projection_script_sha256": file_sha256(
                    Path(__file__).with_name("probe_precision.py")
                ),
                "script_sha256": file_sha256(__file__),
                "oracle_manifest_sha256": file_sha256(args.oracle / "manifest.json"),
                "oracle_tensors_sha256": file_sha256(args.oracle / "tensors.pt"),
                "acceptance_eligible": False,
            },
        }
        return original_save(directory, metadata=metadata, tensors=tensors, routes=routes)

    artifacts.save_artifact = save
    mlite.build_runtime_session = build
    result = main(["run", *remaining])
    if not dist.is_initialized() or dist.get_rank() == 0:
        output = Path(remaining[remaining.index("--output") + 1])
        report = compare_artifacts(args.oracle, output, scope="forward", diagnostic=True)
        report["operator_probe"] = {
            "acceptance_eligible": False,
            "kind": "identical-input-native-end-blocks",
            "submodules": args.submodules,
            "projection_fp32": args.projection_fp32,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(
            json.dumps(
                {"operator_metrics": [v for v in report["tensor_results"] if "/end_" in v["name"]]},
                indent=2,
            )
        )
    return result


if __name__ == "__main__":
    raise SystemExit(run())
