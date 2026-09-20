"""Full-width Qwen3 initialization evidence without training or pretrained weights."""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--arm", choices=list("ABCD"), required=True)
    parser.add_argument("--hf-config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch.distributed as dist

    from mor_mlite.pretraining.config import Experiment, validate_base_model
    from mor_mlite.pretraining.cost import parameter_counts
    from mor_mlite.pretraining.data import sha256_file
    from mor_mlite.pretraining.memory import MemoryMonitor
    from mor_mlite.pretraining.runtime import build, distributed_environment
    from mor_mlite.pretraining.smoke import fingerprint
    from mor_mlite.pretraining.train import models
    from mor_mlite.provenance import source_snapshot

    model_config = json.loads((args.hf_config / "config.json").read_text())
    data_manifest = json.loads((args.data / "manifest.json").read_text())
    validate_base_model(model_config, data_manifest)
    distributed_environment()
    memory = MemoryMonitor()
    memory.start()
    config = Experiment(args.arm, world_size=int(os.environ["WORLD_SIZE"]))
    _, handle = build(config, hf_config_dir=args.hf_config, total_steps=100)
    model = models(handle)[0]
    backbone = fingerprint(model)
    counts = parameter_counts(model)
    if len(model.layers) != config.physical_layers:
        raise AssertionError("full-width physical layer count mismatch")
    report = {
        "scope": "full-width initialization only, no forward/backward/Adam-state allocation",
        "arm": args.arm,
        "world_size": config.world_size,
        "rank": dist.get_rank(),
        "backbone": backbone,
        "parameters": counts,
        "memory": memory.finish(),
        "source": source_snapshot(),
        "model_config_sha256": sha256_file(args.hf_config / "config.json"),
        "probe_sha256": sha256_file(Path(__file__)),
    }
    if dist.get_rank() == 0:
        args.output.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    (args.output / f"rank-{dist.get_rank()}.json").write_text(json.dumps(report) + "\n")
    dist.barrier()
    if dist.get_rank() == 0:
        print(
            json.dumps({k: v for k, v in report.items() if k not in ("backbone", "source")}),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
