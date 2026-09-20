"""Production runtime construction; no historical single-node parity shortcut."""

from __future__ import annotations

import os

from mor_mlite.pretraining.config import Experiment

ATTENTION_BACKEND = "fused"


def distributed_environment():
    if "SLURM_PROCID" in os.environ:
        for target, source in (
            ("RANK", "SLURM_PROCID"),
            ("LOCAL_RANK", "SLURM_LOCALID"),
            ("WORLD_SIZE", "SLURM_NTASKS"),
        ):
            os.environ.setdefault(target, os.environ[source])
        hosts = expand_hostlist(os.environ["SLURM_JOB_NODELIST"])
        os.environ.setdefault("MASTER_ADDR", hosts[0])
        os.environ.setdefault("MASTER_PORT", str(15000 + int(os.environ["SLURM_JOB_ID"]) % 40000))
    import torch
    from megatron.lite.primitive.deterministic import set_deterministic

    # The ImplConfig flag selects distributed groups, but does not enable
    # PyTorch/TE deterministic kernels. Configure both before CUDA work.
    os.environ["MEGATRON_LITE_DETERMINISTIC"] = "1"
    set_deterministic(seed=1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def build(config: Experiment, *, hf_config_dir, total_steps):
    import torch
    from megatron.lite.model.registry import register_model
    from megatron.lite.runtime import RuntimeConfig, create_runtime
    from megatron.lite.runtime.contracts import MegatronLiteConfig, OptimizerConfig, ParallelConfig

    if int(os.environ["WORLD_SIZE"]) != config.world_size:
        raise ValueError("allocated world does not match frozen experiment")
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    register_model(
        "qwen3_moe_pretrain",
        package="mor_mlite.qwen3_moe_mor",
        hf_model_types=["qwen3_moe_pretrain"],
        impls={"lite": "mor_mlite.pretraining.protocol"},
    )
    parallel = ParallelConfig(tp=1, cp=1, pp=1, vpp=1, etp=1, ep=config.ep)
    optimizer = OptimizerConfig(
        lr=config.lr,
        min_lr=config.min_lr,
        adam_beta1=config.adam_beta1,
        adam_beta2=config.adam_beta2,
        adam_eps=config.adam_eps,
        clip_grad=config.clip_grad,
        weight_decay=config.weight_decay,
        decoupled_weight_decay=True,
        total_training_steps=total_steps,
        lr_decay_style="cosine",
        lr_warmup_steps_ratio=config.warmup_fraction,
    )
    # A and C are ordinary single-pass stacks; B uses the same physical stack
    # as C but invokes its middle slice three times; only D has depth routers.
    if config.arm == "A":
        start, recurrent, rounds, end = 3, 42, 1, 3
    else:
        start, recurrent, rounds, end = 3, 14, 1 if config.arm == "C" else 3, 3
    impl = {
        "parallel": parallel,
        "optimizer": "dist_opt",
        "use_thd": True,
        "use_deepep": False,
        "deterministic": True,
        "local_attention_backend": "te",
        "cross_entropy_fusion": True,
        "n_start_layers": start,
        "n_recurrent_layers": recurrent,
        "num_recursions": rounds,
        "n_end_layers": end,
        "depth_router_seed": config.seed,
        "dense_dp_size": config.world_size,
        "experiment_arm": config.arm,
        "initialization_seed": config.seed,
        "pretraining_moe_aux": config.moe_aux,
        "depth_router_aux_loss_coef": config.depth_aux,
    }
    backend = MegatronLiteConfig(
        model_name="qwen3_moe_pretrain",
        impl="lite",
        hf_path=str(hf_config_dir),
        parallel=parallel,
        optimizer=optimizer,
        # This is TE's algorithm selector, distinct from our adapter name "te".
        attention_backend_override=ATTENTION_BACKEND,
        load_hf_weights=False,
        impl_cfg=impl,
    )
    runtime = create_runtime(
        RuntimeConfig(backend="mlite", hf_path=str(hf_config_dir), backend_cfg=backend)
    )
    handle = runtime.build_model()
    # Native runtime seeds CUDA internally during construction. Restore the
    # experiment RNG after our name-seeded parameter initialization.
    import random

    import numpy as np

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    if handle.dp_size != config.world_size:
        raise ValueError("data-parallel view differs from the global batch contract")
    return runtime, handle


def expand_hostlist(expression):
    """Expand Slurm's bracket ranges without requiring host tools in a container."""
    import re

    groups = re.split(r",(?![^\[]*\])", expression)
    result = []
    for group in groups:
        match = re.search(r"\[([^\[\]]+)\]", group)
        if not match:
            if not group or "[" in group or "]" in group:
                raise ValueError("invalid Slurm hostlist")
            result.append(group)
            continue
        for part in match[1].split(","):
            bounds = part.split("-")
            if len(bounds) not in (1, 2) or not all(x.isdigit() for x in bounds):
                raise ValueError("invalid Slurm host range")
            lo, hi = int(bounds[0]), int(bounds[-1])
            if lo > hi:
                raise ValueError("reversed Slurm host range")
            for number in range(lo, hi + 1):
                expanded = (
                    group[: match.start()]
                    + str(number).zfill(len(bounds[0]))
                    + group[match.end() :]
                )
                result.extend(expand_hostlist(expanded))
    return result


def topology_evidence(handle, topology_text):
    """Use Slurm's topology/block inventory, not inferred hostname prefixes."""
    import socket

    import torch.distributed as dist

    from mor_mlite.pretraining.tuning import validate_ep_locality

    hostname = socket.gethostname().split(".")[0]
    domain = None
    for line in topology_text.splitlines():
        fields = dict(part.split("=", 1) for part in line.split() if "=" in part)
        if "BlockName" not in fields or "Nodes" not in fields:
            continue
        hosts = expand_hostlist(fields["Nodes"])
        if hostname in hosts:
            if domain is not None:
                raise ValueError("ambiguous Slurm NVL block mapping")
            domain = fields["BlockName"]
    if domain is None:
        raise ValueError("hostname has no NVL domain in actual Slurm topology inventory")
    ps = handle._parallel_state
    record = {
        "rank": dist.get_rank(),
        "hostname": hostname,
        "nvl_domain": domain,
        "ep_members": dist.get_process_group_ranks(ps.ep_group),
    }
    records = [None] * dist.get_world_size()
    dist.all_gather_object(records, record)
    report = validate_ep_locality(records, world_size=dist.get_world_size())
    report["records"] = records
    return report
