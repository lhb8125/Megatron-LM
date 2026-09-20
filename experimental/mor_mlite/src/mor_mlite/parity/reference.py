"""Generate a complete tiny-model parity artifact."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from mor_mlite.config import DepthRouterConfig, MoRArchitectureConfig
from mor_mlite.config_loader import load_preset_config
from mor_mlite.data import make_synthetic_batch
from mor_mlite.determinism import configure_determinism
from mor_mlite.optim import MasterWeightAdamW
from mor_mlite.provenance import source_snapshot
from mor_mlite.reference_checkpoint import (
    load_reference_checkpoint,
    save_reference_checkpoint,
)
from mor_mlite.routing import RoutePlan
from mor_mlite.tiny import TinyMoRConfig, TinyMoRModel
from mor_mlite.versions import collect_version_manifest

from .artifacts import load_artifact, require_fresh_artifact_directory, save_artifact


@dataclass(slots=True)
class ReferenceRunConfig:
    output: Path
    precision: str = "fp32"
    device: str = "auto"
    seed: int = 1234
    steps: int = 1
    num_microbatches: int = 2
    route_mode: str = "learned"
    replay_from: Path | None = None
    checkpoint_roundtrip: bool = True
    strict: bool = True
    lr: float = 1e-3
    adam_eps: float = 1e-6
    clip_grad: float = 1.0
    seq_lens: tuple[int, ...] = (9, 6, 3)
    preset_config: Path | None = None
    architecture: MoRArchitectureConfig | None = None
    depth_router: DepthRouterConfig | None = None

    def validate(self) -> None:
        if self.precision not in {"fp32", "bf16"}:
            raise ValueError("precision must be fp32 or bf16")
        if self.route_mode not in {"learned", "replay"}:
            raise ValueError("route_mode must be learned or replay")
        if self.route_mode == "replay" and self.replay_from is None:
            raise ValueError("replay mode requires replay_from")
        if self.steps < 1 or self.num_microbatches < 1:
            raise ValueError("steps and num_microbatches must be positive")
        if (
            not math.isfinite(self.lr)
            or not math.isfinite(self.adam_eps)
            or not math.isfinite(self.clip_grad)
            or self.lr <= 0.0
            or self.adam_eps <= 0.0
            or self.clip_grad <= 0.0
        ):
            raise ValueError("lr, adam_eps, and clip_grad must be finite and positive")


def _snapshot(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().clone()


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = torch.device(name)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return selected


def _route_index(path: Path | None) -> dict[tuple[str, int, int], dict[int, RoutePlan]]:
    if path is None:
        return {}
    _, _, routes = load_artifact(path)
    result: dict[tuple[str, int, int], dict[int, RoutePlan]] = {}
    for raw in routes:
        phase = str(raw.get("phase", "train"))
        step = int(raw.get("step", 0))
        microbatch = int(raw.get("microbatch", 0))
        plan = RoutePlan.from_dict(raw)
        result.setdefault((phase, step, microbatch), {})[plan.round_index] = plan
    return result


def _metadata_for_candidates(batch, candidate_ids: torch.Tensor, *, field: str) -> torch.Tensor:
    all_ids = batch.extras["global_token_ids"].detach().cpu().tolist()
    values = batch.extras[field].detach().cpu().tolist()
    value_by_id = {
        int(token_id): int(value) for token_id, value in zip(all_ids, values, strict=True)
    }
    return torch.tensor(
        [value_by_id[int(token_id)] for token_id in candidate_ids.detach().cpu().tolist()],
        dtype=torch.long,
    )


def _record_forward(
    *,
    prefix: str,
    step: int,
    microbatch: int,
    batch,
    output,
    model_config: TinyMoRConfig,
    tensors: dict[str, torch.Tensor],
    routes: list[dict[str, Any]],
) -> None:
    base = f"{prefix}step_{step:03d}/mb_{microbatch:03d}"
    tensors[f"forward/{base}/logits"] = _snapshot(output.logits)
    tensors[f"loss/{base}/lm"] = _snapshot(output.lm_loss.reshape(()))
    tensors[f"loss/{base}/aux"] = _snapshot(output.aux_loss.reshape(()))
    tensors[f"loss/{base}/total"] = _snapshot(output.total_loss.reshape(()))
    for round_index, hidden in enumerate(output.hidden_by_round):
        tensors[f"forward/{base}/hidden_round_{round_index}"] = _snapshot(hidden)
    for round_index, plan in enumerate(output.route_plans):
        raw = plan.to_dict()
        raw["step"] = step
        raw["microbatch"] = microbatch
        raw["phase"] = prefix.rstrip("/") or "train"
        candidate_ids = output.router_candidate_ids[round_index]
        raw_logits = output.router_logits[round_index].detach().float()
        scores = (
            torch.sigmoid(raw_logits / model_config.depth_router.temperature)
            * model_config.depth_router.alpha
        )
        raw["candidate_global_token_ids"] = candidate_ids.detach().cpu().tolist()
        raw["candidate_sample_ids"] = _metadata_for_candidates(
            batch, candidate_ids, field="sample_ids"
        ).tolist()
        raw["candidate_original_positions"] = _metadata_for_candidates(
            batch, candidate_ids, field="original_position_ids"
        ).tolist()
        raw["candidate_scores"] = scores.cpu().tolist()
        routes.append(raw)
        tensors[f"forward/{base}/router_scores_round_{round_index}"] = _snapshot(scores)
        tensors[f"forward/{base}/selected_gates_round_{round_index}"] = _snapshot(
            plan.selected_gates
        )


def run_reference(config: ReferenceRunConfig) -> Path:
    started_source = source_snapshot()
    config.validate()
    require_fresh_artifact_directory(config.output)
    configure_determinism(config.seed, strict=config.strict)
    device = _device(config.device)
    dtype = torch.float32 if config.precision == "fp32" else torch.bfloat16
    preset = load_preset_config("tiny", config.preset_config)
    model_config = TinyMoRConfig(
        **dict(preset.model),
        architecture=config.architecture or preset.architecture,
        depth_router=config.depth_router or preset.depth_router,
    )
    model = TinyMoRModel(model_config, seed=config.seed).to(device=device, dtype=dtype)
    optimizer = MasterWeightAdamW(model.parameters(), lr=config.lr, eps=config.adam_eps)
    replay = _route_index(config.replay_from)
    tensors: dict[str, torch.Tensor] = {
        f"initial/{name}": _snapshot(parameter) for name, parameter in model.named_parameters()
    }
    routes: list[dict[str, Any]] = []
    last_communication: dict[str, int] = {}

    def run_step(step: int, *, prefix: str = "") -> None:
        nonlocal last_communication
        optimizer.zero_grad()
        for microbatch in range(config.num_microbatches):
            batch = make_synthetic_batch(
                seq_lens=config.seq_lens,
                vocab_size=model_config.vocab_size,
                seed=config.seed + step * 100 + microbatch,
                extreme_routing=microbatch == config.num_microbatches - 1,
            ).to(device)
            phase = prefix.rstrip("/") or "train"
            plans = replay.get((phase, step, microbatch)) if config.route_mode == "replay" else None
            if config.route_mode == "replay" and plans is None:
                raise KeyError(f"replay artifact has no RoutePlan for step={step}, mb={microbatch}")
            output = model(batch, route_mode=config.route_mode, replay_plans=plans)
            last_communication = dict(output.communication)
            (output.total_loss / config.num_microbatches).backward()
            _record_forward(
                prefix=prefix,
                step=step,
                microbatch=microbatch,
                batch=batch,
                output=output,
                model_config=model_config,
                tensors=tensors,
                routes=routes,
            )
        for name, parameter in model.named_parameters():
            # An expert that received no tokens has no autograd edge. Its
            # mathematical gradient is zero; retain it in the full vector
            # without materializing .grad or changing optimizer skip semantics.
            tensors[f"gradient/{prefix}step_{step:03d}/{name}"] = (
                _snapshot(parameter.grad)
                if parameter.grad is not None
                else torch.zeros_like(parameter)
            )
        master_before = {
            name: _snapshot(master)
            for (name, _), master in zip(
                model.named_parameters(), optimizer.master_parameters, strict=True
            )
        }
        grad_norm = optimizer.step(clip_grad=config.clip_grad)
        tensors[f"gradient/{prefix}step_{step:03d}/global_norm"] = _snapshot(grad_norm.reshape(()))
        for (name, parameter), master in zip(
            model.named_parameters(), optimizer.master_parameters, strict=True
        ):
            tensors[f"update/{prefix}step_{step:03d}/{name}"] = (
                _snapshot(master) - master_before[name]
            )
            tensors[f"post_step/{prefix}step_{step:03d}/{name}"] = _snapshot(parameter)

    for step in range(config.steps):
        run_step(step)

    checkpoint_metadata = {
        "model": model_config.to_dict(),
        "dependency_versions": collect_version_manifest(),
    }
    if config.checkpoint_roundtrip:
        checkpoint = config.output / "reference-checkpoint.pt"
        save_reference_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            step=config.steps,
            metadata=checkpoint_metadata,
        )
        resumed_model = TinyMoRModel(model_config, seed=config.seed + 999).to(
            device=device, dtype=dtype
        )
        resumed_optimizer = MasterWeightAdamW(
            resumed_model.parameters(), lr=config.lr, eps=config.adam_eps
        )
        restored_step, restored_metadata = load_reference_checkpoint(
            checkpoint, model=resumed_model, optimizer=resumed_optimizer
        )
        if restored_step != config.steps or restored_metadata["model"] != model_config.to_dict():
            raise RuntimeError("reference checkpoint metadata did not round-trip")
        model = resumed_model
        optimizer = resumed_optimizer
        run_step(config.steps, prefix="resume/")

    metadata = {
        "source_snapshot": started_source,
        "backend": "reference",
        "preset": "tiny",
        "preset_config_source": str(preset.source),
        "precision": config.precision,
        "route_mode": config.route_mode,
        "seed": config.seed,
        "steps": config.steps,
        "num_microbatches": config.num_microbatches,
        "checkpoint_next_step": config.checkpoint_roundtrip,
        "initialized_parameters": [name for name, _ in model.named_parameters()],
        "model_config": model_config.to_dict(),
        "architecture": model_config.architecture.to_dict(),
        "optimizer": {
            "name": "adamw",
            "lr": config.lr,
            "adam_eps": config.adam_eps,
            "clip_grad": config.clip_grad,
        },
        "global_batch": {
            "sequence_lengths": list(config.seq_lens),
            "sharded_across_dense_dp": False,
        },
        "versions": collect_version_manifest(),
        "communication": {
            **last_communication,
            # The reference optimizer is local.  This compatibility value is
            # an upper bound, not a claim that bucket synchronization ran.
            "physical_bucket_sync_dispatch_max": 1,
        },
    }
    return save_artifact(config.output, metadata=metadata, tensors=tensors, routes=routes)


__all__ = ["ReferenceRunConfig", "run_reference"]
