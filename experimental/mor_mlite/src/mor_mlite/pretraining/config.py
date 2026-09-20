"""Pure-Python scientific and launch contracts for the four-arm experiment."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Experiment:
    arm: str
    world_size: int = 64
    micro_batch_size: int = 1
    global_batch_size: int = 2048
    seq_length: int = 4096
    ep: int = 16
    segment: int = 4
    seed: int = 1234
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_fraction: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    weight_decay: float = 0.1
    clip_grad: float = 1.0
    moe_aux: float = 0.001
    depth_aux: float = 0.001

    def __post_init__(self):
        if self.arm not in "ABCD" or len(self.arm) != 1:
            raise ValueError("arm must be A, B, C, or D")
        if self.world_size not in (32, 64) or self.ep != 16 or self.segment != 4:
            raise ValueError("experiment requires 32/64 ranks, EP16, segment=4")
        if self.micro_batch_size not in (1, 2, 4, 8):
            raise ValueError("MBS must be 1, 2, 4, or 8")
        if self.global_batch_size != 2048 or self.seq_length != 4096:
            raise ValueError("scientific contract requires GBS2048 and sequence length4096")
        if self.global_batch_size % (self.world_size * self.micro_batch_size):
            raise ValueError("global batch must divide evenly over DP and microbatches")
        values = (self.lr, self.min_lr, self.adam_eps, self.clip_grad)
        if any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError("optimizer values must be positive and finite")
        if not 0 < self.warmup_fraction < 1 or self.min_lr > self.lr:
            raise ValueError("invalid learning rate schedule")
        if not all(0 <= b < 1 for b in (self.adam_beta1, self.adam_beta2)):
            raise ValueError("Adam betas must be in [0,1)")
        if any(
            not math.isfinite(v) or v < 0 for v in (self.weight_decay, self.moe_aux, self.depth_aux)
        ):
            raise ValueError("regularization coefficients must be nonnegative and finite")

    @property
    def physical_layers(self):
        return 48 if self.arm == "A" else 20

    @property
    def logical_layers(self):
        return 20 if self.arm == "C" else 48

    @property
    def accumulation_steps(self):
        return self.global_batch_size // (self.world_size * self.micro_batch_size)

    @property
    def tokens_per_step(self):
        return self.global_batch_size * self.seq_length

    def layer_indices(self):
        if self.arm in ("A", "C"):
            return tuple(range(self.physical_layers))
        return tuple(range(3)) + tuple(range(3, 17)) * 3 + tuple(range(17, 20))

    def to_dict(self):
        return asdict(self)

    def learning_rate(self, consumed_tokens: int, total_tokens: int):
        if total_tokens <= 0 or not 0 <= consumed_tokens <= total_tokens:
            raise ValueError("invalid token cursor")
        warmup = total_tokens * self.warmup_fraction
        if consumed_tokens < warmup:
            return self.lr * consumed_tokens / warmup
        progress = (consumed_tokens - warmup) / (total_tokens - warmup)
        return self.min_lr + (self.lr - self.min_lr) * (1 + math.cos(math.pi * progress)) / 2


def parameter_seed(seed: int, canonical_name: str) -> int:
    """Stable across processes, model construction order, and arm B/C/D."""
    raw = hashlib.sha256(f"{seed}:{canonical_name}".encode()).digest()
    return int.from_bytes(raw[:8], "little") % (2**63 - 1)


def validate_base_model(model_config, data_manifest):
    expected = {
        "model_type": "qwen3_moe",
        "num_hidden_layers": 48,
        "hidden_size": 2048,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "vocab_size": 151936,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 768,
        "tie_word_embeddings": False,
    }
    for key, value in expected.items():
        if model_config.get(key) != value:
            raise ValueError(f"formal/tuning run requires Qwen3-30B-A3B architecture: {key}")
    if model_config.get("eos_token_id") != data_manifest["eos"]:
        raise ValueError("model/tokenized data EOS mismatch")
    if model_config["vocab_size"] < data_manifest["vocab_size"]:
        raise ValueError("model vocabulary is smaller than the frozen tokenizer")


def validate_data_purpose(manifest, mode):
    if mode not in ("train", "tune"):
        raise ValueError("unknown training mode")
    if (
        mode == "train"
        and manifest.get("provenance", {}).get("purpose") == "performance-pilot-only"
    ):
        raise ValueError("performance pilot data cannot be used for formal training")


def milestones(total_steps: int, tokens_per_step: int) -> tuple[int, ...]:
    if total_steps < 1 or tokens_per_step < 1:
        raise ValueError("empty training budget")
    tokens = [10**9, 10**10]
    tokens += list(range(2 * 10**10, total_steps * tokens_per_step, 10**10))
    return tuple(
        sorted(
            {0, total_steps, *(min(total_steps, math.ceil(t / tokens_per_step)) for t in tokens)}
        )
    )
