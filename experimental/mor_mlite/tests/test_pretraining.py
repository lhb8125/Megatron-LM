"""CPU checks of the new experiment contract, without requiring MLite."""

import json
import struct
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from mor_mlite.pretraining.config import Experiment, milestones, parameter_seed
from mor_mlite.pretraining.data import IndexedCorpus, TokenStream, sha256_file, validation_document
from mor_mlite.pretraining.tuning import Trial, next_mbs, select_mbs, validate_ep_locality


def test_attention_selector_against_pinned_native_source():
    import ast
    import os
    from pathlib import Path

    from mor_mlite.pretraining.runtime import ATTENTION_BACKEND

    native = os.environ.get("MOR_NATIVE_ROOT")
    if not native:
        pytest.skip("set MOR_NATIVE_ROOT to check the pinned MLite selector without CUDA")
    path = Path(native) / "experimental/lite/megatron/lite/runtime/backends/mlite/runtime.py"
    module = ast.parse(path.read_text())
    selector = next(
        n
        for n in module.body
        if isinstance(n, ast.FunctionDef) and n.name == "_apply_attention_backend_env"
    )
    assignment = next(
        n
        for n in selector.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "env_overrides" for t in n.targets)
    )
    supported = ast.literal_eval(assignment.value)
    assert supported[ATTENTION_BACKEND] == ("0", "1", "0")
    assert "te" not in supported


def test_hostlist_expansion():
    from mor_mlite.pretraining.runtime import expand_hostlist

    assert expand_hostlist("nvl72068-T[01-03,05],nvl72069-T01") == [
        "nvl72068-T01",
        "nvl72068-T02",
        "nvl72068-T03",
        "nvl72068-T05",
        "nvl72069-T01",
    ]
    assert expand_hostlist("rack[1-2]n[01-02]") == ["rack1n01", "rack1n02", "rack2n01", "rack2n02"]
    with pytest.raises(ValueError):
        expand_hostlist("node[3-1]")


def test_job_snapshot_includes_configs_and_refuses_mutation(tmp_path):
    from mor_mlite.pretraining.snapshot import freeze

    package = tmp_path / "package"
    (package / "src/mor_mlite").mkdir(parents=True)
    (package / "configs").mkdir()
    (package / "src/mor_mlite/__init__.py").write_text("# code v1\n")
    with pytest.raises(ValueError, match="topology"):
        freeze(package, tmp_path / "missing-config")
    (package / "configs/topologies.json").write_text("{}")
    store = tmp_path / "snapshots"
    preview = freeze(package, store, write=False)
    assert not store.exists()
    frozen = freeze(package, store)
    assert frozen == preview
    assert (frozen / "configs/topologies.json").read_text() == "{}"
    assert freeze(package, store) == frozen
    (frozen / "src/mor_mlite/__init__.py").write_text("# unexpected mutation\n")
    with pytest.raises(ValueError, match="changed"):
        freeze(package, store)
    (package / "src/mor_mlite/__init__.py").write_text("# code v2\n")
    assert freeze(package, store) != frozen


def test_deterministic_kernels_are_enabled_before_device_setup(monkeypatch):
    import sys

    torch = pytest.importorskip("torch")
    from mor_mlite.pretraining.runtime import distributed_environment

    calls = []
    monkeypatch.delenv("SLURM_PROCID", raising=False)
    monkeypatch.setenv("LOCAL_RANK", "2")
    monkeypatch.setenv("MEGATRON_LITE_DETERMINISTIC", "0")
    monkeypatch.setattr(torch.cuda, "set_device", lambda rank: calls.append(("device", rank)))
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    monkeypatch.setitem(
        sys.modules,
        "megatron.lite.primitive.deterministic",
        SimpleNamespace(set_deterministic=lambda seed: calls.append(("deterministic", seed))),
    )
    distributed_environment()
    assert calls == [("deterministic", 1234), ("device", 2)]


def test_performance_pilot_is_forbidden_for_formal_training():
    from mor_mlite.pretraining.config import validate_data_purpose

    manifest = {"provenance": {"purpose": "performance-pilot-only"}}
    validate_data_purpose(manifest, "tune")
    with pytest.raises(ValueError, match="pilot"):
        validate_data_purpose(manifest, "train")


@pytest.mark.parametrize("future_leak", [False, True])
def test_native_causal_probe_rejects_suffix_leakage(monkeypatch, future_leak):
    import sys
    from contextlib import nullcontext

    torch = pytest.importorskip("torch")
    from mor_mlite.pretraining import causal_probe, train

    model = SimpleNamespace(
        config=SimpleNamespace(vocab_size=256), experiment_arm="D", causal_route_traces=[]
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda _: None)
    monkeypatch.setattr(train, "models", lambda _: [model])
    monkeypatch.setattr(train, "packed", lambda ids: SimpleNamespace(ids=ids, extras={}))
    monkeypatch.setitem(
        sys.modules,
        "mor_mlite.qwen3_moe_mor.protocol",
        SimpleNamespace(unpack_forward_output=lambda model, batch, values: values),
    )

    class Runtime:
        def eval_mode(self, _):
            return nullcontext()

        def forward_backward(self, handle, batches, capture, **kwargs):
            batch = next(batches)
            assert batch.extras["mor_return_full_logits"] is True
            values = torch.tensor(batch.ids, dtype=torch.float32).flatten()
            values = values.mean().expand_as(values) if future_leak else values.cumsum(0)
            rows = torch.arange(values.numel())
            model.causal_route_traces = [
                {"selected_rows": rows},
                {"selected_rows": rows[rows % 2 == 1]},
                {"selected_rows": rows[rows % 4 == 3]},
            ]
            capture({"logits": values[:, None].expand(-1, 256), "loss": torch.tensor(0.0)})

    if future_leak:
        with pytest.raises(AssertionError):
            causal_probe.check(Runtime(), None)
    else:
        evidence = causal_probe.check(Runtime(), None)
        assert evidence["comparisons_per_rank"] == 8
        assert evidence["max_abs_error_local"] == 0
        assert evidence["zero_active_rank_rounds"] > 0


def test_campaign_is_stage_major_and_pairs_have_signed_differences():
    from mor_mlite.pretraining.campaign import next_stage, paired_results

    assert next_stage({}, total_steps=5960)["arm"] == "A"
    assert next_stage({"A": 120}, total_steps=5960)["arm"] == "B"
    assert next_stage(dict.fromkeys("ABCD", 120), total_steps=5960)["to_step"] == 1193
    assert next_stage(dict.fromkeys("ABCD", 1193), total_steps=5960)["to_step"] == 5960
    assert next_stage(dict.fromkeys("ABCD", 5960), total_steps=5960) is None
    rows = {
        arm: [{"tokens": 0, "targets": 100, "nll": nll}]
        for arm, nll in (("A", 5.0), ("B", 4.0), ("C", 4.5), ("D", 4.2))
    }
    paired = paired_results(rows)[0]["delta_nll"]
    assert paired["B-A"] == -1
    assert paired["B-C"] == -0.5
    assert paired["D-B"] == pytest.approx(0.2)
    rows["D"][0]["targets"] = 99
    with pytest.raises(ValueError, match="target"):
        paired_results(rows)


def test_raw_index_writer_roundtrip(tmp_path):
    from mor_mlite.pretraining.prepare_raw import write_index

    prefix = tmp_path / "raw"
    prefix.with_suffix(".bin").write_bytes(np.array([1, 9, 3, 4, 9], dtype="<i4").tobytes())
    write_index(prefix.with_suffix(".idx"), [2, 3])
    corpus = IndexedCorpus(prefix)
    assert corpus.document(0).tolist() == [1, 9]
    assert corpus.document(1).tolist() == [3, 4, 9]
    with pytest.raises(FileExistsError):
        write_index(prefix.with_suffix(".idx"), [2, 3])


def test_global_batch_hash_is_partition_independent():
    import hashlib

    from mor_mlite.pretraining.data import global_input_digest

    records = [(i, hashlib.sha256(bytes([i])).hexdigest()) for i in range(8)]
    expected = global_input_digest(records, first_sample=0, batch_size=8)
    assert expected == global_input_digest(
        records[::2] + records[1::2], first_sample=0, batch_size=8
    )
    with pytest.raises(ValueError, match="missing"):
        global_input_digest(records[:-1] + records[:1], first_sample=0, batch_size=8)


def test_flop_accounting_respects_recurrence_and_active_tokens():
    from mor_mlite.pretraining.cost import forward_flops

    cfg = SimpleNamespace(
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=256,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
    )
    a = forward_flops("A", cfg, [12])
    b = forward_flops("B", cfg, [12])
    c = forward_flops("C", cfg, [12])
    d = forward_flops("D", cfg, [12], [[12], [8], [4]])
    assert a == b
    assert c < d < b
    assert forward_flops("D", cfg, [12], [[12], [0], [0]]) > c * 0.5
    with pytest.raises(ValueError):
        forward_flops("D", cfg, [12])


def test_validation_keeps_incomplete_tail(tmp_path, monkeypatch):
    from contextlib import nullcontext

    torch = pytest.importorskip("torch")
    from mor_mlite.pretraining import train

    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda _: None)
    monkeypatch.setattr(train, "packed", lambda tokens: tokens)
    monkeypatch.setattr(train, "models", lambda _: [None])
    monkeypatch.setattr(train, "observed_flops", lambda *args: 0)

    class Runtime:
        def eval_mode(self, _):
            return nullcontext()

        def forward_backward(self, handle, data, loss_fn, **kwargs):
            count = next(data).shape[-1]
            loss_fn(
                {
                    "loss": torch.tensor(2.0 if count > 1 else 0.0),
                    "mor_router_aux_loss": torch.tensor(0.0),
                }
            )

    class Stream:
        def __len__(self):
            return 5

        def read(self, offset, count):
            assert offset + count <= 5
            return np.arange(offset, offset + count)

    result = train.evaluate(
        Runtime(),
        SimpleNamespace(dp_rank=0),
        Stream(),
        SimpleNamespace(seq_length=4, world_size=1, arm="B"),
    )
    assert result["validation_input_tokens"] == 5
    assert result["targets"] == 3
    assert result["nll"] == 2
    assert result["discarded_validation_tail_tokens"] == 0


def test_selective_causal_policy_multiple_samples():
    torch = pytest.importorskip("torch")
    from mor_mlite.pretraining.execution import run_causal

    class Layer(torch.nn.Module):
        def forward(self, x, *, position_ids, packed_seq_params):
            out = []
            cursor = 0
            for length in packed_seq_params:
                part = x[cursor : cursor + length]
                denom = torch.arange(1, length + 1, dtype=x.dtype)[:, None, None]
                out.append(part + 0.1 * part.cumsum(0) / denom)
                cursor += length
            assert cursor == len(x)
            return torch.cat(out)

    routers = [
        SimpleNamespace(
            proj=torch.nn.Linear(1, 1, bias=False), config=SimpleNamespace(temperature=1, alpha=0.1)
        )
        for _ in range(3)
    ]
    for router in routers:
        router.proj.weight.data.fill_(1)

    def execute(x, lengths):
        samples = torch.repeat_interleave(torch.arange(len(lengths)), torch.tensor(lengths))
        positions = torch.cat([torch.arange(n) for n in lengths])
        with torch.no_grad():
            return run_causal(
                x,
                layers=[Layer()],
                routers=routers,
                sample_ids=samples,
                positions=positions,
                padding_mask=torch.zeros(len(x), dtype=torch.bool),
                position_ids=positions[None],
                packed_seq_params=lengths,
                make_packed=lambda counts: counts.tolist(),
                start=0,
                recurrent=1,
                end=0,
            )

    x = torch.tensor([-2.0, 4.0, -3.0, 4.0, 1.0, -2.0, 3.0, -4.0]).reshape(-1, 1, 1)
    full, routes = execute(x, [4, 4])
    assert 0 < routes[1]["selected_rows"].numel() < len(x)
    for sample in range(2):
        for length in range(1, 5):
            start = 4 * sample
            prefix, sub_routes = execute(x[start : start + length], [length])
            torch.testing.assert_close(prefix, full[start : start + length], rtol=1e-5, atol=1e-6)
            for a, b in zip(routes, sub_routes, strict=True):
                rows = a["selected_rows"]
                expected = rows[(rows >= start) & (rows < start + length)] - start
                assert torch.equal(expected, b["selected_rows"])


def test_raw_shard_resume_and_tampering(tmp_path, monkeypatch):
    from mor_mlite.pretraining import prepare_raw as prep

    raw = tmp_path / "source"
    raw.write_bytes(b"raw fixture")
    monkeypatch.setattr(prep, "text_batches", lambda _: iter([["ab", "c", "de"]]))

    class Tokenizer:
        eos_token_id = 9

        def __call__(self, texts, **kwargs):
            return {"input_ids": [[ord(c) for c in text] for text in texts]}

    target = tmp_path / "shard"
    state = prep.encode_shard(raw, target, Tokenizer(), 4, {"seed": 1234})
    assert state["tokens"] == 5  # whole document boundary, no truncation
    assert state["documents"] == 2
    assert prep.encode_shard(raw, target, Tokenizer(), 4, {"seed": 1234}) == state
    with pytest.raises(ValueError, match="different"):
        prep.encode_shard(raw, target, Tokenizer(), 5, {"seed": 1234})
    (target / "text.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        prep.encode_shard(raw, target, Tokenizer(), 4, {"seed": 1234})


@pytest.mark.parametrize("world", [32, 64])
@pytest.mark.parametrize("mbs", [1, 2, 4, 8])
def test_accumulation(world, mbs):
    config = Experiment("B", world_size=world, micro_batch_size=mbs)
    assert world * mbs * config.accumulation_steps == 2048
    assert config.tokens_per_step == 8388608


def test_architecture():
    a, b, c, d = [Experiment(x) for x in "ABCD"]
    assert [x.physical_layers for x in (a, b, c, d)] == [48, 20, 20, 20]
    assert [len(x.layer_indices()) for x in (a, b, c, d)] == [48, 48, 20, 48]
    assert b.layer_indices().count(3) == 3
    assert b.layer_indices().count(17) == 1
    assert len(set(b.layer_indices())) == len(set(c.layer_indices())) == 20
    assert parameter_seed(1234, "layers.3.weight") != parameter_seed(1234, "layers.4.weight")


def test_formal_model_rejects_tiny_or_wrong_tokenizer():
    from mor_mlite.pretraining.config import validate_base_model

    cfg = {
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
        "eos_token_id": 151643,
    }
    manifest = {"eos": 151643, "vocab_size": 151669}
    validate_base_model(cfg, manifest)
    with pytest.raises(ValueError, match="architecture"):
        validate_base_model({**cfg, "hidden_size": 128}, manifest)
    with pytest.raises(ValueError, match="EOS"):
        validate_base_model(cfg, {**manifest, "eos": 151645})


def test_scheduler_and_milestones():
    c = Experiment("A")
    assert c.learning_rate(0, 10000) == 0
    assert c.learning_rate(100, 10000) == c.lr
    assert c.learning_rate(10000, 10000) == c.min_lr
    assert milestones(5740, c.tokens_per_step)[:3] == (0, 120, 1193)
    assert milestones(5740, c.tokens_per_step)[-1] == 5740
    with pytest.raises(ValueError):
        c.learning_rate(10001, 10000)


def trial(mbs=1, speed=100, memory=0.5):
    return Trial(mbs, speed, memory, 20, 30, True, True, True)


def test_tuning_stops_and_tie_break():
    assert next_mbs(trial(memory=0.8)) == 2
    assert next_mbs(trial(memory=0.8001)) is None
    assert next_mbs(trial(mbs=8)) is None
    assert select_mbs([trial(), trial(2, 102), trial(4, 103)]) == 1
    assert select_mbs([trial(), trial(2, 104), trial(4, 110, 0.88)]) == 4
    assert select_mbs([trial(), trial(2, 200, 0.91)]) == 1
    with pytest.raises(ValueError):
        select_mbs([trial(memory=0.81), trial(2)])
    with pytest.raises(ValueError):
        select_mbs([replace(trial(), checkpoint_ok=False)])
    with pytest.raises(ValueError):
        select_mbs([trial(speed=float("nan"))])


def test_actual_topology_not_just_segment():
    rows = [
        {
            "rank": r,
            "hostname": f"node{r // 4}",
            "nvl_domain": f"domain{r // 16}",
            "ep_members": list(range(r // 16 * 16, (r // 16 + 1) * 16)),
        }
        for r in range(32)
    ]
    assert validate_ep_locality(rows, world_size=32)["passed"]
    rows[0]["nvl_domain"] = "elsewhere"
    with pytest.raises(ValueError, match="NVL"):
        validate_ep_locality(rows, world_size=32)


def write_corpus(tmp_path):
    docs = [
        np.array([1, 2, 3, 9], dtype="<i4"),
        np.array([4, 5, 9], dtype="<i4"),
        np.array([1, 2, 3, 9], dtype="<i4"),
    ]
    prefix = tmp_path / "corpus"
    prefix.with_suffix(".bin").write_bytes(b"".join(d.tobytes() for d in docs))
    prefix.with_suffix(".idx").write_bytes(
        b"MMIDIDX\0\0"
        + struct.pack("<QBQQ", 1, 4, 3, 4)
        + np.array([4, 3, 4], dtype="<i4").tobytes()
        + np.array([0, 16, 28], dtype="<i8").tobytes()
        + np.array([0, 1, 2, 3], dtype="<i8").tobytes()
    )
    return prefix, docs


def test_index_split_and_stream(tmp_path):
    prefix, docs = write_corpus(tmp_path)
    corpus = IndexedCorpus(prefix)
    assert len(corpus) == 3
    assert np.array_equal(corpus.document(1), docs[1])
    assert validation_document(docs[0].tobytes()) == validation_document(docs[2].tobytes())
    np.save(tmp_path / "train_documents.npy", np.array([1, 0, 2], dtype=np.int64))
    np.save(tmp_path / "train_offsets.npy", np.array([0, 3, 7, 11], dtype=np.int64))
    files = {f: sha256_file(tmp_path / f) for f in ("train_documents.npy", "train_offsets.npy")}
    (tmp_path / "manifest.json").write_text(json.dumps({"prefix": str(prefix), "files": files}))
    stream = TokenStream(tmp_path)
    assert stream.read(1, 8).tolist() == [5, 9, 1, 2, 3, 9, 1, 2]
    with pytest.raises(ValueError, match="one-pass"):
        stream.read(10, 2)

    # Global ordering is identical despite different microbatch partitions.
    def partition(dp, mbs):
        return np.concatenate(
            [
                stream.microbatch(
                    0, m, dp_rank=r, dp_size=dp, mbs=mbs, gbs=4, seq_length=2
                ).reshape(-1)
                for m in range(4 // (dp * mbs))
                for r in range(dp)
            ]
        )

    assert np.array_equal(partition(1, 1), partition(2, 2))
    prefix.with_suffix(".bin").write_bytes(b"short")
    with pytest.raises(ValueError):
        IndexedCorpus(prefix)


def test_repeated_module_gradient_is_sum():
    torch = pytest.importorskip("torch")
    from mor_mlite.pretraining.execution import run_fixed

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.2, dtype=torch.float64))

        def forward(self, x, **kwargs):
            return x * self.weight

    layer = Layer()
    x = torch.tensor([2.0], dtype=torch.float64)
    y = run_fixed(x, [layer], (0, 0, 0), position_ids=None, packed_seq_params=None)
    y.sum().backward()
    assert torch.allclose(layer.weight.grad, torch.tensor(3 * 2 * 1.2**2, dtype=torch.float64))


def test_causal_policy_prefix_and_zero_active():
    torch = pytest.importorskip("torch")
    from mor_mlite.pretraining.execution import run_causal

    class Layer(torch.nn.Module):
        def forward(self, x, **kwargs):
            # Independent causal prefix average as a small attention oracle.
            return x + x.cumsum(0) / torch.arange(1, len(x) + 1, dtype=x.dtype)[:, None, None]

    routers = []
    for sign in (1, -1, -1):
        router = SimpleNamespace(
            proj=torch.nn.Linear(2, 1, bias=False), config=SimpleNamespace(temperature=1, alpha=0.1)
        )
        router.proj.weight.data.fill_(sign)
        routers.append(router)
    layers = [Layer(), Layer(), Layer()]

    def execute(x):
        n = len(x)
        with torch.no_grad():
            return run_causal(
                x,
                layers=layers,
                routers=routers,
                sample_ids=torch.zeros(n, dtype=torch.long),
                positions=torch.arange(n),
                padding_mask=torch.zeros(n, dtype=torch.bool),
                position_ids=torch.arange(n)[None],
                packed_seq_params=None,
                make_packed=lambda counts: None,
                start=1,
                recurrent=1,
                end=1,
            )

    inputs = torch.ones(6, 1, 2)
    full, trace = execute(inputs)
    assert trace[0]["selected_rows"].tolist() == list(range(6))
    assert trace[1]["selected_rows"].numel() == trace[2]["selected_rows"].numel() == 0
    changed = inputs.clone()
    changed[3:] = -100
    alternate, _ = execute(changed)
    assert torch.allclose(full[:3], alternate[:3], atol=1e-6, rtol=1e-5)
    for length in range(1, 7):
        prefix, _ = execute(inputs[:length])
        assert torch.allclose(full[:length], prefix, atol=1e-6, rtol=1e-5)
