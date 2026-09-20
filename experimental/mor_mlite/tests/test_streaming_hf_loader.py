from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

CHECKPOINT_SOURCE = (
    Path(__file__).parents[1] / "src" / "mor_mlite" / "qwen3_moe_mor" / "checkpoint.py"
)


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} is missing")


def test_production_loader_does_not_delegate_to_generic_materializing_loader() -> None:
    tree = ast.parse(CHECKPOINT_SOURCE.read_text(encoding="utf-8"))
    imported_hf_helpers = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "megatron.lite.primitive.ckpt.hf_weights"
        for alias in node.names
    }
    assert "load_hf_weights" not in imported_hf_helpers
    assert {"SafeTensorReader", "split_dim", "split_gate_up", "unwrap_model"} <= (
        imported_hf_helpers
    )

    loader = _function(tree, "load_hf_weights")
    called_names = {
        node.func.id
        for node in ast.walk(loader)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_fold_source_groups_streaming" in called_names


def test_streaming_fold_reads_and_transforms_one_logical_group_per_iteration() -> None:
    tree = ast.parse(CHECKPOINT_SOURCE.read_text(encoding="utf-8"))
    fold = _function(tree, "_fold_source_groups_streaming")
    loops = [node for node in ast.walk(fold) if isinstance(node, ast.For)]
    assert any(
        isinstance(loop.iter, ast.Name)
        and loop.iter.id == "source_groups"
        and isinstance(loop.target, ast.Name)
        and loop.target.id == "source_group"
        for loop in loops
    )
    called_names = {
        node.func.id
        for node in ast.walk(fold)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_read_source_group" in called_names
    assert any(
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "torch"
        and node.attr == "float32"
        for node in ast.walk(fold)
    )


def _runtime_types():
    torch = pytest.importorskip("torch")
    pytest.importorskip("megatron.lite.model.qwen3_moe.lite.checkpoint")

    from megatron.lite.model.qwen3_moe.config import Qwen3MoEConfig

    from mor_mlite.config import MoRArchitectureConfig
    from mor_mlite.qwen3_moe_mor.checkpoint import Qwen3MoEMoRWeightSpec

    architecture = MoRArchitectureConfig.tiny()
    config = Qwen3MoEConfig(
        num_hidden_layers=architecture.logical_num_layers,
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=257,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        layer_types=["full_attention"] * architecture.logical_num_layers,
    )
    return torch, config, architecture, Qwen3MoEMoRWeightSpec


def test_qwen_recurrent_qkv_groups_are_folded_in_fp32_one_group_at_a_time() -> None:
    torch, config, architecture, weight_spec_type = _runtime_types()
    from mor_mlite.qwen3_moe_mor.checkpoint import _fold_source_groups_streaming

    spec = weight_spec_type(config, architecture, folding_policy="mean")
    native_name = "layers.1.attn.qkv.linear.weight"
    groups = spec.source_groups(native_name)
    assert groups == (
        (
            "model.layers.1.self_attn.q_proj.weight",
            "model.layers.1.self_attn.k_proj.weight",
            "model.layers.1.self_attn.v_proj.weight",
        ),
        (
            "model.layers.3.self_attn.q_proj.weight",
            "model.layers.3.self_attn.k_proj.weight",
            "model.layers.3.self_attn.v_proj.weight",
        ),
        (
            "model.layers.5.self_attn.q_proj.weight",
            "model.layers.5.self_attn.k_proj.weight",
            "model.layers.5.self_attn.v_proj.weight",
        ),
    )

    tensors = {}
    for layer, value in ((1, 1.0), (3, 3.0), (5, 8.0)):
        prefix = f"model.layers.{layer}.self_attn"
        tensors[f"{prefix}.q_proj.weight"] = torch.full((32, 32), value, dtype=torch.bfloat16)
        tensors[f"{prefix}.k_proj.weight"] = torch.full((16, 32), value + 1, dtype=torch.bfloat16)
        tensors[f"{prefix}.v_proj.weight"] = torch.full((16, 32), value + 2, dtype=torch.bfloat16)

    class Reader:
        def __init__(self) -> None:
            self.reads: list[str] = []

        @staticmethod
        def first_available(names):
            return next(iter(names))

        @staticmethod
        def has_tensor(_name):
            return False

        def get_tensor(self, name, *, device):
            assert device == "cpu"
            self.reads.append(name)
            return tensors[name].clone()

    transform_group_sizes: list[int] = []
    native_transform = spec._base.hf_to_native

    def observed_transform(name, values):
        transform_group_sizes.append(len(values))
        return native_transform(name, values)

    spec._base.hf_to_native = observed_transform
    reader = Reader()
    folded = _fold_source_groups_streaming(reader, spec, native_name, groups)

    expected_groups = [
        native_transform(native_name, [tensors[name] for name in group]).float() for group in groups
    ]
    expected = torch.stack(expected_groups).mean(dim=0)
    assert transform_group_sizes == [3, 3, 3]
    assert reader.reads == [name for group in groups for name in group]
    assert folded.dtype == torch.float32
    assert torch.equal(folded, expected)


def test_start_end_and_expert_source_groups_follow_exact_fold_map() -> None:
    _, config, architecture, weight_spec_type = _runtime_types()
    spec = weight_spec_type(config, architecture, folding_policy="mean")

    assert spec.source_groups("layers.0.mlp_norm.weight") == (
        ("model.layers.0.post_attention_layernorm.weight",),
    )
    assert spec.source_groups("layers.3.mlp_norm.weight") == (
        ("model.layers.7.post_attention_layernorm.weight",),
    )
    assert spec.source_groups("layers.2.moe.experts._fc1_weight_3") == (
        (
            "model.layers.2.mlp.experts.3.gate_proj.weight",
            "model.layers.2.mlp.experts.3.up_proj.weight",
        ),
        (
            "model.layers.4.mlp.experts.3.gate_proj.weight",
            "model.layers.4.mlp.experts.3.up_proj.weight",
        ),
        (
            "model.layers.6.mlp.experts.3.gate_proj.weight",
            "model.layers.6.mlp.experts.3.up_proj.weight",
        ),
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"pp_size": 2}, "PP=1"),
        ({"virtual_pipeline_size": 2}, "VPP=1"),
        ({"etp_size": 2}, "ETP=1"),
    ],
)
def test_streaming_loader_scope_rejects_unsupported_parallelism(
    overrides: dict[str, int], message: str
) -> None:
    torch, config, architecture, weight_spec_type = _runtime_types()
    from mor_mlite.qwen3_moe_mor.checkpoint import _validate_streaming_load_scope

    spec = weight_spec_type(config, architecture)
    parallel = SimpleNamespace(
        pp_size=1,
        pp_rank=0,
        virtual_pipeline_size=None,
        etp_size=1,
        etp_rank=0,
        tp_size=1,
        tp_rank=0,
        ep_size=1,
        ep_rank=0,
    )
    for name, value in overrides.items():
        setattr(parallel, name, value)

    class FakeModel(torch.nn.Module):
        pass

    with pytest.raises(ValueError, match=message):
        _validate_streaming_load_scope(FakeModel(), spec, parallel)


def _write_checkpoint_config(root: Path, config, **overrides) -> None:
    payload = {"model_type": "qwen3_moe", **config.to_dict(), **overrides}
    (root / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"")


def test_checkpoint_config_validates_every_layout_and_forward_field(tmp_path: Path) -> None:
    _, config, _, _ = _runtime_types()
    from mor_mlite.qwen3_moe_mor.checkpoint import _validate_hf_checkpoint_directory

    root = tmp_path / "valid"
    root.mkdir()
    _write_checkpoint_config(root, config)
    assert _validate_hf_checkpoint_directory(str(root), config) == root.resolve()

    for field, incompatible in (
        ("vocab_size", config.vocab_size - 1),
        ("hidden_size", config.hidden_size * 2),
        ("num_key_value_heads", 1),
        ("head_dim", config.head_dim * 2),
        ("moe_intermediate_size", config.moe_intermediate_size * 2),
        ("rope_theta", config.rope_theta / 2),
        ("rms_norm_eps", config.rms_norm_eps * 2),
    ):
        _write_checkpoint_config(root, config, **{field: incompatible})
        with pytest.raises(ValueError, match=field):
            _validate_hf_checkpoint_directory(str(root), config)

    _write_checkpoint_config(root, config, use_sliding_window=True)
    with pytest.raises(ValueError, match="use_sliding_window"):
        _validate_hf_checkpoint_directory(str(root), config)


def test_checkpoint_config_accepts_only_equivalent_theta_in_rope_parameters(
    tmp_path: Path,
) -> None:
    _, config, _, _ = _runtime_types()
    from mor_mlite.qwen3_moe_mor.checkpoint import _validate_hf_checkpoint_directory

    root = tmp_path / "rope"
    root.mkdir()
    _write_checkpoint_config(root, config, rope_parameters={"rope_theta": config.rope_theta})
    assert _validate_hf_checkpoint_directory(str(root), config) == root.resolve()

    payload = {"model_type": "qwen3_moe", **config.to_dict()}
    payload.pop("rope_theta")
    payload["rope_parameters"] = {"rope_theta": config.rope_theta}
    (root / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    assert _validate_hf_checkpoint_directory(str(root), config) == root.resolve()


@pytest.mark.parametrize(
    "rope_parameters",
    (
        {"rope_theta": 1_000_000.0, "rope_type": "default"},
        {"rope_theta": 1_000_000.0, "factor": 1.0},
        {"rope_theta": 1_000_000.0, "partial_rotary_factor": 1.0},
        {"rope_theta": 1_000_000.0, "unknown_future_semantics": False},
    ),
)
def test_checkpoint_config_rejects_all_other_rope_parameter_semantics(
    tmp_path: Path, rope_parameters: dict[str, object]
) -> None:
    _, config, _, _ = _runtime_types()
    from mor_mlite.qwen3_moe_mor.checkpoint import _validate_hf_checkpoint_directory

    root = tmp_path / "rope"
    root.mkdir()
    _write_checkpoint_config(root, config, rope_parameters=rope_parameters)
    with pytest.raises(ValueError, match="rope_parameters enables semantics unsupported"):
        _validate_hf_checkpoint_directory(str(root), config)


def test_checkpoint_config_rejects_invalid_or_conflicting_rope_parameters(
    tmp_path: Path,
) -> None:
    _, config, _, _ = _runtime_types()
    from mor_mlite.qwen3_moe_mor.checkpoint import _validate_hf_checkpoint_directory

    root = tmp_path / "rope"
    root.mkdir()
    _write_checkpoint_config(root, config, rope_parameters=[config.rope_theta])
    with pytest.raises(ValueError, match="rope_parameters must be an object or null"):
        _validate_hf_checkpoint_directory(str(root), config)

    _write_checkpoint_config(
        root,
        config,
        rope_parameters={"rope_theta": config.rope_theta / 2},
    )
    with pytest.raises(ValueError, match="conflicts with top-level rope_theta"):
        _validate_hf_checkpoint_directory(str(root), config)


def test_tp_qkv_and_vocab_shards_are_exact_after_folding() -> None:
    torch, config, architecture, weight_spec_type = _runtime_types()
    from mor_mlite.qwen3_moe_mor.checkpoint import _shard_dense_for_target

    spec = weight_spec_type(config, architecture)
    parallel = SimpleNamespace(tp_size=2, tp_rank=1)
    native_name = "layers.1.attn.qkv.linear.weight"
    q = torch.arange(32 * 32, dtype=torch.float32).reshape(32, 32)
    k = 10_000 + torch.arange(16 * 32, dtype=torch.float32).reshape(16, 32)
    v = 20_000 + torch.arange(16 * 32, dtype=torch.float32).reshape(16, 32)
    packed = spec._base.hf_to_native(native_name, [q, k, v])
    shard = _shard_dense_for_target(native_name, packed, spec, parallel)
    assert torch.equal(shard, packed.chunk(2, dim=0)[1])

    embedding = torch.arange(config.vocab_size * 2, dtype=torch.float32).reshape(
        config.vocab_size, 2
    )
    vocab_shard = _shard_dense_for_target("embed.embedding.weight", embedding, spec, parallel)
    assert vocab_shard.shape == (192, 2)
    assert torch.equal(vocab_shard[:65], embedding[192:])
    assert torch.count_nonzero(vocab_shard[65:]) == 0


@pytest.mark.parametrize("native_name", ("embed.embedding.weight", "head.col.linear.weight"))
@pytest.mark.parametrize("rows", (256, 384))
def test_vocab_sources_require_exact_logical_rows_before_tp_padding(
    native_name: str, rows: int
) -> None:
    torch, config, architecture, weight_spec_type = _runtime_types()
    from mor_mlite.qwen3_moe_mor.checkpoint import _shard_dense_for_target

    spec = weight_spec_type(config, architecture)
    parallel = SimpleNamespace(tp_size=2, tp_rank=0)
    source = torch.zeros((rows, config.hidden_size), dtype=torch.float32)

    with pytest.raises(ValueError, match=r"exactly logical vocab_size=257 rows"):
        _shard_dense_for_target(native_name, source, spec, parallel)


def test_source_loader_accepts_only_unscaled_bf16_fp16_fp32() -> None:
    torch, _, _, _ = _runtime_types()
    from mor_mlite.qwen3_moe_mor.checkpoint import _read_source_group

    class Reader:
        def __init__(self, dtype, metadata=()):
            self.dtype = dtype
            self.metadata = set(metadata)
            self.get_calls = 0

        @staticmethod
        def first_available(names):
            return next(iter(names))

        def has_tensor(self, name):
            return name in self.metadata

        def get_tensor(self, _name, *, device):
            assert device == "cpu"
            self.get_calls += 1
            return torch.ones((2, 2), dtype=self.dtype)

    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        reader = Reader(dtype)
        values = _read_source_group(reader, "native.weight", ("hf.weight",))
        assert values[0].dtype == dtype

    unsupported = [torch.float64, torch.int8]
    float8 = getattr(torch, "float8_e4m3fn", None)
    if float8 is not None:
        unsupported.append(float8)
    for dtype in unsupported:
        with pytest.raises(TypeError, match="unscaled BF16, FP16, or FP32 only"):
            _read_source_group(Reader(dtype), "native.weight", ("hf.weight",))

    for scale_name in (
        "hf.weight_scale_inv",
        "hf.scale",
        "hf.weight_scale",
        "hf.weight_shape",
    ):
        reader = Reader(torch.float32, metadata=(scale_name,))
        with pytest.raises(TypeError, match="unsupported scale metadata"):
            _read_source_group(reader, "native.weight", ("hf.weight",))
        assert reader.get_calls == 0


@pytest.mark.parametrize(
    "dtype_name",
    ("float64", "float8_e4m3fn", "float8_e5m2"),
)
def test_header_preflight_rejects_unsupported_source_dtype_before_payload_read(
    tmp_path: Path, dtype_name: str
) -> None:
    torch, _, _, _ = _runtime_types()
    save_file = pytest.importorskip("safetensors.torch").save_file
    from mor_mlite.qwen3_moe_mor.checkpoint import _preflight_source_tensor_headers

    dtype = getattr(torch, dtype_name, None)
    if dtype is None:
        pytest.skip(f"torch has no {dtype_name}")
    save_file({"hf.weight": torch.zeros((2, 2), dtype=dtype)}, str(tmp_path / "model.safetensors"))

    with pytest.raises(TypeError, match="unsupported safetensors dtype"):
        _preflight_source_tensor_headers(tmp_path, ("hf.weight",))


def test_header_preflight_rejects_scale_metadata(tmp_path: Path) -> None:
    torch, _, _, _ = _runtime_types()
    save_file = pytest.importorskip("safetensors.torch").save_file
    from mor_mlite.qwen3_moe_mor.checkpoint import _preflight_source_tensor_headers

    save_file(
        {
            "hf.weight": torch.zeros((2, 2), dtype=torch.float32),
            "hf.weight_scale_inv": torch.ones((1,), dtype=torch.float32),
        },
        str(tmp_path / "model.safetensors"),
    )

    with pytest.raises(TypeError, match="unsupported scale metadata"):
        _preflight_source_tensor_headers(tmp_path, ("hf.weight",))


def test_load_preflights_all_local_source_dtypes_before_first_target_copy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    torch = pytest.importorskip("torch")
    save_file = pytest.importorskip("safetensors.torch").save_file
    checkpoint = pytest.importorskip("mor_mlite.qwen3_moe_mor.checkpoint")

    save_file(
        {
            "hf.first": torch.ones((2, 2), dtype=torch.float32),
            "hf.second": torch.ones((2, 2), dtype=torch.float64),
        },
        str(tmp_path / "model.safetensors"),
    )

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight0 = torch.nn.Parameter(torch.zeros((2, 2)))
            self.weight1 = torch.nn.Parameter(torch.zeros((2, 2)))

    class BaseTransform:
        @staticmethod
        def hf_to_native(_name, values):
            return values[0]

    class Spec:
        num_experts = 1

        def __init__(self) -> None:
            self._base = BaseTransform()
            self._base_weight_map = {
                "weight0": ["hf.first"],
                "weight1": ["hf.second"],
            }

        def weight_map(self):
            return self._base_weight_map

        def source_groups(self, native_name):
            return (tuple(self._base_weight_map[native_name]),)

        @staticmethod
        def expert_global_id(_native_name):
            return None

        @staticmethod
        def tp_spec(_native_name):
            return None

    model = Model()
    monkeypatch.setattr(checkpoint, "_spec", lambda _config: Spec())
    monkeypatch.setattr(checkpoint, "_validate_streaming_load_scope", lambda *_args: None)
    monkeypatch.setattr(
        checkpoint,
        "_validate_hf_checkpoint_directory",
        lambda *_args: tmp_path,
    )

    with pytest.raises(TypeError, match="unsupported safetensors dtype"):
        checkpoint.load_hf_weights(
            model,
            str(tmp_path),
            SimpleNamespace(),
            SimpleNamespace(ep_size=1, ep_rank=0),
        )
    assert torch.count_nonzero(model.weight0) == 0
    assert torch.count_nonzero(model.weight1) == 0


def test_ep_loader_reads_only_local_experts_and_preserves_depth_router(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = pytest.importorskip("mor_mlite.qwen3_moe_mor.checkpoint")

    class Weights(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_parameter("weight0", torch.nn.Parameter(torch.zeros(1)))
            self.register_parameter("weight1", torch.nn.Parameter(torch.zeros(1)))

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layer = torch.nn.Module()
            layer.moe = torch.nn.Module()
            layer.moe.experts = torch.nn.Module()
            layer.moe.experts.fc1 = Weights()
            self.layers = torch.nn.ModuleList([layer])
            self.depth_routers = torch.nn.ModuleList([torch.nn.Linear(1, 1, bias=False)])

    class BaseTransform:
        @staticmethod
        def hf_to_native(_name, values):
            return values[0]

    class Spec:
        num_experts = 4

        def __init__(self) -> None:
            self._base = BaseTransform()
            self._base_weight_map = {
                f"layers.0.moe.experts._fc1_weight_{expert}": [f"hf.expert.{expert}"]
                for expert in range(4)
            }

        def weight_map(self):
            return self._base_weight_map

        def source_groups(self, native_name):
            return (tuple(self._base_weight_map[native_name]),)

        @staticmethod
        def expert_global_id(native_name):
            return int(native_name.rsplit("_", 1)[1])

        @staticmethod
        def expert_local_name(_native_name, local_index):
            return f"layers.0.moe.experts.fc1.weight{local_index}"

    reads: list[str] = []

    class Reader:
        def __init__(self, _path, *, device):
            assert device == "cpu"

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        @staticmethod
        def has_tensor(name):
            return name in {f"hf.expert.{expert}" for expert in range(4)}

        @staticmethod
        def first_available(names):
            return next(iter(names))

        @staticmethod
        def get_tensor(name, *, device):
            assert device == "cpu"
            reads.append(name)
            return torch.tensor([float(name.rsplit(".", 1)[1])])

    model = Model()
    router_before = model.depth_routers[0].weight.detach().clone()
    monkeypatch.setattr(checkpoint, "_spec", lambda _config: Spec())
    monkeypatch.setattr(checkpoint, "SafeTensorReader", Reader)
    monkeypatch.setattr(checkpoint, "_validate_streaming_load_scope", lambda *_args: None)
    monkeypatch.setattr(checkpoint, "_validate_hf_checkpoint_directory", lambda *_args: tmp_path)
    monkeypatch.setattr(checkpoint, "_preflight_source_tensor_headers", lambda *_args: None)
    checkpoint.load_hf_weights(
        model,
        str(tmp_path),
        SimpleNamespace(),
        SimpleNamespace(ep_size=2, ep_rank=1),
    )

    assert reads == ["hf.expert.2", "hf.expert.3"]
    assert model.layers[0].moe.experts.fc1.weight0.item() == 2.0
    assert model.layers[0].moe.experts.fc1.weight1.item() == 3.0
    assert torch.equal(model.depth_routers[0].weight, router_before)
