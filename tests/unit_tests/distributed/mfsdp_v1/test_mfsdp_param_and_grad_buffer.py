# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import math
from types import SimpleNamespace

import pytest
import torch

from megatron.core.distributed.fsdp.src.megatron_fsdp import param_and_grad_buffer as pgb_module
from megatron.core.distributed.fsdp.src.megatron_fsdp.distributed_data_parallel_config import (
    DistributedDataParallelConfig,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import (
    Bucket,
    BucketingPolicy,
    DataParallelBuffer,
    MaxPoolAllocator,
    ParamAndGradBuffer,
    ParameterGroup,
    _build_ubr_arena_layout,
    _get_parameter_groups,
    _get_ubr_registration_groups,
    _max_pool_assignment_signature,
    _mem_pool_registration_signature,
    _tensor_mem_pool_location,
)


class _ExpertTestModule(torch.nn.Module):
    """
    Mock module whose params are routed under `.experts.` to trigger
    is_expert_param=True. The outer `layer` attribute puts a dot before
    `experts` in the parameter path (e.g. `layer.experts.linear_fc1`).
    """

    def __init__(self, shapes):
        super().__init__()
        self.layer = torch.nn.Module()
        self.layer.experts = torch.nn.ParameterDict(
            {name: torch.nn.Parameter(torch.empty(shape)) for name, shape in shapes.items()}
        )


class _MultiUseEmbeddingTestModule(torch.nn.Module):
    """Mock module with an MTP-style multi-use embedding and a regular root parameter."""

    def __init__(self):
        super().__init__()
        self.embedding_weight = torch.nn.Parameter(torch.empty(32, 16))
        self.embedding_weight.zero_out_wgrad = True
        self.final_norm_weight = torch.nn.Parameter(torch.empty(16))


def _get_bucket_signatures(module):
    bucket_groups, _, _ = _get_parameter_groups(
        module, BucketingPolicy(suggested_bucket_size=None), meta_device_init_fp8_params={}
    )
    param_to_name = {param: name for name, param in module.named_parameters()}
    return [
        {
            "chunk_size_factor": group.chunk_size_factor,
            "params": [(param_to_name[param], tuple(param.shape)) for param in group.params],
        }
        for group in bucket_groups
    ]


class _TestMemoryPool:
    def __init__(self, segments):
        self.segments = segments

    def snapshot(self):
        return self.segments


class _TestTensorLocation:
    def __init__(self, pointer, numel, element_size):
        self.pointer = pointer
        self.tensor_numel = numel
        self.tensor_element_size = element_size

    def data_ptr(self):
        return self.pointer

    def numel(self):
        return self.tensor_numel

    def element_size(self):
        return self.tensor_element_size


class _TestDistIndex:
    def __init__(self, use_hybrid_fsdp):
        self.use_hybrid_fsdp = use_hybrid_fsdp
        self.dense = object()
        self.expert = object()
        self.dense_ag = object()
        self.expert_ag = object()
        self.outer = object()

    def get_fsdp_group(self, is_expert_parallel=False, independent_all_gather=False):
        if is_expert_parallel:
            return self.expert_ag if independent_all_gather else self.expert
        return self.dense_ag if independent_all_gather else self.dense

    def get_outer_fsdp_group(self):
        return self.outer


def test_dense_inner_ubr_scope_selects_hsdp_dense_helper_group_only():
    """HSDP inner parameter AG uses the base dense FSDP group, not outer/expert groups."""
    dist_index = _TestDistIndex(use_hybrid_fsdp=True)

    assert _get_ubr_registration_groups(dist_index, "dense_inner") == [dist_index.dense]


def test_dense_inner_ubr_scope_prefers_independent_ag_without_hsdp():
    """Non-HSDP parameter AG uses its independent communicator when one is provided."""
    dist_index = _TestDistIndex(use_hybrid_fsdp=False)

    assert _get_ubr_registration_groups(dist_index, "dense_inner") == [dist_index.dense_ag]


def test_all_ubr_scope_preserves_every_registration_group():
    """The default scope remains backward compatible with the previous group list."""
    dist_index = _TestDistIndex(use_hybrid_fsdp=True)

    assert _get_ubr_registration_groups(dist_index, "all") == [
        dist_index.dense,
        dist_index.expert,
        dist_index.dense_ag,
        dist_index.expert_ag,
        dist_index.outer,
    ]


def test_ubr_scope_rejects_unknown_value():
    dist_index = _TestDistIndex(use_hybrid_fsdp=True)

    with pytest.raises(ValueError, match="Invalid FSDP UBR registration scope"):
        _get_ubr_registration_groups(dist_index, "outer")


def test_symmetric_rs_is_explicit_and_requires_dense_manual_registration():
    """The native RS path remains opt-in and invalid registration setups fail early."""
    assert not DistributedDataParallelConfig().fsdp_ubr_enable_symmetric_rs

    with pytest.raises(ValueError, match="requires nccl_ub=True"):
        DistributedDataParallelConfig(fsdp_ubr_enable_symmetric_rs=True)

    config = DistributedDataParallelConfig(
        nccl_ub=True,
        fsdp_ubr_registration_scope="dense_inner",
        fsdp_ubr_enable_symmetric_rs=True,
        fsdp_manual_registration=True,
        megatron_fsdp_max_pool_double_buffer=True,
    )
    assert config.fsdp_ubr_enable_symmetric_rs
    assert config.fsdp_double_buffer


def test_manual_registration_context_can_target_independent_rs_pool(monkeypatch):
    """Dense RS allocations must not fall back into the parameter registration pool."""
    buffer = object.__new__(ParamAndGradBuffer)
    buffer.ddp_config = SimpleNamespace(nccl_ub=True, fsdp_manual_registration=True)
    param_pool = object()
    rs_pool = object()
    monkeypatch.setattr(pgb_module, "NCCL_ALLOCATOR", "MCORE")
    monkeypatch.setattr(pgb_module, "NCCL_MEMORY_POOL", param_pool)

    param_context = buffer.get_mem_alloc_context(mem_pool=param_pool)
    rs_context = buffer.get_mem_alloc_context(mem_pool=rs_pool)

    assert param_context.args == (param_pool,)
    assert rs_context.args == (rs_pool,)


def test_reduce_scatter_output_is_native_rank_offset_shard_view():
    """Symmetric RS does not need a separate same-offset staging allocation."""
    buffer = DataParallelBuffer.__new__(DataParallelBuffer)
    buffer.shard_bucket_index = SimpleNamespace(bucket_data_index=8, size=4)
    bucket = Bucket(torch.arange(16, dtype=torch.float32))

    output = buffer.get_shard_from_bucket(bucket)

    assert torch.equal(output, bucket.data[8:12])
    assert output.untyped_storage().data_ptr() == bucket.data.untyped_storage().data_ptr()
    assert output.storage_offset() == 8


def test_mem_pool_registration_signature_uses_registration_order():
    """Signature order must match ProcessGroupNCCL's registration order."""
    pool = _TestMemoryPool(
        [
            {"total_size": 4096, "registration_counter": 7, "address": 0x1000},
            {"total_size": 1024, "registration_counter": 3, "address": 0x2000},
            {"total_size": 2048, "registration_counter": 5, "address": 0x3000},
        ]
    )

    assert _mem_pool_registration_signature(pool) == (1024, 2048, 4096)


def test_mem_pool_registration_signature_ignores_local_addresses():
    """Different rank-local addresses do not change the collective layout."""
    first = _TestMemoryPool([{"total_size": 1024, "address": 0x1000}])
    second = _TestMemoryPool([{"total_size": 1024, "address": 0x9000}])

    assert _mem_pool_registration_signature(first) == _mem_pool_registration_signature(second)


def test_ubr_arena_layout_aligns_requests_in_logical_order():
    requests = [
        (17, "max_pool", object(), 17, torch.uint8, "first"),
        (512, "persistent", object(), 128, torch.float32, "second"),
        (33, "max_pool", object(), 33, torch.uint8, "third"),
    ]

    layout, arena_size = _build_ubr_arena_layout(requests, alignment=256)

    assert [offset for offset, _ in layout] == [0, 256, 768]
    assert [request[-1] for _, request in layout] == ["first", "second", "third"]
    assert arena_size == 1024


def test_tensor_mem_pool_location_uses_registration_order_and_offset():
    """Tensor locations use NCCL segment order and reject cross-boundary views."""
    pool = _TestMemoryPool(
        [
            {"total_size": 4096, "registration_counter": 7, "address": 0x10000},
            {"total_size": 1024, "registration_counter": 3, "address": 0x20000},
        ]
    )

    tensor = _TestTensorLocation(pointer=0x10100, numel=32, element_size=4)
    crossing = _TestTensorLocation(pointer=0x10FF0, numel=8, element_size=4)

    assert _tensor_mem_pool_location(pool, tensor) == (1, 0x100, 128)
    assert _tensor_mem_pool_location(pool, crossing) is None


def test_max_pool_materialize_uses_exact_largest_padded_bucket(monkeypatch):
    """Eager materialization uses exact runtime sizes in deterministic slot order."""
    parameter_groups = [
        ParameterGroup(
            [torch.nn.Parameter(torch.empty(4, dtype=torch.bfloat16))],
            dtype=torch.bfloat16,
            fsdp_unit_id=0,
        ),
        ParameterGroup(
            [torch.nn.Parameter(torch.empty(8, dtype=torch.bfloat16))],
            dtype=torch.bfloat16,
            fsdp_unit_id=1,
        ),
    ]
    allocator = MaxPoolAllocator("test_pool", parameter_groups, size=2)

    allocations = []

    class _TestGlobalMemoryBuffer:
        def get_tensor(self, tensor_shape, dtype, name, mem_alloc_context=None):
            allocations.append((tuple(tensor_shape), dtype, name, mem_alloc_context))
            return torch.empty(tensor_shape, dtype=dtype)

    monkeypatch.setattr(pgb_module, "get_global_memory_buffer", _TestGlobalMemoryBuffer)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    allocator.materialize({0: (64, torch.uint8), 1: (8, torch.float32)})

    assert allocations == [
        ((8,), torch.float32, "test_pool_0_torch.float32_0", None),
        ((8,), torch.float32, "test_pool_1_torch.float32_0", None),
        ((64,), torch.uint8, "test_pool_0_torch.uint8_0", None),
        ((64,), torch.uint8, "test_pool_1_torch.uint8_0", None),
    ]
    assert allocator.allocation_tracker == {
        ("test_pool_0_torch.float32_0", torch.float32): 8,
        ("test_pool_0_torch.uint8_0", torch.uint8): 64,
        ("test_pool_1_torch.float32_0", torch.float32): 8,
        ("test_pool_1_torch.uint8_0", torch.uint8): 64,
    }


def test_max_pool_assignment_signature_tracks_logical_bucket_slots():
    """Runtime signatures distinguish the pool slot used by each logical bucket."""
    first_param = torch.nn.Parameter(torch.empty(4, dtype=torch.bfloat16))
    second_param = torch.nn.Parameter(torch.empty(4, dtype=torch.bfloat16))
    parameter_groups = [
        ParameterGroup([first_param], dtype=torch.bfloat16, fsdp_unit_id=0),
        ParameterGroup([second_param], dtype=torch.bfloat16, fsdp_unit_id=1),
    ]
    param_to_name = {first_param: "layers.0.weight", second_param: "layers.1.weight"}
    allocator = MaxPoolAllocator("test_pool", parameter_groups, size=2)

    allocator.bucket_alloc_index = {0: (0, 0), 1: (1, 0)}
    first_signature = _max_pool_assignment_signature((allocator,), parameter_groups, param_to_name)

    allocator.bucket_alloc_index = {0: (1, 0), 1: (0, 0)}
    second_signature = _max_pool_assignment_signature((allocator,), parameter_groups, param_to_name)

    assert first_signature != second_signature
    assert {assignment[5] for assignment in first_signature} == {0, 1}


def test_max_pool_bucket_filter_keeps_dense_and_expert_slots_disjoint():
    """Dense-inner UBR cannot share registered MaxPool slots with expert traffic."""
    dense_param = torch.nn.Parameter(torch.empty(4, dtype=torch.bfloat16))
    expert_param = torch.nn.Parameter(torch.empty(8, dtype=torch.bfloat16))
    parameter_groups = [
        ParameterGroup([dense_param], dtype=torch.bfloat16, fsdp_unit_id=0),
        ParameterGroup([expert_param], dtype=torch.bfloat16, is_expert_param=True, fsdp_unit_id=0),
    ]

    dense_allocator = MaxPoolAllocator(
        "dense_pool",
        parameter_groups,
        size=2,
        bucket_filter=lambda _, group: not group.is_expert_param,
    )
    expert_allocator = MaxPoolAllocator(
        "expert_pool",
        parameter_groups,
        size=2,
        bucket_filter=lambda _, group: group.is_expert_param,
    )

    assert set(dense_allocator.bucket_alloc_index) == {0}
    assert set(expert_allocator.bucket_alloc_index) == {1}
    assert dense_allocator.materialization_requests({0: (16, torch.bfloat16)}) == [
        (16, torch.bfloat16, "dense_pool_0_torch.bfloat16_0"),
        (16, torch.bfloat16, "dense_pool_1_torch.bfloat16_0"),
    ]
    assert expert_allocator.materialization_requests({1: (32, torch.bfloat16)}) == [
        (32, torch.bfloat16, "expert_pool_0_torch.bfloat16_0"),
        (32, torch.bfloat16, "expert_pool_1_torch.bfloat16_0"),
    ]


def test_multi_use_embedding_gets_separate_bucket_when_sharded():
    """Root-handled multi-use embeddings must not share a bucket with regular-hook params."""
    module = _MultiUseEmbeddingTestModule()
    bucket_groups, param_to_group, _ = _get_parameter_groups(
        module,
        BucketingPolicy(
            suggested_bucket_size=None, data_parallel_sharding_strategy="optim_grads_params"
        ),
        meta_device_init_fp8_params={},
    )

    embedding_group = param_to_group[module.embedding_weight]
    final_norm_group = param_to_group[module.final_norm_weight]
    assert embedding_group != final_norm_group
    assert bucket_groups[embedding_group].params == [module.embedding_weight]


def test_grouped_expert_weights_split_when_chunk_size_factors_differ():
    """Grouped expert weights with mismatched chunk size factors get routed to separate buckets."""
    num_local_experts = 4
    hidden_size = 12
    moe_ffn_hidden_size = 8
    shapes = {
        "linear_fc1": (num_local_experts, 2 * moe_ffn_hidden_size, hidden_size),
        "linear_fc2": (num_local_experts, hidden_size, moe_ffn_hidden_size),
    }
    module = _ExpertTestModule(shapes)

    assert _get_bucket_signatures(module) == [
        {
            "chunk_size_factor": torch.Size(shapes["linear_fc1"])[1:].numel(),
            "params": [("layer.experts.linear_fc1", shapes["linear_fc1"])],
        },
        {
            "chunk_size_factor": torch.Size(shapes["linear_fc2"])[1:].numel(),
            "params": [("layer.experts.linear_fc2", shapes["linear_fc2"])],
        },
    ]


def test_per_expert_2d_weights_merge_via_lcm():
    """Per-expert 2D weights merge into a single bucket via LCM chunk size factor."""
    hidden_size = 12
    moe_ffn_hidden_size = 8
    shapes = {
        "linear_fc1": (2 * moe_ffn_hidden_size, hidden_size),
        "linear_fc2": (hidden_size, moe_ffn_hidden_size),
    }
    module = _ExpertTestModule(shapes)

    assert _get_bucket_signatures(module) == [
        {
            "chunk_size_factor": math.lcm(
                torch.Size(shapes["linear_fc1"])[1:].numel(),
                torch.Size(shapes["linear_fc2"])[1:].numel(),
            ),
            "params": [
                ("layer.experts.linear_fc1", shapes["linear_fc1"]),
                ("layer.experts.linear_fc2", shapes["linear_fc2"]),
            ],
        }
    ]
