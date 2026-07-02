# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for TemporaryBucketAllocator. Pure CPU, no torch.distributed."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[2]))
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.allocator import (
    Bucket,
    TemporaryBucketAllocator,
    TracePoolAllocator,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.dp_buffer import DataParallelBuffer


def _run_allocator_tests(allocator: TemporaryBucketAllocator) -> None:
    """Three-phase test covering allocate, free, and re-allocate for any allocator.

    Phase 1 -- allocate:
      - Returned bucket has correct size, dtype, and is tracked internally.
      - Duplicate allocate on the same id returns the same object (no realloc).
      - Different ids produce independent buckets.

    Phase 2 -- free:
      - Freed id is removed from internal tracking.
      - Underlying tensor storage is resized to 0.
      - Freeing a non-existent id is silently ignored.
      - Freeing one id does not affect others.

    Phase 3 -- re-allocate after free:
      - Re-allocating a freed id returns a usable bucket with correct size.
    """

    # ---- Phase 1: allocate ----
    b0 = allocator.allocate(
        param_group_id=0, size=1024, dtype=torch.float32, device=torch.device("cpu")
    )
    assert b0.data.numel() == 1024
    assert b0.data.dtype == torch.float32
    assert 0 in allocator.buckets

    b0_again = allocator.allocate(
        param_group_id=0, size=1024, dtype=torch.float32, device=torch.device("cpu")
    )
    assert b0_again is b0
    assert b0_again.data.data_ptr() == b0.data.data_ptr()

    b1 = allocator.allocate(
        param_group_id=1, size=512, dtype=torch.bfloat16, device=torch.device("cpu")
    )
    assert b1 is not b0
    assert b1.data.numel() == 512
    assert b1.data.dtype == torch.bfloat16

    # ---- Phase 2: free ----
    tensor_ref = b0.data
    allocator.free(0)
    assert 0 not in allocator.buckets
    assert tensor_ref._typed_storage()._size() == 0

    assert 1 in allocator.buckets
    assert b1.data.numel() == 512

    allocator.free(999)

    # ---- Phase 3: re-allocate ----
    b0_new = allocator.allocate(
        param_group_id=0, size=1024, dtype=torch.float32, device=torch.device("cpu")
    )
    assert b0_new.data.numel() == 1024
    assert 0 in allocator.buckets

    # cleanup
    allocator.free(0)
    allocator.free(1)


class TestTemporaryBucketAllocator:

    def test_full_lifecycle(self):
        _run_allocator_tests(TemporaryBucketAllocator())


class TestTracePoolFullIterationGradBuffers:

    @staticmethod
    def _make_buffer(allocator, key, size):
        buffer = DataParallelBuffer.__new__(DataParallelBuffer)
        buffer.is_distributed = True
        buffer.allocator = allocator
        buffer.alloc_key = key
        buffer.dtype = torch.float32
        buffer.device = torch.device("cpu")
        buffer.buffer_index = SimpleNamespace(bucket_meta=SimpleNamespace(size=size))
        bucket = allocator.allocate(key=key, size=size, dtype=buffer.dtype, device=buffer.device)
        buffer._unsharded_buffer = bucket.data
        return buffer

    def test_non_overlapping_grad_buffers_share_stable_slot(self):
        allocator = TracePoolAllocator()
        first = self._make_buffer(allocator, (0, "main_grad"), 16)
        first_trace_tensor = first._unsharded_buffer
        first.release_unsharded_buffer_for_reuse()

        second = self._make_buffer(allocator, (1, "main_grad"), 16)
        second_trace_tensor = second._unsharded_buffer
        second.release_unsharded_buffer_for_reuse()

        assert first._unsharded_buffer is first_trace_tensor
        assert second._unsharded_buffer is second_trace_tensor
        assert first_trace_tensor._typed_storage()._size() == 0
        assert second_trace_tensor._typed_storage()._size() == 0

        allocator.plan()

        assert len(allocator._slots) == 1
        assert first.rebind_unsharded_buffer_to_allocator(zero=True)
        assert second.rebind_unsharded_buffer_to_allocator(zero=True)
        assert first._unsharded_buffer.data_ptr() == second._unsharded_buffer.data_ptr()
        assert first._unsharded_buffer.numel() == 16
        assert second._unsharded_buffer.numel() == 16

    def test_complete_trace_preserves_later_phase_conflicts(self):
        allocator = TracePoolAllocator()
        first = self._make_buffer(allocator, (0, "main_grad"), 16)
        first.release_unsharded_buffer_for_reuse()
        second = self._make_buffer(allocator, (1, "main_grad"), 16)
        second.release_unsharded_buffer_for_reuse()

        first_trace_tensor = first._unsharded_buffer
        assert first.fetch_buffer() is first_trace_tensor
        assert first_trace_tensor._typed_storage()._size() == 16
        assert second.fetch_buffer()._typed_storage()._size() == 16
        first.release_unsharded_buffer_for_reuse()
        second.release_unsharded_buffer_for_reuse()

        allocator.plan()

        assert len(allocator._slots) == 2
        assert first.rebind_unsharded_buffer_to_allocator(zero=True)
        assert second.rebind_unsharded_buffer_to_allocator(zero=True)
        assert first._unsharded_buffer.data_ptr() != second._unsharded_buffer.data_ptr()

    def test_slot_inherits_largest_key_trace_stream(self, monkeypatch):
        allocator = TracePoolAllocator()
        large = self._make_buffer(allocator, "large", 32)
        large.release_unsharded_buffer_for_reuse()
        small = self._make_buffer(allocator, "small", 16)
        small.release_unsharded_buffer_for_reuse()

        allocator._trace_streams = {"large": "large-stream", "small": "small-stream"}
        allocated_streams = []

        def fake_allocate(size, dtype, device, stream):
            allocated_streams.append(stream)
            return torch.empty(size, dtype=dtype, device=device)

        monkeypatch.setattr(allocator, "_allocate_slot_tensor", fake_allocate)
        allocator.plan()

        assert len(allocator._slots) == 1
        assert allocated_streams == ["large-stream"]
        assert allocator._slots[0].stream == "large-stream"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
