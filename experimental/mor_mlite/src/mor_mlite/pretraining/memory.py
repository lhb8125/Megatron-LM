"""Rank-local whole-device sampling plus allocator high-water marks."""

from __future__ import annotations

import threading


class MemoryMonitor:
    def __init__(self, interval=0.02):
        import torch

        self.torch = torch
        self.device = torch.cuda.current_device()
        self.interval = interval
        self.stop_event = threading.Event()
        self.peak_device = 0
        self.samples = 0
        self.failure = None
        self.thread = threading.Thread(target=self._poll, daemon=True)

    def sample(self):
        free, total = self.torch.cuda.mem_get_info(self.device)
        self.total = total
        self.peak_device = max(self.peak_device, total - free)
        self.samples += 1

    def _poll(self):
        try:
            self.torch.cuda.set_device(self.device)
            while not self.stop_event.wait(self.interval):
                self.sample()
        except (RuntimeError, OSError) as exc:
            self.failure = repr(exc)

    def start(self):
        self.torch.cuda.reset_peak_memory_stats(self.device)
        self.sample()
        self.thread.start()

    def finish(self):
        self.stop_event.set()
        self.thread.join(timeout=10)
        if self.thread.is_alive() or self.failure:
            raise RuntimeError(f"whole-device memory sampler failed: {self.failure}")
        self.sample()
        return {
            "whole_device_peak_bytes": self.peak_device,
            "capacity_bytes": self.total,
            "peak_device_fraction": self.peak_device / self.total,
            "allocated_peak_bytes": self.torch.cuda.max_memory_allocated(self.device),
            "reserved_peak_bytes": self.torch.cuda.max_memory_reserved(self.device),
            "samples": self.samples,
            "interval_seconds": self.interval,
            "whole_device_metric": "sampled CUDA mem_get_info; short-lived peaks may be missed",
        }
