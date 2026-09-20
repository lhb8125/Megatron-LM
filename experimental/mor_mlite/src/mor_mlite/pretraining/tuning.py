"""Fail-closed MBS selection and actual EP locality validation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise


@dataclass(frozen=True)
class Trial:
    mbs: int
    tokens_per_second: float
    peak_device_fraction: float
    warmup_steps: int
    measured_steps: int
    updates_ok: bool
    evaluation_ok: bool
    checkpoint_ok: bool

    @property
    def eligible(self):
        return (
            self.mbs in (1, 2, 4, 8)
            and self.warmup_steps == 20
            and self.measured_steps == 30
            and self.updates_ok
            and self.evaluation_ok
            and self.checkpoint_ok
            and math.isfinite(self.tokens_per_second)
            and self.tokens_per_second > 0
            and math.isfinite(self.peak_device_fraction)
            and 0 < self.peak_device_fraction <= 0.90
        )


def next_mbs(trial: Trial) -> int | None:
    if not trial.eligible or trial.peak_device_fraction > 0.80 or trial.mbs == 8:
        return None
    return trial.mbs * 2


def select_mbs(trials: list[Trial]) -> int:
    if not trials or trials[0].mbs != 1:
        raise ValueError("MBS sweep must begin at 1")
    for left, right in pairwise(trials):
        if next_mbs(left) != right.mbs:
            raise ValueError("invalid MBS sweep: skipped size or memory stop condition")
    valid = [t for t in trials if t.eligible]
    if not valid:
        raise ValueError("no MBS passed training/evaluation/checkpoint/memory checks")
    fastest = max(t.tokens_per_second for t in valid)
    return min(
        t.mbs
        for t in valid
        if t.tokens_per_second == fastest or (fastest - t.tokens_per_second) / fastest < 0.03
    )


def validate_ep_locality(rank_records: list[dict], *, world_size: int, ep: int = 16):
    if sorted(r["rank"] for r in rank_records) != list(range(world_size)):
        raise ValueError("missing or duplicate rank topology evidence")
    by_rank = {r["rank"]: r for r in rank_records}
    groups = set()
    for record in rank_records:
        members = tuple(sorted(record["ep_members"]))
        if len(members) != ep or len(set(members)) != ep or record["rank"] not in members:
            raise ValueError("invalid actual EP group membership")
        peers = [by_rank[r] for r in members]
        if any(tuple(sorted(p["ep_members"])) != members for p in peers):
            raise ValueError("inconsistent EP membership reports")
        domains = {p["nvl_domain"] for p in peers}
        if len(domains) != 1 or not next(iter(domains)):
            raise ValueError("EP group crosses NVL domains or lacks hardware evidence")
        hosts = {p["hostname"] for p in peers}
        if len(hosts) != 4 or any(sum(p["hostname"] == h for p in peers) != 4 for h in hosts):
            raise ValueError("EP16 must occupy four complete four-GPU nodes")
        groups.add(members)
    if len(groups) != world_size // ep:
        raise ValueError("EP groups do not partition the allocated world")
    return {"world_size": world_size, "ep": ep, "groups": sorted(groups), "passed": True}
