"""Ordered milestone selection and paired held-out reporting (read-only)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def next_stage(completed, *, total_steps, tokens_per_step=8_388_608):
    """All four 1B runs precede any 10B run; all 10B precede final continuation."""
    if set(completed) - set("ABCD") or total_steps < 1:
        raise ValueError("invalid campaign state")
    if any(type(step) is not int or not 0 <= step <= total_steps for step in completed.values()):
        raise ValueError("invalid completed-step cursor")
    stages = sorted(
        {min(total_steps, math.ceil(t / tokens_per_step)) for t in (10**9, 10**10)} | {total_steps}
    )
    for stage in stages:
        for arm in "ABCD":
            if completed.get(arm, 0) < stage:
                return {
                    "arm": arm,
                    "from_step": completed.get(arm, 0),
                    "to_step": stage,
                    "stop_tokens": stage * tokens_per_step,
                }
    return None


def paired_results(evaluations):
    if set(evaluations) != set("ABCD"):
        raise ValueError("paired comparison needs all four arms")
    indexed = {}
    for arm, rows in evaluations.items():
        mapping = {row["tokens"]: row for row in rows}
        if len(mapping) != len(rows):
            raise ValueError("duplicate evaluation milestone")
        indexed[arm] = mapping
    common = set.intersection(*(set(rows) for rows in indexed.values()))
    result = []
    for tokens in sorted(common):
        rows = {arm: indexed[arm][tokens] for arm in "ABCD"}
        if len({row["targets"] for row in rows.values()}) != 1:
            raise ValueError("paired validation has different target counts")
        if not all(math.isfinite(row["nll"]) for row in rows.values()):
            raise ValueError("non-finite paired validation NLL")
        result.append(
            {
                "tokens": tokens,
                "nll": {a: r["nll"] for a, r in rows.items()},
                "ppl": {a: math.exp(r["nll"]) for a, r in rows.items()},
                "delta_nll": {
                    f"{a}-{b}": rows[a]["nll"] - rows[b]["nll"]
                    for a, b in (("B", "A"), ("B", "C"), ("D", "B"))
                },
            }
        )
    return result


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--run",
        type=Path,
        action="append",
        required=True,
        help="Completed formal run directory; include all continuations",
    )
    args = parser.parse_args()
    completed, evaluations, contracts = {}, {a: [] for a in "ABCD"}, []
    inputs = {a: {} for a in "ABCD"}
    for run in args.run:
        manifest = json.loads((run / "manifest.json").read_text())
        completion = [
            json.loads(line) for line in (run / "complete.jsonl").read_text().splitlines()
        ]
        if len(completion) != 1 or manifest["mode"] != "train":
            raise ValueError("campaign accepts completed formal runs only")
        arm = manifest["experiment"]["arm"]
        step = completion[0]["step"]
        if not (run / f"checkpoint-{step:06d}" / "pretraining-state.json").is_file():
            raise ValueError("completion is missing its resumable final checkpoint")
        completed[arm] = max(completed.get(arm, 0), step)
        for line in (run / "loss.jsonl").read_text().splitlines():
            row = json.loads(line)
            if row["step"] in inputs[arm]:
                raise ValueError("campaign duplicates a training update")
            inputs[arm][row["step"]] = row["input_sha256"]
        evaluations[arm].extend(
            json.loads(line) for line in (run / "evaluation.jsonl").read_text().splitlines()
        )
        contracts.append(
            (
                manifest["data_sha256"],
                manifest["source_sha256"],
                manifest["environment_sha256"],
                manifest["model_config_sha256"],
                json.dumps(
                    {
                        k: v
                        for k, v in manifest["experiment"].items()
                        if k not in ("arm", "micro_batch_size")
                    },
                    sort_keys=True,
                ),
                manifest["total_steps"],
            )
        )
    if len(set(contracts)) != 1:
        raise ValueError("campaign mixes data, source, environment, or token budgets")
    for arm, step in completed.items():
        if sorted(inputs[arm]) != list(range(1, step + 1)):
            raise ValueError("campaign is missing consumed training updates")
    for step in set.union(*(set(rows) for rows in inputs.values())):
        hashes = {rows[step] for rows in inputs.values() if step in rows}
        if len(hashes) != 1:
            raise ValueError(f"arms consumed different inputs at step {step}")
    print(
        json.dumps(
            {
                "completed": completed,
                "next": next_stage(completed, total_steps=contracts[0][-1]),
                "paired": paired_results(evaluations),
                "limitations": "single seed, one-pass LM loss; no downstream or equal-FLOP claim",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
