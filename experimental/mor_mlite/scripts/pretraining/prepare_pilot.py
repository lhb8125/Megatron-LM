"""Prepare a sealed full-corpus shard for correctness/performance runs only."""

import argparse
import json
from pathlib import Path

from mor_mlite.pretraining.data import prepare, sha256_file
from mor_mlite.pretraining.prepare_raw import DATASET, REVISION, TOKENIZER, TOKENIZER_REVISION


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    state = json.loads((args.shard / "complete.json").read_text())
    contract = state["contract"]
    for key, value in {
        "dataset": DATASET,
        "revision": REVISION,
        "tokenizer": TOKENIZER,
        "tokenizer_revision": TOKENIZER_REVISION,
    }.items():
        if contract.get(key) != value:
            raise ValueError(f"sealed shard has unexpected provenance: {key}")
    for filename, expected in state["files"].items():
        if sha256_file(args.shard / filename) != expected:
            raise ValueError(f"sealed shard changed: {filename}")
    result = prepare(
        args.shard / "text",
        args.output,
        tokenizer_dir=args.tokenizer,
        seed=1234,
        provenance={
            "purpose": "performance-pilot-only",
            "dataset_source": json.dumps(contract),
            "tokenizer_source": f"{TOKENIZER}@{TOKENIZER_REVISION}",
            "verification_evidence": "Completed raw-encoding shard; hashes reverified; official pinned tokenizer. This subset is not the formal one-pass corpus.",
            "shard_manifest_sha256": sha256_file(args.shard / "complete.json"),
        },
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
