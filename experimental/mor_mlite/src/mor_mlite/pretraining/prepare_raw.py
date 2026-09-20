"""Pinned, resumable raw SlimPajama -> Qwen indexed-data token-budget pipeline.

Each source shard is an atomic unit. An interrupted shard remains in a unique
staging directory; completed shards are hash-verified and reused on restart.
Nothing is removed or overwritten. All tokenizer/model files are official HF
files at an immutable revision; model weights are never downloaded.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import tempfile
import time
from array import array
from pathlib import Path

import numpy as np

from mor_mlite.pretraining.data import IndexedCorpus, prepare, sha256_file

DATASET = "gmongaras/SlimPajama-627B_Reupload"
REVISION = "c34c22dbb10ae6b264a2f357a909d1a537141b36"
TOKENIZER = "Qwen/Qwen3-30B-A3B-Base"
TOKENIZER_REVISION = "1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9"


def transient_download_error(error):
    """Retry transport/server failures, never auth, missing revisions or validation."""
    from requests.exceptions import ConnectionError as RequestsConnectionError
    from requests.exceptions import Timeout

    status = getattr(getattr(error, "response", None), "status_code", None)
    if status is not None:
        return status in (408, 429) or 500 <= status < 600
    return isinstance(error, (Timeout, RequestsConnectionError, TimeoutError, ConnectionError))


def download_with_retry(operation, *, attempts=8, sleep=time.sleep, **kwargs):
    if attempts < 1:
        raise ValueError("download attempts must be positive")
    for attempt in range(1, attempts + 1):
        try:
            return operation(**kwargs)
        except Exception as error:
            if attempt == attempts or not transient_download_error(error):
                raise
            delay = min(300, 5 * 2 ** (attempt - 1))
            # Exception messages can include signed URLs; log only the class.
            print(
                json.dumps(
                    {
                        "download_retry": attempt,
                        "error": type(error).__name__,
                        "delay_seconds": delay,
                    }
                ),
                flush=True,
            )
            sleep(delay)


def completed_shard(target, token_budget, contract):
    state = json.loads((target / "complete.json").read_text())
    if state["contract"] != contract or state["requested_tokens"] != token_budget:
        raise ValueError("completed shard has a different preparation contract")
    if set(state["files"]) != {"text.bin", "text.idx"}:
        raise ValueError("completed shard lacks indexed file fingerprints")
    for filename, digest in state["files"].items():
        if sha256_file(target / filename) != digest:
            raise ValueError("completed indexed shard was changed")
    return state


def write_index(path, lengths):
    sizes = np.asarray(lengths, dtype="<i4")
    if not len(sizes) or np.any(sizes <= 0):
        raise ValueError("indexed output requires nonempty documents")
    offsets = np.empty(len(sizes), dtype="<i8")
    offsets[0] = 0
    np.cumsum(sizes[:-1], dtype=np.int64, out=offsets[1:])
    offsets *= 4
    with Path(path).open("xb") as stream:
        stream.write(b"MMIDIDX\0\0" + struct.pack("<QBQQ", 1, 4, len(sizes), len(sizes) + 1))
        stream.write(sizes.tobytes())
        stream.write(offsets.tobytes())
        stream.write(np.arange(len(sizes) + 1, dtype="<i8").tobytes())


def text_batches(path, batch_size=256):
    import pyarrow.parquet as pq

    with pq.ParquetFile(path) as parquet:
        for batch in parquet.iter_batches(batch_size=batch_size, columns=["text"]):
            texts = batch.column(0).to_pylist()
            if not all(isinstance(text, str) for text in texts):
                raise ValueError("raw document has no string text")
            yield texts


def encode_shard(raw, target, tokenizer, token_budget, contract):
    """Stop after the first whole document reaching the remaining budget."""
    target = Path(target)
    if target.exists():
        return completed_shard(target, token_budget, contract)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=target.name + ".partial-", dir=target.parent))
    lengths = array("q")
    tokens = 0
    with (staging / "text.bin").open("xb") as binary:
        for texts in text_batches(raw):
            batch = tokenizer(texts, add_special_tokens=False, return_attention_mask=False)[
                "input_ids"
            ]
            for ids in batch:
                ids.append(tokenizer.eos_token_id)
                if len(ids) >= 2**31:
                    raise ValueError("document exceeds indexed-data int32 length")
                binary.write(np.asarray(ids, dtype="<i4").tobytes())
                lengths.append(len(ids))
                tokens += len(ids)
                if tokens >= token_budget:
                    break
            if len(lengths) % 65536 < len(texts):
                print(
                    json.dumps({"shard": target.name, "documents": len(lengths), "tokens": tokens}),
                    flush=True,
                )
            if tokens >= token_budget:
                break
    write_index(staging / "text.idx", lengths)
    state = {
        "contract": contract,
        "requested_tokens": token_budget,
        "tokens": tokens,
        "documents": len(lengths),
        "raw_sha256": sha256_file(raw),
        "files": {name: sha256_file(staging / name) for name in ("text.bin", "text.idx")},
    }
    (staging / "complete.json").write_text(json.dumps(state, indent=2) + "\n")
    staging.rename(target)
    return state


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=50_000_000_000)
    parser.add_argument("--threads", type=int, default=32)
    args = parser.parse_args()
    if args.tokens <= 0 or args.threads <= 0:
        parser.error("tokens and threads must be positive")
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN must be configured before downloading")
    # Set before importing the Rust tokenizer thread pool.
    os.environ["RAYON_NUM_THREADS"] = str(args.threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    # Configure before importing huggingface_hub, which reads these at import.
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from transformers import AutoTokenizer

    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    api = HfApi(token=os.environ["HF_TOKEN"])
    download_with_retry(api.whoami)
    info = download_with_retry(api.dataset_info, repo_id=DATASET, revision=REVISION)
    if info.sha != REVISION:
        raise ValueError("dataset revision did not resolve exactly")
    files = sorted(
        x.rfilename
        for x in info.siblings
        if x.rfilename.startswith("data/train-") and x.rfilename.endswith(".parquet")
    )
    if not files:
        raise ValueError("pinned dataset has no expected training shards")
    contract = {
        "dataset": DATASET,
        "revision": REVISION,
        "files": files,
        "tokenizer": TOKENIZER,
        "tokenizer_revision": TOKENIZER_REVISION,
        "target_tokens": args.tokens,
        "eos": 151643,
        "chat_template": False,
        "sampling": "sorted-train-shards-whole-document-prefix-v1",
        "source_limit": "third-party Parquet reupload of SlimPajama; original is unavailable and original byte identity is not independently verified",
    }
    contract_path = root / "preparation-contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError("output directory is bound to a different preparation")
    else:
        contract_path.write_text(json.dumps(contract, indent=2) + "\n")
    tokenizer_dir = root / "tokenizer"
    download_with_retry(
        snapshot_download,
        repo_id=TOKENIZER,
        revision=TOKENIZER_REVISION,
        local_dir=tokenizer_dir,
        allow_patterns=[
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json",
        ],
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir, local_files_only=True, trust_remote_code=False
    )
    if tokenizer.eos_token_id != 151643:
        raise ValueError("official Base tokenizer EOS mismatch")
    consumed = 0
    shards = []
    for index, name in enumerate(files):
        target = root / "indexed-shards" / f"{index:04d}"
        if target.exists():
            state = completed_shard(target, args.tokens - consumed, contract)
            print(json.dumps({"reused_shard": target.name, "tokens": state["tokens"]}), flush=True)
        else:
            raw = download_with_retry(
                hf_hub_download,
                repo_id=DATASET,
                filename=name,
                repo_type="dataset",
                revision=REVISION,
                cache_dir=root / "hf-cache",
            )
            state = encode_shard(raw, target, tokenizer, args.tokens - consumed, contract)
        consumed += state["tokens"]
        shards.append(target)
        if consumed >= args.tokens:
            break
    if consumed < args.tokens:
        raise ValueError(f"source exhausted at {consumed} tokens; requested {args.tokens}")
    merged = root / "indexed"
    if not merged.exists():
        staging = Path(tempfile.mkdtemp(prefix="indexed.partial-", dir=root))
        lengths = array("q")
        with (staging / "text.bin").open("xb") as out:
            for shard in shards:
                corpus = IndexedCorpus(shard / "text")
                lengths.extend(int(n) for n in corpus.lengths)
                with (shard / "text.bin").open("rb") as source:
                    shutil.copyfileobj(source, out, length=8 * 1024 * 1024)
        write_index(staging / "text.idx", lengths)
        (staging / "complete.json").write_text(
            json.dumps({"contract": contract, "tokens": consumed}) + "\n"
        )
        staging.rename(merged)
    if json.loads((merged / "complete.json").read_text())["contract"] != contract:
        raise ValueError("merged data has a different preparation contract")
    prepared = root / "prepared"
    if prepared.exists():
        raise FileExistsError("preparation already exists; inspect manifest instead of overwriting")
    prepare(
        merged / "text",
        prepared,
        tokenizer_dir=tokenizer_dir,
        seed=1234,
        provenance={
            "dataset_source": json.dumps(contract),
            "tokenizer_source": f"{TOKENIZER}@{TOKENIZER_REVISION}",
            "verification_evidence": "Direct raw-text encoding with official pinned tokenizer; shard complete.json binds raw/indexed hashes; no chat template",
        },
    )


if __name__ == "__main__":
    main()
