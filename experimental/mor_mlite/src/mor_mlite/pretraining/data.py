"""Read-only indexed corpus, content-addressed split, one-pass token stream."""

from __future__ import annotations

import hashlib
import json
import struct
import tempfile
from pathlib import Path

import numpy as np


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def global_input_digest(records, *, first_sample, batch_size):
    """Hash delivered samples in global order, independent of rank/MBS layout."""
    ordered = sorted(records)
    if [i for i, _ in ordered] != list(range(first_sample, first_sample + batch_size)):
        raise ValueError("global batch has missing, duplicate, or unexpected samples")
    digest = hashlib.sha256()
    for sample, value in ordered:
        digest.update(struct.pack("<Q", sample) + bytes.fromhex(value))
    return digest.hexdigest()


class IndexedCorpus:
    """MMIDIDX v1 int32 text-only reader, without importing CUDA/Megatron."""

    def __init__(self, prefix):
        self.prefix = str(prefix)
        index = Path(self.prefix + ".idx")
        with index.open("rb") as f:
            if f.read(9) != b"MMIDIDX\x00\x00":
                raise ValueError("not a Megatron indexed dataset")
            version, dtype, self.sequences, self.document_indices = struct.unpack(
                "<QBQQ", f.read(25)
            )
        if version != 1 or dtype != 4:
            raise ValueError("requires text-only version1/int32 data")
        expected = 34 + self.sequences * 12 + self.document_indices * 8
        if index.stat().st_size != expected:
            raise ValueError("truncated, multimodal, or unsupported index")
        self.lengths = np.memmap(index, dtype="<i4", mode="r", offset=34, shape=(self.sequences,))
        self.pointers = np.memmap(
            index, dtype="<i8", mode="r", offset=34 + 4 * self.sequences, shape=(self.sequences,)
        )
        self.documents = np.memmap(
            index,
            dtype="<i8",
            mode="r",
            offset=34 + 12 * self.sequences,
            shape=(self.document_indices,),
        )
        size = Path(self.prefix + ".bin").stat().st_size
        if size % 4 or self.document_indices < 2 or not self.sequences:
            raise ValueError("invalid binary size or empty corpus")
        self.tokens = np.memmap(self.prefix + ".bin", dtype="<i4", mode="r")
        if self.documents[0] != 0 or self.documents[-1] != self.sequences:
            raise ValueError("document boundaries do not cover the sequences")
        if np.any(np.diff(self.documents) <= 0) or np.any(self.lengths <= 0):
            raise ValueError("empty or unordered documents/sequences")
        expected_pointers = np.empty(self.sequences, dtype=np.int64)
        expected_pointers[0] = 0
        np.cumsum(self.lengths[:-1], dtype=np.int64, out=expected_pointers[1:])
        expected_pointers *= 4
        if not np.array_equal(self.pointers, expected_pointers):
            raise ValueError("non-contiguous or invalid sequence pointers")
        if int(self.pointers[-1]) + int(self.lengths[-1]) * 4 != size:
            raise ValueError("index does not exactly cover the binary")

    def __len__(self):
        return self.document_indices - 1

    def document(self, document_id):
        if not 0 <= document_id < len(self):
            raise IndexError(document_id)
        first, end = map(int, self.documents[document_id : document_id + 2])
        start = int(self.pointers[first]) // 4
        stop = int(self.pointers[end]) // 4 if end < self.sequences else len(self.tokens)
        return self.tokens[start:stop]


def validation_document(token_bytes: bytes | memoryview) -> bool:
    """Same content always maps to the same split, including duplicate documents."""
    return int.from_bytes(hashlib.sha256(token_bytes).digest()[:8], "little") % 1000 == 0


def prepare(prefix, output, *, tokenizer_dir, provenance, seed=1234):
    """Audit the entire corpus; never modify or re-tokenize the source files.

    provenance is an explicit, auditable description of how this binary was
    produced, not a flag to waive tokenizer verification.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    provenance = dict(provenance)
    for key in ("dataset_source", "tokenizer_source", "verification_evidence"):
        if not isinstance(provenance.get(key), str) or not provenance[key].strip():
            raise ValueError(f"missing data provenance: {key}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_dir), local_files_only=True, trust_remote_code=False
    )
    vocab_size = len(tokenizer)
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("tokenizer has no EOS")
    corpus = IndexedCorpus(prefix)
    target = output
    target.parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=target.name + ".partial-", dir=target.parent))
    split = np.lib.format.open_memmap(
        output / "split.npy", mode="w+", dtype=np.uint8, shape=(len(corpus),)
    )
    lengths = np.lib.format.open_memmap(
        output / "document_lengths.npy", mode="w+", dtype=np.int64, shape=(len(corpus),)
    )
    digest = hashlib.sha256()
    for i in range(len(corpus)):
        doc = corpus.document(i)
        if int(doc.min()) < 0 or int(doc.max()) >= vocab_size or int(doc[-1]) != eos:
            raise ValueError(
                f"token range/EOS mismatch in document {i}; check tokenizer provenance"
            )
        raw = memoryview(doc).cast("B")
        digest.update(raw)
        split[i] = validation_document(raw)
        lengths[i] = len(doc)
        if i % 1_000_000 == 0:
            print(json.dumps({"documents_audited": i, "total": len(corpus)}), flush=True)
    split.flush()
    lengths.flush()
    if not np.any(split == 1) or not np.any(split == 0):
        raise ValueError("content split produced an empty partition")
    files = {}
    totals = {}
    for name, value in (("train", 0), ("validation", 1)):
        ids = np.flatnonzero(split == value).astype(np.int64)
        if name == "train":
            np.random.default_rng(seed).shuffle(ids)
        offsets = np.empty(len(ids) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(lengths[ids], out=offsets[1:])
        for suffix, array in (("documents", ids), ("offsets", offsets)):
            filename = f"{name}_{suffix}.npy"
            np.save(output / filename, array, allow_pickle=False)
            files[filename] = sha256_file(output / filename)
        totals[name] = int(offsets[-1])
    tokenizer_files = {
        p.name: sha256_file(p)
        for p in Path(tokenizer_dir).iterdir()
        if p.is_file() and ("token" in p.name or p.name in ("vocab.json", "merges.txt"))
    }
    if not tokenizer_files:
        raise ValueError("no tokenizer files available for provenance")
    manifest = {
        "schema": 1,
        "prefix": str(Path(prefix).resolve()),
        "seed": seed,
        "bin_sha256": digest.hexdigest(),
        "idx_sha256": sha256_file(str(prefix) + ".idx"),
        "stored_tokens": len(corpus.tokens),
        "documents": len(corpus),
        "tokens": totals,
        "files": files,
        "vocab_size": vocab_size,
        "eos": eos,
        "tokenizer_files": tokenizer_files,
        "provenance": provenance,
        "split": "sha256(token-content)-mod1000-zero-validation",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    output.rename(target)
    return manifest


def verify_sources(prepared):
    manifest = json.loads((Path(prepared) / "manifest.json").read_text())
    for suffix in ("bin", "idx"):
        if sha256_file(manifest["prefix"] + "." + suffix) != manifest[suffix + "_sha256"]:
            raise ValueError(f"source indexed data changed: {suffix}")
    for filename, expected in manifest["files"].items():
        if sha256_file(Path(prepared) / filename) != expected:
            raise ValueError(f"prepared data changed: {filename}")
    return {
        "source_hashes_verified": True,
        "manifest_sha256": sha256_file(Path(prepared) / "manifest.json"),
    }


class TokenStream:
    def __init__(self, prepared, split="train"):
        if split not in ("train", "validation"):
            raise ValueError(split)
        self.root = Path(prepared)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.corpus = IndexedCorpus(self.manifest["prefix"])
        for suffix in ("documents", "offsets"):
            filename = f"{split}_{suffix}.npy"
            if sha256_file(self.root / filename) != self.manifest["files"][filename]:
                raise ValueError(f"prepared data changed: {filename}")
        self.ids = np.load(self.root / f"{split}_documents.npy", mmap_mode="r")
        self.offsets = np.load(self.root / f"{split}_offsets.npy", mmap_mode="r")

    def __len__(self):
        return int(self.offsets[-1])

    def read(self, offset, count):
        if count < 0 or offset < 0 or offset + count > len(self):
            raise ValueError("read would repeat or escape the one-pass token stream")
        result = np.empty(count, dtype=np.int64)
        written = 0
        while written < count:
            row = int(np.searchsorted(self.offsets, offset, side="right") - 1)
            inner = offset - int(self.offsets[row])
            doc = self.corpus.document(int(self.ids[row]))
            take = min(count - written, len(doc) - inner)
            result[written : written + take] = doc[inner : inner + take]
            written += take
            offset += take
        return result

    def microbatch(self, step, microstep, *, dp_rank, dp_size, mbs, gbs, seq_length):
        if not 0 <= dp_rank < dp_size or gbs % (dp_size * mbs):
            raise ValueError("invalid data partition")
        if not 0 <= microstep < gbs // (dp_size * mbs) or step < 0:
            raise ValueError("invalid training cursor")
        sample = step * gbs + microstep * dp_size * mbs + dp_rank * mbs
        return self.read(sample * seq_length, mbs * seq_length).reshape(mbs, seq_length)
