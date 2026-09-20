from __future__ import annotations

import json
from pathlib import Path

import pytest

from mor_mlite.hf import resolve_hf_checkpoint, validate_local_hf_checkpoint


def _checkpoint(root: Path, *, sharded: bool = False) -> Path:
    root.mkdir(parents=True)
    (root / "config.json").write_text(json.dumps({"model_type": "qwen3_moe"}), encoding="utf-8")
    if sharded:
        shard = "model-00001-of-00001.safetensors"
        (root / shard).write_bytes(b"test")
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"model.embed_tokens.weight": shard}}),
            encoding="utf-8",
        )
    else:
        (root / "model.safetensors").write_bytes(b"test")
    return root


def test_validate_local_hf_checkpoint_accepts_config_file(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "hf")
    assert validate_local_hf_checkpoint(checkpoint / "config.json") == checkpoint.resolve()


def test_validate_local_hf_checkpoint_checks_every_index_shard(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "hf", sharded=True)
    (checkpoint / "model-00001-of-00001.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="missing shard"):
        validate_local_hf_checkpoint(checkpoint)


@pytest.mark.parametrize(
    "unsafe_name",
    (
        "../outside.safetensors",
        "nested/model.safetensors",
        r"nested\model.safetensors",
        "/tmp/model.safetensors",
        "..",
        "",
    ),
)
def test_validate_local_hf_checkpoint_rejects_unsafe_index_shard_paths(
    tmp_path: Path, unsafe_name: str
) -> None:
    checkpoint = tmp_path / "hf"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps({"model_type": "qwen3_moe"}), encoding="utf-8"
    )
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.embed_tokens.weight": unsafe_name}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="safe relative basenames"):
        validate_local_hf_checkpoint(checkpoint)


def test_validate_local_hf_checkpoint_rejects_non_string_index_shard_path(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "hf"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps({"model_type": "qwen3_moe"}), encoding="utf-8"
    )
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.embed_tokens.weight": ["model.safetensors"]}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="safe relative basenames"):
        validate_local_hf_checkpoint(checkpoint)


def test_validate_local_hf_checkpoint_allows_basename_symlink_shard(tmp_path: Path) -> None:
    checkpoint = tmp_path / "hf"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps({"model_type": "qwen3_moe"}), encoding="utf-8"
    )
    target = tmp_path / "shared-shard.safetensors"
    target.write_bytes(b"test")
    shard = checkpoint / "model-00001-of-00001.safetensors"
    shard.symlink_to(target)
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.embed_tokens.weight": shard.name}}),
        encoding="utf-8",
    )

    assert validate_local_hf_checkpoint(checkpoint) == checkpoint.resolve()


def test_resolve_hf_checkpoint_downloads_hub_id_then_validates(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "downloaded")
    calls: list[dict[str, object]] = []

    def downloader(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(checkpoint)

    resolved = resolve_hf_checkpoint("org/model", downloader=downloader)
    assert resolved.source == "org/model"
    assert resolved.local_path == checkpoint.resolve()
    assert resolved.downloaded
    assert calls == [
        {
            "repo_id": "org/model",
            "allow_patterns": (
                "config.json",
                "model.safetensors",
                "model.safetensors.index.json",
                "model-*.safetensors",
            ),
        }
    ]


def test_resolve_hub_id_returns_downloaded_local_snapshot(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "snapshot")
    calls: list[dict[str, object]] = []

    def downloader(**kwargs) -> str:
        calls.append(kwargs)
        return str(checkpoint)

    resolved = resolve_hf_checkpoint("org/model", downloader=downloader)
    assert resolved.source == "org/model"
    assert resolved.local_path == checkpoint.resolve()
    assert resolved.downloaded
    assert calls == [
        {
            "repo_id": "org/model",
            "allow_patterns": (
                "config.json",
                "model.safetensors",
                "model.safetensors.index.json",
                "model-*.safetensors",
            ),
        }
    ]
