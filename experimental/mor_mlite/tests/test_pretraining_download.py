import json

import pytest
from requests import HTTPError, Response
from requests.exceptions import ConnectionError, ReadTimeout

from mor_mlite.pretraining import prepare_raw as prep


@pytest.mark.parametrize("error", [ReadTimeout("private signed URL"), ConnectionError("private")])
def test_transient_download_retries_without_logging_secrets(error, capsys):
    calls, delays = [], []

    def operation(**kwargs):
        calls.append(kwargs)
        if len(calls) < 3:
            raise error
        return "cached_file"

    assert (
        prep.download_with_retry(operation, sleep=delays.append, revision="pinned") == "cached_file"
    )
    assert calls == [{"revision": "pinned"}] * 3
    assert delays == [5, 10]
    assert "private" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "status,retry", [(401, False), (403, False), (404, False), (429, True), (503, True)]
)
def test_retry_http_classification(status, retry):
    response = Response()
    response.status_code = status
    assert prep.transient_download_error(HTTPError(response=response)) is retry


def test_retry_is_bounded_and_validation_errors_are_not_retried():
    delays = []

    def timeout():
        raise ReadTimeout()

    with pytest.raises(ReadTimeout):
        prep.download_with_retry(timeout, attempts=3, sleep=delays.append)
    assert delays == [5, 10]
    delays.clear()

    def invalid():
        raise ValueError("wrong hash")

    with pytest.raises(ValueError):
        prep.download_with_retry(invalid, sleep=delays.append)
    assert delays == []


def test_completed_shard_reused_offline_but_tampering_rejected(tmp_path):
    state = {
        "contract": {"revision": "pinned"},
        "requested_tokens": 100,
        "tokens": 123,
        "files": {},
    }
    for name in ("text.bin", "text.idx"):
        (tmp_path / name).write_bytes(b"sealed")
        state["files"][name] = prep.sha256_file(tmp_path / name)
    (tmp_path / "complete.json").write_text(json.dumps(state))
    assert prep.encode_shard(None, tmp_path, None, 100, state["contract"]) == state
    with pytest.raises(ValueError, match="contract"):
        prep.completed_shard(tmp_path, 99, state["contract"])
    (tmp_path / "text.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        prep.completed_shard(tmp_path, 100, state["contract"])
