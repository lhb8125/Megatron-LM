import json

import pytest
from test_pretraining_probes import load_probe


@pytest.fixture
def failure(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"tokens": {"train": 5954 * 8_388_608}}))
    return {
        "status": "failed", "active": {},
        "completed": {"A": 200, "B": 200, "C": 520, "D": 520},
        "binding": {"data_manifest": str(manifest), "max_updates": "400",
                    "parallel_arms": "4", "output_prefix": "${PROJECT_ROOT}/run",
                    "remote_root": "/remote"},
        "intent": {"arm": "A", "from_step": 200, "to_step": 600,
                   "stop_tokens": 600 * 8_388_608, "output": "/remote/run-a-000200-000600"},
    }


LOG = "in launch_remote_slurm\n sync_with_retry(\nRemoteSyncError: sync_project failed"


def test_presubmit_proof(failure):
    module = load_probe("reconcile_presubmit")
    assert module.validate_failure(failure, LOG, [], [], False)["from_step"] == 200


@pytest.mark.parametrize("local,remote,exists", [(["job"], [], False), ([], ["job"], False), ([], [], True)])
def test_presubmit_rejects_submission_evidence(failure, local, remote, exists):
    with pytest.raises(ValueError, match="no retry"):
        load_probe("reconcile_presubmit").validate_failure(failure, LOG, local, remote, exists)


@pytest.mark.parametrize("log", ["sbatch failed", LOG + '"job_id"', LOG + '"scheduler_id"'])
def test_presubmit_rejects_ambiguous_log(failure, log):
    with pytest.raises(ValueError, match="no retry"):
        load_probe("reconcile_presubmit").validate_failure(failure, log, [], [], False)


def test_presubmit_rejects_changed_intent(failure):
    failure["intent"]["to_step"] = 700
    with pytest.raises(ValueError, match="differs"):
        load_probe("reconcile_presubmit").validate_failure(failure, LOG, [], [], False)


def test_presubmit_rejects_running_state(failure):
    failure["status"] = "watching"
    with pytest.raises(ValueError, match="not a stopped"):
        load_probe("reconcile_presubmit").validate_failure(failure, LOG, [], [], False)


def preauth_row():
    return {"job_id": "local", "status": "failed", "scheduler_id": None,
            "metadata_json": json.dumps({"launch_error":
                "remote command exited with status 255\n"
                "ssh_exchange_identification: Connection closed by remote host\n"
                "No JSON envelope found in remote output:\n"})}


def test_preauth_reconcile_requires_registry_and_empty_remote(failure):
    row = preauth_row()
    module = load_probe("reconcile_presubmit")
    log = json.dumps(row)
    assert module.validate_failure(failure, log, [row], [], False)["arm"] == "A"
    for remote, exists in [(["remote-job"], False), ([], True)]:
        with pytest.raises(ValueError, match="no retry"):
            module.validate_failure(failure, log, [row], remote, exists)


@pytest.mark.parametrize("field,value", [("scheduler_id", "123"), ("status", "queued"),
                                        ("metadata_json", '{}'), ("job_id", "other")])
def test_preauth_reconcile_rejects_ambiguous_local_record(failure, field, value):
    row = preauth_row()
    log = json.dumps(row)
    row[field] = value
    with pytest.raises(ValueError, match="no retry"):
        load_probe("reconcile_presubmit").validate_failure(failure, log, [row], [], False)
