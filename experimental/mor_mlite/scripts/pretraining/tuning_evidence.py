"""Keep representative-data screening separate from final-data qualification.

Screening can establish the safe starting candidate. It never replaces that
candidate's final-data 50-update/evaluation/checkpoint qualification, and its
input hashes are never compared as though they came from the final corpus.
"""

from pathlib import Path


def compatible(screen, final):
    """Only corpus identity, cursor budget and token schedule may differ."""

    def identity(contract):
        return {
            key: value
            for key, value in contract.items()
            if key
            not in {"data_sha256", "total_steps", "total_tokens", "discarded_training_tokens"}
        }

    if identity(screen) != identity(final):
        raise ValueError("screening changed model/source/environment/parallelism or optimizer")


def load_screening(paths, *, world_size, source):
    from inspect_trials import inspect

    from mor_mlite.pretraining.config import Experiment
    from mor_mlite.pretraining.tuning import select_mbs

    result = {}
    common = None
    for path in paths:
        trial, contract, hashes = inspect(Path(path))
        arm = contract["experiment"]["arm"]
        if (
            contract["experiment"]
            != Experiment(arm, world_size=world_size, micro_batch_size=trial.mbs).to_dict()
            or contract["source_sha256"] != source
        ):
            raise ValueError("screening source/config mismatch")
        identity = (
            {k: v for k, v in contract.items() if k not in {"experiment", "parameters"}},
            hashes,
        )
        if common is not None and identity != common:
            raise ValueError("screening runs changed their own data/schedule/environment/inputs")
        common = identity
        result.setdefault(arm, []).append((trial, contract, hashes))
    for records in result.values():
        records.sort(key=lambda record: record[0].mbs)
        select_mbs([record[0] for record in records])
    return result


def check_screening_contract(records, final):
    for trial, contract, _ in records:
        comparable = {
            **final,
            "experiment": {**final["experiment"], "micro_batch_size": trial.mbs},
        }
        compatible(contract, comparable)


def combined_trials(screening, final):
    by_size = {trial.mbs: trial for trial in screening}
    if len({trial.mbs for trial in final}) != len(final):
        raise ValueError("duplicate final-data MBS trial")
    by_size.update({trial.mbs: trial for trial in final})
    return [by_size[size] for size in sorted(by_size)]


def next_candidate(screening, final):
    from mor_mlite.pretraining.tuning import next_mbs, select_mbs

    if not final:
        return select_mbs(screening) if screening else 1
    merged = combined_trials(screening, final)
    selected = select_mbs(merged)
    following = next_mbs(merged[-1])
    if following is not None:
        return following
    # Even a historically faster size must qualify on the final dataset before
    # it can be frozen; no pilot-only candidate can start formal training.
    if selected not in {trial.mbs for trial in final if trial.eligible}:
        return selected
    return None
