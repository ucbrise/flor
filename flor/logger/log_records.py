from flor.state import State
from flor.constants import LOG_RECORDS
from flor import flags
from typing import Optional, List, Dict, Any

from flor.query import database

import csv
import pandas as pd
from pathlib import Path, PurePath

replay_logs: List[Dict[str, Any]] = []
record_logs: List[Dict[str, Any]] = []


def deferred_init():
    with open(LOG_RECORDS, "r") as f:
        replay_logs.extend(list(csv.DictReader(f)))


def put(name, value):
    assert State.epoch is not None
    d = {
        "epoch": int(State.epoch),
        "step": int(State.step) if State.step is not None else None,
        "name": name,
        "value": value,
    }
    record_logs.append(d)


def put_dp(name, value):
    d = {
        "epoch": -1,
        "step": -1,
        "name": name,
        "value": value,
    }
    record_logs.append(d)


def get(name):
    return [d for d in replay_logs if d["name"] == name]


def flush(projid: str, tstamp: str):
    if flags.NAME and not flags.REPLAY:
        if len(record_logs) > 0:
            pd.DataFrame(record_logs).to_csv(LOG_RECORDS, index=False)
            print("Flor wrote log records locally.")
    elif flags.NAME and flags.REPLAY:
        assert State.repo is not None
        for rlg in record_logs:
            rlg["tstamp"] = str(PurePath(tstamp).stem)
            rlg["projid"] = projid
            rlg["runid"] = flags.NAME
            for version in State.repo.iter_commits():
                if "RECORD::" in str(version.message):
                    rlg["vid"] = str(version.hexsha)
                    break
            assert (
                "vid" in rlg
            ), "Failed to find a recent FLOR commit, are you calling from a flor.shadow branch?"
        database.write_log_records(record_logs)


def exists():
    return Path(LOG_RECORDS).exists()


def eval(select, where=None) -> pd.DataFrame:
    assert replay_logs is not None
    df = pd.DataFrame(replay_logs)
    if where is not None:
        df = df.query(where)
    return df[select]
