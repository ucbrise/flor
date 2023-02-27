from flor.state import State
from flor.constants import LOG_RECORDS
from flor import flags
from typing import Optional, List, Dict, Any


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
    print(f"flushing log records {projid}, {tstamp}")
    if flags.NAME and not flags.REPLAY:
        if record_logs:
            pd.DataFrame(record_logs).to_csv(LOG_RECORDS, index=False)
    elif flags.NAME and flags.REPLAY:
        assert State.repo is not None
        assert State.db_conn is not None
        State.db_conn.commit()
        if record_logs:
            df = pd.DataFrame(record_logs)
            df.insert(0, "vid", str(State.repo.head.commit.hexsha))
            df.insert(0, "tstamp", str(PurePath(tstamp).stem))
            df.insert(0, "projid", projid)
            df.to_sql("log_records", con=State.db_conn, if_exists="append", index=False)


def exists():
    return Path(LOG_RECORDS).exists()


def eval(select, where=None) -> pd.DataFrame:
    assert replay_logs is not None
    df = pd.DataFrame(replay_logs)
    if where is not None:
        df = df.query(where)
    return df[select]
