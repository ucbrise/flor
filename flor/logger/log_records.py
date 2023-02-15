from flor.state import State
from flor.constants import FLORFILE
from typing import Optional, List, Dict, Any


import csv
import pandas as pd
from pathlib import Path

import atexit

records: List[Dict[str, Any]] = []


def deferred_init():
    with open(FLORFILE, "r") as f:
        records.extend(list(csv.DictReader(f)))
    atexit.register(flush)


def put(name, value, ow=True):
    assert State.epoch is not None
    d = {
        "epoch": int(State.epoch),
        "step": len(records) + 1,
        "name": name,
        "value": value,
    }


def get(name):
    pass


def flush():
    pd.DataFrame(records).to_csv(FLORFILE)


def exists():
    return Path(FLORFILE).exists()


def eval(select, where=None) -> pd.DataFrame:
    assert records is not None
    df = pd.DataFrame(records)
    if where is not None:
        df = df.query(where)
    return df[select]
