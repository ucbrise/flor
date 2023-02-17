import csv
import pandas as pd
from flor.query.unpack import unpack


def log_records(cache_dir=None):
    if cache_dir is None:
        cache_dir = unpack()
    assert cache_dir is not None
    data = []

    for path in cache_dir.iterdir():
        if path.suffix == ".csv":
            with open(path, "r") as f:
                data.extend(list(csv.DictReader(f)))

    return pd.DataFrame(data).sort_values(by=["tstamp", "epoch", "value"])


def pull_pivot():
    facts = log_records()
