import csv
import pandas as pd
from flor.query.unpack import unpack

facts = None


def log_records(cache_dir=None):
    global facts
    if cache_dir is None:
        cache_dir = unpack()
    assert cache_dir is not None
    data = []

    for path in cache_dir.iterdir():
        if path.suffix == ".csv":
            with open(path, "r") as f:
                data.extend(list(csv.DictReader(f)))
    facts = pd.DataFrame(data).sort_values(by=["tstamp", "epoch", "value"])
    return facts


def full_pivot():
    global facts
    if facts is None:
        facts = log_records()
    return facts
