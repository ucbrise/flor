import csv
import pandas as pd
import numpy as np
from flor.query.unpack import unpack, resolve_cache
from pathlib import Path

from flor.query.pivot import *

facts = None


def log_records(cache_dir=None):
    global facts
    if cache_dir is None:
        cache_dir = unpack()
    assert cache_dir is not None
    if isinstance(cache_dir, str):
        cache_dir = resolve_cache(cache_dir)
    assert isinstance(cache_dir, Path)
    data = []

    for path in cache_dir.iterdir():
        if path.suffix == ".csv":
            with open(path, "r") as f:
                data.extend(list(csv.DictReader(f)))
    facts = (
        pd.DataFrame(data)
        .astype(
            {
                "projid": str,
                "tstamp": np.datetime64,
                "vid": str,
                "epoch": int,
                "step": int,
                "name": str,
                "value": object,
            }
        )
        .sort_values(by=["tstamp", "epoch", "step"])
    )
    return facts


def full_pivot(*args, **kwargs):
    global facts
    if facts is None:
        facts = log_records(*args, **kwargs)

    data_prep_names = set([])
    data_prep_gb = facts.groupby(by=["name", "vid"])
    for rowid, agg in data_prep_gb.count()["value"].items():
        name, hexsha = tuple(rowid)  # type: ignore
        if agg == 1:
            data_prep_names |= {
                name,
            }

    outer_loop_names = set([])
    outer_loop_gb = facts.groupby(by=["name", "vid", "epoch"])
    for rowid, agg in outer_loop_gb.count()["value"].items():
        name, hexsha, _ = tuple(rowid)  # type: ignore
        if name not in data_prep_names and agg == 1:
            outer_loop_names |= {
                name,
            }

    inner_loop_names = set(
        [
            name
            for name in facts["name"]
            if name not in data_prep_names and name not in outer_loop_names
        ]
    )

    print(
        f"\nData prep: {data_prep_names},\nouter_loop: {outer_loop_names},\ninner_loop: {inner_loop_names}"
    )

    data_prep_pivot(facts, data_prep_names)
