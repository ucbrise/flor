import csv
import pandas as pd
import numpy as np
from flor.query.unpack import unpack, clear_stash
from flor.query import database
from pathlib import Path

from flor.query.pivot import *

facts = None


def log_records():
    global facts
    if facts is not None:
        return facts
    unpack()

    facts = (
        pd.DataFrame(
            database.get_log_records(),
            columns=["projid", "tstamp", "vid", "epoch", "step", "name", "value"],
        )
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


def full_pivot():
    global facts
    if facts is None:
        facts = log_records()

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

    pivots = []

    dp_pivot = data_prep_pivot(facts, data_prep_names)
    if dp_pivot is not None:
        dp_pivot = dp_pivot[
            [c for c in dp_pivot.columns if c != "epoch" and c != "step"]
        ]
        pivots.append((("projid", "tstamp", "vid"), dp_pivot))
    ol_pivot = outer_loop_pivot(facts, outer_loop_names)
    if ol_pivot is not None:
        ol_pivot = ol_pivot[[c for c in ol_pivot.columns if c != "step"]]
        pivots.append((("projid", "tstamp", "vid", "epoch"), ol_pivot))
    il_pivot = inner_loop_pivot(facts, inner_loop_names)
    if il_pivot is not None:
        pivots.append((("projid", "tstamp", "vid", "epoch", "step"), il_pivot))

    if pivots:
        left_keys, rolling_df = pivots[0]
        for right_keys, right_df in pivots[1:]:
            rolling_df = rolling_df.merge(right_df, how="outer", on=tuple(left_keys))
            left_keys = right_keys
        rolling_df = rolling_df.drop_duplicates(
            subset=["projid", "tstamp", "vid", "epoch", "step"]
        )
        keys = ["projid", "tstamp", "vid", "epoch", "step"]
        keys.extend([c for c in rolling_df.columns if c not in keys])
        return rolling_df[keys].sort_values(by=["tstamp", "epoch", "step"])


__all__ = ["facts", "log_records", "full_pivot", "clear_stash"]
