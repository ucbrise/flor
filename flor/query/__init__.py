import csv
import pandas as pd
import numpy as np
from flor.query.unpack import unpack, clear_stash
from flor.query import database
from flor.shelf import cwd_shelf
from pathlib import Path

from flor.query.pivot import *

facts = None


def log_records(skip_unpack=False):
    global facts
    if facts is not None:
        return facts
    if not skip_unpack:
        unpack()
    else:
        # do unpack initialize
        assert cwd_shelf.in_shadow_branch()
        database.start_db(cwd_shelf.get_projid())

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
        facts = log_records(skip_unpack=True)

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

    dp_keys = ("projid", "tstamp", "vid")
    dp_pivot = data_prep_pivot(facts, data_prep_names)

    ol_keys = dp_keys + ("epoch",)
    ol_pivot = outer_loop_pivot(facts, outer_loop_names)

    all_keys = ol_keys + ("step",)
    il_pivot = inner_loop_pivot(facts, inner_loop_names)

    if dp_pivot and ol_pivot:
        # join dp, ol
        if il_pivot:
            # join (join dp, ol), il
            return "dp_ol_il"
        return "dp_ol"
    elif dp_pivot and il_pivot:
        return "dp_il"
    elif ol_pivot and il_pivot:
        return "ol_il"
    elif dp_pivot:
        return dp_pivot
    elif ol_pivot:
        return ol_pivot
    elif il_pivot:
        return il_pivot


__all__ = ["facts", "log_records", "full_pivot", "clear_stash"]


# if pivots:
#     left_keys, rolling_df = pivots[0]
#     for right_keys, right_df in pivots[1:]:
#         rolling_df = rolling_df.merge(right_df, how="cross", on=tuple(left_keys))
#         left_keys = right_keys
#     left_keys = list(left_keys)
#     left_keys.extend([c for c in rolling_df.columns if c not in left_keys])
#     return rolling_df[left_keys].sort_values(by=["tstamp", "epoch", "step"])
