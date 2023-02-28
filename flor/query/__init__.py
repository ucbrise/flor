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

    def post_proc(df, df_keys):
        df_keys = list(df_keys)
        df_keys.extend([c for c in df.columns if c not in df_keys])
        return df[df_keys].sort_values(
            [k for k in df_keys if k in ("tstamp", "epoch", "step")]
        )

    if dp_pivot is not None and ol_pivot is not None:
        dp_ol = dp_pivot.merge(ol_pivot, how="outer", on=dp_keys)
        if il_pivot:
            return post_proc(dp_ol.merge(il_pivot, how="outer", on=ol_keys), all_keys)
        return post_proc(dp_ol, ol_keys)
    elif dp_pivot is not None and il_pivot is not None:
        return post_proc(dp_pivot.merge(il_pivot, how="outer", on=dp_keys), all_keys)
    elif ol_pivot is not None and il_pivot is not None:
        return post_proc(ol_pivot.merge(il_pivot, how="outer", on=ol_keys), all_keys)
    elif dp_pivot is not None:
        return post_proc(dp_pivot, dp_keys)
    elif ol_pivot is not None:
        return post_proc(ol_pivot, ol_keys)
    elif il_pivot is not None:
        return post_proc(il_pivot, all_keys)


__all__ = ["facts", "log_records", "full_pivot", "clear_stash"]


# if pivots:
#     left_keys, rolling_df = pivots[0]
#     for right_keys, right_df in pivots[1:]:
#         rolling_df = rolling_df.merge(right_df, how="cross", on=tuple(left_keys))
#         left_keys = right_keys
#     left_keys = list(left_keys)
#     left_keys.extend([c for c in rolling_df.columns if c not in left_keys])
#     return rolling_df[left_keys].sort_values(by=["tstamp", "epoch", "step"])
