import csv
from typing import List, Set, Dict
import pandas as pd
import numpy as np
from flor.query.unpack import unpack, clear_stash
from flor.query import database
from flor.shelf import cwd_shelf
from pathlib import Path

from flor.query.pivot import *
from flor.query.engine import *
from flor.constants import *

facts = None
pivot_vars: Dict[str, Set[str]] = {
    "DATA_PREP": set([]),
    "OUTR_LOOP": set([]),
    "INNR_LOOP": set([]),
}


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

    data_prep_gb = facts.groupby(by=list(DATA_PREP + ("name",)))
    for rowid, agg in data_prep_gb.count()["value"].items():
        name = str(tuple(rowid)[-1])  # type: ignore
        if agg == 1:
            pivot_vars["DATA_PREP"] |= {
                name,
            }

    outer_loop_gb = facts.groupby(by=list(OUTR_LOOP + ("name",)))
    for rowid, agg in outer_loop_gb.count()["value"].items():
        name = str(tuple(rowid)[-1])  # type: ignore
        if name not in pivot_vars["DATA_PREP"] and agg == 1:
            pivot_vars["OUTR_LOOP"] |= {
                name,
            }

    pivot_vars["INNR_LOOP"] |= set(
        [
            name
            for name in facts["name"]
            if name not in pivot_vars["DATA_PREP"]
            and name not in pivot_vars["OUTR_LOOP"]
        ]
    )

    dp_keys = DATA_PREP
    dp_pivot = data_prep_pivot(facts, pivot_vars["DATA_PREP"])

    ol_keys = OUTR_LOOP
    ol_pivot = outer_loop_pivot(facts, pivot_vars["OUTR_LOOP"])

    all_keys = INNR_LOOP
    il_pivot = inner_loop_pivot(facts, pivot_vars["INNR_LOOP"])

    def post_proc(df, df_keys):
        df_keys = list(df_keys)
        df_keys.extend([c for c in df.columns if c not in df_keys])
        return df[df_keys].sort_values(
            [k for k in df_keys if k in ("tstamp", "epoch", "step")]
        )

    if ol_pivot is not None and il_pivot is not None:
        ol_il = il_pivot.merge(ol_pivot, how="outer", on=ol_keys)
        if dp_pivot is not None:
            return post_proc(ol_il.merge(dp_pivot, how="outer", on=dp_keys), all_keys)
        return post_proc(ol_il, all_keys)
    elif dp_pivot is not None and ol_pivot is not None:
        return post_proc(ol_pivot.merge(dp_pivot, how="outer", on=dp_keys), ol_keys)
    elif dp_pivot is not None and il_pivot is not None:
        return post_proc(il_pivot.merge(dp_pivot, how="outer", on=dp_keys), all_keys)
    elif dp_pivot is not None:
        return post_proc(dp_pivot, dp_keys)
    elif ol_pivot is not None:
        return post_proc(ol_pivot, ol_keys)
    elif il_pivot is not None:
        return post_proc(il_pivot, all_keys)


def replay(apply_vars: List[str], where_clause: str, path: str):
    """
    apply_vars : ['device', 'optimizer', 'learning_rate', ...]
    where_clause: stated in Pandas/SQL, passed to full_pivot
    path: `train_rnn.py` or such denoting main python script
    """
    assert Path(path).suffix == ".py"
    df = full_pivot()
    assert df is not None

    loglvl = get_dims(pivot_vars, apply_vars)
    if loglvl == DATA_PREP:
        schedule = df.query(where_clause)[list(DATA_PREP)].drop_duplicates()
        versions = schedule["vid"].drop_duplicates()
        print(f"Replaying {len(versions)} at DATA_PREP loglevel: {DATA_PREP}")
        res = input("Continue [Y/n]? ")
        if res.strip().lower() != "n":
            batch_replay(apply_vars, path, versions, DATA_PREP)
    elif loglvl == OUTR_LOOP:
        schedule = df.query(where_clause)[list(OUTR_LOOP)].drop_duplicates()
        versions = schedule["vid"].drop_duplicates()
        print(f"Replaying {len(versions)} at OUTR_LOOP loglevel: {OUTR_LOOP}")
        res = input("Continue [Y/n]? ")
        if res.strip().lower() != "n":
            batch_replay(apply_vars, path, versions, OUTR_LOOP)
    elif loglvl == INNR_LOOP:
        schedule = df.query(where_clause)[list(INNR_LOOP)].drop_duplicates()
        versions = schedule["vid"].drop_duplicates()
        print(f"Replaying {len(versions)} at INNR_LOOP loglevel: {INNR_LOOP}")
        res = input("Continue [Y/n]? ")
        if res.strip().lower() != "n":
            batch_replay(apply_vars, path, versions, INNR_LOOP)
    else:
        raise


__all__ = ["facts", "log_records", "full_pivot", "clear_stash", "replay"]
