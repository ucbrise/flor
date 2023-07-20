import ast
from typing import List, Optional, Set, Dict
import pandas as pd
import numpy as np
from flor.shelf import home_shelf, cwd_shelf
from flor.query.unpack import unpack, clear_stash, get_stash
from flor.query import database

from flor.shelf import cwd_shelf
from pathlib import Path
from IPython.display import display

from flor.query.pivot import *
from flor.query.engine import *
from flor.constants import *
from flor.state import State

from flor.hlast.visitors import LoggedExpVisitor
from shutil import copyfile


pivot_vars: Dict[str, Set[str]] = {
    "DATA_PREP": set([]),
    "OUTR_LOOP": set([]),
    "INNR_LOOP": set([]),
}


def log_records(skip_unpack=False):
    if not skip_unpack:
        unpack()
    else:
        # do unpack initialize
        assert cwd_shelf.in_shadow_branch()
        if State.db_conn is None:
            database.start_db(cwd_shelf.get_projid())

    facts = (
        pd.DataFrame(
            database.get_log_records(),
            columns=[
                "projid",
                "runid",
                "tstamp",
                "vid",
                "epoch",
                "step",
                "name",
                "value",
            ],
        )
        .astype(
            {
                "projid": str,
                "runid": str,
                "tstamp": str,
                "vid": str,
                "epoch": int,
                "step": int,
                "name": str,
                "value": object,
            }
        )
        .sort_values(by=["tstamp", "epoch", "step"])
    )
    facts["projid"] = facts["projid"].map(lambda x: x.replace("\x1b[m", ""))
    return facts


def full_pivot(facts: pd.DataFrame):
    data_prep_gb = facts.drop_duplicates()[list(DATA_PREP) + ["name", "value"]].groupby(
        by=list(DATA_PREP + ("name",))
    )

    skip_set = set([])
    for rowid, agg in data_prep_gb.count()["value"].items():
        name = str(tuple(rowid)[-1])  # type: ignore
        if agg == 1 and name not in skip_set:
            pivot_vars["DATA_PREP"] |= {
                name,
            }
        else:
            skip_set.add(name)

    outer_loop_gb = (
        facts[list(OUTR_LOOP) + ["name", "value"]]
        .drop_duplicates()
        .groupby(by=list(OUTR_LOOP + ("name",)))
    )
    skip_set = set([])
    for rowid, agg in outer_loop_gb.count()["value"].items():
        name = str(tuple(rowid)[-1])  # type: ignore
        if name not in pivot_vars["DATA_PREP"] and agg == 1 and name not in skip_set:
            pivot_vars["OUTR_LOOP"] |= {
                name,
            }
        else:
            skip_set.add(name)

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


def replay(
    apply_vars: List[str], where_clause: Optional[str] = None, path: str = "train.py"
):
    """
    apply_vars : ['device', 'optimizer', 'learning_rate', ...]
    where_clause: stated in Pandas/SQL, passed to full_pivot
    path: `train_rnn.py` or such denoting main python script
    """
    assert Path(path).suffix == ".py"
    facts = log_records(skip_unpack=True)
    df = full_pivot(facts)
    assert df is not None

    stash = clear_stash()
    assert stash is not None

    with open(path, "r") as f:
        tree = ast.parse(f.read())
    lev = LoggedExpVisitor()
    lev.visit(tree)

    vars_in_latest = []
    for var in apply_vars:
        if var not in facts["name"].values:
            assert (
                var in lev.names
            ), f"FLOR: could not find logged var `{var}` in any version of `{path}`."
            copyfile(path, stash / Path(var).with_suffix(".py"))
            vars_in_latest.append(var)
            State.hls_hits.add(var)
            State.grouped_names[var] = lev.names[var]

    for var in vars_in_latest:
        # find where in source code
        # Prompt the user
        # TODO: Infer which one with fast static analysis.
        msg = input(
            f"What is the log level of logging statement `{var}`? Leave blank to infer `{lev.get_loglevel([var,])}`: "
        )
        msg = msg.strip().upper()
        if msg in pivot_vars:
            pivot_vars[msg].add(var)
        elif msg == "" or not msg:
            pivot_vars[
                lev.get_loglevel(
                    [
                        var,
                    ]
                )
            ].add(var)
        else:
            raise

    loglvl = get_dims(pivot_vars, apply_vars)
    dp_schedule = (
        (df.query(where_clause) if where_clause is not None else df)[list(DATA_PREP)]
        .drop_duplicates()
        .merge(
            pd.DataFrame(database.get_schedule(DATA_PREP)).astype(
                {
                    "projid": str,
                    "runid": str,
                    "tstamp": str,
                    "vid": str,
                    "prep_secs": float,
                    "eval_secs": float,
                }
            ),
            how="inner",
            on=DATA_PREP,
        )[list(DATA_PREP) + ["prep_secs", "eval_secs"]]
    )
    if loglvl == DATA_PREP:
        versions = dp_schedule["vid"].drop_duplicates()
        display(dp_schedule)
        res = input(
            f"Continue replaying {len(versions)} versions at DATA_PREP level for {'{:.2f}'.format(3 * len(versions) + sum(dp_schedule['prep_secs']) + sum(dp_schedule['eval_secs']))} seconds?[Y/n]? "
        )
        if res.strip().lower() != "n":
            batch_replay(apply_vars, path, versions, DATA_PREP)
    elif loglvl == OUTR_LOOP:
        size_bytes = home_shelf.get_checkpoint_bytes_per_epoch(cwd_shelf.get_projid())
        schedule = (
            (df.query(where_clause) if where_clause is not None else df)[
                list(OUTR_LOOP)
            ]
            .drop_duplicates()
            .merge(
                pd.DataFrame(database.get_schedule(OUTR_LOOP)).astype(
                    {
                        "projid": str,
                        "runid": str,
                        "tstamp": str,
                        "vid": str,
                        "epoch": int,
                        "seconds": float,
                    }
                ),
                how="inner",
                on=OUTR_LOOP,
            )[
                list(OUTR_LOOP)
                + [
                    "seconds",
                ]
            ]
        )
        schedule["seconds"] = size_bytes * DESERIALIZATION_COEFF
        versions = schedule["vid"].drop_duplicates()
        display(schedule)
        res = input(
            f"Continue replaying {len(versions)} versions at OUTR_LOOP level for {'{:.2f}'.format((3 * len(versions) + sum(dp_schedule['prep_secs']) + sum(dp_schedule['eval_secs'])) + sum(schedule['seconds']))} seconds [Y/n]? "
        )
        if res.strip().lower() != "n":
            batch_replay(apply_vars, path, versions, OUTR_LOOP)
    elif loglvl == INNR_LOOP:
        schedule = (
            (df.query(where_clause) if where_clause is not None else df)[
                list(OUTR_LOOP)
            ]
            .drop_duplicates()
            .merge(
                pd.DataFrame(database.get_schedule(OUTR_LOOP)).astype(
                    {
                        "projid": str,
                        "runid": str,
                        "tstamp": str,
                        "vid": str,
                        "epoch": int,
                        "seconds": float,
                    }
                ),
                how="inner",
                on=OUTR_LOOP,
            )[
                list(OUTR_LOOP)
                + [
                    "seconds",
                ]
            ]
        )
        versions = schedule["vid"].drop_duplicates()
        display(schedule)
        res = input(
            f"Continue replaying {len(versions)} versions at INNR_LOOP level for {'{:.2f}'.format((3 * len(versions) + sum(dp_schedule['prep_secs']) + sum(dp_schedule['eval_secs'])) + sum(schedule['seconds']))} seconds [Y/n]? "
        )
        if res.strip().lower() != "n":
            batch_replay(apply_vars, path, versions, INNR_LOOP)
    else:
        raise


__all__ = ["log_records", "full_pivot", "clear_stash", "get_stash", "replay"]
