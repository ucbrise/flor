from typing import List, Dict, Set
import pandas as pd
import ast
import os

from flor.constants import *
from flor.hlast import apply

from flor.hlast.visitors import LoggedExpVisitor
from flor import database


def get_dims(pivot_vars: Dict[str, Set[str]], apply_vars: List[str]):
    if any([applied_v in pivot_vars["INNR_LOOP"] for applied_v in apply_vars]):
        return INNR_LOOP  # FULL SCAN --replay_flor 1/1
    if any([applied_v in pivot_vars["OUTR_LOOP"] for applied_v in apply_vars]):
        return OUTR_LOOP  # INDEX SCAN --replay_flor
    return DATA_PREP  # INDEX LOOKUP --replay_flor 0/1


def apply_variables(apply_vars, path):
    apply(diff_vars(apply_vars, path), path)


def batch_replay(apply_vars: List[str], path: str, versions: pd.Series, loglvl):

    mode = "--replay_flor"
    vid_vars_mode = []

    for hexsha in versions:
        if loglvl == DATA_PREP:
            vid_vars_mode.append((hexsha, ", ".join(apply_vars), mode + " 0/1"))
        elif loglvl == OUTR_LOOP:
            vid_vars_mode.append((hexsha, ", ".join(apply_vars), mode))
        elif loglvl == INNR_LOOP:
            vid_vars_mode.append((hexsha, ", ".join(apply_vars), mode + " 1/2"))
            vid_vars_mode.append((hexsha, ", ".join(apply_vars), mode + " 2/2"))
        else:
            raise

    db_conn = database.start_db()

    database.add_replay(db_conn, os.getcwd(), path, vid_vars_mode)
    print(f"Flordb registered {len(vid_vars_mode)} replay jobs.")

    db_conn.close()


def diff_vars(apply_vars: List[str], path: str):
    with open(path, "r") as f:
        tree = ast.parse(f.read())
    visitor = LoggedExpVisitor()
    visitor.visit(tree)
    return [v for v in apply_vars if v not in visitor.names]


__all__ = ["get_dims", "batch_replay"]
