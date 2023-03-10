from typing import List, Dict, Set
from enum import Enum
from pathlib import Path
import pandas as pd
import ast
from time import time

from flor.constants import *
from flor.state import State
from flor.hlast import apply

from flor.hlast.visitors import LoggedExpVisitor

import subprocess


def get_dims(pivot_vars: Dict[str, Set[str]], apply_vars: List[str]):
    if any([applied_v in pivot_vars["INNR_LOOP"] for applied_v in apply_vars]):
        return INNR_LOOP
    if any([applied_v in pivot_vars["OUTR_LOOP"] for applied_v in apply_vars]):
        return OUTR_LOOP
    return DATA_PREP


def batch_replay(apply_vars: List[str], path: str, versions: pd.Series, loglvl):
    # TODO: argv processing
    base_cmd = ["python", path, "--replay_flor"]

    start_time = time()
    assert State.repo is not None
    for hexsha in versions:
        State.repo.git.checkout(hexsha)
        apply(diff_vars(apply_vars, path), path)

        if loglvl == DATA_PREP:
            subprocess.run(
                base_cmd
                + [
                    "0/1",
                ]
            )
        elif loglvl == OUTR_LOOP:
            subprocess.run(base_cmd)
        elif loglvl == INNR_LOOP:
            subprocess.run(
                base_cmd
                + [
                    "1/2",
                ]
            )
            subprocess.run(
                base_cmd
                + [
                    "2/2",
                ]
            )
        else:
            raise
        State.repo.git.stash()
    print(f"Time elapsed: {time() - start_time} seconds")


def diff_vars(apply_vars: List[str], path: str):
    with open(path, "r") as f:
        tree = ast.parse(f.read())
    visitor = LoggedExpVisitor()
    visitor.visit(tree)
    return [v for v in apply_vars if v not in visitor.names]


__all__ = ["get_dims", "batch_replay"]
