# type: ignore
import ast
from argparse import Namespace
from shutil import copyfile
from pathlib import Path
from sys import stdout
from typing import Dict, List, Set
import os

from flor.hlast.gtpropagate import propagate  # type: ignore
from flor.state import State
import flor.query as q
from .visitors import NoGradVisitor, NoGradTransformer, LoggedExpVisitor


def backprop(lineno: int, source: str, target: str, out=None):
    syntactic_prop(lineno, source, target, out)


def syntactic_prop(lineno: int, source, target, out=None):
    if out is None:
        with open(str(source), "r") as src, open(str(target), "r") as dst:
            return propagate(
                Namespace(
                    lineno=lineno,
                    source=src,
                    target=dst,
                    out=str(target),
                    gumtree=dict(),
                )
            )
    else:
        with open(str(source), "r") as src, open(str(target), "r") as dst:
            return propagate(
                Namespace(
                    lineno=lineno, source=src, target=dst, out=out, gumtree=dict()
                )
            )


def apply(names: List[str], dst: str):
    """
    Caller checks out a previous version
    """

    fp = Path(dst)
    facts = q.log_records(skip_unpack=True)

    # Get latest timestamp for each variable name
    historical_names = facts[facts["name"].isin(names)][
        ["name", "tstamp", "vid", "value"]
    ]
    historical_names = historical_names[historical_names["value"].notna()]
    hist_name2tstamp = historical_names[["name", "tstamp", "vid"]].drop_duplicates()

    stash = q.get_stash()
    assert stash is not None
    assert State.repo is not None
    copyfile(fp, stash / fp)

    for _, row in hist_name2tstamp.iterrows():
        if len(State.hls_hits) == len(names):
            break
        n = row["name"]
        v = row["vid"]
        State.repo.git.checkout(v, "--", fp)
        lev = LoggedExpVisitor()
        with open(fp, "r") as f:
            lev.visit(ast.parse(f.read()))
        if n in lev.names:
            State.grouped_names[n] = lev.names[n]
            State.hls_hits.add(n)
            copyfile(src=fp, dst=stash / Path(n).with_suffix(".py"))

    copyfile(stash / fp, fp)

    for p in os.listdir(stash):
        State.hls_hits.add(".".join(p.split(".")[0:-1]))

    assert len(os.listdir(stash)) > len(
        names
    ), f"Failed to find log statement for vars {[n for n in names if n not in State.hls_hits]}"

    # Next, from the stash you will apply each file to our main one
    for name in names:
        with open(stash / Path(name).with_suffix(".py"), "r") as f:
            tree = ast.parse(f.read())

        if name in State.grouped_names:
            lineno = int(State.grouped_names[name])
        else:
            lev = LoggedExpVisitor()
            lev.visit(tree)
            lineno = int(lev.names[name])
        # lev possibly unbound
        backprop(
            lineno,
            str(stash / Path(name).with_suffix(".py")).replace("\x1b[m", ""),
            dst,
        )
        print(f"Applied {name} to {dst}")


__all__ = ["backprop", "apply"]

if __name__ == "__main__":
    backprop(78, "cases/train_rnn/now.py", "cases/train_rnn/before.py")
