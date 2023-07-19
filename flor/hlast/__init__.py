import ast
from argparse import Namespace
from os import PathLike
from shutil import copyfile
from pathlib import Path, PurePath
from sys import stdout
from typing import Dict, List, Set
import pandas as pd
import os

from flor.hlast.gtpropagate import propagate, LogLinesVisitor  # type: ignore
from flor.state import State
import flor.query as q
from .visitors import LoggedExpVisitor, NoGradVisitor, NoGradTransformer


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


class StmtToPropVisitor(ast.NodeVisitor):
    def __init__(self, lineno) -> None:
        super().__init__()
        self.value = ""
        self.value_valid = False
        self.lineno = int(lineno)

    def generic_visit(self, node: ast.AST):
        if isinstance(node, ast.stmt):
            assert node.end_lineno is not None
            if int(node.lineno) == int(self.lineno):
                self.value = str(ast.unparse(node))
                self.value_valid = True
            else:
                super().generic_visit(node)
        else:
            super().generic_visit(node)


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
    parse_noGrad = []
    for name in names:
        with open(stash / Path(name).with_suffix(".py"), "r") as f:
            tree = ast.parse(f.read())

        ng_visitor = NoGradVisitor()
        ng_visitor.visit(tree)

        if name not in ng_visitor.names:
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
        else:
            parse_noGrad.append(ng_visitor.tree)
            print(f"Applied {name} to {dst}")

    if len(parse_noGrad) > 1:
        raise NotImplementedError(
            "TODO: Will need to merge code blocks, union of logging statements"
        )
    elif len(parse_noGrad) == 1:
        their_tree = parse_noGrad[0]
        with open(dst, "r") as f:
            my_tree = ast.parse(f.read())
        ng_transformer = NoGradTransformer(their_tree)

        with open(dst, "w") as f:
            f.write(ast.unparse(ng_transformer.visit(my_tree)))


__all__ = ["backprop", "apply"]

if __name__ == "__main__":
    backprop(78, "cases/train_rnn/now.py", "cases/train_rnn/before.py")
