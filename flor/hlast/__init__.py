import ast
from argparse import Namespace
from os import PathLike
from shutil import copy2
from pathlib import Path, PurePath
from sys import stdout
from typing import List

from flor.hlast.gtpropagate import propagate, LogLinesVisitor  # type: ignore
from flor.state import State
import flor.query as q

_LVL = None


def backprop(lineno: int, source: str, target: str, out=None):
    global _LVL
    with open(source, "r") as src:
        content = src.read()
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
    fp = Path(dst)
    facts = q.log_records() if q.facts is None else q.facts
    # Get latest timestamp for each variable name
    name2tstamp = (
        facts[facts["name"].isin(names)][["name", "tstamp"]]
        .groupby(by=["name"])
        .max()
        .reset_index()
    )
    name2vid = {
        row["name"]: row["vid"]
        for _, row in facts.merge(name2tstamp, how="inner")[["name", "vid"]]
        .drop_duplicates()
        .iterrows()
    }
    stash = q.clear_stash()
    assert stash is not None
    assert State.repo is not None
    for n, v in name2vid.items():
        State.repo.git.checkout(v, "--", dst)
        copy2(dst, stash / PurePath(n).with_suffix(".py"))
    State.repo.git.checkout(State.active_branch)
    print("wrote stash")


__all__ = ["backprop", "apply"]

if __name__ == "__main__":
    backprop(78, "cases/train_rnn/now.py", "cases/train_rnn/before.py")
