from os import PathLike
from sys import stdout
from .gtpropagate import propagate
from .semantics import in_logging_hotzone
from argparse import Namespace


def backprop(lineno: int, source: PathLike, target: PathLike, out=None):
    with open(source, "r") as src:
        content = src.read()
    if in_logging_hotzone(lineno, content):
        print("SEMANTIC PROP")
        semantic_prop(lineno, source, target, out)
    else:
        print("SYNTACTIC PROP")
        syntactic_prop(lineno, source, target, out)


def semantic_prop(lineno: int, source: PathLike, target: PathLike, out=None):
    ...


def syntactic_prop(lineno: int, source: PathLike, target: PathLike, out=None):
    if out is None:
        with open(str(source), "r") as src, open(str(target), "r") as dst:
            return propagate(
                Namespace(
                    lineno=int(lineno),
                    source=src,
                    target=dst,
                    out=stdout,
                    gumtree=dict(),
                )
            )
    else:
        with open(str(source), "r") as src, open(str(target), "r") as dst, open(
            out, "r"
        ) as f:
            return propagate(
                Namespace(
                    lineno=int(lineno), source=src, target=dst, out=f, gumtree=dict()
                )
            )
