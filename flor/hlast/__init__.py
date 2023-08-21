import ast
from argparse import Namespace
from shutil import copyfile
from pathlib import Path
from sys import stdout
from typing import Dict, List, Set
import os

from flor.hlast.gtpropagate import propagate  # type: ignore
from .visitors import NoGradVisitor, NoGradTransformer, LoggedExpVisitor


def backprop(lineno: int, source, target, out=None):
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


__all__ = ["backprop"]

if __name__ == "__main__":
    backprop(78, "cases/train_rnn/now.py", "cases/train_rnn/before.py")
