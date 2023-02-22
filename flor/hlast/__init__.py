import ast
from argparse import Namespace
from os import PathLike
from sys import stdout

from flor.hlast.gtpropagate import propagate, LogLinesVisitor

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


__all__ = ["backprop"]

if __name__ == "__main__":
    backprop(78, "cases/train_rnn/now.py", "cases/train_rnn/before.py")
