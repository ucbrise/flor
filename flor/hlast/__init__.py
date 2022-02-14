import ast
from argparse import Namespace
from os import PathLike
from sys import stdout

from flor.hlast.gtpropagate import propagate, LogLinesVisitor
from flor.hlast.semantics import in_logging_hotzone

_LVL = None


def backprop(lineno: int, source: str, target: str, out=None):
    global _LVL
    with open(source, "r") as src:
        content = src.read()
    _LVL, smt = (
        in_logging_hotzone(lineno, content) if lineno is not None else None,
        None,
    )
    if lineno is not None and smt:
        semantic_prop(lineno, source, target, out)
    else:
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


def semantic_prop(lineno: int, source, target, out=None):

    with open(source, "r") as src, open(target, "r") as dst:
        src_content = src.read()
        dst_content = dst.read()
    src_tree = ast.parse(src_content)
    dst_tree = ast.parse(dst_content)

    if lineno is None:
        llv = LogLinesVisitor()

    seeker = StmtToPropVisitor(lineno)
    seeker.visit(src_tree)
    assert seeker.value_valid and seeker.value is not None
    assert _LVL is not None
    new_dst_tree = semantic_injection(dst_tree, seeker.value, _LVL == "step-level")

    if out is None:
        print(ast.unparse(new_dst_tree))
        print("BY SEMANTIC")
    else:
        out.write(ast.unparse(new_dst_tree))
        out.close()


def semantic_injection(target_tree_mut: ast.AST, stmt: str, se: bool):
    if se:
        # Step-level logging
        return StepLevelInjection(stmt).visit(target_tree_mut)
    else:
        # Epoch-level logging
        return EpochLevelInjection(stmt).visit(target_tree_mut)


class StmtToPropVisitor(ast.NodeVisitor):
    def __init__(self, lineno) -> None:
        super().__init__()
        self.value = None
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


class EpochLevelInjection(ast.NodeTransformer):
    def __init__(self, payload):
        super().__init__()
        self.payload = payload

    def visit_While(self, node: ast.While):
        if "flor.it(" in ast.unparse(node.test):
            node.body.append(ast.parse(self.payload).body.pop())
        return self.generic_visit(node)

    def visit_For(self, node: ast.For):
        if "flor.it(" in ast.unparse(node.iter):
            node.body.append(ast.parse(self.payload).body.pop())
            self.value_loaded = True
        return self.generic_visit(node)


class StepLevelInjection(ast.NodeTransformer):
    def __init__(self, payload: str):
        self.payload = payload

    def visit_If(self, node: ast.If):
        if "flor.SkipBlock.step_into" in ast.unparse(node.test):
            stmts = [stmt for stmt in node.body if isinstance(stmt, ast.For)]
            assert len(stmts) == 1
            stmt = stmts.pop()
            stmt.body.append(ast.parse(self.payload).body.pop())
        return self.generic_visit(node)


__all__ = ["backprop"]

if __name__ == "__main__":
    backprop(78, "cases/train_rnn/now.py", "cases/train_rnn/before.py")
