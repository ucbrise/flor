from _ast import AST, Constant
from typing import Any, Dict, Optional
import ast

from .. import utils


class WithExpVisitor(ast.NodeVisitor):
    def __init__(self):
        super().__init__()
        self.found = False

    def visit_With(self, node: ast.With):
        """
        Checks for the existence of a with statement:
        with flor.checkpointing(...):
            ...
        """
        pred = (
            isinstance(node.items[0].context_expr, ast.Call)
            and isinstance(node.items[0].context_expr.func, ast.Attribute)
            and isinstance(node.items[0].context_expr.func.value, ast.Name)
            and node.items[0].context_expr.func.value.id == "flor"
            and node.items[0].context_expr.func.attr == "checkpointing"
        )
        if pred:
            self.found = True
        else:
            self.generic_visit(node)


class LoggedExpVisitor(ast.NodeVisitor):
    def __init__(self):
        super().__init__()
        self.names: Dict[str, int] = {}

        self.line2level: Dict[int, int] = {}
        self.lvl = 0

    def visit_For(self, node: ast.For):
        iter_s = ast.unparse(node.iter).strip()
        if iter_s.startswith("flor.loop"):
            start_lvl = self.lvl
            try:
                self.lvl += 1
                self.generic_visit(node)
            finally:
                self.lvl = start_lvl
        else:
            self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        pred = (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "flor"
            and node.func.attr == "log"
        )
        if not pred:
            return self.generic_visit(node)
        if len(node.args) == 2 and isinstance(node.args[0], ast.Constant):
            self.names[str(node.args[0].value)] = node.lineno
            self.line2level[node.lineno] = self.lvl
        else:
            raise IndexError("FLOR: Did you give flor.log a key? It takes 2 args.")

    def generic_visit(self, node: AST) -> Any:
        if hasattr(node, "lineno"):
            self.line2level[node.lineno] = self.lvl
        return super().generic_visit(node)


class NoGradVisitor(ast.NodeVisitor):
    def __init__(self):
        super().__init__()
        self.feeding = False
        self.names = {}
        self.tree = None

    def visit_Call(self, node: ast.Call):
        pred = (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "flor"
            and node.func.attr == "log"
        )
        if not pred or not self.feeding:
            return self.generic_visit(node)
        if len(node.args) == 2 and isinstance(node.args[0], ast.Constant):
            self.names[str(node.args[0].value)] = node.lineno
        else:
            raise IndexError("FLOR: Did you give flor.log a key? It takes 2 args.")

    def visit_With(self, node: ast.With):
        if [True for each in node.items if "torch.no_grad" in ast.unparse(each)]:
            try:
                feeding = self.feeding
                self.feeding = True
                self.tree = node
                for stmt in node.body:
                    self.visit(stmt)
            finally:
                self.feeding = feeding  # type: ignore


class NoGradTransformer(ast.NodeTransformer):
    def __init__(self, old_tree) -> None:
        super().__init__()
        self.their_tree = old_tree

    def visit_With(self, node: ast.With):
        if [True for each in node.items if "torch.no_grad" in ast.unparse(each)]:
            return self.generic_visit(self.their_tree)
        else:
            return self.generic_visit(node)
