from _ast import AST, Constant
from typing import Any, Dict, Optional
import ast

from .. import utils


class WithExpVisitor(ast.NodeVisitor):
    """
    Signals to the replay planner that the script has a checkpointable scope.

    Pre-v4: the only signal was `with flor.checkpointing(...):`.
    v4: any `flor.loop` is also a checkpointable scope (piggy-backed via
    torch.save / torch.load hooks), so its presence is sufficient.
    """

    def __init__(self):
        super().__init__()
        self.found = False

    def visit_With(self, node: ast.With):
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

    def visit_For(self, node: ast.For):
        iter_s = ast.unparse(node.iter).strip()
        if iter_s.startswith("flor.loop"):
            self.found = True
        self.generic_visit(node)


class LoggedExpVisitor(ast.NodeVisitor):
    def __init__(self):
        super().__init__()
        self.names: Dict[str, int] = {}

        self.line2level: Dict[int, int] = {}
        self.lvl = 0
        # loop_names[i] is the name of the flor.loop at nesting depth (i+1)
        # in the order they appear on the dominant nesting path. So for
        #   for epoch in flor.loop("epoch", ...):
        #       for step in flor.loop("step", ...):
        #           ...
        # we get loop_names == ["epoch", "step"].
        self.loop_names: list[str] = []

    def visit_For(self, node: ast.For):
        iter_s = ast.unparse(node.iter).strip()
        if iter_s.startswith("flor.loop"):
            start_lvl = self.lvl
            loop_name: Optional[str] = None
            try:
                call = node.iter
                if isinstance(call, ast.Call) and call.args:
                    first = call.args[0]
                    if isinstance(first, ast.Constant) and isinstance(
                        first.value, str
                    ):
                        loop_name = first.value
            except Exception:
                pass
            try:
                self.lvl += 1
                # Record the loop name at this depth on first encounter.
                if loop_name is not None and len(self.loop_names) < self.lvl:
                    self.loop_names.append(loop_name)
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
