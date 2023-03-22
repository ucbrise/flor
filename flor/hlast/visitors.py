from typing import Dict
import ast


class LoggedExpVisitor(ast.NodeVisitor):
    def __init__(self):
        super().__init__()
        self.names: Dict[str, int] = {}

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
        else:
            raise IndexError("FLOR: Did you give flor.log a key? It takes 2 args.")


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
    def __init__(self, their_tree) -> None:
        super().__init__()
        self.their_tree = their_tree

    def visit_With(self, node: ast.With):
        if [True for each in node.items if "torch.no_grad" in ast.unparse(each)]:
            return self.generic_visit(self.their_tree)
        else:
            return self.generic_visit(node)
    