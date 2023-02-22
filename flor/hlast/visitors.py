from typing import Optional, Set
import ast


class LoggedExpVisitor(ast.NodeVisitor):
    def __init__(self):
        super().__init__()
        self.names: Set[str] = set([])

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
            self.names.add(str(node.args[0].value))
        else:
            raise IndexError("FLOR: Did you give flor.log a key? It takes 2 args.")
