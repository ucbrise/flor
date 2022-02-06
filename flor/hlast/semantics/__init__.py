import ast
from typing import Any


class StepLoggingVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.value = None
        self.value_loaded = False

    def visit_If(self, node: ast.If):
        self.generic_visit(node)

        if "flor.SkipBlock.step_into" in ast.unparse(node.test):
            stmts = [stmt for stmt in node.body if isinstance(stmt, ast.For)]
            assert len(stmts) == 1
            stmt = stmts.pop()
            self.value = stmt.body[-1].end_lineno
            self.value_loaded = True


class EpochLoggingVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.value = None
        self.value_loaded = False

    def visit_While(self, node: ast.While):
        self.generic_visit(node)
        if "flor.it(" in ast.unparse(node.test):
            self.value = node.body[-1].end_lineno
            self.value_loaded = True

    def visit_For(self, node: ast.For):
        self.generic_visit(node)
        if "flor.it(" in ast.unparse(node.iter):
            self.value = node.body[-1].end_lineno
            self.value_loaded = True


def in_logging_hotzone(lineno: int, content: str):
    """
    `Content` is an opened and read python file.
    """
    slv = StepLoggingVisitor()
    slv.visit(ast.parse(content))
    assert slv.value_loaded and slv.value is not None
    elv = EpochLoggingVisitor()
    elv.visit(ast.parse(content))
    assert elv.value_loaded and elv.value is not None
    if int(lineno) == int(slv.value):
        return ("step-level", True)
    elif int(lineno) == int(elv.value):
        return ("epoch-level", True)
    return ("garbage", False)


__all__ = ["in_logging_hotzone"]

if __name__ == "__main__":
    with open("cases/train_rnn/now.py", "r") as f:
        content = f.read()
        for i in range(1, 1000):
            if in_logging_hotzone(i, content):
                print(i)
