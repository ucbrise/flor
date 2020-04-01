import ast


class StatementCounter(ast.NodeVisitor):

    def __init__(self):
        self.count = 0

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Attribute):
                self.count += 1

    def visit_Assign(self, node):
        self.count += 1