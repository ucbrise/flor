from .log_stmt import LogStmt
from .. import generate as gen
import ast

class Assign(LogStmt):

    def __init__(self, node: 'ast.Assign'):
        super().__init__()
        self.node = node

    def __make_locals__(self):
        if isinstance(self.node, ast.Assign):
            return self.__make_tuple__(self.node.targets)
        elif isinstance(self.node, ast.AugAssign):
            return self.__make_tuple__([self.node.target])
        else:
            raise RuntimeError()

    def to_string(self):
        return "{{'locals': [{}], 'lineage': '{}'}}".format(
            self.__make_locals__(),
            gen.proc_lhs(self.node)
        )

    def parse(self):
        return ast.parse(str(self)).body[0]

    def __str__(self):
        return super().to_string(self.to_string())