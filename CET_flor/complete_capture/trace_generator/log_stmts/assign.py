from .log_stmt import LogStmt
from .. import util as gen
import ast

class Assign(LogStmt):

    def __init__(self, node: 'ast.Assign', counter):
        super().__init__()
        self.node = node
        self.counter = counter

    def __make_locals__(self):
        if isinstance(self.node, ast.Assign):
            return self.__make_tuple__(self.node.targets)
        elif isinstance(self.node, ast.AugAssign) or isinstance(self.node, ast.AnnAssign):
            return self.__make_tuple__([self.node.target])
        else:
            raise RuntimeError()

    def to_string(self):
        lsn = self.counter['value']
        self.counter['value'] += 1
        return "{{'locals': [{}], 'lineage': '{}', 'lsn': {}}}".format(
            self.__make_locals__(),
            gen.proc_lhs(self.node),
            lsn
        )

    def parse(self):
        return ast.parse(str(self)).body[0]

    def __str__(self):
        return super().to_string(self.to_string())