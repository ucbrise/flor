from .log_stmt import LogStmt
import ast
from .. import util as gen

class BoolExp(LogStmt):

    def __init__(self, node, counter):
        """
        :param node: node.test in a If Stmt
        """
        super().__init__()
        self.node = node
        self.counter = counter

    def to_string(self, flag=True):
        lsn = self.counter['value']
        self.counter['value'] += 1
        if flag:
            return "{{'conditional_fork': '{}', 'lsn': {}}}".format(gen.proc_lhs(self.node), lsn)
        else:
            return "{{'conditional_fork': '{}', 'lsn': {}}}".format(gen.neg(gen.proc_lhs(self.node)), lsn)

    def parse(self, flag):
        return ast.parse(super().to_string(self.to_string(flag))).body[0]

    def __str__(self):
        return super().to_string(self.to_string())