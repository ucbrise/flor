from .log_stmt import LogStmt
import ast
from .. import generate as gen

class BoolExp(LogStmt):

    def __init__(self, node):
        """
        :param node: node.test in a If Stmt
        """
        super().__init__()
        self.node = node

    def to_string(self, flag=True):
        if flag:
            return "{{'conditional_fork': '{}'}}".format(gen.proc_lhs(self.node))
        else:
            return "{{'conditional_fork': '{}'}}".format(gen.neg(gen.proc_lhs(self.node)))

    def parse(self, flag):
        return ast.parse(super().to_string(self.to_string(flag))).body[0]

    def __str__(self):
        return super().to_string(self.to_string())