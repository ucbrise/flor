from .log_stmt import LogStmt
from .. import util as gen
import ast

class Raise(LogStmt):

    def __init__(self, node: ast.Raise, counter):
        super().__init__()
        self.node = node
        self.counter = counter

    def __make_msg__(self):
        if self.node.exc:
            ts = self.__make_tuple__(self.node.exc.args)
            ks = self.__make_keywords__(self.node.exc.keywords)
            if ts and ks:
                return ts + ', ' + ks
            elif ts:
                return ts
            else:
                return ks
        return ''

    def parse(self):
        return ast.parse(str(self)).body[0]

    def to_string(self):
        lsn = self.counter['value']
        self.counter['value'] += 1
        if self.node.exc and isinstance(self.node.exc, ast.Call):
            return "{{'lsn': {}, 'exception_condition': {{'{}': [{}]}}}}".format(
                lsn,
                gen.proc_lhs(self.node.exc.func),
                self.__make_msg__()
            )
        else:
            return "{{'exception_raised': None, 'lsn': {}}}".format(lsn)

    def __str__(self):
        return super().to_string(self.to_string())