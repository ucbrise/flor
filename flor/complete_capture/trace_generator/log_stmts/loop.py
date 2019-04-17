from .log_stmt import LogStmt
from .. import util as gen
import ast

class Loop(LogStmt):

    def __init__(self, node, counter):
        super().__init__()
        self.node = node
        self.counter = counter

    def to_string_start(self):
        lsn = self.counter['value']
        self.counter['value'] += 1
        d = "{{'start_loop': {}, 'lsn': {}}}".format(self.node.lineno, lsn)
        return super().to_string(d)

    def to_string_end(self):
        lsn = self.counter['value']
        self.counter['value'] += 1
        d = "{{'end_loop': {}, 'lsn': {}}}".format(self.node.lineno, lsn)
        return super().to_string(d)

    def parse_start(self):
        return ast.parse(self.to_string_start()).body[0]

    def parse_end(self):
        return ast.parse(self.to_string_end()).body[0]