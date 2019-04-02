from .log_stmt import LogStmt
from .. import generate as gen
import ast

class Loop(LogStmt):

    def __init__(self, node):
        super().__init__()
        self.node = node

    def to_string_start(self):
        d = "{{'start_loop': {}}}".format(self.node.lineno)
        return super().to_string(d)

    def to_string_end(self):
        d = "{{'end_loop': {}}}".format(self.node.lineno)
        return super().to_string(d)

    def parse_start(self):
        return ast.parse(self.to_string_start()).body[0]

    def parse_end(self):
        return ast.parse(self.to_string_end()).body[0]