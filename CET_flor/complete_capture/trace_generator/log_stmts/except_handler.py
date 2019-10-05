from .log_stmt import LogStmt
from .. import util as gen
import ast

HEADER = """
from inspect import stack as stackkcats
"""

class ExceptHandler(LogStmt):

    def __init__(self, counter):
        super().__init__()
        self.counter = counter

    def parse(self):
        return ast.parse(str(self)).body

    def to_string(self):
        lsn  = self.counter['value']
        self.counter['value'] += 1
        return "{{'lsn': {}, 'catch_stack_frame': [str(each.function) for each in stackkcats()]}}".format(lsn)

    def __str__(self):
        return HEADER + "\n" + super().to_string(self.to_string())