from .log_stmt import LogStmt
from .. import util as gen
import ast

HEADER = """
from flor import Flog
"""


class LibRoot(LogStmt):

    def __init__(self, filepath, counter):
        super().__init__()
        self.filepath = filepath
        self.counter = counter

    def parse_heads(self):
        return ast.parse(self.to_string_head()).body

    def to_string_head(self):
        lsn = self.counter['value']
        self.counter['value'] += 1
        return HEADER

