from .log_stmt import LogStmt
from .. import util as gen
import ast

HEADER = """
from flor import Flog
if Flog.flagged(option='nofork'): flog = Flog(False)
"""

FOOTER = """
Flog.flagged(option='nofork') and flog.writer.close()
"""

class ClientRoot(LogStmt):

    def __init__(self, filepath, counter):
        super().__init__()
        self.filepath = filepath
        self.counter = counter

    def parse_heads(self):
        return ast.parse(self.to_string_head()).body

    def parse_foot(self):
        return ast.parse(self.to_string_foot()).body

    def to_string_head(self):
        lsn = self.counter['value']
        self.counter['value'] += 1
        return (HEADER + "\n" + super().to_string("{{'file_path': '{}', 'lsn': {}}}".format(self.filepath, lsn))
                + "\n")

    def to_string_foot(self):
        return FOOTER + "\n"

