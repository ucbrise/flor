from .log_stmt import LogStmt
from .. import generate as gen
import ast

HEADER = """
from flor import Flog
if Flog.flagged(): flog = Flog(False)
"""

class ClientRoot(LogStmt):

    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath

    def parse_heads(self):
        return ast.parse(self.to_string_head()).body

    def to_string_head(self):
        return (HEADER + "\n" + super().to_string("{{'file_path': '{}'}}".format(self.filepath)) + "\n")

