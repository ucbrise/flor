from .log_stmt import LogStmt
from .. import util as gen
import ast

HEADER = """
from flor import Flog
if Flog.flagged(option='nofork'): flog = Flog()
"""

HEADER_SOP_FD = """
from flor import Flog
if Flog.flagged(option='nofork'): flog = Flog(False)
"""

FOOTER = """
Flog.flagged(option='nofork') and flog.writer.close()
"""

class FuncDef(LogStmt):

    def __init__(self, node: ast.FunctionDef, filepath, counter, classname=None):
        super().__init__()
        self.node = node
        self.filepath = filepath
        self.classname = classname
        self.arg_names = []
        self.counter = counter
        if node.args:
            for arg in node.args.args:
                self.arg_names.append("raw.{}".format(arg.arg))
            if node.args.vararg:
                self.arg_names.append("vararg.{}".format(node.args.vararg.arg))
            if node.args.kwarg:
                self.arg_names.append("kwarg.{}".format(node.args.kwarg.arg))

    def __get_params__(self):
        lsn = self.counter['value']
        self.counter['value'] += 1
        return super().to_string(("{" + "'lsn': {},".format(lsn) + "'params': "
                                  + str(list(map(lambda i, x: "{{'{}.{}': {}}}".format(i, gen.proc_lhs(x),
                                                                                       gen.proc_rhs(x.split('.')[1])),
                                                 range(len(self.arg_names)), self.arg_names)))
                                  + "}").replace('"', ''))

    def parse_heads(self):
        return ast.parse(self.to_string_head()).body

    def parse_foot(self):
        return ast.parse(self.to_string_foot()).body

    def to_string_head(self):
        if self.classname:
            lsn1 = self.counter['value']
            lsn2, lsn3 = lsn1 + 1, lsn1 + 2
            self.counter['value'] = lsn3 + 1
            return (HEADER + "\n"
                   + super().to_string("{{'file_path': '{}', 'lsn': {}}}".format(self.filepath, lsn1)) + "\n"
                    + super().to_string("{{'class_name': '{}', 'lsn': {}}}".format(self.classname, lsn2)) + "\n"
                    + super().to_string("{{'start_function': '{}', 'lsn':{}}}".format(self.node.name, lsn3)) + "\n"
                    + self.__get_params__() + "\n")
        else:
            lsn1 = self.counter['value']
            lsn2 = lsn1 + 1
            self.counter['value'] = lsn2 + 1
            return (HEADER + "\n"
                    + super().to_string("{{'file_path': '{}', 'lsn': {}}}".format(self.filepath, lsn1)) + "\n"
                    + super().to_string("{{'start_function': '{}', 'lsn': {}}}".format(self.node.name, lsn2)) + "\n"
                    + self.__get_params__() + "\n")

    def to_string_foot(self):
        lsn = self.counter['value']
        self.counter['value'] += 1
        return ( super().to_string("{{'end_function': '{}', 'lsn': {}}}".format(self.node.name, lsn)) + "\n"
                 + FOOTER + "\n")

class StackOverflowPreventing_FuncDef(LogStmt):

    def __init__(self, node: ast.FunctionDef):
        super().__init__()
        self.node = node
    
    def parse_heads(self):
        """
        Returns a list of statements
        """
        return ast.parse(self.to_string_head()).body

    def parse_foot(self):
        """
        Returns a single statement
        """
        return ast.parse(self.to_string_foot()).body

    def to_string_head(self):
        return (HEADER_SOP_FD + "\n"
                + 'Flog.flagged() and flog.block_recursive_serialization()\n')

    def to_string_foot(self):
        return ('Flog.flagged() and flog.unblock_recursive_serialization()\n'
                + FOOTER + "\n")