from .log_stmt import LogStmt
from .. import generate as gen
import ast

HEADER = """
from flor import Flog
if Flog.flagged(): flog = Flog()
"""

class FuncDef(LogStmt):

    def __init__(self, node: ast.FunctionDef, filepath, classname=None):
        super().__init__()
        self.node = node
        self.filepath = filepath
        self.classname = classname
        self.arg_names = []
        if node.args:
            for arg in node.args.args:
                self.arg_names.append(arg.arg)
            if node.args.vararg:
                self.arg_names.append(node.args.vararg.arg)
            if node.args.kwarg:
                self.arg_names.append(node.args.kwarg.arg)

    def __get_params__(self):
        return super().to_string(("{" + "'params': "
                                  + str(list(map(lambda x: "{{'{}': {}}}".format(gen.proc_lhs(x), gen.proc_rhs(x)), self.arg_names)))
                                  + "}").replace('"', ''))

    def parse_heads(self):
        return ast.parse(self.to_string_head()).body

    def parse_foot(self):
        return ast.parse(self.to_string_foot()).body[0]

    def to_string_head(self):
        if self.classname:
            return (HEADER + "\n"
                   + super().to_string("{{'file_path': '{}'}}".format(self.filepath)) + "\n"
                    + super().to_string("{{'class_name': '{}'}}".format(self.classname)) + "\n"
                    + super().to_string("{{'start_function': '{}'}}".format(self.node.name)) + "\n"
                    + self.__get_params__() + "\n")
        else:
            return (HEADER + "\n"
                    + super().to_string("{{'file_path': '{}'}}".format(self.filepath)) + "\n"
                    + super().to_string("{{'start_function': '{}'}}".format(self.node.name)) + "\n"
                    + self.__get_params__() + "\n")

    def to_string_foot(self):
        return super().to_string("{{'end_function': '{}'}}".format(self.node.name))

