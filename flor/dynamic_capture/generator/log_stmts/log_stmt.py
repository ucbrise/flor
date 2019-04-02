from .. import generate as gen
import ast

class LogStmt():

    def __init__(self):
        self.LOG_CMD = 'Flog.flagged() and flog.write'

    def __make_tuple__(self, t):
        if not t:
            return "{'[]': []}"
        make_pair = lambda x: "{{'{}': {}}}".format(
            gen.proc_lhs(x), gen.proc_rhs(x)) if not (
                isinstance(x, ast.Tuple) or isinstance(x, ast.List)) else self.__make_tuple__(x.elts)
        return ', '.join(map(make_pair, t))

    def __make_keywords__(self, k):
        make_pair = lambda x: "{{'{}': {}}}".format(x.arg, gen.proc_rhs(x.value))
        return ', '.join(map(make_pair, k))

    def to_string(self, child : str):
        return "{}({})".format(
            self.LOG_CMD,
            child
        )