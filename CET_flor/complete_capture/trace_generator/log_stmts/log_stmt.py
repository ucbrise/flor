from .. import util as gen
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

    @staticmethod
    def is_instance(node : ast.Expr):
        """
        NOTE this method should be modified on change to LOG STMT FORMAT
        Identifies instances of Flor LOG Statements
        :param node:
        :return:
        """
        if not isinstance(node, ast.Expr):
            return False
        boolop = node.value
        if not isinstance(boolop, ast.BoolOp):
            return False
        if not isinstance(boolop.op, ast.And):
            return False
        values = boolop.values
        if len(values) != 2:
            return False
        flog_flagged, flog_write = values
        if not isinstance(flog_flagged, ast.Call) or not isinstance(flog_write, ast.Call):
            return False
        flog_flagged_func, flog_write_func = flog_flagged.func, flog_write.func
        if not isinstance(flog_flagged_func, ast.Attribute):
            return False
        if not isinstance(flog_write_func, ast.Attribute):
            return False
        if not isinstance(flog_flagged_func.value, ast.Name) or flog_flagged_func.value.id != 'Flog':
            return False
        if flog_flagged_func.attr != 'flagged':
            return False
        if not isinstance(flog_write_func.value, ast.Name) or flog_write_func.value.id != 'flog':
            return False
        if flog_write_func.attr != 'write':
            return False
        return True
