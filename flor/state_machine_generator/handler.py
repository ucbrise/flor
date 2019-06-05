import ast

from flor.complete_capture.trace_generator.log_stmts.log_stmt import LogStmt

class Handler:

    def __init__(self, node : ast.AST):
        self.node = node

    def is_flog_write(self):
        return LogStmt.is_instance(self.node)

    def is_highlight(self):
        #TODO: This is where we will want to be able to read a broad set of HIGHLIGHTS
        """
        For now, we are hardcoding the simple highlight:
            GET(name, desired_value)
        """
        if not isinstance(self.node, ast.Call):
            return False
        func = self.node.func
        args = self.node.args
        keywords = self.node.keywords
        if not isinstance(func, ast.Name) or func.id != 'GET':
            return False
        if len(args) != 2:
            return False
        name, desired_value = args
        if not isinstance(name, ast.Str):
            return False
        if keywords:
            return False
        return True

    def fetch_highlight_name(self):
        assert self.is_highlight()
        name, _ = self.node.args
        return name.s

    def fetch_lsn(self):
        assert LogStmt.is_instance(self.node)
        dc = self.node.value.values[1].args[0]
        for i, astree in enumerate(dc.keys):
            if isinstance(astree, ast.Str) and astree.s == 'lsn':
                return dc.values[i].n
        raise RuntimeError()