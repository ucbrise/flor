import ast
import astor

from flor.state_machine_generator.handler import Handler
from flor.log_scanner.scanner import Scanner

class Visitor(ast.NodeVisitor):

    #TODO: implement WITH statement

    def __init__(self, filepath, logpath):
        super().__init__()
        self.filepath = filepath
        self.class_ctx = None
        self.func_ctx = None
        self.func_name = None
        self.lsn = None
        self.pos = None
        self.kw = None

        self.scanner = Scanner(logpath)
        self.collecting_queue = []

    def visit_ClassDef(self, node):
        # Good for filling class CTX
        prev = self.class_ctx
        self.class_ctx = node.name
        self.generic_visit(node)
        self.class_ctx = prev

    def visit_FunctionDef(self, node):
        prev = self.func_ctx
        self.func_ctx = node.name
        self.generic_visit(node)
        self.func_ctx = prev

    def visit_Call(self, node):
        h = Handler(node)
        if h.is_highlight():
            name = h.fetch_highlight_name()
            # Need to choose a State Machine
            assert self.lsn is not None
            if self.func_name is not None:
                # ActualParam Case
                pos_kw = {}
                if self.kw is not None:
                    pos_kw['kw'] = self.kw
                else:
                    assert self.pos is not None
                    pos_kw['pos'] = self.pos

                self.collecting_queue.append(self.scanner.register_ActualParam(
                    name = name,
                    file_path = self.filepath,
                    class_ctx = self.class_ctx,
                    func_ctx = self.func_ctx,
                    prev_lsn = self.lsn,
                    func_name = self.func_name,
                    pos_kw = pos_kw
                ))
            else:
                #RootExpression Case
                self.collecting_queue.append(self.scanner.register_RootExpression(
                    name = name,
                    file_path = self.filepath,
                    class_ctx = self.class_ctx,
                    func_ctx = self.func_ctx,
                    prev_lsn = self.lsn,
                    tuple_idx = self.pos
                ))

            # Just in case there are nested highlights
            self.generic_visit(node.args[1])
        else:
            prev = self.func_name
            self.func_name = astor.to_source(node.func).strip().replace('\n', '').replace('\t', '')
            if '.' in self.func_name:
                self.func_name = self.func_name.split('.')[-1]
            self.generic_visit(node)
            self.func_name = prev

    def visit_keyword(self, node):
        prev = self.kw
        self.kw = node.arg
        self.generic_visit(node)
        self.kw = prev

    def visit_assign(self, node):
        self.pos = None
        self.generic_visit(node)
        self.pos = None

    def visit_Expr(self, node):
        """
        Normally, expressions are not logged (but if callees the callee is logged)
		Here, we should expect flog.write() invocations and print statements
		and clf.fits() and so on.
		EVERY flog.write is an expression.
		Here, we should test and send the AST to the appropriate extractor or handler
        """
        h = Handler(node)
        if h.is_flog_write():
            self.lsn = h.fetch_lsn()
            while self.collecting_queue:
                fsm = self.collecting_queue.pop(0)
                fsm.follow_lsn = self.lsn
            #TODO: We could be in prev or follow LSN. Need to respond slightly differently to each
        else:
            self.pos = None
            self.generic_visit(node)
            self.pos = None

    def generic_visit(self, node):
        """Called if no explicit visitor function exists for a node."""
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for i, item in enumerate(value):
                    self.pos = i
                    if isinstance(item, ast.AST):
                        self.visit(item)
                self.pos = None
            elif isinstance(value, ast.AST):
                self.visit(value)
