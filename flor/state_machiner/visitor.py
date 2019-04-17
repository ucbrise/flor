import ast
import astor

from .flog_write_handler.handler import Handler
from flor.log_scanner.scanner import Scanner

class Ctx:
    def __init__(self):
        self.class_ctx = None
        self.func_ctx = None
        self.lsn = None

        self.func_name = None
        self.pos_kw = None

        self.parent_ctx = None

    def set_lsn(self, class_ctx, func_ctx, lsn):
        self.class_ctx = class_ctx
        self.func_ctx = func_ctx
        self.lsn = lsn

    def set_parent_func_name(self, func_name):
        # This is the func_name for prev_lsn, so parent with defined LSN must exist
        assert self.lsn is None and self.parent_ctx is not None
        assert self.parent_ctx.lsn is not None and self.parent_ctx.func_name is None
        self.parent_ctx.func_name = func_name

    def set_parent(self, other):
        self.parent_ctx = other
        # self.class_ctx = other.class_ctx
        # self.func_ctx = other.func_ctx
        # self.func_name = other.func_name
        # self.pos_kw = other.pos_kw
        # self.parent_ctx = other

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

        # self.current_ctx = Ctx()

        self.scanner = Scanner(logpath)
        self.collecting_queue = []

        # Do I want to store state about...
        # self.top_level = ast.STMT
        # self.parent = ast.AST

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
        self.class_ctx = prev

    def visit_Call(self, node):
        h = Handler(node)
        if h.is_highlight():
            name = h.fetch_highlight_name()
            # Need to choose a State Machine
            try:
                assert self.lsn is not None
            except:
                print('hold up')
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



class OtherVisitor(ast.NodeVisitor):

    def __init__(self, in_execution, in_file):
        super().__init__()
        self.__in_execution__ = in_execution
        self.__in_file__ = in_file

        self.__structs__: List[Struct] = []
        self.__struct_index__ = []
        self.__struct_map__ = {}

        self.__assign_line_no__ = -1
        self.__expr_line_no__ = -1
        self.__val__ = None
        self.__pruned_names__ = []
        self.__keyword_name__ = None
        self.__call_stack__ = []
        self.__pos_arg_stack__ = []

    def consolidate_structs(self):
        if self.__struct_map__:
            # for idempotency
            return
        new = []
        for idx, struct in enumerate(self.__structs__):
            distinct = True
            match = None
            dest_idx = idx
            for prev_idx, prev_struct in enumerate(new):
                if (struct.instruction_no == prev_struct.instruction_no
                        and struct.typ == prev_struct.typ
                        and struct.value == prev_struct.value
                        and struct.keyword_name == prev_struct.keyword_name
                        and struct.caller == prev_struct.caller
                        and struct.pos == prev_struct.pos):
                    distinct = False
                    match = prev_struct
                    dest_idx = prev_idx
                    break
            if distinct:
                new.append(struct)
            else:
                if type(match.assignee) == list:
                    match.assignee.append(struct.assignee)
                else:
                    match.assignee = [match.assignee, struct.assignee]
            self.__struct_map__[idx] = dest_idx
        self.__structs__ = new

    def visit_Attribute(self, node):
        if type(node.ctx) == ast.Store:
            return astunparse.unparse(node).strip()
        elif type(node.ctx) == ast.Load:
            value = self.visit(node.value)
            attr = node.attr
            if value == 'flor':
                if attr == 'log':
                    return 'flor.log'
            elif value == 'log' or value == 'flor.log':

                caller = pos = None
                if self.__call_stack__[0:-1]:
                    [*_, caller] = self.__call_stack__[0:-1]
                if self.__pos_arg_stack__:
                    [*_, pos] = self.__pos_arg_stack__

                if self.__assign_line_no__ >= 0:
                    # ASSIGN CONTEXT
                    # assert self.__pruned_names__, "Static Analyzer: Failed to retrieve name of assignee variable"
                    if not self.__pruned_names__:
                        self.__pruned_names__ = None

                    if attr == 'read':
                        self.__structs__.append(Struct(assignee=self.__pruned_names__,
                                                       value=astunparse.unparse(self.__val__).strip(),
                                                       typ='read',
                                                       instruction_no=self.__assign_line_no__,
                                                       keyword_name=self.__keyword_name__,
                                                       caller=caller, pos=pos, in_execution=self.__in_execution__,
                                                       in_file=self.__in_file__))
                        self.__struct_index__.append(len(self.__struct_index__))
                    elif attr == 'write':
                        self.__structs__.append(Struct(assignee=self.__pruned_names__,
                                                       value=astunparse.unparse(self.__val__).strip(),
                                                       typ='write',
                                                       instruction_no=self.__assign_line_no__,
                                                       keyword_name=self.__keyword_name__,
                                                       caller=caller, pos=pos, in_execution=self.__in_execution__,
                                                       in_file=self.__in_file__))
                        self.__struct_index__.append(len(self.__struct_index__))
                    elif attr == 'param':
                        self.__structs__.append(Struct(assignee=self.__pruned_names__,
                                                       value=astunparse.unparse(self.__val__).strip(),
                                                       typ='param',
                                                       instruction_no=self.__assign_line_no__,
                                                       keyword_name=self.__keyword_name__,
                                                       caller=caller, pos=pos, in_execution=self.__in_execution__,
                                                       in_file=self.__in_file__))
                        self.__struct_index__.append(len(self.__struct_index__))
                    elif attr == 'metric':
                        self.__structs__.append(Struct(assignee=self.__pruned_names__,
                                                       value=astunparse.unparse(self.__val__).strip(),
                                                       typ='metric',
                                                       instruction_no=self.__assign_line_no__,
                                                       keyword_name=self.__keyword_name__,
                                                       caller=caller, pos=pos, in_execution=self.__in_execution__,
                                                       in_file=self.__in_file__))
                        self.__struct_index__.append(len(self.__struct_index__))
                else:
                    # EXPR CONTEXT
                    if attr == 'read':
                        self.__structs__.append(Struct(value=astunparse.unparse(self.__val__).strip(),
                                                       typ='read',
                                                       instruction_no=self.__expr_line_no__,
                                                       keyword_name=self.__keyword_name__,
                                                       caller=caller, pos=pos, in_execution=self.__in_execution__,
                                                       in_file=self.__in_file__))
                        self.__struct_index__.append(len(self.__struct_index__))
                    elif attr == 'write':
                        self.__structs__.append(Struct(value=astunparse.unparse(self.__val__).strip(),
                                                       typ='write',
                                                       instruction_no=self.__expr_line_no__,
                                                       keyword_name=self.__keyword_name__,
                                                       caller=caller, pos=pos, in_execution=self.__in_execution__,
                                                       in_file=self.__in_file__))
                        self.__struct_index__.append(len(self.__struct_index__))
                    elif attr == 'param':
                        self.__structs__.append(Struct(value=astunparse.unparse(self.__val__).strip(),
                                                       typ='param',
                                                       instruction_no=self.__expr_line_no__,
                                                       keyword_name=self.__keyword_name__,
                                                       caller=caller, pos=pos, in_execution=self.__in_execution__,
                                                       in_file=self.__in_file__))
                    elif attr == 'metric':
                        self.__structs__.append(Struct(value=astunparse.unparse(self.__val__).strip(),
                                                       typ='metric',
                                                       instruction_no=self.__expr_line_no__,
                                                       keyword_name=self.__keyword_name__,
                                                       caller=caller, pos=pos, in_execution=self.__in_execution__,
                                                       in_file=self.__in_file__))
                        self.__struct_index__.append(len(self.__struct_index__))

            return "{}.{}".format(value, attr)



    def visit_withitem(self, node):
        if node.optional_vars is None:
            self.__pruned_names__ = None
        else:
            self.__pruned_names__ = self.visit(node.optional_vars)
        self.visit(node.context_expr)
        self.__pruned_names__ = []


    def visit_keyword(self, node):
        self.__keyword_name__ = node.arg
        self.visit(node.value)
        self.__keyword_name__ = None

    def visit_With(self, node):
        self.__assign_line_no__ = node.lineno
        for item in node.items:
            self.visit(item)
        self.__assign_line_no__ = -1
        for each in node.body:
            self.visit(each)


    def visit_Expr(self, node):
        self.__call_stack__ = []
        self.__pos_arg_stack__ = []

        self.__expr_line_no__ = node.lineno
        self.visit(node.value)
        self.__expr_line_no__ = -1

        self.__call_stack__ = []
        self.__pos_arg_stack__ = []

    def visit_Return(self, node):
        self.visit_Expr(node)

    def visit_Call(self, node):
        if len(node.args) > 0:
            self.__val__ = node.args[0]
        self.__call_stack__.append(astunparse.unparse(node.func).strip())
        self.visit(node.func)
        for i, arg in enumerate(node.args):
            self.__pos_arg_stack__.append(i)
            self.visit(arg)
            self.__pos_arg_stack__.pop()
        for kwd in node.keywords:
            self.visit(kwd)
        self.__val__ = None
        self.__call_stack__.pop()

    def visit_Subscript(self, node):
        self.visit(node.slice)
        return astunparse.unparse(node).strip()

    def visit_Starred(self, node):
        return astunparse.unparse(node).strip()

    def visit_Name(self, node):
        return astunparse.unparse(node).strip()

    def visit_List(self, node):
        if type(node.ctx) == ast.Store:
            res = []
            for each in node.elts:
                res.append(self.visit(each))
            return res
        elif type(node.ctx) == ast.Load:
            for idx, each in enumerate(node.elts):
                self.visit(each)
        else:
            raise TypeError("Invalid context")

    def visit_Tuple(self, node):
        if type(node.ctx) == ast.Store:
            res = []
            for each in node.elts:
                res.append(self.visit(each))
            return tuple(res)
        elif type(node.ctx) == ast.Load:
            for idx, each in enumerate(node.elts):
                self.visit(each)
        else:
            raise TypeError("Invalid context")

    def visit_Assign(self, node):
        """
        Assign(targets, value)
        https://docs.python.org/3/library/ast.html
        Target is a special kind of expr
        value is an expr
        :param node:
        :return:

        RULES: Cannot get a name from a Starred.
        """
        assert len(node.targets) >= 1

        self.__call_stack__ = []
        self.__pos_arg_stack__ = []

        self.__assign_line_no__ = node.lineno
        for target in node.targets:
            self.__pruned_names__ = self.visit(target)
            self.visit(node.value)

        self.__pruned_names__ = []
        self.__assign_line_no__ = -1

        self.__call_stack__ = []
        self.__pos_arg_stack__ = []