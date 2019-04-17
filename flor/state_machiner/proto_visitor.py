import ast, astor

class Ctx:
    def __init__(self):
        self.class_ctx = None
        self.func_ctx = None
        self.lsn = None
        self.func_name = None
        self.pos_kw = None

        self.parent_ctx = None

    def set_parent(self, other):
        self.class_ctx = other.class_ctx
        self.func_ctx = other.func_ctx
        self.func_name = other.func_name
        self.pos_kw = other.pos_kw
        self.parent_ctx = other

class HighlightVisitor(ast.NodeVisitor):

    def __init__(self, filepath=''):
        super().__init__()
        self.filepath = filepath
        self.classname = None
        self.fd = None #FuncDec

        self.current_ctx = Ctx()

        self.collecting_queue = [] # Used for followLSN

    def visit_keyword(self, node):
        pass

    def visit_Call(self, node):
        skip = False
        if isinstance(node.func, ast.Attribute):
            attr = node.func
            if isinstance(attr.value, ast.Name) and attr.value.id == 'flog' and attr.attr == 'write':
                skip = True
                arg = node.args[0]
                assert isinstance(arg, ast.Dict)
                for i, v in arg.keys:
                    if v.s == 'lsn':
                        lsn = arg.values[i].n
                        self.current_ctx.lsn = lsn
                        next_ctx = Ctx()
                        next_ctx.set_parent(self.current_ctx)
                        self.current_ctx = next_ctx
                        break
                if self.collecting_queue:
                    pass #TODO

            elif isinstance(attr.value, ast.Name) and attr.value.id == 'GET':
                skip = True
        if not skip:
            #ordinary function call, could be collecting ActualParams
            # TODO: could be attribute or name (or something else)
            # TODO: attribute if it is method... name if ordinary function
            # let's just get the string and process it.
            func_name = astor.to_source(node.func).strip().replace('\n', '').replace('\t', '')
            if '.' in func_name:
                func_name = func_name.split('.')[-1]
            self.current_ctx.func_name = func_name
        self.generic_visit(node)





    def generic_visit(self, node):
        """Called if no explicit visitor function exists for a node."""
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        self.visit(item)
            elif isinstance(value, ast.AST):
                self.visit(value)




