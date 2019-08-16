import ast, astor
from flor.complete_capture.trace_generator import *
from flor.utils import write_debug_msg
from .helper import Header
from copy import deepcopy


class Transformer(ast.NodeTransformer):

    def __init__(self, filepath=''):
        super().__init__()
        self.refuse_transform = False
        self.filepath = filepath
        self.classname = None
        self.fd = None

        # relative_counter: used to uniquely identify log statements, relative to their context
        self.relative_counter = {'value': 0}
        self.header_license = True

    def visit_ClassDef(self, node):
        prev_class_name = self.classname
        self.classname = node.name
        new_node = self.generic_visit(node)
        self.classname = prev_class_name
        return new_node

    def visit_FunctionDef(self, node):
        # Lambda test is a patch
        # PyTorch has some JIT functions that don't work if they're flor-transformed, we wish to skip those
        if all(map(lambda decorator: 'jit' not in astor.to_source(decorator), node.decorator_list)) \
            and '__' != node.name[0:2] or node.name == '__init__':
            # ONLY WRAP PUBLIC METHODS TO AVOID STACK OVERFLOW
            prev_refuse_transform = self.refuse_transform
            self.refuse_transform = False
            prev = self.fd
            relative_counter = self.relative_counter['value']
            self.relative_counter['value'] = 0
            self.fd = FuncDef(node, self.filepath,
                              self.relative_counter, self.classname)
            heads = self.fd.parse_heads()
            foot = self.fd.parse_foot()
            save_node = deepcopy(node)
            new_node = self.generic_visit(node)

            if self.refuse_transform:
                #Avoid transforming functions that contain yield/await/etc
                self.fd = prev
                self.relative_counter['value'] = relative_counter
                self.refuse_transform = prev_refuse_transform
                write_debug_msg("Voluntarily refusing to transform function in file: {}".format(self.filepath))
                return save_node

            # Does function contain docstring?
            contains_docstring = ast.get_docstring(new_node) is not None
            _docstring = new_node.body[0]

            heads.extend(new_node.body)
            new_node.body = heads
            if isinstance(new_node.body[-1], ast.Pass):
                new_node.body.pop()
            new_node.body = [ast.Try(new_node.body, [], [], foot)]

            if contains_docstring:
                new_node.body.insert(0, _docstring) 

            self.fd = prev
            self.relative_counter['value'] = relative_counter
            self.refuse_transform = prev_refuse_transform
            return new_node
        else:
            return node

    def visit_If(self, node):
        node.body.insert(0, self.visit(
            BoolExp(node.test, self.relative_counter).parse(True)))
        node.orelse.insert(0, self.visit(
            BoolExp(node.test, self.relative_counter).parse(False)))
        return self.generic_visit(node)

    def visit_For(self, node):
        loop = Loop(node, self.relative_counter)
        node.body.insert(0, self.visit(loop.parse_start()))
        node.body.append(self.visit(loop.parse_end()))
        return self.generic_visit(node)

    def visit_While(self, node):
        loop = Loop(node, self.relative_counter)
        node.body.insert(0, self.visit(loop.parse_start()))
        node.body.append(self.visit(loop.parse_end()))
        return self.generic_visit(node)

    def visit_Return(self, node):
        nodes_module = Return(node).parse()
        if len(nodes_module.body) <= 1:
            return nodes_module.body
        ret_stmt = nodes_module.body.pop()
        nodes_module = self.generic_visit(nodes_module)
        nodes_module.body.append(ret_stmt)
        return nodes_module.body

    def visit_Yield(self, node):
        self.refuse_transform = True
        return node

    def visit_Await(self, node):
        self.refuse_transform = True
        return node

    def visit_ExceptHandler(self, node):
        body = ExceptHandler(self.relative_counter).parse()
        new_node = self.generic_visit(node)
        body.extend(new_node.body)
        new_node.body = body
        return new_node

    def generic_visit(self, node):
        for field, old_value in ast.iter_fields(node):
            if isinstance(old_value, list):
                header_obj = None
                new_values = []
                if self.header_license:
                    header_obj = Header(ClientRoot(self.filepath, self.relative_counter))
                    if ast.get_docstring(node) is not None:
                        header_obj.docstring = node.body.pop(0)
                    self.header_license = False
                for value in old_value:
                    if isinstance(value, ast.Raise):
                        r = self.visit(
                            Raise(value, self.relative_counter).parse())
                        new_values.append(r)
                    elif isinstance(value, ast.Return):
                        values = self.visit(value)
                        assert values and isinstance(values, list)
                        new_values.extend(values)
                        continue
                    if isinstance(value, ast.AST):
                        value = self.visit(value)
                        if value is None:
                            continue
                        elif not isinstance(value, ast.AST):
                            new_values.extend(value)
                            continue
                    new_values.append(value)
                    if isinstance(value, ast.Assign) or \
                            isinstance(value, ast.AugAssign) or \
                            isinstance(value, ast.AnnAssign):
                        # OUTPUT ASSIGN STATEMENT
                        value = self.visit(
                            Assign(value, self.relative_counter).parse())
                        new_values.append(value)
                old_value[:] = new_values
                if header_obj is not None:
                    assert isinstance(node, ast.Module)
                    header_obj.proc_imports(node)
            elif isinstance(old_value, ast.AST):
                new_node = self.visit(old_value)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)
        return node

