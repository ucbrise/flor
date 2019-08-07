import ast
import astor
from flor.complete_capture.trace_generator import *
from flor.utils import write_debug_msg
from copy import deepcopy

class ClientTransformer(ast.NodeTransformer):
    # TODO: Implement YIELD for Client

    def __init__(self, filepath=''):
        super().__init__()
        self.filepath = filepath
        self.classname = None
        self.fd = None  # FuncDef

        self.client_header_license = True
        self.relative_counter = {'value': 0}

    def visit_ClassDef(self, node):
        prev_class_name = self.classname
        self.classname = node.name
        new_node = self.generic_visit(node)
        self.classname = prev_class_name
        return new_node

    def visit_FunctionDef(self, node):
        if '__' != node.name[0:2] or node.name == '__init__':
            # ONLY WRAP PUBLIC METHODS TO AVOID STACK OVERFLOW
            prev = self.fd
            relative_counter = self.relative_counter['value']
            self.relative_counter['value'] = 0
            self.fd = FuncDef(node, self.filepath,
                              self.relative_counter, self.classname)
            heads = self.fd.parse_heads()
            foot = self.fd.parse_foot()
            new_node = self.generic_visit(node)
            heads.extend(new_node.body)
            new_node.body = heads
            if isinstance(new_node.body[-1], ast.Pass):
                new_node.body.pop()
            new_node.body.append(foot)
            self.fd = prev
            self.relative_counter['value'] = relative_counter
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
        # nodes_module.body.insert(-1, self.visit(self.fd.parse_foot()))
        if len(nodes_module.body) <= 1:
            return nodes_module.body
        ret_stmt = nodes_module.body.pop()
        nodes_module = self.generic_visit(nodes_module)
        nodes_module.body.append(ret_stmt)
        return nodes_module.body

    def visit_ExceptHandler(self, node):
        body = ExceptHandler(self.relative_counter).parse()
        new_node = self.generic_visit(node)
        body.extend(new_node.body)
        new_node.body = body
        return new_node

    def generic_visit(self, node):
        for field, old_value in ast.iter_fields(node):
            if isinstance(old_value, list):
                if self.client_header_license:
                    new_values = ClientRoot(
                        self.filepath, self.relative_counter).parse_heads()
                    self.client_header_license = False
                else:
                    new_values = []
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
            elif isinstance(old_value, ast.AST):
                new_node = self.visit(old_value)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)
        return node


class LibTransformer(ast.NodeTransformer):

    def __init__(self, filepath=''):
        super().__init__()
        self.active = False
        self.refuse_transform = False
        self.filepath = filepath
        self.classname = None
        self.fd = None

        # relative_counter: used to uniquely identfy log statements, relative to their context
        self.relative_counter = {'value': 0}

    def visit_ClassDef(self, node):
        prev_class_name = self.classname
        self.classname = node.name
        new_node = self.generic_visit(node)
        self.classname = prev_class_name
        return new_node

    def visit_FunctionDef(self, node):
        # TODO: Relative counter needs more work
        if '__' != node.name[0:2] or node.name == '__init__':
            # ONLY WRAP PUBLIC METHODS TO AVOID STACK OVERFLOW
            self.active = True
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
                self.active = False
                self.relative_counter['value'] = relative_counter
                self.refuse_transform = prev_refuse_transform
                write_debug_msg("Voluntarily refusing to transform function in file: {}".format(self.filepath))
                return save_node

            # Does function contain docstring?
            if ast.get_docstring(new_node) is not None:
                heads.insert(0, new_node.body[0])

            heads.extend(new_node.body)
            new_node.body = heads
            if isinstance(new_node.body[-1], ast.Pass):
                new_node.body.pop()
            new_node.body = [ast.Try(new_node.body, [], [], [foot])]

            # new_node.body.append(foot)
            self.fd = prev
            self.active = False
            self.relative_counter['value'] = relative_counter
            self.refuse_transform = prev_refuse_transform
            return new_node
        else:
            return node

    def visit_If(self, node):
        if self.active:
            node.body.insert(0, self.visit(
                BoolExp(node.test, self.relative_counter).parse(True)))
            node.orelse.insert(0, self.visit(
                BoolExp(node.test, self.relative_counter).parse(False)))
        return self.generic_visit(node)

    def visit_For(self, node):
        if self.active:
            loop = Loop(node, self.relative_counter)
            node.body.insert(0, self.visit(loop.parse_start()))
            node.body.append(self.visit(loop.parse_end()))
        return self.generic_visit(node)

    def visit_While(self, node):
        if self.active:
            loop = Loop(node, self.relative_counter)
            node.body.insert(0, self.visit(loop.parse_start()))
            node.body.append(self.visit(loop.parse_end()))
        return self.generic_visit(node)

    def visit_Return(self, node):
        nodes_module = Return(node).parse()
        # nodes_module.body.insert(-1, self.visit(self.fd.parse_foot()))
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
        if self.active:
            for field, old_value in ast.iter_fields(node):
                if isinstance(old_value, list):
                    new_values = []
                    for value in old_value:
                        if isinstance(value, ast.Raise):
                            r = self.visit(
                                Raise(value, self.relative_counter).parse())
                            new_values.append(r)
                        elif isinstance(value, ast.Return):
                            _v = value
                            assign_type = isinstance(value, ast.Assign)
                            if not isinstance(value, ast.Return):
                                value = value.value
                            values = self.visit(value)
                            assert values and isinstance(values, list), "BAD VALUES: {}".format(astor.to_source(values))
                            if assign_type:
                                assignee = values.pop()
                            new_values.extend(values)
                            if assign_type:
                                _v.value = assignee.value
                                new_values.append(_v)
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
                elif isinstance(old_value, ast.AST):
                    new_node = self.visit(old_value)
                    if new_node is None:
                        delattr(node, field)
                    else:
                        setattr(node, field, new_node)
            return node
        else:
            return super().generic_visit(node)
