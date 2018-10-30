import ast
import astor
import astunparse

from typing import List

class Struct:
    def __init__(self, assignee=None, value=None, typ=None,
                 instruction_no=None, keyword_name=None,
                 caller=None, pos=None):
        # TODO: Comment
        # ALERT: value is an AST node, it must be evaluated on transformer
        self.assignee = assignee
        self.value = value
        self.type = typ
        self.instruction_no = instruction_no
        self.keyword_name = keyword_name
        self.caller = caller
        self.pos = pos

class Visitor(ast.NodeVisitor):
    def __init__(self):
        super().__init__()
        self.__structs__: List[Struct] = []
        self.__assign_line_no__ = -1
        self.__expr_line_no__ = -1
        self.__val__ = None
        self.__pruned_names__ = []
        self.__keyword_name__ = None
        self.__call_stack__ = []
        self.__pos_arg_stack__ = []


    def consolidate_structs(self):
        new = []
        for struct in self.__structs__:
            distinct = True
            match = None
            for prev_struct in new:
                if (struct.instruction_no == prev_struct.instruction_no
                        and struct.type == prev_struct.type
                        and struct.value == prev_struct.value):
                    distinct = False
                    match = prev_struct
                    break
            if distinct:
                new.append(struct)
            else:
                if type(match.name) == list:
                    match.name.append(struct.name)
                else:
                    match.name = [match.name, struct.name]
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
                    assert self.__pruned_names__, "Static Analyzer: Failed to retrieve name of assignee variable"

                    if attr == 'read':
                        self.__structs__.append(Struct(assignee=self.__pruned_names__,
                                                       value=self.__val__,
                                                       typ='read',
                                                       instruction_no=self.__assign_line_no__,
                                                       keyword_name=self.__keyword_name__,
                                                       caller=caller, pos=pos))
                    elif attr == 'write':
                        self.__structs__.append(Struct(assignee=self.__pruned_names__,
                                                       value=self.__val__,
                                                       typ='write',
                                                       instruction_no=self.__assign_line_no__,
                                                       keyword_name=self.__keyword_name__,
                                                       caller=caller, pos=pos))
                    elif attr == 'parameter':
                        self.__structs__.append(Struct(assignee=self.__pruned_names__,
                                                       value=self.__val__,
                                                       typ='parameter',
                                                       instruction_no=self.__assign_line_no__,
                                                       keyword_name=self.__keyword_name__,
                                                       caller=caller, pos=pos))
                    elif attr == 'metric':
                        self.__structs__.append(Struct(assignee=self.__pruned_names__,
                                                       value=self.__val__,
                                                       typ='metric',
                                                       instruction_no=self.__assign_line_no__,
                                                       keyword_name=self.__keyword_name__,
                                                       caller=caller, pos=pos))
                else:
                    # EXPR CONTEXT
                    if attr == 'read':
                        self.__structs__.append(Struct(value=self.__val__,
                                                       typ='read',
                                                       instruction_no=self.__expr_line_no__,
                                                       keyword_name=self.__keyword_name__,
                                                       caller=caller, pos=pos))
                    elif attr == 'write':
                        self.__structs__.append(Struct(value=self.__val__,
                                                       typ='write',
                                                       instruction_no=self.__expr_line_no__,
                                                       keyword_name=self.__keyword_name__,
                                                       caller=caller, pos=pos))
                    elif attr == 'parameter':
                        self.__structs__.append(Struct(value=self.__val__,
                                                       typ='parameter',
                                                       instruction_no=self.__expr_line_no__,
                                                       keyword_name=self.__keyword_name__,
                                                       caller=caller, pos=pos))
                    elif attr == 'metric':
                        self.__structs__.append(Struct(value=self.__val__,
                                                       typ='metric',
                                                       instruction_no=self.__expr_line_no__,
                                                       keyword_name=self.__keyword_name__,
                                                       caller=caller, pos=pos))

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
            src = self.__pruned_names__
            for idx, each in enumerate(node.elts):
                self.__pruned_names__ = src[idx]
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
            src = self.__pruned_names__
            for idx, each in enumerate(node.elts):
                self.__pruned_names__ = src[idx]
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
