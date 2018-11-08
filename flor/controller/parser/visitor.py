import ast
import astunparse

from typing import List
from flor.context.struct import Struct


class Visitor(ast.NodeVisitor):

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
                    assert self.__pruned_names__, "Static Analyzer: Failed to retrieve name of assignee variable"

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
                    elif attr == 'parameter':
                        self.__structs__.append(Struct(assignee=self.__pruned_names__,
                                                       value=astunparse.unparse(self.__val__).strip(),
                                                       typ='parameter',
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
                    elif attr == 'parameter':
                        self.__structs__.append(Struct(value=astunparse.unparse(self.__val__).strip(),
                                                       typ='parameter',
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
                if src:
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
                if src:
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
