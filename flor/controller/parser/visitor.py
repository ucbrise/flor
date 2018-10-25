import ast
import astor
import astunparse

from typing import List

class Struct:
    def __init__(self, name=None, value=None, typ=None, instruction_no=None):
        # ALERT: value is an AST node, it must be evaluated on transformer
        self.name = name
        self.value = value
        self.type = typ
        self.instruction_no = instruction_no

class Visitor(ast.NodeVisitor):
    # TODO: log.write in with clause

    def __init__(self):
        super().__init__()
        self.__structs__: List[Struct] = []
        self.__assign_line_no__ = -1
        self.__expr_line_no__ = -1
        self.__val__ = None
        self.__pruned_names__ = []
        self.__keywoard_name__ = None

    def visit_Attribute(self, node):
        if type(node.ctx) == ast.Store:
            return astunparse.unparse(node)
        elif type(node.ctx) == ast.Load:
            value = self.visit(node.value)
            attr = node.attr
            if value == 'flor':
                if attr == 'log':
                    return 'flor.log'
            elif value == 'log' or value == 'flor.log':
                if self.__assign_line_no__ >= 0:
                    # Assign context
                    if attr == 'read':
                        if self.__pruned_names__:
                            self.__structs__.append(Struct(name=self.__pruned_names__,
                                                           value=self.__val__,
                                                           typ='read',
                                                           instruction_no=self.__assign_line_no__))
                        else:
                            assert (type(self.__val__) == ast.Subscript
                                    or type(self.__val__) == ast.Attribute
                                    or type(self.__val__) == ast.Name)
                            self.__structs__.append(Struct(name=self.visit(self.__val__),
                                                           value=self.__val__,
                                                           typ='read',
                                                           instruction_no=self.__assign_line_no__))
                    elif attr == 'write':
                        if self.__pruned_names__:
                            self.__structs__.append(Struct(name=self.__pruned_names__,
                                                           value=self.__val__,
                                                           typ='write',
                                                           instruction_no=self.__assign_line_no__))
                        else:
                            assert (type(self.__val__) == ast.Subscript
                                    or type(self.__val__) == ast.Attribute
                                    or type(self.__val__) == ast.Name)
                            self.__structs__.append(Struct(name=self.visit(self.__val__),
                                                           value=self.__val__,
                                                           typ='read',
                                                           instruction_no=self.__assign_line_no__))
                    else:
                        if self.__keywoard_name__ is not None:
                            if attr == 'parameter':
                                self.__structs__.append(Struct(name=self.__keywoard_name__,
                                                               value=self.__val__,
                                                               typ='parameter',
                                                               instruction_no=self.__assign_line_no__))
                            elif attr == 'metric':
                                self.__structs__.append(Struct(name=self.__keywoard_name__,
                                                               value=self.__val__,
                                                               typ='metric',
                                                               instruction_no=self.__assign_line_no__))
                        else:
                            if attr == 'parameter':
                                self.__structs__.append(Struct(name=self.visit(self.__val__),
                                                               value=self.__val__,
                                                               typ='parameter',
                                                               instruction_no=self.__assign_line_no__))
                            elif attr == 'metric':
                                self.__structs__.append(Struct(name=self.visit(self.__val__),
                                                               value=self.__val__,
                                                               typ='metric',
                                                               instruction_no=self.__assign_line_no__))
                else:
                    # EXPR context
                    if attr == 'read':
                        assert (type(self.__val__) == ast.Subscript
                                or type(self.__val__) == ast.Attribute
                                or type(self.__val__) == ast.Name)
                        self.__structs__.append(Struct(name=self.visit(self.__val__),
                                                       value=self.__val__,
                                                       typ='read',
                                                       instruction_no=self.__expr_line_no__))
                    elif attr == 'write':
                        assert (type(self.__val__) == ast.Subscript
                                or type(self.__val__) == ast.Attribute
                                or type(self.__val__) == ast.Name)
                        self.__structs__.append(Struct(name=self.visit(self.__val__),
                                                       value=self.__val__,
                                                       typ='write',
                                                       instruction_no=self.__expr_line_no__))
                    else:
                        if self.__keywoard_name__ is not None:
                            if attr == 'parameter':
                                self.__structs__.append(Struct(name=self.__keywoard_name__,
                                                               value=self.__val__,
                                                               typ='parameter',
                                                               instruction_no=self.__expr_line_no__))
                            elif attr == 'metric':
                                self.__structs__.append(Struct(name=self.__keywoard_name__,
                                                               value=self.__val__,
                                                               typ='metric',
                                                               instruction_no=self.__expr_line_no__))
                        else:
                            if attr == 'parameter':
                                self.__structs__.append(Struct(name=self.visit(self.__val__),
                                                               value=self.__val__,
                                                               typ='parameter',
                                                               instruction_no=self.__expr_line_no__))
                            elif attr == 'metric':
                                self.__structs__.append(Struct(name=self.visit(self.__val__),
                                                               value=self.__val__,
                                                               typ='metric',
                                                               instruction_no=self.__expr_line_no__))
            return "{}.{}".format(value, attr)

    def visit_withitem(self, node):
        if node.optional_vars is None:
            self.__pruned_names__ = None
        else:
            self.__pruned_names__ = self.visit(node.optional_vars)
        self.visit(node.context_expr)
        self.__pruned_names__ = []


    def visit_keyword(self, node):
        self.__keywoard_name__ = node.arg
        self.visit(node.value)
        self.__keywoard_name__ = None

    def visit_With(self, node):
        self.__assign_line_no__ = node.lineno
        for item in node.items:
            self.visit(item)
        self.__assign_line_no__ = -1
        for each in node.body:
            self.visit(each)


    def visit_Expr(self, node):
        self.__expr_line_no__ = node.lineno
        self.visit(node.value)

    def visit_Call(self, node):
        if len(node.args) > 0:
            self.__val__ = node.args[0]
        self.visit(node.func)
        for arg in node.args:
            self.visit(arg)
        for kwd in node.keywords:
            self.visit(kwd)
        self.__val__ = None

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

        self.__assign_line_no__ = node.lineno
        for target in node.targets:
            self.__pruned_names__ = self.visit(target)
            self.visit(node.value)

        self.__pruned_names__ = []
        self.__assign_line_no__ = -1
