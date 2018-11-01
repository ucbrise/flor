import ast
import astor
import astunparse

from typing import List, Dict
from flor.context.struct import Struct
from flor.controller.parser.flor_log import FlorLog


class Transformer(ast.NodeTransformer):

    def __init__(self, structs: List[Struct]):
        super().__init__()
        self.__structs__ = structs

    def visit_Attribute(self, node):
        if isinstance(node.ctx, ast.Store):
            return super().generic_visit(node)
        elif isinstance(node.ctx, ast.Load):
            value = astunparse.unparse(node.value).strip()
            attr = node.attr
            if value == 'log' or value == 'flor.log':
                if attr in ['read', 'write', 'parameter', 'metric']:
                    return FlorLog
            return super().generic_visit(node)

    def visit_Call(self, node):
        if self.visit(node.func) is FlorLog:
            struct = self.__structs__.pop(0)
            interim  = ast.parse("flor.internal_log({}, {})".format(
                struct.value, struct.to_dict())).body[0].value
            return super().generic_visit(interim)
        return super().generic_visit(node)

