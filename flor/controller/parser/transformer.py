import ast
import astor
import astunparse

from typing import List, Dict
from flor.context.struct import Struct
from flor.controller.parser.flor_log import FlorLog


class Transformer(ast.NodeTransformer):

    def __init__(self, structs: List[Struct], args):
        super().__init__()
        self.__structs__ = structs
        self.__args__ = args

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
            from_arg = ' or '.join(["{} is {}".format(struct.value, each) for each in self.__args__])
            if from_arg:
                from_arg = ast.parse(from_arg).body[0].value
            interim  = ast.parse("flor.internal_log({}, {})".format(
                struct.value, struct.to_dict())).body[0].value
            if from_arg:
                interim.args[1].keys.append(ast.Str(s='from_arg'))
                interim.args[1].values.append(from_arg)
            # print(astor.dump_tree(interim))
            return super().generic_visit(interim)
        return super().generic_visit(node)

