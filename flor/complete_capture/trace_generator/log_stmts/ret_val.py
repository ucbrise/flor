from .. import util as gen
import ast

class Return():

    def __init__(self, node: ast.Return):
        self.node = node    #node.value

    def to_string(self):
        if self.node.value:
            out = "__return__ = {}\n".format(gen.proc_lhs(self.node.value, esc=False))
            out += "return __return__"
        else:
            out = gen.proc_lhs(self.node, esc=False)
        return out

    def parse(self):
        return ast.parse(self.to_string())

class Yield():
    def __init__(self, node):
        self.node = node

    def to_string(self):
        if self.node.value:
            out = "__return__ = {}\n".format(gen.proc_lhs(self.node.value, esc=False))
            out += "yield __return__"
        else:
            out = gen.proc_lhs(self.node, esc=False)
        return out

    def parse(self):
        return ast.parse(self.to_string())