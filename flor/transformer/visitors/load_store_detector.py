import ast

from flor.transformer.visitors.eval_order import EvalOrderVisitor
from flor.transformer.utils import set_union, node_in_nodes


class LoadStoreDetector(EvalOrderVisitor):

    def __init__(self, writes=None):
        super().__init__()
        if writes is None:
            self.writes = []
        else:
            self.writes = writes
        self.unmatched_reads = []

    def set_checkpoint(self):
        """
        Returns a copy of the  current state of the Node Visitor
        """
        return list(self.writes), list(self.unmatched_reads)

    def branch_to(self, state):
        """
        counter-part to set_checkpoint
        Sets the Node Visitor state to the argument
        Returns the current state in case the caller wants to save it
        """
        writes, unmatched_reads = state
        temp = list(self.writes), list(self.unmatched_reads)
        self.writes = writes
        self.unmatched_reads = unmatched_reads
        return temp

    def merge_branches(self, state1, state2):
        """
        if False:
            x = 10
        else:
            y = x

        --------------
        self.writes = [x, y]
        self.unmatched_reads = [x]
        """
        self.writes = set_union(state1[0], state2[0])
        self.unmatched_reads = set_union(state1[1], state2[1])

    def do_unimplemented_behavior(self):
        raise NotImplementedError()

    def visit_arg(self, node):
        self.visit(ast.Name(
            id=node.arg,
            ctx=ast.Store()
        ))

    def visit_Name(self, node):
        self.proc_node(node)

    def visit_Attribute(self, node):
        self.visit(node.value)
        self.proc_node(node)


    def proc_node(self, node):
        if isinstance(node.ctx, ast.Load):
            if not node_in_nodes(node, self.writes):
                if not node_in_nodes(node, self.unmatched_reads):
                    self.unmatched_reads.append(node)
        else:
            if not node_in_nodes(node, self.writes):
                self.writes.append(node)