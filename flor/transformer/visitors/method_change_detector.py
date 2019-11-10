import ast
from flor.transformer.utils import node_in_nodes


class MethodChangeDetector(ast.NodeVisitor):

    def __init__(self):
        self.mutated_objects = []

    def visit_Expr(self, node):
        """
        TODO: Missing Additional checks

        How would this method treat
        torch.cuda.synchronize()?

        :param node:
        :return:
        """
        if isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Attribute):
                mutated_node = node.value.func.value
                if not node_in_nodes(mutated_node, self.mutated_objects):
                    self.mutated_objects.append(mutated_node)