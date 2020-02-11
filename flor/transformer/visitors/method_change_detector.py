import ast
from flor.transformer.utils import node_in_nodes


class MethodChangeDetector(ast.NodeVisitor):

    def __init__(self):
        self.mutated_objects = []

    def visit_Call(self, node):
        """
        TODO: Make sure you add dynamic checks in proc_side_effects
        Some calls correspond to modules rather than objects.
        These cannot be disambiguated statically.
        """
        if isinstance(node.func, ast.Attribute):
            mutated_node = node.func.value
            if not node_in_nodes(mutated_node, self.mutated_objects):
                self.mutated_objects.append(mutated_node)

    def visit_Subscript(self, node):
        store_node = node.value
        if not node_in_nodes(store_node, self.mutated_objects):
            self.mutated_objects.append(store_node)

    # def visit_Expr(self, node):
    #     """
    #     TODO: Missing Additional checks
    #
    #     How would this method treat
    #     torch.cuda.synchronize()?
    #
    #     :param node:
    #     :return:
    #     """
    #     if isinstance(node.value, ast.Call):
    #         if isinstance(node.value.func, ast.Attribute):
    #             mutated_node = node.value.func.value
    #             if not node_in_nodes(mutated_node, self.mutated_objects):
    #                 self.mutated_objects.append(mutated_node)
