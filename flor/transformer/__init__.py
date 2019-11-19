import ast
from flor.transformer.visitors import get_change_and_read_set, LoadStoreDetector
from flor.transformer.code_gen import *
from flor.transformer.utils import set_intersection, set_union, node_in_nodes


class Transformer(ast.NodeTransformer):

    class RefuseTransformError(RuntimeError):
        pass

    def __init__(self):
        # Assign_Updates
        #   These are names (LHS of assign) with the following assign semantics:
        #   On re-assign, the value of the name is _updated_
        #       (rather than a new value with a same name being created that shadows the first)
        self.assign_updates = []

        # Loop_Context
        #   Are we in a loop context?
        self.loop_context = False

    def visit_Assign(self, node):
        lsd = LoadStoreDetector(writes=self.assign_updates)
        lsd.visit(node)
        lsd = LoadStoreDetector()
        [lsd.visit(n) for n in node.targets]
        output = [self.generic_visit(node), ]
        for name in lsd.writes:
            output.append(make_test_force(name))
        return output

    def visit_FunctionDef(self, node):
        temp = self.assign_updates
        self.assign_updates = []
        lsd = LoadStoreDetector(writes=self.assign_updates)
        lsd.visit(node.args)
        output = self.generic_visit(node)
        self.assign_updates = temp

        output.body = [make_block_initialize('namespace_stack'), ] + output.body

        output.body = [ast.Try(body=output.body,
                              handlers=[],
                              orelse=[],
                              finalbody=[make_block_destroy('namespace_stack')]), ]

        return output

    def visit_Expr(self, node):
        if self.loop_context and is_side_effecting(node) and not is_expr_excepted(node):
            # In the context of a MAY-MEMOIZE loop
            raise self.RefuseTransformError()
        return self.generic_visit(node)

    def _vistit_loop(self, node):
        lsd_change_set, mcd_change_set, read_set = get_change_and_read_set(node)
        change_set = set_union(lsd_change_set, mcd_change_set)
        memoization_set = set_intersection(set_union(self.assign_updates, read_set), change_set)

        new_node = self.generic_visit(node)

        underscored_memoization_set = []
        for element in memoization_set:
            if node_in_nodes(element, lsd_change_set):
                underscored_memoization_set.append(element)
            else:
                underscored_memoization_set.append(ast.Name('_', ast.Store()))

        # Inner Block
        block_initialize = make_block_initialize('skip_stack')
        cond_block = make_cond_block()
        proc_side_effects = make_proc_side_effects(underscored_memoization_set,
                                                   memoization_set)
        cond_block.body = new_node.body
        new_node.body = [block_initialize, cond_block, proc_side_effects]

        # Outer Block
        block_initialize = make_block_initialize('skip_stack')
        cond_block = make_cond_block()
        proc_side_effects = make_proc_side_effects(underscored_memoization_set,
                                                   memoization_set)

        cond_block.body = [new_node, ]

        return [block_initialize, cond_block, proc_side_effects]

    def proc_loop(self, node):
        temp = self.loop_context
        self.loop_context = True

        try:
            new_node = self._vistit_loop(node)
            return new_node
        except self.RefuseTransformError:
            if temp:
                raise
            return ast.NodeTransformer().generic_visit(node)
        except AssertionError:
            print("Assertion Error")
            return ast.NodeTransformer().generic_visit(node)
        finally:
            self.loop_context = temp

    def visit_For(self, node):
        return self.proc_loop(node)

    def visit_While(self, node):
        return self.proc_loop(node)

