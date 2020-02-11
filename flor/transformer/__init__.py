import ast
from flor.transformer.visitors import get_change_and_read_set, LoadStoreDetector
from flor.transformer.code_gen import *
from flor.transformer.utils import set_intersection, set_union, node_in_nodes


class Transformer(ast.NodeTransformer):
    static_key = 0

    class RefuseTransformError(RuntimeError):
        pass

    @staticmethod
    def transform(filepath):
        import astor
        import os
        with open(filepath, 'r') as f:
            contents = f.read()
        transformer = Transformer()
        new_contents = transformer.visit(ast.parse(contents))
        new_contents = astor.to_source(new_contents)
        new_filepath, ext = os.path.splitext(filepath)
        new_filepath += '_transformed' + ext
        with open(new_filepath, 'w') as f:
            f.write(new_contents)
        print(f"wrote {new_filepath}")

    def __init__(self):
        # These are names defined before the loop
        # If these values are updated in the loop body, we want to save them
        #       Even if they look like "blind writes"
        self.assign_updates = []

        # Loop_Context
        #   Are we in a loop context?
        self.loop_context = False

    def get_incr_static_key(self):
        sk = Transformer.static_key
        Transformer.static_key += 1
        return sk

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
        lsd.visit(node.args)                                # This is possibly redundant but harmless because of set semantics of lsd
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
        memoization_set = set_intersection(set_union(self.assign_updates, read_set), change_set)    # read_set: unmatched_reads

        new_node = self.generic_visit(node)

        underscored_memoization_set = []
        for element in memoization_set:
            if not node_in_nodes(element, mcd_change_set):
                underscored_memoization_set.append(element)
            else:
                underscored_memoization_set.append(ast.Name('_', ast.Store()))

        # Outer Block
        block_initialize = make_block_initialize('skip_stack', make_arg(self.get_incr_static_key()))
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
        except AssertionError as e:
            print(f"Assertion Error: {e}")
            return ast.NodeTransformer().generic_visit(node)
        finally:
            self.loop_context = temp



    def visit_For(self, node):
        return self.proc_loop(node)

    def visit_While(self, node):
        return self.proc_loop(node)

