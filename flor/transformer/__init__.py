from flor.transformer.visitors import get_change_and_read_set, LoadStoreDetector, StatementCounter
from flor.transformer.code_gen import *
from flor.transformer.utils import set_intersection, set_union, node_in_nodes
import copy
import astor
import os

class Transformer(ast.NodeTransformer):
    static_key = 0

    class RefuseTransformError(RuntimeError):
        pass

    @staticmethod
    def transform(filepaths, inplace=False, root_script=None):

        if not isinstance(filepaths, list):
            root_script = filepaths
            filepaths = [filepaths,]
        elif len(filepaths) == 1:
            root_script = filepaths[0]

        for filepath in filepaths:
            with open(filepath, 'r') as f:
                contents = f.read()
            transformer = Transformer()
            new_contents = transformer.visit(ast.parse(contents))
            new_contents.body.insert(0, ast.Import(names=[ast.alias('flor', asname=None)]))
            if root_script and os.path.samefile(filepath, root_script):
                new_contents.body.append(ast.If(test=ast.UnaryOp(op=ast.Not(), operand=ast.Attribute(value=ast.Name('flor'), attr='SKIP')),
                                                body=[ast.Expr(value=ast.Call(func=ast.Attribute(value=ast.Name('flor'), attr='flush'),
                                                                              args=[], keywords=[]))], orelse=[]))
            new_contents = astor.to_source(new_contents)
            new_filepath, ext = os.path.splitext(filepath)
            new_filepath += ('_transformed' if not inplace else '') + ext
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

        if not memoization_set:
            #TODO: should probably raise a RefuseTransformError
            return new_node

        underscored_memoization_set = []
        for element in memoization_set:
            if not node_in_nodes(element, mcd_change_set):
                underscored_memoization_set.append(element)
            else:
                underscored_memoization_set.append(ast.Name('_', ast.Store()))

        # Outer Block
        block_initialize = make_block_initialize('skip_stack', [make_arg(self.get_incr_static_key()),])
        cond_block = make_cond_block()
        proc_side_effects = make_proc_side_effects(underscored_memoization_set,
                                                   memoization_set)

        cond_block.body = [new_node, ]

        return [block_initialize, cond_block, proc_side_effects]

    def proc_loop(self, node):
        temp = self.loop_context

        sc = StatementCounter()
        sc.visit(node)
        if sc.count <= 3:
            # import astor
            # print(astor.to_source(node))
            self.loop_context = False
            noud = self.generic_visit(node)
            self.loop_context = temp

            blinit = make_block_initialize('skip_stack', [make_arg(self.get_incr_static_key()), make_arg(0)])
            blestroy = make_block_destroy('skip_stack')

            return [blinit, noud, blestroy]


        self.loop_context = True

        temp_assign_updates = list(self.assign_updates)
        node_clone = copy.deepcopy(node)

        try:
            new_node = self._vistit_loop(node)
            return new_node
        except self.RefuseTransformError:
            if temp:
                raise
            self.loop_context = False
            self.assign_updates = temp_assign_updates
            new_node = self.generic_visit(node_clone)

            blinit = make_block_initialize('skip_stack', [make_arg(self.get_incr_static_key()), make_arg(0)])
            blestroy = make_block_destroy('skip_stack')

            return [blinit, new_node, blestroy]
        except AssertionError as e:
            print(f"Assertion Error: {e}")
            return ast.NodeTransformer().generic_visit(node)
        finally:
            self.loop_context = temp



    def visit_For(self, node):
        return self.proc_loop(node)

    def visit_While(self, node):
        return self.proc_loop(node)


class PartitionTransformer(ast.NodeTransformer):

    def __init__(self, outermost_sk):
        self.outermost_sk = outermost_sk
        self.enabled = False
        self.transformed = False

    @classmethod
    def transform(cls, filepaths, xp_name=None, memo=None, outermost_sk=None):

        if outermost_sk is None:
            assert xp_name is not None and memo is not None
            import flor
            flor.initialize(xp_name, mode='reexec', memo=memo)
            from flor.writer import Writer
            log_record = Writer.stateful_adaptive_ext
            outermost_sk = int(log_record['outermost_sk'])

        outermost_sk = int(outermost_sk)
        if not isinstance(filepaths, list):
            filepaths = [filepaths,]

        for filepath in filepaths:
            with open(filepath, 'r') as f:
                contents = f.read()
            transformer = cls(outermost_sk)
            new_contents = transformer.visit(ast.parse(contents))
            if not transformer.transformed:
                continue
            new_contents = astor.to_source(new_contents)
            with open(filepath, 'w') as f:
                f.write(new_contents)
            print(f"rewrote {filepath}")

    def visit_For(self, node):
        if self.enabled:
            self.transformed = True
            self.enabled = False
            node.iter = ast.Call(func=ast.Attribute(value=ast.Name(id='flor'), attr='partition'),
                args=[
                    node.iter,
                    ast.Attribute(value=ast.Name(id='flor'), attr='PID'),
                    ast.Attribute(value=ast.Name(id='flor'), attr='NPARTS')],
                keywords=[])
        return node

    def visit_While(self, node):
        if self.enabled:
            self.transformed = True
            self.enabled = False
            # test = ast.Expr(node.test)
            node = ast.For(target=ast.Name(id='_'),
                               iter=ast.Call(func=ast.Attribute(value=ast.Name(id='flor'), attr='partition'),
                                    args=[
                                        ast.Call(func=ast.Name(id='range'),
                                            args=[ast.Call(func=ast.Attribute(value=ast.Name(id='flor'), attr='get_epochs'), args=[], keywords=[])],
                                            keywords=[]),
                                        ast.Attribute(value=ast.Name(id='flor'), attr='PID'),
                                        ast.Attribute(value=ast.Name(id='flor'), attr='NPARTS')],
                                    keywords=[]),
                               body=[ast.Expr(node.test),] + node.body, orelse=[])
        return node


    def visit_Call(self, node):
        src = astor.to_source(node)
        if 'flor.skip_stack.new' in astor.to_source(node.func).strip():
            if (len(node.args) > 0
                and isinstance(node.args[0], ast.Num)
                and node.args[0].n == self.outermost_sk):
                self.enabled = True
        return node

class SampleTransformer(PartitionTransformer):

    def visit_For(self, node):
        if self.enabled:
            node = ast.For(target=ast.Attribute(value=ast.Name(id='flor'), attr='PID'),
                    iter=ast.Call(func=ast.Attribute(value=ast.Name(id='flor'), attr='sample'), args=[node.iter, ast.Attribute(value=ast.Name(id='flor'), attr='RATE')], keywords=[]),
                    body=[super().visit_For(node),],
                    orelse=[])
        return node

    def visit_While(self, node):
        if self.enabled:
            node = ast.For(target=ast.Attribute(value=ast.Name(id='flor'), attr='PID'),
                           iter=ast.Call(func=ast.Attribute(value=ast.Name(id='flor'), attr='sample'),
                                         args=[ast.Call(func=ast.Name(id='range'),
                                                args=[ast.Call(func=ast.Attribute(value=ast.Name(id='flor'), attr='get_epochs'), args=[], keywords=[])],
                                                keywords=[]),
                                               ast.Attribute(value=ast.Name(id='flor'), attr='RATE')], keywords=[]),
                           body=[super().visit_While(node), ],
                           orelse=[])
        return node
