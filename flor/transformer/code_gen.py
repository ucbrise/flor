import ast
import astunparse
from copy import deepcopy

"""
This file generates ASTs for splicing into existing code
"""


def make_arg(arg):
    if isinstance(arg, (int, float)):
        return ast.Num(arg)
    elif isinstance(arg, str):
        return ast.Str(arg)
    else:
        raise NotImplementedError()


def make_attr_call(attr1, attr2, args=None):
    """
    flor._attr1_._attr2_(arg)
    """
    if args is None:
        return ast.Call(
            func=ast.Attribute(
                value=ast.Attribute(
                    value=ast.Name('flor', ast.Load()),
                    attr=attr1,
                    ctx=ast.Load()
                ),
                attr=attr2,
                ctx=ast.Load()
            ),
            args=[],                                        # arg is None
            keywords=[]
        )
    else:
        return ast.Call(
            func=ast.Attribute(
                value=ast.Attribute(
                    value=ast.Name('flor', ast.Load()),
                    attr=attr1,
                    ctx=ast.Load()
                ),
                attr=attr2,
                ctx=ast.Load()
            ),
            args=args,                                     # arg is not None
            keywords=[]
        )


def make_block_initialize(attr, args=None):
    """
    Traditionally used for initializing namespace blocks and skipblocks

    flor._attr_.new(arg)
    """
    assert args is None or isinstance(args, list)
    return ast.Expr(
        value=make_attr_call(attr, 'new', args),
    )


def make_block_destroy(attr):
    """
    The counter-part to make_block_initialize
    flor._attr_.pop()
    """
    return ast.Expr(
        value=make_attr_call(attr, 'pop')
    )


def make_cond_block():
    """
    if flor.skip_stack.peek().should_execute(not flor.SKIP):
        pass

    TODO: Extend to accept predicate
    """
    previous_arg = ast.UnaryOp(
                op=ast.Not(),
                operand=ast.Attribute(
                    value=ast.Name('flor', ast.Load()),
                    attr='SKIP',
                    ctx=ast.Load()
                )
            )
    safe_arg = ast.NameConstant(value=True)
    safe_arg = previous_arg

    return ast.If(
        test=ast.Call(
            func=ast.Attribute(
                value=make_attr_call('skip_stack', 'peek'),
                attr='should_execute',
                ctx=ast.Load()
            ),
            args=[safe_arg],
            keywords=[]
        ),
        body=[ast.Pass()],
        orelse=[]
    )


def make_proc_side_effects(left_arg_list, right_arg_list):
    """
    left_arg_list: list of ASTs, already under_scored
    right_arg_list: list of ASTs

    *arg_list = flor.skip_stack.pop().proc_side_effects(*arg_list)
    """
    assert len(left_arg_list) == len(right_arg_list)
    if left_arg_list:
        names = [astunparse.unparse(e).strip() for e in right_arg_list]
        mask = mask_lattice(names)

        right_arg_list = [arg for i,arg in enumerate(right_arg_list) if i in mask]
        left_arg_list = [arg for i, arg in enumerate(left_arg_list) if i in mask]

        load_list = deepcopy(right_arg_list)
        store_list = deepcopy(left_arg_list)
        for each in load_list:
            each.ctx = ast.Load()
        for each in store_list:
            each.ctx = ast.Store()
        if len(store_list) > 1:
            store_list = [ast.Tuple(elts=store_list)]
        return ast.Assign(
            targets=store_list,
            value=ast.Call(
                func=ast.Attribute(
                    value=make_attr_call('skip_stack','pop'),
                    attr='proc_side_effects',
                    ctx=ast.Load()
                ),
                args=load_list,
                keywords=[]
            )
        )
    else:
        # Case when there is nothing to proc_side_effect.
        return ast.Expr(
            value=make_attr_call('skip_stack', 'pop')
        )


def make_test_force(store_node):
    """
    flor.namespace_stack.test_force(_store_node_, '_store_node_')
    """
    store_node = deepcopy(store_node)
    store_node.ctx = ast.Load()

    store_node_name = ast.Str(astunparse.unparse(store_node).strip())

    namespace_stack_new = make_attr_call('namespace_stack', 'test_force', [store_node, store_node_name])
    return ast.Expr(namespace_stack_new)


def is_side_effecting(node):
    """
    This determines whether node is a statement with possibly arbitrary side-effects
    """
    node = node.value
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name)


def is_expr_excepted(node):
    """
    Depends on is_side_effecting being true
    :param node:
    :return:
    """
    node = node.value
    if node.func.id in ['print', 'print_once', 'print_dict']:
        return True
    return False

def mask_lattice(names):
    """
    names: [self, self.training, self.model, optimizer]
    """
    def is_prefixed_by(name, prefix):
        if len(prefix) >= len(name):
            return False
        name = name[0:len(prefix)]
        for l,r in zip(name, prefix):
            if l != r:
                return False
        return True
    names = [name.split('.') for name in names]
    mask = []
    for i, out_name in enumerate(names):
        mask.append(i)
        for j, in_name in enumerate(names):
            if i == j:
                continue
            if is_prefixed_by(out_name, in_name):
                mask.pop()
                break
    return mask
