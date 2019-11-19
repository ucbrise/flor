import ast
from copy import deepcopy


def make_attr_call(attr1, attr2):
    """
    flor._attr1_._attr2_()
    """
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
        args=[],
        keywords=[]
    )


def make_block_initialize(attr):
    """
    flor._attr_.new()
    """
    return ast.Expr(
        value=make_attr_call(attr,'new')
    )


def make_block_destroy(attr):
    """
    flor._attr_.pop()
    """
    return ast.Expr(
        value=make_attr_call(attr,'pop')
    )


def make_cond_block():
    """
    if flor.skip_stack.peek().should_execute(not flor.SKIP):
        pass

    TODO: Extend to accept predicate
    """
    return ast.If(
        test=ast.Call(
            func=ast.Attribute(
                value=make_attr_call('skip_stack', 'peek'),
                attr='should_execute',
                ctx=ast.Load()
            ),
            args=[ast.UnaryOp(
                op=ast.Not(),
                operand=ast.Attribute(
                    value=ast.Name('flor', ast.Load()),
                    attr='SKIP',
                    ctx=ast.Load()
                )
            )],
            keywords=[]
        ),
        body=[ast.Pass()],
        orelse=[]
    )


def make_proc_side_effects(left_arg_lsit, right_arg_list):
    """
    *arg_list = flor.skip_stack.pop().proc_side_effects(*arg_list)
    """
    load_list = deepcopy(right_arg_list)
    store_list = deepcopy(left_arg_lsit)
    for each in load_list:
        each.ctx = ast.Load()
    for each in store_list:
        each.ctx = ast.Store()
    return ast.Assign(
        targets=[ast.Tuple(elts=store_list)],
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


def make_test_force(store_node):
    """
    flor.namespace_stack.test_force(_store_node_, '_store_node_')
    """
    store_node = deepcopy(store_node)
    store_node.ctx = ast.Load()

    store_node_name = ast.Str(store_node.id)

    namespace_stack_new = make_attr_call('namespace_stack', 'test_force')
    namespace_stack_new.args = [store_node,store_node_name]
    return ast.Expr(namespace_stack_new)


def make_decorator(arg_list: [str]):
    arg_list = [ast.Str(x) for x in arg_list]
    return ast.Call(
        func=ast.Attribute(
            value=ast.Name('flor', ast.Load()),
            attr='side_effects',
            ctx=ast.Load()
        ),
        args=arg_list,
        keywords=[]
    )


def is_side_effecting(node):
    node = node.value
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name)


def is_expr_excepted(node):
    """
    Depends on is_side_effecting being true
    :param node:
    :return:
    """
    node = node.value
    if node.func.id == 'print':
        return True
    return False
