import ast
from copy import deepcopy


class False_Break(ValueError): pass  # Dummy exception used by node_eq


def node_eq(node1, node2):
    """
    Sub-routine to node_equals.
    Compares to ASTrees rooted at node1 and node2.
    Returns True if they are equal in the Flor-relevant sense.
    May raise an exception as a way of forcing a false up the stack

    TODO: Testing
    :param node1: an Abstract Syntax Tree
    :param node2: an Abstract Syntax Tree
    :return: T/F
    """
    for (field1, value1), (field2, value2) in zip(ast.iter_fields(node1), ast.iter_fields(node2)):
        if field1 == 'ctx' and field2 == 'ctx':
            continue
        if field1 != field2:
            raise False_Break()
        if type(value1) != type(value2):
            raise False_Break()
        if isinstance(value1, list):
            if len(value1) != len(value2):
                raise False_Break()
            for item1, item2 in zip(value1, value2):
                if type(item1) != type(item2):
                    raise False_Break()
                if isinstance(item1, ast.AST):
                    node_eq(item1, item2)
                else:
                    if item1 != item2:
                        raise False_Break()
        elif isinstance(value1, ast.AST):
            node_eq(value1, value2)
        else:
            if value1 != value2:
                raise False_Break()


def node_equals(node1, node2):
    try:
        node_eq(node1, node2)
        return True
    except False_Break:
        return False


def node_in_nodes(node, nodes):
    """
    It's the equivalent of
    node in nodes
    """
    for other in nodes:
        if node_equals(node, other):
            return True
    return False


def set_union(nodes1, nodes2):
    """
    Conjecture nodes1 and nodes2 are both lists
    """
    output = [n for n in nodes1]
    for node2 in nodes2:
        if not node_in_nodes(node2, nodes1):
            output.append(node2)
    return output


def underscoring_set_union(lsd, mcd):
    output = [n for n in lsd]
    underscore = ast.Name('_', ast.Store())
    for node in mcd:
        if not node_in_nodes(node, output):
            output.append(deepcopy(underscore))
    return output


def set_intersection(nodes1, nodes2):
    output = []
    for node1 in nodes1:
        for node2 in nodes2:
            if node_equals(node1, node2) and not node_in_nodes(node1, output):
                output.append(node1)
    return output
