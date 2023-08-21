#!/usr/bin/env python
# type: ignore

from argparse import ArgumentParser, FileType, Namespace
from ast import AST, Name, iter_fields, parse, unparse, walk, stmt, literal_eval
from collections import defaultdict
from copy import deepcopy
import sys
import ast

from .gumtree import GumTree, python
from .visitors import LoggedExpVisitor


def propagate(args: Namespace):
    tree, target = [parse(f.read()) for f in (args.source, args.target)]
    args.source.close()
    args.target.close()
    args.out = open(args.out, "w")

    replicate(tree, find(tree, lineno=args.lineno), target, **args.gumtree)  # type: ignore
    print(unparse(target), file=args.out)


def replicate(tree: AST, node: stmt, target: AST, **kwargs):
    """`
    First we do code-block alignment using the GumTree
    algorithm from Falleri et al.
    """
    adapter = python.Adapter(tree, target)
    mapping = GumTree(adapter, **kwargs).mapping(tree, target)
    # asserting `tree` is the root of `node` in the `adapter`
    # assert tree == adapter.root(node) and isinstance(node, stmt)

    """
    Then we insert the back-propagated statement into the target block
    QUERY:
        Is the back-propagation always intra-block, meaning from
        same block in version v to same block in version ancestor(v)


    It's working. It's not off by one, it's finding an 
    adjacent injection site.

    The contextual copy ignores content of target
    """
    block, index = find_insert_loc(adapter, node, mapping)
    assert block is not None
    # if node in mapping:
    lev = LoggedExpVisitor()
    lev.visit(node)
    pnv = PairNodeVisitor(lev.name)

    if node in mapping:
        # ABORT
        edon = mapping[node]

        original = block.pop(index)  # type: ignore
        original_s = deepcopy(original)
        original = pnv.visit(node, original)
        assert pnv.success
        block.insert(index, original)  # type: ignore
    else:
        new = make_contextual_copy(adapter, node, mapping)
        block.insert(index, new)  # type: ignore


class PairNodeVisitor(ast.NodeTransformer):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.success = False

    def make_wrapper(self, child):
        if isinstance(child, AST):
            return (
                ast.parse(f"flor.log('{self.name}',{unparse(child).strip()})")
                .body[0]
                .value  # type: ignore
            )
        elif isinstance(child, (list, tuple)):
            return (
                ast.parse(
                    f"flor.log('{self.name}',{[unparse(c).strip() for c in child]})"
                )
                .body[0]
                .value  # type:ignore
            )
        else:
            raise NotImplementedError

    def equals(self, node1: AST, node2: AST):
        if type(node1) != type(node2):
            return False
        for a in node1._fields:
            if type(getattr(node1, a)) != type(getattr(node2, a)):
                return False
        return True

    def visit(self, node1: AST, node2: AST):
        """Visit a node."""
        if not self.equals(node1, node2):

            field = None

            for f in node1._fields:
                v = getattr(node1, f)
                s = unparse(v).strip() if isinstance(v, AST) else str(v)
                if "flor.log" in s:
                    field = f
                    break
            assert field is not None
            child = getattr(node2, field)
            logging_child = self.make_wrapper(child)
            setattr(node2, field, logging_child)
            self.success = True
            return node2
        return self.generic_visit(node1, node2)

    def generic_visit(self, node1, node2):
        for (fld1, old_val1), (fld2, old_val2) in zip(
            iter_fields(node1), iter_fields(node2)
        ):
            if isinstance(old_val1, list):
                assert isinstance(old_val2, list)
                new_values = []
                for val1, val2 in zip(old_val1, old_val2):
                    if isinstance(val1, AST):
                        assert isinstance(val2, AST)
                        val2 = self.visit(val1, val2)
                        if val2 is None:
                            continue
                        elif not isinstance(val2, AST):
                            new_values.extend(val2)
                            continue
                    new_values.append(val2)
                old_val2[:] = new_values
            elif isinstance(old_val1, AST):
                assert isinstance(old_val2, AST)
                new_node = self.visit(old_val1, old_val2)
                if new_node is None:
                    delattr(node1, fld1)
                    delattr(node2, fld2)
                else:
                    setattr(node2, fld2, new_node)
        return node2


def find_insert_loc(adapter, node, mapping):
    parent = adapter.parent(node)

    context = None
    for sibling in adapter.children(parent):
        if id(sibling) == id(node):
            break
        if sibling in mapping:
            context = sibling

    if context is not None:
        ref = mapping[context]
        block = adapter.parent(ref)
        index = 1 + block.index(ref)
    elif parent in mapping:
        block = mapping[parent]
        index = 0
    else:
        exit("Unable to map context!")

    return block, index


def make_contextual_copy(adapter, node, mapping):
    renames = defaultdict(lambda: defaultdict(int))
    for source, target in mapping.items():
        if isinstance(source, Name) and not adapter.contains(source, node):
            renames[source.id][target.id] += 1

    new = deepcopy(node)
    for n in walk(new):
        if isinstance(n, Name) and n.id in renames:
            n.id = max(renames[n.id], key=renames[n.id].get)  # type: ignore
    return new


def find(t: AST, *, lineno: int):
    adapter = python.Adapter(t)  # FIXME: odd dependency
    res = None
    for n in adapter.postorder(t):
        if getattr(n, "lineno", lineno) == lineno and isinstance(n, stmt):
            res = n
    return res
