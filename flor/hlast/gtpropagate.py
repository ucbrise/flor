#!/usr/bin/env python

from argparse import Namespace
from ast import AST, Name, parse, unparse, walk, stmt, literal_eval
from collections import defaultdict
from copy import deepcopy
import ast

from .gumtree import GumTree, python


def propagate(args: Namespace):
    tree, target = [parse(f.read()) for f in (args.source, args.target)]
    replicate(tree, find(tree, lineno=args.lineno), target, **args.gumtree)
    with open(args.out, "w") as f:
        print(unparse(target), file=f)


def replicate(tree: AST, node: AST, target: AST, **kwargs):
    adapter = python.Adapter(tree, target)
    mapping = GumTree(adapter, **kwargs).mapping(tree, target)
    assert tree == adapter.root(node) and isinstance(node, stmt)

    if node in mapping:
        raise FileExistsError("Nothing to do")

    block, index = find_insert_loc(adapter, node, mapping)
    new = make_contextual_copy(adapter, node, mapping)
    block.insert(index, new)  # type: ignore


def find_insert_loc(adapter, node, mapping):
    parent = adapter.parent(node)

    # Find the context by checking the mapping of sibling nodes
    context = None
    for sibling in adapter.children(parent):
        if id(sibling) == id(node):
            break
        if sibling in mapping:
            context = sibling

    # Determine the block and index for insertion based on the context found
    if context is not None:
        ref = mapping[context]
        block = adapter.parent(ref)
        index = 1 + block.index(ref)
    elif parent in mapping:
        block = mapping[parent]
        index = 0
    else:
        raise ValueError("Unable to map context!")  # Raise a specific exception

    assert block is not None
    return block, index


def make_contextual_copy(adapter, node, mapping):
    renames = defaultdict(lambda: defaultdict(int))
    for source, target in mapping.items():
        if isinstance(source, ast.Name) and not adapter.contains(source, node):
            renames[source.id][target.id] += 1

    new_node = deepcopy(node)

    for n in ast.walk(new_node):
        if isinstance(n, ast.Name) and n.id in renames:
            most_common_target_id = max(
                renames[n.id].items(), key=lambda item: item[1]
            )[0]
            n.id = most_common_target_id

    return new_node


class FindNodeVisitor(ast.NodeVisitor):
    def __init__(self, lineno):
        self.lineno = lineno
        self.result = None

    def generic_visit(self, node):
        if getattr(node, "lineno", None) == self.lineno:
            self.result = node
        else:
            super().generic_visit(node)


def find(t: AST, lineno: int):
    visitor = FindNodeVisitor(lineno)
    visitor.visit(t)
    assert visitor.result is not None

    return visitor.result
