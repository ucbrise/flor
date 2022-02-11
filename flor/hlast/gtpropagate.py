#!/usr/bin/env python

from argparse import ArgumentParser, FileType, Namespace
from ast import AST, Name, parse, walk, stmt, literal_eval
from astunparse import unparse
from collections import defaultdict
from copy import deepcopy
from typing import TextIO
import sys

from flor.hlast.gumtree import GumTree, Mapping, python

# https://github.com/PyCQA/pylint/issues/3882
# pylint: disable=unsubscriptable-object


def add_arguments(parser: ArgumentParser):
    parser.add_argument("lineno", type=int)
    parser.add_argument("source", type=FileType("r"))
    parser.add_argument("target", type=FileType("r+"))
    parser.add_argument("--out", type=FileType("w"), default=sys.stdout)
    parser.add_argument("--minor", type=int, default=sys.version_info[1])
    parser.add_argument("--gumtree", type=literal_eval, default="{}")
    return parser


def propagate(args: Namespace):
    tree, target = [parse(f.read()) for f in (args.source, args.target)]
    replicate(tree, find(tree, lineno=args.lineno), target, **args.gumtree)  # type: ignore
    print(unparse(target), file=args.out)


def replicate(tree: AST, node: stmt, target: AST, **kwargs):
    """
    First we do code-block alignment using the GumTree
    algorithm from Falleri et al.
    """
    adapter = python.Adapter(tree, target)
    mapping = GumTree(adapter, **kwargs).mapping(tree, target)
    # asserting `tree` is the root of `node` in the `adapter`
    assert tree == adapter.root(node) and isinstance(node, stmt)

    """
    Then we insert the back-propagated statement into the target block
    QUERY:
        Is the back-propagation always intra-block, meaning from
        same block in version v to same block in version ancestor(v)
    """
    if node in mapping:
        # exit("Already in target!")
        print("hello")

    block, index = find_insert_loc(adapter, node, mapping)
    new = make_contextual_copy(adapter, node, mapping)
    block.insert(index, new)  # type: ignore


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


if __name__ == "__main__":
    propagate(add_arguments(ArgumentParser()).parse_args(sys.argv[1:]))
