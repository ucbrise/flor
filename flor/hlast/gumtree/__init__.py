# type: ignore
from apted import APTED, Config
from itertools import product, zip_longest
from collections.abc import Iterable
from typing import Generic, Optional, TypeVar
from bidict._mut import MutableBidict

from .adapter import Adapter
from .idmap import IdMap
from .priorityq import PriorityQ

# https://github.com/PyCQA/pylint/issues/3882
# pylint: disable=unsubscriptable-object


Tree = TypeVar("Tree")


class HeightPQ(PriorityQ[Tree, int]):
    def __init__(self, adapter: Adapter[Tree], it=[]):
        super().__init__(it, key=adapter.height, reverse=True)  # type: ignore
        self.adapter = adapter

    def peek_max(self) -> int:
        return len(self) and self.adapter.height(self.peek())

    def open(self, tree: Tree):
        for child in self.adapter.children(tree):
            self.push(child)

    def pop(self) -> list[Tree]:
        trees = []
        assert self, "Empty!"
        height = self.peek_max()
        while self.peek_max() == height:
            trees.append(super().pop())
        return trees


class Mapping(MutableBidict[Tree, Tree]):  # type: ignore
    _fwdm_cls = IdMap[Tree, Tree]  # type: ignore
    _invm_cls = IdMap[Tree, Tree]  # type: ignore
    _repr_delegate = list

    def __init__(self, adapter: Adapter[Tree], it: Iterable[tuple[Tree, Tree]] = ()):
        self.adapter = adapter
        super().__init__(it)

    def put_tree(self, t1: Tree, t2: Tree):
        self.putall(zip(*map(self.adapter.postorder, [t1, t2])))


class GumTree(Generic[Tree]):

    """

    Fine-grained and accurate source code differencing
    Falleri et al., ASE 2014.

    """

    defaults = {"min_height": 2, "min_dice": 0.50, "max_size": 100}

    def __init__(self, adapter: Adapter[Tree], *, opt=None, **params):
        assert not set(params) - set(self.defaults), "Invalid parameters!"
        self.params = dict(self.defaults, **params)
        self.opt = opt or self.apted
        self.adapter = adapter

    def mapping(self, t1: Tree, t2: Tree) -> Mapping[Tree, Tree]:  # type: ignore
        m = self.topdown(t1, t2)
        self.bottomup(t1, t2, m)
        return m

    def topdown(self, t1: Tree, t2: Tree) -> Mapping[Tree, Tree]:  # type: ignore
        min_height = self.params["min_height"]

        parent = self.adapter.parent
        isomorphic = self.adapter.isomorphic

        def different(l, r):
            return id(l) != id(r)

        adapter = self.adapter
        l1, l2 = HeightPQ(adapter, [t1]), HeightPQ(adapter, [t2])
        a, m = [], Mapping(adapter)

        # Note: Algorithm uses >, but the example seems to use >= instead.
        while max(l.peek_max() for l in [l1, l2]) >= min_height:
            if l1.peek_max() != l2.peek_max():
                pq = max(l1, l2, key=HeightPQ.peek_max)
                for t in pq.pop():
                    pq.open(t)
            else:
                h1, h2 = l1.pop(), l2.pop()
                for n1, n2 in product(h1, h2):
                    if isomorphic(n1, n2):
                        if any(
                            isomorphic(n1, t) for t in h2 if different(t, n2)
                        ) or any(isomorphic(t, n2) for t in h1 if different(t, n1)):
                            a.append((n1, n2))
                        else:
                            m.put_tree(n1, n2)
                for t1 in h1:
                    if t1 not in m:
                        l1.open(t1)
                for t2 in h2:
                    if t2 not in m.inv:
                        l2.open(t2)

        a.sort(key=lambda ts: self.dice(*map(parent, ts), m), reverse=True)  # type: ignore
        for n1, n2 in a:
            # Note: Algorithm removes from A, but I think this is more efficient
            if n1 not in m and n2 not in m.inv:
                m.put_tree(n1, n2)

        return m

    def bottomup(self, t1: Tree, t2: Tree, m: Mapping[Tree, Tree]):  # type: ignore
        min_dice = self.params["min_dice"]
        max_size = self.params["max_size"]

        label = self.adapter.label
        postorder = self.adapter.postorder
        num_descendants = self.adapter.num_descendants

        # FIXME: Paper mentions candidates must have descendants matched, but
        # it was hard to do and if min_dice > 0 they will be dropped anyways.
        assert min_dice > 0

        def candidate(n1: Tree, m: Mapping[Tree]):  # type: ignore
            return max(
                (
                    c2
                    for c2 in postorder(t2)
                    if label(n1) == label(c2) and c2 not in m.inv
                ),
                key=lambda c2: self.dice(n1, c2, m),
                default=None,
            )

        for n1 in postorder(t1):
            if n1 not in m:
                n2 = candidate(n1, m)
                if n2 and self.dice(n1, n2, m) > min_dice:
                    m.put(n1, n2)
                    # Note: Paper mentions removing already matched descendants
                    if max(num_descendants(t) for t in [n1, n2]) < max_size:
                        for ta, tb in self.opt(n1, n2):
                            if (
                                ta is not None
                                and tb is not None
                                and ta not in m
                                and tb not in m.inv
                                and label(ta) == label(tb)
                            ):
                                m.put(ta, tb)

    def dice(self, t1: Tree, t2: Tree, m: Mapping[Tree]):  # type: ignore
        num_descendants = self.adapter.num_descendants
        descendants = self.adapter.descendants
        contains = self.adapter.contains

        # Note: Formula is unclear, I think this is what they meant ¯\_(ツ)_/¯
        return (
            2
            * sum(1 for n1 in descendants(t1) if n1 in m and contains(m[n1], t2))
            / (num_descendants(t1) + num_descendants(t2) or 1)
        )

    def apted(self, t1: Tree, t2: Tree):
        return APTED(t1, t2, AptedConfig(self.adapter)).compute_edit_mapping()


class AptedConfig(Config):
    def __init__(self, adapter):
        self.adapter = adapter

    def rename(self, n1: Tree, n2: Tree):
        return int(self.adapter.label(n1) != self.adapter.label(n2))

    def children(self, n: Tree):
        return list(self.adapter.children(n))
