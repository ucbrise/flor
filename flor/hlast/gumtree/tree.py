from dataclasses import dataclass, field
from .adapter import BaseAdapter


@dataclass(frozen=True)
class Node:
    label: str
    value: str = ''
    children: list['Node'] = field(default_factory=list)
    parent: 'Node' = field(init=False, compare=False, repr=False)

    def __post_init__(self):
        for child in self.children:
            object.__setattr__(child, 'parent', self)
        super().__setattr__('parent', None)
    
    def __getitem__(self, i):  # testing
        return self.children[i]

Tree = Node


class Adapter(BaseAdapter[Tree]):
    def parent(self, n: Tree):
        return n.parent

    def children(self, n: Tree):
        return n.children

    def label(self, n: Tree) -> str:
        return n.label

    def value(self, n: Tree) -> str:
        return n.value
    
    def isomorphic(self, n1: Tree, n2: Tree) -> bool:
        return n1 == n2


adapter = Adapter()
