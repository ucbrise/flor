# type: ignore

from ast import AST, iter_fields
from typing import Any, Iterable, Optional, Union

from .adapter import BaseAdapter, memoize, materialize

# https://github.com/PyCQA/pylint/issues/3882
# pylint: disable=unsubscriptable-object

from ast import expr_context, boolop, operator, unaryop, cmpop

Enums = (expr_context, boolop, operator, unaryop, cmpop)


Node = Union[AST, list]


class Adapter(BaseAdapter):
    def __init__(self, *roots: Node):
        """
        DFS tree generator
        implements:
          self[Tree].children(Node)
          self[Tree].label(Node)
          self[Tree[.value(Node)
        """

        self._parents = {}
        for root in roots:
            self._save_parents(root)

    def parent(self, n: Node) -> Optional[Node]:
        return self._parents[id(n)]

    def _save_parents(self, node, parent=None):
        assert id(node) not in self._parents
        self._parents[id(node)] = parent
        for child in self.children(node):
            self._save_parents(child, parent=node)

    @memoize
    @materialize
    def children(self, n: Node) -> Iterable[Node]:
        if isinstance(n, AST):
            it = iter_fields(n)
        if isinstance(n, list):
            it = enumerate(n)

        for _, value in it:
            if isinstance(value, (AST, list)) and not isinstance(value, Enums):
                yield value

    @memoize
    def label(self, n: Node) -> str:
        return type(n).__name__

    @memoize
    def value(self, n: Node) -> Any:
        terminals = []
        if isinstance(n, AST):
            for name, value in iter_fields(n):
                if (
                    isinstance(value, list)
                    and value
                    and all(isinstance(e, Enums) for e in value)
                ):
                    terminals.append((name, value))
                if (
                    value is not None
                    and not isinstance(value, (AST, list))
                    or isinstance(value, Enums)
                ):
                    terminals.append((name, value))
        return terminals
