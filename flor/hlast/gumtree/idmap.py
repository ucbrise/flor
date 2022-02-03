
from collections.abc import MutableMapping
from typing import Generic, TypeVar, Iterable, NamedTuple


K, V = TypeVar('K'), TypeVar('V')


class IdMap(MutableMapping[K, V]):
    class Item(NamedTuple):
        key: K
        value: V

    def __init__(self, it: Iterable[tuple[K, V]] = ()):
        self._map = {}
        for k, v in it:
            self[k] = v

    def __getitem__(self, key: K) -> V:
        return self._map[id(key)].value

    def __setitem__(self, key: K, value: V):
        self._map[id(key)] = self.Item(key, value)

    def __delitem__(self, key: K):
        del self._map[id(key)]

    def __iter__(self) -> Iterable[K]:
        for item in self._map.values():
            yield item.key

    def __len__(self) -> int:
        return len(self._map)

    def items(self) -> Iterable[tuple[K, V]]:
        return self._map.values()