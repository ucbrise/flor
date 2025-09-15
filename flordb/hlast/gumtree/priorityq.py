from heapq import heapify, heappush, heappop
from typing import Callable, Generic, Iterable, TypeVar
from operator import lt, gt


T, I = TypeVar("T"), TypeVar("I")
Key = Callable[[T], I]


def identity(x):
    return x


class PriorityQ(Generic[T, I]):
    def __init__(self, it: Iterable[T], *, key: Key = identity, reverse=False):
        self._Item = self._itemizer(key, reverse)
        self._heap = list(map(self._Item, it))
        heapify(self._heap)

    def push(self, value: T):
        item = self._Item(value)
        heappush(self._heap, item)

    def pop(self) -> T:
        return heappop(self._heap).value  # type: ignore

    def peek(self) -> T:
        return self._heap[0].value  # type: ignore

    def __len__(self) -> int:
        return len(self._heap)

    def _itemizer(self, key, reverse):
        op = gt if reverse else lt

        class Item:
            def __init__(self, value):
                self.key, self.value = key(value), value

            def __lt__(self, other: "Item"):
                return op(self.key, other.key)

        return Item
