from ..constants import *
from ..abstract import Entry
from ....logger import Future

from abc import ABC, abstractmethod


class Data(Entry, Future, ABC):
    def __init__(self, sk, gk, v):
        Entry.__init__(self, sk, gk)
        Future.__init__(self, v)
        self.value = v

    def is_left(self):
        return False

    def is_right(self):
        return True

    def jsonify(self):
        d = super().jsonify()
        d[VAL] = self.value
        return d

    @abstractmethod
    def make_val(self):
        ...

    @abstractmethod
    def would_mat(self):
        ...
