from ..constants import *
from ..abstract import Entry

from abc import ABC, abstractmethod


class Data(Entry, ABC):
    def __init__(self, sk, gk, v):
        super().__init__(sk, gk)
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
