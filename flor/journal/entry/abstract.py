from .constants import *
from abc import ABC, abstractmethod

import json


class Entry(ABC):
    next_lsn = 0

    def __init__(self, sk, gk):
        self.sk = sk
        self.gk = gk
        self.lsn = Entry.next_lsn
        Entry.next_lsn += 1

    def jsonify(self):
        d = dict()
        d[STATIC_KEY] = str(self.sk)
        d[GLOBAL_KEY] = int(self.gk)
        d[GLOBAL_LSN] = int(self.lsn)
        return d

    @abstractmethod
    def is_left(self):
        ...

    @abstractmethod
    def is_right(self):
        ...

    @staticmethod
    @abstractmethod
    def is_superclass(json_dict: dict):
        ...

    @classmethod
    @abstractmethod
    def cons(cls, json_dict: dict):
        ...
