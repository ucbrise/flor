from ..constants import *
from .abstract import Metadata

VALID_BRACKETS = (LBRACKET, RBRACKET)


class Bracket(Metadata):
    def __init__(self, sk, gk, meta, predicate=None, timestamp=None):
        assert meta in VALID_BRACKETS
        super().__init__(sk, gk, meta)
        self.predicate = predicate
        self.timestamp = timestamp

    def is_left(self):
        return self.meta == LBRACKET

    def is_right(self):
        return self.meta == RBRACKET

    @staticmethod
    def is_superclass(json_dict: dict):
        return METADATA in json_dict and json_dict[METADATA] in VALID_BRACKETS

    @classmethod
    def cons(cls, json_dict: dict):
        return cls(json_dict[STATIC_KEY], json_dict[GLOBAL_KEY], json_dict[METADATA])
