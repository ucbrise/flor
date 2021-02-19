from ..constants import *
from ..abstract import Entry

from abc import ABC


class Metadata(Entry, ABC):
    def __init__(self, sk, gk, meta):
        super().__init__(sk, gk)
        self.meta = meta

    def jsonify(self):
        d = super().jsonify()
        d[METADATA] = self.meta
        return d
