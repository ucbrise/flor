from ..constants import *
from ..abstract import Entry
from ....logger import Future

from abc import ABC
import json


class Metadata(Entry, Future, ABC):
    def __init__(self, sk, gk, meta):
        Entry.__init__(self, sk, gk)
        self.meta = meta

    def jsonify(self):
        d = super().jsonify()
        d[METADATA] = self.meta
        return d

    def promise(self):
        ...

    def fulfill(self):
        return json.dumps(self.jsonify())
