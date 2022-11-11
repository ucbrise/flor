from ..constants import *
from .abstract import Data

import json


class Value(Data):
    def __init__(self, sk, gk, v):
        super().__init__(sk, gk, v)

    def would_mat(self):
        """
        For timing serialization costs
        """
        d = self.jsonify()
        json.dumps(d)

    def make_val(self):
        pass

    @staticmethod
    def is_superclass(json_dict: dict):
        return VAL in json_dict

    @classmethod
    def cons(cls, json_dict: dict):
        return cls(json_dict[STATIC_KEY], json_dict[GLOBAL_KEY], json_dict[VAL])

    def promise(self):
        self.promised = self.value

    def fulfill(self):
        super().fulfill()
        return json.dumps(self.jsonify())
