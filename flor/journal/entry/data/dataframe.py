import json
from pathlib import PurePath
from typing import Union
import uuid

import pandas as pd

from flor.shelf import home_shelf

from ..constants import *
from .abstract import Data


class DataFrame(Data):
    def __init__(self, sk, gk, v=None, r: Union[None, PurePath] = None):
        assert bool(v is not None) != bool(r is not None)
        super().__init__(sk, gk, v)
        self.ref = r
        self.val_saved = v is None and r is not None

    def make_val(self):
        assert self.ref is not None
        self.value = pd.read_csv(self.ref)

    def would_mat(self):
        """
        For timing serialization costs
        """
        return

    def jsonify(self):
        assert self.ref is not None
        assert (
            self.val_saved and self.ref.suffix == ".csv"
        ), "Must call DataFrame.set_ref_and_dump(...) before jsonify()"
        d = super().jsonify()
        del d[VAL]
        d["csv_ref"] = str(self.ref)
        return d

    def set_ref(self, csv_ref: PurePath):
        self.ref = csv_ref

    def dump(self):
        assert self.value is not None
        assert (
            isinstance(self.ref, PurePath) and self.ref.suffix == ".csv"
        ), "Must first set a reference path with a `.csv` suffix"
        self.value.to_csv(self.ref)
        self.val_saved = True
        self.value = None

    def set_ref_and_dump(self, csv_ref: PurePath):
        self.set_ref(csv_ref)
        self.dump()

    @staticmethod
    def is_superclass(json_dict: dict):
        return "csv_ref" in json_dict or (
            "ref" in json_dict and PurePath(json_dict["ref"]).suffix == ".csv"
        )

    @classmethod
    def cons(cls, json_dict: dict):
        return cls(
            json_dict[STATIC_KEY], json_dict[GLOBAL_KEY], v=None, r=json_dict["csv_ref"]
        )

    def promise(self):
        ref = home_shelf.get_csv_ref()
        assert ref is not None
        self.set_ref_and_dump(ref)
        self.promised = self.jsonify()

    def fulfill(self):
        super().fulfill()
        return json.dumps(self.promised)
