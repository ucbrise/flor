import json
from copy import deepcopy
from pathlib import PurePath
from typing import Union

import cloudpickle

from flor.shelf import home_shelf

from ..constants import *
from .abstract import Data


class Reference(Data):
    def __init__(self, sk, gk, v=None, r: Union[None, PurePath] = None):
        assert bool(v is not None) != bool(r is not None)
        super().__init__(sk, gk, v)
        self.ref = r
        self.val_saved = v is None and r is not None

    def make_val(self):
        assert self.ref is not None
        with open(self.ref, "rb") as f:
            self.value = cloudpickle.load(f)

    def would_mat(self):
        """
        For timing serialization costs
        """
        cloudpickle.dumps(self.value)

    def jsonify(self):
        assert self.ref is not None
        assert (
            self.val_saved and self.ref.suffix == PKL_SFX
        ), "Must call Reference.set_ref_and_dump(...) before jsonify()"
        d = super().jsonify()
        del d[VAL]
        d[REF] = str(self.ref)
        return d

    def set_ref(self, pkl_ref: PurePath):
        self.ref = pkl_ref

    def dump(self):
        assert (
            isinstance(self.ref, PurePath) and self.ref.suffix == PKL_SFX
        ), "Must first set a reference path with a `.pkl` suffix"
        with open(self.ref, "wb") as f:
            cloudpickle.dump(self.value, f)
        self.val_saved = True
        self.value = None

    def set_ref_and_dump(self, pkl_ref: PurePath):
        self.set_ref(pkl_ref)
        self.dump()

    @staticmethod
    def is_superclass(json_dict: dict):
        return REF in json_dict

    @classmethod
    def cons(cls, json_dict: dict):
        return cls(
            json_dict[STATIC_KEY], json_dict[GLOBAL_KEY], v=None, r=json_dict[REF]
        )

    def promise(self):
        self.promised = deepcopy(self.value)
        self.value = self.promised

    def fulfill(self):
        super().fulfill()
        ref = home_shelf.get_pkl_ref()
        assert ref is not None
        self.set_ref_and_dump(ref)
        return json.dumps(self.jsonify())
