from ..constants import *
from .abstract import Metadata

from typing import List


class EOF(Metadata):
    def __init__(self, sparse: List[int], itc: int, commit_sha: str):
        super().__init__(None, None, EOF_NAME)
        self.sparse_checkpoints = sparse
        self.iterations_count = itc
        self.commit_sha = commit_sha

    def is_left(self):
        return False

    def is_right(self):
        return False

    def jsonify(self):
        d = dict()
        d[METADATA] = EOF_NAME
        d[SPARSE_CHECKPOINTS] = self.sparse_checkpoints
        d[ITERATIONS_COUNT] = int(self.iterations_count)
        d["COMMIT_SHA"] = self.commit_sha
        return d

    @staticmethod
    def is_superclass(json_dict: dict):
        return METADATA in json_dict and json_dict[METADATA] == EOF_NAME

    @classmethod
    def cons(cls, json_dict: dict):
        sparse = json_dict[SPARSE_CHECKPOINTS]
        assert isinstance(sparse, List)
        return cls(sparse, int(json_dict[ITERATIONS_COUNT]), json_dict["COMMIT_SHA"])
