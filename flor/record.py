import os
import cloudpickle
import json
from typing import Union, List


STATIC_KEY = 'static_key'
GLOBAL_KEY = 'global_key'
GLOBAL_LSN = 'global_lsn'
VAL = 'value'
REF = 'ref'
LBRACKET = 'LBRACKET'
RBRACKET = 'RBRACKET'
METADATA = 'metadata'
SPARSE_CHECKPOINTS = 'sparse_checkpoints'
ITERATIONS_COUNT = 'iterations_count'


class Record:
    next_lsn = 0

    def __init__(self, sk, gk):
        self.sk = sk
        self.gk = gk
        self.lsn = Record.next_lsn
        Record.next_lsn += 1

    def jsonify(self):
        d = dict()
        d[STATIC_KEY] = str(self.sk)
        d[GLOBAL_KEY] = int(self.gk)
        d[GLOBAL_LSN] = int(self.lsn)
        return d


class DataVal(Record):
    """
    {
        static_key: ...,
        global_key: ...,
        global_lsn: ...,
        value: ...
    }
    """
    def __init__(self, sk, gk, v):
        super().__init__(sk, gk)
        self.value = v

    @staticmethod
    def is_left():
        return False

    @staticmethod
    def is_right():
        return True

    def jsonify(self):
        d = super().jsonify()
        d[VAL] = self.value
        return d

    def make_val(self):
        ...

    def would_mat(self):
        """
        For timing serialization costs
        """
        d = self.jsonify()
        json.dumps(d)

    @staticmethod
    def is_superclass(json_dict):
        assert bool(VAL in json_dict) != bool(REF in json_dict)
        return VAL in json_dict

    @classmethod
    def cons(cls, json_dict):
        return cls(json_dict[STATIC_KEY],
                   json_dict[GLOBAL_KEY],
                   json_dict[VAL])


class DataRef(Record):
    """
    {
        static_key: ...,
        global_key: ...,
        global_lsn: ...,
        ref: ...
    }
    """
    def __init__(self, sk, gk, v=None, r=None):
        assert bool(v is not None) != bool(r is not None)
        super().__init__(sk, gk)
        self.value = v
        self.ref = r

    def set_ref_and_dump(self, pkl_ref: str):
        """
        The caller is responsible for serializing val into ref
        """
        self.ref = pkl_ref
        with open(pkl_ref, 'wb') as f:
            cloudpickle.dump(self.value, f)
        del self.value

    def make_val(self):
        with open(self.ref, 'rb') as f:
            self.value = cloudpickle.load(f)

    @staticmethod
    def is_left():
        return False

    @staticmethod
    def is_right():
        return True

    def would_mat(self):
        """
        For timing serialization costs
        """
        d = super().jsonify()
        cloudpickle.dumps(self.value)
        json.dumps(d)

    def jsonify(self):
        assert (self.ref is not None
                and os.path.splitext(self.ref)[1] == '.pkl'), \
            "Must call DataRef.set_ref_and_dump(...) before Jsonifying"
        d = super().jsonify()
        d[REF] = str(self.ref)
        return d

    @staticmethod
    def is_superclass(json_dict):
        assert bool(VAL in json_dict) != bool(REF in json_dict)
        return REF in json_dict

    @classmethod
    def cons(cls, json_dict):
        return cls(json_dict[STATIC_KEY],
                   json_dict[GLOBAL_KEY],
                   v=None,
                   r=json_dict[REF])


class Metadata(Record):
    def __init__(self, sk, gk, meta):
        super().__init__(sk, gk)
        self.meta = meta

    def jsonify(self):
        d = super().jsonify()
        d[METADATA] = str(self.meta)
        return d


class Bracket(Metadata):
    """
    {
        static_key: ...,
        global_key: ...,
        global_lsn: ...,
        metadata: LBRACKET | RBRACKET
    }
    """
    LEGAL_BRACKETS = [LBRACKET, RBRACKET]

    def __init__(self, sk, gk, mode=None, predicate=None, timestamp=None):
        assert mode in Bracket.LEGAL_BRACKETS
        super().__init__(sk, gk, mode)
        self.predicate = predicate
        self.timestamp = timestamp

    def is_left(self):
        return self.meta == LBRACKET

    def is_right(self):
        return self.meta == RBRACKET

    @staticmethod
    def is_superclass(json_dict):
        return (METADATA in json_dict and
                json_dict[METADATA] in Bracket.LEGAL_BRACKETS)

    @classmethod
    def cons(cls, json_dict):
        return cls(json_dict[STATIC_KEY],
                   json_dict[GLOBAL_KEY],
                   json_dict[METADATA])


class EOF:
    NAME = "EOF"

    def __init__(self, sparse: List[int], itc: int):
        self.sparse_checkpoints = sparse
        self.iterations_count = itc

    def jsonify(self):
        d = dict()
        d[METADATA] = EOF.NAME
        d[SPARSE_CHECKPOINTS] = self.sparse_checkpoints
        d[ITERATIONS_COUNT] = int(self.iterations_count)
        return d

    @staticmethod
    def is_left():
        return False

    @staticmethod
    def is_right():
        return False

    @staticmethod
    def is_superclass(json_dict):
        return (METADATA in json_dict and
                json_dict[METADATA] == EOF.NAME)

    @classmethod
    def cons(cls, json_dict):
        sparse = json_dict[SPARSE_CHECKPOINTS]
        assert isinstance(sparse, List)
        return cls(sparse,
                   int(json_dict[ITERATIONS_COUNT]))


def make_record(json_dict: dict) -> Union[DataRef, DataVal, Bracket, EOF]:
    if METADATA in json_dict:
        # Metadata Record
        if Bracket.is_superclass(json_dict):
            return Bracket.cons(json_dict)
        else:
            assert EOF.is_superclass(json_dict)
            return EOF.cons(json_dict)
    else:
        # Data Record
        if DataVal.is_superclass(json_dict):
            return DataVal.cons(json_dict)
        else:
            assert DataRef.is_superclass(json_dict)
            return DataRef.cons(json_dict)


__all__ = ['DataRef', 'DataVal',
           'Bracket', 'EOF', 'make_record',
           'LBRACKET', 'RBRACKET']
