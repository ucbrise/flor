from . import flags
from .skipblock.seemblock import SeemBlock as sb
from typing import Any, Dict, TypeVar, Union, Tuple, List

LITERALS = Union[int, float, bool, str]
T = TypeVar("T", int, float, bool, str)
kvs: Dict[str, List[LITERALS]] = dict()


def pin(name: str, value: T) -> T:
    """
    KVS: {
        (E,S,name) : [value, ...]
        EPOCH is GLOBAL ITERATIONS INDEX
    }
    """
    if flags.NAME is None:
        return value
    k = f"{sb.journal.get_iterations_count()}.{name}"
    if flags.REPLAY:
        assert k in kvs
        assert type(value) == type(kvs[k])
        return kvs[k].pop(0)  # type: ignore
    else:
        if k in kvs:
            kvs[k].append(value)
        else:
            kvs[k] = [
                value,
            ]
        return value
