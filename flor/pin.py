from . import flags
from typing import Any, Dict, TypeVar, Union

LITERALS = Union[int, float, bool, str]
T = TypeVar("T", int, float, bool, str)
kvs: Dict[str, LITERALS] = dict()


def pin(name: str, value: T) -> T:
    """
    Helper method for pinning random number generator seeds
    """
    if flags.NAME is None:
        return value
    if flags.REPLAY:
        assert name in kvs
        assert type(value) == type(kvs[name])
        return kvs[name]  # type: ignore
    else:
        assert name not in kvs
        kvs[name] = value
        return value
