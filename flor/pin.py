from . import flags
from .shelf import get_pkl_ref
from .skipblock.seemblock import SeemBlock as sb
from .iterator import replay_clock
from typing import Any, Dict, TypeVar, Union, Tuple, List
import cloudpickle

LITERALS = Union[int, float, bool, str]
T = TypeVar("T", int, float, bool, str)
kvs: Dict[str, List[LITERALS]] = dict()
anti_kvs = dict()


def _get_key(name):
    return f"{sb.journal.get_iterations_count() if not flags.REPLAY else replay_clock.epoch}.{'b' if flags.REPLAY else 'a'}.{name}"  # TODO: Debug get iterations count


def _swap(c):
    e, r, n = c.split(".")
    r = "a" if r != "a" else "b"
    return ".".join((e, r, n))


def _saved_kvs_pop(k):
    if k not in anti_kvs:
        anti_kvs[k] = [
            kvs[k].pop(0),
        ]
    else:
        anti_kvs[k].append(kvs[k].pop(0))
    return anti_kvs[k][-1]


def pkl(value: T) -> str:
    p = str(get_pkl_ref())
    with open(p, "wb") as f:
        cloudpickle.dump(value, f)
    return p


def pin(name: str, value: T) -> T:
    """
    Replay uses historical values
    """
    if flags.NAME is None:
        return value
    k = _get_key(name)
    if flags.REPLAY:
        assert _swap(k) in kvs
        return _saved_kvs_pop(_swap(k))  # type: ignore
    else:
        if k in kvs:
            kvs[k].append(value)
        else:
            kvs[k] = [
                value,
            ]
        return value


def log(name: str, value):
    """
    Record & Replay grow the kvs written to .replay.json
    """
    if flags.NAME is None:
        # FLOR must be enabled
        return value
    # we share the same kvs, but use it differently
    k = _get_key(name)
    # log, unlike pin, is write-only
    if k in kvs:
        kvs[k].append(value)
    else:
        kvs[k] = [
            value,
        ]
    return value
