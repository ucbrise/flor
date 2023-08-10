import json
from typing import Any


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


def duck_cast(v: str, default: Any):
    if isinstance(default, bool):
        return bool(v)
    if isinstance(default, int):
        return int(v)
    elif isinstance(default, float):
        return float(v)
    elif isinstance(default, str):
        return str(v) if v else ""
    elif isinstance(default, list):
        return list(v) if v else []
    elif isinstance(default, tuple):
        return tuple(v) if v else tuple([])
    else:
        raise TypeError(f"Unsupported type: {type(default)}")


def add2copy(src, name, value):
    d = dict(src)
    d[name] = value
    return d
