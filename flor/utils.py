import json
from typing import Any
from pathlib import Path


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


def to_string(src, name, value):
    if src:
        return (
            f"{', '.join([f'{k}: {v}' for k,v in src.items()])}, {name}: {str(value)}"
        )
    else:
        return f"{name}: {str(value)}"


def to_filename(src, name, ext):
    rolling_name = [
        name,
    ]
    for k, v in src.items():
        rolling_name.append(str(k))
        rolling_name.append(str(v))
    return Path(".".join(rolling_name)).with_suffix(ext)
