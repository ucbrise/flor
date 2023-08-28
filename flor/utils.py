import json
from typing import Any
from pathlib import Path

import pandas as pd

import warnings

warnings.filterwarnings('ignore', category=UserWarning, message='Could not infer format.*')



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
    return Path("_".join(rolling_name)).with_suffix(ext)


def split_and_retrieve_elements(array, count=10):
    # Find the middle index
    middle_index = len(array) // 2

    # Split the array into two halves
    left_half = array[:middle_index]
    right_half = array[middle_index:]

    # Retrieve the first 10 elements from the left half and last 10 elements from the right half
    first_10_left = left_half[:count]
    last_10_right = right_half[-count:]

    return first_10_left, last_10_right

def is_integer(string):
    try:
        int(string)
        return True
    except ValueError:
        return False

def cast_dtypes(df: pd.DataFrame, columns=None):
    for col in (columns if columns is not None else df.columns):
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='ignore')
        if df[col].dtype == 'object':
            df[col] = pd.to_datetime(df[col], errors='ignore')
    return df