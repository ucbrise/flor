import json
from typing import Any
from pathlib import Path

import pandas as pd

import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, message="Could not infer format.*"
)


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


def duck_cast(v: str, default: Any):
    if isinstance(default, bool):
        return v.lower() in ("yes", "y", "true", "t", "1")
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
        return f"{', '.join([f'{k}: {v}' if v is not None else f'{k}: {i}' for k,(i,v) in src.items()])}, {name}: {str(value)}"
    else:
        return f"{name}: {str(value)}"


def to_filename(src, name, ext):
    rolling_name = [
        name,
    ]
    for k, (i, v) in src.items():
        rolling_name.append(str(k))
        if v is not None:
            rolling_name.append(str(v))
        else:
            rolling_name.append(str(i))
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
    target_columns = columns if columns is not None else df.columns
    for col in target_columns:
        if df[col].dtype == "object":
            # Attempt to convert to numeric first
            try:
                df[col] = pd.to_numeric(df[col], errors="raise")
            except ValueError:
                # If it fails, try coerce (turns non-numeric into NaN)
                pass

            # After numeric attempt, if still object, try datetime
            if df[col].dtype == "object":
                try:
                    df[col] = pd.to_datetime(df[col], errors="raise")
                except ValueError:
                    # If datetime also fails, revert to no conversion or coerce
                    pass
    return df


def latest(df: pd.DataFrame):
    # return df where tstamp is the latest tstamp (and only the latest tstamp)
    if not df.empty:
        return df[df["tstamp"] == df["tstamp"].max()]
    return df


def discretize(cost_estimate: float):
    if cost_estimate < 10:
        return "under 10 seconds"
    elif cost_estimate < 100:
        return "under 2 minutes"
    elif cost_estimate < 1000:
        return "up to 15 minutes"
    else:
        return "more than 15 minutes"
