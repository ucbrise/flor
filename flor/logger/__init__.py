from typing import Dict, Any
from flor.logger.checkpoint_logger import Logger
from flor.logger.future import Future
from copy import deepcopy
import json


from flor.logger import exp_json, log_records
from flor.state import State
from flor import flags
from flor.constants import *


def log(name, value) -> Any:
    def is_jsonable(x):
        try:
            json.dumps(x)
            return True
        except (TypeError, OverflowError):
            return False

    serializable_value = value if is_jsonable(value) else str(value)
    if State.loop_nesting_level:
        log_records.put(name, serializable_value)
    else:
        exp_json.put(name, serializable_value, ow=False)
    return value


def arg(name, default=None) -> Any:
    historical_args = None
    if flags.NAME and flags.REPLAY:
        with open(REPLAY_JSON, "r") as f:
            d = json.load(f)
            if "CLI" in d:
                historical_args = d["CLI"]
    if default is None:
        if historical_args is None or name not in historical_args:
            if name in vars(flags.parser.nsp):
                v = getattr(flags.parser.nsp, name)
                # CLI is logged by EXP_JSON
                return v
            else:
                raise RuntimeError(f"Args without defaults need CLI input: {name}")
        else:
            return historical_args[name]
    else:
        if historical_args is None or name not in historical_args:
            if name in vars(flags.parser.nsp):
                # CLI takes precedence over default
                v = getattr(flags.parser.nsp, name)
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
                elif isinstance(default, bytes):
                    return bytes(v)
                elif isinstance(default, bytearray):
                    return bytearray(v)
                else:
                    raise TypeError(f"Unsupported type: {type(default)}")
            else:
                log(name, default)
                return default
        else:
            v = historical_args[name]
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
            elif isinstance(default, bytes):
                return bytes(v)
            elif isinstance(default, bytearray):
                return bytearray(v)
            else:
                raise TypeError(f"Unsupported type: {type(default)}")


def pinned(name, callback, *args) -> Any:
    if not flags.REPLAY:
        value = callback(*args)
        if flags.NAME:
            exp_json.put(name, value, ow=False)
    else:
        value = exp_json.get(name)
    return value
