from typing import Dict, Any
from flor.logger.checkpoint_logger import Logger
from flor.logger.future import Future
from copy import deepcopy
import json


from flor.logger import exp_json, log_records
from flor.state import State
from flor import flags


def log(name, value):
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


def pinned(name, callback, *args):
    if not flags.REPLAY:
        value = callback(*args)
        if flags.NAME:
            exp_json.put(name, value, ow=False)
    else:
        value = exp_json.get(name)
    return value
