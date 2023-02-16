from typing import Dict, Any
from flor.logger.checkpoint_logger import Logger
from flor.logger.future import Future
from copy import deepcopy

from flor.logger import exp_json, log_records
from flor.logger.csv import CSV_Writer
from flor.state import State
from flor import flags

import atexit

csv_writers: Dict[str, CSV_Writer] = {}


def log(name, value, **kwargs):
    if "csv" in kwargs or name in csv_writers:
        if flags.NAME and flags.DATALOGGING:
            if not csv_writers.get(name, False):
                csv_writers[name] = CSV_Writer(name, kwargs["csv"])
            assert name in csv_writers
            csv_writers[name].put(value)
        return value
    else:
        # default case, treat as plaintext
        if State.loop_nesting_level:
            log_records.put(name, value)
        else:
            exp_json.put(name, value, ow=False)
        return value


def pinned(name, callback, *args):
    if not flags.REPLAY:
        value = callback(*args)
        if flags.NAME:
            exp_json.put(name, value, ow=False)
    else:
        value = exp_json.get(name)
    return value


@atexit.register
def flush():
    if flags.NAME:
        for name in csv_writers:
            csv_writers[name].flush()
