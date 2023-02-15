from typing import Dict
from flor.logger.logger import Logger
from flor.logger.future import Future
from copy import deepcopy

from flor.logger import exp_json, log_records
from flor.logger.csv import CSV_Writer
from flor.state import State

csv_writers: Dict[str, CSV_Writer] = {}

import atexit


def log(name, value, **kwargs):
    if "csv" in kwargs:
        if not csv_writers.get(name, False):
            csv_writers[name] = CSV_Writer(name, kwargs["csv"])
            atexit.register(csv_writers[name].flush)
    else:
        # default case, treat as plaintext
        if State.loop_nesting_level:
            log_records.put(name, value)
        else:
            exp_json.put(name, value, ow=False)
        return value
