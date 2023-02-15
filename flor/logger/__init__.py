from flor.logger.logger import Logger
from flor.logger.future import Future
from copy import deepcopy

from flor.logger import exp_json
from flor.state import State


def log(name, value, **kwargs):
    if "csv" in kwargs:
        pass
    else:
        # default case, treat as plaintext
        if State.loop_nesting_level:
            pass
        else:
            exp_json.put(name, value, ow=False)
