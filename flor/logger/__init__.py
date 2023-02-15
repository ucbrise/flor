from flor.constants import FLORFILE
from flor.logger.logger import Logger
from flor.logger.future import Future
from copy import deepcopy


def log(name, value, **kwargs):
    if "csv" in kwargs:
        pass
    else:
        # default case, treat as txt
        pass
