#!/usr/bin/env python3
from flor import global_state
import sys
import logging
# Am I running from an interactive environment?
try:
    from IPython import get_ipython
    get_ipython
    global_state.interactive = True
except:
    pass

logging.basicConfig(format='%(name)-12s: %(levelname)-8s %(message)s',level=logging.WARNING)

from flor.interface.input.decorators import func, track_action
from flor.interface.input.headers import setNotebookName, internal_log
from flor.interface.input.experiment import Experiment
from flor.interface.input.logger import log
from flor.interface.input.execution_tracker import track_execution
from flor.interface.input.context import Context

__all__ = ["func", "track_action", "setNotebookName",
           "Experiment", "log", "track_execution", 'internal_log',
           'Context']
