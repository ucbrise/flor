#!/usr/bin/env python3

import logging

logging.basicConfig(format='%(name)-12s: %(levelname)-8s %(message)s',level=logging.WARNING)

import flor.global_state as global_state

try:
    get_ipython()
    from time import strftime
    global_state.interactive = True
    global_state.log_name = strftime("%H%M%S_%d%m%Y") + "_log.json"
    from flor.controller.parser import injected
    injected.file = open(global_state.log_name, "w")
    logging.debug("Running in interactive mode")
except NameError:
    global_state.interactive = False
    logging.debug("Running in non-interactive mode")

from flor.interface.input.headers import setNotebookName
from flor.controller.parser.injected import internal_log, log_enter, log_exit
from flor.interface.input.logger import log
from flor.interface.input.execution_tracker import track
from flor.interface.input.context import Context


__all__ = ['setNotebookName', 'log', 'track', 'internal_log',
           'Context', 'log_enter', 'log_exit']





