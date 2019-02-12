#!/usr/bin/env python3

import logging

logging.basicConfig(format='%(name)-12s: %(levelname)-8s %(message)s',level=logging.WARNING)

import flor.global_state as global_state

try:
    get_ipython()
    from time import strftime
    global_state.interactive = True
    from flor.util import get_notebook_name
    from os import path
    global_state.log_name = path.basename(get_notebook_name())
    while global_state.log_name == 'Untitled.ipynb':
        ok = input("This notebook has default name Untitled. Please rename before continuing. [ok]")
        global_state.log_name = path.basename(get_notebook_name())
    # TODO: Check that no other log with the same name exists so we don't overwrite
    global_state.log_name = '.'.join(global_state.log_name.split('.')[0:-1]) + '_log.json'
    # global_state.log_name = strftime("%H%M%S_%d%m%Y") + "_log.json"
    from flor.controller.parser import injected
    injected.file = open(global_state.log_name, "a")
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





