#!/usr/bin/env python3

from . import global_state

# Am I running from an interactive environment?
try:
    get_ipython
    global_state.interactive = True
except:
    pass

from .decorators import func
from .headers import setNotebookName
from .experiment import Experiment

__all__ = ["func", "setNotebookName", "Experiment"]