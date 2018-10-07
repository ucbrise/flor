#!/usr/bin/env python3
import requests
from . import global_state

# Am I running from an interactive environment?
try:
    get_ipython
    global_state.interactive = True
except:
    pass

from .decorators import func, track_action
from .headers import setNotebookName
from .experiment import Experiment

__all__ = ["func", "track_action", "setNotebookName", "Experiment"]
