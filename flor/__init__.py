#!/usr/bin/env python3
from flor import global_state

# Am I running from an interactive environment?
try:
    from IPython import get_ipython
    get_ipython
    global_state.interactive = True
except:
    pass

from flor.interface.input.decorators import func, track_action
from flor.interface.input.headers import setNotebookName
from flor.interface.input.experiment import Experiment

__all__ = ["func", "track_action", "setNotebookName", "Experiment"]
