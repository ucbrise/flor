#!/usr/bin/env python3

from . import global_state

# Am I running from an interactive environment?
try:
    get_ipython
    global_state.interactive = True
except:
    pass

import warnings
with warnings.catch_warnings():
    from .decorators import func
    from .headers import setNotebookName
    from .headers import listVersionSummaries
    from .headers import diffExperimentVersions
    from .headers import materialize
    from .experiment import Experiment

__all__ = ["func", "setNotebookName", "diffExperimentVersions",
           "materialize", "listVersionSummaries", "Experiment"]