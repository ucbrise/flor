#!/usr/bin/env python3
import requests
from . import global_state

# Am I running from an interactive environment?
try:
    get_ipython
    global_state.interactive = True
except:
    pass

from .decorators import func
from .headers import setNotebookName
from .headers import versionSummaries
from .headers import diffExperimentVersions
from .headers import checkoutArtifact
from .headers import run
from .headers import fork
from .experiment import Experiment

__all__ = ["func", "setNotebookName", "diffExperimentVersions", "run", "fork",
           "checkoutArtifact", "versionSummaries", "Experiment"]
