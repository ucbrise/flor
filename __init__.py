#!/usr/bin/env python3
import requests
from . import global_state

# Am I running from an interactive environment?
try:
    get_ipython
    global_state.interactive = True
except:
    pass


######################### GROUND GROUND GROUND ###################################################
# Is Ground Server initialized?
# Localhost hardcoded into the url
try:
    requests.get('http://localhost:9000')
except:
    # No, Ground not initialized
    raise requests.exceptions.ConnectionError('Please start Ground first')
######################### </> GROUND GROUND GROUND ###################################################

from .decorators import func
from .headers import setNotebookName
from .headers import versionSummaries
from .headers import diffExperimentVersions
from .headers import checkoutArtifact
from  .headers import run
from .experiment import Experiment

__all__ = ["func", "setNotebookName", "diffExperimentVersions", "run",
           "checkoutArtifact", "versionSummaries", "Experiment"]
