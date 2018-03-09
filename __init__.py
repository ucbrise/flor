#!/usr/bin/env python3
import requests, sys, time
from ground import client
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
    # This code is fitted to the not-checkpointed implementation of Ground server
    # It will not generalize to other or future implementations of Ground
    # But that is ok for now
    gc = client.GroundClient()

    def node(s):
        s = 'jarvis' + s
        try:
            e = gc.get_node(s)
            if e is None:
                gc.create_node(s, s)
        except:
            gc.create_node(s, s)

    def edge(s, t):
        s = 'jarvis' + s
        t = 'jarvis' + t
        try:
            e = gc.get_edge(s+t)
            if e is None:
                gc.create_edge(s + t, s + t, gc.get_node(s).get_id(), gc.get_node(t).get_id())
        except:
            gc.create_edge(s+t, s+t, gc.get_node(s).get_id(), gc.get_node(t).get_id())

    # Let's set up the schema
    [node(s) for s in ['Experiment', 'Trial', 'Literal', 'Artifact', 'Action']]

    edge('Experiment', 'Trial')
    edge('Trial', 'Literal')
    edge('Trial', 'Artifact')
    edge('Literal', 'Action')
    edge('Artifact', 'Action')
    edge('Action', 'Artifact')

    # Ground application layer ready

except:
    # No, Ground not initialized
    raise requests.exceptions.ConnectionError('Please start Ground first')

######################### </> GROUND GROUND GROUND ###################################################

from .decorators import func
from .headers import setNotebookName
from .headers import versionSummaries
from .headers import diffExperimentVersions
from .headers import checkoutArtifact
from .experiment import Experiment
import ray

__all__ = ["func", "setNotebookName", "diffExperimentVersions",
           "checkoutArtifact", "versionSummaries", "Experiment"]
