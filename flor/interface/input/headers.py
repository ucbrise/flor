#!/usr/bin/env python3
from flor import global_state

def setNotebookName(name):
    """
    Manually set the name of the notebook
    """
    global_state.nb_name = name
