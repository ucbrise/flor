#!/usr/bin/env python3

import os
import warnings
from tqdm import tqdm
import pandas as pd
import tempfile
import shutil
import dill

from typing import Dict, Union, Optional, List

from . import global_state
from . import util
from flor.stateful import State
from flor.object_model.artifact import Artifact
import flor.above_ground as ag
import subprocess

def setNotebookName(name):
    global_state.nb_name = name
