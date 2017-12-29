#!/usr/bin/env python3

from . import global_state

# Am I running from an interactive environment?
try:
    get_ipython
    global_state.interactive = True
except:
    pass

from .decorators import func
from .headers import *

