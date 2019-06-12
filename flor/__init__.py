#!/usr/bin/env python3

import logging
import os

logging.basicConfig(format='%(name)-12s: %(levelname)-8s %(message)s',level=logging.WARNING)

from flor.face_library.flog import Flog
from flor.utils import cond_mkdir, refresh_tree
from flor.constants import *
from flor.__main__ import install

cond_mkdir(FLOR_DIR)
refresh_tree(FLOR_CUR)

__all__ = ['Flog', 'install']

if not os.path.exists(os.path.join(FLOR_DIR, '.conda_map')):
    print("Flor hasn't been installed.")
    print("From Python: You may run the function flor.install()")
    print("From CLI: You may run the pyflor_install script")