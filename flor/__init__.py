#!/usr/bin/env python3

import logging

logging.basicConfig(format='%(name)-12s: %(levelname)-8s %(message)s',level=logging.WARNING)

from flor.face_library.flog import Flog
from flor.utils import cond_mkdir, refresh_tree
from flor.constants import *

cond_mkdir(FLOR_DIR)
refresh_tree(FLOR_CUR)

__all__ = ['Flog',]
