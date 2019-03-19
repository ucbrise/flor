#!/usr/bin/env python3

import logging

logging.basicConfig(format='%(name)-12s: %(levelname)-8s %(message)s',level=logging.WARNING)

from flor.face_library.flog import Flog
from flor.face_user.open_log import OpenLog

__all__ = ['OpenLog', 'Flog']
