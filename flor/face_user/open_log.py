from flor.constants import *
from flor.face_library.flog import Flog
from flor.utils import cond_mkdir, refresh_tree, cond_rmdir
from flor.model import get, put
import os
import datetime
import sys

class OpenLog:

    def __init__(self, name, depth_limit=1):
        cond_mkdir(FLOR_DIR)
        cond_mkdir(os.path.join(FLOR_DIR, name))
        refresh_tree(FLOR_CUR)
        open(os.path.join(FLOR_CUR, name), 'a').close()

        if depth_limit is not None:
            put('depth_limit', depth_limit)

        Flog().write({'session_start': format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))})

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        Flog().write({'session_end': format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))})
        refresh_tree(FLOR_CUR)
        cond_rmdir(MODEL_DIR)