from flor.constants import FLOR_DIR, FLOR_CUR
from flor.face_library.flog import Flog
import os
import shutil
import datetime

class OpenLog:

    def __init__(self, name):
        self.__cond_mkdir__(FLOR_DIR)
        self.__cond_mkdir__(os.path.join(FLOR_DIR, name))
        self.__set_current__(name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        Flog().write({'session_end': format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))})
        self.__clear_current__()


    def __cond_mkdir__(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)

    def __set_current__(self, name):
        """
        Creates a file with the given name
        Used as a means for passing information between processes
        :param name:
        :return:
        """
        if os.path.isdir(FLOR_CUR):
            shutil.rmtree(FLOR_CUR)
        os.mkdir(FLOR_CUR)
        f = open(os.path.join(FLOR_CUR, name), 'a').close()
        Flog().write({'session_start': format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))})

    def __clear_current__(self):
        if os.path.isdir(FLOR_CUR):
            shutil.rmtree(FLOR_CUR)
        os.mkdir(FLOR_CUR)