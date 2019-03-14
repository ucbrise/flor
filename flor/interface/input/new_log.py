from flor.global_state import *
import flor.global_state
import json
import os, shutil
import datetime

class Flog:

    def __init__(self):
        self.writer = open(self.__get_current__(), 'a')

    def write(self, s):
        self.writer.write(json.dumps(s) + '\n')
        self.writer.flush()

    @staticmethod
    def __get_current__():
        name = os.listdir(FLOR_CUR).pop()
        return os.path.join(FLOR_DIR, name, 'log.json')

    @staticmethod
    def flagged():
        return not not os.listdir(FLOR_CUR)


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