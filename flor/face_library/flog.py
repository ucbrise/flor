from flor.constants import FLOR_CUR, FLOR_DIR
import os
import json

class Flog:

    def __init__(self):
        self.writer = open(self.__get_current__(), 'a')

    def write(self, s):
        self.writer.write(json.dumps(s) + '\n')
        self.flush()
        return True

    def flush(self):
        self.writer.flush()

    def serialize(self, x):
        import cloudpickle
        try:
            out = str(cloudpickle.dumps(x))
            return out
        except:
            return "ERROR: failed to serialize"

    @staticmethod
    def __get_current__():
        name = os.listdir(FLOR_CUR).pop()
        return os.path.join(FLOR_DIR, name, 'log.json')

    @staticmethod
    def flagged():
        return not not os.listdir(FLOR_CUR)