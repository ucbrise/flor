from flor.constants import *
from flor.controller import Controller
import os
import json
import cloudpickle

class Flog:

    def __init__(self):
        self.writer = open(self.__get_current__(), 'a')
        self.controller = Controller()

    def write(self, s):
        #TODO: Can I dump with json rather than dumps
        decision = self.controller.do(s)
        if decision is Exit:
            return True
        self.writer.write(json.dumps(s) + '\n')
        self.flush()
        return True

    def flush(self):
        self.writer.flush()

    def serialize(self, x):
        license = self.controller.get_license_to_serialize()
        if not license:
            return "PASS"
        try:
            # import cloudpickle
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