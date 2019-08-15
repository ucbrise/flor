from flor.constants import *
import os
import json
import pickle as cloudpickle

class Flog:

    """
    This class is instantiated only within a function

    ...
    That's a problem, we need this class, or equivalent
    To be instantiated in the header of client-side code
    even outside the scope of the function

    What behavior do we care about?

    """

    serializing = False
    depth_limit = None
    xp_name = None

    def __init__(self, init_in_func_ctx=True):
        """
        We have to read the current name of the experiment
        The log.json file in the corresponding directory need not exist in advance
        The Controller initialization
            Reads and modifies the depth limit automatically
            Because we assume we're in the context of a function

        Recommended Correction:
        Flagging -- On initialization, parameterize on context of initialization.

        Modification preserves intended behavior in previous context
        Generalizes
        On context outside function, no modification of depth limit

        """
        self.init_in_func_ctx = init_in_func_ctx
        self.writer = open(self.__get_current__(), 'a')

        if Flog.depth_limit is not None and init_in_func_ctx:
            Flog.depth_limit -= 1

    def __do__(self, d):
        prev_depth_limit = Flog.depth_limit
        if 'end_function' in d:
            if Flog.depth_limit is not None:
                Flog.depth_limit += 1
        if prev_depth_limit is not None and prev_depth_limit < 0:
            return Exit
        else:
            return Continue

    def __get_license_to_serialize__(self):
        return Flog.depth_limit is None or Flog.depth_limit >= 0

    def write(self, s):
        #TODO: Can I dump with json rather than dumps
        if self.init_in_func_ctx:
            decision = self.__do__(s)
            if decision is Exit:
                return False
        self.writer.write(json.dumps(s) + '\n')
        self.flush()
        return True

    def flush(self):
        self.writer.flush()

    def serialize(self, x, name: str = None):
        # We need a license because Python evaluates arguments before calling a function
        if self.init_in_func_ctx:
            license = self.__get_license_to_serialize__()
            if not license:
                return "PASS"
        try:
            Flog.serializing = True
            out = str(cloudpickle.dumps(x))
            return out
        except:
            return "ERROR: failed to serialize"
        finally:
            Flog.serializing = False

    @staticmethod
    def __get_current__():
        name = os.listdir(FLOR_CUR).pop()
        return os.path.join(FLOR_DIR, name, 'log.json')

    @staticmethod
    def flagged(option: str = None):
        experiment_is_active = not not os.listdir(FLOR_CUR)
        if not experiment_is_active:
            return False
        if Flog.serializing:
            # Stack overflow avoidance
            return False