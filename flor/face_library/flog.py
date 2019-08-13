from flor.constants import *
from .controller import Controller
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
        self.controller = Controller(init_in_func_ctx)

        # variable below is used by the controller,
        # to avoid infinite recursions due to repeated calls to __get_state__
        self.block_succeeded = False

    def write(self, s):
        #TODO: Can I dump with json rather than dumps
        if self.init_in_func_ctx:
            decision = self.controller.do(s)
            if decision is Exit:
                return False
        self.writer.write(json.dumps(s) + '\n')
        self.flush()
        return True

    def flush(self):
        self.writer.flush()

    def serialize(self, x, name: str = None):
        # We need a license because Python evaluates arguments before calling a function
        reset_succeeded = False
        try:
            if name == "self":
                reset_succeeded = self.controller.cond_reset()
            if self.init_in_func_ctx:
                license = self.controller.get_license_to_serialize()
                if not license:
                    return "PASS"
            try:
                out = str(cloudpickle.dumps(x))
                return out
            except:
                return "ERROR: failed to serialize"
        finally:
            self.controller.unreset(reset_succeeded)

    def block_recursive_serialization(self):
        self.block_succeeded = self.controller.cond_inf_recursion_block()
    
    def unblock_recursive_serialization(self):
        self.controller.inf_recursion_unblock(self.block_succeeded)

    @staticmethod
    def __get_current__():
        name = os.listdir(FLOR_CUR).pop()
        return os.path.join(FLOR_DIR, name, 'log.json')

    @staticmethod
    def flagged(option: str = None):
        if option == 'nofork':
            return not not os.listdir(FLOR_CUR)
        return not not os.listdir(FLOR_CUR)