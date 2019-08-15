from flor.constants import *
import os
import json
import pickle as cloudpickle

class Flog:

    serializing = False
    depth_limit = None

    xp_name = None
    log_path = None

    def __init__(self):
        self.writer = open(Flog.log_path, 'a')

    def write(self, s):
        self.writer.write(json.dumps(s) + '\n')
        self.writer.flush()
        return True

    def serialize(self, x, name: str = None):
        try:
            Flog.serializing = True
            out = str(cloudpickle.dumps(x))
            return out
        except:
            return "ERROR: failed to serialize"
        finally:
            Flog.serializing = False

    @staticmethod
    def flagged(option: str = None):
        experiment_is_active = Flog.xp_name is not None
        if not experiment_is_active:
            return False
        if Flog.serializing:
            # Stack overflow avoidance
            return False

        # There is an active experiment and no risk of stack overflow

        depth_limit = Flog.depth_limit

        if option == 'start_function':
            # If flagged() reduces the depth below zero, we don't want the next expression to run
            # So we update the local depth_limit
            if Flog.depth_limit is not None:
                Flog.depth_limit -= 1
                depth_limit = Flog.depth_limit
        elif option == 'end_function':
            # If flagged() increases the depth to zero, this effect should be visible to the next call of flagged() but not this one
            # Since the update should happen after the full evaluation of the boolean expression
            # So we don't update the local depth_limit
            # The guarantee: either all flog statements in a function run or none do.
            if Flog.depth_limit is not None:
                Flog.depth_limit += 1
        return depth_limit is None or depth_limit >= 0

