from flor.constants import *
import os
import json
import pickle as cloudpickle
from datetime import datetime

class Flog:

    serializing = False
    depth_limit = None

    xp_name = None
    log_path = None

    lsn = 0

    state = Play

    pinned_state = []
    seeds = []

    def __init__(self, indexed_log=False):
        if indexed_log:
            if Flog.state is Play:
                log_path = os.path.join(os.path.dirname(Flog.log_path), "temp.json")
                self.writer = open(log_path, 'a')
        else:
            self.writer = open(Flog.log_path, 'a')

    def write(self, s):
        s['global_lsn'] = Flog.lsn
        self.writer.write(json.dumps(s) + '\n')
        self.writer.flush()
        Flog.lsn += 1
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

    def read_next(self, mode):
        if mode == 'pin_state':
            return Flog.pinned_state.pop(0)
        elif mode == 'random_seed':
            return Flog.seeds.pop(0)
        else:
            raise RuntimeError()

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

    @staticmethod
    def scan_indexed_log():
        temp_log_path = os.path.join(os.path.dirname(Flog.log_path), "temp.json")
        perm_log_path = os.path.join(os.path.dirname(Flog.log_path), "{}.json".format(datetime.now().strftime("%Y%m%d-%H%M%S")))
        os.rename(temp_log_path, perm_log_path)
        with open(perm_log_path, 'r') as f:
            for line in f:
                log_record = json.loads(line.strip())
                if 'source' in log_record:
                    if log_record['source'] == 'pin_state':
                        Flog.pinned_state.append(cloudpickle.loads(eval(log_record['state'])))
                    elif log_record['source'] == 'random_seed':
                        Flog.seeds.append(log_record['seed'])
