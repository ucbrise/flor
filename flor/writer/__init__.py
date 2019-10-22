import numpy, random
import os
import cloudpickle
import copy
import json
from flor.stateful import *
from flor.serial_wrapper import SerialWrapper



class Writer:
    serializing = False
    lsn = 0
    pinned_state = []
    seeds = []
    store_load = []
    max_buffer = 1
    write_buffer = []
    fork_now = False

    if MODE is EXEC:
        fd = open(LOG_PATH, 'w')
    else:
        with open(MEMO_PATH, 'r') as f:
            for line in f:
                log_record = json.loads(line.strip())
                if 'source' in log_record:
                    if log_record['source'] == 'pin_state':
                        pinned_state.append(cloudpickle.loads(eval(log_record['state'])))
                    elif log_record['source'] == 'random_seed':
                        seeds.append(log_record['seed'])
                    elif log_record['source'] == 'store':
                        store_load.append(eval(log_record['value']))

    @staticmethod
    def serialize(obj):
        if not isinstance(obj, (int, float, bool, str)):
            try:
                Writer.serializing = True
                return copy.deepcopy(obj)
            except:
                try:
                    return str(cloudpickle.dumps(obj))
                except:
                    return "ERROR: failed to serialize"
            finally:
                Writer.serializing = False
        else:
            return obj

    @staticmethod
    def serial_serialize(obj):
        try:
            Writer.serializing = True
            out = str(cloudpickle.dumps(obj))
            return out
        except:
            return "ERROR: failed to serialize"
        finally:
            Writer.serializing = False

    @staticmethod
    def write(obj):
        obj['global_lsn'] = Writer.lsn
        Writer.write_buffer.append(obj)
        Writer.lsn += 1
        if len(Writer.write_buffer) >= Writer.max_buffer or Writer.fork_now:
            Writer.fork_now = False
            pid = os.fork()
            if not pid:
                os.nice(1)
                Writer.serializing = True
                for each in Writer.write_buffer:
                    if 'value' in each:
                        if isinstance(each['value'], SerialWrapper):
                            each['value'] = str(cloudpickle.dumps(each['value']))
                    else:
                        if isinstance(each['state'], SerialWrapper):
                            each['state'] = str(cloudpickle.dumps(each['state']))
                    Writer.fd.write(json.dumps(each) + '\n')
                Writer.fd.flush()
                Writer.serializing = False
                os._exit(0)
            else:
                Writer.write_buffer = []

    @staticmethod
    def flush():
        if Writer.write_buffer:
            pid = os.fork()
            if not pid:
                os.nice(1)
                Writer.serializing = True
                for each in Writer.write_buffer:
                    if 'value' in each:
                        if isinstance(each['value'], SerialWrapper):
                            each['value'] = str(cloudpickle.dumps(each['value']))
                    else:
                        if isinstance(each['state'], SerialWrapper):
                            each['state'] = str(cloudpickle.dumps(each['state']))
                    Writer.fd.write(json.dumps(each) + '\n')
                Writer.fd.flush()
                Writer.serializing = False
                os._exit(0)
            else:
                Writer.write_buffer = []


    @staticmethod
    def store(obj):
        # Store the object in the memo
        d = {
            'source': 'store',
            'value': Writer.serialize(obj)
        }
        Writer.write(d)

    @staticmethod
    def load():
        value = Writer.store_load.pop(0)
        return cloudpickle.loads(value)

    @staticmethod
    def pin_state(library):
        if MODE is EXEC:
            if library is numpy:
                d = {'source': 'pin_state',
                     'library': 'numpy',
                     'state': Writer.serial_serialize(library.random.get_state())}
                Writer.write(d)
            elif library is random:
                d = {'source': 'pin_state',
                     'library': 'random',
                     'state': Writer.serial_serialize(library.getstate())}
                Writer.write(d)
            else:
                raise RuntimeError("Library must be `numpy` or `random`, but `{}` was given".format(library.__name__))
        elif MODE is REEXEC:
            state = Writer.pinned_state.pop(0)
            if library is numpy:
                library.random.set_state(state)
            elif library is random:
                library.setstate(state)
            else:
                raise RuntimeError("Library must be `numpy` or `random`, but `{}` was given".format(library.__name__))
        else:
            raise RuntimeError()

    @staticmethod
    def random_seed(*args, **kwargs):
        if MODE is EXEC:
            if args or kwargs:
                seed = numpy.random.randint(*args, **kwargs)
            else:
                seed = numpy.random.randint(0, 2 ** 32)
            d =  {
            'source': 'random_seed',
            'seed': seed
            }
            Writer.write(d)
            return seed
        elif MODE is REEXEC:
            seed = Writer.seeds.pop(0)
            return seed
        else:
            raise RuntimeError()

pin_state = Writer.pin_state
random_seed = Writer.random_seed
store = Writer.store
load = Writer.load

__all__ = ['pin_state', 'random_seed', 'store', 'load', 'flush']