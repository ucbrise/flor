import numpy, random
import os
import uuid
import cloudpickle
import json
from flor.stateful import *

class Writer:
    serializing = False
    lsn = 0
    pinned_state = []
    seeds = []
    store_load = []
    max_buffer = 500
    write_buffer = []

    if MODE is EXEC:
        # fd = open(LOG_PATH, 'w')
        fd = None
    else:
        with open(MEMO_PATH, 'r') as f:
            for line in f:
                log_record = json.loads(line.strip())
                if 'source' in log_record:
                    if log_record['source'] == 'pin_state':
                        pinned_state.append(log_record['state'])  # THIS IS JUST A FILENAME
                    elif log_record['source'] == 'random_seed':
                        seeds.append(log_record['seed'])
                    elif log_record['source'] == 'store':
                        # THIS IS FILENAME, or LBRACK, or ERROR
                        store_load.append((log_record['global_key'], log_record['value']))
            # We now do a Group By global_key on store_load
            new_store_load = []
            current_group = {'key': None, 'list': None}
            for k, v in store_load:
                if current_group['key'] != k or current_group['list'][0] == 'LBRACKET':
                    # New Group
                    new_store_load.append((current_group['key'], current_group['list']))
                    current_group = {'key': k, 'list': []}
                current_group['list'].append(v)
            new_store_load.append((current_group['key'], current_group['list']))
            assert new_store_load.pop(0) == (None, None)

            store_load = new_store_load
            del new_store_load
            del current_group


    @staticmethod
    def serialize(obj):
        try:
            Writer.serializing = True

            # ADD SOME INDIRECTION
            # MAKE THIS INTO INDEX

            while True:
                unique_filename = uuid.uuid4().hex + '.pkl'
                unique_filename = os.path.join(LOG_DATA_PATH, unique_filename)
                if not os.path.exists(unique_filename):
                    break

            with open(unique_filename, 'wb') as f:
                cloudpickle.dump(obj, f)

            return unique_filename
        except:
            return "ERROR: failed to serialize"
        finally:
            Writer.serializing = False

    @staticmethod
    def write(obj):
        obj['global_lsn'] = Writer.lsn
        Writer.write_buffer.append(obj)
        Writer.lsn += 1  # append to buffer and increment lsn
        if len(Writer.write_buffer) >= Writer.max_buffer:
            Writer.forked_write()  # if buffer exceeds a certain size, or fork_now is triggered
            # note: fork_now is there as a mechanism for forcing fork, we aren't using it yet

    @staticmethod
    def forked_write():
        pid = os.fork()
        if not pid:
            path = LOG_PATH.split('.')
            path.insert(-1, str(Writer.lsn))
            path = '.'.join(path)
            fd = open(path, 'w')
            os.nice(1)  # child process gets lower priority and starts flushing
            for each in Writer.write_buffer:
                if 'value' in each and not isinstance(each['value'], str):  # the dict can have 'value' or 'state'
                    each['value'] = Writer.serialize(each['value'])
                fd.write(json.dumps(each) + '\n')
            fd.close()
            os._exit(0)
        else:
            Writer.write_buffer = []  # parent process resets buffer


    @staticmethod
    def flush():
        if Writer.write_buffer:
            Writer.forked_write()  # at the end of flor execution, flushes buffer to disk


    @staticmethod
    def store(obj, global_key):
        # Store the object in the memo
        if obj is not LBRACKET:
            d = {
                'source': 'store',
                'global_key': global_key,
                'value': obj
            }
        else:
            d = {
                'source': 'store',
                'global_key': global_key,
                'value': 'LBRACKET'
            }
        Writer.write(d)

    @staticmethod
    def load(global_key):
        while True:
            its_key, paths = Writer.store_load.pop(0)
            if its_key == global_key:
                break
        # paths can only contain PATHS or ERRORS
        values = []
        for path in paths:
            if '.pkl' not in path:
                # ERROR CASE
                raise RuntimeError("Necessary state corrupted, unrecoverable")
            else:
                # PATH CASE
                with open(path, 'rb') as f:
                    values.append(cloudpickle.load(f))

        return values

    @staticmethod
    def lbrack_load():
        its_key, [v, ] = Writer.store_load.pop(0)
        assert v == 'LBRACKET', str(v)
        return its_key

    @staticmethod
    def pin_state(library):
        if MODE is EXEC:
            if library is numpy:
                d = {'source': 'pin_state',
                     'library': 'numpy',
                     'state': Writer.serialize(library.random.get_state())}
                Writer.write(d)
            elif library is random:
                d = {'source': 'pin_state',
                     'library': 'random',
                     'state': Writer.serialize(library.getstate())}
                Writer.write(d)
            else:
                raise RuntimeError("Library must be `numpy` or `random`, but `{}` was given".format(library.__name__))
        elif MODE is REEXEC:
            path = Writer.pinned_state.pop(0)
            with open(path, 'rb') as f:
                state = cloudpickle.load(f)
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
flush = Writer.flush

__all__ = ['pin_state', 'random_seed', 'Writer', 'flush']

