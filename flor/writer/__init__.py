import numpy
import pandas as pd
import random
import os
import cloudpickle
import json
from types import SimpleNamespace
import flor
from flor.stateful import *


class Writer:
    serializing = False
    lsn = 0
    pinned_state = []
    seeds = []
    store_load = []
    predicates = []
    condition = True
    collected = []
    columns = []
    flor_vars = SimpleNamespace()

    if MODE is EXEC:
        fd = open(LOG_PATH, 'w')
    elif MODE is REEXEC or MODE is ETL:
        with open(MEMO_PATH, 'r') as f:
            for line in f:
                log_record = json.loads(line.strip())
                if 'source' in log_record:
                    if log_record['source'] == 'pin_state':
                        pinned_state.append(cloudpickle.loads(eval(log_record['state'])))
                    elif log_record['source'] == 'random_seed':
                        seeds.append(log_record['seed'])
                    elif log_record['source'] == 'store':
                        store_load.append((log_record['global_key'], eval(log_record['value'])))
            # We now do a Group By global_key on store_load
            new_store_load = []
            current_group = {'key': None, 'list': None}
            for k, v in store_load:
                if current_group['key'] != k:
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
            out = str(cloudpickle.dumps(obj))
            return out
        except:
            return "ERROR: failed to serialize"
        finally:
            Writer.serializing = False

    @staticmethod
    def write(obj):
        obj['global_lsn'] = Writer.lsn
        Writer.fd.write(json.dumps(obj) + '\n')
        Writer.fd.flush()
        Writer.lsn += 1

    @staticmethod
    def store(obj, global_key):
        # Store the object in the memo
        if MODE is not EXEC:
            return
        d = {
            'source': 'store',
            'global_key': global_key,
            'value': Writer.serialize(obj)
        }
        Writer.write(d)

    @staticmethod
    def load(global_key):
        its_key, values = Writer.store_load.pop(0)
        assert its_key == global_key
        return [cloudpickle.loads(v) for v in values]

    @staticmethod
    def eval_pred(pred_str):
        try:
            return eval(pred_str)
        except Exception as e:
            print(e)
            return True

    @staticmethod
    def update_condition():
        if not Writer.predicates:
            # Initialize condition to True
            Writer.condition = True
        else:
            Writer.condition = all([Writer.eval_pred(pred) for pred in Writer.predicates])

    @staticmethod
    def get(expr, name, pred=None, maps=None):
        setattr(Writer.flor_vars, name, expr)
        if type(pred) == str:
            cond(pred)
        Writer.update_condition()
        if Writer.condition:
            Writer.collected.append({name: expr})
            if maps:
                for name in maps:
                    f = maps[name]
                    Writer.collected.append({name: f(expr)})
        return expr

    @staticmethod
    def cond(pred=None):
        if type(pred) == str and str(pred) not in Writer.predicates:
            Writer.predicates.append(str(pred))
            Writer.update_condition()
        elif type(pred) == bool:
            Writer.condition = Writer.condition and pred

    @staticmethod
    def var(name):
        # TODO: default value
        return getattr(Writer.flor_vars, name, None)

    @staticmethod
    def to_rows():
        rows = []
        row = []
        for each in Writer.collected:
            k = list(each.keys()).pop()
            if k not in Writer.columns:
                Writer.columns.append(k)
            if k not in map(lambda x: list(x.keys()).pop(), row):
                row.append(each)
            else:
                rows.append(row)
                new_row = []
                for r in row:
                    if k not in r:
                        new_row.append(r)
                    else:
                        new_row.append(each)
                        break
                row = new_row
        rows.append(row)
        # post-proc
        rows2 = []
        for row in rows:
            row2 = []
            for each in row:
                row2.append(list(each.values()).pop())
            rows2.append(row2)
        return rows2

    @staticmethod
    def to_df():
        rows = Writer.to_rows()
        return pd.DataFrame(rows, columns=Writer.columns)

    @staticmethod
    def export():
        if MODE is ETL:
            Writer.to_df().to_csv(CSV_PATH)

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
        elif MODE is REEXEC or MODE is ETL:
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
            d = {
                'source': 'random_seed',
                'seed': seed
            }
            Writer.write(d)
            return seed
        elif MODE is REEXEC or MODE is ETL:
            seed = Writer.seeds.pop(0)
            return seed
        else:
            raise RuntimeError()


pin_state = Writer.pin_state
random_seed = Writer.random_seed
get = Writer.get
cond = Writer.cond
var = Writer.var
export = Writer.export

__all__ = ['pin_state', 'random_seed', 'get', 'cond', 'export']
