#!/usr/bin/env python3
from flor.context.struct import Struct

# TODO: OUT-OF-CORE. Log could potentially be very big
log_sequence = []

dict_of_returns = {}

class FlorEnter:
    pass

class FlorExit:
    pass

def internal_log(v, d):
    d['runtime_value'] = v
    struct = Struct.from_dict(d)
    log_sequence.append(struct)
    return v

def log_enter(locl=None, vararg=None, kwarg=None):
    varargs = []
    kwargs = []

    consumes = False
    consumes_from = None

    if locl is not None:
        if vararg is not None:
            varargs = list(locl[vararg])
            del locl[vararg]
        if kwarg is not None:
            kwargs = list(locl[kwarg].values())
            del locl[kwarg]

        for arg in list(locl.values()) + varargs + kwargs:
            for returned_value in dict_of_returns:
                func_names = dict_of_returns[returned_value]
                consumes = returned_value is arg
                if type(returned_value) == list or type(returned_value) == tuple:
                    consumes = consumes or any([x is arg for x in returned_value])
                if consumes:
                    consumes_from = list(set(["{}".format(each) for each in func_names]))
                    break
            if consumes_from:
                break

    log_sequence.append(FlorEnter)
    if consumes_from:
        log_sequence.append({"consumes_from": consumes_from})



def log_exit(v=None, func_name=None):
    if func_name:
        # Return Context
        if v in dict_of_returns:
            dict_of_returns[v] |= {func_name,}
        else:
            dict_of_returns[v] = {func_name,}
    log_sequence.append(FlorExit)
    return v