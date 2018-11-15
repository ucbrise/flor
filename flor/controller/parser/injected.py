#!/usr/bin/env python3
from flor.context.struct import Struct

# TODO: OUT-OF-CORE. Log could potentially be very big

class StructuredLog:

    def __init__(self):

        # self.log_sequence = None
        self.log_tree = None
        self.dict_of_returns = {}
        self.parents = []

structured_log = StructuredLog()

class FlorEnter:
    pass

class FlorExit:
    pass

def internal_log(v, d):
    d['runtime_value'] = v
    # struct = Struct.from_dict(d)
    structured_log.log_tree['log_sequence'].append(d)
    return v

def log_enter(locl=None, vararg=None, kwarg=None):
    structured_log.parents.append(structured_log.log_tree)
    structured_log.log_tree = {}

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
            for returned_value in structured_log.dict_of_returns:
                func_names = structured_log.dict_of_returns[returned_value]
                consumes = returned_value is arg
                if type(returned_value) == list or type(returned_value) == tuple:
                    consumes = consumes or any([x is arg for x in returned_value])
                if consumes:
                    consumes_from = list(set(["{}".format(each) for each in func_names]))
                    break
            if consumes_from:
                break

    structured_log.log_tree['block_type'] = ''

    if consumes_from:
        structured_log.log_tree["consumes_from"] = consumes_from

    structured_log.log_tree['log_sequence'] = []
    # structured_log.log_sequence = structured_log.log_tree['log_sequence']

    # log_sequence.append(FlorEnter)




def log_exit(v=None, func_name=None):

    if func_name:
        # Return Context
        if v in structured_log.dict_of_returns:
            structured_log.dict_of_returns[v] |= {func_name,}
        else:
            structured_log.dict_of_returns[v] = {func_name,}

        structured_log.log_tree['block_type'] = "function_body :: {}".format(func_name)
    else:
        structured_log.log_tree['block_type'] = "loop_body"

    if not structured_log.log_tree['log_sequence']:
        del structured_log.log_tree['log_sequence']

    parent = structured_log.parents.pop()
    if parent:
        original = structured_log.log_tree
        structured_log.log_tree = parent

        if structured_log.log_tree['log_sequence'] and 'children' in structured_log.log_tree['log_sequence'][-1]:
            structured_log.log_tree['log_sequence'][-1]['children'].append(original)
        else:
            structured_log.log_tree['log_sequence'].append({'children': [original, ]})


    # log_sequence.append(FlorExit)
    return v