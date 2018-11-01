#!/usr/bin/env python3
from flor import global_state
from flor.context.struct import Struct

def setNotebookName(name):
    global_state.nb_name = name

def internal_log(v, d):
    struct = Struct.from_dict(d)
    print("v: {}".format(v))
    print("struct: {}".format(struct))
    print()

    return v
