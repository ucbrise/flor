#!/usr/bin/env python3
from flor.context.struct import Struct

def internal_log(v, d):
    struct = Struct.from_dict(d)
    print("v: {}".format(v))
    print("struct: {}".format(struct))
    print()

    return v

def log_enter():
    print("ENTER")

def log_exit(v=None):
    print("EXIT")
    return v