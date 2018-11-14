#!/usr/bin/env python3
from flor.context.struct import Struct

# TODO: OUT-OF-CORE. Log could potentially be very big
log_sequence = []

class FlorEnter:
    pass

class FlorExit:
    pass

def internal_log(v, d):
    d['runtime_value'] = v
    struct = Struct.from_dict(d)
    log_sequence.append(struct)
    return v

def log_enter():
    log_sequence.append(FlorEnter)

def log_exit(v=None):
    log_sequence.append(FlorExit)
    return v