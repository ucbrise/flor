from . import shared

def put(k, v):
    setattr(shared.shared_state, k, v)

