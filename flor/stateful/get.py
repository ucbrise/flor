from . import shared

def get(k):
    return getattr(shared.shared_state, k)