import random

def sample(iterator, rate):
    iterator = list(iterator)
    out_iterator = random.sample(iterator, min(len(iterator), round(len(iterator)*rate)))

    import flor
    flor.NPARTS = len(iterator)
    return out_iterator