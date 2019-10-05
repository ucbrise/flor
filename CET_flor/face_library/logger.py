from flor.face_library.flog import Flog, Play, Replay
import numpy as np
import random

def pin_state(library):
    if not Flog.flagged():
        return
    flog = Flog(indexed_log=True)
    if flog.state is Play:
        if library is np:
            d = {'source': 'pin_state',
                'library': 'numpy',
                 'state': flog.serialize(library.random.get_state())}
            flog.write(d)
            flog.writer.close()
        elif library is random:
            d = {'source': 'pin_state',
                'library': 'random',
                 'state': flog.serialize(library.getstate())}
            flog.write(d)
        else:
            raise RuntimeError("Library must be `numpy` or `random`, but `{}` was given".format(library.__name__))
    elif flog.state is Replay:
        state = flog.read_next('pin_state')
        if library is np:
            library.random.set_state(state)
        elif library is random:
            library.setstate(state)
        else:
            raise RuntimeError("Library must be `numpy` or `random`, but `{}` was given".format(library.__name__))
    else:
        raise RuntimeError()

def pin_generator(gen):
    """
    As pin_state
    but for such generators as numpy.random.generator.Generator
    :param gen:
    :return:
    """
    pass

def random_seed(*args, **kwargs):
    if not Flog.flagged():
        return
    flog = Flog(indexed_log=True)
    if flog.state is Play:
        if args or kwargs:
            seed = np.random.randint(*args, **kwargs)
        else:
            seed = np.random.randint(0, 2**32)
        d = {
            'source': 'random_seed',
            'seed': seed
        }
        flog.write(d)
        flog.writer.close()
        return seed
    elif flog.state is Replay:
        seed = flog.read_next('random_seed')
        return seed
    else:
        raise RuntimeError