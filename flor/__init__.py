import sys

from flor.constants import *
from flor.initializer import initialize, is_initialized
import flor.utils as utils
from flor.transformer import Transformer
from flor.parallelizer import partition as part

import torch
from torch import cuda

try:
    if cuda.is_available() and not cuda.is_initialized():
        cuda.init()
except:
    if cuda.is_available() and not torch.distributed.is_initialized():
        cuda.init()

class NotInitializedError(RuntimeError):
    pass


def foo(*args, **kwargs):
    raise NotInitializedError("[FLOR] Missing experiment name, mode, and memo.")


class NullClass:
    def __init__(self, *args, **kwargs):
        raise NotInitializedError("[FLOR] Missing experiment name, mode, and memo.")

    @staticmethod
    def new():
        raise NotInitializedError("[FLOR] Missing experiment name, mode, and memo.")

    @staticmethod
    def peek():
        raise NotInitializedError("[FLOR] Missing experiment name, mode, and memo.")

    @staticmethod
    def pop():
        raise NotInitializedError("[FLOR] Missing experiment name, mode, and memo.")

    @staticmethod
    def test_force(*args):
        raise NotInitializedError("[FLOR] Missing experiment name, mode, and memo.")


pin_state = foo
random_seed = foo
flush = foo
partition = foo
get_epochs = foo
sample = foo
SKIP = False
PID = None
NPARTS = None
RATE = None
SkipBlock = NullClass

namespace_stack = NullClass
skip_stack = NullClass

user_settings = None

if [each for each in sys.argv if '--flor' == each[0:len('--flor')]]:
    # Fetch the flags we need without disrupting user code
    flor_settings = {
        'mode': ['exec', 'reexec'], # default: exec
        'predinit': ['weak', 'strong'],
        'name': ANY,
        'memo': ANY,
        'maxb': ANY,  # buffer limit
        'rd': ANY,     # root directory for .flor subdir,
        'pid': ANY,     # partition id, for parallelism
        'ngpus': ANY,   # num_gpus, for parallelism
        'rate': ANY     # sampling rate
    }

    argvs = []
    flor_arg = None
    for each in sys.argv:
        if '--flor' != each[0:len('--flor')]:
            argvs.append(each)
        else:
            flor_arg = each.split('=')[1]
            assert flor_arg != '', "[FLOR] Enter a setting and value: {}".format(flor_settings)
    sys.argv = argvs

    user_settings = {}

    # Validate the user entered valid settings
    flor_arg = flor_arg.split(',')
    flor_arg = [each.split(':') for each in flor_arg]
    for (k, v) in flor_arg:
        assert k in flor_settings, "[FLOR] Invalid setting: {}".format(k)
        assert flor_settings[k] is ANY or v in flor_settings[k], "[FLOR] Invalid value for setting `{}`. Value must be one of {}".format(k, flor_settings[k])
        assert k not in user_settings, "[FLOR] Duplicate setting entered: {}".format(k)
        user_settings[k] = v

    # Check that required flags are set
    assert 'name' in user_settings, "[FLOR] Missing required parameter: name."

    initialize(**user_settings)

def it(iterator_or_predicate):
    assert is_initialized(), "Please initialize flor first"
    from . import stateful as flags

    if isinstance(iterator_or_predicate, bool):
        predicate = iterator_or_predicate
        if flags.MODE is REEXEC and flags.PID is not None:
            assert flags.NPARTS is not None
            assert flags.iterations_count >= 0
            for _ in part(range(flags.iterations_count), flags.PID, flags.NPARTS):
                yield True
            return False
        else:
            return predicate
    else:
        iterator = iterator_or_predicate
        if flags.MODE is REEXEC and flags.PID is not None:
            assert flags.NPARTS is not None
            for each in part(iterator, flags.PID, flags.NPARTS):
                yield each
        else:
            for each in iterator:
                yield each
    from flor.writer import flush
    if flags.MODE is EXEC:
        flush()

__all__ = ['pin_state',
           'random_seed',
           'flush',
           'SKIP',
           'SkipBlock',
           'initialize',
           'is_initialized',
           'user_settings',
           'utils',
           'Transformer',
           'partition',
           'get_epochs',
           'PID',
           'NPARTS',
           'RATE',
           'sample'
           ]
