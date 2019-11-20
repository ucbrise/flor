import sys

from flor.constants import *

from flor.initializer import initialize, is_initialized


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
SKIP = False
SkipBlock = NullClass

namespace_stack = NullClass
skip_stack = NullClass

__all__ = ['pin_state', 'random_seed', 'flush', 'SKIP', 'SkipBlock', 'initialize', 'is_initialized', ]

if [each for each in sys.argv if '--flor' == each[0:len('--flor')]]:
    # Fetch the flags we need without disrupting user code
    flor_settings = {
        'mode': ['exec', 'reexec'], # default: exec
        'name': ANY,
        'memo': ANY
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
