import os
import flor

from flor.constants import *
import flor.stateful as flags
from datetime import datetime
import flor.utils as utils


def initialize(name, mode='exec', memo=None, maxb=5000):
    """
    Flor won't work properly unless these values are set correctly
    :param name:
    :param mode:
    :param memo:
    :return:
    """
    assert flags.NAME is None, "[FLOR] initialized more than once"
    assert mode in ['exec', 'reexec'], "[FLOR] Invalid Mode"
    buffer_limit = int(maxb)
    flags.NAME = name
    flags.LOG_PATH = os.path.join(os.path.expanduser('~'), '.flor', flags.NAME,
                                  "{}.json".format(datetime.now().strftime("%Y%m%d-%H%M%S")))
    flags.LOG_DATA_PATH = os.path.join(os.path.expanduser('~'), '.flor', flags.NAME, "data")

    if mode == 'reexec':
        assert memo is not None, "[FLOR] On Re-execution, please specify a memoized file"
        flags.MEMO_PATH = os.path.join(os.path.expanduser('~'), '.flor', flags.NAME, memo)
        assert os.path.exists(flags.MEMO_PATH)
        flags.MODE = REEXEC

    utils.cond_mkdir(os.path.join(os.path.expanduser('~'), '.flor'))
    utils.cond_mkdir(os.path.join(os.path.expanduser('~'), '.flor', flags.NAME))
    utils.cond_mkdir(flags.LOG_DATA_PATH)

    # FINISH INITIALIZATION
    from flor.writer import Writer, pin_state, random_seed, flush
    from flor.skipblock.skip_block import SkipBlock
    from flor.skipblock.namespace_stack import NamespaceStack
    from flor.skipblock.skip_stack import SkipStack

    Writer.max_buffer = buffer_limit

    flor.SKIP = flags.MODE is REEXEC
    flor.pin_state = pin_state
    flor.random_seed = random_seed
    flor.SkipBlock = SkipBlock
    flor.flush = flush

    flor.namespace_stack = NamespaceStack
    flor.skip_stack = SkipStack


def is_initialized():
    return flags.NAME is not None
