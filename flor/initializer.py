import os
import flor

from flor.constants import *
from . import stateful as flags
from datetime import datetime
import flor.utils as utils


def initialize(name, mode='exec', memo=None, maxb=None, rd=None, predinit='weak'):
    """
    Flor won't work properly unless these values are set correctly
    :param name:
    :param mode:
    :param memo:
    :param predinit: Whether to use weakk or strong predecessor initialization for Parallel/Sampling replay
    :return:
    """
    assert flags.NAME is None, "[FLOR] initialized more than once"
    assert mode in ['exec', 'reexec'], "[FLOR] Invalid Mode"
    assert predinit in ['weak', 'strong'], "[FLOR] Invalid Predecessor Initialization mode"
    root_path = rd
    flor_path = utils.PATH(root_path, '.flor')

    flags.NAME = name
    flags.LOG_PATH = utils.PATH(root_path, os.path.join('.flor', flags.NAME,
                                             "{}.json".format(datetime.now().strftime("%Y%m%d-%H%M%S"))))

    flags.LOG_DATA_PATH = utils.PATH(root_path, os.path.join('.flor', flags.NAME, "data"))

    if mode == 'reexec':
        assert memo is not None, "[FLOR] On Re-execution, please specify a memoized file"
        flags.MEMO_PATH = utils.PATH(root_path, os.path.join('.flor', flags.NAME, memo))
        assert os.path.exists(flags.MEMO_PATH.absolute), f"{flags.MEMO_PATH.absolute} does not exist"
        flags.MODE = REEXEC
        flags.PRED_INIT_MODE = WEAK if predinit == 'weak' else STRONG

    utils.cond_mkdir(flor_path.absolute)
    utils.cond_mkdir(os.path.join(flor_path.absolute, flags.NAME))
    utils.cond_mkdir(flags.LOG_DATA_PATH.absolute)

    # FINISH INITIALIZATION
    from flor.writer import Writer, pin_state, random_seed, flush
    from flor.skipblock.skip_block import SkipBlock
    from flor.skipblock.namespace_stack import NamespaceStack
    from flor.skipblock.skip_stack import SkipStack
    import signal

    signal.signal(signal.SIGCHLD, signal.SIG_IGN)

    Writer.initialize()
    if maxb is not None:
        Writer.max_buffer = int(maxb)

    flor.SKIP = flags.MODE is REEXEC
    flor.pin_state = pin_state
    flor.random_seed = random_seed
    flor.SkipBlock = SkipBlock
    flor.flush = flush

    flor.namespace_stack = NamespaceStack
    flor.skip_stack = SkipStack

    from flor.parallelizer import parallelize
    flor.parallelize = parallelize

    flor.get_epochs = lambda : int(Writer.stateful_adaptive_ext['iterations_count'])

def is_initialized():
    return flags.NAME is not None
