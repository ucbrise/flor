from flor import shelf

import sys
from pathlib import PurePath

NAME = None
REPLAY = False
INDEX = 'latest.json'
WEAK = 'weak'
STRONG = 'strong'
MODE = WEAK
PID = (1, 1)
EPSILON = 1/15
RESUMING = False

"""
[--flor NAME [EPSILON] [--replay_flor [INDEX.json] [weak | strong] [i/n]]]
"""


def set_REPLAY(index=None, mode=None, pid=None):
    """
    When set: enables FLOR REPLAY
    """
    global REPLAY, INDEX, MODE, PID
    REPLAY = True
    if index is not None:
        assert isinstance(index, str)
        assert PurePath(index).suffix == '.json'
        assert shelf.verify(PurePath(index))
        INDEX = index
    if mode is not None:
        assert mode in (WEAK, STRONG)
        MODE = mode
    if pid is not None:
        assert isinstance(pid, tuple) and len(pid) == 2
        p, n = pid
        assert isinstance(p, int) and isinstance(n, int)
        assert p >= 1
        assert n >= 1
        assert p <= n
        PID = pid


class Parser:
    """
    [--flor NAME [EPSILON] [--replay_flor [INDEX.json] [weak | strong] [i/n]]]
    """

    @staticmethod
    def parse_name():
        global NAME, EPSILON
        flor_flags = []
        feeding = False
        for _ in range(len(sys.argv)):
            arg = sys.argv.pop(0)

            if arg == '--flor':
                feeding = True
            elif arg[0] == '-':
                feeding = False

            if feeding:
                flor_flags.append(arg)
            else:
                sys.argv.append(arg)

        assert len(flor_flags) <= 3
        if flor_flags:
            assert flor_flags.pop(0) == '--flor'
            assert flor_flags, "Missing NAME argument in --flor NAME"
            for flag in flor_flags:
                if flag[0:2] == '0.':
                    EPSILON = float(flag)
                else:
                    NAME = flag

    @staticmethod
    def parse_replay():
        flor_flags = []
        feeding = False
        for _ in range(len(sys.argv)):
            arg = sys.argv.pop(0)

            if arg == '--replay_flor':
                feeding = True
            elif arg[0] == '-':
                feeding = False

            if feeding:
                flor_flags.append(arg)
            else:
                sys.argv.append(arg)

        assert len(flor_flags) <= 4
        if flor_flags:
            assert flor_flags.pop(0) == '--replay_flor'
            index = None
            mode = None
            pid = None
            for flor_flag in flor_flags:
                if PurePath(flor_flag).suffix == '.json':
                    index = flor_flag
                elif flor_flag in (WEAK, STRONG):
                    mode = flor_flag
                elif len(flor_flag.split('/')) == 2:
                    p, n = flor_flag.split('/')
                    pid = int(p), int(n)
                else:
                    raise RuntimeError("Invalid argument passed to --replay_flor" +
                                       " [NAME.json] [weak | strong] [I/N]")
            set_REPLAY(index=index, mode=mode, pid=pid)


__all__ = ['NAME',
           'REPLAY',
           'INDEX',
           'WEAK',
           'STRONG',
           'MODE',
           'PID',
           'EPSILON',
           'set_REPLAY',
           'Parser']
