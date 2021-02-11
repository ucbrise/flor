import sys
from pathlib import PurePath

NAME = None
REPLAY = False
INDEX = 'latest.json'
MODE = 'weak'
PID = (1, 1)


def set_NAME(name: str):
    """
    When set: enables FLOR RECORD
    """
    global NAME
    assert isinstance(name, str)
    NAME = name


def set_REPLAY(index=None, mode=None, pid=None):
    """
    When set: enables FLOR REPLAY
    """
    global REPLAY, INDEX, MODE, PID
    REPLAY = True
    if index is not None:
        assert isinstance(index, str)
        assert PurePath(index).suffix == '.json'
        INDEX = index
    if mode is not None:
        assert mode in ('weak', 'strong')
        MODE = mode
    if pid is not None:
        assert isinstance(pid, tuple) and len(pid) == 2
        p, n = pid
        assert isinstance(p, int) and isinstance(n, int)
        PID = pid


class Parser:
    """
    [--flor NAME [--replay_flor [INDEX.json] [weak | strong] [i/n]]]
    """

    @staticmethod
    def parse_name():
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

        assert len(flor_flags) <= 2
        if flor_flags:
            assert flor_flags.pop(0) == '--flor'
            assert flor_flags, "Missing NAME argument in --flor NAME"
            set_NAME(name=flor_flags.pop(0))

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
                elif flor_flag in ['weak', 'strong']:
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
           'MODE',
           'PID',
           'set_NAME',
           'set_REPLAY',
           'Parser']
