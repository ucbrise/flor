from pathlib import Path, PurePath
import sys

REPLAY = False
MEMO = 'latest.json'
MODE = 'weak'
PID = (1, 1)

_p = Path.home()
_p = _p / '.flor'
_p.mkdir(exist_ok=True)
del _p

_flor_flags = []
_feeding = False
for _ in range(len(sys.argv)):
    arg = sys.argv.pop(0)

    if arg == '--replay_flor':
        _feeding = True
    elif arg[0] == '-':
        _feeding = False

    if _feeding:
        _flor_flags.append(arg)
    else:
        sys.argv.append(arg)
del _feeding, _, arg

assert len(_flor_flags) <= 4
if _flor_flags:
    assert _flor_flags.pop(0) == '--replay_flor'
    REPLAY = True
    for _flor_flag in _flor_flags:
        if PurePath(_flor_flag).suffix == '.json':
            MEMO = _flor_flag
        elif _flor_flag in ['weak', 'strong']:
            MODE = _flor_flag
        elif len(_flor_flag.split('/')) == 2:
            _pid, _np = _flor_flag.split('/')
            PID = (int(_pid), int(_np))
            del _pid, _np
        else:
            raise RuntimeError("Invalid argument passed to --replay_flor" +
                               " [NAME.json] [weak | strong] [I/N]")
del _flor_flag, _flor_flags

__all__ = ['REPLAY', 'MEMO', 'MODE', 'PID']

