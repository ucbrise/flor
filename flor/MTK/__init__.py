from inspect import stack
from .iterator import it, load_kvs, report_end as commit, replay_clock
from .skipblock import SkipBlock
from flor.utils import flags

"""
MODEL TRAINING KIT
"""

_nesting_lvl = 0
_chckpts = []  # type: ignore


def checkpoints(*args):
    _chckpts.extend(list(args))


def loop(iter8r, name=None, probed=None):
    """
    Commits after every outer loop
    """
    global _nesting_lvl
    try:
        _nesting_lvl += 1
        assert _nesting_lvl >= 1
        static_id = {
            "name": "outer loop" if _nesting_lvl == 1 else "nested loop",
            "lineno": stack()[1].lineno,
            "src": stack()[1].filename,
        }
        name = str(static_id) if name is None else name
        if _nesting_lvl == 1:
            # Outer loop
            for each in it(iter8r):
                replay_clock.epoch += 1
                yield each
        else:
            assert _nesting_lvl > 1
            # Nested loop
            if SkipBlock.step_into(name, probed):
                for each in iter8r:
                    yield each
            SkipBlock.end(*_chckpts)
    finally:
        _nesting_lvl -= 1


flags.Parser.parse()
