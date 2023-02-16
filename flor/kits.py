from inspect import stack

from flor.skipblock import SkipBlock
from flor.constants import *

from flor.state import State
from flor.iterator import it


class MTK:
    """
    MODEL TRAINING KIT
    """

    SkipBlock = SkipBlock
    chckpts = []  # type: ignore

    @staticmethod
    def checkpoints(*args):
        MTK.chckpts.extend([a for a in args])

    @staticmethod
    def loop(iter8r, name=None, probed=None):
        """
        Commits after every outer loop
        """
        try:
            State.loop_nesting_level += 1
            assert State.loop_nesting_level >= 1
            static_id = {
                "name": "outer loop"
                if State.loop_nesting_level == 1
                else "nested loop",
                "lineno": stack()[1].lineno,
                "src": stack()[1].filename,
            }
            name = str(static_id) if name is None else name
            if State.loop_nesting_level == 1:
                # Outer loop
                State.epoch = 0
                for each in it(iter8r):
                    State.step = 0
                    State.epoch += 1
                    yield each
            else:
                assert State.loop_nesting_level > 1
                # Nested loop
                if SkipBlock.step_into(name, probed):
                    assert State.step is not None
                    for each in iter8r:
                        State.step += 1
                        yield each
                SkipBlock.end(*MTK.chckpts)
        finally:
            State.loop_nesting_level -= 1
