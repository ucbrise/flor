from flor.skipblock import SkipBlock
from flor.constants import *

from flor.state import State
from flor.iterator import it

from time import time
from typing import Iterable, Any


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
    def loop(iter8r, name=None, probed=None) -> Iterable[Any]:
        """
        Commits after every outer loop
        """
        try:
            State.loop_nesting_level += 1
            assert State.loop_nesting_level >= 1
            static_id = f"""{'outer loop' if State.loop_nesting_level == 1 else 'nested loop'}"""
            name = str(static_id) if name is None else name
            if State.loop_nesting_level == 1:
                # Outer loop
                assert State.import_time is not None
                State.seconds["PREP"] = time() - State.import_time
                State.seconds["EPOCHS"] = []
                State.epoch = 0

                for each in it(iter8r):
                    start_time = time()
                    State.step = 0
                    State.epoch += 1
                    yield each
                    State.seconds["EPOCHS"].append(time() - start_time)
                State.seconds["EVAL"] = time()
            else:
                assert State.loop_nesting_level > 1
                # Nested loop
                if SkipBlock.step_into(name, probed):
                    assert State.step is not None
                    try:
                        while True:
                            each = next(iter8r)
                            State.step += 1
                            yield each
                    except StopIteration:
                        pass
                    finally:
                        SkipBlock.end(*MTK.chckpts)
                else:
                    SkipBlock.end(*MTK.chckpts)

        finally:
            State.loop_nesting_level -= 1
