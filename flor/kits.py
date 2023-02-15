from inspect import stack
import time
from typing import Optional
import pandas as pd

from .iterator import it, load_kvs, Repo, _close_record, _deferred_init
from .skipblock import SkipBlock
from .constants import *

from flor.journal.entry import *
from flor import flags, shelf

from flor.state import State


class MTK:
    """
    MODEL TRAINING KIT
    """

    load_kvs = load_kvs
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
                for each in it(iter8r):
                    assert State.epoch is not None
                    State.epoch += 1
                    yield each
            else:
                assert State.loop_nesting_level > 1
                # Nested loop
                if SkipBlock.step_into(name, probed):
                    for each in iter8r:
                        yield each
                SkipBlock.end(*MTK.chckpts)
        finally:
            State.loop_nesting_level -= 1


if __name__ == "__main__":
    for epoch in MTK.loop(range(5)):
        for batch in MTK.loop(range(10)):
            pass
