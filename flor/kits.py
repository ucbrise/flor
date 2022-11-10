from inspect import stack
import pandas as pd

from .iterator import it, load_kvs, report_end, replay_clock
from .skipblock import SkipBlock


class MTK:
    """
    MODEL TRAINING KIT
    """

    nesting_lvl = 0
    load_kvs = load_kvs
    chckpts = []  # type: ignore

    @staticmethod
    def checkpoints(*args):
        MTK.chckpts.extend(list(args))

    @staticmethod
    def loop(iter8r, name=None, probed=None):
        """
        Commits after every outer loop
        """
        try:
            MTK.nesting_lvl += 1
            assert MTK.nesting_lvl >= 1
            static_id = {
                "name": "outer loop" if MTK.nesting_lvl == 1 else "nested loop",
                "lineno": stack()[1].lineno,
                "src": stack()[1].filename,
            }
            name = str(static_id) if name is None else name
            if MTK.nesting_lvl == 1:
                # Outer loop
                for each in it(iter8r):
                    replay_clock.epoch += 1
                    yield each
            else:
                assert MTK.nesting_lvl > 1
                # Nested loop
                if SkipBlock.step_into(name, probed):
                    for each in iter8r:
                        yield each
                SkipBlock.end(*MTK.chckpts)
        finally:
            MTK.nesting_lvl -= 1

    @staticmethod
    def commit():
        report_end()


class DPK:
    """
    DATA PREP KIT
    """

    @staticmethod
    def checkpoints(*args):
        """
        TODO: add dataframe type to Journal Entries
        """
        logger = SkipBlock.logger
        for a in args:
            if isinstance(a, pd.DataFrame):
                ...
            else:
                ...
        report_end()


if __name__ == "__main__":
    for epoch in MTK.loop(range(5)):
        for batch in MTK.loop(range(10)):
            pass
