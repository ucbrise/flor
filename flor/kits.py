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


class DPK:
    """
    DATA PREP KIT
    """

    load_kvs = load_kvs
    next_id = 0
    lbracket = None

    @staticmethod
    def checkpoints(*args):
        if flags.NAME is not None and not flags.REPLAY:
            """
            RECORD-only
                For replay we skip execution and just load checkpoint
            """
            static_id = f"{stack()[1].lineno}@{stack()[1].filename}"
            start_time = time.time()
            if DPK.lbracket is None:
                _deferred_init()
            DPK.lbracket = Bracket(
                static_id,
                DPK.next_id,
                LBRACKET,
                predicate=True,
                timestamp=start_time,
            )
            SkipBlock.journal.as_tree().feed_entry(DPK.lbracket)
            SkipBlock.logger.append(DPK.lbracket)

            for a in args:
                data_record = DPK._val_to_record(a, static_id)
                SkipBlock.journal.as_tree().feed_entry(data_record)  # type: ignore
                SkipBlock.logger.append(data_record)
            DPK.next_id += 1

    @staticmethod
    def commit():
        if DPK.lbracket is not None:
            assert DPK.lbracket.timestamp
            commit_sha, index_path = _close_record()
            print(f"committed {commit_sha[0:6]}... at {index_path}")
            print(
                f"---------------- {time.time() - DPK.lbracket.timestamp} ---------------------"
            )

    @staticmethod
    def _val_to_record(arg, static_id):
        my_lsn = DPK.next_id

        if type(arg) in [type(None), int, float, bool, str]:
            return DataVal(static_id, my_lsn, arg)
        elif isinstance(arg, pd.DataFrame):
            return DataFrame(static_id, my_lsn, arg)
        else:
            if hasattr(arg, "state_dict"):
                try:
                    return Torch(static_id, my_lsn, arg.state_dict())  # type: ignore
                except:
                    pass
            return DataRef(static_id, my_lsn, arg)


if __name__ == "__main__":
    for epoch in MTK.loop(range(5)):
        for batch in MTK.loop(range(10)):
            pass
