from inspect import stack
from time import time
import pandas as pd

from .iterator import it, load_kvs, Repo, replay_clock, _close_record
from .skipblock import SkipBlock
from .constants import *

from flor.journal.entry import *
from flor import flags, shelf


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


class DPK:
    """
    DATA PREP KIT
    """

    load_kvs = load_kvs
    lsn = 0
    initialized = False

    @staticmethod
    def checkpoints(*args):
        if flags.NAME is not None and not flags.REPLAY:
            """
            RECORD-only
                For replay we skip execution and just load checkpoint
            """
            start_time = time.time()
            DPK._defer_init()

            for a in args:
                data_record = DPK._val_to_record(a)
                SkipBlock.journal.as_tree().feed_entry(data_record)  # type: ignore
                SkipBlock.logger.append(data_record)
            commit_sha, index_path = _close_record()
            print(f"committed {commit_sha[0:6]}... at {index_path}")
            print(f"---------------- {time.time() - start_time} ---------------------")

    @staticmethod
    def _defer_init():
        assert flags.NAME is not None
        if not DPK.initialized:
            if not flags.REPLAY and flags.MODE is None:
                repo = Repo()
                assert (
                    SHADOW_BRANCH_PREFIX
                    == repo.active_branch.name[0 : len(SHADOW_BRANCH_PREFIX)]
                ), f"Please run FLOR from a shadow branch (branch name: `{SHADOW_BRANCH_PREFIX}.[...]`)\nso we may commit dirty pages automatically"
            shelf.mk_job(flags.NAME)
            if flags.REPLAY:
                SkipBlock.journal.read()
            else:
                index_path = (
                    flags.INDEX
                    if flags.MODE is RECORD_MODE.chkpt_restore
                    else shelf.get_index()
                )
                SkipBlock.logger.set_path(index_path)

            DPK.initialized = True

    @staticmethod
    def _val_to_record(arg):
        static_id = f"{stack()[1].lineno}@{stack()[1].filename}"
        my_lsn = DPK.lsn
        DPK.lsn += 1

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
