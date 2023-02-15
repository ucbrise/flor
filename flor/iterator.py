import json
import os
from typing import Iterable, List, Union
import pandas as pd

from flor import flags
from flor.shelf import home_shelf as shelf, cwd_shelf
from flor.skipblock import SkipBlock

from flor.constants import *
from flor.utils import *
from pathlib import Path
import numpy as np

from flor.utils import gen_commit2tstamp_mapper
from flor.state import State


runtime_initialized = False


def it(value: Union[Iterable, bool]):
    """
    Main loop wrapper
    :param value:
        Iterable when iterating over a for loop
        Bool when looping with while
    """
    assert isinstance(value, (Iterable, bool))
    if flags.NAME is None:
        if isinstance(value, bool):
            return value
        else:
            assert isinstance(value, Iterable)
            for each in value:
                yield each
            return

    _deferred_init()

    if not flags.REPLAY:
        # Record mode
        if isinstance(value, bool):
            return value
        else:
            for each in value:
                yield each
    else:
        # Replay mode
        segment = SkipBlock.journal.get_segment_window()
        for capsule in segment:
            flags.RESUMING = capsule.init_only
            if isinstance(value, bool):
                yield True
            else:
                assert isinstance(value, Iterable)
                if flags.RESUMING:
                    if capsule.epoch is None:
                        continue
                    else:
                        State.epoch = value[capsule.epoch]  # type: ignore
                        assert hasattr(
                            value, "__getitem__"
                        ), "TODO: Implement next() calls to consume iterator"
                        yield value[capsule.epoch]  # type: ignore
                else:
                    assert capsule.epoch is not None
                    State.epoch = value[capsule.epoch]  # type: ignore
                    assert hasattr(
                        value, "__getitem__"
                    ), "TODO: Implement next() calls to consume iterator"
                    yield value[capsule.epoch]  # type: ignore


def _deferred_init(_nil=[]):
    """
    At most once execution
    """
    global runtime_initialized
    if not runtime_initialized:
        assert flags.NAME is not None
        if not flags.REPLAY and flags.MODE is None:
            assert (
                cwd_shelf.in_shadow_branch()
            ), f"Please run FLOR from a shadow branch (branch name: `{SHADOW_BRANCH_PREFIX}.[...]`)\nso we may commit dirty pages automatically"
        SkipBlock.bind()
        if flags.REPLAY:
            SkipBlock.journal.read()
        else:
            index_path = shelf.get_index()
            SkipBlock.logger.set_path(index_path)
            assert SkipBlock.logger.path is not None
        runtime_initialized = True


def load_kvs():
    """
    TODO: Move to other file
    """
    with open(REPLAY_JSON, "r", encoding="utf-8") as f:
        d = json.load(f)

    p = Path.home()
    p = p / ".flor"
    p = p / d["NAME"]  # type: ignore
    p = p / "replay_jsons"

    seq = []

    for q in p.iterdir():
        # q will contain the timestamp: 2022-02-07T20:42:25.json
        tstamp = q.stem
        # 2022-02-07T20:42:25
        with open(str(q), "r", encoding="utf-8") as f:
            d = json.load(f)

        _kvs = d["KVS"]

        for k in _kvs:
            if len(k.split(".")) >= 3:
                z = k.split(".")
                e = z.pop(0)
                r = z.pop(0)
                n = ".".join(z)
                for s, x in enumerate(_kvs[k]):
                    # pvresnx
                    seq.append((d["NAME"], d["MEMO"], tstamp, r, e, s, n, x))

    df1 = pd.DataFrame(
        seq,
        columns=["projid", "vid", "tstamp", "alpha", "epoch", "step", "name", "value"],
        # dtype=(str, str, np.datetime64, str, int, int, str, object),
    ).astype(
        {
            "projid": str,
            "vid": str,
            "tstamp": np.datetime64,
            "alpha": str,
            "epoch": int,
            "step": int,
            "name": str,
            "value": object,
        }
    )
    # TODO: RESUME
    time2sha, sha2time = gen_commit2tstamp_mapper()

    df1["vid"] = df1["vid"].apply(lambda x: time2sha.get(os.path.basename(x), x))

    return df1.sort_values(by=["tstamp", "epoch", "step"])


__all__ = ["it"]
