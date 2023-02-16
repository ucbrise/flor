from flor.constants import *
from flor.state import State

import os
import json
import pandas as pd
import numpy as np


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


def gen_commit2tstamp_mapper():
    """
    os.path.basename(absolute)
    """
    assert State.repo is not None
    commits = [
        c
        for c in State.repo.iter_commits()
        if "flor.shadow" in str(c.message) and ".json" == c.message[-len(".json") :]
    ]

    def get_index(message: str):
        return message.split("::")[1]

    sha2time = dict()
    time2sha = dict()

    for c in commits:
        if SHADOW_BRANCH_PREFIX in str(c.message):
            index = get_index(str(c.message))
            sha2time[c.hexsha] = index
            time2sha[index] = c.hexsha

    return (time2sha, sha2time)
