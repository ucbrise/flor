from flor.constants import *
from flor.state import State

from flor.shelf import home_shelf, cwd_shelf

import os
import json
import shutil
import pandas as pd
import numpy as np

from git.repo import Repo


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


def unpack():
    with open(REPLAY_JSON, "r") as f:
        name = json.load(f)["NAME"]
    dst = home_shelf.get_job()
    dst.mkdir(exist_ok=True)

    dst = dst / "repo.git"
    if dst.exists():
        shutil.rmtree(dst)

    replay_jsons = home_shelf.get_job() / "replay_jsons"
    if not replay_jsons.exists():
        replay_jsons.mkdir()

    r = State.repo
    assert r is not None
    assert cwd_shelf.in_shadow_branch()

    r.clone(dst)
    r = Repo(dst)
    commits = [c for c in r.iter_commits()]
    cwd = os.getcwd()
    os.chdir(dst)
    active = r.active_branch  # check behavior
    for version in commits:
        r.git.checkout(version)
        hexsha, message = version.hexsha, version.message
        messages = message.split("::")  # type: ignore
        if len(messages) != 2:
            print(f"Did not parse >>{messages}<<")
            continue
        else:
            _, tstamp_json = messages
        try:
            shutil.copy2(".replay.json", os.path.join(replay_jsons, tstamp_json))  # type: ignore
            print(f'copied {(str(version.hexsha) + "::" + str(tstamp_json))}')
        except FileNotFoundError:
            # print(f"version {version.hexsha[0:6]}... does not contain {args.source}")
            continue
        except:
            continue

    r.git.checkout(active)
    os.chdir(cwd)
