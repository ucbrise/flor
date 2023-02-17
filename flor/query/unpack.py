from pathlib import PurePath, Path
from typing import Any, Dict
from flor.shelf import cwd_shelf
from flor.state import State
from flor.constants import *

import os
import shutil
import json
import csv
import pandas as pd


def clear_stash():
    global stash
    (Path.home() / ".flor").mkdir(exist_ok=True)
    (Path.home() / ".flor" / cwd_shelf.get_projid()).mkdir(exist_ok=True)
    stash = Path.home() / ".flor" / cwd_shelf.get_projid() / "stash"
    if stash.exists():
        shutil.rmtree(stash)
    stash.mkdir()


def filtered_versions():
    r = State.repo
    assert r is not None

    all_versions = [version for version in r.iter_commits()]
    flor_versions = [
        version for version in all_versions if "::" in str(version.message)
    ]

    return {"ALL": all_versions, "FLOR": flor_versions}


def unpack():
    """
    MAIN FUNCTION
    """
    assert cwd_shelf.in_shadow_branch()
    clear_stash()

    r = State.repo
    assert r is not None
    active = r.active_branch  # check behavior
    try:
        commits = filtered_versions()
        for version in commits["ALL"]:
            try:
                print(f"STEPPING IN {version.hexsha}")
                r.git.checkout(version)
                cp_log_records(version)
            except Exception as e:
                print(e)
    finally:
        r.git.checkout(active)
        return stash


def cp_log_records(version):
    hexsha, message = version.hexsha, version.message
    if PurePath(message).suffix == ".json":
        # Older  Versions
        messages = message.split("::")
        tstamp_json = messages[-1]
        tstamp_json = PurePath(tstamp_json)
        replay_json = get_replay_json()
        if replay_json is not None:
            lr_csv = get_log_records_csv()
            data = normalize(replay_json, lr_csv, hexsha, tstamp_json)
            pd.DataFrame(data).to_csv(
                stash / tstamp_json.with_suffix(".csv"), index=False
            )
    else:
        # Newer Versions, Non-Flor Versions
        replay_json = get_replay_json()
        if replay_json is not None:
            lr_csv = get_log_records_csv()
            tstamp_json = get_tstamp_json(replay_json)
            data = normalize(replay_json, lr_csv, hexsha, tstamp_json)
            assert tstamp_json is not None
            pd.DataFrame(data).to_csv(
                stash / tstamp_json.with_suffix(".csv"), index=False
            )


def get_replay_json():
    candidate_paths = {">=2.5.1": REPLAY_JSON, "<2.5": Path(".replay.json")}
    for vs, p in candidate_paths.items():
        if p.exists():
            return p
    return None


def get_log_records_csv():
    candidate_paths = {">=2.5.1": LOG_RECORDS, "<2.5": Path("log_records.csv")}
    for vs, p in candidate_paths.items():
        if p.exists():
            return p
    return None


def normalize(replay_json, lr_csv, hexsha, tstamp):
    """
    Draw data from replay_json.kvs, replay_json.data_prep, lr_csv
    """
    assert replay_json is not None
    data = []

    with open(replay_json, "r") as f:
        d: Dict[str, Any] = json.load(f)

    user_vars = [name for name in d if not name.isupper()]
    for user_var in user_vars:
        # append p,v,-1,-1,n,v
        data.append(
            {
                "projid": cwd_shelf.get_projid(),
                "tstamp": tstamp.stem,
                "vid": hexsha,
                "epoch": -1,
                "step": -1,
                "name": user_var,
                "value": d[user_var],
            }
        )
    if "KVS" in d:
        # LEGACY
        _kvs = d["KVS"]

        for k in _kvs:
            if len(k.split(".")) >= 3:
                z = k.split(".")
                e = z.pop(0)
                _ = z.pop(0)
                n = ".".join(z)
                for s, x in enumerate(_kvs[k]):
                    # pvresnx
                    # seq.append((d["NAME"], d["MEMO"], tstamp, r, e, s, n, x))
                    data.append(
                        {
                            "projid": cwd_shelf.get_projid(),
                            "tstamp": tstamp.stem,
                            "vid": hexsha,
                            "epoch": int(e),
                            "step": int(s),
                            "name": n,
                            "value": x,
                        }
                    )
    if lr_csv is not None:
        with open(lr_csv, "r") as f:
            data.extend(
                [
                    {
                        "projid": cwd_shelf.get_projid(),
                        "tstamp": tstamp.stem,
                        "vid": hexsha,
                        "epoch": int(each["epoch"]),
                        "step": int(each["step"]),
                        "name": str(each["name"]),
                        "value": each["value"],
                    }
                    for each in csv.DictReader(f)
                ]
            )
    return data


def get_tstamp_json(replay_json):
    assert replay_json is not None
    with open(replay_json, "r") as f:
        kvs = json.load(f)
    if "TSTAMP" in kvs:
        return PurePath(kvs["TSTAMP"]).with_suffix(".json")
    elif "tstamp" in kvs:
        return PurePath(kvs["tstamp"]).with_suffix(".json")
