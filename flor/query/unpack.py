from pathlib import PurePath, Path
from typing import Any, Dict
from flor.shelf import cwd_shelf
from flor.state import State
from flor.constants import *
from flor import flags

from flor.query import database

import os
import shutil
import json
import csv
import pandas as pd


def get_stash():
    projid = cwd_shelf.get_projid()
    projid = str(projid).replace("\x1b[m", "")
    out = Path.home() / ".flor" / projid / "stash"
    return out


def clear_stash():
    global stash

    projid = cwd_shelf.get_projid()
    projid = str(projid).replace("\x1b[m", "")

    (Path.home() / ".flor").mkdir(exist_ok=True)

    (Path.home() / ".flor" / projid).mkdir(exist_ok=True)
    stash = ((Path.home() / ".flor") / projid) / "stash"
    if stash.exists():
        shutil.rmtree(stash)
    stash.mkdir()
    return stash


def filtered_versions():
    r = State.repo
    assert r is not None

    all_versions = [version for version in r.iter_commits()]
    flor_versions = [
        version for version in all_versions if "::" in str(version.message)
    ]
    record_versions = [
        version
        for version in flor_versions
        if str(version.message).count("RECORD::") == 1
    ]

    return {"ALL": all_versions, "FLOR": flor_versions, "RECORD": record_versions}


def resolve_cache(cache_short_path):
    assert cwd_shelf.in_shadow_branch()
    return Path.home() / ".flor" / cwd_shelf.get_projid() / cache_short_path


def unpack():
    """
    MAIN FUNCTION
    """
    assert (
        cwd_shelf.in_shadow_branch()
    ), "Please unpack log records from within a `flor.shadow` branch"
    clear_stash()

    r = State.repo
    assert r is not None
    if State.db_conn is None:
        database.start_db(cwd_shelf.get_projid())
    wmrk = database.get_watermark()
    active_branch = State.active_branch
    try:
        commits = filtered_versions()
        for version in commits["RECORD"]:
            if wmrk is not None and version.hexsha in wmrk:
                break
            try:
                # print(f"STEPPING IN {version.hexsha}")
                r.git.checkout(version)
                cp_seconds(version)
                cp_log_records(version)
            except Exception as e:
                print("Line Exception", e)
    finally:
        r.git.checkout(active_branch)


def cp_seconds(version):
    assert State.db_conn is not None
    hexsha, message = version.hexsha, version.message
    if message.count("RECORD::") == 1 and SECONDS_JSON.exists():
        with open(REPLAY_JSON, "r") as f:
            replay_json = json.load(f)
        with open(SECONDS_JSON, "r") as f:
            seconds_json = json.load(f)
        assert isinstance(seconds_json, dict) and seconds_json
        assert "PREP" in seconds_json

        def prep_normalize(prep_secs, eval_secs):
            nonlocal replay_json
            data = []
            data.append(
                {
                    c: v
                    for c, v in zip(
                        list(DATA_PREP) + ["prep_secs", "eval_secs"],
                        [
                            cwd_shelf.get_projid(),
                            replay_json["NAME"],
                            PurePath(replay_json["TSTAMP"]).stem,
                            hexsha,
                            float(prep_secs),
                            float(eval_secs),
                        ],
                    )
                }
            )
            return data

        def outr_normalize(all_epochs_secs):
            nonlocal replay_json
            data = []
            for i, epoch_secs in enumerate(all_epochs_secs):
                epoch = i + 1  # TODO: confirm not off-by-one
                data.append(
                    {
                        c: v
                        for c, v in zip(
                            list(OUTR_LOOP)
                            + [
                                "seconds",
                            ],
                            [
                                cwd_shelf.get_projid(),
                                replay_json["NAME"],
                                PurePath(replay_json["TSTAMP"]).stem,
                                hexsha,
                                int(epoch),
                                float(epoch_secs),
                            ],
                        )
                    }
                )
            return data

        # ..send PREP to data_prep
        pd.DataFrame(
            prep_normalize(
                seconds_json["PREP"],
                seconds_json["EVAL"] if "EVAL" in seconds_json else -1.0,
            )
        ).to_sql("data_prep", con=State.db_conn, if_exists="append", index=False)

        # .. send EPOCHS to outr_loop
        if "EPOCHS" in seconds_json:
            pd.DataFrame(outr_normalize(seconds_json["EPOCHS"])).to_sql(
                "outr_loop", con=State.db_conn, if_exists="append", index=False
            )


def cp_log_records(version):
    assert State.db_conn is not None
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
            df = pd.DataFrame(data)
            df.to_csv(stash / tstamp_json.with_suffix(".csv"), index=False)
            df.to_sql("log_records", con=State.db_conn, if_exists="append", index=False)
    else:
        # Newer Versions, Non-Flor Versions
        replay_json = get_replay_json()
        if replay_json is not None:
            lr_csv = get_log_records_csv()
            tstamp_json = get_tstamp_json(replay_json)
            if tstamp_json is not None:
                data = normalize(replay_json, lr_csv, hexsha, tstamp_json)
                df = pd.DataFrame(data)
                df.to_sql(
                    "log_records", con=State.db_conn, if_exists="append", index=False
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
                "runid": d["NAME"],
                "tstamp": tstamp.stem,
                "vid": hexsha,
                "epoch": -1,
                "step": -1,
                "name": user_var,
                "value": d[user_var],
            }
        )
    if "CLI" in d:
        for user_var in d["CLI"]:
            data.append(
                {
                    "projid": cwd_shelf.get_projid(),
                    "runid": d["NAME"],
                    "tstamp": tstamp.stem,
                    "vid": hexsha,
                    "epoch": -1,
                    "step": -1,
                    "name": user_var,
                    "value": d["CLI"][user_var],
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
                            "runid": d["NAME"],
                            "tstamp": tstamp.stem,
                            "vid": hexsha,
                            "epoch": int(e) if e else -1,
                            "step": int(s) if s else -1,
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
                        "runid": d["NAME"],
                        "tstamp": tstamp.stem,
                        "vid": hexsha,
                        "epoch": int(each["epoch"]) if each["epoch"] else -1,
                        "step": int(each["step"]) if each["step"] else -1,
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
