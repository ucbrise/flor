import json
import os
from typing import Iterable, List, Union
import pandas as pd

from git.exc import InvalidGitRepositoryError
from git.repo import Repo

from . import flags, shelf
from . import pin
from .skipblock import SkipBlock

from .constants import *
from .utils import *
from pathlib import Path, PurePath
import numpy as np

from .utils import gen_commit2tstamp_mapper


class replay_clock:
    epoch = 0


ignore_report = False
runtime_initialized = False


def it(value: Union[Iterable, bool]):
    """
    Main loop wrapper
    :param value:
        Iterable when iterating over a for loop
        Bool when looping with while
    """
    global ignore_report
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
            if not value:
                _close_record()
                ignore_report = True
            return value
        else:
            for each in value:
                yield each
            _close_record()
            ignore_report = True
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
                        replay_clock.epoch = value[capsule.epoch]  # type: ignore
                        assert hasattr(
                            value, "__getitem__"
                        ), "TODO: Implement next() calls to consume iterator"
                        yield value[capsule.epoch]  # type: ignore
                else:
                    assert capsule.epoch is not None
                    replay_clock.epoch = value[capsule.epoch]  # type: ignore
                    assert hasattr(
                        value, "__getitem__"
                    ), "TODO: Implement next() calls to consume iterator"
                    yield value[capsule.epoch]  # type: ignore
        _write_replay_file(name=flags.NAME, memo=str(flags.INDEX))


def _deferred_init(_nil=[]):
    """
    At most once execution
    """
    global runtime_initialized
    if not runtime_initialized:
        assert flags.NAME is not None
        if not flags.REPLAY and flags.MODE is None:
            repo = Repo()
            assert (
                SHADOW_BRANCH_PREFIX
                == repo.active_branch.name[0 : len(SHADOW_BRANCH_PREFIX)]
            ), f"Please run FLOR from a shadow branch (branch name: `{SHADOW_BRANCH_PREFIX}.[...]`)\nso we may commit dirty pages automatically"
        shelf.mk_job(flags.NAME)
        SkipBlock.bind()
        if flags.REPLAY:
            SkipBlock.journal.read()
        else:
            index_path = (
                flags.INDEX
                if flags.MODE is RECORD_MODE.chkpt_restore
                else shelf.get_index()
            )
            SkipBlock.logger.set_path(index_path)
            assert SkipBlock.logger.path is not None
        runtime_initialized = True


def report_end():
    """
    Call me when the execution finishes
    """
    global runtime_initialized, ignore_report
    if not ignore_report and flags.NAME is not None:
        if not flags.REPLAY:
            # Record mode
            repo = Repo()

            d = {}
            d["SOURCE"] = "report_end"
            d["NAME"] = flags.NAME
            pin.kvs.update(pin.anti_kvs)
            d["KVS"] = pin.kvs
            with open(".replay.json", "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False, indent=4)
            repo.git.add("-A")
            commit = repo.index.commit(
                f"{repo.active_branch.name}@{flags.NAME}::report_end"
            )
            sha = commit.hexsha
            print(f"Logged to `.replay.json` and committed to {sha[0:8]}...")


def _close_record():
    commit_sha = _save_run() if flags.MODE is None else get_active_commit_sha()
    SkipBlock.logger.append(SkipBlock.journal.get_eof(commit_sha))
    SkipBlock.logger.close()
    return commit_sha, SkipBlock.logger.path


def _save_run() -> str:
    assert SkipBlock.logger.path is not None
    repo = Repo()
    _write_replay_file()
    repo.git.add("-A")
    commit = repo.index.commit(
        f"{repo.active_branch.name}@{flags.NAME}::{SkipBlock.logger.path.name}"
    )
    commit_sha = commit.hexsha
    return commit_sha


def _write_replay_file(name=None, memo=None):
    d = {}
    d["NAME"] = flags.NAME if name is None else name
    d["MEMO"] = str(SkipBlock.logger.path) if memo is None else memo
    pin.kvs.update(pin.anti_kvs)
    d["KVS"] = pin.kvs
    with open(FLORFILE, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=4)

    p = (Path.home() / ".flor") / flags.NAME / "replay_jsons"  # type: ignore
    p.mkdir(exist_ok=True)
    memo = os.path.basename(d["MEMO"])
    memo, _ = os.path.splitext(memo)
    memo += ".json"
    p = p / memo
    with open(p, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=4)


def load_kvs():
    with open(FLORFILE, "r", encoding="utf-8") as f:
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
