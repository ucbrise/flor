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
from pathlib import Path, PurePath
import numpy as np


from sh import tail


class replay_clock:
    epoch = 0


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
            if not value:
                _close_record()
            return value
        else:
            for each in value:
                yield each
            _close_record()
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
    if not _nil:
        assert flags.NAME is not None
        if not flags.REPLAY:
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
            SkipBlock.logger.set_path(shelf.get_index())
            assert SkipBlock.logger.path is not None
        _nil.append(True)


def _close_record():
    commit_sha = _save_run()
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

    # I want to build a mapper from FLORFILE to GIT HASH
    vid_mapper = dict()
    for path in df1["vid"].drop_duplicates().to_list():
        eof = json.loads(tail("-1", path, _iter=True).next())
        vid_mapper[path] = eof["COMMIT_SHA"]

    df1["vid"] = df1["vid"].apply(lambda x: vid_mapper[x])

    return df1.sort_values(by=["tstamp", "epoch", "step"], ascending=False)


__all__ = ["it"]
