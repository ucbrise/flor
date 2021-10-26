import json
import os
from typing import Iterable, List, Union

from git.exc import InvalidGitRepositoryError
from git.repo import Repo

from . import flags, shelf
from .skipblock import SkipBlock

from .constants import *
from .pin import kvs


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
                        assert hasattr(
                            value, "__getitem__"
                        ), "TODO: Implement next() calls to consume iterator"
                        yield value[capsule.epoch]  # type: ignore
                else:
                    assert capsule.epoch is not None
                    assert hasattr(
                        value, "__getitem__"
                    ), "TODO: Implement next() calls to consume iterator"
                    yield value[capsule.epoch]  # type: ignore


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


def _write_replay_file():
    d = {}
    d["NAME"] = flags.NAME
    d["MEMO"] = str(SkipBlock.logger.path)
    d["KVS"] = kvs
    with open(FLORFILE, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=4)


__all__ = ["it"]
