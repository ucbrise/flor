import json
import os
from typing import Iterable, List, Union

from git.exc import InvalidGitRepositoryError
from git.repo import Repo

from . import flags, shelf
from .skipblock import SkipBlock


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
        shelf.mk_job(flags.NAME)
        SkipBlock.bind()
        if flags.REPLAY:
            SkipBlock.journal.read()
        else:
            SkipBlock.logger.set_path(shelf.get_index())
            assert SkipBlock.logger.path is not None
        _nil.append(True)


def _close_record():
    commit_sha = _save_versions()
    SkipBlock.logger.append(SkipBlock.journal.get_eof(commit_sha))
    SkipBlock.logger.close()
    return commit_sha, SkipBlock.logger.path


def _save_versions() -> str:
    try:
        repo = Repo()
    except InvalidGitRepositoryError:
        repo = Repo.init()

    if not repo.branches:
        repo.index.commit("initial commit")

    active_branch = repo.active_branch
    repo.git.stash("-u")
    if flags.NAME not in [b.name for b in repo.branches]:
        repo.create_head(flags.NAME)

    repo.git.checkout(flags.NAME)
    stash_msg_list: str = repo.git.stash("list")

    repo.git.merge(active_branch.name, "-X", "theirs", "--squash")

    if stash_msg_list:
        repo.git.stash("apply")

    _write_replay_file()

    repo.git.add("-A")
    commit = repo.index.commit(
        f"flor::{active_branch.name}@{active_branch.commit.hexsha}"
    )
    commit_sha = commit.hexsha

    repo.git.checkout(active_branch.name)
    if stash_msg_list:
        repo.git.stash("pop")

    return commit_sha


def _write_replay_file():
    d = {}
    d["NAME"] = flags.NAME
    d["MEMO"] = str(SkipBlock.logger.path)
    with open("flor_replay.json", "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=4)


__all__ = ["it"]
