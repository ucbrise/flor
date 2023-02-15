from git.repo import Repo
from git.exc import InvalidGitRepositoryError
import os

from flor import flags
from flor.state import State

from flor.constants import *
from flor.logger import exp_json
from flor.shelf import home_shelf
from pathlib import Path

import atexit


def get_projid():
    if State.common_dir is None:
        r = Repo()
        State.common_dir = Path(r.common_dir)
    return (
        os.path.basename(os.path.dirname(str(State.common_dir)))
        + ":"
        + str(State.active_branch)
    )


def in_shadow_branch():
    try:
        if State.active_branch is None:
            r = Repo()
            State.active_branch = str(r.active_branch)
        cond = (
            SHADOW_BRANCH_PREFIX == State.active_branch[0 : len(SHADOW_BRANCH_PREFIX)]
        )
        if cond:
            get_projid()
        return cond
    except InvalidGitRepositoryError:
        return False


@atexit.register
def flush():
    # This is the last flush
    path = home_shelf.close()
    if flags.NAME and in_shadow_branch():
        repo = Repo(State.common_dir)
        repo.git.add("-A")
        commit = repo.index.commit(
            f"{'REPLAY' if flags.REPLAY else 'RECORD'}::{get_projid()}@{flags.NAME}::{path if path else 'None'}"
        )
        commit_sha = commit.hexsha
        exp_json.put("COMMIT", commit_sha)
        exp_json.put("PROJID", get_projid())
        exp_json.flush()
