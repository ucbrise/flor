from git.repo import Repo
from git.exc import InvalidGitRepositoryError
import os

from flor import flags
from flor.state import State

from flor.constants import *
from flor.logger import exp_json, log_records
from flor.shelf import home_shelf
from pathlib import Path

import atexit

PATH = Path(".flor")


def get_projid():
    if State.common_dir is None:
        r = Repo()
        State.common_dir = Path(r.common_dir)
    return (
        os.path.basename(os.path.dirname(str(State.common_dir)))
        + "_"
        + str(State.active_branch)
    )


def in_shadow_branch():
    """
    Initialize
    """
    try:
        if State.active_branch is None:
            r = Repo()
            State.repo = r
            State.active_branch = str(r.active_branch)
        cond = (
            SHADOW_BRANCH_PREFIX == State.active_branch[0 : len(SHADOW_BRANCH_PREFIX)]
        )
        if cond:
            PATH.mkdir(exist_ok=True)
            get_projid()
        return cond
    except InvalidGitRepositoryError:
        return False


@atexit.register
def flush():
    path = home_shelf.close()
    try:
        log_records.flush()
    except Exception as e:
        print(e)
    if flags.NAME and in_shadow_branch():
        projid = get_projid()
        exp_json.put("PROJID", projid)
        exp_json.put("EPOCHS", State.epoch)
        exp_json.flush()
        repo = Repo(State.common_dir)
        repo.git.add("-A")
        commit = repo.index.commit(
            f"{'REPLAY' if flags.REPLAY else 'RECORD'}::{flags.NAME}"
        )
        if State.db_conn:
            State.db_conn.close()
