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
        r = Repo()
        State.repo = r
        try:
            State.active_branch = str(r.active_branch)
        except TypeError:
            bs = State.repo.git.branch(
                "--contains", str(State.repo.head.commit.hexsha)
            ).split("\n")
            branches = [e.strip() for e in bs if "flor.shadow" in e]
            if len(branches) == 1:
                State.active_branch = branches.pop()
            else:
                raise NotImplementedError()
        cond = check_branch_cond()
        if cond:
            PATH.mkdir(exist_ok=True)
            get_projid()
        return cond
    except InvalidGitRepositoryError:
        return False


def check_branch_cond():
    if State.repo is not None:
        return "RECORD::" in str(
            State.repo.head.commit.message
        ) or "flor.shadow" in State.repo.git.branch(
            "--contains", str(State.repo.head.commit.hexsha)
        )
    return False


@atexit.register
def flush():
    path = home_shelf.close()
    cond = in_shadow_branch()
    projid = get_projid()

    if flags.NAME and not flags.REPLAY:
        assert cond
        log_records.flush(projid, str(State.timestamp))
        exp_json.put("PROJID", projid)
        exp_json.put("EPOCHS", State.epoch)
        exp_json.flush()
        repo = State.repo
        assert repo is not None
        repo.git.add("-A")
        repo.index.commit(f"RECORD::{flags.NAME}")
    elif flags.NAME and flags.REPLAY:
        assert cond
        for k in [k for k in exp_json.record_d if not k.isupper()]:
            log_records.put_dp(k, exp_json.record_d[k])

        log_records.flush(projid, str(State.timestamp))

    if State.db_conn:
        State.db_conn.commit()
        State.db_conn.close()
