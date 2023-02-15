from git.repo import Repo
from git.exc import InvalidGitRepositoryError
import os

from flor import flags

from flor.constants import *
from flor.logger import exp_json
from flor.shelf import home_shelf

import atexit


def get_projid():
    r = Repo()
    active_branch = str(r.active_branch)
    common_dir = os.path.basename(os.path.dirname(str(r.common_dir)))
    return common_dir + ":" + active_branch


def in_shadow_branch():
    try:
        r = Repo()
        active_branch = str(r.active_branch)
        return SHADOW_BRANCH_PREFIX == active_branch[0 : len(SHADOW_BRANCH_PREFIX)]
    except InvalidGitRepositoryError:
        return False


@atexit.register
def flush():
    # This is the last flush
    path = home_shelf.close()
    if flags.NAME:
        try:
            repo = Repo()
        except InvalidGitRepositoryError:
            return
        repo.git.add("-A")
        commit = repo.index.commit(
            f"{get_projid()}@{flags.NAME}::{path if path else 'None'}"
        )
        commit_sha = commit.hexsha
        exp_json.put("COMMIT", commit_sha)
        exp_json.put("PROJID", get_projid())
        exp_json.flush()
