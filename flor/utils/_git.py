from .. import constants as CONST
from git.repo import Repo
import os


def gen_commit2tstamp_mapper():
    """
    os.path.basename(absolute)
    """
    r = Repo()
    commits = r.iter_commits("--all")

    def get_index(message: str):
        return message.split("::")[1]

    sha2time = dict()
    time2sha = dict()

    for c in commits:
        if CONST.SHADOW_BRANCH_PREFIX in str(c.message):
            index = get_index(str(c.message))
            sha2time[c.hexsha] = index
            time2sha[index] = c.hexsha

    return time2sha, sha2time


def get_active_commit_sha():
    r = Repo()
    return r.head.commit.hexsha
