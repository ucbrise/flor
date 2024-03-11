import os
import sys

from . import versions

CURRDIR = versions.get_repo_dir()
assert CURRDIR, "Not a valid Git repository"
PROJID = os.path.basename(CURRDIR)
HOMEDIR = os.path.join(os.path.expanduser("~"), ".flor")
os.makedirs(HOMEDIR, exist_ok=True)
SCRIPTNAME = os.path.basename(sys.argv[0])


__all__ = [
    "CURRDIR",
    "PROJID",
    "HOMEDIR",
    "SCRIPTNAME",
]
