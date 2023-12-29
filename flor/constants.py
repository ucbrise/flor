import os
import sys

CURRDIR = os.getcwd()
PROJID = os.path.basename(CURRDIR)
HOMEDIR = os.path.join(os.path.expanduser("~"), ".flor")
os.makedirs(HOMEDIR, exist_ok=True)
SHADOW_BRANCH_PREFIX = "flor."
SCRIPTNAME = os.path.basename(sys.argv[0])


__all__ = [
    "SHADOW_BRANCH_PREFIX",
    "CURRDIR",
    "PROJID",
    "HOMEDIR",
    "SCRIPTNAME",
]
