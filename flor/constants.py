from datetime import datetime
import os
import sys

CURRDIR = os.getcwd()
PROJID = os.path.basename(CURRDIR)
HOMEDIR = os.path.join(os.path.expanduser("~"), ".flor")
SHADOW_BRANCH_PREFIX = "flor."
SCRIPTNAME = os.path.basename(sys.argv[0])
TIMESTAMP = datetime.now().isoformat(timespec="seconds")


__all__ = [
    "SHADOW_BRANCH_PREFIX",
    "TIMESTAMP",
    "CURRDIR",
    "PROJID",
    "HOMEDIR",
    "SCRIPTNAME",
]
