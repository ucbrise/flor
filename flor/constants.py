from datetime import datetime
import os
import sys

STEM = (
    "branch",
    "op",
    "tstamp",
    "vid",
)
TIMESTAMP = datetime.now().isoformat(timespec="seconds")
CURRDIR = os.getcwd()
PROJID = os.path.basename(CURRDIR)
HOMEDIR = os.path.join(os.path.expanduser("~"), ".flor")
SHADOW_BRANCH_PREFIX = "flor."
SCRIPTNAME = os.path.basename(sys.argv[0])

__all__ = ["SHADOW_BRANCH_PREFIX", "STEM", "TIMESTAMP", "CURRDIR", "PROJID", "HOMEDIR", "SCRIPTNAME"]
