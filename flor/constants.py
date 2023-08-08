from datetime import datetime
import os

STEM = (
    "branch",
    "op",
    "tstamp",
    "vid",
)
TIMESTAMP = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
CURRDIR = os.getcwd()
PROJID = os.path.basename(CURRDIR)
HOMEDIR = os.path.join(os.path.expanduser("~"), ".flor")
SHADOW_BRANCH_PREFIX = "flor"

__all__ = ["SHADOW_BRANCH_PREFIX", "STEM", "TIMESTAMP", "CURRDIR", "PROJID", "HOMEDIR"]
