import os
import sys
from pathlib import Path

from . import versions

CURRDIR = versions.get_repo_dir()
assert (
    CURRDIR
), "Please call flor from within a Git repository. We'll be doing auto-commits."
PROJID = os.path.basename(CURRDIR)

FLORDIR = os.path.join(CURRDIR, ".flor")
RUNS_DIR = os.path.join(FLORDIR, "runs")
OBJSTORE_DIR = os.path.join(FLORDIR, "obj_store")
DB_PATH = os.path.join(FLORDIR, str(Path(PROJID).with_suffix(".db")))

os.makedirs(FLORDIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(OBJSTORE_DIR, exist_ok=True)

SCRIPTNAME = os.path.basename(sys.argv[0])

# `logs.value_type`. Only VALUE_TYPE_LOG is pivoted by flor.dataframe(), which
# is what keeps captured io and profiling out of the metric table by default.
VALUE_TYPE_LOG = 1  # flor.log / flor.arg
VALUE_TYPE_IO = 2  # captured print / logging
VALUE_TYPE_TIME = 3  # time::* profiling
VALUE_TYPE_START = 4  # flor::start -- the earlier run a run's training resumed from

__all__ = [
    "CURRDIR",
    "PROJID",
    "FLORDIR",
    "RUNS_DIR",
    "OBJSTORE_DIR",
    "DB_PATH",
    "SCRIPTNAME",
    "VALUE_TYPE_LOG",
    "VALUE_TYPE_IO",
    "VALUE_TYPE_TIME",
    "VALUE_TYPE_START",
]
