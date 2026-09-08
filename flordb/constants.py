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
# Metrics read out of captured text by `flor capture --extract`, one file
# per run beside the run it was read from. Tracked, like runs/: derived,
# but derived once and then shared, so a teammate sees the same columns.
EXTRACT_DIR = os.path.join(FLORDIR, "extracted")
OBJSTORE_DIR = os.path.join(FLORDIR, "obj_store")
DB_PATH = os.path.join(FLORDIR, str(Path(PROJID).with_suffix(".db")))

os.makedirs(FLORDIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(EXTRACT_DIR, exist_ok=True)
os.makedirs(OBJSTORE_DIR, exist_ok=True)

SCRIPTNAME = os.path.basename(sys.argv[0])

# `logs.value_type`. Only VALUE_TYPE_LOG is pivoted by flor.dataframe(), which
# is what keeps captured io and profiling out of the metric table by default.
VALUE_TYPE_LOG = 1  # flor.log / flor.arg
VALUE_TYPE_IO = 2  # captured print / logging
VALUE_TYPE_TIME = 3  # time::* profiling

__all__ = [
    "CURRDIR",
    "PROJID",
    "FLORDIR",
    "RUNS_DIR",
    "EXTRACT_DIR",
    "OBJSTORE_DIR",
    "DB_PATH",
    "SCRIPTNAME",
    "VALUE_TYPE_LOG",
    "VALUE_TYPE_IO",
    "VALUE_TYPE_TIME",
]
