import os
from pathlib import Path

from .constants import *
from .clock import Clock
from . import cli


# Backend -> extension of the per-iteration snapshots older runs left on the
# shelf. Nothing writes these any more; checkpoint_io still lists them so those
# runs stay loadable.
BACKENDS = (
    ("state_dict", ".pth"),
    ("numpy", ".npy"),
    ("pandas", ".parquet"),
    ("pickle", ".pkl"),
)


def get_shelf():
    if not cli.in_replay_mode():
        tstamp = Clock.get_datetime()
    else:
        assert cli.flags.old_tstamp is not None
        tstamp = cli.flags.old_tstamp

    SHELF = Path(OBJSTORE_DIR) / tstamp
    os.makedirs(SHELF, exist_ok=True)
    return SHELF
