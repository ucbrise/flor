from flor import flags
from flor.constants import FLORFILE
from typing import Optional, List, Dict, Any


import csv
import pandas as pd
from pathlib import Path

_fyle: Optional[List[Dict[str, Any]]] = None


def deferred_init():
    global _fyle
    with open(FLORFILE, "r") as f:
        _fyle = list(csv.DictReader(f))


def logfile_exists():
    return Path(FLORFILE).exists()


def eval(select, where=None) -> pd.DataFrame:
    assert _fyle is not None
    df = pd.DataFrame(_fyle)
    if where is not None:
        df = df.query(where)
    return df[select]


__all__ = ["deferred_init", "logfile_exists"]
