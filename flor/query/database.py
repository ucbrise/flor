import sqlite3
from pathlib import Path

from flor.shelf import home_shelf
from flor.state import State

SUFFIX = ".db"


def start_db(projid: str):
    (dir_n, _) = projid.split("_")
    dbp = home_shelf.florin / Path(dir_n).with_suffix(SUFFIX)
    assert State.db_conn is None, "Did you try to start_db more than once?"
    State.db_conn = sqlite3.connect(dbp)

