import sqlite3
from pathlib import Path

from flor.shelf import home_shelf
from flor.state import State

SUFFIX = ".db"


def start_db(projid: str):
    (dir_n, _) = projid.split("_")
    dbp = home_shelf.florin / Path(dir_n).with_suffix(SUFFIX)
    is_first_start = not dbp.exists()
    assert State.db_conn is None, "Did you try to start_db more than once?"
    State.db_conn = sqlite3.connect(dbp)
    if is_first_start:
        init_db()


def init_db():
    """
    Initializes the database
    1. Creates log_records
    2. Defines pivot views
    3. Updates metadata
    """
    assert State.db_conn is not None
    cur = State.db_conn.cursor()
    cur.executescript(
        """
        BEGIN;
        CREATE TABLE watermark(projid text, commitsha text);
        CREATE TABLE log_records(
            projid text,
            tstamp text,
            vid text, 
            epoch integer,
            step integer,
            name text,
            value text
            );
        COMMIT;
    """
    )
    cur.close()


def update_watermark(projid: str, commitsha: str):
    assert State.db_conn is not None
    cur = State.db_conn.cursor()
    res = cur.execute(
        "SELECT commitsha, projid FROM watermark WHERE projid = ?", (projid,)
    ).fetchall()
    if res:
        c, p = res[0]
        cur.execute(
            "UPDATE watermark SET commitsha = ? WHERE projid = ?",
            (commitsha, projid),
        )
        return str(c)
    else:
        cur.execute("INSERT INTO watermark VALUES(?, ?)", (projid, commitsha))
    State.db_conn.commit()
    cur.close()

def get_watermark(projid: str):
    assert State.db_conn is not None
    cur = State.db_conn.cursor()
    res = cur.execute("SELECT commitsha, projid FROM watermark WHERE projid = ?", (projid,)).fetchall()
    if not res:
        return None
    c,p = res[0]
    return str(c) 


def get_log_records():
    assert State.db_conn is not None
    cur = State.db_conn.cursor()
    res = cur.execute("SELECT * FROM log_records").fetchall()
    cur.close()
    return res
