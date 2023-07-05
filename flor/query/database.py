import sqlite3
from pathlib import Path
from typing import Optional

from flor.shelf import home_shelf
from flor.state import State
from flor.constants import *

SUFFIX = ".db"
dbp: Optional[Path] = None


def exists():
    return dbp is not None and dbp.exists()


def start_db(projid: str):
    global dbp
    dir_n = "_".join(projid.split("_")[0:-1])
    dbp = home_shelf.florin / Path(dir_n).with_suffix(SUFFIX)
    is_first_start = not dbp.exists()
    assert State.db_conn is None, "Did you try to start_db more than once?"
    State.db_conn = sqlite3.connect(dbp)
    if is_first_start:
        init_db()


def init_db():
    """
    Initializes the database
    """
    assert State.db_conn is not None
    cur = State.db_conn.cursor()
    cur.executescript(
        """
        BEGIN;
        CREATE TABLE log_records(
            projid text,
            runid text,
            tstamp text,
            vid text, 
            epoch integer,
            step integer,
            name text,
            value text
            );
        CREATE TABLE data_prep(
            projid text,
            runid text,
            tstamp text,
            vid text,
            prep_secs real,
            eval_secs real
        );
        CREATE TABLE outr_loop(
            projid text,
            runid text,
            tstamp text,
            vid text,
            epoch integer,
            seconds real
        );
        COMMIT;
        """
    )
    cur.close()


def get_watermark():
    assert State.db_conn is not None
    cur = State.db_conn.cursor()
    res = cur.execute("SELECT DISTINCT vid FROM log_records").fetchall()
    if not res:
        return None
    cur.close()
    return [str(row[0]) for row in res]


def get_schedule(keys):
    # TODO: repair and test
    assert State.db_conn is not None
    cur = State.db_conn.cursor()
    if keys == DATA_PREP:
        res = []
        for r in cur.execute(
            "SELECT " + ", ".join(DATA_PREP) + ", prep_secs, eval_secs FROM data_prep;"
        ).fetchall():
            res.append(
                {
                    c: v
                    for c, v in zip(
                        list(DATA_PREP) + ["prep_secs", "eval_secs"],
                        r,
                    )
                }
            )
    elif keys == OUTR_LOOP:
        res = []
        for r in cur.execute(
            "SELECT " + ", ".join(OUTR_LOOP) + ", seconds FROM outr_loop;"
        ).fetchall():
            res.append(
                {
                    c: v
                    for c, v in zip(
                        list(OUTR_LOOP)
                        + [
                            "seconds",
                        ],
                        r,
                    )
                }
            )
    else:
        raise
    if not res:
        return None
    cur.close()
    return res


def get_log_records():
    assert State.db_conn is not None
    cur = State.db_conn.cursor()
    res = cur.execute("SELECT * FROM log_records").fetchall()
    cur.close()
    return res


def write_log_records(list_of_dicts):
    assert State.db_conn is not None
    cur = State.db_conn.cursor()

    cur.executemany(
        "INSERT INTO log_records VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                d["projid"],
                d["runid"],
                d["tstamp"],
                d["vid"],
                d["epoch"],
                d["step"],
                d["name"],
                d["value"],
            )
            for d in list_of_dicts
        ],
    )
    State.db_conn.commit()
    print("Flor wrote log records to sqlite")
    cur.close()
