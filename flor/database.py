import os
import sqlite3
from pathlib import Path
from typing import Optional
from datetime import datetime

from flor.shelf import home_shelf
from flor.constants import *

SUFFIX = ".db"
dbp: Optional[Path] = None


def start_db():
    global dbp
    dbp = home_shelf.florin / Path("main").with_suffix(SUFFIX)
    is_first_start = not dbp.exists()
    db_conn = sqlite3.connect(dbp)
    if is_first_start:
        init_db(db_conn)
    return db_conn


def server_status(db_conn, status="ACTIVE"):
    sql = f"""
    SELECT DISTINCT * FROM status WHERE status = '{status}'; 
    """
    cur = db_conn.cursor()
    try:
        res = cur.execute(
            sql,
        )
        if res is not None:
            all_res = res.fetchall()
            if all_res:
                for pid, status, tstamp in all_res:
                    print(f"PID {pid}: {status} since {tstamp}.")
            else:
                print("FLOR Server Inactive.")
        else:
            print("FLOR Server Inactive.")
    except Exception as e:
        print(e)
    finally:
        cur.close()


def server_active(db_conn):
    sql = """
    INSERT INTO status VALUES(?, ?, ?)
    """
    cur = db_conn.cursor()
    try:
        cur.execute(sql, (int(os.getpid()), "ACTIVE", str(datetime.now())))
        db_conn.commit()
    except Exception as e:
        print(e)
    finally:
        cur.close()


def server_completed(db_conn):
    sql = """
    UPDATE status SET status = ?, tstamp = ? WHERE pid = ?;
    """
    cur = db_conn.cursor()
    try:
        cur.execute(
            sql,
            (
                "COMPLETED",
                str(datetime.now()),
                int(os.getpid()),
            ),
        )
        db_conn.commit()
    except Exception as e:
        print(e)
    finally:
        cur.close()


def init_db(db_conn: sqlite3.Connection):
    """
    Initializes the database
    """
    cur = db_conn.cursor()
    cur.executescript(
        """
        BEGIN;
        CREATE TABLE status(
            pid integer,
            status text,
            tstamp text);
        CREATE TABLE config(
            name text,
            value text
        );
        CREATE TABLE queue(
            path text,
            args text,
            pid integer
        );
        COMMIT;
        """
    )
    cur.close()
