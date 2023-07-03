import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

from flor.shelf import home_shelf
from flor.constants import *
import math
import random

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
    SELECT DISTINCT * FROM workers WHERE status = '{status}'; 
    """
    cur = db_conn.cursor()
    try:
        res = cur.execute(
            sql,
        )
        if res is not None:
            all_res = res.fetchall()
            if all_res:
                for pid, tstamp, gpu, status in all_res:
                    if gpu is not None and not math.isnan(gpu):
                        print(f"PID {pid} on GPU {gpu}: {status} since {tstamp}.")
                    else:
                        print(f"PID {pid} on CPU: {status} since {tstamp}.")
            else:
                print("FLOR Server Inactive.")
        else:
            print("FLOR Server Inactive.")
    except Exception as e:
        print(e)
    finally:
        cur.close()


def server_active(db_conn, gpu_id):
    sql = """
    INSERT INTO workers VALUES(?, ?, ?, ?)
    """
    cur = db_conn.cursor()
    try:
        pid = int(os.getpid())
        tstamp = str(datetime.now())
        status = "ACTIVE"
        cur.execute(sql, (pid, tstamp, gpu_id, status))
        db_conn.commit()
        return pid, tstamp
    except Exception as e:
        print(e)
        return None, None
    finally:
        cur.close()


def server_completed(db_conn):
    cur = db_conn.cursor()
    try:
        # Step 1: Remove all occurrences of this worker in the pool table
        cur.execute(
            "DELETE FROM pool WHERE pid = ?",
            (int(os.getpid()),),
        )

        # Step 2: Update the workers table
        cur.execute(
            "UPDATE workers SET status = 'COMPLETED' WHERE pid = ?",
            (int(os.getpid()),),
        )

        # If all operations were successful, commit the changes
        db_conn.commit()

    except Exception as e:
        print(e)
        # If an error occurred, roll back any changes
        db_conn.rollback()
    finally:
        cur.close()


def init_db(db_conn: sqlite3.Connection):
    """
    Initializes the database
    TODO: add GPU column to `workers`
    """
    cur = db_conn.cursor()
    cur.executescript(
        """
        BEGIN;
        CREATE TABLE workers(
            pid integer,
            tstamp text,
            gpu integer,
            status text
        );
        CREATE TABLE jobs(
            jobid integer,
            path text,
            script text,
            args text,
            done integer
        );
        CREATE TABLE replay(
            jobid integer,
            path text,
            script text,
            vid text,
            appvars text,
            mode text, 
            done integer
        );
        CREATE TABLE pool(
            pid integer,
            tstamp text,
            jobid integer
        );
        COMMIT;
        """
    )
    cur.close()


def add_jobs(
    db_conn: sqlite3.Connection, run_dir: str, script: str, batched_args: List[str]
):
    sql = """
    INSERT INTO jobs VALUES(?, ?, ?, ?, ?)
    """
    cur = db_conn.cursor()
    params = [
        (random.randint(0, 999999999), run_dir, script, args, 0)
        for args in batched_args
    ]
    try:
        cur.executemany(sql, params)
        db_conn.commit()
    except Exception as e:
        print(e)
    finally:
        cur.close()


def add_replay(
    db_conn: sqlite3.Connection,
    run_dir: str,
    script: str,
    vid_vars_mod: List[Any],
):
    cur = db_conn.cursor()
    params = []
    for vid, apply_vars, mode in vid_vars_mod:
        params.append(
            (random.randint(0, 999999999), run_dir, script, vid, apply_vars, mode, 0)
        )
    try:
        cur.executemany("INSERT INTO replay VALUES (?, ?, ?, ?, ?, ?, ?)", params)
        db_conn.commit()
    except Exception as e:
        print(e)
        db_conn.rollback()
    finally:
        cur.close()


def finish_job(db_conn: sqlite3.Connection, jobid):
    sql = """
    UPDATE jobs SET done = 1 WHERE jobid = ?
    """
    cur = db_conn.cursor()
    try:
        cur.execute(sql, (jobid,))
        db_conn.commit()
    except Exception as e:
        print(e)
    finally:
        cur.close()


def finish_replay(db_conn: sqlite3.Connection, jobid):
    cur = db_conn.cursor()
    try:
        cur.execute("UPDATE replay SET done = 1 WHERE jobid = ?", (jobid,))
        db_conn.commit()
    except Exception as e:
        print(e)
        db_conn.rollback()
    finally:
        cur.close()


def step_worker(db_conn: sqlite3.Connection, pid, tstamp):
    sqls = [
        """
        SELECT jobid, path, script, args FROM jobs WHERE
            done = 0 AND
            jobid not in (SELECT jobid FROM pool)
            LIMIT 1;
        """,
        """
        INSERT INTO pool VALUES(?, ?, ?)
        """,
        """
        SELECT jobid, path, script, vid, appvars, mode FROM replay 
          WHERE done = 0 AND jobid not in (SELECT jobid FROM pool) LIMIT 1;
        """,
    ]
    cur = db_conn.cursor()
    try:
        res = cur.execute(sqls[0])
        if res is not None:
            for jobid, path, script, args in res.fetchall():
                cur.execute(sqls[1], (pid, tstamp, jobid))
                db_conn.commit()
                return "jobs", (jobid, path, script, args)
        res = cur.execute(sqls[2])
        if res is not None:
            for jobid, path, script, vid, apply_vars, mode in res.fetchall():
                cur.execute(sqls[1], (pid, tstamp, jobid))
                db_conn.commit()
                return "replay", (jobid, path, script, vid, apply_vars, mode)
        return None
    except Exception as e:
        print(e)
        return None
    finally:
        cur.close()
