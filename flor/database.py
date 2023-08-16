from typing import Dict, Optional, Tuple
from .constants import *
from . import orm

import sqlite3
from pathlib import Path
import os


def unpack(output_buffer, cursor):
    if not output_buffer:
        return

    print(
        "Unpacking ",
        output_buffer[-1] if output_buffer else None,
    )
    # Structure the output buffer
    stem = output_buffer[-1]
    for obj in output_buffer[:-1]:
        orm.parse_entries(obj)
        orm.parse_log(stem, obj, orm.parse_loop(obj))

    loops: Dict[Tuple[Optional[int], str, int, int], int] = {}
    for (
        ctx_id,
        parent_ctx_id,
        loop_name,
        loop_entries,
        loop_iteration,
    ) in read_from_loops(cursor):
        print("ctx_id", ctx_id)
        print("parent_ctx_id", parent_ctx_id)
        print("loop_name", loop_name)
        print("loop_entries", loop_entries)
        print("loop_iteration", loop_iteration)
        loops[(parent_ctx_id, loop_name, int(loop_entries), int(loop_iteration))] = int(
            ctx_id
        )
        print("ctx ID: ", ctx_id)

    for log_record in orm.logs:
        print(log_record)


def create_tables(cursor):
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS loops (
        ctx_id INTEGER,
        parent_ctx_id INTEGER,
        loop_name TEXT,
        loop_entries INTEGER,
        loop_iteration INTEGER
    )
    """
    )

    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS logs (
        projid TEXT,
        tstamp TEXT,
        filename TEXT,
        ctx_id INTEGER,
        value_name TEXT,
        value TEXT,
        value_type INTEGER
    )
    """
    )


def insert_data_into_loops(
    ctx_id, parent_ctx_id, loop_name, loop_entries, loop_iteration, cursor
):
    cursor.execute(
        """
    INSERT INTO loops (ctx_id, parent_ctx_id, loop_name, loop_entries, loop_iteration)
    VALUES (?, ?, ?, ?, ?)
    """,
        (ctx_id, parent_ctx_id, loop_name, loop_entries, loop_iteration),
    )


def insert_data_into_logs(
    projid, tstamp, filename, ctx_id, value_name, value, value_type, cursor
):
    cursor.execute(
        """
    INSERT INTO logs (projid, tstamp, filename, ctx_id, value_name, value, value_type)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """,
        (projid, tstamp, filename, ctx_id, value_name, value, value_type),
    )


def read_from_loops(cursor):
    cursor.execute("SELECT * FROM loops")
    return cursor.fetchall()


def read_from_logs(cursor):
    cursor.execute("SELECT * FROM logs")
    return cursor.fetchall()
