from .constants import *

import sqlite3
from pathlib import Path
import os


def unpack(output_buffer):
    print("Unpacking len:", len(output_buffer))


def create_tables():
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
    ctx_id, parent_ctx_id, loop_name, loop_entries, loop_iteration
):
    cursor.execute(
        """
    INSERT INTO loops (ctx_id, parent_ctx_id, loop_name, loop_entries, loop_iteration)
    VALUES (?, ?, ?, ?, ?)
    """,
        (ctx_id, parent_ctx_id, loop_name, loop_entries, loop_iteration),
    )


def insert_data_into_logs(
    projid, tstamp, filename, ctx_id, value_name, value, value_type
):
    cursor.execute(
        """
    INSERT INTO logs (projid, tstamp, filename, ctx_id, value_name, value, value_type)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """,
        (projid, tstamp, filename, ctx_id, value_name, value, value_type),
    )


def read_from_loops():
    cursor.execute("SELECT * FROM loops")
    return cursor.fetchall()


def read_from_logs():
    cursor.execute("SELECT * FROM logs")
    return cursor.fetchall()


# Example usage:
if __name__ == "__main__":
    connection = sqlite3.connect(os.path.join(HOMEDIR, Path(PROJID).with_suffix(".db")))
    cursor = connection.cursor()

    create_tables()

    # Example of inserting data
    # insert_data_into_loops(1, None, "loop1", 10, 5)
    # insert_data_into_logs(
    #     "proj1", "2023-08-16", "file.txt", 1, "value_name", "value", 1
    # )

    # Example of reading data
    loops_data = read_from_loops()
    logs_data = read_from_logs()

    connection.commit()
    connection.close()

    # Process or print the retrieved data as needed
    print(loops_data)
    print(logs_data)
