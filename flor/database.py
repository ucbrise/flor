from .constants import *

import sqlite3
from pathlib import Path
import os


def start_db() -> sqlite3.Connection:
    conn = create_connection(os.path.join(HOMEDIR, Path(PROJID).with_suffix(".db")))
    assert conn is not None
    return conn


def add_jobs(db_conn: sqlite3.Connection, from_path, script, cli_args):
    pass


def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except sqlite3.Error as e:
        print(e)

    return conn


def create_table(conn, create_table_sql):
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except sqlite3.Error as e:
        print(e)


def add_row(conn, table_name, data):
    placeholders = ", ".join(["?"] * len(data))
    sql = f"INSERT INTO {table_name} VALUES ({placeholders})"
    cur = conn.cursor()
    cur.execute(sql, data)
    conn.commit()


def create_and_add_table(conn, table_name, columns, data):
    columns_definition = ", ".join(
        [f"{col_name} {col_type}" for col_name, col_type in columns.items()]
    )
    create_table_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_definition})"

    create_table(conn, create_table_sql)
    add_row(conn, table_name, data)


# Example usage:
if __name__ == "__main__":
    database = "mydatabase.db"

    # Create a connection
    conn = create_connection(database)

    # Create a predefined table
    create_table(
        conn, "CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT)"
    )

    # Add a row to the predefined table
    add_row(conn, "users", (1, "John Doe"))

    # Add a new table and a row during runtime
    columns = {"product_id": "INTEGER PRIMARY KEY", "product_name": "TEXT"}
    data = (1, "Laptop")
    create_and_add_table(conn, "products", columns, data)

    if conn is not None:
        conn.close()
