from functools import reduce
from typing import Dict, Optional
import pandas as pd
import sqlite3
import os
from pathlib import Path

from .constants import *
from . import orm

from . import utils
from typing import Any, Dict, List, Optional, Tuple


def conn_and_cursor():
    conn = sqlite3.connect(os.path.join(HOMEDIR, Path(PROJID).with_suffix(".db")))
    cursor = conn.cursor()
    return conn, cursor


def insert_context(cursor, context):
    # Recursive
    parent_context_id = (
        insert_context(cursor, context.p_ctx) if context.p_ctx is not None else None
    )
    cursor.execute(
        """INSERT INTO contexts (ctx_id, p_ctx_id) VALUES (?, ?)""",
        (context.ctx_id, parent_context_id),
    )
    if isinstance(context, orm.Loop):
        cursor.execute(
            """INSERT INTO loops (l_ctx_id, l_name, l_iteration, l_value) VALUES (?, ?, ?, ?)""",
            (context.ctx_id, context.name, context.iteration, context.value),
        )
    elif isinstance(context, orm.Func):
        cursor.execute(
            """INSERT INTO funcs (f_ctx_id, f_name, f_int_arg, f_txt_arg) VALUES (?, ?, ?, ?)""",
            (context.ctx_id, context.name, context.int_arg, context.txt_arg),
        )
    else:
        raise
    return context.ctx_id


def unpack(output_buffer: List[orm.Log], cursor):
    if not output_buffer:
        return
    for each in output_buffer:
        ctx_id = insert_context(cursor, each.ctx) if each.ctx is not None else None
        cursor.execute(
            """INSERT INTO logs (projid, tstamp, filename, ctx_id, value_name, value, value_type) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                each.projid,
                each.tstamp,
                each.filename,
                ctx_id,
                each.name,
                each.value,
                each.type,
            ),
        )


def create_tables(cursor):
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS contexts (
            ctx_id INTEGER,
            p_ctx_id INTEGER
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS loops (
            l_ctx_id INTEGER,
            l_name TEXT,
            l_iteration INTEGER,
            l_value TEXT
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS funcs (
            f_ctx_id INTEGER,
            f_name TEXT,
            f_int_arg INTEGER,
            f_txt_arg TEXT

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


def deduplicate_table(cursor, table_name):
    # Create a temporary table to store unique rows
    cursor.execute(
        f"""CREATE TEMPORARY TABLE temp_table AS SELECT DISTINCT * FROM {table_name}"""
    )

    # Delete the original table's contents
    cursor.execute(f"""DELETE FROM {table_name}""")

    # Insert the unique rows back into the original table
    cursor.execute(f"""INSERT INTO {table_name} SELECT * FROM temp_table""")

    cursor.execute("DROP TABLE temp_table")


def read_from_logs(cursor, where_clause=None):
    if where_clause is None:
        cursor.execute("SELECT DISTINCT * FROM logs")
    else:
        cursor.execute(f"SELECT DISTINCT * FROM logs WHERE {where_clause}")
    return cursor.fetchall()


def read_known_tstamps(cursor):
    cursor.execute("SELECT DISTINCT tstamp FROM logs")
    return cursor.fetchall()


def query(cursor, user_query, aspandas=False):
    cursor.execute(user_query)
    res = cursor.fetchall()
    if res and aspandas:
        return utils.cast_dtypes(pd.DataFrame(res, columns=get_column_names(cursor)))
    elif res:
        return res


def get_column_names(cursor):
    column_names = [description[0] for description in cursor.description]
    return column_names


def pivot(conn, *args):
    def _pivot_star():
        df = pd.read_sql(
            "SELECT DISTINCT value_name FROM logs WHERE value_type = 1 AND ctx_id IS NULL",
            conn,
        )
        value_names = df["value_name"].values
        if len(value_names) == 0:
            print("No default values to pivot on")
            return pd.DataFrame()

        # Build the dynamic part of the SQL query
        dynamic_sql = ", ".join(
            [
                f"MAX(CASE WHEN value_name = '{value_name}' THEN value ELSE NULL END) AS '{value_name}'"
                for value_name in value_names
            ]
        )

        # Construct the final SQL query
        final_sql = f"""
        SELECT projid,
            tstamp,
            filename,
            {dynamic_sql}
        FROM logs
        WHERE value_type = 1 AND ctx_id IS NULL
        GROUP BY projid, tstamp, filename;
        """

        # Execute the final SQL query
        return pd.read_sql(
            final_sql,
            conn,
            parse_dates=[
                "tstamp",
            ],
            coerce_float=True,
        )

    if not args:
        return _pivot_star()

    dataframes = []
    loops = pd.read_sql("SELECT * FROM LOOPS", conn, coerce_float=False)
    for value_name in args:
        logs = pd.read_sql(
            f'SELECT * FROM logs WHERE value_name = "{value_name}"',
            conn,
            parse_dates=["tstamp"],
            coerce_float=True,
        )
        logs = logs[["projid", "tstamp", "filename", "ctx_id", "value"]]
        logs = logs.rename(columns={"value": value_name})
        while logs["ctx_id"].notna().any():
            logs = pd.merge(
                right=logs,
                left=loops,
                how="inner",
                on=[
                    "ctx_id",
                ],
            )
            loop_name = logs["loop_name"].unique()[0]
            logs = logs.drop(columns=["loop_name", "loop_entries"])
            logs = logs.rename(
                columns={
                    "loop_iteration": loop_name,
                }
            )
            logs["ctx_id"] = logs["parent_ctx_id"]
            logs = logs.drop(columns=["parent_ctx_id"])

        logs = logs.drop(columns=["ctx_id"])
        dataframes.append(logs)

    def join_on_common_columns(df1, df2):
        common_columns = set(df1.columns) & set(df2.columns)
        return pd.merge(df1, df2, on=list(common_columns), how="outer")

    all_joined = reduce(join_on_common_columns, dataframes)
    cols = [c for c in all_joined.columns if c not in args]
    cols += list(args)
    return all_joined[cols]
