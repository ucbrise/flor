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
    if isinstance(context, dict):
        parent_context_id = (
            insert_context(cursor, context["p_ctx"])
            if context["p_ctx"] is not None
            else None
        )
    else:
        parent_context_id = (
            insert_context(cursor, context.p_ctx) if context.p_ctx is not None else None
        )
    if isinstance(context, orm.Loop):
        cursor.execute(
            """INSERT INTO loops (ctx_id, p_ctx_id, l_name, iteration, l_value) VALUES (?, ?, ?, ?, ?)""",
            (
                int(context.ctx_id),
                parent_context_id,
                str(context.name),
                int(context.iteration) if context.iteration is not None else None,
                str(context.value) if context.value is not None else None,
            ),
        )
        return int(context.ctx_id)
    else:
        cursor.execute(
            """INSERT INTO loops (ctx_id, p_ctx_id, l_name, iteration, l_value) VALUES (?, ?, ?, ?, ?)""",
            (
                int(context["ctx_id"]),
                parent_context_id,
                str(context["name"]),
                int(context["iteration"]) if context["iteration"] is not None else None,
                str(context["value"]) if context["value"] is not None else None,
            ),
        )
        return int(context["ctx_id"])


def unpack(output_buffer, cursor):
    if not output_buffer:
        return
    for each in output_buffer:
        if isinstance(each, orm.Log):
            ctx_id = insert_context(cursor, each.ctx) if each.ctx is not None else None
            cursor.execute(
                """INSERT INTO logs (projid, tstamp, filename, ctx_id, value_name, value, value_type) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    each.projid,
                    each.tstamp,
                    each.filename,
                    ctx_id,
                    each.name,
                    str(each.value),
                    each.type,
                ),
            )
        else:
            # Parse JSON dict
            ctx_id = (
                insert_context(cursor, each["ctx"]) if each["ctx"] is not None else None
            )
            cursor.execute(
                """INSERT INTO logs (projid, tstamp, filename, ctx_id, value_name, value, value_type) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    each["projid"],
                    each["tstamp"],
                    each["filename"],
                    ctx_id,
                    each["name"],
                    str(each["value"]),
                    each["type"],
                ),
            )


def create_tables(cursor):
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS loops (
            ctx_id INTEGER,
            p_ctx_id INTEGER,
            l_name TEXT,
            iteration INTEGER,
            l_value TEXT
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
    loops = pd.read_sql("SELECT * FROM loops", conn, coerce_float=False)
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
                left=loops,
                right=logs,
                how="inner",
                on=["ctx_id"],
            )
            loop_name = logs["l_name"].unique()[0]
            logs = logs.drop(
                columns=[
                    "l_name",
                ]
            )
            if logs["iteration"].notna().all():
                logs["iteration"] = logs["iteration"].astype(int)
            logs = logs.rename(
                columns={"iteration": loop_name, "l_value": f"{loop_name}_value"}
            )
            if logs[loop_name].isna().any():
                logs = logs.drop(columns=[loop_name])
            if logs[f"{loop_name}_value"].isna().all():
                logs = logs.drop(columns=[f"{loop_name}_value"])

            assert loop_name in logs or f"{loop_name}_value" in logs

            logs["ctx_id"] = logs["p_ctx_id"]
            logs = logs.drop(columns=["p_ctx_id"])

        logs = logs.drop(columns=["ctx_id"])
        dataframes.append(logs)

    def join_on_common_columns(df1, df2):
        common_columns = set(df1.columns) & set(df2.columns)
        return pd.merge(df1, df2, on=list(common_columns), how="outer")

    all_joined = reduce(join_on_common_columns, dataframes)
    cols = [c for c in all_joined.columns if c not in args]
    cols += list(args)
    return all_joined[cols]
